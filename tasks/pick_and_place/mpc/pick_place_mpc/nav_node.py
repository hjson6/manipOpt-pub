"""The base's navigation on the live robot (nav/navigator.py): goals by name (/nav/goal,
std_msgs/String: pick | place | home; the docks as taught at commissioning, saved with
the map (nav/docks.py); home is the map's origin), the pose in the map (the SLAM's map > odom from /tf times
/odom, so it works with either SLAM option), the people tracks
(/perception/people). Plans (in a worker thread), follows the route with the base MPC,
docks. From the arm's task: /task/arm_stowed (Bool; off the docking line only when
true) and /task/pause_base (Bool; stop while true).

Publishes /nav/cmd_vel (Twist, before the safety layer) at 10 Hz, /nav/status
(String "<state> <goal> <docking|normal>": the safety layer's margins; on change and
every second) and /nav/plan [length m, heading travel rad] of each goal's first plan.
Parameters: map, map_dir (the map the robot localizes in), speed (the top speed on
routes, m/s), base_safety (the distances to people x this).
See docs/implementation_notes.md#navigatorpy.
"""
import os
import time
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from std_msgs.msg import Bool, Float64MultiArray, String
from tf2_msgs.msg import TFMessage

from nav.navigator import Navigator
from nav import docks as nav_docks
from pick_place_common.telemetry import Telemetry
from pick_place_mpc.slam_node import DEFAULT_MAP_DIR
from slam.grid import OccupancyGrid
from slam.pose_graph import compose

CONTROL_PERIOD_S = 0.1
STATUS_PERIOD_S = 1.0
PEOPLE_STALE_S = 0.5  # older tracks are not used (the safety layer stops on it)
LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)


class NavNode(Node):
    def __init__(self):
        super().__init__("nav_node")
        self.declare_parameter("map", "stations")
        self.declare_parameter("map_dir", os.environ.get("MANIPOPT_MAP_DIR", DEFAULT_MAP_DIR))
        self.declare_parameter("speed", 0.5)
        self.declare_parameter("base_safety", 1.0)
        path = Path(self.get_parameter("map_dir").value) / self.get_parameter("map").value
        grid = OccupancyGrid.load(path.with_suffix(".yaml"))
        docks = nav_docks.load(path)  # taught at commissioning
        self.nav = Navigator(grid, docks, (0.0, 0.0, 0.0), background=True,  # home: the map's origin
                             speed=float(self.get_parameter("speed").value),
                             safety=float(self.get_parameter("base_safety").value))
        self.map_to_odom = None
        self.odom = None  # (x, y, yaw), (v, omega)
        self.people = []
        self.people_t = -np.inf
        self.pending_goal = None
        self.last_status = None
        self.last_status_t = -np.inf
        self.telemetry = Telemetry("nav", ["t", "x", "y", "yaw", "v_meas", "w_meas", "v_cmd", "w_cmd", "state", "goal",
                                           "progress", "status", "n_people", "ms"])
        self.cmd_pub = self.create_publisher(Twist, "/nav/cmd_vel", 10)
        self.status_pub = self.create_publisher(String, "/nav/status", 10)
        self.plan_pub = self.create_publisher(Float64MultiArray, "/nav/plan", 10)
        self.plan_sent = None
        self.create_subscription(TFMessage, "/tf", self._on_tf, 50)
        self.create_subscription(Odometry, "/odom", self._on_odom, 50)
        self.create_subscription(Float64MultiArray, "/perception/people", self._on_people, qos_profile_sensor_data)
        self.create_subscription(String, "/nav/goal", self._on_goal, 10)
        self.create_subscription(Bool, "/task/arm_stowed", self._on_arm_stowed, LATCHED)
        self.create_subscription(Bool, "/task/pause_base", self._on_pause, LATCHED)
        self.create_timer(CONTROL_PERIOD_S, self._tick)
        self.get_logger().info(f"navigation on the map {path}, top speed {self.nav.speed:.2f} m/s, people's "
                               f"distances x{self.nav.safety:g}; goals: pick, place, home")

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_tf(self, msg):
        for tr in msg.transforms:
            if tr.header.frame_id == "map" and tr.child_frame_id == "odom":
                q = tr.transform.rotation
                self.map_to_odom = np.array([tr.transform.translation.x, tr.transform.translation.y,
                                             2 * np.arctan2(q.z, q.w)])

    def _on_odom(self, msg):
        q = msg.pose.pose.orientation
        self.odom = (np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, 2 * np.arctan2(q.z, q.w)]),
                     (msg.twist.twist.linear.x, msg.twist.twist.angular.z))

    def _on_people(self, msg):
        d = msg.data
        n = int(d[1])
        rows = np.asarray(d[2:2 + 7 * n]).reshape(n, 7)
        self.people = [tuple(r[1:5]) for r in rows]
        self.people_t = self._now()

    def _on_arm_stowed(self, msg):
        if bool(msg.data) != self.nav.arm_ready:
            self.get_logger().info(f"arm {'stowed' if msg.data else 'out'}")
        self.nav.arm_ready = bool(msg.data)

    def _on_pause(self, msg):
        if bool(msg.data) != self.nav.paused:
            self.get_logger().info(f"base {'paused' if msg.data else 'resumed'} by the arm's task")
        self.nav.paused = bool(msg.data)

    def _on_goal(self, msg):
        name = msg.data.strip()
        if name not in ("pick", "place", "home"):
            self.get_logger().warn(f"unknown goal '{name}'")
            return
        self.pending_goal = name

    def _tick(self):
        if self.map_to_odom is None or self.odom is None:
            return
        t = self._now()
        pose = compose(self.map_to_odom, self.odom[0])
        if self.pending_goal is not None:
            goal, self.pending_goal = self.pending_goal, None
            self.nav.set_goal(goal, pose)
            self.get_logger().info(f"goal {goal}: {self.nav.state}")
        people = self.people if t - self.people_t < PEOPLE_STALE_S else []
        t0 = time.perf_counter()
        v, w = self.nav.step(t, pose, self.odom[1], people)
        ms = (time.perf_counter() - t0) * 1e3
        cmd = Twist()
        cmd.linear.x, cmd.angular.z = float(v), float(w)
        self.cmd_pub.publish(cmd)
        status = f"{self.nav.state} {self.nav.goal} {'docking' if self.nav.docking else 'normal'}"
        if status != self.last_status or t - self.last_status_t >= STATUS_PERIOD_S:
            if status != self.last_status:
                self.get_logger().info(f"nav: {status}")
            self.status_pub.publish(String(data=status))
            self.last_status, self.last_status_t = status, t
        if self.nav.planned is not None and self.nav.planned is not self.plan_sent:
            self.plan_sent = self.nav.planned
            self.plan_pub.publish(Float64MultiArray(data=[float(v) for v in self.nav.planned]))
        progress = self.nav.ref.progress if self.nav.ref is not None else -1
        solve = self.nav.last[0] if self.nav.last is not None else -1
        self.telemetry.row(round(t, 3), *np.round(pose, 4), *np.round(self.odom[1], 3), round(v, 3), round(w, 3),
                           self.nav.state, self.nav.goal, progress, solve, len(people), round(ms, 2))


def main():
    rclpy.init()
    node = NavNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.telemetry.flush()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
