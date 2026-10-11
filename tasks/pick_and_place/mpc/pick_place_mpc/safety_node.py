"""The base's safety layer on the live robot (nav/safety.py), between the navigation and
the drives: each command (/nav/cmd_vel) and each round of both lidars' raw scans
(/env/lidar_scan, in base_link with the taught mounts) is checked against the fields
for the commanded and the measured motion (/odom); docking margins while
/nav/status says docking. Watchdogs: no scan, odometry, people tracks
(/perception/people) or localization (/slam/pose, else the map > odom transform)
within their time: stop. People's points (the people detector's foreground of the last
round, /perception/foreground_base) get the wider margin (nav/safety.py), scaled by the
base_safety parameter.

Publishes /base/cmd_vel (Twist, to base_node) and /safety/base [t, code (0 clear,
1 warn, 2 stop, 3 fault), speed cap, turn cap, v in, w in, v out, w out].
See docs/implementation_notes.md#safetypy.
"""
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float64MultiArray, String
from tf2_msgs.msg import TFMessage

from nav import safety
from pick_place_common.scene import LIDAR_MOUNTS_BASE
from pick_place_common.telemetry import Telemetry
from pick_place_mpc.scan_merger_node import ScanPairs
from slam.scan import scan_points

STALE_S = {"scan": 0.25, "odom": 0.1, "people": 0.5, "localization": 0.5, "command": 0.3}
CODES = {"clear": 0, "warn": 1, "stop": 2, "fault": 3}
BIG = 1e3  # an uncapped speed, in the status message


class SafetyNode(Node):
    def __init__(self):
        super().__init__("safety_node")
        self.declare_parameter("base_safety", 1.0)
        scale = float(self.get_parameter("base_safety").value)
        self.extra = safety.person_extra(scale)
        self.get_logger().info(f"people's stop margin {safety.MARGIN_M + self.extra:.2f} m (base_safety {scale:g}"
                               + (f", held at the {safety.PERSON_MARGIN_MIN_M} m floor)"
                                  if scale * (safety.MARGIN_M + safety.PERSON_EXTRA_M) < safety.PERSON_MARGIN_MIN_M
                                  else ")"))
        self.pairs = ScanPairs()
        self.points = np.zeros((0, 2))
        self.people = None
        self.seen = {k: -np.inf for k in STALE_S}
        self.measured = (0.0, 0.0)
        self.cmd = (0.0, 0.0)
        self.docking = False
        self.own_pose = False
        self.last_state = None
        self.telemetry = Telemetry("safety", ["t", "state", "v_cap", "w_cap", "v_in", "w_in", "v_out", "w_out", "fault"])
        self.cmd_pub = self.create_publisher(Twist, "/base/cmd_vel", 10)
        self.state_pub = self.create_publisher(Float64MultiArray, "/safety/base", 10)
        self.create_subscription(Float64MultiArray, "/env/lidar_scan", self._on_scan, 10)
        self.create_subscription(Twist, "/nav/cmd_vel", self._on_cmd, 10)
        self.create_subscription(String, "/nav/status", self._on_status, 10)
        self.create_subscription(Odometry, "/odom", self._on_odom, 50)
        self.create_subscription(Float64MultiArray, "/perception/people", self._on_people, qos_profile_sensor_data)
        self.create_subscription(Float64MultiArray, "/perception/foreground_base", self._on_foreground,
                                 qos_profile_sensor_data)
        self.create_subscription(PoseStamped, "/slam/pose", self._on_pose, 10)
        self.create_subscription(TFMessage, "/tf", self._on_tf, 50)

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_scan(self, msg):
        got = self.pairs.add(msg.data)
        if got is None:
            return
        _t, angles, ranges = got
        self.points = scan_points(ranges, angles, LIDAR_MOUNTS_BASE)
        self.seen["scan"] = self._now()
        self._apply()

    def _on_cmd(self, msg):
        self.cmd = (msg.linear.x, msg.angular.z)
        self.seen["command"] = self._now()
        self._apply()

    def _on_status(self, msg):
        self.docking = msg.data.endswith("docking")

    def _on_odom(self, msg):
        self.measured = (msg.twist.twist.linear.x, msg.twist.twist.angular.z)
        self.seen["odom"] = self._now()

    def _on_foreground(self, msg):
        n = int(msg.data[1])
        self.people = np.asarray(msg.data[2:2 + 2 * n], dtype=float).reshape(n, 2)

    def _on_people(self, _msg):
        self.seen["people"] = self._now()

    def _on_pose(self, _msg):
        self.own_pose = True
        self.seen["localization"] = self._now()

    def _on_tf(self, msg):
        if self.own_pose:
            return
        if any(tr.header.frame_id == "map" and tr.child_frame_id == "odom" for tr in msg.transforms):
            self.seen["localization"] = self._now()

    def _apply(self):
        now = self._now()
        stale = [k for k, s in STALE_S.items() if now - self.seen[k] > s]
        fault = "+".join(k for k in stale if k != "command")
        v, w = self.cmd if "command" not in stale else (0.0, 0.0)
        if fault:
            state, cap = "fault", (0.0, 0.0)
        else:
            state, cap = safety.check(self.points, v, w, self.docking, measured=self.measured, people=self.people,
                                      extra=self.extra)
        out = safety.limit(v, w, "stop" if state == "fault" else state, cap)
        twist = Twist()
        twist.linear.x, twist.angular.z = float(out[0]), float(out[1])
        self.cmd_pub.publish(twist)
        caps = [float(min(c, BIG)) for c in cap]
        self.state_pub.publish(Float64MultiArray(data=[now, float(CODES[state]), *caps, float(v), float(w), *map(float, out)]))
        if (state, fault) != self.last_state:
            if state in ("stop", "fault") or (self.last_state or ("",))[0] in ("stop", "fault"):
                self.get_logger().info(f"safety: {state}{' (' + fault + ')' if fault else ''}")
            self.last_state = (state, fault)
        self.telemetry.row(round(now, 3), state, *[round(c, 3) for c in caps], round(v, 3), round(w, 3),
                           round(out[0], 3), round(out[1], 3), fault)


def main():
    rclpy.init()
    node = SafetyNode()
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
