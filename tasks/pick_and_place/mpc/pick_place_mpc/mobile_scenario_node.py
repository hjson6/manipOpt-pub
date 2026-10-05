"""Scenario actor for the mobile drives (the environment, like dynamic_obstacle_node;
nothing on the method side reads its inputs): a technician drives the base along a
route with a joystick (/base/cmd_vel, steered on the true pose, /sim/base_truth), or,
with route:=nav, an operator sends the navigation its goals in turn (/nav/goal, the
next when /nav/status says the last ended); and people walk scripted paths in the
room (/env/people for person_1..6, room frame: the stations layout's crowd, some
giving way to the robot, or the cell layout's walkers). Routes and paths:
mobile_scenarios.py. Logs "route done" when the drive (or the last goal) is over.
wait_for_go: everything holds (the people standing at their starts) until /mpc/go, as
the robot does (the mobile job).
"""
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Empty, Float64MultiArray, String

from pick_place_common.mobile_scenarios import ACC, ALPHA, CROWDS, PEOPLE, ROUTES, Crowd, Driver, person_at, walking
from pick_place_common.scene import BASE_HOME_POSE, BASE_PARK_POSE

CONTROL_PERIOD_S = 0.02


class MobileScenarioNode(Node):
    def __init__(self):
        super().__init__("mobile_scenario_node")
        self.declare_parameter("route", "loop")  # loop | cross | nav | none
        self.declare_parameter("laps", 1)
        self.declare_parameter("goals", "pick place home")  # route:=nav: drives laps x these
        self.declare_parameter("people", 3)  # the first N people
        self.declare_parameter("start_delay_s", 5.0)
        self.declare_parameter("layout", "stations")
        self.declare_parameter("crowd", "stations")  # mobile_scenarios.CROWDS
        self.declare_parameter("people_speed", 1.0)  # x each person's walking speed
        self.declare_parameter("wait_for_go", False)
        route = self.get_parameter("route").value
        laps = int(self.get_parameter("laps").value)
        cell = self.get_parameter("layout").value == "cell"
        start = BASE_PARK_POSE if cell else BASE_HOME_POSE
        self.driver = None if route in ("none", "nav") else Driver(ROUTES[route] * laps, start, reverse_out=cell)
        self.goals = self.get_parameter("goals").value.split() * laps if route == "nav" else []
        self.goal_sent = None  # (goal, time) of the drive under way
        n = int(self.get_parameter("people").value)
        k = float(self.get_parameter("people_speed").value)
        self.people = walking([p for p in PEOPLE if p[0] != "person_obstacle"][:n], k) if cell else []
        self.crowd = None if cell else Crowd(walking(CROWDS[self.get_parameter("crowd").value][:n], k))
        self.last_t = None
        self.start_delay = float(self.get_parameter("start_delay_s").value)
        self.waiting = bool(self.get_parameter("wait_for_go").value)
        self.start = self.get_clock().now().nanoseconds * 1e-9 + self.start_delay
        if self.waiting:
            self.create_subscription(Empty, "/mpc/go", self._on_go, 10)
        self.truth = None
        self.v = self.w = 0.0
        self.done_logged = False
        self.cmd_pub = self.create_publisher(Twist, "/base/cmd_vel", 10)
        self.goal_pub = self.create_publisher(String, "/nav/goal", 10)
        self.create_subscription(String, "/nav/status", self._on_nav_status, 10)
        self.people_pub = self.create_publisher(Float64MultiArray, "/env/people", 10)
        self.create_subscription(Float64MultiArray, "/sim/base_truth", self._on_truth, 10)
        self.create_timer(CONTROL_PERIOD_S, self._tick)
        self.get_logger().info(f"route {route} x{laps}, {n} people walking at x{k:g}, layout "
                               f"{'cell' if cell else 'stations'}")

    def _on_go(self, _msg):
        if self.waiting:
            self.waiting = False
            self.start = self.get_clock().now().nanoseconds * 1e-9
            self.get_logger().info("go: the people start walking")

    def _on_nav_status(self, msg):
        state, goal = msg.data.split()[:2]
        if self.goal_sent is None or goal != self.goal_sent[0] or state not in ("docked", "parked", "failed"):
            return
        if self.get_clock().now().nanoseconds * 1e-9 - self.goal_sent[1] < 2.0:
            return  # the status before the goal took
        self.get_logger().info(f"drive to {goal}: {state}")
        self.goal_sent = None
        if not self.goals and not self.done_logged:
            self.done_logged = True
            self.get_logger().info("route done")

    def _on_truth(self, msg):
        x, y, _z, qw, _qx, _qy, qz = msg.data[1:8]
        self.truth = np.array([x, y, 2 * np.arctan2(qz, qw)])

    def _tick(self):
        t = self.get_clock().now().nanoseconds * 1e-9 - self.start
        if self.waiting and self.truth is not None:
            if self.crowd is not None:
                people = [v for p in self.crowd.step(0.0) for v in p]
            else:
                people = [v for _, path, speed, dwell in self.people for v in person_at(path, speed, dwell, 0.0)]
            self.people_pub.publish(Float64MultiArray(data=[float(v) for v in people]))
            return
        if t < 0.0 or self.truth is None:
            return
        if self.crowd is not None:
            dt = 0.0 if self.last_t is None else t - self.last_t
            people = [v for p in self.crowd.step(dt, self.truth[:2]) for v in p]
        else:
            people = [v for _, path, speed, dwell in self.people for v in person_at(path, speed, dwell, t)]
        self.last_t = t
        self.people_pub.publish(Float64MultiArray(data=[float(v) for v in people]))
        if self.goals and self.goal_sent is None:
            goal = self.goals.pop(0)
            self.goal_pub.publish(String(data=goal))
            self.goal_sent = (goal, self.get_clock().now().nanoseconds * 1e-9)
            self.get_logger().info(f"goal {goal} ({len(self.goals)} more)")
        if self.driver is None:
            return
        sv, sw = self.driver.command(self.truth)
        self.v += np.clip(sv - self.v, -ACC * CONTROL_PERIOD_S, ACC * CONTROL_PERIOD_S)
        self.w += np.clip(sw - self.w, -ALPHA * CONTROL_PERIOD_S, ALPHA * CONTROL_PERIOD_S)
        cmd = Twist()
        cmd.linear.x, cmd.angular.z = float(self.v), float(self.w)
        self.cmd_pub.publish(cmd)
        if self.driver.done and abs(self.v) < 1e-3 and not self.done_logged:
            self.done_logged = True
            self.get_logger().info("route done")


def main():
    rclpy.init()
    node = MobileScenarioNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
