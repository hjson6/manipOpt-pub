"""Validation only (like people_monitor_node): each navigation drive scored on the truth
(pick_place_common/nav_scoring.py, as scripts/dev/nav_sim.py does offline): the base's
true pose (/sim/base_truth, its speeds by differences) and the true crowd (/env/people),
both in the room frame; the commands after the safety layer (/base/cmd_vel), its state
(/safety/base), the navigation's (/nav/status) and its plan (/nav/plan) for the motion
checks. A drive starts
when the status names a new goal and ends docked, parked or failed; then its arrival
error against the taught dock (or home) and its line are logged; the totals when the
node stops.
"""
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String

from pick_place_common.nav_scoring import DriveScore, summary
from pick_place_common.scene import BASE_HOME_POSE, PICK_DOCK_BASE, PLACE_DOCK_BASE
from slam.pose_graph import wrap

PLANT_PERIOD_S = 0.02
TARGETS = {"pick": PICK_DOCK_BASE, "place": PLACE_DOCK_BASE, "home": BASE_HOME_POSE}
SAFETY_STATES = {0: "clear", 1: "warn", 2: "stop", 3: "fault"}


class NavMonitorNode(Node):
    def __init__(self):
        super().__init__("nav_monitor_node")
        self.scores = []
        self.score = None
        self.state, self.goal, self.docking = "idle", None, False
        self.safety_state = ""
        self.people = []
        self.prev = None  # (step, pose)
        self.pose = None
        self.cmd_t = None
        self.create_subscription(Float64MultiArray, "/sim/base_truth", self._on_truth, 50)
        self.create_subscription(Float64MultiArray, "/env/people", self._on_people, 10)
        self.create_subscription(Twist, "/base/cmd_vel", self._on_cmd, 10)
        self.create_subscription(Float64MultiArray, "/safety/base", self._on_safety, 10)
        self.create_subscription(String, "/nav/status", self._on_status, 10)
        self.create_subscription(Float64MultiArray, "/nav/plan", self._on_plan, 10)

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_people(self, msg):
        d = np.asarray(msg.data).reshape(-1, 3)
        self.people = [tuple(p) for p in d if np.isfinite(p[0])]

    def _on_plan(self, msg):
        if self.score is not None:
            self.score.plan(*msg.data[:2])

    def _on_safety(self, msg):
        self.safety_state = SAFETY_STATES.get(int(msg.data[1]), "")

    def _on_status(self, msg):
        state, goal, fields = msg.data.split()[:3]
        self.docking = fields == "docking"
        if goal in TARGETS and state not in ("docked", "parked", "failed") and self.score is None:
            self.score = DriveScore(goal, self._now())
        if self.score is not None and state in ("docked", "parked", "failed") and goal == self.score.r["goal"]:
            self._finish(state)
        self.state, self.goal = state, goal

    def _finish(self, state):
        r = self.score.r
        g = TARGETS[r["goal"]]
        c, s = np.cos(g[2]), np.sin(g[2])
        dx, dy = self.pose[0] - g[0], self.pose[1] - g[1]
        along, lateral, yaw = c * dx + s * dy, -s * dx + c * dy, np.degrees(wrap(self.pose[2] - g[2]))
        r["state"], r["err"] = state, (along, lateral, yaw)
        self.scores.append(self.score)
        self.get_logger().info(f"drive {len(self.scores)} to {r['goal']}: {state} {self._now() - r['t0']:.1f} s "
                               f"{r['dist']:.1f} m; error along/lateral {1e3 * along:+.0f}/{1e3 * lateral:+.0f} mm, yaw "
                               f"{yaw:+.2f} deg; {self.score.line()}")
        self.score = None

    def _on_truth(self, msg):
        step = int(msg.data[0])
        x, y, _z, qw, _qx, _qy, qz = msg.data[1:8]
        self.pose = np.array([x, y, 2 * np.arctan2(qz, qw)])
        prev, self.prev = self.prev, (step, self.pose)
        if prev is None or step <= prev[0] or self.score is None:
            return
        dt = (step - prev[0]) * PLANT_PERIOD_S
        d = self.pose - prev[1]
        v = (np.cos(self.pose[2]) * d[0] + np.sin(self.pose[2]) * d[1]) / dt
        w = wrap(d[2]) / dt
        self.score.truth(self._now(), self.pose, float(v), float(w), self.people, self.docking, self.safety_state,
                         self.state)

    def _on_cmd(self, msg):
        now = self._now()
        if self.score is not None and self.cmd_t is not None:
            self.score.command(now, msg.linear.x, msg.angular.z, now - self.cmd_t, self.safety_state, self.state)
        self.cmd_t = now

    def report(self):
        if self.scores:
            ok = sum(s.r["state"] in ("docked", "parked") for s in self.scores)
            e = np.array([s.r["err"] for s in self.scores])
            self.get_logger().info(f"navigation over the run: arrived {ok}/{len(self.scores)}, error |along| max "
                                   f"{1e3 * np.abs(e[:, 0]).max():.0f} mm, |lateral| max {1e3 * np.abs(e[:, 1]).max():.0f} "
                                   f"mm, |yaw| max {np.abs(e[:, 2]).max():.2f} deg; {summary(self.scores)}")


def main():
    rclpy.init()
    node = NavMonitorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.report()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
