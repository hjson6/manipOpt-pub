"""Validation only: the mobile job's people safety at the arm, on the truth. The plant's
model (stations layout) is put at the base's true pose (/sim/base_truth), the arm's
joint angles (/sim/joint_states) and the people's true poses (/env/people); the
smallest distance between the arm's collision geoms and any person's geoms
(body_clearance.py), apart while the arm works (the base docked or parked, /nav/status,
the arm moving) and while the base drives: then apart when the base moves and its
nearest point closes on that person (its speed from the truth), or at any time (people
walking up to it count too), and while the base moves with the arm out of the carry
pose (docking and undocking, the arm working meanwhile). Each supervisor hold (/mpc/hold rising)
with its cause: a person within HOLD_EXPLAINED_M of the arm, or perception (no people
detections yet, or none for PERCEPTION_LATE_S), else unexplained. Boxes placed
(/task/action place_held). Logs a summary every `summary_s` and the totals when the
node stops.
"""
import time

import mujoco
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Bool, Float64MultiArray, String

from pick_place_common.body_clearance import BodyClearance
from pick_place_common import frames
from pick_place_common.mujoco_sim_node import ARM_JOINT_NAMES, load_scene_model
from pick_place_common.scene import CARRY_TCP_POSITION

CHECK_PERIOD_S = 0.1
MOVING_RAD_S = 0.1  # joint speed norm: the arm moving
HOLD_EXPLAINED_M = 2.5  # a person this near the arm explains a hold (wider with arm_safety, assumed_speed)
ISO_SPEED = 1.6
PERCEPTION_LATE_S = 0.5  # the supervisor's own watchdog
NEAR_M = 3.0  # people farther from the base (in plan) are not measured exactly
MOVING_MS, MOVING_RADS = 0.05, 0.1  # the base moving
CLOSING_MS = 0.05
OUT_M = 0.05  # the tool this far from the carry pose: the arm is out


class JobMonitorNode(Node):
    def __init__(self):
        super().__init__("job_monitor_node")
        self.declare_parameter("summary_s", 60.0)
        self.declare_parameter("arm_safety", 1.0)
        self.declare_parameter("assumed_speed", ISO_SPEED)
        self.explained_m = HOLD_EXPLAINED_M * max(1.0, float(self.get_parameter("arm_safety").value)) * max(
            1.0, float(self.get_parameter("assumed_speed").value) / ISO_SPEED)
        self.m = load_scene_model(layout="stations")
        self.d = mujoco.MjData(self.m)
        self.qa = [self.m.joint(n).qposadr[0] for n in ARM_JOINT_NAMES]
        self.free = self.m.joint("base_free").qposadr[0]
        self.people = [self.m.body_mocapid[self.m.body(f"person_{i}").id] for i in range(1, 7)]
        self.clearance = BodyClearance(self.m, near_m=NEAR_M)
        self.tcp_site = self.m.site("tcp_site").id
        self.base = None
        self.base_prev = None  # (time, pose) for the base's speed
        self.v_world, self.w_world = np.zeros(3), np.zeros(3)
        self.q = self.qdot = None
        self.poses = []
        self.hold = False
        self.docked = True  # until /nav/status says otherwise
        self.detections_at = None
        self.window = self._fresh()
        self.total = self._fresh()
        self.placed = 0
        self.t0 = time.monotonic()
        self.create_subscription(Float64MultiArray, "/sim/base_truth", self._on_base, 10)
        self.create_subscription(JointState, "/sim/joint_states", self._on_joints, 10)
        self.create_subscription(Float64MultiArray, "/env/people", self._on_people, 10)
        self.create_subscription(Bool, "/mpc/hold", self._on_hold, 10)
        self.create_subscription(String, "/task/action", self._on_action, 10)
        self.create_subscription(String, "/nav/status", self._on_nav, 10)
        self.create_subscription(Float64MultiArray, "/perception/lidar_detections", self._on_detections,
                                 qos_profile_sensor_data)
        self.create_timer(CHECK_PERIOD_S, self._check)
        self.create_timer(float(self.get_parameter("summary_s").value), self._summary)

    @staticmethod
    def _fresh():
        return {"work": (np.inf, ""), "drive": (np.inf, ""), "drive_closing": (np.inf, ""), "overlap": (np.inf, ""),
                "holds": 0,
                "person": 0, "perception": 0, "unexplained": 0, "hold_dists": []}

    def _on_nav(self, msg):
        self.docked = msg.data.split()[0] in ("docked", "parked", "idle")

    def _on_detections(self, _msg):
        self.detections_at = time.monotonic()

    def _on_base(self, msg):
        self.base = np.asarray(msg.data[1:8], dtype=float)
        t = float(msg.data[0]) * 0.02  # the plant's step
        yaw = 2 * np.arctan2(self.base[6], self.base[3])
        if self.base_prev is not None and t > self.base_prev[0]:
            dt = t - self.base_prev[0]
            self.v_world = np.r_[(self.base[:2] - self.base_prev[1][:2]) / dt, 0.0]
            self.w_world = np.array([0.0, 0.0, ((yaw - self.base_prev[2] + np.pi) % (2 * np.pi) - np.pi) / dt])
        self.base_prev = (t, self.base[:3].copy(), yaw)

    def _on_joints(self, msg):
        self.q, self.qdot = np.array(msg.position[:7]), np.array(msg.velocity[:7])

    def _on_people(self, msg):
        self.poses = np.asarray(msg.data, dtype=float).reshape(-1, 3)

    def _on_action(self, msg):
        if msg.data.startswith("place_held"):
            self.placed += 1

    def _place(self):
        """The model at the truth; False until every input has come."""
        if self.base is None or self.q is None:
            return False
        self.d.qpos[self.free:self.free + 7] = self.base
        self.d.qpos[self.qa] = self.q
        for k, mid in enumerate(self.people):
            p = self.poses[k] if k < len(self.poses) else (np.nan, np.nan, np.nan)  # not in the scenario: parked
            if np.isfinite(p[0]):
                self.d.mocap_pos[mid] = (p[0], p[1], 0.0)
                self.d.mocap_quat[mid] = (np.cos(p[2] / 2), 0.0, 0.0, np.sin(p[2] / 2))
            else:
                self.d.mocap_pos[mid] = (0.0, 0.0, -10.0)
        mujoco.mj_forward(self.m, self.d)
        return True

    def _nearest(self):
        """(distance, arm part, person, arm point, person point) of the closest pair."""
        return self.clearance.nearest(self.d)

    def _check(self):
        working = self.docked
        if self.qdot is None or (working and np.linalg.norm(self.qdot) < MOVING_RAD_S) or not self._place():
            return
        dist, arm, person, p_arm, p_person = self._nearest()
        if not np.isfinite(dist):
            return
        keys = ["work"] if working else ["drive"]
        if not working and (np.linalg.norm(self.v_world) > MOVING_MS or abs(self.w_world[2]) > MOVING_RADS):
            if self.clearance.closing_speed(self.d, p_arm, p_person, self.v_world, self.w_world) > CLOSING_MS:
                keys.append("drive_closing")
            tcp = frames.to_arm(frames.arm_base_pose(self.m, self.d), self.d.site_xpos[self.tcp_site])
            if np.linalg.norm(tcp - CARRY_TCP_POSITION) > OUT_M:
                keys.append("overlap")
        for acc in (self.window, self.total):
            for key in keys:
                if dist < acc[key][0]:
                    acc[key] = (dist, f"{arm} / {person}")

    def _on_hold(self, msg):
        rising = msg.data and not self.hold
        self.hold = bool(msg.data)
        if not rising or not self._place():
            return
        dist = self._nearest()[0]
        late = self.detections_at is None or time.monotonic() - self.detections_at > PERCEPTION_LATE_S
        cause = "person" if dist <= self.explained_m else "perception" if late else "unexplained"
        for acc in (self.window, self.total):
            acc["holds"] += 1
            acc[cause] += 1
            acc["hold_dists"].append(dist)
        self.get_logger().info(f"arm hold ({cause}): nearest person {dist:.2f} m from the arm")

    def _report(self, acc, what):
        d = [v for v in acc["hold_dists"] if np.isfinite(v)]
        (w, wp), (dr, dp), (dc, dcp), (ov, ovp) = acc["work"], acc["drive"], acc["drive_closing"], acc["overlap"]
        self.get_logger().info(
            f"arm and people {what}: closest while the arm worked {1e3 * w:.0f} mm ({wp}); while the base "
            f"drove, closing on them {1e3 * dc:.0f} mm ({dcp}), any {1e3 * dr:.0f} mm ({dp}); while it moved with "
            f"the arm out {1e3 * ov:.0f} mm ({ovp}) (people within {NEAR_M:.0f} m measured); holds {acc['holds']}: "
            f"a person within {self.explained_m:g} m {acc['person']}, perception {acc['perception']}, unexplained "
            f"{acc['unexplained']}" + (f" (nearest person at holds: max {max(d):.2f} m)" if d else "")
            + f"; boxes placed {self.placed} in {(time.monotonic() - self.t0) / 60:.1f} min")

    def _summary(self):
        self._report(self.window, "over the last window")
        self.window = self._fresh()


def main():
    rclpy.init()
    node = JobMonitorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._report(node.total, "over the run")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
