"""Obstacle supervisor: turns workspace-camera detections into requests for the
motion layer. Its only obstacle input is /env/workspace_detections, never
the ground truth.

Per frame: track and classify (perception/obstacle_tracking.py). Person
tracks: speed and separation monitoring against a computed protective
distance; /mpc/speed_scale falls with the margin and /mpc/hold is asserted
at zero. People are never avoided or dodged. Static tracks: the nearest one
to the tool goes to /mpc/static_obstacle for the detour planner. No
detections for PERCEPTION_TIMEOUT_S: hold. A software stand-in, not a
safety function (docs/design_notes.md).
"""
import csv
import json
import time
from pathlib import Path

import numpy as np
import pinocchio as pin
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64, Float64MultiArray, String

from perception import obstacle_tracking as ot
from pick_place_common.scene import WORKSPACE_CAM_RATE_HZ

MJCF_PATH = str(Path(__file__).resolve().parent.parent.parent / "common" / "models" / "panda_robot.xml")

# Arm points checked against person tracks: (frame, radius). Radii as the OCP
# proxies; the TCP sized for the gripper and a held box.
ARM_POINTS = [("link3", 0.10), ("link5", 0.09), ("link7", 0.09), ("tcp_site", 0.10)]

CONTROL_TICK_S = 0.02  # one tick to publish, one to apply
PERCEPTION_TIMEOUT_S = 0.5  # no detections this long: hold
SPEED_BAND = 0.25  # log a scale change only across bands this wide


class ObstacleSupervisorNode(Node):
    def __init__(self):
        super().__init__("obstacle_supervisor_node")
        # 1.6 m/s walking speed, halved for the half-scale cell.
        self.declare_parameter("human_speed", 0.8)
        self.declare_parameter("a_max", 3.0)  # = task_node.MAX_ACCEL
        self.declare_parameter("uncertainty_m", 0.05)  # camera and tracking position pad
        self.declare_parameter("ramp_m", 0.6)  # margin over which speed recovers to 100%
        self.declare_parameter("crawl", 0.15)
        self.declare_parameter("resume_margin_m", 0.05)  # hysteresis: extra clearance to leave hold
        self.declare_parameter("csv_path", "")
        p = self.get_parameter
        self.human_speed = float(p("human_speed").value)
        self.a_max = float(p("a_max").value)
        self.uncertainty = float(p("uncertainty_m").value)
        self.ramp = float(p("ramp_m").value)
        self.crawl = float(p("crawl").value)
        self.resume_margin = float(p("resume_margin_m").value)
        csv_path = p("csv_path").value
        self.csv = csv.writer(open(csv_path, "w", newline="")) if csv_path else None
        if self.csv:
            self.csv.writerow(["t", "latency_ms", "n_tracks", "id", "label", "visible", "x", "y", "speed",
                               "gap", "required", "v_toward", "scale", "hold"])

        self.model = pin.buildModelFromMJCF(MJCF_PATH)
        self.data = self.model.createData()
        self.frame_ids = [(self.model.getFrameId(n), r, n) for n, r in ARM_POINTS]

        self.q = self.qdot = None
        self.tracker = ot.Tracker()
        self.latency_ema = 0.05
        self.last_detection_wall = None
        self.last_detection_mono = None
        self.stale = False
        self.last_capture = None
        self._last_watchdog = None
        self.hold = False
        self.scale = 1.0
        self.last_band = 4
        self.last_labels = {}

        self.create_subscription(JointState, "/sim/joint_states", self._on_state, 10)
        self.create_subscription(Float64MultiArray, "/env/workspace_detections", self._on_detections,
                                 qos_profile_sensor_data)
        self.hold_pub = self.create_publisher(Bool, "/mpc/hold", 10)
        self.scale_pub = self.create_publisher(Float64, "/mpc/speed_scale", 10)
        self.static_pub = self.create_publisher(Float64MultiArray, "/mpc/static_obstacle", 10)
        # Display only (the sim's obstacle view): tracks and decision as JSON.
        self.tracks_pub = self.create_publisher(String, "/supervisor/tracks", 10)
        self.create_timer(0.1, self._watchdog)
        self._publish()

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    @staticmethod
    # In-process durations use the monotonic clock: WSL2 steps the system (ROS)
    # clock forward ~1.5 s every ~34 s, which tripped the perception fail-safe
    # (known_issues B4). Times compared across processes stay on the ROS clock.
    def _mono():
        return time.monotonic()

    def _on_state(self, msg):
        self.q = np.array(msg.position)
        self.qdot = np.array(msg.velocity)

    def _arm_points(self):
        """[(name, radius, position, velocity)] of the checked arm points."""
        pin.forwardKinematics(self.model, self.data, self.q, self.qdot)
        pin.updateFramePlacements(self.model, self.data)
        out = []
        for fid, r, name in self.frame_ids:
            v = pin.getFrameVelocity(self.model, self.data, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).linear
            out.append((name, r, np.array(self.data.oMf[fid].translation), np.array(v)))
        return out

    def _separation(self, tr, arm_points, latency):
        """(gap, required, v_toward, worst_point) for one track: the arm point with the
        smallest gap - required. The obstacle is a column from the floor to z_max
        (the camera sees only its top), modelled as a capsule.
        """
        best = None
        for name, r_pt, pos, vel in arm_points:
            axis_top = max(tr.z_max - tr.radius, 0.0)
            nearest = np.array([tr.pos[0], tr.pos[1], np.clip(pos[2], 0.0, axis_top)])
            delta = nearest - pos
            dist = float(np.linalg.norm(delta))
            gap = dist - tr.radius - r_pt
            v_toward = float(vel @ delta / dist) if dist > 1e-6 else 0.0
            required, _ = ot.protective_distance(
                self.human_speed, v_toward, latency, self.a_max, self.uncertainty)
            if best is None or gap - required < best[0] - best[1]:
                best = (gap, required, v_toward, name)
        return best

    def _on_detections(self, msg):
        t_capture, n = msg.data[0], int(msg.data[1])
        blobs = np.array(msg.data[2:], dtype=float).reshape(n, 6)
        prev_wall, prev_capture = self.last_detection_wall, self.last_capture
        self.last_detection_wall = self._now()
        self.last_detection_mono = self._mono()
        self.last_capture = t_capture
        if self.stale:
            self.stale = False
            # Capture gap ~0.1 s with a long arrival gap: frames were late, not missing.
            self.get_logger().info(
                f"workspace detections resumed: arrival gap {self.last_detection_wall - prev_wall:.2f} s, "
                f"capture gap {t_capture - prev_capture:.2f} s")
        self.latency_ema += 0.2 * ((self.last_detection_wall - t_capture) - self.latency_ema)
        tracks = self.tracker.update(t_capture, blobs)
        if self.q is None:
            return
        # Age of what we act on: capture-to-receipt latency, one frame period (the
        # person moves unseen between frames), one tick to publish and one to apply.
        latency = self.latency_ema + 1.0 / WORKSPACE_CAM_RATE_HZ + CONTROL_TICK_S * 2
        arm_points = self._arm_points()

        scale, worst, worst_sep = 1.0, None, None
        held_margin = self.resume_margin if self.hold else 0.0
        static_candidates = []
        for tr in tracks:
            self._log_label_change(tr)
            if tr.label == ot.STATIC:
                static_candidates.append(tr)
                continue
            gap, required, v_toward, pt = self._separation(tr, arm_points, latency)
            s = ot.speed_scale(gap, required + held_margin, self.ramp, self.crawl)
            if self.csv:
                self.csv.writerow([f"{t_capture:.3f}", f"{latency*1e3:.0f}", len(tracks), tr.id, tr.label,
                                   int(tr.visible), *tr.pos[:2].round(3), round(tr.speed, 3),
                                   round(gap, 3), round(required, 3), round(v_toward, 3), round(s, 3),
                                   int(s == 0.0)])
            if s < scale or worst is None:
                scale, worst, worst_sep = min(s, scale), tr, (gap, required, v_toward, pt)
        self._set_state(scale, worst, worst_sep)
        self._publish_tracks(tracks, worst, worst_sep, arm_points)

        ee = arm_points[-1][2]
        if static_candidates:
            # The latched estimate: the arm passing over the object occludes it and the
            # live blob shrinks and drifts.
            nearest = min(static_candidates, key=lambda t: np.linalg.norm(t.static_pos - ee))
            self.static_pub.publish(Float64MultiArray(data=[*map(float, nearest.static_pos), nearest.static_r]))
        else:
            self.static_pub.publish(Float64MultiArray(data=[0.0, 0.0, 0.0, 0.0]))

    def _publish_tracks(self, tracks, worst, worst_sep, arm_points):
        worst_info = None
        if worst is not None:
            gap, required, v_toward, pt = worst_sep
            pos = next(p for name, _r, p, _v in arm_points if name == pt)
            worst_info = {"id": worst.id, "gap": gap, "required": required, "v_toward": v_toward,
                          "point": pt, "point_xyz": [float(v) for v in pos]}
        self.tracks_pub.publish(String(data=json.dumps({
            "hold": bool(self.hold), "scale": float(self.scale), "worst": worst_info,
            "tracks": [{"id": t.id, "label": t.label, "x": float(t.pos[0]), "y": float(t.pos[1]),
                        "r": float(t.radius), "top": float(t.z_max), "vx": float(t.vel[0]),
                        "vy": float(t.vel[1]), "visible": bool(t.visible)} for t in tracks]})))

    def _log_label_change(self, tr):
        if self.last_labels.get(tr.id) != tr.label:
            self.get_logger().info(
                f"track {tr.id} classified {tr.label.upper()} at ({tr.pos[0]:+.2f},{tr.pos[1]:+.2f}), "
                f"top {tr.z_max:.2f} m, r {tr.radius:.2f} m, speed {tr.speed:.2f} m/s, seen {tr.n_seen} frames")
            self.last_labels[tr.id] = tr.label

    def _set_state(self, scale, worst, sep):
        hold = scale == 0.0
        band = int(scale / SPEED_BAND) if scale < 1.0 else 99
        if worst is not None:
            gap, required, v_toward, pt = sep
            detail = (f"track {worst.id} ({worst.label}{'' if worst.visible else ', occluded/coasting'}) "
                      f"gap {gap:.2f} m vs required {required:.2f} m at {pt}, "
                      f"robot closing {v_toward:+.2f} m/s")
        else:
            detail = "no person-class track"
        if hold != self.hold:
            self.get_logger().warn(f"{'HOLD' if hold else 'RESUME'}: {detail}")
        elif band != self.last_band and not hold:
            self.get_logger().info(f"speed scale {scale:.2f}: {detail}")
        self.hold, self.scale, self.last_band = hold, scale, band
        self._publish()

    def _watchdog(self):
        # Loop-stall probe (10 Hz timer). Also the heartbeat: task_node holds if the
        # supervisor goes silent, so state is republished even without detections.
        now = self._mono()
        if self._last_watchdog is not None and now - self._last_watchdog > 0.35:
            self.get_logger().warn(f"supervisor loop stalled for {now - self._last_watchdog:.2f} s")
        self._last_watchdog = now
        if self.last_detection_wall is None:
            self._publish()
            return
        if self._mono() - self.last_detection_mono <= PERCEPTION_TIMEOUT_S:
            self._publish()
            return
        if not self.stale:
            self.stale = True
            self.get_logger().error(
                f"HOLD: no workspace detections for {PERCEPTION_TIMEOUT_S}s (perception down) -- fail-safe")
        self.hold, self.scale = True, 0.0
        self._publish()

    def _publish(self):
        self.hold_pub.publish(Bool(data=bool(self.hold)))
        self.scale_pub.publish(Float64(data=float(self.scale)))


def main():
    rclpy.init()
    node = ObstacleSupervisorNode()
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
