"""Validation only: compares the workspace camera's and the safety lidars'
detections with the environment actor's ground truth and reports error,
latency and false positives, per source. The one place truth and detections meet; nothing on the control
path subscribes to /env/dynamic_obstacle.

Truth (the room frame) is moved into the arm frame with the base's true pose
(/sim/base_truth), and interpolated to each frame's capture time, so latency
does not show up as position error. One log line per blob, a summary every
`summary_every` frames, and optionally a CSV row per blob (`csv_path`).
"""
import bisect
import csv

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float64MultiArray

from pick_place_common.scene import ARM_IN_BASE, BASE_ARM_MOUNT, CELL_POSE, compose, invert

# A blob further than this (XY) from every truth is a false positive.
MATCH_RADIUS_M = 0.25
HISTORY_S = 5.0
EDGE = object()


class DetectionMonitorNode(Node):
    def __init__(self):
        super().__init__("detection_monitor_node")
        self.declare_parameter("csv_path", "")
        self.declare_parameter("summary_every", 50)
        self.declare_parameter("verbose", True)
        self.summary_every = int(self.get_parameter("summary_every").value)
        self.verbose = bool(self.get_parameter("verbose").value)
        csv_path = self.get_parameter("csv_path").value
        self.csv = csv.writer(open(csv_path, "w", newline="")) if csv_path else None
        if self.csv:
            self.csv.writerow(["source", "t_capture", "latency_ms", "sx", "sy", "s_top", "sr", "n_px",
                               "tx", "ty", "t_top", "tr", "err_xy", "err_top", "err_r", "kind"])

        self.truth_t, self.truth = [], []  # publish time, [x, y, z, r, top]
        self.sources = {name: {"pending": [], "frames": 0, "tp": 0, "fp": 0, "missed": 0,
                               "err_xy": [], "err_z": [], "err_r": [], "latency": []}
                        for name in ("camera", "lidar")}
        self.arm_pose, self.arm_z = CELL_POSE, BASE_ARM_MOUNT[2]  # the arm base in the room
        self.create_subscription(Float64MultiArray, "/sim/base_truth", self._on_base_truth, 10)
        self.create_subscription(Float64MultiArray, "/env/dynamic_obstacle", self._on_truth, 50)
        self.create_subscription(Float64MultiArray, "/perception/camera_detections",
                                 lambda m: self._on_detections("camera", m), qos_profile_sensor_data)
        self.create_subscription(Float64MultiArray, "/perception/lidar_detections",
                                 lambda m: self._on_detections("lidar", m), qos_profile_sensor_data)

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_base_truth(self, msg):
        x, y, z, qw, qx, qy, qz = msg.data[1:8]
        yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        self.arm_pose, self.arm_z = compose((x, y, yaw), ARM_IN_BASE), z + BASE_ARM_MOUNT[2]

    def _on_truth(self, msg):
        # Keyed on the actor's publish stamp, not receipt time.
        t = msg.data[4]
        self.truth_t.append(t)
        d = msg.data
        # Tops, not centres: the overhead camera measures tops.
        top = d[5] if len(d) > 5 else d[2] + d[3]
        x, y, _ = compose(invert(self.arm_pose), (d[0], d[1], 0.0))
        self.truth.append([x, y, d[2] - self.arm_z, d[3], top - self.arm_z])
        while self.truth_t and self.truth_t[0] < t - HISTORY_S:
            self.truth_t.pop(0)
            self.truth.pop(0)
        self._drain()

    def _drain(self):
        # Evaluate a frame only once truth brackets its capture time.
        for name, src in self.sources.items():
            while src["pending"] and self.truth_t and self.truth_t[-1] >= src["pending"][0][0]:
                self._evaluate(name, *src["pending"].pop(0))

    def _truth_at(self, t):
        """Truth at time t, interpolated; None if none around then or absent; EDGE if
        the obstacle appears or leaves around then."""
        if not self.truth_t:
            return None
        i = bisect.bisect_left(self.truth_t, t)
        if i == 0 or i == len(self.truth_t):
            return None
        t0, t1 = self.truth_t[i - 1], self.truth_t[i]
        a, b = np.array(self.truth[i - 1]), np.array(self.truth[i])
        if (a[3] <= 0.0) != (b[3] <= 0.0):
            return EDGE  # appearing or leaving between the two samples: not scored
        if a[3] <= 0.0:
            return None
        w = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
        return a + (b - a) * w

    def _on_detections(self, name, msg):
        t_capture = msg.data[0]
        src = self.sources[name]
        src["pending"].append((t_capture, (self._now() - t_capture) * 1e3, msg))
        if self.count_publishers("/env/dynamic_obstacle") == 0:
            # No actor at all: every blob is a false positive.
            while src["pending"]:
                self._evaluate(name, *src["pending"].pop(0))
            return
        self._drain()

    @staticmethod
    def _blobs(name, msg):
        """(x, y, r, top or nan, size) per blob."""
        n = int(msg.data[1])
        if name == "camera":
            b = np.array(msg.data[2:], dtype=float).reshape(n, 6)
            return [(x, y, r, top, n_px) for x, y, _z, r, n_px, top in b]
        b = np.array(msg.data[2:2 + 4 * n], dtype=float).reshape(n, 4)
        return [(x, y, r, np.nan, legs) for x, y, r, legs in b]

    def _evaluate(self, name, t_capture, latency_ms, msg):
        src = self.sources[name]
        truth = self._truth_at(t_capture)
        if truth is EDGE:
            return
        src["frames"] += 1
        src["latency"].append(latency_ms)
        matched = False
        # Heights are compared top to top (the camera's only).
        for sx, sy, sr, sz, size in self._blobs(name, msg):
            if truth is not None and np.hypot(sx - truth[0], sy - truth[1]) < MATCH_RADIUS_M:
                matched = True
                e_xy = float(np.hypot(sx - truth[0], sy - truth[1]))
                e_z, e_r = float(sz - truth[4]), float(sr - truth[3])
                src["tp"] += 1
                src["err_xy"].append(e_xy)
                src["err_r"].append(e_r)
                if np.isfinite(e_z):
                    src["err_z"].append(e_z)
                kind = "tp"
                if self.verbose:
                    self.get_logger().info(
                        f"{name}: sensed ({sx:+.3f},{sy:+.3f}) top {sz:.3f} r={sr:.3f} | truth "
                        f"({truth[0]:+.3f},{truth[1]:+.3f}) top {truth[4]:.3f} r={truth[3]:.3f} | "
                        f"err xy={e_xy*1e3:.0f}mm top={e_z*1e3:+.0f}mm r={e_r*1e3:+.0f}mm | "
                        f"latency {latency_ms:.0f}ms")
            else:
                src["fp"] += 1
                kind = "fp"
                self.get_logger().warn(
                    f"{name}: FALSE POSITIVE blob at ({sx:+.3f},{sy:+.3f}) top {sz:.3f} r={sr:.3f} "
                    f"size={int(size)} (truth {'none active' if truth is None else 'elsewhere'})")
            if self.csv:
                t = truth if truth is not None else [np.nan] * 5
                self.csv.writerow([name, f"{t_capture:.3f}", f"{latency_ms:.1f}", sx, sy, sz, sr, int(size),
                                   t[0], t[1], t[4], t[3], np.hypot(sx - t[0], sy - t[1]), sz - t[4],
                                   sr - t[3], kind])
        if truth is not None and not matched:
            src["missed"] += 1  # may be out of view, masked or occluded
            if self.verbose:
                self.get_logger().info(
                    f"{name}: no blob for truth ({truth[0]:+.3f},{truth[1]:+.3f}) top {truth[4]:.2f}")
            if self.csv:
                self.csv.writerow([name, f"{t_capture:.3f}", f"{latency_ms:.1f}", *[np.nan] * 5,
                                   truth[0], truth[1], truth[4], truth[3], *[np.nan] * 3, "miss"])
        if src["frames"] % self.summary_every == 0:
            self._summary(name)

    def _summary(self, name):
        src = self.sources[name]

        def stat(v, scale=1e3):
            return f"mean {np.mean(np.abs(v))*scale:.0f} max {np.max(np.abs(v))*scale:.0f}" if v else "n/a"
        self.get_logger().info(
            f"SUMMARY {name} frames={src['frames']} true_pos={src['tp']} false_pos={src['fp']} "
            f"missed_while_truth_active={src['missed']} | err xy mm [{stat(src['err_xy'])}] "
            f"top mm [{stat(src['err_z'])}] r mm [{stat(src['err_r'])}] | "
            f"latency ms mean {np.mean(src['latency']):.0f} max {np.max(src['latency']):.0f}")


def main():
    rclpy.init()
    node = DetectionMonitorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        for name, src in node.sources.items():
            if src["frames"]:
                node._summary(name)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
