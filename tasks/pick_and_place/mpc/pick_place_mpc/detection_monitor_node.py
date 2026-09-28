"""Validation only: compares the workspace camera's detections with the
environment actor's ground truth and reports error, latency and false
positives. The one place truth and detections meet; nothing on the control
path subscribes to /env/dynamic_obstacle.

Truth is interpolated to each frame's capture time, so latency does not show
up as position error. One log line per blob, a summary every
`summary_every` frames, and optionally a CSV row per blob (`csv_path`).
"""
import bisect
import csv

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float64MultiArray

# A blob further than this (XY) from every truth is a false positive.
MATCH_RADIUS_M = 0.25
HISTORY_S = 5.0


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
            self.csv.writerow(["t_capture", "latency_ms", "sx", "sy", "s_top", "sr", "n_px",
                               "tx", "ty", "t_top", "tr", "err_xy", "err_top", "err_r", "kind"])

        self.truth_t, self.truth = [], []  # publish time, [x, y, z, r, top]
        self.pending = []  # frames waiting for truth newer than their capture
        self.create_subscription(Float64MultiArray, "/env/dynamic_obstacle", self._on_truth, 50)
        self.create_subscription(Float64MultiArray, "/env/workspace_detections", self._on_detections,
                                 qos_profile_sensor_data)

        self.frames = 0
        self.true_pos = self.false_pos = self.missed = 0
        self.err_xy, self.err_z, self.err_r, self.latency = [], [], [], []

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_truth(self, msg):
        # Keyed on the actor's publish stamp, not receipt time.
        t = msg.data[4]
        self.truth_t.append(t)
        d = msg.data
        # Tops, not centres: the overhead camera measures tops.
        top = d[5] if len(d) > 5 else d[2] + d[3]
        self.truth.append([d[0], d[1], d[2], d[3], top])
        while self.truth_t and self.truth_t[0] < t - HISTORY_S:
            self.truth_t.pop(0)
            self.truth.pop(0)
        self._drain()

    def _drain(self):
        # Evaluate a frame only once truth brackets its capture time.
        while self.pending and self.truth_t and self.truth_t[-1] >= self.pending[0][0]:
            self._evaluate(*self.pending.pop(0))

    def _truth_at(self, t):
        """Truth at time t, interpolated; None if none around then or absent."""
        if not self.truth_t:
            return None
        i = bisect.bisect_left(self.truth_t, t)
        if i == 0 or i == len(self.truth_t):
            return None
        t0, t1 = self.truth_t[i - 1], self.truth_t[i]
        a, b = np.array(self.truth[i - 1]), np.array(self.truth[i])
        if a[3] <= 0.0 or b[3] <= 0.0:
            return None
        w = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
        return a + (b - a) * w

    def _on_detections(self, msg):
        t_capture = msg.data[0]
        self.pending.append((t_capture, (self._now() - t_capture) * 1e3, msg))
        if self.count_publishers("/env/dynamic_obstacle") == 0:
            # No actor at all: every blob is a false positive.
            while self.pending:
                self._evaluate(*self.pending.pop(0))
            return
        self._drain()

    def _evaluate(self, t_capture, latency_ms, msg):
        n = int(msg.data[1])
        blobs = np.array(msg.data[2:], dtype=float).reshape(n, 6)
        truth = self._truth_at(t_capture)
        self.frames += 1
        self.latency.append(latency_ms)

        matched = False
        # Heights are compared top to top.
        for sx, sy, _sz, sr, n_px, sz in blobs:
            if truth is not None and np.hypot(sx - truth[0], sy - truth[1]) < MATCH_RADIUS_M:
                matched = True
                e_xy = float(np.hypot(sx - truth[0], sy - truth[1]))
                e_z, e_r = float(sz - truth[4]), float(sr - truth[3])
                self.true_pos += 1
                self.err_xy.append(e_xy); self.err_z.append(e_z); self.err_r.append(e_r)
                kind = "tp"
                if self.verbose:
                    self.get_logger().info(
                        f"sensed ({sx:+.3f},{sy:+.3f}) top {sz:.3f} r={sr:.3f} | truth "
                        f"({truth[0]:+.3f},{truth[1]:+.3f}) top {truth[4]:.3f} r={truth[3]:.3f} | "
                        f"err xy={e_xy*1e3:.0f}mm top={e_z*1e3:+.0f}mm r={e_r*1e3:+.0f}mm | "
                        f"latency {latency_ms:.0f}ms")
            else:
                self.false_pos += 1
                kind = "fp"
                self.get_logger().warn(
                    f"FALSE POSITIVE blob at ({sx:+.3f},{sy:+.3f}) top {sz:.3f} r={sr:.3f} "
                    f"px={int(n_px)} (truth {'none active' if truth is None else 'elsewhere'})")
            if self.csv:
                t = truth if truth is not None else [np.nan] * 5
                self.csv.writerow([f"{t_capture:.3f}", f"{latency_ms:.1f}", sx, sy, sz, sr, int(n_px),
                                   t[0], t[1], t[4], t[3], np.hypot(sx - t[0], sy - t[1]), sz - t[4],
                                   sr - t[3], kind])
        if truth is not None and not matched:
            self.missed += 1  # may be out of view or masked
            if self.verbose:
                self.get_logger().info(
                    f"no blob for truth ({truth[0]:+.3f},{truth[1]:+.3f}) top {truth[4]:.2f}")
        if self.frames % self.summary_every == 0:
            self._summary()

    def _summary(self):
        def stat(v, scale=1e3):
            return f"mean {np.mean(np.abs(v))*scale:.0f} max {np.max(np.abs(v))*scale:.0f}" if v else "n/a"
        self.get_logger().info(
            f"SUMMARY frames={self.frames} true_pos={self.true_pos} false_pos={self.false_pos} "
            f"missed_while_truth_active={self.missed} | err xy mm [{stat(self.err_xy)}] "
            f"top mm [{stat(self.err_z)}] r mm [{stat(self.err_r)}] | "
            f"latency ms mean {np.mean(self.latency):.0f} max {np.max(self.latency):.0f}")


def main():
    rclpy.init()
    node = DetectionMonitorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.frames:
            node._summary()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
