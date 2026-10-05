"""Validation only (like detection_monitor_node): the robot's pose in the map, as the
method has it (the map > odom transform of whichever SLAM runs, times the odometry),
against the true pose (/sim/base_truth). The map frame is anchored where its map was
started, the dock (BASE_PARK_POSE in the cell layout, BASE_HOME_POSE in the
stations layout; parameter map_origin overrides).
Odometry and truth are paired by the plant's step (/sim/wheel_states carries it).
Also scores both of slam_node's estimators (/slam/estimates: the scan matcher and the
filter, with the filter's NEES; pick_place_common/loc_scoring.py).
Logs a summary every `summary_s` and a CSV row per tick (MANIPOPT_TELEMETRY_DIR).
"""
import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from tf2_msgs.msg import TFMessage

from pick_place_common.loc_scoring import LocScore, line as loc_line
from pick_place_common.scene import BASE_HOME_POSE, BASE_PARK_POSE
from pick_place_common.telemetry import Telemetry
from slam.pose_graph import compose, relative, wrap


class LocalizationMonitorNode(Node):
    def __init__(self):
        super().__init__("localization_monitor_node")
        self.declare_parameter("layout", "stations")
        self.declare_parameter("map_origin", [float("nan")] * 3)
        self.declare_parameter("summary_s", 10.0)
        origin = np.array(self.get_parameter("map_origin").value, dtype=float)
        dock = BASE_PARK_POSE if self.get_parameter("layout").value == "cell" else BASE_HOME_POSE
        self.origin = origin if np.all(np.isfinite(origin)) else np.array(dock, dtype=float)
        self.summary_s = float(self.get_parameter("summary_s").value)
        self.map_to_odom = None
        self.stamp_step = {}  # wheel-state stamp (ns) -> plant step
        self.odom = {}  # step -> odom pose
        self.truth = {}  # step -> true pose in the map frame
        self.err, self.yaw_err = [], []
        self.all_err = []
        self.estimates = {}  # step -> /slam/estimates
        self.scores = {"icp": LocScore(), "ekf": LocScore()}
        self.telemetry = Telemetry("localization", ["step", "est_x", "est_y", "est_yaw", "true_x", "true_y",
                                                    "true_yaw", "err_m", "err_deg"])
        self.est_telemetry = Telemetry("estimates", ["step", "true_x", "true_y", "true_yaw", "icp_x", "icp_y", "icp_yaw",
                                                     "ekf_x", "ekf_y", "ekf_yaw", "ekf_sxx", "ekf_sxy", "ekf_sxyaw",
                                                     "ekf_syy", "ekf_syyaw", "ekf_syawyaw"])
        self.create_subscription(TFMessage, "/tf", self._on_tf, 50)
        self.create_subscription(JointState, "/sim/wheel_states", self._on_wheels, 50)
        self.create_subscription(Odometry, "/odom", self._on_odom, 50)
        self.create_subscription(Float64MultiArray, "/sim/base_truth", self._on_truth, 50)
        self.create_subscription(Float64MultiArray, "/slam/estimates", self._on_estimates, 50)
        self.create_timer(self.summary_s, self._summary)

    def _on_tf(self, msg):
        for tr in msg.transforms:
            if tr.header.frame_id == "map" and tr.child_frame_id == "odom":
                q = tr.transform.rotation
                self.map_to_odom = np.array([tr.transform.translation.x, tr.transform.translation.y,
                                             2 * np.arctan2(q.z, q.w)])

    @staticmethod
    def _ns(stamp):
        return stamp.sec * 1_000_000_000 + stamp.nanosec

    def _on_wheels(self, msg):
        self.stamp_step[self._ns(msg.header.stamp)] = int(msg.header.frame_id)
        if len(self.stamp_step) > 500:
            self.stamp_step.pop(next(iter(self.stamp_step)))

    def _on_odom(self, msg):
        step = self.stamp_step.get(self._ns(msg.header.stamp))
        if step is None:
            return
        q = msg.pose.pose.orientation
        self.odom[step] = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, 2 * np.arctan2(q.z, q.w)])
        self._match(step)

    def _on_truth(self, msg):
        step = int(msg.data[0])
        x, y, _z, qw, _qx, _qy, qz = msg.data[1:8]
        self.truth[step] = relative(self.origin, (x, y, 2 * np.arctan2(qz, qw)))
        self._match_estimates(step)
        self._match(step)

    def _on_estimates(self, msg):
        step = int(msg.data[0])
        self.estimates[step] = np.array(msg.data[1:])
        self._match_estimates(step)
        for k in [k for k in self.estimates if k < step - 100]:
            del self.estimates[k]

    def _match_estimates(self, step):
        if step not in self.estimates or step not in self.truth:
            return
        e, true = self.estimates.pop(step), self.truth[step]
        c = e[6:12]
        cov = np.array([[c[0], c[1], c[2]], [c[1], c[3], c[4]], [c[2], c[4], c[5]]])
        self.scores["icp"].add(e[0:3], true)
        self.scores["ekf"].add(e[3:6], true, cov)
        self.est_telemetry.row(step, *np.round(true, 5), *np.round(e[:6], 5), *c)

    def _match(self, step):
        if step not in self.odom or step not in self.truth or self.map_to_odom is None:
            return
        est = compose(self.map_to_odom, self.odom.pop(step))
        true = self.truth[step]
        e = float(np.hypot(*(est[:2] - true[:2])))
        ey = float(np.degrees(abs(wrap(est[2] - true[2]))))
        self.err.append(e)
        self.yaw_err.append(ey)
        self.all_err.append((e, ey))
        self.telemetry.row(step, *np.round(est, 5), *np.round(true, 5), round(e, 5), round(ey, 4))
        for d in (self.odom, self.truth):
            for k in [k for k in d if k < step - 100]:
                del d[k]

    def _summary(self):
        if not self.err:
            return
        e, ey = np.array(self.err), np.array(self.yaw_err)
        self.get_logger().info(f"localization: position error p50/p95/max {1e3 * np.median(e):.0f}/"
                               f"{1e3 * np.percentile(e, 95):.0f}/{1e3 * e.max():.0f} mm, yaw p50/max "
                               f"{np.median(ey):.2f}/{ey.max():.2f} deg over the last {self.summary_s:.0f} s")
        self.err, self.yaw_err = [], []

    def final(self):
        for name, sc in self.scores.items():
            if sc.rows:
                self.get_logger().info(f"localization over the run, {name}: {loc_line(sc.summary())}")
        if self.all_err:
            a = np.array(self.all_err)
            self.get_logger().info(f"localization over the run: position error p50/p95/max {1e3 * np.median(a[:, 0]):.0f}/"
                                   f"{1e3 * np.percentile(a[:, 0], 95):.0f}/{1e3 * a[:, 0].max():.0f} mm, yaw "
                                   f"p50/max {np.median(a[:, 1]):.2f}/{a[:, 1].max():.2f} deg ({len(a)} ticks)")


def main():
    rclpy.init()
    node = LocalizationMonitorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.final()
        node.telemetry.flush()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
