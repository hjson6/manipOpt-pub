"""Our own SLAM on the live robot (slam/): mapping (graph SLAM, the map saved when the
node stops) or localization in a saved map. Takes both lidars' raw scans
(/env/lidar_scan) with their taught calibration and the odometry (/odom, interpolated
to each scan's capture time); publishes the map > odom transform (/tf), the pose
(/slam/pose) and the map (/map; in mapping, each keyframe drawn at its pose then,
for display; the saved map is rebuilt at the optimized poses, on /slam/save_map or
when the node stops).

In localization both estimators run on the same data (shadow): the scan matcher alone
(icp, slam/localizer.py) and the filter (ekf, slam/base_ekf.py: the wheels and the gyro,
/sim/wheel_states and /sim/imu, at every plant step, the scan matches weighted); fusion
picks the one whose map > odom goes out. Also publishes the filter's pose and covariance
(/slam/covariance, every step) and both estimates for validation (/slam/estimates, every
step: [step, icp x y yaw, ekf x y yaw, covariance xx xy xyaw yy yyaw yawyaw]).

Parameters: mode (mapping | localization), map (name in map_dir), map_dir,
initial_pose [x, y, yaw] for localization (the dock, the map's origin, by default),
fusion (icp | ekf).
See docs/implementation_notes.md#slam.
"""
import os
from collections import deque
from pathlib import Path

import numpy as np
import rclpy
from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, TransformStamped
from nav_msgs.msg import OccupancyGrid as GridMsg
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger
from tf2_ros import TransformBroadcaster

from pick_place_common.scene import BASE_WHEEL_RADIUS_M, BASE_WHEEL_TRACK_M, LIDAR_MOUNTS_BASE
from pick_place_common.telemetry import Telemetry
from pick_place_mpc.scan_merger_node import ScanPairs
from slam.base_ekf import BaseEKF
from slam.grid import OccupancyGrid
from slam.localizer import GridLocalizer
from slam.mapper import GraphSlam
from slam.pose_graph import compose, relative, wrap
from slam.scan import scan_points, transform

DEFAULT_MAP_DIR = str(Path(__file__).resolve().parents[4] / "data" / "maps")
TF_PERIOD_S = 0.05
CONTROL_PERIOD_S = 0.02  # the plant's step
MAP_PUBLISH_PERIOD_S = 2.0
ODOM_HISTORY = 200


def to_time(t):
    return Time(sec=int(t), nanosec=int((t % 1.0) * 1e9))


class SlamNode(Node):
    def __init__(self):
        super().__init__("slam_node")
        self.declare_parameter("mode", "mapping")
        self.declare_parameter("map", "room")
        self.declare_parameter("map_dir", os.environ.get("MANIPOPT_MAP_DIR", DEFAULT_MAP_DIR))
        self.declare_parameter("initial_pose", [0.0, 0.0, 0.0])
        self.declare_parameter("fusion", "ekf")
        self.mode = self.get_parameter("mode").value
        self.fusion = self.get_parameter("fusion").value
        if self.fusion not in ("icp", "ekf"):
            raise ValueError(f"fusion must be icp or ekf, got {self.fusion!r}")
        self.map_path = Path(self.get_parameter("map_dir").value) / self.get_parameter("map").value
        self.odom = deque(maxlen=ODOM_HISTORY)  # (t, x, y, yaw)
        self.pairs = ScanPairs()
        self.map_to_odom = np.zeros(3)
        self.scans = 0
        self.pending = None  # a scan waiting for the odometry (and the filter) of its tick
        self.ekf = None
        if self.mode == "mapping":
            self.slam = GraphSlam()
            self.display = None
            self.drawn = 0
        else:
            grid = OccupancyGrid.load(self.map_path.with_suffix(".yaml"))
            start = self.get_parameter("initial_pose").value
            self.localizer = GridLocalizer(grid, start)
            self.ekf = BaseEKF(start, BASE_WHEEL_RADIUS_M, BASE_WHEEL_TRACK_M)
            self.icp_map_to_odom = np.zeros(3)
            self.samples = {}  # plant step -> {"wheels", "gyro", "t"}
            self.ticks = deque(maxlen=ODOM_HISTORY)  # (t, step) the filter has had
            self.flushed_t = -np.inf
            self.slips_seen, self.slipping = 0, False
            self.display = grid
            self.get_logger().info(f"localizing in {self.map_path} from {start}; {self.fusion} drives, both run")
        self.telemetry = Telemetry("slam", ["t", "x", "y", "yaw", "quality", "ms", "ekf_x", "ekf_y", "ekf_yaw",
                                            "ekf_status", "ekf_d2", "ekf_ms", "lag_ms"])
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        self.map_pub = self.create_publisher(GridMsg, "/map", latched)
        self.pose_pub = self.create_publisher(PoseStamped, "/slam/pose", 10)
        self.tf = TransformBroadcaster(self)
        if self.ekf is not None:
            self.cov_pub = self.create_publisher(PoseWithCovarianceStamped, "/slam/covariance", 10)
            self.est_pub = self.create_publisher(Float64MultiArray, "/slam/estimates", 10)
            self.create_subscription(JointState, "/sim/wheel_states", self._on_wheels, 50)
            self.create_subscription(Imu, "/sim/imu", self._on_imu, 50)
        self.create_subscription(Odometry, "/odom", self._on_odom, 50)
        self.create_subscription(Float64MultiArray, "/env/lidar_scan", self._on_scan, 10)
        self.create_timer(TF_PERIOD_S, self._publish_tf)
        self.create_timer(MAP_PUBLISH_PERIOD_S, self._publish_map)
        self.saved = False
        self.create_service(Trigger, "/slam/save_map", self._on_save)
        if self.display is not None:
            self._publish_map()

    def _on_odom(self, msg):
        q = msg.pose.pose.orientation
        self.odom.append((msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9, msg.pose.pose.position.x,
                          msg.pose.pose.position.y, 2 * np.arctan2(q.z, q.w)))
        self._flush()
        self._try_pending()

    def _ready(self, t):
        """The odometry, and the filter, have reached time t."""
        return bool(self.odom) and self.odom[-1][0] >= t - 1e-6 and (self.ekf is None or self.ekf.t is not None
                                                                         and self.ekf.t >= t - 1e-6)

    def _try_pending(self):
        if self.pending is not None and self._ready(self.pending[0]):
            pending, self.pending = self.pending, None
            self._process(*pending)

    def _on_wheels(self, msg):
        self._sample(int(msg.header.frame_id), "wheels", np.array(msg.position), msg.header.stamp)

    def _on_imu(self, msg):
        self._sample(int(msg.header.frame_id), "gyro", msg.angular_velocity.z, msg.header.stamp)

    def _sample(self, step, key, value, stamp):
        """The filter's tick once both of a step's samples are in (as base_node pairs them)."""
        s = self.samples.setdefault(step, {})
        s[key] = value
        s["t"] = stamp.sec + stamp.nanosec * 1e-9
        for old in [k for k in self.samples if k < step - 5]:
            del self.samples[old]
        if "wheels" not in s or "gyro" not in s:
            return
        del self.samples[step]
        if self.ticks and step <= self.ticks[-1][1]:
            return
        dt = (step - self.ticks[-1][1]) * CONTROL_PERIOD_S if self.ticks else None  # stamps are wall time
        self.ekf.tick(s["t"], s["wheels"], s["gyro"], dt)
        self.ticks.append((s["t"], step))
        n = self.ekf.counts["slips"]
        if n > self.slips_seen and not self.slipping:  # an episode starts
            self.get_logger().info(f"ekf: wheel slip ({n} ticks, {self.ekf.counts['skids']} skids so far)")
        self.slipping, self.slips_seen = n > self.slips_seen, n
        self._flush()
        self._try_pending()

    def _flush(self):
        """Every step both the odometry and the filter have: both estimates out, and in
        ekf, the map > odom transform from the filter."""
        if self.ekf is None:
            return
        new = []
        for t, step in reversed(self.ticks):
            if t <= self.flushed_t + 1e-6:
                break
            new.append((t, step))
        for t, step in reversed(new):
            odom = self._odom_at(t)
            ekf = self.ekf.pose_at(t)
            if odom is None or ekf is None:
                return
            self.flushed_t = t
            icp = compose(self.icp_map_to_odom, odom)
            if self.fusion == "ekf":
                self.map_to_odom = compose(ekf, relative(odom, np.zeros(3)))
            c = self.ekf.cov
            self.est_pub.publish(Float64MultiArray(data=[float(step), *icp, *ekf, c[0, 0], c[0, 1], c[0, 2], c[1, 1],
                                                         c[1, 2], c[2, 2]]))
            msg = PoseWithCovarianceStamped()
            msg.header.stamp = to_time(t)
            msg.header.frame_id = "map"
            msg.pose.pose.position.x, msg.pose.pose.position.y = float(ekf[0]), float(ekf[1])
            msg.pose.pose.orientation.z, msg.pose.pose.orientation.w = float(np.sin(ekf[2] / 2)), float(np.cos(ekf[2] / 2))
            cov = np.zeros((6, 6))
            cov[np.ix_([0, 1, 5], [0, 1, 5])] = c
            msg.pose.covariance = cov.ravel().tolist()
            self.cov_pub.publish(msg)

    def _odom_at(self, t):
        """The odometry pose at time t, interpolated (None if t is outside the history)."""
        if len(self.odom) < 2 or not self.odom[0][0] <= t <= self.odom[-1][0] + 1e-6:
            return None
        h = np.array(self.odom)
        i = int(np.clip(np.searchsorted(h[:, 0], t), 1, len(h) - 1))
        a, b = h[i - 1], h[i]
        f = float(np.clip((t - a[0]) / max(b[0] - a[0], 1e-9), 0.0, 1.0))
        return np.array([a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2]), a[3] + f * wrap(b[3] - a[3])])

    def _on_scan(self, msg):
        got = self.pairs.add(msg.data)
        if got is None:
            return
        if not self._ready(got[0]):
            self.pending = got  # its tick's odometry is still on the way
            return
        self._process(*got)

    def _process(self, t, angles, ranges):
        odom = self._odom_at(t)
        if odom is None:
            return
        t0 = self.get_clock().now().nanoseconds
        pts, origins = scan_points(ranges, angles, LIDAR_MOUNTS_BASE, origins=True)
        ekf_row = [np.nan] * 6
        if self.mode == "mapping":
            pose = self.slam.update(odom, pts, origins)
            quality = float(len(self.slam.loops))
            self.map_to_odom = compose(pose, relative(odom, np.zeros(3)))
        else:
            pose = self.localizer.update(odom, pts)
            quality = self.localizer.quality
            self.icp_map_to_odom = compose(pose, relative(odom, np.zeros(3)))
            t1 = self.get_clock().now().nanoseconds
            status = self.ekf.scan(t, lambda prior: self.localizer.match(pts, prior))
            ekf_ms = (self.get_clock().now().nanoseconds - t1) * 1e-6
            if status in ("reset", "late"):
                self.get_logger().info(f"ekf: scan {status} ({self.ekf.counts})")
            ekf_pose = self.ekf.pose_at(t)
            ekf_row = [*np.round(ekf_pose if ekf_pose is not None else [np.nan] * 3, 5), status,
                       round(float(self.ekf.last[2]), 2) if self.ekf.last else np.nan, round(ekf_ms, 2)]
            if self.fusion == "icp":
                self.map_to_odom = self.icp_map_to_odom
            elif ekf_pose is not None:
                pose = ekf_pose
            self._flush()
        ms = (self.get_clock().now().nanoseconds - t0) * 1e-6
        self.scans += 1
        lag_ms = (self.get_clock().now().nanoseconds * 1e-9 - t) * 1e3
        self.telemetry.row(round(t, 4), *np.round(pose, 5), round(quality, 3), round(ms, 2), *ekf_row, round(lag_ms, 1))
        p = PoseStamped()
        p.header.stamp = to_time(t)
        p.header.frame_id = "map"
        p.pose.position.x, p.pose.position.y = float(pose[0]), float(pose[1])
        p.pose.orientation.z, p.pose.orientation.w = float(np.sin(pose[2] / 2)), float(np.cos(pose[2] / 2))
        self.pose_pub.publish(p)

    def _publish_tf(self):
        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = "map"
        tf.child_frame_id = "odom"
        tf.transform.translation.x, tf.transform.translation.y = float(self.map_to_odom[0]), float(self.map_to_odom[1])
        tf.transform.rotation.z = float(np.sin(self.map_to_odom[2] / 2))
        tf.transform.rotation.w = float(np.cos(self.map_to_odom[2] / 2))
        self.tf.sendTransform(tf)

    def _publish_map(self):
        if self.mode == "mapping":
            kfs = self.slam.keyframes
            if not kfs:
                return
            if self.display is None:
                self.display = OccupancyGrid((-12.0, -12.0), (480, 480), 0.05)
            for k in kfs[self.drawn:]:
                pose = self.slam.graph.nodes[k["node"]]
                self.display.integrate(pose, k["full"], k["origins"])
            self.drawn = len(kfs)
        g = self.display
        msg = GridMsg()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.info.resolution = g.res
        msg.info.height, msg.info.width = g.log_odds.shape
        msg.info.origin.position.x, msg.info.origin.position.y = float(g.origin[0]), float(g.origin[1])
        msg.info.origin.orientation.w = 1.0
        data = np.full(g.log_odds.shape, -1, np.int8)
        data[g.free] = 0
        data[g.occupied] = 100
        msg.data = data.ravel().tolist()
        self.map_pub.publish(msg)

    def _on_save(self, _req, res):
        self.save()
        res.success, res.message = self.saved, str(self.map_path)
        return res

    def save(self):
        """Mapping: the map at the optimized keyframe poses, as map_dir/<map>.{pgm,yaml,npz}."""
        if self.mode != "mapping" or not self.slam.keyframes or self.saved:
            return
        self.map_path.parent.mkdir(parents=True, exist_ok=True)
        self.slam.map().save(self.map_path)
        np.save(self.map_path.with_name(self.map_path.name + "_keyframes.npy"), self.slam.keyframe_poses())
        self.saved = True
        self.get_logger().info(f"map saved: {self.map_path} ({len(self.slam.keyframes)} keyframes, "
                               f"{len(self.slam.loops)} loop closures, {self.scans} scans)")


def main():
    rclpy.init()
    node = SlamNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save()
        node.telemetry.flush()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
