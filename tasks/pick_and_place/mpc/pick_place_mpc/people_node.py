"""People on the moving base (perception/map_people.py, people_tracker.py): each round
of lidar scans (/env/lidar_scan) at the robot's pose in the map (the SLAM's map > odom
from /tf times the odometry at the scan's time, so it works with either SLAM option)
against the saved map; tracks in the map frame.

Publishes /perception/people [t, n, (id, x, y, vx, vy, radius, standing) * n] (map
frame, confirmed tracks), /perception/people_foreground [t, n, (x, y) * n] and the same
points in the base frame of the scan (/perception/foreground_base, for the base's
safety layer, which keeps people's legs farther). With
arm_detections:=true also each scan's people in the arm frame, for the arm's
obstacle supervisor (/perception/lidar_detections [t, n, (x, y, radius, n_legs) * n],
as lidar_detection_node gives them on a parked base), and the arm frame's pose in the
odometry frame at the scan's time (/perception/arm_in_odom [t, x, y, yaw]: the
supervisor tracks people there, still while the base moves).
Parameters: map, map_dir (the map the robot localizes in), arm_detections.
See docs/implementation_notes.md#map_peoplepy.
"""
import os
from collections import deque
from pathlib import Path

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float64MultiArray
from tf2_msgs.msg import TFMessage

from perception.map_people import MapPeopleDetector
from perception.people_tracker import PeopleTracker
from pick_place_common.scene import ARM_IN_BASE, LIDAR_MOUNTS_BASE
from pick_place_common.telemetry import Telemetry
from pick_place_mpc.scan_merger_node import ScanPairs
from pick_place_mpc.slam_node import DEFAULT_MAP_DIR
from slam.grid import OccupancyGrid
from slam.pose_graph import compose, relative, wrap


class PeopleNode(Node):
    def __init__(self):
        super().__init__("people_node")
        self.declare_parameter("map", "room")
        self.declare_parameter("map_dir", os.environ.get("MANIPOPT_MAP_DIR", DEFAULT_MAP_DIR))
        self.declare_parameter("arm_detections", False)
        path = Path(self.get_parameter("map_dir").value) / self.get_parameter("map").value
        self.map_points = OccupancyGrid.load(path.with_suffix(".yaml")).occupied_points()
        self.detector = None
        self.tracker = PeopleTracker()
        self.pairs = ScanPairs()
        self.odom = deque(maxlen=200)
        self.map_to_odom = None
        self.pending = None
        self.telemetry = Telemetry("people", ["t", "n", "tracks"])
        self.pub = self.create_publisher(Float64MultiArray, "/perception/people", qos_profile_sensor_data)
        self.fg_pub = self.create_publisher(Float64MultiArray, "/perception/people_foreground", qos_profile_sensor_data)
        self.fg_base_pub = self.create_publisher(Float64MultiArray, "/perception/foreground_base",
                                                 qos_profile_sensor_data)
        self.arm_pub = None
        if self.get_parameter("arm_detections").value:
            self.arm_pub = self.create_publisher(Float64MultiArray, "/perception/lidar_detections",
                                                 qos_profile_sensor_data)
            self.arm_odom_pub = self.create_publisher(Float64MultiArray, "/perception/arm_in_odom",
                                                      qos_profile_sensor_data)
        self.create_subscription(TFMessage, "/tf", self._on_tf, 50)
        self.create_subscription(Odometry, "/odom", self._on_odom, 50)
        self.create_subscription(Float64MultiArray, "/env/lidar_scan", self._on_scan, 10)
        self.get_logger().info(f"people against the map {path} ({len(self.map_points)} surface cells)")

    def _on_tf(self, msg):
        for tr in msg.transforms:
            if tr.header.frame_id == "map" and tr.child_frame_id == "odom":
                q = tr.transform.rotation
                self.map_to_odom = np.array([tr.transform.translation.x, tr.transform.translation.y,
                                             2 * np.arctan2(q.z, q.w)])

    def _on_odom(self, msg):
        q = msg.pose.pose.orientation
        self.odom.append((msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9, msg.pose.pose.position.x,
                          msg.pose.pose.position.y, 2 * np.arctan2(q.z, q.w)))
        if self.pending is not None and self.odom[-1][0] >= self.pending[0] - 1e-6:
            pending, self.pending = self.pending, None
            self._process(*pending)

    def _odom_at(self, t):
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
        if self.odom and self.odom[-1][0] < got[0] - 1e-6:
            self.pending = got
            return
        self._process(*got)

    def _process(self, t, angles, ranges):
        odom = self._odom_at(t)
        if odom is None or self.map_to_odom is None:
            return
        if self.detector is None:
            self.detector = MapPeopleDetector(self.map_points, LIDAR_MOUNTS_BASE, angles)
        pose = compose(self.map_to_odom, odom)
        people = self.detector.update(pose, ranges)
        tracks = self.tracker.update(t, people, self.detector.seen_empty)
        if self.arm_pub is not None:
            self.arm_odom_pub.publish(Float64MultiArray(data=[t, *map(float, compose(odom, ARM_IN_BASE))]))
            arm = compose(pose, ARM_IN_BASE)
            rows = [(*relative(arm, (x, y, 0.0))[:2], r, k) for x, y, r, k in people]
            self.arm_pub.publish(Float64MultiArray(data=[t, float(len(rows)), *[float(v) for r in rows for v in r]]))
        rows = [(tr.id, *tr.x, tr.radius, float(tr.standing)) for tr in tracks]
        self.pub.publish(Float64MultiArray(data=[t, float(len(rows)), *[float(v) for r in rows for v in r]]))
        fg = self.detector.foreground
        self.fg_pub.publish(Float64MultiArray(data=[t, float(len(fg)), *fg.ravel().tolist()]))
        c, s = np.cos(pose[2]), np.sin(pose[2])
        rel = fg - pose[:2]
        fg_base = np.column_stack([c * rel[:, 0] + s * rel[:, 1], -s * rel[:, 0] + c * rel[:, 1]])
        self.fg_base_pub.publish(Float64MultiArray(data=[t, float(len(fg_base)), *fg_base.ravel().tolist()]))
        self.telemetry.row(round(t, 4), len(rows), " ".join(f"{i}:{x:.3f}:{y:.3f}:{vx:.2f}:{vy:.2f}"
                                                            for i, x, y, vx, vy, *_ in rows))


def main():
    rclpy.init()
    node = PeopleNode()
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
