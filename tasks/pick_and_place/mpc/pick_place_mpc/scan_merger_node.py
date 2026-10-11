"""The two safety lidars' raw scans (/env/lidar_scan, one message per scanner per
round) merged into one 360 deg scan in base_link (frame base_scan) with their taught
calibration (scene.LIDAR_MOUNTS_BASE), as /scan (sensor_msgs/LaserScan), stamped at
the round's capture time: the scan slam_toolbox reads (slam/scan.py).
"""
import numpy as np
import rclpy
from builtin_interfaces.msg import Time
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64MultiArray

from pick_place_common.scene import LIDAR_MOUNTS_BASE
from slam.scan import merge_scan, scan_points

HEADER = 9  # t, index, x, y, z, yaw, angle_min, angle_step, n
N_BEAMS = 720


def unpack(d):
    """(t, scanner index, beam angles, ranges) from one /env/lidar_scan message."""
    t, index, n = d[0], int(d[1]), int(d[8])
    return t, index, d[6] + d[7] * np.arange(n), np.array(d[HEADER:HEADER + n])


class ScanPairs:
    """Collects one scan per scanner of the same capture time."""

    def __init__(self, n=len(LIDAR_MOUNTS_BASE)):
        self.n = n
        self.t = None
        self.ranges = {}

    def add(self, d):
        """(t, angles, [ranges per scanner]) once a round is complete, else None."""
        t, index, angles, r = unpack(d)
        if t != self.t:
            self.t, self.ranges = t, {}
        self.ranges[index] = r
        if len(self.ranges) == self.n:
            return t, angles, [self.ranges[i] for i in range(self.n)]
        return None


class ScanMergerNode(Node):
    def __init__(self):
        super().__init__("scan_merger_node")
        self.pairs = ScanPairs()
        self.pub = self.create_publisher(LaserScan, "/scan", qos_profile_sensor_data)
        self.create_subscription(Float64MultiArray, "/env/lidar_scan", self._on_scan, 10)

    def _on_scan(self, msg):
        got = self.pairs.add(msg.data)
        if got is None:
            return
        t, angles, ranges = got
        r = merge_scan(scan_points(ranges, angles, LIDAR_MOUNTS_BASE), N_BEAMS)
        out = LaserScan()
        out.header.stamp = Time(sec=int(t), nanosec=int((t % 1.0) * 1e9))
        out.header.frame_id = "base_scan"
        out.angle_min, out.angle_increment = -np.pi, 2 * np.pi / N_BEAMS
        out.angle_max = out.angle_min + (N_BEAMS - 1) * out.angle_increment
        out.range_min, out.range_max = 0.05, 10.0
        out.ranges = np.where(np.isfinite(r), r, np.inf).astype(np.float32).tolist()
        self.pub.publish(out)


def main():
    rclpy.init()
    node = ScanMergerNode()
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
