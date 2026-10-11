"""People from the safety lidars' raw scans (perception/lidar_detection.py), with
the scanners' calibrated poses (scene.py). Learns the background from the first
scans: start it with the cell empty.

Publishes /perception/lidar_detections [t_capture_s, n, (x, y, radius, n_legs) * n]
(base frame) and, for the obstacle window, /perception/lidar_foreground
[t_capture_s, n, (x, y) * n].
"""
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float64MultiArray

from perception.lidar_detection import LidarPeopleDetector
from pick_place_common.scene import LIDAR_SCANNERS

HEADER = 9  # t, index, x, y, z, yaw, angle_min, angle_step, n


class LidarDetectionNode(Node):
    def __init__(self):
        super().__init__("lidar_detection_node")
        self.detector = None
        self.people_pub = self.create_publisher(Float64MultiArray, "/perception/lidar_detections",
                                                qos_profile_sensor_data)
        self.fg_pub = self.create_publisher(Float64MultiArray, "/perception/lidar_foreground",
                                            qos_profile_sensor_data)
        self.create_subscription(Float64MultiArray, "/env/lidar_scan", self._on_scan, 10)

    def _on_scan(self, msg):
        d = msg.data
        t, index, n = d[0], int(d[1]), int(d[8])
        if self.detector is None:
            angles = d[6] + d[7] * np.arange(n)
            self.detector = LidarPeopleDetector(LIDAR_SCANNERS, angles)
        was_ready = self.detector.ready
        self.detector.update(index, np.array(d[HEADER:HEADER + n]))
        if not self.detector.ready:
            return
        if not was_ready:
            self.get_logger().info("lidar background learnt; detecting people")
        if index != len(LIDAR_SCANNERS) - 1:
            return  # once per round of scans
        people = self.detector.people()
        self.people_pub.publish(Float64MultiArray(
            data=[t, float(len(people)), *[v for p in people for v in p]]))
        fg = np.vstack(self.detector.foreground)
        self.fg_pub.publish(Float64MultiArray(data=[t, float(len(fg)), *fg.ravel().tolist()]))


def main():
    rclpy.init()
    node = LidarDetectionNode()
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
