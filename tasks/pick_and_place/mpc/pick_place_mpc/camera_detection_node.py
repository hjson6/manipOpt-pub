"""People and objects from the ceiling camera's raw depth (perception/
obstacle_detection.py), with the camera's calibrated pose (scene.py).

The robot is masked kinematically: the method's own robot model at the measured
joint angles nearest the frame's capture (robot_geometry.py), and the held box
from its sensed size and in-hand offset (/task/held_box), projected into the
camera. Ignored as before: the pick zone up to just above the last pile scan
(/task/pile_top), the tray, the furniture.

Publishes /perception/camera_detections [t_capture_s, n, (x, y, z, radius, n_px,
z_max) * n] and, for the obstacle window, /perception/camera_debug [t_capture_s,
w, h, class per pixel (0 none, 1 ignored, 2 robot, 3 foreground)].
"""
from array import array
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, Float64MultiArray, UInt8MultiArray

from perception import obstacle_detection
from pick_place_common.scene import (
    AISLE_MASK_MARGIN_M, AISLE_MASK_PILE_ABOVE_M, AISLE_MASK_TRAY, DETECTION_MIN_BLOB_PX, FOREGROUND_MIN_HEIGHT_M,
    FURNITURE_MASKS, PICK_ZONE_MAX_HEIGHT_M, ROOM_FLOOR_Z, SOURCE_SCAN_BOUNDS, WORKSPACE_CAM_FOVY_DEG,
    WORKSPACE_CAM_POS, WORKSPACE_CAM_RATE_HZ)
from pick_place_mpc.robot_geometry import RobotGeometry

CAM_MAT = np.eye(3)  # straight down, xyaxes="1 0 0 0 1 0"
ROBOT_DILATE_PX = 3  # ~6 cm on the floor: joint noise and the state's age at capture
STATE_HISTORY = 50
DEBUG_HZ = 2.0


class CameraDetectionNode(Node):
    def __init__(self):
        super().__init__("camera_detection_node")
        self.robot = RobotGeometry()
        self.states = deque(maxlen=STATE_HISTORY)  # (stamp_s, q)
        self.pile_top = PICK_ZONE_MAX_HEIGHT_M
        self.held = None  # (offset, axes, hx, hy, height) in the TCP frame
        self.frames = 0
        self.create_subscription(JointState, "/sim/joint_states", self._on_state, qos_profile_sensor_data)
        self.create_subscription(Float64, "/task/pile_top", self._on_pile_top, 10)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Float64MultiArray, "/task/held_box", self._on_held_box, latched)
        # Newest frame only, but reliable: best effort lost most of these 0.6 MB frames.
        self.create_subscription(Float64MultiArray, "/env/workspace_depth", self._on_depth, 1)
        self.pub = self.create_publisher(Float64MultiArray, "/perception/camera_detections", qos_profile_sensor_data)
        self.debug_pub = self.create_publisher(UInt8MultiArray, "/perception/camera_debug", 1)

    def _on_state(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.states.append((t, np.array(msg.position[:7])))

    def _on_pile_top(self, msg):
        self.pile_top = float(msg.data)

    def _on_held_box(self, msg):
        d = np.array(msg.data)
        self.held = None if d[0] < 0.5 else (d[1:4], d[4:13].reshape(3, 3), d[13], d[14], d[15])

    def _robot_mask(self, t_capture, w, h):
        _t, q = min(self.states, key=lambda s: abs(s[0] - t_capture))
        self.robot.update(q)
        sets = self.robot.point_sets()
        if self.held is not None:
            offset, axes, hx, hy, height = self.held
            tcp, rot = self.robot.tcp()
            corners = [offset + axes @ np.array([sx * hx, sy * hy, 0.0]) for sx in (-1, 1) for sy in (-1, 1)]
            corners += [c + np.array([0.0, 0.0, height]) for c in corners]  # tool z points down
            sets.append(np.array([tcp + rot @ c for c in corners]))
        return obstacle_detection.convex_silhouette(sets, WORKSPACE_CAM_POS, CAM_MAT, WORKSPACE_CAM_FOVY_DEG, w, h)

    def _on_depth(self, msg):
        if not self.states:
            return
        d = np.asarray(msg.data, dtype=np.float64)
        t_capture, w, h = float(d[0]), int(d[1]), int(d[2])
        depth = d[3:3 + w * h].reshape(h, w)
        self.frames += 1
        debug = self.frames % max(1, round(WORKSPACE_CAM_RATE_HZ / DEBUG_HZ)) == 0
        result = obstacle_detection.detect_blobs(
            depth, WORKSPACE_CAM_POS, CAM_MAT, WORKSPACE_CAM_FOVY_DEG, ROOM_FLOOR_Z, FOREGROUND_MIN_HEIGHT_M,
            robot_mask=self._robot_mask(t_capture, w, h), robot_dilate_px=ROBOT_DILATE_PX,
            mask_boxes=((*SOURCE_SCAN_BOUNDS, self.pile_top + AISLE_MASK_PILE_ABOVE_M), AISLE_MASK_TRAY,
                        *FURNITURE_MASKS),
            mask_margin=AISLE_MASK_MARGIN_M, min_blob_px=DETECTION_MIN_BLOB_PX, return_masks=debug)
        blobs, masks = result if debug else (result, None)
        self.pub.publish(Float64MultiArray(data=[t_capture, float(len(blobs)), *blobs.ravel().tolist()]))
        if masks is not None:
            cls = np.zeros((h, w), dtype=np.uint8)
            cls[masks["ignored"]] = 1
            cls[masks["robot"]] = 2
            cls[masks["foreground"]] = 3
            header = np.frombuffer(np.array([t_capture], dtype=np.float64).tobytes(), dtype=np.uint8)
            size = np.array([w, h], dtype=np.uint16).view(np.uint8)
            self.debug_pub.publish(UInt8MultiArray(data=array("B", np.concatenate([header, size, cls.ravel()]).tobytes())))


def main():
    rclpy.init()
    node = CameraDetectionNode()
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
