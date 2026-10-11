"""The mobile base's driver on the method side: odometry from the wheel encoders
and the IMU (core/base_odometry.py, the taught wheel radius and track), and
body velocity commands turned into wheel speeds for the motor drivers.

Subscribes /sim/wheel_states, /sim/imu, /base/cmd_vel (geometry_msgs/Twist in
base_link). Publishes /odom (nav_msgs/Odometry, odom > base_link), the same
transform on /tf, the robot's own fixed frames on /tf_static (base_link > base_scan,
the merged scan's frame; > lidar_0/1; > arm_base, the arm's link0), and
/sim/wheel_command [left, right] (rad/s) every tick. No command for CMD_TIMEOUT_S:
zero speed.

See docs/implementation_notes.md#base_nodepy.
"""
import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float64MultiArray
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

from core.base_odometry import BaseOdometry, wheel_speeds
from pick_place_common.scene import BASE_ARM_MOUNT, BASE_WHEEL_RADIUS_M, BASE_WHEEL_TRACK_M, LIDAR_MOUNTS_BASE

CONTROL_PERIOD_S = 0.02  # the plant's tick; samples are stamped with its step
CMD_TIMEOUT_S = 0.25
WHEEL_SPEED_MAX = 15.0  # rad/s, below the drives' 20
V_MAX, W_MAX = 1.0, 1.5  # m/s, rad/s: what this driver accepts


class BaseNode(Node):
    def __init__(self):
        super().__init__("base_node")
        self.declare_parameter("use_gyro", True)
        self.odo = BaseOdometry(BASE_WHEEL_RADIUS_M, BASE_WHEEL_TRACK_M,
                                use_gyro=bool(self.get_parameter("use_gyro").value))
        self.pending = {}  # step -> {"wheels": angles, "gyro": rate}
        self.last_step = None
        self.cmd = (0.0, 0.0)
        self.cmd_t = -np.inf
        best_effort = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.wheel_pub = self.create_publisher(Float64MultiArray, "/sim/wheel_command", best_effort)
        self.odom_pub = self.create_publisher(Odometry, "/odom", 10)
        self.tf = TransformBroadcaster(self)
        self.tf_static = StaticTransformBroadcaster(self)
        mounts = [("base_scan", (0.0, 0.0, 0.0, 0.0)), ("arm_base", BASE_ARM_MOUNT)]
        mounts += [(f"lidar_{i}", m) for i, m in enumerate(LIDAR_MOUNTS_BASE)]
        self.tf_static.sendTransform([self._transform("base_link", child, self.get_clock().now().to_msg(), x, y, yaw, z)
                                      for child, (x, y, z, yaw) in mounts])
        self.create_subscription(JointState, "/sim/wheel_states", self._on_wheels, 10)
        self.create_subscription(Imu, "/sim/imu", self._on_imu, 10)
        self.create_subscription(Twist, "/base/cmd_vel", self._on_cmd_vel, 10)

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_cmd_vel(self, msg: Twist):
        self.cmd = (float(np.clip(msg.linear.x, -V_MAX, V_MAX)), float(np.clip(msg.angular.z, -W_MAX, W_MAX)))
        self.cmd_t = self._now()

    def _on_wheels(self, msg: JointState):
        self._sample(int(msg.header.frame_id), "wheels", np.array(msg.position), msg.header.stamp)
        v, w = self.cmd if self._now() - self.cmd_t <= CMD_TIMEOUT_S else (0.0, 0.0)
        speeds = wheel_speeds(v, w, BASE_WHEEL_RADIUS_M, BASE_WHEEL_TRACK_M, WHEEL_SPEED_MAX)
        self.wheel_pub.publish(Float64MultiArray(data=speeds.tolist()))

    def _on_imu(self, msg: Imu):
        self._sample(int(msg.header.frame_id), "gyro", msg.angular_velocity.z, msg.header.stamp)

    def _sample(self, step, key, value, stamp):
        """Update once both of a step's samples are in (or the wheels alone, with no IMU)."""
        s = self.pending.setdefault(step, {})
        s[key] = value
        s["stamp"] = stamp
        for old in [k for k in self.pending if k < step - 5]:
            del self.pending[old]
        if "wheels" not in s or ("gyro" not in s and self.odo.use_gyro):
            return
        del self.pending[step]
        dt = 0.0 if self.last_step is None else (step - self.last_step) * CONTROL_PERIOD_S
        if self.last_step is not None and step <= self.last_step:
            return
        self.last_step = step
        x, y, yaw = self.odo.update(s["wheels"], dt, s.get("gyro"))
        self._publish(s["stamp"], x, y, yaw)

    @staticmethod
    def _transform(parent, child, stamp, x, y, yaw, z=0.0):
        tf = TransformStamped()
        tf.header.stamp = stamp
        tf.header.frame_id = parent
        tf.child_frame_id = child
        tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = float(x), float(y), float(z)
        tf.transform.rotation.w, tf.transform.rotation.z = float(np.cos(yaw / 2)), float(np.sin(yaw / 2))
        return tf

    def _publish(self, stamp, x, y, yaw):
        q = (np.cos(yaw / 2), np.sin(yaw / 2))
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"
        odom.pose.pose.position.x, odom.pose.pose.position.y = float(x), float(y)
        odom.pose.pose.orientation.w, odom.pose.pose.orientation.z = float(q[0]), float(q[1])
        odom.twist.twist.linear.x, odom.twist.twist.angular.z = (float(v) for v in self.odo.twist)
        self.odom_pub.publish(odom)
        self.tf.sendTransform(self._transform("odom", "base_link", stamp, x, y, yaw))


def main():
    rclpy.init()
    node = BaseNode()
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
