"""Validation only: the people tracks (/perception/people, map frame) against the true
crowd (/env/people, room frame, moved to the map frame anchored at the dock) and the
robot's true pose. Per round of tracks: each person within RANGE_M of the robot
tracked within MATCH_M or not (occlusion is not known live, so this rate is a lower
bound), and tracks with no person near (false people). Logs a summary every
`summary_s` and one for the run.
"""
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float64MultiArray

from pick_place_common.scene import BASE_HOME_POSE
from slam.pose_graph import relative

RANGE_M = 5.0
MATCH_M = 0.5


class PeopleMonitorNode(Node):
    def __init__(self):
        super().__init__("people_monitor_node")
        self.declare_parameter("summary_s", 10.0)
        self.origin = np.array(BASE_HOME_POSE, dtype=float)
        self.people = np.zeros((0, 2))
        self.robot = None
        self.window = [0, 0, 0, []]  # near, tracked, false, position errors
        self.total = [0, 0, 0, []]
        self.create_subscription(Float64MultiArray, "/env/people", self._on_people, 10)
        self.create_subscription(Float64MultiArray, "/sim/base_truth", self._on_truth, 10)
        self.create_subscription(Float64MultiArray, "/perception/people", self._on_tracks, qos_profile_sensor_data)
        self.create_timer(float(self.get_parameter("summary_s").value), self._summary)

    def _to_map(self, x, y):
        return relative(self.origin, (x, y, 0.0))[:2]

    def _on_people(self, msg):
        d = np.array(msg.data).reshape(-1, 3)
        self.people = np.array([self._to_map(x, y) for x, y, _ in d if np.isfinite(x)]).reshape(-1, 2)

    def _on_truth(self, msg):
        self.robot = self._to_map(msg.data[1], msg.data[2])

    def _on_tracks(self, msg):
        if self.robot is None:
            return
        n = int(msg.data[1])
        tracks = np.array(msg.data[2:2 + 7 * n]).reshape(n, 7)[:, 1:3]
        used = set()
        for p in self.people:
            hit = None
            if n:
                d = np.hypot(*(tracks - p).T)
                j = int(np.argmin(d))
                if d[j] < MATCH_M:
                    hit = j
                    used.add(j)
            if np.hypot(*(p - self.robot)) <= RANGE_M:
                for acc in (self.window, self.total):
                    acc[0] += 1
                    if hit is not None:
                        acc[1] += 1
                        acc[3].append(float(np.hypot(*(tracks[hit] - p))))
        false = n - len(used)
        self.window[2] += false
        self.total[2] += false

    def _report(self, acc, what):
        near, tracked, false, err = acc
        if near == 0 and false == 0:
            return
        e = 1e3 * np.array(err) if err else np.array([np.nan])
        self.get_logger().info(f"people {what}: within {RANGE_M:.0f} m tracked {tracked}/{near} "
                               f"({100 * tracked / max(near, 1):.1f}%), position error p50/p95 {np.median(e):.0f}/"
                               f"{np.percentile(e, 95):.0f} mm, false tracks {false}")

    def _summary(self):
        self._report(self.window, "over the last window")
        self.window = [0, 0, 0, []]


def main():
    rclpy.init()
    node = PeopleMonitorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._report(node.total, "over the run")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
