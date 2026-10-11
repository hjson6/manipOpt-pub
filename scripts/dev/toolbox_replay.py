"""Plays a recorded drive (record_drive.py) to slam_toolbox in simulated time and
collects its result: the map (/map), its pose at each scan (its map > odom
correction times the recorded odometry) and its CPU per scan. mapping: the
sync node, then the pose graph is serialized for localization; localization: the
localization node in a serialized map, started at the true start pose plus
--init-offset. The scans are the two lidars merged into one 360 deg scan in
base_link (slam/scan.py), as the live system feeds it.
usage: python toolbox_replay.py <recording> [--mode mapping|localization] [--map <name>]
       [--init-offset dx dy dyaw_deg] [--rate 2.0]
Writes data/maps/toolbox_<recording>.{pgm,yaml,posegraph,data} and ..._result.npz.
"""
import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from builtin_interfaces.msg import Time
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import OccupancyGrid as GridMsg
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import LaserScan
from tf2_msgs.msg import TFMessage

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tasks/pick_and_place/common")]
from pick_place_common.scene import LIDAR_MOUNTS_BASE  # noqa: E402
from slam.grid import L_MAX, L_MIN, OccupancyGrid  # noqa: E402
from slam.pose_graph import compose, relative  # noqa: E402
from slam.scan import merge_scan, scan_points  # noqa: E402

CONFIG = REPO / "tasks/pick_and_place/mpc/config"
MAPS = REPO / "data" / "maps"
T0 = 100.0  # sim time offset: a zero stamp means "no time" to ROS
N_BEAMS = 720


def stamp(t):
    s = T0 + t
    return Time(sec=int(s), nanosec=int(round((s % 1.0) * 1e9)) % 1_000_000_000)


def tf_msg(parent, child, t, pose):
    m = TransformStamped()
    m.header.stamp = stamp(t)
    m.header.frame_id = parent
    m.child_frame_id = child
    m.transform.translation.x, m.transform.translation.y = float(pose[0]), float(pose[1])
    m.transform.rotation.z, m.transform.rotation.w = float(np.sin(pose[2] / 2)), float(np.cos(pose[2] / 2))
    return m


def cpu_seconds(pgid):
    """CPU time of every process in the group (ros2 run starts the node as a child)."""
    total = 0
    for stat in Path("/proc").glob("[0-9]*/stat"):
        try:
            f = stat.read_text().rsplit(")", 1)[1].split()
        except OSError:
            continue
        if int(f[2]) == pgid:
            total += int(f[11]) + int(f[12])
    return total / os.sysconf("SC_CLK_TCK")


class Player(Node):
    def __init__(self):
        super().__init__("recording_player", parameter_overrides=[])
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        self.clock = self.create_publisher(Clock, "/clock", 10)
        self.tf = self.create_publisher(TFMessage, "/tf", 100)
        self.tf_static = self.create_publisher(TFMessage, "/tf_static", latched)
        self.scan = self.create_publisher(LaserScan, "/scan", 10)
        self.map_to_odom = []  # (t, pose)
        self.map = None
        self.create_subscription(TFMessage, "/tf", self._on_tf, 100)
        self.create_subscription(GridMsg, "/map", self._on_map, latched)

    def _on_tf(self, msg):
        for tr in msg.transforms:
            if tr.header.frame_id == "map" and tr.child_frame_id == "odom":
                q = tr.transform.rotation
                yaw = 2 * np.arctan2(q.z, q.w)
                t = tr.header.stamp.sec + tr.header.stamp.nanosec * 1e-9 - T0
                self.map_to_odom.append((t, np.array([tr.transform.translation.x, tr.transform.translation.y, yaw])))

    def _on_map(self, msg):
        self.map = msg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recording")
    ap.add_argument("--mode", default="mapping", choices=("mapping", "localization"))
    ap.add_argument("--map", default="")
    ap.add_argument("--init-offset", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    ap.add_argument("--rate", type=float, default=2.0)
    args = ap.parse_args()
    r = np.load(REPO / "data" / "recordings" / f"{args.recording}.npz")
    MAPS.mkdir(parents=True, exist_ok=True)
    name = f"toolbox_{args.recording}" if args.mode == "mapping" else f"toolbox_loc_{args.recording}_in_{args.map}"
    if args.mode == "mapping":
        cmd = ["ros2", "run", "slam_toolbox", "sync_slam_toolbox_node", "--ros-args", "--params-file",
               str(CONFIG / "slam_toolbox_mapping.yaml"), "-p", "use_sim_time:=true"]
        start = None
    else:
        t0_map = np.load(MAPS / f"{args.map}_result.npz")["truth_start"]
        start = relative(t0_map, r["scan_truth"][0]) + (args.init_offset[0], args.init_offset[1],
                                                         np.radians(args.init_offset[2]))
        cmd = ["ros2", "run", "slam_toolbox", "localization_slam_toolbox_node", "--ros-args", "--params-file",
               str(CONFIG / "slam_toolbox_localization.yaml"), "-p", "use_sim_time:=true",
               "-p", f"map_file_name:={MAPS / args.map}", "-p",
               f"map_start_pose:=[{start[0]:.4f}, {start[1]:.4f}, {start[2]:.4f}]"]
    log = open(MAPS / f"{name}.log", "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    rclpy.init()
    node = Player()
    try:
        node.tf_static.publish(TFMessage(transforms=[tf_msg("base_link", "base_scan", 0.0, (0.0, 0.0, 0.0))]))
        t_ticks, odom = r["t"], r["odom"]
        scan_idx = {int(round(ts / 0.02)): i for i, ts in enumerate(r["scan_t"])}
        for _ in range(int(3.0 / 0.02)):  # let the node come up, clock at the start
            node.clock.publish(Clock(clock=stamp(0.0)))
            node.tf.publish(TFMessage(transforms=[tf_msg("odom", "base_link", 0.0, odom[0])]))
            rclpy.spin_once(node, timeout_sec=0.0)
            time.sleep(0.02)
        cpu0 = cpu_seconds(proc.pid)
        wall = time.perf_counter()
        for k, (t, o) in enumerate(zip(t_ticks, odom)):
            node.clock.publish(Clock(clock=stamp(t)))
            node.tf.publish(TFMessage(transforms=[tf_msg("odom", "base_link", t, o)]))
            i = scan_idx.get(k + 1)
            if i is not None:
                pts = scan_points(r["scans"][i], r["beam_angles"], LIDAR_MOUNTS_BASE)
                ranges = merge_scan(pts, N_BEAMS)
                msg = LaserScan()
                msg.header.stamp = stamp(t)
                msg.header.frame_id = "base_scan"
                msg.angle_min, msg.angle_increment = -np.pi, 2 * np.pi / N_BEAMS
                msg.angle_max = msg.angle_min + (N_BEAMS - 1) * msg.angle_increment
                msg.range_min, msg.range_max = 0.05, 10.0
                msg.ranges = np.where(np.isfinite(ranges), ranges, np.inf).astype(np.float32).tolist()
                node.scan.publish(msg)
            rclpy.spin_once(node, timeout_sec=0.0)
            due = wall + (k + 1) * 0.02 / args.rate
            while time.perf_counter() < due:
                rclpy.spin_once(node, timeout_sec=max(0.0, due - time.perf_counter()))
        end = time.perf_counter() + 5.0  # the last scans and the map
        while time.perf_counter() < end:
            node.clock.publish(Clock(clock=stamp(t_ticks[-1] + 0.02)))
            rclpy.spin_once(node, timeout_sec=0.05)
        cpu = cpu_seconds(proc.pid) - cpu0
        if args.mode == "mapping":
            from slam_toolbox.srv import SerializePoseGraph
            cli = node.create_client(SerializePoseGraph, "/slam_toolbox/serialize_map")
            cli.wait_for_service(timeout_sec=10.0)
            fut = cli.call_async(SerializePoseGraph.Request(filename=str(MAPS / name)))
            rclpy.spin_until_future_complete(node, fut, timeout_sec=30.0)
        # The pose at each scan: the correction in effect a scan period later times the odometry.
        mo = node.map_to_odom
        ts_mo = np.array([t for t, _ in mo]) if mo else np.zeros(0)
        idx = np.round(r["scan_t"] / 0.02).astype(int) - 1
        poses = []
        for ts, o in zip(r["scan_t"], odom[idx]):
            j = np.searchsorted(ts_mo, ts + 1.0 / 15, side="right") - 1
            poses.append(compose(mo[j][1], o) if j >= 0 else o)
        out = dict(poses=np.array(poses), ms=np.full(len(poses), 1e3 * cpu / len(poses)),
                   map_yaml=str(MAPS / f"{name}.yaml"), cpu_s=cpu)
        if args.mode == "mapping":
            out["truth_start"] = r["scan_truth"][0]
        if node.map is not None:
            info = node.map.info
            grid = OccupancyGrid((info.origin.position.x, info.origin.position.y), (info.height, info.width),
                                 info.resolution)
            occ = np.array(node.map.data, dtype=np.int16).reshape(info.height, info.width)
            grid.log_odds[occ >= 65] = L_MAX
            grid.log_odds[(occ >= 0) & (occ <= 25)] = L_MIN
            grid.save(MAPS / name)
        np.savez(MAPS / f"{name}_result.npz", **out)
        print(f"{name}: {len(poses)} scans, {len(mo)} map>odom updates, map {'saved' if node.map else 'missing'}, "
              f"CPU {cpu:.1f} s ({1e3 * cpu / len(poses):.1f} ms per scan)")
    finally:
        os.killpg(proc.pid, signal.SIGINT)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
