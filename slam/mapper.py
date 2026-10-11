"""Our own 2D graph SLAM: each scan is matched (point-to-line ICP) against a local map
of the latest keyframes, starting from the odometry's prediction; every 0.3 m or 15
deg a keyframe joins the pose graph, and keyframes near an old part of the path are
matched against it for loop closures, after which the graph is optimized. The map
is an occupancy grid of the keyframes' scans at their optimized poses.
Framework-free.

See docs/implementation_notes.md#slam.
"""
import numpy as np

from slam import icp
from slam.grid import OccupancyGrid
from slam.pose_graph import PoseGraph, compose, relative
from slam.scan import transform, voxel_downsample

KEYFRAME_DIST_M = 0.3
KEYFRAME_ANGLE_RAD = np.radians(15)
SUBMAP_KEYFRAMES = 12
VOXEL_M = 0.05
MATCH_MIN_FITNESS = 0.35  # below: keep the odometry's prediction
LOOP_RADIUS_M = 3.0
LOOP_MIN_GAP = 25  # keyframes between a closure's ends
LOOP_CANDIDATES = 3
LOOP_MIN_FITNESS = 0.55
LOOP_MAX_RMS_M = 0.035
LOOP_MAX_JUMP_M = 1.0
STEP_INFO = np.diag([1 / 0.02 ** 2, 1 / 0.02 ** 2, 1 / np.radians(1.0) ** 2])  # keyframe to keyframe


class GraphSlam:
    def __init__(self):
        self.graph = PoseGraph()
        self.keyframes = []  # dict(points, full, origins, node)
        self.pose = np.zeros(3)  # current scan's pose in the map frame
        self.last_odom = None
        self._target = None
        self.loops = []  # (i, j) closures accepted
        self.rejected = 0
        self.matches_lost = 0

    def update(self, odom, points, origins):
        """One scan (points and their scanners' positions in base_link) with the odometry
        pose at its time; returns the scan's pose in the map frame."""
        odom = np.asarray(odom, dtype=float)
        pts = voxel_downsample(points, VOXEL_M)
        if self.last_odom is None:
            self.last_odom = odom
            self._add_keyframe(np.zeros(3), pts, points, origins)
            return self.pose.copy()
        predicted = compose(self.pose, relative(self.last_odom, odom))
        self.last_odom = odom
        est, _info, _rms, fit = icp.match(pts, self._submap(), predicted)
        if fit < MATCH_MIN_FITNESS or np.hypot(*(est[:2] - predicted[:2])) > 0.3:
            est = predicted
            self.matches_lost += 1
        self.pose = est
        last = self.graph.nodes[self.keyframes[-1]["node"]]
        step = relative(last, est)
        if np.hypot(step[0], step[1]) > KEYFRAME_DIST_M or abs(step[2]) > KEYFRAME_ANGLE_RAD:
            self._add_keyframe(est, pts, points, origins)
        return self.pose.copy()

    def _add_keyframe(self, pose, pts, full, origins):
        node = self.graph.add_node(pose)
        if self.keyframes:
            prev = self.keyframes[-1]["node"]
            z = relative(self.graph.nodes[prev], pose)
            self.graph.add_edge(prev, node, z, STEP_INFO)
        self.keyframes.append(dict(points=pts, full=full, origins=origins, node=node))
        self._target = None
        if self._close_loop(len(self.keyframes) - 1):
            self.graph.optimize()
            self.pose = self.graph.nodes[node].copy()
            self._target = None

    def _submap_points(self, kfs):
        return voxel_downsample(np.vstack([transform(self.graph.nodes[k["node"]], k["points"]) for k in kfs]), VOXEL_M)

    def _submap(self):
        if self._target is None:
            self._target = icp.Target(self._submap_points(self.keyframes[-SUBMAP_KEYFRAMES:]))
        return self._target

    def _close_loop(self, cur):
        if cur < LOOP_MIN_GAP:
            return False
        nodes = np.array(self.graph.nodes)
        here = nodes[self.keyframes[cur]["node"]]
        old = np.arange(cur - LOOP_MIN_GAP)
        d = np.hypot(*(nodes[[self.keyframes[i]["node"] for i in old], :2] - here[:2]).T)
        found = False
        for i in old[np.argsort(d)][:LOOP_CANDIDATES]:
            if d[i] > LOOP_RADIUS_M:
                break
            target = icp.Target(self._submap_points(self.keyframes[max(0, i - 3):i + 4]))
            est, info, rms, fit = icp.match(self.keyframes[cur]["points"], target, here)
            if fit < LOOP_MIN_FITNESS or rms > LOOP_MAX_RMS_M or np.hypot(*(est[:2] - here[:2])) > LOOP_MAX_JUMP_M:
                self.rejected += 1
                continue
            a = self.keyframes[i]["node"]
            self.graph.add_edge(a, self.keyframes[cur]["node"], relative(nodes[a], est), info, kind="loop")
            self.loops.append((i, cur))
            found = True
        return found

    def keyframe_poses(self):
        return np.array([self.graph.nodes[k["node"]] for k in self.keyframes])

    def map(self, resolution=0.05):
        """The occupancy grid of all keyframes at their optimized poses."""
        poses = self.keyframe_poses()
        pts = np.vstack([transform(p, k["full"]) for p, k in zip(poses, self.keyframes)])
        grid = OccupancyGrid.around(pts, resolution)
        for p, k in zip(poses, self.keyframes):
            grid.integrate(p, k["full"], k["origins"])
        return grid
