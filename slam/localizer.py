"""Localization in a saved map: each scan is matched (point-to-line ICP, slam/icp.py)
against the map's surfaces (the occupied cells' mean hit positions, or their centres
for a map without them), starting from the odometry's prediction from the last
estimate. ICP's shrinking match distance leaves out points far from any mapped
surface (people, things moved since mapping). A poor match keeps the prediction and
counts as lost. Framework-free.

See docs/implementation_notes.md#slam.
"""
import numpy as np

from slam import icp
from slam.pose_graph import compose, relative
from slam.scan import voxel_downsample

VOXEL_M = 0.05
MIN_FITNESS = 0.4  # share of scan points within 5 cm of a mapped surface
MAX_CORRECTION_M = 0.3
MAP_NORMAL_K = 12  # a mapped wall is about two cells thick: a line only over more neighbours


class GridLocalizer:
    def __init__(self, grid, initial_pose):
        self.target = icp.Target(grid.occupied_points(), k=MAP_NORMAL_K, neighbour_max=0.35)
        self.pose = np.array(initial_pose, dtype=float)
        self.last_odom = None
        self.quality = 0.0
        self.lost = 0

    def update(self, odom, points):
        """One scan (points in base_link) with the odometry pose at its time; returns the
        pose in the map frame."""
        odom = np.asarray(odom, dtype=float)
        predicted = self.pose if self.last_odom is None else compose(self.pose, relative(self.last_odom, odom))
        self.last_odom = odom
        est, _info, fit = self.match(points, predicted)
        self.quality = fit
        if fit < MIN_FITNESS or np.hypot(*(est[:2] - predicted[:2])) > MAX_CORRECTION_M:
            est = predicted
            self.lost += 1
        self.pose = est
        return est.copy()

    def match(self, points, prior):
        """(pose, information, fitness) of a scan matched against the map from prior;
        leaves the localizer as it was."""
        est, info, _rms, fit = icp.match(voxel_downsample(points, VOXEL_M), self.target, prior)
        return est, info, fit
