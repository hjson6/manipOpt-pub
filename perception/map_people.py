"""People from the safety lidars on a moving base: in the map frame (the localized
pose), the beams the map does not explain (more than FOREGROUND_M from every mapped
surface) are foreground; runs of them along each scan are legs and legs close
together are people (perception/lidar_detection.py's leg and pairing rules).
Framework-free (numpy, scipy).

See docs/implementation_notes.md#map_peoplepy.
"""
import numpy as np
from scipy.spatial import cKDTree

from perception.lidar_detection import group_people, legs_from_scan

FOREGROUND_M = 0.15  # farther than this from the mapped surfaces: not part of the map


class MapPeopleDetector:
    def __init__(self, map_points, mounts, beam_angles, range_max=10.0):
        """map_points: the map's surfaces (map frame); mounts: per scanner (x, y, z, yaw)
        in base_link; beam_angles: relative to each scanner's sector centre."""
        self.tree = cKDTree(np.asarray(map_points, dtype=float))
        self.mounts = [tuple(m) for m in mounts]
        self.angles = np.asarray(beam_angles, dtype=float)
        self.range_max = range_max
        self.foreground = np.zeros((0, 2))
        self.legs = []
        self.views = []  # per scanner: (sensor xy, sector heading, ranges) of the last round

    def update(self, pose, ranges):
        """One round of scans (ranges per scanner, nan: no return) at the base's map pose
        (x, y, yaw); returns [(x, y, radius, n_legs)] in the map frame."""
        c, s = np.cos(pose[2]), np.sin(pose[2])
        fg_all, self.legs, self.views = [], [], []
        for (mx, my, _mz, myaw), r in zip(self.mounts, ranges):
            r = np.asarray(r, dtype=float)
            sensor = np.array([pose[0] + c * mx - s * my, pose[1] + s * mx + c * my])
            a = pose[2] + myaw + self.angles
            pts = sensor + r[:, None] * np.column_stack([np.cos(a), np.sin(a)])
            ok = np.isfinite(r) & (r < self.range_max)
            d = np.full(len(r), 0.0)
            d[ok], _ = self.tree.query(pts[ok])
            fg = ok & (d > FOREGROUND_M)
            fg_all.append(pts[fg])
            self.views.append((sensor, pose[2] + myaw, r))
            self.legs.append(legs_from_scan(sensor, pts, fg))
        self.foreground = np.vstack(fg_all) if fg_all else np.zeros((0, 2))
        return group_people(self.legs)

    def seen_empty(self, xy, margin=0.3):
        """True if a beam of the last round went on past xy by more than margin: the
        lidars saw through that spot, so nobody stands there."""
        for sensor, heading, r in self.views:
            d = np.asarray(xy, dtype=float) - sensor
            dist = float(np.hypot(*d))
            rel = (np.arctan2(d[1], d[0]) - heading + np.pi) % (2 * np.pi) - np.pi
            k = int(round((rel - self.angles[0]) / (self.angles[1] - self.angles[0])))
            if 0 <= k < len(r) and dist > 0.1:
                beam = r[max(k - 1, 0):k + 2]
                if np.all(np.isnan(beam) | (beam > dist + margin)):
                    return True
        return False
