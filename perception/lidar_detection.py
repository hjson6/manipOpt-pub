"""People from planar safety-lidar scans (plain numpy).

Per scanner: a background range per beam, learnt from the first scans of the
empty cell (the base, the table legs: static); foreground beams return nearer
than it by FOREGROUND_MARGIN_M. Consecutive foreground beams closer than
CLUSTER_GAP_M make a leg; its centre is the visible arc's centroid moved away
from the scanner by the arc's mean depth (2/pi of its half width). Legs seen
by both scanners are merged, and legs within PAIR_M make one person.
See docs/implementation_notes.md#lidar_detectionpy.
"""
import numpy as np

BACKGROUND_SCANS = 20
FOREGROUND_MARGIN_M = 0.08
CLUSTER_GAP_M = 0.10
MIN_POINTS = 2
LEG_RADIUS_LIMITS_M = (0.03, 0.12)
SAME_LEG_M = 0.10
PAIR_M = 0.5


def scan_points(pose, angles, ranges):
    """Beam end points (n, 2) in the base frame; nan where there is no return."""
    x, y, _z, yaw = pose
    a = yaw + np.asarray(angles)
    r = np.asarray(ranges, dtype=float)
    return np.column_stack([x + r * np.cos(a), y + r * np.sin(a)])


class LidarPeopleDetector:
    def __init__(self, poses, angles):
        """poses: per scanner (x, y, z, yaw of the sector's centre); angles: beam
        angles relative to that centre."""
        self.poses = [tuple(p) for p in poses]
        self.angles = np.asarray(angles, dtype=float)
        self.learning = [[] for _ in poses]
        self.background = [None] * len(poses)
        self.legs = [np.zeros((0, 3))] * len(poses)  # per scanner: (x, y, radius)
        self.foreground = [np.zeros((0, 2))] * len(poses)

    @property
    def ready(self):
        return all(b is not None for b in self.background)

    def update(self, index, ranges):
        """Take one scan of scanner `index`; while learning, it goes to the background."""
        r = np.asarray(ranges, dtype=float)
        if self.background[index] is None:
            self.learning[index].append(r)
            if len(self.learning[index]) >= BACKGROUND_SCANS:
                stack = np.array(self.learning[index])
                # A beam with returns in most scans has their median; else none (inf).
                seen = np.isfinite(stack).mean(axis=0) >= 0.5
                bg = np.full(len(r), np.inf)
                bg[seen] = np.nanmedian(stack[:, seen], axis=0)
                self.background[index] = bg
                self.learning[index] = []
            return
        fg = np.isfinite(r) & (r < self.background[index] - FOREGROUND_MARGIN_M)
        pts = scan_points(self.poses[index], self.angles, r)
        self.foreground[index] = pts[fg]
        self.legs[index] = self._legs(index, pts, fg)

    def _legs(self, index, pts, fg):
        return legs_from_scan(self.poses[index][:2], pts, fg)

    def people(self):
        """[(x, y, radius, n_legs)]: each person's centre and the radius round it that
        holds the legs seen, from the latest scan of every scanner."""
        return group_people(self.legs)


def legs_from_scan(sensor_xy, pts, fg):
    """Legs (x, y, radius) from one scan's beam end points in beam order: runs of
    foreground beams closer than CLUSTER_GAP_M with at least MIN_POINTS beams, each
    centred on the visible arc's centroid moved away from the sensor by the arc's mean
    depth."""
    sensor = np.asarray(sensor_xy, dtype=float)
    legs, run = [], []
    for i in np.flatnonzero(fg):
        if run and np.linalg.norm(pts[i] - pts[run[-1]]) > CLUSTER_GAP_M:
            legs.append(run)
            run = []
        run.append(i)
    if run:
        legs.append(run)
    out = []
    for run in legs:
        if len(run) < MIN_POINTS:
            continue
        p = pts[run]
        c = p.mean(axis=0)
        half = np.clip(np.linalg.norm(p[-1] - p[0]) / 2.0, *LEG_RADIUS_LIMITS_M)
        u = (c - sensor) / max(np.linalg.norm(c - sensor), 1e-9)
        out.append((*(c + (2.0 / np.pi) * half * u), half))
    return np.array(out).reshape(-1, 3)


def group_people(legs_per_scanner):
    """[(x, y, radius, n_legs)] from every scanner's legs: legs within SAME_LEG_M are one
    leg, legs within PAIR_M of each other one person."""
    legs = []  # (x, y, r, weight)
    for scanner_legs in legs_per_scanner:
        for x, y, r in scanner_legs:
            for k, (lx, ly, lr, w) in enumerate(legs):
                if np.hypot(x - lx, y - ly) < SAME_LEG_M:
                    legs[k] = ((lx * w + x) / (w + 1), (ly * w + y) / (w + 1), max(lr, r), w + 1)
                    break
            else:
                legs.append((x, y, r, 1))
    groups = []
    for leg in legs:
        near = [g for g in groups if any(np.hypot(leg[0] - o[0], leg[1] - o[1]) < PAIR_M for o in g)]
        merged = [leg] + [o for g in near for o in g]
        groups = [g for g in groups if g not in near] + [merged]
    out = []
    for g in groups:
        xy = np.array([(o[0], o[1]) for o in g])
        c = xy.mean(axis=0)
        radius = max(np.hypot(o[0] - c[0], o[1] - c[1]) + o[2] for o in g)
        out.append((float(c[0]), float(c[1]), float(radius), len(g)))
    return out
