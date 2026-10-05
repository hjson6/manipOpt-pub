"""Lidar scans to points in base_link, with the scanners' calibration (scene.py's
LIDAR_MOUNTS_BASE), and the merged 360 deg scan both SLAM options take.
Framework-free (numpy).
"""
import numpy as np


def scan_points(ranges, beam_angles, mounts, range_max=10.0, origins=False):
    """Points (N, 2) in base_link from each scanner's ranges (nan: no return);
    mounts: [(x, y, z, yaw)] per scanner, as `ranges`. origins: also each point's
    scanner position (N, 2)."""
    pts, orig = [], []
    for r, (mx, my, _mz, myaw) in zip(ranges, mounts):
        r = np.asarray(r, dtype=float)
        ok = np.isfinite(r) & (r < range_max)
        a = myaw + beam_angles[ok]
        pts.append(np.column_stack([mx + r[ok] * np.cos(a), my + r[ok] * np.sin(a)]))
        orig.append(np.tile([mx, my], (ok.sum(), 1)))
    p = np.vstack(pts) if pts else np.zeros((0, 2))
    return (p, np.vstack(orig) if orig else np.zeros((0, 2))) if origins else p


def merge_scan(points, n_beams=720, range_min=0.05, range_max=10.0):
    """A 360 deg scan from base_link's origin: the nearest point per beam (nan: none);
    beam k at angle -pi + k * 2 pi / n_beams."""
    r = np.hypot(points[:, 0], points[:, 1])
    a = np.arctan2(points[:, 1], points[:, 0])
    k = np.floor((a + np.pi) / (2 * np.pi) * n_beams).astype(int) % n_beams
    ok = (r >= range_min) & (r <= range_max)
    ranges = np.full(n_beams, np.inf)
    np.minimum.at(ranges, k[ok], r[ok])
    ranges[~np.isfinite(ranges)] = np.nan
    return ranges


def voxel_downsample(points, cell):
    """One point (the mean) per cell x cell square."""
    if len(points) == 0:
        return points
    keys = np.floor(points / cell).astype(np.int64)
    _, inv = np.unique(keys[:, 0] * 1_000_003 + keys[:, 1], return_inverse=True)
    sums = np.zeros((inv.max() + 1, 2))
    np.add.at(sums, inv, points)
    return sums / np.bincount(inv)[:, None]


def transform(pose, points):
    """Points from a frame at pose (x, y, yaw) into its parent."""
    c, s = np.cos(pose[2]), np.sin(pose[2])
    return points @ np.array([[c, s], [-s, c]]) + pose[:2]
