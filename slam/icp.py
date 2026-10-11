"""Point-to-line ICP in 2D (Censi's PL-ICP, Gauss-Newton with Huber weights): the
pose that puts a scan's points on the lines of a target point set (another scan or
a local map). Framework-free (numpy, scipy).

See docs/implementation_notes.md#slam.
"""
import numpy as np
from scipy.spatial import cKDTree

from slam.scan import transform

NORMAL_K = 6
LINE_RATIO = 0.15  # smallest over largest spread of a neighbourhood: line-like below
NEIGHBOUR_MAX_M = 0.25
SIGMA_M = 0.03  # per-point line distance noise, for the information matrix


class Target:
    """A point set to match against: KD-tree and per-point line normals."""

    def __init__(self, points, k=NORMAL_K, neighbour_max=NEIGHBOUR_MAX_M):
        self.points = np.asarray(points, dtype=float)
        self.tree = cKDTree(self.points)
        k = min(k, len(self.points))
        d, idx = self.tree.query(self.points, k=k)
        nb = self.points[idx]
        c = nb - nb.mean(axis=1, keepdims=True)
        w, v = np.linalg.eigh(np.einsum("nki,nkj->nij", c, c) / k)
        self.normals = v[:, :, 0]
        self.ok = (w[:, 0] < LINE_RATIO * w[:, 1]) & (d[:, -1] < neighbour_max)


def match(src, target, init, max_iter=30, max_dist=0.5, min_dist=0.08, huber=0.05):
    """(pose, info, rms, fitness): the pose of src's frame in target's frame starting
    from init; info is the 3x3 information of the estimate, fitness the share of src
    points within 5 cm of a target line."""
    src = np.asarray(src, dtype=float)
    pose = np.array(init, dtype=float)
    h = np.eye(3)
    dist = max_dist
    for _ in range(max_iter):
        p = transform(pose, src)
        d, j = target.tree.query(p, distance_upper_bound=dist)
        ok = np.isfinite(d)
        ok[ok] &= target.ok[j[ok]]
        if ok.sum() < 10:
            break
        n = target.normals[j[ok]]
        r = np.einsum("ij,ij->i", n, p[ok] - target.points[j[ok]])
        rel = p[ok] - pose[:2]
        jac = np.column_stack([n[:, 0], n[:, 1], n[:, 1] * rel[:, 0] - n[:, 0] * rel[:, 1]])
        w = np.where(np.abs(r) < huber, 1.0, huber / np.maximum(np.abs(r), 1e-12))
        h = jac.T @ (w[:, None] * jac)
        delta = -np.linalg.solve(h + 1e-9 * np.eye(3), jac.T @ (w * r))
        pose += delta
        pose[2] = (pose[2] + np.pi) % (2 * np.pi) - np.pi
        dist = max(min_dist, dist * 0.75)
        if np.abs(delta[:2]).max() < 1e-5 and abs(delta[2]) < 1e-5:
            break
    p = transform(pose, src)
    d, j = target.tree.query(p, distance_upper_bound=0.2)
    ok = np.isfinite(d)
    ok[ok] &= target.ok[j[ok]]
    r = np.einsum("ij,ij->i", target.normals[j[ok]], p[ok] - target.points[j[ok]]) if ok.any() else np.zeros(0)
    inl = np.abs(r) < 0.05
    rms = float(np.sqrt(np.mean(r[inl] ** 2))) if inl.any() else np.inf
    fitness = float(inl.sum() / max(len(src), 1))
    # Neighbouring points' errors are correlated; scale the information down to ~100 points.
    info = h / SIGMA_M ** 2 * min(1.0, 100.0 / max(int(ok.sum()), 1))
    return pose, info, rms, fitness
