"""A held box in the tray scan: its silhouette is projected from its known pose,
and heightmap cells it may hide are unknown (NaN), never floor.
Run with: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest perception/test
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO)]
from perception import heightmap  # noqa: E402

CAM_POS = np.array([0.0, 0.0, 0.75])
CAM_MAT = np.eye(3)
FOVY, W, H = 55.0, 320, 240


def _render(box_top, box_half, box_height, floor_boxes=()):
    """Depth image of a floor at z = 0 with floor_boxes ((x0, x1, y0, y1, top))
    and a held box hanging under the camera (top at box_top)."""
    f = (H / 2.0) / np.tan(np.radians(FOVY) / 2.0)
    vs, us = np.mgrid[0:H, 0:W]
    ray_x, ray_y = (us - W / 2.0) / f, -(vs - H / 2.0) / f
    depth = np.full((H, W), CAM_POS[2])
    for x0, x1, y0, y1, top in floor_boxes:
        d = CAM_POS[2] - top
        on = (ray_x * d > x0) & (ray_x * d < x1) & (ray_y * d > y0) & (ray_y * d < y1)
        depth = np.where(on, np.minimum(depth, d), depth)
    for z in np.linspace(box_top - box_height, box_top, 20):
        d = CAM_POS[2] - z
        on = (np.abs(ray_x * d) < box_half[0]) & (np.abs(ray_y * d) < box_half[1])
        depth = np.where(on, np.minimum(depth, d), depth)
    return depth


def _corners(top, half, height, grow=0.01):
    return [np.array([sx * (half[0] + grow), sy * (half[1] + grow), z])
            for sx in (-1, 1) for sy in (-1, 1) for z in (top, top - height - grow)]


def test_hidden_cells_are_unknown_and_seen_cells_are_right():
    half, top, height = (0.05, 0.04), 0.5, 0.10
    placed = (0.20, 0.30, -0.10, 0.00, 0.08)
    depth = _render(top, half, height, [placed])
    pts, shape = heightmap.build_dense_grid_xy((-0.35, 0.35), (-0.25, 0.25), 0.01)
    mask = heightmap.box_silhouette(_corners(top, half, height), CAM_POS, CAM_MAT, FOVY, W, H)
    h, _ = heightmap.infer_heights_parallax_corrected(depth, CAM_POS, CAM_MAT, FOVY, W, H, pts, 0.05, 0.0,
                                                      hidden=mask, top_z=0.2)
    pts = np.asarray(pts)
    under = (np.abs(pts[:, 0]) < 0.05) & (np.abs(pts[:, 1]) < 0.04)
    assert np.isnan(h[under]).all()  # directly under the box: never floor
    seen = ~np.isnan(h)
    assert seen.sum() > 0.5 * len(h)
    on_placed = (pts[:, 0] > 0.21) & (pts[:, 0] < 0.29) & (pts[:, 1] > -0.09) & (pts[:, 1] < -0.01)
    assert np.allclose(h[seen & on_placed], 0.08, atol=0.005)
    assert np.all(h[seen & ~on_placed] < 0.085)  # the held box itself never shows up


def test_no_mask_keeps_the_old_result():
    depth = _render(0.5, (0.05, 0.04), 0.10)
    pts, _ = heightmap.build_dense_grid_xy((-0.2, 0.2), (-0.2, 0.2), 0.01)
    a, _ = heightmap.infer_heights_parallax_corrected(depth, CAM_POS, CAM_MAT, FOVY, W, H, pts, 0.05, 0.0)
    assert not np.isnan(a).any()
