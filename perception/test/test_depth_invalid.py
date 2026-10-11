"""Invalid depth pixels (0, as depth cameras report no reading) in the perception
functions, and the simulated camera noise that produces them.
Run with: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest perception/test
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tasks/pick_and_place/common")]
from perception import heightmap, obstacle_detection  # noqa: E402
from pick_place_common.depth_noise import add_depth_noise  # noqa: E402

CAM_POS = np.array([0.0, 0.0, 1.0])
CAM_MAT = np.eye(3)  # looking straight down: camera -z is world -z


def test_fill_invalid_takes_the_nearest_valid_pixel():
    d = np.full((5, 5), 0.8)
    d[2, 2] = 0.0
    d[0, 0] = 0.5
    filled = heightmap.fill_invalid(d)
    assert filled[2, 2] == 0.8 and filled[0, 0] == 0.5
    assert heightmap.fill_invalid(np.full((3, 3), 0.7)) is not None


def test_detect_blobs_ignores_invalid_pixels():
    depth = np.full((60, 80), 1.0)  # empty floor at z = 0
    depth[10:14, 10:30] = 0.0  # a patch of no readings would unproject to the camera height
    blobs = obstacle_detection.detect_blobs(depth, CAM_POS, CAM_MAT, 60.0, 0.0, 0.05, min_blob_px=5)
    assert len(blobs) == 0


def test_top_extent_ignores_invalid_pixels():
    depth = np.full((120, 160), 1.0)
    depth[40:80, 60:100] = 0.9  # a top 0.1 m above the floor
    clean = heightmap.top_extent(depth, CAM_POS, CAM_MAT, 60.0, 160, 120, (-1, 1), (-1, 1), 0.1, 0.005)
    holed = depth.copy()
    holed[55:58, 70:90] = 0.0
    holed[40, 60:100] = 0.0  # an edge row lost
    got = heightmap.top_extent(holed, CAM_POS, CAM_MAT, 60.0, 160, 120, (-1, 1), (-1, 1), 0.1, 0.005)
    assert got is not None
    pitch = 0.9 / ((120 / 2) / np.tan(np.radians(30)))
    assert all(abs(a - b) <= pitch + 1e-9 for a, b in zip(clean, got))


def test_depth_noise_size_and_invalid_edges():
    depth = np.full((120, 160), 0.65)
    depth[40:80, 60:100] = 0.55
    noisy = add_depth_noise(depth, np.random.default_rng(0))
    valid = noisy > 0
    flat = valid.copy()
    flat[38:82, 58:102] = False
    resid = noisy[flat] - np.median(noisy[flat])
    assert 0.0003 < resid.std() < 0.002  # ~0.7 mm at 0.65 m
    edge_rows = noisy[39:41, 60:100]
    assert (edge_rows == 0).mean() > 0.2  # depth edges lose pixels
    assert (noisy[5:30, 5:50] == 0).mean() < 0.02
