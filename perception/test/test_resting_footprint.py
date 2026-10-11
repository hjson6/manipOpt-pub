import numpy as np

from perception.heightmap import find_resting_footprint

TOL = 0.01


def test_floor_spot_first():
    h = np.zeros((20, 20))
    h[:10, :] = 0.1
    r, c, z, s = find_resting_footprint(h, (8, 8), TOL)
    assert z == 0.0 and r >= 10 and s == 1.0


def test_bridges_two_boxes_across_a_gap():
    h = np.zeros((20, 30))
    h[2:18, 2:14] = 0.09  # two boxes of the same height, a two-cell gap between them
    h[2:18, 16:28] = 0.09
    r, c, z, s = find_resting_footprint(h, (14, 24), TOL)
    assert abs(z - 0.09) < 1e-9 and 0.8 < s < 1.0


def test_not_on_one_edge():
    h = np.zeros((20, 20))
    h[:, :6] = 0.09  # a narrow box under one side only: the box would tip
    assert find_resting_footprint(h, (20, 20), TOL) is None
    h[:, 14:] = 0.09  # a second box under the other side: now it is carried
    assert find_resting_footprint(h, (20, 20), TOL) is not None


def test_small_overhang_is_fine():
    h = np.zeros((20, 20))
    h[:, :] = 0.3  # nothing low anywhere: only the box top counts
    h[2:18, 2:16] = 0.09
    h[:, 16:] = 0.0  # ...and a gap beside it the box may overhang
    r, c, z, s = find_resting_footprint(h, (16, 17), TOL)
    assert abs(z - 0.09) < 1e-9 and s < 1.0


def test_unknown_cells_rejected():
    h = np.zeros((10, 10))
    h[:, :] = np.nan
    assert find_resting_footprint(h, (5, 5), TOL) is None
