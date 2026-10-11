"""Isolated tests for perception/heightmap.py's dense-heightmap search
(build_dense_grid_xy / find_best_footprint) -- pure functions over
hand-built numpy arrays, no MuJoCo/camera/ROS2 involved at all. This is
deliberately the fast, deterministic layer to get right before wiring
the same search into the live scan-decide-act workflow (mirrors this
project's standalone acados+MuJoCo harness discipline: validate the
algorithm in isolation first).

Run with: python3 -m pytest perception/test/test_heightmap_search.py -v
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from perception import heightmap as cp

FLOOR_Z = 0.0
FLATNESS_TOL = 0.01
FOOTPRINT = (3, 3)


def test_build_dense_grid_xy_shape_and_positions():
    points, shape = cp.build_dense_grid_xy((0.0, 0.05), (0.0, 0.03), 0.01)
    assert shape == (4, 6)  # y: 0,0.01,0.02,0.03 -> 4 rows; x: 0..0.05 -> 6 cols
    assert len(points) == 4 * 6
    # row-major: first row is y=0, x sweeping 0..0.05
    assert points[0] == (0.0, 0.0)
    assert points[5] == (0.05, 0.0)
    assert points[6] == (0.0, 0.01)


def _empty_map(rows, cols, fill=FLOOR_Z):
    return np.full((rows, cols), fill, dtype=float)


def test_flatness_filters_out_bumpy_region_even_if_taller():
    # A clean flat 3x3 plateau at 0.15, versus a "bumpy" 3x3 region whose
    # max (0.30) is higher but whose own height range blows flatness_tol
    # -- a real box could rest on/be grasped from the first, not the
    # second (that's an edge/gap, not a surface).
    heightmap = _empty_map(6, 6)
    heightmap[0:3, 0:3] = 0.15
    heightmap[0:3, 3:6] = np.array([
        [0.30, 0.30, 0.30],
        [0.00, 0.00, 0.00],
        [0.30, 0.30, 0.30],
    ])
    result = cp.find_best_footprint(heightmap, FOOTPRINT, mode="highest",
                                     flatness_tol=FLATNESS_TOL)
    assert result is not None
    row, col, height = result
    assert (row, col) == (0, 0)
    assert abs(height - 0.15) < 1e-9


def test_pickup_highest_selects_tallest_flat_region_then_next_after_removal():
    # Two flat "columns": one at 0.2 (top layer still there), one at 0.1
    # (already down to its last layer), separated by floor so windows
    # can't accidentally straddle both.
    heightmap = _empty_map(8, 8)
    heightmap[0:3, 0:3] = 0.2
    heightmap[0:3, 5:8] = 0.1

    row, col, height = cp.find_best_footprint(
        heightmap, FOOTPRINT, mode="highest", flatness_tol=FLATNESS_TOL)
    assert (row, col) == (0, 0)
    assert abs(height - 0.2) < 1e-9

    # Simulate picking that box: its column drops back to floor, exactly
    # as a real sensed column would once the top box is gone.
    heightmap[0:3, 0:3] = FLOOR_Z
    row, col, height = cp.find_best_footprint(
        heightmap, FOOTPRINT, mode="highest", flatness_tol=FLATNESS_TOL)
    assert (row, col) == (0, 5)
    assert abs(height - 0.1) < 1e-9


def test_placement_lowest_prefers_empty_over_occupied():
    heightmap = _empty_map(6, 6)
    heightmap[0:3, 0:3] = 0.5  # already occupied region
    row, col, height = cp.find_best_footprint(
        heightmap, FOOTPRINT, mode="lowest", flatness_tol=FLATNESS_TOL,
        floor_z=FLOOR_Z)
    assert abs(height - FLOOR_Z) < 1e-9
    # must not be inside the occupied region
    assert not (0 <= row < 3 and 0 <= col < 3)


def test_placement_adjacency_prefers_corner_over_midwall_over_open():
    # 9x9 empty tray, footprint 3x3 -> candidate top-lefts at rows/cols
    # {0, 3, 6}. All tied at height=floor_z, so adjacency alone decides.
    heightmap = _empty_map(9, 9)

    # Corner (row=0,col=0): west+north off-grid -> 2 wall bonuses.
    # Mid-wall (row=0,col=3): only north off-grid -> 1 wall bonus.
    # Open/center (row=3,col=3): no side off-grid -> 0 wall bonus.
    # The global search should pick the corner.
    row, col, height = cp.find_best_footprint(
        heightmap, FOOTPRINT, mode="lowest", flatness_tol=FLATNESS_TOL,
        floor_z=FLOOR_Z)
    assert (row, col) in {(0, 0), (0, 6), (6, 0), (6, 6)}  # any true corner

    # Now place a box at that corner and confirm the NEXT placement
    # prefers the cell adjacent to it over an equally-empty, isolated
    # cell elsewhere. Default neighbor_tol (== flatness_tol) matters
    # here: it must stay tight enough that the already-placed box itself
    # (height 0.1) is never mistaken for a "near-lowest" empty candidate
    # -- a large neighbor_tol would sweep it back in as its own
    # candidate, since 0.1 - 0.0 would then read as "close enough."
    heightmap[row:row + 3, col:col + 3] = 0.1  # placed box, one layer tall
    row2, col2, height2 = cp.find_best_footprint(
        heightmap, FOOTPRINT, mode="lowest", flatness_tol=FLATNESS_TOL,
        floor_z=FLOOR_Z)
    assert abs(height2 - FLOOR_Z) < 1e-9  # still filling flat, not stacking yet
    # adjacent to the first box (shares a full side with it)
    touches_first_box = (
        (row2 == row and abs(col2 - col) == 3) or
        (col2 == col and abs(row2 - row) == 3)
    )
    assert touches_first_box, f"expected a neighbor of ({row},{col}), got ({row2},{col2})"


def test_placement_adjacency_weighs_contact_length_not_just_presence():
    # A spot that only grazes ONE cell of a neighbor's edge must not
    # outscore a genuinely fresh, untouched corner -- otherwise the
    # search accepts a barely-touching placement that leaves a sliver of
    # floor behind it too narrow for any future box, the exact source of
    # wasted destination-tray space with mixed box sizes (a full-edge
    # touch is still worth preferring, see the adjacency test above).
    heightmap = _empty_map(9, 9)
    # (0, 3)'s west ring is column 2, rows 0-2 (3 cells) -- occupy only
    # one of them. Under the old binary "any contact" scoring this
    # spot (1 wall + 1 neighbor-of-any-size = 3.0) used to beat a fresh
    # double-wall corner (2.0) outright; proportional scoring instead
    # gives it only 1 + 2*(1/3) = 1.667, below the fresh corner's 2.0.
    heightmap[0, 2] = 0.5

    row, col, height = cp.find_best_footprint(
        heightmap, FOOTPRINT, mode="lowest", flatness_tol=FLATNESS_TOL,
        floor_z=FLOOR_Z)
    assert (row, col) != (0, 3), "picked the barely-touching spot over a fresh corner"
    assert (row, col) in {(0, 0), (0, 6), (6, 0), (6, 6)}


def test_placement_fills_available_floor_before_any_stacking():
    # A 2x3-footprint-sized floor (mirrors this session's 2x3 destination
    # grid), 2 layers tall -- 6 footprints' worth of floor area, 8
    # "boxes" to place. This is a genuine continuous search (unlike the
    # old fixed 6-slot design), so later placements are free to land at
    # any offset, not just the 6 grid-aligned slot origins -- once
    # several neighboring layer-0 boxes are level, a window straddling
    # more than one of them is just as flat as one sitting exactly on a
    # single box, and both are equally valid rests. So the invariant
    # that actually matters (and is what the fixed-slot design's
    # counting argument was really standing in for) isn't "which exact
    # positions get used," it's heights: every bit of floor area gets
    # used at layer 0 before any box goes to layer 1.
    fh, fw = FOOTPRINT
    n_rows, n_cols = 2 * fh, 3 * fw
    heightmap = _empty_map(n_rows, n_cols)
    layer_pitch = 0.1
    n_footprints = (n_rows // fh) * (n_cols // fw)  # 6

    heights = []
    for _ in range(8):
        row, col, height = cp.find_best_footprint(
            heightmap, FOOTPRINT, mode="lowest", flatness_tol=FLATNESS_TOL,
            floor_z=FLOOR_Z)
        heights.append(height)
        heightmap[row:row + fh, col:col + fw] = height + layer_pitch

    layer0 = [h for h in heights if abs(h - FLOOR_Z) < 1e-9]
    layer1 = [h for h in heights if abs(h - (FLOOR_Z + layer_pitch)) < 1e-9]
    assert len(layer0) == n_footprints, (
        f"expected all {n_footprints} footprints' worth of floor filled at "
        f"layer 0 before any stacking, got {len(layer0)}"
    )
    assert len(layer1) == 8 - n_footprints
    # every layer-0 placement happened before every layer-1 one
    assert heights == sorted(heights), f"placements weren't strictly bottom-up: {heights}"


# --- find_topmost_boxes: variable-size source-side detection ---------------

# Three distinct boxes, deliberately different sizes/heights, spatially
# separated by at least one floor row/col so they can't touch (that
# specific interaction is its own test below).
_BOX_A = dict(row0=0, col0=0, row1=3, col1=5, height=0.20)   # 3x5=15 cells
_BOX_B = dict(row0=0, col0=7, row1=4, col1=12, height=0.15)  # 4x5=20 cells
_BOX_C = dict(row0=8, col0=2, row1=12, col1=6, height=0.30)  # 4x4=16 cells


def _multi_box_heightmap():
    heightmap = np.full((12, 12), FLOOR_Z)
    for box in (_BOX_A, _BOX_B, _BOX_C):
        heightmap[box["row0"]:box["row1"], box["col0"]:box["col1"]] = box["height"]
    return heightmap


def test_find_topmost_boxes_detects_distinct_sizes_and_positions():
    heightmap = _multi_box_heightmap()
    records = cp.find_topmost_boxes(heightmap, FLOOR_Z, FLATNESS_TOL,
                                     min_footprint_cells=4)
    assert len(records) == 3

    found = {(r0, c0, r1, c1): (h, a) for r0, c0, r1, c1, h, a in records}
    for box in (_BOX_A, _BOX_B, _BOX_C):
        key = (box["row0"], box["col0"], box["row1"], box["col1"])
        assert key in found, f"missing detection for {box}"
        height, area = found[key]
        expected_area = (box["row1"] - box["row0"]) * (box["col1"] - box["col0"])
        assert abs(height - box["height"]) < 1e-9
        assert area == expected_area


def test_find_topmost_boxes_largest_area_is_box_b():
    heightmap = _multi_box_heightmap()
    records = cp.find_topmost_boxes(heightmap, FLOOR_Z, FLATNESS_TOL,
                                     min_footprint_cells=4)
    best = max(records, key=lambda r: r[5])  # area is index 5
    row0, col0, row1, col1, height, area = best
    assert (row0, col0, row1, col1) == (_BOX_B["row0"], _BOX_B["col0"],
                                          _BOX_B["row1"], _BOX_B["col1"])
    assert area == 20  # the biggest of the three


def test_find_topmost_boxes_same_height_adjacent_boxes_merge():
    # Two boxes, same height, sharing a border -- a real, physically
    # honest ambiguity for depth-only sensing (documented in
    # find_topmost_boxes' own docstring), not a bug: confirm this
    # produces exactly one merged region spanning both, rather than
    # silently misbehaving some other way.
    heightmap = np.full((6, 10), FLOOR_Z)
    heightmap[0:4, 0:3] = 0.2   # box 1: rows 0-3, cols 0-2
    heightmap[0:4, 3:6] = 0.2   # box 2: rows 0-3, cols 3-5, touching box 1
    records = cp.find_topmost_boxes(heightmap, FLOOR_Z, FLATNESS_TOL,
                                     min_footprint_cells=4)
    assert len(records) == 1
    row0, col0, row1, col1, height, area = records[0]
    assert (row0, col0, row1, col1) == (0, 0, 4, 6)
    assert area == 24


def test_find_topmost_boxes_reveals_next_layer_after_removal():
    heightmap = _multi_box_heightmap()
    records = cp.find_topmost_boxes(heightmap, FLOOR_Z, FLATNESS_TOL,
                                     min_footprint_cells=4)
    best = max(records, key=lambda r: r[5])
    row0, col0, row1, col1, _, _ = best
    heightmap[row0:row1, col0:col1] = FLOOR_Z  # simulate having picked it

    records2 = cp.find_topmost_boxes(heightmap, FLOOR_Z, FLATNESS_TOL,
                                      min_footprint_cells=4)
    assert len(records2) == 2
    remaining_keys = {(r0, c0, r1, c1) for r0, c0, r1, c1, h, a in records2}
    assert (row0, col0, row1, col1) not in remaining_keys


def test_find_topmost_boxes_filters_small_noise():
    heightmap = _multi_box_heightmap()
    heightmap[6, 6] = 0.05  # a single-cell speck, not a real object
    records = cp.find_topmost_boxes(heightmap, FLOOR_Z, FLATNESS_TOL,
                                     min_footprint_cells=4)
    assert len(records) == 3  # the speck (area 1) is below min_footprint_cells


def _overhung_pile():
    # Big low box A (rows 0-5, cols 0-7, h 0.09); small tall box B (h 0.195)
    # resting mostly on a neighbour but overhanging A's last column.
    heightmap = np.full((8, 14), FLOOR_Z)
    heightmap[0:6, 0:8] = 0.09     # A
    heightmap[0:5, 9:13] = 0.09    # neighbour under B, separated from A by a floor column
    heightmap[0:4, 7:11] = 0.195   # B, covering A's col 7
    return heightmap


def test_pick_order_puts_box_under_a_higher_neighbour_last():
    heightmap = _overhung_pile()
    records = cp.find_topmost_boxes(heightmap, FLOOR_Z, FLATNESS_TOL,
                                     min_footprint_cells=4, min_fill_frac=0.6)
    by_height = sorted(records, key=lambda r: r[4])
    big_low = max((r for r in records if abs(r[4] - 0.09) < 1e-9), key=lambda r: r[5])
    tall = by_height[-1]
    assert big_low[5] > tall[5]  # area alone would pick the low box first
    order = cp.pick_order(heightmap, records, FLATNESS_TOL)
    assert order[0] == tall
    assert order.index(big_low) > order.index(tall)


def test_pick_order_is_area_order_without_touching_higher_tops():
    heightmap = _multi_box_heightmap()
    records = cp.find_topmost_boxes(heightmap, FLOOR_Z, FLATNESS_TOL,
                                     min_footprint_cells=4)
    order = cp.pick_order(heightmap, records, FLATNESS_TOL)
    assert order == sorted(records, key=lambda b: (b[5], b[4]), reverse=True)


def test_find_topmost_boxes_filters_non_rectangular_shapes():
    # An L-shape: 10 occupied cells inside a 4x4=16 bounding box
    # (fill fraction 0.625) -- not what a rigid box's top face looks
    # like, should be rejected even though it clears the area filter.
    heightmap = np.full((8, 8), FLOOR_Z)
    heightmap[0:4, 0:1] = 0.2       # vertical arm: 4 cells
    heightmap[3:4, 0:4] = 0.2       # horizontal arm: 4 more new cells (row 3 col 0 shared)
    # total occupied cells = 4 + 3 (col0 row3 already counted) = 7, bbox 4x4=16
    records = cp.find_topmost_boxes(heightmap, FLOOR_Z, FLATNESS_TOL,
                                     min_footprint_cells=4, min_fill_frac=0.85)
    assert records == []
