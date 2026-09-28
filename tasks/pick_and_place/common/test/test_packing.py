"""plan_compact on hand-built trays. Run with:
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tasks/pick_and_place/common/test
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pick_place_common.packing import plan_compact  # noqa: E402

BOUNDS = ((0.0, 0.40), (0.0, 0.20))
C = 0.003


def overlapping(a, b):
    return abs(a[0] - b[0]) < a[2] + b[2] + C - 1e-9 and abs(a[1] - b[1]) < a[3] + b[3] + C - 1e-9


def test_first_box_goes_in_a_corner():
    x, y, hx, hy = plan_compact(0.05, 0.04, [], BOUNDS, C, rotate=False)
    assert (hx, hy) == (0.05, 0.04)
    assert min(x - hx, BOUNDS[0][1] - (x + hx)) == pytest_approx(C)
    assert min(y - hy, BOUNDS[1][1] - (y + hy)) == pytest_approx(C)


def test_next_box_packs_against_the_first():
    first = plan_compact(0.05, 0.04, [], BOUNDS, C, rotate=False)
    second = plan_compact(0.05, 0.04, [first], BOUNDS, C, rotate=False)
    assert not overlapping(first, second)
    gap = max(abs(first[0] - second[0]) - first[2] - second[2], abs(first[1] - second[1]) - first[3] - second[3])
    assert gap == pytest_approx(C)


def test_turning_fits_a_box_that_does_not_fit_unturned():
    # A 0.09 x 0.18 box in a 0.10 deep tray fits only turned.
    narrow = ((0.0, 0.40), (0.0, 0.10))
    assert plan_compact(0.045, 0.09, [], narrow, C, rotate=False) is None
    turned = plan_compact(0.045, 0.09, [], narrow, C, rotate=True)
    assert turned is not None and turned[2:] == (0.09, 0.045)


def test_is_free_rejects_spots():
    spot = plan_compact(0.05, 0.04, [], BOUNDS, C, is_free=lambda cx, cy, hx, hy: cx > 0.2, rotate=False)
    assert spot[0] > 0.2


def test_never_overlaps_or_leaves_the_tray():
    sizes = [(0.075, 0.06), (0.0675, 0.0675), (0.0525, 0.0525), (0.06, 0.045), (0.045, 0.045),
             (0.0375, 0.0525), (0.0375, 0.0375), (0.03, 0.03)]
    bounds = ((-0.515, -0.085), (-0.565, -0.285))
    placed = []
    for i, (hx, hy) in enumerate(sizes):
        s = plan_compact(hx, hy, placed, bounds, C, upcoming=sizes[i + 1:i + 4])
        if s is None:
            continue
        (x0, x1), (y0, y1) = bounds
        assert s[0] - s[2] >= x0 + C - 1e-9 and s[0] + s[2] <= x1 - C + 1e-9
        assert s[1] - s[3] >= y0 + C - 1e-9 and s[1] + s[3] <= y1 - C + 1e-9
        assert all(not overlapping(s, p) for p in placed)
        placed.append(s)
    assert len(placed) == 8


def pytest_approx(v):
    import pytest
    return pytest.approx(v, abs=1e-9)
