"""Online placement in a rectangular tray.

plan_compact (used by task_node): each box `clearance` from walls or placed
boxes, at the spot and turn (0 or 90 deg) whose perimeter touches walls and
boxes most,
looking ahead greedily over the next known boxes. The caller's `where` (the
scan and the robot) has the last word and may move a spot; None means no
floor spot (the caller then stacks).
plan_placement: two rows along the long walls, kept for the offline
comparison (scripts/dev/packing_compare.py).

Why compact: docs/system_overview.md, section 7.
"""
import itertools

import numpy as np

ROWS = ("back", "front")
CONTACT_STEP_M = 0.01  # touching perimeter is compared in steps of this


def touching(box, others, bounds, reach):
    """Perimeter of box (cx, cy, hx, hy) within `reach` of a wall or of `others`."""
    (x0, x1), (y0, y1) = bounds
    cx, cy, hx, hy = box
    total = 0.0
    if cx - hx - x0 <= reach:
        total += 2 * hy
    if x1 - (cx + hx) <= reach:
        total += 2 * hy
    if cy - hy - y0 <= reach:
        total += 2 * hx
    if y1 - (cy + hy) <= reach:
        total += 2 * hx
    for px, py, phx, phy in others:
        gap_x = max(px - phx - (cx + hx), (cx - hx) - (px + phx))
        gap_y = max(py - phy - (cy + hy), (cy - hy) - (py + phy))
        if 0.0 <= gap_x <= reach:
            total += max(0.0, min(cy + hy, py + phy) - max(cy - hy, py - phy))
        if 0.0 <= gap_y <= reach:
            total += max(0.0, min(cx + hx, px + phx) - max(cx - hx, px - phx))
    return total


def split_rows(placed, bounds, clearance, tol=0.002):
    """The boxes of `placed` [(cx, cy, hx, hy)] in each row (flush against the
    far or near long wall); others are left out.
    """
    (_x0, _x1), (y0, y1) = bounds
    rows = {"back": [], "front": []}
    for b in placed:
        _cx, cy, _hx, hy = b
        if cy + hy >= y1 - clearance - tol:
            rows["back"].append(b)
        elif cy - hy <= y0 + clearance + tol:
            rows["front"].append(b)
    return rows


def _row_spot(hx, hy, row, rows, others, bounds, clearance):
    """Where a box goes next in a row (flush after its last box, against the
    row's wall), or None if it does not fit.
    """
    (x0, x1), (y0, y1) = bounds
    c = clearance
    items = rows[row]
    start = max(px + phx for px, _py, phx, _phy in items) + c if items else x0 + c
    cx = start + hx
    cy = y1 - c - hy if row == "back" else y0 + c + hy
    if cx + hx > x1 - c + 1e-9:
        return None
    for px, py, phx, phy in list(rows["back"]) + list(rows["front"]) + list(others):
        if abs(cx - px) < hx + phx + c - 1e-9 and abs(cy - py) < hy + phy + c - 1e-9:
            return None
    return (cx, cy, hx, hy)


def plan_placement(hx, hy, placed, bounds, clearance, is_free=None, upcoming=(), touch_tol=0.001):
    """Centre (x, y) for the current box (half extents hx, hy), or None.

    placed: [(cx, cy, hx, hy)] boxes on the floor.
    bounds: ((x_min, x_max), (y_min, y_max)), the tray's inner wall faces;
        rows run along x.
    is_free(cx, cy): extra check for the current box (the sensed scan).
    upcoming: [(hx, hy)] boxes known to come next, in pick order.
    """
    base = split_rows(placed, bounds, clearance)
    in_rows = {id(b) for r in base.values() for b in r}
    others = [b for b in placed if id(b) not in in_rows]
    reach = clearance + touch_tol
    best = None
    for choice in itertools.product(ROWS, repeat=1 + len(upcoming)):
        rows = {k: list(v) for k, v in base.items()}
        first = _row_spot(hx, hy, choice[0], rows, others, bounds, clearance)
        if first is None or (is_free is not None and not is_free(first[0], first[1])):
            continue
        rows[choice[0]].append(first)
        fitted = 1
        for (uhx, uhy), row in zip(upcoming, choice[1:]):
            spot = _row_spot(uhx, uhy, row, rows, others, bounds, clearance)
            if spot is not None:
                rows[row].append(spot)
                fitted += 1
        longest = max((max(px + phx for px, _py, phx, _phy in v) if v else bounds[0][0]) for v in rows.values())
        layout = rows["back"] + rows["front"] + others
        contact = sum(touching(b, layout[:i] + layout[i + 1:], bounds, reach) for i, b in enumerate(layout))
        key = (fitted, -round(longest, 4), round(contact, 4))
        if best is None or key > best[0]:
            best = (key, (float(first[0]), float(first[1])))
    return None if best is None else best[1]


def _fits(box, placed, bounds, clearance):
    (x0, x1), (y0, y1) = bounds
    cx, cy, hx, hy = box
    c = clearance - 1e-9
    if cx - hx < x0 + c or cx + hx > x1 - c or cy - hy < y0 + c or cy + hy > y1 - c:
        return False
    return not any(abs(cx - px) < hx + phx + c and abs(cy - py) < hy + phy + c
                   for px, py, phx, phy in placed)


def _flush_spots(hx, hy, placed, bounds, clearance):
    """Spots `clearance` from a wall or a placed box on each axis."""
    (x0, x1), (y0, y1) = bounds
    c = clearance
    xs = {x0 + c + hx, x1 - c - hx}
    ys = {y0 + c + hy, y1 - c - hy}
    for px, py, phx, phy in placed:
        xs |= {px + phx + c + hx, px - phx - c - hx}
        ys |= {py + phy + c + hy, py - phy - c - hy}
    return [(x, y, hx, hy) for x in sorted(xs) for y in sorted(ys)
            if _fits((x, y, hx, hy), placed, bounds, clearance)]


def _strip_area(placed, bounds, far):
    """Tray area from the short wall at the far corner to the block's furthest edge."""
    (x0, x1), (y0, y1) = bounds
    if x1 - x0 >= y1 - y0:
        reach = max(abs((cx + hx if far[0] == x0 else cx - hx) - far[0]) for cx, _cy, hx, _hy in placed)
        return reach * (y1 - y0)
    reach = max(abs((cy + hy if far[1] == y0 else cy - hy) - far[1]) for _cx, cy, _hx, hy in placed)
    return reach * (x1 - x0)


def _bbox_area(placed):
    xs = [v for cx, _cy, hx, _hy in placed for v in (cx - hx, cx + hx)]
    ys = [v for _cx, cy, _hx, hy in placed for v in (cy - hy, cy + hy)]
    return (max(xs) - min(xs)) * (max(ys) - min(ys)) if placed else 0.0


def _layout_contact(placed, bounds, reach):
    return sum(touching(b, placed[:i] + placed[i + 1:], bounds, reach) for i, b in enumerate(placed))


def _orientations(hx, hy, rotate):
    return [(hx, hy), (hy, hx)] if rotate and abs(hx - hy) > 1e-6 else [(hx, hy)]


def plan_compact(hx, hy, placed, bounds, clearance, where=None, upcoming=(), rotate=True,
                 touch_tol=0.001, max_checks=30, group_first=False, needs_push=None, push_area=0.0, strip=False):
    """(cx, cy, fhx, fhy) for the current box, fhx/fhy its footprint half extents
    along x/y (swapped from hx/hy if turned 90 deg), or None if it does not fit.

    Spots flush with walls or placed boxes, best first by: most of the upcoming
    boxes still fitting, most of the box's perimeter touching walls or placed
    boxes, nearest flush (a spot the robot cannot reach flush is scored where it
    ends up), smallest group rectangle, nearest the tray corner farthest from the
    robot (the base at the origin).
    placed: [(cx, cy, fhx, fhy)] boxes on the floor.
    bounds: ((x_min, x_max), (y_min, y_max)), the tray's inner wall faces.
    where(cx, cy, fhx, fhy): where the box really ends up if placed there, (x, y),
        or None if it cannot go there (the scan, the robot); a moved spot is
        scored again where it ends up.
    upcoming: [(hx, hy)] boxes known to come next, in pick order.
    group_first: rank by the group rectangle before the touching perimeter (the boxes grow
        as one block from the far corner instead of spreading to the tray's corners).
    needs_push(cx, cy, fhx, fhy, turned): whether a box set down there (turned 90 deg or
        not) needs a push (the wrist cannot lower it flush); with group_first such a spot
        counts push_area (m2) more block.
    strip: with group_first, the block is measured by how far it reaches from the far
        short wall (it fills the tray's width first; the free space stays one rectangle,
        and the boxes still to come fit far more often) instead of by its bounding
        rectangle.
    """
    reach = clearance + touch_tol
    placed = list(placed)
    (x0, x1), (y0, y1) = bounds
    far = max(((x, y) for x in (x0, x1) for y in (y0, y1)), key=lambda c: c[0] ** 2 + c[1] ** 2)

    def rank(spot, others, moved=0.0, own_hx=hx):
        touch = round(touching(spot, others, bounds, reach) / CONTACT_STEP_M)
        group = -round(_bbox_area(others + [spot]), 4)
        corner = -round(float(np.hypot(spot[0] - far[0], spot[1] - far[1])), 4)
        if group_first:
            size = _strip_area(others + [spot], bounds, far) if strip else _bbox_area(others + [spot])
            if needs_push is not None and push_area > 0.0 and needs_push(*spot, abs(spot[2] - own_hx) > 1e-6):
                size += push_area
            group = -round(size, 4)
            return (group, touch, -round(moved / clearance), corner)
        return (touch, -round(moved / clearance), -round(_bbox_area(others + [spot]), 6), corner)

    def greedy(uhx, uhy, layout):
        opts = [s for a, b in _orientations(uhx, uhy, rotate) for s in _flush_spots(a, b, layout, bounds, clearance)]
        return max(opts, key=lambda s: rank(s, layout, own_hx=uhx), default=None)

    def score(spot, moved=0.0):
        layout = placed + [spot]
        fitted = 1
        for uhx, uhy in upcoming:
            u = greedy(uhx, uhy, layout)
            if u is not None:
                layout.append(u)
                fitted += 1
        return (fitted,) + rank(spot, placed, moved)

    cands = [(score(s), s, False) for a, b in _orientations(hx, hy, rotate)
             for s in _flush_spots(a, b, placed, bounds, clearance)]
    checks = 0
    while cands:
        cands.sort(key=lambda c: c[0])
        _key, spot, known = cands.pop()
        if known or where is None:
            return tuple(float(v) for v in spot)
        if checks >= max_checks:
            continue  # a spot already checked may still be left
        checks += 1
        got = where(*spot)
        if got is None:
            continue
        moved = (float(got[0]), float(got[1]), spot[2], spot[3])
        if abs(moved[0] - spot[0]) < 1e-4 and abs(moved[1] - spot[1]) < 1e-4:
            return tuple(float(v) for v in spot)
        if _fits(moved, placed, bounds, clearance):
            cands.append((score(moved, float(np.hypot(moved[0] - spot[0], moved[1] - spot[1]))), moved, True))
    return None
