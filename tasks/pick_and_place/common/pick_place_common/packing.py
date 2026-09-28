"""Online placement in a rectangular tray.

plan_compact (used by task_node): each box `clearance` from walls or placed
boxes, at the spot and turn (0 or 90 deg) that keeps the placed group's
bounding rectangle smallest, then most touching perimeter, looking ahead
greedily over the next known boxes. The caller's `is_free` check (the scan)
has the last word; None means no floor spot (the caller then stacks).
plan_placement: two rows along the long walls, kept for the offline
comparison (scripts/dev/packing_compare.py).

Why compact: docs/system_overview.md, section 7.
"""
import itertools

ROWS = ("back", "front")


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


def _bbox_area(placed):
    xs = [v for cx, _cy, hx, _hy in placed for v in (cx - hx, cx + hx)]
    ys = [v for _cx, cy, _hx, hy in placed for v in (cy - hy, cy + hy)]
    return (max(xs) - min(xs)) * (max(ys) - min(ys)) if placed else 0.0


def _layout_contact(placed, bounds, reach):
    return sum(touching(b, placed[:i] + placed[i + 1:], bounds, reach) for i, b in enumerate(placed))


def _orientations(hx, hy, rotate):
    return [(hx, hy), (hy, hx)] if rotate and abs(hx - hy) > 1e-6 else [(hx, hy)]


def plan_compact(hx, hy, placed, bounds, clearance, is_free=None, upcoming=(), rotate=True,
                 touch_tol=0.001):
    """(cx, cy, fhx, fhy) for the current box, fhx/fhy its footprint half extents
    along x/y (swapped from hx/hy if turned 90 deg), or None if it does not fit.

    placed: [(cx, cy, fhx, fhy)] boxes on the floor.
    bounds: ((x_min, x_max), (y_min, y_max)), the tray's inner wall faces.
    is_free(cx, cy, fhx, fhy): extra check for the current box (the sensed scan).
    upcoming: [(hx, hy)] boxes known to come next, in pick order.
    """
    reach = clearance + touch_tol
    placed = list(placed)

    def greedy(uhx, uhy, layout):
        opts = [s for a, b in _orientations(uhx, uhy, rotate) for s in _flush_spots(a, b, layout, bounds, clearance)]
        if not opts:
            return None
        return min(opts, key=lambda s: (round(_bbox_area(layout + [s]), 6),
                                        -round(_layout_contact(layout + [s], bounds, reach), 4)))

    best = None
    for a, b in _orientations(hx, hy, rotate):
        for spot in _flush_spots(a, b, placed, bounds, clearance):
            if is_free is not None and not is_free(*spot):
                continue
            layout = placed + [spot]
            fitted = 1
            for uhx, uhy in upcoming:
                u = greedy(uhx, uhy, layout)
                if u is not None:
                    layout.append(u)
                    fitted += 1
            key = (fitted, -round(_bbox_area(layout), 6), round(_layout_contact(layout, bounds, reach), 4))
            if best is None or key > best[0]:
                best = (key, tuple(float(v) for v in spot))
    return None if best is None else best[1]
