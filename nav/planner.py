"""Route planning for the differential-drive base on the saved map: a costmap (the
map's occupied and unknown cells, the distance to them), the chassis as three
circles, and Hybrid A* over (x, y, heading) with the base's own moves (0.2 m
forward, straight or on a 0.8 m radius arc; 15 deg turns in place; short, costly
reversing; each change of steering costs a little, against S-bends), a 2D grid search
as the heuristic, and a shot to the goal whenever it is free: rotate-straight-rotate, or
for a docking line's start (entry=True) first rotate-straight-arc onto the line.
Framework-free (numpy, scipy).

See docs/implementation_notes.md#plannerpy.
"""
import heapq

import numpy as np
from scipy import ndimage

# The chassis (0.80 x 0.56 m, the arm tucked inside) as three circles along its axis.
FOOTPRINT_X = (-0.27, 0.0, 0.27)
FOOTPRINT_R = 0.31
STEP_M = 0.2
ARC_R_M = 0.8
TURN_RAD = np.radians(15)
XY_RES = 0.1
YAW_BINS = 72
TURN_COST = 0.15  # per 15 deg turn in place, in metres
ARC_COST = 1.05
REVERSE_COST = 3.0
STEER_CHANGE_COST = 0.3  # metres, per change of the steering (straight, left, right, turn) between moves
CLEARANCE_SOFT_M = 0.35  # below this clearance (circle edge to obstacle) the cost rises
CLEARANCE_WEIGHT = 1.0
GOAL_XY_M = 0.05
ENTRY_R_M = 0.6  # the arc onto a docking line
ENTRY_LEAD_M = (0.0, 0.3, 0.6, 1.0)  # straight along the line before its start, tried in turn
ENTRY_SWEEP_MAX = np.radians(150)
SHOT_REVERSE_M = 0.5  # a shot this short may back to the goal
REJOIN_MIN_M = 0.3  # a curve back onto a docking line: at least this long, and long enough that its
# bend stays wider than REJOIN_RADIUS_MIN_M (an S over offset e: about sqrt(8 R e))
REJOIN_YAW_MAX = np.radians(60)  # heading off the line's more than this: no curve
REJOIN_RADIUS_MIN_M = 0.3
MAX_EXPANSIONS = 200_000


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def rotate_straight_rotate(p, goal, reverse_ok=False, min_dist=0.0):
    """Poses turning in place to face the goal (or away from it, driving back, if that is
    the smaller turn and reverse_ok), straight to it, turning to its heading. Closer than
    min_dist: the turn to its heading only."""
    p, goal = np.asarray(p, dtype=float), np.asarray(goal, dtype=float)
    d = goal[:2] - p[:2]
    dist = float(np.hypot(*d))
    if dist <= max(min_dist, 1e-6):
        dist, head = 0.0, p[2]
    else:
        head = float(np.arctan2(d[1], d[0]))
        if reverse_ok and abs(wrap(head - p[2])) > np.pi / 2:
            head = wrap(head + np.pi)
        head = p[2] + wrap(head - p[2])
    rot1 = np.linspace(0, head - p[2], max(2, int(abs(head - p[2]) / 0.1) + 1))
    line = np.linspace(0, dist, max(2, int(dist / 0.05) + 1))
    ux, uy = (d / dist) if dist > 0 else (0.0, 0.0)
    rot2 = np.linspace(0, wrap(goal[2] - head), max(2, int(abs(wrap(goal[2] - head)) / 0.1) + 1))
    end = goal[:2] if dist > 0 else p[:2]
    return np.vstack([np.column_stack([np.full(len(rot1), p[0]), np.full(len(rot1), p[1]), p[2] + rot1]),
                      np.column_stack([p[0] + line * ux, p[1] + line * uy, np.full(len(line), head)]),
                      np.column_stack([np.full(len(rot2), end[0]), np.full(len(rot2), end[1]), head + rot2])])


def rotate_straight_arc(p, goal, r=ENTRY_R_M, lead=0.0):
    """Poses turning in place at p, straight, then on an arc of radius r that ends tangent to
    the line through goal along its heading, lead before it, then straight along it to goal:
    a docking line joined without stopping to turn. None if p is too near that arc's circle
    or the arc would sweep more than ENTRY_SWEEP_MAX."""
    p, goal = np.asarray(p, dtype=float), np.asarray(goal, dtype=float)
    u = np.array([np.cos(goal[2]), np.sin(goal[2])])
    n = np.array([-u[1], u[0]])
    side = 1.0 if n @ (p[:2] - goal[:2]) >= 0.0 else -1.0  # the arc turns towards the line: left if p is left of it
    t = goal[:2] - lead * u
    c = t + side * r * n
    dc = c - p[:2]
    dist = float(np.hypot(*dc))
    if dist <= r + 0.05:
        return None
    alpha = np.arcsin(r / dist)
    head = float(np.arctan2(dc[1], dc[0]) - side * alpha)
    a = p[:2] + np.sqrt(dist * dist - r * r) * np.array([np.cos(head), np.sin(head)])
    b0 = float(np.arctan2(*(a - c)[::-1]))
    sweep = (float(np.arctan2(*(t - c)[::-1])) - b0) * side % (2 * np.pi)
    if sweep > ENTRY_SWEEP_MAX:
        return None
    head = p[2] + wrap(head - p[2])
    rot = np.linspace(0, head - p[2], max(2, int(abs(head - p[2]) / 0.1) + 1))
    straight = np.linspace(0.0, 1.0, max(2, int(np.hypot(*(a - p[:2])) / 0.05) + 1))[:, None]
    ang = b0 + side * np.linspace(0.0, sweep, max(2, int(sweep * r / 0.05) + 1))
    arc = np.column_stack([c[0] + r * np.cos(ang), c[1] + r * np.sin(ang), head + (ang - b0)])
    tail = np.linspace(0.0, 1.0, max(2, int(lead / 0.05) + 1))[:, None]
    return np.vstack([np.column_stack([np.full(len(rot), p[0]), np.full(len(rot), p[1]), p[2] + rot]),
                      np.column_stack([p[:2] + straight * (a - p[:2]), np.full(len(straight), head)]),
                      arc,
                      np.column_stack([t + tail * (goal[:2] - t), np.full(len(tail), arc[-1, 2])])])


def curve_onto_line(p, goal, end_before):
    """Poses driven forward on one smooth curve (a cubic Hermite) from p onto the line
    through goal along its heading, joining it end_before (m) or more before goal, on its
    heading: no turn on the spot. None if p heads off the line by more than REJOIN_YAW_MAX,
    is too near goal, or the curve bends tighter than REJOIN_RADIUS_MIN_M."""
    p, goal = np.asarray(p, dtype=float), np.asarray(goal, dtype=float)
    u = np.array([np.cos(goal[2]), np.sin(goal[2])])
    n = np.array([-u[1], u[0]])
    rel = p[:2] - goal[:2]
    along, lateral = float(rel @ u), float(rel @ n)
    if abs(wrap(p[2] - goal[2])) > REJOIN_YAW_MAX:
        return None
    join = along + max(REJOIN_MIN_M, np.sqrt(8.0 * REJOIN_RADIUS_MIN_M * abs(lateral)))
    if join > -end_before:
        return None
    j = goal[:2] + join * u
    k = float(np.hypot(*(j - p[:2])))
    t0, t1 = k * np.array([np.cos(p[2]), np.sin(p[2])]), k * u
    s = np.linspace(0.0, 1.0, max(3, int(k / 0.02) + 1))[:, None]
    xy = ((2 * s**3 - 3 * s**2 + 1) * p[:2] + (s**3 - 2 * s**2 + s) * t0 + (-2 * s**3 + 3 * s**2) * j
          + (s**3 - s**2) * t1)
    d1 = (6 * s**2 - 6 * s) * p[:2] + (3 * s**2 - 4 * s + 1) * t0 + (-6 * s**2 + 6 * s) * j + (3 * s**2 - 2 * s) * t1
    d2 = (12 * s - 6) * p[:2] + (6 * s - 4) * t0 + (-12 * s + 6) * j + (6 * s - 2) * t1
    speed = np.hypot(d1[:, 0], d1[:, 1])
    curvature = np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / np.maximum(speed, 1e-9) ** 3
    if curvature.max() > 1.0 / REJOIN_RADIUS_MIN_M:
        return None
    head = np.arctan2(d1[:, 1], d1[:, 0])
    head = p[2] + np.r_[0.0, np.cumsum(wrap(np.diff(head)))] + wrap(head[0] - p[2])
    return np.column_stack([xy, head])


class Costmap:
    """Distance (m) from each cell to the nearest occupied or unknown cell of the map."""

    def __init__(self, grid):
        self.origin = grid.origin
        self.res = grid.res
        blocked = ~grid.free
        self.dist = ndimage.distance_transform_edt(~blocked) * grid.res
        self.shape = blocked.shape

    def with_obstacles(self, centres, radii):
        """A copy with discs (people standing in the way) blocked as well."""
        out = Costmap.__new__(Costmap)
        out.origin, out.res, out.shape = self.origin, self.res, self.shape
        h, w = self.shape
        ys, xs = np.mgrid[0:h, 0:w]
        cx = self.origin[0] + (xs + 0.5) * self.res
        cy = self.origin[1] + (ys + 0.5) * self.res
        out.dist = self.dist.copy()
        for (x, y), r in zip(centres, np.broadcast_to(radii, len(centres))):
            out.dist = np.minimum(out.dist, np.maximum(np.hypot(cx - x, cy - y) - r, 0.0))
        return out

    def clearance(self, xy):
        """Distance to the nearest blocked cell at map points (0 outside the map)."""
        u = np.floor((np.atleast_2d(xy) - self.origin) / self.res).astype(int)
        h, w = self.shape
        ok = (u[:, 0] >= 0) & (u[:, 0] < w) & (u[:, 1] >= 0) & (u[:, 1] < h)
        out = np.zeros(len(u))
        out[ok] = self.dist[u[ok, 1], u[ok, 0]]
        return out

    def footprint_clearance(self, poses, x=FOOTPRINT_X, r=FOOTPRINT_R):
        """Smallest clearance of the chassis's circles at each pose (x, y, yaw); negative:
        a collision."""
        poses = np.atleast_2d(poses)
        c, s = np.cos(poses[:, 2]), np.sin(poses[:, 2])
        pts = np.concatenate([np.column_stack([poses[:, 0] + c * dx, poses[:, 1] + s * dx]) for dx in x])
        return self.clearance(pts).reshape(len(x), -1).min(axis=0) - r


class HybridAStar:
    def __init__(self, costmap, margin=0.08):
        self.cm = costmap
        self.margin = margin

    def _free(self, poses):
        return np.all(self.cm.footprint_clearance(poses) >= self.margin)

    def _key(self, p):
        return (int(round(p[0] / XY_RES)), int(round(p[1] / XY_RES)), int(round(wrap(p[2]) / (2 * np.pi / YAW_BINS))) % YAW_BINS)

    def _heuristic_grid(self, goal):
        """Shortest free 2D distances to the goal (8-connected Dijkstra over cells where the
        robot's inner circle fits): the heuristic, aware of walls and tables."""
        res = self.cm.res
        ok = self.cm.dist >= 0.28 + self.margin
        h, w = ok.shape
        g = np.floor((np.asarray(goal[:2]) - self.cm.origin) / res).astype(int)
        dist = np.full(ok.shape, np.inf)
        if not (0 <= g[0] < w and 0 <= g[1] < h):
            return dist
        dist[g[1], g[0]] = 0.0
        heap = [(0.0, g[1], g[0])]
        steps = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                 (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]
        while heap:
            d, r, c = heapq.heappop(heap)
            if d > dist[r, c]:
                continue
            for dr, dc, w_ in steps:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and (ok[rr, cc] or (rr, cc) == (g[1], g[0])):
                    nd = d + w_ * res
                    if nd < dist[rr, cc]:
                        dist[rr, cc] = nd
                        heapq.heappush(heap, (nd, rr, cc))
        return dist

    def _moves(self, p):
        """(next pose, samples along the move, cost, kind, steer) for each move from p."""
        x, y, th = p
        out = []
        for kind, k in (("fwd", 0.0), ("fwd", 1 / ARC_R_M), ("fwd", -1 / ARC_R_M), ("rev", 0.0)):
            sign = -1.0 if kind == "rev" else 1.0
            s = np.linspace(STEP_M / 4, STEP_M, 4) * sign
            if k == 0.0:
                pts = np.column_stack([x + s * np.cos(th), y + s * np.sin(th), np.full(4, th)])
            else:
                a = th + s * k
                pts = np.column_stack([x + (np.sin(a) - np.sin(th)) / k, y - (np.cos(a) - np.cos(th)) / k, a])
            cost = STEP_M * (REVERSE_COST if kind == "rev" else (ARC_COST if k else 1.0))
            out.append((pts[-1], pts, cost, kind, (kind, np.sign(k))))
        for d in (TURN_RAD, -TURN_RAD):
            out.append((np.array([x, y, th + d]), np.array([[x, y, th + d / 2], [x, y, th + d]]), TURN_COST, "turn",
                        ("turn", np.sign(d))))
        return out

    def _shot(self, p, goal, entry=False):
        """entry: rotate, straight and an arc onto goal's line (the shortest free lead);
        else, or if none is free: rotate to face the goal (or away, backing a short way),
        drive straight, rotate to its heading. The samples if free. Within GOAL_XY_M: the
        turn only."""
        if entry:
            for lead in ENTRY_LEAD_M:
                samples = rotate_straight_arc(p, goal, lead=lead)
                if samples is not None and self._free(samples):
                    return samples
        near = np.hypot(goal[0] - p[0], goal[1] - p[1]) < SHOT_REVERSE_M
        samples = rotate_straight_rotate(p, goal, reverse_ok=near, min_dist=GOAL_XY_M)
        return samples if self._free(samples) else None

    def plan(self, start, goal, entry=False):
        """Poses (n, 3) from start to goal, densely sampled, or None; entry: goal starts a
        docking line, join it on an arc where one fits."""
        start, goal = np.asarray(start, dtype=float), np.asarray(goal, dtype=float)
        if not self._free(goal[None]):
            return None
        hgrid = self._heuristic_grid(goal)

        def h(p):
            u = np.floor((p[:2] - self.cm.origin) / self.cm.res).astype(int)
            hh, ww = hgrid.shape
            g = hgrid[u[1], u[0]] if 0 <= u[0] < ww and 0 <= u[1] < hh else np.inf
            return max(g, float(np.hypot(*(p[:2] - goal[:2]))))

        open_ = [(h(start), 0.0, 0, start)]
        parent = {0: (None, start[None])}
        nodes = {0: start}
        steer = {0: None}
        best = {self._key(start): 0.0}
        n = 0
        while open_ and n < MAX_EXPANSIONS:
            _f, g, i, p = heapq.heappop(open_)
            n += 1
            shot = self._shot(p, goal, entry) if np.hypot(*(p[:2] - goal[:2])) < 4.0 or n % 50 == 0 else None
            if shot is not None:
                return self._trace(parent, i, shot)
            for q, samples, cost, kind, st in self._moves(p):
                if not self._free(samples):
                    continue
                clear = float(self.cm.footprint_clearance(q[None])[0])
                cost += CLEARANCE_WEIGHT * max(0.0, CLEARANCE_SOFT_M - clear) * STEP_M / CLEARANCE_SOFT_M
                cost += STEER_CHANGE_COST * (steer[i] is not None and st != steer[i])
                k = self._key(q)
                ng = g + cost
                if ng >= best.get(k, np.inf):
                    continue
                best[k] = ng
                j = len(nodes)
                nodes[j] = q
                steer[j] = st
                parent[j] = (i, samples)
                heapq.heappush(open_, (ng + h(q), ng, j, q))
        return None

    @staticmethod
    def _trace(parent, i, tail):
        segs = [tail]
        while i is not None:
            pi, samples = parent[i]
            segs.append(samples)
            i = pi
        path = np.vstack(segs[::-1])
        keep = np.r_[True, np.any(np.abs(np.diff(path, axis=0)) > 1e-9, axis=1)]
        return path[keep]
