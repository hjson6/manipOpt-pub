"""Scenario side of the mobile drives (plant and validation, never the method): the
routes a technician drives at commissioning or a test drives, steered by a driver
that sees the true pose (a stand-in for a person with a joystick, and for the
navigation still to come), and people walking scripted paths in the room (a crowd
that waits for each other, some giving way to the robot, some not). The hand-written
crowds are fixed loops; random_crowd draws new ones (for training learned policies,
which must not learn one script), clear of the room's walls, tables and fixtures.
"""
import numpy as np

from pick_place_common.scene import BASE_HOME_POSE, PICK_DOCK_BASE, PLACE_DOCK_BASE, ROOM_SIZE

V_MAX, W_MAX, ACC, ALPHA = 0.5, 0.8, 0.5, 1.5
LOOKAHEAD_M = 0.6
REVERSE_OUT_M = 1.7  # out of the parked cell, backwards
TURN_IN_PLACE_RAD = np.radians(50)

# Room-frame waypoints, clear of walls, tables and fixtures for the chassis turning on them.
LOOP = [(5.7, 1.0), (5.7, 4.0), (5.0, 4.6), (2.0, 4.6), (1.3, 4.0), (1.3, 1.0), (3.5, 1.0)]
CROSS = [(5.7, 1.0), (5.7, 4.0), (5.0, 4.6), (4.4, 4.0), (4.4, 1.0), (1.3, 1.0), (1.3, 4.0), (2.0, 4.6), (3.5, 4.6)]
ROUTES = {"loop": LOOP, "cross": CROSS}
# People for the cell layout (step 2's recordings): (body, loop of room points, speed m/s,
# seconds standing at each point).
PEOPLE = [("person_obstacle", [(4.6, 1.0), (4.6, 4.6), (1.6, 4.6), (1.6, 1.0)], 1.2, 0.0),
          ("person_1", [(6.2, 0.6), (6.2, 2.6)], 1.0, 4.0),
          ("person_2", [(6.4, 4.8), (4.6, 4.8), (4.6, 3.8)], 1.3, 2.0),
          ("person_3", [(1.0, 1.4)], 0.0, 0.0)]

# The crowd for the stations layout: (body, loop of room points, speed m/s, seconds
# standing at each point, seconds standing before the first step, gives way to the robot,
# and optionally a trigger: stands at the first point until the robot comes this near the
# second, then walks there).
CROWD = [
    ("person_1", [(2.2, 2.0), (4.8, 2.0), (4.8, 3.6), (2.2, 3.6)], 1.2, 0.0, 0.0, True),  # loops the middle
    ("person_2", [(1.6, 3.0), (5.6, 3.0)], 1.0, 1.0, 0.0, False),  # crosses the room, never gives way
    ("person_3", [(5.9, 2.0), (4.2, 1.6)], 1.1, 25.0, 20.0, True),  # stands away from home, then walks over
    ("person_4", [(1.6, 3.3)], 0.0, 0.0, 0.0, False),  # stands by the pick station
    ("person_5", [(1.6, 4.0), (5.6, 1.2)], 0.8, 3.0, 5.0, False),  # slow diagonal, never gives way
    ("person_6", [(5.0, 3.6), (5.4, 2.2), (4.0, 3.2)], 1.1, 2.0, 0.0, True),  # near the place station
]
# The mobile job's crowd: the same paths, but everyone is considerate of the robot (as people
# at work are: they go round it when it is in their way and do not stand by it while it
# works), they stand a while at each of their points, and the bystander by the pick station
# comes and goes. Obstructive people are the test scenarios' (walkers, dock_block, step_in).
PAUSE_S = 6.0  # the job's crowd stands this long at each point
JOB_CROWD = [(c[0], *(([(1.6, 3.3), (3.0, 4.2)], 0.9, 30.0, 0.0) if c[0] == "person_4" else
                      (c[1], c[2], max(c[3], PAUSE_S), c[4])), True) for c in CROWD]
# The scenario set (step 7): a few walkers; someone standing in the pick station's
# docking line (on its approach, behind the docked robot) 40 s at a time; someone waiting
# beside the route between the stations who steps onto it as the robot comes.
WALKERS = [CROWD[0], CROWD[1], CROWD[4]]
DOCK_BLOCK = [("person_1", [(1.9, 1.95), (2.6, 0.9)], 0.8, 40.0, 0.0, False)]
STEP_IN = [("person_1", [(3.6, 1.93), (3.0, 2.72)], 1.0, 6.0, 0.0, False, 2.5)]
CROWDS = {"stations": CROWD, "job": JOB_CROWD, "walkers": WALKERS, "dock_block": DOCK_BLOCK, "step_in": STEP_IN}
REARM_M = 0.4  # a triggered person steps in again only once the robot has gone this much farther
PERSON_GAP_M = 0.7  # someone this close ahead: wait...
PERSON_WAIT_MAX_S = 2.0  # ...this long at most (no one waits for a waiting person for ever)...
PERSON_PASS_S = 1.5  # ...then walks on past them for this long
WAIT_M = 2.5  # a considerate person does not stand nearer a parked robot than this (beyond the arm's holds)
PARKED_S = 3.0  # the robot moved under 5 cm this long: parked (docked, working)
PATIENCE_S = 1.0  # a considerate person walks round the robot's body in their way after this
BLOCKED_GAP_M = 0.75  # the robot's centre this close ahead: blocked by its body, whoever you are
BLOCKED_MAX_S = 3.0  # blocked this long: step aside, then walk on round it
SIDESTEP_M = 0.9
GOAL_TAKEN_M = 0.6  # a spot this near the robot's centre is under it: on to the next one
WALL_KEEP_M = 0.3
# Random crowds (random_crowd; the RL plan's layer 1, section 5).
PERSON_R = 0.30  # a person's footprint: points and paths keep this, and CLEAR_M, from walls, tables, fixtures
CLEAR_M = 0.05
RANDOM_SPEED = (0.8, 1.3)
RANDOM_DWELL_S = (0.0, 30.0)
RANDOM_START_S = (0.0, 20.0)
RANDOM_GIVES_WAY = 0.6
RANDOM_NEAR_STATION = 0.4  # of the points: 0.6-1.8 m from a dock
RANDOM_STEP_IN, RANDOM_DOCK_BLOCK = 0.2, 0.1  # of the crowds: one more who steps in, one who stands in a docking line
STATION_LINE_M = 0.9  # a docking line's length (nav.navigator.PRE_DOCK_M)
LINE_KEEP_M = 0.8  # where walkers stand: this far from a docking line (the docked chassis's corner and a person)
START_FROM_HOME_M = 1.2  # nobody starts this near the robot parked at home
START_APART_M = 0.8
POINT_APART_M = 1.0
SKIP_GEOMS = ("floor", "wall_", "person", "dynamic_obstacle", "cbox", "chassis", "pedestal", "wheel", "caster")


def walking(specs, speed=1.0):
    """The specs with each person's walking speed x speed."""
    return [(s[0], s[1], s[2] * speed, *s[3:]) for s in specs]


def room_obstacles(m):
    """[(name, (x0, x1, y0, y1))] of the static geoms above the floor (tables, trays,
    fixtures), their footprints in the room, from a compiled scene."""
    import mujoco
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    base_root = m.body_rootid[m.body("base_link").id]
    out = []
    for g in range(m.ngeom):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        b = m.geom_bodyid[g]
        body = m.body(b).name
        if any(name.startswith(s) or body.startswith(s) for s in SKIP_GEOMS) or (b and m.body_rootid[b] == base_root):
            continue
        z0, z1 = d.geom_xpos[g][2] - m.geom_size[g][2], d.geom_xpos[g][2] + m.geom_size[g][2]
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX or z0 > 1.5 or z1 < 0.0:
            continue  # the ceiling camera's mount, things parked below the floor
        r = d.geom_xmat[g].reshape(3, 3)
        corners = np.array([d.geom_xpos[g] + r @ (np.array([sx, sy, 0]) * m.geom_size[g])
                            for sx in (-1, 1) for sy in (-1, 1)])
        out.append((name or f"geom{g}", (corners[:, 0].min(), corners[:, 0].max(), corners[:, 1].min(),
                                         corners[:, 1].max())))
    return out


def gap_point(p, rects):
    """The distance from point p to the walls and the nearest rectangle, and its name."""
    best = (min(p[0], ROOM_SIZE[0] - p[0], p[1], ROOM_SIZE[1] - p[1]), "wall")
    for name, (x0, x1, y0, y1) in rects:
        dx, dy = max(x0 - p[0], 0.0, p[0] - x1), max(y0 - p[1], 0.0, p[1] - y1)
        best = min(best, (float(np.hypot(dx, dy)), name))
    return best


def gap_path(pts, rects, step=0.05):
    pts = np.asarray(pts, dtype=float)
    best = (np.inf, "")
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(2, int(np.linalg.norm(b - a) / step))
        for f in np.linspace(0, 1, n):
            best = min(best, gap_point(a + f * (b - a), rects))
    return best


def _clear(pts, rects):
    gap = gap_path(pts, rects)[0] if len(pts) > 1 else gap_point(pts[0], rects)[0]
    return gap >= PERSON_R + CLEAR_M


def _start_ok(p, starts):
    return (np.hypot(p[0] - BASE_HOME_POSE[0], p[1] - BASE_HOME_POSE[1]) >= START_FROM_HOME_M
            and all(np.hypot(*(np.asarray(p) - s)) >= START_APART_M for s in starts))


def _point(rng, rects, centre=None, radius=(0.0, 0.0), tries=200):
    """A free point in the room, or round centre within radius."""
    for _ in range(tries):
        if centre is None:
            p = rng.uniform((0.0, 0.0), ROOM_SIZE)
        else:
            a, r = rng.uniform(0.0, 2 * np.pi), rng.uniform(*radius)
            p = np.asarray(centre, dtype=float) + r * np.array([np.cos(a), np.sin(a)])
        if _clear([p], rects):
            return p
    return None


def line_gap(p):
    """The distance from point p to the nearer docking line (the docked pose to the pre-dock pose)."""
    p, gaps = np.asarray(p, dtype=float), []
    for dock in (PICK_DOCK_BASE, PLACE_DOCK_BASE):
        a, b = np.asarray(dock[:2], dtype=float), _station_line(dock, STATION_LINE_M)
        f = np.clip((p - a) @ (b - a) / ((b - a) @ (b - a)), 0.0, 1.0)
        gaps.append(float(np.linalg.norm(p - (a + f * (b - a)))))
    return min(gaps)


def _station_line(dock, back, lateral=0.0):
    """A point on a dock's line, back from the docked pose, lateral to its left."""
    c, s = np.cos(dock[2]), np.sin(dock[2])
    return np.array([dock[0] - back * c - lateral * s, dock[1] - back * s + lateral * c])


def _rounded(pts):
    return [tuple(round(float(v), 3) for v in p) for p in pts]


def _loop(rng, rects, starts, tries=100):
    """2-4 points, a fraction near the stations but off their docking lines, walked as a
    loop (2: back and forth)."""
    for _ in range(tries):
        k = int(rng.integers(2, 5))
        pts = []
        for _j in range(k):
            dock = (PICK_DOCK_BASE, PLACE_DOCK_BASE)[int(rng.integers(2))]
            near = rng.random() < RANDOM_NEAR_STATION
            p = _point(rng, rects, dock[:2], (0.6, 1.8)) if near else _point(rng, rects)
            if p is None or line_gap(p) < LINE_KEEP_M:
                break
            pts.append(p)
        if len(pts) < k:
            continue
        closed = pts + pts[:1] if k > 2 else pts
        if (all(np.linalg.norm(a - b) >= POINT_APART_M for a, b in zip(closed[:-1], closed[1:]))
                and _clear(closed, rects) and _start_ok(pts[0], starts)):
            return _rounded(pts)
    return None


def _step_in(rng, rects, starts, tries=100):
    """Waiting beside the way between the stations' docking lines, onto it as the robot comes."""
    a, b = _station_line(PICK_DOCK_BASE, STATION_LINE_M), _station_line(PLACE_DOCK_BASE, STATION_LINE_M)
    u = (b - a) / np.linalg.norm(b - a)
    n = np.array([-u[1], u[0]])
    for _ in range(tries):
        on = a + rng.uniform(0.25, 0.75) * (b - a) + rng.uniform(-0.3, 0.3) * n
        wait = on + rng.choice((-1.0, 1.0)) * rng.uniform(0.7, 1.0) * n
        if _clear([wait, on], rects) and _start_ok(wait, starts):
            return _rounded([wait, on])
    return None


def _dock_block(rng, rects, starts, tries=100):
    """Standing in a docking line a while, then a while elsewhere, and back."""
    for _ in range(tries):
        dock = (PICK_DOCK_BASE, PLACE_DOCK_BASE)[int(rng.integers(2))]
        stand = _station_line(dock, rng.uniform(0.6, 1.2), rng.uniform(-0.15, 0.15))
        away = _point(rng, rects, stand, (1.5, 2.5))
        if away is not None and _clear([stand, away], rects) and _start_ok(stand, starts):
            return _rounded([stand, away])
    return None


def random_crowd(rng, rects, n_max=6):
    """A crowd drawn at random, clear of rects (room_obstacles) and the walls: 1 to n_max
    people (at most the scene's six), each on a loop of 2-4 points, walking 0.8-1.3 m/s,
    standing 0-30 s at each, starting after 0-20 s, 60% giving way to the robot; in 20% of
    crowds one of them waits beside the way between the stations and steps in as the robot
    comes (as step_in), in 10% one stands a while in a docking line (as dock_block)."""
    n_max = int(min(n_max, 6))
    extras = [kind for kind, share in (("step_in", RANDOM_STEP_IN), ("dock_block", RANDOM_DOCK_BLOCK))
              if rng.random() < share][:max(0, n_max - 1)]
    n = int(rng.integers(1, n_max - len(extras) + 1))
    specs, starts = [], []
    for kind in ["loop"] * n + extras:
        pts = {"loop": _loop, "step_in": _step_in, "dock_block": _dock_block}[kind](rng, rects, starts)
        if pts is None:
            continue
        name = f"person_{len(specs) + 1}"
        speed = float(round(rng.uniform(*RANDOM_SPEED), 2))
        if kind == "loop":
            specs.append((name, pts, speed, float(round(rng.uniform(*RANDOM_DWELL_S), 1)),
                          float(round(rng.uniform(*RANDOM_START_S), 1)), bool(rng.random() < RANDOM_GIVES_WAY)))
        elif kind == "step_in":
            specs.append((name, pts, speed, float(round(rng.uniform(4.0, 8.0), 1)), 0.0, False,
                          float(round(rng.uniform(2.0, 3.0), 2))))
        else:
            specs.append((name, pts, speed, float(round(rng.uniform(15.0, 40.0), 1)), 0.0, False))
        starts.append(np.asarray(pts[0]))
    return specs


class Crowd:
    """People walking their loops; each waits for someone just ahead. Everyone walks on
    into the robot's path and stops only when its body is right in front of them, then
    steps aside and walks round it: after PATIENCE_S if they give way (considerate),
    BLOCKED_MAX_S if not. The considerate also do not stand within WAIT_M of the robot
    while it is parked: they skip such a spot, or leave it once it has parked there."""

    def __init__(self, specs):
        self.specs = specs
        self.pos = np.array([s[1][0] for s in specs], dtype=float)
        self.target = [1 % len(s[1]) for s in specs]
        self.wait = np.array([s[4] + (s[3] if len(s[1]) > 1 else 0.0) for s in specs])
        self.yaw = np.zeros(len(specs))
        self.waited = np.zeros(len(specs))
        self.passing = np.zeros(len(specs))
        self.blocked = np.zeros(len(specs))
        self.detour = [None] * len(specs)
        self.armed = [len(s) > 6 for s in specs]
        self.robot_at = None  # where the robot was when it last moved 5 cm, and since when
        self.robot_parked_s = 0.0

    def step(self, dt, robot_xy=None):
        """Advance dt; returns [(x, y, yaw)] per person."""
        if robot_xy is not None:
            r = np.asarray(robot_xy, dtype=float)
            if self.robot_at is None or np.hypot(*(r - self.robot_at)) > 0.05:
                self.robot_at, self.robot_parked_s = r, 0.0
            else:
                self.robot_parked_s += dt
        parked = robot_xy is not None and self.robot_parked_s >= PARKED_S
        for i, (_, path, speed, dwell, _start, gives_way, *trigger) in enumerate(self.specs):
            if len(path) < 2 or speed <= 0.0:
                continue
            if trigger and robot_xy is not None and self.target[i] == 1 and np.allclose(self.pos[i], path[0]):
                gap = float(np.hypot(*(np.asarray(path[1]) - robot_xy)))
                if not self.armed[i]:
                    self.armed[i] = gap > trigger[0] + REARM_M
                    continue
                if gap > trigger[0]:
                    continue  # waiting for the robot
                self.armed[i], self.wait[i] = False, 0.0
            near = parked and gives_way and np.hypot(*(self.pos[i] - robot_xy)) < WAIT_M
            if self.wait[i] > 0.0 and not near:  # a considerate person does not stand by a working robot
                self.wait[i] -= dt
                continue
            goal = self.detour[i] if self.detour[i] is not None else np.array(path[self.target[i]], dtype=float)
            if gives_way and parked and self.detour[i] is None:
                goal = self._considerate_goal(i, path, np.asarray(robot_xy, dtype=float))
                if goal is None:
                    continue  # waiting, well clear of the robot, for a spot it is at
            elif self.detour[i] is None and robot_xy is not None and np.hypot(*(goal - robot_xy)) < GOAL_TAKEN_M:
                nxt = (self.target[i] + 1) % len(path)
                if np.hypot(*(np.asarray(path[nxt]) - self.pos[i])) > 0.05:
                    self.target[i] = nxt
                    continue
                # On to where they stand already: beside the taken spot instead.
                self.detour[i] = self._beside(goal, robot_xy)
                goal = self.detour[i]
            d = goal - self.pos[i]
            dist = float(np.linalg.norm(d))
            if dist < speed * dt and self.detour[i] is not None:
                self.pos[i], self.detour[i] = goal, None
                continue
            if dist < speed * dt:
                self.pos[i] = goal
                self.target[i] = (self.target[i] + 1) % len(path)
                self.wait[i] = dwell
                continue
            u = d / dist
            self.yaw[i] = float(np.arctan2(u[1], u[0]))
            others = [self.pos[j] for j in range(len(self.specs)) if j != i]
            if self.passing[i] > 0.0:
                self.passing[i] -= dt
            elif any(self._ahead(self.pos[i], u, o, PERSON_GAP_M) for o in others):
                if self.waited[i] < PERSON_WAIT_MAX_S:
                    self.waited[i] += dt
                    continue
                self.passing[i] = PERSON_PASS_S
            self.waited[i] = 0.0
            if robot_xy is not None and self._ahead(self.pos[i], u, robot_xy, BLOCKED_GAP_M):
                self.blocked[i] += dt
                if self.blocked[i] > (PATIENCE_S if gives_way else BLOCKED_MAX_S):
                    self.detour[i], self.blocked[i] = self._sidestep(self.pos[i], u, robot_xy), 0.0
                continue
            self.blocked[i] = 0.0
            self.pos[i] = np.clip(self.pos[i] + u * speed * dt, WALL_KEEP_M, np.asarray(ROOM_SIZE) - WALL_KEEP_M)
        return [(float(x), float(y), float(a)) for (x, y), a in zip(self.pos, self.yaw)]

    def _considerate_goal(self, i, path, robot):
        """A considerate person's goal by a parked robot: their spot, or the next one at least
        WAIT_M from it; if none is, None where they stand that far from it, else a step away."""
        n = len(path)
        for k in range(n):
            j = (self.target[i] + k) % n
            if np.hypot(*(np.asarray(path[j]) - robot)) >= WAIT_M:
                self.target[i] = j
                return np.array(path[j], dtype=float)
        away = self.pos[i] - robot
        return None if np.hypot(*away) >= WAIT_M else self.pos[i] + away

    @staticmethod
    def _sidestep(p, u, robot_xy):
        """A point SIDESTEP_M off, the free way (not toward the robot, inside the room) nearest
        the way on (two perpendicular sides could both be toward a robot close by)."""
        best = None
        for a in np.linspace(0.0, 2 * np.pi, 16, endpoint=False):
            d = np.array([np.cos(a), np.sin(a)])
            q = p + SIDESTEP_M * d
            inside = WALL_KEEP_M <= q[0] <= ROOM_SIZE[0] - WALL_KEEP_M and WALL_KEEP_M <= q[1] <= ROOM_SIZE[1] - WALL_KEEP_M
            if inside and not Crowd._ahead(p, d, robot_xy, BLOCKED_GAP_M + 0.1) and (best is None or d @ u > best[0]):
                best = (float(d @ u), q)
        return best[1] if best is not None else p - 0.5 * u

    @staticmethod
    def _beside(goal, robot_xy):
        """A spot just clear of the robot on goal's side of it, inside the room."""
        d = np.asarray(goal, dtype=float) - robot_xy
        q = robot_xy + d / max(float(np.linalg.norm(d)), 1e-6) * (GOAL_TAKEN_M + 0.3)
        return np.clip(q, WALL_KEEP_M, np.asarray(ROOM_SIZE) - WALL_KEEP_M)

    @staticmethod
    def _ahead(p, u, o, gap):
        v = np.asarray(o, dtype=float) - p
        return np.linalg.norm(v) < gap and v @ u > 0.3 * np.linalg.norm(v)


def person_at(path, speed, dwell, t):
    """(x, y, yaw) along a closed path, walking at speed and standing dwell at each point."""
    pts = np.array(path, dtype=float)
    if len(pts) == 1 or speed <= 0.0:
        return pts[0, 0], pts[0, 1], 0.0
    legs = [(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]
    period = sum(np.linalg.norm(b - a) / speed + dwell for a, b in legs)
    u = t % period
    for a, b in legs:
        if u < dwell:
            d = b - a
            return a[0], a[1], float(np.arctan2(d[1], d[0]))
        u -= dwell
        walk = np.linalg.norm(b - a) / speed
        if u < walk:
            p = a + (b - a) * u / walk
            d = b - a
            return p[0], p[1], float(np.arctan2(d[1], d[0]))
        u -= walk
    return pts[0, 0], pts[0, 1], 0.0


class Driver:
    """Pure pursuit on the true pose, turning in place at sharp corners; reverses out of
    the dock first."""

    def __init__(self, waypoints, start, reverse_out=True):
        """reverse_out: back 1.7 m out of the cell first (the base starts parked there)."""
        self.start = np.array(start, dtype=float)
        back = self.start[:2] - REVERSE_OUT_M * np.array([np.cos(self.start[2]), np.sin(self.start[2])])
        self.path = [back] + [np.array(w) for w in waypoints]
        self.i = 0
        self.reversing = reverse_out
        if not reverse_out:
            self.path[0] = np.array(start[:2])
            self.i = 1
        self.done = False

    def command(self, pose):
        x, y, yaw = pose
        if self.reversing:
            if np.hypot(x - self.start[0], y - self.start[1]) < REVERSE_OUT_M:
                return -0.25, 0.0
            self.reversing = False
            self.i = 1
        if self.i >= len(self.path):
            self.done = True
            return 0.0, 0.0
        target = self.path[self.i]
        dist = np.hypot(*(target - (x, y)))
        if dist < 0.15:
            self.i += 1
            return self.command(pose)
        if dist > LOOKAHEAD_M and self.i > 1:
            prev = self.path[self.i - 1]
            seg = target - prev
            s = np.clip(np.dot((x, y) - prev, seg) / np.dot(seg, seg), 0.0, 1.0)
            ahead = prev + seg * min(1.0, s + LOOKAHEAD_M / np.linalg.norm(seg))
        else:
            ahead = target
        err = np.arctan2(ahead[1] - y, ahead[0] - x) - yaw
        err = (err + np.pi) % (2 * np.pi) - np.pi
        if abs(err) > TURN_IN_PLACE_RAD:
            return 0.0, float(np.clip(2.0 * err, -W_MAX, W_MAX))
        v = min(V_MAX, 0.8 * dist + 0.1)
        return v, float(np.clip(2.0 * v * np.sin(err) / max(np.hypot(*(ahead - (x, y))), 0.2), -W_MAX, W_MAX))
