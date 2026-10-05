"""Navigation to named goals (the stations' docks, home) on the saved map: undock
straight back if docked, plan to the goal's pre-dock pose (Hybrid A*), follow the
route with the MPC (people predicted at constant velocity), joining a dock's line on an
arc where one fits and rolling on into the approach when it arrives lined up, else aligning
on the pre-dock pose or home (turn, straight, turn), then approach the dock straight and
slowly; an arrival
outside the tolerance backs out straight, aligns and approaches again. The docking
moves leave people to the safety layer (stop and wait, no swerving by the table).
The arm may work while the base docks or undocks; off the docking line (route,
align) the base moves only with the arm stowed (arm_ready, from the task), and
paused stops it anywhere (someone near the arm while it is out).
Optionally plans in a worker thread, so a slow plan never holds up the control steps.
Framework-free (acados for the MPC).

See docs/implementation_notes.md#navigatorpy.
"""
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from nav.base_mpc import NO_PERSON, BaseMPC, BaseMPCConfig
from nav.planner import FOOTPRINT_R, FOOTPRINT_X, Costmap, HybridAStar, curve_onto_line, rotate_straight_rotate
from nav.reference import RouteReference, wrap

PRE_DOCK_M = 0.9
REJOIN_END_M = 0.35  # a curve back onto the docking line joins it this far or more before the dock
DOCK_SPEED = 0.10  # the docking line's last DOCK_SLOW_M, by the table
DOCK_SLOW_M = 0.25
DOCK_FAST_SPEED = 0.30  # the rest of it
UNDOCK_SPEED = 0.30
DOCK_TOL_LATERAL_M = 0.02
DOCK_TOL_ALONG_M = 0.02
DOCK_TOL_YAW_RAD = np.radians(1.5)
DOCK_RETRIES = 2
ALIGN_TOL_LATERAL_M = 0.03  # the straight approach absorbs this much
ALIGN_TOL_YAW_RAD = np.radians(3.0)
PARK_TOL_M = 0.02  # at home no approach follows: parked this close
PARK_TOL_YAW_RAD = np.radians(1.0)
ALIGN_TRIES = 3
ALIGN_MIN_M = 0.02  # closer: turn to the heading only
ALIGN_SPEED = 0.3
ROUTE_TOL_M, ROUTE_TOL_RAD = 0.06, np.radians(4)
PERSON_OWN_M = 0.3
PERSON_RADIUS_M = 0.45  # a person's own 0.3 m plus the distance the robot keeps (x safety)
STAND_SLACK_M = 0.02
NEAR_M = (0.9, 1.8, 3.0)  # the nearest person's centre this far (x safety): speed limit...
NEAR_SPEED = (0.2, 0.5)  # ...from these to the top speed
TURN_PER_SPEED = 1.2  # rad/s per m/s of top speed: turns keep pace with driving...
TURN_RANGE = (0.6, 1.2)  # ...within these
WHEEL_RIM_MAX = 1.4  # m/s, the drives' wheel-speed limit with a little in hand
PREDICT_MAX_S = 3.0
STILL_V, STILL_W = 0.02, 0.03
DONE_SETTLE_S = 0.5
BLOCKED_S = 4.0  # no progress this long: plan again round the people standing in the way
STEP_BACK_NEAR_M = 1.0  # blocked again on a route round someone standing this near ahead: step back...
STEP_BACK_M = 0.4  # ...this far
STEP_BACK_SPEED = 0.2
HELD_RETRY_S = 1.0  # stepped back: plan again this often (a route round them, or they moved)
BLOCKER_RADIUS_M = 0.45
STANDING_MS = 0.2


def behind(pose, d):
    return np.array([pose[0] - d * np.cos(pose[2]), pose[1] - d * np.sin(pose[2]), pose[2]])


def line(a, b, heading, step=0.02):
    """Poses on the straight line from a to b, all at one heading (driven forward or
    backward along it)."""
    n = max(2, int(np.hypot(b[0] - a[0], b[1] - a[1]) / step) + 1)
    f = np.linspace(0.0, 1.0, n)[:, None]
    xy = np.asarray(a[:2]) + f * (np.asarray(b[:2]) - np.asarray(a[:2]))
    return np.column_stack([xy, np.full(n, heading)])


class Navigator:
    def __init__(self, grid, docks, home, mpc=None, background=False, speed=0.5, accel=0.5, safety=1.0):
        """docks: {name: dock pose (map frame)}; home: a free pose to park at;
        background: plan in a worker thread (waiting in "planning" for a goal's first
        route, driving on the current one while planning round people); speed: the top
        speed on routes (m/s; the turn rate follows it); accel: the acceleration limit
        (m/s^2; routes brake to their end at 0.8 of it); safety: scales the distances to
        people (where it slows down, the room it keeps, round people standing)."""
        self.speed = float(speed)
        self.a_brake = 0.8 * float(accel)
        self.turn = float(np.clip(TURN_PER_SPEED * self.speed, *TURN_RANGE))
        self.safety = float(safety)
        self.person_radius = PERSON_OWN_M + (PERSON_RADIUS_M - PERSON_OWN_M) * self.safety
        self.blocker_radius = PERSON_OWN_M + (BLOCKER_RADIUS_M - PERSON_OWN_M) * self.safety
        self.near_m = tuple(self.safety * np.array(NEAR_M))
        self.costmap = Costmap(grid)
        self.planner = HybridAStar(self.costmap)
        self.docks = {k: np.asarray(v, dtype=float) for k, v in docks.items()}
        self.home = np.asarray(home, dtype=float)
        self.mpc = mpc or BaseMPC(BaseMPCConfig(
            v_max=self.speed, v_min=-UNDOCK_SPEED, w_max=max(1.0, self.turn), a_max=float(accel),
            rim_speed_max=min(WHEEL_RIM_MAX, max(0.8, self.speed + 0.25 * self.turn))))
        # idle | undock | planning | route | step_back | held | align | approach | backout | docked | parked | failed
        self.state = "idle"
        self.at = None  # the station docked at
        self.goal = None
        self.ref = None
        self.retries = 0
        self.aligns = 0
        self.settle_t = None
        self.last = None
        self.people = []
        self.rerouted = False  # the route leads round people standing in the way
        self.stepped_from = None
        self.progress_t = None
        self.progress_at = -1
        self.planned = None  # (length m, heading travel rad) of this goal's first plan, docking lines included
        self.undock_m = 0.0
        self.pool = ThreadPoolExecutor(1, thread_name_prefix="planner") if background else None
        self.pending = None  # (future, standing people) of a plan being made
        self.arm_ready = True  # the arm stowed: needed off the docking line
        self.paused = False
        self.counts = dict(aligns=0, replans=0, step_backs=0, rejoins=0)  # this goal's, for scoring
        self.rejoining = False  # aligning on a curve onto the docking line: rolls on into the approach
        self.align_from = []

    @property
    def docking(self):
        """Close to a table: the safety layer's docking margins."""
        return self.state in ("approach", "backout", "undock")

    @property
    def precise(self):
        """Docking moves: the MPC follows the line and leaves people to the safety layer."""
        return self.docking or self.state == "align"

    def set_goal(self, name, pose):
        """Go to a dock or home; pose: the robot's current map pose."""
        self.goal = name
        self.counts = dict(aligns=0, replans=0, step_backs=0, rejoins=0)
        self.align_from = []  # (along, lateral, yaw deg) from the dock where each align started
        self.retries = 0
        self.planned = None
        self.pending = None
        self.undock_m = PRE_DOCK_M if self.at is not None else 0.0
        if self.at is not None:
            self._start("undock", self._back(pose))
        else:
            self._plan(pose)

    def _back(self, pose):
        """Straight back from the dock, on the robot's own heading, to the pre-dock distance."""
        along, _, _ = self.dock_error(pose, self.at or self.goal)
        return RouteReference(line(pose, behind(pose, max(PRE_DOCK_M + along, 0.05)), pose[2]), v_cruise=UNDOCK_SPEED,
                              v_reverse=UNDOCK_SPEED)

    def _start(self, state, ref):
        self.state, self.ref, self.settle_t = state, ref, None
        self.progress_t, self.progress_at = None, -1
        self.mpc.warm = False

    def _target(self):
        return behind(self.docks[self.goal], PRE_DOCK_M) if self.goal in self.docks else self.home

    def _plan(self, pose, people=(), keep_route=False):
        """A route to the goal's pre-dock pose (home); people standing: kept clear of;
        keep_route: drive on the current route while the new one is being made."""
        standing = [p[:2] for p in people if np.hypot(p[2], p[3]) < STANDING_MS]
        planner = self.planner
        if standing:
            free = [pose] + ([self.stepped_from] if self.state == "held" else [])
            planner = HybridAStar(self.costmap.with_obstacles(standing, self._blocker_radii(free, standing)))
        entry = self.goal in self.docks
        if self.pool is None:
            return self._planned(planner.plan(pose, self._target(), entry), standing)
        self.pending = (self.pool.submit(planner.plan, np.array(pose, dtype=float), self._target(), entry), standing)
        if not keep_route:
            self.state = "planning"
        return True

    def _planned(self, path, standing):
        """Take a finished plan (None: none found; the current route stays if planning
        round people)."""
        if path is None:
            if not standing:
                self.state = "failed"
            return False
        self.at = None
        self.rerouted = bool(standing)
        self._start("route", RouteReference(path, v_cruise=self.speed, w_turn=self.turn, a_brake=self.a_brake,
                                            v_end=DOCK_FAST_SPEED if self.goal in self.docks else 0.0))
        if self.planned is None:
            turn = float(np.abs(wrap(np.diff(self.ref.path[:, 2]))).sum())
            self.planned = (self.ref.length + self.undock_m + (PRE_DOCK_M if self.goal in self.docks else 0.0), turn)
        return True

    def _blocker_radii(self, poses, centres):
        """BLOCKER_RADIUS_M, less where a person stands closer: the robot's own pose stays
        free, so the route can lead away (held: also the pose it stepped back from, so the
        route round them found there is found again)."""
        circles = np.array([[q[0] + dx * np.cos(q[2]), q[1] + dx * np.sin(q[2])] for q in poses for dx in FOOTPRINT_X])
        gap = [np.hypot(*(circles - c).T).min() - FOOTPRINT_R - self.planner.margin - 2 * self.costmap.res for c in centres]
        return np.clip(gap, 0.0, self.blocker_radius)

    def step(self, t, pose, twist, people):
        """One control step: pose (map), twist (v, omega) measured, people [(x, y, vx, vy)]
        tracks (map frame). Returns (v, omega) to command (before the safety layer)."""
        if self.pending is not None and self.pending[0].done():
            (future, standing), self.pending = self.pending, None
            if self.state in ("planning", "route", "held"):  # not once it is aligning or docking
                self._planned(future.result(), standing)
        if self.state == "held":
            if self.pending is None and t - self.progress_t > HELD_RETRY_S:
                self.progress_t = t
                self._plan(pose, people, keep_route=True)
            return 0.0, 0.0
        if self.state in ("idle", "planning", "docked", "parked", "failed") or self.ref is None:
            return 0.0, 0.0
        if self.paused or not (self.arm_ready or self.docking):
            self.progress_t = t  # waiting, not blocked
            self.mpc.warm = False
            return 0.0, 0.0
        self.people = people
        progress = self.ref.update(pose)
        if progress != self.progress_at:
            self.progress_at, self.progress_t = progress, t
        elif self.state == "route" and t - self.progress_t > BLOCKED_S and people and self.pending is None:
            self.progress_t = t
            if self.rerouted and self._standing_ahead(pose, people):
                self.counts["step_backs"] += 1
                self.stepped_from = np.array(pose, dtype=float)
                self._start("step_back", RouteReference(line(pose, behind(pose, STEP_BACK_M), pose[2]),
                                                        v_cruise=STEP_BACK_SPEED, v_reverse=STEP_BACK_SPEED))
            else:
                self.counts["replans"] += 1
                self._plan(pose, people, keep_route=True)
        elif self.state == "step_back" and t - self.progress_t > BLOCKED_S:
            self._next(pose)  # no room behind: plan from here
        cfg = self.mpc.cfg
        ref = self.ref.horizon(cfg.N, cfg.dt)
        pred = np.tile(NO_PERSON, (cfg.N + 1, cfg.n_people, 1)).astype(float)
        near = [] if self.precise else sorted(people, key=lambda p: np.hypot(p[0] - pose[0], p[1] - pose[1]))[:cfg.n_people]
        ts = np.minimum(np.arange(cfg.N + 1) * cfg.dt, PREDICT_MAX_S)
        for j, (px, py, vx, vy) in enumerate(near):
            pred[:, j, 0] = px + vx * ts
            pred[:, j, 1] = py + vy * ts
            # Never closer than the robot standing still would be: it does not flee people.
            gap = np.hypot(pred[:, j, 0] - pose[0], pred[:, j, 1] - pose[1]) - cfg.robot_radius - STAND_SLACK_M
            pred[:, j, 2] = np.clip(gap, -cfg.robot_radius, self.person_radius)
        d_near = min((np.hypot(p[0] - pose[0], p[1] - pose[1]) for p in people), default=np.inf)
        v_near = float(np.interp(d_near, self.near_m, (*NEAR_SPEED, max(NEAR_SPEED[-1], self.speed))))
        status, xs = self.mpc.solve([*pose, *twist], ref, pred, v_max=v_near)
        self.last = (status, xs, ref)
        if status not in (0, 2):
            return 0.0, 0.0
        if self._arrived(t, pose, twist):
            rolling = self.state == "route" or (self.state == "align" and self.rejoining)
            self._next(pose)
            if rolling and self.state == "approach":
                return self.step(t, pose, twist, people)  # rolls on: the approach's first command
            return 0.0, 0.0
        return float(xs[1, 3]), float(xs[1, 4])

    @staticmethod
    def _standing_ahead(pose, people):
        """Someone standing within STEP_BACK_NEAR_M ahead: a route round them may start
        with a turn the safety layer refuses so near."""
        c, s = np.cos(pose[2]), np.sin(pose[2])
        for px, py, vx, vy in people:
            dx, dy = px - pose[0], py - pose[1]
            if np.hypot(vx, vy) < STANDING_MS and np.hypot(dx, dy) < STEP_BACK_NEAR_M and c * dx + s * dy > 0.0:
                return True
        return False

    def _arrived(self, t, pose, twist):
        g = self.ref.path[-1]
        if (self.state == "route" or (self.state == "align" and self.rejoining)) and self.goal in self.docks \
                and self.ref.progress >= len(self.ref.path) - 2 and self._aligned(pose):
            return True  # lined up on the docking line: rolls on into the approach
        tol_xy = DOCK_TOL_ALONG_M if self.state == "approach" else ROUTE_TOL_M
        close = np.hypot(pose[0] - g[0], pose[1] - g[1]) < max(tol_xy, 0.01) and abs(wrap(pose[2] - g[2])) < ROUTE_TOL_RAD
        still = abs(twist[0]) < STILL_V and abs(twist[1]) < STILL_W
        at_end = self.ref.progress >= len(self.ref.path) - 2
        if (close or at_end) and still:
            self.settle_t = t if self.settle_t is None else self.settle_t
            return t - self.settle_t >= DONE_SETTLE_S
        self.settle_t = None
        return False

    def dock_error(self, pose, name=None):
        """(along, lateral, yaw) of pose from a dock, in the dock's frame."""
        d = self.docks[name or self.goal]
        c, s = np.cos(d[2]), np.sin(d[2])
        dx, dy = pose[0] - d[0], pose[1] - d[1]
        return c * dx + s * dy, -s * dx + c * dy, wrap(pose[2] - d[2])

    def _align(self, pose):
        """Onto a dock's line on one forward curve where one fits and is free (rolling on
        into the approach); else turn, straight, turn onto the pre-dock pose, or home
        (backwards if it lies behind)."""
        self.aligns += 1
        self.rejoining = False
        if self.goal in self.docks:
            along, lateral, yaw = self.dock_error(pose)
            self.align_from.append((round(along, 3), round(lateral, 3), round(float(np.degrees(yaw)), 1)))
            path = curve_onto_line(pose, self.docks[self.goal], REJOIN_END_M)
            if path is not None and self.planner._free(path):
                self.counts["rejoins"] += 1
                self.rejoining = True
                self._start("align", RouteReference(path, v_cruise=ALIGN_SPEED, w_turn=self.turn))
                return
        self.counts["aligns"] += 1
        path = rotate_straight_rotate(pose, self._target(), reverse_ok=True, min_dist=ALIGN_MIN_M)
        keep = np.r_[True, np.any(np.abs(np.diff(path, axis=0)) > 1e-9, axis=1)]
        self._start("align", RouteReference(path[keep], v_cruise=ALIGN_SPEED, w_turn=self.turn, v_reverse=ALIGN_SPEED))

    def _aligned(self, pose):
        """On the dock's axis (the approach takes up the rest), or at home."""
        g = self._target()
        c, s = np.cos(g[2]), np.sin(g[2])
        dx, dy = pose[0] - g[0], pose[1] - g[1]
        dock = self.goal in self.docks
        along = 0.0 if dock else c * dx + s * dy
        tol, tol_yaw = (ALIGN_TOL_LATERAL_M, ALIGN_TOL_YAW_RAD) if dock else (PARK_TOL_M, PARK_TOL_YAW_RAD)
        return np.hypot(along, -s * dx + c * dy) <= tol and abs(wrap(pose[2] - g[2])) <= tol_yaw

    def _aligned_next(self, pose):
        if self.goal in self.docks:
            self._approach(pose)
        else:
            self.state = "parked"

    def _approach(self, pose):
        """Along the dock's axis, from the robot's place on it to the dock."""
        d = self.docks[self.goal]
        along, _, _ = self.dock_error(pose)
        self._start("approach", RouteReference(line(behind(d, max(-along, 0.05)), d, d[2]), v_cruise=DOCK_FAST_SPEED,
                                               slow_end=(DOCK_SLOW_M, DOCK_SPEED)))

    def _next(self, pose):
        if self.state == "undock":
            self.at = None
            self._plan(pose)
        elif self.state == "step_back":
            self.state, self.progress_t = "held", -np.inf  # until a route round them comes
        elif self.state == "route":
            self.aligns = 0
            self._aligned_next(pose) if self._aligned(pose) else self._align(pose)
        elif self.state == "align":
            self._aligned_next(pose) if self._aligned(pose) or self.aligns >= ALIGN_TRIES else self._align(pose)
        elif self.state == "approach":
            along, lateral, yaw = self.dock_error(pose)
            self.at = self.goal  # by the table: whatever comes next starts by backing out
            if abs(lateral) <= DOCK_TOL_LATERAL_M and abs(along) <= DOCK_TOL_ALONG_M and abs(yaw) <= DOCK_TOL_YAW_RAD:
                self.state = "docked"
            elif self.retries < DOCK_RETRIES:
                self.retries += 1
                self._start("backout", self._back(pose))
            else:
                self.state = "failed"
        elif self.state == "backout":
            self.aligns = 0
            self._align(pose)
