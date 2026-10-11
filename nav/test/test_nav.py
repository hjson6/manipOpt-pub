"""Unit tests for nav/ (no simulator, no solver): the safety fields, the planner on a
synthetic room, the route reference, planning in the background, and the crowd's
give-way rules."""
import time

import numpy as np

from nav import safety
from nav.base_mpc import BaseMPCConfig
from nav.navigator import Navigator
from nav.planner import Costmap, HybridAStar, rotate_straight_rotate, wrap
from nav.reference import RouteReference
from pick_place_common.mobile_scenarios import BLOCKED_MAX_S, Crowd
from slam.grid import OccupancyGrid

AHEAD = np.array([[safety.HALF_LENGTH + 0.08, 0.0]])  # 8 cm in front of the chassis


def room(width=6.0, height=4.0, table=None):
    """A free rectangle with walls; table: (x0, x1, y0, y1) blocked inside."""
    g = OccupancyGrid((-0.5, -0.5), (int((height + 1) / 0.05), int((width + 1) / 0.05)), 0.05)
    ys, xs = np.mgrid[0:g.log_odds.shape[0], 0:g.log_odds.shape[1]]
    cx, cy = g.origin[0] + (xs + 0.5) * g.res, g.origin[1] + (ys + 0.5) * g.res
    inside = (cx > 0) & (cx < width) & (cy > 0) & (cy < height)
    if table is not None:
        inside &= ~((cx > table[0]) & (cx < table[1]) & (cy > table[2]) & (cy < table[3]))
    g.log_odds[:] = np.where(inside, -2.0, 3.5)
    return g


def test_standing_still_has_no_field():
    assert not safety.protective(AHEAD, 0.0, 0.0).any()
    assert safety.check(AHEAD, 0.0, 0.0)[0] == "clear"


def test_driving_stops_for_what_it_closes_on():
    assert safety.check(AHEAD, 0.3, 0.0)[0] == "stop"
    behind = np.array([[-safety.HALF_LENGTH - 0.08, 0.0]])
    beside = np.array([[0.0, safety.HALF_WIDTH + 0.04]])
    assert safety.check(behind, 0.3, 0.0)[0] == "clear"
    assert safety.check(beside, 0.3, 0.0)[0] != "stop"  # driving past, not closer
    assert safety.check(behind, -0.1, 0.0)[0] == "stop"
    far = np.array([[safety.HALF_LENGTH + 0.8, 0.0]])
    assert safety.check(far, 0.5, 0.0)[0] == "clear"  # its stop and the buffer fit before it
    state, cap = safety.check(far, 1.0, 0.0)
    assert state == "warn" and 0.45 < cap[0] < 0.55  # slowed to what fits
    assert np.allclose(safety.limit(1.0, 0.2, state, cap), (cap[0], 0.2 * cap[0]))
    near = np.array([[safety.HALF_LENGTH + 0.3, 0.0]])
    assert safety.warning_caps(near, 1.0, 0.0)[0] == safety.WARN_SPEED  # the floor


def test_turning_covers_the_corners_way_to_a_stop():
    """A turn stops for what a corner is about to reach, not for the whole circle, and
    never for what it turns away from."""
    corner = np.array([safety.HALF_LENGTH, safety.HALF_WIDTH])
    ahead_ccw = corner + 0.06 * np.array([-safety.HALF_WIDTH, safety.HALF_LENGTH]) / safety.SWEEP_R
    assert safety.check(ahead_ccw[None], 0.0, 0.5)[0] == "stop"
    assert safety.check(ahead_ccw[None], 0.0, -0.5)[0] != "stop"
    leg = np.array([[0.40, 0.43]])  # the table leg beside the pick pre-dock: 15 cm off the corner
    assert safety.check(leg, 0.05, -0.11)[0] != "stop"  # setting off on a slow arc away
    assert safety.check(leg, 0.0, 0.8)[0] == "warn"  # turning toward it fast: slowed (to WARN_TURN)
    assert safety.protective(leg, 0.0, 1.0).all()  # at 1 rad/s within 10 cm before it stops...
    assert safety.check(leg, 0.0, 1.0)[0] == "warn"  # ...so the command is slowed first
    assert safety.check(leg, 0.0, 1.0, measured=(0.0, 1.0))[0] == "stop"  # already turning that fast: stop


def test_people_kept_farther():
    """A person's leg 15 cm ahead stops a slow approach; a table leg there does not (the
    person's upper body overhangs the leg; the folded arm is near the chassis's edge)."""
    leg = np.array([[safety.HALF_LENGTH + 0.15, 0.0]])
    assert safety.check(leg, 0.1, 0.0)[0] != "stop"
    assert safety.check(leg, 0.1, 0.0, people=leg + 0.05)[0] == "stop"  # foreground a round behind
    assert safety.check(leg, 0.1, 0.0, people=leg + 0.5)[0] != "stop"  # someone else's
    assert safety.check(leg, -0.1, 0.0, people=leg)[0] != "stop"  # backing away from them


def test_measured_motion_keeps_the_field():
    """Commanded to stop while still rolling: the field of the measured speed applies."""
    assert safety.check(AHEAD, 0.0, 0.0, measured=(0.3, 0.0))[0] == "stop"


def test_people_margin_scales_with_a_floor():
    """base_safety scales a person's whole stop margin, never below their overhang."""
    assert np.isclose(safety.person_extra(1.0), safety.PERSON_EXTRA_M)
    assert np.isclose(safety.MARGIN_M + safety.person_extra(2.0), 2.0 * (safety.MARGIN_M + safety.PERSON_EXTRA_M))
    assert np.isclose(safety.MARGIN_M + safety.person_extra(0.1), safety.PERSON_MARGIN_MIN_M)
    near = np.array([[safety.HALF_LENGTH + 0.12, 0.0]])
    assert safety.check(near, 0.0, 0.0, people=near, extra=safety.person_extra(1.0))[0] == "clear"
    assert safety.check(near, 0.1, 0.0, people=near, extra=safety.person_extra(1.0))[0] == "stop"
    assert safety.check(near + [0.3, 0.0], 0.1, 0.0, people=near + [0.3, 0.0], extra=safety.person_extra(0.5))[0] \
        != "stop"


def test_docking_margins_smaller():
    table = np.array([[safety.HALF_LENGTH + 0.05, 0.0]])
    assert safety.check(table, 0.1, 0.0)[0] == "stop"
    assert safety.check(table, 0.1, 0.0, docking=True)[0] != "stop"


def test_rotate_straight_rotate():
    p = rotate_straight_rotate((0, 0, 0), (1.0, 0, np.pi / 2))
    assert np.allclose(p[-1], (1.0, 0, np.pi / 2))
    back = rotate_straight_rotate((0, 0, 0), (-0.3, 0, 0), reverse_ok=True)
    assert np.allclose(back[:, 2], 0.0)  # backs straight, no turn
    turn_only = rotate_straight_rotate((0, 0, 0), (0.01, 0, 0.2), min_dist=0.02)
    assert np.allclose(turn_only[:, :2], 0.0) and np.isclose(turn_only[-1, 2], 0.2)


def test_planner_round_a_table():
    g = room(table=(2.5, 3.5, 0.0, 2.6))
    planner = HybridAStar(Costmap(g))
    start, goal = np.array([1.0, 1.0, 0.0]), np.array([5.0, 1.0, 0.0])
    path = planner.plan(start, goal)
    assert path is not None
    assert np.allclose(path[0], start) and np.hypot(*(path[-1, :2] - goal[:2])) < 0.06
    assert abs(wrap(path[-1, 2] - goal[2])) < np.radians(3)
    assert (planner.cm.footprint_clearance(path) >= planner.margin - 0.05).all()
    assert path[:, 1].max() > 2.6 + 0.3  # round the table's free end


def test_reference_advances_and_brakes():
    path = np.column_stack([np.linspace(0, 2, 11), np.zeros(11), np.zeros(11)])
    ref = RouteReference(path)
    h = ref.horizon(20, 0.1)
    assert np.all(np.diff(h[:, 0]) >= 0) and h[0, 3] > 0.0
    ref.update((1.0, 0.02, 0.0))
    assert abs(ref.path[ref.progress, 0] - 1.0) < 0.03
    assert ref.horizon(20, 0.1)[-1, 3] < 0.5  # braking to the end


class FakeMPC:
    """Stands in for the solver: predicts the reference itself."""

    def __init__(self):
        self.cfg = BaseMPCConfig()
        self.warm = False

    def solve(self, x0, ref, people, v_max=None):
        return 0, np.array(ref, dtype=float)


def test_planning_in_the_background():
    """The control steps go on while the route is planned: standing still for a goal's
    first route, on the current route while planning round someone standing."""
    g = room(table=(2.5, 3.5, 0.0, 2.6))
    docks, home, pose = {"far": (5.5, 1.0, 0.0)}, (1.0, 1.0, 0.0), np.array([1.0, 1.0, 0.0])
    sync = Navigator(g, docks, home, mpc=FakeMPC())
    sync.set_goal("far", pose)
    assert sync.state == "route"
    nav = Navigator(g, docks, home, mpc=FakeMPC(), background=True)
    nav.set_goal("far", pose)
    assert nav.state == "planning"
    slowest = 0.0
    for k in range(500):
        t0 = time.perf_counter()
        cmd = nav.step(0.1 * k, pose, (0.0, 0.0), [])
        slowest = max(slowest, time.perf_counter() - t0)
        if nav.state != "planning":
            break
        assert cmd == (0.0, 0.0)
        time.sleep(0.01)
    assert nav.state == "route" and np.allclose(nav.ref.path, sync.ref.path)
    assert slowest < 0.05
    old = nav.ref
    nav._plan(pose, [(2.0, 1.0, 0.0, 0.0)], keep_route=True)
    assert nav.state == "route" and nav.pending is not None
    assert nav.step(60.0, pose, (0.0, 0.0), []) != (0.0, 0.0) or nav.pending is None
    while nav.pending is not None:
        nav.step(61.0, pose, (0.0, 0.0), [])
        time.sleep(0.01)
    assert nav.ref is not old


def test_a_late_route_does_not_take_over_docking():
    """A route round someone that comes after the base has gone on to align (or dock) is
    dropped: it starts behind the base, and as a route the table ahead is an obstacle."""
    g = room(table=(2.5, 3.5, 0.0, 2.6))
    docks, home, pose = {"far": (5.5, 1.0, 0.0)}, (1.0, 1.0, 0.0), np.array([1.0, 1.0, 0.0])
    nav = Navigator(g, docks, home, mpc=FakeMPC(), background=True)
    nav.set_goal("far", pose)
    while nav.state == "planning":
        nav.step(0.0, pose, (0.0, 0.0), [])
        time.sleep(0.01)
    nav._plan(pose, [(2.0, 1.0, 0.0, 0.0)], keep_route=True)
    nav._align(pose)
    while nav.pending is not None:
        nav.step(1.0, pose, (0.0, 0.0), [])
        time.sleep(0.01)
    assert nav.state == "align"


def test_steps_back_when_a_route_round_someone_is_blocked():
    """Blocked on a route round someone standing just ahead: back off, then hold and plan."""
    g = room()
    docks, home, pose = {"far": (5.0, 2.0, 0.0)}, (1.0, 2.0, 0.0), np.array([2.0, 2.0, 0.0])
    nav = Navigator(g, docks, home, mpc=FakeMPC())
    nav.set_goal("far", pose)
    person = [(2.8, 2.0, 0.0, 0.0)]
    nav.rerouted = True
    nav.step(0.0, pose, (0.0, 0.0), person)
    nav.step(5.0, pose, (0.0, 0.0), person)
    assert nav.state == "step_back" and nav.ref.sign[0] < 0
    nav._next(pose)
    assert nav.state == "held" and nav.step(5.1, pose, (0.0, 0.0), person) == (0.0, 0.0)
    nav.step(6.2, pose, (0.0, 0.0), [])
    assert nav.state == "route"


def test_off_the_docking_line_only_with_the_arm_stowed():
    """Undocking goes on with the arm out; the route waits for it; a pause stops the base."""
    g = room()
    docks, home = {"a": (2.0, 1.0, 0.0), "b": (4.5, 3.0, 0.0)}, (1.0, 3.0, 0.0)
    nav = Navigator(g, docks, home, mpc=FakeMPC())
    nav.at, nav.arm_ready = "a", False
    pose = np.array(docks["a"])
    nav.set_goal("b", pose)
    assert nav.state == "undock" and nav.step(0.0, pose, (0.0, 0.0), []) != (0.0, 0.0)
    nav.paused = True
    assert nav.step(0.1, pose, (0.0, 0.0), []) == (0.0, 0.0)
    nav.paused = False
    nav._next(pose)
    assert nav.state == "route" and nav.step(0.2, pose, (0.0, 0.0), []) == (0.0, 0.0)
    nav.arm_ready = True
    assert nav.step(0.3, pose, (0.0, 0.0), []) != (0.0, 0.0)


def test_crowd_steps_round_a_blocking_robot():
    """Someone walking into the robot's body waits, then steps aside and walks on."""
    crowd = Crowd([("p", [(2.0, 4.0), (6.0, 4.0)], 1.0, 0.0, 0.0, False)])
    robot = (3.0, 4.0)
    xs = []
    for _ in range(int((3 * BLOCKED_MAX_S + 8.0) / 0.02)):
        xs.append(crowd.step(0.02, robot)[0][:2])
    xs = np.array(xs)
    assert np.hypot(*(xs - robot).T).min() > 0.5
    assert xs[:, 0].max() > 5.0  # got past
