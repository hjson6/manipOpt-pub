"""The base's safety layer, separate from the planner and the MPC (as the arm's
supervisor is), in the manner of ISO 3691-4 / ANSI/RIA R15.08 (a software stand-in,
not certified): the protective field follows the commanded motion. It is the chassis
swept along its current arc for as long as the stop takes (reaction time, then braking),
plus a margin: driving straight a box ahead as long as the stopping distance, turning
the stretch of their circle the corners cover before they stop. Anything the lidars see
there that the robot is moving closer to stops the base. Warning fields beyond (the
corridor ahead, a ring round a turn in place) cap the speed or the turn to what stops,
with a buffer, short of the nearest point there (never below a floor). Docking switches to small
margins, so the table being docked to (5 cm away) is not an obstacle at docking speed.
Points that are people (the scan's foreground against the map, given) get a wider
margin: a person's arms and torso overhang their legs, which is all the scanners see,
and the folded arm reaches near the chassis's edge. Framework-free.

See docs/implementation_notes.md#safetypy.
"""
import numpy as np

HALF_LENGTH, HALF_WIDTH = 0.40, 0.28  # the chassis, base_link at its centre
T_REACT_S = 0.15  # scan period, processing and the drives' response
A_BRAKE = 1.0  # m/s^2 the drives guarantee, at the chassis's fastest point
MARGIN_M = 0.10
DOCK_MARGIN_M = 0.02
SIDE_MARGIN_M = 0.06
TURN_WIDEN_M = 0.25  # per rad/s of turning, the warning box
WARN_EXTRA_M = 0.5  # the stop's buffer in the corridor ahead
WARN_SPEED = 0.3
WARN_SWEEP_M = 0.15  # the corners' stop's buffer round a turn in place
WARN_TURN = 0.5  # rad/s
SELF_MARGIN_M = 0.02  # points on the robot itself are left out
V_TRANSLATE = 0.02  # slower and turning slower than W_TURN: standing, no field
W_TURN = 0.02
PATH_STEP_M = 0.01  # the fastest point's travel between samples of the stop
APPROACH_M = 0.001
SWEEP_R = float(np.hypot(HALF_LENGTH, HALF_WIDTH))  # the corners' circle when turning in place
PERSON_EXTRA_M = 0.10  # people's legs this much farther: their upper body overhangs
PERSON_MARGIN_MIN_M = 0.14  # a person's whole margin never below their overhang (0.135 m)
PERSON_MATCH_M = 0.15  # a scan point this near a foreground point is a person's


def stop_distance(v, docking=False):
    v = abs(v)
    return v * T_REACT_S + v * v / (2 * A_BRAKE) + (DOCK_MARGIN_M if docking else MARGIN_M)


def fastest(v, w):
    """A bound on the speed of the chassis's fastest point."""
    return abs(v) + abs(w) * SWEEP_R


def stop_time(v, w):
    """Seconds at the current speeds that cover the stop: the reaction at speed, then
    braking the fastest point at A_BRAKE."""
    return T_REACT_S + fastest(v, w) / (2 * A_BRAKE)


def stop_path(v, w):
    """Poses (n, 3) in base_link along the current arc until stopped."""
    f = stop_time(v, w)
    n = int(np.ceil(fastest(v, w) * f / PATH_STEP_M)) + 1
    t = np.linspace(0.0, f, max(n, 2))[1:]
    th = w * t
    if abs(w) > 1e-6:
        x, y = v / w * np.sin(th), v / w * (1.0 - np.cos(th))
    else:
        x, y = v * t, np.zeros_like(t)
    return np.column_stack([x, y, th])


def chassis_distance(points, pose=(0.0, 0.0, 0.0)):
    """Distance from points (base_link) to the chassis at pose (base_link); 0 inside."""
    c, s = np.cos(pose[2]), np.sin(pose[2])
    dx, dy = points[:, 0] - pose[0], points[:, 1] - pose[1]
    lx, ly = c * dx + s * dy, -s * dx + c * dy
    return np.hypot(np.maximum(np.abs(lx) - HALF_LENGTH, 0.0), np.maximum(np.abs(ly) - HALF_WIDTH, 0.0))


def protective(points, v, w, docking=False, grow=0.0):
    """Points (base_link) inside the protective field of the motion (v, omega): within the
    margin (plus grow, a radius round each point) of the chassis on its way to a stop,
    and nearer there than now."""
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    hit = np.zeros(len(pts), dtype=bool)
    if abs(v) < V_TRANSLATE and abs(w) < W_TURN:
        return hit
    margin = (DOCK_MARGIN_M if docking else MARGIN_M) + grow
    path = stop_path(v, w)
    reach = SWEEP_R + float(np.hypot(*path[-1, :2])) + margin + 0.01
    near = np.hypot(pts[:, 0], pts[:, 1]) < reach
    if not near.any():
        return hit
    p = pts[near]
    d_min = np.min([chassis_distance(p, pose) for pose in path], axis=0)
    hit[near] = (d_min < margin) & (d_min < chassis_distance(p) - APPROACH_M)
    return hit


def fit_speed(d, margin):
    """The highest speed whose stop (stop_distance with margin) fits in d."""
    room = d - margin
    if room <= 0.0:
        return 0.0
    return A_BRAKE * (-T_REACT_S + np.sqrt(T_REACT_S ** 2 + 2.0 * room / A_BRAKE))


def warning_caps(points, v, w):
    """(speed cap, turn rate cap; inf: none) for the points (base_link): moving, the
    nearest point in the corridor ahead (the chassis's width, wider when turning) leaves
    room for the stop and WARN_EXTRA_M; turning in place, the nearest point round it
    leaves the corners room for their stop and WARN_SWEEP_M. Never below WARN_SPEED and
    WARN_TURN: there the protective field decides."""
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    v_cap = w_cap = np.inf
    if abs(v) >= V_TRANSLATE and len(pts):
        hw = HALF_WIDTH + SIDE_MARGIN_M + TURN_WIDEN_M * abs(w)
        ahead = np.sign(v) * pts[:, 0] - HALF_LENGTH
        corridor = (ahead > 0.0) & (np.abs(pts[:, 1]) <= hw)
        if corridor.any():
            v_cap = max(WARN_SPEED, fit_speed(ahead[corridor].min(), MARGIN_M + WARN_EXTRA_M))
    if abs(w) > W_TURN and abs(v) < 0.05 and len(pts):
        r = np.hypot(pts[:, 0], pts[:, 1]).min() - SWEEP_R
        w_cap = max(WARN_TURN, fit_speed(r, MARGIN_M + WARN_SWEEP_M) / SWEEP_R)
    return v_cap, w_cap


def person_points(pts, people):
    """Which of pts (base_link) are people's: within PERSON_MATCH_M of a foreground point
    (people, base_link; the last round of the people detector, maybe a round behind)."""
    hit = np.zeros(len(pts), dtype=bool)
    people = np.asarray(people, dtype=float).reshape(-1, 2)
    if not len(people) or not len(pts):
        return hit
    for chunk in np.array_split(np.arange(len(pts)), max(1, len(pts) // 256)):
        dd = np.hypot(pts[chunk, None, 0] - people[None, :, 0], pts[chunk, None, 1] - people[None, :, 1])
        hit[chunk] = dd.min(axis=1) < PERSON_MATCH_M
    return hit


def person_extra(scale=1.0):
    """The extra margin for people's points with their whole margin (MARGIN_M +
    PERSON_EXTRA_M) scaled, never below PERSON_MARGIN_MIN_M."""
    return max(PERSON_MARGIN_MIN_M, scale * (MARGIN_M + PERSON_EXTRA_M)) - MARGIN_M


def check(points, v, w, docking=False, measured=None, people=None, extra=PERSON_EXTRA_M):
    """('clear' | 'warn' | 'stop', (speed cap, turn rate cap)) for the lidar points
    (base_link): the warning fields at the commanded speeds set the caps, the protective
    field of the capped command and of the measured speeds (v, omega) stops; points of
    people (people: foreground points, base_link) with extra more margin."""
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    own = (np.abs(pts[:, 0]) <= HALF_LENGTH + SELF_MARGIN_M) & (np.abs(pts[:, 1]) <= HALF_WIDTH + SELF_MARGIN_M)
    pts = pts[~own]
    cap = warning_caps(pts, v, w)
    person = person_points(pts, people) if people is not None else np.zeros(len(pts), dtype=bool)
    for mv, mw in [limit(v, w, "warn", cap)] + ([tuple(measured)] if measured is not None else []):
        if (protective(pts[~person], mv, mw, docking).any()
                or protective(pts[person], mv, mw, docking, grow=extra).any()):
            return "stop", (0.0, 0.0)
    return ("warn" if cap[0] < abs(v) or cap[1] < abs(w) else "clear"), cap


def limit(v, w, state, cap):
    """The commanded speeds after the safety layer: zero on stop, scaled to the caps in
    warning (the same curvature)."""
    if state == "stop":
        return 0.0, 0.0
    scale = min(1.0, cap[0] / max(abs(v), 1e-9), cap[1] / max(abs(w), 1e-9))
    return v * scale, w * scale
