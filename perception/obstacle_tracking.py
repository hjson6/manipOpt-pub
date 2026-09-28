"""Obstacle tracking, classification and the separation policy (numpy only).

Per frame (detections in the base frame): associate blobs with tracks
(nearest centroid, gated), alpha-beta filter, classify. The caller turns
person tracks into hold / speed scale with protective_distance and
speed_scale.

Classes: `static` only after a still confirmation window, and never for
anything person-tall or anything that has ever moved. Everything else,
including anything uncertain, is `person` (the fail-safe side).
"""
from dataclasses import dataclass, field

import numpy as np

GATE_BASE_M = 0.15  # association radius floor
GATE_SPEED_MPS = 2.5  # plus this * dt (a person at 1.6 m/s, with margin)
ALPHA, BETA = 0.85, 0.5  # high: a walking person does not manoeuvre much
LOST_TIMEOUT_S = 1.0  # coast an unseen track this long (occlusion)
STATIC_LOST_TIMEOUT_S = 10.0  # longer for static: the arm occludes it while passing beside it

MOVING_SPEED_MPS = 0.25  # faster = moving
CONFIRM_S = 1.0  # seen still this long to become static
STATIC_SPEED_MPS = 0.10  # still: slower than this
STATIC_DRIFT_M = 0.05  # and within this of where the still window began
STATIC_BREAK_M = 0.15  # static reverts if it moves this far from its anchor
PERSON_LATCH_S = 3.0  # stays person this long after it last moved
MOVING_MIN_FRAMES = 2  # consecutive fast frames for "moving" (one noisy velocity is not)
STATIC_BREAK_FRAMES = 6  # fast frames to revert static (occlusion glitches last a few)

# Anything this tall is a person, whatever it does: 0.5 m in this half-scale
# cell (1.0 m full size); cartons and carts are well below.
PERSON_MIN_HEIGHT_M = 0.5

PERSON, STATIC = "person", "static"


@dataclass
class Track:
    id: int
    pos: np.ndarray  # filtered [x, y, z]
    radius: float
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    t_first: float = 0.0
    t_last_seen: float = 0.0
    t_last_moving: float = -1e9
    fast_frames: int = 0
    anchor: np.ndarray = None  # where the current still window began
    still_since: float = 0.0  # and when
    n_seen: int = 1
    label: str = PERSON  # fail-safe until proven static
    visible: bool = True
    z_max: float = 0.0  # top of the object (a column from the floor)
    ever_moved: bool = False  # has moved once: a person for good
    still_r_max: float = 0.0  # largest radius in the current still window
    # Latched at confirmation, before any occlusion; what a static track reports.
    static_pos: np.ndarray = None
    static_r: float = 0.0

    @property
    def speed(self):
        return float(np.linalg.norm(self.vel[:2]))


class Tracker:
    def __init__(self):
        self.tracks = []
        self._next_id = 1
        self._t = None

    def update(self, t, blobs):
        """blobs: (N, >= 4) rows [x, y, z, radius, (n_px, z_max)] at capture time t.
        Returns the tracks.
        """
        dt = 0.0 if self._t is None else max(t - self._t, 1e-3)
        self._t = t
        for tr in self.tracks:
            tr.pos = tr.pos + tr.vel * dt  # predict
            tr.visible = False

        free = list(range(len(blobs)))
        # Greedy nearest-first association within the gate.
        gate = GATE_BASE_M + GATE_SPEED_MPS * dt
        pairs = sorted(
            ((np.linalg.norm(blobs[j, :2] - tr.pos[:2]), i, j)
             for i, tr in enumerate(self.tracks) for j in free),
            key=lambda p: p[0])
        used_tracks = set()
        for d, i, j in pairs:
            if d > gate or i in used_tracks or j not in free:
                continue
            used_tracks.add(i)
            free.remove(j)
            self._measure(self.tracks[i], blobs[j], t, dt)
        for j in free:
            self._spawn(blobs[j], t)

        self.tracks = [tr for tr in self.tracks
                       if t - tr.t_last_seen <= (STATIC_LOST_TIMEOUT_S if tr.label == STATIC else LOST_TIMEOUT_S)]
        for tr in self.tracks:
            self._classify(tr, t)
        return self.tracks

    def _spawn(self, b, t):
        pos = np.array(b[:3], dtype=float)
        self.tracks.append(Track(
            id=self._next_id, pos=pos, radius=float(b[3]), t_first=t, t_last_seen=t,
            anchor=pos.copy(), still_since=t, still_r_max=float(b[3]),
            z_max=float(b[5]) if len(b) > 5 else float(b[2] + b[3])))
        self._next_id += 1

    def _measure(self, tr, b, t, dt):
        z = np.array(b[:3], dtype=float)
        innov = z - tr.pos
        tr.pos = tr.pos + ALPHA * innov
        if dt > 0:
            tr.vel = tr.vel + BETA * innov / dt
        tr.radius = float(b[3])
        tr.z_max = float(b[5]) if len(b) > 5 else float(b[2] + b[3])
        tr.still_r_max = max(tr.still_r_max, tr.radius)
        tr.t_last_seen = t
        tr.visible = True
        tr.n_seen += 1

    def _classify(self, tr, t):
        if tr.speed > MOVING_SPEED_MPS:
            tr.fast_frames += 1
        else:
            tr.fast_frames = 0
        if tr.fast_frames >= MOVING_MIN_FRAMES:
            tr.t_last_moving = t
            tr.ever_moved = True
        if tr.ever_moved or tr.z_max >= PERSON_MIN_HEIGHT_M:
            tr.label = PERSON
            return
        if tr.label == STATIC:
            # Hysteresis: the arm occluding part of the blob shifts its centroid by cm,
            # which is not motion. Revert only on sustained speed or real displacement.
            if (tr.fast_frames >= STATIC_BREAK_FRAMES
                    or np.linalg.norm(tr.pos[:2] - tr.anchor[:2]) >= STATIC_BREAK_M):
                tr.label = PERSON
                tr.t_last_moving = t
                tr.anchor = tr.pos.copy()
                tr.still_since = t
            return
        # Restart the still window when fast or drifted.
        if (tr.speed >= STATIC_SPEED_MPS
                or np.linalg.norm(tr.pos[:2] - tr.anchor[:2]) >= STATIC_DRIFT_M):
            tr.anchor = tr.pos.copy()
            tr.still_since = t
            tr.still_r_max = tr.radius
        confirmed_still = (
            t - tr.still_since >= CONFIRM_S
            and t - tr.t_last_moving > PERSON_LATCH_S)
        if confirmed_still:
            tr.label = STATIC
            tr.static_pos = tr.anchor.copy()
            tr.static_r = tr.still_r_max
        else:
            tr.label = PERSON  # moving, latched or unconfirmed


def protective_distance(human_speed, robot_speed_toward, latency_s, a_max, uncertainty_m):
    """Required separation, ISO 13855 style (S = K*T + C): the person walks
    through the reaction and stopping time; the robot closes during the
    reaction time, then brakes; plus an uncertainty pad. Returns
    (required_m, t_stop_s).
    """
    v = max(robot_speed_toward, 0.0)
    t_stop = v / a_max
    required = (human_speed * (latency_s + t_stop)
                + v * latency_s + v * v / (2.0 * a_max)
                + uncertainty_m)
    return required, t_stop


def speed_scale(gap_m, required_m, ramp_m, crawl=0.15):
    """Speed limit from the margin d = gap - required: 0 (hold) at d <= 0, `crawl`
    just above, 1.0 once ramp_m clear.
    """
    d = gap_m - required_m
    if d <= 0.0:
        return 0.0
    return crawl + (1.0 - crawl) * min(d / ramp_m, 1.0)
