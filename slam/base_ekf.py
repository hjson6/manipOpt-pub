"""The base's pose in the map by an extended Kalman filter: the wheel encoders (speed) and
the gyro (turn rate) predict at 50 Hz, the lidar's scan match corrects at 15 Hz, weighted
by its own information matrix. Slip shows where the wheels disagree with the gyro or
change speed faster than the drives can move the chassis. A late scan is applied at its
capture time and the filter re-run to now. Taught: the wheel radius and track, the
encoder resolution, datasheet noise. Framework-free (numpy).

See docs/implementation_notes.md#base_ekfpy.
"""
import math
from collections import deque

import numpy as np

X, Y, TH, V, W, B = range(6)
ENCODER_COUNTS_PER_REV = 16384
GYRO_NOISE = 0.002  # rad/s per sample (datasheet)
GYRO_BIAS_MAX = 0.005  # rad/s (datasheet)
GYRO_BIAS_WALK = 2e-4  # rad/s per sqrt(s)
ACCEL_MAX, ALPHA_MAX = 0.5, 1.5  # m/s^2, rad/s^2: the base's limits
WHEEL_TOLERANCE = 0.01  # effective radius, each wheel (tyre tolerance and slip)
TRACK_TOLERANCE = 0.02
DIST_NOISE = 0.007  # m per sqrt(m) driven: the wheels' scale error, between scans
DECEL_MAX = 6.0  # m/s^2: the wheels' speed changing faster than the drives can move the chassis
SKID_DECEL_MIN = 1.0  # m/s^2: the least a skid decelerates (the worst floor, friction 0.1)
SKID_MIN_S = 0.2
SKID_SPEED_NOISE = 0.5  # m/s: the wheels' speed during a skid (locked, or nearly right on a dry floor)
STILL_AFTER_S = 0.3
STILL_NOISE = 1e-4
SCAN_FLOOR = (0.003, 0.003, np.radians(0.1))  # a match's error beyond its information
MAP_SIGMA = (0.0025, 0.0025, np.radians(0.04))  # the map against the room (mapping error p50 2-3 mm)
GATE = 11.34  # chi-square, 3 degrees of freedom, 99%
MIN_FITNESS = 0.4
RESET_AFTER = 5  # good matches rejected in a row: the covariance was too small
HISTORY_S = 1.0


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def motion(x, dt):
    """The state after dt at constant speed and turn rate (heading at the step's midpoint)."""
    v, w = x[V], x[W]
    thm = x[TH] + 0.5 * w * dt
    out = x.copy()
    out[X] += v * dt * math.cos(thm)
    out[Y] += v * dt * math.sin(thm)
    out[TH] = wrap(x[TH] + w * dt)
    return out


def motion_jacobian(x, dt):
    v, w = x[V], x[W]
    thm = x[TH] + 0.5 * w * dt
    c, s = math.cos(thm), math.sin(thm)
    F = np.eye(6)
    F[X, TH], F[X, V], F[X, W] = -v * dt * s, dt * c, -0.5 * v * dt * dt * s
    F[Y, TH], F[Y, V], F[Y, W] = v * dt * c, dt * s, 0.5 * v * dt * dt * c
    F[TH, W] = dt
    return F


class BaseEKF:
    def __init__(self, pose, wheel_radius, track, sigma0=(0.05, 0.05, np.radians(2.0))):
        self.r, self.track = wheel_radius, track
        self.x = np.array([*pose, 0.0, 0.0, 0.0], dtype=float)
        self.P = np.diag([sigma0[0] ** 2, sigma0[1] ** 2, sigma0[2] ** 2, 1e-4, 1e-4, GYRO_BIAS_MAX ** 2])
        step = 2 * np.pi / ENCODER_COUNTS_PER_REV * wheel_radius
        self.q_v, self.q_w = step / np.sqrt(2), np.sqrt(2) * step / track  # per (one tick's) second
        self.hist = deque()  # (t, x, P, z, skid time left) after each tick
        self.skid_s = 0.0  # time left of a skid: the wheels do not measure the chassis's speed
        self.t = None
        self.prev_wheels = None
        self.prev_v = 0.0
        self.still_s = 0.0
        self.pending = 0  # scans rejected in a row with a good match
        self.counts = {"slips": 0, "skids": 0, "accepted": 0, "rejected": 0, "poor": 0, "resets": 0, "late": 0}
        self.last = None  # the last scan's (status, innovation, d2)

    @property
    def pose(self):
        return self.x[:3].copy()

    @property
    def cov(self):
        """The pose's covariance against the room: the filter's in the map, plus the map's own."""
        return self.P[:3, :3] + np.diag(np.square(MAP_SIGMA))

    def tick(self, t, wheels, gyro_z, dt=None):
        """One sample at time t: wheel angles [left, right] (rad) and the gyro's mean yaw rate
        since the last; dt: the sample's own period if t is a message stamp (wall time) rather
        than the sensor's. Returns the pose."""
        wheels = np.asarray(wheels, dtype=float)
        if self.prev_wheels is None or self.t is None or t <= self.t:
            self.prev_wheels, self.t = wheels, t
            self.hist.append((t, self.x.copy(), self.P.copy(), None, self.skid_s))
            return self.pose
        dt = t - self.t if dt is None else dt
        dl, dr = (wheels - self.prev_wheels) * self.r
        self.prev_wheels = wheels
        self.still_s = self.still_s + dt if dl == 0.0 and dr == 0.0 else 0.0
        v = 0.5 * (dl + dr) / dt
        z = (dt, v, (dr - dl) / self.track / dt, (v - self.prev_v) / dt, float(gyro_z), self.still_s >= STILL_AFTER_S)
        self.prev_v = v
        self._step(z, count=True)
        self.t = t
        self.hist.append((t, self.x.copy(), self.P.copy(), z, self.skid_s))
        while self.hist and self.hist[0][0] < t - HISTORY_S:
            self.hist.popleft()
        return self.pose

    def _step(self, z, count=False):
        dt, v_enc, w_enc, a_enc, gyro, still = z
        P = self.P
        P[V, V] += (ACCEL_MAX * dt) ** 2
        P[W, W] += (ALPHA_MAX * dt) ** 2
        P[B, B] += GYRO_BIAS_WALK ** 2 * dt
        if still and self.skid_s <= 0.0:
            self._update(V, 0.0, STILL_NOISE)
            self._update(W, 0.0, STILL_NOISE)
        else:
            sv = math.hypot(self.q_v / dt, WHEEL_TOLERANCE * v_enc)
            sw = math.sqrt((self.q_w / dt) ** 2 + (1.4142 * WHEEL_TOLERANCE * v_enc / self.track) ** 2
                           + (TRACK_TOLERANCE * w_enc) ** 2)
            turn_slip = abs(w_enc - (gyro - self.x[B])) > 3.0 * math.sqrt(sw ** 2 + GYRO_NOISE ** 2 + P[B, B])
            skid = abs(a_enc) > DECEL_MAX
            d = v_enc - self.x[V]
            if skid and self.skid_s <= 0.0:  # a skid starts: the slide is unknown, up to the worst floor's
                along = np.array([math.cos(self.x[TH]), math.sin(self.x[TH])])
                P[:2, :2] += np.outer(along, along) * (self.x[V] ** 2 / (2 * SKID_DECEL_MIN)) ** 2
                P[V, V] += self.x[V] ** 2
                self.skid_s = max(abs(self.x[V]) / SKID_DECEL_MIN, SKID_MIN_S)
                if count:
                    self.counts["skids"] += 1
            if turn_slip or skid or self.skid_s > 0.0:
                if count:
                    self.counts["slips"] += 1
            if self.skid_s > 0.0:  # the wheels hardly measure the chassis: the scans will
                P[V, V] += (DECEL_MAX * dt) ** 2
                sv = SKID_SPEED_NOISE
                self.skid_s = max(self.skid_s - dt, 0.0)
            elif turn_slip:
                P[V, V] += d * d
                sv = max(sv, abs(d))
            self._update(V, v_enc, sv)
        self._update_gyro(gyro)
        self._propagate(dt)

    def _update(self, i, z, sigma):
        """z measures state i directly."""
        h = self.P[:, i].copy()
        k = h / (h[i] + sigma * sigma)
        self.x += k * (z - self.x[i])
        self.P -= np.outer(k, h)

    def _update_gyro(self, gyro):
        h = self.P[:, W] + self.P[:, B]
        k = h / (h[W] + h[B] + GYRO_NOISE ** 2)
        self.x += k * (gyro - self.x[W] - self.x[B])
        self.P -= np.outer(k, h)

    def _propagate(self, dt):
        F = motion_jacobian(self.x, dt)
        self.x = motion(self.x, dt)
        P = F @ self.P @ F.T
        q = DIST_NOISE ** 2 * abs(self.x[V]) * dt
        P[X, X] += q
        P[Y, Y] += q
        self.P = 0.5 * (P + P.T)

    def _index(self, t):
        """The history entry at time t (the nearest tick within half a tick), or None."""
        if not self.hist:
            return None
        ts = np.array([h[0] for h in self.hist])
        i = int(np.argmin(np.abs(ts - t)))
        half = 0.5 * (ts[1] - ts[0]) if len(ts) > 1 else 0.01
        return i if abs(ts[i] - t) <= half + 1e-6 else None

    def pose_at(self, t):
        """The filter's pose at time t from its history, or None."""
        i = self._index(t)
        return None if i is None else self.hist[i][1][:3].copy()

    def scan(self, t, match):
        """A scan captured at time t: match(prior pose) -> (pose, info, fitness) is the scan
        match from the filter's pose then; applied there and re-run to now. Returns the
        status: accepted, rejected (the gate), poor (the match), reset or late."""
        i = self._index(t)
        if i is None:
            self.counts["late"] += 1
            self.last = ("late", None, np.nan)
            return "late"
        _t, x, P, _z, skid_s = self.hist[i]
        pose, info, fitness = match(x[:3].copy())
        if fitness < MIN_FITNESS:
            self.counts["poor"] += 1
            self.last = ("poor", None, np.nan)
            return "poor"
        nu = np.asarray(pose, dtype=float) - x[:3]
        nu[2] = wrap(nu[2])
        R = np.linalg.inv(info + 1e-12 * np.eye(3)) + np.diag(np.square(SCAN_FLOOR))
        S = P[:3, :3] + R
        d2 = float(nu @ np.linalg.solve(S, nu))
        status = "accepted"
        if d2 > GATE:
            self.pending += 1
            if self.pending < RESET_AFTER:
                self.counts["rejected"] += 1
                self.last = ("rejected", nu, d2)
                return "rejected"
            P = P.copy()
            P[:3, :3] += np.diag(np.square(nu))
            S = P[:3, :3] + R
            status = "reset"
            self.counts["resets"] += 1
        else:
            self.counts["accepted"] += 1
        self.pending = 0
        K = np.linalg.solve(S, P[:3, :]).T
        x = x + K @ nu
        x[TH] = wrap(x[TH])
        P = P - K @ P[:3, :]
        self.x, self.P, self.skid_s = x, 0.5 * (P + P.T), skid_s
        self.hist[i] = (_t, self.x.copy(), self.P.copy(), _z, skid_s)
        for j in range(i + 1, len(self.hist)):
            tj, _x, _P, zj, _s = self.hist[j]
            if zj is not None:
                self._step(zj)
            self.hist[j] = (tj, self.x.copy(), self.P.copy(), zj, self.skid_s)
        self.last = (status, nu, d2)
        return status
