"""The MPC's reference along a planned route: the path (poses, densely sampled) gets a
time profile (cruising speed on straights, slower on arcs, turns in place at a set
rate, braking to a stop at the end) and each solve samples it over the horizon from
the robot's progress along the path, which only moves forward. Framework-free.
"""
import numpy as np

V_CRUISE = 0.5
W_TURN = 0.6  # rad/s, turning in place and the most on arcs
V_REVERSE = 0.15
A_BRAKE = 0.4  # m/s^2 for the stop at the end, and into a slow end
PROGRESS_WINDOW = 100  # samples searched ahead for the robot's progress


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


STEP_M = 0.02
STEP_RAD = np.radians(2)


def resample(path):
    """The path with no two consecutive poses more than STEP_M or STEP_RAD apart."""
    out = [path[0]]
    for a, b in zip(path[:-1], path[1:]):
        n = int(max(np.hypot(b[0] - a[0], b[1] - a[1]) / STEP_M, abs(wrap(b[2] - a[2])) / STEP_RAD)) + 1
        f = np.linspace(0.0, 1.0, n + 1)[1:, None]
        out.extend(np.column_stack([a[:2] + f * (b[:2] - a[:2]), a[2] + f[:, 0] * wrap(b[2] - a[2])]))
    return np.array(out)


class RouteReference:
    def __init__(self, path, v_cruise=V_CRUISE, w_turn=W_TURN, v_reverse=V_REVERSE, slow_end=None, a_brake=A_BRAKE,
                 v_end=0.0):
        """path: (n, 3) poses (x, y, yaw); resampled finely, so the robot's progress
        cannot stall between two distant samples. slow_end: (metres, speed) the last
        metres at most that speed (docking: fast, then slow by the table); v_end: the speed
        at the end (rolling on into what follows), else it brakes to a stop."""
        self.path = resample(np.asarray(path, dtype=float))
        d = np.diff(self.path, axis=0)
        ds = np.hypot(d[:, 0], d[:, 1])
        dth = np.abs(wrap(d[:, 2]))
        forward = np.einsum("ij,ij->i", d[:, :2], np.column_stack([np.cos(self.path[:-1, 2]), np.sin(self.path[:-1, 2])]))
        self.sign = np.where(forward < -1e-6, -1.0, 1.0)
        remaining = np.r_[np.cumsum(ds[::-1])[::-1], 0.0]
        # Speed per segment: cruise, limited by the turn rate on arcs and by braking to the end.
        curv = np.where(ds > 1e-6, dth / np.maximum(ds, 1e-9), np.inf)
        v = np.minimum(v_cruise, np.where(np.isfinite(curv), w_turn / np.maximum(curv, 1e-9), 0.0))
        v = np.minimum(v, np.sqrt(v_end ** 2 + 2 * a_brake * remaining[1:]) + 0.05)
        if slow_end is not None:
            slow_m, v_slow = slow_end
            v = np.minimum(v, np.sqrt(v_slow ** 2 + 2 * a_brake * np.maximum(remaining[1:] - slow_m, 0.0)))
        v = np.where(self.sign < 0, np.minimum(v, v_reverse), v)
        dt = np.where(ds > 1e-6, ds / np.maximum(v, 1e-3), dth / w_turn)
        self.t = np.r_[0.0, np.cumsum(dt)]
        self.progress = 0
        self.length = float(remaining[0])

    def update(self, pose):
        """Advance the progress to the path sample nearest the robot's pose (only forward,
        within a window); returns it."""
        lo, hi = self.progress, min(self.progress + PROGRESS_WINDOW, len(self.path))
        seg = self.path[lo:hi]
        cost = np.hypot(seg[:, 0] - pose[0], seg[:, 1] - pose[1]) + 0.3 * np.abs(wrap(seg[:, 2] - pose[2]))
        self.progress = lo + int(np.argmin(cost))
        return self.progress

    def horizon(self, n, dt):
        """(n+1, 5) reference states [x, y, yaw, v, omega] from the progress on, dt apart."""
        t0 = self.t[self.progress]
        ts = t0 + dt * np.arange(n + 1)
        i = np.clip(np.searchsorted(self.t, ts, side="right") - 1, 0, len(self.path) - 1)
        j = np.minimum(i + 1, len(self.path) - 1)
        span = np.maximum(self.t[j] - self.t[i], 1e-9)
        f = np.clip((ts - self.t[i]) / span, 0.0, 1.0)
        a, b = self.path[i], self.path[j]
        x = a[:, 0] + f * (b[:, 0] - a[:, 0])
        y = a[:, 1] + f * (b[:, 1] - a[:, 1])
        th = a[:, 2] + f * wrap(b[:, 2] - a[:, 2])
        moving = j > i
        ds = np.hypot(b[:, 0] - a[:, 0], b[:, 1] - a[:, 1])
        sign = self.sign[np.minimum(i, len(self.sign) - 1)]
        v = np.where(moving & (ts < self.t[-1]), sign * ds / span, 0.0)
        w = np.where(moving & (ts < self.t[-1]), wrap(b[:, 2] - a[:, 2]) / span, 0.0)
        return np.column_stack([x, y, th, v, w])

    def done(self, pose, tol_xy=0.05, tol_yaw=np.radians(3)):
        g = self.path[-1]
        return np.hypot(pose[0] - g[0], pose[1] - g[1]) < tol_xy and abs(wrap(pose[2] - g[2])) < tol_yaw
