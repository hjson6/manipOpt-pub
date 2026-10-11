"""People tracks in the map frame: a constant-velocity Kalman filter per person (state
x, y, vx, vy), detections assigned to tracks by optimal matching (Hungarian) within
a gate around each prediction. A track is confirmed after CONFIRM_HITS matches,
coasts on its prediction through short occlusions and is dropped after LOST_S
unseen, or at once when the lidars saw through its predicted spot (a person who
turned leaves no ghost walking on); it is standing while its speed stays under
STANDING_SPEED_MS.
Framework-free (numpy, scipy).

See docs/implementation_notes.md#people_trackerpy.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

GATE_M = 0.6
CONFIRM_HITS = 3
LOST_S = 1.0
ACCEL_SIGMA = 2.0  # m/s^2: how sharply people change their walk
MEAS_SIGMA_M = 0.05
STANDING_SPEED_MS = 0.2
STANDING_AFTER_S = 1.0


class PersonTrack:
    def __init__(self, tid, xy, t):
        self.id = tid
        self.x = np.array([xy[0], xy[1], 0.0, 0.0])
        self.p = np.diag([MEAS_SIGMA_M ** 2, MEAS_SIGMA_M ** 2, 1.0, 1.0])
        self.hits = 1
        self.t = t
        self.last_seen = t
        self.moving_t = t
        self.radius = 0.3

    @property
    def confirmed(self):
        return self.hits >= CONFIRM_HITS

    @property
    def speed(self):
        return float(np.hypot(self.x[2], self.x[3]))

    @property
    def standing(self):
        return self.t - self.moving_t >= STANDING_AFTER_S

    def predict(self, t):
        dt = max(t - self.t, 0.0)
        f = np.eye(4)
        f[0, 2] = f[1, 3] = dt
        g = np.array([[dt ** 2 / 2, 0.0], [0.0, dt ** 2 / 2], [dt, 0.0], [0.0, dt]])
        self.x = f @ self.x
        self.p = f @ self.p @ f.T + g @ g.T * ACCEL_SIGMA ** 2
        self.t = t

    def correct(self, xy, radius):
        h = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
        s = h @ self.p @ h.T + np.eye(2) * MEAS_SIGMA_M ** 2
        k = self.p @ h.T @ np.linalg.inv(s)
        self.x = self.x + k @ (np.asarray(xy) - h @ self.x)
        self.p = (np.eye(4) - k @ h) @ self.p
        self.hits += 1
        self.last_seen = self.t
        self.radius = radius
        if self.speed > STANDING_SPEED_MS:
            self.moving_t = self.t


class PeopleTracker:
    def __init__(self):
        self.tracks = []
        self.next_id = 1

    def update(self, t, people, seen_empty=None):
        """people: [(x, y, radius, ...)] detected at time t (map frame); seen_empty(xy):
        whether the sensors saw through a spot; returns the confirmed tracks."""
        for tr in self.tracks:
            tr.predict(t)
        det = np.array([p[:3] for p in people], dtype=float).reshape(-1, 3)
        matched_tracks, matched_det = set(), set()
        if self.tracks and len(det):
            pred = np.array([tr.x[:2] for tr in self.tracks])
            cost = np.hypot(pred[:, None, 0] - det[None, :, 0], pred[:, None, 1] - det[None, :, 1])
            rows, cols = linear_sum_assignment(np.where(cost < GATE_M, cost, 1e6))
            for r, c in zip(rows, cols):
                if cost[r, c] < GATE_M:
                    self.tracks[r].correct(det[c, :2], det[c, 2])
                    matched_tracks.add(r)
                    matched_det.add(c)
        for c in range(len(det)):
            if c not in matched_det:
                self.tracks.append(PersonTrack(self.next_id, det[c, :2], t))
                self.next_id += 1
        self.tracks = [tr for i, tr in enumerate(self.tracks)
                       if t - tr.last_seen <= (LOST_S if tr.confirmed else 0.3)
                       and (i in matched_tracks or seen_empty is None or not seen_empty(tr.x[:2]))]
        return [tr for tr in self.tracks if tr.confirmed]
