"""Plant conditions that make the base's localization harder (handover_notes/ekf_plan.md):
a slippery patch, a worn tyre, a drifting gyro bias, the localization's scans dropping
out. Plant side only; the method is never told. See docs/implementation_notes.md#plant_conditionspy.
"""
import numpy as np

CONDITIONS = ("spill", "worn_tyre", "gyro_drift", "dropout")
SPILL_RECT = ((4.65, 5.55), (3.55, 4.95))  # room frame (x, y): where the robot turns in front of the place table
SPILL_FRICTION = 0.2  # wet floor; dry: 1.0
WORN_TYRE = 0.02  # the left tyre's radius this much smaller
GYRO_DRIFT_TAU_S = 60.0  # the bias wanders (first-order Gauss-Markov)...
GYRO_DRIFT_SIGMA = 0.005  # ...within this (rad/s, steady state)
DROPOUT_S = (1.0, 2.0)  # each gap's length
DROPOUT_EVERY_S = (8.0, 20.0)  # time from one gap's end to the next


def parse(text):
    """The set of conditions in a comma- or space-separated list ("" or "none": none)."""
    names = {n for n in str(text).replace(",", " ").split() if n != "none"}
    unknown = names - set(CONDITIONS)
    if unknown:
        raise ValueError(f"unknown plant condition(s) {sorted(unknown)}: {', '.join(CONDITIONS)}")
    return names


def on_spill(xy):
    (x0, x1), (y0, y1) = SPILL_RECT
    return x0 <= xy[0] <= x1 and y0 <= xy[1] <= y1


class Dropouts:
    """The localization's scan feed lost for DROPOUT_S at a time, seeded; the schedule starts
    at the first scan's time."""

    def __init__(self, rng):
        self.rng = rng
        self.start = self.end = None
        self.lost_s = 0.0

    def dropped(self, t):
        """True if a scan captured at time t is lost (t non-decreasing)."""
        if self.start is None:
            self.start = t + self.rng.uniform(*DROPOUT_EVERY_S)
            self.end = self.start + self.rng.uniform(*DROPOUT_S)
        while t >= self.end:
            self.lost_s += self.end - self.start
            self.start = self.end + self.rng.uniform(*DROPOUT_EVERY_S)
            self.end = self.start + self.rng.uniform(*DROPOUT_S)
        return t >= self.start
