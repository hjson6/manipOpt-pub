"""Localization scored on the truth (validation only): position and heading error, pose
jumps (the estimate's change per tick beyond the true motion), and, given the estimate's
covariance, its consistency (NEES). Shared by nav_sim.py and localization_monitor_node.
"""
import numpy as np

NEES_BAND = (0.216, 9.348)  # chi-square, 3 degrees of freedom, two-sided 95%


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class LocScore:
    def __init__(self):
        self.rows = []  # err x, y (map frame), err yaw, jump m, jump rad, nees
        self.prev = None

    def add(self, est, true, cov=None):
        e = np.array([est[0] - true[0], est[1] - true[1], wrap(est[2] - true[2])])
        jump = (np.hypot(*(e[:2] - self.prev[:2])), abs(wrap(e[2] - self.prev[2]))) if self.prev is not None else (0.0, 0.0)
        self.prev = e
        nees = float(e @ np.linalg.solve(cov, e)) if cov is not None else np.nan
        self.rows.append((*e, *jump, nees))

    def summary(self):
        """Dict of the scores (mm, deg)."""
        if not self.rows:
            return {}
        r = np.array(self.rows)
        pos, yaw = 1e3 * np.hypot(r[:, 0], r[:, 1]), np.degrees(np.abs(r[:, 2]))
        out = {"ticks": len(r), "pos_p50": np.median(pos), "pos_p95": np.percentile(pos, 95), "pos_max": pos.max(),
               "yaw_p50": np.median(yaw), "yaw_p95": np.percentile(yaw, 95), "yaw_max": yaw.max(),
               "jump_max": 1e3 * r[:, 3].max(), "jump_p99": 1e3 * np.percentile(r[:, 3], 99),
               "jump_yaw_max": np.degrees(r[:, 4].max())}
        nees = r[:, 5][np.isfinite(r[:, 5])]
        if len(nees):
            out.update(nees_mean=nees.mean(), nees_in_band=float(np.mean((nees >= NEES_BAND[0]) & (nees <= NEES_BAND[1]))))
        return {k: float(v) for k, v in out.items()}


def line(s):
    """One line of a summary."""
    if not s:
        return "no samples"
    out = (f"position p50/p95/max {s['pos_p50']:.1f}/{s['pos_p95']:.1f}/{s['pos_max']:.1f} mm, yaw p50/p95/max "
           f"{s['yaw_p50']:.2f}/{s['yaw_p95']:.2f}/{s['yaw_max']:.2f} deg, jumps max {s['jump_max']:.1f} mm "
           f"(p99 {s['jump_p99']:.2f}) {s['jump_yaw_max']:.2f} deg")
    if "nees_mean" in s:
        out += f", NEES mean {s['nees_mean']:.1f}, in the 95% band {100 * s['nees_in_band']:.0f}%"
    return out
