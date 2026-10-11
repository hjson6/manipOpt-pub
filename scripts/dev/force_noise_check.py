"""Replay a noise-free run's wrist forces with the load cell's noise and bias added,
and count false touch-down rule firings (task_node._on_wrist_force).

usage: python force_noise_check.py <run_dir> [<run_dir> ...]   (runs with sensor_noise:=false)

Per touch leg (task.csv touch_base set, MOVING_TO_SLOT, no hold), a firing is false
when the noise-free reading is still far from firing: fz above 80% of the baseline
and the sideways load 0.5 N under its limit. Variants:
  now        the rules as in task_node
  tare       sideways load taken relative to the hover's filtered reading
  avg3       tare, and both rules on the mean of the last 3 readings
light: every touch leg's vertical load rescaled to the lightest spec box (0.2 kg).
carry: share of carry ticks (true fz > 1 N, outside touch legs) whose raw sideways
load exceeds TOUCH_SIDE_FORCE_N (margin only; the rule is not armed there).
Exit 1 if the chosen variant (--variant, default avg3) has any false firing, light included.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_analyze import load  # noqa: E402

MOVING_TO_SLOT = 7
SIGMA, BIAS = 0.2, 0.5  # mujoco_sim_node FORCE_SIGMA_N, FORCE_BIAS_MAX_N
SIDE_N, MU, FRAC, MIN_W = 1.0, 0.8, 0.5, 0.5  # task_node TOUCH_*
EMA = 0.1
LIGHT_N = 0.2 * 9.81


def ema(x):
    out = np.empty_like(x)
    acc = x[0]
    for i, v in enumerate(x):
        acc = (1 - EMA) * acc + EMA * v
        out[i] = acc
    return out


def windows(task, steps):
    on = (task["state"] == MOVING_TO_SLOT) & np.isfinite(task["touch_base"]) & (task["hold"] == 0)
    on_steps = set(task["step"][on].astype(int))
    mask = np.array([s in on_steps for s in steps])
    idx = np.flatnonzero(mask)
    if not len(idx):
        return []
    cuts = np.flatnonzero(np.diff(idx) > 1)
    return [(a, b) for a, b in zip(np.r_[idx[0], idx[cuts + 1]], np.r_[idx[cuts], idx[-1]])]


def fires(f, fz_filt, a, b, variant):
    """First tick in [a, b] where a rule fires, per rule: (contact, side)."""
    base_z = fz_filt[a - 1]
    base_xy = np.array([ema(f[:a, 0])[-1], ema(f[:a, 1])[-1]]) if variant != "now" else np.zeros(2)
    seg = f[a:b + 1]
    if variant == "avg3":
        seg = np.array([f[max(a, i - 2):i + 1].mean(0) for i in range(a, b + 1)])
    side = np.hypot(seg[:, 0] - base_xy[0], seg[:, 1] - base_xy[1])
    taken = np.maximum(0.0, base_z - seg[:, 2])
    side_hit = np.flatnonzero(side > MU * taken + SIDE_N)
    low = (seg[:, 2] < FRAC * base_z) & (base_z > MIN_W)
    contact_hit = np.flatnonzero(low)
    return (a + contact_hit[0] if len(contact_hit) else None, a + side_hit[0] if len(side_hit) else None)


def check(run, trials, rng, variants, light=False):
    sim, task = load(run / "sim.csv"), load(run / "task.csv")
    steps = sim["step"].astype(int)
    f0 = np.stack([sim["fx"], sim["fy"], sim["fz"]], 1)
    wins = windows(task, steps)
    if light:
        for a, b in wins:
            f0[a - 80:b + 1, 2] *= LIGHT_N / ema(f0[:a, 2])[-1]
    fz0 = ema(f0[:, 2])
    early = []  # per window: last tick index where the noise-free reading is far from firing
    for a, b in wins:
        base = fz0[a - 1]
        side0 = np.hypot(f0[a:b + 1, 0], f0[a:b + 1, 1])
        far = (f0[a:b + 1, 2] > 0.8 * base) & (side0 < MU * np.maximum(0, base - f0[a:b + 1, 2]) + SIDE_N - 0.5)
        stop = np.flatnonzero(~far)
        early.append(a + (stop[0] if len(stop) else b - a + 1) - 1)
    in_win = np.zeros(len(steps), bool)
    for a, b in wins:
        in_win[a:b + 1] = True
    carry = (f0[:, 2] > 1.0) & ~in_win
    res = {v: [0, 0] for v in variants}
    carry_hits = 0
    for _ in range(trials):
        f = f0 + rng.uniform(-BIAS, BIAS, 3) + rng.normal(0, SIGMA, f0.shape)
        fz = ema(f[:, 2])
        carry_hits += int(np.sum(np.hypot(f[carry, 0], f[carry, 1]) > SIDE_N))
        for v in variants:
            for (a, b), last_far in zip(wins, early):
                c, s = fires(f, fz, a, b, v)
                res[v][0] += c is not None and c <= last_far
                res[v][1] += s is not None and s <= last_far
    legs = len(wins) * trials
    lens = [e - a + 1 for (a, _), e in zip(wins, early)]
    print(f"{run.name}{' (light)' if light else ''}: {len(wins)} touch legs, pre-contact ticks per leg median {np.median(lens):.0f} "
          f"max {max(lens)}; {trials} noise draws")
    for v in variants:
        print(f"  {v:7s} false contact {res[v][0]}/{legs} legs, false edge {res[v][1]}/{legs} legs")
    print(f"  carry: raw sideways > {SIDE_N} N on {carry_hits / max(1, trials * carry.sum()):.2%} of "
          f"{carry.sum()} ticks")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--variant", default="avg3")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    variants = ["now", "tare", "avg3"]
    bad = 0
    for r in args.runs:
        for light in (False, True):
            res = check(Path(r), args.trials, rng, variants, light)
            bad += sum(res[args.variant])
    print(f"{args.variant}: {'PASS' if bad == 0 else 'FAIL'} ({bad} false firings)")
    sys.exit(int(bad > 0))


if __name__ == "__main__":
    main()
