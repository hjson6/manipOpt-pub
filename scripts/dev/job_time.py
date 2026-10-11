"""Where a mobile job's time goes (telemetry of job_run.sh runs): the job from the first
navigation goal to parking at home, per box; the navigation's states (planning, route,
align, approach, docked, undock); the arm's (working at a station, tucking, stowed,
unfolding); the time the arm moved while the base docked or undocked (overlap), and
the time the base waited off the docking line for the arm (gated).
usage: python job_time.py <run_dir> [<run_dir> ...]
"""
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

TASK_STATES = ("returning", "awaiting_scan", "pausing", "moving_to_box", "traveling_to_dest", "awaiting_dest_scan",
               "pausing_dest", "moving_to_slot", "parked", "tucking", "driving", "untucking")
MOVING = ("undock", "approach", "backout")


def series(path, key):
    rows = list(csv.DictReader(open(path)))
    return np.array([float(r["wall_t"]) for r in rows]), [r[key] for r in rows]


def state_at(t, ts, states):
    i = np.clip(np.searchsorted(ts, t, side="right") - 1, 0, len(ts) - 1)
    return [states[j] for j in i]


def summary(run):
    """{total, boxes, nav: {state: s}, arm: {state: s}, work, overlap, gated} of one run."""
    tn, nav = series(run / "nav.csv", "state")
    tt, task = series(run / "task.csv", "state")
    task = [TASK_STATES[int(s)] for s in task]
    done = re.search(r"job done: (\d+) boxes", (run / "launch.log").read_text())
    t0 = next(t for t, s in zip(tn, nav) if s != "idle")
    t1 = next((t for t, s in zip(tn, nav) if t > t0 and s == "parked"), tn[-1])
    grid = np.arange(t0, t1, 0.02)
    nav_time, arm_time = defaultdict(float), defaultdict(float)
    overlap = gated = 0.0
    for ns, a in zip(state_at(grid, tn, nav), state_at(grid, tt, task)):
        nav_time[ns] += 0.02
        arm_time[a] += 0.02
        overlap += 0.02 if ns in MOVING and a != "driving" else 0.0
        gated += 0.02 if ns in ("route", "align") and a != "driving" else 0.0
    work = sum(v for k, v in arm_time.items() if k not in ("tucking", "driving", "untucking", "parked"))
    return dict(total=t1 - t0, boxes=int(done.group(1)) if done else None, nav=nav_time, arm=arm_time, work=work,
                overlap=overlap, gated=gated)


def main():
    for run in map(Path, sys.argv[1:]):
        r = summary(run)
        total, boxes, arm = r["total"], r["boxes"], r["arm"]
        print(f"{run.name}: {total / 60:.1f} min, {boxes} boxes" + (f", {total / boxes:.0f} s per box" if boxes else ""))
        print("  base: " + ", ".join(f"{k} {v:.0f} s" for k, v in sorted(r["nav"].items(), key=lambda kv: -kv[1])))
        print(f"  arm: working {r['work']:.0f} s, tucking {arm['tucking']:.0f} s, stowed {arm['driving']:.0f} s, "
              f"unfolding {arm['untucking']:.0f} s")
        print(f"  both moving (the arm out while the base docks or undocks) {r['overlap']:.0f} s; the base waiting "
              f"for the arm off the docking line {r['gated']:.1f} s")


if __name__ == "__main__":
    main()
