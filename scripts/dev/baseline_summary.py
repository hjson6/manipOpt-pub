"""Step 5 baseline table: one line per bench_run.sh directory, then totals.

usage: python baseline_summary.py <run_dir> [<run_dir> ...]

Columns:
  done      the task parked with the pile empty
  boxes     boxes placed (floor + stacked)
  time      simulated time from the controller's start to parking [s], and per box
  arrive    TCP error when the controller declared arrival [mm], median / max
  t/o       legs that hit their time limit
  abort     places aborted and re-planned (side force, blocked)
  fail      solver failures
  wall      arm or tool contacts with a tray wall (events)
  yaw       box turn after pushes [deg], max |.|
  gap       final facing gaps from the sim's place/push lines [mm], median (min)
  j7        share of ticks with joint-7 torque jumping > 10 Nm (wrist chatter)
  solve     solve time p50 / p99 [ms]
  hold      obstacle holds / resumes
"""
import csv
import re
import sys
from pathlib import Path

import numpy as np


def summarise(run):
    log = (run / "launch.log").read_text(errors="replace")
    mpc = list(csv.DictReader(open(run / "mpc.csv")))
    sim = list(csv.DictReader(open(run / "sim.csv")))
    placed = re.findall(r"placed (cbox_\d) \(bottom at z=(-?[\d.]+) m", log)
    floor = sum(float(z) < 0.01 for _, z in placed)
    settled = np.array([int(r["settled"]) for r in mpc])
    err = np.array([float(r["ee_err"]) for r in mpc])
    edges = np.where(np.diff(settled) == 1)[0] + 1
    status = np.array([int(float(r["status"])) for r in mpc])
    solve = np.array([float(r["solve_ms"]) for r in mpc])
    t = np.array([float(r["sim_t"]) for r in sim])
    tau7 = np.array([float(r["tau7"]) for r in sim])
    yaws = [abs(float(y)) for y in re.findall(r"pushed cbox_\d: [^;]*yaw ([+-][\d.]+) deg", log)]
    yaws = [min(y, abs(y - 90)) for y in yaws]  # a box turned 90 deg at placement
    gaps = []
    for line in re.findall(r"side gaps -x/\+x/-y/\+y ([\d./-]+) mm", log):
        gaps += [float(g) for g in line.split("/") if g not in ("-",) and float(g) < 60]
    done = "container empty; parked" in log
    n = len(placed)
    dur = t[-1] - t[0] if len(t) else float("nan")
    return dict(
        name=run.name, done=done, boxes=f"{n} ({floor} floor)", n=n, floor=floor,
        time=f"{dur:.0f} / {dur / max(n, 1):.1f}", dur=dur if done else np.nan,
        arrive=f"{1e3 * np.median(err[edges]):.1f} / {1e3 * err[edges].max():.1f}" if len(edges) else "-",
        arr_max=1e3 * err[edges].max() if len(edges) else np.nan,
        to=log.count("not reached in"), abort=log.count("place blocked"),
        fail=int(np.sum(status != 0)),
        wall=len(re.findall(r"dest_tray_wall_\w+ / (?:link\d|tool)", log)),
        yaw=max(yaws) if yaws else 0.0,
        gap=f"{np.median(gaps):.1f} ({min(gaps):.1f})" if gaps else "-",
        j7=float(np.mean(np.abs(np.diff(tau7)) > 10)) if len(tau7) > 1 else 0.0,
        solve=f"{np.percentile(solve, 50):.1f} / {np.percentile(solve, 99):.1f}",
        hold=f"{len(re.findall(r'task_node.*supervisor HOLD', log))}/{len(re.findall(r'task_node.*supervisor RESUME', log))}",
    )


def main():
    rows = [summarise(Path(p)) for p in sys.argv[1:]]
    cols = ["name", "done", "boxes", "time", "arrive", "to", "abort", "fail", "wall", "yaw", "gap", "j7",
            "solve", "hold"]
    widths = {c: max(len(c), *(len(f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c])) for r in rows))
              for c in cols}
    fmt = lambda r, c: (f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c])).ljust(widths[c])
    print("  ".join(c.ljust(widths[c]) for c in cols))
    for r in rows:
        print("  ".join(fmt(r, c) for c in cols))
    k = len(rows)
    print(f"\nruns {k}: finished {sum(r['done'] for r in rows)}/{k}, boxes placed "
          f"{sum(r['n'] for r in rows)}/{8 * k} ({sum(r['floor'] for r in rows)} on the floor), "
          f"time per finished run {np.nanmean([r['dur'] for r in rows]):.0f} s, "
          f"worst arrival {np.nanmax([r['arr_max'] for r in rows]):.1f} mm, time-outs {sum(r['to'] for r in rows)}, "
          f"aborts {sum(r['abort'] for r in rows)}, solver failures {sum(r['fail'] for r in rows)}, "
          f"wall contacts {sum(r['wall'] for r in rows)}, worst push turn {max(r['yaw'] for r in rows):.1f} deg, "
          f"worst j7 chatter {max(r['j7'] for r in rows):.3f}")


if __name__ == "__main__":
    main()
