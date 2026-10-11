"""Runs nav_sim over a set of scenarios and seeds in parallel and tables the drives (the RL
plan's layer 1, step 0: today's rules on the evaluation set; later the same set in each
mode). Each run's --json and log go to <out>/runs/<set>/ (git-ignored; a run already
there is reused); the table to <out>/<name>.json (per scenario and seed, the drives'
metrics) and <name>.md (per scenario: the mean over seeds with a 95% interval
bootstrapped over seeds, since a seed's drives share one world; the closest approaches
pooled over the drives, p5 with its interval and the minimum). --only runs just those
scenarios; the table has every scenario of the set with runs there.
usage: python scripts/dev/nav_batch.py [--set baseline] [--seeds 0-19] [--drives 20] [--workers 12]
       [--out results/rl] [--name baseline_rules] [--only job4 quiet] [--table-only]
"""
import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
COMMON = ["--arm", "carry", "--box", "--fusion", "ekf", "--speed", "1.0"]
JOB4 = ["--people", "job", "--crowd", "4"]
SETS = {"baseline": {
    "quiet": ["--crowd", "0"],
    "job2": ["--people", "job", "--crowd", "2"],
    "job4": JOB4,
    "walkers3": ["--people", "walkers", "--crowd", "3"],
    "stations6": ["--people", "stations", "--crowd", "6"],
    "dock_block": ["--people", "dock_block", "--crowd", "1"],
    "step_in": ["--people", "step_in", "--crowd", "1"],
    "job4_close": JOB4 + ["--base-safety", "0.6"],  # the close setting: navigator and stop field at 0.6
    "job4_hybrid": JOB4 + ["--base-safety", "0.6", "--stop-safety", "1.0"],  # RL's boldest
    "walkers3_hybrid": ["--people", "walkers", "--crowd", "3", "--base-safety", "0.6", "--stop-safety", "1.0"],
    "stations6_hybrid": ["--people", "stations", "--crowd", "6", "--base-safety", "0.6", "--stop-safety", "1.0"],
    "random6": ["--people", "random", "--crowd", "6"],  # a crowd of up to six drawn per seed (random_crowd)
    "random6_hybrid": ["--people", "random", "--crowd", "6", "--base-safety", "0.6", "--stop-safety", "1.0"],
}}
# The tuned baseline (nav_policy tuned, policies/nav/tuned_v1.json) on the same crowds.
SETS["baseline"].update({f"{k}_tuned": v + ["--nav-policy", "tuned"] for k, v in list(SETS["baseline"].items())
                         if k in ("quiet", "job2", "job4", "walkers3", "stations6", "dock_block", "step_in",
                                  "random6")})
BOOT = 2000


def seeds(text):
    lo, _, hi = text.partition("-")
    return list(range(int(lo), int(hi or lo) + 1))


def run_one(scenario, args, seed, drives, runs):
    out = runs / f"{scenario}_s{seed}.json"
    if out.exists():
        return out, 0.0
    t0 = time.perf_counter()
    with open(runs / f"{scenario}_s{seed}.log", "w") as log:
        cmd = [sys.executable, str(REPO / "scripts/dev/nav_sim.py"), *args, *COMMON, "--seed", str(seed), "--drives",
               str(drives), "--json", str(out)]
        p = subprocess.run(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
    if p.returncode:
        raise RuntimeError(f"{scenario} seed {seed} failed: see {runs / f'{scenario}_s{seed}.log'}")
    return out, time.perf_counter() - t0


def metrics(run):
    """Per run (one scenario and seed): the per-drive means the gates read, and the
    closest approaches per drive for pooling."""
    ds = run["drive_scores"]
    ok = [d for d in ds if d["state"] in ("docked", "parked")]
    docked = np.array([d["err"] for d in ds if d["state"] == "docked"]).reshape(-1, 3)
    clear = [round(d["min_clear"], 4) for d in ds if d["min_clear"] is not None]
    body = [round(d["body_closing"][0], 4) for d in ds if d["body_closing"][0] is not None]
    extra = [d["extra_turn_deg"] for d in ds if d["extra_turn_deg"] is not None]
    return {
        "drives": len(ds), "arrived": len(ok), "timeouts": sum(d["state"] == "timeout" for d in ds),
        "failed": sum(d["state"] == "failed" for d in ds),
        "time_s": float(np.mean([d["t"] for d in ds])),
        "stop_onsets": float(np.mean([d["stop_onsets"] for d in ds])),
        "route_stop_onsets": float(np.mean([d["stop_onsets_by"].get("route", 0) for d in ds])),
        "route_warn_s": float(np.mean([d["warn_s_by"].get("route", 0.0) for d in ds])),
        "stop_s": float(np.mean([d["stop_s"] for d in ds])),
        "warn_s": float(np.mean([d["warn_s"] for d in ds])),
        "extra_turn_deg_median": float(np.median(extra)) if extra else None,
        "swerves": float(np.mean([d["wiggles"] for d in ds])),
        "oscillations": float(np.mean([d["oscillations"] for d in ds])),
        "replans": float(np.mean([d["events"].get("replans", 0) for d in ds])),
        "step_backs": float(np.mean([d["events"].get("step_backs", 0) for d in ds])),
        "intrusions": int(sum(d["intrusions"] for d in ds)), "contacts": int(sum(d["contacts"] for d in ds)),
        "dock_lateral_max_mm": float(1e3 * np.abs(docked[:, 1]).max()) if len(docked) else None,
        "dock_along_max_mm": float(1e3 * np.abs(docked[:, 0]).max()) if len(docked) else None,
        "dock_yaw_max_deg": float(np.abs(docked[:, 2]).max()) if len(docked) else None,
        "min_clear_per_drive": clear, "body_closing_per_drive": body,
    }


def boot_mean(values, rng):
    v = np.asarray([x for x in values if x is not None], dtype=float)
    if not len(v):
        return None
    idx = rng.integers(0, len(v), (BOOT, len(v)))
    lo, hi = np.percentile(v[idx].mean(axis=1), [2.5, 97.5])
    return [float(v.mean()), float(lo), float(hi)]


def boot_p5(per_seed, rng):
    """p5 of the drives pooled over seeds, with its interval resampling seeds."""
    per_seed = [np.asarray(x, dtype=float) for x in per_seed if len(x)]
    if not per_seed:
        return None
    pooled = np.concatenate(per_seed)
    idx = rng.integers(0, len(per_seed), (BOOT, len(per_seed)))
    p5 = [np.percentile(np.concatenate([per_seed[i] for i in row]), 5) for row in idx]
    lo, hi = np.percentile(p5, [2.5, 97.5])
    return [float(np.percentile(pooled, 5)), float(lo), float(hi), float(pooled.min())]


def table(per_run, scenarios):
    rng = np.random.default_rng(0)
    out = {}
    for sc in scenarios:
        ms = [m for (s, _), m in sorted(per_run.items()) if s == sc]
        if not ms:
            continue
        row = {k: boot_mean([m[k] for m in ms], rng) for k in (
            "time_s", "stop_onsets", "route_stop_onsets", "stop_s", "warn_s", "route_warn_s", "extra_turn_deg_median",
            "swerves", "oscillations", "replans", "step_backs")}
        row.update(seeds=len(ms), drives=sum(m["drives"] for m in ms), arrived=sum(m["arrived"] for m in ms),
                   timeouts=sum(m["timeouts"] for m in ms), failed=sum(m["failed"] for m in ms),
                   intrusions=sum(m["intrusions"] for m in ms), contacts=sum(m["contacts"] for m in ms),
                   min_clear_mm=boot_p5([1e3 * np.asarray(m["min_clear_per_drive"]) for m in ms], rng),
                   body_closing_mm=boot_p5([1e3 * np.asarray(m["body_closing_per_drive"]) for m in ms], rng))
        for k in ("dock_lateral_max_mm", "dock_along_max_mm", "dock_yaw_max_deg"):
            v = [m[k] for m in ms if m[k] is not None]
            row[k] = max(v) if v else None
        out[sc] = row
    return out


def fmt(v, digits=1):
    if v is None:
        return "-"
    return f"{v[0]:.{digits}f} [{v[1]:.{digits}f}, {v[2]:.{digits}f}]"


def markdown(rows, header):
    lines = [header, "",
             "| scenario | drive s | stop onsets, all / on routes | stopped s | warned s, all / on routes | "
             "extra turning deg | swerves | closest, chassis p5 / min mm | arm or box p5 / min mm | arrived | "
             "intrusions, contacts | docked max mm, deg |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for sc, r in rows.items():
        close = lambda v: "-" if v is None else f"{v[0]:.0f} [{v[1]:.0f}, {v[2]:.0f}] / {v[3]:.0f}"  # noqa: E731
        dock = "-" if r["dock_lateral_max_mm"] is None else (
            f"{r['dock_lateral_max_mm']:.0f} / {r['dock_along_max_mm']:.0f}, {r['dock_yaw_max_deg']:.2f}")
        lines.append(f"| {sc} | {fmt(r['time_s'])} | {fmt(r['stop_onsets'], 2)} / {fmt(r['route_stop_onsets'], 2)} | "
                     f"{fmt(r['stop_s'], 2)} | {fmt(r['warn_s'], 2)} / {fmt(r['route_warn_s'], 2)} | "
                     f"{fmt(r['extra_turn_deg_median'], 0)} | {fmt(r['swerves'], 2)} | "
                     f"{close(r['min_clear_mm'])} | {close(r['body_closing_mm'])} | {r['arrived']}/{r['drives']} | "
                     f"{r['intrusions']}, {r['contacts']} | {dock} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="baseline", choices=tuple(SETS))
    ap.add_argument("--seeds", default="0-19")
    ap.add_argument("--drives", type=int, default=20)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="results/rl")
    ap.add_argument("--name", default="baseline_rules")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--table-only", action="store_true")
    args = ap.parse_args()
    scenarios = SETS[args.set]
    out = REPO / args.out
    runs = out / "runs" / args.set
    runs.mkdir(parents=True, exist_ok=True)
    jobs = [(sc, s) for s in seeds(args.seeds) for sc in scenarios if args.only is None or sc in args.only]
    if not args.table_only:
        sys.path[:0] = [str(REPO)]
        from nav.base_mpc import BaseMPC
        BaseMPC(build_dir=str(REPO))  # built once here, so the workers only load it
        t0, done = time.perf_counter(), 0
        with ThreadPoolExecutor(args.workers) as pool:
            futs = [(j, pool.submit(run_one, j[0], scenarios[j[0]], j[1], args.drives, runs)) for j in jobs]
            for (sc, s), fut in futs:
                _, wall = fut.result()
                done += 1
                print(f"[{done}/{len(jobs)}, {time.perf_counter() - t0:.0f} s] {sc} seed {s}"
                      + (f" ({wall:.0f} s)" if wall else " (reused)"), flush=True)
    per_run = {}
    for sc, s in [(sc, s) for s in seeds(args.seeds) for sc in scenarios]:
        f = runs / f"{sc}_s{s}.json"
        if f.exists():
            per_run[(sc, s)] = metrics(json.loads(f.read_text()))
    rows = table(per_run, scenarios)
    commit = subprocess.run(["git", "describe", "--always", "--dirty"], cwd=REPO, capture_output=True,
                            text=True).stdout.strip()
    meta = {"set": args.set, "seeds": args.seeds, "drives": args.drives, "commit": commit, "common": COMMON,
            "scenarios": scenarios}
    (out / f"{args.name}.json").write_text(json.dumps(
        {"meta": meta, "table": rows,
         "runs": {f"{sc}_s{s}": m for (sc, s), m in sorted(per_run.items())}}, indent=1) + "\n")
    header = (f"# nav_sim, set {args.set}: seeds {args.seeds}, {args.drives} drives each, `{' '.join(COMMON)}` "
              f"(commit {commit})\n\nPer drive, the mean over seeds [95% interval, bootstrapped over seeds]; extra "
              f"turning: each seed's median. Closest approaches while the robot closes on someone: the drives' "
              f"p5 [interval] / minimum.")
    (out / f"{args.name}.md").write_text(markdown(rows, header))
    print(f"wrote {out / args.name}.json and .md")


if __name__ == "__main__":
    main()
