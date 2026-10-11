"""Compares a mode of the route's policy (tuned, rl) with the rules and the hybrid on the
evaluation set, paired by seed (handover_notes/rl/rl_crowd_nav_plan.md, sections 9-10):
per crowd, each metric's change against the rules and against the hybrid, the mean over
seeds with a 95% interval bootstrapped over seeds (a seed's drives share a world), and the
safety gates (hits: none in the hand-written crowds, no more than the rules' in the random
ones; the closest approaches' p5 no smaller than the hybrid's; every drive arrived).
Reads nav_batch.py's table (per run: <scenario>_s<seed>), writes a markdown report.
usage: python rl/evaluate.py --mode tuned [--json results/rl/baseline_rules.json] [--md results/rl/tuned_compare.md]
"""
import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
CROWDS = ("job2", "job4", "walkers3", "stations6", "dock_block", "step_in", "random6", "quiet")
RANDOM = ("random6",)
METRICS = (("time_s", "drive s"), ("route_stop_onsets", "stop onsets on routes"), ("swerves", "swerves"),
           ("extra_turn_deg_median", "extra turning deg"), ("route_warn_s", "warned s on routes"))
BOOT = 2000


def paired(runs, a, b, key):
    """b - a per seed: (mean, lo, hi, relative mean) or None."""
    seeds = sorted({int(k.rsplit("_s", 1)[1]) for k in runs if k.rsplit("_s", 1)[0] == a}
                   & {int(k.rsplit("_s", 1)[1]) for k in runs if k.rsplit("_s", 1)[0] == b})
    if not seeds:
        return None
    va = np.array([runs[f"{a}_s{s}"][key] for s in seeds], dtype=float)
    vb = np.array([runs[f"{b}_s{s}"][key] for s in seeds], dtype=float)
    d = vb - va
    idx = np.random.default_rng(0).integers(0, len(d), (BOOT, len(d)))
    lo, hi = np.percentile(d[idx].mean(axis=1), [2.5, 97.5])
    base = va.mean()
    return float(d.mean()), float(lo), float(hi), float(d.mean() / base) if base else np.nan


def p5(runs, sc, key):
    vals = [x for k, m in runs.items() if k.rsplit("_s", 1)[0] == sc for x in m[key]]
    return 1e3 * float(np.percentile(vals, 5)) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True)
    ap.add_argument("--json", default="results/rl/baseline_rules.json")
    ap.add_argument("--md", default="")
    args = ap.parse_args()
    data = json.loads((REPO / args.json).read_text())
    runs = data["runs"]
    lines = [f"# The route's policy `{args.mode}` against the rules and the hybrid (paired by seed)", "",
             f"From `{args.json}` (commit {data['meta']['commit']}). Change per drive, mean over seeds "
             f"[95% interval over seeds] (relative to the reference).", ""]
    gates = []
    for sc in CROWDS:
        mode, hyb = f"{sc}_{args.mode}", f"{sc}_hybrid"
        if not any(k.startswith(mode + "_s") for k in runs):
            continue
        lines += [f"## {sc}", "", "| metric | against the rules | against the hybrid |", "|---|---|---|"]
        for key, name in METRICS:
            cells = []
            for ref in (sc, hyb):
                r = paired(runs, ref, mode, key)
                cells.append("-" if r is None else f"{r[0]:+.2f} [{r[1]:+.2f}, {r[2]:+.2f}] ({100 * r[3]:+.0f}%)")
            lines.append(f"| {name} | {cells[0]} | {cells[1]} |")
        m_runs = [m for k, m in runs.items() if k.rsplit("_s", 1)[0] == mode]
        r_runs = [m for k, m in runs.items() if k.rsplit("_s", 1)[0] == sc]
        hits = sum(m["intrusions"] + m["contacts"] for m in m_runs)
        rule_hits = sum(m["intrusions"] + m["contacts"] for m in r_runs)
        arrived = sum(m["arrived"] for m in m_runs), sum(m["drives"] for m in m_runs)
        clear = {who: (p5(runs, who, "min_clear_per_drive"), p5(runs, who, "body_closing_per_drive"))
                 for who in (sc, hyb, mode)}
        lines += ["", f"Hits (intrusions + contacts) {hits} (the rules {rule_hits}); arrived "
                      f"{arrived[0]}/{arrived[1]}; closest while closing, p5, chassis / arm or box: " + "; ".join(
                          f"{who} " + " / ".join("-" if v is None else f"{v:.0f} mm" for v in clear[who])
                          for who in (sc, hyb, mode)), ""]
        ok_hits = hits <= rule_hits if sc in RANDOM else hits == 0
        ok_clear = all(clear[mode][i] is None or clear[hyb][i] is None or clear[mode][i] >= clear[hyb][i] - 10.0
                       for i in (0, 1))
        gates.append((sc, ok_hits, ok_clear, arrived[0] == arrived[1]))
    lines += ["## Safety gates", "", "| crowd | hits | closest p5 (no smaller than the hybrid's, 10 mm slack) | "
                                     "every drive arrived |", "|---|---|---|---|"]
    lines += [f"| {sc} | {'ok' if h else 'FAIL'} | {'ok' if c else 'FAIL'} | {'ok' if a else 'FAIL'} |"
              for sc, h, c, a in gates]
    text = "\n".join(lines) + "\n"
    if args.md:
        (REPO / args.md).write_text(text)
        print(f"wrote {args.md}")
    print(text)


if __name__ == "__main__":
    main()
