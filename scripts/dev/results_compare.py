"""Paired comparison of two methods or scenarios on the same seeds, from
results_summary.py's metrics.csv files (handover_notes/baseline_benchmark_plan.md).

usage: python results_compare.py <metrics_a.csv>:<scenario_a> <metrics_b.csv>:<scenario_b> [out.md]
  e.g. results/mpc/metrics.csv:plain results/lmpc/metrics.csv:plain

Per metric: median of the seed means for A and B, median per-seed difference
B - A, how many seeds B is lower, the Wilcoxon signed-rank p-value on the seed
means, and the difference over A's within-seed SD.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from results_summary import NOISE, noise, seed_means  # noqa: E402

METRICS = ["boxes", "done", "time_per_box_s", "pred_q_rms_mrad", "pred_qd_rms", "pred_qd_med",
           "track_rms_mm", "track_p99_mm", "tsway_rms_mm", "arrive_med_mm", "settle_med_s",
           "bias_mean_mm", "solve_p50_ms", "solve_p99_ms", "over_budget", "jerk_p99", "dtau_p99",
           "effort_per_box", "tilt_p99_deg", "gap_med_mm"]


def rows_of(spec):
    import csv
    path, scen = spec.rsplit(":", 1)
    rows = []
    for r in csv.DictReader(open(path)):
        if r["scenario"] != scen:
            continue
        r = {k: (v if k in ("run", "scenario") else (float(v) if v not in ("", "nan") else np.nan))
             for k, v in r.items()}
        r["seed"] = int(r["seed"])
        rows.append(r)
    return rows, scen


def main():
    (ra, sa), (rb, sb) = rows_of(sys.argv[1]), rows_of(sys.argv[2])
    lines = [f"# {sys.argv[2]} vs {sys.argv[1]}", "",
             "Seed means; diff = B - A per seed. p: Wilcoxon signed-rank, two-sided. "
             "diff/noise: median diff over A's within-seed SD.", "",
             "| metric | A median | B median | median diff | B lower (seeds) | p | diff/noise |",
             "|---|---|---|---|---|---|---|"]
    for key in METRICS:
        ma, mb = seed_means(ra, sa, key), seed_means(rb, sb, key)
        seeds = sorted(s for s in ma if s in mb and not np.isnan(ma[s]) and not np.isnan(mb[s]))
        if not seeds:
            continue
        a, b = np.array([ma[s] for s in seeds]), np.array([mb[s] for s in seeds])
        d = b - a
        p = wilcoxon(a, b).pvalue if np.any(d != 0) and len(seeds) >= 6 else np.nan
        within = noise(ra, sa, key)[0] if key in NOISE else np.nan
        ratio = np.median(d) / within if within and not np.isnan(within) else np.nan
        lines.append(f"| {key} | {np.median(a):.3g} | {np.median(b):.3g} | {np.median(d):+.3g} | "
                     f"{int(np.sum(d < 0))}/{len(seeds)} | {p:.3g} | {ratio:+.2g} |")
    text = "\n".join(lines) + "\n"
    print(text)
    if len(sys.argv) > 3:
        Path(sys.argv[3]).write_text(text)


if __name__ == "__main__":
    main()
