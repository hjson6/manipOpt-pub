"""The scenario figure of mobile jobs (job_run.sh runs): per scenario, the time per box
by what the base was doing (driving a route, docking moves, standing docked) and the
closest anyone came to the chassis while it moved and to the arm or a held box.
usage: python job_figure.py <name>=<run_dir> [...] --out figures/mobile/scenarios.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import job_results  # noqa: E402
import job_time  # noqa: E402

TOP_MM = 600
PARTS = [("route", ("route", "planning", "step_back", "held")),
         ("docking moves", ("approach", "undock", "align", "backout")), ("docked", ("docked",))]


def mm(v):
    """mm, NaN for nobody near (the measures' 2 m cutoff, or none at all)."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return np.nan
    return v if v < 2000.0 or v not in (2000.0, 3000.0) else np.nan


def main():
    out = sys.argv[sys.argv.index("--out") + 1]
    runs = [a.split("=", 1) for a in sys.argv[1:] if "=" in a]
    names = [n for n, _ in runs]
    times = [job_time.summary(Path(p)) for _, p in runs]
    res = [job_results.parse(Path(p)) for _, p in runs]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(runs))
    bottom = np.zeros(len(runs))
    for label, states in PARTS:
        vals = np.array([sum(t["nav"].get(s, 0.0) for s in states) / max(t["boxes"] or 1, 1) for t in times])
        a1.bar(x, vals, bottom=bottom, label=label)
        bottom += vals
    a1.set_xticks(x, names, rotation=20)
    a1.set_ylabel("s per box")
    a1.set_title("time per box, by what the base was doing")
    a1.legend()
    w = 0.25
    for k, (key, label) in enumerate([("base_moving", "chassis, while it moved"),
                                      ("arm_closing", "arm or box, base closing on them"),
                                      ("arm_overlap", "arm or box, base moving with the arm out")]):
        vals = [mm(r.get(key)) for r in res]
        a2.bar(x + (k - 1) * w, np.minimum(np.nan_to_num(vals), TOP_MM), w, label=label)
        for i, v in enumerate(vals):
            if np.isnan(v):
                a2.text(x[i] + (k - 1) * w, 5, "none near", rotation=90, ha="center", va="bottom", fontsize=7)
            elif v > TOP_MM:
                a2.text(x[i] + (k - 1) * w, TOP_MM - 5, f"{v / 1000:.1f} m", rotation=90, ha="center", va="top",
                        fontsize=7, color="white")
    a2.set_xticks(x, names, rotation=20)
    a2.set_ylim(0, TOP_MM * 1.4)  # room for the legend over the clipped bars
    a2.set_ylabel("closest person (mm)")
    a2.set_title("closest approach (truth, whole bodies for the arm)")
    a2.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
