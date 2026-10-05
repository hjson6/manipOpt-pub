"""A live run's localization from its telemetry (localization_monitor_node's estimates.csv,
slam_node's slam.csv): both estimators against the truth, the filter's scan statuses and
the scans' lag. usage: python loc_live.py <run dir> [--json file] [--plot file.png]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tasks/pick_and_place/common"))
from pick_place_common.loc_scoring import LocScore, line  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--json", default="")
    ap.add_argument("--plot", default="")
    args = ap.parse_args()
    run = Path(args.run)
    rows = list(csv.DictReader(open(run / "estimates.csv")))
    scores = {"icp": LocScore(), "ekf": LocScore()}
    rows = sorted(rows, key=lambda r: int(r["step"]))
    for r in rows:
        f = {k: float(v) for k, v in r.items()}
        true = (f["true_x"], f["true_y"], f["true_yaw"])
        c = [f[k] for k in ("ekf_sxx", "ekf_sxy", "ekf_sxyaw", "ekf_syy", "ekf_syyaw", "ekf_syawyaw")]
        cov = np.array([[c[0], c[1], c[2]], [c[1], c[3], c[4]], [c[2], c[4], c[5]]])
        scores["icp"].add((f["icp_x"], f["icp_y"], f["icp_yaw"]), true)
        scores["ekf"].add((f["ekf_x"], f["ekf_y"], f["ekf_yaw"]), true, cov)
    out = {name: sc.summary() for name, sc in scores.items()}
    for name in out:
        print(f"{name}: {line(out[name])}")
    slam = list(csv.DictReader(open(run / "slam.csv")))
    status = [r["ekf_status"] for r in slam]
    lag = np.array([float(r["lag_ms"]) for r in slam])
    out["scans"] = {s: status.count(s) for s in sorted(set(status))}
    out["lag_ms"] = {"p50": float(np.median(lag)), "p95": float(np.percentile(lag, 95)), "max": float(lag.max())}
    print(f"scans {out['scans']}; lag p50/p95/max {out['lag_ms']['p50']:.0f}/{out['lag_ms']['p95']:.0f}/"
          f"{out['lag_ms']['max']:.0f} ms")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(out, indent=1))
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        a = np.array([[float(r[k]) for k in ("step", "true_x", "true_y", "icp_x", "icp_y", "ekf_x", "ekf_y",
                                             "ekf_sxx", "ekf_syy")] for r in rows])
        t = (a[:, 0] - a[0, 0]) * 0.02
        e_icp = 1e3 * np.hypot(a[:, 3] - a[:, 1], a[:, 4] - a[:, 2])
        e_ekf = 1e3 * np.hypot(a[:, 5] - a[:, 1], a[:, 6] - a[:, 2])
        band = 2e3 * np.sqrt(a[:, 7] + a[:, 8])
        fig, axes = plt.subplots(2, 1, figsize=(12, 7))
        mid = t[-1] / 2
        for ax, sel in ((axes[0], t > 5.0), (axes[1], (t > mid) & (t < mid + 30.0))):
            ax.plot(t[sel], e_icp[sel], "C3", lw=0.5, label="scan matching (icp)")
            ax.plot(t[sel], e_ekf[sel], "C0", lw=0.8, label="filter (ekf)")
            ax.plot(t[sel], band[sel], "C0:", lw=0.8, label="filter's 2 sigma")
            ax.set_ylim(0.0, 1.2 * max(np.percentile(e_icp[sel], 99.9), np.percentile(band[sel], 99.9)))
            ax.set_ylabel("position error (mm)")
            ax.grid(alpha=0.3)
            ax.legend(loc="upper right", fontsize=8)
        axes[1].set_xlabel("time (s)")
        axes[0].set_title(f"live: {run.name}, both estimators against the truth (below: 30 s of it)")
        fig.tight_layout()
        Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot, dpi=95)
        print(f"plot: {args.plot}")


if __name__ == "__main__":
    main()
