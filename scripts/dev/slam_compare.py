"""Comparison figure of the two SLAM options on the recorded drives (after slam_eval.py
and toolbox_replay.py have run): both maps of one drive with the true path, and the
position error over time when mapping and when localizing in a saved map.
usage: python slam_compare.py [--map-rec map_people] [--loc-rec cross_people] [--out figures/slam/compare.png]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "scripts/dev"), str(REPO / "tasks/pick_and_place/common")]
import slam_eval  # noqa: E402
from slam.grid import OccupancyGrid  # noqa: E402
from slam.pose_graph import relative  # noqa: E402

MAPS = REPO / "data" / "maps"


def show(ax, grid, title, truth, est):
    img = np.full(grid.log_odds.shape, 0.5)
    img[grid.free] = 1.0
    img[grid.occupied] = 0.0
    ext = [grid.origin[0], grid.origin[0] + grid.log_odds.shape[1] * grid.res,
           grid.origin[1], grid.origin[1] + grid.log_odds.shape[0] * grid.res]
    ax.imshow(img, cmap="gray", origin="lower", extent=ext, vmin=0, vmax=1)
    ax.plot(truth[:, 0], truth[:, 1], "b-", lw=1, label="true path")
    ax.plot(est[:, 0], est[:, 1], "r--", lw=1, label="estimate")
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map-rec", default="map_people")
    ap.add_argument("--loc-rec", default="cross_people")
    ap.add_argument("--map", default="map_quiet")
    ap.add_argument("--out", default=str(REPO / "figures/slam/compare.png"))
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r, _ = slam_eval.load(args.map_rec)
    truth, _ = slam_eval.truth_in_map(r)
    own = np.load(MAPS / f"own_{args.map_rec}_traj.npz")["est"]
    tb = np.load(MAPS / f"toolbox_{args.map_rec}_result.npz")["poses"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    show(axes[0, 0], OccupancyGrid.load(MAPS / f"own_{args.map_rec}.yaml"), f"own graph SLAM, {args.map_rec}", truth, own)
    show(axes[0, 1], OccupancyGrid.load(MAPS / f"toolbox_{args.map_rec}.yaml"), f"slam_toolbox, {args.map_rec}", truth, tb)
    t = r["scan_t"]
    ax = axes[1, 0]
    for est, label, c in ((own, "own", "C0"), (tb, "slam_toolbox", "C3")):
        ax.plot(t, 1e3 * np.hypot(*(est[:, :2] - truth[:, :2]).T), c, lw=0.8, label=label)
    ax.set_title(f"mapping {args.map_rec}: position error", fontsize=10)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("mm")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax = axes[1, 1]
    rl, odom = slam_eval.load(args.loc_rec)
    for kind, c in (("own", "C0"), ("toolbox", "C3")):
        t0 = np.load(MAPS / (f"own_{args.map}_traj.npz" if kind == "own" else f"toolbox_{args.map}_result.npz"))["truth_start"]
        tl = np.array([relative(t0, p) for p in rl["scan_truth"]])
        if kind == "own":
            est, _ = slam_eval.localize(argparse.Namespace(recording=args.loc_rec, localize=f"own_{args.map}",
                                                           init_offset=(0.05, -0.05, 2.0), toolbox=""))
        else:
            est = np.load(MAPS / f"toolbox_loc_{args.loc_rec}_in_toolbox_{args.map}_result.npz")["poses"]
        ax.plot(rl["scan_t"], 1e3 * np.hypot(*(est[:, :2] - tl[:, :2]).T), c, lw=0.8,
                label="own" if kind == "own" else "slam_toolbox")
    ax.set_title(f"localizing {args.loc_rec} in the map of {args.map} (start 7 cm / 2 deg off)", fontsize=10)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("mm")
    ax.set_ylim(0, 80)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=95)
    print(f"plot: {args.out}")


if __name__ == "__main__":
    main()
