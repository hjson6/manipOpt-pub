"""Offline SLAM evaluation on a recorded drive (record_drive.py), no ROS: runs our own
graph SLAM (slam/mapper.py) on the scans and the method's odometry, or reads
slam_toolbox's result (toolbox_replay.py), and compares with the truth: trajectory
error (the map frame is anchored at the start, so truth is moved there), the walls'
positions and angles in the map, the map against one built from the true poses, and
CPU per scan. --localize <map>: localization in a saved map instead (slam/localizer.py),
starting from the true start pose plus --init-offset (an operator's rough estimate).
--walls <map>: only the walls of a map saved by a live run (anchored at the dock).
usage: python slam_eval.py <recording> [--toolbox <result.npz>] [--localize <map stem>
       [--init-offset dx dy dyaw_deg]] [--walls <map>] [--plot dir]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tasks/pick_and_place/common")]
from pick_place_common.scene import LIDAR_MOUNTS_BASE, ROOM_SIZE  # noqa: E402
from slam.grid import OccupancyGrid  # noqa: E402
from slam.localizer import GridLocalizer  # noqa: E402
from slam.mapper import GraphSlam  # noqa: E402
from slam.pose_graph import compose, relative, wrap  # noqa: E402
from slam.scan import scan_points, transform  # noqa: E402

REC = REPO / "data" / "recordings"
WALL_BAND_M = 0.25
WALL_END_M = 0.6
SHELF_X = (6.3, 8.7)


def load(name):
    r = np.load(REC / f"{name}.npz" if not str(name).endswith(".npz") else name)
    idx = np.round(r["scan_t"] / 0.02).astype(int) - 1
    return r, r["odom"][idx]


def run_own(r, odom):
    slam = GraphSlam()
    est, ms = [], []
    for scan, o in zip(r["scans"], odom):
        pts, orig = scan_points(scan, r["beam_angles"], LIDAR_MOUNTS_BASE, origins=True)
        t = time.perf_counter()
        est.append(slam.update(o, pts, orig))
        ms.append((time.perf_counter() - t) * 1e3)
    return slam, np.array(est), np.array(ms)


def truth_in_map(r):
    t0 = r["scan_truth"][0]
    return np.array([relative(t0, p) for p in r["scan_truth"]]), t0


def wall_errors(grid, t0):
    """Per wall: (signed offset of the occupied cells' fitted line from the true inner
    face, m; angle error, deg; cells)."""
    pts = np.array([compose(t0, (x, y, 0.0))[:2] for x, y in grid.occupied_points()])
    w, h = ROOM_SIZE
    out = {}
    for name, axis, face in (("west", 0, 0.0), ("east", 0, w), ("south", 1, 0.0), ("north", 1, h)):
        other = 1 - axis
        span = (0.0, h) if axis == 0 else (0.0, w)
        sel = (np.abs(pts[:, axis] - face) < WALL_BAND_M) & (pts[:, other] > span[0] + WALL_END_M) & (
            pts[:, other] < span[1] - WALL_END_M)
        if name == "north":
            sel &= (pts[:, 0] < SHELF_X[0]) | (pts[:, 0] > SHELF_X[1])
        p = pts[sel]
        if len(p) < 20:
            out[name] = (np.nan, np.nan, len(p))
            continue
        a, b = np.polyfit(p[:, other], p[:, axis], 1)
        mid = np.mean(span)
        out[name] = (float(a * mid + b - face), float(np.degrees(np.arctan(a))), len(p))
    return out


def reference_grid(r, res=0.05):
    """The same scans integrated at the true poses: the map a perfect SLAM would make."""
    t0 = r["scan_truth"][0]
    poses = [relative(t0, p) for p in r["scan_truth"]]
    allp = [transform(p, scan_points(s, r["beam_angles"], LIDAR_MOUNTS_BASE)) for p, s in zip(poses[::10], r["scans"][::10])]
    grid = OccupancyGrid.around(np.vstack(allp), res)
    for p, s in zip(poses[::3], r["scans"][::3]):
        pts, orig = scan_points(s, r["beam_angles"], LIDAR_MOUNTS_BASE, origins=True)
        grid.integrate(p, pts, orig)
    return grid


def report(label, est, truth, grid, ref, t0, ms=None, extra=""):
    e = np.hypot(*(est[:, :2] - truth[:, :2]).T)
    ey = np.degrees(np.abs(wrap(est[:, 2] - truth[:, 2])))
    from scipy.spatial import cKDTree
    d, _ = cKDTree(ref.occupied_points()).query(grid.occupied_points())
    walls = wall_errors(grid, t0)
    print(f"{label}: position error p50/p95/max {1e3 * np.median(e):.0f}/{1e3 * np.percentile(e, 95):.0f}/"
          f"{1e3 * e.max():.0f} mm, yaw p50/max {np.median(ey):.2f}/{ey.max():.2f} deg; map vs truth-pose map "
          f"p50/p95 {1e3 * np.median(d):.0f}/{1e3 * np.percentile(d, 95):.0f} mm"
          + (f"; CPU per scan p50/p95/max {np.median(ms):.1f}/{np.percentile(ms, 95):.1f}/{ms.max():.0f} ms" if ms is not None else "")
          + extra)
    print("   walls (offset mm, angle deg): " + ", ".join(
        f"{k} {1e3 * v[0]:+.0f}/{v[1]:+.2f}" for k, v in walls.items()))
    return e, walls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recording")
    ap.add_argument("--toolbox", default="")
    ap.add_argument("--plot", default="")
    ap.add_argument("--localize", default="")
    ap.add_argument("--walls", default="", help="a saved map (live run): its walls, anchored at the dock")
    ap.add_argument("--layout", default="cell", choices=("cell", "stations"), help="with --walls: where the dock is")
    ap.add_argument("--init-offset", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    args = ap.parse_args()
    if args.localize:
        return localize(args)
    if args.walls:
        from pick_place_common.scene import BASE_HOME_POSE, BASE_PARK_POSE
        dock = BASE_PARK_POSE if args.layout == "cell" else BASE_HOME_POSE
        walls = wall_errors(OccupancyGrid.load(REPO / "data" / "maps" / f"{args.walls}.yaml"), np.array(dock))
        print(f"{args.walls} walls (offset mm, angle deg): " + ", ".join(
            f"{k} {1e3 * v[0]:+.0f}/{v[1]:+.2f}" for k, v in walls.items()))
        return None
    r, odom = load(args.recording)
    truth, t0 = truth_in_map(r)
    ref = reference_grid(r)
    odom_map = np.array([relative(odom[0], o) for o in odom])
    eo = np.hypot(*(odom_map[:, :2] - truth[:, :2]).T)
    ref_walls = wall_errors(ref, t0)
    print("reference map (true poses), walls (offset mm, angle deg): " + ", ".join(
        f"{k} {1e3 * v[0]:+.0f}/{v[1]:+.2f}" for k, v in ref_walls.items()))
    print(f"{args.recording}: {len(r['scans'])} scans, {np.sum(np.hypot(*np.diff(truth[:, :2], axis=0).T)):.1f} m; "
          f"odometry alone p50/max {1e3 * np.median(eo):.0f}/{1e3 * eo.max():.0f} mm")
    results = {}
    if args.toolbox:
        tb = np.load(args.toolbox)
        grid = OccupancyGrid.load(tb["map_yaml"].item())
        est = tb["poses"]
        results["toolbox"] = (est, grid, report("slam_toolbox", est, truth, grid, ref, t0, tb["ms"] if "ms" in tb else None))
    else:
        slam, est, ms = run_own(r, odom)
        grid = slam.map()
        kf = slam.keyframe_poses()
        extra = f"; {len(slam.keyframes)} keyframes, {len(slam.loops)} loop closures ({slam.rejected} rejected), {slam.matches_lost} scans on odometry"
        results["own"] = (est, grid, report("own (online)", est, truth, grid, ref, t0, ms, extra))
        out = REPO / "data" / "maps"
        out.mkdir(parents=True, exist_ok=True)
        grid.save(out / f"own_{Path(args.recording).stem}")
        np.savez(out / f"own_{Path(args.recording).stem}_traj.npz", est=est, keyframes=kf, ms=ms, truth_start=t0)
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        out = Path(args.plot)
        out.mkdir(parents=True, exist_ok=True)
        for name, (est, grid, _) in results.items():
            fig, ax = plt.subplots(figsize=(9, 7.5))
            img = np.full(grid.log_odds.shape, 0.5)
            img[grid.free] = 1.0
            img[grid.occupied] = 0.0
            ext = [grid.origin[0], grid.origin[0] + grid.log_odds.shape[1] * grid.res,
                   grid.origin[1], grid.origin[1] + grid.log_odds.shape[0] * grid.res]
            ax.imshow(img, cmap="gray", origin="lower", extent=ext, vmin=0, vmax=1)
            ax.plot(truth[:, 0], truth[:, 1], "b-", lw=1, label="true path")
            ax.plot(est[:, 0], est[:, 1], "r--", lw=1, label=f"{name} estimate")
            ax.plot(odom_map[:, 0], odom_map[:, 1], ":", color="orange", lw=1, label="odometry")
            ax.set_title(f"{name} on {Path(args.recording).stem} (map frame = start pose)")
            ax.legend(loc="upper right", fontsize=8)
            ax.set_aspect("equal")
            fig.tight_layout()
            fig.savefig(out / f"{name}_{Path(args.recording).stem}.png", dpi=100)
            print(f"plot: {out / f'{name}_{Path(args.recording).stem}.png'}")


def localize(args):
    r, odom = load(args.recording)
    maps = REPO / "data" / "maps"
    meta = maps / (f"{args.localize}_result.npz" if args.localize.startswith("toolbox") else f"{args.localize}_traj.npz")
    t0 = np.load(meta)["truth_start"]
    truth = np.array([relative(t0, p) for p in r["scan_truth"]])
    dx, dy, dyaw = args.init_offset
    if args.toolbox:  # slam_toolbox's localization (toolbox_replay.py --mode localization)
        tb = np.load(args.toolbox)
        est, ms, quality, lost = tb["poses"], tb["ms"], [np.nan], "n/a"
    else:
        grid = OccupancyGrid.load(maps / f"{args.localize}.yaml")
        loc = GridLocalizer(grid, truth[0] + (dx, dy, np.radians(dyaw)))
        est, ms, quality = [], [], []
        for scan, o in zip(r["scans"], odom):
            pts = scan_points(scan, r["beam_angles"], LIDAR_MOUNTS_BASE)
            t = time.perf_counter()
            est.append(loc.update(o, pts))
            ms.append((time.perf_counter() - t) * 1e3)
            quality.append(loc.quality)
        est, ms, lost = np.array(est), np.array(ms), loc.lost
    e = np.hypot(*(est[:, :2] - truth[:, :2]).T)
    ey = np.degrees(np.abs(wrap(est[:, 2] - truth[:, 2])))
    settled = slice(15, None)  # after the first second
    print(f"localize {Path(args.recording).stem} in {args.localize} from offset {args.init_offset}: position error "
          f"p50/p95/max {1e3 * np.median(e[settled]):.0f}/{1e3 * np.percentile(e[settled], 95):.0f}/"
          f"{1e3 * e[settled].max():.0f} mm, yaw p50/max {np.median(ey[settled]):.2f}/{ey[settled].max():.2f} deg "
          f"(first scan {1e3 * e[0]:.0f} mm); match quality p5 {np.percentile(quality, 5):.2f}, {lost} scans on "
          f"odometry; CPU p50/p95/max {np.median(ms):.1f}/{np.percentile(ms, 95):.1f}/{ms.max():.0f} ms")
    return est, truth


if __name__ == "__main__":
    main()
