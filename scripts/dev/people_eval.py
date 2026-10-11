"""Offline people detection on a moving base, no ROS: a crowd recording (record_drive.py
--crowd) through the method's whole chain: localization in a saved map
(slam/localizer.py, from the dock), people against the map (perception/map_people.py)
and tracking (perception/people_tracker.py). Scored against the truth: the share of
people within 5 m that the lidars see (at least VISIBLE_BEAMS beams on them) with a
confirmed track within MATCH_M, position and velocity error, and false people (with
their distance to the nearest mapped surface: walls, tables, fixtures).
usage: python people_eval.py <recording> [--map own_st_map_quiet] [--plot dir]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "scripts/dev"), str(REPO / "tasks/pick_and_place/common")]
import slam_eval  # noqa: E402
from perception.map_people import MapPeopleDetector  # noqa: E402
from perception.people_tracker import PeopleTracker  # noqa: E402
from pick_place_common.scene import LIDAR_MOUNTS_BASE  # noqa: E402
from slam.grid import OccupancyGrid  # noqa: E402
from slam.localizer import GridLocalizer  # noqa: E402
from slam.pose_graph import relative  # noqa: E402
from slam.scan import scan_points  # noqa: E402

RANGE_M = 5.0
VISIBLE_BEAMS = 3
MATCH_M = 0.5
MOVING_MS = 0.3


def run(name, map_name):
    r, odom = slam_eval.load(name)
    maps = REPO / "data" / "maps"
    grid = OccupancyGrid.load(maps / f"{map_name}.yaml")
    t0 = np.load(maps / f"{map_name}_traj.npz")["truth_start"]
    truth = np.array([relative(t0, p) for p in r["scan_truth"]])
    loc = GridLocalizer(grid, truth[0])
    det = MapPeopleDetector(grid.occupied_points(), LIDAR_MOUNTS_BASE, r["beam_angles"])
    trk = PeopleTracker()
    c, s = np.cos(-t0[2]), np.sin(-t0[2])
    rows = []
    for k, (scan, o, t) in enumerate(zip(r["scans"], odom, r["scan_t"])):
        pose = loc.update(o, scan_points(scan, r["beam_angles"], LIDAR_MOUNTS_BASE))
        tracks = trk.update(t, det.update(pose, scan), det.seen_empty)
        crowd = r["crowd"][k]
        p = np.array([relative(t0, (x, y, 0.0))[:2] for x, y, _vx, _vy in crowd]).reshape(-1, 2)
        v = np.column_stack([c * crowd[:, 2] - s * crowd[:, 3], s * crowd[:, 2] + c * crowd[:, 3]]) if len(crowd) else p
        rows.append(dict(t=t, robot=truth[k], est=pose, people=p, vel=v, beams=r["crowd_beams"][k],
                         tracks=[(tr.id, *tr.x, tr.standing) for tr in tracks], fg=det.foreground))
    return rows, grid


def score(rows, grid):
    from scipy.spatial import cKDTree
    surfaces = cKDTree(grid.occupied_points())
    near, seen, pos_err, vel_err, fp, fp_where = 0, 0, [], [], 0, []
    per_person, run, longest = {}, {}, {}
    for row in rows:
        tr = np.array([t[1:5] for t in row["tracks"]]).reshape(-1, 4)
        used = set()
        for i, (pxy, vxy, beams) in enumerate(zip(row["people"], row["vel"], row["beams"])):
            d_robot = np.hypot(*(pxy - row["robot"][:2]))
            hit = None
            if len(tr):
                d = np.hypot(*(tr[:, :2] - pxy).T)
                j = int(np.argmin(d))
                if d[j] < MATCH_M:
                    hit = j
                    used.add(j)
            if d_robot <= RANGE_M and beams >= VISIBLE_BEAMS:
                near += 1
                per_person.setdefault(i, [0, 0])[0] += 1
                run[i] = 0 if hit is not None else run.get(i, 0) + 1
                longest[i] = max(longest.get(i, 0), run[i])
                if hit is not None:
                    seen += 1
                    per_person[i][1] += 1
                    pos_err.append(np.hypot(*(tr[hit, :2] - pxy)))
                    if np.hypot(*vxy) > MOVING_MS:
                        vel_err.append(np.hypot(*(tr[hit, 2:4] - vxy)))
        for j in range(len(tr)):
            if j not in used:
                fp += 1
                fp_where.append((row["t"], *tr[j, :2], surfaces.query(tr[j, :2])[0]))
    return dict(near=near, seen=seen, pos_err=np.array(pos_err), vel_err=np.array(vel_err), fp=fp,
                fp_where=np.array(fp_where).reshape(-1, 4), per_person=per_person, scans=len(rows), longest=longest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recording")
    ap.add_argument("--map", default="own_st_map_quiet")
    ap.add_argument("--plot", default="")
    args = ap.parse_args()
    rows, grid = run(args.recording, args.map)
    s = score(rows, grid)
    pe, ve = 1e3 * s["pos_err"], s["vel_err"]
    print(f"{args.recording}: {s['scans']} scans; people within {RANGE_M:.0f} m and in view: detected "
          f"{s['seen']}/{s['near']} ({100 * s['seen'] / max(s['near'], 1):.1f}%); position error p50/p95 "
          f"{np.median(pe):.0f}/{np.percentile(pe, 95):.0f} mm; speed error (moving) p50/p95 "
          f"{np.median(ve):.2f}/{np.percentile(ve, 95):.2f} m/s; false people {s['fp']} in "
          f"{len(set(s['fp_where'][:, 0]))} scans")
    print("   per person (tracked/in view, longest miss in scans): " + ", ".join(
        f"{i + 1}: {b}/{a} ({s['longest'].get(i, 0)})" for i, (a, b) in sorted(s["per_person"].items())))
    if len(s["fp_where"]):
        print("   false people: nearest mapped surface p50/min "
              f"{1e3 * np.median(s['fp_where'][:, 3]):.0f}/{1e3 * s['fp_where'][:, 3].min():.0f} mm; e.g. "
              + ", ".join(f"({x:+.2f}, {y:+.2f}) at {t:.1f} s" for t, x, y, _ in s["fp_where"][:4]))
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        out = Path(args.plot)
        out.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        img = np.full(grid.log_odds.shape, 0.5)
        img[grid.free] = 1.0
        img[grid.occupied] = 0.0
        ext = [grid.origin[0], grid.origin[0] + grid.log_odds.shape[1] * grid.res,
               grid.origin[1], grid.origin[1] + grid.log_odds.shape[0] * grid.res]
        snap = rows[len(rows) // 2]
        ax = axes[0]
        ax.imshow(img, cmap="gray", origin="lower", extent=ext, vmin=0, vmax=1)
        ax.plot(*snap["fg"].T, ".", color="orange", ms=2, label="foreground")
        ax.plot(*snap["people"].T, "bo", mfc="none", ms=12, label="true people")
        for tid, x, y, vx, vy, standing in snap["tracks"]:
            ax.plot(x, y, "r+", ms=10)
            ax.arrow(x, y, vx, vy, color="r", head_width=0.08)
        ax.plot(*snap["robot"][:2], "gs", ms=8, label="robot")
        ax.set_title(f"{args.recording} at {snap['t']:.1f} s: red tracks (velocity arrows)", fontsize=10)
        ax.legend(fontsize=7, loc="upper right")
        ax.set_aspect("equal")
        ax = axes[1]
        for i in range(len(rows[0]["people"])):
            t = [row["t"] for row in rows]
            d = [np.hypot(*(row["people"][i] - row["robot"][:2])) for row in rows]
            ok = [row["beams"][i] >= VISIBLE_BEAMS for row in rows]
            det_ = [any(np.hypot(x - row["people"][i][0], y - row["people"][i][1]) < MATCH_M
                        for _, x, y, *_r in row["tracks"]) for row in rows]
            ax.plot(t, d, lw=0.6, color=f"C{i}", label=f"person {i + 1}")
            ax.plot(np.array(t)[np.array(ok) & ~np.array(det_) & (np.array(d) <= RANGE_M)],
                    np.array(d)[np.array(ok) & ~np.array(det_) & (np.array(d) <= RANGE_M)], "x", color=f"C{i}")
        ax.axhline(RANGE_M, color="k", lw=0.5)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("distance from the robot (m); x: in view but not tracked")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / f"people_{args.recording}.png", dpi=95)
        print(f"plot: {out / f'people_{args.recording}.png'}")


if __name__ == "__main__":
    main()
