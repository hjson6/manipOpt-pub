"""Offline check of the lidar people detection (plant ray cast + perception):
  grid   the person on a grid out to 5 m from the base, random headings: detected
         wherever at least 2 beams of a scanner hit them (not occluded), centre error
  paths  the visit and walk paths, every 5 cm
  empty  100 empty-cell scans per scanner: false positives
usage: python lidar_check.py [--draw]   (--draw: figures/lidar/lidar_check.png)
"""
import sys
from pathlib import Path

import mujoco
import numpy as np

from perception.lidar_detection import MIN_POINTS, LidarPeopleDetector
from pick_place_common import lidar_sim
from pick_place_common.mujoco_sim_node import load_scene_model
from pick_place_common.scene import (
    CELL_POSE, CHASSIS_BOUNDS, LIDAR_SCANNERS, PICK_TABLE_BOUNDS, PLACE_TABLE_BOUNDS, ROOM_FIXTURES, ROOM_FLOOR_Z,
    ROOM_SIZE, compose)
from pick_place_mpc import dynamic_obstacle_node as actor

REPO = Path(__file__).resolve().parents[2]
m = load_scene_model(world="arm")
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
lidar_sim.check_sites(m, d)
person = m.body("person_obstacle")
pid = m.body_mocapid[person.id]
person_geoms = set(range(person.geomadr[0], person.geomadr[0] + person.geomnum[0]))
rng = np.random.default_rng(0)


def place(xy, heading=0.0):
    d.mocap_pos[pid] = (*xy, ROOM_FLOOR_Z) if xy is not None else (0.9, 0.9, -3.0)
    d.mocap_quat[pid] = (np.cos(heading / 2), 0.0, 0.0, np.sin(heading / 2))
    mujoco.mj_forward(m, d)


def frame(det):
    """One noisy scan per scanner; (people, visible)."""
    visible = False
    for i in range(len(LIDAR_SCANNERS)):
        r, g = lidar_sim.scan(m, d, i, rng, hit_geoms=True)
        visible |= sum(int(k) in person_geoms for k in g) >= MIN_POINTS
        det.update(i, r)
    return det.people(), visible


def detector():
    det = LidarPeopleDetector(LIDAR_SCANNERS, lidar_sim.BEAM_ANGLES)
    place(None)
    while not det.ready:
        for i in range(len(LIDAR_SCANNERS)):
            det.update(i, lidar_sim.scan(m, d, i, rng))
    return det


def blocked(xy):
    """Furniture, or not on the room's free floor (walls, pillar, shelf)."""
    if any(b[0][0] - 0.35 < xy[0] < b[0][1] + 0.35 and b[1][0] - 0.35 < xy[1] < b[1][1] + 0.35
           for b in (CHASSIS_BOUNDS, PICK_TABLE_BOUNDS, PLACE_TABLE_BOUNDS)):
        return True
    x, y, _ = compose(CELL_POSE, (xy[0], xy[1], 0.0))
    return not (0.35 < x < ROOM_SIZE[0] - 0.35 and 0.35 < y < ROOM_SIZE[1] - 0.35) or any(
        x0 - 0.35 < x < x1 + 0.35 and y0 - 0.35 < y < y1 + 0.35 for (x0, x1), (y0, y1) in ROOM_FIXTURES)


def run(det, positions, label):
    errs, missed, hidden, extra, rows = [], [], 0, 0, []
    for xy in positions:
        place(xy, rng.uniform(-np.pi, np.pi))
        people, visible = frame(det)
        near = [p for p in people if np.hypot(p[0] - xy[0], p[1] - xy[1]) < 0.5]
        extra += len(people) - len(near)
        if not visible:
            hidden += 1
            rows.append((*xy, "hidden"))
            continue
        if near:
            errs.append(min(np.hypot(p[0] - xy[0], p[1] - xy[1]) for p in near))
            rows.append((*xy, "seen"))
        else:
            missed.append(xy)
            rows.append((*xy, "missed"))
    seen = len(positions) - hidden
    e = np.array(errs) * 1e3
    print(f"{label}: {len(positions)} poses, occluded {hidden}, detected {len(errs)}/{seen}"
          + (f", centre error median {np.median(e):.0f} mm p95 {np.percentile(e, 95):.0f} max {e.max():.0f}" if len(e) else "")
          + f", other blobs {extra}")
    for xy in missed[:10]:
        print(f"  missed at ({xy[0]:+.2f}, {xy[1]:+.2f})")
    return len(missed) == 0 and extra == 0 and (not len(e) or np.median(e) < 50), rows


def main():
    det = detector()
    place(None)
    fp = 0
    for _ in range(100):
        fp += len(frame(det)[0])
    print(f"empty: {fp} false positives in 100 scans per scanner")
    ok = fp == 0
    g = np.arange(-5.0, 5.01, 0.25)
    grid = [(x, y) for x in g for y in g if np.hypot(x, y) <= 5.0 and not blocked((x, y))]
    good, rows = run(det, grid, "grid")
    ok &= good
    entry, stand = np.array(actor.VISIT_ENTRY), np.array(actor.VISIT_STAND)
    n = int(np.linalg.norm(stand - entry) / 0.05)
    visit = [tuple(entry + (stand - entry) * u) for u in np.linspace(0.0, 1.0, n)]
    walk = [(x, actor.WALK_Y) for x in np.arange(actor.WALK_START[0], actor.WALK_END[0], 0.05)]
    good, rows_v = run(det, visit, "visit path")
    ok &= good
    good, rows_w = run(det, walk, "walk path")
    ok &= good
    if "--draw" in sys.argv:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 7))
        colors = {"seen": "#3a9d5d", "hidden": "#bbbbbb", "missed": "#d62728"}
        for x, y, k in rows + rows_v + rows_w:
            ax.plot(x, y, ".", color=colors[k], ms=5)
        for (x0, x1), (y0, y1) in (CHASSIS_BOUNDS, PICK_TABLE_BOUNDS, PLACE_TABLE_BOUNDS):
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False))
        for x, y, _z, _yaw in LIDAR_SCANNERS:
            ax.plot(x, y, "s", color="#e6b800", ms=8)
        ax.set_aspect("equal")
        ax.set_title("Lidar people detection offline: green seen, grey occluded, red missed")
        out = REPO / "figures/lidar/lidar_check.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out, dpi=100)
        print("saved", out)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
