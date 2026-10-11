"""Offline reach check for docking (mobile-manipulator plan, before step 4): with the
base docked at a table, can the arm reach every point of the zone, tool down, at
every working height? IK with the joint limits (wrist_pose.place_tcp) over a grid
of each zone (pick 0.50 x 0.50 m, place 0.61 x 0.46 m), TCP heights from a box top
to the scan pose, heading as task_node's aligned rule.
Docking: "front", the table ahead of the chassis (the arm's -y; the zone's long
side along the table edge), or "side", the table on the chassis's left (the arm's
+x). gap = chassis (or arm side) to the table edge; the zone starts EDGE_M in.
The old cell's zones (step 1, base parked) are the reference.
usage: python dock_reach_check.py
"""
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wrist_pose  # noqa: E402
from pick_place_common.mujoco_sim_node import load_scene_model  # noqa: E402
from pick_place_common.scene import (  # noqa: E402
    BASE_ARM_MOUNT, BASE_CHASSIS_HALF, PICK_ZONE_SIZE, PLACE_ZONE_BOUNDS, PLACE_ZONE_SIZE, SOURCE_SCAN_BOUNDS)
from pick_place_mpc.task_node import TaskNode  # noqa: E402

HEIGHTS = (0.07, 0.15, 0.25, 0.35, 0.50)  # box tops to the scan pose, above the table
STEP_M = 0.05
EDGE_M = 0.03
FRONT_TO_ARM = BASE_CHASSIS_HALF[0] - BASE_ARM_MOUNT[0]  # 0.20 m
SIDE_TO_ARM = BASE_CHASSIS_HALF[1]  # 0.28 m


def zone(kind, size, gap, lateral=0.0):
    """Zone bounds in the arm frame for a docking."""
    w, dpt = max(size), min(size)  # the long side along the table edge
    if kind == "front":
        near = FRONT_TO_ARM + gap + EDGE_M
        return (lateral - w / 2, lateral + w / 2), (-near - dpt, -near)
    near = SIDE_TO_ARM + gap + EDGE_M
    return (near, near + dpt), (lateral - w / 2, lateral + w / 2)


def reach(m, d, bounds):
    (x0, x1), (y0, y1) = bounds
    pts = [(x, y, z) for x in np.arange(x0, x1 + 1e-9, STEP_M) for y in np.arange(y0, y1 + 1e-9, STEP_M)
           for z in HEIGHTS]
    ok = []
    for p in pts:
        wrist_pose.reset_scene(m, d)
        e_pos, e_rot = wrist_pose.place_tcp(m, d, p, TaskNode._aligned_yaw(float(np.arctan2(p[1], p[0]))))
        ok.append(e_pos < 1e-3 and e_rot < 1e-2)
    ok = np.array(ok)
    worst = [p for p, k in zip(pts, ok) if not k]
    return ok.mean(), len(pts), worst


def main():
    m = load_scene_model(world="arm")
    d = mujoco.MjData(m)
    rows = [("old cell, pick zone", SOURCE_SCAN_BOUNDS), ("old cell, place zone", PLACE_ZONE_BOUNDS)]
    for kind in ("front", "side"):
        for gap in (0.05, 0.10):
            for name, size in (("pick", PICK_ZONE_SIZE), ("place", PLACE_ZONE_SIZE)):
                rows.append((f"{kind} dock, gap {100 * gap:.0f} cm, {name} zone", zone(kind, size, gap)))
    for label, b in rows:
        frac, n, worst = reach(m, d, b)
        far = max(worst, key=lambda p: np.hypot(p[0], p[1]), default=None)
        print(f"{label:34s} x {b[0][0]:+.2f}..{b[0][1]:+.2f} y {b[1][0]:+.2f}..{b[1][1]:+.2f}: "
              f"{100 * frac:5.1f}% of {n} reached"
              + (f"; missed e.g. ({far[0]:+.2f}, {far[1]:+.2f}, {far[2]:.2f})" if far else ""))


if __name__ == "__main__":
    main()
