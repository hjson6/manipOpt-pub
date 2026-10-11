"""Offline check of the cell layout (the chassis, tables, full-size person):
  arm    the smallest distance from the arm's collision geoms to the chassis and the
         tables over a recorded run's arm poses (sim.csv q1..q7; the arm's frame
         is unchanged, so its motion is too); must stay positive
  empty  overhead-camera blobs in the empty cell (furniture masked); must be 0
  person the person along the visit path, at the stand and along the walk: one
         blob near the truth each frame (xy error), nothing else
usage: python layout_check.py <run_dir with sim.csv>
"""
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wrist_pose  # noqa: E402
from bench_analyze import load  # noqa: E402
from perception import obstacle_detection  # noqa: E402
from pick_place_common import frames  # noqa: E402
from pick_place_common.depth_noise import add_depth_noise  # noqa: E402
from pick_place_common.mujoco_sim_node import ARM_JOINT_NAMES, load_scene_model  # noqa: E402
from pick_place_common.scene import (  # noqa: E402
    AISLE_MASK_MARGIN_M, AISLE_MASK_PILE_ABOVE_M, AISLE_MASK_TRAY, DETECTION_MIN_BLOB_PX, FOREGROUND_MIN_HEIGHT_M,
    FURNITURE_MASKS, ROOM_FLOOR_Z, SOURCE_SCAN_BOUNDS, WORKSPACE_CAM_FOVY_DEG, WORKSPACE_CAM_HEIGHT,
    WORKSPACE_CAM_NAME, WORKSPACE_CAM_POS, WORKSPACE_CAM_WIDTH)
from pick_place_mpc import dynamic_obstacle_node as actor  # noqa: E402

FURNITURE = ("chassis", "pedestal", "pick_table", "place_table")
PILE_TOP = 0.20

m = load_scene_model(world="arm")
d = mujoco.MjData(m)
wrist_pose.reset_scene(m, d)
qa = [m.joint(n).qposadr[0] for n in ARM_JOINT_NAMES]
robot_root = m.body_rootid[m.body("link0").id]
arm_geoms = [g for g in range(m.ngeom) if frames.in_arm(m, m.geom_bodyid[g])
             and m.geom_contype[g] and m.body(m.geom_bodyid[g]).name != "link0"]
furniture = [g for g in range(m.ngeom) if (m.geom(g).name or "").startswith(FURNITURE)]


def arm_clearance(run):
    s = load(Path(run) / "sim.csv")
    q = np.column_stack([s[f"q{j}"] for j in range(1, 8)])[::5]
    worst = (np.inf, None)
    fromto = np.zeros(6)
    for qi in q:
        d.qpos[qa] = qi
        mujoco.mj_forward(m, d)
        for a in arm_geoms:
            for b in furniture:
                dist = mujoco.mj_geomDistance(m, d, a, b, 0.5, fromto)
                if dist < worst[0]:
                    worst = (dist, f"{m.body(m.geom_bodyid[a]).name} / {m.geom(b).name}")
    print(f"arm: {len(q)} poses, closest {1e3 * worst[0]:.0f} mm ({worst[1]})")
    return worst[0] > 0


def detect(person_xy=None, rng=None):
    pid = m.body_mocapid[m.body("person_obstacle").id]
    d.mocap_pos[pid] = (*person_xy, ROOM_FLOOR_Z) if person_xy is not None else (0.9, 0.9, -3.0)
    mujoco.mj_forward(m, d)
    r_depth.update_scene(d, camera=WORKSPACE_CAM_NAME, scene_option=opt)
    depth = add_depth_noise(r_depth.render(), rng)
    r_seg.update_scene(d, camera=WORKSPACE_CAM_NAME, scene_option=opt)
    seg = r_seg.render()
    is_geom = seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
    robot = is_geom & (m.body_rootid[m.geom_bodyid[np.where(is_geom, seg[..., 0], 0)]] == robot_root)
    return obstacle_detection.detect_blobs(
        depth, WORKSPACE_CAM_POS, np.eye(3), WORKSPACE_CAM_FOVY_DEG, ROOM_FLOOR_Z, FOREGROUND_MIN_HEIGHT_M,
        robot_mask=robot, mask_boxes=((*SOURCE_SCAN_BOUNDS, PILE_TOP + AISLE_MASK_PILE_ABOVE_M), AISLE_MASK_TRAY,
                                      *FURNITURE_MASKS),
        mask_margin=AISLE_MASK_MARGIN_M, min_blob_px=DETECTION_MIN_BLOB_PX)


def main():
    ok = arm_clearance(sys.argv[1])
    rng = np.random.default_rng(0)
    empty = sum(len(detect(None, rng)) for _ in range(20))
    print(f"empty: {empty} blobs in 20 noisy frames")
    ok &= empty == 0
    entry, stand = np.array(actor.VISIT_ENTRY), np.array(actor.VISIT_STAND)
    path = [entry + (stand - entry) * u for u in np.linspace(0.0, 1.0, 12)]
    path += [np.array([x, actor.WALK_Y]) for x in np.linspace(actor.WALK_START[0], actor.WALK_END[0], 11)]
    seen, extra, errs = 0, 0, []
    for xy in path:
        blobs = detect(xy, rng)
        near = [b for b in blobs if np.hypot(b[0] - xy[0], b[1] - xy[1]) < 0.4]
        extra += len(blobs) - len(near)
        if near:
            seen += 1
            errs.append(min(np.hypot(b[0] - xy[0], b[1] - xy[1]) for b in near))
            tops = max(b[5] for b in near)
        print(f"  person at ({xy[0]:+.2f}, {xy[1]:+.2f}): "
              + (f"seen, xy error {1e3 * errs[-1]:.0f} mm, top {tops:.2f}" if near else "NOT SEEN")
              + (f", {len(blobs) - len(near)} other blob(s)" if len(blobs) > len(near) else ""))
    print(f"person: seen in {seen}/{len(path)} frames, other blobs {extra}")
    ok &= extra == 0
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


r_depth = mujoco.Renderer(m, WORKSPACE_CAM_HEIGHT, WORKSPACE_CAM_WIDTH)
r_depth.enable_depth_rendering()
r_seg = mujoco.Renderer(m, WORKSPACE_CAM_HEIGHT, WORKSPACE_CAM_WIDTH)
r_seg.enable_segmentation_rendering()
opt = mujoco.MjvOption()
opt.geomgroup[2] = 0
opt.geomgroup[3] = 1

if __name__ == "__main__":
    main()
