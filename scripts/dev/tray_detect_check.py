"""Offline check of perception/tray_detection.py: the tray shifted at random
within +-5 cm (as tray_seed does), the first tray scan's noisy wrist-camera
frames (one per half of the place zone, the arm there by IK, the tool's pixels
masked), inner wall faces and wall top vs the truth.
usage: python tray_detect_check.py [n_offsets, default 50]
"""
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wrist_pose  # noqa: E402
from perception import tray_detection as td  # noqa: E402
from pick_place_common.depth_noise import add_depth_noise  # noqa: E402
from pick_place_common.mujoco_sim_node import load_scene_model, shift_tray  # noqa: E402
from pick_place_common.scene import (  # noqa: E402
    CONTAINER_CAM_HEIGHT, CONTAINER_CAM_NAME, CONTAINER_CAM_WIDTH,
    DEST_TRAY_X_MAX, DEST_TRAY_X_MIN, DEST_TRAY_Y_MAX, DEST_TRAY_Y_MIN, DEST_TRAY_Z_MAX, FLOOR_Z,
    PLACE_ZONE_BOUNDS, PLACE_ZONE_MAX_HEIGHT_M, SCAN_RESOLUTION)
from pick_place_mpc import task_node as tn  # noqa: E402

n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
rng = np.random.default_rng(0)
poses = [tn._scan_tcp(part, PLACE_ZONE_MAX_HEIGHT_M) for part in tn._halves(PLACE_ZONE_BOUNDS)]
errs, tops, misses = [], [], 0
for k in range(n):
    m = load_scene_model(world="arm")
    d = mujoco.MjData(m)
    off = shift_tray(m, np.random.default_rng(1000 + k))
    wrist_pose.reset_scene(m, d)
    r = mujoco.Renderer(m, CONTAINER_CAM_HEIGHT, CONTAINER_CAM_WIDTH)
    r.enable_depth_rendering()
    cid = m.camera(CONTAINER_CAM_NAME).id
    frames = []
    for tcp in poses:
        wrist_pose.place_tcp(m, d, tcp, tn.TaskNode._aligned_yaw(float(np.arctan2(tcp[1], tcp[0]))))
        r.update_scene(d, camera=CONTAINER_CAM_NAME)
        depth = add_depth_noise(r.render(), rng)
        depth[tn._tool_mask()] = 0.0
        frames.append((depth, d.cam_xpos[cid].copy(), d.cam_xmat[cid].reshape(3, 3).copy(), m.cam_fovy[cid],
                       CONTAINER_CAM_WIDTH, CONTAINER_CAM_HEIGHT))
    tray = td.detect_tray(frames, *PLACE_ZONE_BOUNDS, SCAN_RESOLUTION, FLOOR_Z)
    if tray is None:
        misses += 1
        continue
    tray = td.refine_walls(tray, frames)
    truth = np.array([DEST_TRAY_X_MIN + off[0], DEST_TRAY_X_MAX + off[0], DEST_TRAY_Y_MIN + off[1], DEST_TRAY_Y_MAX + off[1]])
    errs.append(np.array([tray.x_min, tray.x_max, tray.y_min, tray.y_max]) - truth)
    tops.append(tray.wall_top - DEST_TRAY_Z_MAX)
    r.close()
e = 1e3 * np.array(errs)
print(f"{n} offsets, {misses} not found")
for i, name in enumerate(("x_min", "x_max", "y_min", "y_max")):
    print(f"  {name}: mean {e[:, i].mean():+.2f} mm, max |err| {np.abs(e[:, i]).max():.2f} mm")
print(f"  all walls: mean |err| {np.abs(e).mean():.2f} mm, max {np.abs(e).max():.2f} mm; "
      f"wall top: mean {1e3 * np.mean(tops):+.2f} mm, max |err| {1e3 * np.abs(tops).max():.2f} mm")
