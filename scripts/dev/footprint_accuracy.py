"""Offline: how well the pile scan measures each box's top footprint, box by box in
pick order (each measured box is then removed, as if picked).
Estimates: heightmap cell count at the scan resolution, the same at half the
resolution, and the extent of the depth pixels on the box top plus one pixel.
The arm is at the pile scan pose for what is left (task_node._scan_tcp), with the
tool's pixels masked as the method does.
usage: python footprint_accuracy.py [--noise] [--seed N]
"""
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wrist_pose  # noqa: E402
from perception import heightmap as hm  # noqa: E402
from pick_place_common.depth_noise import add_depth_noise  # noqa: E402
from pick_place_common.mujoco_sim_node import load_scene_model  # noqa: E402
from pick_place_common.scene import (  # noqa: E402
    CONTAINER_CAM_HEIGHT, CONTAINER_CAM_NAME,
    CONTAINER_CAM_WIDTH, FLATNESS_TOL, FLOOR_Z, GRASP_INLIER_FRAC, SCAN_RESOLUTION,
    SOURCE_MIN_FILL_FRAC, SOURCE_MIN_FOOTPRINT_CELLS, SOURCE_SCAN_BOUNDS)
from pick_place_mpc import task_node as tn  # noqa: E402

NOISE = '--noise' in sys.argv
RNG = np.random.default_rng(int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 0)
m = load_scene_model(world="arm")
d = mujoco.MjData(m)
boxes = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b): b for b in range(m.nbody)
         if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith("cbox_")}
wrist_pose.reset_scene(m, d)
r = mujoco.Renderer(m, CONTAINER_CAM_HEIGHT, CONTAINER_CAM_WIDTH)
r.enable_depth_rendering()
opt = mujoco.MjvOption()
cam_id = m.camera(CONTAINER_CAM_NAME).id
fovy = m.cam_fovy[cam_id]
f_px = (CONTAINER_CAM_HEIGHT / 2.0) / np.tan(np.radians(fovy) / 2.0)


def scan(res):
    top = max(d.xpos[b][2] + m.geom_size[m.body_geomadr[b]][2] for b in boxes.values() if d.xpos[b][0] < 2.0)
    p = tn._scan_tcp(SOURCE_SCAN_BOUNDS, top)
    wrist_pose.place_tcp(m, d, p, tn.TaskNode._aligned_yaw(float(np.arctan2(p[1], p[0]))))
    pts, shape = hm.build_dense_grid_xy(*SOURCE_SCAN_BOUNDS, res)
    r.update_scene(d, camera=CONTAINER_CAM_NAME, scene_option=opt)
    depth = r.render()
    if NOISE:
        depth = add_depth_noise(depth, RNG)
    depth[tn._tool_mask()] = 0.0
    cam_pos, cam_mat = d.cam_xpos[cam_id].copy(), d.cam_xmat[cam_id].reshape(3, 3).copy()
    h, _ = hm.infer_heights_parallax_corrected(depth, cam_pos, cam_mat, fovy, CONTAINER_CAM_WIDTH,
                                               CONTAINER_CAM_HEIGHT, pts, (FLOOR_Z + top) / 2.0, FLOOR_Z)
    return h.reshape(shape), depth, cam_pos, cam_mat


def pixel_extent(depth, cam_pos, cam_mat, rec, res):
    """Half extents from heightmap.top_extent over the record's cells plus one."""
    row0, col0, row1, col1, height, _ = rec
    (x0, _), (y0, _) = SOURCE_SCAN_BOUNDS
    # As task_node._top_footprint: the flat patch connected to the top's centre.
    e = hm.top_extent(depth, cam_pos, cam_mat, fovy, CONTAINER_CAM_WIDTH, CONTAINER_CAM_HEIGHT,
                      (x0 + (col0 - 1) * res, x0 + col1 * res), (y0 + (row0 - 1) * res, y0 + row1 * res),
                      height, FLATNESS_TOL / 2,
                      seed_xy=hm.footprint_center_xy(row0, col0, (row1 - row0, col1 - col0), *SOURCE_SCAN_BOUNDS, res))
    return (e[1] - e[0]) / 2, (e[3] - e[2]) / 2


def main():
    errs = {"grid 5mm": [], "grid 2.5mm": [], "pixels": []}
    for _ in range(len(boxes)):
        hmap, depth, cam_pos, cam_mat = scan(SCAN_RESOLUTION)
        recs = hm.find_topmost_boxes(hmap, FLOOR_Z, FLATNESS_TOL, SOURCE_MIN_FOOTPRINT_CELLS,
                                     GRASP_INLIER_FRAC, SOURCE_MIN_FILL_FRAC)
        if not recs:
            break
        rec = hm.pick_order(hmap, recs, FLATNESS_TOL)[0]
        row0, col0, row1, col1, height, _ = rec
        cx, cy = hm.footprint_center_xy(row0, col0, (row1 - row0, col1 - col0), *SOURCE_SCAN_BOUNDS, SCAN_RESOLUTION)
        name = min(boxes, key=lambda n: np.hypot(*(d.xpos[boxes[n]][:2] - [cx, cy])) + abs(
            d.xpos[boxes[n]][2] + m.geom_size[m.body_geomadr[boxes[n]]][2] - height))
        true = m.geom_size[m.body_geomadr[boxes[name]]][:2]
        est5 = ((col1 - col0) * SCAN_RESOLUTION / 2, (row1 - row0) * SCAN_RESOLUTION / 2)
        h2, *_ = scan(SCAN_RESOLUTION / 2)
        recs2 = hm.find_topmost_boxes(h2, FLOOR_Z, FLATNESS_TOL, 4 * SOURCE_MIN_FOOTPRINT_CELLS,
                                      GRASP_INLIER_FRAC, SOURCE_MIN_FILL_FRAC)
        (x0, _), (y0, _) = SOURCE_SCAN_BOUNDS
        rec2 = min(recs2, key=lambda q: np.hypot(x0 + (q[1] + q[3]) / 2 * SCAN_RESOLUTION / 2 - cx,
                                                  y0 + (q[0] + q[2]) / 2 * SCAN_RESOLUTION / 2 - cy))
        est25 = ((rec2[3] - rec2[1]) * SCAN_RESOLUTION / 4, (rec2[2] - rec2[0]) * SCAN_RESOLUTION / 4)
        estp = pixel_extent(depth, cam_pos, cam_mat, rec, SCAN_RESOLUTION)
        line = f"{name}: true {2e3 * true[0]:.1f} x {2e3 * true[1]:.1f} mm"
        for k, e in (("grid 5mm", est5), ("grid 2.5mm", est25), ("pixels", estp)):
            err = 2e3 * (np.array(e) - true)
            errs[k] += list(err)
            line += f" | {k} {err[0]:+.1f}/{err[1]:+.1f}"
        print(line)
        a = m.jnt_qposadr[m.body_jntadr[boxes[name]]]
        d.qpos[a:a + 3] = [3.0, 3.0, 0.1 * len(errs["pixels"])]  # picked: out of the way
        mujoco.mj_forward(m, d)
    for k, e in errs.items():
        e = np.abs(e)
        print(f"{k:11s} width error (mm): mean {e.mean():.1f}, max {e.max():.1f}")


main()
