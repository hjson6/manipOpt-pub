"""Offline check of the wrist camera mount (scene.py CONTAINER_CAM_*): at the pile
scan (measured and highest pile), the tray scans (first: each half of the place
zone; later: the found tray) and a held-box tray rescan, all with the aligned heading, the
arm is put at the scan pose (task_node._scan_tcp, IK), the camera renders, and:
  view    the scanned area's corners (floor and top) inside the image
  tool    tool pixels the fixed self-mask misses (must be 0)
  robot   heightmap cells whose pixel is on the robot (must be 0; not for the first tray
          scan's pictures, which only find the tray: tray_detect_check.py covers them)
  diff    heightmap vs the same frame rendered without the robot, both noise-free
          (depth noise drops pixels along edges, so the tool's outline changes the
          draw): cells that differ by more than 5 mm (must be 0), i.e. what the tool
          and arm add; also, with noise, cells off the true top by more than 1 cm
          (vertical ray: includes the parallax and occlusion any camera has)
Saves RGB (mask outlined) and depth images to <out_dir>. --mobile: the pile scans at the
mobile job's pick dock (the stations layout, the arm facing the pick table) instead.
usage: python wrist_cam_check.py [out_dir] [--mobile]
"""
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wrist_pose  # noqa: E402
from perception import heightmap as hm  # noqa: E402
from pick_place_common.depth_noise import add_depth_noise  # noqa: E402
from pick_place_common.mujoco_sim_node import load_scene_model  # noqa: E402
from pick_place_common.scene import (  # noqa: E402
    BOX_HEIGHT_MAX_M, CONTAINER_CAM_FOVY_DEG, CONTAINER_CAM_HEIGHT, CONTAINER_CAM_NAME, CONTAINER_CAM_WIDTH, DEST_TRAY_X_MAX, DEST_TRAY_X_MIN, DEST_TRAY_Y_MAX, DEST_TRAY_Y_MIN,
    DEST_TRAY_Z_MAX, FLOOR_Z, PICK_DOCK_ARM, PLACE_ZONE_BOUNDS, PLACE_ZONE_MAX_HEIGHT_M,
    SCAN_RESOLUTION, pick_zone)
from pick_place_mpc import task_node as tn  # noqa: E402

W, H, FOVY = CONTAINER_CAM_WIDTH, CONTAINER_CAM_HEIGHT, CONTAINER_CAM_FOVY_DEG
MOBILE = "--mobile" in sys.argv
ARGS = [a for a in sys.argv[1:] if a != "--mobile"]
OUT = Path(ARGS[0]) if ARGS else Path("/tmp/wrist_cam_check")
SOURCE_SCAN_BOUNDS, _grid, _shape, PICK_ZONE_MAX_HEIGHT_M = pick_zone(MOBILE)
OUT.mkdir(parents=True, exist_ok=True)
TOOL_MASK = tn._tool_mask()
TRAY = ((DEST_TRAY_X_MIN, DEST_TRAY_X_MAX), (DEST_TRAY_Y_MIN, DEST_TRAY_Y_MAX))

m = load_scene_model(world="arm", layout="stations", dock=PICK_DOCK_ARM) if MOBILE else load_scene_model(world="arm")
d = mujoco.MjData(m)
rgb_r, dep_r, seg_r = (mujoco.Renderer(m, H, W) for _ in range(3))
dep_r.enable_depth_rendering()
seg_r.enable_segmentation_rendering()
opt = mujoco.MjvOption()
cam = m.camera(CONTAINER_CAM_NAME).id
robot_root = m.body_rootid[m.body("link0").id]
geom_robot = m.body_rootid[m.geom_bodyid] == robot_root
tool_geoms = {m.geom("tool").id, m.geom("tool_visual").id}
robot_group = np.zeros(m.nbody, np.uint8)
robot_group[robot_root] = 1
NOISE_SEED = 0  # noqa: the check with noise is informational
DIFF_TOL_M = 0.005
no_robot = mujoco.MjvOption()
no_robot.geomgroup[2] = 0


def pile_top():
    return max(d.xpos[b][2] + m.geom_size[m.body_geomadr[b]][2] for b in range(m.nbody)
               if (m.body(b).name or "").startswith("cbox_"))


def true_tops(points, held_body=None):
    """Top surface under each (x, y), robot (and held box) excluded."""
    excl = robot_group.copy()
    if held_body is not None:
        excl[held_body] = 1  # rays ignore bodies whose group is excluded: use bodyexclude
    out = []
    gid = np.zeros(1, np.int32)
    for x, y in points:
        best = FLOOR_Z
        start = np.array([x, y, 2.0])
        body_ex = -1
        for _ in range(4):
            dist = mujoco.mj_ray(m, d, start, np.array([0, 0, -1.0]), None, 1, body_ex, gid)
            if dist < 0:
                break
            g = gid[0]
            if geom_robot[g] or m.geom_bodyid[g] == held_body:
                start = start + np.array([0, 0, -(dist + 1e-4)])
                continue
            best = max(start[2] - dist, FLOOR_Z)
            break
        out.append(best)
    return np.array(out)


def check(name, tcp, psi, bounds, top_z, floor_z, held=None, heightmap=True):
    wrist_pose.reset_scene(m, d)
    perr, rerr = wrist_pose.place_tcp(m, d, tcp, psi)
    held_body = None
    hidden = None
    if held is not None:
        held_body = m.body(held).id
        a = m.jnt_qposadr[m.body_jntadr[held_body]]
        hz = m.geom_size[m.body_geomadr[held_body]][2]
        d.qpos[a:a + 3] = d.site_xpos[m.site("tcp_site").id] - [0, 0, hz + 0.001]
        mujoco.mj_forward(m, d)
    cp, cm = d.cam_xpos[cam].copy(), d.cam_xmat[cam].reshape(3, 3).copy()
    if held is not None:
        hx, hy, hz = m.geom_size[m.body_geomadr[held_body]] + [tn.HELD_MASK_GROW_M, tn.HELD_MASK_GROW_M, 0]
        c = d.xpos[held_body]
        corners = [c + [sx * hx, sy * hy, sz * hz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
        hidden = hm.box_silhouette(corners, cp, cm, FOVY, W, H)
    views = []
    for r in (rgb_r, dep_r, seg_r):
        r.update_scene(d, camera=CONTAINER_CAM_NAME, scene_option=opt)
        views.append(r.render())
    rgb, clean, seg = views
    depth = add_depth_noise(clean, np.random.default_rng(NOISE_SEED))
    depth[TOOL_MASK] = 0.0
    clean = clean.copy()
    clean[TOOL_MASK] = 0.0
    dep_r.update_scene(d, camera=CONTAINER_CAM_NAME, scene_option=no_robot)
    ideal = dep_r.render()
    is_geom = seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
    gids = np.where(is_geom, seg[..., 0], 0)
    robot_px = is_geom & geom_robot[gids]
    tool_px = is_geom & np.isin(gids, list(tool_geoms))
    corners = [(x, y, z) for x in bounds[0] for y in bounds[1] for z in (floor_z, top_z)]
    uv = [hm.project_world_point(cp, cm, FOVY, W, H, c) for c in corners]
    in_view = all(p is not None and 0 <= p[0] < W and 0 <= p[1] < H for p in uv)
    pts, shape = hm.build_dense_grid_xy(*bounds, SCAN_RESOLUTION)
    hmap, pmap = hm.infer_heights_parallax_corrected(depth, cp, cm, FOVY, W, H, pts, (floor_z + top_z) / 2,
                                                     floor_z, hidden=hidden, top_z=top_z)
    on_robot = sum(1 for p in pmap if p is not None and robot_px[p] and not TOOL_MASK[p])
    ref, _ = hm.infer_heights_parallax_corrected(ideal, cp, cm, FOVY, W, H, pts, (floor_z + top_z) / 2,
                                                 floor_z, hidden=hidden, top_z=top_z)
    mine, _ = hm.infer_heights_parallax_corrected(clean, cp, cm, FOVY, W, H, pts, (floor_z + top_z) / 2,
                                                  floor_z, hidden=hidden, top_z=top_z)
    both = ~np.isnan(mine) & ~np.isnan(ref)
    diff = np.abs(mine - ref)[both]
    n_diff = int(np.sum(diff > DIFF_TOL_M))
    truth = true_tops(pts, held_body)
    known = ~np.isnan(hmap)
    err = np.abs(hmap - truth)
    bad = int(np.sum(known & (err > 0.01)))
    print(f"{name:14s} IK {1e3 * perr:.1f} mm {np.degrees(rerr):.2f} deg | cam {cp.round(3).tolist()} | "
          f"view {'ok' if in_view else 'CUT'} | tool missed {int(np.sum(tool_px & ~TOOL_MASK))} px, "
          f"tool shown {int(tool_px.sum())} px | robot px {int(robot_px.sum())}, cells on robot {on_robot} | "
          f"diff>5mm {n_diff} cells (max {1e3 * diff.max():.1f} mm) | off truth >1cm {bad}/{int(known.sum())}"
          + (f", hidden {int((~known).sum())} cells" if held is not None else ""))
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cnt, _ = cv2.findContours(TOOL_MASK.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, cnt, -1, (0, 0, 255), 1)
    dn = cv2.normalize(hm.fill_invalid(depth), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(str(OUT / f"{name}.png"), np.hstack([img, cv2.applyColorMap(255 - dn, cv2.COLORMAP_JET)]))
    return (in_view and (on_robot == 0 or not heightmap) and int(np.sum(tool_px & ~TOOL_MASK)) == 0
            and (n_diff == 0 or not heightmap))


def heading(pos, offset=0.0):
    return tn.TaskNode._aligned_yaw(float(np.arctan2(pos[1], pos[0]))) + offset


def main():
    wrist_pose.reset_scene(m, d)
    top = pile_top()
    ok = True
    p = tn._scan_tcp(SOURCE_SCAN_BOUNDS, top)
    ok &= check("pile", p, heading(p), SOURCE_SCAN_BOUNDS, top, FLOOR_Z)
    p = tn._scan_tcp(SOURCE_SCAN_BOUNDS, PICK_ZONE_MAX_HEIGHT_M)
    ok &= check("pile_first", p, heading(p), SOURCE_SCAN_BOUNDS, PICK_ZONE_MAX_HEIGHT_M, FLOOR_Z)
    if MOBILE:
        print(f"images in {OUT}; {'PASS' if ok else 'FAIL'}")
        sys.exit(0 if ok else 1)
    for i, part in enumerate(tn._halves(PLACE_ZONE_BOUNDS)):
        p = tn._scan_tcp(part, PLACE_ZONE_MAX_HEIGHT_M)
        ok &= check(f"tray_first_{i + 1}", p, heading(p), part, PLACE_ZONE_MAX_HEIGHT_M, FLOOR_Z, heightmap=False)
    p = tn._scan_tcp(TRAY, DEST_TRAY_Z_MAX, FLOOR_Z)
    ok &= check("tray", p, heading(p), TRAY, DEST_TRAY_Z_MAX, FLOOR_Z)
    p = tn._scan_tcp(TRAY, DEST_TRAY_Z_MAX, FLOOR_Z)
    p[2] = max(p[2], DEST_TRAY_Z_MAX + BOX_HEIGHT_MAX_M + tn.LIFT_MARGIN_M)
    ok &= check("tray_held", p, heading(p), TRAY, DEST_TRAY_Z_MAX, FLOOR_Z, held="cbox_0")
    print(f"images in {OUT}; {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
