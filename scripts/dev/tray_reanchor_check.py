"""Offline check of tray_detection.reanchor (the mobile job's later visits to the place
station): the tray found once at its nominal pose (the first scan's two frames), then
moved as a different docking moves it relative to the arm (up to +-1.5 cm and +-0.4
deg), found again from one frame at the tray scan pose; wall errors against the
moved tray (mid-wall), and how often it is refused. --held: a box in the gripper
(its pixels blanked, as task_node does) and the scan as high as a held box needs.
usage: python tray_reanchor_check.py [n_trials, default 30] [--held]
"""
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wrist_pose  # noqa: E402
from perception import heightmap  # noqa: E402
from perception import tray_detection as td  # noqa: E402
from pick_place_common.depth_noise import add_depth_noise  # noqa: E402
from pick_place_common.mujoco_sim_node import load_scene_model  # noqa: E402
from pick_place_common.scene import (  # noqa: E402
    CONTAINER_CAM_HEIGHT, CONTAINER_CAM_NAME, CONTAINER_CAM_WIDTH,
    DEST_TRAY_X_MAX, DEST_TRAY_X_MIN, DEST_TRAY_Y_MAX, DEST_TRAY_Y_MIN, FLOOR_Z,
    PLACE_ZONE_BOUNDS, PLACE_ZONE_MAX_HEIGHT_M, SCAN_RESOLUTION)
from pick_place_mpc import task_node as tn  # noqa: E402

SHIFT_M, TURN_DEG = 0.015, 0.4
HELD_SCAN_Z = 0.404  # the held-box scan height of the live run (over the pile's rest)


def hold_box(m, d, name="cbox_0"):
    """The box hanging from the TCP (top at the TCP, axes along the tool's); its eight
    corners."""
    tcp = d.site_xpos[m.site("tcp_site").id].copy()
    rot = d.site_xmat[m.site("tcp_site").id].reshape(3, 3).copy()
    b = m.body(name)
    half = m.geom_size[[g for g in range(m.ngeom) if m.geom_bodyid[g] == b.id][0]]
    adr = m.jnt_qposadr[b.jntadr[0]]
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, rot.ravel())
    d.qpos[adr:adr + 3] = tcp + rot @ np.array([0.0, 0.0, half[2]])
    d.qpos[adr + 3:adr + 7] = q
    mujoco.mj_forward(m, d)
    return [tcp + rot @ np.array([sx * half[0], sy * half[1], sz * 2 * half[2]])
            for sx in (-1, 1) for sy in (-1, 1) for sz in (0, 1)]


def frame(m, d, r, rng, tcp, held=False):
    cid = m.camera(CONTAINER_CAM_NAME).id
    wrist_pose.place_tcp(m, d, tcp, tn.TaskNode._aligned_yaw(float(np.arctan2(tcp[1], tcp[0]))))
    corners = hold_box(m, d) if held else None
    r.update_scene(d, camera=CONTAINER_CAM_NAME)
    depth = add_depth_noise(r.render(), rng)
    depth[tn._tool_mask()] = 0.0
    cam_pos, cam_mat = d.cam_xpos[cid].copy(), d.cam_xmat[cid].reshape(3, 3).copy()
    if held:
        grow = [c + 0.01 * np.sign(c - np.mean(corners, axis=0)) for c in corners]
        depth[heightmap.box_silhouette(grow, cam_pos, cam_mat, m.cam_fovy[cid], CONTAINER_CAM_WIDTH,
                                       CONTAINER_CAM_HEIGHT)] = 0.0
    return (depth, cam_pos, cam_mat, m.cam_fovy[cid], CONTAINER_CAM_WIDTH, CONTAINER_CAM_HEIGHT)


def move_tray(m, off, turn):
    """Turn the dest_tray_* geoms by turn (rad) about the tray's centre and shift them by
    off (arm frame)."""
    c = np.array([(DEST_TRAY_X_MIN + DEST_TRAY_X_MAX) / 2, (DEST_TRAY_Y_MIN + DEST_TRAY_Y_MAX) / 2])
    rot = np.array([[np.cos(turn), -np.sin(turn)], [np.sin(turn), np.cos(turn)]])
    q_turn = np.array([np.cos(turn / 2), 0.0, 0.0, np.sin(turn / 2)])
    for g in range(m.ngeom):
        if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("dest_tray_"):
            m.geom_pos[g][:2] = c + rot @ (m.geom_pos[g][:2] - c) + off
            q = np.zeros(4)
            mujoco.mju_mulQuat(q, q_turn, m.geom_quat[g].copy())
            m.geom_quat[g] = q


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    held = "--held" in sys.argv
    n = int(args[0]) if args else 30
    rng = np.random.default_rng(0)
    first = [tn._scan_tcp(part, PLACE_ZONE_MAX_HEIGHT_M) for part in tn._halves(PLACE_ZONE_BOUNDS)]
    errs, refused, reasons = [], 0, []
    for k in range(n):
        m = load_scene_model(world="arm")
        d = mujoco.MjData(m)
        wrist_pose.reset_scene(m, d)
        r = mujoco.Renderer(m, CONTAINER_CAM_HEIGHT, CONTAINER_CAM_WIDTH)
        r.enable_depth_rendering()
        frames = [frame(m, d, r, rng, tcp) for tcp in first]
        tray = td.refine_walls(td.detect_tray(frames, *PLACE_ZONE_BOUNDS, SCAN_RESOLUTION, FLOOR_Z), frames)
        trial = np.random.default_rng(2000 + k)
        off = trial.uniform(-SHIFT_M, SHIFT_M, 2)
        move_tray(m, off, np.radians(trial.uniform(-TURN_DEG, TURN_DEG)))
        wrist_pose.reset_scene(m, d)
        tcp = tn._scan_tcp(tray.bounds, tray.wall_top, tray.floor_z)
        if held:
            tcp[2] = max(tcp[2], HELD_SCAN_Z)
        again, why = td.reanchor(tray, [frame(m, d, r, rng, tcp, held)])
        r.close()
        if again is None:
            refused += 1
            reasons.append(why)
            continue
        truth = np.array([DEST_TRAY_X_MIN + off[0], DEST_TRAY_X_MAX + off[0], DEST_TRAY_Y_MIN + off[1],
                          DEST_TRAY_Y_MAX + off[1]])
        errs.append(np.array([again.x_min, again.x_max, again.y_min, again.y_max]) - truth)
    e = 1e3 * np.array(errs).reshape(-1, 4)
    print(f"{n} dockings (up to {1e3 * SHIFT_M:.0f} mm, {TURN_DEG} deg){', box held' if held else ''}, "
          f"{refused} refused {sorted(set(reasons))[:4]}")
    if len(e):
        print(f"  walls (mid-wall): mean |err| {np.abs(e).mean():.2f} mm, max {np.abs(e).max():.2f} mm")


if __name__ == "__main__":
    main()
