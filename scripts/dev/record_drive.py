"""Records a drive of the mobile base for SLAM, no ROS. The plant's base (its drives,
encoders, IMU, the seed's mismatch) follows a waypoint route, steered by a scripted
driver that sees the true pose (as a technician drives at commissioning; the method
never gets it); the arm is held at home. Saves the two lidars' raw ranges (15 Hz),
the encoders and the gyro (50 Hz), the method's odometry, and the true base pose and
people (for evaluation only) to data/recordings/<name>.npz. --layout stations: the
mobile job's room (the base starts at home); --crowd N: the first N of its crowd, with
each person's true velocity and the number of lidar beams on them per scan.
usage: python record_drive.py <name> [--route loop|cross] [--laps 2] [--people N] [--seed S]
       [--layout cell|stations] [--crowd N]
"""
import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tasks/pick_and_place/common")]
from core.base_odometry import BaseOdometry, wheel_speeds  # noqa: E402
from pick_place_common import frames, lidar_sim  # noqa: E402
from pick_place_common.base_drive import BASE_BODY, WHEEL_JOINTS, BaseDrive  # noqa: E402
from pick_place_common.mobile_scenarios import ACC, ALPHA, CROWD, PEOPLE, ROUTES, Crowd, Driver, person_at  # noqa: E402
from pick_place_common.mujoco_sim_node import (  # noqa: E402
    ARM_ACTUATOR_NAMES, ARM_JOINT_NAMES, HOLD_KD, HOLD_KP, HOME_KEYFRAME, load_scene_model)
from pick_place_common.plant_mismatch import apply_base_mismatch, apply_plant_mismatch, refresh  # noqa: E402
from pick_place_common.scene import (  # noqa: E402
    BASE_HOME_POSE, BASE_PARK_POSE, BASE_WHEEL_RADIUS_M, BASE_WHEEL_TRACK_M, LIDAR_RATE_HZ, LIDAR_SCANNERS)

DT = 0.02
OUT = REPO / "data" / "recordings"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--route", default="loop", choices=ROUTES)
    ap.add_argument("--laps", type=int, default=2)
    ap.add_argument("--people", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layout", default="cell", choices=("cell", "stations"))
    ap.add_argument("--crowd", type=int, default=0)
    args = ap.parse_args()

    m = load_scene_model(layout=args.layout)
    d = mujoco.MjData(m)
    apply_plant_mismatch(m, np.random.default_rng(args.seed), ARM_JOINT_NAMES, [f"cbox_{i}" for i in range(8)])
    apply_base_mismatch(m, np.random.default_rng([args.seed, 2]), WHEEL_JOINTS, BASE_BODY)
    refresh(m, d)
    mujoco.mj_resetDataKeyframe(m, d, m.key(HOME_KEYFRAME).id)
    frames.free_bodies_at_rest(m, d)
    mujoco.mj_forward(m, d)
    qa = [m.joint(n).qposadr[0] for n in ARM_JOINT_NAMES]
    va = [m.joint(n).dofadr[0] for n in ARM_JOINT_NAMES]
    ai = [m.actuator(n).id for n in ARM_ACTUATOR_NAMES]
    q_hold = d.qpos[qa].copy()
    base = BaseDrive(m, d, np.random.default_rng([args.seed, 3]))
    odo = BaseOdometry(BASE_WHEEL_RADIUS_M, BASE_WHEEL_TRACK_M)
    lidar_rng = np.random.default_rng([args.seed, 1])
    people = [(m.body_mocapid[m.body(name).id], path, speed, dwell) for name, path, speed, dwell in PEOPLE[:args.people]]
    cell = args.layout == "cell"
    driver = Driver(ROUTES[args.route] * args.laps, BASE_PARK_POSE if cell else BASE_HOME_POSE, reverse_out=cell)
    crowd = Crowd(CROWD[:args.crowd])
    crowd_mocap = [m.body_mocapid[m.body(s[0]).id] for s in crowd.specs]
    crowd_bodies = np.array([m.body(s[0]).id for s in crowd.specs])
    prev_crowd = None

    def truth():
        b = base.truth(d)
        return np.array([b[0], b[1], 2 * np.arctan2(b[6], b[3])])

    rec = {k: [] for k in ("t", "enc", "gyro", "odom", "truth", "people", "scan_t", "scans", "scan_truth",
                           "crowd", "crowd_beams")}
    v = w = 0.0
    k = 0
    next_scan = 0.0
    settle = int(1.0 / DT)
    while True:
        t = k * DT
        pose = truth()
        sv, sw = (0.0, 0.0) if k < settle else driver.command(pose)
        if driver.done and abs(v) < 1e-3 and abs(w) < 1e-3:
            break
        v += np.clip(sv - v, -ACC * DT, ACC * DT)
        w += np.clip(sw - w, -ALPHA * DT, ALPHA * DT)
        base.set_command(wheel_speeds(v, w, BASE_WHEEL_RADIUS_M, BASE_WHEEL_TRACK_M, 15.0), d.time)
        base.update(d)
        for mid, path, speed, dwell in people:
            px, py, pyaw = person_at(path, speed, dwell, t)
            d.mocap_pos[mid] = (px, py, 0.0)
            d.mocap_quat[mid] = (np.cos(pyaw / 2), 0.0, 0.0, np.sin(pyaw / 2))
        here = crowd.step(DT, pose[:2])
        for mid, (px, py, pyaw) in zip(crowd_mocap, here):
            d.mocap_pos[mid] = (px, py, 0.0)
            d.mocap_quat[mid] = (np.cos(pyaw / 2), 0.0, 0.0, np.sin(pyaw / 2))
        xy = np.array([h[:2] for h in here]).reshape(-1, 2)
        vel = np.zeros_like(xy) if prev_crowd is None else (xy - prev_crowd) / DT
        prev_crowd = xy
        d.ctrl[ai] = d.qfrc_bias[va] + HOLD_KP * (q_hold - d.qpos[qa]) - HOLD_KD * d.qvel[va]
        base.step(m, d, 10)
        k += 1
        gyro, _ = base.imu(d, DT)
        enc = base.encoders(d)
        odo.update(enc, DT, gyro[2])
        rec["t"].append(k * DT)
        rec["enc"].append(enc)
        rec["gyro"].append(gyro)
        rec["odom"].append(odo.pose.copy())
        rec["truth"].append(truth())
        rec["people"].append([d.mocap_pos[mid][:2].copy() for mid, *_ in people] or np.zeros((0, 2)))
        if k * DT >= next_scan:
            next_scan += 1.0 / LIDAR_RATE_HZ
            mujoco.mj_forward(m, d)
            rec["scan_t"].append(k * DT)
            hits = [lidar_sim.scan(m, d, i, lidar_rng, hit_geoms=True) for i in range(len(LIDAR_SCANNERS))]
            rec["scans"].append(np.array([r for r, _ in hits], dtype=np.float32))
            rec["scan_truth"].append(truth())
            body = np.concatenate([np.where(g >= 0, m.geom_bodyid[np.maximum(g, 0)], -1) for _, g in hits])
            rec["crowd"].append(np.column_stack([xy, vel]))
            rec["crowd_beams"].append([int(np.sum(body == b)) for b in crowd_bodies])
        if k * DT > 900:
            raise RuntimeError("the driver did not finish the route")
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{args.name}.npz"
    np.savez_compressed(out, **{k: np.array(v) for k, v in rec.items()}, route=args.route, laps=args.laps,
                        seed=args.seed, beam_angles=lidar_sim.BEAM_ANGLES, layout=args.layout)
    dist = np.sum(np.hypot(*np.diff(np.array(rec["truth"])[:, :2], axis=0).T))
    print(f"{out}: {k * DT:.0f} s, {dist:.1f} m driven, {len(rec['scans'])} scans, people {args.people}, "
          f"crowd {args.crowd}")


if __name__ == "__main__":
    main()
