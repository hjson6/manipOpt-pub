"""Offline check, no ROS: the arm under its MPC (the same OCP, solved from the true
state each tick, lockstep) holds a pose, tool down and level, while the mobile
base is parked, drives a sinusoidal speed profile forward and back, and turns in
place. The arm's model assumes a fixed base, so the base's motion is an unmodelled
disturbance; this measures how far the tool is pushed off its goal (arm frame).
usage: python arm_on_moving_base.py [--pose home|carry] [--v 0.5] [--period 6] [--w 1.0] [--payload KG]
"""
import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np
import pinocchio as pin

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tasks/pick_and_place/common"), str(REPO / "tasks/pick_and_place/mpc")]
from core.base_odometry import wheel_speeds  # noqa: E402
from core.dynamics import forward_kinematics, load_manipulator  # noqa: E402
from core.ocp import MPCConfig, build_ocp  # noqa: E402
from pick_place_common import frames  # noqa: E402
from pick_place_common.base_drive import BaseDrive  # noqa: E402
from pick_place_common.mujoco_sim_node import (  # noqa: E402
    ARM_ACTUATOR_NAMES, ARM_JOINT_NAMES, HOME_KEYFRAME, load_scene_model)
from pick_place_common.scene import BASE_WHEEL_RADIUS_M, BASE_WHEEL_TRACK_M  # noqa: E402
from pick_place_mpc import mpc_controller_node as mcn  # noqa: E402

POSES = {"home": None, "carry": (0.25, 0.0, 0.10)}  # TCP goals in the arm frame (None: home's own)
START = (2.5, 2.0, 0.0)  # open floor, room frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose", default="home", choices=POSES)
    ap.add_argument("--v", type=float, default=0.5)
    ap.add_argument("--period", type=float, default=6.0)
    ap.add_argument("--w", type=float, default=1.0)
    ap.add_argument("--payload", type=float, default=0.0, help="kg at the tool (plant and the OCP's payload)")
    args = ap.parse_args()

    cfg = MPCConfig()
    model = load_manipulator(mcn.MJCF_PATH, [p.frame_name for p in mcn.PROXY_FRAMES])
    solver = build_ocp(model, mcn.PROXY_FRAMES, cfg)
    fk = forward_kinematics(model, cfg.ee_frame)
    pdata = model.model.createData()
    q_center = (model.model.lowerPositionLimit + model.model.upperPositionLimit) / 2
    yref = np.zeros(7 + 7 + 7 + 9)

    m = load_scene_model()
    d = mujoco.MjData(m)
    if args.payload > 0:
        b = m.body("attachment").id
        m.body_mass[b] += args.payload
        m.body_ipos[b] = (0.0, 0.0, 0.15)
        m.body_inertia[b] += 1e-3
        mujoco.mj_setConst(m, d)
    mujoco.mj_resetDataKeyframe(m, d, m.key(HOME_KEYFRAME).id)
    frames.free_bodies_at_rest(m, d)
    fa = m.jnt_qposadr[m.joint("base_free").id]
    d.qpos[fa:fa + 3] = (START[0], START[1], 0.0)
    d.qpos[fa + 3:fa + 7] = (np.cos(START[2] / 2), 0, 0, np.sin(START[2] / 2))
    mujoco.mj_forward(m, d)
    qa = [m.joint(n).qposadr[0] for n in ARM_JOINT_NAMES]
    va = [m.joint(n).dofadr[0] for n in ARM_JOINT_NAMES]
    aa = [m.actuator(n).id for n in ARM_ACTUATOR_NAMES]
    base = BaseDrive(m, d, np.random.default_rng(0), noise=False)

    goal = np.array(fk(d.qpos[qa])).flatten() if POSES[args.pose] is None else np.array(POSES[args.pose])
    psi = 0.0
    orient = np.array([0.0, 0.0, -1.0, np.cos(psi), np.sin(psi), 0.0, 1.0])
    p = np.concatenate([np.tile([*mcn.NO_OBSTACLE_POSITION, mcn.NO_OBSTACLE_RADIUS], cfg.n_obstacles), goal, orient,
                        [args.payload]])
    for k in range(cfg.N + 1):
        solver.set(k, "p", p)
    if np.hypot(goal[0], goal[1]) > 0.15:
        yref[14] = np.arctan2(goal[1], goal[0]) - q_center[0]
        for k in range(cfg.N):
            solver.cost_set(k, "yref", yref)

    def guess(x0):
        g = pin.computeGeneralizedGravity(model.model, pdata, x0[:7])
        for k in range(cfg.N + 1):
            solver.set(k, "x", x0)
        for k in range(cfg.N):
            solver.set(k, "u", g)

    t_settle, t_drive, t_turn = 6.0, 2 * args.period, 2 * np.pi / args.w * 2
    phases = [("settle", t_settle), ("parked", 3.0), ("drive", t_drive), ("stop", 2.0), ("turn", t_turn), ("stop2", 2.0)]
    rows, fails, t = [], 0, 0.0
    x0 = np.concatenate([d.qpos[qa], d.qvel[va]])
    guess(x0)
    for name, dur in phases:
        t0 = t
        while t < t0 + dur - 1e-9:
            x0 = np.concatenate([d.qpos[qa], d.qvel[va]])
            solver.set(0, "lbx", x0)
            solver.set(0, "ubx", x0)
            st = solver.solve()
            if st != 0:
                fails += 1
                guess(x0)
            d.ctrl[aa] = solver.get(0, "u")
            tau = t - t0
            v = w = 0.0
            if name == "drive":
                v = args.v * np.sin(2 * np.pi * tau / args.period)
            elif name == "turn":
                w = args.w * np.sin(2 * np.pi * tau / (t_turn / 2)) if tau < t_turn else 0.0
            base.set_command(wheel_speeds(v, w, BASE_WHEEL_RADIUS_M, BASE_WHEEL_TRACK_M, 20.0), d.time)
            base.update(d)
            base.step(m, d, 10)
            t += 0.02
            tcp = np.array(fk(d.qpos[qa])).flatten()
            b = base.truth(d)
            rows.append((name, np.linalg.norm(tcp - goal), np.abs(d.qvel[va]).max(), np.abs(d.ctrl[aa]).max(),
                         np.degrees(np.arccos(np.clip(frames.arm_base_pose(m, d)[1][2, 2], -1, 1))), st,
                         np.hypot(*d.qvel[m.joint("base_free").dofadr[0]:][:2])))
    print(f"pose {args.pose} (goal {np.round(goal, 3).tolist()}), payload {args.payload} kg, drive +-{args.v} m/s "
          f"period {args.period} s (peak accel {args.v * 2 * np.pi / args.period:.2f} m/s^2), turn +-{args.w} rad/s; "
          f"{fails} solver failures")
    for name, _ in phases[1:]:
        r = [x for x in rows if x[0] == name]
        e = np.array([x[1] for x in r])
        print(f"  {name:7s} tool off goal p50 {1e3 * np.median(e):5.1f} / max {1e3 * e.max():5.1f} mm; "
              f"joint speed max {max(x[2] for x in r):.3f} rad/s; |tau| max {max(x[3] for x in r):.1f} Nm; "
              f"base tilt max {max(x[4] for x in r):.3f} deg; base speed max {max(x[6] for x in r):.2f} m/s")


if __name__ == "__main__":
    main()
