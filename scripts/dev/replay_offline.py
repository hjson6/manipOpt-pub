"""Offline, lockstep replay of a recorded run's controller inputs.

Feeds the goal / orientation / obstacle parameters that mpc_controller saw
on each active tick of a live run (bench_run.sh's mpc.csv) to the same
acados OCP, closed around a MuJoCo plant stepped in lockstep (no ROS, no
wall clock). Isolates what the controller itself does from what the live
timing does: with --lag 0 every torque is computed from the state right
before it is applied; --lag-prob P makes a fraction P of ticks apply the
previous tick's torque instead (one tick of dead time, as seen live).

usage: python replay_offline.py <run_dir> [--ticks N] [--start K] [--lag-prob P]
                                 [--set name=value ...] [--seed S]
--set overrides MPCConfig fields that are runtime values (weights, bounds).
Prints the bench_analyze smoothness metrics for the replay.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import mujoco

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tasks/pick_and_place/common"), str(REPO / "tasks/pick_and_place/mpc")]
from bench_analyze import load, moving_avg  # noqa: E402
from pick_place_common.mujoco_sim_node import load_scene_model, ARM_JOINT_NAMES, ARM_ACTUATOR_NAMES  # noqa: E402


def smooth_metrics(qd, tau, dt=0.02):
    acc = np.diff(qd, axis=0) / dt
    jerk = np.diff(acc, axis=0) / dt
    hf = np.linalg.norm(qd - moving_avg(qd, 5), axis=1)[3:-3]
    nq = np.linalg.norm(qd, axis=1)
    return dict(
        qd=f"{np.percentile(nq,99):.2f}/{nq.max():.2f}",
        acc=f"{np.percentile(np.linalg.norm(acc,axis=1),99):.1f}",
        jerk=f"{np.percentile(np.linalg.norm(jerk,axis=1),99):.0f}",
        hf=f"{np.sqrt(np.mean(hf**2)):.3f}",
        dtau=f"{np.percentile(np.linalg.norm(np.diff(tau,axis=0),axis=1),99):.2f}",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--ticks", type=int, default=3000)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--lag-prob", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--init", default="none", help="none | x0 (fill the guess with x0 before the first solve)")
    ap.add_argument("--save", default=None)
    ap.add_argument("--preview", action="store_true",
                    help="give stage k the recorded goal k ticks later (the future reference) instead of "
                         "the same goal at every stage; the recording's 2-tick look-ahead is undone")
    args = ap.parse_args()

    from pick_place_mpc import mpc_controller_node as mcn
    from core.dynamics import load_manipulator, forward_kinematics
    from core.ocp import MPCConfig, build_ocp

    cfg = MPCConfig()
    for kv in args.set:
        k, v = kv.split("=")
        cur = getattr(cfg, k)
        if "," in v:
            val = np.array([float(x) for x in v.split(",")])
        elif isinstance(cur, bool):
            val = v.lower() in ("1", "true", "yes")
        elif isinstance(cur, np.ndarray):
            val = np.full(len(cur), float(v))
        else:
            val = float(v)
        setattr(cfg, k, val)
    model = load_manipulator(mcn.MJCF_PATH, [p.frame_name for p in mcn.PROXY_FRAMES])
    solver = build_ocp(model, mcn.PROXY_FRAMES, cfg)
    fk = forward_kinematics(model, cfg.ee_frame)
    q_center = (model.model.lowerPositionLimit + model.model.upperPositionLimit) / 2
    yref = np.zeros(7 + 7 + 7 + 9)
    base_idx = 14

    rec = load(str(Path(args.run) / "mpc.csv"))
    n = len(rec["goal_x"])
    goal = np.column_stack([rec["goal_x"], rec["goal_y"], rec["goal_z"]])
    orient = np.column_stack([rec[f"orient{i}"] for i in range(7)])
    obs = np.column_stack([rec[f"obs{i}"] for i in range(4 * cfg.n_obstacles)])

    m = load_scene_model()
    d = mujoco.MjData(m)
    if os.environ.get("MISMATCH"):
        from pick_place_common.plant_mismatch import apply_plant_mismatch, refresh
        apply_plant_mismatch(m, np.random.default_rng(int(os.environ["MISMATCH"])), ARM_JOINT_NAMES,
                             [f"cbox_{i}" for i in range(8)])
        refresh(m, d)
    if os.environ.get("J7_ARMATURE"):
        m.dof_armature[m.jnt_dofadr[m.joint("joint7").id]] *= float(os.environ["J7_ARMATURE"])
    mujoco.mj_resetDataKeyframe(m, d, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home"))
    for body_id in range(1, m.nbody):
        if m.body_jntnum[body_id] != 1:
            continue
        j = m.body_jntadr[body_id]
        if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE:
            continue
        a = m.jnt_qposadr[j]
        d.qpos[a:a + 3] = m.body_pos[body_id]
        d.qpos[a + 3:a + 7] = [1, 0, 0, 0]
    mujoco.mj_forward(m, d)
    qa = [m.joint(nm).qposadr[0] for nm in ARM_JOINT_NAMES]
    va = [m.joint(nm).dofadr[0] for nm in ARM_JOINT_NAMES]
    aa = [m.actuator(nm).id for nm in ARM_ACTUATOR_NAMES]
    nsub = round(0.02 / m.opt.timestep)

    rng = np.random.default_rng(args.seed)
    prev_u = np.zeros(7)
    qds, taus, stats, fails = [], [], [], 0
    tcps, goals = [], []
    end = min(n, args.start + args.ticks)
    t_solve = []
    for k in range(args.start, end):
        x0 = np.concatenate([d.qpos[qa], d.qvel[va]])
        if k == args.start and args.init == "x0":
            for s in range(cfg.N + 1):
                solver.set(s, "x", x0)
        solver.set(0, "lbx", x0)
        solver.set(0, "ubx", x0)
        g = goal[k]
        if np.hypot(g[0], g[1]) > 0.15:
            yref[base_idx] = np.arctan2(g[1], g[0]) - q_center[0]
            for s in range(cfg.N):
                solver.cost_set(s, "yref", yref)
        if args.preview:
            for s in range(cfg.N + 1):
                j = min(max(k - 2 + s, 0), n - 1)
                solver.set(s, "p", np.concatenate([obs[k], goal[j], orient[j], [0.0]]))
            g = goal[max(k - 2, 0)]
        else:
            p = np.concatenate([obs[k], g, orient[k], [0.0]])
            for s in range(cfg.N + 1):
                solver.set(s, "p", p)
        t0 = time.perf_counter()
        st = solver.solve()
        t_solve.append((time.perf_counter() - t0) * 1e3)
        u = solver.get(0, "u") if st == 0 else prev_u * 0.96
        fails += st != 0
        applied = prev_u if rng.random() < args.lag_prob else u
        d.ctrl[aa] = applied
        mujoco.mj_step(m, d, nstep=nsub)
        prev_u = u
        qds.append(d.qvel[va].copy())
        taus.append(applied.copy())
        tcp = np.array(fk(d.qpos[qa])).flatten()
        stats.append(float(np.linalg.norm(tcp - g)))
        tcps.append(tcp)
        goals.append(g.copy())
    qds, taus = np.array(qds), np.array(taus)
    tcps, goals = np.array(tcps), np.array(goals)
    # sideways error while the goal moves straight down (descents into the
    # pile and the tray): horizontal distance TCP - goal
    gv = np.r_[np.zeros((1, 3)), np.diff(goals, axis=0)]
    down = (gv[:, 2] < -0.002) & (np.hypot(gv[:, 0], gv[:, 1]) < 0.0005)
    side = np.hypot(*(tcps - goals)[:, :2].T)[down] * 1e3
    res = dict(ticks=end - args.start, fail=fails, solve_p99=f"{np.percentile(t_solve,99):.1f}",
               descent_side=f"{side.mean():.1f}/{side.max():.1f}mm" if len(side) else "-",
               ee_err_p99=f"{np.percentile(stats,99)*100:.1f}cm", **smooth_metrics(qds, taus))
    print("  ".join(f"{k}={v}" for k, v in res.items()))
    if args.save:
        np.savez(args.save, qd=qds, tau=taus, err=np.array(stats))


if __name__ == "__main__":
    main()
