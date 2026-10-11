"""Offline, lockstep replay of a recorded run's controller inputs.

Feeds the goal / orientation / obstacle parameters that mpc_controller saw
on each active tick of a live run (bench_run.sh's mpc.csv) to the same
acados OCP, closed around a MuJoCo plant stepped in lockstep (no ROS, no
wall clock). Isolates what the controller itself does from what the live
timing does: with --lag 0 every torque is computed from the state right
before it is applied; --lag-prob P makes a fraction P of ticks apply the
previous tick's torque instead (one tick of dead time, as seen live).

usage: python replay_offline.py <run_dir> [--ticks N] [--start K] [--lag-prob P]
                                 [--set name=value ...] [--seed S] [--enc-noise SIGMA] [--fd-vel]
--delay applies each torque one tick after the state it was computed from (the
real-time plant); --predict then solves from the state predicted one tick ahead
with the controller's model and the torque being applied.
--enc-noise adds Gaussian noise to the measured joint positions; --fd-vel gives the
controller the finite difference of the measured positions as velocity (the plant's
sensor_noise model), instead of the true joint velocity.
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
from pick_place_common import frames  # noqa: E402
from pick_place_common.base_drive import BaseDrive  # noqa: E402
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
    ap.add_argument("--enc-noise", type=float, default=0.0)
    ap.add_argument("--fd-vel", action="store_true")
    ap.add_argument("--delay", action="store_true")
    ap.add_argument("--delay-substeps", type=int, default=10,
                    help="with --delay: the new torque starts this many physics substeps (2 ms) into "
                         "the tick; the previous one runs until then (10: a whole tick)")
    ap.add_argument("--predict", action="store_true")
    ap.add_argument("--ekf", type=float, default=0.0,
                    help="with --predict: estimate the state with an EKF on the measured joint positions; "
                         "the value is the process noise on velocity per tick [rad/s]")
    ap.add_argument("--ekf-meas-q", action="store_true",
                    help="with --ekf: predict ahead from the measured q and the estimated velocity")
    ap.add_argument("--ekf-r", type=float, default=0.0,
                    help="with --ekf: measurement noise std assumed for the joint positions [rad] "
                         "(default: --enc-noise)")
    ap.add_argument("--ekf-dist", type=float, default=0.0,
                    help="with --ekf: also estimate a torque disturbance per joint (random walk, this "
                         "std per tick [Nm]) and predict with it")
    ap.add_argument("--goal-ahead", type=int, default=0,
                    help="solve for the goal this many ticks later (with --delay --predict: 1)")
    ap.add_argument("--vel-blend", type=float, default=1.0,
                    help="with --predict: velocity estimate = blend x measured + (1 - blend) x the model's "
                         "prediction from the last tick (1: measured only)")
    ap.add_argument("--fd-substeps", type=int, default=0,
                    help="with --fd-vel: difference over the last this many physics substeps, not the 20 ms tick")
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

    m = load_scene_model(world="arm")
    d = mujoco.MjData(m)
    if os.environ.get("MISMATCH"):
        from pick_place_common.plant_mismatch import apply_plant_mismatch, refresh
        apply_plant_mismatch(m, np.random.default_rng(int(os.environ["MISMATCH"])), ARM_JOINT_NAMES,
                             [f"cbox_{i}" for i in range(8)])
        refresh(m, d)
    if os.environ.get("J7_ARMATURE"):
        m.dof_armature[m.jnt_dofadr[m.joint("joint7").id]] *= float(os.environ["J7_ARMATURE"])
    mujoco.mj_resetDataKeyframe(m, d, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home"))
    frames.free_bodies_at_rest(m, d)
    mujoco.mj_forward(m, d)
    base = BaseDrive(m, d, np.random.default_rng(0), noise=False)  # parked: the drives hold the wheels

    def step(n):
        base.update(d)
        base.step(m, d, n)
    qa = [m.joint(nm).qposadr[0] for nm in ARM_JOINT_NAMES]
    va = [m.joint(nm).dofadr[0] for nm in ARM_JOINT_NAMES]
    aa = [m.actuator(nm).id for nm in ARM_ACTUATOR_NAMES]
    nsub = round(0.02 / m.opt.timestep)

    rng = np.random.default_rng(args.seed)
    if args.predict:
        import casadi as ca
        f = ca.Function("f", [model.x, model.u, model.payload_mass], [model.xdot])
        xs, us = ca.SX.sym("x", 14), ca.SX.sym("u", 7)
        hh, xk = 0.02 / 4, xs
        for _ in range(4):
            k1 = f(xk, us, 0); k2 = f(xk + hh / 2 * k1, us, 0)
            k3 = f(xk + hh / 2 * k2, us, 0); k4 = f(xk + hh * k3, us, 0)
            xk = xk + hh / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        predict = ca.Function("predict", [xs, us], [xk])
        hd, xd = args.delay_substeps * 0.002 / 4, xs
        for _ in range(4):
            k1 = f(xd, us, 0); k2 = f(xd + hd / 2 * k1, us, 0)
            k3 = f(xd + hd / 2 * k2, us, 0); k4 = f(xd + hd * k3, us, 0)
            xd = xd + hd / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        predict_delay = ca.Function("predict_delay", [xs, us], [xd])
        # One tick with the torque switching from u1 to u2 after delay_substeps (EKF prediction).
        u2s = ca.SX.sym("u2", 7)
        n1 = args.delay_substeps if args.delay else 10
        xt = xs
        for n_sub, uu in ((n1, us), (10 - n1, u2s)):
            if n_sub == 0:
                continue
            hh2 = n_sub * 0.002 / 4
            for _ in range(4):
                k1 = f(xt, uu, 0); k2 = f(xt + hh2 / 2 * k1, uu, 0)
                k3 = f(xt + hh2 / 2 * k2, uu, 0); k4 = f(xt + hh2 * k3, uu, 0)
                xt = xt + hh2 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        tick_fn = ca.Function("tick", [xs, us, u2s], [xt, ca.jacobian(xt, xs), ca.jacobian(xt, us) + ca.jacobian(xt, u2s)])
        tick_torques = (np.zeros(7), np.zeros(7))
        from core.state_estimator import DelayEKF
        core_ekf = DelayEKF(model, 0.02, (args.delay_substeps if args.delay else 10) * 0.002, args.ekf,
                            args.ekf_r or max(args.enc_noise, 1e-6))
        x_est, p_est = None, None
        nd = 7 if args.ekf_dist > 0 else 0
        ekf_q = np.diag(np.r_[np.full(7, 1e-6), np.full(7, args.ekf), np.full(nd, args.ekf_dist)] ** 2)
        ekf_r = np.eye(7) * (args.ekf_r or max(args.enc_noise, 1e-6)) ** 2
        ekf_h = np.hstack([np.eye(7), np.zeros((7, 7 + nd))])
        dist = np.zeros(7)
    applied = np.zeros(7)
    x_expected = None  # the model's prediction of this tick's state, made last tick
    prev_u = np.zeros(7)
    qds, taus, stats, fails = [], [], [], 0
    tcps, goals = [], []
    end = min(n, args.start + args.ticks)
    t_solve = []
    q_prev = None
    for k in range(args.start, end):
        q_meas = d.qpos[qa] + rng.normal(0.0, args.enc_noise, 7) if args.enc_noise else d.qpos[qa].copy()
        qd_meas = d.qvel[va].copy()
        if args.fd_vel:
            qd_meas = np.zeros(7) if q_prev is None else (q_meas - q_prev) / (
                args.fd_substeps * m.opt.timestep if args.fd_substeps else 0.02)
        q_prev = q_meas
        x0 = np.concatenate([q_meas, qd_meas])
        if args.predict and args.ekf > 0 and not args.ekf_dist:
            # The controller's estimator (core/state_estimator.py).
            if core_ekf.x is None:
                core_ekf.reset(q_meas, qd_meas)
            else:
                core_ekf.update(q_meas, tick_torques[0], tick_torques[1], 0.0)
            x0 = core_ekf.ahead(prev_u if args.delay else applied, 0.0, q_meas if args.ekf_meas_q else None)
        elif args.predict and args.ekf > 0:
            if x_est is None:
                x_est, p_est = x0.copy(), np.eye(14 + nd) * 1e-4
            else:
                xn, jx, ju = (np.array(j) for j in tick_fn(x_est, tick_torques[0] + dist, tick_torques[1] + dist))
                a_mat = np.eye(14 + nd)
                a_mat[:14, :14] = jx
                if nd:
                    a_mat[:14, 14:] = ju
                x_est = xn.flatten()
                p_est = a_mat @ p_est @ a_mat.T + ekf_q
            gain = p_est @ ekf_h.T @ np.linalg.inv(ekf_h @ p_est @ ekf_h.T + ekf_r)
            corr = gain @ (q_meas - x_est[:7])
            x_est = x_est + corr[:14]
            if nd:
                dist = dist + corr[14:]
            p_est = (np.eye(14 + nd) - gain @ ekf_h) @ p_est
            x0 = np.array((predict_delay if args.delay else predict)(x_est, (prev_u if args.delay else applied) + dist)).flatten()
        elif args.predict:
            if x_expected is not None and args.vel_blend < 1.0:
                x0[7:] = args.vel_blend * x0[7:] + (1.0 - args.vel_blend) * x_expected[7:]
            x_expected = np.array(predict(x0, prev_u if args.delay else applied)).flatten()
            x0 = x_expected.copy()
        if k == args.start and args.init == "x0":
            for s in range(cfg.N + 1):
                solver.set(s, "x", x0)
        solver.set(0, "lbx", x0)
        solver.set(0, "ubx", x0)
        g = goal[min(k + args.goal_ahead, n - 1)]
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
            p = np.concatenate([obs[k], g, orient[min(k + args.goal_ahead, n - 1)], [0.0]])
            for s in range(cfg.N + 1):
                solver.set(s, "p", p)
        t0 = time.perf_counter()
        st = solver.solve()
        t_solve.append((time.perf_counter() - t0) * 1e3)
        u = solver.get(0, "u") if st == 0 else prev_u * 0.96
        fails += st != 0
        if args.delay:
            applied = prev_u
        else:
            applied = prev_u if rng.random() < args.lag_prob else u
        d.ctrl[aa] = applied
        if args.delay and args.delay_substeps < nsub:
            # The previous torque until the new one arrives, then the new one.
            d.ctrl[aa] = prev_u
            step(args.delay_substeps)
            d.ctrl[aa] = u
            step(nsub - args.delay_substeps)
            applied = u
            tick_torques = (prev_u.copy(), u.copy())
        elif args.fd_substeps:
            step(nsub - args.fd_substeps)
            q_prev = d.qpos[qa] + rng.normal(0.0, args.enc_noise, 7) if args.enc_noise else d.qpos[qa].copy()
            step(args.fd_substeps)
        else:
            step(nsub)
        if not (args.delay and args.delay_substeps < nsub):
            tick_torques = (applied.copy(), applied.copy())
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
        np.savez(args.save, qd=qds, tau=taus, err=np.array(stats), tcp=tcps, goal=goal[args.start:end])


if __name__ == "__main__":
    main()
