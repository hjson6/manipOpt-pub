"""Offline odometry check, no ROS: the plant's base (wheel drives, encoders, IMU,
the seed's wheel mismatch) drives scripted profiles on open floor with the arm
held, and the method's odometry (core/base_odometry.py) is compared with the
true pose: drift per metre driven and heading error, with and without the gyro.
--conditions (plant_conditions.py): each profile also with them, to check they do what
they say; "spill" adds a profile on the wet patch that ends in a protective stop.
usage: python odom_check.py [--seeds 0 1 2] [--plot figures/base] [--conditions spill worn_tyre gyro_drift]
"""
import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tasks/pick_and_place/common")]
from core.base_odometry import BaseOdometry, wheel_speeds  # noqa: E402
from pick_place_common import frames, plant_conditions  # noqa: E402
from pick_place_common.base_drive import BASE_BODY, WHEEL_JOINTS, BaseDrive  # noqa: E402
from pick_place_common.mujoco_sim_node import (  # noqa: E402
    ARM_ACTUATOR_NAMES, ARM_JOINT_NAMES, HOLD_KD, HOLD_KP, HOME_KEYFRAME, load_scene_model)
from pick_place_common.plant_mismatch import apply_base_mismatch, refresh  # noqa: E402
from pick_place_common.scene import BASE_WHEEL_RADIUS_M, BASE_WHEEL_TRACK_M  # noqa: E402

DT = 0.02
SUBSTEPS = 10
V_MAX, W_MAX = 0.5, 1.0  # m/s, rad/s
ACC, ALPHA = 0.5, 1.5  # m/s^2, rad/s^2
WHEEL_SPEED_MAX = 15.0


def segments(name):
    """[(v, omega, seconds at that speed[, stop at once at its end])], start pose (x, y, yaw)."""
    if name == "wet_stop":  # onto the wet patch, a turn on it, then a protective stop on it
        (x0, x1), (y0, y1) = plant_conditions.SPILL_RECT
        return [(V_MAX, 0.0, 1.0), (0.0, W_MAX, np.pi / 2 / W_MAX), (V_MAX, 0.0, 1.2, True)], (x0 - 0.05, y0 + 0.4, 0.0)
    if name == "straight":
        return [(V_MAX, 0.0, 6.0), (0.0, 0.0, 1.0), (-V_MAX, 0.0, 6.0)], (1.5, 1.5, 0.0)
    if name == "square":
        side, turn = (V_MAX, 0.0, 4.0), (0.0, W_MAX, np.pi / 2 / W_MAX)
        return [side, turn] * 4, (2.0, 1.5, 0.0)
    if name == "circle":
        return [(0.4, 0.4, 2 * np.pi / 0.4)], (3.5, 1.5, 0.0)
    raise ValueError(name)


def drive(seed, profile, conditions=()):
    m = load_scene_model(layout="stations")  # the open middle of the room
    apply_base_mismatch(m, np.random.default_rng([seed, 2]), WHEEL_JOINTS, BASE_BODY,
                        worn=plant_conditions.WORN_TYRE if "worn_tyre" in conditions else 0.0)
    d = mujoco.MjData(m)
    refresh(m, d)
    mujoco.mj_resetDataKeyframe(m, d, m.key(HOME_KEYFRAME).id)
    frames.free_bodies_at_rest(m, d)
    segs, start = segments(profile)
    fa = m.jnt_qposadr[m.joint("base_free").id]
    d.qpos[fa:fa + 3] = (start[0], start[1], 0.0)
    d.qpos[fa + 3:fa + 7] = (np.cos(start[2] / 2), 0.0, 0.0, np.sin(start[2] / 2))
    mujoco.mj_forward(m, d)
    qa = [m.joint(n).qposadr[0] for n in ARM_JOINT_NAMES]
    va = [m.joint(n).dofadr[0] for n in ARM_JOINT_NAMES]
    ai = [m.actuator(n).id for n in ARM_ACTUATOR_NAMES]
    q_hold = d.qpos[qa].copy()
    base = BaseDrive(m, d, np.random.default_rng([seed, 3]), conditions=conditions)
    odo = {k: BaseOdometry(BASE_WHEEL_RADIUS_M, BASE_WHEEL_TRACK_M, use_gyro=(k == "gyro")) for k in ("gyro", "wheels")}

    def truth():
        b = base.truth(d)
        return np.array([b[0], b[1], 2 * np.arctan2(b[6], b[3])])

    def tick(v, w):
        base.set_command(wheel_speeds(v, w, BASE_WHEEL_RADIUS_M, BASE_WHEEL_TRACK_M, WHEEL_SPEED_MAX), d.time)
        base.update(d)
        d.ctrl[ai] = d.qfrc_bias[va] + HOLD_KP * (q_hold - d.qpos[qa]) - HOLD_KD * d.qvel[va]
        base.step(m, d, SUBSTEPS)
        gyro, _ = base.imu(d, DT)
        enc = base.encoders(d)
        for k, o in odo.items():
            o.update(enc, DT, gyro[2])

    for _ in range(100):  # settle; the gyro's bias is learnt standing still
        tick(0.0, 0.0)
    t0 = truth()
    bias0 = base.gyro_bias[2]
    for o in odo.values():
        o.pose[:] = 0.0
        o.distance = 0.0
    rows = []
    v = w = 0.0
    plan = [(sv, sw, n, hard) for sv, sw, s, *h in segs for n, hard in [(int(round(s / DT)), bool(h and h[0]))]]
    for sv, sw, n, hard in plan + [(0.0, 0.0, 100, False)]:
        for _ in range(n):
            v += np.clip(sv - v, -ACC * DT, ACC * DT)
            w += np.clip(sw - w, -ALPHA * DT, ALPHA * DT)
            tick(v, w)
            rows.append(np.r_[truth(), odo["gyro"].pose, odo["wheels"].pose])
        if hard:
            v = w = 0.0
        # brake to the segment's end speed before the next: profiles chain at rest
        while abs(v) > 1e-9 or abs(w) > 1e-9:
            v += np.clip(-v, -ACC * DT, ACC * DT)
            w += np.clip(-w, -ALPHA * DT, ALPHA * DT)
            tick(v, w)
            rows.append(np.r_[truth(), odo["gyro"].pose, odo["wheels"].pose])
    rows = np.array(rows)
    # Express truth in the odom frame (odometry starts at zero at t0).
    c, s = np.cos(-t0[2]), np.sin(-t0[2])
    dx, dy = rows[:, 0] - t0[0], rows[:, 1] - t0[1]
    rows[:, 0], rows[:, 1], rows[:, 2] = c * dx - s * dy, s * dx + c * dy, rows[:, 2] - t0[2]
    return rows, odo["gyro"].distance, odo["gyro"].bias, base.gyro_bias[2], bias0


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--plot", default="")
    ap.add_argument("--conditions", nargs="*", default=[], choices=("spill", "worn_tyre", "gyro_drift"))
    args = ap.parse_args()
    results = {}
    print(f"{'profile':8s} {'seed':>4s} {'dist m':>7s} | {'gyro: end err mm':>16s} {'%':>5s} {'max mm':>7s} "
          f"{'yaw deg':>7s} | {'wheels: end mm':>14s} {'%':>5s} {'yaw deg':>7s} | bias est/true (start) mrad/s")
    runs = [(p, ()) for p in ("straight", "square", "circle")]
    if args.conditions:
        cond = tuple(args.conditions)
        runs = [r for p, _ in runs for r in ((p, ()), (p, cond))]
        if "spill" in cond:
            runs += [("wet_stop", ()), ("wet_stop", cond)]
    for profile, cond in runs:
        for seed in args.seeds:
            rows, dist, bias_est, bias_true, bias0 = drive(seed, profile, cond)
            if cond:
                profile_name = f"{profile}, {'+'.join(cond)}"
            else:
                profile_name = profile
            if not cond:
                results[(profile, seed)] = rows
            out = []
            for c in (3, 6):
                e = np.hypot(rows[:, c] - rows[:, 0], rows[:, c + 1] - rows[:, 1])
                out.append((1e3 * e[-1], 100 * e[-1] / dist, 1e3 * e.max(),
                            np.degrees(wrap(rows[-1, c + 2] - rows[-1, 2]))))
            g, wh = out
            print(f"{profile_name:8s} {seed:4d} {dist:7.2f} | {g[0]:16.1f} {g[1]:5.2f} {g[2]:7.1f} {g[3]:7.2f} | "
                  f"{wh[0]:14.1f} {wh[1]:5.2f} {wh[3]:7.2f} | {1e3 * bias_est:+.2f} / {1e3 * bias_true:+.2f} "
                  f"({1e3 * bias0:+.2f})")
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        out_dir = Path(args.plot)
        out_dir.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for profile in ("straight", "square", "circle"):
            for seed in args.seeds:
                rows = results[(profile, seed)]
                travelled = np.r_[0.0, np.cumsum(np.hypot(*np.diff(rows[:, :2], axis=0).T))]
                axes[0].plot(travelled, 1e3 * np.hypot(rows[:, 3] - rows[:, 0], rows[:, 4] - rows[:, 1]),
                             {"straight": "C0", "square": "C1", "circle": "C2"}[profile], lw=0.8,
                             label=profile if seed == args.seeds[0] else None)
        axes[0].set_xlabel("distance driven (m)")
        axes[0].set_ylabel("position error, wheels + gyro (mm)")
        axes[0].set_title(f"odometry error, seeds {', '.join(map(str, args.seeds))}")
        axes[0].grid(alpha=0.3)
        axes[0].legend(fontsize=8)
        for ax, profile in zip(axes[1:], ("square", "circle")):
            rows = results[(profile, args.seeds[0])]
            ax.plot(rows[:, 0], rows[:, 1], "k-", lw=2, label="true")
            ax.plot(rows[:, 3], rows[:, 4], "C0--", label="odometry, wheels + gyro")
            ax.plot(rows[:, 6], rows[:, 7], "C3:", label="odometry, wheels only")
            ax.set_title(f"{profile} (seed {args.seeds[0]})")
            ax.set_aspect("equal")
            ax.set_xlabel("x (m, odom)")
            ax.set_ylabel("y (m, odom)")
            ax.grid(alpha=0.3)
        axes[1].legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "odom_check.png", dpi=110)
        print(f"plot: {out_dir / 'odom_check.png'}")


if __name__ == "__main__":
    main()
