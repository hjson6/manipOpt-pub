"""Sweeping motion in recorded runs (sim.csv, mpc.csv, task.csv), next to
bench_analyze's stutter measures.
  self      self-motion: joints 1-7 faster than 0.5 rad/s (norm) while the TCP moves
            slower than 5 cm/s and turns slower than 0.3 rad/s, for >= 0.2 s,
            not holding: the arm swinging without the tool going anywhere
  excess    per move (between rests), a turning joint's (1, 3, 5, 7) travel beyond
            its net change (back and forth), worst joint [rad]; over 0.3 rad listed
            (2, 4 and 6 go up and down with every lift and set-down)
  creep     joints drifting while the TCP holds still (TCP slower than 2 cm/s and
            0.1 rad/s, joints faster than 0.02 rad/s): joint travel per span
            [rad]; spans over 0.15 rad listed
  detour    per move, the TCP's path over the reference's path (moves with a
            reference path over 5 cm); over 1.25 listed
  track     |TCP - reference| while moving, p99 / max [mm]
usage: python motion_check.py [--plot <dir>] <run_dir> [<run_dir> ...]
"""
import sys
from pathlib import Path

import numpy as np
import pinocchio as pin

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_analyze import MJCF, load  # noqa: E402

DT = 0.02
REST_QD = 0.05
REST_TICKS = 10
SELF_QD = 0.5
SELF_V = 0.05
SELF_W = 0.3
SELF_TICKS = 10
EXCESS_FLAG = 0.3
TURNING_JOINTS = [0, 2, 4, 6]
CREEP_V = 0.02
CREEP_W = 0.1
CREEP_QD = 0.02
CREEP_FLAG = 0.15
DETOUR_FLAG = 1.25
DETOUR_MIN_PATH = 0.05

model = pin.buildModelFromMJCF(MJCF)
data = model.createData()
tcp_id = model.getFrameId("tcp_site")


def fk(q):
    pos, rot = np.zeros((len(q), 3)), np.zeros((len(q), 3, 3))
    for i, qi in enumerate(q):
        pin.framesForwardKinematics(model, data, qi)
        pos[i] = data.oMf[tcp_id].translation
        rot[i] = data.oMf[tcp_id].rotation
    return pos, rot


def runs_of(mask, min_len):
    """(start, end) index pairs of True runs at least min_len long."""
    out, start = [], None
    for i, m in enumerate(np.append(mask, False)):
        if m and start is None:
            start = i
        elif not m and start is not None:
            if i - start >= min_len:
                out.append((start, i))
            start = None
    return out


def check(run):
    run = Path(run)
    sim, mpc, task = load(run / "sim.csv"), load(run / "mpc.csv"), load(run / "task.csv")
    q = np.column_stack([sim[f"q{j}"] for j in range(1, 8)])
    qd = np.column_stack([sim[f"qd{j}"] for j in range(1, 8)])
    step = sim["step"].astype(int)
    pos, rot = fk(q)
    v = np.r_[0.0, np.linalg.norm(np.diff(pos, axis=0), axis=1) / DT]
    w = np.r_[0.0, [np.linalg.norm(pin.log3(rot[i - 1].T @ rot[i])) / DT for i in range(1, len(rot))]]
    k = np.ones(5) / 5
    v, w = np.convolve(v, k, "same"), np.convolve(w, k, "same")
    hold = np.zeros(len(step), bool)
    ti = {int(s): i for i, s in enumerate(task["step"])}
    ref_speed = np.full(len(step), np.nan)
    for i, s in enumerate(step):
        j = ti.get(int(s))
        if j is not None:
            hold[i] = task["hold"][j] > 0.5
            ref_speed[i] = task["ref_speed"][j]
    goal = np.full((len(step), 3), np.nan)
    mi = {int(s): i for i, s in enumerate(mpc["state_step"])}
    for i, s in enumerate(step):
        j = mi.get(int(s))
        if j is not None:
            goal[i] = (mpc["goal_x"][j], mpc["goal_y"][j], mpc["goal_z"][j])
    qn = np.linalg.norm(qd, axis=1)

    selfm = runs_of((qn > SELF_QD) & (v < SELF_V) & (w < SELF_W) & ~hold, SELF_TICKS)
    rest = qn < REST_QD
    moves = runs_of(~rest, REST_TICKS)
    flagged, worst_excess, worst_detour = [], (0.0, None), (0.0, None)
    for a, b in moves:
        dq = np.abs(np.diff(q[a:b + 1], axis=0)).sum(axis=0)
        net = np.abs(q[min(b, len(q) - 1)] - q[a])
        excess = np.zeros(7)
        excess[TURNING_JOINTS] = (dq - net)[TURNING_JOINTS]
        j = int(np.argmax(excess))
        t = sim["sim_t"][a]
        if excess[j] > worst_excess[0]:
            worst_excess = (float(excess[j]), f"q{j + 1} at {t:.1f} s")
        g = goal[a:b + 1]
        g = g[~np.isnan(g).any(axis=1)]
        ref_path = np.linalg.norm(np.diff(g, axis=0), axis=1).sum() if len(g) > 1 else 0.0
        path = np.linalg.norm(np.diff(pos[a:b + 1], axis=0), axis=1).sum()
        detour = path / ref_path if ref_path > DETOUR_MIN_PATH else np.nan
        if np.isfinite(detour) and detour > worst_detour[0]:
            worst_detour = (float(detour), f"at {t:.1f} s ({1e3 * path:.0f} vs {1e3 * ref_path:.0f} mm)")
        if excess[j] > EXCESS_FLAG or (np.isfinite(detour) and detour > DETOUR_FLAG):
            flagged.append((t, sim["sim_t"][min(b, len(q) - 1)], f"q{j + 1} excess {excess[j]:.2f} rad, "
                            f"detour {detour:.2f}"))
    creep = []
    for a, b in runs_of((v < CREEP_V) & (w < CREEP_W) & (qn > CREEP_QD), 5):
        travel = float(np.abs(np.diff(q[a:b + 1], axis=0)).sum())
        if travel > CREEP_FLAG:
            creep.append((sim["sim_t"][a], sim["sim_t"][min(b, len(q) - 1)], travel))
    moving = ~rest & ~hold & ~np.isnan(goal).any(axis=1)
    err = np.linalg.norm(pos - goal, axis=1)[moving] * 1e3
    return dict(run=run.name, sim=sim, v=v, ref_speed=ref_speed, hold=hold, qd=qd, selfm=selfm, moves=len(moves),
                creep=creep,
                flagged=flagged, worst_excess=worst_excess, worst_detour=worst_detour,
                track=(np.percentile(err, 99) if len(err) else np.nan, err.max() if len(err) else np.nan))


def plot(r, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = r["sim"]["sim_t"]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    for j in range(7):
        a1.plot(t, r["qd"][:, j], lw=0.6, label=f"q{j + 1}")
    a1.set_ylabel("joint speed (rad/s)")
    a1.legend(ncol=7, fontsize=7, loc="upper right")
    a2.plot(t, r["v"], lw=0.7, color="k", label="TCP speed")
    a2.plot(t, r["ref_speed"], lw=0.7, color="tab:blue", alpha=0.7, label="reference speed")
    a2.set_ylabel("m/s")
    a2.set_xlabel("sim time (s)")
    for ax in (a1, a2):
        for a, b in [(s, e) for s, e in zip(*np.flatnonzero(np.diff(np.r_[0, r["hold"].astype(int), 0])).reshape(-1, 2).T)]:
            ax.axvspan(t[a], t[min(b, len(t) - 1)], color="tab:red", alpha=0.12, lw=0)
        for a, b in r["selfm"]:
            ax.axvspan(t[a], t[min(b, len(t) - 1)], color="tab:purple", alpha=0.35, lw=0)
        for t0, t1, _ in r["flagged"]:
            ax.axvspan(t0, t1, color="tab:orange", alpha=0.25, lw=0)
        for t0, t1, _ in r["creep"]:
            ax.axvspan(t0, t1, color="tab:green", alpha=0.35, lw=0)
    a2.legend(fontsize=7, loc="upper right")
    fig.suptitle(f"{r['run']}: red hold, purple self-motion, orange flagged move (excess travel or detour), green creep")
    plt.tight_layout()
    out = Path(out_dir) / f"{r['run']}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=90)
    plt.close(fig)


def main():
    args = sys.argv[1:]
    plot_dir = None
    if args and args[0] == "--plot":
        plot_dir, args = args[1], args[2:]
    for run in args:
        r = check(run)
        print(f"{r['run']}: moves {r['moves']}, self-motion {len(r['selfm'])}, "
              f"worst excess {r['worst_excess'][0]:.2f} rad ({r['worst_excess'][1]}), "
              f"worst detour {r['worst_detour'][0]:.2f} ({r['worst_detour'][1]}), "
              f"track p99/max {r['track'][0]:.0f}/{r['track'][1]:.0f} mm, flagged moves {len(r['flagged'])}, "
              f"creep spans {len(r['creep'])}")
        for t0, t1, travel in r["creep"]:
            print(f"    creep {t0:.1f}-{t1:.1f} s: joints travelled {travel:.2f} rad with the TCP still")
        for a, b in r["selfm"]:
            print(f"    self-motion {r['sim']['sim_t'][a]:.1f}-{r['sim']['sim_t'][min(b, len(r['v']) - 1)]:.1f} s")
        for t0, t1, why in r["flagged"]:
            print(f"    flagged {t0:.1f}-{t1:.1f} s: {why}")
        if plot_dir:
            plot(r, plot_dir)


if __name__ == "__main__":
    main()
