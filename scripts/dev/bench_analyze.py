"""Summarise bench_run.sh output directories, one line per run.

usage: python bench_analyze.py <run_dir> [<run_dir> ...]

Columns:
  boxes     boxes placed in the tray (the plant's "placed" lines in the log)
  flake     start flake: mean |qdot| > 2 rad/s over 3-9 s after /mpc/go
  fail      solver ticks with status != 0 (all run)
  solve     solve time p50 / p99 / max [ms], and ticks over the 20 ms budget
  lag       fraction of active steps whose torque came from the latest state
            (lag 0), one step old (lag 1), older (2+)
  period    physics step wall period p99 / max [ms], steps > 40 ms
  stale_mv  steps on the plant's PD fallback while the arm moves (|qdot|>0.1)
  qd        |qdot| p99 / max [rad/s]
  acc       |qddot| p99 [rad/s^2] (per step, finite differences, sim time)
  jerk      |qdddot| p99 [rad/s^3]
  hf        RMS of |qdot - 100 ms moving average| [rad/s]: jitter/shake energy
  hf_max    worst 1 s window of the same
  dtau      |tau(k) - tau(k-1)| p99 [Nm] (torque chatter)
  tilt      carried box tilt p99 / max [deg] (tool axis vs vertical, between
            "picked up" and "placed")
  stops     times the arm came to rest (|qdot| < 0.05 for >= 0.2 s) per box
  sway      1-5 Hz joint-speed content while moving: RMS of (100 ms average
            - 500 ms average) of qdot, worst joint [rad/s]: the slow sway
            of the whole arm that `hf` does not see
  dips      reference (goal) speed dropping by > 0.15 m/s and recovering
            by > 0.15 m/s within 1.5 s without stopping, per box
  tsway     1-5 Hz content of the tracking error (TCP minus goal) while
            moving, RMS / p99 [mm]: the arm wobbling about its reference
  resume    deepest reference-speed valley (a drop followed by a rise)
            within 3 s after a supervisor RESUME [m/s]; a normal stop at the
            end of the leg is not a valley
"""
import re
import sys
from pathlib import Path

import numpy as np

MJCF = str(Path(__file__).resolve().parents[2] / "tasks/pick_and_place/common/models/panda_robot.xml")


def load(path):
    with open(path) as f:
        head = f.readline().strip().split(",")
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    if data.ndim == 1:
        data = data[None, :]
    return {h: data[:, i] for i, h in enumerate(head)}


def moving_avg(x, n):
    k = np.ones(n) / n
    return np.vstack([np.convolve(c, k, mode="same") for c in x.T]).T


def analyze(run):
    run = Path(run)
    log = (run / "launch.log").read_text(errors="replace") if (run / "launch.log").exists() else ""
    sim, mpc = load(run / "sim.csv"), load(run / "mpc.csv")
    t_go = mpc["wall_t"][0]  # first solve = right after /mpc/go
    act = sim["wall_t"] >= t_go
    s = {k: v[act] for k, v in sim.items()}
    qd = np.column_stack([s[f"qd{i}"] for i in range(1, 8)])
    tau = np.column_stack([s[f"tau{i}"] for i in range(1, 8)])
    dt = 0.02
    acc = np.diff(qd, axis=0) / dt
    jerk = np.diff(acc, axis=0) / dt
    hf = np.linalg.norm(qd - moving_avg(qd, 5), axis=1)[3:-3]
    win = 50
    hf_max = max(np.sqrt(np.mean(hf[i:i + win] ** 2)) for i in range(0, max(1, len(hf) - win), 10))
    t_rel = s["wall_t"] - t_go
    early = (t_rel > 3) & (t_rel < 9)
    flake = s["qdot_norm"][early].mean() > 2.0 if early.any() else True
    active = s["stale"] == 0
    lag = s["cmd_lag"][active]
    moving = s["qdot_norm"] > 0.1
    sm = mpc["solve_ms"]
    # carried-box tilt from the measured joint angles
    tilt = np.array([0.0])
    carry = np.zeros(len(s["wall_t"]), bool)
    if "q1" in s:
        start = None
        for ts, what in re.findall(r"\[(\d+\.\d+)\] \[mujoco_sim_node\]: (picked up|placed)", log):
            if what == "picked up":
                start = float(ts)
            elif start is not None:
                carry |= (s["wall_t"] >= start) & (s["wall_t"] < float(ts))
                start = None
        if carry.any():
            import pinocchio as pin
            pm = pin.buildModelFromMJCF(MJCF)
            pd = pm.createData()
            fid = pm.getFrameId("tcp_site")
            qs = np.column_stack([s[f"q{i}"] for i in range(1, 8)])[carry]
            tilt = []
            for q in qs:
                pin.framesForwardKinematics(pm, pd, q)
                z = pd.oMf[fid].rotation[:, 2]
                tilt.append(np.degrees(np.arccos(np.clip(-z[2], -1, 1))))
            tilt = np.array(tilt)
    rest = s["qdot_norm"] < 0.05
    stops, run_len = 0, 0
    for r in rest:
        run_len = run_len + 1 if r else 0
        stops += run_len == 10
    boxes = len(re.findall(r"\[mujoco_sim_node\]: placed ", log))
    moving = s["qdot_norm"] > 0.1
    tsway = (0.0, 0.0)
    if "q1" in s:
        import pinocchio as pin
        pm = pin.buildModelFromMJCF(MJCF)
        pd = pm.createData()
        fid = pm.getFrameId("tcp_site")
        goal_at = {int(k): np.array([x, y, z]) for k, x, y, z in
                   zip(mpc["state_step"], mpc["goal_x"], mpc["goal_y"], mpc["goal_z"])}
        err, mv = [], []
        for i, k in enumerate(s["step"].astype(int)):
            if k in goal_at:
                pin.framesForwardKinematics(pm, pd, np.array([s[f"q{j}"][i] for j in range(1, 8)]))
                err.append(pd.oMf[fid].translation - goal_at[k])
                mv.append(moving[i])
        if len(err) > 30:
            err, mv = np.array(err), np.array(mv)
            b = np.linalg.norm(moving_avg(err, 5) - moving_avg(err, 25), axis=1)[mv]
            tsway = (np.sqrt(np.mean(b ** 2)) * 1e3, np.percentile(b, 99) * 1e3)
    band = moving_avg(qd, 5) - moving_avg(qd, 25)
    sway = float(np.max(np.sqrt(np.mean(band[moving] ** 2, axis=0)))) if moving.any() else 0.0
    g = np.column_stack([mpc["goal_x"], mpc["goal_y"], mpc["goal_z"]])
    gv = moving_avg(np.r_[0.0, np.linalg.norm(np.diff(g, axis=0), axis=1) / dt][:, None], 3)[:, 0]
    dips = 0
    i = 0
    while i < len(gv) - 1:
        # a local minimum of the reference speed that is not a stop
        w0, w1 = max(0, i - 75), min(len(gv), i + 75)
        if (gv[i] > 0.03 and gv[i] == gv[max(0, i - 5):i + 6].min()
                and gv[w0:i].max() - gv[i] > 0.15 and gv[i:w1].max() - gv[i] > 0.15
                and gv[i:w1].min() > 0.02):
            dips += 1
            i += 25
        i += 1
    resume = 0.0
    for ts in re.findall(r"\[(\d+\.\d+)\] \[task_node\]: supervisor RESUME", log):
        k = np.where((mpc["wall_t"] > float(ts)) & (mpc["wall_t"] < float(ts) + 3.0))[0]
        if len(k) > 2:
            v = gv[k]
            valley = np.minimum(np.maximum.accumulate(v) - v, np.maximum.accumulate(v[::-1])[::-1] - v)
            resume = max(resume, float(np.max(valley)))
    per = s["period_ms"][1:]
    return dict(
        run=run.name,
        boxes=boxes,
        done=int("parked indefinitely" in log),
        flake=int(flake),
        fail=int((mpc["status"] != 0).sum()),
        solve=f"{np.percentile(sm,50):.1f}/{np.percentile(sm,99):.1f}/{sm.max():.0f} ({int((sm>20).sum())})",
        lag=f"{np.mean(lag==0):.2f}/{np.mean(lag==1):.2f}/{np.mean(lag>=2):.2f}",
        period=f"{np.percentile(per,99):.0f}/{per.max():.0f} ({int((per>40).sum())})",
        stale_mv=int(((s["stale"] == 1) & moving).sum()),
        qd=f"{np.percentile(s['qdot_norm'],99):.2f}/{s['qdot_norm'].max():.2f}",
        acc=f"{np.percentile(np.linalg.norm(acc,axis=1),99):.1f}",
        jerk=f"{np.percentile(np.linalg.norm(jerk,axis=1),99):.0f}",
        hf=f"{np.sqrt(np.mean(hf**2)):.3f}",
        hf_max=f"{hf_max:.3f}",
        dtau=f"{np.percentile(np.linalg.norm(np.diff(tau,axis=0),axis=1),99):.2f}",
        tilt=f"{np.percentile(tilt,99):.1f}/{tilt.max():.1f}",
        stops=f"{stops / max(boxes, 1):.1f}",
        sway=f"{sway:.3f}",
        dips=f"{dips / max(boxes, 1):.1f}",
        tsway=f"{tsway[0]:.1f}/{tsway[1]:.1f}",
        resume=f"{resume:.2f}",
        dur=f"{t_rel[-1]:.0f}",
    )


if __name__ == "__main__":
    rows = [analyze(r) for r in sys.argv[1:]]
    keys = list(rows[0].keys())
    widths = {k: max(len(k), *(len(str(r[k])) for r in rows)) for k in keys}
    print("  ".join(k.ljust(widths[k]) for k in keys))
    for r in rows:
        print("  ".join(str(r[k]).ljust(widths[k]) for k in keys))
