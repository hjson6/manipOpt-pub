"""Baseline results for the learning-based MPC comparison: per-run metrics and a
per-scenario summary (handover_notes/baseline_benchmark_plan.md).

usage: python results_summary.py <results_dir>
  reads <results_dir>/runs/<scenario>_s<seed>[_repeat]/ (bench_run.sh output),
  writes <results_dir>/metrics.csv and <results_dir>/summary.md

Added to bench_analyze.py / baseline_summary.py:
  pred_q/pred_qd   one-step model error: the controller's ODE (with the payload
                   estimate) integrated one tick from the measured state with the
                   torque the plant applied, vs the measured next state; norm over joints
  track            TCP error vs the reference while moving, RMS / p99 [mm]
  settle_s         reference stopped -> settled pulse [s] (includes the 0.5 s dwell)
  bias             |goal bias| at each settle [mm]
  effort           integral of |tau|^2 per box [N^2 m^2 s]
  clear_true       min person-to-arm distance, geometry [m] (obstacle runs);
                   clear_moving: the same while |qdot| > 0.1 rad/s
  gap_sensed       min supervisor gap to a person track [m]
  det_err          workspace detection error, median / p95 [mm]
"""
import csv
import re
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_analyze import MJCF, analyze, load  # noqa: E402
from baseline_summary import summarise  # noqa: E402
from pick_place_common import frames  # noqa: E402
from pick_place_common.mujoco_sim_node import load_scene_model  # noqa: E402

DT = 0.02
NQ = 7
MOVING_QDOT = 0.1
CARRY_KG = 0.05
RK4_SUBSTEPS = 4
PERSON_PRESENT_Z = -1.0
CLEARANCE_EVERY = 2  # steps
CLEARANCE_MAX_M = 2.0


def _sim_by_step(sim):
    return {int(k): i for i, k in enumerate(sim["step"])}


_model = {}


def _one_step():
    """The controller's own ODE (core/dynamics.py), RK4 over one tick, batched."""
    if not _model:
        import casadi as ca
        from core.dynamics import load_manipulator
        m = load_manipulator(MJCF, [])
        f = ca.Function("f", [m.x, m.u, m.payload_mass], [m.xdot])
        x, u, pm = ca.SX.sym("x", 2 * NQ), ca.SX.sym("u", NQ), ca.SX.sym("pm")
        h, xk = DT / RK4_SUBSTEPS, x
        for _ in range(RK4_SUBSTEPS):
            k1 = f(xk, u, pm)
            k2 = f(xk + h / 2 * k1, u, pm)
            k3 = f(xk + h / 2 * k2, u, pm)
            k4 = f(xk + h * k3, u, pm)
            xk = xk + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        _model["step"] = ca.Function("step", [x, u, pm], [xk])
    return _model["step"]


def prediction_error(sim, mpc):
    at = _sim_by_step(sim)
    payload = dict(zip(mpc["state_step"].astype(int), mpc["payload"]))
    act = sim["wall_t"] >= mpc["wall_t"][0]
    rows = [i for i in np.where(act)[0] if int(sim["step"][i]) + 1 in at]
    nxt = [at[int(sim["step"][i]) + 1] for i in rows]
    q = np.column_stack([sim[f"q{k}"] for k in range(1, 8)])
    qd = np.column_stack([sim[f"qd{k}"] for k in range(1, 8)])
    tau = np.column_stack([sim[f"tau{k}"] for k in range(1, 8)])
    pm = np.array([payload.get(int(sim["step"][i]), 0.0) for i in rows])
    x0 = np.hstack([q[rows], qd[rows]])
    step = _one_step().map(len(rows))
    x1 = np.array(step(x0.T, tau[rows].T, pm[None, :])).T
    eq = np.linalg.norm(x1[:, :NQ] - q[nxt], axis=1)
    eqd = np.linalg.norm(x1[:, NQ:] - qd[nxt], axis=1)
    carry = pm > CARRY_KG
    rms = lambda x: float(np.sqrt(np.mean(x ** 2))) if len(x) else np.nan
    return dict(
        pred_n=len(eq),
        pred_q_rms_mrad=1e3 * rms(eq), pred_q_p99_mrad=1e3 * float(np.percentile(eq, 99)),
        pred_qd_rms=rms(eqd), pred_qd_med=float(np.median(eqd)), pred_qd_p99=float(np.percentile(eqd, 99)),
        pred_qd_rms_empty=rms(eqd[~carry]), pred_qd_rms_carry=rms(eqd[carry]),
    )


def tracking(sim, mpc):
    at = _sim_by_step(sim)
    idx = [(r, at.get(int(s))) for r, s in enumerate(mpc["state_step"])]
    rows = [r for r, i in idx if i is not None and sim["qdot_norm"][i] > MOVING_QDOT and mpc["status"][r] == 0]
    e = mpc["ee_err"][rows] * 1e3
    return dict(track_rms_mm=float(np.sqrt(np.mean(e ** 2))), track_p99_mm=float(np.percentile(e, 99)))


def settling(mpc):
    g = np.column_stack([mpc["goal_x"], mpc["goal_y"], mpc["goal_z"]])
    moved = np.r_[True, np.linalg.norm(np.diff(g, axis=0), axis=1) > 1e-6]
    edges = np.where(np.diff(mpc["settled"]) == 1)[0] + 1
    times, bias = [], []
    for e in edges:
        j = np.where(moved[:e + 1])[0][-1]
        times.append((mpc["state_step"][e] - mpc["state_step"][j]) * DT)
        bias.append(1e3 * np.linalg.norm([mpc["bias_x"][e], mpc["bias_y"][e], mpc["bias_z"][e]]))
    times, bias = np.array(times), np.array(bias)
    return dict(
        settles=len(edges),
        settle_med_s=float(np.median(times)) if len(times) else np.nan,
        settle_p90_s=float(np.percentile(times, 90)) if len(times) else np.nan,
        settle_max_s=float(times.max()) if len(times) else np.nan,
        bias_mean_mm=float(bias.mean()) if len(bias) else np.nan,
        bias_max_mm=float(bias.max()) if len(bias) else np.nan,
    )


def effort(sim, mpc, boxes):
    act = sim["wall_t"] >= mpc["wall_t"][0]
    tau = np.column_stack([sim[f"tau{i}"] for i in range(1, 8)])[act]
    return dict(effort_per_box=float(np.sum(tau ** 2) * DT / max(boxes, 1)))


_scene = {}


def true_clearance(sim):
    if "person_z" not in sim:
        return dict(clear_true_m=np.nan, clear_moving_m=np.nan)
    present = np.where(sim["person_z"] > PERSON_PRESENT_Z)[0][::CLEARANCE_EVERY]
    if not len(present):
        return dict(clear_true_m=np.nan, clear_moving_m=np.nan)
    if not _scene:
        m = load_scene_model(world="arm")
        d = mujoco.MjData(m)
        link1 = m.body("link1").id

        def under(b):
            while b > 0:
                if b == link1:
                    return True
                b = m.body_parentid[b]
            return False
        arm = [g for g in range(m.ngeom) if under(m.geom_bodyid[g])]
        person = [g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body("person_obstacle").id]
        qadr = [m.jnt_qposadr[m.joint(f"joint{i}").id] for i in range(1, 8)]
        _scene.update(m=m, d=d, arm=arm, person=person, qadr=qadr,
                      mocap=m.body_mocapid[m.body("person_obstacle").id])
    m, d = _scene["m"], _scene["d"]
    best, best_moving = np.inf, np.inf
    for i in present:
        d.qpos[_scene["qadr"]] = [sim[f"q{k}"][i] for k in range(1, 8)]
        x, y, z, yaw = frames.person_in_arm(sim, i)
        d.mocap_pos[_scene["mocap"]] = (x, y, z)
        d.mocap_quat[_scene["mocap"]] = (np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))
        mujoco.mj_kinematics(m, d)
        dist = min(mujoco.mj_geomDistance(m, d, a, p, CLEARANCE_MAX_M, None)
                   for a in _scene["arm"] for p in _scene["person"])
        best = min(best, dist)
        if sim["qdot_norm"][i] > MOVING_QDOT:
            best_moving = min(best_moving, dist)
    return dict(clear_true_m=float(best), clear_moving_m=float(best_moving))


def obstacle(run):
    out = dict(gap_sensed_m=np.nan, det_err_med_mm=np.nan, det_err_p95_mm=np.nan)
    if (run / "sup.csv").exists():
        rows = [r for r in csv.DictReader(open(run / "sup.csv")) if r["gap"]]
        gaps = [float(r["gap"]) for r in rows if r["label"] != "static"]
        if gaps:
            out["gap_sensed_m"] = min(gaps)
    if (run / "monitor.csv").exists():
        e = [float(r["err_xy"]) for r in csv.DictReader(open(run / "monitor.csv"))
             if r["err_xy"] not in ("", "nan") and r.get("source", "camera") == "camera"]
        if e:
            out["det_err_med_mm"] = 1e3 * float(np.median(e))
            out["det_err_p95_mm"] = 1e3 * float(np.percentile(e, 95))
    return out


def _rep(name):
    m = re.search(r"_r(\d+)$", name)
    return int(m.group(1)) if m else 1


def _time_per_box(sim, mpc, log):
    """Sim time from the first solve to the last placement, per box placed."""
    placed = [float(t) for t in re.findall(r"\[(\d+\.\d+)\] \[mujoco_sim_node\]: placed cbox_", log)]
    if not placed:
        return np.nan
    t = np.interp([mpc["wall_t"][0], placed[-1]], sim["wall_t"], sim["sim_t"])
    return float((t[1] - t[0]) / len(placed))


def metrics(run):
    run = Path(run)
    sim, mpc = load(run / "sim.csv"), load(run / "mpc.csv")
    a, b = analyze(run), summarise(run)
    sm = mpc["solve_ms"]
    log = (run / "launch.log").read_text(errors="replace")
    m = dict(
        run=run.name, scenario=re.sub(r"_s\d+.*$", "", run.name),
        seed=int(re.search(r"_s(\d+)", run.name).group(1)), rep=_rep(run.name),
        done=int(b["done"]), boxes=b["n"], boxes_floor=b["floor"],
        time_s=b["dur"], time_per_box_s=_time_per_box(sim, mpc, log),
        arrive_med_mm=float(b["arrive"].split(" / ")[0]) if b["arrive"] != "-" else np.nan,
        arrive_max_mm=b["arr_max"], timeouts=b["to"], aborts=b["abort"], solver_fail=b["fail"],
        solve_p50_ms=float(np.percentile(sm, 50)), solve_p99_ms=float(np.percentile(sm, 99)),
        solve_max_ms=float(sm.max()), over_budget=int((sm > 20).sum()),
        jerk_p99=float(a["jerk"]), dtau_p99=float(a["dtau"]),
        hf=float(a["hf"]) if "hf" in a else np.nan, j7=b["j7"],
        tilt_p99_deg=float(a["tilt"].split("/")[0]), tilt_max_deg=float(a["tilt"].split("/")[1]),
        tsway_rms_mm=float(a["tsway"].split("/")[0]) if "tsway" in a else np.nan,
        wall=b["wall"], push_yaw_deg=b["yaw"],
        gap_med_mm=float(b["gap"].split(" ")[0]) if b["gap"] != "-" else np.nan,
        holds=len(re.findall(r"task_node.*supervisor HOLD", log)),
    )
    m.update(prediction_error(sim, mpc))
    m.update(tracking(sim, mpc))
    m.update(settling(mpc))
    m.update(effort(sim, mpc, max(b["n"], 1)))
    m.update(true_clearance(sim))
    m.update(obstacle(run))
    return m


SUMMARY = [
    ("done", "runs finished", "count"), ("boxes", "boxes placed per run", "med"),
    ("time_per_box_s", "time per placed box [s]", "med"),
    ("pred_q_rms_mrad", "prediction error q, RMS [mrad]", "med"),
    ("pred_qd_rms", "prediction error qdot, RMS [rad/s]", "med"),
    ("pred_qd_med", "prediction error qdot, median [rad/s]", "med"),
    ("pred_qd_rms_empty", "  empty hand, RMS", "med"), ("pred_qd_rms_carry", "  carrying, RMS", "med"),
    ("track_rms_mm", "tracking error moving, RMS [mm]", "med"),
    ("track_p99_mm", "tracking error moving, p99 [mm]", "med"),
    ("tsway_rms_mm", "tracking 1-5 Hz, RMS [mm]", "med"),
    ("arrive_med_mm", "arrival error, median [mm]", "med"), ("arrive_max_mm", "arrival error, max [mm]", "max"),
    ("settle_med_s", "settle time, median [s]", "med"), ("settle_p90_s", "settle time, p90 [s]", "med"),
    ("bias_mean_mm", "goal bias at settle, mean [mm]", "med"),
    ("timeouts", "legs over time limit", "sum"), ("aborts", "aborted places", "sum"),
    ("solver_fail", "solver failures (ticks)", "sum"),
    ("solve_p50_ms", "solve time p50 [ms]", "med"), ("solve_p99_ms", "solve time p99 [ms]", "med"),
    ("solve_max_ms", "solve time max [ms]", "max"), ("over_budget", "ticks over 20 ms", "sum"),
    ("jerk_p99", "joint jerk p99 [rad/s^3]", "med"), ("dtau_p99", "torque step p99 [Nm]", "med"),
    ("j7", "joint-7 chatter share", "med"), ("tilt_max_deg", "carried-box tilt max [deg]", "max"),
    ("effort_per_box", "torque effort per box", "med"),
    ("wall", "tray-wall contacts", "sum"), ("push_yaw_deg", "push yaw max [deg]", "max"),
    ("gap_med_mm", "facing gap median [mm]", "med"),
    ("holds", "obstacle holds", "sum"), ("clear_true_m", "min true person clearance [m]", "min"),
    ("clear_moving_m", "  while the arm moves [m]", "min"),
    ("gap_sensed_m", "min sensed gap [m]", "min"), ("det_err_med_mm", "detection error median [mm]", "med"),
]

# Metrics whose seed-to-seed comparison matters most: noise within a seed vs between seeds.
NOISE = ["time_per_box_s", "pred_qd_rms", "track_rms_mm", "tsway_rms_mm", "arrive_med_mm",
         "settle_med_s", "solve_p99_ms", "jerk_p99", "effort_per_box", "boxes"]


def _num(v):
    return np.nan if v is None else float(v)


def seed_means(rows, scen, key):
    """{seed: mean over that seed's runs} for one scenario."""
    by = {}
    for r in rows:
        if r["scenario"] == scen:
            by.setdefault(r["seed"], []).append(_num(r[key]))
    return {s: float(np.nanmean(v)) if not np.all(np.isnan(v)) else np.nan for s, v in by.items()}


def noise(rows, scen, key):
    """(within-seed SD, pooled over seeds with 2+ runs; SD of the seed means)."""
    by = {}
    for r in rows:
        if r["scenario"] == scen and not np.isnan(_num(r[key])):
            by.setdefault(r["seed"], []).append(_num(r[key]))
    within = [np.var(v, ddof=1) for v in by.values() if len(v) > 1]
    means = [np.mean(v) for v in by.values()]
    return (float(np.sqrt(np.mean(within))) if within else np.nan,
            float(np.std(means, ddof=1)) if len(means) > 1 else np.nan)


def cell(rows, scen, key, how):
    if how == "count":
        runs = [r for r in rows if r["scenario"] == scen]
        return f"{sum(int(r[key]) for r in runs)}/{len(runs)}"
    if how in ("sum", "max", "min"):
        v = np.array([_num(r[key]) for r in rows if r["scenario"] == scen])
        v = v[~np.isnan(v)]
        if not len(v):
            return "-"
        return f"{v.sum():g}" if how == "sum" else f"{(v.max() if how == 'max' else v.min()):.3g}"
    v = np.array([x for x in seed_means(rows, scen, key).values() if not np.isnan(x)])
    if not len(v):
        return "-"
    return f"{np.median(v):.3g} [{v.min():.3g}-{v.max():.3g}]"


def main():
    out = Path(sys.argv[1])
    runs = sorted((p for p in (out / "runs").iterdir() if (p / "mpc.csv").exists()),
                  key=lambda p: (p.name.split("_s")[0], int(re.search(r"_s(\d+)", p.name).group(1)), _rep(p.name)))
    rows = []
    for r in runs:
        try:
            rows.append(metrics(r))
            print(f"ok   {r.name}")
        except Exception as e:  # a crashed run still gets listed
            print(f"FAIL {r.name}: {e!r}")
    keys = list(rows[0])
    with open(out / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()} for r in rows)
    scen = sorted({r["scenario"] for r in rows})
    n = {s: (len({r["seed"] for r in rows if r["scenario"] == s}), sum(r["scenario"] == s for r in rows))
         for s in scen}
    lines = ["# MPC baseline results", "",
             "Generated by `scripts/dev/results_summary.py`; protocol and metric definitions in "
             "`handover_notes/baseline_benchmark_plan.md`. Each seed's runs are averaged first; cells are "
             "median [min-max] over the seed means, or a count / total / extreme over all runs where marked.", "",
             "| metric | " + " | ".join(f"{s} ({n[s][0]} seeds, {n[s][1]} runs)" for s in scen) + " |",
             "|---|" + "---|" * len(scen)]
    for key, label, how in SUMMARY:
        lines.append(f"| {label} | " + " | ".join(cell(rows, s, key, how) for s in scen) + " |")
    lines += ["", "## Noise: within a seed vs between seeds", "",
              "SD between two runs of the same seed (pooled) and SD of the seed means. A paired "
              "comparison on seed means can only resolve differences well above the within-seed SD.", "",
              "| metric | " + " | ".join(f"{s} within / between" for s in scen) + " |",
              "|---|" + "---|" * len(scen)]
    for key in NOISE:
        lines.append(f"| {key} | " + " | ".join(
            "{:.3g} / {:.3g}".format(*noise(rows, s, key)) for s in scen) + " |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out / 'metrics.csv'} and {out / 'summary.md'}")


if __name__ == "__main__":
    main()
