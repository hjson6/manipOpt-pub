"""Offline closed-loop navigation, no ROS: the plant (the base's drives and sensors, the
seed's mismatch, the arm tucked, the crowd) and the method's whole chain (odometry,
localization in the saved map, people on the map with tracks, the navigator with
its planner and MPC, the safety layer) drive a sequence of goals. Scored on the
truth: arrival at the docks, the closest any person came to the chassis while it
moved, people inside the protective field, contacts, and the motion: wiggles (a turn
reversed and reversed again within 2 s: one swerve), oscillations (wiggles in a row),
turn-backs (turning in place one way, then back) and, apart, realignments (the same,
aligning on a dock's axis). The world itself is pick_place_common/nav_world.py; this is
its command line.
With --arm carry (the mobile job's driving pose; --box: the largest box held) also the
robot's upper structure against the people's whole bodies (body_clearance.py).
Localization: today's scan matching (icp) and the filter (ekf, slam/base_ekf.py) both run on the
same data every tick, --fusion picks the one that drives; scans reach them --scan-latency
late (as live); both scored on the truth every tick (loc_scoring.py); --fast runs and scores
only the one that drives. The docks are the ones
taught at commissioning (nav/docks.py); --teach is that commissioning: the navigation steered
on the truth (a technician parking the robot by eye) to each station's spot, the filter's pose
there saved as the dock. --conditions makes it
harder (plant_conditions.py; with "spill", a protective stop the first time in each drive the
base crosses the wet patch above WET_STOP_SPEED, as if someone stepped in); --json writes the scores.
--stop-safety scales the safety layer's person margin apart from --base-safety (the navigator's
distances). --fixture writes only what must repeat run to run: the sha256 of the per-tick
trajectory (truth, the estimate that drives, the command, the safety state, the people) and the
per-drive scores; no wall-clock timings, no paths (nav/test/test_nav_fixtures.py).
--nav-policy tuned: the navigation's constants tuned by optimizer (nav/tuned.py, --nav-model
names the file, default tuned_v1); rules (the default) is today's code path.
usage: python nav_sim.py [--goals pick place home ...] [--drives 20] [--crowd 4] [--seed S]
       [--map stations] [--plot dir] [--arm tucked|carry] [--box] [--people stations|job|...] [--speed 1.0]
       [--base-safety 1.0] [--stop-safety S] [--people-speed 1.0] [--conditions spill worn_tyre gyro_drift dropout]
       [--json file] [--fixture file] [--fusion icp|ekf] [--fast] [--scan-latency 0.02] [--dump file.npz] [--teach]
       [--nav-policy rules|tuned] [--nav-model tuned_v1]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tasks/pick_and_place/common")]
from core.learned_policy import PolicyFileError  # noqa: E402
from nav import docks as nav_docks, tuned as nav_tuned  # noqa: E402
from pick_place_common import plant_conditions  # noqa: E402
from pick_place_common.loc_scoring import line as loc_line  # noqa: E402
from pick_place_common.mobile_scenarios import CROWDS, random_crowd, walking  # noqa: E402
from pick_place_common.nav_scoring import record, summary, turning_summary  # noqa: E402
from pick_place_common.nav_world import DT, NavWorld  # noqa: E402
from pick_place_common.scene import BASE_HOME_POSE, PICK_DOCK_BASE, PLACE_DOCK_BASE  # noqa: E402
from slam.pose_graph import compose, wrap  # noqa: E402

SLIP_TICK_M = 0.003  # the wheels' arc off the true travel by more than this in a tick: slipping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--goals", nargs="*", default=["pick", "place"])
    ap.add_argument("--drives", type=int, default=20)
    ap.add_argument("--crowd", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--map", default="stations")
    ap.add_argument("--plot", default="")
    ap.add_argument("--arm", choices=("tucked", "carry"), default="tucked")
    ap.add_argument("--box", action="store_true")
    ap.add_argument("--people", choices=tuple(CROWDS) + ("random",), default="stations")  # random: up to --crowd
    ap.add_argument("--no-person-margin", action="store_true")  # people's points as anything else's
    ap.add_argument("--speed", type=float, default=1.0)  # the base's top speed, m/s
    ap.add_argument("--accel", type=float, default=0.5)  # its acceleration limit, m/s^2
    ap.add_argument("--base-safety", type=float, default=1.0)  # its distances to people x this
    ap.add_argument("--stop-safety", type=float, default=None)  # the stop field's person margin x this (else base's)
    ap.add_argument("--people-speed", type=float, default=1.0)  # people walk at this x their speed
    ap.add_argument("--conditions", nargs="*", default=[], choices=plant_conditions.CONDITIONS)
    ap.add_argument("--json", default="")
    ap.add_argument("--fixture", default="")
    ap.add_argument("--fusion", choices=("icp", "ekf"), default="icp")  # the estimator that drives
    ap.add_argument("--fast", action="store_true")  # only the estimator that drives runs (training)
    ap.add_argument("--scan-latency", type=float, default=0.02)  # s from a scan's capture to its use
    ap.add_argument("--dump", default="")  # per tick: time, truth, both estimates, the filter's covariance
    ap.add_argument("--teach", action="store_true")
    ap.add_argument("--nav-policy", choices=nav_tuned.MODES, default="rules")
    ap.add_argument("--nav-model", default="")
    args = ap.parse_args()
    try:
        tuned = nav_tuned.select(args.nav_policy, args.nav_model, args.base_safety)
    except PolicyFileError as e:
        sys.exit(f"nav policy: {e}")
    print(f"nav policy: {args.nav_policy}")
    conditions = set(args.conditions)

    map_path = REPO / "data" / "maps" / args.map
    if args.people == "random":
        def crowd(rects):
            return walking(random_crowd(np.random.default_rng([args.seed, 7]), rects, args.crowd), args.people_speed)
    else:
        crowd = walking(CROWDS[args.people][:args.crowd], args.people_speed)
    w = NavWorld(map_path, seed=args.seed, crowd=crowd,
                 arm=args.arm, box=args.box, speed=args.speed, accel=args.accel, base_safety=args.base_safety,
                 stop_safety=args.stop_safety, conditions=conditions, fusion=args.fusion,
                 scan_latency=args.scan_latency, person_margin=not args.no_person_margin, teach=args.teach,
                 fast=args.fast, record=bool(args.dump or args.plot))
    if tuned is not None:
        nav_tuned.apply_tuned(w.nav, tuned)
        print(f"tuned constants: {tuned}")
    if args.people == "random":
        specs = w.crowd.specs
        print(f"random crowd: {len(specs)} people, {sum(s[5] for s in specs)} giving way; "
              + ", ".join(f"{s[0]} {len(s[1])} points{' (steps in)' if len(s) > 6 else ''}" for s in specs))
    goals = [args.goals[i % len(args.goals)] for i in range(args.drives)]
    t_wall = time.perf_counter()
    for goal in goals:
        score = w.drive(goal)
        rec = score.r
        if w.nav.align_from:
            print(f"    aligns from (along, lateral m, yaw deg off the dock): {w.nav.align_from}")
        e = rec.get("err", (np.nan,) * 3)
        print(f"drive {len(w.drives):2d} to {goal:5s}: {rec['state']:7s} {rec['t']:5.1f} s {rec['dist']:5.1f} m; error "
              f"along/lateral {1e3 * e[0]:+5.0f}/{1e3 * e[1]:+5.0f} mm, yaw {e[2]:+.2f} deg, retries {rec.get('retries', 0)}; "
              f"{score.line()}", flush=True)
        if rec["min_clear"] < 0.3:
            print(f"    closest (t, v, w, person speed, person in robot frame, safety, nav): {rec['closest']}")
    drives = w.drives
    print(f"{len(drives)} drives in {time.perf_counter() - t_wall:.0f} s wall, {w.k * DT:.0f} s simulated ("
          + ", ".join(f"{st} {t:.0f} s" for st, t in sorted(w.state_time.items(), key=lambda kv: -kv[1])) + ")")
    rs = [sc.r for sc in drives]
    ok = [r for r in rs if r["state"] in ("docked", "parked")]
    out = [f"arrived {len(ok)}/{len(rs)}"]
    for what, state in (("docked", "docked"), ("parked at home", "parked")):
        e = np.array([r["err"] for r in rs if r["state"] == state]).reshape(-1, 3)
        if len(e):
            out.append(f"{what}: |lateral| max {1e3 * np.abs(e[:, 1]).max():.0f} mm, |along| max "
                       f"{1e3 * np.abs(e[:, 0]).max():.0f} mm, |yaw| max {np.abs(e[:, 2]).max():.2f} deg")
    print("; ".join(out + [summary(drives)]))
    print(turning_summary(drives))
    slip = np.array(w.slip)
    ran = [name for name in ("icp", "ekf") if not args.fast or name == args.fusion]
    loc_out = {name: w.scores[name].summary() for name in ran}
    fit = np.array(w.fitness)
    if "icp" in loc_out:
        loc_out["icp"].update(lost=w.loc.lost, scans=len(fit), ms_mean=float(np.mean(w.loc_ms)),
                              ms_p95=float(np.percentile(w.loc_ms, 95)), fitness_p5=float(np.percentile(fit, 5)),
                              fitness_p50=float(np.median(fit)))
    if "ekf" in loc_out:
        loc_out["ekf"].update(**w.ekf.counts, lost=w.ekf.counts["poor"], ms_mean=float(np.mean(w.ekf_scan_ms)),
                              ms_p95=float(np.percentile(w.ekf_scan_ms, 95)), ms_tick=float(np.mean(w.ekf_tick_ms)),
                              ms_tick_p95=float(np.percentile(w.ekf_tick_ms, 95)))
    for name, ls in loc_out.items():
        ls.update(dropped=w.dropped, slip_ticks=int(np.sum(slip > SLIP_TICK_M)), slip_tick_max=1e3 * float(np.max(slip)),
                  wet_stops=w.wet_stops)
    print((f"scans {len(fit)} (dropped {w.dropped}), fitness p5/p50 {np.percentile(fit, 5):.2f}/{np.median(fit):.2f}; "
           if len(fit) else f"scans dropped {w.dropped}; ")
          + f"wheels slipping {int(np.sum(slip > SLIP_TICK_M))} ticks (worst {1e3 * np.max(slip):.1f} mm); wet stops "
            f"{w.wet_stops}")
    if "icp" in loc_out:
        print(f"localization (icp{', drives' if args.fusion == 'icp' else ''}): {loc_line(loc_out['icp'])}; lost "
              f"{w.loc.lost}, {np.mean(w.loc_ms):.1f} ms per scan")
    if "ekf" in loc_out:
        c = w.ekf.counts
        print(f"localization (ekf{', drives' if args.fusion == 'ekf' else ''}): {loc_line(loc_out['ekf'])}; scans "
              f"accepted {c['accepted']}, rejected {c['rejected']}, poor {c['poor']}, resets {c['resets']}, late "
              f"{c['late']}; slips {c['slips']} (skids {c['skids']}); {np.mean(w.ekf_tick_ms):.3f} ms per tick, "
              f"{np.mean(w.ekf_scan_ms):.3f} ms per scan besides the match")
    if args.teach:
        nav_docks.save(map_path, w.taught, note=f"nav_sim.py --teach, seed {args.seed}: parked by eye (the truth), "
                                                "the filter's pose saved")
        print(f"taught {sorted(w.taught)}: " + "; ".join(f"{k} {np.round(v, 4).tolist()} (the spot "
                                                        f"{np.round(w.targets[k], 4).tolist()})"
                                                        for k, v in w.taught.items())
              + f" -> {nav_docks.path(map_path)}")
    if args.dump:
        np.save(args.dump, np.array(w.dump))
    if args.json:
        e = np.array([r["err"] for r in rs if r["state"] == "docked"]).reshape(-1, 3)
        dock = {"dock_lateral_max": float(1e3 * np.abs(e[:, 1]).max()), "dock_along_max": float(1e3 * np.abs(e[:, 0]).max()),
                "dock_yaw_max": float(np.abs(e[:, 2]).max())} if len(e) else {}
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        policy = {"mode": args.nav_policy, **({"tuned": tuned} if tuned else {})}
        Path(args.json).write_text(json.dumps({"args": vars(args), "arrived": len(ok), "drives": len(rs),
                                               "nav_policy": policy, "localization": loc_out, **dock,
                                               "drive_scores": [record(sc) for sc in drives]}, indent=1, default=float))
    if args.fixture:
        same = {k: v for k, v in vars(args).items() if k not in ("json", "fixture", "plot", "dump")}
        Path(args.fixture).parent.mkdir(parents=True, exist_ok=True)
        Path(args.fixture).write_text(json.dumps({"args": same, "trajectory_sha256": w.digest.hexdigest(),
                                                  "arrived": len(ok), "drive_scores": [record(sc) for sc in drives]},
                                                 indent=1, sort_keys=True) + "\n")
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        tr = np.array(w.traj)
        out = Path(args.plot)
        out.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 8))
        walls = np.array([compose(BASE_HOME_POSE, (x, y, 0.0))[:2] for x, y in w.grid.occupied_points()])
        ax.plot(walls[:, 0], walls[:, 1], ",", color="0.4")
        ax.plot(tr[:, 1], tr[:, 2], "b-", lw=0.6)
        for name, p in (("pick dock", PICK_DOCK_BASE), ("place dock", PLACE_DOCK_BASE), ("home", BASE_HOME_POSE)):
            ax.plot(p[0], p[1], "rs")
            ax.annotate(name, p[:2])
        ax.set_xlim(-0.2, 10.2)
        ax.set_ylim(-0.2, 8.2)
        ax.set_aspect("equal")
        ax.set_title(f"{len(drives)} drives, crowd of {args.crowd}, seed {args.seed} (room frame, true paths; grey: the map)")
        fig.savefig(out / "nav_sim.png", dpi=95)
        print(f"plot: {out / 'nav_sim.png'}")
        dm = np.array(w.dump)
        t, true = dm[:, 0], dm[:, 1:4]
        cov = dm[:, 10:19].reshape(-1, 3, 3)
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        for name, cols, color in (("scan matching (icp)", slice(4, 7), "C3"), ("filter (ekf)", slice(7, 10), "C0")):
            e = dm[:, cols] - true
            axes[0].plot(t, 1e3 * np.hypot(e[:, 0], e[:, 1]), color, lw=0.6, label=name)
            axes[1].plot(t, np.degrees(wrap(e[:, 2])), color, lw=0.6, label=name)
        axes[0].plot(t, 2e3 * np.sqrt(cov[:, 0, 0] + cov[:, 1, 1]), "C0:", lw=0.8, label="filter's 2 sigma")
        axes[1].fill_between(t, -2 * np.degrees(np.sqrt(cov[:, 2, 2])), 2 * np.degrees(np.sqrt(cov[:, 2, 2])), color="C0",
                             alpha=0.15, lw=0, label="filter's 2 sigma")
        e_all = [dm[:, c] - true for c in (slice(4, 7), slice(7, 10))]
        axes[0].set_ylim(0.0, 1.3 * max(max(1e3 * np.hypot(e[:, 0], e[:, 1]).max() for e in e_all),
                                       2e3 * np.percentile(np.sqrt(cov[:, 0, 0] + cov[:, 1, 1]), 99)))
        yaw_top = 1.3 * max(max(np.degrees(np.abs(wrap(e[:, 2]))).max() for e in e_all),
                            2 * np.degrees(np.percentile(np.sqrt(cov[:, 2, 2]), 99)))
        axes[1].set_ylim(-yaw_top, yaw_top)
        axes[0].set_ylabel("position error (mm)")
        axes[1].set_ylabel("heading error (deg)")
        axes[1].set_xlabel("time (s)")
        axes[0].set_title(f"localization against the truth: {len(drives)} drives, crowd of {args.crowd}, seed {args.seed}"
                          + (f", {', '.join(sorted(conditions))}" if conditions else "") + f"; {args.fusion} drives")
        for ax in axes:
            ax.grid(alpha=0.3)
            ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "localization.png", dpi=95)
        print(f"plot: {out / 'localization.png'}")


if __name__ == "__main__":
    main()
