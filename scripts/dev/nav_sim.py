"""Offline closed-loop navigation, no ROS: the plant (the base's drives and sensors, the
seed's mismatch, the arm tucked, the crowd) and the method's whole chain (odometry,
localization in the saved map, people on the map with tracks, the navigator with
its planner and MPC, the safety layer) drive a sequence of goals. Scored on the
truth: arrival at the docks, the closest any person came to the chassis while it
moved, people inside the protective field, contacts, and the motion: wiggles (a turn
reversed and reversed again within 2 s: one swerve), oscillations (wiggles in a row),
turn-backs (turning in place one way, then back) and, apart, realignments (the same,
aligning on a dock's axis).
With --arm carry (the mobile job's driving pose; --box: the largest box held) also the
robot's upper structure against the people's whole bodies (body_clearance.py).
Localization: today's scan matching (icp) and the filter (ekf, slam/base_ekf.py) both run on the
same data every tick, --fusion picks the one that drives; scans reach them --scan-latency
late (as live); both scored on the truth every tick (loc_scoring.py). The docks are the ones
taught at commissioning (nav/docks.py); --teach is that commissioning: the navigation steered
on the truth (a technician parking the robot by eye) to each station's spot, the filter's pose
there saved as the dock. --conditions makes it
harder (plant_conditions.py; with "spill", a protective stop the first time in each drive the
base crosses the wet patch above WET_STOP_SPEED, as if someone stepped in); --json writes the scores.
usage: python nav_sim.py [--goals pick place home ...] [--drives 20] [--crowd 4] [--seed S]
       [--map stations] [--plot dir] [--arm tucked|carry] [--box] [--people stations|job|...] [--speed 1.0]
       [--base-safety 1.0] [--people-speed 1.0] [--conditions spill worn_tyre gyro_drift dropout]
       [--json file] [--fusion icp|ekf] [--scan-latency 0.02] [--dump file.npz] [--teach]
"""
import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tasks/pick_and_place/common")]
from core.base_odometry import BaseOdometry, wheel_speeds  # noqa: E402
from nav import docks as nav_docks, safety  # noqa: E402
from nav.navigator import Navigator  # noqa: E402
from perception.map_people import MapPeopleDetector  # noqa: E402
from perception.people_tracker import PeopleTracker  # noqa: E402
from pick_place_common import frames, lidar_sim, plant_conditions  # noqa: E402
from pick_place_common.loc_scoring import LocScore, line as loc_line  # noqa: E402
from pick_place_common.nav_scoring import DriveScore, summary, turning_summary  # noqa: E402
from pick_place_common.base_drive import BASE_BODY, WHEEL_JOINTS, BaseDrive  # noqa: E402
from pick_place_common.body_clearance import BodyClearance  # noqa: E402
from pick_place_common.mobile_scenarios import CROWDS, Crowd, walking  # noqa: E402
from pick_place_common.mujoco_sim_node import (  # noqa: E402
    ARM_ACTUATOR_NAMES, ARM_JOINT_NAMES, HOLD_KD, HOLD_KP, HOME_KEYFRAME, load_scene_model)
from pick_place_common.plant_mismatch import apply_base_mismatch, apply_plant_mismatch, refresh  # noqa: E402
from pick_place_common.scene import (  # noqa: E402
    ARM_CARRY_Q, ARM_TUCKED_Q, BASE_HOME_POSE, BASE_WHEEL_RADIUS_M, BASE_WHEEL_TRACK_M, LIDAR_MOUNTS_BASE, LIDAR_RATE_HZ,
    LIDAR_SCANNERS, PICK_DOCK_BASE, PLACE_DOCK_BASE, room_to_map)
from slam.base_ekf import BaseEKF  # noqa: E402
from slam.grid import OccupancyGrid  # noqa: E402
from slam.localizer import GridLocalizer  # noqa: E402
from slam.pose_graph import compose, relative, wrap  # noqa: E402
from slam.scan import scan_points  # noqa: E402

DT = 0.02
NAV_EVERY = 5  # ticks: the MPC at 10 Hz
DRIVE_TIMEOUT_S = 120.0
WHEEL_SPEED_MAX = 15.0
LOC_STALE_S = 0.5  # safety_node's localization watchdog: no pose this long, stop
WET_STOP_SPEED, WET_STOP_S = 0.3, 1.0
SLIP_TICK_M = 0.003  # the wheels' arc off the true travel by more than this in a tick: slipping


def hold_largest_box(m, d):
    """The largest box hung from the TCP (top at the TCP, axes along the tool's) and welded
    there as the plant's carry weld does; its body name."""
    sizes = {f"cbox_{i}": m.geom_size[[g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body(f"cbox_{i}").id][0]]
             for i in range(8)}
    name = max(sizes, key=lambda n: np.linalg.norm(sizes[n]))
    half = sizes[name]
    site = m.site("tcp_site").id
    tcp, rot = d.site_xpos[site].copy(), d.site_xmat[site].reshape(3, 3).copy()
    b = m.body(name)
    adr = m.jnt_qposadr[b.jntadr[0]]
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, rot.ravel())
    d.qpos[adr:adr + 3] = tcp + rot @ np.array([0.0, 0.0, half[2]])
    d.qpos[adr + 3:adr + 7] = q
    mujoco.mj_forward(m, d)
    eq = m.equality("box_carry").id
    ee = m.body("attachment").id
    m.eq_obj2id[eq] = b.id
    neg_p, neg_q, rel_p, rel_q = np.zeros(3), np.zeros(4), np.zeros(3), np.zeros(4)
    mujoco.mju_negPose(neg_p, neg_q, d.xpos[ee], d.xquat[ee])
    mujoco.mju_mulPose(rel_p, rel_q, neg_p, neg_q, d.xpos[b.id], d.xquat[b.id])
    m.eq_data[eq, 0:3] = 0.0
    m.eq_data[eq, 3:6] = rel_p
    m.eq_data[eq, 6:10] = rel_q
    d.eq_active[eq] = True
    return name


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
    ap.add_argument("--people", choices=tuple(CROWDS), default="stations")
    ap.add_argument("--no-person-margin", action="store_true")  # people's points as anything else's
    ap.add_argument("--speed", type=float, default=1.0)  # the base's top speed, m/s
    ap.add_argument("--accel", type=float, default=0.5)  # its acceleration limit, m/s^2
    ap.add_argument("--base-safety", type=float, default=1.0)  # its distances to people x this
    ap.add_argument("--people-speed", type=float, default=1.0)  # people walk at this x their speed
    ap.add_argument("--conditions", nargs="*", default=[], choices=plant_conditions.CONDITIONS)
    ap.add_argument("--json", default="")
    ap.add_argument("--fusion", choices=("icp", "ekf"), default="icp")  # the estimator that drives
    ap.add_argument("--scan-latency", type=float, default=0.02)  # s from a scan's capture to its use
    ap.add_argument("--dump", default="")  # per tick: time, truth, both estimates, the filter's covariance
    ap.add_argument("--teach", action="store_true")
    args = ap.parse_args()
    conditions = set(args.conditions)

    m = load_scene_model(layout="stations")
    d = mujoco.MjData(m)
    apply_plant_mismatch(m, np.random.default_rng(args.seed), ARM_JOINT_NAMES, [f"cbox_{i}" for i in range(8)])
    apply_base_mismatch(m, np.random.default_rng([args.seed, 2]), WHEEL_JOINTS, BASE_BODY,
                        worn=plant_conditions.WORN_TYRE if "worn_tyre" in conditions else 0.0)
    refresh(m, d)
    mujoco.mj_resetDataKeyframe(m, d, m.key(HOME_KEYFRAME).id)
    frames.free_bodies_at_rest(m, d)
    qa = [m.joint(n).qposadr[0] for n in ARM_JOINT_NAMES]
    va = [m.joint(n).dofadr[0] for n in ARM_JOINT_NAMES]
    ai = [m.actuator(n).id for n in ARM_ACTUATOR_NAMES]
    d.qpos[qa] = ARM_CARRY_Q if args.arm == "carry" else ARM_TUCKED_Q
    mujoco.mj_forward(m, d)
    q_hold = d.qpos[qa].copy()
    held = hold_largest_box(m, d) if args.box else None
    base = BaseDrive(m, d, np.random.default_rng([args.seed, 3]), conditions=conditions)
    odo = BaseOdometry(BASE_WHEEL_RADIUS_M, BASE_WHEEL_TRACK_M)
    lidar_rng = np.random.default_rng([args.seed, 1])
    crowd = Crowd(walking(CROWDS[args.people][:args.crowd], args.people_speed))
    dropouts = plant_conditions.Dropouts(np.random.default_rng([args.seed, 5])) if "dropout" in conditions else None
    bodies = BodyClearance(m, extra_bodies=[held] if held else ()) if args.arm == "carry" else None
    crowd_mocap = [m.body_mocapid[m.body(s[0]).id] for s in crowd.specs]
    fd = m.joint("base_free").dofadr[0]

    map_path = REPO / "data" / "maps" / args.map
    grid = OccupancyGrid.load(map_path.with_suffix(".yaml"))
    targets = {"pick": room_to_map(PICK_DOCK_BASE), "place": room_to_map(PLACE_DOCK_BASE)}  # the technician's spots
    docks = targets if args.teach else nav_docks.load(map_path)
    taught = {}
    nav = Navigator(grid, docks, room_to_map(BASE_HOME_POSE), speed=args.speed, accel=args.accel,
                    safety=args.base_safety)
    extra = safety.person_extra(args.base_safety)
    loc = GridLocalizer(grid, (0.0, 0.0, 0.0))  # starts at home, the map's origin
    ekf = BaseEKF((0.0, 0.0, 0.0), BASE_WHEEL_RADIUS_M, BASE_WHEEL_TRACK_M)
    det = MapPeopleDetector(grid.occupied_points(), LIDAR_MOUNTS_BASE, lidar_sim.BEAM_ANGLES)
    trk = PeopleTracker()

    def truth():
        b = base.truth(d)
        return np.array([b[0], b[1], 2 * np.arctan2(b[6], b[3])])

    goals = [args.goals[i % len(args.goals)] for i in range(args.drives)]
    k, next_scan = 0, 0.0
    map_to_odom = np.zeros(3)
    loc_t = 0.0
    scores = {"icp": LocScore(), "ekf": LocScore()}
    loc_ms, fitness, dropped = [], [], 0
    ekf_tick_ms, ekf_scan_ms = [], []
    latency = int(round(args.scan_latency / DT))
    queue = deque()  # (due tick, capture time, points, odometry then)
    dump = []
    slip, prev_tp, prev_enc, wet_stops = [], None, None, 0
    last_points = np.zeros((0, 2))
    person_points = np.zeros((0, 2))
    tracks = []
    cmd = (0.0, 0.0)
    drives = []
    state_time = {}
    traj = []
    t_wall = time.perf_counter()
    for i in range(50):  # settle; the gyro bias is learnt
        base.set_command((0.0, 0.0), d.time)
        base.update(d)
        d.ctrl[ai] = d.qfrc_bias[va] + HOLD_KP * (q_hold - d.qpos[qa]) - HOLD_KD * d.qvel[va]
        base.step(m, d, 10)
        gyro, _ = base.imu(d, DT)
        odo.update(base.encoders(d), DT, gyro[2])
        ekf.tick((i - 49) * DT, base.encoders(d), gyro[2])
    for goal in goals:
        score = DriveScore(goal, k * DT)
        rec = score.r
        started = False
        last_state = "clear"
        wet_stop = -1.0 if "spill" in conditions else None  # its start, once it happened
        while True:
            # Plant: the people, the base's drives, the arm's hold.
            here = crowd.step(DT, truth()[:2])
            for mid, (px, py, pyaw) in zip(crowd_mocap, here):
                d.mocap_pos[mid] = (px, py, 0.0)
                d.mocap_quat[mid] = (np.cos(pyaw / 2), 0.0, 0.0, np.sin(pyaw / 2))
            base.set_command(wheel_speeds(*cmd, BASE_WHEEL_RADIUS_M, BASE_WHEEL_TRACK_M, WHEEL_SPEED_MAX), d.time)
            base.update(d)
            d.ctrl[ai] = d.qfrc_bias[va] + HOLD_KP * (q_hold - d.qpos[qa]) - HOLD_KD * d.qvel[va]
            base.step(m, d, 10)
            k += 1
            gyro, _ = base.imu(d, DT)
            wheels = base.encoders(d)
            odom = odo.update(wheels, DT, gyro[2]).copy()
            t0 = time.perf_counter()
            ekf.tick(k * DT, wheels, gyro[2])
            ekf_tick_ms.append(1e3 * (time.perf_counter() - t0))
            # Method: scans at 15 Hz (localization, people), control at 10 Hz.
            scanned = k * DT >= next_scan
            if scanned:
                next_scan += 1.0 / LIDAR_RATE_HZ
                mujoco.mj_forward(m, d)
                scans = [lidar_sim.scan(m, d, i, lidar_rng) for i in range(len(LIDAR_SCANNERS))]
                last_points = scan_points(scans, lidar_sim.BEAM_ANGLES, LIDAR_MOUNTS_BASE)
                if dropouts is not None and dropouts.dropped(k * DT):
                    dropped += 1
                else:
                    queue.append((k + latency, k * DT, last_points, odom))
            while queue and queue[0][0] <= k:
                _due, t_cap, pts, odom_cap = queue.popleft()
                t0 = time.perf_counter()
                est = loc.update(odom_cap, pts)
                loc_ms.append(1e3 * (time.perf_counter() - t0))
                fitness.append(loc.quality)
                map_to_odom, loc_t = compose(est, relative(odom_cap, np.zeros(3))), k * DT
                icp_s = []

                def match(prior, pts=pts):
                    t1 = time.perf_counter()
                    out = loc.match(pts, prior)
                    icp_s.append(time.perf_counter() - t1)
                    return out
                t0 = time.perf_counter()
                ekf.scan(t_cap, match)
                ekf_scan_ms.append(1e3 * (time.perf_counter() - t0 - sum(icp_s)))
            ests = {"icp": compose(map_to_odom, odom), "ekf": ekf.pose}
            pose = np.array(room_to_map(truth())) if args.teach else ests[args.fusion]
            if scanned:
                tracks = trk.update(k * DT, det.update(pose, scans), det.seen_empty)
                c, s = np.cos(pose[2]), np.sin(pose[2])
                fg = det.foreground - pose[:2]
                person_points = np.column_stack([c * fg[:, 0] + s * fg[:, 1], -s * fg[:, 0] + c * fg[:, 1]])
            if not started:
                nav.set_goal(goal, pose)
                started = True
            if k % NAV_EVERY == 0:
                v, w = nav.step(k * DT, pose, odo.twist, [tr.x for tr in tracks])
                last_state, cap = safety.check(last_points, v, w, nav.docking, measured=odo.twist,
                                               people=None if args.no_person_margin else person_points,
                                               extra=extra)
                if k * DT - loc_t > LOC_STALE_S:
                    last_state = "stop"
                if wet_stop is not None:
                    tp = truth()
                    if wet_stop < 0.0 and plant_conditions.on_spill(tp[:2]) and abs(odo.twist[0]) > WET_STOP_SPEED:
                        wet_stop, wet_stops = k * DT, wet_stops + 1
                    if wet_stop >= 0.0 and k * DT - wet_stop < WET_STOP_S:
                        last_state = "stop"
                cmd = safety.limit(v, w, last_state, cap)
                score.command(k * DT, *cmd, NAV_EVERY * DT, last_state, nav.state)
                state_time[nav.state] = state_time.get(nav.state, 0.0) + NAV_EVERY * DT
            # Truth: people against the chassis and its protective field.
            tp = truth()
            for name, sc in scores.items():
                sc.add(ests[name], room_to_map(tp), ekf.cov if name == "ekf" else None)
            if args.dump or args.plot:
                dump.append(np.r_[k * DT, room_to_map(tp), ests["icp"], ests["ekf"], ekf.cov.ravel(),
                                  [ekf.counts[c] for c in ("slips", "skids", "rejected", "resets")], odo.twist[0]])
            enc = base.encoders(d) * BASE_WHEEL_RADIUS_M
            if prev_tp is not None:  # the wheels' arc against the true travel along the heading
                along = np.cos(tp[2]) * (tp[0] - prev_tp[0]) + np.sin(tp[2]) * (tp[1] - prev_tp[1])
                slip.append(abs(along - 0.5 * np.sum(enc - prev_enc)))
            prev_tp, prev_enc = tp, enc
            vel = d.qvel[fd:fd + 6]
            v_body = float(np.cos(tp[2]) * vel[0] + np.sin(tp[2]) * vel[1])
            score.truth(k * DT, tp, v_body, float(vel[5]), here, nav.docking, last_state, nav.state)
            if bodies is not None and k % NAV_EVERY == 0 and here:
                dist, part, who, p_robot, p_person = bodies.nearest(d)
                if np.isfinite(dist):
                    w_world = d.xmat[m.body(BASE_BODY).id].reshape(3, 3) @ vel[3:6]
                    toward = bodies.closing_speed(d, p_robot, p_person, vel[:3], w_world)
                    moving = abs(v_body) > 0.05 or abs(vel[5]) > 0.1
                    score.body(dist, toward, moving, f"{part} / {who}")
            traj.append(np.r_[k * DT, tp, pose, cmd])
            if nav.state in ("docked", "parked", "failed") or k * DT - rec["t0"] > DRIVE_TIMEOUT_S:
                break
        if nav.planned is not None:
            score.plan(*nav.planned)
        score.events(nav.counts)
        if nav.align_from:
            print(f"    aligns from (along, lateral m, yaw deg off the dock): {nav.align_from}")
        tp = truth()
        rec["t"] = k * DT - rec["t0"]
        rec["state"] = nav.state if k * DT - rec["t0"] <= DRIVE_TIMEOUT_S else "timeout"
        dock = {"pick": PICK_DOCK_BASE, "place": PLACE_DOCK_BASE}.get(goal, BASE_HOME_POSE)
        c, s = np.cos(dock[2]), np.sin(dock[2])
        dx, dy = tp[0] - dock[0], tp[1] - dock[1]
        rec["err"] = (c * dx + s * dy, -s * dx + c * dy, np.degrees(wrap(tp[2] - dock[2])))
        rec["retries"] = nav.retries
        if args.teach and goal in targets and rec["state"] == "docked":
            taught[goal] = tuple(float(v) for v in ekf.pose)
        drives.append(score)
        e = rec.get("err", (np.nan,) * 3)
        print(f"drive {len(drives):2d} to {goal:5s}: {rec['state']:7s} {rec['t']:5.1f} s {rec['dist']:5.1f} m; error "
              f"along/lateral {1e3 * e[0]:+5.0f}/{1e3 * e[1]:+5.0f} mm, yaw {e[2]:+.2f} deg, retries {rec.get('retries', 0)}; "
              f"{score.line()}", flush=True)
        if rec["min_clear"] < 0.3:
            print(f"    closest (t, v, w, person speed, person in robot frame, safety, nav): {rec['closest']}")
    print(f"{len(drives)} drives in {time.perf_counter() - t_wall:.0f} s wall, {k * DT:.0f} s simulated ("
          + ", ".join(f"{st} {t:.0f} s" for st, t in sorted(state_time.items(), key=lambda kv: -kv[1])) + ")")
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
    slip = np.array(slip)
    loc_out = {name: sc.summary() for name, sc in scores.items()}
    fit = np.array(fitness)
    loc_out["icp"].update(lost=loc.lost, scans=len(fit), ms_mean=float(np.mean(loc_ms)),
                          ms_p95=float(np.percentile(loc_ms, 95)), fitness_p5=float(np.percentile(fit, 5)),
                          fitness_p50=float(np.median(fit)))
    loc_out["ekf"].update(**ekf.counts, lost=ekf.counts["poor"], ms_mean=float(np.mean(ekf_scan_ms)),
                          ms_p95=float(np.percentile(ekf_scan_ms, 95)), ms_tick=float(np.mean(ekf_tick_ms)),
                          ms_tick_p95=float(np.percentile(ekf_tick_ms, 95)))
    for name, ls in loc_out.items():
        ls.update(dropped=dropped, slip_ticks=int(np.sum(slip > SLIP_TICK_M)), slip_tick_max=1e3 * float(np.max(slip)),
                  wet_stops=wet_stops)
    print(f"scans {len(fit)} (dropped {dropped}), fitness p5/p50 {np.percentile(fit, 5):.2f}/{np.median(fit):.2f}; "
          f"wheels slipping {int(np.sum(slip > SLIP_TICK_M))} ticks (worst {1e3 * np.max(slip):.1f} mm); wet stops {wet_stops}")
    print(f"localization (icp{', drives' if args.fusion == 'icp' else ''}): {loc_line(loc_out['icp'])}; lost {loc.lost}, "
          f"{np.mean(loc_ms):.1f} ms per scan")
    c = ekf.counts
    print(f"localization (ekf{', drives' if args.fusion == 'ekf' else ''}): {loc_line(loc_out['ekf'])}; scans accepted "
          f"{c['accepted']}, rejected {c['rejected']}, poor {c['poor']}, resets {c['resets']}, late {c['late']}; slips "
          f"{c['slips']} (skids {c['skids']}); {np.mean(ekf_tick_ms):.3f} ms per tick, {np.mean(ekf_scan_ms):.3f} ms per "
          f"scan besides the match")
    if args.teach:
        nav_docks.save(map_path, taught, note=f"nav_sim.py --teach, seed {args.seed}: parked by eye (the truth), "
                                              "the filter's pose saved")
        print(f"taught {sorted(taught)}: " + "; ".join(f"{k} {np.round(v, 4).tolist()} (the spot "
                                                      f"{np.round(targets[k], 4).tolist()})" for k, v in taught.items())
              + f" -> {nav_docks.path(map_path)}")
    if args.dump:
        np.save(args.dump, np.array(dump))
    if args.json:
        e = np.array([r["err"] for r in rs if r["state"] == "docked"]).reshape(-1, 3)
        dock = {"dock_lateral_max": float(1e3 * np.abs(e[:, 1]).max()), "dock_along_max": float(1e3 * np.abs(e[:, 0]).max()),
                "dock_yaw_max": float(np.abs(e[:, 2]).max())} if len(e) else {}
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps({"args": vars(args), "arrived": len(ok), "drives": len(rs),
                                               "localization": loc_out, **dock}, indent=1, default=float))
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        tr = np.array(traj)
        out = Path(args.plot)
        out.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 8))
        walls = np.array([compose(BASE_HOME_POSE, (x, y, 0.0))[:2] for x, y in grid.occupied_points()])
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
        dm = np.array(dump)
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
