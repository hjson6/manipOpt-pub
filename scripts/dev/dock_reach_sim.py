"""Offline comparison for the whole-body step, no ROS: the arm moving while the base
docks or undocks, three ways. seq: as the job does it (dock, then the arm; or the arm
to the carry pose, then undock); split: both at once, the base under its MPC as the
navigator drives it, the arm under its own MPC with its goals and the station's
hulls moved into the arm frame along the base's predicted motion; wb: the whole-body
MPC (core/wb_ocp.py) drives both. Scenarios (poses from the live job): pick-in (empty
hand, the carry pose to the pile scan pose while approaching the pick dock), pick-out
(a box held at the lift pose over the pile, to the carry pose while backing out),
place-in (a box held, to the tray scan pose), place-out (empty hand, the tray scan
pose to the carry pose). Out of the carry pose the arm goes straight up first, then
to the scan pose (fixed at the station: in the station frame W, the arm frame of the
nominal dock), started when the base is within --start-m of the dock; into it,
across at height, then down (both in the arm's own frame).
Lockstep: the controllers solve from the true state each 20 ms tick. Truth: the time
until both are done, the docking error, the tool against its reference, the arm and
a held box against the station (table, pile, tray: the smallest distance, contacts),
solve times.
usage: python dock_reach_sim.py [--scenario all|pick-in|pick-out|place-in|place-out]
       [--strategy all|seq|split|wb] [--start-m 0.3] [--seed 0]
"""
import argparse
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import pinocchio as pin
from ruckig import InputParameter, OutputParameter, Result, Ruckig, Trajectory

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tasks/pick_and_place/common"), str(REPO / "tasks/pick_and_place/mpc"),
                str(Path(__file__).resolve().parent)]
from core.base_odometry import wheel_speeds  # noqa: E402
from core.dynamics import forward_kinematics, load_manipulator  # noqa: E402
from core.ocp import MPCConfig, build_ocp  # noqa: E402
from core.wb_ocp import WholeBodyConfig, WholeBodyMPC  # noqa: E402
from nav.base_mpc import NO_PERSON, BaseMPC  # noqa: E402
from nav.navigator import (  # noqa: E402
    DOCK_SPEED, DOCK_TOL_ALONG_M, PRE_DOCK_M, STILL_V, STILL_W, UNDOCK_SPEED, behind, line)
from nav.reference import RouteReference, wrap  # noqa: E402
from nav_sim import hold_largest_box  # noqa: E402
from pick_place_common import frames  # noqa: E402
from pick_place_common.base_drive import BaseDrive  # noqa: E402
from pick_place_common.mujoco_sim_node import (  # noqa: E402
    ARM_ACTUATOR_NAMES, ARM_JOINT_NAMES, HOME_KEYFRAME, load_scene_model)
from pick_place_common.scene import (  # noqa: E402
    ARM_CARRY_Q, ARM_IN_BASE, BASE_WHEEL_RADIUS_M, BASE_WHEEL_TRACK_M, CARRY_TCP_POSITION, DEST_TRAY_X_MAX,
    DEST_TRAY_X_MIN, DEST_TRAY_Y_MAX, DEST_TRAY_Y_MIN, DEST_TRAY_Z_MAX, DEST_TRAY_Z_MIN, MOBILE_PICK_ZONE_BOUNDS,
    PICK_DOCK_ARM, PICK_DOCK_BASE, PLACE_DOCK_ARM, PLACE_DOCK_BASE, compose, invert, tool_down_rot)
from pick_place_mpc import mpc_controller_node as mcn  # noqa: E402
from pick_place_mpc import task_node as tn  # noqa: E402

DT = 0.02
NAV_EVERY = 5
WHEEL_SPEED_MAX = 15.0
SETTLE_TICKS = 25
ARM_TOL_M, ARM_STILL = 0.005, 0.05
TIMEOUT_S = 40.0
CARRY = np.array(CARRY_TCP_POSITION)
# From the quiet live job (arm frame at the dock): scan poses, the lift, headings. The pick
# station's, measured at the side-on dock (the arm at OLD_PICK_ARM), carried to the facing one:
# the pile scan by task_node's rule (the pile 0.20 m tall), the lift over the same table spot.
OLD_PICK_ARM = (0.85, 2.2, np.pi)


def _at_pick_dock(p):
    xy = compose(invert(PICK_DOCK_ARM), compose(OLD_PICK_ARM, (p[0], p[1], 0.0)))[:2]
    return (float(xy[0]), float(xy[1]), p[2])


def _aligned(p):
    return tn.TaskNode._aligned_yaw(float(np.arctan2(p[1], p[0])))


_PILE_SCAN = tuple(float(v) for v in tn._scan_tcp(MOBILE_PICK_ZONE_BOUNDS, 0.20))
_LIFT = _at_pick_dock((0.44, 0.305, 0.353))
SCENARIOS = {
    "pick-in": dict(station="pick", into=True, box=False, target=_PILE_SCAN, psi=_aligned(_PILE_SCAN)),
    "pick-out": dict(station="pick", into=False, box=True, start=_LIFT, psi=_aligned(_LIFT)),
    "place-in": dict(station="place", into=True, box=True, target=(-0.198, -0.421, 0.539), psi=np.pi / 2),
    "place-out": dict(station="place", into=False, box=False, start=(-0.198, -0.421, 0.539), psi=np.pi / 2),
}
CARRY_PSI = -np.pi / 2  # the carry pose's heading in the live job
TRAY_HULL = ((DEST_TRAY_X_MIN + DEST_TRAY_X_MAX) / 2, (DEST_TRAY_Y_MIN + DEST_TRAY_Y_MAX) / 2,
             (DEST_TRAY_Z_MIN + DEST_TRAY_Z_MAX) / 2,
             float(np.linalg.norm([(DEST_TRAY_X_MAX - DEST_TRAY_X_MIN) / 2, (DEST_TRAY_Y_MAX - DEST_TRAY_Y_MIN) / 2,
                                   (DEST_TRAY_Z_MAX - DEST_TRAY_Z_MIN) / 2])) + tn.TRAY_HULL_PAD_M)
REACH_M = 0.80  # the W leg starts only with its target this near the shoulder
SHOULDER_Z = 0.333
DEBUG = False


def rel(a, b):
    """Planar pose b in the frame of pose a."""
    return compose(invert(a), b)


class Leg:
    """A reference leg in cylindrical coordinates round its frame's origin (as task_node
    plans in the arm frame), Ruckig from a start state to a target at rest; frame 'arm'
    (moves with the base) or 'W' (the station)."""

    def __init__(self, frame, p0, v0, target, psi0, psi1, q7):
        self.frame = frame
        self.target = np.asarray(target, dtype=float)
        self.p0 = np.asarray(p0, dtype=float)
        c0, cv0, ca0 = tn._cart_to_cyl(self.p0, v0, np.zeros(3), 0.0)
        c1, _, _ = tn._cart_to_cyl(self.target, np.zeros(3), np.zeros(3), 0.0)
        # The heading turns the way that keeps joint 7 (~ joint 1 - heading - 135 deg) in range.
        w = wrap(psi1 - psi0)
        self.dpsi = min((w, w + 2 * np.pi, w - 2 * np.pi), key=lambda dp: abs(q7 + (c1[0] - c0[0]) - dp))
        self.psi0, self.psi1 = psi0, psi0 + self.dpsi
        r_lim = max(c0[1], c1[1], tn.MIN_PLAN_RADIUS_M)
        self.inp = InputParameter(3)
        self.inp.current_position, self.inp.current_velocity, self.inp.current_acceleration = c0, cv0, [0.0] * 3
        self.inp.target_position = c1
        lim = np.array([r_lim, 1.0, 1.0])
        self.inp.max_velocity = (tn.MAX_VEL / lim).tolist()
        self.inp.max_acceleration = (tn.MAX_ACCEL / lim).tolist()
        self.inp.max_jerk = (tn.MAX_JERK / lim).tolist()
        self.otg = Ruckig(3, DT)
        self.out = OutputParameter(3)
        self.done = False
        self.total = float(np.linalg.norm(self.target - self.p0))

    def step(self):
        if not self.done:
            res = self.otg.update(self.inp, self.out)
            self.out.pass_to_input(self.inp)
            self.done = res == Result.Finished

    def horizon(self, n):
        traj = Trajectory(3)
        Ruckig(3, DT).calculate(self.inp, traj)
        out = []
        for k in range(n + 1):
            pos, _, _ = traj.at_time(min(k * DT, traj.duration))
            out.append(tn._cyl_to_cart(pos, np.zeros(3), np.zeros(3))[0])
        return np.array(out)

    def psi_at(self, p):
        f = 1.0 if self.total < 1e-6 else float(np.clip(1 - np.linalg.norm(self.target - p) / self.total, 0, 1))
        return self.psi0 + self.dpsi * f


def orient(psi):
    return np.array([0.0, 0.0, -1.0, np.cos(psi), np.sin(psi), 0.0, 1.0])


def run(name, strategy, args, arm_model, solvers, rng_seed=0):
    sc = SCENARIOS[name]
    dock_arm = np.array(PICK_DOCK_ARM if sc["station"] == "pick" else PLACE_DOCK_ARM, dtype=float)
    dock_base = np.array(PICK_DOCK_BASE if sc["station"] == "pick" else PLACE_DOCK_BASE, dtype=float)
    m = load_scene_model(layout="stations")
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key(HOME_KEYFRAME).id)
    frames.free_bodies_at_rest(m, d)
    start_base = behind(dock_base, PRE_DOCK_M) if sc["into"] else dock_base
    fa = m.joint("base_free").qposadr[0]
    d.qpos[fa:fa + 3] = (start_base[0], start_base[1], 0.0)
    d.qpos[fa + 3:fa + 7] = (np.cos(start_base[2] / 2), 0, 0, np.sin(start_base[2] / 2))
    qa = [m.joint(n).qposadr[0] for n in ARM_JOINT_NAMES]
    va = [m.joint(n).dofadr[0] for n in ARM_JOINT_NAMES]
    ai = [m.actuator(n).id for n in ARM_ACTUATOR_NAMES]
    fd = m.joint("base_free").dofadr[0]
    d.qpos[qa] = ARM_CARRY_Q
    mujoco.mj_forward(m, d)
    if not sc["into"]:
        ik_arm(m, d, qa, sc["start"], sc["psi"])
    held = hold_largest_box(m, d) if sc["box"] else None
    payload = float(m.body_mass[m.body(held).id]) if held else 0.0
    base = BaseDrive(m, d, np.random.default_rng(rng_seed), noise=False)
    for i in range(1, 7):
        d.mocap_pos[m.body_mocapid[m.body(f"person_{i}").id]] = (0.0, 0.0, -10.0)
    mujoco.mj_forward(m, d)
    robot = robot_geoms(m, held)
    station = station_geoms(m, held)
    fk = forward_kinematics(arm_model, "tcp_site")
    pdata = arm_model.model.createData()

    def truth():
        b = base.truth(d)
        return np.array([b[0], b[1], 2 * np.arctan2(b[6], b[3])])

    def twist():
        tp = truth()
        vel = d.qvel[fd:fd + 6]
        return float(np.cos(tp[2]) * vel[0] + np.sin(tp[2]) * vel[1]), float(vel[5])

    def arm_in_w(pose_room):
        """The arm frame's pose in W, from base_link's room pose."""
        return np.array(rel(dock_arm, compose(pose_room, ARM_IN_BASE)))

    # Base reference: the approach to the dock, or straight back to the pre-dock distance.
    if sc["into"]:
        base_ref = RouteReference(line(start_base, dock_base, dock_base[2]), v_cruise=DOCK_SPEED)
    else:
        base_ref = RouteReference(line(dock_base, behind(dock_base, PRE_DOCK_M), dock_base[2]), v_cruise=UNDOCK_SPEED)
    hulls = [] if sc["into"] else [pile_hull(m, d, dock_arm, held) if sc["station"] == "pick" else TRAY_HULL]
    hull_grown = [0.0] * len(hulls)

    # Arm legs.
    tcp0 = np.array(fk(d.qpos[qa])).flatten()
    if sc["into"]:
        target_w = np.array(sc["target"])
        legs = [("arm", np.array([CARRY[0], CARRY[1], target_w[2]]), CARRY_PSI),
                ("arm" if strategy == "overlap" else "W", target_w, sc["psi"])]
    else:
        legs = [("arm", np.array([CARRY[0], CARRY[1], max(tcp0[2], CARRY[2])]), CARRY_PSI), ("arm", CARRY, CARRY_PSI)]
    leg_i, leg = -1, None
    psi_now = sc["psi"] if not sc["into"] else CARRY_PSI
    hold_goal = ("arm", tcp0.copy(), psi_now)  # what the arm holds between legs

    base_on = strategy != "seq" or sc["into"]  # seq out: the arm first
    arm_on = strategy != "seq" or not sc["into"]
    base_done = arm_done = False
    base_settle = arm_settle = 0
    cmd = (0.0, 0.0)
    base_pred = None  # (t0, (N+1, 5) states, dt) of the base MPC
    t = 0.0
    log = dict(track=[], clear=np.inf, contacts=0, solve=[], fails=0, dock=None, t_base=None, t_arm=None)
    bmpc, ampc, wb = solvers["base"], solvers["arm"], solvers["wb"]
    bmpc.warm = False
    cfg = ampc[1]
    guess_arm = True
    while t < TIMEOUT_S:
        pose = truth()
        v_meas, w_meas = twist()
        aw = arm_in_w(pose)
        # Leg sequencing.
        if arm_on and not arm_done and (leg is None or leg.done):
            nxt = leg_i + 1
            if nxt < len(legs):
                frame, tgt, psi1 = legs[nxt]
                remaining = np.hypot(*(pose[:2] - dock_base[:2])) if sc["into"] else 0.0
                ok = True
                if nxt == len(legs) - 1 and sc["into"]:
                    tgt_arm = np.array(compose(invert(aw), (tgt[0], tgt[1], 0.0))[:2]) if frame == "W" else tgt[:2]
                    reach = np.hypot(np.hypot(*tgt_arm), tgt[2] - SHOULDER_Z)
                    ok = (base_done or remaining <= args.start_m) and reach <= REACH_M
                if ok:
                    p_ref = hold_goal[1] if leg is None else leg.target
                    p0, v0, psi0 = p_ref, np.zeros(3), psi_now
                    if (hold_goal[0] if leg is None else leg.frame) != frame:
                        # An arm-frame point into W, moving there with the base.
                        p0, v0 = point_in_w(aw, base_in_w(pose, dock_arm), p_ref, v_meas, w_meas)
                        psi0 = psi_now + aw[2]
                    leg_i, leg = nxt, Leg(frame, p0, v0, tgt, psi0, psi1, d.qpos[qa[6]])
        # The reference horizon (in the leg's frame) and what the arm aims at.
        if leg is not None:
            leg.step()
            horizon = leg.horizon(cfg.N)
            frame = leg.frame
            psis = [leg.psi_at(p) for p in horizon]
            psi_now = psis[0]
        else:
            frame, horizon, psis = hold_goal[0], np.tile(hold_goal[1], (cfg.N + 1, 1)), [hold_goal[2]] * (cfg.N + 1)
        final = legs[-1]
        at_final = leg is not None and leg_i == len(legs) - 1 and leg.done

        # Controllers.
        x_arm = np.r_[d.qpos[qa], d.qvel[va]]
        if strategy == "wb" and not (base_done and arm_done):
            x0 = np.r_[base_in_w(pose, dock_arm), v_meas, w_meas, x_arm]
            if guess_arm:
                wb.guess(x0, pin.computeGeneralizedGravity(arm_model.model, pdata, x_arm[:7]))
                guess_arm = False
            base_ref.update(pose)
            bref = base_ref.horizon(cfg.N, DT)
            bref_w = np.array([[*base_in_w(r[:3], dock_arm), r[3], r[4]] for r in bref])
            if not base_on or base_done:
                bref_w = np.tile([*base_in_w(pose, dock_arm), 0.0, 0.0], (cfg.N + 1, 1))
            route = (np.cos(bref_w[-1, 2]), np.sin(bref_w[-1, 2]))
            obst = obstacle_slots(hulls, hull_grown, None, cfg)
            for k in range(cfg.N + 1):
                g = horizon[k]
                aw_k = arm_in_w_from_base(bref_w[k, :3])
                g_arm = np.array(compose(invert(aw_k), (g[0], g[1], 0.0))[:2]) if frame == "W" else g[:2]
                p = wb.params(obstacles=obst, goal=g, orient=orient(psis[k]), payload=payload,
                              sel=1.0 if frame == "W" else 0.0, route=route)
                az = float(np.arctan2(g_arm[1], g_arm[0])) if np.hypot(*g_arm) > 0.15 else None
                wb.set_stage(k, p, bref_w[k], az)
            t0 = time.perf_counter()
            st = wb.solve(x0)
            log["solve"].append((time.perf_counter() - t0) * 1e3)
            if st not in (0, 2):
                log["fails"] += 1
                guess_arm = True
                tau = pin.computeGeneralizedGravity(arm_model.model, pdata, x_arm[:7])
                cmd = (0.0, 0.0)
            else:
                u0 = wb.solver.get(0, "u")
                x1 = wb.solver.get(1, "x")
                tau = u0[2:]
                cmd = (float(x1[3]), float(x1[4])) if base_on and not base_done else (0.0, 0.0)
            d.ctrl[ai] = tau
        else:
            # Base: its MPC at 10 Hz along the docking line, as the navigator runs it.
            if base_on and not base_done and round(t / DT) % NAV_EVERY == 0:
                base_ref.update(pose)
                bc = bmpc.cfg
                ref = base_ref.horizon(bc.N, bc.dt)
                ppl = np.tile(NO_PERSON, (bc.N + 1, bc.n_people, 1)).astype(float)
                st, xs = bmpc.solve([*pose, v_meas, w_meas], ref, ppl)
                cmd = (float(xs[1, 3]), float(xs[1, 4])) if st in (0, 2) else (0.0, 0.0)
                base_pred = (t, xs, bc.dt)
            if base_done or not base_on:
                cmd = (0.0, 0.0)
            # Arm: its own MPC, goals in the arm frame along the base's predicted motion.
            solver = ampc[0]
            if guess_arm:
                g0 = pin.computeGeneralizedGravity(arm_model.model, pdata, x_arm[:7])
                for k in range(cfg.N + 1):
                    solver.set(k, "x", x_arm)
                for k in range(cfg.N):
                    solver.set(k, "u", g0)
                guess_arm = False
            solver.set(0, "lbx", x_arm)
            solver.set(0, "ubx", x_arm)
            yref = np.zeros(7 + 7 + 7 + 9)
            for k in range(cfg.N + 1):
                aw_k = aw if base_pred is None or not (base_on and not base_done) else arm_in_w(
                    compose(pose, rel(predict(base_pred, t, pose), predict(base_pred, t + k * DT, pose))))
                g = horizon[k]
                if frame == "W":
                    xy = compose(invert(aw_k), (g[0], g[1], 0.0))
                    g_arm = np.array([xy[0], xy[1], g[2]])
                    psi_arm = psis[k] - aw_k[2]
                else:
                    g_arm, psi_arm = g, psis[k]
                obst = obstacle_slots(hulls, hull_grown, aw, cfg)
                solver.set(k, "p", np.r_[obst, g_arm, orient(psi_arm), payload])
                if k < cfg.N and np.hypot(*g_arm[:2]) > 0.15:
                    yref[14] = np.arctan2(g_arm[1], g_arm[0]) - ampc[2][0]
                    solver.cost_set(k, "yref", yref)
            t0 = time.perf_counter()
            st = solver.solve()
            log["solve"].append((time.perf_counter() - t0) * 1e3)
            if st != 0:
                log["fails"] += 1
                guess_arm = True
                d.ctrl[ai] = pin.computeGeneralizedGravity(arm_model.model, pdata, x_arm[:7])
            else:
                d.ctrl[ai] = solver.get(0, "u")

        # Hulls grow in behind the arm (as task_node), never over its clearance.
        update_hulls(hulls, hull_grown, m, d, aw)

        # Plant.
        base.set_command(wheel_speeds(*cmd, BASE_WHEEL_RADIUS_M, BASE_WHEEL_TRACK_M, WHEEL_SPEED_MAX), d.time)
        base.update(d)
        base.step(m, d, 10)
        t += DT

        # Truth and completion.
        pose = truth()
        v_meas, w_meas = twist()
        aw = arm_in_w(pose)
        tcp = np.array(fk(d.qpos[qa])).flatten()
        ref0 = horizon[1] if len(horizon) > 1 else horizon[0]
        if frame == "W":
            c, s = np.cos(aw[2]), np.sin(aw[2])
            tcp_f = np.array([aw[0] + c * tcp[0] - s * tcp[1], aw[1] + s * tcp[0] + c * tcp[1], tcp[2]])
        else:
            tcp_f = tcp
        log["track"].append(float(np.linalg.norm(tcp_f - ref0)))
        if DEBUG and round(t / DT) % 25 == 0:
            print(f"  t {t:5.2f} leg {leg_i} {frame} base {np.round(base_in_w(pose, dock_arm), 3)} v {v_meas:+.2f} "
                  f"tcp {np.round(tcp_f, 3)} ref {np.round(ref0, 3)} q {np.round(d.qpos[qa], 2)}")
        if round(t / DT) % 5 == 0:
            log["clear"] = min(log["clear"], clearance(m, d, robot, station))
        log["contacts"] += contacts(d, robot, station)
        if base_on and not base_done:
            g = base_ref.path[-1]
            close = (np.hypot(*(pose[:2] - g[:2])) < max(DOCK_TOL_ALONG_M, 0.01) and abs(wrap(pose[2] - g[2])) <
                     np.radians(4))
            at_end = base_ref.progress >= len(base_ref.path) - 2
            still = abs(v_meas) < STILL_V and abs(w_meas) < STILL_W
            base_settle = base_settle + 1 if (close or at_end) and still else 0
            if base_settle >= SETTLE_TICKS:
                base_done, log["t_base"] = True, t
                if sc["into"]:
                    c, s = np.cos(dock_base[2]), np.sin(dock_base[2])
                    dx, dy = pose[0] - dock_base[0], pose[1] - dock_base[1]
                    log["dock"] = (c * dx + s * dy, -s * dx + c * dy, wrap(pose[2] - dock_base[2]))
                if strategy == "seq" and sc["into"]:
                    arm_on = True
        if arm_on and not arm_done and at_final:
            fin = final[1]
            if final[0] == "W":
                c, s = np.cos(aw[2]), np.sin(aw[2])
                tcp_w = np.array([aw[0] + c * tcp[0] - s * tcp[1], aw[1] + s * tcp[0] + c * tcp[1], tcp[2]])
                err = np.linalg.norm(tcp_w - fin)
            else:
                err = np.linalg.norm(tcp - fin)
            arm_settle = arm_settle + 1 if err < ARM_TOL_M and np.abs(d.qvel[va]).max() < ARM_STILL else 0
            if arm_settle >= SETTLE_TICKS:
                arm_done, log["t_arm"] = True, t
                hold_goal = (final[0], fin, final[2])
                if strategy == "seq" and not sc["into"]:
                    base_on = True
        if base_done and arm_done:
            break
        if leg is not None and leg.done and leg_i == len(legs) - 1:
            hold_goal = (leg.frame, leg.target, leg.psi1)
    log["t"] = t
    log["done"] = base_done and arm_done
    return log


def base_in_w(pose_room, dock_arm):
    return np.array(rel(dock_arm, pose_room))


def point_in_w(aw, bw, p_arm, v, w):
    """An arm-frame point in W (the arm frame at aw in W, base_link at bw) and its
    velocity there, carried by the base (v forward, w turning)."""
    c, s = np.cos(aw[2]), np.sin(aw[2])
    p = np.array([aw[0] + c * p_arm[0] - s * p_arm[1], aw[1] + s * p_arm[0] + c * p_arm[1], p_arm[2]])
    r = p[:2] - bw[:2]
    vel = np.array([v * np.cos(bw[2]) - w * r[1], v * np.sin(bw[2]) + w * r[0], 0.0])
    return p, vel


def pile_hull(m, d, dock_arm, held):
    """The pile's hull as task_node makes it from a scan: round the boxes' tops (W), out
    to the farthest plus the largest box's half-diagonal and the pad."""
    pts = []
    for i in range(8):
        name = f"cbox_{i}"
        if name == held:
            continue
        b = m.body(name).id
        x, y, _ = rel(dock_arm, (d.xpos[b][0], d.xpos[b][1], 0.0))
        g = [k for k in range(m.ngeom) if m.geom_bodyid[k] == b][0]
        top = d.xpos[b][2] + m.geom_size[g][2] - 0.75
        pts.append((x, y, top))
    pts = np.array(pts)
    c = pts.mean(axis=0)
    return (*c, float(np.max(np.linalg.norm(pts - c, axis=1))) + tn.BOX_HALF_DIAGONAL_MAX_M + tn._HULL_PAD)


def arm_in_w_from_base(b):
    return np.array(compose(b, ARM_IN_BASE))


def predict(base_pred, t, pose_now):
    """The base's room pose at time t from its MPC's last prediction."""
    t0, xs, dt = base_pred
    f = (t - t0) / dt
    i = int(np.clip(np.floor(f), 0, len(xs) - 2))
    a = float(np.clip(f - i, 0.0, 1.0))
    p = xs[i, :3] + a * (xs[i + 1, :3] - xs[i, :3])
    return np.array([p[0], p[1], p[2]])


def obstacle_slots(hulls, grown, aw, cfg):
    """The hulls as the OCP's slots: in W (aw None) or moved into the arm frame aw."""
    slots = []
    for (x, y, z, _), r in zip(hulls, grown):
        if r <= 0.0:
            slots += [*mcn.NO_OBSTACLE_POSITION, mcn.NO_OBSTACLE_RADIUS]
            continue
        if aw is not None:
            x, y, _ = compose(invert(aw), (x, y, 0.0))
        slots += [x, y, z, r]
    while len(slots) < 4 * cfg.n_obstacles:
        slots += [*mcn.NO_OBSTACLE_POSITION, mcn.NO_OBSTACLE_RADIUS]
    return np.array(slots, dtype=float)


def update_hulls(hulls, grown, m, d, aw):
    """task_node's grow-in: a hull's radius never exceeds the proxies' clearance from its
    centre (W), and never shrinks."""
    if not hulls:
        return
    pos, rot = frames.arm_base_pose(m, d)
    proxies = []
    for name, r in tn.PROXY_SPHERES:
        b = m.body(name).id
        p_arm = frames.to_arm((pos, rot), d.xpos[b])
        c, s = np.cos(aw[2]), np.sin(aw[2])
        proxies.append((np.array([aw[0] + c * p_arm[0] - s * p_arm[1], aw[1] + s * p_arm[0] + c * p_arm[1],
                                  p_arm[2]]), r))
    for i, h in enumerate(hulls):
        c = np.array(h[:3])
        clear = min(float(np.linalg.norm(p - c)) - r for p, r in proxies) - tn.OCP_SAFETY_MARGIN_M - tn.HULL_GROW_PAD_M
        grown[i] = max(grown[i], min(h[3], clear))


def ik_arm(m, d, qa, pos_arm, psi):
    """The arm at a TCP position (arm frame) with the tool down at heading psi (arm frame),
    by damped least squares (as wrist_pose.place_tcp, in the arm frame)."""
    va = np.array([m.joint(n).dofadr[0] for n in ARM_JOINT_NAMES])
    lo, hi = m.jnt_range[[m.joint(n).id for n in ARM_JOINT_NAMES]].T
    home = m.key(HOME_KEYFRAME).qpos[qa].copy()
    home[0] = np.arctan2(pos_arm[1], pos_arm[0])
    home[6] = np.clip((home[0] - psi - np.radians(135) + np.pi) % (2 * np.pi) - np.pi, lo[6], hi[6])
    d.qpos[qa] = home
    site = m.site("tcp_site").id
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    for _ in range(400):
        mujoco.mj_forward(m, d)
        arm = frames.arm_base_pose(m, d)
        pos, target = frames.to_room(arm, pos_arm, tool_down_rot(psi))
        r = d.site_xmat[site].reshape(3, 3)
        e = np.r_[pos - d.site_xpos[site], 0.5 * sum(np.cross(r[:, i], target[:, i]) for i in range(3))]
        if np.linalg.norm(e) < 1e-7:
            break
        mujoco.mj_jacSite(m, d, jp, jr, site)
        j = np.vstack([jp, jr])[:, va]
        jinv = j.T @ np.linalg.inv(j @ j.T + 1e-4 * np.eye(6))
        dq = jinv @ e + (np.eye(7) - jinv @ j) @ (0.05 * (home - d.qpos[qa]))
        d.qpos[qa] = np.clip(d.qpos[qa] + dq, lo, hi)
    mujoco.mj_forward(m, d)


def robot_geoms(m, held):
    out = [g for g in range(m.ngeom) if (m.geom_contype[g] or m.geom_conaffinity[g])
           and frames.in_arm(m, m.geom_bodyid[g])]
    if held:
        out += [g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body(held).id]
    return out


def station_geoms(m, held):
    """Everything the arm could touch but its own robot and a held box (the station's
    table, tray, pile)."""
    base_root = m.body_rootid[m.body("base_link").id]
    out = []
    for g in range(m.ngeom):
        if not (m.geom_contype[g] or m.geom_conaffinity[g]) or m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        b = m.geom_bodyid[g]
        name = m.body(b).name
        if b and m.body_rootid[b] == base_root or name.startswith("person") or name == held:
            continue
        out.append(g)
    return out


def clearance(m, d, robot, station):
    best = np.inf
    ft = np.zeros(6)
    for rg in robot:
        for sg in station:
            if np.linalg.norm(d.geom_xpos[rg][:2] - d.geom_xpos[sg][:2]) > 1.5:
                continue
            best = min(best, mujoco.mj_geomDistance(m, d, rg, sg, 0.5, ft))
    return best


def contacts(d, robot, station):
    rs, ss = set(robot), set(station)
    n = 0
    for i in range(d.ncon):
        c = d.contact[i]
        if (c.geom1 in rs and c.geom2 in ss) or (c.geom2 in rs and c.geom1 in ss):
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="all", choices=["all", *SCENARIOS])
    ap.add_argument("--strategy", default="all", choices=["all", "seq", "overlap", "split", "wb"])
    ap.add_argument("--start-m", type=float, default=0.3)
    ap.add_argument("--build", default=".")
    args = ap.parse_args()
    arm_model = load_manipulator(mcn.MJCF_PATH, [p.frame_name for p in mcn.PROXY_FRAMES])
    cfg = MPCConfig()
    q_center = (arm_model.model.lowerPositionLimit + arm_model.model.upperPositionLimit) / 2
    solvers = {"arm": (build_ocp(arm_model, mcn.PROXY_FRAMES, cfg), cfg, q_center),
               "base": BaseMPC(build_dir=args.build),
               "wb": WholeBodyMPC(arm_model, mcn.PROXY_FRAMES, WholeBodyConfig(), build_dir=args.build)}
    names = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    strategies = ["seq", "overlap", "split", "wb"] if args.strategy == "all" else [args.strategy]
    for name in names:
        for strat in strategies:
            r = run(name, strat, args, arm_model, solvers)
            tr = 1e3 * np.array(r["track"])
            dock = ("-" if r["dock"] is None else
                    f"{1e3 * r['dock'][0]:+.0f}/{1e3 * r['dock'][1]:+.0f} mm {np.degrees(r['dock'][2]):+.2f} deg")
            s = np.array(r["solve"])
            print(f"{name:9s} {strat:5s}: {'done' if r['done'] else 'NOT DONE'} in {r['t']:5.1f} s (base "
                  f"{r['t_base'] or float('nan'):4.1f}, arm {r['t_arm'] or float('nan'):4.1f}); dock {dock}; tool off "
                  f"reference p95 {np.percentile(tr, 95):4.1f} max {tr.max():5.1f} mm; station clearance "
                  f"{1e3 * r['clear']:4.0f} mm, contacts {r['contacts']}; solve p50 {np.median(s):.1f} max "
                  f"{s.max():.1f} ms, failures {r['fails']}")


if __name__ == "__main__":
    main()
