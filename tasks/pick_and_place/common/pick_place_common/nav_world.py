"""The navigation's world offline, no ROS: scripts/dev/nav_sim.py's loop as a class, for
nav_sim and for training learned policies on the method's own code. The plant (the base's
drives and sensors, the seed's mismatch, the arm held, a crowd) and the method's whole
chain (odometry, localization in the saved map, people on the map with tracks, the
navigator with its planner and MPC, the safety layer), scored on the truth
(nav_scoring.py). tick() is one plant step and what the method does in it; start(goal)
and finish() bracket a drive, drive(goal) runs one. fast: only the estimator that drives
is run and scored (the other is a shadow, known_issues U4); the drive is unchanged.

See docs/implementation_notes.md#learned-policies-on-top-the-navigations-world-rl-layer-1-step-1.
"""
import hashlib
import time
from collections import deque
from pathlib import Path

import mujoco
import numpy as np

from core.base_odometry import BaseOdometry, wheel_speeds
from nav import docks as nav_docks, safety
from nav.navigator import Navigator
from perception.map_people import MapPeopleDetector
from perception.people_tracker import PeopleTracker
from pick_place_common import frames, lidar_sim, plant_conditions
from pick_place_common.base_drive import BASE_BODY, WHEEL_JOINTS, BaseDrive
from pick_place_common.body_clearance import BodyClearance
from pick_place_common.loc_scoring import LocScore
from pick_place_common.mobile_scenarios import Crowd, room_obstacles
from pick_place_common.mujoco_sim_node import (
    ARM_ACTUATOR_NAMES, ARM_JOINT_NAMES, HOLD_KD, HOLD_KP, HOME_KEYFRAME, load_scene_model)
from pick_place_common.nav_scoring import DriveScore
from pick_place_common.plant_mismatch import apply_base_mismatch, apply_plant_mismatch, refresh
from pick_place_common.scene import (
    ARM_CARRY_Q, ARM_TUCKED_Q, BASE_HOME_POSE, BASE_WHEEL_RADIUS_M, BASE_WHEEL_TRACK_M, LIDAR_MOUNTS_BASE,
    LIDAR_RATE_HZ, LIDAR_SCANNERS, PICK_DOCK_BASE, PLACE_DOCK_BASE, room_to_map)
from slam.base_ekf import BaseEKF
from slam.grid import OccupancyGrid
from slam.localizer import GridLocalizer
from slam.pose_graph import compose, relative, wrap
from slam.scan import scan_points

DT = 0.02
NAV_EVERY = 5  # ticks: the MPC at 10 Hz
DRIVE_TIMEOUT_S = 120.0
WHEEL_SPEED_MAX = 15.0
LOC_STALE_S = 0.5  # safety_node's localization watchdog: no pose this long, stop
WET_STOP_SPEED, WET_STOP_S = 0.3, 1.0
SAFETY_CODES = {"clear": 0, "warn": 1, "stop": 2}
DOCKS_ROOM = {"pick": PICK_DOCK_BASE, "place": PLACE_DOCK_BASE}


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


class NavWorld:
    def __init__(self, map_path, seed=0, crowd=(), arm="tucked", box=False, speed=1.0, accel=0.5, base_safety=1.0,
                 stop_safety=None, conditions=(), fusion="icp", scan_latency=0.02, person_margin=True, teach=False,
                 fast=False, record=False):
        """map_path: the saved map without its suffix; crowd: the people's specs
        (mobile_scenarios, walking speeds applied), or a function drawing them from the
        room's obstacles (random_crowd); arm: tucked | carry, box: the largest
        held; base_safety: the navigator's distances to people x this; stop_safety: the
        safety layer's person margin x this (default: base_safety); fusion: the estimator
        that drives (icp | ekf); teach: steer on the truth, the docks are the technician's
        spots (commissioning); record: keep the per-tick trajectory and estimates (plots)."""
        self.seed, self.fusion, self.teach, self.fast, self.record = seed, fusion, teach, fast, record
        self.conditions = set(conditions)
        self.person_margin = person_margin
        m = self.m = load_scene_model(layout="stations")
        d = self.d = mujoco.MjData(m)
        apply_plant_mismatch(m, np.random.default_rng(seed), ARM_JOINT_NAMES, [f"cbox_{i}" for i in range(8)])
        apply_base_mismatch(m, np.random.default_rng([seed, 2]), WHEEL_JOINTS, BASE_BODY,
                            worn=plant_conditions.WORN_TYRE if "worn_tyre" in self.conditions else 0.0)
        refresh(m, d)
        mujoco.mj_resetDataKeyframe(m, d, m.key(HOME_KEYFRAME).id)
        frames.free_bodies_at_rest(m, d)
        self.qa = [m.joint(n).qposadr[0] for n in ARM_JOINT_NAMES]
        self.va = [m.joint(n).dofadr[0] for n in ARM_JOINT_NAMES]
        self.ai = [m.actuator(n).id for n in ARM_ACTUATOR_NAMES]
        d.qpos[self.qa] = ARM_CARRY_Q if arm == "carry" else ARM_TUCKED_Q
        mujoco.mj_forward(m, d)
        self.q_hold = d.qpos[self.qa].copy()
        held = hold_largest_box(m, d) if box else None
        self.base = BaseDrive(m, d, np.random.default_rng([seed, 3]), conditions=self.conditions)
        self.odo = BaseOdometry(BASE_WHEEL_RADIUS_M, BASE_WHEEL_TRACK_M)
        self.lidar_rng = np.random.default_rng([seed, 1])
        self.rects = room_obstacles(m)  # tables, trays, fixtures: footprints in the room
        self.crowd = Crowd(crowd(self.rects) if callable(crowd) else list(crowd))
        self.dropouts = (plant_conditions.Dropouts(np.random.default_rng([seed, 5]))
                         if "dropout" in self.conditions else None)
        self.bodies = BodyClearance(m, extra_bodies=[held] if held else ()) if arm == "carry" else None
        self.crowd_mocap = [m.body_mocapid[m.body(s[0]).id] for s in self.crowd.specs]
        self.fd = m.joint("base_free").dofadr[0]

        self.map_path = Path(map_path)
        self.grid = OccupancyGrid.load(self.map_path.with_suffix(".yaml"))
        self.targets = {name: room_to_map(p) for name, p in DOCKS_ROOM.items()}  # the technician's spots
        docks = self.targets if teach else nav_docks.load(self.map_path)
        self.taught = {}
        self.nav = Navigator(self.grid, docks, room_to_map(BASE_HOME_POSE), speed=speed, accel=accel,
                             safety=base_safety)
        self.extra = safety.person_extra(base_safety if stop_safety is None else stop_safety)
        self.loc = GridLocalizer(self.grid, (0.0, 0.0, 0.0))  # starts at home, the map's origin
        self.ekf = BaseEKF((0.0, 0.0, 0.0), BASE_WHEEL_RADIUS_M, BASE_WHEEL_TRACK_M)
        self.det = MapPeopleDetector(self.grid.occupied_points(), LIDAR_MOUNTS_BASE, lidar_sim.BEAM_ANGLES)
        self.trk = PeopleTracker()

        self.k, self.next_scan = 0, 0.0
        self.map_to_odom = np.zeros(3)
        self.loc_t = 0.0
        self.scores = {"icp": LocScore(), "ekf": LocScore()}
        self.loc_ms, self.fitness, self.dropped = [], [], 0
        self.ekf_tick_ms, self.ekf_scan_ms = [], []
        self.latency = int(round(scan_latency / DT))
        self.queue = deque()  # (due tick, capture time, points, odometry then)
        self.dump, self.traj = [], []
        self.slip, self.prev_tp, self.prev_enc, self.wet_stops = [], None, None, 0
        self.last_points = np.zeros((0, 2))
        self.person_points = np.zeros((0, 2))
        self.tracks = []
        self.cmd = (0.0, 0.0)
        self.drives = []
        self.state_time = {}
        self.digest = hashlib.sha256()
        self.score = None
        self._settle()

    def truth(self):
        b = self.base.truth(self.d)
        return np.array([b[0], b[1], 2 * np.arctan2(b[6], b[3])])

    def _hold_arm(self):
        d = self.d
        d.ctrl[self.ai] = d.qfrc_bias[self.va] + HOLD_KP * (self.q_hold - d.qpos[self.qa]) - HOLD_KD * d.qvel[self.va]

    def _settle(self):
        base, d = self.base, self.d
        for i in range(50):  # the gyro bias is learnt
            base.set_command((0.0, 0.0), d.time)
            base.update(d)
            self._hold_arm()
            base.step(self.m, d, 10)
            gyro, _ = base.imu(d, DT)
            self.odo.update(base.encoders(d), DT, gyro[2])
            self.ekf.tick((i - 49) * DT, base.encoders(d), gyro[2])

    def start(self, goal):
        """A new drive to goal (pick | place | home); the navigator gets it on the next tick."""
        self.goal = goal
        self.score = DriveScore(goal, self.k * DT)
        self.started = False
        self.last_state = "clear"
        self.wet_stop = -1.0 if "spill" in self.conditions else None  # its start, once it happened

    @property
    def done(self):
        """The drive is over: docked, parked, failed or timed out."""
        return self.nav.state in ("docked", "parked", "failed") or self.k * DT - self.score.r["t0"] > DRIVE_TIMEOUT_S

    def tick(self):
        """One plant step (DT) and what the method does in it."""
        m, d, base, nav, ekf, loc, score = self.m, self.d, self.base, self.nav, self.ekf, self.loc, self.score
        # Plant: the people, the base's drives, the arm's hold.
        here = self.here = self.crowd.step(DT, self.truth()[:2])
        for mid, (px, py, pyaw) in zip(self.crowd_mocap, here):
            d.mocap_pos[mid] = (px, py, 0.0)
            d.mocap_quat[mid] = (np.cos(pyaw / 2), 0.0, 0.0, np.sin(pyaw / 2))
        base.set_command(wheel_speeds(*self.cmd, BASE_WHEEL_RADIUS_M, BASE_WHEEL_TRACK_M, WHEEL_SPEED_MAX), d.time)
        base.update(d)
        self._hold_arm()
        base.step(m, d, 10)
        self.k += 1
        k = self.k
        gyro, _ = base.imu(d, DT)
        wheels = base.encoders(d)
        odom = self.odo.update(wheels, DT, gyro[2]).copy()
        t0 = time.perf_counter()
        ekf.tick(k * DT, wheels, gyro[2])
        self.ekf_tick_ms.append(1e3 * (time.perf_counter() - t0))
        # Method: scans at 15 Hz (localization, people), control at 10 Hz.
        scanned = k * DT >= self.next_scan
        if scanned:
            self.next_scan += 1.0 / LIDAR_RATE_HZ
            mujoco.mj_forward(m, d)
            scans = [lidar_sim.scan(m, d, i, self.lidar_rng) for i in range(len(LIDAR_SCANNERS))]
            self.last_points = scan_points(scans, lidar_sim.BEAM_ANGLES, LIDAR_MOUNTS_BASE)
            if self.dropouts is not None and self.dropouts.dropped(k * DT):
                self.dropped += 1
            else:
                self.queue.append((k + self.latency, k * DT, self.last_points, odom))
        while self.queue and self.queue[0][0] <= k:
            _due, t_cap, pts, odom_cap = self.queue.popleft()
            if not self.fast or self.fusion == "icp":
                t0 = time.perf_counter()
                est = loc.update(odom_cap, pts)
                self.loc_ms.append(1e3 * (time.perf_counter() - t0))
                self.fitness.append(loc.quality)
                self.map_to_odom = compose(est, relative(odom_cap, np.zeros(3)))
            self.loc_t = k * DT
            if not self.fast or self.fusion == "ekf":
                icp_s = []

                def match(prior, pts=pts):
                    t1 = time.perf_counter()
                    out = loc.match(pts, prior)
                    icp_s.append(time.perf_counter() - t1)
                    return out
                t0 = time.perf_counter()
                ekf.scan(t_cap, match)
                self.ekf_scan_ms.append(1e3 * (time.perf_counter() - t0 - sum(icp_s)))
        ests = {"icp": compose(self.map_to_odom, odom), "ekf": ekf.pose}
        pose = self.pose = np.array(room_to_map(self.truth())) if self.teach else ests[self.fusion]
        if scanned:
            self.tracks = self.trk.update(k * DT, self.det.update(pose, scans), self.det.seen_empty)
            c, s = np.cos(pose[2]), np.sin(pose[2])
            fg = self.det.foreground - pose[:2]
            self.person_points = np.column_stack([c * fg[:, 0] + s * fg[:, 1], -s * fg[:, 0] + c * fg[:, 1]])
        if not self.started:
            nav.set_goal(self.goal, pose)
            self.started = True
        if k % NAV_EVERY == 0:
            v, w = nav.step(k * DT, pose, self.odo.twist, [tr.x for tr in self.tracks])
            self.last_state, cap = safety.check(self.last_points, v, w, nav.docking, measured=self.odo.twist,
                                                people=self.person_points if self.person_margin else None,
                                                extra=self.extra)
            if k * DT - self.loc_t > LOC_STALE_S:
                self.last_state = "stop"
            if self.wet_stop is not None:
                tp = self.truth()
                if (self.wet_stop < 0.0 and plant_conditions.on_spill(tp[:2])
                        and abs(self.odo.twist[0]) > WET_STOP_SPEED):
                    self.wet_stop, self.wet_stops = k * DT, self.wet_stops + 1
                if self.wet_stop >= 0.0 and k * DT - self.wet_stop < WET_STOP_S:
                    self.last_state = "stop"
            self.cmd = safety.limit(v, w, self.last_state, cap)
            score.command(k * DT, *self.cmd, NAV_EVERY * DT, self.last_state, nav.state)
            self.state_time[nav.state] = self.state_time.get(nav.state, 0.0) + NAV_EVERY * DT
        # Truth: people against the chassis and its protective field.
        tp = self.truth()
        for name, sc in self.scores.items():
            if not self.fast or name == self.fusion:
                sc.add(ests[name], room_to_map(tp), ekf.cov if name == "ekf" else None)
        if self.record:
            self.dump.append(np.r_[k * DT, room_to_map(tp), ests["icp"], ests["ekf"], ekf.cov.ravel(),
                                   [ekf.counts[c] for c in ("slips", "skids", "rejected", "resets")],
                                   self.odo.twist[0]])
            self.traj.append(np.r_[k * DT, tp, pose, self.cmd])
        enc = base.encoders(d) * BASE_WHEEL_RADIUS_M
        if self.prev_tp is not None:  # the wheels' arc against the true travel along the heading
            along = np.cos(tp[2]) * (tp[0] - self.prev_tp[0]) + np.sin(tp[2]) * (tp[1] - self.prev_tp[1])
            self.slip.append(abs(along - 0.5 * np.sum(enc - self.prev_enc)))
        self.prev_tp, self.prev_enc = tp, enc
        vel = d.qvel[self.fd:self.fd + 6]
        v_body = float(np.cos(tp[2]) * vel[0] + np.sin(tp[2]) * vel[1])
        self.v_body, self.w_body = v_body, float(vel[5])
        score.truth(k * DT, tp, v_body, float(vel[5]), here, nav.docking, self.last_state, nav.state)
        if self.bodies is not None and k % NAV_EVERY == 0 and here:
            dist, part, who, p_robot, p_person = self.bodies.nearest(d)
            if np.isfinite(dist):
                w_world = d.xmat[m.body(BASE_BODY).id].reshape(3, 3) @ vel[3:6]
                toward = self.bodies.closing_speed(d, p_robot, p_person, vel[:3], w_world)
                moving = abs(v_body) > 0.05 or abs(vel[5]) > 0.1
                score.body(dist, toward, moving, f"{part} / {who}")
        self.digest.update(np.r_[k * DT, tp, pose, self.cmd, SAFETY_CODES[self.last_state], np.ravel(here)].tobytes())

    def finish(self):
        """Closes the drive: its arrival against the true dock, the navigator's counts; its
        scores (DriveScore)."""
        nav, score, rec = self.nav, self.score, self.score.r
        if nav.planned is not None:
            score.plan(*nav.planned)
        score.events(nav.counts)
        tp = self.truth()
        rec["t"] = self.k * DT - rec["t0"]
        rec["state"] = nav.state if self.k * DT - rec["t0"] <= DRIVE_TIMEOUT_S else "timeout"
        dock = DOCKS_ROOM.get(self.goal, BASE_HOME_POSE)
        c, s = np.cos(dock[2]), np.sin(dock[2])
        dx, dy = tp[0] - dock[0], tp[1] - dock[1]
        rec["err"] = (c * dx + s * dy, -s * dx + c * dy, np.degrees(wrap(tp[2] - dock[2])))
        rec["retries"] = nav.retries
        if self.teach and self.goal in self.targets and rec["state"] == "docked":
            self.taught[self.goal] = tuple(float(v) for v in self.ekf.pose)
        self.drives.append(score)
        return score

    def drive(self, goal):
        """One drive to goal, to its end; its scores."""
        self.start(goal)
        while True:
            self.tick()
            if self.done:
                return self.finish()
