"""The plant: owns the MuJoCo simulation and exposes it over ROS 2 topics
(joint states, torque commands, the wrist load cell, the grasp weld, the mobile
base's wheel drives, encoders and IMU, and the scene state that
sim_sensors_node renders the cameras and windows from). MuJoCo's world is the
room; what goes to the method is in the arm's frame or the sensors' own.

See docs/implementation_notes.md#mujoco_sim_nodepy.
"""
import os
import re
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import mujoco
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu, JointState
from geometry_msgs.msg import Vector3
from std_msgs.msg import Float64MultiArray, String

from pick_place_common import frames, plant_conditions
from pick_place_common.base_drive import BASE_BODY, WHEEL_JOINTS, BaseDrive, has_base
from pick_place_common.plant_mismatch import apply_base_mismatch, apply_plant_mismatch, refresh
from pick_place_common.scene import (
    ARM_CARRY_Q, ARM_TUCKED_Q, BASE_HOME_POSE, BASE_IN_ARM, BASE_PARK_POSE, CELL_POSE, PICK_STATION_POSE, PLACE_STATION_POSE,
    ROOM_FLOOR_Z, compose, invert)
from pick_place_common.telemetry import Telemetry

CONTROL_PERIOD_S = 0.02  # must match MPCConfig.dt
COMMAND_STALE_STEPS = 10  # hold if the command is from an older state than this
# Lockstep with mpc_controller (docs/design_notes.md): each step waits up to
# LOCKSTEP_MAX_WAIT_S for the command computed from the last published state,
# paced to real time, while the controller is live.
LOCKSTEP_MAX_WAIT_S = 0.05
LOCKSTEP_LIVE_STEPS = 3
LOCKSTEP_POLL_S = 0.002
LOCKSTEP_STALL_S = 0.01  # a longer gap between polls means this thread was busy (a render)
LOCKSTEP_MAX_CATCHUP_S = 0.1  # drop the backlog after a long pause
# Real time (lockstep off): the plant steps every CONTROL_PERIOD_S of wall time; the
# command from state k starts RT_SWITCH_SUBSTEPS (14 ms) into the next tick if it has
# arrived by then, else the previous one runs on (a late tick). Must match
# mpc_controller_node.RT_SWITCH_S.
RT_SWITCH_SUBSTEPS = 7
RT_MAX_BEHIND_S = 0.1  # further behind wall time than this: skip ahead (time lost)
RT_REPORT_EVERY_STEPS = 1500  # 30 s
RT_GIL_SWITCH_S = 0.0005
# Position-hold PD gains, split by torque headroom (joints 1-4: 87 Nm, 5-7: 12 Nm).
HOLD_KP = np.array([80, 80, 80, 80, 15, 15, 15])
HOLD_KD = np.array([8, 8, 8, 8, 1.5, 1.5, 1.5])
ENCODER_SIGMA_RAD = 5e-5
VELOCITY_FD_SUBSTEPS = 4  # driver velocity: difference over the last 8 ms (docs/implementation_notes.md)
FORCE_SIGMA_N = 0.2  # per axis
GRASP_ABOVE_M = 0.004  # the grasp stand-in grips a box top this far below the TCP at most
GRASP_INTO_M = 0.015  # ...or the TCP pressed this far into it
FORCE_BIAS_MAX_N = 0.5  # per axis, drawn once per run (tare drift)

# The model files use a meshdir placeholder for the menagerie assets. MuJoCo
# resolves <include> on disk, so load_scene_model writes the files to a temp dir.
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MESHDIR_PLACEHOLDER = "__MENAGERIE_PANDA_ASSETS_DIR__"
MENAGERIE_PANDA_ASSETS_DIR = os.environ.get(
    "MENAGERIE_PANDA_ASSETS_DIR",
    str(Path.home() / "mujoco_menagerie" / "franka_emika_panda" / "assets"),
)
ROOM_FILE = "room_scene.xml"
ROBOT_FILE = "panda_robot.xml"
DEFAULT_CELL_FILE = "cell_container.xml"
ARM_MOUNT_SITE = "arm_mount"


def _pose_attr(xyz, yaw):
    return f'pos="{xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}" euler="0 0 {np.degrees(yaw):.6f}"'


def _check_and_place(text, world, layout, cell_file, dock=None):
    """The room file's cell and base poses must be scene.py's; world "arm": the same scene
    expressed in the arm frame with the base parked (offline tools; dock: the arm's pose in
    the room, the cell's by default), "room": as written. layout "stations": the cell's two
    tables apart at their stations, the base at home (or docked, world "arm")."""
    arm_pose = CELL_POSE if dock is None else dock
    for name, pose, z in (("frame", CELL_POSE, -ROOM_FLOOR_Z), ("body", BASE_PARK_POSE, 0.0)):
        tag = "cell" if name == "frame" else "base_link"
        m = re.search(rf'<{name} name="{tag}" pos="([^"]+)" euler="0 0 ([^"]+)"', text)
        xyz, yaw = np.array(m.group(1).split(), float), np.radians(float(m.group(2)))
        if np.abs(xyz - (pose[0], pose[1], z)).max() > 1e-6 or abs(np.sin((yaw - pose[2]) / 2)) > 1e-6:
            raise RuntimeError(f"{ROOM_FILE}: the {tag} pose disagrees with scene.py")
        if world == "arm":
            local = (0.0, 0.0, 0.0) if name == "frame" else BASE_IN_ARM
            text = text.replace(m.group(0), f'<{name} name="{tag}" {_pose_attr((local[0], local[1], z + ROOM_FLOOR_Z), local[2])}')
    if world == "arm":
        room = invert(arm_pose)
        text = text.replace('<frame name="room">', f'<frame name="room" {_pose_attr((room[0], room[1], ROOM_FLOOR_Z), room[2])}>')
    if layout == "stations":
        if world == "arm" and dock is None:
            raise ValueError("the stations layout in the arm frame needs the dock")
        at = (lambda q: compose(invert(arm_pose), q)) if world == "arm" else (lambda q: q)
        z = 0.0 if world == "arm" else -ROOM_FLOOR_Z
        pick = cell_file.replace("cell_container", "station_pick")
        cell = re.search(r'<frame name="cell"[^>]*>\s*<include file="[^"]+"/>\s*</frame>', text)
        text = text.replace(cell.group(0), "\n    ".join(
            f'<frame name="{n}" {_pose_attr((*at(p)[:2], z), at(p)[2])}><include file="{f}"/></frame>'
            for n, p, f in (("pick_station", PICK_STATION_POSE, pick), ("place_station", PLACE_STATION_POSE,
                                                                       "station_place.xml"))))
        if world == "room":
            base = re.search(r'<body name="base_link" pos="[^"]+" euler="[^"]+"', text)
            text = text.replace(base.group(0), f'<body name="base_link" {_pose_attr((*BASE_HOME_POSE[:2], 0.0), BASE_HOME_POSE[2])}')
    elif layout != "cell":
        raise ValueError(f"layout must be 'cell' or 'stations', got {layout!r}")
    return text


def load_scene_model(cell_file: str = DEFAULT_CELL_FILE, world: str = "room", layout: str = "cell",
                     dock=None) -> mujoco.MjModel:
    """The room with the cell `cell_file` in it and the arm attached to the mobile base.
    world "arm": everything re-expressed in the arm's frame, the base parked at the cell
    (offline tools that work in the arm frame); the plant uses "room". layout
    "stations": the cell's tables at their stations apart (scene.*_STATION_POSE), the
    base at home; "cell": the parked cell (the arm task's layout). dock: with world "arm",
    the arm's pose in the room to express the scene from (a station's dock)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        for path in MODELS_DIR.glob("*.xml"):
            text = path.read_text().replace(MESHDIR_PLACEHOLDER, MENAGERIE_PANDA_ASSETS_DIR)
            if path.name == ROOM_FILE:
                text = _check_and_place(text.replace(f'file="{DEFAULT_CELL_FILE}"', f'file="{cell_file}"'), world,
                                        layout, cell_file, dock)
            (Path(tmpdir) / path.name).write_text(text)
        room = mujoco.MjSpec.from_file(str(Path(tmpdir) / ROOM_FILE))
        robot = mujoco.MjSpec.from_file(str(Path(tmpdir) / ROBOT_FILE))
    room.site(ARM_MOUNT_SITE).attach_body(robot.body("link0"), "", "")
    return room.compile()


TRAY_SHIFT_MAX_M = 0.05


def shift_tray(model, rng):
    """Move the dest_tray_* geoms by a random offset within TRAY_SHIFT_MAX_M along the
    cell's axes; returns it. The controller never reads it."""
    off = rng.uniform(-TRAY_SHIFT_MAX_M, TRAY_SHIFT_MAX_M, 2)
    cell = np.zeros(9)
    mujoco.mju_quat2Mat(cell, model.geom_quat[model.geom("dest_tray_floor").id])
    off_room = cell.reshape(3, 3)[:2, :2] @ off
    for g in range(model.ngeom):
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("dest_tray_"):
            model.geom_pos[g][:2] += off_room
    return off


ARM_JOINT_NAMES = [f"joint{i}" for i in range(1, 8)]
ARM_ACTUATOR_NAMES = [f"actuator{i}" for i in range(1, 8)]
HOME_KEYFRAME = "home"  # joint4's range is entirely negative; can't start all-zero
EE_BODY_NAME = "attachment"  # wrist body: camera mount and carry weld anchor (not the TCP)
TCP_SITE_NAME = "tcp_site"
PARKED_POSITION = (0.9, 0.9, -3.0)  # absent obstacle bodies: below the floor


class MujocoSimNode(Node):
    def __init__(self):
        super().__init__("mujoco_sim_node")

        self.declare_parameter("cell_file", DEFAULT_CELL_FILE)
        self.declare_parameter("layout", "cell")  # cell (the arm task) | stations (the mobile job)
        self.model = load_scene_model(self.get_parameter("cell_file").value,
                                      layout=self.get_parameter("layout").value)
        self.data = mujoco.MjData(self.model)
        # The plant stops matching the controller's model (plant_mismatch.py).
        self.declare_parameter("plant_mismatch", True)
        self.declare_parameter("mismatch_seed", 0)
        # Off: the arm is nominal (oracle), the boxes still get the seed's masses.
        arm_mismatch = bool(self.get_parameter("plant_mismatch").value)
        boxes = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(self.model.nbody)
                 if (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith("cbox_")]
        summary = apply_plant_mismatch(self.model, np.random.default_rng(
            int(self.get_parameter("mismatch_seed").value)), ARM_JOINT_NAMES, boxes, arm=arm_mismatch)
        self.has_base = has_base(self.model)
        # Conditions that make localization harder (plant_conditions.py): "spill worn_tyre ...".
        self.declare_parameter("conditions", "none")
        self.conditions = plant_conditions.parse(self.get_parameter("conditions").value)
        if self.has_base:
            # Own stream: the arm's and boxes' draws stay those of earlier runs.
            summary += "; base: " + apply_base_mismatch(self.model, np.random.default_rng(
                [int(self.get_parameter("mismatch_seed").value), 2]), WHEEL_JOINTS, BASE_BODY,
                worn=plant_conditions.WORN_TYRE if "worn_tyre" in self.conditions else 0.0)
            if self.conditions:
                summary += f"; conditions: {', '.join(sorted(self.conditions))}"
        refresh(self.model, self.data)
        self.get_logger().info(f"plant mismatch {'on' if arm_mismatch else 'off (arm nominal)'}: {summary}")
        self.declare_parameter("tray_seed", -1)  # -1: nominal tray
        tray_seed = int(self.get_parameter("tray_seed").value)
        if tray_seed >= 0:
            off = shift_tray(self.model, np.random.default_rng(tray_seed))
            self.get_logger().info(f"tray shifted by ({1e3 * off[0]:+.0f}, {1e3 * off[1]:+.0f}) mm (tray_seed {tray_seed})")
        home_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, HOME_KEYFRAME)
        mujoco.mj_resetDataKeyframe(self.model, self.data, home_id)

        frames.free_bodies_at_rest(self.model, self.data)
        self.declare_parameter("arm_start", "home")  # home | tucked (folded over the chassis) | carry
        start = {"tucked": ARM_TUCKED_Q, "carry": ARM_CARRY_Q}.get(self.get_parameter("arm_start").value)
        if start is not None:
            for name, q in zip(ARM_JOINT_NAMES, start):
                self.data.qpos[self.model.joint(name).qposadr[0]] = q
        mujoco.mj_forward(self.model, self.data)  # populate xpos before the first publish

        self.joint_qpos_adr = [self.model.joint(n).qposadr[0] for n in ARM_JOINT_NAMES]
        self.joint_qvel_adr = [self.model.joint(n).dofadr[0] for n in ARM_JOINT_NAMES]
        self.actuator_ids = [self.model.actuator(n).id for n in ARM_ACTUATOR_NAMES]

        self.ee_body_id = self.model.body(EE_BODY_NAME).id
        # What the controller tracks; used for the viewer marker.
        self.tcp_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, TCP_SITE_NAME)

        # One shared carry weld, repointed at whichever box is held (_set_box_held).
        self.box_carry_eq = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "box_carry")

        self.n_substeps = max(1, round(CONTROL_PERIOD_S / self.model.opt.timestep))
        self.declare_parameter("lockstep", False)
        self.lockstep = bool(self.get_parameter("lockstep").value)
        # Real time: the stepping thread and the executor share self.data.
        self.sim_lock = threading.RLock()
        self.rt_late = self.rt_late_run = self.rt_late_run_max = 0
        self.rt_lost_s = 0.0
        self.rt_stop = False
        self.latest_torque_cmd = np.zeros(len(ARM_ACTUATOR_NAMES))
        self.applied_torque = np.zeros(len(ARM_ACTUATOR_NAMES))
        self.cmd_ms = float("nan")  # the last command's arrival after its state went out
        self.last_step_ms = 0.0  # wall time of the previous _step
        # Stale-command fallback: gravity compensation plus a PD pull to the pose where
        # it started. Compensation alone lets residual velocity coast.
        self.hold_qpos = None
        # Stamped into each joint state and echoed on the command computed from it.
        self.step_count = 0
        self.cmd_src_step = -1
        self.last_step_wall = None
        self.next_step_due = None
        self.last_state_sent = 0.0
        self.last_poll = 0.0
        self.resumed_at = 0.0
        self.telemetry = Telemetry(
            "sim", ["sim_t", "step", "period_ms", "stale", "cmd_lag", "qdot_norm",
                    *[f"qd{i}" for i in range(1, 8)], *[f"tau{i}" for i in range(1, 8)],
                    *[f"q{i}" for i in range(1, 8)], "person_x", "person_y", "person_z", "person_yaw",
                    "fx", "fy", "fz", "late", "cmd_ms", "step_ms",
                    "base_x", "base_y", "base_z", "base_qw", "base_qx", "base_qy", "base_qz",
                    "wheel_cmd_l", "wheel_cmd_r", "wheel_qd_l", "wheel_qd_r"])
        self.hold_tau_max = self.model.actuator_ctrlrange[self.actuator_ids, 1]

        self.declare_parameter("noise_seed", 0)
        # Joint encoders and the wrist load cell (sensor_noise); own stream, so depth draws are unchanged.
        self.declare_parameter("sensor_noise", True)
        self.sensor_noise = bool(self.get_parameter("sensor_noise").value)
        self.sensor_rng = np.random.default_rng([int(self.get_parameter("noise_seed").value), 1])
        self.force_bias = (self.sensor_rng.uniform(-FORCE_BIAS_MAX_N, FORCE_BIAS_MAX_N, 3)
                           if self.sensor_noise else np.zeros(3))
        self.q_fd_start = None
        self.last_force = np.zeros(3)
        if self.sensor_noise:
            self.get_logger().info(f"sensor noise on: wrist force bias {np.round(self.force_bias, 2).tolist()} N")

        self.state_pub = self.create_publisher(JointState, "/sim/joint_states", 10)
        # Everything sim_sensors_node needs to draw the scene: [step, held box body (-1),
        # sphere radius, qpos, mocap_pos, mocap_quat, the state's time stamp (s)].
        self.scene_state_pub = self.create_publisher(Float64MultiArray, "/sim/scene_state", 1)
        # Wrist load cell, world axes: [fx, fy, fz, step]. Reads the held box's weight.
        self.wrist_force_pub = self.create_publisher(Float64MultiArray, "/sim/wrist_force", 10)
        self.wrist_force_adr = self.model.sensor_adr[self.model.sensor("wrist_force").id]
        self.wrist_site_id = self.model.site("attachment_site").id
        # Newest command only, best effort (no queue). Real time: on its own node and thread,
        # so a command does not wait behind the other callbacks.
        self.cmd_node = None if self.lockstep else rclpy.create_node("mujoco_sim_commands")
        (self.cmd_node or self).create_subscription(
            JointState, "/sim/joint_command", self._on_joint_command,
            QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.create_subscription(String, "/task/action", self._on_task_action, 10)
        self.base = None
        self.wheel_cmd = np.zeros(2)
        if self.has_base:
            self.base = BaseDrive(self.model, self.data, np.random.default_rng(
                [int(self.get_parameter("noise_seed").value), 3]), noise=self.sensor_noise,
                conditions=self.conditions)
            # Wheel speeds (rad/s) [left, right] to the motor drivers; newest only.
            (self.cmd_node or self).create_subscription(
                Float64MultiArray, "/sim/wheel_command", self._on_wheel_command,
                QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.BEST_EFFORT))
            self.wheel_pub = self.create_publisher(JointState, "/sim/wheel_states", 10)
            self.imu_pub = self.create_publisher(Imu, "/sim/imu", 10)
            # The base's true pose in the room, for validation only: [step, x, y, z, qw, qx, qy, qz].
            self.base_truth_pub = self.create_publisher(Float64MultiArray, "/sim/base_truth", 10)
            if self.sensor_noise:
                self.get_logger().info(f"base: gyro bias {np.round(self.base.gyro_bias, 4).tolist()} rad/s")
        # The one ground-truth read: the grasped box's half-extents (task_node).
        self.grasped_box_size_pub = self.create_publisher(Vector3, "/sim/grasped_box_size", 10)
        self.grasped_box_offset_pub = self.create_publisher(Vector3, "/sim/grasped_box_offset", 10)
        # Box on the carry weld; place_held releases it.
        self.held_box_name = None
        # Contact and disturbance evaluation: logged only, never sent to the controller.
        self.robot_body_ids = {self.model.body(n).id for n in (*[f"link{i}" for i in range(8)], EE_BODY_NAME)}
        self.box_body_ids = [b for b in range(self.model.nbody)
                             if (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith("cbox_")]
        self.contact_log = {}
        self.box_rest_pos = {b: self.data.xpos[b].copy() for b in self.box_body_ids}
        self.release_check = None
        self.last_placed_body = None

        # Environment actor bodies (dynamic_obstacle_node): a sphere and a person, both mocap.
        self.dynamic_obstacle_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "dynamic_obstacle")
        self.dynamic_obstacle_mocap_id = (
            self.model.body_mocapid[self.dynamic_obstacle_body_id]
            if self.dynamic_obstacle_body_id >= 0 else -1
        )
        person_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "person_obstacle")
        self.person_mocap_id = self.model.body_mocapid[person_body] if person_body >= 0 else -1
        # More people (scenario actors, room frame): /env/people [x, y, yaw] per person_<i>, nan: absent.
        self.people_mocap = [self.model.body_mocapid[self.model.body(f"person_{i}").id] for i in range(1, 7)
                             if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"person_{i}") >= 0]
        if self.people_mocap:
            self.create_subscription(Float64MultiArray, "/env/people", self._on_people, 10)
        if self.dynamic_obstacle_mocap_id >= 0:
            self.dynamic_obstacle_geom_id = int(
                self.model.body_geomadr[self.dynamic_obstacle_body_id])
            self.create_subscription(
                Float64MultiArray, "/env/dynamic_obstacle", self._on_dynamic_obstacle, 10)
        if self.lockstep:
            self.create_timer(LOCKSTEP_POLL_S, self._try_step)
        else:
            # The stepping thread waits up to a switch interval (5 ms) per hand-over of the GIL.
            sys.setswitchinterval(RT_GIL_SWITCH_S)
            self.rt_thread = threading.Thread(target=self._realtime_loop, daemon=True)
            self.rt_thread.start()
        self.get_logger().info(f"stepping: {'lockstep with the controller' if self.lockstep else 'real time'}")

    def _worst_overlap(self, box_name):
        """Deepest overlap (m) of a box with the tray or another box, from world
        axis-aligned bounds; 0 if apart. Also records the facing gaps.
        """
        arm = frames.arm_base_pose(self.model, self.data)

        def bounds(geom_id):  # arm-frame axes
            c, r = frames.to_arm(arm, self.data.geom_xpos[geom_id], self.data.geom_xmat[geom_id].reshape(3, 3))
            half = np.abs(r) @ self.model.geom_size[geom_id]
            return c - half, c + half
        if not hasattr(self, "_overlap_geoms"):
            self._overlap_geoms = []
            for g in range(self.model.ngeom):
                gname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
                body = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, self.model.geom_bodyid[g]) or ""
                if gname.startswith("dest_tray") or body.startswith("cbox_"):
                    self._overlap_geoms.append((g, gname or body))
        mine = self.model.body_geomadr[self.model.body(box_name).id]
        lo, hi = bounds(mine)
        worst, name = 0.0, "nothing"
        self.last_nearest_gap = (np.inf, "nothing")
        self.last_side_gaps = [np.inf] * 4  # -x, +x, -y, +y
        for g, gname in self._overlap_geoms:
            if g == mine:
                continue
            olo, ohi = bounds(g)
            overlap = np.minimum(hi, ohi) - np.maximum(lo, olo)  # per axis; < 0 = apart
            depth = float(np.min(overlap))
            if depth > worst:
                worst, name = depth, gname
            # Facing: apart along one axis, overlapping along the other two.
            if int(np.sum(overlap < 0)) == 1:
                if -depth < self.last_nearest_gap[0]:
                    self.last_nearest_gap = (-depth, gname)
                axis = int(np.argmin(overlap))
                if axis < 2:
                    side = 2 * axis + int(olo[axis] > lo[axis])
                    self.last_side_gaps[side] = min(self.last_side_gaps[side], -depth)
        return worst, name

    def _contact_label(self, geom_id, held_id):
        body = self.model.geom_bodyid[geom_id]
        if body == held_id:
            return "held box"
        return (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
                or mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body))

    def _log_contacts(self):
        """Accumulate contacts that involve the robot or the held box: steps in contact,
        deepest penetration, largest normal force.
        """
        held = self.model.body(self.held_box_name).id if self.held_box_name else -1
        seen = set()
        force = np.zeros(6)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            b1, b2 = self.model.geom_bodyid[c.geom1], self.model.geom_bodyid[c.geom2]
            if not ({b1, b2} & self.robot_body_ids or held in (b1, b2)):
                continue
            key = " / ".join(sorted((self._contact_label(c.geom1, held), self._contact_label(c.geom2, held))))
            mujoco.mj_contactForce(self.model, self.data, i, force)
            steps, pen, f = self.contact_log.get(key, (0, 0.0, 0.0))
            self.contact_log[key] = (steps + (key not in seen), max(pen, -c.dist), max(f, abs(force[0])))
            seen.add(key)

    def _event_report(self, exclude):
        """Contacts since the last event, and boxes (other than `exclude`) moved more
        than 2 mm since then; resets both.
        """
        parts = [f"{k} {s} steps, {1e3 * p:.1f} mm, {f:.0f} N"
                 for k, (s, p, f) in sorted(self.contact_log.items(), key=lambda kv: -kv[1][2])[:4]]
        self.contact_log = {}
        moved = []
        for b in self.box_body_ids:
            if b != exclude:
                dist = np.linalg.norm(self.data.xpos[b] - self.box_rest_pos[b])
                if dist > 0.002:
                    moved.append(f"{mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b)} {1e3 * dist:.0f} mm")
                self.box_rest_pos[b] = self.data.xpos[b].copy()
        return (f"contacts: {'; '.join(parts) or 'none'}; "
                f"boxes disturbed: {', '.join(moved) or 'none'}")

    def _report_release(self):
        """A released box 1 s later: how far it moved while settling, tilt, yaw change."""
        body, _, pos0, mat0 = self.release_check
        self.release_check = None
        mat = self.data.xmat[body].reshape(3, 3)
        tilt = np.degrees(np.arccos(np.clip(mat[2, 2], -1.0, 1.0)))
        yaw = np.degrees(np.arctan2(mat[1, 0], mat[0, 0]) - np.arctan2(mat0[1, 0], mat0[0, 0]))
        d = self.data.xpos[body] - pos0
        self.box_rest_pos[body] = self.data.xpos[body].copy()
        c, r = frames.to_arm(frames.arm_base_pose(self.model, self.data), self.data.xpos[body], mat)
        self.get_logger().info(
            f"released {mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body)} settled: "
            f"moved {1e3 * np.linalg.norm(d):.1f} mm (dz {1e3 * d[2]:+.1f}), tilt {tilt:.1f} deg, "
            f"yaw {(yaw + 180) % 360 - 180:+.1f} deg; at {1e3 * c[0]:.1f}/{1e3 * c[1]:.1f}/"
            f"{np.degrees(frames.yaw_of(r)):.2f} (arm frame)")

    def _truth_at_release(self, body):
        """Validation only (place_budget.py): the released box, the TCP and every box, arm frame."""
        arm = frames.arm_base_pose(self.model, self.data)

        def pose(p, r):
            c, ra = frames.to_arm(arm, p, r)
            return f"{1e3 * c[0]:.1f}/{1e3 * c[1]:.1f}/{np.degrees(frames.yaw_of(ra)):.2f}"
        tcp_p, tcp_r = self.data.site_xpos[self.tcp_site_id], self.data.site_xmat[self.tcp_site_id].reshape(3, 3)
        boxes = " ".join(f"{mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b)}="
                         f"{pose(self.data.xpos[b], self.data.xmat[b].reshape(3, 3))}" for b in self.box_body_ids)
        return (f"truth at release {mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body)}: "
                f"box {pose(self.data.xpos[body], self.data.xmat[body].reshape(3, 3))} "
                f"tcp {pose(tcp_p, tcp_r)} (x/y mm, yaw deg); boxes {boxes}")

    def _box_half_extents(self, box_name):
        """A box's (hx, hy, hz) half-extents, from its geom."""
        body_id = self.model.body(box_name).id
        geom_id = self.model.body_geomadr[body_id]
        return self.model.geom_size[geom_id]

    def _on_joint_command(self, msg: JointState):
        with self.sim_lock:
            self.latest_torque_cmd = np.array(msg.effort)
            # A command without a state stamp (the MoveIt bridge) answers the latest state.
            self.cmd_src_step = int(msg.header.frame_id) if msg.header.frame_id else self.step_count
            self.hold_qpos = None  # fresh command: drop the hold target
            if self.cmd_src_step == self.step_count:
                self.cmd_ms = (time.perf_counter() - self.last_state_sent) * 1e3
        if self.lockstep:
            self._try_step()

    def _on_wheel_command(self, msg: Float64MultiArray):
        with self.sim_lock:
            self.base.set_command(msg.data, self.data.time)

    def _on_people(self, msg: Float64MultiArray):
        with self.sim_lock:
            for k, mid in enumerate(self.people_mocap):
                x, y, yaw = msg.data[3 * k:3 * k + 3] if len(msg.data) >= 3 * k + 3 else (np.nan,) * 3
                if np.isfinite(x):
                    self.data.mocap_pos[mid] = (x, y, 0.0)
                    self.data.mocap_quat[mid] = (np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2))
                else:
                    self.data.mocap_pos[mid] = PARKED_POSITION

    def _on_dynamic_obstacle(self, msg: Float64MultiArray):
        with self.sim_lock:
            self._set_dynamic_obstacle(msg)

    def _set_dynamic_obstacle(self, msg: Float64MultiArray):
        # [x, y, z, radius, t_pub, top_z, kind, heading] from dynamic_obstacle_node.
        # kind 1 moves the person, kind 0 the sphere; radius 0 parks both.
        x, y, z, radius = msg.data[:4]
        kind = int(msg.data[6]) if len(msg.data) > 6 else 0
        heading = float(msg.data[7]) if len(msg.data) > 7 else 0.0
        present = radius > 0.0
        person = present and kind == 1
        sphere = present and kind == 0
        if self.person_mocap_id >= 0:
            if person:
                self.data.mocap_pos[self.person_mocap_id] = (x, y, z)  # feet on the room floor
                self.data.mocap_quat[self.person_mocap_id] = (
                    np.cos(heading / 2), 0.0, 0.0, np.sin(heading / 2))
            else:
                self.data.mocap_pos[self.person_mocap_id] = PARKED_POSITION
        if sphere:
            self.data.mocap_pos[self.dynamic_obstacle_mocap_id] = (x, y, z)
            # Render at the actor's radius.
            self.model.geom_size[self.dynamic_obstacle_geom_id, 0] = radius
        else:
            self.data.mocap_pos[self.dynamic_obstacle_mocap_id] = PARKED_POSITION

    def _on_task_action(self, msg: String):
        with self.sim_lock:
            self._task_action(msg)

    def _task_action(self, msg: String):
        # "pick_at x y z": grip the box the tool is on (the pose is only logged). "place_held":
        # release the held box.
        parts = msg.data.split()
        if parts and parts[0] == "pick_at" and len(parts) == 4:
            target = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            box_name = self._box_under_tool()
            if box_name is None:
                tcp = self.data.site_xpos[self.tcp_site_id]
                self.get_logger().warn(f"pick_at {target}: the tool (TCP {np.round(tcp, 3).tolist()}) is on no "
                                       f"box top; nothing grasped")
                return
            self.held_box_name = box_name
            self.carry_overlap = (0.0, "nothing")
            self._set_box_held(True, box_name)
            hx, hy, hz = self._box_half_extents(box_name)
            self.grasped_box_size_pub.publish(Vector3(x=float(hx), y=float(hy), z=float(hz)))
            # Box centre relative to the TCP, in the TCP frame (in-hand sensing): the
            # grasp from a 5 mm heightmap is a few mm off centre.
            tcp_r = self.data.site_xmat[self.tcp_site_id].reshape(3, 3)
            offset = tcp_r.T @ (self.data.xpos[self.model.body(box_name).id] - self.data.site_xpos[self.tcp_site_id])
            self.grasped_box_offset_pub.publish(Vector3(x=float(offset[0]), y=float(offset[1]), z=float(offset[2])))
            hz = self._box_half_extents(box_name)[2]
            top = self.data.xpos[self.model.body(box_name).id][2] + hz
            self.get_logger().info(
                f"picked up {box_name} (the tool is on it; asked at {np.round(target, 3).tolist()}); "
                f"TCP {1e3 * (self.data.site_xpos[self.tcp_site_id][2] - top):+.1f} mm above its true top; "
                f"true size {2e3 * hx:.0f} x {2e3 * hy:.0f} x {2e3 * hz:.0f} mm; "
                f"since last event {self._event_report(self.model.body(box_name).id)}")
        elif parts and parts[0] == "place_held" and len(parts) == 1:
            if self.held_box_name is None:
                self.get_logger().warn("place_held: no box currently held")
                return
            box_name = self.held_box_name
            self._set_box_held(False, box_name)
            hz = self._box_half_extents(box_name)[2]
            bottom = self.data.xpos[self.model.body(box_name).id][2] - hz
            depth, other = self._worst_overlap(box_name)
            self.get_logger().info(
                f"placed {box_name} (bottom at z={bottom:.4f} m; overlap at release "
                f"{1e3 * depth:.1f} mm with {other}, worst while carried "
                f"{1e3 * self.carry_overlap[0]:.1f} mm with {self.carry_overlap[1]}; nearest facing gap "
                f"{1e3 * self.last_nearest_gap[0]:.1f} mm to {self.last_nearest_gap[1]}; side gaps -x/+x/-y/+y "
                f"{'/'.join(f'{1e3 * g:.1f}' if np.isfinite(g) else '-' for g in self.last_side_gaps)} mm); "
                f"while carried {self._event_report(self.model.body(box_name).id)}")
            body = self.model.body(box_name).id
            self.get_logger().info(self._truth_at_release(body))
            self.last_placed_body = body
            self.release_check = (body, self.step_count + 50, self.data.xpos[body].copy(),
                                  self.data.xmat[body].reshape(3, 3).copy())
            self.held_box_name = None
        elif parts and parts[0] == "place_aborted" and len(parts) == 1:
            self.get_logger().info(f"place aborted; during the attempt {self._event_report(-1)}")
        elif parts and parts[0] == "pushed" and len(parts) == 1:
            if self.last_placed_body is None:
                return
            body = self.last_placed_body
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body)
            mat = self.data.xmat[body].reshape(3, 3)
            tilt = np.degrees(np.arccos(np.clip(mat[2, 2], -1.0, 1.0)))
            yaw = np.degrees(frames.yaw_of(mat) - frames.yaw_of(frames.arm_base_pose(self.model, self.data)[1]))
            moved = np.linalg.norm(self.data.xpos[body][:2] - self.box_rest_pos[body][:2])
            depth, other = self._worst_overlap(name)
            self.box_rest_pos[body] = self.data.xpos[body].copy()
            self.get_logger().info(
                f"pushed {name}: moved {1e3 * moved:.1f} mm, tilt {tilt:.1f} deg, yaw {(yaw + 90) % 180 - 90:+.1f} deg "
                f"(0 = square); side gaps -x/+x/-y/+y "
                f"{'/'.join(f'{1e3 * g:.1f}' if np.isfinite(g) else '-' for g in self.last_side_gaps)} mm; "
                f"overlap {1e3 * depth:.1f} mm with {other}; {self._event_report(body)}")
        else:
            self.get_logger().warn(f"unrecognized /task/action: {msg.data!r}")

    def _box_under_tool(self):
        """The grasp stand-in grips only with contact, as suction would: the box whose top
        the TCP is on (within GRASP_ABOVE_M above it or GRASP_INTO_M into it) and inside
        whose top face it is. Where the tool really is decides, not what was asked."""
        mujoco.mj_forward(self.model, self.data)
        tcp = self.data.site_xpos[self.tcp_site_id]
        for body_id in range(1, self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if not name or not name.startswith("cbox_"):
                continue
            hx, hy, hz = self._box_half_extents(name)
            rot = self.data.xmat[body_id].reshape(3, 3)
            local = rot.T @ (tcp - self.data.xpos[body_id])
            if abs(local[0]) <= hx and abs(local[1]) <= hy and -GRASP_INTO_M <= local[2] - hz <= GRASP_ABOVE_M:
                return name
        return None

    def _set_box_held(self, held: bool, box_name: str = "box"):
        """Engage the carry weld on a box at its current pose relative to the wrist (no
        snap), or release it; a released box rests by contact.
        """
        if not held:
            self.data.eq_active[self.box_carry_eq] = False
            return
        mujoco.mj_forward(self.model, self.data)
        box_body_id = self.model.body(box_name).id
        self.model.eq_obj2id[self.box_carry_eq] = box_body_id  # repoint the shared carry weld

        p1, q1 = self.data.xpos[self.ee_body_id], self.data.xquat[self.ee_body_id]
        p2, q2 = self.data.xpos[box_body_id], self.data.xquat[box_body_id]
        neg_p1, neg_q1 = np.zeros(3), np.zeros(4)
        mujoco.mju_negPose(neg_p1, neg_q1, p1, q1)
        rel_pos, rel_quat = np.zeros(3), np.zeros(4)
        mujoco.mju_mulPose(rel_pos, rel_quat, neg_p1, neg_q1, p2, q2)

        self.model.eq_data[self.box_carry_eq, 0:3] = 0.0  # anchor: unused
        self.model.eq_data[self.box_carry_eq, 3:6] = rel_pos
        self.model.eq_data[self.box_carry_eq, 6:10] = rel_quat
        self.data.eq_active[self.box_carry_eq] = True

    def _try_step(self):
        """Step physics when due and the controller has answered the last published
        state (or waited LOCKSTEP_MAX_WAIT_S). Called by a fast poll timer and by
        _on_joint_command.
        """
        now = time.perf_counter()
        if now - self.last_poll > LOCKSTEP_STALL_S:
            self.resumed_at = now
        self.last_poll = now
        if self.next_step_due is None:
            self.next_step_due = now
        if now < self.next_step_due:
            return
        fresh = self.cmd_src_step == self.step_count
        live = 0 <= self.step_count - self.cmd_src_step <= LOCKSTEP_LIVE_STEPS
        # Time the wait from when the state went out or the thread came back from a
        # stall, not from the due time: after a stall that stepped several times on
        # old commands, a kick after every scan.
        if (not fresh and live
                and now - max(self.last_state_sent, self.resumed_at) < LOCKSTEP_MAX_WAIT_S):
            return
        self._step()
        self.next_step_due = max(self.next_step_due + CONTROL_PERIOD_S, now - LOCKSTEP_MAX_CATCHUP_S)

    def _person_pose(self):
        if self.person_mocap_id < 0:
            return (0.0, 0.0, PARKED_POSITION[2], 0.0)
        pos, quat = self.data.mocap_pos[self.person_mocap_id], self.data.mocap_quat[self.person_mocap_id]
        return (*np.round(pos, 4), round(float(2 * np.arctan2(quat[3], quat[0])), 4))

    def _command_torque(self):
        """(torque, stale): the latest command, or the position hold if it is older than
        COMMAND_STALE_STEPS."""
        stale = self.cmd_src_step < 0 or self.step_count - self.cmd_src_step > COMMAND_STALE_STEPS
        if not stale:
            return np.array(self.latest_torque_cmd, dtype=float), False
        if self.hold_qpos is None:
            self.hold_qpos = self.data.qpos[self.joint_qpos_adr].copy()
        q = self.data.qpos[self.joint_qpos_adr]
        qdot = self.data.qvel[self.joint_qvel_adr]
        tau = self.data.qfrc_bias[self.joint_qvel_adr] + HOLD_KP * (self.hold_qpos - q) - HOLD_KD * qdot
        return np.clip(tau, -self.hold_tau_max, self.hold_tau_max), True

    def _sleep_until(self, t):
        dt = t - time.perf_counter()
        if dt > 0:
            time.sleep(dt)

    def _realtime_loop(self):
        """Step every CONTROL_PERIOD_S of wall time. The torque in effect runs for the
        first RT_SWITCH_SUBSTEPS; then the answer to the last published state if it
        has come, else the same torque again (a late tick)."""
        with self.sim_lock:
            u_cur, _ = self._command_torque()
        due = time.perf_counter()
        switch_s = RT_SWITCH_SUBSTEPS * self.model.opt.timestep
        while not self.rt_stop:
            self._sleep_until(due + switch_s)
            with self.sim_lock:
                u_new, stale = self._command_torque()
                late = not stale and self.cmd_src_step != self.step_count
                if late:
                    u_new = u_cur
            self._sleep_until(due + CONTROL_PERIOD_S)
            with self.sim_lock:
                self._step([(u_cur, RT_SWITCH_SUBSTEPS), (u_new, self.n_substeps - RT_SWITCH_SUBSTEPS)],
                           stale, late)
            u_cur = u_new
            self.rt_late += late
            self.rt_late_run = self.rt_late_run + 1 if late else 0
            self.rt_late_run_max = max(self.rt_late_run_max, self.rt_late_run)
            due += CONTROL_PERIOD_S
            behind = time.perf_counter() - due
            if behind > RT_MAX_BEHIND_S:
                self.rt_lost_s += behind
                due = time.perf_counter()
            if self.step_count % RT_REPORT_EVERY_STEPS == 0:
                self.get_logger().info(
                    f"real time: {self.rt_late} late ticks of {self.step_count} "
                    f"(longest run {self.rt_late_run_max}), {self.rt_lost_s:.2f} s lost")

    def _step(self, segments=None, stale=None, late=False):
        """One control period. segments: [(torque, substeps), ...]; None: the latest
        command (or the hold) for the whole period (lockstep)."""
        with self.sim_lock:
            self._step_locked(segments, stale, late)

    def _step_locked(self, segments, stale, late):
        t_step = time.perf_counter()
        self._advance(segments, stale, late)
        self.last_step_ms = (time.perf_counter() - t_step) * 1e3

    def _advance(self, segments, stale, late):
        if segments is None:
            tau, stale = self._command_torque()
            segments = [(tau, self.n_substeps)]
        self.data.ctrl[self.actuator_ids] = segments[-1][0]
        if self.base is not None:
            self.wheel_cmd = self.base.update(self.data)
        if self.telemetry.enabled:
            now = time.perf_counter()
            period_ms = (now - self.last_step_wall) * 1e3 if self.last_step_wall else 0.0
            self.last_step_wall = now
            qd = self.data.qvel[self.joint_qvel_adr]
            self.telemetry.row(
                round(self.data.time, 3), self.step_count, round(period_ms, 2), int(stale),
                self.step_count - self.cmd_src_step, round(float(np.linalg.norm(qd)), 4),
                *np.round(qd, 4), *np.round(self.data.ctrl[self.actuator_ids], 3),
                *np.round(self.data.qpos[self.joint_qpos_adr], 5), *self._person_pose(),
                *np.round(self.last_force, 3), int(late), round(self.cmd_ms, 2), round(self.last_step_ms, 2),
                *(np.round(self.base.truth(self.data), 5) if self.base else [np.nan] * 7),
                *np.round(self.wheel_cmd, 4),
                *(np.round(self.data.qvel[self.base.dof_adr], 4) if self.base else [np.nan] * 2))
        # The driver's velocity: encoders read VELOCITY_FD_SUBSTEPS before the end.
        fd_at = self.n_substeps - VELOCITY_FD_SUBSTEPS if self.sensor_noise else None
        done = 0
        for tau, n in segments:
            self.data.ctrl[self.actuator_ids] = tau
            for chunk in ((fd_at - done, n - (fd_at - done)) if fd_at is not None and done < fd_at < done + n
                          else (n,)):
                if self.base is not None:
                    self.base.step(self.model, self.data, chunk)
                else:
                    mujoco.mj_step(self.model, self.data, nstep=chunk)
                done += chunk
                if done == fd_at:
                    self.q_fd_start = self._encoders()
        self.applied_torque = np.array(segments[-1][0], dtype=float)
        self.step_count += 1
        if self.held_box_name is not None:
            # Deepest overlap while carrying, reported on place.
            depth, other = self._worst_overlap(self.held_box_name)
            if depth > self.carry_overlap[0]:
                self.carry_overlap = (depth, other)
        self._log_contacts()
        if self.release_check is not None and self.step_count >= self.release_check[1]:
            self._report_release()
        self._publish_state()

    def _publish_base(self, stamp):
        """Wheel encoders and the IMU, stamped with the step like the joint states; the
        base's true pose for validation."""
        wheels = JointState()
        wheels.header.stamp = stamp
        wheels.header.frame_id = str(self.step_count)
        wheels.name = list(WHEEL_JOINTS)
        wheels.position = self.base.encoders(self.data).tolist()
        self.wheel_pub.publish(wheels)
        gyro, accel = self.base.imu(self.data, CONTROL_PERIOD_S)
        imu = Imu()
        imu.header.stamp = stamp
        imu.header.frame_id = str(self.step_count)
        imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z = gyro.tolist()
        imu.linear_acceleration.x, imu.linear_acceleration.y, imu.linear_acceleration.z = accel.tolist()
        imu.orientation_covariance[0] = -1.0  # no orientation estimate
        self.imu_pub.publish(imu)
        self.base_truth_pub.publish(Float64MultiArray(data=[float(self.step_count), *self.base.truth(self.data).tolist()]))

    def _encoders(self):
        return self.data.qpos[self.joint_qpos_adr] + self.sensor_rng.normal(0.0, ENCODER_SIGMA_RAD, 7)

    def _publish_state(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = str(self.step_count)
        self.last_state_sent = time.perf_counter()
        # The load cell in the arm frame's axes (its mount rotation is the robot's own model).
        f = frames.arm_base_pose(self.model, self.data)[1].T @ self.data.site_xmat[self.wrist_site_id].reshape(
            3, 3) @ self.data.sensordata[self.wrist_force_adr:self.wrist_force_adr + 3]
        if self.sensor_noise:
            q = self._encoders()
            qd = np.zeros_like(q) if self.q_fd_start is None else (q - self.q_fd_start) / (
                VELOCITY_FD_SUBSTEPS * self.model.opt.timestep)
            f = f + self.force_bias + self.sensor_rng.normal(0.0, FORCE_SIGMA_N, 3)
        else:
            q, qd = self.data.qpos[self.joint_qpos_adr], self.data.qvel[self.joint_qvel_adr]
        msg.position = q.tolist()
        msg.velocity = qd.tolist()
        msg.effort = self.applied_torque.tolist()  # the torque in effect (it runs on into the next tick)
        self.state_pub.publish(msg)
        self.last_force = f
        self.wrist_force_pub.publish(Float64MultiArray(data=[*f.tolist(), float(self.step_count)]))
        if self.base is not None:
            self._publish_base(msg.header.stamp)
        held = self.model.body(self.held_box_name).id if self.held_box_name else -1
        radius = (self.model.geom_size[self.dynamic_obstacle_geom_id, 0]
                  if self.dynamic_obstacle_mocap_id >= 0 else 0.0)
        self.scene_state_pub.publish(Float64MultiArray(data=np.concatenate([
            [self.step_count, held, radius], self.data.qpos, self.data.mocap_pos.ravel(),
            self.data.mocap_quat.ravel(), [msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9]]).tolist()))


def main():
    rclpy.init()
    node = MujocoSimNode()
    if node.cmd_node is not None:
        cmd_executor = SingleThreadedExecutor()
        cmd_executor.add_node(node.cmd_node)
        threading.Thread(target=cmd_executor.spin, daemon=True).start()
    try:
        rclpy.spin(node)
    finally:
        node.rt_stop = True
        if not node.lockstep:
            node.rt_thread.join(timeout=1.0)
            node.get_logger().info(
                f"real time: {node.rt_late} late ticks of {node.step_count} "
                f"(longest run {node.rt_late_run_max}), {node.rt_lost_s:.2f} s lost")
        node.telemetry.flush()
        node.destroy_node()
        # rclpy's SIGINT handler already shut the context down; a second shutdown
        # throws (and crashed the viewer thread).
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
