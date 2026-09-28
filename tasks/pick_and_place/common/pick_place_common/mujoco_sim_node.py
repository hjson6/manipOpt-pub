"""The plant: owns the MuJoCo simulation and exposes it over ROS 2 topics
(joint states, torque commands, depth-camera scans, the grasp weld). Nothing
else imports mujoco.

See docs/implementation_notes.md#mujoco_sim_nodepy.
"""
import json
import multiprocessing
import os
import tempfile
import threading
import time
from pathlib import Path

# Cap the passive viewer's render thread at vsync; must be set before its GL
# context exists (docs/design_notes.md).
os.environ.setdefault("vblank_mode", "1")

import numpy as np
import cv2
import mujoco
import mujoco.viewer
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Vector3
from std_msgs.msg import Empty, Float32MultiArray, Float64MultiArray, String

from pick_place_common.scene import (
    CONTAINER_CAM_HEIGHT, CONTAINER_CAM_MOUNT_ID, CONTAINER_CAM_MOUNT_OFFSET,
    CONTAINER_CAM_NAME, CONTAINER_CAM_REF_Z, CONTAINER_CAM_WIDTH, DEST_CAM_REF_Z,
    DEST_FLOOR_Z, DEST_GRID_POINTS, FLOOR_Z, SOURCE_GRID_POINTS,
    AISLE_MASK_BOXES, AISLE_MASK_MARGIN_M, DETECTION_MIN_BLOB_PX,
    FOREGROUND_MIN_HEIGHT_M, WORKSPACE_CAM_FOVY_DEG, WORKSPACE_CAM_HEIGHT,
    WORKSPACE_CAM_NAME, WORKSPACE_CAM_POS, WORKSPACE_CAM_RATE_HZ,
    WORKSPACE_CAM_WIDTH,
)
from perception import heightmap, obstacle_detection
from pick_place_common.depth_noise import add_depth_noise
from pick_place_common.plant_mismatch import apply_plant_mismatch, refresh
from pick_place_common import window_process
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
# Position-hold PD gains, split by torque headroom (joints 1-4: 87 Nm, 5-7: 12 Nm).
HOLD_KP = np.array([80, 80, 80, 80, 15, 15, 15])
HOLD_KD = np.array([8, 8, 8, 8, 1.5, 1.5, 1.5])

# The model files use a meshdir placeholder for the menagerie assets. MuJoCo
# resolves <include> on disk, so load_scene_model writes both files to a temp dir.
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MESHDIR_PLACEHOLDER = "__MENAGERIE_PANDA_ASSETS_DIR__"
MENAGERIE_PANDA_ASSETS_DIR = os.environ.get(
    "MENAGERIE_PANDA_ASSETS_DIR",
    str(Path.home() / "mujoco_menagerie" / "franka_emika_panda" / "assets"),
)


def load_scene_model(scene_file: str = "panda_scene_container.xml") -> mujoco.MjModel:
    with tempfile.TemporaryDirectory() as tmpdir:
        for fname in (scene_file, "panda_robot.xml"):
            text = (MODELS_DIR / fname).read_text().replace(
                MESHDIR_PLACEHOLDER, MENAGERIE_PANDA_ASSETS_DIR
            )
            (Path(tmpdir) / fname).write_text(text)
        return mujoco.MjModel.from_xml_path(str(Path(tmpdir) / scene_file))


DASHBOARD_WINDOW = "pick & place decisions"
OBSTACLE_WINDOW = "obstacle detection (live)"
OBSTACLE_VIEW_HALF_WIDTH_M = 1.4  # floor shown each side of the image centre in the obstacle view
# Obstacle view pixel classes: none, ignored, robot, obstacle.
_OBSTACLE_LUT = np.array([[0, 0, 0], [200, 90, 20], [60, 170, 60], [40, 40, 230]], dtype=np.uint8)
# JET as a lookup table: cv2.applyColorMap costs ~4 ms per frame here.
_JET_LUT = cv2.applyColorMap(np.arange(256, dtype=np.uint8)[:, None], cv2.COLORMAP_JET)[:, 0, :]


def _draw_world_rect(image, x0, x1, y0, y1, z, cam_pos, cam_mat, fovy, color=(255, 255, 255), thickness=2):
    """Outline the world rectangle [x0, x1] x [y0, y1] at height z on a
    container_cam image taken from cam_pos/cam_mat.
    """
    pts = []
    for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
        uv = heightmap.project_world_point(
            cam_pos, cam_mat, fovy, CONTAINER_CAM_WIDTH, CONTAINER_CAM_HEIGHT, (x, y, z))
        if uv is None:
            return
        pts.append([int(round(uv[0])), int(round(uv[1]))])
    cv2.polylines(image, [np.array(pts, dtype=np.int32)], isClosed=True, color=color, thickness=thickness)


ARM_JOINT_NAMES = [f"joint{i}" for i in range(1, 8)]
ARM_ACTUATOR_NAMES = [f"actuator{i}" for i in range(1, 8)]
HOME_KEYFRAME = "home"  # joint4's range is entirely negative; can't start all-zero
EE_BODY_NAME = "attachment"  # wrist body: camera mount and carry weld anchor (not the TCP)
TCP_SITE_NAME = "tcp_site"
MARKER_RADIUS_M = 0.02
PARKED_POSITION = (0.9, 0.9, -3.0)  # absent obstacle bodies: below the floor
DEST_SCAN_MASK_GROUP = 3  # unused geom group; the held box is moved into it for the tray scan


class MujocoSimNode(Node):
    def __init__(self):
        super().__init__("mujoco_sim_node")

        self.declare_parameter("scene_file", "panda_scene_container.xml")
        self.scene_file = self.get_parameter("scene_file").value
        self.model = load_scene_model(self.scene_file)
        self.data = mujoco.MjData(self.model)
        # The plant stops matching the controller's model (plant_mismatch.py).
        self.declare_parameter("plant_mismatch", True)
        self.declare_parameter("mismatch_seed", 0)
        if self.get_parameter("plant_mismatch").value:
            boxes = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(self.model.nbody)
                     if (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith("cbox_")]
            summary = apply_plant_mismatch(self.model, np.random.default_rng(
                int(self.get_parameter("mismatch_seed").value)), ARM_JOINT_NAMES, boxes)
            refresh(self.model, self.data)
            self.get_logger().info(f"plant mismatch on: {summary}")
        home_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, HOME_KEYFRAME)
        mujoco.mj_resetDataKeyframe(self.model, self.data, home_id)

        # The home keyframe covers the arm only; put free bodies (the boxes) at their
        # declared poses, not the origin.
        for body_id in range(1, self.model.nbody):
            if self.model.body_jntnum[body_id] != 1:
                continue
            jnt_id = self.model.body_jntadr[body_id]
            if self.model.jnt_type[jnt_id] != mujoco.mjtJoint.mjJNT_FREE:
                continue
            qpos_adr = self.model.jnt_qposadr[jnt_id]
            self.data.qpos[qpos_adr: qpos_adr + 3] = self.model.body_pos[body_id]
            self.data.qpos[qpos_adr + 3: qpos_adr + 7] = [1, 0, 0, 0]

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
        self.latest_torque_cmd = np.zeros(len(ARM_ACTUATOR_NAMES))
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
                    *[f"q{i}" for i in range(1, 8)]])
        self.hold_tau_max = self.model.actuator_ctrlrange[self.actuator_ids, 1]

        # Depth-camera noise on both cameras (depth_noise.py); seeded for repeatable runs.
        self.declare_parameter("depth_noise", True)
        self.declare_parameter("noise_seed", 0)
        self.depth_noise = bool(self.get_parameter("depth_noise").value)
        self.noise_rng = np.random.default_rng(int(self.get_parameter("noise_seed").value))

        self.declare_parameter("render", True)
        self.viewer = None
        if self.get_parameter("render").value:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        self.state_pub = self.create_publisher(JointState, "/sim/joint_states", 10)
        # Wrist load cell, world axes: [fx, fy, fz, step]. Reads the held box's weight.
        self.wrist_force_pub = self.create_publisher(Float64MultiArray, "/sim/wrist_force", 10)
        self.wrist_force_adr = self.model.sensor_adr[self.model.sensor("wrist_force").id]
        self.wrist_site_id = self.model.site("attachment_site").id
        # Depth 1: only the newest command matters.
        self.create_subscription(JointState, "/sim/joint_command", self._on_joint_command, 1)
        self.create_subscription(String, "/task/action", self._on_task_action, 10)
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

        self.container_cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, CONTAINER_CAM_NAME)
        # Wrist camera: a mocap body synced to the gripper at each scan.
        self.container_cam_mount_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, CONTAINER_CAM_MOUNT_ID)
        self.container_cam_mocap_id = (
            self.model.body_mocapid[self.container_cam_mount_id]
            if self.container_cam_mount_id >= 0 else -1
        )

        # Environment actor bodies (dynamic_obstacle_node): a sphere and a person, both mocap.
        self.dynamic_obstacle_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "dynamic_obstacle")
        self.dynamic_obstacle_mocap_id = (
            self.model.body_mocapid[self.dynamic_obstacle_body_id]
            if self.dynamic_obstacle_body_id >= 0 else -1
        )
        person_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "person_obstacle")
        self.person_mocap_id = self.model.body_mocapid[person_body] if person_body >= 0 else -1
        if self.dynamic_obstacle_mocap_id >= 0:
            self.dynamic_obstacle_geom_id = int(
                self.model.body_geomadr[self.dynamic_obstacle_body_id])
            self.create_subscription(
                Float64MultiArray, "/env/dynamic_obstacle", self._on_dynamic_obstacle, 10)
        self._setup_workspace_camera()
        self.container_renderer = None
        self.container_rgb_renderer = None
        # Decision window: pick-up on top, placement below, each the last scan's frame
        # with task_node's decision (/task/decision). The obstacle row is live.
        self.declare_parameter("visualize_pickup", True)
        self.declare_parameter("obstacle_view", True)
        self.declare_parameter("obstacle_view_hz", 2.0)
        self.dashboard_enabled = False
        self.dashboard = {side: {"panels": None, "pose": None, "decision": None, "status": ""}
                          for side in ("pick", "place")}
        self.obstacle_panels = None
        self.supervisor_view = None
        self.supervisor_view_at = None
        # Shared with the window thread (_dashboard_loop).
        self.dashboard_lock = threading.Lock()
        self.dashboard_dirty = threading.Event()
        self.dashboard_write = False
        self.dashboard_stop = False
        self.obstacle_raw = None
        self.obstacle_view = False
        if self.container_cam_id >= 0:
            self.container_renderer = mujoco.Renderer(
                self.model, height=CONTAINER_CAM_HEIGHT, width=CONTAINER_CAM_WIDTH)
            self.container_renderer.enable_depth_rendering()
            # Separate RGB renderer: toggling one renderer's depth mode costs 20-70 ms.
            self.container_rgb_renderer = mujoco.Renderer(
                self.model, height=CONTAINER_CAM_HEIGHT, width=CONTAINER_CAM_WIDTH)
            # Hide the arm's visual meshes (group 2): the gripper dominated the depth image.
            self.container_cam_scene_option = mujoco.MjvOption()
            self.container_cam_scene_option.geomgroup[2] = 0
            self.container_cam_scene_option.geomgroup[DEST_SCAN_MASK_GROUP] = 0  # the held box during a tray scan
            self.occupancy_pub = self.create_publisher(
                Float64MultiArray, "/sim/container_occupancy", 10)
            # [cam pos (3), cam rotation (9), fovy, width, height, depth (row-major)]
            self.container_depth_pub = self.create_publisher(
                Float32MultiArray, "/sim/container_depth", 10)
            self.destination_depth_pub = self.create_publisher(
                Float32MultiArray, "/sim/destination_depth", 10)
            self.create_subscription(Empty, "/sim/scan_container", self._on_scan_container, 10)
            self.destination_occupancy_pub = self.create_publisher(
                Float64MultiArray, "/sim/destination_occupancy", 10)
            self.create_subscription(Empty, "/sim/scan_destination", self._on_scan_destination, 10)
            if self.get_parameter("visualize_pickup").value:
                # Composed on a thread and shown by a separate process: in lockstep, time this
                # thread spends on the window pauses the simulation.
                self.obstacle_view_requested = bool(self.get_parameter("obstacle_view").value)
                ready = threading.Event()
                self.dashboard_thread = threading.Thread(target=self._dashboard_loop, args=(ready,), daemon=True)
                self.dashboard_thread.start()
                ready.wait(12.0)
                if self.dashboard_enabled:
                    self.obstacle_view = self.obstacle_view_requested
                    self.obstacle_view_every = max(1, round(
                        WORKSPACE_CAM_RATE_HZ / float(self.get_parameter("obstacle_view_hz").value)))
                    self.create_subscription(String, "/supervisor/tracks", self._on_supervisor_tracks, 10)
                    self.create_subscription(String, "/task/decision", self._on_task_decision, 10)
                    self._refresh_dashboard()
                else:
                    # e.g. no DISPLAY
                    self.get_logger().warn("visualize_pickup requested but the window could not be created; disabling")

        self.create_timer(LOCKSTEP_POLL_S, self._try_step)

    def _setup_workspace_camera(self):
        """World-fixed obstacle-sensing camera, if the scene has one. Separate depth
        and segmentation renderers (switching modes is slow). Publishes blobs in the
        base frame, no object identities.
        """
        self.declare_parameter("workspace_sensing", True)
        self.workspace_cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, WORKSPACE_CAM_NAME)
        if self.workspace_cam_id < 0 or not self.get_parameter("workspace_sensing").value:
            return
        # The detector uses the calibrated pose (scene.py), not the sim's own; fail if
        # they disagree.
        xml_pos = self.model.cam_pos[self.workspace_cam_id]
        xml_fovy = self.model.cam_fovy[self.workspace_cam_id]
        if (np.abs(xml_pos - WORKSPACE_CAM_POS).max() > 1e-6
                or abs(xml_fovy - WORKSPACE_CAM_FOVY_DEG) > 1e-6):
            raise RuntimeError(
                "workspace_cam in the scene XML disagrees with scene.py's "
                "WORKSPACE_CAM_POS/FOVY -- update one to match the other")
        self.workspace_cam_mat = np.eye(3)  # xyaxes="1 0 0 0 1 0"
        self.workspace_depth_renderer = mujoco.Renderer(
            self.model, height=WORKSPACE_CAM_HEIGHT, width=WORKSPACE_CAM_WIDTH)
        self.workspace_depth_renderer.enable_depth_rendering()
        self.workspace_seg_renderer = mujoco.Renderer(
            self.model, height=WORKSPACE_CAM_HEIGHT, width=WORKSPACE_CAM_WIDTH)
        self.workspace_seg_renderer.enable_segmentation_rendering()
        # Robot self-filter: every geom in the arm's kinematic tree. The held box is
        # added per frame.
        link0 = self.model.body("link0").id
        robot_root = self.model.body_rootid[link0]
        self.robot_geom_lut = self.model.body_rootid[self.model.geom_bodyid] == robot_root
        # Draw the arm as its collision hulls (group 3): ~1 ms vs ~8 ms for the meshes.
        self.workspace_scene_option = mujoco.MjvOption()
        self.workspace_scene_option.geomgroup[2] = 0
        self.workspace_scene_option.geomgroup[3] = 1
        # Best effort: a sensor stream where only the newest frame matters.
        self.workspace_pub = self.create_publisher(
            Float64MultiArray, "/env/workspace_detections", qos_profile_sensor_data)
        self.create_timer(1.0 / WORKSPACE_CAM_RATE_HZ, self._on_workspace_scan)
        self._workspace_proc_ms = []

    def _on_workspace_scan(self):
        """Render the workspace camera and publish blobs: [t_capture_s, n_blobs,
        (x, y, z, radius, n_px, z_max) * n_blobs], base frame.
        """
        t_capture = self.get_clock().now().nanoseconds * 1e-9
        wall0 = time.perf_counter()
        # A late frame means the executor was blocked.
        last = getattr(self, "_last_workspace_frame", None)
        if last is not None and t_capture - last > 0.3:
            self.get_logger().warn(f"workspace_cam frame timer late: {t_capture - last:.2f} s since previous frame")
        self._last_workspace_frame = t_capture
        opt = self.workspace_scene_option
        self.workspace_depth_renderer.update_scene(
            self.data, camera=WORKSPACE_CAM_NAME, scene_option=opt)
        depth = self._sensor(self.workspace_depth_renderer.render())
        self.workspace_seg_renderer.update_scene(
            self.data, camera=WORKSPACE_CAM_NAME, scene_option=opt)
        seg = self.workspace_seg_renderer.render()

        # Lookup table, not np.isin (which sorts).
        is_self = self.robot_geom_lut.copy()
        if self.held_box_name is not None:
            body_id = self.model.body(self.held_box_name).id
            adr = self.model.body_geomadr[body_id]
            is_self[adr: adr + self.model.body_geomnum[body_id]] = True
        # int(): comparing with the pybind enum is ~17 ms vs ~0.1 ms.
        is_geom = seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
        robot_mask = is_self[np.where(is_geom, seg[..., 0], 0)] & is_geom

        result = obstacle_detection.detect_blobs(
            depth, WORKSPACE_CAM_POS, self.workspace_cam_mat, WORKSPACE_CAM_FOVY_DEG,
            FLOOR_Z, FOREGROUND_MIN_HEIGHT_M, robot_mask=robot_mask,
            mask_boxes=AISLE_MASK_BOXES, mask_margin=AISLE_MASK_MARGIN_M,
            min_blob_px=DETECTION_MIN_BLOB_PX, return_masks=self.obstacle_view)
        blobs, masks = result if self.obstacle_view else (result, None)
        self.workspace_pub.publish(Float64MultiArray(
            data=[t_capture, float(len(blobs)), *blobs.ravel().tolist()]))
        self._workspace_frames = getattr(self, "_workspace_frames", 0) + 1
        if masks is not None and self._workspace_frames % self.obstacle_view_every == 0:
            with self.dashboard_lock:
                self.obstacle_raw = (depth.copy(), masks, blobs)
            self._refresh_dashboard(live=True)

        self._workspace_proc_ms.append((time.perf_counter() - wall0) * 1e3)
        if len(self._workspace_proc_ms) == 50:
            ms = np.array(self._workspace_proc_ms)
            self.get_logger().info(
                f"workspace_cam perception cycle: mean {ms.mean():.1f} ms, max {ms.max():.1f} ms "
                f"over {len(ms)} frames")
            self._workspace_proc_ms = []

    def _scan_columns(self, columns_xy, ref_z, floor_z):
        """Render container_cam and infer a top-surface height at each (x, y) in
        columns_xy. On demand only, never per physics step.
        """
        mujoco.mj_forward(self.model, self.data)
        if self.container_cam_mocap_id >= 0:
            # Camera mode="track" does not move the camera in this MuJoCo build, so the
            # mount is synced here; the second forward pass updates cam_xpos.
            self.data.mocap_pos[self.container_cam_mocap_id] = (
                self.data.xpos[self.ee_body_id] + CONTAINER_CAM_MOUNT_OFFSET
            )
            mujoco.mj_forward(self.model, self.data)
        cam_pos = self.data.cam_xpos[self.container_cam_id]
        cam_mat = self.data.cam_xmat[self.container_cam_id].reshape(3, 3)
        fovy = self.model.cam_fovy[self.container_cam_id]
        self.container_renderer.update_scene(self.data, camera=CONTAINER_CAM_NAME, scene_option=self.container_cam_scene_option)
        depth = self._sensor(self.container_renderer.render())
        heights, pixel_map = heightmap.infer_heights_parallax_corrected(
            depth, cam_pos, cam_mat, fovy, CONTAINER_CAM_WIDTH, CONTAINER_CAM_HEIGHT,
            columns_xy, ref_z, floor_z)
        return depth, heights, pixel_map

    def _on_scan_container(self, _msg: Empty):
        depth, heights, pixel_map = self._scan_columns(
            SOURCE_GRID_POINTS, CONTAINER_CAM_REF_Z, FLOOR_Z)
        self._publish_depth(self.container_depth_pub, depth)
        self.occupancy_pub.publish(Float64MultiArray(data=heights.tolist()))

        if self.dashboard_enabled:
            self.container_rgb_renderer.update_scene(
                self.data, camera=CONTAINER_CAM_NAME, scene_option=self.container_cam_scene_option)
            rgb = self.container_rgb_renderer.render()
            self._show_scan("pick", depth, rgb, "container_cam (sensor view)")

    def _on_scan_destination(self, _msg: Empty):
        # Mask the held box: it hangs in the camera's view of the tray.
        held_geom_ids = []
        saved_groups = []
        if self.held_box_name is not None:
            body_id = self.model.body(self.held_box_name).id
            held_geom_ids = list(range(
                self.model.body_geomadr[body_id],
                self.model.body_geomadr[body_id] + self.model.body_geomnum[body_id]))
            saved_groups = [int(self.model.geom_group[g]) for g in held_geom_ids]
            for g in held_geom_ids:
                self.model.geom_group[g] = DEST_SCAN_MASK_GROUP
        try:
            depth, heights, pixel_map = self._scan_columns(
                DEST_GRID_POINTS, DEST_CAM_REF_Z, DEST_FLOOR_Z)
            if self.dashboard_enabled:
                self.container_rgb_renderer.update_scene(
                    self.data, camera=CONTAINER_CAM_NAME, scene_option=self.container_cam_scene_option)
                rgb = self.container_rgb_renderer.render()
                self._show_scan("place", depth, rgb, "container_cam (sensor view, held box masked)")
        finally:
            for g, grp in zip(held_geom_ids, saved_groups):
                self.model.geom_group[g] = grp
        self._publish_depth(self.destination_depth_pub, depth)
        self.destination_occupancy_pub.publish(Float64MultiArray(data=heights.tolist()))

    def _sensor(self, depth):
        return add_depth_noise(depth, self.noise_rng) if self.depth_noise else depth

    def _publish_depth(self, pub, depth):
        """The raw frame and camera pose, as a camera driver gives them; sent before
        the heightmap."""
        cam_pos, cam_mat, fovy = self._camera_pose()
        pub.publish(Float32MultiArray(data=np.concatenate([
            cam_pos, cam_mat.ravel(), [fovy, CONTAINER_CAM_WIDTH, CONTAINER_CAM_HEIGHT],
            depth.ravel()]).astype(np.float32).tolist()))

    def _camera_pose(self):
        """container_cam's current pose and fovy, valid right after a scan."""
        cam_pos = self.data.cam_xpos[self.container_cam_id]
        cam_mat = self.data.cam_xmat[self.container_cam_id].reshape(3, 3)
        fovy = self.model.cam_fovy[self.container_cam_id]
        return cam_pos, cam_mat, fovy

    def _worst_overlap(self, box_name):
        """Deepest overlap (m) of a box with the tray or another box, from world
        axis-aligned bounds; 0 if apart. Also records the facing gaps.
        """
        def bounds(geom_id):
            half = np.abs(self.data.geom_xmat[geom_id].reshape(3, 3)) @ self.model.geom_size[geom_id]
            c = self.data.geom_xpos[geom_id]
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
        self.get_logger().info(
            f"released {mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body)} settled: "
            f"moved {1e3 * np.linalg.norm(d):.1f} mm (dz {1e3 * d[2]:+.1f}), tilt {tilt:.1f} deg, "
            f"yaw {(yaw + 180) % 360 - 180:+.1f} deg")

    def _box_half_extents(self, box_name):
        """A box's (hx, hy, hz) half-extents, from its geom."""
        body_id = self.model.body(box_name).id
        geom_id = self.model.body_geomadr[body_id]
        return self.model.geom_size[geom_id]

    def _show_scan(self, side, depth, rgb, title):
        """Keep a scan's frame and camera pose as the base of that side's half of the
        dashboard; the decision is drawn when task_node publishes it.
        """
        depth_norm = cv2.normalize(heightmap.fill_invalid(depth), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_color = cv2.applyColorMap(255 - depth_norm, cv2.COLORMAP_JET)  # closer = warmer
        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.putText(rgb_bgr, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(depth_color, "depth + decision", (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        # Copies: _camera_pose returns views of mjData.
        pose = tuple(np.array(v, copy=True) if isinstance(v, np.ndarray) else v for v in self._camera_pose())
        with self.dashboard_lock:
            d = self.dashboard[side]
            d["panels"], d["pose"], d["decision"], d["status"] = (rgb_bgr, depth_color), pose, None, ""
        self._refresh_dashboard()

    def _on_supervisor_tracks(self, msg: String):
        view = json.loads(msg.data)
        with self.dashboard_lock:
            self.supervisor_view = view
            self.supervisor_view_at = time.monotonic()

    def _ws_zoom(self):
        """(u0, v0, scale) of the crop the obstacle row shows, and its magnification."""
        f = (WORKSPACE_CAM_HEIGHT / 2.0) / np.tan(np.deg2rad(WORKSPACE_CAM_FOVY_DEG) / 2.0)
        half_w = OBSTACLE_VIEW_HALF_WIDTH_M / (WORKSPACE_CAM_POS[2] - FLOOR_Z) * f
        scale = (WORKSPACE_CAM_WIDTH / 2.0) / half_w
        return WORKSPACE_CAM_WIDTH / 2.0 - half_w, WORKSPACE_CAM_HEIGHT / 2.0 - half_w * 0.75, scale

    def _ws_crop(self, img, interpolation):
        u0, v0, scale = self._ws_zoom()
        w, h = WORKSPACE_CAM_WIDTH / scale, WORKSPACE_CAM_HEIGHT / scale
        crop = img[int(round(v0)):int(round(v0 + h)), int(round(u0)):int(round(u0 + w))]
        return cv2.resize(crop, (WORKSPACE_CAM_WIDTH, WORKSPACE_CAM_HEIGHT), interpolation=interpolation)

    def _ws_px(self, x, y, z):
        """World point -> pixel in the zoomed obstacle panels, or None."""
        uv = heightmap.project_world_point(
            WORKSPACE_CAM_POS, self.workspace_cam_mat, WORKSPACE_CAM_FOVY_DEG,
            WORKSPACE_CAM_WIDTH, WORKSPACE_CAM_HEIGHT, (x, y, z))
        if uv is None:
            return None
        u0, v0, scale = self._ws_zoom()
        return int(round((uv[0] - u0) * scale)), int(round((uv[1] - v0) * scale))

    def _ws_circle(self, img, x, y, z, r, color, thickness=1):
        c, e = self._ws_px(x, y, z), self._ws_px(x + r, y, z)
        if c is not None and e is not None:
            cv2.circle(img, c, max(2, int(round(np.hypot(e[0] - c[0], e[1] - c[1])))), color, thickness,
                       cv2.LINE_AA)
        return c

    def _draw_obstacle_panels(self, depth, masks, blobs):
        """Left: the detector's view (height in grey; ignored regions blue, robot
        green, obstacle red; blobs as columns). Right: depth, with the supervisor's
        tracks added in _obstacle_row.
        """
        z = np.clip(masks["points"][..., 2] - FLOOR_Z, 0.0, 1.0)
        grey = cv2.cvtColor((40 + 180 * z).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        cls = np.zeros(z.shape, dtype=np.uint8)
        cls[masks["ignored"]] = 1
        cls[masks["robot"]] = 2
        cls[masks["foreground"]] = 3
        tinted = cv2.addWeighted(grey, 0.45, _OBSTACLE_LUT[cls], 0.55, 0.0)
        left = self._ws_crop(np.where(cls[..., None] > 0, tinted, grey), cv2.INTER_NEAREST)
        for x, y, _zc, r, _n, top in blobs:
            c = self._ws_circle(left, x, y, top, r, (0, 230, 255), 2)
            if c is not None:
                cv2.putText(left, f"top {top:.2f} m", (c[0] + 6, c[1] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 230, 255), 1, cv2.LINE_AA)
        cv2.putText(left, f"detector: {len(blobs)} blob(s)", (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(left, "red obstacle  green robot  blue ignored", (8, WORKSPACE_CAM_HEIGHT - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
        shown = self._ws_crop(depth, cv2.INTER_LINEAR)
        depth_norm = cv2.normalize(heightmap.fill_invalid(shown), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        right = _JET_LUT[255 - depth_norm]  # closer = warmer
        self.obstacle_panels = (left, right)

    def _obstacle_row(self):
        """The live obstacle window: detector panel, and depth with the supervisor's
        tracks and decision.
        """
        w, h = WORKSPACE_CAM_WIDTH, WORKSPACE_CAM_HEIGHT
        if self.obstacle_panels is None:
            blank = np.zeros((h, w, 3), np.uint8)
            cv2.putText(blank, "OBSTACLES: no workspace camera frame yet", (8, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
            return cv2.hconcat([blank, blank])
        left, right = self.obstacle_panels[0], self.obstacle_panels[1].copy()
        with self.dashboard_lock:
            view, view_at = self.supervisor_view, self.supervisor_view_at
        fresh = view is not None and time.monotonic() - view_at < 1.0
        if not fresh:
            status, colour = "supervisor: no data", (160, 160, 160)
        else:
            for t in view["tracks"]:
                person = t["label"] == "person"
                colour_t = (60, 60, 255) if person else (255, 255, 0)
                c = self._ws_circle(right, t["x"], t["y"], t["top"], t["r"], colour_t, 2)
                if c is None:
                    continue
                ahead = self._ws_px(t["x"] + 0.5 * t["vx"], t["y"] + 0.5 * t["vy"], t["top"])
                if ahead is not None and np.hypot(t["vx"], t["vy"]) > 0.05:
                    cv2.arrowedLine(right, c, ahead, colour_t, 2, cv2.LINE_AA, tipLength=0.3)
                speed = np.hypot(t["vx"], t["vy"])
                cv2.putText(right, f"#{t['id']} {t['label'].upper()} {speed:.2f} m/s"
                                   f"{'' if t['visible'] else ' (hidden)'}", (c[0] + 6, c[1] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, colour_t, 1, cv2.LINE_AA)
            worst = view["worst"]
            if worst is not None:
                tr = next((t for t in view["tracks"] if t["id"] == worst["id"]), None)
                a = self._ws_px(*worst["point_xyz"])
                b = None if tr is None else self._ws_px(tr["x"], tr["y"], worst["point_xyz"][2])
                if a is not None and b is not None:
                    cv2.line(right, a, b, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(right, f"gap {worst['gap']:.2f} m, needs {worst['required']:.2f} m "
                                   f"(at {worst['point']})", (8, h - 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
            if view["hold"]:
                status, colour = "HOLD: waiting for the person", (0, 0, 255)
            elif view["scale"] < 0.999:
                status, colour = f"slowed to {100 * view['scale']:.0f}% speed", (0, 165, 255)
            else:
                status, colour = "clear: full speed", (0, 200, 0)
        cv2.putText(right, "supervisor: tracks + decision", (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(right, (0, h - 20), (w, h), colour, -1)
        cv2.putText(right, status, (8, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        return cv2.hconcat([left, right])

    def _on_task_decision(self, msg: String):
        decision = json.loads(msg.data)
        side = decision.get("side")
        if side in self.dashboard:
            with self.dashboard_lock:
                self.dashboard[side]["decision"] = decision
            self._refresh_dashboard()

    def _refresh_dashboard(self, live=False):
        """Ask the window thread to redraw (any thread). live: only the obstacle row
        changed.
        """
        if not self.dashboard_enabled:
            return
        if not live:
            self.dashboard_write = True
        self.dashboard_dirty.set()

    def _dashboard_loop(self, ready):
        """Window thread: starts window_process, then composes and sends the image
        whenever asked.
        """
        try:
            ctx = multiprocessing.get_context("spawn")
            self.window_conn, child = ctx.Pipe()
            titles = [DASHBOARD_WINDOW] + ([OBSTACLE_WINDOW] if self.obstacle_view_requested else [])
            self.window_proc = ctx.Process(target=window_process.run, args=(child, titles), daemon=True)
            self.window_proc.start()
            self.dashboard_enabled = bool(self.window_conn.poll(10.0) and self.window_conn.recv())
        except Exception as e:
            self.get_logger().warn(f"decision window: {e}")
        ready.set()
        if not self.dashboard_enabled:
            return
        while not self.dashboard_stop:
            if self.dashboard_dirty.wait(0.05):
                self.dashboard_dirty.clear()
                try:
                    self._compose_dashboard()
                except (BrokenPipeError, EOFError, OSError):
                    return
                except Exception as e:
                    self.get_logger().warn(f"decision window redraw failed: {e}")
        try:
            self.window_conn.send(None)
        except OSError:
            pass

    def _compose_dashboard(self):
        """Compose what changed and send it to the window process. Runs on the window
        thread.
        """
        with self.dashboard_lock:
            snapshot = {side: dict(d) for side, d in self.dashboard.items()}
            raw, self.obstacle_raw = self.obstacle_raw, None
            write, self.dashboard_write = self.dashboard_write, False
        if raw is not None:
            self._draw_obstacle_panels(*raw)
            self.window_conn.send((OBSTACLE_WINDOW, self._obstacle_row()))
        if not write:
            return
        halves = []
        for side, name in (("pick", "PICK-UP"), ("place", "PLACEMENT")):
            d = snapshot[side]
            if d["panels"] is None:
                blank = np.zeros((CONTAINER_CAM_HEIGHT, CONTAINER_CAM_WIDTH, 3), np.uint8)
                cv2.putText(blank, f"{name}: waiting for the first scan", (8, CONTAINER_CAM_HEIGHT // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
                halves.append(cv2.hconcat([blank, blank]))
                continue
            rgb_bgr, depth_color = (img.copy() for img in d["panels"])
            dec = d["decision"]
            if dec is not None:
                # Outlines at their own heights (perspective): placed boxes at their tops,
                # the choice at the surface it rests on.
                for other in dec.get("others", []):
                    _draw_world_rect(depth_color, *other, *d["pose"], color=(200, 200, 200), thickness=1)
                if dec.get("rect"):
                    _draw_world_rect(depth_color, *dec["rect"], *d["pose"])
                label = dec.get("label", "")
            else:
                label = "deciding..."
            cv2.putText(depth_color, f"{name}: {label}", (8, CONTAINER_CAM_HEIGHT - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            if d["status"]:
                cv2.putText(depth_color, d["status"], (8, CONTAINER_CAM_HEIGHT - 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            halves.append(cv2.hconcat([rgb_bgr, depth_color]))
        composite = cv2.vconcat(halves)
        self.window_conn.send((DASHBOARD_WINDOW, composite))
        if os.environ.get("MANIPOPT_TELEMETRY_DIR"):
            # Latest window content, for checking a run afterwards.
            cv2.imwrite(str(Path(os.environ["MANIPOPT_TELEMETRY_DIR"]) / "decisions.png"), composite)

    def _on_joint_command(self, msg: JointState):
        self.latest_torque_cmd = np.array(msg.effort)
        # A command without a state stamp (the MoveIt bridge) answers the latest state.
        self.cmd_src_step = int(msg.header.frame_id) if msg.header.frame_id else self.step_count
        self.hold_qpos = None  # fresh command: drop the hold target
        self._try_step()

    def _on_dynamic_obstacle(self, msg: Float64MultiArray):
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
                self.data.mocap_pos[self.person_mocap_id] = (x, y, 0.0)
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
        # "pick_at x y z": grasp whichever box is at that sensed pose. "place_held":
        # release the held box.
        parts = msg.data.split()
        if parts and parts[0] == "pick_at" and len(parts) == 4:
            target = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            box_name = self._resolve_box_at(target)
            if box_name is None:
                self.get_logger().warn(f"pick_at {target}: no box found near this pose")
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
                f"picked up {box_name} (resolved from sensed pose {target}); "
                f"TCP {1e3 * (self.data.site_xpos[self.tcp_site_id][2] - top):+.1f} mm above its true top; "
                f"true size {2e3 * hx:.0f} x {2e3 * hy:.0f} x {2e3 * hz:.0f} mm; "
                f"since last event {self._event_report(self.model.body(box_name).id)}")
            with self.dashboard_lock:
                self.dashboard["pick"]["status"] = f"-> grasped {box_name}"
            self._refresh_dashboard()
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
            self.last_placed_body = body
            self.release_check = (body, self.step_count + 50, self.data.xpos[body].copy(),
                                  self.data.xmat[body].reshape(3, 3).copy())
            self.held_box_name = None
            with self.dashboard_lock:
                self.dashboard["place"]["status"] = f"-> placed {box_name}"
            self._refresh_dashboard()
        elif parts and parts[0] == "place_aborted" and len(parts) == 1:
            self.get_logger().info(f"place aborted; during the attempt {self._event_report(-1)}")
        elif parts and parts[0] == "pushed" and len(parts) == 1:
            if self.last_placed_body is None:
                return
            body = self.last_placed_body
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body)
            mat = self.data.xmat[body].reshape(3, 3)
            tilt = np.degrees(np.arccos(np.clip(mat[2, 2], -1.0, 1.0)))
            yaw = np.degrees(np.arctan2(mat[1, 0], mat[0, 0]))
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

    def _resolve_box_at(self, target_xyz, xy_tol=0.02, z_tol=0.08):
        """Nearest cbox_* in 3D (boxes stack, so z matters). The target is the box top
        and xpos its centre, so z_tol covers the tallest half-height (0.06 m).
        """
        mujoco.mj_forward(self.model, self.data)
        best_name, best_dist = None, None
        for body_id in range(1, self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if not name or not name.startswith("cbox_"):
                continue
            dx, dy, dz = self.data.xpos[body_id] - target_xyz
            if abs(dx) > xy_tol or abs(dy) > xy_tol or abs(dz) > z_tol:
                continue
            dist = float(np.linalg.norm([dx, dy, dz]))
            if best_dist is None or dist < best_dist:
                best_name, best_dist = name, dist
        return best_name

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

    def _step(self):
        stale = self.cmd_src_step < 0 or self.step_count - self.cmd_src_step > COMMAND_STALE_STEPS
        if stale:
            if self.hold_qpos is None:
                self.hold_qpos = self.data.qpos[self.joint_qpos_adr].copy()
            q = self.data.qpos[self.joint_qpos_adr]
            qdot = self.data.qvel[self.joint_qvel_adr]
            tau = (
                self.data.qfrc_bias[self.joint_qvel_adr]
                + HOLD_KP * (self.hold_qpos - q)
                - HOLD_KD * qdot
            )
            self.data.ctrl[self.actuator_ids] = np.clip(tau, -self.hold_tau_max, self.hold_tau_max)
        else:
            self.data.ctrl[self.actuator_ids] = self.latest_torque_cmd
        if self.telemetry.enabled:
            now = time.perf_counter()
            period_ms = (now - self.last_step_wall) * 1e3 if self.last_step_wall else 0.0
            self.last_step_wall = now
            qd = self.data.qvel[self.joint_qvel_adr]
            self.telemetry.row(
                round(self.data.time, 3), self.step_count, round(period_ms, 2), int(stale),
                self.step_count - self.cmd_src_step, round(float(np.linalg.norm(qd)), 4),
                *np.round(qd, 4), *np.round(self.data.ctrl[self.actuator_ids], 3),
                *np.round(self.data.qpos[self.joint_qpos_adr], 5))
        mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)
        self.step_count += 1
        if self.held_box_name is not None:
            # Deepest overlap while carrying, reported on place.
            depth, other = self._worst_overlap(self.held_box_name)
            if depth > self.carry_overlap[0]:
                self.carry_overlap = (depth, other)
        self._log_contacts()
        if self.release_check is not None and self.step_count >= self.release_check[1]:
            self._report_release()
        if self.viewer is not None:
            if not self.viewer.is_running():
                # Window closed: keep simulating headless.
                self.viewer = None
            else:
                self._update_markers()
                self.viewer.sync()
        self._publish_state()

    def _update_markers(self):
        # Marker at the TCP (what the controller tracks).
        mujoco.mjv_initGeom(
            self.viewer.user_scn.geoms[0],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([MARKER_RADIUS_M, 0, 0]),
            pos=self.data.site_xpos[self.tcp_site_id].copy(),
            mat=np.eye(3).flatten(),
            rgba=np.array([0.0, 0.0, 1.0, 0.8], dtype=np.float32),
        )
        self.viewer.user_scn.ngeom = 1

    def _publish_state(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = str(self.step_count)
        self.last_state_sent = time.perf_counter()
        msg.position = self.data.qpos[self.joint_qpos_adr].tolist()
        msg.velocity = self.data.qvel[self.joint_qvel_adr].tolist()
        self.state_pub.publish(msg)
        f = self.data.site_xmat[self.wrist_site_id].reshape(3, 3) @ self.data.sensordata[
            self.wrist_force_adr:self.wrist_force_adr + 3]
        self.wrist_force_pub.publish(Float64MultiArray(data=[*f.tolist(), float(self.step_count)]))


def main():
    rclpy.init()
    node = MujocoSimNode()
    try:
        rclpy.spin(node)
    finally:
        node.telemetry.flush()
        if node.viewer is not None:
            node.viewer.close()
        if node.dashboard_enabled:
            node.dashboard_stop = True
            node.dashboard_thread.join(timeout=1.0)
            node.window_proc.join(timeout=1.0)
        node.destroy_node()
        # rclpy's SIGINT handler already shut the context down; a second shutdown
        # throws (and crashed the viewer thread).
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
