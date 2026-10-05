"""The plant's cameras and windows, in their own process: renders the wrist
camera's scans and the overhead camera (with obstacle detection) from the scene
state mujoco_sim_node publishes, and shows the decision and obstacle windows and
the MuJoCo viewer. Kept apart so drawing never delays the physics. Renders in
the room (MuJoCo's world); camera poses go out in the arm frame.

See docs/implementation_notes.md#sim_sensors_nodepy.
"""
import json
from array import array
import multiprocessing
import os
import threading
import time
from pathlib import Path

# Cap the passive viewer's render thread at vsync; must be set before its GL
# context exists (docs/design_notes.md).
os.environ.setdefault("vblank_mode", "1")
# The windows' Qt6 on WSLg's Wayland took ~3 cores per window, starving the plant and the
# controller; on X11 (xcb) next to nothing. Inherited by window_process.
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import numpy as np
import cv2
import mujoco
import mujoco.viewer
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Empty, Float32MultiArray, Float64, Float64MultiArray, String, UInt8MultiArray

from pick_place_common import lidar_sim, plant_conditions
from pick_place_common.scene import (
    CONTAINER_CAM_HEIGHT, CONTAINER_CAM_POS_TCP, CONTAINER_CAM_ROT_TCP,
    CONTAINER_CAM_FOVY_DEG, CONTAINER_CAM_NAME, CONTAINER_CAM_WIDTH, DEST_CAM_REF_Z,
    DEST_FLOOR_Z, DEST_GRID_POINTS, FLOOR_Z, pick_zone,
    ROOM_FLOOR_Z, WORKSPACE_CAM_FOVY_DEG, WORKSPACE_CAM_HEIGHT,
    WORKSPACE_CAM_NAME, WORKSPACE_CAM_POS, WORKSPACE_CAM_RATE_HZ,
    WORKSPACE_CAM_WIDTH, LIDAR_RATE_HZ, LIDAR_SCANNERS, PICK_TABLE_BOUNDS, PLACE_TABLE_BOUNDS, CHASSIS_BOUNDS,
)
from perception import heightmap, obstacle_detection
from pick_place_common.depth_noise import add_depth_noise
from pick_place_common.mujoco_sim_node import (
    DEFAULT_CELL_FILE, HOME_KEYFRAME, TCP_SITE_NAME, load_scene_model, shift_tray)
from pick_place_common import frames, window_process

DASHBOARD_WINDOW = "pick & place decisions"
OBSTACLE_WINDOW = "obstacle detection (live)"
OBSTACLE_VIEW_HALF_WIDTH_M = 1.4  # floor shown each side of the image centre in the obstacle view
# Obstacle view pixel classes: none, ignored, robot, obstacle.
_OBSTACLE_LUT = np.array([[0, 0, 0], [200, 90, 20], [60, 170, 60], [40, 40, 230]], dtype=np.uint8)
# JET as a lookup table: cv2.applyColorMap costs ~4 ms per frame here.
_JET_LUT = cv2.applyColorMap(np.arange(256, dtype=np.uint8)[:, None], cv2.COLORMAP_JET)[:, 0, :]
MARKER_RADIUS_M = 0.02
VIEWER_PERIOD_S = 0.02  # one frame per plant tick
OBSTACLE_LIDAR_VIEW_HZ = 5.0
LIDAR_VIEW_BOUNDS = ((-2.6, 2.6), (-3.2, 0.7))  # x, y shown in the lidar panel


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


class SimSensorsNode(Node):
    def __init__(self):
        super().__init__("sim_sensors_node")
        cv2.setNumThreads(1)  # its worker threads spin between calls (window_process.py)
        self.declare_parameter("cell_file", DEFAULT_CELL_FILE)
        self.declare_parameter("layout", "cell")
        self.model = load_scene_model(self.get_parameter("cell_file").value,
                                      layout=self.get_parameter("layout").value)
        _b, self.pick_grid, _s, pile_top = pick_zone(self.get_parameter("layout").value == "stations")  # task_node's
        # Same geometry as the plant: the tray shift moves geoms, not bodies.
        self.declare_parameter("tray_seed", -1)
        tray_seed = int(self.get_parameter("tray_seed").value)
        if tray_seed >= 0:
            shift_tray(self.model, np.random.default_rng(tray_seed))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key(HOME_KEYFRAME).id)
        frames.free_bodies_at_rest(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.snap = self.data  # the state rendered from (set by _snapshot)
        self.tcp_site_id = self.model.site(TCP_SITE_NAME).id
        # The latest /sim/scene_state: (step, held body, sphere radius, qpos, mocap_pos, mocap_quat).
        self.scene_lock = threading.Lock()
        self.scene = None
        self.scene_stamp = None
        self.held_body = -1
        # Own node and thread: camera renders on the main executor held updates up to 0.35 s.
        self.scene_node = rclpy.create_node("sim_sensors_scene")
        self.scene_node.create_subscription(Float64MultiArray, "/sim/scene_state", self._on_scene_state, 1)
        sphere = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "dynamic_obstacle")
        self.sphere_geom_id = int(self.model.body_geomadr[sphere]) if sphere >= 0 else -1

        # Depth-camera noise on both cameras (depth_noise.py); seeded for repeatable runs.
        self.declare_parameter("depth_noise", True)
        self.declare_parameter("noise_seed", 0)
        self.depth_noise = bool(self.get_parameter("depth_noise").value)
        self.noise_rng = np.random.default_rng(int(self.get_parameter("noise_seed").value))
        self.pile_top = pile_top  # the method's last pile scan (/task/pile_top)
        self.create_subscription(Float64, "/task/pile_top", self._on_pile_top, 10)
        self.create_subscription(String, "/task/action", self._on_task_action, 10)

        self.container_cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, CONTAINER_CAM_NAME)
        if self.container_cam_id >= 0:
            self._check_container_cam()
        self._setup_workspace_camera()
        self._setup_lidars()
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
        self.lidar_dirty = False
        self.obstacle_view = False
        if self.container_cam_id >= 0:
            self.container_renderer = mujoco.Renderer(
                self.model, height=CONTAINER_CAM_HEIGHT, width=CONTAINER_CAM_WIDTH)
            self.container_renderer.enable_depth_rendering()
            # Separate RGB renderer: toggling one renderer's depth mode costs 20-70 ms.
            self.container_rgb_renderer = mujoco.Renderer(
                self.model, height=CONTAINER_CAM_HEIGHT, width=CONTAINER_CAM_WIDTH)
            self.container_cam_scene_option = mujoco.MjvOption()
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
                # Composed on a thread and shown by a separate process.
                self.obstacle_view_requested = bool(self.get_parameter("obstacle_view").value)
                ready = threading.Event()
                self.dashboard_thread = threading.Thread(target=self._dashboard_loop, args=(ready,), daemon=True)
                self.dashboard_thread.start()
                ready.wait(12.0)
                if self.dashboard_enabled:
                    self.obstacle_view = self.obstacle_view_requested
                    self.create_subscription(String, "/supervisor/tracks", self._on_supervisor_tracks, 10)
                    self.create_subscription(String, "/task/decision", self._on_task_decision, 10)
                    self._refresh_dashboard()
                else:
                    # e.g. no DISPLAY
                    self.get_logger().warn("visualize_pickup requested but the window could not be created; disabling")

        self.declare_parameter("render", True)
        self.viewer = None
        self.viewer_stop = False
        if self.get_parameter("render").value:
            self.view_data = mujoco.MjData(self.model)
            mujoco.mj_copyData(self.view_data, self.model, self.data)
            self.viewer = mujoco.viewer.launch_passive(self.model, self.view_data)
            threading.Thread(target=self._viewer_loop, daemon=True).start()

    def _on_scene_state(self, msg: Float64MultiArray):
        with self.scene_lock:
            self.scene = np.array(msg.data)

    def _apply_scene(self, data):
        """Copy the latest scene state into data (positions only); False if none yet."""
        with self.scene_lock:
            scene = self.scene
        if scene is None:
            return False
        nq, nm = self.model.nq, self.model.nmocap
        self.held_body = int(scene[1])
        if self.sphere_geom_id >= 0 and scene[2] > 0:
            self.model.geom_size[self.sphere_geom_id, 0] = scene[2]
        data.qpos[:] = scene[3:3 + nq]
        data.mocap_pos[:] = scene[3 + nq:3 + nq + 3 * nm].reshape(nm, 3)
        data.mocap_quat[:] = scene[3 + nq + 3 * nm:3 + nq + 7 * nm].reshape(nm, 4)
        self.scene_stamp = scene[3 + nq + 7 * nm] if len(scene) > 3 + nq + 7 * nm else None
        mujoco.mj_forward(self.model, data)
        return True

    def _snapshot(self):
        """The simulation state to render from: the latest scene state."""
        self._apply_scene(self.data)
        return self.data

    def _on_task_action(self, msg: String):
        if not self.dashboard_enabled:
            return
        word = msg.data.split()[0] if msg.data else ""
        status = {"pick_at": ("pick", "-> grasped"), "place_held": ("place", "-> placed")}.get(word)
        if status:
            with self.dashboard_lock:
                self.dashboard[status[0]]["status"] = status[1]
            self._refresh_dashboard()

    def _viewer_loop(self):
        """Refresh the MuJoCo viewer from the scene state."""
        next_t = time.monotonic()
        while not self.viewer_stop and self.viewer is not None:
            if not self.viewer.is_running():
                self.viewer = None  # window closed
                return
            with self.viewer.lock():
                self._apply_scene(self.view_data)
            mujoco.mjv_initGeom(
                self.viewer.user_scn.geoms[0], type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=np.array([MARKER_RADIUS_M, 0, 0]), pos=self.view_data.site_xpos[self.tcp_site_id].copy(),
                mat=np.eye(3).flatten(), rgba=np.array([0.0, 0.0, 1.0, 0.8], dtype=np.float32))
            self.viewer.user_scn.ngeom = 1  # marker at the TCP (what the controller tracks)
            if "spill" in self.conditions:
                (x0, x1), (y0, y1) = plant_conditions.SPILL_RECT
                mujoco.mjv_initGeom(
                    self.viewer.user_scn.geoms[1], type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=np.array([(x1 - x0) / 2, (y1 - y0) / 2, 0.001]),
                    pos=np.array([(x0 + x1) / 2, (y0 + y1) / 2, 0.001]), mat=np.eye(3).flatten(),
                    rgba=np.array([0.3, 0.6, 1.0, 0.45], dtype=np.float32))
                self.viewer.user_scn.ngeom = 2
            self.viewer.sync()
            now = time.monotonic()
            next_t = max(next_t + VIEWER_PERIOD_S, now)
            time.sleep(next_t - now)

    def _check_container_cam(self):
        """The wrist camera in the XML must be where the method's calibration
        (scene.py) says, relative to the TCP."""
        mujoco.mj_forward(self.model, self.data)
        c = self.container_cam_id
        tcp_r = self.data.site_xmat[self.tcp_site_id].reshape(3, 3)
        pos = tcp_r.T @ (self.data.cam_xpos[c] - self.data.site_xpos[self.tcp_site_id])
        rot = tcp_r.T @ self.data.cam_xmat[c].reshape(3, 3)
        if (np.abs(pos - CONTAINER_CAM_POS_TCP).max() > 1e-6 or np.abs(rot - CONTAINER_CAM_ROT_TCP).max() > 1e-6
                or abs(self.model.cam_fovy[c] - CONTAINER_CAM_FOVY_DEG) > 1e-6):
            raise RuntimeError("container_cam in panda_robot.xml disagrees with scene.py's CONTAINER_CAM_*")

    def _setup_workspace_camera(self):
        """World-fixed ceiling camera, if the scene has one and it is on: raw depth
        with noise at WORKSPACE_CAM_RATE_HZ, as a driver gives it
        (/env/workspace_depth [t_capture_s, w, h, depth row-major]); the detection is
        the method's (camera_detection_node).
        """
        self.declare_parameter("workspace_sensing", True)
        self.workspace_cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, WORKSPACE_CAM_NAME)
        self.workspace_on = self.workspace_cam_id >= 0 and bool(self.get_parameter("workspace_sensing").value)
        self.workspace_depths = {}  # recent frames by capture time, for the obstacle window
        self.camera_debug = None
        if not self.workspace_on:
            return
        # The detector uses the calibrated pose (scene.py, the arm frame with the base parked
        # at the cell), not the sim's own; fail if they disagree.
        xml_pos, xml_rot = frames.to_arm(
            frames.arm_base_pose(self.model, self.data), self.data.cam_xpos[self.workspace_cam_id],
            self.data.cam_xmat[self.workspace_cam_id].reshape(3, 3))
        xml_fovy = self.model.cam_fovy[self.workspace_cam_id]
        if (np.abs(xml_pos - WORKSPACE_CAM_POS).max() > 1e-3 or np.abs(xml_rot - np.eye(3)).max() > 1e-3
                or abs(xml_fovy - WORKSPACE_CAM_FOVY_DEG) > 1e-6):
            raise RuntimeError(
                "workspace_cam in the scene XML disagrees with scene.py's "
                "WORKSPACE_CAM_POS/FOVY: update one to match the other")
        self.workspace_cam_mat = np.eye(3)  # xyaxes="1 0 0 0 1 0"
        self.workspace_depth_renderer = mujoco.Renderer(
            self.model, height=WORKSPACE_CAM_HEIGHT, width=WORKSPACE_CAM_WIDTH)
        self.workspace_depth_renderer.enable_depth_rendering()
        # Draw the arm as its collision hulls (group 3): ~1 ms vs ~8 ms for the meshes.
        self.workspace_scene_option = mujoco.MjvOption()
        self.workspace_scene_option.geomgroup[2] = 0
        self.workspace_scene_option.geomgroup[3] = 1
        # Newest frame only, but reliable: best effort lost most of these 0.6 MB frames.
        self.workspace_pub = self.create_publisher(Float64MultiArray, "/env/workspace_depth", 1)
        self.create_subscription(UInt8MultiArray, "/perception/camera_debug", self._on_camera_debug, 1)
        self.create_subscription(Float64MultiArray, "/perception/camera_detections", self._on_camera_blobs,
                                 qos_profile_sensor_data)
        self.camera_blobs = np.zeros((0, 6))
        self.create_timer(1.0 / WORKSPACE_CAM_RATE_HZ, self._on_workspace_scan)
        self._workspace_proc_ms = []

    def _setup_lidars(self):
        """Safety-lidar scans at LIDAR_RATE_HZ on the scene thread (camera renders on
        the main executor must not delay them): /env/lidar_scan, one message per
        scanner, [t_capture_s, index, x, y, z, yaw, angle_min, angle_step, n, ranges (nan: no return)]."""
        self.declare_parameter("lidar", True)
        self.declare_parameter("conditions", "none")
        self.conditions = plant_conditions.parse(self.get_parameter("conditions").value)
        self.dropouts = None
        self.lidar_last = [None] * len(LIDAR_SCANNERS)
        self.lidar_people = []
        self.lidar_foreground = np.zeros((0, 2))
        self.lidar_frames = 0
        if not self.get_parameter("lidar").value:
            return
        self.lidar_rng = np.random.default_rng([int(self.get_parameter("noise_seed").value), 1])
        self.lidar_data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.lidar_data)
        lidar_sim.check_sites(self.model, self.lidar_data)
        self.lidar_pub = self.scene_node.create_publisher(Float64MultiArray, "/env/lidar_scan", 10)
        if "dropout" in self.conditions:
            # The localization's copy (the launch remaps slam_node to it), with gaps: a network hiccup
            # between the scanners and the navigation computer; the safety fields keep theirs.
            self.dropouts = plant_conditions.Dropouts(np.random.default_rng([int(self.get_parameter("noise_seed").value), 5]))
            self.lossy_pub = self.scene_node.create_publisher(Float64MultiArray, "/env/lidar_scan_lossy", 10)
        self.scene_node.create_timer(1.0 / LIDAR_RATE_HZ, self._on_lidar_scan)
        self.create_subscription(Float64MultiArray, "/perception/lidar_detections", self._on_lidar_people,
                                 qos_profile_sensor_data)
        self.create_subscription(Float64MultiArray, "/perception/lidar_foreground", self._on_lidar_foreground,
                                 qos_profile_sensor_data)

    def _on_lidar_scan(self):
        if not self._apply_scene(self.lidar_data):
            return
        # Stamped with the plant state's time, as a scanner stamps its own measurement.
        t = self.scene_stamp if self.scene_stamp is not None else self.get_clock().now().nanoseconds * 1e-9
        a = lidar_sim.BEAM_ANGLES
        lost = self.dropouts is not None and self.dropouts.dropped(t)
        for i, pose in enumerate(LIDAR_SCANNERS):
            r = lidar_sim.scan(self.model, self.lidar_data, i, self.lidar_rng)
            msg = Float64MultiArray(data=[t, float(i), *pose, float(a[0]), float(a[1] - a[0]), float(len(r)), *r.tolist()])
            self.lidar_pub.publish(msg)
            if self.dropouts is not None and not lost:
                self.lossy_pub.publish(msg)
            self.lidar_last[i] = r
        self.lidar_frames += 1
        if self.obstacle_view and self.lidar_frames % max(1, round(LIDAR_RATE_HZ / OBSTACLE_LIDAR_VIEW_HZ)) == 0:
            with self.dashboard_lock:
                self.lidar_dirty = True
            self._refresh_dashboard(live=True)

    def _on_lidar_people(self, msg: Float64MultiArray):
        n = int(msg.data[1])
        with self.dashboard_lock:
            self.lidar_people = np.array(msg.data[2:2 + 4 * n]).reshape(n, 4)

    def _on_lidar_foreground(self, msg: Float64MultiArray):
        n = int(msg.data[1])
        with self.dashboard_lock:
            self.lidar_foreground = np.array(msg.data[2:2 + 2 * n]).reshape(n, 2)

    def _draw_lidar_panel(self):
        """Top-down lidar view: returns grey, the detector's foreground orange,
        people red (centre and radius), scanners yellow, furniture outlined."""
        w, h = WORKSPACE_CAM_WIDTH, WORKSPACE_CAM_HEIGHT
        img = np.full((h, w, 3), 25, np.uint8)
        (x0, x1), (y0, y1) = LIDAR_VIEW_BOUNDS
        k = min(w / (x1 - x0), h / (y1 - y0))

        def px(x, y):
            return int(round((x - x0) * k)), int(round((y1 - y) * k))
        for (bx0, bx1), (by0, by1) in (CHASSIS_BOUNDS, PICK_TABLE_BOUNDS, PLACE_TABLE_BOUNDS):
            cv2.rectangle(img, px(bx0, by1), px(bx1, by0), (110, 110, 110), 1)
        with self.dashboard_lock:
            people, fg, last = self.lidar_people, self.lidar_foreground, list(self.lidar_last)
        for pose, r in zip(LIDAR_SCANNERS, last):
            if r is None:
                continue
            a = pose[3] + lidar_sim.BEAM_ANGLES
            for x, y in zip(pose[0] + r * np.cos(a), pose[1] + r * np.sin(a)):
                if np.isfinite(x):
                    cv2.circle(img, px(x, y), 1, (150, 150, 150), -1)
        for x, y in fg:
            cv2.circle(img, px(x, y), 2, (0, 140, 255), -1)
        for x, y, r, _n in people:
            cv2.circle(img, px(x, y), max(2, int(round(r * k))), (60, 60, 255), 2, cv2.LINE_AA)
        for x, y, _z, _yaw in LIDAR_SCANNERS:
            cv2.rectangle(img, px(x - 0.04, y + 0.04), px(x + 0.04, y - 0.04), (0, 220, 255), -1)
        with self.dashboard_lock:
            view, view_at = self.supervisor_view, self.supervisor_view_at
        status, colour = "supervisor: no data", (160, 160, 160)
        if view is not None and time.monotonic() - view_at < 1.0:
            for t in view["tracks"]:
                if t.get("source") != "lidar":
                    continue
                c = px(t["x"], t["y"])
                cv2.circle(img, c, max(2, int(round(t["r"] * k))), (255, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(img, f"{np.hypot(t['vx'], t['vy']):.1f} m/s", (c[0] + 6, c[1] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1, cv2.LINE_AA)
            if view["hold"]:
                status, colour = "HOLD: waiting for the person", (0, 0, 255)
            elif view["scale"] < 0.999:
                status, colour = f"slowed to {100 * view['scale']:.0f}% speed", (0, 165, 255)
            else:
                status, colour = "clear: full speed", (0, 200, 0)
        cv2.putText(img, f"lidar: {len(people)} person(s); cyan: protective radius", (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, "grey return  orange foreground  red person", (8, h - 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(img, (0, h - 20), (w, h), colour, -1)
        cv2.putText(img, status, (8, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        return img

    def _on_workspace_scan(self):
        """Render the ceiling camera and publish its raw depth."""
        t_capture = self.get_clock().now().nanoseconds * 1e-9
        wall0 = time.perf_counter()
        # A late frame means the executor was blocked.
        last = getattr(self, "_last_workspace_frame", None)
        if last is not None and t_capture - last > 0.3:
            self.get_logger().warn(f"workspace_cam frame timer late: {t_capture - last:.2f} s since previous frame")
        self._last_workspace_frame = t_capture
        self.workspace_depth_renderer.update_scene(
            self._snapshot(), camera=WORKSPACE_CAM_NAME, scene_option=self.workspace_scene_option)
        depth = self._sensor(self.workspace_depth_renderer.render())
        h, w = depth.shape
        self.workspace_pub.publish(Float64MultiArray(data=array("d", np.concatenate(
            [[t_capture, w, h], depth.ravel()]).astype(np.float64).tobytes())))
        if self.obstacle_view:
            with self.dashboard_lock:
                self.workspace_depths[t_capture] = depth
                for t in sorted(self.workspace_depths)[:-5]:
                    del self.workspace_depths[t]
        self._workspace_proc_ms.append((time.perf_counter() - wall0) * 1e3)
        if len(self._workspace_proc_ms) == 50:
            ms = np.array(self._workspace_proc_ms)
            self.get_logger().info(
                f"workspace_cam render cycle: mean {ms.mean():.1f} ms, max {ms.max():.1f} ms "
                f"over {len(ms)} frames")
            self._workspace_proc_ms = []

    def _on_camera_blobs(self, msg: Float64MultiArray):
        n = int(msg.data[1])
        with self.dashboard_lock:
            self.camera_blobs = np.array(msg.data[2:2 + 6 * n]).reshape(n, 6)

    def _on_camera_debug(self, msg: UInt8MultiArray):
        """The detector's classes for a frame: draw it with that frame's depth."""
        b = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        t = float(np.frombuffer(b[:8].tobytes(), dtype=np.float64)[0])
        w, h = np.frombuffer(b[8:12].tobytes(), dtype=np.uint16)
        cls = b[12:12 + int(w) * int(h)].reshape(int(h), int(w))
        with self.dashboard_lock:
            depth = self.workspace_depths.get(t)
            blobs = self.camera_blobs
            if depth is not None:
                self.obstacle_raw = (depth, cls, blobs)
        if depth is not None:
            self._refresh_dashboard(live=True)

    def _scan_columns(self, columns_xy, ref_z, floor_z):
        """Render container_cam and infer a top-surface height at each (x, y) in
        columns_xy. On demand only, never per physics step.
        """
        snap = self._snapshot()
        cam_pos, cam_mat, fovy = self._camera_pose()
        self.container_renderer.update_scene(snap, camera=CONTAINER_CAM_NAME, scene_option=self.container_cam_scene_option)
        depth = self._sensor(self.container_renderer.render())
        heights, pixel_map = heightmap.infer_heights_parallax_corrected(
            depth, cam_pos, cam_mat, fovy, CONTAINER_CAM_WIDTH, CONTAINER_CAM_HEIGHT,
            columns_xy, ref_z, floor_z)
        return depth, heights, pixel_map

    def _on_pile_top(self, msg: Float64):
        self.pile_top = float(msg.data)

    def _on_scan_container(self, _msg: Empty):
        depth, heights, pixel_map = self._scan_columns(
            self.pick_grid, (FLOOR_Z + self.pile_top) / 2.0, FLOOR_Z)
        self._publish_depth(self.container_depth_pub, depth)
        self.occupancy_pub.publish(Float64MultiArray(data=heights.tolist()))

        if self.dashboard_enabled:
            self.container_rgb_renderer.update_scene(
                self.snap, camera=CONTAINER_CAM_NAME, scene_option=self.container_cam_scene_option)
            rgb = self.container_rgb_renderer.render()
            self._show_scan("pick", depth, rgb, "container_cam (sensor view)")

    def _on_scan_destination(self, _msg: Empty):
        depth, heights, pixel_map = self._scan_columns(
            DEST_GRID_POINTS, DEST_CAM_REF_Z, DEST_FLOOR_Z)
        if self.dashboard_enabled:
            self.container_rgb_renderer.update_scene(
                self.snap, camera=CONTAINER_CAM_NAME, scene_option=self.container_cam_scene_option)
            rgb = self.container_rgb_renderer.render()
            self._show_scan("place", depth, rgb, "container_cam (sensor view)")
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
        """container_cam's pose (arm frame) and fovy at the last scan."""
        cam_pos, cam_mat = frames.to_arm(
            frames.arm_base_pose(self.model, self.snap), self.snap.cam_xpos[self.container_cam_id],
            self.snap.cam_xmat[self.container_cam_id].reshape(3, 3))
        fovy = self.model.cam_fovy[self.container_cam_id]
        return cam_pos, cam_mat, fovy

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
        half_w = OBSTACLE_VIEW_HALF_WIDTH_M / (WORKSPACE_CAM_POS[2] - ROOM_FLOOR_Z) * f
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

    def _draw_obstacle_panels(self, depth, cls, blobs):
        """Left: the detector's view (height in grey; ignored regions blue, robot
        green, obstacle red; blobs as columns). Right: depth, with the supervisor's
        tracks added in _obstacle_row.
        """
        points = obstacle_detection.unproject_depth(depth, WORKSPACE_CAM_POS, self.workspace_cam_mat,
                                                    WORKSPACE_CAM_FOVY_DEG)
        z = np.clip((points[..., 2] - ROOM_FLOOR_Z) / 2.0, 0.0, 1.0)
        grey = cv2.cvtColor((40 + 180 * z).astype(np.uint8), cv2.COLOR_GRAY2BGR)
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
            lidar, self.lidar_dirty = self.lidar_dirty, False
            write, self.dashboard_write = self.dashboard_write, False
        if raw is not None:
            self._draw_obstacle_panels(*raw)
        if raw is not None or lidar:
            panels = [self._obstacle_row()] if self.workspace_on else []
            if self.get_parameter("lidar").value:
                panels.append(self._draw_lidar_panel())
            row = cv2.hconcat(panels)
            self.window_conn.send((OBSTACLE_WINDOW, row))
            now = time.monotonic()
            if os.environ.get("MANIPOPT_TELEMETRY_DIR") and now - getattr(self, "_obstacle_png_at", 0.0) > 1.0:
                self._obstacle_png_at = now  # latest window content, for checking a run afterwards
                cv2.imwrite(str(Path(os.environ["MANIPOPT_TELEMETRY_DIR"]) / "obstacle.png"), row)
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


def main():
    rclpy.init()
    node = SimSensorsNode()
    scene_executor = SingleThreadedExecutor()
    scene_executor.add_node(node.scene_node)
    threading.Thread(target=scene_executor.spin, daemon=True).start()
    try:
        rclpy.spin(node)
    finally:
        node.viewer_stop = True
        if node.viewer is not None:
            node.viewer.close()
        if node.dashboard_enabled:
            node.dashboard_stop = True
            node.dashboard_thread.join(timeout=1.0)
            node.window_proc.join(timeout=1.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
