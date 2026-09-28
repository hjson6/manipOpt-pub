"""MPC method's task sequencing: scans the pile and the tray, picks and places
boxes, and streams a Ruckig reference (planned in cylindrical coordinates
round the base) to mpc_controller.

See docs/implementation_notes.md#task_nodepy.
"""
import json
import time
from pathlib import Path

import numpy as np
import pinocchio as pin
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from ruckig import ControlInterface, InputParameter, OutputParameter, Result, Ruckig, RuckigError, Trajectory
from std_msgs.msg import Empty, Float32MultiArray, Float64, Float64MultiArray, Bool, String

from core.collision import NO_OBSTACLE_POSITION, NO_OBSTACLE_RADIUS
from perception import heightmap
from pick_place_common.packing import plan_compact
from pick_place_common.telemetry import Telemetry
from pick_place_common.scene import (
    DEST_FLOOR_Z, DEST_GRID_SHAPE, DEST_HULL_CENTER,
    DEST_TRAY_X_MAX, DEST_TRAY_X_MIN, DEST_TRAY_Y_MAX, DEST_TRAY_Y_MIN, DEST_TRAY_Z_MAX,
    TOOL_RADIUS_M, WRIST_ABOVE_TCP_M, WRIST_EXTENT_TOOL, BOX_HEIGHT_MAX_M, BOX_HEIGHT_MIN_M,
    DEST_HULL_RADIUS, DEST_SCAN_BOUNDS, DEST_SCAN_POSITION, FLATNESS_TOL,
    FLOOR_Z, GRASP_INLIER_FRAC, LIFT_HEIGHT, PARK_POSITION,
    SCAN_RESOLUTION, SOURCE_GRID_POINTS, SOURCE_GRID_SHAPE,
    SOURCE_MIN_FILL_FRAC, SOURCE_MIN_FOOTPRINT_CELLS, SOURCE_SCAN_BOUNDS,
)

CONTROL_PERIOD_S = 0.02  # must match MPCConfig.dt
HOME_EE_POSITION = np.array([0.5545, 0.0, 0.6245])
# Same model and TCP frame as the controller's OCP.
MJCF_PATH = str(Path(__file__).resolve().parent.parent.parent / "common" / "models" / "panda_robot.xml")
TCP_FRAME = "tcp_site"
# Must match mpc_controller_node.PROXY_FRAMES and MPCConfig.safety_margin.
PROXY_SPHERES = (("link3", 0.10), ("link5", 0.09), ("link7", 0.09), ("attachment", 0.08))
OCP_SAFETY_MARGIN_M = 0.03
HULL_GROW_PAD_M = 0.01

MAX_VEL = np.array([0.5, 0.5, 0.5])
MAX_ACCEL = np.array([3.0, 3.0, 3.0])
MAX_JERK = np.array([20.0, 20.0, 20.0])
HORIZON_STEPS = 15  # one goal per MPC stage; must match MPCConfig.N
PLACE_CLEARANCE_M = 0.005  # gap to walls and placed boxes; more than the tray scan's ~3 mm error
PLACE_SNAP_MAX_M = 0.02  # snap only gaps smaller than this
WRIST_MARGIN_M = 0.008  # wrist to wall
PUSH_LIFT_M = 0.02  # tool tip above the box top while moving round it
PUSH_START_GAP_M = 0.01  # tool to box face before a push
PUSH_BACKOFF_M = 0.01
PUSH_SPEED_MPS = 0.05
PUSH_TRY_M = 0.002  # smaller shifts are left as a gap
PUSH_TOOL_MARGIN_M = 0.008  # tool tracking error while tilting down (a wall was tapped at 9 mm)
PUSH_MIN_M = 0.006  # a gap below this is accepted when the push is not possible
PUSH_LEG_TIMEOUT_S = 6.0  # a jammed push gives up and moves on
PUSH_TILT_RAD = np.radians(30.0)  # IK search: 27 mm wall clearance at a back-row push, near the normal posture
PUSH_SAFE_TCP_Z = 0.15  # link7 (0.086 m above the TCP) clears the 0.22 m walls
WRIST_HEIGHT_M = 0.055  # link7 spans 0.100-0.155 m above the TCP
FOREARM_ABOVE_FLANGE_M = 0.036  # links 5-6 start ~0.136 m above the TCP
FOREARM_WALL_MIN_M = 0.09  # flange to a wall on the robot's side while the forearm is below its top (IK search)
J7_PLACE_LIMIT_RAD = np.radians(150.0)  # joint 7's range is +-166 deg
PLACE_LEG_TIMEOUT_S = 8.0  # a blocked place releases where it is
TOUCH_SPEED_MPS = 0.02  # final descent of a place, until the surface takes the box
TOUCH_HOVER_M = 0.01  # hover this far above the tallest possible box top
TOUCH_BELOW_M = 0.005  # the touch leg aims this far below the lowest possible box top
TOUCH_FORCE_FRAC = 0.5  # contact: wrist load below this share of the box's weight
TOUCH_MIN_WEIGHT_N = 0.5  # below this the load cell cannot tell contact
TOUCH_SIDE_FORCE_N = 1.0  # a sideways wrist load while lowering: the box is on a neighbour's edge
TOUCH_FRICTION_MU = 0.8  # ...beyond the landing surface's friction (box friction 0.6)
PLACE_RETRIES = 2  # re-scan and re-plan a blocked place this often, then stop
PLACE_RETRY_GROW_M = 0.0025  # the held box's footprint grows by this per side at each retry
SET_DOWN_MIN_GAP_M = 0.001  # a set-down spot shifted for the wrist keeps this from placed boxes
MIN_SPEED_SCALE = 0.05  # Ruckig rejects a zero max velocity; a full stop is a hold
SUPERVISOR_TIMEOUT_S = 1.0  # hold if the supervisor is silent this long
REF_AT_TARGET_TOL_M = 1e-3  # a settle only counts as an arrival within this
# Arc points for long swings, used only in x/y/z mode (during a detour).
ARC_MIN_SWING_DEG = 50.0
ARC_STEP_DEG = 40.0
ARC_MIN_RADIUS_M = 0.40
PASS_SPEED_MPS = 0.5  # cruise speed through pass-through points
PASS_SKIP_M = 0.03  # on resume, skip a pass-through point this close
BLEND_M = 0.05  # departing corners start the swing this far below the top
MIN_PLAN_RADIUS_M = 0.05  # floor on the radius in cylindrical coordinates


def _cart_to_cyl(p, v, a, az_near):
    """x/y/z (position, velocity, acceleration) -> (azimuth, radius, z) and rates,
    azimuth unwrapped to within pi of az_near.
    """
    p, v, a = (np.asarray(x, dtype=float) for x in (p, v, a))
    r = max(float(np.hypot(p[0], p[1])), MIN_PLAN_RADIUS_M)
    az = float(np.arctan2(p[1], p[0]))
    az = az_near + (az - az_near + np.pi) % (2 * np.pi) - np.pi
    e_r = np.array([np.cos(az), np.sin(az), 0.0])
    e_t = np.array([-np.sin(az), np.cos(az), 0.0])
    r_d, az_d = float(v @ e_r), float(v @ e_t) / r
    r_dd = float(a @ e_r) + r * az_d ** 2
    az_dd = (float(a @ e_t) - 2.0 * r_d * az_d) / r
    return [az, r, p[2]], [az_d, r_d, v[2]], [az_dd, r_dd, a[2]]


def _cyl_to_cart(c, cv, ca):
    az, r, z = c
    az_d, r_d, z_d = cv
    az_dd, r_dd, z_dd = ca
    e_r = np.array([np.cos(az), np.sin(az), 0.0])
    e_t = np.array([-np.sin(az), np.cos(az), 0.0])
    p = r * e_r + np.array([0.0, 0.0, z])
    v = r_d * e_r + r * az_d * e_t + np.array([0.0, 0.0, z_d])
    a = (r_dd - r * az_d ** 2) * e_r + (r * az_dd + 2.0 * r_d * az_d) * e_t + np.array([0.0, 0.0, z_dd])
    return p, v, a


# Detour round the supervisor's static obstacle (_plan_via). The flange proxy
# and margin must match PROXY_FRAMES ("attachment") and MPCConfig.safety_margin.
DETOUR_EXTRA_M = 0.02
DETOUR_FLANGE_PROXY_M = 0.08
DETOUR_SAFETY_MARGIN_M = 0.03
DETOUR_STEP_M = 0.05
DETOUR_MAX_EXTRA_M = 0.45
DETOUR_MAX_Z_M = 1.0
REACH_MIN_M, REACH_MAX_M = 0.30, 0.78  # comfortable reach annulus round the base
BASE_EXCLUSION_M = 0.10  # detour legs stay this far from the base axis

ORIENT_AXIS_TARGET = np.array([0.0, 0.0, -1.0])  # tool +z straight down
ORIENT_AXIS2_TARGET = np.array([0.0, -1.0, 0.0])  # tool +x heading at the pile; elsewhere see _heading_at

SCAN_PAUSE_TICKS = 50  # 1 s after a scan, so the log is readable

DEPTH_WAIT_TICKS = 25  # then decide from the heightmap alone
REMEASURE_WINDOW_M = 0.006  # search a placed box's top this far round where it was recorded
REMEASURE_SIZE_TOL_M = 0.0025  # measured half-size this close to the known one: both edges are its own
REMEASURE_MAX_MOVE_M = 0.015  # a bigger correction is not believed
SCAN_TIMEOUT_TICKS = 100  # re-request an unanswered scan after 2 s
DEST_NO_SPOT_MAX_RETRIES = 30  # then stop: the tray is blocked or full

N_OBSTACLE_SLOTS = 3  # must match MPCConfig.n_obstacles
# Largest box (cbox_0); pads the pile hull. Update if a bigger box is added.
_BOX_HALF_DIAGONAL = float(np.linalg.norm([0.075, 0.060, 0.045]))
_HULL_PAD = 0.05


class Waypoint:
    """One waypoint of a leg: what to do there and which hulls are on."""
    __slots__ = ("pos", "action", "hull_active", "dest_idx", "dest_hull_active", "pass_through",
                 "blend", "max_speed", "timeout_s", "tool_axis", "heading_offset", "touch")

    def __init__(self, pos, action=None, hull_active=False, dest_idx=None,
                 dest_hull_active=False, pass_through=False, blend=False,
                 max_speed=None, timeout_s=None, tool_axis=None, heading_offset=0.0, touch=False):
        self.pos = pos
        self.touch = touch  # descend until the wrist load drops (a place)
        self.heading_offset = heading_offset  # added to the aligned heading (a box turned 90 deg)
        self.tool_axis = ORIENT_AXIS_TARGET if tool_axis is None else tool_axis  # reached by the leg's end
        self.max_speed = max_speed  # m/s cap on this leg
        self.timeout_s = timeout_s  # arrive anyway after this long (not counting holds)
        self.blend = blend  # start the next leg BLEND_M early
        self.pass_through = pass_through  # no stop; the arm stops only to grasp, release and scan
        self.action = action  # None, "pick", or "place"
        self.hull_active = hull_active  # pile hull (slot 0)
        self.dest_idx = dest_idx  # (x, y, stack height) of the chosen spot, for the log
        self.dest_hull_active = dest_hull_active  # tray hull (slot 1)


def _build_source_legs(box_pos):
    """Approach, pick and lift legs for a box whose top is at box_pos."""
    approach = np.array([box_pos[0], box_pos[1], LIFT_HEIGHT])
    return [
        Waypoint(approach, hull_active=False, pass_through=True),
        Waypoint(box_pos, action="pick", hull_active=False),
        Waypoint(approach, hull_active=False, pass_through=True, blend=True),
    ]


def _build_dest_legs(hover_pos, touch_pos, dest_idx, push_legs=(), heading_offset=0.0):
    """Release-above, hover, touch-down (place), optional push and lift-off legs
    for the chosen spot. The box turns by heading_offset on the way to above the
    spot."""
    release_above = np.array([hover_pos[0], hover_pos[1], LIFT_HEIGHT])
    return [
        Waypoint(release_above, hull_active=False, dest_idx=dest_idx, pass_through=True,
                 heading_offset=heading_offset),
        Waypoint(hover_pos, action="hover", hull_active=False, heading_offset=heading_offset,
                 timeout_s=PLACE_LEG_TIMEOUT_S),
        Waypoint(touch_pos, action="place", hull_active=False, timeout_s=PLACE_LEG_TIMEOUT_S,
                 heading_offset=heading_offset, max_speed=TOUCH_SPEED_MPS, touch=True),
        *push_legs,
        Waypoint(release_above, hull_active=False, pass_through=True, blend=True),
    ]


def _push_centring(tilted, axis, height):
    """Sideways TCP shift that centres a tilted tool's contact on the pushed face.
    The tool touches the face from its tip (mid-height) up to the box top, drifting
    sideways as it rises; its middle, a quarter of the height up, would push
    off-centre and turn the box."""
    k = 1 - axis
    shift = np.zeros(3)
    shift[k] = -(tilted[k] / tilted[2]) * height / 4.0
    return shift


def _push_legs(start_xy, centre, half, axis, push, surface, height, tilted):
    """Legs that push a released box (centre, half-extents) by `push` metres along
    `axis`, starting from the TCP at start_xy: up clear of the walls, over the
    box's far side, down to mid-height with the tool along `tilted`, push, back
    off, up.
    """
    sign = np.sign(push)
    safe_z = max(surface + height + PUSH_LIFT_M, PUSH_SAFE_TCP_Z)
    start = np.array([centre[0], centre[1], safe_z])
    start[axis] -= sign * (half[axis] + TOOL_RADIUS_M + PUSH_START_GAP_M)
    down = start.copy()
    down[2] = surface + height / 2.0
    end = down.copy()
    end[axis] += push + sign * PUSH_START_GAP_M
    back = end.copy()
    back[axis] -= sign * PUSH_BACKOFF_M
    shift = _push_centring(tilted, axis, height)
    start, down, end, back = start + shift, down + shift, end + shift, back + shift
    return [
        Waypoint(np.array([start_xy[0], start_xy[1], safe_z]), timeout_s=PUSH_LEG_TIMEOUT_S),
        Waypoint(start, timeout_s=PUSH_LEG_TIMEOUT_S),
        Waypoint(down, timeout_s=PUSH_LEG_TIMEOUT_S, tool_axis=tilted),
        Waypoint(end, action="push", max_speed=PUSH_SPEED_MPS, timeout_s=PUSH_LEG_TIMEOUT_S, tool_axis=tilted),
        Waypoint(back, timeout_s=PUSH_LEG_TIMEOUT_S, tool_axis=tilted),
        Waypoint(np.array([back[0], back[1], safe_z]), timeout_s=PUSH_LEG_TIMEOUT_S),
    ]


class TaskNode(Node):
    """Per box: RETURNING -> AWAITING_SCAN -> PAUSING -> MOVING_TO_BOX (approach,
    pick, lift) -> TRAVELING_TO_DEST -> AWAITING_DEST_SCAN -> PAUSING_DEST ->
    MOVING_TO_SLOT (release-above, place, lift-off). PARKED when done.
    """
    (RETURNING, AWAITING_SCAN, PAUSING, MOVING_TO_BOX,
     TRAVELING_TO_DEST, AWAITING_DEST_SCAN, PAUSING_DEST, MOVING_TO_SLOT,
     PARKED) = range(9)

    def __init__(self):
        super().__init__("task_node")
        self.state = self.RETURNING
        self.tick_count = 0
        self.pause_until_tick = None
        self.scan_requested_at_tick = None
        self.latest_heightmap = None  # last pile scan, for the pile hull
        self.latest_dest_hmap = None
        self.axis_cmd = ORIENT_AXIS_TARGET.copy()
        self.leg_axis0 = self.leg_axis_g = ORIENT_AXIS_TARGET.copy()
        self.leg_start = None
        self.leg_speed = None
        self.leg_heading_offset = 0.0
        self.leg_timeout_ticks = None
        self.leg_ticks = 0
        self.leg_wps = []
        self.leg_idx = 0
        self.boxes_moved = 0
        # Held box half-extents: footprint from the pick scan, height the spec's maximum
        # until measured at touch-down (box_height).
        self.current_box_size = None
        self.box_height = None
        self.pick_sensed = None  # (x, y, hx, hy) of the chosen top in the pick scan
        self.place_ctx = None  # what the touch-down needs to finish the placement
        self.wrist_fz = None
        self.wrist_fz_filt = None
        self.touch_baseline = None
        self.place_attempts = 0
        # Grows the static obstacle while carrying (current_box_size stays set after a place).
        self.holding_box = False
        self.dest_no_spot_count = 0  # consecutive placement scans with no flat spot
        # (x, y, hx, hy, top z, surface z) of every box placed so far.
        self.placed_boxes = []
        self.current_box_offset = None
        self.pending_place = None
        self.upcoming_sizes = []

        self.otg = Ruckig(3, CONTROL_PERIOD_S)
        self.inp = InputParameter(3)
        self.out = OutputParameter(3)
        self.inp.current_position = HOME_EE_POSITION.tolist()
        self.inp.current_velocity = [0.0, 0.0, 0.0]
        self.inp.current_acceleration = [0.0, 0.0, 0.0]
        self.inp.target_position = PARK_POSITION.tolist()
        # The waypoint; inp.target_position may be an arc or detour point on the way.
        self.leg_target = PARK_POSITION.copy()
        self.via_active = False
        self.detour_active = False
        self.leg_pass = False
        self.pass_vel = np.zeros(3)
        self.was_holding = False
        self.leg_in_dir = np.zeros(3)
        self.leg_blend = False
        # Planner coordinates: cylindrical unless detouring. lim_div scales the
        # Cartesian limits per planner axis (the azimuth's by the radius).
        self.ref_cyl = False
        self.lim_div = np.ones(3)
        self.target_cart = PARK_POSITION.copy()
        # Heading profile state (_start_heading_leg).
        self.psi_cmd = float(np.arctan2(ORIENT_AXIS2_TARGET[1], ORIENT_AXIS2_TARGET[0]))
        self.leg_az0, self.leg_daz = 0.0, 0.0
        self.leg_psi0 = self.leg_psi_g = self.psi_cmd
        self.inp.target_velocity = [0.0, 0.0, 0.0]
        self.inp.target_acceleration = [0.0, 0.0, 0.0]
        self.inp.max_velocity = MAX_VEL.tolist()
        self.inp.max_acceleration = MAX_ACCEL.tolist()
        self.inp.max_jerk = MAX_JERK.tolist()

        self.pub = self.create_publisher(Float64MultiArray, "/mpc/goal", 10)
        self.action_pub = self.create_publisher(String, "/task/action", 10)
        # Pick and place decisions as JSON, for the sim's decision window.
        self.decision_pub = self.create_publisher(String, "/task/decision", 10)
        self.obstacle_pub = self.create_publisher(Float64MultiArray, "/mpc/obstacle_params", 10)
        self.dynamic_obstacle_slot = [*NO_OBSTACLE_POSITION, NO_OBSTACLE_RADIUS]
        # Current waypoint's hull flags, to republish when the static obstacle changes.
        self._last_hull_active = False
        self._last_dest_hull_active = False
        # Slot 2: the supervisor's static obstacle ([x, y, z, r], r = 0 for none).
        # People never come this way; they are handled by hold and slow-down.
        self.create_subscription(Float64MultiArray, "/mpc/static_obstacle",
                                  self._on_dynamic_obstacle, 10)
        self.orient_pub = self.create_publisher(Float64MultiArray, "/mpc/orientation_goal", 10)
        self.scan_pub = self.create_publisher(Empty, "/sim/scan_container", 10)
        self.create_subscription(Float64MultiArray, "/sim/container_occupancy",
                                  self._on_container_occupancy, 10)
        self.latest_depth = None  # the pile scan's frame, sent just before its heightmap
        self.latest_depth_tick = -1
        self.pending_pile_scan = None
        self.create_subscription(Float32MultiArray, "/sim/container_depth", self._on_container_depth, 10)
        self.dest_depth = None  # the tray scan's frame, likewise
        self.dest_depth_tick = -1
        self.pending_dest_scan = None
        self.create_subscription(Float32MultiArray, "/sim/destination_depth", self._on_destination_depth, 10)
        self.create_subscription(Float64MultiArray, "/sim/wrist_force", self._on_wrist_force, 10)
        self.dest_scan_pub = self.create_publisher(Empty, "/sim/scan_destination", 10)
        self.create_subscription(Float64MultiArray, "/sim/destination_occupancy",
                                  self._on_destination_occupancy, 10)
        self.create_subscription(Bool, "/mpc/settled", self._on_settled, 10)
        # Obstacle supervisor: hold and speed limit, applied to the reference only.
        self.hold_requested = False
        self.speed_scale = 1.0
        self.supervisor_seen_at = None
        self.supervisor_stale = False
        self.create_subscription(Bool, "/mpc/hold", self._on_hold, 10)
        self.create_subscription(Float64, "/mpc/speed_scale", self._on_speed_scale, 10)
        # The reference stays on the measured TCP until the controller starts.
        self.started = False
        self.pin_model = pin.buildModelFromMJCF(MJCF_PATH)
        self.pin_data = self.pin_model.createData()
        self.tcp_frame_id = self.pin_model.getFrameId(TCP_FRAME)
        self.proxy_frames = [(self.pin_model.getFrameId(n), r) for n, r in PROXY_SPHERES]
        self.proxy_pos = None
        # Pile and tray hulls at full size, and the radius each has grown to.
        self.hull_full = [None, None]
        self.hull_grown = [0.0, 0.0]
        self.measured_tcp = None
        self.create_subscription(JointState, "/sim/joint_states", self._on_joint_state, 10)
        # Start on the controller's first solve, not on /mpc/go: `ros2 topic pub
        # --once` can fire before the controller is discovered.
        self.create_subscription(Float64MultiArray, "/mpc/solve_diagnostics",
                                  lambda _msg: self._on_go(), 10)
        # Ticked once per plant step (_on_joint_state), not by a timer, so goals line
        # up with the lockstep plant. Tick counts are simulated time.
        self.state_step = -1
        self.telemetry = Telemetry("task", ["step", "state", "ref_speed", "speed_scale", "hold", "cyl", "via"])
        self.dynamic_slot_inflated = [*NO_OBSTACLE_POSITION, NO_OBSTACLE_RADIUS]

    def _on_joint_state(self, msg: JointState):
        pin.framesForwardKinematics(self.pin_model, self.pin_data, np.array(msg.position[:7]))
        self.measured_tcp = np.array(self.pin_data.oMf[self.tcp_frame_id].translation)
        self.proxy_pos = [(np.array(self.pin_data.oMf[f].translation), r) for f, r in self.proxy_frames]
        self.state_step = int(msg.header.frame_id) if msg.header.frame_id else self.state_step + 1
        self._tick()

    def _on_go(self):
        if self.started or self.measured_tcp is None:
            return
        self.started = True
        self._set_ref_state(self.measured_tcp, np.zeros(3), np.zeros(3), cyl=True)
        self._advance_reference(Waypoint(PARK_POSITION, hull_active=True))

    def _request_scan(self):
        self.state = self.AWAITING_SCAN
        self.scan_requested_at_tick = self.tick_count
        self.scan_pub.publish(Empty())

    def _request_dest_scan(self):
        self.state = self.AWAITING_DEST_SCAN
        self.scan_requested_at_tick = self.tick_count
        self.dest_scan_pub.publish(Empty())

    def _on_container_occupancy(self, msg: Float64MultiArray):
        # The depth frame is larger and may come later: wait for it (_on_container_depth,
        # or DEPTH_WAIT_TICKS in _tick).
        if self.state != self.AWAITING_SCAN:
            return
        self.pending_pile_scan = (msg, self.tick_count)
        if self.latest_depth_tick >= self.scan_requested_at_tick:
            self._process_pile_scan()

    def _process_pile_scan(self):
        msg, _ = self.pending_pile_scan
        self.pending_pile_scan = None
        if self.state != self.AWAITING_SCAN:
            return
        hmap = np.array(msg.data).reshape(SOURCE_GRID_SHAPE)
        self.latest_heightmap = hmap
        boxes = heightmap.find_topmost_boxes(
            hmap, floor_z=FLOOR_Z, flatness_tol=FLATNESS_TOL,
            min_footprint_cells=SOURCE_MIN_FOOTPRINT_CELLS,
            inlier_frac=GRASP_INLIER_FRAC, min_fill_frac=SOURCE_MIN_FILL_FRAC)

        if not boxes:
            self.get_logger().info("container empty; parked indefinitely")
            self._publish_decision("pick", None, "container empty, done")
            self.state = self.PARKED
            self._advance_reference(Waypoint(PARK_POSITION, hull_active=False, dest_hull_active=True))
            return

        pick_order = heightmap.pick_order(hmap, boxes, FLATNESS_TOL)
        row0, col0, row1, col1, height, area = pick_order[0]
        footprint_cells = (row1 - row0, col1 - col0)
        # The rest with a fully visible top, in pick order, for the placement look-ahead.
        hidden = {id(r) for r, f in zip(boxes, heightmap.touches_higher(hmap, boxes, FLATNESS_TOL)) if f}
        self.upcoming_sizes = [self._top_footprint(r)[2:] for r in pick_order[1:] if id(r) not in hidden]
        # Grasp at the centre of the sensed top; the footprint is the top's size.
        x, y, hx_s, hy_s = self._top_footprint(pick_order[0])
        box_pos = np.array([x, y, height])
        self.pick_sensed = (x, y, hx_s, hy_s)
        self.leg_wps = _build_source_legs(box_pos)
        self.leg_idx = 0
        (sx0, _), (sy0, _) = SOURCE_SCAN_BOUNDS
        self._publish_decision(
            "pick", [sx0 + col0 * SCAN_RESOLUTION, sx0 + (col1 - 1) * SCAN_RESOLUTION,
                     sy0 + row0 * SCAN_RESOLUTION, sy0 + (row1 - 1) * SCAN_RESOLUTION, height],
            f"biggest of {len(boxes)} seen")

        self.get_logger().info(
            f"scanning container... selected ({x:.3f}, {y:.3f}) "
            f"(sensed height {height:.3f}m, footprint area {area} cells)"
        )
        self.state = self.PAUSING
        self.pause_until_tick = self.tick_count + SCAN_PAUSE_TICKS

    @staticmethod
    def _parse_depth(msg):
        d = np.asarray(msg.data, dtype=float)
        w, h = int(d[13]), int(d[14])
        return d[0:3], d[3:12].reshape(3, 3), float(d[12]), w, h, d[15:].reshape(h, w)

    def _on_container_depth(self, msg: Float32MultiArray):
        self.latest_depth = self._parse_depth(msg)
        self.latest_depth_tick = self.tick_count
        if self.pending_pile_scan is not None:
            self._process_pile_scan()

    def _on_destination_depth(self, msg: Float32MultiArray):
        self.dest_depth = self._parse_depth(msg)
        self.dest_depth_tick = self.tick_count
        if self.pending_dest_scan is not None:
            self._process_dest_scan()

    def _remeasure_placed(self):
        """Move the placed boxes' records to where the tray scan's depth frame shows
        their tops. On each axis both measured edges are used if the measured size
        matches the known one; otherwise a neighbour of the same height ran into the
        search window, and the edge nearer the record is used with the known size."""
        if self.dest_depth is None or self.dest_depth_tick < self.scan_requested_at_tick:
            self.get_logger().warn("no depth frame for this tray scan; placed boxes not re-measured")
            return
        cam_pos, cam_mat, fovy, w, h, depth = self.dest_depth
        m = REMEASURE_WINDOW_M
        moved = []
        for i, (cx, cy, hx, hy, top, surf) in enumerate(self.placed_boxes):
            e = heightmap.top_extent(depth, cam_pos, cam_mat, fovy, w, h, (cx - hx - m, cx + hx + m),
                                     (cy - hy - m, cy + hy + m), top, FLATNESS_TOL / 2.0, seed_xy=(cx, cy))
            if e is None:
                continue
            new = []
            for lo, hi, c, half in ((e[0], e[1], cx, hx), (e[2], e[3], cy, hy)):
                if abs((hi - lo) / 2.0 - half) <= REMEASURE_SIZE_TOL_M:
                    new.append((lo + hi) / 2.0)
                elif abs(lo - (c - half)) <= abs(hi - (c + half)):
                    new.append(lo + half)
                else:
                    new.append(hi - half)
            d = float(np.hypot(new[0] - cx, new[1] - cy))
            if d > REMEASURE_MAX_MOVE_M:
                self.get_logger().warn(f"placed box {i + 1}: scan puts it {1e3 * d:.0f} mm from its record; kept")
                continue
            self.placed_boxes[i] = (new[0], new[1], hx, hy, top, surf)
            moved.append(f"{i + 1}: {1e3 * (new[0] - cx):+.1f}/{1e3 * (new[1] - cy):+.1f}")
        if moved:
            self.get_logger().info(f"tray scan: placed boxes re-measured (x/y mm) {', '.join(moved)}")

    def _top_footprint(self, rec):
        """(cx, cy, hx, hy) of a detected top: its edges from the depth pixels on it
        (heightmap.top_extent), or from the heightmap cells without a frame."""
        row0, col0, row1, col1, height, _ = rec
        (x0, _), (y0, _) = SOURCE_SCAN_BOUNDS
        r = SCAN_RESOLUTION
        if self.latest_depth is not None and self.latest_depth_tick >= self.scan_requested_at_tick:
            cam_pos, cam_mat, fovy, w, h, depth = self.latest_depth
            e = heightmap.top_extent(depth, cam_pos, cam_mat, fovy, w, h,
                                     (x0 + (col0 - 1) * r, x0 + col1 * r), (y0 + (row0 - 1) * r, y0 + row1 * r),
                                     height, FLATNESS_TOL / 2.0,
                                     seed_xy=heightmap.footprint_center_xy(
                                         row0, col0, (row1 - row0, col1 - col0), *SOURCE_SCAN_BOUNDS, r))
            if e is not None:
                return (e[0] + e[1]) / 2, (e[2] + e[3]) / 2, (e[1] - e[0]) / 2, (e[3] - e[2]) / 2
        self.get_logger().warn("no depth frame for this scan; footprint from the heightmap cells")
        x, y = heightmap.footprint_center_xy(row0, col0, (row1 - row0, col1 - col0), *SOURCE_SCAN_BOUNDS, r)
        return x, y, (col1 - col0) * r / 2, (row1 - row0) * r / 2

    def _on_wrist_force(self, msg: Float64MultiArray):
        side = float(np.hypot(msg.data[0], msg.data[1]))
        self.wrist_fz = float(msg.data[2])
        self.wrist_fz_filt = self.wrist_fz if self.wrist_fz_filt is None else (
            0.9 * self.wrist_fz_filt + 0.1 * self.wrist_fz)
        touching = (self.state == self.MOVING_TO_SLOT and self.leg_idx < len(self.leg_wps)
                    and self.leg_wps[self.leg_idx].touch and not self._hold_active())
        # The surface it lands on can push sideways only by friction on the load it has
        # taken; more than that is something else (a neighbour's edge).
        taken = 0.0 if self.touch_baseline is None else max(0.0, self.touch_baseline - self.wrist_fz)
        if touching and side > TOUCH_FRICTION_MU * taken + TOUCH_SIDE_FORCE_N:
            self._retry_place(f"{side:.1f} N sideways ({msg.data[0]:+.2f}, {msg.data[1]:+.2f}) while lowering at "
                              f"TCP {np.round(self.measured_tcp, 3).tolist()}: the box is on something's edge")
        elif (touching and self.touch_baseline is not None and self.touch_baseline > TOUCH_MIN_WEIGHT_N
                and self.wrist_fz < TOUCH_FORCE_FRAC * self.touch_baseline):
            self._touch_down(f"wrist load {self.wrist_fz:.2f} of {self.touch_baseline:.2f} N")

    def _retry_place(self, why):
        """A place is blocked: lift straight up, go back to the tray scan point and
        plan again with the box taken a little bigger."""
        self.action_pub.publish(String(data="place_aborted"))
        self.place_attempts += 1
        if self.place_attempts > PLACE_RETRIES:
            self.get_logger().error(f"place failed ({why}) {self.place_attempts} times; stopping with the box held")
            self.state = self.PARKED
            self._set_ref_state(self.measured_tcp, np.zeros(3), np.zeros(3), cyl=self.ref_cyl)
            self._set_target(self.measured_tcp, np.zeros(3))
            return
        self.current_box_size[:2] += PLACE_RETRY_GROW_M
        self.get_logger().warn(f"place blocked ({why}); lifting and re-planning with the footprint "
                               f"grown to {2e3 * self.current_box_size[0]:.0f} x {2e3 * self.current_box_size[1]:.0f} mm")
        self._set_ref_state(self.measured_tcp, np.zeros(3), np.zeros(3), cyl=self.ref_cyl)
        up = np.array([self.measured_tcp[0], self.measured_tcp[1], LIFT_HEIGHT])
        self.leg_wps = [Waypoint(up), Waypoint(DEST_SCAN_POSITION, action="rescan")]
        self.leg_idx = 0
        self.state = self.MOVING_TO_SLOT
        self._advance_reference(self.leg_wps[0])

    def _plan_hz(self):
        """Half-height to plan a placement with: measured, else the spec's lowest
        (the wrist then comes lowest, the conservative case for wall clearance)."""
        return (self.box_height if self.box_height is not None else BOX_HEIGHT_MIN_M) / 2.0

    def _touch_down(self, why):
        """The surface has the box: freeze the reference, measure the height, plan
        the pushes with it and release."""
        x, y, hx, hy, surface, psi, shift = self.place_ctx
        height = float(self.measured_tcp[2] - surface)
        if not BOX_HEIGHT_MIN_M - 0.01 <= height <= BOX_HEIGHT_MAX_M + 0.01:
            self._retry_place(f"touch-down ({why}) at a box height of {1e3 * height:.0f} mm, outside the spec: "
                              f"resting on something else")
            return
        self.box_height = height
        self.get_logger().info(f"touch-down ({why}): box height {1e3 * self.box_height:.1f} mm")
        self._set_ref_state(self.measured_tcp, np.zeros(3), np.zeros(3), cyl=self.ref_cyl)
        self._set_target(self.measured_tcp, np.zeros(3))
        _, pushes, notes, final, _ = self._clearance_and_pushes(
            x, y, hx, hy, self.box_height / 2.0, surface, psi, shift=shift)
        for note in notes:
            self.get_logger().info(note)
        self.leg_wps = self.leg_wps[:self.leg_idx + 1] + pushes + [self.leg_wps[-1]]
        self.pending_place = (final[0], final[1], float(hx), float(hy), surface + self.box_height, surface)
        self.leg_timeout_ticks = None
        self._arrive()

    def _on_destination_occupancy(self, msg: Float64MultiArray):
        # As for the pile scan, wait for the depth frame.
        if self.state != self.AWAITING_DEST_SCAN:
            return
        self.pending_dest_scan = (msg, self.tick_count)
        if self.dest_depth_tick >= self.scan_requested_at_tick:
            self._process_dest_scan()

    def _process_dest_scan(self):
        msg, _ = self.pending_dest_scan
        self.pending_dest_scan = None
        if self.state != self.AWAITING_DEST_SCAN:
            return
        self._remeasure_placed()
        if self.current_box_size is None:
            self.get_logger().warn(
                "destination scan arrived with no grasped box size known yet; retrying")
            self._request_dest_scan()
            return
        hmap = np.array(msg.data).reshape(DEST_GRID_SHAPE)
        self.latest_dest_hmap = hmap
        hx, hy, hz = self.current_box_size
        floor_boxes = [b[:4] for b in self.placed_boxes if b[5] < DEST_FLOOR_Z + 0.005]
        # Floor spots come from the packing rule, checked against the scan. The
        # heightmap search is the fallback, and also what stacks.
        packed = plan_compact(
            float(hx), float(hy), floor_boxes,
            ((DEST_TRAY_X_MIN, DEST_TRAY_X_MAX), (DEST_TRAY_Y_MIN, DEST_TRAY_Y_MAX)),
            PLACE_CLEARANCE_M,
            # Clear in the scan, and the robot can get the box there (set down, push back).
            is_free=lambda cx, cy, fhx, fhy: (self._footprint_clear(cx, cy, fhx, fhy, DEST_FLOOR_Z, hmap)
                                              and not self._place_options(cx, cy, fhx, fhy, DEST_FLOOR_Z)[0][0]),
            upcoming=self.upcoming_sizes[:3])
        if packed is not None:
            self.dest_no_spot_count = 0
            cx, cy, fhx, fhy = packed
            turned = abs(fhx - hx) > 1e-6
            self._commit_placement(cx, cy, DEST_FLOOR_Z,
                                   f"compact{', turned 90 deg' if turned else ''} "
                                   f"(looked ahead at {len(self.upcoming_sizes[:3])})", fhx, fhy)
            return
        footprint_cells = (
            round(2 * hy / SCAN_RESOLUTION) + 1,
            round(2 * hx / SCAN_RESOLUTION) + 1,
        )
        # Lowest flat spot, strict: any disagreeing cell means something is in the way.
        result = heightmap.find_best_footprint(
            hmap, footprint_cells, mode="lowest", flatness_tol=FLATNESS_TOL,
            floor_z=DEST_FLOOR_Z)
        if result is None:
            # Noise can hide a flat spot: retry, up to DEST_NO_SPOT_MAX_RETRIES.
            self.dest_no_spot_count += 1
            if self.dest_no_spot_count >= DEST_NO_SPOT_MAX_RETRIES:
                self.get_logger().error(
                    f"destination blocked or full: no flat placement spot in "
                    f"{self.dest_no_spot_count} consecutive scans "
                    f"({self.boxes_moved} boxes moved); stopping with the "
                    f"box still held over the tray. Clear the tray and restart.")
                self.state = self.PARKED
                return
            if self.dest_no_spot_count == 1 or self.dest_no_spot_count % 10 == 0:
                self.get_logger().warn(
                    f"destination scan found no flat placement spot; retrying "
                    f"({self.dest_no_spot_count}/{DEST_NO_SPOT_MAX_RETRIES})")
            self._request_dest_scan()
            return
        self.dest_no_spot_count = 0
        row, col, surface_height = result

        x, y = heightmap.footprint_center_xy(
            row, col, footprint_cells, *DEST_SCAN_BOUNDS, SCAN_RESOLUTION)
        x, y = self._snap_flush(x, y, hx, hy, surface_height, hmap)
        self._commit_placement(x, y, surface_height,
                               "no floor spot in the packing: free floor spot" if surface_height < DEST_FLOOR_Z + 0.005
                               else "no floor spot in the packing: on top of a box", hx, hy)

    def _place_options(self, x, y, hx, hy, surface):
        """Ways to place the held box at (x, y) with footprint hx/hy, best first: one
        per tool heading that turns the box that way (joint 7 in range), each
        (incomplete, set-down shift, heading offset, heading, shift, push legs,
        notes, where the box ends up).
        """
        hz = self._plan_hz()
        turned = abs(hx - self.current_box_size[0]) > 1e-6
        az = float(np.arctan2(y, x))
        offsets = (np.pi / 2, -np.pi / 2) if turned else (0.0,)
        j7 = {o: abs((az - self._aligned_yaw(az) - o - np.radians(135) + np.pi) % (2 * np.pi) - np.pi)
              for o in offsets}  # see _aligned_yaw
        ok = [o for o in offsets if j7[o] <= J7_PLACE_LIMIT_RAD] or [min(offsets, key=j7.get)]
        options = []
        for offset in ok:
            psi = self._aligned_yaw(az) + offset
            shift, legs, notes, final, complete = self._clearance_and_pushes(x, y, hx, hy, hz, surface, psi)
            options.append((not complete, float(np.hypot(*shift)), offset, psi, shift, legs, notes, final))
        return sorted(options, key=lambda o: o[:2])

    def _commit_placement(self, x, y, surface_height, how, hx, hy):
        """Build and start the release legs for a box centred at (x, y) on a surface
        at surface_height, footprint half extents hx/hy along x/y (swapped from the
        box's own when it is turned 90 deg).
        """
        _, _, offset, psi, shift, push_legs, notes, final = self._place_options(x, y, hx, hy, surface_height)[0]
        for note in notes:
            self.get_logger().info(note)
        xy = (x + shift[0], y + shift[1])
        hover = self._place_target(xy, surface_height + BOX_HEIGHT_MAX_M + TOUCH_HOVER_M, psi)
        touch = self._place_target(xy, surface_height + BOX_HEIGHT_MIN_M - TOUCH_BELOW_M, psi)
        # Pushes are re-planned with the measured height at touch-down (_touch_down).
        self.place_ctx = (x, y, hx, hy, surface_height, psi, shift)
        self.pending_place = None
        stack_height = surface_height - DEST_FLOOR_Z
        self.leg_wps = _build_dest_legs(hover, touch, (x, y, stack_height), push_legs, offset)
        self.leg_idx = 0
        self._publish_decision(
            "place", [x - hx, x + hx, y - hy, y + hy, surface_height],
            how,
            others=[[b[0] - b[2], b[0] + b[2], b[1] - b[3], b[1] + b[3], b[4]] for b in self.placed_boxes])

        self.get_logger().info(
            f"scanning destination... selected ({x:.3f}, {y:.3f}) "
            f"(sensed surface {surface_height:.3f}m; {how})"
        )
        self.state = self.PAUSING_DEST
        self.pause_until_tick = self.tick_count + SCAN_PAUSE_TICKS

    def _publish_decision(self, side, rect, label, others=()):
        self.decision_pub.publish(String(data=json.dumps(
            {"side": side, "rect": None if rect is None else [float(v) for v in rect], "label": label,
             "others": [[float(v) for v in o] for o in others]})))

    def _place_target(self, centre_xy, tcp_z, psi):
        """TCP target (x, y, tcp_z) that puts the held box's centre over centre_xy
        with the tool at heading psi, from the in-hand offset."""
        x_t = np.array([np.cos(psi), np.sin(psi), 0.0])
        z_t = np.array([0.0, 0.0, -1.0])
        rot = np.column_stack([x_t, np.cross(z_t, x_t), z_t])
        off = rot @ self.current_box_offset if self.current_box_offset is not None else np.zeros(3)
        return np.array([centre_xy[0] - off[0], centre_xy[1] - off[1], tcp_z])

    @staticmethod
    def _wrist_rect(psi):
        """link7's (lo, hi) x/y reach past the TCP in world, tool down at heading psi."""
        c, s = np.cos(psi), np.sin(psi)
        px, mx, py, my = WRIST_EXTENT_TOOL
        # Tool down: tool y = z x x = (sin psi, -cos psi) in world.
        corners = np.array([[px, py], [px, -my], [-mx, py], [-mx, -my]]) @ np.array([[c, s], [s, -c]])
        return corners.min(axis=0), corners.max(axis=0)

    def _wrist_limits(self, tcp_z, psi):
        """Per axis, the (min, max) TCP coordinate that keeps link7 inside the tray
        walls with the tool at tcp_z; None where link7 passes above the wall tops.
        """
        if tcp_z + WRIST_ABOVE_TCP_M >= DEST_TRAY_Z_MAX + WRIST_MARGIN_M:
            return None
        lo, hi = self._wrist_rect(psi)
        walls = ((DEST_TRAY_X_MIN, DEST_TRAY_X_MAX), (DEST_TRAY_Y_MIN, DEST_TRAY_Y_MAX))
        return [(w0 + WRIST_MARGIN_M - lo[k], w1 - WRIST_MARGIN_M - hi[k]) for k, (w0, w1) in enumerate(walls)]

    def _push_clearance(self, tip, tool_axis, psi):
        """Smallest clearance (m) at a push pose, tip at `tip` and the tool along
        tool_axis: the tool and link7 against the tray walls and against what the
        scan shows, the flange against the walls on the robot's side (the forearm
        hangs towards the base). Below zero is a hit.
        """
        hmap = self.latest_dest_hmap
        (xmin, _), (ymin, _) = DEST_SCAN_BOUNDS
        walls = ((DEST_TRAY_X_MIN, DEST_TRAY_X_MAX), (DEST_TRAY_Y_MIN, DEST_TRAY_Y_MAX))
        wlo, whi = self._wrist_rect(psi)
        best = np.inf
        for t in np.arange(0.0, WRIST_ABOVE_TCP_M + WRIST_HEIGHT_M + 1e-9, 0.01):
            p = tip - t * tool_axis
            if t < WRIST_ABOVE_TCP_M:
                r = TOOL_RADIUS_M + PUSH_TOOL_MARGIN_M
                lo, hi = np.full(2, -r), np.full(2, r)
            else:
                lo, hi = wlo, whi
            if p[2] < DEST_TRAY_Z_MAX + WRIST_MARGIN_M:
                for k, (w0, w1) in enumerate(walls):
                    best = min(best, p[k] + lo[k] - w0, w1 - (p[k] + hi[k]))
            c0 = max(int(np.floor((p[0] + lo[0] - xmin) / SCAN_RESOLUTION)), 0)
            c1 = int(np.ceil((p[0] + hi[0] - xmin) / SCAN_RESOLUTION))
            r0 = max(int(np.floor((p[1] + lo[1] - ymin) / SCAN_RESOLUTION)), 0)
            r1 = int(np.ceil((p[1] + hi[1] - ymin) / SCAN_RESOLUTION))
            patch = hmap[r0:r1 + 1, c0:c1 + 1]
            if patch.size:
                best = min(best, p[2] - float(np.max(patch)) - WRIST_MARGIN_M)
        flange = tip - WRIST_ABOVE_TCP_M * tool_axis
        if flange[2] + FOREARM_ABOVE_FLANGE_M < DEST_TRAY_Z_MAX + WRIST_MARGIN_M:
            for k, (w0, w1) in enumerate(walls):
                if w0 > 0.0:  # the base is beyond this wall
                    best = min(best, flange[k] - w0 - FOREARM_WALL_MIN_M)
                if w1 < 0.0:
                    best = min(best, w1 - flange[k] - FOREARM_WALL_MIN_M)
        return best

    def _clearance_and_pushes(self, x, y, hx, hy, hz, surface, psi, shift=None):
        """Where to set the box down so the wrist clears the walls (or the given
        shift), the legs that push it back to (x, y) after release, notes for the
        log, where the box ends up, and whether it ends up at (x, y). Each push
        takes the lean (none, or tilted either way in the plane of the pushed face)
        with the most clearance; a push with no clear lean is skipped.
        """
        tcp = self._place_target((x, y), surface + 2 * hz, psi)
        if shift is None:
            lim = self._wrist_limits(surface + 2 * hz, psi)
            if lim is None:
                return (0.0, 0.0), [], [], (x, y), True
            shift = [float(np.clip(tcp[k], *lim[k]) - tcp[k]) for k in range(2)]
            if max(abs(v) for v in shift) < 1e-4:
                return (0.0, 0.0), [], [], (x, y), True
            sx, sy = x + shift[0], y + shift[1]
            if not self._footprint_clear(sx, sy, hx, hy, surface, self.latest_dest_hmap) or any(
                    abs(sx - px) < hx + phx + SET_DOWN_MIN_GAP_M and abs(sy - py) < hy + phy + SET_DOWN_MIN_GAP_M
                    for px, py, phx, phy, top, _s in self.placed_boxes if top > surface + 0.005):
                return ((0.0, 0.0), [], ["wrist clearance: shifted spot not clear; placing flush"],
                        (x, y), False)
        half = np.array([hx, hy])
        centre = np.array([x + shift[0], y + shift[1]])
        tcp_xy = tcp[:2] + np.array(shift)
        psi_push = self._aligned_yaw(float(np.arctan2(y, x)))
        legs, notes = [], [f"wrist clearance: set down {1e3 * shift[0]:+.0f}/{1e3 * shift[1]:+.0f} mm (x/y) from flush"]
        complete = True
        for axis in (0, 1):
            push = -shift[axis]
            if abs(push) < PUSH_TRY_M:
                continue
            sign = np.sign(push)
            start = np.array([centre[0], centre[1], surface + hz])
            start[axis] -= sign * (half[axis] + TOOL_RADIUS_M + PUSH_START_GAP_M)
            end = start.copy()
            end[axis] += push + sign * PUSH_START_GAP_M
            k = 1 - axis
            leans = []
            for lean in (0.0, 1.0, -1.0):
                tool_axis = np.array([0.0, 0.0, -np.cos(PUSH_TILT_RAD * abs(lean))])
                tool_axis[k] = -lean * np.sin(PUSH_TILT_RAD)  # tool z points flange -> tip
                centring = _push_centring(tool_axis, axis, 2 * hz)
                # The descent tilts the tool on the way down: check it along the way too.
                high = start.copy()
                high[2] = max(surface + 2 * hz + PUSH_LIFT_M, PUSH_SAFE_TCP_Z)
                poses = [(end + centring, tool_axis)]
                for f in (0.25, 0.5, 0.75, 1.0):
                    s_f = f * f * (3.0 - 2.0 * f)
                    a = (1.0 - s_f) * ORIENT_AXIS_TARGET + s_f * tool_axis
                    poses.append((high + f * (start - high) + centring, a / np.linalg.norm(a)))
                cl = min(self._push_clearance(p, a, psi_push) for p, a in poses)
                leans.append((cl, lean, tool_axis))
            cl, lean, tool_axis = max(leans, key=lambda v: v[0])
            if cl < 0.0:
                notes.append(f"push along {'xy'[axis]}: no clear lean (best {1e3 * cl:+.0f} mm); "
                             f"box left {1e3 * abs(push):.0f} mm from flush")
                complete = complete and abs(push) < PUSH_MIN_M
                continue
            notes.append(f"push along {'xy'[axis]} {1e3 * push:+.0f} mm, "
                         f"{'no tilt' if lean == 0 else 'tilted towards ' + ('+' if lean > 0 else '-') + 'xy'[k]}, "
                         f"clearance {1e3 * cl:.0f} mm")
            new = _push_legs(tcp_xy, centre, half, axis, push, surface, 2 * hz, tool_axis)
            if legs and np.allclose(legs[-1].pos, new[0].pos):
                new = new[1:]  # a zero-length leg never gets a settle pulse
            legs += new
            tcp_xy = new[-1].pos[:2]
            centre[axis] += push
        return tuple(shift), legs, notes, (float(centre[0]), float(centre[1])), complete

    @staticmethod
    def _footprint_clear(x, y, hx, hy, surface, hmap):
        """Whether the scan shows a footprint at (x, y) clear down to `surface`,
        ignoring one sample round its edge (it may read a neighbour or a wall).
        """
        (xmin, _), (ymin, _) = DEST_SCAN_BOUNDS
        c0 = int(np.ceil((x - hx - xmin) / SCAN_RESOLUTION)) + 1
        c1 = int(np.floor((x + hx - xmin) / SCAN_RESOLUTION)) - 1
        r0 = int(np.ceil((y - hy - ymin) / SCAN_RESOLUTION)) + 1
        r1 = int(np.floor((y + hy - ymin) / SCAN_RESOLUTION)) - 1
        patch = hmap[max(r0, 0):r1 + 1, max(c0, 0):c1 + 1]
        return patch.size > 0 and float(np.max(patch)) <= surface + FLATNESS_TOL

    def _snap_flush(self, x, y, hx, hy, surface, hmap):
        """Slide a placement to PLACE_CLEARANCE_M from the nearer wall or standing box
        on each axis, if the moved footprint is still clear in the scan.
        """
        def snap_axis(c, half, lo_wall, hi_wall, other_c, other_half, faces):
            lo_face, hi_face = lo_wall, hi_wall
            for fc, fhalf, oc, ohalf in faces:
                if abs(oc - other_c) >= ohalf + other_half - 1e-3:  # not facing this box on this axis
                    continue
                if fc < c:
                    lo_face = max(lo_face, fc + fhalf)
                else:
                    hi_face = min(hi_face, fc - fhalf)
            gap_lo, gap_hi = (c - half) - lo_face, hi_face - (c + half)
            if gap_lo < PLACE_CLEARANCE_M and gap_hi - (PLACE_CLEARANCE_M - gap_lo) >= PLACE_CLEARANCE_M:
                return c + (PLACE_CLEARANCE_M - gap_lo)
            if gap_hi < PLACE_CLEARANCE_M and gap_lo - (PLACE_CLEARANCE_M - gap_hi) >= PLACE_CLEARANCE_M:
                return c - (PLACE_CLEARANCE_M - gap_hi)
            if min(gap_lo, gap_hi) < PLACE_SNAP_MAX_M:
                return c - (gap_lo - PLACE_CLEARANCE_M) if gap_lo <= gap_hi else c + (gap_hi - PLACE_CLEARANCE_M)
            return c

        standing = [b for b in self.placed_boxes if b[4] > surface + 0.005]
        x0, y0 = x, y
        x = snap_axis(x, hx, DEST_TRAY_X_MIN, DEST_TRAY_X_MAX, y, hy,
                      [(b[0], b[2], b[1], b[3]) for b in standing])
        y = snap_axis(y, hy, DEST_TRAY_Y_MIN, DEST_TRAY_Y_MAX, x, hx,
                      [(b[1], b[3], b[0], b[2]) for b in standing])
        if (x, y) == (x0, y0):
            return x, y
        if not self._footprint_clear(x, y, hx, hy, surface, hmap):
            self.get_logger().warn(
                f"flush placement: moved footprint not clear in the scan; keeping ({x0:.3f}, {y0:.3f})")
            return x0, y0
        self.get_logger().info(
            f"flush placement: moved ({1e3 * (x - x0):+.1f}, {1e3 * (y - y0):+.1f}) mm "
            f"to {1e3 * PLACE_CLEARANCE_M:.0f} mm from the nearest wall/box")
        return x, y

    def _announce_dest(self, wp: "Waypoint"):
        if wp.dest_idx is not None:
            x, y, stack_height = wp.dest_idx
            self.get_logger().info(
                f"destination slot selected: ({x:.3f}, {y:.3f}), stack height {stack_height:.3f}m"
            )

    def _now_s(self):
        return time.monotonic()  # WSL2 steps the system clock

    def _on_hold(self, msg: Bool):
        self.supervisor_seen_at = self._now_s()
        if msg.data != self.hold_requested:
            self.get_logger().warn(
                f"supervisor {'HOLD' if msg.data else 'RESUME'} (reference at "
                f"{np.round(self._ref_cart()[0], 3).tolist()}, "
                f"speed {np.linalg.norm(self._ref_cart()[1]):.2f} m/s)")
        self.hold_requested = bool(msg.data)

    def _on_speed_scale(self, msg: Float64):
        self.supervisor_seen_at = self._now_s()
        self.speed_scale = float(msg.data)

    def _hold_active(self) -> bool:
        stale = (self.supervisor_seen_at is not None
                 and self._now_s() - self.supervisor_seen_at > SUPERVISOR_TIMEOUT_S)
        if stale != self.supervisor_stale:
            self.supervisor_stale = stale
            self.get_logger().error(
                "obstacle supervisor went silent -- holding (fail-safe)" if stale
                else "obstacle supervisor is back")
        return self.hold_requested or stale

    def _apply_motion_limits(self, inp):
        """Apply the supervisor's hold and speed limit to a Ruckig input. Hold brakes
        at full acceleration (what the protective distance assumes) through the
        velocity interface and keeps the target, so release just continues the
        move. The speed scale limits velocity only, so braking is unchanged.
        """
        scale = min(max(self.speed_scale, MIN_SPEED_SCALE), 1.0)
        if self.leg_speed is not None:
            scale = min(scale, self.leg_speed / float(np.max(MAX_VEL)))
        inp.max_velocity = (MAX_VEL * scale / self.lim_div).tolist()
        inp.max_acceleration = (MAX_ACCEL / self.lim_div).tolist()
        inp.max_jerk = (MAX_JERK / self.lim_div).tolist()
        if self._hold_active():
            inp.control_interface = ControlInterface.Velocity
            inp.target_velocity = [0.0, 0.0, 0.0]
            inp.target_acceleration = [0.0, 0.0, 0.0]
        else:
            inp.control_interface = ControlInterface.Position
            # Pass speed scaled, not clipped per axis: clipping changed its direction as
            # the limit rose, and the restart hunted.
            vmax = 0.95 * np.array(inp.max_velocity)
            inp.target_velocity = np.clip(self.pass_vel * scale, -vmax, vmax).tolist()
            inp.target_acceleration = [0.0, 0.0, 0.0]

    def _reference_at_target(self) -> bool:
        return bool(not self.via_active and
                    np.linalg.norm(self._ref_cart()[0] - self.leg_target)
                    < REF_AT_TARGET_TOL_M)

    def _on_settled(self, _msg: Bool):
        # A settle away from the target is a hold, not an arrival. Pass-through
        # waypoints advance in _tick.
        if not self._reference_at_target() or self.leg_pass:
            return
        self._arrive()

    def _arrive(self):
        """The reference reached the current waypoint: act and move on."""
        if self.state == self.RETURNING:
            # Above the pile: the wrist camera can see it now.
            self._request_scan()
            return
        if self.state == self.TRAVELING_TO_DEST:
            self._request_dest_scan()
            return
        if self.state not in (self.MOVING_TO_BOX, self.MOVING_TO_SLOT):
            return

        wp = self.leg_wps[self.leg_idx]
        if wp.action == "rescan":
            self._request_dest_scan()
            return
        if wp.action == "pick":
            self.action_pub.publish(String(data=f"pick_at {wp.pos[0]} {wp.pos[1]} {wp.pos[2]}"))
            self.holding_box = True
            x, y, hx, hy = self.pick_sensed
            self.current_box_size = np.array([hx, hy, BOX_HEIGHT_MAX_M / 2.0])
            self.box_height = None
            self.place_attempts = 0
            # In-hand offset: sensed top centre from the TCP, in the TCP frame (x/y only;
            # the TCP holds the top, the height comes at touch-down).
            rot = np.array(self.pin_data.oMf[self.tcp_frame_id].rotation)
            self.current_box_offset = rot.T @ np.array([x - self.measured_tcp[0], y - self.measured_tcp[1], 0.0])
            self.get_logger().info(
                f"grasped: sensed footprint {2e3 * hx:.0f} x {2e3 * hy:.0f} mm, top centre "
                f"{1e3 * np.hypot(x - self.measured_tcp[0], y - self.measured_tcp[1]):.1f} mm from the TCP")
        elif wp.action == "place":
            self.action_pub.publish(String(data="place_held"))
            self.holding_box = False
            if self.pending_place is not None:
                self.placed_boxes.append(self.pending_place)
                self.pending_place = None
        elif wp.action == "push":
            self.action_pub.publish(String(data="pushed"))

        self.leg_idx += 1
        if self.leg_idx < len(self.leg_wps):
            next_wp = self.leg_wps[self.leg_idx]
            self._announce_dest(next_wp)
            self._advance_reference(next_wp)
        elif self.state == self.MOVING_TO_BOX:
            # A hull is off on legs that end inside it (see the notes): the pile hull is
            # on only while leaving the pile, the tray hull only while leaving the tray.
            self.state = self.TRAVELING_TO_DEST
            self._advance_reference(Waypoint(DEST_SCAN_POSITION, hull_active=True, dest_hull_active=False))
        else:
            self.boxes_moved += 1
            self.state = self.RETURNING
            self._advance_reference(Waypoint(PARK_POSITION, hull_active=False, dest_hull_active=True))

    def _advance_reference(self, wp: "Waypoint"):
        self.leg_target = np.array(wp.pos, dtype=float)
        self.leg_start = self._ref_cart()[0].copy()
        self.leg_axis0 = self.axis_cmd
        self.leg_axis_g = np.asarray(wp.tool_axis, dtype=float)
        self.leg_heading_offset = wp.heading_offset
        # The hover before a touch-down has settled: the wrist reads the box's weight.
        self.touch_baseline = self.wrist_fz_filt if wp.touch else None
        self.leg_speed = wp.max_speed
        self.leg_timeout_ticks = None if wp.timeout_s is None else round(wp.timeout_s / CONTROL_PERIOD_S)
        self.leg_ticks = 0
        self.leg_pass = wp.pass_through
        self.leg_blend = wp.blend
        d = self.leg_target - self._ref_cart()[0]
        self.leg_in_dir = d / max(float(np.linalg.norm(d)), 1e-9)
        self._retarget()
        self._start_heading_leg()
        self._last_hull_active = wp.hull_active
        self._last_dest_hull_active = wp.dest_hull_active
        self._publish_obstacle_params(wp.hull_active, wp.dest_hull_active)

    def _retarget(self):
        """Aim the reference at leg_target: straight in cylindrical coordinates, or,
        while the static obstacle blocks the way, via arc points and a detour point
        in x/y/z. Recomputed from the current reference state, so no path state is
        kept.
        """
        cur, cur_v, cur_a = self._ref_cart()
        if not self._swing_blocked(cur, self.leg_target):
            if self.detour_active:
                self.get_logger().info("detour: path clear again")
            self.detour_active = False
            self.via_active = False
            self._set_ref_state(cur, cur_v, cur_a, cyl=True)
            after = None
            if self.leg_pass:
                after = self._next_leg_position()
            self._set_target(self.leg_target,
                             self._pass_velocity_cyl(cur, self.leg_target, after, along_out=not self.leg_blend))
            return
        self._set_ref_state(cur, cur_v, cur_a, cyl=False)
        arc = self._next_arc_point(cur, self.leg_target)
        seg_goal = arc if arc is not None else self.leg_target
        detour = self._plan_via(cur, seg_goal)
        if detour is not None and not self.detour_active:
            self.get_logger().warn(
                f"detour: straight path {np.round(cur, 3).tolist()} -> {np.round(seg_goal, 3).tolist()} "
                f"blocked by static obstacle; via {np.round(detour, 3).tolist()}")
        elif detour is None and self.detour_active:
            self.get_logger().info("detour: path clear again")
        self.detour_active = detour is not None
        target = detour if detour is not None else seg_goal
        self.via_active = target is not self.leg_target
        # Pass-through speed towards whatever comes after the target.
        after = None
        if detour is None and arc is not None:
            nxt = self._next_arc_point(arc, self.leg_target)
            after = nxt if nxt is not None else self.leg_target
        elif detour is None and self.leg_pass:
            nxt_wp = self._next_leg_position()
            if nxt_wp is not None:
                nxt = self._next_arc_point(self.leg_target, nxt_wp)
                after = nxt if nxt is not None else nxt_wp
        self._set_target(target, self._pass_velocity(cur, np.asarray(target, dtype=float), after))

    def _ref_cart(self, inp=None):
        """Reference (position, velocity, acceleration) in x/y/z."""
        inp = inp or self.inp
        c, cv, ca = inp.current_position, inp.current_velocity, inp.current_acceleration
        if self.ref_cyl:
            return _cyl_to_cart(c, cv, ca)
        return np.array(c, dtype=float), np.array(cv, dtype=float), np.array(ca, dtype=float)

    def _set_ref_state(self, p, v, a, cyl):
        """Set the Ruckig state from x/y/z, in cylindrical or x/y/z planner coordinates."""
        if cyl:
            az_near = (self.inp.current_position[0] if self.ref_cyl
                       else float(np.arctan2(p[1], p[0])))
            c, cv, ca = _cart_to_cyl(p, v, a, az_near)
        else:
            c, cv, ca = (np.asarray(x, dtype=float).tolist() for x in (p, v, a))
        # Coordinate-change round-off on a reference at rest breaks Ruckig's time sync.
        cv, ca = ([0.0 if abs(x) < 1e-9 else float(x) for x in y] for y in (cv, ca))
        self.inp.current_position, self.inp.current_velocity, self.inp.current_acceleration = c, cv, ca
        self.ref_cyl = cyl

    def _set_target(self, p, v):
        """Set the Ruckig target (x/y/z position, pass velocity) in the active planner
        coordinates, with matching limits.
        """
        self.target_cart = np.asarray(p, dtype=float)
        # Cap the pass speed at what is reachable in the distance left, set once here:
        # recomputed every tick, the target chased the reference.
        v = np.asarray(v, dtype=float)
        speed = float(np.linalg.norm(v))
        if speed > 1e-6:
            p_ref, v_ref, _ = self._ref_cart()
            reach = np.sqrt(float(v_ref @ v_ref) + float(np.min(MAX_ACCEL)) * np.linalg.norm(self.target_cart - p_ref))
            v = v * min(1.0, reach / speed)
        if self.ref_cyl:
            c, cv, _ = _cart_to_cyl(p, v, np.zeros(3), self.inp.current_position[0])
            self.inp.target_position = c
            self.pass_vel = np.array(cv)
            r_lim = max(self.inp.current_position[1], c[1], MIN_PLAN_RADIUS_M)
            self.lim_div = np.array([r_lim, 1.0, 1.0])
        else:
            self.inp.target_position = self.target_cart.tolist()
            self.pass_vel = np.asarray(v, dtype=float)
            self.lim_div = np.ones(3)

    def _swing_blocked(self, cur, goal):
        """Whether the static obstacle blocks any straight piece of the path from cur
        to goal.
        """
        if self.dynamic_obstacle_slot[3] <= 0.0:
            return False
        a = cur
        while True:
            nxt = self._next_arc_point(a, goal)
            b = nxt if nxt is not None else goal
            if self._plan_via(a, b, quiet=True) is not None:
                return True
            if nxt is None:
                return False
            a = b

    def _next_leg_position(self):
        """Where the task goes after the current waypoint, if known now, else None."""
        if self.state not in (self.MOVING_TO_BOX, self.MOVING_TO_SLOT):
            return None
        if self.leg_idx + 1 < len(self.leg_wps):
            return np.asarray(self.leg_wps[self.leg_idx + 1].pos, dtype=float)
        return DEST_SCAN_POSITION if self.state == self.MOVING_TO_BOX else PARK_POSITION

    @staticmethod
    def _pass_velocity(prev, p, nxt, along_out=False):
        """Velocity through p between prev and nxt: slower for sharper turns, capped
        by the shorter segment.
        """
        if nxt is None:
            return np.zeros(3)
        d_in, d_out = p - prev, np.asarray(nxt, dtype=float) - p
        l_in, l_out = float(np.linalg.norm(d_in)), float(np.linalg.norm(d_out))
        if l_in < 1e-3 or l_out < 1e-3:
            return np.zeros(3)
        u_in, u_out = d_in / l_in, d_out / l_out
        speed = PASS_SPEED_MPS * np.sqrt(max(0.0, (1.0 + float(u_in @ u_out)) / 2.0))
        speed = min(speed, 0.5 * np.sqrt(2.0 * float(np.min(MAX_ACCEL)) * min(l_in, l_out)))
        # Arriving corners (along_out) are passed moving straight down; along the
        # incoming direction the box swung 3-4 cm into the walls on the way down.
        return (u_out if along_out else u_in) * speed

    @classmethod
    def _pass_velocity_cyl(cls, prev, p, nxt, along_out=False):
        """_pass_velocity in arc-length coordinates round the base, so the velocity is
        tangent to the arc.
        """
        if nxt is None:
            return np.zeros(3)
        az_p, r_p = float(np.arctan2(p[1], p[0])), max(float(np.hypot(p[0], p[1])), MIN_PLAN_RADIUS_M)

        def local(q):
            az = float(np.arctan2(q[1], q[0]))
            return np.array([r_p * ((az - az_p + np.pi) % (2 * np.pi) - np.pi), np.hypot(q[0], q[1]), q[2]])
        v_loc = cls._pass_velocity(local(prev), local(p), local(np.asarray(nxt, dtype=float)), along_out)
        e_r = np.array([np.cos(az_p), np.sin(az_p), 0.0])
        e_t = np.array([-np.sin(az_p), np.cos(az_p), 0.0])
        return v_loc[0] * e_t + v_loc[1] * e_r + np.array([0.0, 0.0, v_loc[2]])

    def _next_arc_point(self, cur, goal):
        """Next arc point round the base from cur towards goal, or None if the rest of
        the swing is short enough to go straight.
        """
        az0, az1 = np.arctan2(cur[1], cur[0]), np.arctan2(goal[1], goal[0])
        sweep = (az1 - az0 + np.pi) % (2 * np.pi) - np.pi  # shortest way round
        step = np.radians(ARC_STEP_DEG)
        if abs(sweep) < max(np.radians(ARC_MIN_SWING_DEG), 1.25 * step):
            return None
        frac = step / abs(sweep)
        r0, r1 = np.hypot(*cur[:2]), np.hypot(*goal[:2])
        r = max(r0 + (r1 - r0) * frac, ARC_MIN_RADIUS_M)
        az = az0 + np.sign(sweep) * step
        return np.array([r * np.cos(az), r * np.sin(az), max(cur[2], goal[2])])

    @staticmethod
    def _dist_point_segment(p, a, b):
        ab = b - a
        t = float(np.clip((p - a) @ ab / max(ab @ ab, 1e-12), 0.0, 1.0))
        closest = a + t * ab
        return float(np.linalg.norm(p - closest)), closest

    def _plan_via(self, start, goal, quiet=False):
        """Via-point round the static obstacle (left, right or over; shortest path
        wins), or None if the path is clear or no detour fits the reach envelope.
        """
        x, y, z, r = self.dynamic_obstacle_slot
        if r <= 0.0:
            return None
        obs = np.array([x, y, z])
        clearance = r + DETOUR_FLANGE_PROXY_M + DETOUR_SAFETY_MARGIN_M + DETOUR_EXTRA_M
        if self.holding_box:
            clearance += float(np.linalg.norm(self.current_box_size)) \
                if self.current_box_size is not None else _BOX_HALF_DIAGONAL
        # The target itself is in the keep-out: no via-point can fix that.
        if np.linalg.norm(goal - obs) < clearance:
            return None
        d, closest = self._dist_point_segment(obs, start, goal)
        if d >= clearance:
            return None
        seg = (goal - start)[:2]
        if np.linalg.norm(seg) < 1e-6:
            return None
        n = np.array([-seg[1], seg[0], 0.0]) / np.linalg.norm(seg)
        best = None
        for name, direction in (("left", n), ("right", -n), ("over", np.array([0.0, 0.0, 1.0]))):
            for off in np.arange(clearance, clearance + DETOUR_MAX_EXTRA_M, DETOUR_STEP_M):
                via = closest + direction * off
                if name != "over":
                    via[2] = 0.5 * (start[2] + goal[2])
                reach = float(np.linalg.norm(via[:2]))
                if not (REACH_MIN_M <= reach <= REACH_MAX_M) or via[2] > DETOUR_MAX_Z_M:
                    continue
                if (self._dist_point_segment(obs, start, via)[0] < clearance
                        or self._dist_point_segment(obs, via, goal)[0] < clearance):
                    continue
                # Neither new leg may cut through the space over the base.
                if (self._dist_point_segment(np.zeros(3), np.r_[start[:2], 0.0], np.r_[via[:2], 0.0])[0] < BASE_EXCLUSION_M
                        or self._dist_point_segment(np.zeros(3), np.r_[via[:2], 0.0], np.r_[goal[:2], 0.0])[0] < BASE_EXCLUSION_M):
                    continue
                length = float(np.linalg.norm(via - start) + np.linalg.norm(goal - via))
                if best is None or length < best[0]:
                    best = (length, via, name)
                break
        if best is None:
            if not quiet:
                self.get_logger().warn("detour: no clear via-point inside the reach envelope; relying on the OCP")
            return None
        if not quiet:
            self.get_logger().info(f"detour: going {best[2]} the obstacle, path {best[0]:.2f} m")
        return best[1]

    @staticmethod
    def _aligned_yaw(az):
        """Gripper heading (world angle of tool +x) at base azimuth az: -90 or +90
        deg, square to the pile and tray, whichever keeps joint 7 nearer mid-range.
        Measured with the tool down: joint7 ~= joint1 - heading - 135 deg.
        """
        def j7(psi):
            return abs((az - psi - np.radians(135) + np.pi) % (2 * np.pi) - np.pi)
        return min((-np.pi / 2, np.pi / 2), key=j7)

    def _start_heading_leg(self):
        """Heading profile for a new leg: from the current heading to the aligned one
        at the goal, turning the same way as the base.
        """
        cur = self._ref_cart()[0]
        self.leg_az0 = float(np.arctan2(cur[1], cur[0]))
        az_g = float(np.arctan2(self.leg_target[1], self.leg_target[0]))
        self.leg_daz = float((az_g - self.leg_az0 + np.pi) % (2 * np.pi) - np.pi)
        psi_g = self._aligned_yaw(az_g) + self.leg_heading_offset
        want = self.psi_cmd + self.leg_daz
        self.leg_psi0 = self.psi_cmd
        self.leg_psi_g = psi_g + 2 * np.pi * round((want - psi_g) / (2 * np.pi))

    def _heading_at(self, pos):
        """Heading at reference position pos: turns with the base in proportion to
        the swing done, so the box arrives square (docs/design_notes.md).
        """
        if abs(self.leg_daz) < np.radians(1.0):
            # No swing: turn with the distance covered instead (a box turned at the tray).
            total = 0.0 if self.leg_start is None else float(np.linalg.norm(self.leg_target - self.leg_start))
            frac = 1.0 if total < 1e-6 else float(np.clip(1.0 - np.linalg.norm(self.leg_target - pos) / total, 0.0, 1.0))
            frac = frac * frac * (3.0 - 2.0 * frac)
        else:
            az = float(np.arctan2(pos[1], pos[0]))
            done = (az - self.leg_az0 + np.pi) % (2 * np.pi) - np.pi
            frac = float(np.clip(done / self.leg_daz, 0.0, 1.0))
            # Smoothstep: the heading rate starts and ends at zero, even mid-motion.
            frac = frac * frac * (3.0 - 2.0 * frac)
        return self.leg_psi0 + (self.leg_psi_g - self.leg_psi0) * frac

    def _tool_axis_at(self, pos):
        """Tool axis at reference position pos: eased from the leg's start axis to its
        target axis with the distance covered."""
        if self.leg_start is None or np.allclose(self.leg_axis0, self.leg_axis_g):
            return self.leg_axis_g.copy()
        total = float(np.linalg.norm(self.leg_target - self.leg_start))
        frac = 1.0 if total < 1e-6 else float(np.clip(1.0 - np.linalg.norm(self.leg_target - pos) / total, 0.0, 1.0))
        frac = frac * frac * (3.0 - 2.0 * frac)
        z = (1.0 - frac) * self.leg_axis0 + frac * self.leg_axis_g
        return z / np.linalg.norm(z)

    def _publish_goal(self, horizon):
        """Publish one position and heading goal per MPC stage, stamped with the plant
        step they are for.
        """
        stamp = float(self.state_step + 1)
        horizon = np.asarray(horizon, dtype=float).reshape(-1, 3)
        psis = [self._heading_at(p) for p in horizon]
        axes = [self._tool_axis_at(p) for p in horizon]
        self.psi_cmd = psis[0]
        self.axis_cmd = axes[0]
        self.pub.publish(Float64MultiArray(data=[*horizon.ravel().tolist(), stamp]))
        orient = []
        for psi, z in zip(psis, axes):
            h = np.array([np.cos(psi), np.sin(psi), 0.0])
            h -= (h @ z) * z  # heading target square to the tool axis
            h /= np.linalg.norm(h)
            orient += [*z.tolist(), *h.tolist(), 1.0]
        self.orient_pub.publish(Float64MultiArray(data=[*orient, stamp]))

    def _publish_obstacle_params(self, hull_active: bool, dest_hull_active: bool = False):
        idle_slot = [*NO_OBSTACLE_POSITION, NO_OBSTACLE_RADIUS]
        occupied = None
        if hull_active and self.latest_heightmap is not None:
            flat_heights = self.latest_heightmap.flatten()
            mask = flat_heights > FLOOR_Z + 1e-6
            if mask.any():
                occupied = np.array([
                    [SOURCE_GRID_POINTS[i][0], SOURCE_GRID_POINTS[i][1], flat_heights[i]]
                    for i in range(len(flat_heights)) if mask[i]
                ])
        if occupied is not None:
            center = occupied.mean(axis=0)
            radius = float(np.max(np.linalg.norm(occupied - center, axis=1))) + _BOX_HALF_DIAGONAL + _HULL_PAD
            slot0 = [float(center[0]), float(center[1]), float(center[2]), radius]
        else:
            slot0 = idle_slot
        slot1 = [*(float(c) for c in DEST_HULL_CENTER), DEST_HULL_RADIUS] if dest_hull_active else idle_slot
        for i, slot in enumerate((slot0, slot1)):
            full = slot if slot[3] > 0.0 else None
            if full is None or self.hull_full[i] is None or not np.allclose(full[:3], self.hull_full[i][:3]):
                self.hull_grown[i] = 0.0  # a new or moved hull grows in again
            self.hull_full[i] = full
        self.dynamic_slot_inflated = self._dynamic_obstacle_slot_inflated(hull_active)
        self._send_obstacle_params()

    def _send_obstacle_params(self):
        """Publish the obstacle slots, called every tick. A hull grows in behind the
        arm: its radius never exceeds the arm's clearance from its centre and never
        shrinks during a leg. Switched on at full size around the arm, it jolted
        joints 2 and 4.
        """
        idle_slot = [*NO_OBSTACLE_POSITION, NO_OBSTACLE_RADIUS]
        slots = []
        for i in (0, 1):
            full = self.hull_full[i]
            if full is None:
                slots += idle_slot
                continue
            if self.proxy_pos is not None:
                c = np.array(full[:3])
                clearance = min(float(np.linalg.norm(p - c)) - r for p, r in self.proxy_pos) \
                    - OCP_SAFETY_MARGIN_M - HULL_GROW_PAD_M
                self.hull_grown[i] = max(self.hull_grown[i], min(full[3], clearance))
            slots += [*full[:3], self.hull_grown[i]] if self.hull_grown[i] > 0.0 else idle_slot
        data = slots + self.dynamic_slot_inflated + idle_slot * (N_OBSTACLE_SLOTS - 3)
        self.obstacle_pub.publish(Float64MultiArray(data=data))

    def _dynamic_obstacle_slot_inflated(self, hull_active: bool):
        """Static obstacle slot, grown by the held box's half-diagonal on the travel
        leg (the OCP has no proxy for the box). Not on the short local legs: there
        it left the arm unable to settle at the place target.
        """
        x, y, z, r = self.dynamic_obstacle_slot
        if self.holding_box and hull_active and r > 0.0:
            half_diag = (
                float(np.linalg.norm(self.current_box_size))
                if self.current_box_size is not None else _BOX_HALF_DIAGONAL
            )
            r = r + half_diag
        return [x, y, z, r]

    def _on_dynamic_obstacle(self, msg: Float64MultiArray):
        # Sent to the controller at once, and only when it changes.
        slot = [float(v) for v in msg.data[:4]]
        if slot[3] <= 0.0:
            slot = [*NO_OBSTACLE_POSITION, NO_OBSTACLE_RADIUS]
        if np.allclose(slot, self.dynamic_obstacle_slot, atol=0.01):
            return
        self.get_logger().info(
            "static obstacle slot: " + ("cleared" if slot[3] == NO_OBSTACLE_RADIUS else
                                        f"({slot[0]:+.3f}, {slot[1]:+.3f}, {slot[2]:.3f}) r={slot[3]:.3f}"))
        self.dynamic_obstacle_slot = slot
        self._publish_obstacle_params(self._last_hull_active, self._last_dest_hull_active)
        self._retarget()

    def _tick(self):
        if not self.started:
            # Hold the reference on the arm until the controller starts.
            if self.measured_tcp is not None:
                self._set_ref_state(self.measured_tcp, np.zeros(3), np.zeros(3), cyl=False)
                self._set_target(self.measured_tcp, np.zeros(3))
                self._publish_goal(np.tile(self.measured_tcp, (HORIZON_STEPS + 1, 1)))
            return
        self.tick_count += 1
        if self.pause_until_tick is not None and self.tick_count >= self.pause_until_tick:
            self.pause_until_tick = None
            # PAUSING vs PAUSING_DEST tells which half to resume.
            self.state = self.MOVING_TO_BOX if self.state == self.PAUSING else self.MOVING_TO_SLOT
            self._advance_reference(self.leg_wps[0])
        if self.pending_pile_scan is not None and self.tick_count - self.pending_pile_scan[1] > DEPTH_WAIT_TICKS:
            self._process_pile_scan()
        if self.pending_dest_scan is not None and self.tick_count - self.pending_dest_scan[1] > DEPTH_WAIT_TICKS:
            self._process_dest_scan()
        if (self.state in (self.AWAITING_SCAN, self.AWAITING_DEST_SCAN)
                and self.tick_count - self.scan_requested_at_tick >= SCAN_TIMEOUT_TICKS):
            # No scan response: ask again.
            if self.state == self.AWAITING_SCAN:
                self._request_scan()
            else:
                self._request_dest_scan()

        # Hold released: re-plan from where the reference stopped, and skip a
        # pass-through point that is close or already passed.
        hold = self._hold_active()
        if self.was_holding and not hold:
            to_go = self.leg_target - self._ref_cart()[0]
            if (self.leg_pass and not self.via_active
                    and (np.linalg.norm(to_go) < PASS_SKIP_M or to_go @ self.leg_in_dir < 0.0)):
                self._arrive()
            else:
                self._retarget()
        self.was_holding = hold
        if (self.leg_timeout_ticks is not None and not hold
                and self.state in (self.MOVING_TO_BOX, self.MOVING_TO_SLOT)):
            self.leg_ticks += 1
            if self.leg_ticks > self.leg_timeout_ticks:
                wp = self.leg_wps[self.leg_idx]
                self.get_logger().warn(
                    f"leg to {np.round(self.leg_target, 3).tolist()} ({wp.action or 'move'}) not reached in "
                    f"{wp.timeout_s:.0f} s; TCP at {np.round(self.measured_tcp, 3).tolist()}; moving on")
                self.leg_timeout_ticks = None
                if wp.touch:
                    self._retry_place("the wrist load never dropped: the box is caught on something")
                elif wp.action == "hover":
                    self._retry_place("the box did not get down to above the spot")
                else:
                    self._arrive()
        if (self.via_active and self.detour_active and not self._hold_active()
                and np.linalg.norm(np.array(self.inp.current_position) - np.array(self.inp.target_position))
                < 0.01):
            self._retarget()
        self._apply_motion_limits(self.inp)
        result = self.otg.update(self.inp, self.out)
        self.out.pass_to_input(self.inp)
        # Ruckig leaves ~1e-16 on a reference at rest, which breaks the horizon's time sync.
        self.inp.current_velocity = [0.0 if abs(x) < 1e-9 else x for x in self.inp.current_velocity]
        self.inp.current_acceleration = [0.0 if abs(x) < 1e-9 else x for x in self.inp.current_acceleration]
        reached = (result == Result.Finished or np.linalg.norm(
            self._ref_cart()[0] - self.target_cart) < REF_AT_TARGET_TOL_M)
        if (not reached and self.leg_blend and not self.via_active
                and np.linalg.norm(self.leg_target - self._ref_cart()[0]) < BLEND_M):
            reached = True
        if not reached and self.leg_pass and not self.via_active:
            # Close to a pass-through point and moving away counts as passed.
            p_ref, v_ref, _ = self._ref_cart()
            to_go = self.leg_target - p_ref
            reached = bool(np.linalg.norm(to_go) < PASS_SKIP_M and to_go @ v_ref < 0.0)
        if reached and not self._hold_active():
            if self.via_active and not self.detour_active:
                self._retarget()
            elif not self.via_active and self.leg_pass:
                self._arrive()
        self._publish_goal(self._reference_horizon())
        self._send_obstacle_params()
        if self.telemetry.enabled:
            self.telemetry.row(self.state_step, self.state, round(float(np.linalg.norm(self._ref_cart()[1])), 4),
                               round(self.speed_scale, 3), int(self._hold_active()), int(self.ref_cyl),
                               int(self.via_active))

    def _reference_horizon(self):
        """Reference positions now and at each of the next HORIZON_STEPS ticks. Past a
        pass-through point it goes on at the pass speed; past a target at rest it
        stays there.
        """
        otg = Ruckig(3, CONTROL_PERIOD_S)
        inp = InputParameter(3)
        inp.current_position = self.inp.current_position
        inp.current_velocity = self.inp.current_velocity
        inp.current_acceleration = self.inp.current_acceleration
        inp.target_position = self.inp.target_position
        self._apply_motion_limits(inp)
        traj = Trajectory(3)
        out = [self._planner_to_cart(self.inp.current_position)]
        try:
            otg.calculate(inp, traj)
        except RuckigError as e:
            # Rare Ruckig numerical failure: hold the reference for this tick.
            self.get_logger().warn(f"reference horizon Ruckig solve failed ({e}); holding it this tick")
            return out * (HORIZON_STEPS + 1)
        for k in range(1, HORIZON_STEPS + 1):
            t = k * CONTROL_PERIOD_S
            if t <= traj.duration:
                pos, _, _ = traj.at_time(t)
            else:
                pos, _, _ = traj.at_time(traj.duration)
                pos = np.array(pos) + np.array(inp.target_velocity) * (t - traj.duration)
            out.append(self._planner_to_cart(pos))
        return out

    def _planner_to_cart(self, c):
        """A position in planner coordinates, in x/y/z."""
        if self.ref_cyl:
            return _cyl_to_cart(c, np.zeros(3), np.zeros(3))[0].tolist()
        return list(c)


def main():
    rclpy.init()
    node = TaskNode()
    try:
        rclpy.spin(node)
    finally:
        node.telemetry.flush()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
