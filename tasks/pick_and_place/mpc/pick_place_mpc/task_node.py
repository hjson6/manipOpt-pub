"""MPC method's task sequencing: scans the pile and the tray, picks and places
boxes, and streams a Ruckig reference (planned in cylindrical coordinates
round the base) to mpc_controller. With mobile:=true, the mobile job: the base
drives between the pick and place stations (/nav/goal, /nav/status) with the arm
in the carry pose, and the tray is found again at each visit.

See docs/implementation_notes.md#task_nodepy.
"""
import functools
import json
import time
from pathlib import Path

import numpy as np
import pinocchio as pin
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from scipy.spatial import cKDTree
from sensor_msgs.msg import JointState
from ruckig import ControlInterface, InputParameter, OutputParameter, Result, Ruckig, RuckigError, Trajectory
from std_msgs.msg import Empty, Float32MultiArray, Float64, Float64MultiArray, Bool, String

from core.collision import NO_OBSTACLE_POSITION, NO_OBSTACLE_RADIUS
from perception import heightmap, tray_detection
from pick_place_common.packing import plan_compact
from pick_place_common.telemetry import Telemetry
from pick_place_common.scene import (
    ARM_IN_BASE, BASE_CHASSIS_HALF, CARRY_TCP_POSITION, CONTAINER_CAM_FOVY_DEG, CONTAINER_CAM_HEIGHT, CONTAINER_CAM_POS_TCP, CONTAINER_CAM_WIDTH,
    DEST_WALL_THICKNESS,
    compose, container_cam_pose, tool_down_rot, tool_points_tcp, CONTAINER_CAM_ROT_TCP,
    DEPTH_MIN_RANGE_M, PLACE_ZONE_BOUNDS, PLACE_ZONE_MAX_HEIGHT_M,
    TOOL_RADIUS_M, WRIST_ABOVE_TCP_M, WRIST_EXTENT_TOOL, BOX_HALF_DIAGONAL_MAX_M, BOX_HEIGHT_MAX_M, BOX_HEIGHT_MIN_M,
    FLATNESS_TOL,
    FLOOR_Z, GRASP_INLIER_FRAC, PARK_POSITION,
    SCAN_RESOLUTION, SOURCE_MIN_FILL_FRAC, SOURCE_MIN_FOOTPRINT_CELLS, pick_zone,
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
TRANSIT_VEL = np.array([0.8, 0.8, 0.8])  # mobile job: into and out of the carry pose, high over everything
MAX_ACCEL = np.array([3.0, 3.0, 3.0])
MAX_JERK = np.array([20.0, 20.0, 20.0])
HORIZON_STEPS = 15  # one goal per MPC stage; must match MPCConfig.N
PLACE_CLEARANCE_M = 0.004  # gap to walls and placed boxes; the box-to-box gap's spread is ~2 mm sd
PACK_PUSH_AREA_M2 = 0.005  # a spot needing a push counts this much more block
# After touch-down the held box slides into the wall or box it was planned flush with,
# lifted clear of the floor, until the wrist feels it (a guarded move), then is released.
SLIDE_LIFT_M = 0.005  # clear of the floor: the arm sags a few mm under the box
SLIDE_SPEED_MPS = 0.008
SLIDE_FORCE_N = 1.0  # sideways load: touching; under a placed box's sliding friction (0.2 kg)
SLIDE_EXTRA_M = 0.006  # past the planned gap at most (the pile scan reads boxes ~1 mm large)
SLIDE_SIDE_TOL_M = 0.002  # a side planned within the clearance plus this is flush
SLIDE_MIN_M = 0.001
SLIDE_BACKOFF_M = 0.001  # clear of what it touched: sliding on along it, friction reads as a touch
SLIDE_CONFIRM_TICKS = 3  # the load held this long: a touch, not noise
SLIDE_LEG_TIMEOUT_S = 3.0
PLACE_SNAP_MAX_M = 0.02  # snap only gaps smaller than this
WRIST_MARGIN_M = 0.008  # wrist to wall
PUSH_LIFT_M = 0.02  # tool tip above the box top while moving round it
PUSH_START_GAP_M = 0.01  # tool to box face before a push
PUSH_BACKOFF_M = 0.01
PUSH_SPEED_MPS = 0.05
PUSH_TRY_M = 0.006  # smaller shifts are left as a gap (= PUSH_MIN_M: a push that short is not worth it)
PUSH_TOOL_MARGIN_M = 0.008  # tool tracking error while tilting down (a wall was tapped at 9 mm)
PUSH_MIN_M = 0.006  # a gap below this is accepted when the push is not possible
PUSH_LEG_TIMEOUT_S = 6.0  # a jammed push gives up and moves on
# A push goes at least to the planned end (a heavy box creeps: that is not a wall), then on
# until the box stops (wall or neighbour): the reference ahead of the TCP, the TCP no longer
# advancing, the wrist feeling the box. A wall nearer than planned stops it at PUSH_FORCE_MAX_N.
PUSH_OVERTRAVEL_M = 0.02
PUSH_STALL_LEAD_M = 0.008
PUSH_STALL_SPEED_MPS = 0.005  # over PUSH_STALL_WINDOW_TICKS
PUSH_STALL_WINDOW_TICKS = 10
PUSH_STALL_TICKS = 3
PUSH_CONTACT_N = 2.0
PUSH_FORCE_MAX_N = 40.0
# A push goes straight down if the tool and the arm fit; else the smallest tilt that fits.
PUSH_TILTS_RAD = np.radians(np.arange(5.0, 40.1, 5.0))
# The whole arm against the tray walls at a push pose: points from link3 to the flange (and
# between), as spheres (the forearm, link5, once pressed on a wall at 156 N).
ARM_CHECK_FRAMES = ("link3", "link4", "link5", "link6", "link7")  # tool and wrist: _push_clearance
ARM_CHECK_RADIUS_M = 0.07
ARM_WALL_MARGIN_M = 0.02
# Placement fallback (no floor spot in the packing): stable spots tried, at least this far apart.
FALLBACK_MAX_SPOTS = 20
RELEASE_WALL_MARGIN_M = 0.02  # a box closer to a wall than its half-diagonal plus this: stop above the spot
RELEASE_BOX_MARGIN_M = 0.015  # ...or to a taller placed box than this
FALLBACK_SPOT_SPACING_M = 0.01
PUSH_SAFE_BELOW_WALL_M = 0.07  # a push's safe TCP height below the wall top: link7 (0.086 m above the TCP) clears it
WRIST_HEIGHT_M = 0.055  # link7 spans 0.100-0.155 m above the TCP
FOREARM_ABOVE_FLANGE_M = 0.036  # links 5-6 start ~0.136 m above the TCP
FOREARM_WALL_MIN_M = 0.09  # flange to a wall on the robot's side while the forearm is below its top (IK search)
J7_PLACE_LIMIT_RAD = np.radians(150.0)  # joint 7's range is +-166 deg
HEADING_TURN_RATE_MAX = 1.5  # rad/s, the tool's turn on any leg (joint 7: 2.61)
MIN_TURN_LEG_SPEED_MPS = 0.01
TURN_PER_SWING_MAX = 2.0  # heading turn per base swing above which the heading follows the distance
PLACE_LEG_TIMEOUT_S = 8.0  # a blocked place releases where it is
TOUCH_SPEED_MPS = 0.02  # final descent of a place, until the surface takes the box
TOUCH_HOVER_M = 0.01  # hover this far above the tallest possible box top
TOUCH_BELOW_M = 0.005  # the touch leg aims this far below the lowest possible box top
TOUCH_FORCE_FRAC = 0.5  # contact: wrist load below this share of the box's weight
TOUCH_MIN_WEIGHT_N = 0.5  # below this the load cell cannot tell contact
TOUCH_SIDE_FORCE_N = 1.0  # a sideways wrist load while lowering: the box is on a neighbour's edge
TOUCH_FRICTION_MU = 0.8  # ...beyond the landing surface's friction (box friction 0.6)
TOUCH_AVG_TICKS = 3  # load-cell readings averaged for the touch rules
PLACE_RETRIES = 2  # re-scan and re-plan a blocked place this often, then stop
PICK_HOVER_M = 0.015  # stop this far above the sensed top to tare the load cell
PICK_BELOW_M = 0.02  # the pick leg aims this far below the sensed top; contact ends it first
PICK_TOUCH_SPEED_MPS = 0.03
PICK_CONTACT_N = 3.0  # wrist load change from the hover's: the tool is on the box
PICK_LEG_TIMEOUT_S = 6.0
GRASP_MIN_WEIGHT_N = 2.0  # after the lift, less than this in the wrist: nothing was gripped
PICK_RETRIES = 3  # re-scan the pile after a missed or failed grasp this often, then stop
PLACE_RETRY_GROW_M = 0.0025  # the held box's footprint grows by this per side at each retry
BLOCKED_GROW_M = 0.02  # a blocked set-down footprint, grown by this, is unknown where a held rescan cannot see
BLOCKED_VIEW_MARGIN_M = 0.01  # the rescan after a blocked set-down: the wrist camera this far past it
HELD_MASK_GROW_M = 0.01  # the held box's silhouette in a tray scan, per side (swing, tilt)
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
# Heights from each scan (docs/implementation_notes.md#task_nodepy).
LIFT_MARGIN_M = 0.03  # the held box, taken at the spec's tallest, above what it passes
APPROACH_MARGIN_M = 0.05  # the empty tool above the pile or the tray's contents
SCAN_MARGIN_M = 0.02
SCAN_EDGE_MARGIN_PX = 4  # the scanned area stays this far inside the image
SCAN_Z_STEP_M = 0.005
PICK_WRIST_MARGIN_M = 0.02  # sideways, link7 to a taller neighbour while the tool is on a top
ZONE_EXIT_MARGIN_M = 0.05  # the held box leaves the pick zone's azimuth range by this before descending
HOME_Q = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])  # panda_robot.xml's home keyframe
IK_ITERS = 100
IK_JOINT_MARGIN_RAD = 0.02


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
SCAN_PAUSE_TICKS_MOBILE = 10

TOOL_MASK_DILATE_PX = 3
REMEASURE_WINDOW_M = 0.006  # search a placed box's top this far round where it was recorded
REMEASURE_SIZE_TOL_M = 0.0025  # measured half-size this close to the known one: both edges are its own
REMEASURE_MAX_MOVE_M = 0.015  # a bigger correction is not believed
REMEASURE_PUSHED_WINDOW_M = 0.015  # a pushed box, once: the tool may drag it sideways
REMEASURE_PUSHED_MAX_MOVE_M = 0.025
SCAN_TIMEOUT_TICKS = 100  # re-request an unanswered scan after 2 s
DEST_NO_SPOT_MAX_RETRIES = 30  # then stop: the tray is blocked or full
EMPTY_RESCANS = 3  # pile scans with something seen but no top found, before stopping
# Mobile job.
CARRY_POS = np.array(CARRY_TCP_POSITION)
REDOCKS_MAX = 2  # the tray not found again where it was: dock again this often, then stop
NAV_RETRIES = 1
NAV_RESEND_TICKS = 100  # a goal not taken up: send it again
BASE_PAUSE_KEEP_TICKS = 50  # the base stays paused this long after the arm's hold ends (it flickers)
OVER_BASE_INSET_M = 0.05  # the tool this far inside the chassis outline: the base's fields cover the arm
STOW_XY_M = 0.05  # tucking, the tool this near the carry pose in plan: stowed enough to drive

N_OBSTACLE_SLOTS = 3  # must match MPCConfig.n_obstacles
_HULL_PAD = 0.05
TRAY_HULL_PAD_M = 0.02


class Waypoint:
    """One waypoint of a leg: what to do there and which hulls are on."""
    __slots__ = ("pos", "action", "hull_active", "dest_idx", "dest_hull_active", "pass_through",
                 "blend", "max_speed", "timeout_s", "tool_axis", "heading_offset", "touch", "transit")

    def __init__(self, pos, action=None, hull_active=False, dest_idx=None,
                 dest_hull_active=False, pass_through=False, blend=False,
                 max_speed=None, timeout_s=None, tool_axis=None, heading_offset=0.0, touch=False, transit=False):
        self.pos = pos
        self.transit = transit  # TRANSIT_VEL instead of MAX_VEL
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


@functools.lru_cache(maxsize=1)
def _tool_mask():
    """Wrist-camera pixels the tool covers (rigid mount): invalid in every frame."""
    return heightmap.box_silhouette(
        tool_points_tcp(), CONTAINER_CAM_POS_TCP, CONTAINER_CAM_ROT_TCP, CONTAINER_CAM_FOVY_DEG,
        CONTAINER_CAM_WIDTH, CONTAINER_CAM_HEIGHT, dilate_px=TOOL_MASK_DILATE_PX)


def _scan_tcp(bounds, top_z, floor_z=FLOOR_Z, heading_offset=0.0):
    """TCP position for a scan of `bounds`: the wrist camera over its middle (for the
    heading the arm has there), at the lowest height that keeps it in view from floor_z
    to top_z, above the tool's pixels, with top_z beyond the depth camera's near limit."""
    centre = np.array([np.mean(bounds[0]), np.mean(bounds[1])])
    xy = centre.copy()
    for _ in range(3):
        rot = tool_down_rot(TaskNode._aligned_yaw(float(np.arctan2(xy[1], xy[0]))) + heading_offset)
        xy = centre - (rot @ CONTAINER_CAM_POS_TCP)[:2]
    m, w, h = SCAN_EDGE_MARGIN_PX, CONTAINER_CAM_WIDTH, CONTAINER_CAM_HEIGHT
    bottom = int(np.nonzero(_tool_mask().any(axis=1))[0].min()) if _tool_mask().any() else h
    corners = [(x, y, z) for x in bounds[0] for y in bounds[1] for z in (floor_z, top_z)]
    f = (h / 2.0) / np.tan(np.radians(CONTAINER_CAM_FOVY_DEG) / 2.0)
    z = top_z + DEPTH_MIN_RANGE_M + CONTAINER_CAM_POS_TCP[2]
    while z < top_z + 2.0:
        for _ in range(2):
            cam_pos, cam_mat = container_cam_pose((*xy, z), rot)
            uv = [heightmap.project_world_point(cam_pos, cam_mat, CONTAINER_CAM_FOVY_DEG, w, h, c) for c in corners]
            if any(p is None for p in uv):
                break
            # Centre the area in the rows above the tool.
            dv = (min(p[1] for p in uv) + max(p[1] for p in uv)) / 2.0 - bottom / 2.0
            xy = xy - dv * (cam_pos[2] - (floor_z + top_z) / 2.0) / f * cam_mat[:2, 1]
        if all(p is not None and m <= p[0] <= w - m and m <= p[1] <= bottom - m for p in uv):
            break
        z += SCAN_Z_STEP_M
    return np.array([xy[0], xy[1], z + SCAN_MARGIN_M])


def _halves(bounds):
    """The two halves of a rectangle, split across its longer side."""
    (x0, x1), (y0, y1) = bounds
    if x1 - x0 >= y1 - y0:
        xm = (x0 + x1) / 2.0
        return [((x0, xm), (y0, y1)), ((xm, x1), (y0, y1))]
    ym = (y0 + y1) / 2.0
    return [((x0, x1), (y0, ym)), ((x0, x1), (ym, y1))]


def _build_source_legs(box_pos, approach_z, lift_z):
    """Approach, hover (the load cell tares), guarded descent to contact (the pick)
    and lift legs for a box whose sensed top is at box_pos."""
    x, y, top = box_pos
    return [
        Waypoint(np.array([x, y, approach_z]), hull_active=False, pass_through=True),
        Waypoint(np.array([x, y, top + PICK_HOVER_M]), hull_active=False),
        Waypoint(np.array([x, y, top - PICK_BELOW_M]), action="pick", hull_active=False, touch=True,
                 max_speed=PICK_TOUCH_SPEED_MPS, timeout_s=PICK_LEG_TIMEOUT_S),
        Waypoint(np.array([box_pos[0], box_pos[1], lift_z]), hull_active=False, pass_through=True, blend=True),
    ]


def _build_dest_legs(hover_pos, touch_pos, dest_idx, enter_z, leave_z, push_legs=(), heading_offset=0.0,
                     stop_above=False):
    """Release-above (at enter_z), hover, touch-down (place), optional push and
    lift-off (to leave_z) legs for the chosen spot. The box turns by
    heading_offset on the way to above the spot; stop_above: stop there before going
    down, rather than pass through (a spot near a wall)."""
    release_above = np.array([hover_pos[0], hover_pos[1], enter_z])
    return [
        Waypoint(release_above, hull_active=False, dest_idx=dest_idx, pass_through=not stop_above,
                 heading_offset=heading_offset),
        Waypoint(hover_pos, action="hover", hull_active=False, heading_offset=heading_offset,
                 timeout_s=PLACE_LEG_TIMEOUT_S),
        Waypoint(touch_pos, action="place", hull_active=False, timeout_s=PLACE_LEG_TIMEOUT_S,
                 heading_offset=heading_offset, max_speed=TOUCH_SPEED_MPS, touch=True),
        *push_legs,
        Waypoint(np.array([hover_pos[0], hover_pos[1], leave_z]), hull_active=False, pass_through=True, blend=True,
                 heading_offset=0.0 if push_legs else heading_offset),
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


def _push_legs(start_xy, centre, half, axis, push, surface, height, tilted, safe_tcp_z):
    """Legs that push a released box (centre, half-extents) by `push` metres along
    `axis`, starting from the TCP at start_xy: up clear of the walls, over the
    box's far side, down to mid-height with the tool along `tilted`, push, back
    off, up.
    """
    sign = np.sign(push)
    safe_z = max(surface + height + PUSH_LIFT_M, safe_tcp_z)
    start = np.array([centre[0], centre[1], safe_z])
    start[axis] -= sign * (half[axis] + TOOL_RADIUS_M + PUSH_START_GAP_M)
    down = start.copy()
    down[2] = surface + height / 2.0
    end = down.copy()
    end[axis] += push + sign * (PUSH_START_GAP_M + PUSH_OVERTRAVEL_M)
    back = end.copy()
    back[axis] -= sign * PUSH_BACKOFF_M
    shift = _push_centring(tilted, axis, height)
    start, down, end, back = start + shift, down + shift, end + shift, back + shift
    return [
        Waypoint(np.array([start_xy[0], start_xy[1], safe_z]), timeout_s=PUSH_LEG_TIMEOUT_S),
        Waypoint(start, timeout_s=PUSH_LEG_TIMEOUT_S),
        Waypoint(down, timeout_s=PUSH_LEG_TIMEOUT_S, tool_axis=tilted),
        Waypoint(end, action="push", max_speed=PUSH_SPEED_MPS, timeout_s=PUSH_LEG_TIMEOUT_S, tool_axis=tilted),
        Waypoint(back, action="pushed", timeout_s=PUSH_LEG_TIMEOUT_S, tool_axis=tilted),
        Waypoint(np.array([back[0], back[1], safe_z]), timeout_s=PUSH_LEG_TIMEOUT_S),
    ]


class TaskNode(Node):
    """Per box: RETURNING -> AWAITING_SCAN -> PAUSING -> MOVING_TO_BOX (approach,
    pick, lift; the spot is chosen at the grasp from the last tray scan) ->
    MOVING_TO_SLOT (release-above, place, lift-off) -> TRAVELING_TO_DEST ->
    AWAITING_DEST_SCAN (empty hand). The first tray scan comes before the first
    pick. A blocked place, or no spot, rescans the tray with the box held
    (TRAVELING_TO_DEST -> AWAITING_DEST_SCAN -> PAUSING_DEST). PARKED when done.
    Mobile job: TUCKING (to the carry pose: across at height, then down) -> DRIVING
    (the base to a station) -> UNTUCKING (straight up to the next leg's height) between
    the stations: after the empty-hand tray scan to the pile, after the lift to the
    tray (the place planned there from a scan with the box held), at the end home.
    The base backs out of a dock while the arm tucks, and the arm starts the next
    station's first move on the straight approach; scans wait until it has docked.
    """
    (RETURNING, AWAITING_SCAN, PAUSING, MOVING_TO_BOX,
     TRAVELING_TO_DEST, AWAITING_DEST_SCAN, PAUSING_DEST, MOVING_TO_SLOT,
     PARKED, TUCKING, DRIVING, UNTUCKING) = range(12)

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
        self.leg_vel = MAX_VEL
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
        self.wrist_f_filt = None
        self.wrist_recent = []
        self.wrist_f_avg = np.zeros(3)
        self.push_tare = np.zeros(2)
        self.push_tcps = []
        self.push_stall_ticks = 0
        self.touch_baseline = None
        self.touch_side_tare = np.zeros(2)
        self.place_attempts = 0
        self.next_place_legs = None  # chosen at the grasp
        self.blocked_rect = None  # (x0, x1, y0, y1) of the last blocked set-down
        self.pick_rot = None  # TCP rotation at the grasp
        self.tray = None  # found in the first tray scan
        self.tray_grid = None  # (points, shape) of the tray heightmap
        self.pile_clear_z = None  # TCP height that carries any spec box clear of the rest of the pile
        # Grows the static obstacle while carrying (current_box_size stays set after a place).
        self.holding_box = False
        self.pick_contact = False
        self.pick_failures = 0
        self.grasp_tare = None
        self.dest_no_spot_count = 0  # consecutive placement scans with no flat spot
        self.empty_rescans = 0
        # (x, y, hx, hy, top z, surface z) of every box placed so far.
        self.placed_boxes = []
        self.pushed_unmeasured = set()  # placed_boxes indices
        self.current_box_offset = None
        self.pending_place = None
        self.slide_from = None  # the TCP at touch-down when the box slides before release
        self.slide_tare = np.zeros(2)
        self.slide_ticks = 0
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
        self.leg_wp_index = None
        self.leg_single = True
        self.leg_by_swing = False
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
        self.pile_top_pub = self.create_publisher(Float64, "/task/pile_top", 10)
        # The held box for the ceiling camera's self-filter: [held, offset (3), axes (9), hx, hy, height],
        # TCP frame (camera_detection_node).
        self.held_box_pub = self.create_publisher(
            Float64MultiArray, "/task/held_box", QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self._publish_held_box()
        self.latest_depth = None  # the pile scan's frame, sent just before its heightmap
        self.latest_depth_tick = -1
        self.tool_mask = _tool_mask()
        self.create_subscription(Float32MultiArray, "/sim/container_depth", self._on_container_depth, 10)
        self.dest_depth = None  # the tray scan's frame; the heightmap is made here from it
        self.tray_find_frames = None  # the first tray scan: one frame per half of the place zone
        self.dest_depth_tick = -1
        self.create_subscription(Float32MultiArray, "/sim/destination_depth", self._on_destination_depth, 10)
        self.create_subscription(Float64MultiArray, "/sim/wrist_force", self._on_wrist_force, 10)
        self.dest_scan_pub = self.create_publisher(Empty, "/sim/scan_destination", 10)
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
        self.ik_data = self.pin_model.createData()
        self.measured_q = None
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
        self.telemetry = Telemetry("task", ["step", "state", "ref_speed", "speed_scale", "hold", "cyl", "via",
                                            "touch_base", "station"])
        self.dynamic_slot_inflated = [*NO_OBSTACLE_POSITION, NO_OBSTACLE_RADIUS]
        self.declare_parameter("mobile", False)
        self.declare_parameter("max_boxes", 0)  # mobile: home after this many (0: all)
        self.mobile = bool(self.get_parameter("mobile").value)
        # The pick zone at the pick dock, or the cell's; pile_top: the last pile scan's highest point,
        # the zone's highest until scanned.
        self.source_bounds, self.source_grid, self.source_shape, self.pile_top = pick_zone(self.mobile)
        self.max_boxes = int(self.get_parameter("max_boxes").value)
        self.station = "home" if self.mobile else None  # where the base is docked; None while driving
        self.drive = None  # (goal, what to do there) of the drive under way
        self.nav_seen_goal = False
        self.nav_sent_tick = 0
        self.nav_retries = 0
        self.reanchor_pending = False
        self.redocks = 0
        self.tuck_down = False  # tucking: across at height done, now down into the carry pose
        self.untuck_next = None  # (state, waypoint) after rising out of the carry pose
        self.docked = True  # the base standing at self.station (not from a drive's start to its end)
        self.arrival_started = False  # the station's first move begun on the approach
        self.deferred_scan = None  # a scan asked for before the base had docked
        self.flags_sent = (None, None)  # (arm stowed, pause the base) last sent
        self.held_tick = -BASE_PAUSE_KEEP_TICKS
        if self.mobile:
            self.nav_goal_pub = self.create_publisher(String, "/nav/goal", 10)
            self.create_subscription(String, "/nav/status", self._on_nav_status, 10)
            latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self.stowed_pub = self.create_publisher(Bool, "/task/arm_stowed", latched)
            self.pause_pub = self.create_publisher(Bool, "/task/pause_base", latched)
            self._publish_base_flags()

    def _on_joint_state(self, msg: JointState):
        self.measured_q = np.array(msg.position[:7])
        pin.framesForwardKinematics(self.pin_model, self.pin_data, self.measured_q)
        self.measured_tcp = np.array(self.pin_data.oMf[self.tcp_frame_id].translation)
        self.proxy_pos = [(np.array(self.pin_data.oMf[f].translation), r) for f, r in self.proxy_frames]
        self.state_step = int(msg.header.frame_id) if msg.header.frame_id else self.state_step + 1
        self._tick()

    def _on_go(self):
        if self.started or self.measured_tcp is None:
            return
        self.started = True
        self._set_ref_state(self.measured_tcp, np.zeros(3), np.zeros(3), cyl=True)
        # The tray first, with an empty hand.
        if self.mobile:
            self._drive_to("place", self._at_place)
            return
        self._go_scan_tray()

    def _drive_to(self, goal, then):
        """Mobile job: the arm to the carry pose and the base to goal (a station or home);
        then() once there. The base backs out of a dock while the arm tucks; it leaves
        the docking line once the arm is stowed (nav_node)."""
        if not self._reach_ok([Waypoint(CARRY_POS)], "carry pose"):
            return
        self.drive = (goal, then)
        self._tuck()
        self._send_drive()

    def _tuck(self):
        """The arm to the carry pose: across at height, then down."""
        self.state = self.TUCKING
        self.untuck_next = None
        cur = self._ref_cart()[0]
        self.tuck_down = bool(np.hypot(*(cur[:2] - CARRY_POS[:2])) < 0.01)
        above = np.array([CARRY_POS[0], CARRY_POS[1], max(float(cur[2]), CARRY_POS[2])])
        self._advance_reference(Waypoint(CARRY_POS.copy() if self.tuck_down else above, hull_active=True,
                                         dest_hull_active=True, transit=True, pass_through=not self.tuck_down,
                                         blend=not self.tuck_down))

    def _go(self, state, wp):
        """Head for wp in state; in the mobile job, out of the carry pose first straight
        up to wp's height (a low box swung over a tray's wall would hit it)."""
        cur = self._ref_cart()[0]
        if self.mobile and np.hypot(*(cur[:2] - CARRY_POS[:2])) < 0.01 and wp.pos[2] > cur[2] + 0.01:
            self.untuck_next = (state, wp)
            self.state = self.UNTUCKING
            self._advance_reference(Waypoint(np.array([cur[0], cur[1], wp.pos[2]]), transit=True, pass_through=True,
                                             blend=True))
            return
        self.state = state
        self._advance_reference(wp)

    def _send_drive(self):
        goal = self.drive[0]
        self.docked = False
        self.arrival_started = False
        self.deferred_scan = None
        self.nav_seen_goal = False
        self.nav_sent_tick = self.tick_count
        self.nav_goal_pub.publish(String(data=goal))
        self.get_logger().info(f"to {goal}: the base sets off as the arm stows")

    def _abort_arrival(self, why):
        """The base docking again (or failed) with the arm out: back to the carry pose; the
        base waits off the docking line until it is stowed."""
        self.arrival_started = False
        self.deferred_scan = None
        self.get_logger().warn(f"{why}: the arm back to the carry pose")
        self._tuck()

    def _publish_base_flags(self):
        """To the navigation: the arm stowed (the base leaves the docking line only then; the tool
        over the carry pose counts, whatever its height, so a hold there does not keep it),
        and a pause while the supervisor holds the arm out beyond the chassis and the base
        is not docked (held over the chassis, the base's fields cover it). On change and
        every second."""
        stowed = bool(self.state == self.DRIVING or (self.state == self.TUCKING and self._near_carry()))
        if self.started and self._hold_active():
            self.held_tick = self.tick_count
        pause = bool(self.tick_count - self.held_tick < BASE_PAUSE_KEEP_TICKS and not self.docked and not stowed
                     and not self._over_base())
        if (stowed, pause) != self.flags_sent or self.tick_count % 50 == 0:
            if pause != self.flags_sent[1] and self.flags_sent[1] is not None:
                self.get_logger().info(f"{'pausing' if pause else 'resuming'} the base: the arm "
                                       f"{'held' if pause else 'free or over the chassis'} beyond the chassis")
            self.stowed_pub.publish(Bool(data=stowed))
            self.pause_pub.publish(Bool(data=pause))
            self.flags_sent = (stowed, pause)

    def _near_carry(self):
        """The tool within STOW_XY_M (in plan) of the carry pose: over the deck, whatever height."""
        return self.measured_tcp is not None and bool(np.hypot(*(self.measured_tcp[:2] - CARRY_POS[:2])) < STOW_XY_M)

    def _over_base(self):
        """The tool over the chassis (in plan, OVER_BASE_INSET_M inside its outline)."""
        if self.measured_tcp is None:
            return True
        x, y, _ = compose(ARM_IN_BASE, (float(self.measured_tcp[0]), float(self.measured_tcp[1]), 0.0))
        return bool(abs(x) <= BASE_CHASSIS_HALF[0] - OVER_BASE_INSET_M and abs(y) <= BASE_CHASSIS_HALF[1] - OVER_BASE_INSET_M)

    def _on_nav_status(self, msg):
        if self.drive is None:
            return
        state, goal = msg.data.split()[:2]
        if goal != self.drive[0]:
            return
        if state not in ("docked", "parked", "failed"):
            self.nav_seen_goal = True
            if state == "approach" and self.state == self.DRIVING and not self.arrival_started:
                # The straight approach: the arm sets off on the station's first move.
                self.arrival_started = True
                self.station = goal
                self.get_logger().info(f"approaching {goal}: the arm sets off")
                self.drive[1]()
            elif state in ("backout", "align", "route", "planning") and self.arrival_started:
                self._abort_arrival(f"{state} at {goal}")
            return
        if not self.nav_seen_goal:
            return  # the status from before the goal was taken up
        if state == "failed":
            if self.arrival_started:
                self._abort_arrival(f"navigation to {goal} failed")
            if self.nav_retries < NAV_RETRIES:
                self.nav_retries += 1
                self.get_logger().warn(f"navigation to {goal} failed; trying again")
                self._send_drive()
                return
            self.get_logger().error(f"navigation to {goal} failed; stopping")
            self.state = self.PARKED
            return
        self.nav_retries = 0
        self.station = goal
        self.docked = True
        then, self.drive = self.drive[1], None
        self.get_logger().info(f"{state} at {goal}")
        if self.arrival_started:
            self.arrival_started = False
            if self.deferred_scan is not None:
                request, self.deferred_scan = self.deferred_scan, None
                request()
            return
        then()

    def _at_place(self):
        """At the place station: find the tray (first visit), or scan it from above (the
        box held), where it is found again before the place is planned."""
        if self.tray is None:
            self._go_scan_tray()
            return
        self.reanchor_pending = True
        pos = self._tray_scan_pos(held=self.holding_box)
        if not self._reach_ok([Waypoint(pos)], "tray scan"):
            return
        self._go(self.TRAVELING_TO_DEST, Waypoint(pos, transit=True))

    def _at_pick(self):
        pos = self._pile_scan_pos()
        if not self._reach_ok([Waypoint(pos)], "pile scan"):
            return
        self._go(self.RETURNING, Waypoint(pos, transit=True))

    def _at_home(self):
        self.state = self.PARKED
        self.get_logger().info(f"job done: {self.boxes_moved} boxes moved; parked indefinitely at home")

    def _reanchor_tray(self):
        """The tray found again in this scan (the base docked a little differently): the
        tray, the placed boxes and the last heightmap moved with it. False if it is not
        where it was (docking again, or stopping)."""
        cam_pos, cam_mat, fovy, w, h, depth = self.dest_depth
        depth = depth.copy()
        if self.holding_box:
            depth[self._held_silhouette(cam_pos, cam_mat, fovy, w, h)] = 0.0
        new, why = tray_detection.reanchor(self.tray, [(depth, cam_pos, cam_mat, fovy, w, h)])
        if new is None:
            if self.redocks < REDOCKS_MAX:
                self.redocks += 1
                self.get_logger().warn(f"the tray is not where it was ({why}); docking again "
                                       f"({self.redocks}/{REDOCKS_MAX})")
                self._drive_to("place", self._at_place)
            else:
                self.get_logger().error(f"the tray is not where it was after docking again ({why}); stopping")
                self.state = self.PARKED
            return False
        self.redocks = 0
        shift = new.center[:2] - self.tray.center[:2]
        old_points = np.asarray(self.tray_grid[0])
        self.tray = new
        self.tray_grid = heightmap.build_dense_grid_xy(*new.bounds, SCAN_RESOLUTION)
        self.placed_boxes = [(b[0] + shift[0], b[1] + shift[1], *b[2:]) for b in self.placed_boxes]
        if self.latest_dest_hmap is not None:
            dist, idx = cKDTree(old_points).query(np.asarray(self.tray_grid[0]) - shift)
            vals = self.latest_dest_hmap.ravel()[idx].astype(float)
            vals[dist > SCAN_RESOLUTION] = np.nan
            self.latest_dest_hmap = vals.reshape(self.tray_grid[1])
        self.get_logger().info(f"tray found again: moved {1e3 * shift[0]:+.1f}, {1e3 * shift[1]:+.1f} mm since the "
                               f"last visit")
        return True

    def _request_scan(self):
        if self.mobile and not self.docked:
            self.deferred_scan = self._request_scan  # at the scan pose before the base has docked
            return
        self.state = self.AWAITING_SCAN
        self.scan_requested_at_tick = self.tick_count
        self.scan_pub.publish(Empty())

    def _request_dest_scan(self):
        if self.mobile and not self.docked:
            self.deferred_scan = self._request_dest_scan
            return
        self.state = self.AWAITING_DEST_SCAN
        self.scan_requested_at_tick = self.tick_count
        self.dest_scan_pub.publish(Empty())

    def _process_pile_scan(self):
        cam_pos, cam_mat, fovy, w, h, depth = self.latest_depth
        heights, _ = heightmap.infer_heights_parallax_corrected(
            depth, cam_pos, cam_mat, fovy, w, h, self.source_grid, (FLOOR_Z + self.pile_top) / 2.0, FLOOR_Z)
        hmap = heights.reshape(self.source_shape)
        self.latest_heightmap = hmap
        self.pile_top = max(float(np.max(hmap)), FLOOR_Z)
        self.pile_top_pub.publish(Float64(data=self.pile_top))
        boxes = heightmap.find_topmost_boxes(
            hmap, floor_z=FLOOR_Z, flatness_tol=FLATNESS_TOL,
            min_footprint_cells=SOURCE_MIN_FOOTPRINT_CELLS,
            inlier_frac=GRASP_INLIER_FRAC, min_fill_frac=SOURCE_MIN_FILL_FRAC)

        if not boxes and (hmap > FLOOR_Z + BOX_HEIGHT_MIN_M / 2.0).sum() >= SOURCE_MIN_FOOTPRINT_CELLS:
            # Something is there but no top passed: noise can spoil a small top.
            self.empty_rescans += 1
            if self.empty_rescans <= EMPTY_RESCANS:
                self.get_logger().warn(f"pile scan: something in the pick zone but no box top found; "
                                       f"rescanning ({self.empty_rescans}/{EMPTY_RESCANS})")
                self._request_scan()
                return
            self.get_logger().warn("pile scan: what is left in the pick zone is not a box top it can pick")
        self.empty_rescans = 0
        if not boxes and self.mobile:
            self.get_logger().info("container empty; driving home")
            self._publish_decision("pick", None, "container empty, done")
            self._drive_to("home", self._at_home)
            return
        if not boxes:
            self.get_logger().info("container empty; parked indefinitely")
            self._publish_decision("pick", None, "container empty, done")
            self.state = self.PARKED
            self._advance_reference(Waypoint(self._pile_scan_pos(), hull_active=False, dest_hull_active=True))
            return

        # Grasp at the centre of the sensed top; the footprint is the top's size.
        tops = [(r, self._top_footprint(r)) for r in heightmap.pick_order(hmap, boxes, FLATNESS_TOL)]
        clear = [self._pick_wrist_clear(hmap, fp[0], fp[1], r[4]) for r, fp in tops]
        i = clear.index(True) if any(clear) else 0
        if i > 0:
            self.get_logger().info(f"pick order: {i} skipped, the wrist would hit a taller neighbour")
        elif not clear[0]:
            self.get_logger().warn("pick order: every top has a taller neighbour in the wrist's way")
        (row0, col0, row1, col1, height, area), (x, y, hx_s, hy_s) = tops[i]
        # The rest with a fully visible top, in pick order, for the placement look-ahead.
        hidden = {id(r) for r, f in zip(boxes, heightmap.touches_higher(hmap, boxes, FLATNESS_TOL)) if f}
        self.upcoming_sizes = [fp[2:] for j, (r, fp) in enumerate(tops) if j != i and id(r) not in hidden]
        box_pos = np.array([x, y, height])
        self.pick_sensed = (x, y, hx_s, hy_s)
        pts = np.asarray(self.source_grid).reshape(*self.source_shape, 2)
        rest = (np.abs(pts[..., 0] - x) > hx_s + SCAN_RESOLUTION) | (np.abs(pts[..., 1] - y) > hy_s + SCAN_RESOLUTION)
        rest_top = max(float(np.max(hmap[rest])) if rest.any() else FLOOR_Z, FLOOR_Z)
        # Leaving: the swing starts BLEND_M below the lift point.
        self.pile_clear_z = rest_top + BOX_HEIGHT_MAX_M + LIFT_MARGIN_M + BLEND_M
        lift_z = max(self.pile_clear_z, self._tray_enter_z())
        self.leg_wps = _build_source_legs(box_pos, self.pile_top + APPROACH_MARGIN_M, lift_z)
        self.leg_idx = 0
        if not self._reach_ok(self.leg_wps, "pick"):
            return
        self.get_logger().info(f"heights: pile top {self.pile_top:.3f}, rest of the pile {rest_top:.3f}, "
                               f"approach {self.pile_top + APPROACH_MARGIN_M:.3f}, lift {lift_z:.3f} m")
        (sx0, _), (sy0, _) = self.source_bounds
        self._publish_decision(
            "pick", [sx0 + col0 * SCAN_RESOLUTION, sx0 + (col1 - 1) * SCAN_RESOLUTION,
                     sy0 + row0 * SCAN_RESOLUTION, sy0 + (row1 - 1) * SCAN_RESOLUTION, height],
            f"biggest of {len(boxes)} seen")

        self.get_logger().info(
            f"scanning container... selected ({x:.3f}, {y:.3f}) "
            f"(sensed height {height:.3f}m, footprint area {area} cells)"
        )
        self.state = self.PAUSING
        self.pause_until_tick = self.tick_count + (SCAN_PAUSE_TICKS_MOBILE if self.mobile else SCAN_PAUSE_TICKS)

    def _parse_depth(self, msg):
        """A wrist-camera frame, with the camera's pose from the measured joint angles
        and the hand-eye calibration (the arm is at rest at a scan); the pose the
        plant puts in the message is not used."""
        d = np.asarray(msg.data, dtype=float)
        w, h = int(d[13]), int(d[14])
        depth = d[15:].reshape(h, w)
        depth[self.tool_mask] = 0.0
        rot = np.array(self.pin_data.oMf[self.tcp_frame_id].rotation)
        cam_pos, cam_mat = container_cam_pose(self.measured_tcp, rot)
        return cam_pos, cam_mat, CONTAINER_CAM_FOVY_DEG, w, h, depth

    def _on_container_depth(self, msg: Float32MultiArray):
        self.latest_depth = self._parse_depth(msg)
        self.latest_depth_tick = self.tick_count
        if self.state == self.AWAITING_SCAN and self.latest_depth_tick >= self.scan_requested_at_tick:
            self._process_pile_scan()

    def _on_destination_depth(self, msg: Float32MultiArray):
        self.dest_depth = self._parse_depth(msg)
        self.dest_depth_tick = self.tick_count
        if self.state == self.AWAITING_DEST_SCAN and self.dest_depth_tick >= self.scan_requested_at_tick:
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
        moved = []
        for i, (cx, cy, hx, hy, top, surf) in enumerate(self.placed_boxes):
            pushed = i in self.pushed_unmeasured
            m = REMEASURE_PUSHED_WINDOW_M if pushed else REMEASURE_WINDOW_M
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
            self.pushed_unmeasured.discard(i)
            if d > (REMEASURE_PUSHED_MAX_MOVE_M if pushed else REMEASURE_MAX_MOVE_M):
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
        (x0, _), (y0, _) = self.source_bounds
        r = SCAN_RESOLUTION
        if self.latest_depth is not None and self.latest_depth_tick >= self.scan_requested_at_tick:
            cam_pos, cam_mat, fovy, w, h, depth = self.latest_depth
            e = heightmap.top_extent(depth, cam_pos, cam_mat, fovy, w, h,
                                     (x0 + (col0 - 1) * r, x0 + col1 * r), (y0 + (row0 - 1) * r, y0 + row1 * r),
                                     height, FLATNESS_TOL / 2.0,
                                     seed_xy=heightmap.footprint_center_xy(
                                         row0, col0, (row1 - row0, col1 - col0), *self.source_bounds, r))
            if e is not None:
                return (e[0] + e[1]) / 2, (e[2] + e[3]) / 2, (e[1] - e[0]) / 2, (e[3] - e[2]) / 2
        self.get_logger().warn("no depth frame for this scan; footprint from the heightmap cells")
        x, y = heightmap.footprint_center_xy(row0, col0, (row1 - row0, col1 - col0), *self.source_bounds, r)
        return x, y, (col1 - col0) * r / 2, (row1 - row0) * r / 2

    def _on_wrist_force(self, msg: Float64MultiArray):
        f = np.array(msg.data[:3], dtype=float)
        self.wrist_f_filt = f if self.wrist_f_filt is None else 0.9 * self.wrist_f_filt + 0.1 * f
        self.wrist_recent = (self.wrist_recent + [f])[-TOUCH_AVG_TICKS:]
        f = np.mean(self.wrist_recent, axis=0)
        self.wrist_f_avg = f
        # Sideways load relative to the hover's reading (load-cell tare before contact).
        fx, fy = f[:2] - self.touch_side_tare
        side = float(np.hypot(fx, fy))
        self.wrist_fz = float(f[2])
        picking = (self.state == self.MOVING_TO_BOX and self.leg_idx < len(self.leg_wps)
                   and self.leg_wps[self.leg_idx].action == "pick" and not self.pick_contact
                   and not self._hold_active())
        if picking and self.touch_baseline is not None and abs(self.wrist_fz - self.touch_baseline) > PICK_CONTACT_N:
            self._pick_contact()
            return
        touching = (self.state == self.MOVING_TO_SLOT and self.leg_idx < len(self.leg_wps)
                    and self.leg_wps[self.leg_idx].touch and not self._hold_active())
        # The surface it lands on can push sideways only by friction on the load it has
        # taken; more than that is something else (a neighbour's edge).
        taken = 0.0 if self.touch_baseline is None else max(0.0, self.touch_baseline - self.wrist_fz)
        if touching and side > TOUCH_FRICTION_MU * taken + TOUCH_SIDE_FORCE_N:
            self._retry_place(f"{side:.1f} N sideways ({fx:+.2f}, {fy:+.2f}) while lowering at "
                              f"TCP {np.round(self.measured_tcp, 3).tolist()}: the box is on something's edge")
        elif (touching and self.touch_baseline is not None and self.touch_baseline > TOUCH_MIN_WEIGHT_N
                and self.wrist_fz < TOUCH_FORCE_FRAC * self.touch_baseline):
            self._touch_down(f"wrist load {self.wrist_fz:.2f} of {self.touch_baseline:.2f} N")

    def _check_push_stall(self):
        """End a push when the box has stopped (against a wall or a neighbour): back off
        from where the tool is, not from the planned end."""
        d = self.leg_in_dir
        tcp = self.measured_tcp
        self.push_tcps = (self.push_tcps + [tcp.copy()])[-PUSH_STALL_WINDOW_TICKS:]
        if len(self.push_tcps) < PUSH_STALL_WINDOW_TICKS:
            return
        lead = float((self._ref_cart()[0] - tcp) @ d)
        speed = float((tcp - self.push_tcps[0]) @ d) / ((PUSH_STALL_WINDOW_TICKS - 1) * CONTROL_PERIOD_S)
        force = float((self.wrist_f_avg[:2] - self.push_tare) @ d[:2])  # the arm pushing the tool on
        travelled = float((tcp - self.leg_start) @ d)
        planned = float((self.leg_target - self.leg_start) @ d) - PUSH_OVERTRAVEL_M
        stalled = (lead > PUSH_STALL_LEAD_M and speed < PUSH_STALL_SPEED_MPS and force > PUSH_CONTACT_N
                   and travelled > planned)
        self.push_stall_ticks = self.push_stall_ticks + 1 if stalled else 0
        if self.push_stall_ticks < PUSH_STALL_TICKS and force < PUSH_FORCE_MAX_N:
            return
        self.get_logger().info(f"push: the box stopped ({force:.1f} N, TCP {1e3 * lead:.0f} mm short of the "
                               f"aim point); backing off")
        # The reference is not reset to the tool: the back-off leg starts from where it is, so
        # the push force eases off rather than letting go in one tick.
        back = tcp - d * PUSH_BACKOFF_M
        for wp in self.leg_wps[self.leg_idx + 1:self.leg_idx + 3]:
            wp.pos = np.array([back[0], back[1], wp.pos[2]])
        self.leg_timeout_ticks = None
        self._arrive()

    def _record_push_end(self):
        """The pushed box's record along the push from where the tool stopped (its side on
        the box's face); the next tray scan re-measures it wider."""
        if not self.placed_boxes:
            return
        d = self.leg_in_dir
        axis = int(np.argmax(np.abs(d[:2])))
        cx, cy, hx, hy, top, surf = self.placed_boxes[-1]
        c = [cx, cy]
        c[axis] = float(self.measured_tcp[axis] + np.sign(d[axis]) * (TOOL_RADIUS_M + (hx, hy)[axis]))
        self.placed_boxes[-1] = (c[0], c[1], hx, hy, top, surf)
        self.pushed_unmeasured.add(len(self.placed_boxes) - 1)
        self.get_logger().info(f"push: box recorded {1e3 * (c[axis] - (cx, cy)[axis]):+.1f} mm along "
                               f"{'xy'[axis]} from the plan, where the tool stopped")

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
        x, y, hx, hy, _surface, _psi, shift = self.place_ctx
        cx, cy, g = x + shift[0], y + shift[1], BLOCKED_GROW_M
        self.blocked_rect = (cx - hx - g, cx + hx + g, cy - hy - g, cy + hy + g)
        self.current_box_size[:2] += PLACE_RETRY_GROW_M
        self.get_logger().warn(f"place blocked ({why}); lifting and re-planning with the footprint "
                               f"grown to {2e3 * self.current_box_size[0]:.0f} x {2e3 * self.current_box_size[1]:.0f} mm")
        self._set_ref_state(self.measured_tcp, np.zeros(3), np.zeros(3), cyl=self.ref_cyl)
        up = np.array([self.measured_tcp[0], self.measured_tcp[1], self._tray_enter_z()])
        self.leg_wps = [Waypoint(up), Waypoint(self._blocked_view_pos(), action="rescan")]
        self.leg_idx = 0
        self.state = self.MOVING_TO_SLOT
        self._advance_reference(self.leg_wps[0])

    def _blocked_view_pos(self):
        """The held rescan after a blocked set-down: the wrist camera just past the blocked
        spot and the held box beyond it, so the camera sees the spot (from the usual scan
        pose the box hides it, and no spot is found there again). The usual scan pose if
        out of reach."""
        x0, x1, y0, y1 = self.blocked_rect
        c = np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0])
        z = self._tray_scan_pos(held=True)[2]
        xy = c.copy()
        for _ in range(3):
            psi = self._aligned_yaw(float(np.arctan2(xy[1], xy[0])))
            e = tool_down_rot(psi)[:2, 1]  # from the camera to the TCP
            half = abs(e[0]) * (x1 - x0) / 2.0 + abs(e[1]) * (y1 - y0) / 2.0
            xy = c + (-CONTAINER_CAM_POS_TCP[1] + half + BLOCKED_VIEW_MARGIN_M) * e
        pos = np.array([xy[0], xy[1], z])
        if self._ik_reachable(pos, self._aligned_yaw(float(np.arctan2(xy[1], xy[0])))):
            return pos
        return self._tray_scan_pos(held=True)

    def _plan_hz(self):
        """Half-height to plan a placement with: measured, else the spec's lowest
        (the wrist then comes lowest, the conservative case for wall clearance)."""
        return (self.box_height if self.box_height is not None else BOX_HEIGHT_MIN_M) / 2.0

    def _pick_contact(self):
        """The tool is on the box (the wrist load changed): stop there and grip."""
        self.pick_contact = True
        self.get_logger().info(f"pick: contact at TCP z {self.measured_tcp[2]:.3f} m (sensed top "
                               f"{self.leg_target[2] + PICK_BELOW_M:.3f} m), wrist load "
                               f"{self.wrist_fz - self.touch_baseline:+.1f} N")
        self._set_ref_state(self.measured_tcp, np.zeros(3), np.zeros(3), cyl=self.ref_cyl)
        self._set_target(self.measured_tcp, np.zeros(3))
        self.leg_timeout_ticks = None
        self._arrive()

    def _rescan_pile(self, why):
        """A pick missed or the grasp failed: back above the pile and scan it again."""
        self.pick_failures += 1
        if self.pick_failures > PICK_RETRIES:
            self.get_logger().error(f"pick failed ({why}) {self.pick_failures} times in a row; stopping")
            self.state = self.PARKED
            self._set_ref_state(self.measured_tcp, np.zeros(3), np.zeros(3), cyl=self.ref_cyl)
            self._set_target(self.measured_tcp, np.zeros(3))
            return
        self.get_logger().warn(f"pick failed ({why}); scanning the pile again")
        if self.holding_box:
            self.action_pub.publish(String(data="place_held"))
            self.holding_box = False
            self._publish_held_box()
        self.next_place_legs = None
        pos = self._pile_scan_pos()
        if not self._reach_ok([Waypoint(pos)], "pile scan"):
            return
        self.state = self.RETURNING
        self._advance_reference(Waypoint(pos, hull_active=False))

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
        self.leg_wps[-1].pos[2] = max(self._tray_top(), surface + self.box_height) + APPROACH_MARGIN_M
        self.leg_wps[-1].heading_offset = 0.0 if pushes else self.leg_heading_offset
        slides = [] if pushes else self._slide_legs(x + shift[0], y + shift[1], hx, hy, surface, psi, shift)
        if slides:
            self.leg_wps[self.leg_idx].action = None  # released after the slide
            self.slide_from = self.measured_tcp.copy()
        self.leg_wps = self.leg_wps[:self.leg_idx + 1] + slides + pushes + [self.leg_wps[-1]]
        self.pending_place = (final[0], final[1], float(hx), float(hy), surface + self.box_height, surface)
        self.leg_timeout_ticks = None
        self._arrive()

    def _slide_legs(self, cx, cy, hx, hy, surface, psi, shift):
        """Legs that slide the held box, set down at (cx, cy), into what it was planned
        flush with (a wall or a placed box; one side per axis, none along an axis the wrist
        set it down off), lifted clear of the floor, then lower it (the release). [] on a
        stack or with nothing flush. Not called for a box to be pushed: pressed against a
        wall, the push would drag it along it."""
        if surface > self.tray.floor_z + 0.005:
            return []
        tcp = self.measured_tcp.copy()
        at = tcp + np.array([0.0, 0.0, SLIDE_LIFT_M])
        lim = self._wrist_limits(tcp[2], psi)
        walls = np.array(self.tray.bounds).ravel()
        floor = [b for b in self.placed_boxes if b[5] < self.tray.floor_z + 0.005]
        c, h = (cx, cy), (hx, hy)
        legs = [Waypoint(at.copy(), hull_active=False, heading_offset=self.leg_heading_offset)]
        for k in (0, 1):
            if abs(shift[k]) > 1e-4:
                continue
            o, best = 1 - k, None
            for sign in (-1.0, 1.0):
                gap = sign * (walls[2 * k + int(sign > 0)] - c[k]) - h[k]
                for b in floor:
                    if abs(b[o] - c[o]) < b[2 + o] + h[o] and sign * (b[k] - c[k]) > 0.0:
                        gap = min(gap, sign * (b[k] - c[k]) - b[2 + k] - h[k])
                if gap <= PLACE_CLEARANCE_M + SLIDE_SIDE_TOL_M and (best is None or gap < best[1]):
                    best = (sign, gap)
            if best is None:
                continue
            target = at.copy()
            target[k] += best[0] * (max(best[1], 0.0) + SLIDE_EXTRA_M)
            if lim is not None:
                target[k] = float(np.clip(target[k], *lim[k]))
            if abs(target[k] - at[k]) < SLIDE_MIN_M:
                continue
            legs.append(Waypoint(target, action="slide", hull_active=False, max_speed=SLIDE_SPEED_MPS,
                                 timeout_s=SLIDE_LEG_TIMEOUT_S, heading_offset=self.leg_heading_offset))
            at = target
        if len(legs) == 1:
            return []
        legs.append(Waypoint(np.array([at[0], at[1], tcp[2]]), action="place", hull_active=False,
                             max_speed=TOUCH_SPEED_MPS, timeout_s=PLACE_LEG_TIMEOUT_S,
                             heading_offset=self.leg_heading_offset))
        return legs

    def _check_slide_contact(self):
        """End a slide when the wrist feels the held box touch: back off a little, and move
        the legs after it by where it really stopped."""
        d = self.leg_in_dir
        force = float((self.wrist_f_avg[:2] - self.slide_tare) @ d[:2])
        self.slide_ticks = self.slide_ticks + 1 if force >= SLIDE_FORCE_N else 0
        if self.slide_ticks < SLIDE_CONFIRM_TICKS:
            return
        end = self.measured_tcp - d * SLIDE_BACKOFF_M
        delta = end[:2] - self.leg_target[:2]
        for wp in self.leg_wps[self.leg_idx + 1:]:
            wp.pos = np.array([wp.pos[0] + delta[0], wp.pos[1] + delta[1], wp.pos[2]])
        self.get_logger().info(f"slide: touched ({force:.1f} N) after "
                               f"{1e3 * float((self.measured_tcp - self.leg_start) @ d):.1f} mm")
        self.leg_timeout_ticks = None
        self._arrive()

    def _process_dest_scan(self):
        """Empty hand: re-measure the placed boxes, keep the heightmap for the next
        placement and go back to the pile. Box held: plan the placement now."""
        if self.state != self.AWAITING_DEST_SCAN:
            return
        if self.tray is None and self.tray_find_frames is not None:
            self.tray_find_frames.append(self.dest_depth)
            if len(self.tray_find_frames) < len(_halves(PLACE_ZONE_BOUNDS)):
                self._go_scan_tray()
                return
            found = self._find_tray(self.tray_find_frames)
            self.tray_find_frames = None
            if not found:
                self.get_logger().error("no tray found in the place zone; stopping")
                self.state = self.PARKED
                return
            self._go_scan_tray()
            return
        if self.reanchor_pending and self.tray is not None:
            self.reanchor_pending = False
            if not self._reanchor_tray():
                return
        hmap = self._tray_heightmap(held=self.holding_box)
        if hmap is None:
            self.get_logger().error("no tray found in the place zone; stopping")
            self.state = self.PARKED
            return
        if not self.holding_box:
            self._remeasure_placed()
            self.latest_dest_hmap = hmap
            if self.mobile:
                if self.max_boxes and self.boxes_moved >= self.max_boxes:
                    self._drive_to("home", self._at_home)
                else:
                    self._drive_to("pick", self._at_pick)
                return
            pos = self._pile_scan_pos()
            if not self._reach_ok([Waypoint(pos)], "pile scan"):
                return
            self.state = self.RETURNING
            self._advance_reference(Waypoint(pos, hull_active=False, dest_hull_active=True))
            return
        self.latest_dest_hmap = hmap
        legs = self._plan_place(self.latest_dest_hmap)
        if legs is None:
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
        self.leg_wps = legs
        self.leg_idx = 0
        self.state = self.PAUSING_DEST
        self.pause_until_tick = self.tick_count + (SCAN_PAUSE_TICKS_MOBILE if self.mobile else SCAN_PAUSE_TICKS)

    def _tray_heightmap(self, held=False):
        """Tray heightmap from the scan's depth frame (None if no tray is found). With the box held, cells it
        hides keep the last scan's heights, except round a blocked set-down, which
        are unknown (NaN)."""
        cam_pos, cam_mat, fovy, w, h, depth = self.dest_depth
        ref_z = (FLOOR_Z + self._tray_top()) / 2.0
        if self.tray is None and not self._find_tray([self.dest_depth]):
            return None
        hidden = self._held_silhouette(cam_pos, cam_mat, fovy, w, h) if held else None
        points, shape = self.tray_grid
        heights, _ = heightmap.infer_heights_parallax_corrected(
            depth, cam_pos, cam_mat, fovy, w, h, points, ref_z, self.tray.floor_z,
            hidden=hidden, top_z=self.tray.wall_top)
        hmap = heights.reshape(shape)
        if not held:
            return hmap
        unseen = np.isnan(hmap)
        blocked = np.zeros_like(unseen)
        if self.blocked_rect is not None:
            x0, x1, y0, y1 = self.blocked_rect
            pts = np.asarray(points).reshape(*shape, 2)
            blocked = (pts[..., 0] > x0) & (pts[..., 0] < x1) & (pts[..., 1] > y0) & (pts[..., 1] < y1)
        prev = self.latest_dest_hmap if self.latest_dest_hmap is not None else np.full_like(hmap, np.nan)
        self.get_logger().info(f"tray scan with the box held: {100 * unseen.mean():.0f}% of the tray hidden, "
                               f"{int((unseen & blocked).sum())} hidden cells round the blocked spot unknown")
        return np.where(unseen & ~blocked, prev, hmap)

    def _find_tray(self, frames):
        """The tray's walls, floor and wall top from the first tray scan's frames
        (place zone); frames as _parse_depth gives them."""
        frames = [(depth, cam_pos, cam_mat, fovy, w, h) for cam_pos, cam_mat, fovy, w, h, depth in frames]
        tray = tray_detection.detect_tray(frames, *PLACE_ZONE_BOUNDS, SCAN_RESOLUTION, FLOOR_Z)
        if tray is None:
            return False
        self.tray = tray_detection.refine_walls(tray, frames)
        self.tray_grid = heightmap.build_dense_grid_xy(*self.tray.bounds, SCAN_RESOLUTION)
        t = self.tray
        self.get_logger().info(f"tray found: inner x {t.x_min:.4f}..{t.x_max:.4f}, y {t.y_min:.4f}..{t.y_max:.4f}, "
                               f"floor {t.floor_z:.4f}, wall top {t.wall_top:.4f} m")
        return True

    def _held_silhouette(self, cam_pos, cam_mat, fovy, w, h):
        """Pixels the held box covers: sensed footprint and in-hand offset, measured
        height (else the spec's maximum), pose from the arm's kinematics."""
        rot = np.array(self.pin_data.oMf[self.tcp_frame_id].rotation)
        tcp = np.array(self.pin_data.oMf[self.tcp_frame_id].translation)
        hx, hy = self.current_box_size[:2] + HELD_MASK_GROW_M
        height = self.box_height if self.box_height is not None else BOX_HEIGHT_MAX_M
        corners = []
        for sx in (-1, 1):
            for sy in (-1, 1):
                top = tcp + rot @ (self.current_box_offset + self.pick_rot.T @ np.array([sx * hx, sy * hy, 0.0]))
                corners += [top, top + (height + HELD_MASK_GROW_M) * rot[:, 2]]
        return heightmap.box_silhouette(corners, cam_pos, cam_mat, fovy, w, h)

    def _plan_place(self, hmap):
        """Placement legs for the held box on the tray heightmap, or None if no spot."""
        hx, hy, hz = self.current_box_size
        floor_z = self.tray.floor_z
        floor_boxes = [b[:4] for b in self.placed_boxes if b[5] < floor_z + 0.005]
        # Floor spots come from the packing rule, checked against the scan. The
        # heightmap search is the fallback, and also what stacks.
        rejected = {"scan": 0, "overhang": 0, "robot": 0}
        packed = plan_compact(
            float(hx), float(hy), floor_boxes,
            self.tray.bounds,
            PLACE_CLEARANCE_M,
            where=lambda cx, cy, fhx, fhy: self._floor_spot_end(cx, cy, fhx, fhy, floor_z, hmap, rejected),
            upcoming=self.upcoming_sizes[:3], group_first=True, strip=True,
            needs_push=lambda cx, cy, fhx, fhy, turned: self._needs_push(cx, cy, turned, floor_z),
            push_area=PACK_PUSH_AREA_M2)
        why = ", ".join(f"{k} {v}" for k, v in rejected.items() if v)
        if why:
            self.get_logger().info(f"packing: spots rejected ({why})")
        if packed is not None:
            cx, cy, fhx, fhy = packed
            turned = abs(fhx - hx) > 1e-6
            return self._placement_legs(cx, cy, floor_z,
                                        f"compact{', turned 90 deg' if turned else ''} "
                                        f"(looked ahead at {len(self.upcoming_sizes[:3])})", fhx, fhy)
        # The lowest spot where the box rests stably, either way round (on the floor, on a
        # box, bridging boxes of the same height) and the robot can place it: set down and
        # push back without the wrist or the arm hitting anything (_place_options).
        spots = []
        for fhx, fhy in ((hx, hy), (hy, hx)):
            cells = (round(2 * fhy / SCAN_RESOLUTION) + 1, round(2 * fhx / SCAN_RESOLUTION) + 1)
            spots += [(z, -sup, r, c, cells, fhx, fhy)
                      for r, c, z, sup in heightmap.find_resting_footprints(hmap, cells, FLATNESS_TOL)]
        base = min((v[0] for v in spots), default=0.0)
        spots.sort(key=lambda v: (np.floor((v[0] - base) / FLATNESS_TOL), v[1]))
        tried, best = [], None
        for z, neg_sup, row, col, cells, fhx, fhy in spots:
            x, y = heightmap.footprint_center_xy(row, col, cells, *self.tray.bounds, SCAN_RESOLUTION)
            if any(np.hypot(x - tx, y - ty) < FALLBACK_SPOT_SPACING_M and tf == (fhx, fhy) for tx, ty, tf in tried):
                continue
            tried.append((x, y, (fhx, fhy)))
            if z < floor_z + 0.005:
                x, y = self._snap_flush(x, y, fhx, fhy, z, hmap)  # on boxes: stay where it was found stable
            if self._clear_of_placed(x, y, fhx, fhy, z) and not self._place_options(x, y, fhx, fhy, z)[0][0]:
                best = (x, y, z, -neg_sup, fhx, fhy)
                break
            if len(tried) >= FALLBACK_MAX_SPOTS:
                break
        if best is None:
            return None
        x, y, surface_height, support, fhx, fhy = best
        turned = ", turned 90 deg" if abs(fhx - hx) > 1e-6 else ""
        on = ("free floor spot" if surface_height < floor_z + 0.005
              else f"on top of boxes ({100 * support:.0f}% supported)")
        return self._placement_legs(x, y, surface_height, f"no floor spot in the packing: {on}{turned}", fhx, fhy)

    def _pick_wrist_clear(self, hmap, x, y, top):
        """Whether link7 clears the pile scan's cells with the TCP on a top at (x, y, top)."""
        lo, hi = self._wrist_rect(self._aligned_yaw(float(np.arctan2(y, x))))
        pts = np.asarray(self.source_grid).reshape(*self.source_shape, 2)
        m = PICK_WRIST_MARGIN_M
        under = ((pts[..., 0] > x + lo[0] - m) & (pts[..., 0] < x + hi[0] + m)
                 & (pts[..., 1] > y + lo[1] - m) & (pts[..., 1] < y + hi[1] + m))
        return not (hmap[under] > top + WRIST_ABOVE_TCP_M - WRIST_MARGIN_M).any()

    def _tray_top(self):
        """Highest point in the tray: wall tops and the last scan (the place zone's
        highest before the tray is found)."""
        if self.tray is None:
            return PLACE_ZONE_MAX_HEIGHT_M
        top = self.tray.wall_top
        hmap = self.latest_dest_hmap
        if hmap is not None and np.isfinite(hmap).any():
            top = max(top, float(np.nanmax(hmap)))
        return top

    def _tray_enter_z(self):
        """TCP height that carries the held box (measured height, else the spec's
        tallest) clear over the tray."""
        h = self.box_height if self.box_height is not None else BOX_HEIGHT_MAX_M
        return self._tray_top() + h + LIFT_MARGIN_M

    def _pile_scan_pos(self):
        return _scan_tcp(self.source_bounds, self.pile_top)

    def _tray_scan_pos(self, held=False, heading_offset=0.0):
        """Tray scan pose: sees the tray, clear of the pile's top on the way back,
        and with a box held, clear of the tray's contents under it."""
        bounds = PLACE_ZONE_BOUNDS if self.tray is None else self.tray.bounds
        floor = FLOOR_Z if self.tray is None else self.tray.floor_z
        pos = _scan_tcp(bounds, self._tray_top(), floor, heading_offset)
        pos[2] = max(pos[2], self.pile_top + APPROACH_MARGIN_M)
        if held:
            pos[2] = max(pos[2], self._tray_enter_z(), self.pile_clear_z or 0.0)
        return pos

    def _go_scan_tray(self, heading_offset=0.0):
        """To the tray scan with the aligned heading (turned, the wrist camera's view
        takes in the robot's base); turning back from heading_offset, the leg is slowed
        so joint 7 keeps under HEADING_TURN_RATE_MAX."""
        if self.tray is None:
            # Finding the tray: the whole place zone does not fit one picture in reach.
            if self.tray_find_frames is None:
                self.tray_find_frames = []
            part = _halves(PLACE_ZONE_BOUNDS)[len(self.tray_find_frames)]
            pos = _scan_tcp(part, PLACE_ZONE_MAX_HEIGHT_M)
            pos[2] = max(pos[2], self.pile_top + APPROACH_MARGIN_M)
        else:
            pos = self._tray_scan_pos()
        if not self._reach_ok([Waypoint(pos)], "tray scan"):
            return
        self.state = self.TRAVELING_TO_DEST
        speed = None
        if abs(heading_offset) > 1e-3:
            # Smoothstep over the distance: peak turn rate 1.5 x turn / leg time.
            length = float(np.linalg.norm(pos - self._ref_cart()[0]))
            speed = length * HEADING_TURN_RATE_MAX / (1.5 * abs(heading_offset))
        self._go(self.TRAVELING_TO_DEST, Waypoint(pos, max_speed=speed))

    def _zone_exit(self, start, goal):
        """Where a swing from start (above the pile) to goal leaves the pick zone's
        azimuth range with the held box clear of it, at start's radius and height; None
        if the carry needs no extra height over the pile, or the swing stays in range."""
        if self.pile_clear_z is None or start[2] <= goal[2] + 0.01:
            return None
        (x0, x1), (y0, y1) = self.source_bounds
        azs = [np.arctan2(y, x) for x in (x0, x1) for y in (y0, y1)]
        r = float(np.hypot(start[0], start[1]))
        az0 = float(np.arctan2(start[1], start[0]))
        sweep = (float(np.arctan2(goal[1], goal[0])) - az0 + np.pi) % (2 * np.pi) - np.pi
        m = (ZONE_EXIT_MARGIN_M + BOX_HALF_DIAGONAL_MAX_M) / max(r, MIN_PLAN_RADIUS_M)
        edge = min(azs) - m if sweep < 0 else max(azs) + m
        d = (edge - az0 + np.pi) % (2 * np.pi) - np.pi
        if np.sign(d) != np.sign(sweep) or abs(d) >= abs(sweep):
            return None
        return np.array([r * np.cos(edge), r * np.sin(edge), start[2]])

    def _ik_reachable(self, pos, psi):
        """Whether the TCP reaches pos with the tool down at heading psi, within the
        joint limits."""
        return self._ik_solve(pos, psi) is not None

    def _ik_solve(self, pos, psi, tool_axis=ORIENT_AXIS_TARGET):
        """Joint angles with the TCP at pos, the tool along tool_axis and its +x towards
        heading psi, within the joint limits (damped least squares from the current and
        a home-like seed), or None."""
        z_t = np.asarray(tool_axis, dtype=float) / np.linalg.norm(tool_axis)
        x_t = np.array([np.cos(psi), np.sin(psi), 0.0])
        x_t -= (x_t @ z_t) * z_t
        x_t /= np.linalg.norm(x_t)
        target = pin.SE3(np.column_stack([x_t, np.cross(z_t, x_t), z_t]), np.asarray(pos, dtype=float))
        lo = self.pin_model.lowerPositionLimit + IK_JOINT_MARGIN_RAD
        hi = self.pin_model.upperPositionLimit - IK_JOINT_MARGIN_RAD
        az = float(np.arctan2(pos[1], pos[0]))
        seed = HOME_Q.copy()
        seed[0] = az
        seed[6] = np.clip((az - psi - np.radians(135) + np.pi) % (2 * np.pi) - np.pi, lo[6], hi[6])  # see _aligned_yaw
        for q in ([self.measured_q] if self.measured_q is not None else []) + [seed]:
            q = np.clip(q, lo, hi)
            for _ in range(IK_ITERS):
                pin.framesForwardKinematics(self.pin_model, self.ik_data, q)
                i_m_d = self.ik_data.oMf[self.tcp_frame_id].actInv(target)
                err = pin.log(i_m_d).vector
                if np.linalg.norm(err[:3]) < 1e-3 and np.linalg.norm(err[3:]) < 1e-2:
                    return q
                jac = -pin.Jlog6(i_m_d.inverse()) @ pin.computeFrameJacobian(
                    self.pin_model, self.ik_data, q, self.tcp_frame_id)
                q = np.clip(pin.integrate(self.pin_model, q, -jac.T @ np.linalg.solve(
                    jac @ jac.T + 1e-4 * np.eye(6), err)), lo, hi)
        return None

    def _arm_wall_clearance(self, tip, tool_axis, psi):
        """Smallest clearance (m) from the arm (ARM_CHECK_FRAMES and the points between
        them, ARM_CHECK_RADIUS_M) to the tray's walls with the TCP at tip and the tool
        along tool_axis; -inf if out of reach."""
        q = self._ik_solve(tip, psi, tool_axis)
        if q is None:
            return -np.inf
        pin.framesForwardKinematics(self.pin_model, self.ik_data, q)
        pts = [np.array(self.ik_data.oMf[self.pin_model.getFrameId(f)].translation) for f in ARM_CHECK_FRAMES]
        pts += [(a + b) / 2.0 for a, b in zip(pts, pts[1:])]
        t = self.tray
        w = DEST_WALL_THICKNESS
        walls = [((t.x_min - w, t.x_min), (t.y_min - w, t.y_max + w)), ((t.x_max, t.x_max + w), (t.y_min - w, t.y_max + w)),
                 ((t.x_min - w, t.x_max + w), (t.y_min - w, t.y_min)), ((t.x_min - w, t.x_max + w), (t.y_max, t.y_max + w))]
        best = np.inf
        for (x0, x1), (y0, y1) in walls:
            lo, hi = np.array([x0, y0, t.floor_z]), np.array([x1, y1, t.wall_top])
            for p in pts:
                best = min(best, float(np.linalg.norm(np.maximum(np.maximum(lo - p, 0.0), p - hi))) - ARM_CHECK_RADIUS_M)
        return best

    def _reach_ok(self, wps, what):
        """Check every tool-down waypoint with IK; if one is out of reach, stop."""
        for wp in wps:
            if not np.allclose(wp.tool_axis, ORIENT_AXIS_TARGET):
                continue
            psi = self._aligned_yaw(float(np.arctan2(wp.pos[1], wp.pos[0]))) + wp.heading_offset
            if not self._ik_reachable(wp.pos, psi):
                self.get_logger().error(f"{what}: pose {np.round(wp.pos, 3).tolist()} is out of reach; "
                                        f"stopping (stack too tall?)")
                self.state = self.PARKED
                self._set_ref_state(self.measured_tcp, np.zeros(3), np.zeros(3), cyl=self.ref_cyl)
                self._set_target(self.measured_tcp, np.zeros(3))
                return False
        return True

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
        wrist = {o: self._j7_after(az, self._aligned_yaw(az) + o) for o in offsets}
        ok = ([o for o in offsets if abs(wrist[o][0]) <= J7_PLACE_LIMIT_RAD]
              or [min(offsets, key=lambda o: abs(wrist[o][0]))])
        options = []
        for offset in ok:
            psi = self._aligned_yaw(az) + offset
            shift, legs, notes, final, complete = self._clearance_and_pushes(x, y, hx, hy, hz, surface, psi)
            options.append((not complete, wrist[offset][1] > np.pi / 2, float(np.hypot(*shift)),
                            offset, psi, shift, legs, notes, final))
        return [o[:1] + o[2:] for o in sorted(options, key=lambda o: o[:3])]

    def _j7_after(self, az, psi):
        """Joint 7 at base azimuth az and heading psi, reached from now the way
        _start_heading_leg turns (not wrapped: the wrist turns continuously), and
        how far the wrist turns past turning with the base."""
        if self.measured_q is None:
            return (az - psi - np.radians(135) + np.pi) % (2 * np.pi) - np.pi, 0.0  # see _aligned_yaw
        az0 = float(np.arctan2(self.measured_tcp[1], self.measured_tcp[0]))
        daz = (az - az0 + np.pi) % (2 * np.pi) - np.pi
        psi_u = self._heading_goal(psi, daz, self.measured_q[6])
        return float(self.measured_q[6] + daz - (psi_u - self.psi_cmd)), abs(psi_u - self.psi_cmd - daz)

    def _heading_goal(self, psi, daz, q7):
        """psi + 2 pi k for a swing of daz from now: nearest to turning with the base
        (psi_cmd + daz) among those that keep joint 7 (now q7) within
        J7_PLACE_LIMIT_RAD; joint7 ~= joint1 - heading - 135 deg (_aligned_yaw)."""
        want = self.psi_cmd + daz
        k0 = round((want - psi) / (2 * np.pi))
        cands = [psi + 2 * np.pi * (k0 + d) for d in (0, -1, 1)]
        if q7 is None:
            return cands[0]
        j7 = {p: abs(q7 + daz - (p - self.psi_cmd)) for p in cands}
        ok = [p for p in cands if j7[p] <= J7_PLACE_LIMIT_RAD]
        return min(ok, key=lambda p: abs(p - want)) if ok else min(cands, key=j7.get)

    def _placement_legs(self, x, y, surface_height, how, hx, hy):
        """The release legs for a box centred at (x, y) on a surface at
        surface_height, footprint half extents hx/hy along x/y (swapped from the
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
        self.slide_from = None
        stack_height = surface_height - self.tray.floor_z
        # Near a wall the pass-through's rounded corner starts down while the box (still
        # turning) is over the wall: a big box once caught on the wall's outside at 97 N.
        t = self.tray
        reach = float(np.hypot(hx, hy)) + RELEASE_WALL_MARGIN_M
        stop_above = min(x - t.x_min, t.x_max - x, y - t.y_min, t.y_max - y) < reach
        # Beside a placed box taller than the held one's bottom at the hover, a fast descent
        # lagging sideways catches its top edge: settle above, then straight down.
        bottom = surface_height + BOX_HEIGHT_MAX_M + TOUCH_HOVER_M - self._plan_hz() * 2.0
        stop_above = stop_above or any(
            top > bottom and abs(px - x) < phx + hx + RELEASE_BOX_MARGIN_M and abs(py - y) < phy + hy + RELEASE_BOX_MARGIN_M
            for px, py, phx, phy, top, _s in self.placed_boxes)
        legs = _build_dest_legs(hover, touch, (x, y, stack_height), self._tray_enter_z(),
                                max(self._tray_top(), surface_height + BOX_HEIGHT_MAX_M) + APPROACH_MARGIN_M,
                                push_legs, offset, stop_above)
        self._publish_decision(
            "place", [x - hx, x + hx, y - hy, y + hy, surface_height],
            how,
            others=[[b[0] - b[2], b[0] + b[2], b[1] - b[3], b[1] + b[3], b[4]] for b in self.placed_boxes])

        self.get_logger().info(
            f"placement: selected ({x:.3f}, {y:.3f}) "
            f"(sensed surface {surface_height:.3f}m; {how})"
        )
        off = self.current_box_offset if self.current_box_offset is not None else np.zeros(3)
        self.get_logger().info(
            f"placement aim: box {1e3 * xy[0]:.1f}/{1e3 * xy[1]:.1f} half {1e3 * hx:.1f}/{1e3 * hy:.1f}, "
            f"tcp {1e3 * touch[0]:.1f}/{1e3 * touch[1]:.1f} psi {np.degrees(psi):.2f}, "
            f"in hand {1e3 * off[0]:.1f}/{1e3 * off[1]:.1f} (mm, deg); boxes "
            + " ".join(f"{1e3 * b[0]:.1f}/{1e3 * b[1]:.1f}/{1e3 * b[2]:.1f}/{1e3 * b[3]:.1f}"
                       for b in self.placed_boxes if b[5] < self.tray.floor_z + 0.005))
        return legs

    def _publish_held_box(self):
        if not self.holding_box:
            self.held_box_pub.publish(Float64MultiArray(data=[0.0] * 16))
            return
        hx, hy = self.current_box_size[:2] + HELD_MASK_GROW_M
        height = (self.box_height if self.box_height is not None else BOX_HEIGHT_MAX_M) + HELD_MASK_GROW_M
        self.held_box_pub.publish(Float64MultiArray(data=[
            1.0, *map(float, self.current_box_offset), *map(float, self.pick_rot.T.ravel()),
            float(hx), float(hy), float(height)]))

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
        if tcp_z + WRIST_ABOVE_TCP_M >= self.tray.wall_top + WRIST_MARGIN_M:
            return None
        lo, hi = self._wrist_rect(psi)
        walls = self.tray.bounds
        return [(w0 + WRIST_MARGIN_M - lo[k], w1 - WRIST_MARGIN_M - hi[k]) for k, (w0, w1) in enumerate(walls)]

    def _push_clearance(self, tip, tool_axis, psi):
        """Smallest clearance (m) at a push pose, tip at `tip` and the tool along
        tool_axis: the tool and link7 against the tray walls and against what the
        scan shows, the flange against the walls on the robot's side (the forearm
        hangs towards the base). Below zero is a hit.
        """
        hmap = self.latest_dest_hmap
        (xmin, _), (ymin, _) = self.tray.bounds
        walls = self.tray.bounds
        wlo, whi = self._wrist_rect(psi)
        best = np.inf
        for t in np.arange(0.0, WRIST_ABOVE_TCP_M + WRIST_HEIGHT_M + 1e-9, 0.01):
            p = tip - t * tool_axis
            if t < WRIST_ABOVE_TCP_M:
                r = TOOL_RADIUS_M + PUSH_TOOL_MARGIN_M
                lo, hi = np.full(2, -r), np.full(2, r)
            else:
                lo, hi = wlo, whi
            if p[2] < self.tray.wall_top + WRIST_MARGIN_M:
                for k, (w0, w1) in enumerate(walls):
                    best = min(best, p[k] + lo[k] - w0, w1 - (p[k] + hi[k]))
            c0 = max(int(np.floor((p[0] + lo[0] - xmin) / SCAN_RESOLUTION)), 0)
            c1 = int(np.ceil((p[0] + hi[0] - xmin) / SCAN_RESOLUTION))
            r0 = max(int(np.floor((p[1] + lo[1] - ymin) / SCAN_RESOLUTION)), 0)
            r1 = int(np.ceil((p[1] + hi[1] - ymin) / SCAN_RESOLUTION))
            patch = hmap[r0:r1 + 1, c0:c1 + 1]
            if patch.size:
                best = min(best, -np.inf if np.isnan(patch).any() else p[2] - float(np.max(patch)) - WRIST_MARGIN_M)
        flange = tip - WRIST_ABOVE_TCP_M * tool_axis
        if flange[2] + FOREARM_ABOVE_FLANGE_M < self.tray.wall_top + WRIST_MARGIN_M:
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
            chosen, best_cl = None, -np.inf
            # Straight down first, then the smallest tilt (either way) that fits.
            for tilt, lean in [(0.0, 0.0)] + [(t, ln) for t in PUSH_TILTS_RAD for ln in (1.0, -1.0)]:
                tool_axis = np.array([0.0, 0.0, -np.cos(tilt)])
                tool_axis[k] = -lean * np.sin(tilt)  # tool z points flange -> tip
                centring = _push_centring(tool_axis, axis, 2 * hz)
                # The descent tilts the tool on the way down: check it along the way too.
                high = start.copy()
                high[2] = max(surface + 2 * hz + PUSH_LIFT_M, self.tray.wall_top - PUSH_SAFE_BELOW_WALL_M)
                poses = [(end + centring, tool_axis)]
                for f in (0.25, 0.5, 0.75, 1.0):
                    s_f = f * f * (3.0 - 2.0 * f)
                    a = (1.0 - s_f) * ORIENT_AXIS_TARGET + s_f * tool_axis
                    poses.append((high + f * (start - high) + centring, a / np.linalg.norm(a)))
                cl = min(self._push_clearance(p, a, psi_push) for p, a in poses)
                if cl >= 0.0:  # the tool and the wrist fit: now the whole arm
                    cl = min(cl, min(self._arm_wall_clearance(p, a, psi_push) for p, a in poses) - ARM_WALL_MARGIN_M)
                best_cl = max(best_cl, cl)
                if cl >= 0.0:
                    chosen = (cl, tilt, lean, tool_axis)
                    break
            if chosen is None:
                notes.append(f"push along {'xy'[axis]}: nothing fits (best {1e3 * best_cl:+.0f} mm); "
                             f"box left {1e3 * abs(push):.0f} mm from flush")
                complete = complete and abs(push) < PUSH_MIN_M
                continue
            cl, tilt, lean, tool_axis = chosen
            notes.append(f"push along {'xy'[axis]} {1e3 * push:+.0f} mm, "
                         f"{'no tilt' if tilt == 0 else f'tilted {np.degrees(tilt):.0f} deg towards ' + ('+' if lean > 0 else '-') + 'xy'[k]}, "
                         f"clearance {1e3 * cl:.0f} mm")
            new = _push_legs(tcp_xy, centre, half, axis, push, surface, 2 * hz, tool_axis,
                             self.tray.wall_top - PUSH_SAFE_BELOW_WALL_M)
            if legs and np.allclose(legs[-1].pos, new[0].pos):
                new = new[1:]  # a zero-length leg never gets a settle pulse
            legs += new
            tcp_xy = new[-1].pos[:2]
            centre[axis] += push
        return tuple(shift), legs, notes, (float(centre[0]), float(centre[1])), complete

    def _footprint_clear(self, x, y, hx, hy, surface, hmap):
        """Whether the scan shows a footprint at (x, y) clear down to `surface`,
        ignoring one sample round its edge (it may read a neighbour or a wall).
        """
        (xmin, _), (ymin, _) = self.tray.bounds
        c0 = int(np.ceil((x - hx - xmin) / SCAN_RESOLUTION)) + 1
        c1 = int(np.floor((x + hx - xmin) / SCAN_RESOLUTION)) - 1
        r0 = int(np.ceil((y - hy - ymin) / SCAN_RESOLUTION)) + 1
        r1 = int(np.floor((y + hy - ymin) / SCAN_RESOLUTION)) - 1
        patch = hmap[max(r0, 0):r1 + 1, max(c0, 0):c1 + 1]
        return patch.size > 0 and float(np.max(patch)) <= surface + FLATNESS_TOL

    def _needs_push(self, x, y, turned, surface):
        """Whether the wrist cannot lower a box centred at (x, y) there (turned 90 deg or
        not), at any heading that turns it so: it would be set down off and pushed."""
        z = surface + 2 * self._plan_hz()
        az = float(np.arctan2(y, x))
        for offset in ((np.pi / 2, -np.pi / 2) if turned else (0.0,)):
            lim = self._wrist_limits(z, self._aligned_yaw(az) + offset)
            if lim is None or all(abs(float(np.clip(v, *lim[k])) - v) < PUSH_MIN_M for k, v in enumerate((x, y))):
                return False
        return True

    def _floor_spot_end(self, x, y, hx, hy, floor_z, hmap, rejected):
        """Where the box ends up if placed at the floor spot (x, y): there, or where
        the robot leaves it when a push back is not possible; None if it cannot go
        there. Counts the reason in `rejected`."""
        ok = lambda px, py: (self._footprint_clear(px, py, hx, hy, floor_z, hmap)
                             and self._clear_of_placed(px, py, hx, hy, floor_z))
        if not self._footprint_clear(x, y, hx, hy, floor_z, hmap):
            rejected["scan"] += 1
            return None
        if not self._clear_of_placed(x, y, hx, hy, floor_z):
            rejected["overhang"] += 1
            return None
        incomplete, *_rest, final = self._place_options(x, y, hx, hy, floor_z)[0]
        if not incomplete:
            return (x, y)
        if np.hypot(final[0] - x, final[1] - y) > 1e-4 and ok(*final):
            return final
        rejected["robot"] += 1
        return None

    def _clear_of_placed(self, x, y, hx, hy, surface):
        """Whether a footprint at (x, y) on `surface` keeps PLACE_CLEARANCE_M from every
        stacked box above that surface and beside the held box (an overhang the box
        could catch under)."""
        h = self.box_height if self.box_height is not None else BOX_HEIGHT_MAX_M
        gap = PLACE_CLEARANCE_M - 1e-4  # the packing and _snap_flush space boxes at exactly PLACE_CLEARANCE_M
        return not any(
            abs(x - px) < hx + phx + gap and abs(y - py) < hy + phy + gap
            for px, py, phx, phy, top, bottom in self.placed_boxes
            if surface + FLATNESS_TOL < bottom < surface + h)

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
        (x_lo, x_hi), (y_lo, y_hi) = self.tray.bounds
        x = snap_axis(x, hx, x_lo, x_hi, y, hy,
                      [(b[0], b[2], b[1], b[3]) for b in standing])
        y = snap_axis(y, hy, y_lo, y_hi, x, hx,
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
            scale = min(scale, self.leg_speed / float(np.max(self.leg_vel)))
        inp.max_velocity = (self.leg_vel * scale / self.lim_div).tolist()
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
        if self.state == self.TUCKING:
            if not self.tuck_down:
                self.tuck_down = True
                self._advance_reference(Waypoint(CARRY_POS.copy(), hull_active=True, dest_hull_active=True,
                                                 transit=True))
                return
            self.state = self.DRIVING
            self.station = None
            self.get_logger().info(f"arm in the carry pose; driving to {self.drive[0]}")
            return
        if self.state == self.UNTUCKING:
            (state, wp), self.untuck_next = self.untuck_next, None
            self.state = state
            self._advance_reference(wp)
            return
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
            if not self.pick_contact:
                self._rescan_pile(f"reached {self.measured_tcp[2]:.3f} m, {1e3 * (PICK_BELOW_M):.0f} mm below the "
                                  f"sensed top, without touching a box")
                return
            self.grasp_tare = self.touch_baseline
            tcp = self.measured_tcp
            self.action_pub.publish(String(data=f"pick_at {tcp[0]} {tcp[1]} {tcp[2]}"))
            self.holding_box = True
            x, y, hx, hy = self.pick_sensed
            self.current_box_size = np.array([hx, hy, BOX_HEIGHT_MAX_M / 2.0])
            self.box_height = None
            self.place_attempts = 0
            # In-hand offset: sensed top centre from the TCP, in the TCP frame (x/y only;
            # the TCP holds the top, the height comes at touch-down).
            rot = np.array(self.pin_data.oMf[self.tcp_frame_id].rotation)
            self.current_box_offset = rot.T @ np.array([x - self.measured_tcp[0], y - self.measured_tcp[1], 0.0])
            self.pick_rot = rot
            self._publish_held_box()
            self.blocked_rect = None
            self.get_logger().info(
                f"grasped: sensed footprint {2e3 * hx:.0f} x {2e3 * hy:.0f} mm, top centre "
                f"{1e3 * np.hypot(x - self.measured_tcp[0], y - self.measured_tcp[1]):.1f} mm from the TCP")
            self.next_place_legs = None if self.mobile else self._plan_place(self.latest_dest_hmap)
            if self.mobile:
                pass  # planned at the place station, from a scan there
            elif self.next_place_legs is None:
                self.get_logger().warn("no spot in the last tray scan; rescanning the tray with the box held")
            else:
                exit_pt = self._zone_exit(self.leg_wps[-1].pos, self.next_place_legs[0].pos)
                if exit_pt is not None:
                    self.next_place_legs.insert(0, Waypoint(exit_pt, pass_through=True))
                if not self._reach_ok(self.next_place_legs, "place"):
                    return
        elif wp.action == "place":
            self.action_pub.publish(String(data="place_held"))
            self.holding_box = False
            self._publish_held_box()
            if self.pending_place is not None:
                if self.slide_from is not None:
                    dx, dy = self.measured_tcp[:2] - self.slide_from[:2]
                    p = self.pending_place
                    self.pending_place = (p[0] + dx, p[1] + dy, *p[2:])
                    self.get_logger().info(f"slide: box released {1e3 * dx:+.1f}/{1e3 * dy:+.1f} mm (x/y) "
                                           f"from its set-down")
                    self.slide_from = None
                self.placed_boxes.append(self.pending_place)
                self.pending_place = None
        elif wp.action == "push":
            self._record_push_end()
        elif wp.action == "pushed":
            # Reported once backed off: pressing, the box leans on what stopped it.
            self.action_pub.publish(String(data="pushed"))

        self.leg_idx += 1
        if self.leg_idx < len(self.leg_wps):
            next_wp = self.leg_wps[self.leg_idx]
            self._announce_dest(next_wp)
            self._advance_reference(next_wp)
        elif self.state == self.MOVING_TO_BOX:
            weight = self.wrist_fz - (self.grasp_tare or 0.0)
            if abs(weight) < GRASP_MIN_WEIGHT_N:
                self._rescan_pile(f"lifted with {weight:+.1f} N in the wrist: nothing gripped")
                return
            self.pick_failures = 0
            if self.mobile:
                self._drive_to("place", self._at_place)
                return
            # A hull is off on legs that end inside it (see the notes): the pile hull is
            # on only while leaving the pile, the tray hull only while leaving the tray.
            if self.next_place_legs is None:
                pos = self._tray_scan_pos(held=True)
                if not self._reach_ok([Waypoint(pos)], "tray scan"):
                    return
                self.state = self.TRAVELING_TO_DEST
                self._advance_reference(Waypoint(pos, hull_active=True, dest_hull_active=False))
                return
            self.state = self.MOVING_TO_SLOT
            self.leg_wps, self.next_place_legs = self.next_place_legs, None
            self.leg_idx = 0
            self.leg_wps[0].hull_active = True  # the zone exit, else release-above
            self._announce_dest(self.leg_wps[0])
            self._advance_reference(self.leg_wps[0])
        else:
            self.boxes_moved += 1
            self._go_scan_tray(self.leg_heading_offset)

    def _advance_reference(self, wp: "Waypoint"):
        self.leg_target = np.array(wp.pos, dtype=float)
        # wp's place in leg_wps, if it is one of them (its run of pass-through legs turns as one).
        self.leg_wp_index = next((i for i, w in enumerate(self.leg_wps) if w is wp), None)
        self.leg_start = self._ref_cart()[0].copy()
        self.leg_axis0 = self.axis_cmd
        self.leg_axis_g = np.asarray(wp.tool_axis, dtype=float)
        self.leg_heading_offset = wp.heading_offset
        # The hover before a touch-down has settled: the wrist reads the box's weight.
        self.touch_baseline = self.wrist_f_filt[2] if wp.touch else None
        self.touch_side_tare = self.wrist_f_filt[:2].copy() if wp.touch else np.zeros(2)
        if wp.action == "pick":
            self.pick_contact = False
        if wp.action == "slide":
            self.slide_tare = self.wrist_f_avg[:2].copy()
            self.slide_ticks = 0
        if wp.action == "push":
            self.push_tare = self.wrist_f_filt[:2].copy()
            self.push_tcps = []
            self.push_stall_ticks = 0
        self.leg_speed = wp.max_speed
        self.leg_vel = TRANSIT_VEL if wp.transit else MAX_VEL
        self.leg_timeout_ticks = None if wp.timeout_s is None else round(wp.timeout_s / CONTROL_PERIOD_S)
        self.leg_ticks = 0
        self.leg_pass = wp.pass_through
        self.leg_blend = wp.blend
        d = self.leg_target - self._ref_cart()[0]
        self.leg_in_dir = d / max(float(np.linalg.norm(d)), 1e-9)
        self._retarget()
        self._start_heading_leg()
        self._cap_turn_rate()
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
            # Mobile job: azimuths kept within +-pi (joint 1's range), never across the back.
            c, cv, _ = _cart_to_cyl(p, v, np.zeros(3), 0.0 if self.mobile else self.inp.current_position[0])
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
        if self.state == self.TUCKING:
            return None if self.tuck_down else CARRY_POS.copy()
        if self.state == self.UNTUCKING:
            return np.asarray(self.untuck_next[1].pos, dtype=float) if self.untuck_next else None
        if self.state not in (self.MOVING_TO_BOX, self.MOVING_TO_SLOT):
            return None
        if self.leg_idx + 1 < len(self.leg_wps):
            return np.asarray(self.leg_wps[self.leg_idx + 1].pos, dtype=float)
        if self.state == self.MOVING_TO_BOX and self.next_place_legs is not None:
            return np.asarray(self.next_place_legs[0].pos, dtype=float)
        return self._tray_scan_pos(held=self.state == self.MOVING_TO_BOX)

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
                if self.current_box_size is not None else BOX_HALF_DIAGONAL_MAX_M
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

    def _daz(self, a, b):
        """The swing from azimuth a to b: the shorter way, or in the mobile job within
        +-pi (as the planner goes)."""
        return b - a if self.mobile else (b - a + np.pi) % (2 * np.pi) - np.pi

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
        """Heading profile for a new leg. A run of pass-through legs up to the next stop
        turns as one: towards the aligned heading at that stop, the turn shared out
        over the legs by their base swing (by distance if the run does not swing), so
        the wrist turns one way at a steady rate with the base.
        """
        cur = self._ref_cart()[0]
        chain = [self.leg_target]
        final_offset = self.leg_heading_offset
        if self.leg_wp_index is not None and self.leg_pass:
            for wp in self.leg_wps[self.leg_wp_index + 1:]:
                chain.append(np.asarray(wp.pos, dtype=float))
                final_offset = wp.heading_offset
                if not wp.pass_through:
                    break
        pts = [cur] + chain
        az = [float(np.arctan2(p[1], p[0])) for p in pts]
        dazs = [self._daz(a, b) for a, b in zip(az, az[1:])]
        dists = [float(np.linalg.norm(b - a)) for a, b in zip(pts, pts[1:])]
        total = float(sum(dazs))
        self.leg_az0 = az[0]
        self.leg_daz = float(dazs[0])
        psi_f = self._heading_goal(self._aligned_yaw(az[-1]) + final_offset, total,
                                   None if self.measured_q is None else self.measured_q[6])
        # In step with the base only if it swings enough: a big turn on a small swing would spin the wrist.
        self.leg_by_swing = abs(total) >= np.radians(1.0) and abs(psi_f - self.psi_cmd) <= TURN_PER_SWING_MAX * abs(total)
        if self.leg_by_swing:
            share = self.leg_daz / total
        else:
            share = dists[0] / sum(dists) if sum(dists) > 1e-6 else 1.0
        self.leg_psi0 = self.psi_cmd
        self.leg_psi_g = self.psi_cmd + (psi_f - self.psi_cmd) * share
        self.leg_single = len(chain) == 1

    def _cap_turn_rate(self):
        """Slow the leg so the tool turns no faster than HEADING_TURN_RATE_MAX: a
        heading or tilt change over a short leg otherwise spins the wrist (a turn
        squeezed into a few cm swung the arm at 4.8 rad/s)."""
        length = float(np.linalg.norm(self.leg_target - self.leg_start))
        turn = abs(self.leg_psi_g - self.leg_psi0)
        tilt = float(np.arccos(np.clip(np.dot(self.leg_axis0, self.leg_axis_g)
                                       / (np.linalg.norm(self.leg_axis0) * np.linalg.norm(self.leg_axis_g)), -1.0, 1.0)))
        caps = []
        if not self.leg_by_swing or abs(self.leg_daz) < np.radians(1.0):
            if turn + tilt > 1e-3:
                caps.append(length * HEADING_TURN_RATE_MAX / (1.5 * (turn + tilt)))  # smoothstep peak: 1.5 x mean
        else:
            # Heading in step with the base: rate = turn per azimuth x the base's rate (speed / radius).
            r = min(float(np.hypot(*self.leg_start[:2])), float(np.hypot(*self.leg_target[:2])))
            caps.append(HEADING_TURN_RATE_MAX * max(r, MIN_PLAN_RADIUS_M) * abs(self.leg_daz) / max(turn, 1e-9))
            if tilt > 1e-3:
                caps.append(length * HEADING_TURN_RATE_MAX / (1.5 * tilt))
        cap = min(caps, default=np.inf)
        if cap < (self.leg_speed if self.leg_speed is not None else np.inf):
            self.leg_speed = max(cap, MIN_TURN_LEG_SPEED_MPS)

    def _heading_at(self, pos):
        """Heading at reference position pos: turns with the base in proportion to
        the swing done, so the box arrives square and joint 7 (~ joint 1 - heading)
        moves one way (docs/implementation_notes.md)."""
        if not self.leg_by_swing or abs(self.leg_daz) < np.radians(1.0):
            # Little swing: turn with the distance covered instead (a box turned at the tray).
            total = 0.0 if self.leg_start is None else float(np.linalg.norm(self.leg_target - self.leg_start))
            frac = 1.0 if total < 1e-6 else float(np.clip(1.0 - np.linalg.norm(self.leg_target - pos) / total, 0.0, 1.0))
            if self.leg_single:
                frac = frac * frac * (3.0 - 2.0 * frac)  # from rest to rest: ease in and out
        else:
            az = float(np.arctan2(pos[1], pos[0]))
            done = self._daz(self.leg_az0, az)
            frac = float(np.clip(done / self.leg_daz, 0.0, 1.0))
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
        if self.mobile:
            # Each hull only at its own station (in the arm frame the other is nowhere).
            hull_active = hull_active and self.station == "pick"
            dest_hull_active = dest_hull_active and self.station == "place"
        idle_slot = [*NO_OBSTACLE_POSITION, NO_OBSTACLE_RADIUS]
        occupied = None
        if hull_active and self.latest_heightmap is not None:
            flat_heights = self.latest_heightmap.flatten()
            mask = flat_heights > FLOOR_Z + 1e-6
            if mask.any():
                occupied = np.array([
                    [self.source_grid[i][0], self.source_grid[i][1], flat_heights[i]]
                    for i in range(len(flat_heights)) if mask[i]
                ])
        if occupied is not None:
            center = occupied.mean(axis=0)
            radius = float(np.max(np.linalg.norm(occupied - center, axis=1))) + BOX_HALF_DIAGONAL_MAX_M + _HULL_PAD
            slot0 = [float(center[0]), float(center[1]), float(center[2]), radius]
        else:
            slot0 = idle_slot
        slot1 = ([*(float(c) for c in self.tray.center), self.tray.hull_radius(TRAY_HULL_PAD_M)]
                 if dest_hull_active and self.tray is not None else idle_slot)
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
                if self.current_box_size is not None else BOX_HALF_DIAGONAL_MAX_M
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
        if self.mobile:
            self._publish_base_flags()
        if (self.drive is not None and not self.nav_seen_goal
                and self.tick_count - self.nav_sent_tick >= NAV_RESEND_TICKS):
            self.nav_sent_tick = self.tick_count
            self.nav_goal_pub.publish(String(data=self.drive[0]))
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
                if wp.action == "pick":
                    self._arrive()  # no contact: _arrive re-scans the pile
                elif wp.touch:
                    self._retry_place("the wrist load never dropped: the box is caught on something")
                elif wp.action == "hover":
                    self._retry_place("the box did not get down to above the spot")
                else:
                    self._arrive()
        if (self.state == self.MOVING_TO_SLOT and not hold and self.leg_idx < len(self.leg_wps)
                and self.leg_wps[self.leg_idx].action == "push"):
            self._check_push_stall()
        if (self.state == self.MOVING_TO_SLOT and not hold and self.leg_idx < len(self.leg_wps)
                and self.leg_wps[self.leg_idx].action == "slide"):
            self._check_slide_contact()
        if (self.via_active and self.detour_active and not self._hold_active()
                and np.linalg.norm(np.array(self.inp.current_position) - np.array(self.inp.target_position))
                < 0.01):
            self._retarget()
        self._apply_motion_limits(self.inp)
        result = self._ruckig_step()
        if result is not None:
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
                               int(self.via_active),
                               np.nan if self.touch_baseline is None else round(self.touch_baseline, 3),
                               self.station or "")

    def _ruckig_step(self):
        """One Ruckig step; on its rare numerical failure (time sync near the target, the
        limits just scaled) again from zero acceleration, else None: the reference holds
        this tick."""
        try:
            return self.otg.update(self.inp, self.out)
        except RuckigError as e:
            self.get_logger().warn(f"reference Ruckig step failed ({str(e).strip().splitlines()[0]}); "
                                   f"again from zero acceleration")
        self.inp.current_acceleration = [0.0, 0.0, 0.0]
        try:
            return self.otg.update(self.inp, self.out)
        except RuckigError:
            return None

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
