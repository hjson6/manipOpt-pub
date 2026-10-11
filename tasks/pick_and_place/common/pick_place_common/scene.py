"""Scene geometry and sensing settings for container pick-and-place, shared by
the plant and every method. Lengths in metres. Two frames: the room (the
plant's world; origin at a corner, floor z = 0) and the arm's base frame
(link0; the method's frame for the arm task, the cell's frame when the base is
parked at the cell). Unless named ROOM_ or BASE_, a constant is in the arm frame.

See docs/implementation_notes.md#scenepy.
"""
import numpy as np

from perception import heightmap

FLOOR_Z = 0.0  # the tables' tops, where the zones are, at the arm's mount height
ROOM_FLOOR_Z = -0.75  # the room floor: the arm's mount and the tables' tops are 0.75 m above it

# Room: 7.0 x 5.6 m, and the footprints ((x0, x1), (y0, y1)) of its pillar and shelf. Must
# match room_scene.xml.
ROOM_SIZE = (7.0, 5.6)
ROOM_FIXTURES = (((0.6, 1.0), (4.6, 5.0)), ((3.0, 5.0), (5.2, 5.6)))
# The cell's pose in the room (x, y, yaw): the arm frame with the base parked at the cell.
# Must match the "cell" frame in room_scene.xml.
CELL_POSE = (3.0, 2.8, np.pi / 2)

# Stations layout (the mobile job): each station is one of the cell's tables, with its
# load, at its own pose in the room (the cell's frame there, so a table stands where it
# stood beside the parked arm); the robot starts at home, by the south wall. The pick
# table backs onto the west wall, the tray's onto the east wall, 5 cm off. Used by
# load_scene_model(layout="stations").
PICK_STATION_POSE = (0.79, 2.2, np.pi)
PLACE_STATION_POSE = (6.25, 4.35, np.pi / 2)
BASE_HOME_POSE = (3.5, 1.0, np.pi / 2)

# Docking: where a technician parks the robot at each station, facing its table with the
# chassis DOCK_GAP_M off the table's edge. The place dock is the cell's pose (the arm where
# it stood); the pick dock is below, after the table's footprint.
DOCK_GAP_M = 0.05
PLACE_DOCK_ARM = PLACE_STATION_POSE
# The arm folded back over the chassis for driving: every link inside the footprint,
# above the pedestal (the carry pose without a box).
ARM_TUCKED_Q = (-np.pi / 2, -1.76, 0.0, -3.0, 0.0, 1.6, 0.785)
# The carry pose of the mobile job: the tool straight down 0.40 m behind the arm's base,
# 0.22 m above its mount (arm frame), over the rear deck; the largest box hangs inside
# the footprint, above the pedestal.
CARRY_TCP_POSITION = (0.0, 0.40, 0.22)
ARM_CARRY_Q = (np.pi / 2, -0.2692, 0.0, -2.6853, 0.0, 2.4161, np.pi / 4)

# Mobile base, frame base_link: x forward, origin on the floor midway between the drive
# wheels. Nominal (taught) values; must match base_link in room_scene.xml.
BASE_CHASSIS_HALF = (0.40, 0.28)  # footprint half-sizes; 0.30 m tall, 0.04 m off the floor
BASE_WHEEL_RADIUS_M = 0.10
BASE_WHEEL_TRACK_M = 0.50
BASE_ARM_MOUNT = (0.20, 0.0, -ROOM_FLOOR_Z, np.pi / 2)  # x, y, z, yaw of link0 in base_link


def compose(a, b):
    """Planar pose a * b, each (x, y, yaw)."""
    c, s = np.cos(a[2]), np.sin(a[2])
    return (a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], a[2] + b[2])


def invert(a):
    """Planar pose a^-1."""
    c, s = np.cos(a[2]), np.sin(a[2])
    return (-c * a[0] - s * a[1], s * a[0] - c * a[1], -a[2])


ARM_IN_BASE = (BASE_ARM_MOUNT[0], BASE_ARM_MOUNT[1], BASE_ARM_MOUNT[3])
BASE_IN_ARM = invert(ARM_IN_BASE)
BASE_PARK_POSE = compose(CELL_POSE, BASE_IN_ARM)  # base_link in the room, parked at the cell
PLACE_DOCK_BASE = compose(PLACE_DOCK_ARM, BASE_IN_ARM)  # base_link at the place station, in the room


def room_to_map(pose):
    """A room pose in the stations layout's map frame, which is anchored at home (the
    teaching of commissioning: the map is drawn from home)."""
    c, s = np.cos(BASE_HOME_POSE[2]), np.sin(BASE_HOME_POSE[2])
    dx, dy = pose[0] - BASE_HOME_POSE[0], pose[1] - BASE_HOME_POSE[1]
    return (c * dx + s * dy, -s * dx + c * dy, (pose[2] - BASE_HOME_POSE[2] + np.pi) % (2 * np.pi) - np.pi)


def base_to_arm_xy(x, y):
    """A base_link point (x, y) in the arm frame."""
    return compose(BASE_IN_ARM, (x, y, 0.0))[:2]


# The chassis's footprint in the arm frame (the robot's own furniture).
CHASSIS_BOUNDS = tuple(
    (float(min(v)), float(max(v))) for v in zip(*[base_to_arm_xy(sx * BASE_CHASSIS_HALF[0], sy * BASE_CHASSIS_HALF[1])
                                                    for sx in (-1, 1) for sy in (-1, 1)]))

# The cell's furniture (taught at installation), footprints ((x0, x1), (y0, y1)); tops at
# FLOOR_Z. Must match pick_table and place_table in cell_container.xml.
PICK_TABLE_BOUNDS = ((0.30, 0.74), (-0.05, 0.55))
PLACE_TABLE_BOUNDS = ((-0.65, 0.05), (-0.70, -0.25))

SCAN_RESOLUTION = 0.005  # heightmap cell; 1 cm put grasps up to 7 mm off centre
FLATNESS_TOL = 0.01
GRASP_INLIER_FRAC = 0.85  # placement stays strict
SOURCE_MIN_FOOTPRINT_CELLS = 30  # noise blobs are 9-12 cells, the smallest spec top 100
SOURCE_MIN_FILL_FRAC = 0.85

# Pick zone (taught): where a pile is put, and the highest pile the cell takes.
PICK_ZONE_CENTER = (0.44, 0.25)
PICK_ZONE_SIZE = (0.50, 0.50)
PICK_ZONE_MAX_HEIGHT_M = 0.35  # until the first scan measures the pile; the wrist camera's reach sets it
SOURCE_SCAN_BOUNDS = tuple((c - s / 2.0, c + s / 2.0) for c, s in zip(PICK_ZONE_CENTER, PICK_ZONE_SIZE))
SOURCE_GRID_POINTS, SOURCE_GRID_SHAPE = heightmap.build_dense_grid_xy(
    *SOURCE_SCAN_BOUNDS, SCAN_RESOLUTION)


def _footprint(pose, bounds):
    """((x0, x1), (y0, y1)) of a frame's axis-aligned rectangle seen from the room, or
    from another frame (pose: the rectangle's frame in it)."""
    pts = np.array([compose(pose, (x, y, 0.0))[:2] for x in bounds[0] for y in bounds[1]])
    return tuple((float(lo), float(hi)) for lo, hi in zip(pts.min(axis=0), pts.max(axis=0)))


# The mobile job's pick dock faces the pick table (it backs onto the west wall), centred
# on the pick zone; the pick zone there, in the docked arm's frame: the table's top 3 cm
# in from its edges, the cell zone's width along them (taught with the robot parked).
PICK_TABLE_ROOM = _footprint(PICK_STATION_POSE, PICK_TABLE_BOUNDS)
_PICK_ZONE_ROOM_Y = _footprint(PICK_STATION_POSE, SOURCE_SCAN_BOUNDS)[1]
PICK_DOCK_BASE = (PICK_TABLE_ROOM[0][1] + DOCK_GAP_M + BASE_CHASSIS_HALF[0], float(np.mean(_PICK_ZONE_ROOM_Y)), np.pi)
PICK_DOCK_ARM = compose(PICK_DOCK_BASE, ARM_IN_BASE)
MOBILE_PICK_ZONE_BOUNDS = _footprint(invert(PICK_DOCK_ARM), (
    (PICK_TABLE_ROOM[0][0] + 0.03, PICK_TABLE_ROOM[0][1] - 0.03), _PICK_ZONE_ROOM_Y))
MOBILE_SOURCE_GRID_POINTS, MOBILE_SOURCE_GRID_SHAPE = heightmap.build_dense_grid_xy(
    *MOBILE_PICK_ZONE_BOUNDS, SCAN_RESOLUTION)
MOBILE_PICK_ZONE_MAX_HEIGHT_M = 0.30  # the first scan's pose is out of reach above this, there


def pick_zone(mobile):
    """(scan bounds, grid points, grid shape, highest pile) of the pick zone: at the mobile
    job's pick dock, or the parked cell's."""
    if mobile:
        return MOBILE_PICK_ZONE_BOUNDS, MOBILE_SOURCE_GRID_POINTS, MOBILE_SOURCE_GRID_SHAPE, MOBILE_PICK_ZONE_MAX_HEIGHT_M
    return SOURCE_SCAN_BOUNDS, SOURCE_GRID_POINTS, SOURCE_GRID_SHAPE, PICK_ZONE_MAX_HEIGHT_M
CONTAINER_CAM_NAME = "container_cam"
CONTAINER_CAM_FOVY_DEG = 55.0
DEPTH_MIN_RANGE_M = 0.30  # the depth camera's near limit
# Wrist camera on the wrist housing's side tab, looking along the tool (calibrated mount,
# TCP frame, tool z down); image up points away from the tool. Must match container_cam
# in panda_robot.xml (attachment frame = TCP frame shifted 0.10 m up the tool).
CONTAINER_CAM_POS_TCP = np.array([0.0, -0.10, -0.13])
CONTAINER_CAM_ROT_TCP = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]).T  # columns: camera x, y, z


# Cell spec for boxes (the camera sees only tops): plan with the range until the
# height is measured at touch-down.
BOX_SIDE_MIN_M = 0.05
BOX_SIDE_MAX_M = 0.16
BOX_HEIGHT_MIN_M = 0.07
BOX_HEIGHT_MAX_M = 0.13
BOX_HALF_DIAGONAL_MAX_M = float(np.linalg.norm([BOX_SIDE_MAX_M / 2.0, BOX_SIDE_MAX_M / 2.0, BOX_HEIGHT_MAX_M / 2.0]))

TOOL_RADIUS_M = 0.03  # "tool" geom in panda_robot.xml
TOOL_TIP_ABOVE_TCP_M = 0.003
TOOL_LENGTH_M = 0.097
# link7's collision mesh in the TCP frame (tool z down): lowest point above the TCP, and
# reach past the TCP along tool +x, -x, +y, -y. link6 starts 0.156 m above the TCP.
WRIST_ABOVE_TCP_M = 0.100
WRIST_EXTENT_TOOL = (0.044, 0.044, 0.054, 0.088)

# Nominal heights for the MoveIt method; the MPC method computes them from each scan.
LIFT_HEIGHT = 0.45  # clears pile and tray with a box in hand; higher loses reach
SCAN_HEIGHT = LIFT_HEIGHT + 0.05
PARK_POSITION = np.array([
    float(np.mean(SOURCE_SCAN_BOUNDS[0])), float(np.mean(SOURCE_SCAN_BOUNDS[1])), SCAN_HEIGHT])

# Place zone (taught): where a tray is put, nominal tray +- 8 cm, and the highest the
# tray and its contents get. The MPC method finds the tray in it (perception/tray_detection.py).
PLACE_ZONE_CENTER = (-0.30, -0.425)
PLACE_ZONE_SIZE = (0.61, 0.46)
PLACE_ZONE_MAX_HEIGHT_M = 0.30
PLACE_ZONE_BOUNDS = tuple((c - s / 2.0, c + s / 2.0) for c, s in zip(PLACE_ZONE_CENTER, PLACE_ZONE_SIZE))

# Nominal destination tray: inner wall faces, floor top, wall top. Must match the
# dest_tray_* geoms in cell_container.xml; the plant, the MoveIt method and the
# person's script use it. Re-test the far corner before moving it:
# docs/implementation_notes.md#scenepy
DEST_TRAY_X_MIN, DEST_TRAY_X_MAX = -0.515, -0.085
DEST_TRAY_Y_MIN, DEST_TRAY_Y_MAX = -0.565, -0.285
DEST_TRAY_Z_MIN = 0.0  # floor top
DEST_TRAY_Z_MAX = 0.22  # wall top
DEST_FLOOR_Z = DEST_TRAY_Z_MIN
DEST_WALL_THICKNESS = 0.01

# Not inset by the box size: the search already keeps footprints inside.
DEST_SCAN_BOUNDS = (
    (DEST_TRAY_X_MIN, DEST_TRAY_X_MAX),
    (DEST_TRAY_Y_MIN, DEST_TRAY_Y_MAX),
)
DEST_GRID_POINTS, DEST_GRID_SHAPE = heightmap.build_dense_grid_xy(
    *DEST_SCAN_BOUNDS, SCAN_RESOLUTION)

# Bounding sphere around the tray (obstacle slot 1).
DEST_HULL_CENTER = np.array([
    (DEST_TRAY_X_MIN + DEST_TRAY_X_MAX) / 2.0,
    (DEST_TRAY_Y_MIN + DEST_TRAY_Y_MAX) / 2.0,
    (DEST_TRAY_Z_MIN + DEST_TRAY_Z_MAX) / 2.0,
])
DEST_HULL_RADIUS = float(np.linalg.norm([
    (DEST_TRAY_X_MAX - DEST_TRAY_X_MIN) / 2.0,
    (DEST_TRAY_Y_MAX - DEST_TRAY_Y_MIN) / 2.0,
    (DEST_TRAY_Z_MAX - DEST_TRAY_Z_MIN) / 2.0,
])) + 0.02

DEST_CAM_REF_Z = 0.09  # first height guess for the tray scan's parallax correction
DEST_SCAN_POSITION = np.array([DEST_HULL_CENTER[0], DEST_HULL_CENTER[1], SCAN_HEIGHT])

CONTAINER_CAM_WIDTH = 320
CONTAINER_CAM_HEIGHT = 240


# ---------------------------------------------------------------------------
# Overhead obstacle-sensing camera, fixed in the room above the cell. Pose (arm frame, the
# base parked at the cell) and fovy must match workspace_cam in cell_container.xml.
WORKSPACE_CAM_NAME = "workspace_cam"
WORKSPACE_CAM_POS = np.array([0.0, -0.20, 2.75])  # 3.5 m above the room floor
WORKSPACE_CAM_FOVY_DEG = 70.0
WORKSPACE_CAM_WIDTH = 320
WORKSPACE_CAM_HEIGHT = 240
WORKSPACE_CAM_RATE_HZ = 10.0

# Safety-lidar stand-ins: 270 deg planar scanners with their optical centres on the
# chassis's front-left and rear-right corners, so each sector runs along both faces and
# the two see all round the outline and nothing of the robot; 0.20 m above the floor.
# Calibration in base_link and in the arm frame, each (x, y, z, yaw of the sector's
# centre). Must match the lidar_* sites in room_scene.xml.
LIDAR_HEIGHT_M = 0.20
LIDAR_MOUNTS_BASE = tuple(
    (sx * BASE_CHASSIS_HALF[0], sx * BASE_CHASSIS_HALF[1], LIDAR_HEIGHT_M, yaw)
    for sx, yaw in ((1, np.radians(45.0)), (-1, np.radians(-135.0))))
LIDAR_Z = ROOM_FLOOR_Z + LIDAR_HEIGHT_M
LIDAR_SCANNERS = tuple(
    (*compose(BASE_IN_ARM, (x, y, 0.0))[:2], LIDAR_Z, float(np.arctan2(np.sin(yaw - ARM_IN_BASE[2]),
                                                                      np.cos(yaw - ARM_IN_BASE[2]))))
    for x, y, _z, yaw in LIDAR_MOUNTS_BASE)
LIDAR_FOV_DEG = 270.0
LIDAR_STEP_DEG = 0.5
LIDAR_RANGE_M = (0.0, 10.0)  # safety scanners: from the front window on
LIDAR_RATE_HZ = 15.0
LIDAR_RANGE_SIGMA_M = 0.02
LIDAR_DROPOUT = 0.01  # share of beams with no return

# Obstacle detection ignores the pick zone up to just above the last scanned pile
# top, and the tray.
AISLE_MASK_MARGIN_M = 0.03
AISLE_MASK_PILE_ABOVE_M = 0.07
AISLE_MASK_TRAY = (*PLACE_ZONE_BOUNDS, PLACE_ZONE_MAX_HEIGHT_M)  # (x bounds, y bounds, top z)
FOREGROUND_MIN_HEIGHT_M = 0.05  # above the room floor
# Obstacle detection also ignores the furniture, up to just above its tops (depth noise).
FURNITURE_MASK_ABOVE_M = 0.03
FURNITURE_MASKS = tuple((*b, FLOOR_Z + FURNITURE_MASK_ABOVE_M)
                        for b in (CHASSIS_BOUNDS, PICK_TABLE_BOUNDS, PLACE_TABLE_BOUNDS))
DETECTION_MIN_BLOB_PX = 25  # ~40 px for a 0.14 m carton on the floor


def tool_down_rot(psi):
    """TCP rotation with the tool straight down and tool +x at world heading psi."""
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, s, 0.0], [s, -c, 0.0], [0.0, 0.0, -1.0]])


def container_cam_pose(tcp_pos, tcp_rot):
    """Wrist camera position and rotation (columns: camera axes) at a TCP pose."""
    tcp_rot = np.asarray(tcp_rot)
    return np.asarray(tcp_pos) + tcp_rot @ CONTAINER_CAM_POS_TCP, tcp_rot @ CONTAINER_CAM_ROT_TCP


def tool_points_tcp(n=32):
    """Points on the tool cylinder's surface, TCP frame."""
    a = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    rim = np.column_stack([TOOL_RADIUS_M * np.cos(a), TOOL_RADIUS_M * np.sin(a)])
    zs = -TOOL_TIP_ABOVE_TCP_M - np.linspace(0.0, TOOL_LENGTH_M, 8)
    return np.array([[x, y, z] for z in zs for x, y in rim])
