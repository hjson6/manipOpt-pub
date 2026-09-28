"""Scene geometry and sensing settings for container pick-and-place, shared by
the plant and every method. Lengths in metres, world frame.

See docs/implementation_notes.md#scenepy.
"""
import numpy as np

from perception import heightmap

FLOOR_Z = 0.0

SCAN_RESOLUTION = 0.005  # heightmap cell; 1 cm put grasps up to 7 mm off centre
FLATNESS_TOL = 0.01
GRASP_INLIER_FRAC = 0.85  # placement stays strict
SOURCE_MIN_FOOTPRINT_CELLS = 30  # noise blobs are 9-12 cells, the smallest box top ~144
SOURCE_MIN_FILL_FRAC = 0.85
SOURCE_MAX_HEIGHT = 0.195  # tallest box top in the XML pile

SOURCE_SCAN_BOUNDS = ((0.24375, 0.64125), (0.04125, 0.45375))  # pile extent + 7.5 cm
SOURCE_GRID_POINTS, SOURCE_GRID_SHAPE = heightmap.build_dense_grid_xy(
    *SOURCE_SCAN_BOUNDS, SCAN_RESOLUTION)
CONTAINER_CAM_NAME = "container_cam"
CONTAINER_CAM_MOUNT_ID = "container_cam_mount"
CONTAINER_CAM_MOUNT_OFFSET = np.array([0.0, 0.0, 0.15])  # from the gripper
CONTAINER_CAM_REF_Z = (FLOOR_Z + SOURCE_MAX_HEIGHT) / 2.0


# Cell spec for box heights (the camera sees only tops): plan with the range until the
# height is measured at touch-down.
BOX_HEIGHT_MIN_M = 0.07
BOX_HEIGHT_MAX_M = 0.13

TOOL_RADIUS_M = 0.03  # "tool" geom in panda_robot.xml
# link7's collision mesh in the TCP frame (tool z down): lowest point above the TCP, and
# reach past the TCP along tool +x, -x, +y, -y. link6 starts 0.156 m above the TCP.
WRIST_ABOVE_TCP_M = 0.100
WRIST_EXTENT_TOOL = (0.044, 0.044, 0.054, 0.088)

LIFT_HEIGHT = 0.45  # clears pile and tray with a box in hand; higher loses reach
SCAN_HEIGHT = LIFT_HEIGHT + 0.05
PARK_POSITION = np.array([
    float(np.mean(SOURCE_SCAN_BOUNDS[0])), float(np.mean(SOURCE_SCAN_BOUNDS[1])), SCAN_HEIGHT])

# Destination tray, a fixture: inner wall faces, floor top, wall top. Must match the
# dest_tray_* geoms in panda_scene_container.xml. Re-test the far corner before
# moving it: docs/implementation_notes.md#scenepy
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
# Overhead obstacle-sensing camera. Pose and fovy must match workspace_cam in
# panda_scene_container.xml.
WORKSPACE_CAM_NAME = "workspace_cam"
WORKSPACE_CAM_POS = np.array([0.0, -0.20, 3.5])
WORKSPACE_CAM_FOVY_DEG = 70.0
WORKSPACE_CAM_WIDTH = 320
WORKSPACE_CAM_HEIGHT = 240
WORKSPACE_CAM_RATE_HZ = 10.0

# Obstacle detection ignores the pile and tray, up to just above their contents.
AISLE_MASK_MARGIN_M = 0.03
AISLE_MASK_BOXES = (  # (x bounds, y bounds, top z)
    (*SOURCE_SCAN_BOUNDS, SOURCE_MAX_HEIGHT + 0.07),
    (*DEST_SCAN_BOUNDS, DEST_TRAY_Z_MAX + 0.05),
)
FOREGROUND_MIN_HEIGHT_M = 0.05
DETECTION_MIN_BLOB_PX = 25  # ~40 px for a 0.14 m carton on the floor
