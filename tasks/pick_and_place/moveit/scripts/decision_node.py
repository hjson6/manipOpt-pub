#!/usr/bin/env python3
"""MoveIt method's task sequencing for container pick-and-place, the
counterpart of pick_place_mpc's task_node.

Same scans, vantage points and pick rule as task_node (largest exposed box
first, grasp at the sensed top). Placement is the heightmap search
(find_best_footprint, lowest) only; task_node's row packing, flush snap and
in-hand offset are not used here. Each leg is one mtc_executor_node action:

    MoveTo(PARK) -> scan pile -> Pick -> MoveTo(DEST_SCAN) -> scan tray -> Place


MoveIt avoids only what is in its planning scene, so this node adds the
floor and tray (from scene.py) and the remaining pile, rebuilt from every
pile scan as 2 cm columns, leaving out the target's footprint (the executor
attaches the target itself). Known gap: boxes under a just-picked box are
missing until the next scan; no motion in that window reaches into the pile.

The sequence runs on a worker thread; a MultiThreadedExecutor keeps the
callbacks flowing while it blocks.
"""
import threading
import time

import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from geometry_msgs.msg import Point, Pose
from moveit_msgs.msg import CollisionObject, PlanningScene
from moveit_msgs.srv import ApplyPlanningScene
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Empty, Float64MultiArray

from perception import heightmap
from pick_place_common.scene import (
    DEST_FLOOR_Z, DEST_GRID_SHAPE, DEST_SCAN_BOUNDS, DEST_SCAN_POSITION,
    DEST_TRAY_X_MAX, DEST_TRAY_X_MIN, DEST_TRAY_Y_MAX, DEST_TRAY_Y_MIN,
    DEST_TRAY_Z_MAX, DEST_TRAY_Z_MIN, DEST_WALL_THICKNESS, FLATNESS_TOL,
    FLOOR_Z, GRASP_INLIER_FRAC, PARK_POSITION, SCAN_RESOLUTION,
    SOURCE_GRID_SHAPE, SOURCE_MIN_FILL_FRAC, SOURCE_MIN_FOOTPRINT_CELLS,
    SOURCE_SCAN_BOUNDS,
)
from pick_place_interfaces.action import MoveTo, Pick, Place

WORLD = "world"
# As task_node: scans are request/response over plain topics, so retry.
SCAN_TIMEOUT_S = 2.0
SCAN_MAX_ATTEMPTS = 5
DEST_SEARCH_MAX_ATTEMPTS = 5  # retries when a scan finds no flat spot (noise), bounded
ACTION_TIMEOUT_S = 180.0

FLOOR_GAP = 0.005
PILE_BLOCK_CELLS = 4  # 4 x 0.005 m = 2 cm blocks
PILE_OCCUPIED_ABOVE = FLATNESS_TOL  # a block is occupied above this height
PILE_ID = "source_pile"


def _box(object_id, center, size):
    obj = CollisionObject()
    obj.header.frame_id = WORLD
    obj.id = object_id
    obj.operation = CollisionObject.ADD
    _append_box(obj, center, size)
    return obj


def _append_box(obj, center, size):
    prim = SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[float(s) for s in size])
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = (float(c) for c in center)
    pose.orientation.w = 1.0
    obj.primitives.append(prim)
    obj.primitive_poses.append(pose)


def static_cell_objects():
    """Floor and tray (floor slab and 4 walls) from scene.py's tray constants,
    which match the dest_tray_* geoms.
    """
    t = DEST_WALL_THICKNESS
    x0, x1, y0, y1 = DEST_TRAY_X_MIN, DEST_TRAY_X_MAX, DEST_TRAY_Y_MIN, DEST_TRAY_Y_MAX
    z0, z1 = DEST_TRAY_Z_MIN, DEST_TRAY_Z_MAX
    cx, cy, zc = (x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2
    w, d, h = x1 - x0, y1 - y0, z1 - z0
    return [
        # Floor top FLOOR_GAP below z = 0: flush with panda_link0's collision mesh
        # made the first start state invalid.
        _box("floor", (0.0, 0.0, FLOOR_Z - FLOOR_GAP - 0.05), (3.0, 3.0, 0.1)),
        _box("tray_floor", (cx, cy, z0 - t / 2), (w + 2 * t, d + 2 * t, t)),
        _box("tray_wall_west", (x0 - t / 2, cy, zc), (t, d, h)),
        _box("tray_wall_east", (x1 + t / 2, cy, zc), (t, d, h)),
        _box("tray_wall_south", (cx, y0 - t / 2, zc), (w + 2 * t, t, h)),
        _box("tray_wall_north", (cx, y1 + t / 2, zc), (w + 2 * t, t, h)),
    ]


def pile_object(hmap, exclude_rc=None):
    """The remaining pile as one CollisionObject of 2 cm columns from a sensed
    heightmap, leaving out exclude_rc = (row0, col0, row1, col1). None if
    nothing is left.
    """
    (xb, yb) = SOURCE_SCAN_BOUNDS
    n_rows, n_cols = hmap.shape
    obj = CollisionObject()
    obj.header.frame_id = WORLD
    obj.id = PILE_ID
    obj.operation = CollisionObject.ADD
    b = PILE_BLOCK_CELLS
    for r0 in range(0, n_rows, b):
        for c0 in range(0, n_cols, b):
            r1, c1 = min(r0 + b, n_rows), min(c0 + b, n_cols)
            if exclude_rc is not None:
                er0, ec0, er1, ec1 = exclude_rc
                if r0 < er1 and er0 < r1 and c0 < ec1 and ec0 < c1:
                    continue
            top = float(np.max(hmap[r0:r1, c0:c1]))
            if top <= FLOOR_Z + PILE_OCCUPIED_ABOVE:
                continue
            x_lo = xb[0] + (c0 - 0.5) * SCAN_RESOLUTION  # a block covers its samples +- half a cell
            x_hi = xb[0] + (c1 - 0.5) * SCAN_RESOLUTION
            y_lo = yb[0] + (r0 - 0.5) * SCAN_RESOLUTION
            y_hi = yb[0] + (r1 - 0.5) * SCAN_RESOLUTION
            _append_box(obj,
                        ((x_lo + x_hi) / 2, (y_lo + y_hi) / 2, (FLOOR_Z + top) / 2),
                        (x_hi - x_lo, y_hi - y_lo, top - FLOOR_Z))
    return obj if obj.primitives else None


def _pt(v):
    return Point(x=float(v[0]), y=float(v[1]), z=float(v[2]))


class DecisionNode(Node):
    def __init__(self):
        super().__init__("decision_node")
        self.pick_client = ActionClient(self, Pick, "pick")
        self.place_client = ActionClient(self, Place, "place")
        self.move_to_client = ActionClient(self, MoveTo, "move_to")
        self.scene_client = self.create_client(ApplyPlanningScene, "apply_planning_scene")

        self._scan_lock = threading.Lock()
        self._scan_result = {}
        self._scan_event = {"source": threading.Event(), "dest": threading.Event()}
        self.scan_pub = self.create_publisher(Empty, "/sim/scan_container", 10)
        self.dest_scan_pub = self.create_publisher(Empty, "/sim/scan_destination", 10)
        self.create_subscription(Float64MultiArray, "/sim/container_occupancy",
                                 lambda m: self._on_scan("source", m), 10)
        self.create_subscription(Float64MultiArray, "/sim/destination_occupancy",
                                 lambda m: self._on_scan("dest", m), 10)

        self._pile_present = False
        self._robot_state_seen = threading.Event()
        self.create_subscription(JointState, "/joint_states",
                                 lambda _m: self._robot_state_seen.set(), 10)

        self.done = threading.Event()
        threading.Thread(target=self._run_guarded, daemon=True).start()

    def _on_scan(self, which, msg):
        with self._scan_lock:
            self._scan_result[which] = np.array(msg.data)
        self._scan_event[which].set()

    def _scan(self, which):
        pub, shape = ((self.scan_pub, SOURCE_GRID_SHAPE) if which == "source"
                      else (self.dest_scan_pub, DEST_GRID_SHAPE))
        for attempt in range(SCAN_MAX_ATTEMPTS):
            self._scan_event[which].clear()
            pub.publish(Empty())
            if self._scan_event[which].wait(SCAN_TIMEOUT_S):
                with self._scan_lock:
                    return self._scan_result[which].reshape(shape)
            self.get_logger().warn(f"{which} scan unanswered (attempt {attempt + 1}); retrying")
        raise RuntimeError(f"{which} scan never answered")

    def _wait_future(self, future, timeout):
        ev = threading.Event()
        future.add_done_callback(lambda _f: ev.set())
        if not ev.wait(timeout):
            raise RuntimeError("timed out waiting on a future")
        return future.result()

    def _call(self, client, goal, label):
        handle = self._wait_future(client.send_goal_async(goal), 10.0)
        if not handle.accepted:
            raise RuntimeError(f"{label} goal rejected")
        res = self._wait_future(handle.get_result_async(), ACTION_TIMEOUT_S).result
        if not res.success:
            raise RuntimeError(f"{label} failed: {res.failure_reason}")
        return res

    def _apply_scene(self, objects):
        scene = PlanningScene(is_diff=True)
        scene.world.collision_objects = objects
        resp = self._wait_future(
            self.scene_client.call_async(ApplyPlanningScene.Request(scene=scene)), 10.0)
        if not resp.success:
            raise RuntimeError("apply_planning_scene failed")

    def _set_pile(self, pile):
        # REMOVE of an absent object fails the whole call, so remove only if present.
        if pile is not None:
            self._apply_scene([pile])
        elif self._pile_present:
            self._apply_scene([CollisionObject(id=PILE_ID, operation=CollisionObject.REMOVE)])
        self._pile_present = pile is not None

    def _move_to(self, position, label):
        self._call(self.move_to_client, MoveTo.Goal(tcp_position=_pt(position)), f"move_to {label}")

    def _run_guarded(self):
        try:
            self._run()
        except Exception as e:  # noqa: BLE001 -- report and stop the run
            self.get_logger().error(f"run aborted: {e}")
        finally:
            self.done.set()

    def _run(self):
        for c in (self.pick_client, self.place_client, self.move_to_client):
            c.wait_for_server()
        self.scene_client.wait_for_service()
        # Until the first /joint_states, move_group plans from all-zero joints, which
        # is outside joint 4's range (START_STATE_INVALID). The short sleep lets its
        # state monitor catch up.
        self.get_logger().info("waiting for the plant (/joint_states)...")
        if not self._robot_state_seen.wait(60.0):
            raise RuntimeError("no /joint_states within 60 s; is mujoco_sim_node running?")
        time.sleep(1.0)
        self._apply_scene(static_cell_objects())
        self.get_logger().info("static cell geometry (floor + tray) added to planning scene")

        t_start = time.monotonic()
        cycle_times = []
        boxes_moved = 0
        while rclpy.ok():
            t_cycle = time.monotonic()
            self._move_to(PARK_POSITION, "PARK")

            hmap = self._scan("source")
            boxes = heightmap.find_topmost_boxes(
                hmap, floor_z=FLOOR_Z, flatness_tol=FLATNESS_TOL,
                min_footprint_cells=SOURCE_MIN_FOOTPRINT_CELLS,
                inlier_frac=GRASP_INLIER_FRAC, min_fill_frac=SOURCE_MIN_FILL_FRAC)
            if not boxes:
                self._set_pile(None)
                self.get_logger().info(
                    f"container empty; moved {boxes_moved} boxes in {time.monotonic() - t_start:.1f}s "
                    f"(per box: {', '.join(f'{t:.1f}' for t in cycle_times)} s); parked")
                return
            row0, col0, row1, col1, height, area = heightmap.pick_order(hmap, boxes, FLATNESS_TOL)[0]
            footprint_cells = (row1 - row0, col1 - col0)
            x, y = heightmap.footprint_center_xy(
                row0, col0, footprint_cells, *SOURCE_SCAN_BOUNDS, SCAN_RESOLUTION)
            self.get_logger().info(
                f"box {boxes_moved + 1}: selected ({x:.3f}, {y:.3f}) "
                f"(sensed height {height:.3f}m, footprint area {area} cells)")

            b = PILE_BLOCK_CELLS
            self._set_pile(pile_object(hmap, exclude_rc=(row0 - b, col0 - b, row1 + b, col1 + b)))

            pick = Pick.Goal(top_surface_point=_pt((x, y, height)))
            pick.footprint_size.x = max(footprint_cells[1] - 1, 1) * SCAN_RESOLUTION  # sensed extent between the outermost samples
            pick.footprint_size.y = max(footprint_cells[0] - 1, 1) * SCAN_RESOLUTION
            size = self._call(self.pick_client, pick, "pick").grasped_box_size
            hx, hy, hz = size.x, size.y, size.z

            self._move_to(DEST_SCAN_POSITION, "DEST_SCAN")
            dest_cells = (round(2 * hy / SCAN_RESOLUTION) + 1, round(2 * hx / SCAN_RESOLUTION) + 1)
            result = None
            for attempt in range(DEST_SEARCH_MAX_ATTEMPTS):
                result = heightmap.find_best_footprint(
                    self._scan("dest"), dest_cells, mode="lowest",
                    flatness_tol=FLATNESS_TOL, floor_z=DEST_FLOOR_Z)
                if result is not None:
                    break
                self.get_logger().warn(
                    f"destination scan found no flat placement spot (attempt {attempt + 1}); retrying")
            if result is None:
                raise RuntimeError("no flat placement spot after repeated destination scans")
            row, col, surface_height = result
            dx, dy = heightmap.footprint_center_xy(
                row, col, dest_cells, *DEST_SCAN_BOUNDS, SCAN_RESOLUTION)
            stack_height = surface_height - DEST_FLOOR_Z + 2 * hz
            self.get_logger().info(
                f"box {boxes_moved + 1}: placing at ({dx:.3f}, {dy:.3f}) "
                f"(sensed surface {surface_height:.3f}m, box height {2 * hz:.3f}m, "
                f"stack height now {stack_height:.3f}m)")

            self._call(self.place_client,
                       Place.Goal(surface_point=_pt((dx, dy, surface_height))), "place")
            boxes_moved += 1
            cycle_times.append(time.monotonic() - t_cycle)
            self.get_logger().info(f"box {boxes_moved} done in {cycle_times[-1]:.1f}s")


def main():
    rclpy.init()
    node = DecisionNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        while rclpy.ok() and not node.done.is_set():
            executor.spin_once(timeout_sec=0.1)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
