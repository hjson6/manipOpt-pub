"""Environment actor for the obstacle scenarios: moves a person or a carton in
the scene. Its output is ground truth for mujoco_sim_node (which moves the
body so the cameras see it) and for monitoring; nothing on the control path
reads it.

/env/dynamic_obstacle, every tick, in the room frame: [x, y, z, radius, t_pub,
top_z, kind, heading]. kind 1 = person standing at (x, y) (z = the room floor), radius =
half arm span, top_z = head height, heading = facing (rad). kind 0 = sphere centred
at (x, y, z). radius 0 = absent. t_pub lets a consumer match the pose to a
sensor frame by time.

Modes (`mode`):
- visit (default): a person walks in from outside the camera's view, stands
  beside the tray table on the arm's side of it for `dwell_s`, and walks out;
  `passes` visits, `pause_s` apart (0 passes = for ever). They stop short
  of the arm (PERSON_ROBOT_GAP_M; as the environment, this node may read the
  real arm pose and the base's true pose).


- walk: a person crosses the cell WALK_START -> WALK_END and back.
- static: a carton-sized sphere at `static_position`.
The paths are laid out in the cell's frame (scene.CELL_POSE) and sent in the room's.
Full size: 1.75 m tall, walking at 1.2 m/s. That is not the supervisor's
human_speed (1.6 m/s, ISO 13855), which is a safety figure.
`trigger`: "launch" or "first_pick" (`delay_s` after either).
`duration_s` > 0 removes the obstacle that long after it appeared.
"""
from pathlib import Path

import numpy as np
import pinocchio as pin
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String

from pick_place_common.scene import (
    ARM_IN_BASE, CELL_POSE, DEST_TRAY_X_MAX, DEST_TRAY_X_MIN, DEST_TRAY_Y_MAX, DEST_TRAY_Y_MIN,
    PLACE_TABLE_BOUNDS, ROOM_FLOOR_Z, compose)


CONTROL_PERIOD_S = 0.02  # matches MPCConfig.dt

IDLE_POSITION = (0.9, 0.9, -3.0)  # absent: below the floor (room frame)

# Must match the person_obstacle body in room_scene.xml.
PERSON_HALF_SPAN = 0.30  # half the width across the arms
PERSON_TOP = ROOM_FLOOR_Z + 1.75  # top of the head
FURNITURE_GAP_M = 0.03  # the person's arms to a table or the robot base

# Visit: stand at the tray table's corner on the arm's side, clear of the table
# and the robot, facing the tray; the spot of the fixed cell (beside its old 0.50 m base).
TRAY_CENTER = ((DEST_TRAY_X_MIN + DEST_TRAY_X_MAX) / 2, (DEST_TRAY_Y_MIN + DEST_TRAY_Y_MAX) / 2)
VISIT_STAND = (PLACE_TABLE_BOUNDS[0][1] + PERSON_HALF_SPAN + FURNITURE_GAP_M,
               -0.25 - PERSON_HALF_SPAN - FURNITURE_GAP_M)
VISIT_ENTRY = (1.2, -3.0)  # outside the camera's view, on the open side of the cell

# A person stops rather than come closer than this to the arm (body column
# to link spheres).
PERSON_ROBOT_GAP_M = 0.05
ARM_SPHERES = [("link3", 0.07), ("link4", 0.07), ("link5", 0.07), ("link6", 0.07),
               ("link7", 0.07), ("tcp_site", 0.08)]


MJCF_PATH = str(Path(__file__).resolve().parent.parent.parent / "common" / "models" / "panda_robot.xml")

# Walk: a straight crossing south of the tray table, clear of it.
WALK_Y = PLACE_TABLE_BOUNDS[1][0] - PERSON_HALF_SPAN - 0.20
WALK_START = (-2.50, WALK_Y)
WALK_END = (2.50, WALK_Y)

# Carton: a floating sphere between pile and tray (not re-validated on the
# current layout, known_issues E5).
STATIC_POSITION = (0.32, -0.35, 0.55)
STATIC_RADIUS = 0.07


class DynamicObstacleNode(Node):
    def __init__(self):
        super().__init__("dynamic_obstacle_node")
        self.declare_parameter("mode", "visit")
        self.declare_parameter("trigger", "launch")
        self.declare_parameter("delay_s", 3.0)
        self.declare_parameter("speed", 1.2)
        self.declare_parameter("passes", 3)
        self.declare_parameter("pause_s", 15.0)
        self.declare_parameter("dwell_s", 6.0)
        self.declare_parameter("duration_s", 0.0)
        self.declare_parameter("static_position", list(STATIC_POSITION))
        self.declare_parameter("static_radius", STATIC_RADIUS)
        p = lambda n: self.get_parameter(n).value
        self.mode = p("mode")
        if self.mode not in ("visit", "walk", "static"):
            raise ValueError(f"mode must be 'visit', 'walk' or 'static', got {self.mode!r}")
        self.trigger = p("trigger")
        self.delay_s = float(p("delay_s"))
        self.speed = float(p("speed"))
        self.passes = int(p("passes"))
        self.pause_s = float(p("pause_s"))
        self.dwell_s = float(p("dwell_s"))
        self.duration_s = float(p("duration_s"))
        self.static_position = np.array(p("static_position"), dtype=float)
        self.static_radius = float(p("static_radius"))

        if self.mode == "visit":
            entry, stand = np.array(VISIT_ENTRY), np.array(VISIT_STAND)
            # One pass = (point, seconds to stay) legs, walked at `speed`.
            self.pass_legs = [(entry, 0.0), (stand, self.dwell_s), (entry, 0.0)]
        else:
            self.pass_legs = [(np.array(WALK_START), 0.0), (np.array(WALK_END), 0.0)]
        self.pass_s = self._legs_duration(self.pass_legs)

        self.pub = self.create_publisher(Float64MultiArray, "/env/dynamic_obstacle", 10)
        self.arm_model = pin.buildModelFromMJCF(MJCF_PATH)
        self.arm_data = self.arm_model.createData()
        self.arm_frames = [(self.arm_model.getFrameId(n), r) for n, r in ARM_SPHERES]
        self.q = None
        self.create_subscription(JointState, "/sim/joint_states", self._on_joint_states, 10)
        self.arm_pose = CELL_POSE  # the arm base in the room, (x, y, yaw), from the base's truth
        self.create_subscription(Float64MultiArray, "/sim/base_truth", self._on_base_truth, 10)
        self.visit_state, self.visit_pos, self.visit_t, self.visits_done = "out", None, 0.0, 0  # out, approach, check, leave
        self.pos_heading = 0.0
        if self.trigger == "first_pick":
            self.create_subscription(String, "/task/action", self._on_task_action, 10)
        self.create_timer(CONTROL_PERIOD_S, self._tick)

        self.start_tick = round(self.delay_s / CONTROL_PERIOD_S) if self.trigger == "launch" else None
        self.tick_count = 0
        self.was_present = False
        self.last_heading = 0.0
        self.get_logger().info(
            f"mode={self.mode} trigger={self.trigger} delay={self.delay_s}s speed={self.speed} m/s"
            + (f", {self.pass_s:.1f} s per pass, {self.passes or 'endless'} passes, {self.pause_s} s apart"
               if self.mode != "static" else ""))

    def _legs_duration(self, legs):
        total = 0.0
        for (a, _), (b, dwell) in zip(legs, legs[1:]):
            total += np.linalg.norm(b - a) / self.speed + dwell
        return total

    def _on_joint_states(self, msg: JointState):
        self.q = np.array(msg.position)

    def _on_base_truth(self, msg: Float64MultiArray):
        x, y, _z, qw, qx, qy, qz = msg.data[1:8]
        yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        self.arm_pose = compose((x, y, yaw), ARM_IN_BASE)

    def _arm_xy(self, xy):
        """A cell-frame point in the arm frame (the base may stand off its parking spot)."""
        room = compose(CELL_POSE, (xy[0], xy[1], 0.0))
        a = self.arm_pose
        c, s = np.cos(a[2]), np.sin(a[2])
        dx, dy = room[0] - a[0], room[1] - a[1]
        return c * dx + s * dy, -s * dx + c * dy

    def _clear_of_arm(self, xy):
        """True if a person at xy keeps PERSON_ROBOT_GAP_M from the arm (body as a
        column, arm as link spheres).
        """
        if self.q is None:
            return True
        pin.forwardKinematics(self.arm_model, self.arm_data, self.q)
        pin.updateFramePlacements(self.arm_model, self.arm_data)
        xy = self._arm_xy(xy)
        for fid, r in self.arm_frames:
            p = self.arm_data.oMf[fid].translation
            dz = max(p[2] - PERSON_TOP, 0.0)  # above their head, the height difference counts
            if np.hypot(np.hypot(p[0] - xy[0], p[1] - xy[1]), dz) - PERSON_HALF_SPAN - r < PERSON_ROBOT_GAP_M:
                return False
        return True

    def _visit_step(self, dt):
        """Advance the visit state machine one tick: (xy, heading), or None while out."""
        entry, stand = np.array(VISIT_ENTRY), np.array(VISIT_STAND)
        self.visit_t += dt
        if self.visit_state == "out":
            if self.passes > 0 and self.visits_done >= self.passes:
                return None
            if self.visits_done > 0 and self.visit_t < self.pause_s:
                return None
            self.visit_state, self.visit_pos, self.visit_t = "approach", entry.copy(), 0.0
        if self.visit_state == "approach":
            d = stand - self.visit_pos
            dist = float(np.linalg.norm(d))
            step = d / dist * min(self.speed * dt, dist) if dist > 1e-9 else d
            self.pos_heading = float(np.arctan2(d[1], d[0])) if dist > 1e-9 else self.pos_heading
            if dist < 1e-6 or not self._clear_of_arm(self.visit_pos + step):
                why = "at the tray" if dist < 1e-6 else "stopped short of the robot arm"
                self.get_logger().info(
                    f"visit: checking the tray ({why}) at ({self.visit_pos[0]:+.2f}, {self.visit_pos[1]:+.2f})")
                self.visit_state, self.visit_t = "check", 0.0
            else:
                self.visit_pos = self.visit_pos + step
        if self.visit_state == "check":
            look = np.array(TRAY_CENTER) - self.visit_pos
            self.pos_heading = float(np.arctan2(look[1], look[0]))
            if self.visit_t >= self.dwell_s:
                self.visit_state = "leave"
        if self.visit_state == "leave":
            d = entry - self.visit_pos
            dist = float(np.linalg.norm(d))
            if dist < 1e-6:
                self.visit_state, self.visit_t = "out", 0.0
                self.visits_done += 1
                return None
            self.pos_heading = float(np.arctan2(d[1], d[0]))
            self.visit_pos = self.visit_pos + d / dist * min(self.speed * dt, dist)
        return self.visit_pos.copy(), self.pos_heading

    def _on_task_action(self, msg: String):
        if self.start_tick is None and msg.data.startswith("pick_at"):
            self.start_tick = self.tick_count + round(self.delay_s / CONTROL_PERIOD_S)

    def _person_at(self, u, reverse):
        """(xy, heading, standing) u seconds into one pass."""
        legs = self.pass_legs[::-1] if reverse else self.pass_legs
        if reverse:  # dwell belongs to the point it is spent at
            legs = [(pt, d) for (pt, _), (_, d) in zip(legs, self.pass_legs[::-1][1:] + [(None, 0.0)])]
        for (a, _), (b, dwell) in zip(legs, legs[1:]):
            walk = np.linalg.norm(b - a) / self.speed
            if u <= walk:
                d = b - a
                return a + d * (u / walk), float(np.arctan2(d[1], d[0])), False
            u -= walk
            if u <= dwell:
                look = np.array(TRAY_CENTER) - b  # standing: face the tray
                return b, float(np.arctan2(look[1], look[0])), True
            u -= dwell
        return None, 0.0, False

    def _state(self):
        """[x, y, z, radius, top_z, kind, heading], or None if absent."""
        t = (self.tick_count - self.start_tick) * CONTROL_PERIOD_S
        if self.duration_s > 0.0 and t >= self.duration_s:
            return None
        if self.mode == "static":
            x, y, z = self.static_position
            r = self.static_radius
            return [x, y, z, r, z + r, 0.0, 0.0]
        if self.mode == "visit":
            step = self._visit_step(CONTROL_PERIOD_S)
            if step is None:
                return None
            xy, heading = step
            return [xy[0], xy[1], ROOM_FLOOR_Z, PERSON_HALF_SPAN, PERSON_TOP, 1.0, heading]
        period = self.pass_s + self.pause_s
        k, u = int(t // period), t % period
        if (self.passes > 0 and k >= self.passes) or u > self.pass_s:
            return None
        # Walk passes alternate direction.
        xy, heading, _ = self._person_at(u, reverse=(self.mode == "walk" and k % 2 == 1))
        if xy is None:
            return None
        return [xy[0], xy[1], ROOM_FLOOR_Z, PERSON_HALF_SPAN, PERSON_TOP, 1.0, heading]

    def _tick(self):
        self.tick_count += 1
        state = None
        if self.start_tick is not None and self.tick_count >= self.start_tick:
            state = self._state()
        present = state is not None
        if present != self.was_present:
            self.was_present = present
            self.get_logger().info(f"{self.mode}: {'entered' if present else 'left'} the scene")
        if state is None:
            x, y, z, r, top, kind, heading = (*IDLE_POSITION, 0.0, 0.0, 0.0, 0.0)
        else:
            # Cell frame to the room: z from the arm's mount height to the floor's.
            x, y, z, r, top, kind, heading = state
            x, y, heading = compose(CELL_POSE, (x, y, heading))
            z, top = z - ROOM_FLOOR_Z, top - ROOM_FLOOR_Z
        t_pub = self.get_clock().now().nanoseconds * 1e-9
        self.pub.publish(Float64MultiArray(
            data=[float(x), float(y), float(z), float(r), t_pub, float(top), float(kind), float(heading)]))


def main():
    rclpy.init()
    node = DynamicObstacleNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
