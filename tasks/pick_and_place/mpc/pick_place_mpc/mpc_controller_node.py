"""The receding-horizon MPC loop: on each plant state, set the goals and
obstacles, run one SQP_RTI solve and command the plant's torques directly.
There is no separate replanning mode; obstacles are online parameters.

Why the loop commands the plant itself, the lockstep and the settle logic:
docs/design_notes.md ("Real-time risk", "Lockstep plant").
"""
import time
from pathlib import Path

import numpy as np
import pinocchio as pin
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, Bool, Empty

from core.dynamics import load_manipulator, forward_kinematics
from core.collision import SphereProxy, NO_OBSTACLE_POSITION, NO_OBSTACLE_RADIUS
from core.ocp import MPCConfig, build_ocp
from pick_place_common.telemetry import Telemetry

# Same robot file as the plant, parsed by Pinocchio (a separate model of the
# same robot). The meshdir placeholder does not matter for kinematics.
MJCF_PATH = str(Path(__file__).resolve().parent.parent.parent / "common" / "models" / "panda_robot.xml")

# Collision proxies: first-pass radii, meant to over-approximate each link.
# task_node.PROXY_SPHERES must match.
PROXY_FRAMES = [
    SphereProxy(frame_name="link3", local_offset=(0, 0, 0), radius=0.10),
    SphereProxy(frame_name="link5", local_offset=(0, 0, 0), radius=0.09),
    SphereProxy(frame_name="link7", local_offset=(0, 0, 0), radius=0.09),
    SphereProxy(frame_name="attachment", local_offset=(0, 0, 0), radius=0.08),
]

BUDGET_MISS_TIMEOUT_S = 0.05  # equal to the plant's lockstep wait: later, the plant has moved on
DECEL_RAMP_STEPS = 25  # ticks to ramp torque to zero after a failed or late solve

# scripts/_run_scenario.sh waits for this exact line; keep them in sync.
READY_LOG_MESSAGE = "mpc_controller ready; waiting for /mpc/go to start"

# "Settled" = arrived (a one-shot /mpc/settled pulse); the MPC keeps solving.
SETTLE_DIST_TOL_M = 0.005
SETTLE_QDOT_TOL = 0.05  # rad/s
SETTLE_DWELL_TICKS = 25  # 0.5 s within tolerance
GOAL_CHANGE_TOL_M = 0.001  # smaller changes are the same goal
PAYLOAD_FILTER = 0.1  # per tick; ~0.2 s to follow a grasp or a release
GOAL_BIAS_GAIN = 1.5  # 1/s, on the TCP error while at rest near a still goal
GOAL_BIAS_ACTIVE_M = 0.01
GOAL_BIAS_ACTIVE_QDOT = 0.1  # rad/s
GOAL_BIAS_MAX_M = 0.02


class MPCControllerNode(Node):
    def __init__(self):
        super().__init__("mpc_controller")

        self.cfg = MPCConfig()
        model = load_manipulator(MJCF_PATH, [p.frame_name for p in PROXY_FRAMES])
        self.nq, self.nv = model.nq, model.nv
        self.fk_ee = forward_kinematics(model, self.cfg.ee_frame)
        self.pin_model = model.model
        self.pin_data = model.model.createData()

        self.solver = build_ocp(model, PROXY_FRAMES, self.cfg)
        # Joint 1's posture reference is set each tick to the goal's azimuth (the base
        # faces where it is going), through yref. y layout as in core/ocp.py.
        self.nu = model.nv
        self.q_center = (model.model.lowerPositionLimit + model.model.upperPositionLimit) / 2
        self.yref = np.zeros(self.nu + model.nv + model.nq + 9)
        self.base_posture_idx = self.nu + model.nv  # joint 1 in the q - q_center block
        self.last_base_ref = None

        self.latest_q = np.zeros(self.nq)
        self.latest_qdot = np.zeros(self.nv)
        self.have_state = False
        # Plant step of latest_q, echoed on the command.
        self.latest_state_stamp = ""
        self.latest_state_wall = None
        self.telemetry = Telemetry(
            "mpc", ["status", "solve_ms", "state_age_ms", "state_step", "settled",
                    "goal_x", "goal_y", "goal_z", "ee_err",
                    *[f"orient{i}" for i in range(7)],
                    *[f"obs{i}" for i in range(4 * self.cfg.n_obstacles)]])

        self.obstacle_params = np.concatenate([
            np.tile(np.array([*NO_OBSTACLE_POSITION, NO_OBSTACLE_RADIUS]), self.cfg.n_obstacles),
        ])
        self.goal = np.zeros(3)
        self.goal_buf = {}
        self.orient_buf = {}
        # Per-stage goals (row k for stage k; the last row repeats if short).
        self.goal_horizon = np.zeros((1, 3))
        self.orient_horizon = None
        # No solve before the first goal (the zero placeholder is the robot base).
        self.have_goal = False
        # Re-initialise the solver guess to "stay here, holding gravity" when solving
        # (re)starts; acados' zero guess violates joint 4's range.
        self.guess_stale = True
        # [axis1 (3), axis2 (3), scale]; scale 0 keeps the orientation terms off.
        self.orientation_goal = np.array([0.0, 0.0, -1.0, 0.0, -1.0, 0.0, 0.0])

        # A failed or late solve ramps the last good torque to zero; replaying a
        # command from a stale state is worst just after an obstacle appears.
        self.last_good_u0 = None
        self.decel_step = 0

        self.settled = False
        self.settled_goal = None
        self.settle_ticks = 0

        # Nothing is commanded until /mpc/go; the plant holds the arm meanwhile.
        self.enabled = False

        # Depth 1: after a hiccup, solve for the newest state only. Each state
        # triggers a solve (_on_joint_state); there is no timer.
        self.create_subscription(JointState, "/sim/joint_states", self._on_joint_state, 1)
        self.create_subscription(Float64MultiArray, "/mpc/obstacle_params", self._on_obstacles, 10)
        self.create_subscription(Float64MultiArray, "/mpc/goal", self._on_goal, 10)
        self.create_subscription(Float64MultiArray, "/mpc/orientation_goal", self._on_orientation_goal, 10)
        self.create_subscription(Empty, "/mpc/go", self._on_go, 10)
        # Payload in the model: mass from the wrist load cell (0 with an empty hand).
        self.payload_mass = 0.0
        self.goal_bias = np.zeros(3)
        self.bias_anchor = None
        self.create_subscription(Float64MultiArray, "/sim/wrist_force", self._on_wrist_force, 10)

        self.cmd_pub = self.create_publisher(JointState, "/sim/joint_command", 10)
        self.diag_pub = self.create_publisher(Float64MultiArray, "/mpc/solve_diagnostics", 10)
        # One pulse per arrival, not a state: task_node advances on each message.
        self.settled_pub = self.create_publisher(Bool, "/mpc/settled", 10)

        self.get_logger().info(READY_LOG_MESSAGE)

    def _on_go(self, _msg: Empty):
        if not self.enabled:
            self.enabled = True
            self.get_logger().info("received /mpc/go; starting operation")

    def _on_joint_state(self, msg: JointState):
        self.latest_q = np.array(msg.position[: self.nq])
        self.latest_qdot = np.array(msg.velocity[: self.nv])
        self.latest_state_stamp = msg.header.frame_id
        self.latest_state_wall = time.perf_counter()
        self.have_state = True
        self._control_step()

    def _on_obstacles(self, msg: Float64MultiArray):
        self.obstacle_params = np.array(msg.data)  # [x, y, z, r] per slot, from task_node

    # Goals may end with the plant step they are for. Each solve uses the goal for
    # its state's step, or extrapolates from the two newest for up to
    # GOAL_EXTRAPOLATE_STEPS (a goal standing still, then jumping, was a kick).
    GOAL_BUFFER_STEPS = 50
    GOAL_EXTRAPOLATE_STEPS = 3

    @classmethod
    def _goal_for(cls, buf, step):
        if step is None or step in buf:
            return buf.get(step, buf["latest"])
        stamped = sorted(k for k in buf if k != "latest" and k < step)
        if len(stamped) < 2 or step - stamped[-1] > cls.GOAL_EXTRAPOLATE_STEPS:
            return buf["latest"]
        a, b = stamped[-2], stamped[-1]
        return buf[b] + (buf[b] - buf[a]) * (step - b) / (b - a)

    @staticmethod
    def _stash(buf, data, n):
        """Store a goal message: one n-vector per MPC stage, optionally followed by
        the step stamp.
        """
        step = int(data[-1]) if len(data) % n == 1 else None
        m = len(data) - (len(data) % n)
        value = np.array(data[:m]).reshape(-1, n)
        buf["latest"] = value
        if step is not None:
            buf[step] = value
            for old in [k for k in buf if k != "latest" and k < step - MPCControllerNode.GOAL_BUFFER_STEPS]:
                del buf[old]

    def _on_wrist_force(self, msg: Float64MultiArray):
        mass = max(0.0, float(msg.data[2])) / 9.81
        self.payload_mass += PAYLOAD_FILTER * (mass - self.payload_mass)

    def _on_goal(self, msg: Float64MultiArray):
        self._stash(self.goal_buf, msg.data, 3)
        self.have_goal = True

    def _on_orientation_goal(self, msg: Float64MultiArray):
        self._stash(self.orient_buf, msg.data, 7)

    def _select_goals(self):
        step = int(self.latest_state_stamp) if self.latest_state_stamp else None
        self.goal_horizon = self._goal_for(self.goal_buf, step)
        new_goal = self.goal_horizon[0]
        # Against the settled goal too: a reference creeping on after a hold moves < 1 mm per tick.
        ref_goal = self.settled_goal if self.settled else self.goal
        if np.linalg.norm(new_goal - ref_goal) > GOAL_CHANGE_TOL_M:
            # New goal: re-arm settle detection.
            self.settled = False
            self.settle_ticks = 0
        self.goal = new_goal
        if "latest" in self.orient_buf:
            self.orient_horizon = self._goal_for(self.orient_buf, step)
            self.orientation_goal = self.orient_horizon[0]

    def _update_goal_bias(self):
        """Offset-free tracking: integrate the remaining TCP error into a bias on the
        goal given to the OCP, only while the whole reference stands still, the arm
        is near it and nearly at rest (never against a moving or blocked reference),
        clamped. Reset when the goal moves on. Settling is still judged on the goal.
        """
        if self.bias_anchor is None or np.linalg.norm(self.goal - self.bias_anchor) > GOAL_CHANGE_TOL_M:
            self.goal_bias = np.zeros(3)
            self.bias_anchor = self.goal.copy()
            return
        err = self.goal - np.array(self.fk_ee(self.latest_q)).flatten()
        still = float(np.ptp(np.asarray(self.goal_horizon), axis=0).max()) < 1e-6
        if (still and np.linalg.norm(err) < GOAL_BIAS_ACTIVE_M
                and np.linalg.norm(self.latest_qdot) < GOAL_BIAS_ACTIVE_QDOT):
            self.goal_bias = self.goal_bias + GOAL_BIAS_GAIN * err * self.cfg.dt
            norm = float(np.linalg.norm(self.goal_bias))
            if norm > GOAL_BIAS_MAX_M:
                self.goal_bias *= GOAL_BIAS_MAX_M / norm

    def _control_step(self):
        if not self.enabled:
            return

        if not self.have_state or not self.have_goal:
            return

        self._select_goals()
        x0 = np.concatenate([self.latest_q, self.latest_qdot])
        if self.guess_stale:
            self._reset_guess(x0)
        self._update_goal_bias()

        self.solver.set(0, "lbx", x0)
        self.solver.set(0, "ubx", x0)
        # Stage k gets the goal and heading for its own moment. The base reference is
        # skipped near the base axis, where the azimuth is meaningless.
        if self.last_base_ref is None:
            self.last_base_ref = np.full(self.cfg.N, np.nan)
        for k in range(self.cfg.N + 1):
            g = self.goal_horizon[min(k, len(self.goal_horizon) - 1)] + self.goal_bias
            o = (self.orient_horizon[min(k, len(self.orient_horizon) - 1)]
                 if self.orient_horizon is not None else self.orientation_goal)
            # Same order as core/ocp.py's acados_model.p.
            self.solver.set(k, "p", np.concatenate([self.obstacle_params, g, o, [self.payload_mass]]))
            if k < self.cfg.N and np.hypot(g[0], g[1]) > 0.15:
                base_ref = float(np.arctan2(g[1], g[0]))
                if not abs(base_ref - self.last_base_ref[k]) <= 1e-3:
                    self.yref[self.base_posture_idx] = base_ref - self.q_center[0]
                    self.solver.cost_set(k, "yref", self.yref)
                    self.last_base_ref[k] = base_ref

        t0 = time.perf_counter()
        status = self.solver.solve()
        solve_ms = (time.perf_counter() - t0) * 1e3

        diag = Float64MultiArray()
        diag.data = [float(status), solve_ms]
        self.diag_pub.publish(diag)
        if self.telemetry.enabled:
            ee = np.array(self.fk_ee(self.latest_q)).flatten()
            self.telemetry.row(
                status, round(solve_ms, 2), round((t0 - self.latest_state_wall) * 1e3, 2),
                self.latest_state_stamp, int(self.settled), *np.round(self.goal, 4),
                round(float(np.linalg.norm(ee - self.goal)), 4),
                *np.round(self.orientation_goal, 5), *np.round(self.obstacle_params, 4))

        on_time = solve_ms * 1e-3 <= BUDGET_MISS_TIMEOUT_S
        if status == 0 and on_time:
            u = self.solver.get(0, "u")
            self.last_good_u0 = u
            self.decel_step = 0

            ee_pos = np.array(self.fk_ee(self.latest_q)).flatten()
            at_goal = np.linalg.norm(ee_pos - self.goal) < SETTLE_DIST_TOL_M
            at_rest = np.linalg.norm(self.latest_qdot) < SETTLE_QDOT_TOL
            self.settle_ticks = self.settle_ticks + 1 if (at_goal and at_rest) else 0
            if self.settle_ticks >= SETTLE_DWELL_TICKS and not self.settled:
                self.settled = True
                self.settled_goal = self.goal.copy()
                self.get_logger().info("settled at goal")
                self.settled_pub.publish(Bool(data=True))
        else:
            self.guess_stale = True
            if status != 0:
                self.get_logger().warn(f"acados solve failed (status={status})")
            else:
                self.get_logger().warn(f"acados solve missed budget ({solve_ms:.1f}ms)")
            u = self._decel_command()
            if u is None:
                return

        cmd = JointState()
        cmd.header.frame_id = self.latest_state_stamp
        cmd.effort = u.tolist()
        self.cmd_pub.publish(cmd)

    def _reset_guess(self, x0):
        """Whole-horizon guess: stay at x0, holding against gravity."""
        u_hold = pin.computeGeneralizedGravity(self.pin_model, self.pin_data, x0[: self.nq])
        for k in range(self.cfg.N + 1):
            self.solver.set(k, "x", x0)
        for k in range(self.cfg.N):
            self.solver.set(k, "u", u_hold)
        self.guess_stale = False

    def _decel_command(self):
        if self.last_good_u0 is None or self.decel_step >= DECEL_RAMP_STEPS:
            # Ramp finished (or nothing to ramp): stop commanding. Zero torque would drop
            # the arm; a silent controller makes the plant hold instead.
            self.last_good_u0 = None
            return None
        self.decel_step += 1
        scale = max(0.0, 1.0 - self.decel_step / DECEL_RAMP_STEPS)
        return self.last_good_u0 * scale


def main():
    rclpy.init()
    node = MPCControllerNode()
    try:
        rclpy.spin(node)
    finally:
        node.telemetry.flush()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
