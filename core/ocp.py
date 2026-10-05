"""Builds the acados OCP for the MPC: torque control of the arm, per-stage
goal tracking, soft terminal goal and soft collision margins, SQP_RTI.
mpc_controller_node runs the receding-horizon loop.

See docs/system_overview.md (section 5) and
docs/implementation_notes.md#ocppy.
"""
import hashlib
import json as pyjson
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
import casadi as ca

from core.dynamics import ManipulatorModel, forward_kinematics, forward_kinematics_axis
from core.collision import (
    SphereProxy, proxy_world_center, collision_margin_expr,
    NO_OBSTACLE_POSITION, NO_OBSTACLE_RADIUS,
)


@dataclass
class MPCConfig:
    N: int = 15  # horizon length (steps)
    dt: float = 0.02  # shooting interval [s]
    n_obstacles: int = 3  # obstacle slots (fixed at code generation)
    ee_frame: str = "tcp_site"  # 0.10 m past the flange: where gripper fingers would reach
    goal_tolerance: float = 0.0  # 0 = exact terminal equality (soft)
    safety_margin: float = 0.03  # [m] clearance on every collision constraint
    w_u: float = 1e-3  # torque
    w_qdot: float = 1e-2  # joint speed
    w_terminal_qdot: float = 1e-1  # arrive at rest
    w_ee_track: float = 5000.0  # per-stage pull to the goal; without it RTI stalls short of it
    w_q_center: float = 5.0  # posture: joints toward mid-range (redundancy)
    w_orient: float = 2000.0  # both tool-axis terms, gated by orient_scale
    slack_weight_l1: float = 1e3  # collision slack, linear
    slack_weight_l2: float = 1e4  # collision slack, quadratic
    # Terminal goal is soft so a goal beyond one horizon is not infeasible; heavier
    # than the collision slack so it still wins once in range.
    terminal_slack_weight_l1: float = 1e4
    terminal_slack_weight_l2: float = 1e5
    # [Nm] Panda torque limits; match the <motor> ctrlrange in panda_robot.xml.
    tau_max: np.ndarray = field(default_factory=lambda: np.array([87, 87, 87, 87, 12, 12, 12]))
    # [rad/s] Panda velocity limits (Franka datasheet).
    qdot_max: np.ndarray = field(default_factory=lambda: np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61]))


def _structural_fingerprint(m: ManipulatorModel, proxies: list[SphereProxy], cfg: MPCConfig) -> str:
    """Hash of everything that changes the generated C code: dimensions, the
    expressions (this file's source), safety_margin and proxy radii (compiled in
    as constants). Weights and bounds are left out; _apply_runtime_values sets
    them, so changing one never forces a rebuild.
    """
    this_file_hash = hashlib.sha256(
        Path(__file__).read_bytes() + (Path(__file__).parent / "dynamics.py").read_bytes()).hexdigest()[:16]
    payload = {
        "source": this_file_hash,
        "model": m.name,
        "nq": m.nq,
        "nu": int(m.u.shape[0]),
        "N": cfg.N,
        "dt": cfg.dt,
        "n_obstacles": cfg.n_obstacles,
        "ee_frame": cfg.ee_frame,
        "safety_margin": cfg.safety_margin,
        "proxies": [(p.frame_name, p.radius, tuple(p.local_offset)) for p in proxies],
    }
    return hashlib.sha256(pyjson.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _apply_runtime_values(
    solver: AcadosOcpSolver, cfg: MPCConfig, n_h: int, nu: int, nv: int, pos_limits
) -> None:
    """Push MPCConfig's weights and bounds into a built solver, fresh or cached."""
    W = np.diag(
        [cfg.w_u] * nu + [cfg.w_qdot] * nv + [cfg.w_q_center] * nv
        + [cfg.w_ee_track] * 3 + [cfg.w_orient] * 3 + [cfg.w_orient] * 3
    )
    W_e = np.diag([cfg.w_terminal_qdot] * nv)
    zl = cfg.slack_weight_l1 * np.ones(n_h)
    Zl = cfg.slack_weight_l2 * np.ones(n_h)
    zl_e = cfg.terminal_slack_weight_l1 * np.ones(3)
    Zl_e = cfg.terminal_slack_weight_l2 * np.ones(3)
    lbu = -cfg.tau_max * np.ones(nu)
    ubu = cfg.tau_max * np.ones(nu)
    tol_e = cfg.goal_tolerance * np.ones(3)
    lower_pos, upper_pos = pos_limits
    lbx = np.concatenate([lower_pos, -cfg.qdot_max])
    ubx = np.concatenate([upper_pos, cfg.qdot_max])

    # Stage 0 is overwritten by the x0 equality every solve anyway.
    for k in range(cfg.N + 1):
        solver.constraints_set(k, "lbx", lbx)
        solver.constraints_set(k, "ubx", ubx)
    for k in range(cfg.N):
        solver.cost_set(k, "W", W)
        solver.constraints_set(k, "lbu", lbu)
        solver.constraints_set(k, "ubu", ubu)
    # Stage 0 has no slack (x0 is fixed there), so only stages 1..N-1.
    for k in range(1, cfg.N):
        solver.cost_set(k, "zl", zl); solver.cost_set(k, "zu", zl)
        solver.cost_set(k, "Zl", Zl); solver.cost_set(k, "Zu", Zl)
    solver.cost_set(cfg.N, "W", W_e)
    solver.cost_set(cfg.N, "zl", zl_e); solver.cost_set(cfg.N, "zu", zl_e)
    solver.cost_set(cfg.N, "Zl", Zl_e); solver.cost_set(cfg.N, "Zu", Zl_e)
    solver.constraints_set(cfg.N, "lh", -tol_e)
    solver.constraints_set(cfg.N, "uh", tol_e)


def build_ocp(m: ManipulatorModel, proxies: list[SphereProxy], cfg: MPCConfig) -> AcadosOcpSolver:
    ocp = AcadosOcp()

    nx, nu = m.x.shape[0], m.u.shape[0]

    acados_model = AcadosModel()
    acados_model.name = f"{m.name}_mpc"
    acados_model.x = m.x
    acados_model.u = m.u

    fk_ee = forward_kinematics(m, cfg.ee_frame)
    # Tool +z (down) and +x (heading): together they fix the gripper's orientation.
    fk_ee_axis = forward_kinematics_axis(m, cfg.ee_frame, local_axis=(0.0, 0.0, 1.0))
    fk_ee_axis2 = forward_kinematics_axis(m, cfg.ee_frame, local_axis=(1.0, 0.0, 0.0))
    proxy_fks = {p.frame_name: forward_kinematics(m, p.frame_name) for p in proxies}

    # Online parameters, in this order (mpc_controller_node builds p the same way):
    # [x, y, z, r] per obstacle slot, goal, two orientation targets, orient_scale,
    # payload mass.
    p_obs = ca.SX.sym("p_obs", 4 * cfg.n_obstacles)
    goal_param = ca.SX.sym("goal_p", 3)
    orient_axis_target = ca.SX.sym("orient_axis_target", 3)
    orient_axis2_target = ca.SX.sym("orient_axis2_target", 3)
    orient_scale = ca.SX.sym("orient_scale", 1)
    acados_model.p = ca.vertcat(
        p_obs, goal_param, orient_axis_target, orient_axis2_target, orient_scale, m.payload_mass
    )

    # IRK needs the implicit form xdot - f(x, u) = 0; without it the integrator
    # returns NaN from the first solve.
    xdot_sym = ca.SX.sym("xdot", nx)
    acados_model.xdot = xdot_sym
    acados_model.f_expl_expr = m.xdot
    acados_model.f_impl_expr = xdot_sym - m.xdot
    q = m.x[: m.nq]
    qdot = m.x[m.nq:]

    h_expr = []
    for proxy in proxies:
        p_center = proxy_world_center(proxy_fks[proxy.frame_name], q, proxy.local_offset)
        for k in range(cfg.n_obstacles):
            p_ob = p_obs[4 * k: 4 * k + 3]
            r_ob = p_obs[4 * k + 3]
            h_expr.append(collision_margin_expr(p_center, p_ob, proxy.radius, r_ob,
                                                  cfg.safety_margin))
    acados_model.con_h_expr = ca.vertcat(*h_expr)
    n_h = len(h_expr)

    ee_pos = fk_ee(q)
    acados_model.con_h_expr_e = ee_pos - goal_param

    ocp.model = acados_model

    # Mid-range of the joint limits parsed from the MJCF.
    q_center = (m.model.lowerPositionLimit + m.model.upperPositionLimit) / 2

    ocp.cost.cost_type = "NONLINEAR_LS"
    # y = [tau, qdot, q - q_center, ee_pos - goal, orient_scale * (z_tool - z*),
    # orient_scale * (x_tool - x*)]; orient_scale = 0 makes the orientation terms
    # inert. mpc_controller_node's yref layout must match.
    ee_axis = fk_ee_axis(q)
    ee_axis2 = fk_ee_axis2(q)
    ocp.model.cost_y_expr = ca.vertcat(
        m.u, qdot, q - q_center, ee_pos - goal_param,
        orient_scale * (ee_axis - orient_axis_target),
        orient_scale * (ee_axis2 - orient_axis2_target),
    )
    ocp.cost.yref = np.zeros(nu + m.nv + m.nq + 3 + 3 + 3)
    ocp.cost.W = np.diag(
        [cfg.w_u] * nu + [cfg.w_qdot] * m.nv + [cfg.w_q_center] * m.nq
        + [cfg.w_ee_track] * 3 + [cfg.w_orient] * 3 + [cfg.w_orient] * 3
    )

    ocp.cost.cost_type_e = "NONLINEAR_LS"
    ocp.model.cost_y_expr_e = qdot
    ocp.cost.yref_e = np.zeros(m.nv)
    ocp.cost.W_e = np.diag([cfg.w_terminal_qdot] * m.nv)

    ocp.constraints.x0 = np.zeros(nx)  # overwritten every solve with the measured state

    ocp.constraints.lbu = -cfg.tau_max * np.ones(nu)
    ocp.constraints.ubu = cfg.tau_max * np.ones(nu)
    ocp.constraints.idxbu = np.arange(nu)

    # Hard joint position (from the MJCF) and velocity limits. Stage 0 is pinned
    # to the measured state instead, so a state at a limit is not infeasible.
    idxbx = np.concatenate([np.arange(m.nq), m.nq + np.arange(m.nv)])
    lbx = np.concatenate([m.model.lowerPositionLimit, -cfg.qdot_max])
    ubx = np.concatenate([m.model.upperPositionLimit, cfg.qdot_max])
    ocp.constraints.idxbx = idxbx
    ocp.constraints.lbx = lbx
    ocp.constraints.ubx = ubx
    ocp.constraints.idxbx_e = idxbx
    ocp.constraints.lbx_e = lbx
    ocp.constraints.ubx_e = ubx

    ocp.constraints.lh = np.zeros(n_h)  # margin >= 0
    ocp.constraints.uh = 1e6 * np.ones(n_h)  # one-sided
    ocp.constraints.lh_e = -cfg.goal_tolerance * np.ones(3)
    ocp.constraints.uh_e = cfg.goal_tolerance * np.ones(3)

    # Collision margins and the terminal goal are soft, so the QP is never
    # infeasible.
    ocp.constraints.idxsh = np.arange(n_h)
    ocp.cost.zl = cfg.slack_weight_l1 * np.ones(n_h)
    ocp.cost.zu = cfg.slack_weight_l1 * np.ones(n_h)
    ocp.cost.Zl = cfg.slack_weight_l2 * np.ones(n_h)
    ocp.cost.Zu = cfg.slack_weight_l2 * np.ones(n_h)

    ocp.constraints.idxsh_e = np.arange(3)
    ocp.cost.zl_e = cfg.terminal_slack_weight_l1 * np.ones(3)
    ocp.cost.zu_e = cfg.terminal_slack_weight_l1 * np.ones(3)
    ocp.cost.Zl_e = cfg.terminal_slack_weight_l2 * np.ones(3)
    ocp.cost.Zu_e = cfg.terminal_slack_weight_l2 * np.ones(3)

    ocp.solver_options.N_horizon = cfg.N
    ocp.solver_options.tf = cfg.N * cfg.dt
    ocp.solver_options.integrator_type = "IRK"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.qp_solver_cond_N = cfg.N

    ocp.parameter_values = np.concatenate([
        np.tile(np.array([*NO_OBSTACLE_POSITION, NO_OBSTACLE_RADIUS]), cfg.n_obstacles),
        np.zeros(3),  # goal, set before the first solve
        np.array([0.0, 0.0, -1.0]),  # orient_axis_target
        np.array([0.0, -1.0, 0.0]),  # orient_axis2_target
        np.array([0.0]),  # orient_scale = 0 (inert)
        np.array([0.0]),  # payload mass
    ])

    # Skip code generation and compilation (a minute or two) when the structure
    # is unchanged since the last build.
    json_file = f"{acados_model.name}_ocp.json"
    so_file = f"c_generated_code/libacados_ocp_solver_{acados_model.name}.so"
    fingerprint_file = Path(f"{acados_model.name}_ocp.fingerprint")
    fingerprint = _structural_fingerprint(m, proxies, cfg)
    cache_hit = (
        fingerprint_file.exists()
        and fingerprint_file.read_text().strip() == fingerprint
        and Path(json_file).exists()
        and Path(so_file).exists()
    )

    solver = AcadosOcpSolver(ocp, json_file=json_file, build=not cache_hit, generate=not cache_hit)
    if not cache_hit:
        fingerprint_file.write_text(fingerprint)

    _apply_runtime_values(
        solver, cfg, n_h, nu, m.nv, (m.model.lowerPositionLimit, m.model.upperPositionLimit)
    )
    return solver
