"""Whole-body MPC (acados): the mobile base (unicycle: pose, v, omega; inputs the two
accelerations) and the arm (torque dynamics as core/ocp.py, on a fixed base) in one
OCP, in a station frame W (the arm frame of the nominal dock). The tool's goal per
stage is in W or in the arm's own frame (a selector: a scan pose stays put in the
room while the base drives; the carry pose moves with it); the base tracks a
reference along its docking line (lag and lateral error apart, as nav/base_mpc.py).
Obstacle spheres (the stations' hulls) and people (vertical cylinders) are in W and
apply to every proxy. SQP_RTI.

See docs/implementation_notes.md#wb_ocppy.
"""
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import casadi as ca
import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

from core.collision import NO_OBSTACLE_POSITION, NO_OBSTACLE_RADIUS
from core.dynamics import ManipulatorModel, forward_kinematics, forward_kinematics_axis
from core.ocp import MPCConfig

NO_PERSON = (1e3, 1e3, 0.0)
STRUCTURE_VERSION = 1


@dataclass
class WholeBodyConfig:
    arm: MPCConfig = field(default_factory=MPCConfig)
    arm_in_base: tuple = (0.20, 0.0, np.pi / 2)  # link0 in base_link (x, y, yaw)
    n_people: int = 2
    person_margin: float = 0.10
    v_max: float = 0.20
    v_min: float = -0.20
    w_max: float = 0.5
    a_max: float = 0.5
    alpha_max: float = 1.0
    track: float = 0.50
    rim_speed_max: float = 0.8
    w_lag: float = 2000.0
    w_lat: float = 20000.0
    w_heading: float = 2000.0
    w_v: float = 200.0
    w_w: float = 100.0
    w_a: float = 1.0
    w_alpha: float = 1.0
    w_base_e: float = 3.0  # terminal base weights, times the stage's


def _fingerprint(m, proxies, cfg):
    src = (Path(__file__).read_bytes() + (Path(__file__).parent / "dynamics.py").read_bytes())
    payload = {"src": hashlib.sha256(src).hexdigest()[:16], "v": STRUCTURE_VERSION, "model": m.name,
               "N": cfg.arm.N, "dt": cfg.arm.dt, "n_obstacles": cfg.arm.n_obstacles, "ee": cfg.arm.ee_frame,
               "margin": cfg.arm.safety_margin, "n_people": cfg.n_people, "person_margin": cfg.person_margin,
               "arm_in_base": list(cfg.arm_in_base),
               "proxies": [(p.frame_name, p.radius, tuple(p.local_offset)) for p in proxies]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


class WholeBodyMPC:
    """x = [x_b, y_b, theta_b, v, omega, q (7), qdot (7)] (base_link in W), u = [a, alpha,
    tau (7)]."""

    def __init__(self, m: ManipulatorModel, proxies, cfg=None, build_dir="."):
        self.cfg = c = cfg or WholeBodyConfig()
        a = c.arm
        nq = m.nq
        xb = ca.SX.sym("xb", 5)
        ub = ca.SX.sym("ub", 2)
        px, py, th, v, w = ca.vertsplit(xb)
        q, qdot = m.x[:nq], m.x[nq:]
        x = ca.vertcat(xb, m.x)
        u = ca.vertcat(ub, m.u)

        p_obs = ca.SX.sym("p_obs", 4 * a.n_obstacles)
        goal = ca.SX.sym("goal", 3)
        ax1 = ca.SX.sym("ax1", 3)
        ax2 = ca.SX.sym("ax2", 3)
        oscale = ca.SX.sym("oscale", 1)
        sel = ca.SX.sym("sel", 1)  # 1: goal in W, 0: in the arm frame
        p_people = ca.SX.sym("p_people", 3 * c.n_people)
        route = ca.SX.sym("route", 2)  # the docking line's cos, sin
        params = ca.vertcat(p_obs, goal, ax1, ax2, oscale, m.payload_mass, sel, p_people, route)

        # The arm frame in W.
        lx, ly, lyaw = c.arm_in_base
        ox = px + ca.cos(th) * lx - ca.sin(th) * ly
        oy = py + ca.sin(th) * lx + ca.cos(th) * ly
        psi = th + lyaw
        cp, sp = ca.cos(psi), ca.sin(psi)

        def to_w(pt):
            return ca.vertcat(ox + cp * pt[0] - sp * pt[1], oy + sp * pt[0] + cp * pt[1], pt[2])

        def rot_w(vec):
            return ca.vertcat(cp * vec[0] - sp * vec[1], sp * vec[0] + cp * vec[1], vec[2])

        fk_ee = forward_kinematics(m, a.ee_frame)
        fk_z = forward_kinematics_axis(m, a.ee_frame, local_axis=(0.0, 0.0, 1.0))
        fk_x = forward_kinematics_axis(m, a.ee_frame, local_axis=(1.0, 0.0, 0.0))
        ee = fk_ee(q)
        ee_res = sel * (to_w(ee) - goal) + (1 - sel) * (ee - goal)
        z_res = sel * (rot_w(fk_z(q)) - ax1) + (1 - sel) * (fk_z(q) - ax1)
        x_res = sel * (rot_w(fk_x(q)) - ax2) + (1 - sel) * (fk_x(q) - ax2)

        h = []
        for pr in proxies:
            centre = to_w(forward_kinematics(m, pr.frame_name)(q) + ca.SX(pr.local_offset))
            for k in range(a.n_obstacles):
                o = p_obs[4 * k:4 * k + 3]
                h.append(ca.norm_2(centre - o) - pr.radius - p_obs[4 * k + 3] - a.safety_margin)
            for j in range(c.n_people):
                d_xy = ca.sqrt((centre[0] - p_people[3 * j]) ** 2 + (centre[1] - p_people[3 * j + 1]) ** 2 + 1e-9)
                h.append(d_xy - pr.radius - p_people[3 * j + 2] - c.person_margin)
        self.n_h = len(h)

        cr, sr = route[0], route[1]
        along, across = cr * px + sr * py, -sr * px + cr * py
        q_center = (m.model.lowerPositionLimit + m.model.upperPositionLimit) / 2
        self.q_center = q_center
        y = ca.vertcat(m.u, qdot, q - q_center, ee_res, oscale * z_res, oscale * x_res,
                       along, across, ca.cos(th), ca.sin(th), v, w, ub)
        y_e = ca.vertcat(qdot, along, across, ca.cos(th), ca.sin(th), v, w)

        model = AcadosModel()
        model.name = "wb_mpc"
        model.x, model.u, model.p = x, u, params
        xdot = ca.SX.sym("xdot", x.shape[0])
        f = ca.vertcat(v * ca.cos(th), v * ca.sin(th), w, ub[0], ub[1], m.xdot)
        model.xdot = xdot
        model.f_expl_expr = f
        model.f_impl_expr = xdot - f
        model.con_h_expr = ca.vertcat(*h)
        model.con_h_expr_e = ee_res
        model.cost_y_expr = y
        model.cost_y_expr_e = y_e

        ocp = AcadosOcp()
        ocp.model = model
        self.ny, self.ny_e = y.shape[0], y_e.shape[0]
        ocp.cost.cost_type = ocp.cost.cost_type_e = "NONLINEAR_LS"
        ocp.cost.W = self._W()
        ocp.cost.W_e = self._W_e()
        ocp.cost.yref = np.zeros(self.ny)
        ocp.cost.yref_e = np.zeros(self.ny_e)

        nx = x.shape[0]
        ocp.constraints.x0 = np.zeros(nx)
        tau = np.asarray(a.tau_max, dtype=float)
        ocp.constraints.idxbu = np.arange(2 + nq)
        ocp.constraints.lbu = np.r_[-c.a_max, -c.alpha_max, -tau]
        ocp.constraints.ubu = np.r_[c.a_max, c.alpha_max, tau]
        self.lbx = np.r_[c.v_min, -c.w_max, m.model.lowerPositionLimit, -a.qdot_max]
        self.ubx = np.r_[c.v_max, c.w_max, m.model.upperPositionLimit, a.qdot_max]
        idx = np.r_[3, 4, 5 + np.arange(2 * nq)]
        ocp.constraints.idxbx = ocp.constraints.idxbx_e = idx
        ocp.constraints.lbx = ocp.constraints.lbx_e = self.lbx
        ocp.constraints.ubx = ocp.constraints.ubx_e = self.ubx
        half = c.track / 2
        C = np.zeros((2, nx))
        C[0, 3], C[0, 4], C[1, 3], C[1, 4] = 1.0, half, 1.0, -half
        ocp.constraints.C = C
        ocp.constraints.D = np.zeros((2, 2 + nq))
        ocp.constraints.lg = -c.rim_speed_max * np.ones(2)
        ocp.constraints.ug = c.rim_speed_max * np.ones(2)
        ocp.constraints.lh = np.zeros(self.n_h)
        ocp.constraints.uh = 1e6 * np.ones(self.n_h)
        ocp.constraints.idxsh = np.arange(self.n_h)
        ocp.cost.zl = ocp.cost.zu = a.slack_weight_l1 * np.ones(self.n_h)
        ocp.cost.Zl = ocp.cost.Zu = a.slack_weight_l2 * np.ones(self.n_h)
        tol = a.goal_tolerance * np.ones(3)
        ocp.constraints.lh_e, ocp.constraints.uh_e = -tol, tol
        ocp.constraints.idxsh_e = np.arange(3)
        ocp.cost.zl_e = ocp.cost.zu_e = a.terminal_slack_weight_l1 * np.ones(3)
        ocp.cost.Zl_e = ocp.cost.Zu_e = a.terminal_slack_weight_l2 * np.ones(3)

        self.np_ = params.shape[0]
        ocp.parameter_values = self.params()
        ocp.solver_options.N_horizon = a.N
        ocp.solver_options.tf = a.N * a.dt
        ocp.solver_options.integrator_type = "IRK"
        ocp.solver_options.nlp_solver_type = "SQP_RTI"
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.qp_solver_cond_N = a.N
        ocp.solver_options.print_level = 0

        build = Path(build_dir)
        json_file = str(build / "wb_mpc_ocp.json")
        ocp.code_export_directory = str(build / "c_generated_code_wb_mpc")
        stamp = build / "wb_mpc_ocp.fingerprint"
        fp = _fingerprint(m, proxies, c)
        so = Path(ocp.code_export_directory) / "libacados_ocp_solver_wb_mpc.so"
        hit = stamp.exists() and stamp.read_text().strip() == fp and Path(json_file).exists() and so.exists()
        self.solver = AcadosOcpSolver(ocp, json_file=json_file, build=not hit, generate=not hit, verbose=False)
        if not hit:
            stamp.write_text(fp)
        self.nx, self.nu, self.nq = nx, 2 + nq, nq
        self.yref = np.zeros(self.ny)
        self._apply_runtime()

    def _W(self):
        c, a = self.cfg, self.cfg.arm
        return np.diag([a.w_u] * 7 + [a.w_qdot] * 7 + [a.w_q_center] * 7 + [a.w_ee_track] * 3
                       + [a.w_orient] * 6 + [c.w_lag, c.w_lat, c.w_heading, c.w_heading, c.w_v, c.w_w,
                                             c.w_a, c.w_alpha])

    def _W_e(self):
        c, a = self.cfg, self.cfg.arm
        k = c.w_base_e
        return np.diag([a.w_terminal_qdot] * 7 + [k * c.w_lag, k * c.w_lat, k * c.w_heading, k * c.w_heading,
                                                  k * c.w_v, k * c.w_w])

    def _apply_runtime(self):
        N = self.cfg.arm.N
        for k in range(N):
            self.solver.cost_set(k, "W", self._W())
        self.solver.cost_set(N, "W", self._W_e())
        for k in range(1, N + 1):
            self.solver.constraints_set(k, "lbx", self.lbx)
            self.solver.constraints_set(k, "ubx", self.ubx)

    def params(self, obstacles=None, goal=(0.0, 0.0, 0.0), orient=(0.0, 0.0, -1.0, 0.0, -1.0, 0.0, 0.0),
               payload=0.0, sel=0.0, people=None, route=(1.0, 0.0)):
        """One stage's parameter vector (core/ocp.py's order, then the selector, people,
        the docking line's direction)."""
        a, c = self.cfg.arm, self.cfg
        obs = (np.tile([*NO_OBSTACLE_POSITION, NO_OBSTACLE_RADIUS], a.n_obstacles) if obstacles is None
               else np.asarray(obstacles, dtype=float).ravel())
        ppl = np.tile(NO_PERSON, c.n_people) if people is None else np.asarray(people, dtype=float).ravel()
        return np.r_[obs, goal, orient, payload, sel, ppl, route].astype(float)

    def set_stage(self, k, p, base_ref=None, posture_az=None):
        """Stage k's parameters and references: base_ref [x, y, theta, v, omega] (W);
        posture_az: joint 1's posture reference (rad), or None to keep it mid-range."""
        self.solver.set(k, "p", p)
        cr, sr = p[-2], p[-1]
        if base_ref is None:
            return
        b = np.asarray(base_ref, dtype=float)
        base_y = [cr * b[0] + sr * b[1], -sr * b[0] + cr * b[1], np.cos(b[2]), np.sin(b[2]), b[3], b[4]]
        if k < self.cfg.arm.N:
            yref = self.yref.copy()
            if posture_az is not None:
                yref[14] = posture_az - self.q_center[0]
            yref[30:36] = base_y
            self.solver.set(k, "yref", yref)
        else:
            self.solver.set(k, "yref", np.r_[np.zeros(7), base_y])

    def solve(self, x0):
        self.solver.set(0, "lbx", x0)
        self.solver.set(0, "ubx", x0)
        return self.solver.solve()

    def guess(self, x0, u_hold):
        for k in range(self.cfg.arm.N + 1):
            self.solver.set(k, "x", x0)
        for k in range(self.cfg.arm.N):
            self.solver.set(k, "u", np.r_[0.0, 0.0, u_hold])
