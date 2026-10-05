"""The base's MPC (acados): the unicycle model (x, y, heading, v, omega; inputs the two
accelerations) over a 2 s horizon, tracking a reference along the planned route
(the position error split into lag along the route, cheap while driving, and lateral
error, dear: it keeps its lane and slows for people rather than weaving round them;
where the reference stops or turns in place the lag is as dear: stops land), with
speed, acceleration and wheel-speed limits and the people kept at a distance over
the horizon (each predicted at constant velocity; soft, so the QP stays feasible
when someone steps close). SQP_RTI, as the arm's controller.
Framework-free apart from acados.

See docs/implementation_notes.md#base_mpcpy.
"""
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import casadi as ca
import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

NO_PERSON = (1e3, 1e3, 0.0)  # far away, radius 0
V_STOPPING = 0.05  # reference slower than this: a stop or a turn in place
V_CREEP_BACK = 0.05  # there it may back this fast, onto the spot


@dataclass
class BaseMPCConfig:
    N: int = 20
    dt: float = 0.1
    v_max: float = 0.5
    v_min: float = -0.15  # reversing only slowly
    w_max: float = 1.0
    a_max: float = 0.5
    a_min: float = -1.0
    alpha_max: float = 1.5
    track: float = 0.50
    rim_speed_max: float = 0.8  # |v| + |omega| track / 2
    n_people: int = 6
    robot_radius: float = 0.45  # the chassis's half diagonal, near enough
    w_lag: float = 5.0
    w_lat: float = 40.0
    w_heading: float = 4.0
    w_v: float = 2.0
    w_w: float = 1.0
    w_a: float = 0.5
    w_alpha: float = 1.0
    w_lag_e: float = 10.0
    w_lat_e: float = 80.0
    w_heading_e: float = 8.0
    slack_l1: float = 300.0
    slack_l2: float = 3000.0


STRUCTURE_VERSION = 3  # bump when the OCP's structure changes without the config changing
RUNTIME = ("v_max", "v_min", "w_max", "a_max", "a_min", "alpha_max", "rim_speed_max")  # set on the built solver


def _fingerprint(cfg):
    built = {k: v for k, v in asdict(cfg).items() if k not in RUNTIME}
    return hashlib.sha256(json.dumps({**built, "v": STRUCTURE_VERSION}, sort_keys=True).encode()).hexdigest()[:16]


class BaseMPC:
    def __init__(self, cfg=None, build_dir="."):
        self.cfg = cfg or BaseMPCConfig()
        c = self.cfg
        x = ca.SX.sym("x", 5)
        u = ca.SX.sym("u", 2)
        px, py, th, v, w = ca.vertsplit(x)
        p = ca.SX.sym("p", 3 * c.n_people + 2)  # per person x, y, radius (this stage's prediction); the route's cos, sin
        cr, sr = p[3 * c.n_people], p[3 * c.n_people + 1]
        model = AcadosModel()
        model.name = "base_mpc"
        model.x, model.u, model.p = x, u, p
        model.f_expl_expr = ca.vertcat(v * ca.cos(th), v * ca.sin(th), w, u[0], u[1])
        model.con_h_expr = ca.vertcat(*[
            (px - p[3 * j]) ** 2 + (py - p[3 * j + 1]) ** 2 - (p[3 * j + 2] + c.robot_radius) ** 2
            for j in range(c.n_people)])
        along, across = cr * px + sr * py, -sr * px + cr * py
        model.cost_y_expr = ca.vertcat(along, across, ca.cos(th), ca.sin(th), v, w, u[0], u[1])
        model.cost_y_expr_e = ca.vertcat(along, across, ca.cos(th), ca.sin(th), v, w)
        ocp = AcadosOcp()
        ocp.model = model
        ocp.cost.cost_type = ocp.cost.cost_type_e = "NONLINEAR_LS"
        ocp.cost.W = np.diag([c.w_lag, c.w_lat, c.w_heading, c.w_heading, c.w_v, c.w_w, c.w_a, c.w_alpha])
        ocp.cost.W_e = np.diag([c.w_lag_e, c.w_lat_e, c.w_heading_e, c.w_heading_e, c.w_v, c.w_w])
        ocp.cost.yref = np.zeros(8)
        ocp.cost.yref_e = np.zeros(6)
        ocp.constraints.x0 = np.zeros(5)
        ocp.constraints.idxbu = np.array([0, 1])
        ocp.constraints.lbu = np.array([c.a_min, -c.alpha_max])
        ocp.constraints.ubu = np.array([c.a_max, c.alpha_max])
        ocp.constraints.idxbx = np.array([3, 4])
        ocp.constraints.lbx = np.array([c.v_min, -c.w_max])
        ocp.constraints.ubx = np.array([c.v_max, c.w_max])
        ocp.constraints.idxbx_e = np.array([3, 4])
        ocp.constraints.lbx_e = np.array([c.v_min, -c.w_max])
        ocp.constraints.ubx_e = np.array([c.v_max, c.w_max])
        half = c.track / 2
        ocp.constraints.C = np.array([[0, 0, 0, 1, half], [0, 0, 0, 1, -half]], dtype=float)
        ocp.constraints.D = np.zeros((2, 2))
        ocp.constraints.lg = -c.rim_speed_max * np.ones(2)
        ocp.constraints.ug = c.rim_speed_max * np.ones(2)
        ocp.constraints.lh = np.zeros(c.n_people)
        ocp.constraints.uh = 1e8 * np.ones(c.n_people)
        ocp.constraints.idxsh = np.arange(c.n_people)
        ocp.cost.zl = ocp.cost.zu = c.slack_l1 * np.ones(c.n_people)
        ocp.cost.Zl = ocp.cost.Zu = c.slack_l2 * np.ones(c.n_people)
        ocp.parameter_values = np.r_[np.tile(NO_PERSON, c.n_people), 1.0, 0.0].astype(float)
        ocp.solver_options.N_horizon = c.N
        ocp.solver_options.tf = c.N * c.dt
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.sim_method_num_steps = 2
        ocp.solver_options.nlp_solver_type = "SQP_RTI"
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.print_level = 0
        build = Path(build_dir)
        json_file = str(build / "base_mpc_ocp.json")
        ocp.code_export_directory = str(build / "c_generated_code_base_mpc")
        stamp = build / "base_mpc_ocp.fingerprint"
        fp = _fingerprint(c)
        so = Path(ocp.code_export_directory) / "libacados_ocp_solver_base_mpc.so"
        hit = stamp.exists() and stamp.read_text().strip() == fp and Path(json_file).exists() and so.exists()
        self.solver = AcadosOcpSolver(ocp, json_file=json_file, build=not hit, generate=not hit, verbose=False)
        if not hit:
            stamp.write_text(fp)
        for k in range(c.N):
            self.solver.constraints_set(k, "lbu", np.array([c.a_min, -c.alpha_max]))
            self.solver.constraints_set(k, "ubu", np.array([c.a_max, c.alpha_max]))
            self.solver.constraints_set(k, "lg", -c.rim_speed_max * np.ones(2))
            self.solver.constraints_set(k, "ug", c.rim_speed_max * np.ones(2))
        self.warm = False
        self._W = [ocp.cost.W] * c.N + [ocp.cost.W_e]
        W_stop, W_stop_e = ocp.cost.W.copy(), ocp.cost.W_e.copy()
        W_stop[0, 0], W_stop_e[0, 0] = c.w_lat, c.w_lat_e
        self._W_stop = [W_stop] * c.N + [W_stop_e]
        self._stopping = [False] * (c.N + 1)

    def solve(self, x0, ref, people, v_max=None):
        """x0: [x, y, yaw, v, omega]; ref: (N+1, 5) states along the route; people: (N+1,
        n_people, 3) predicted [x, y, radius] per stage (NO_PERSON for none); v_max: a
        lower speed limit (people near), reached braking if need be. It reverses only
        where the reference does, or creeps back onto a stop (else it may only brake out
        of reversing). Returns (status, the (N+1, 5) predicted states)."""
        c = self.cfg
        x0 = np.asarray(x0, dtype=float)
        reverse = bool((ref[:, 3] < -1e-3).any())
        cap = c.v_max if v_max is None else min(v_max, c.v_max)
        for k in range(1, c.N + 1):
            lo = c.v_min if reverse else max(c.v_min, min(0.0, x0[3] + c.a_max * k * c.dt))
            if abs(ref[k, 3]) < V_STOPPING:
                lo = min(lo, -V_CREEP_BACK)
            hi = max(cap, min(c.v_max, x0[3] + c.a_min * k * c.dt))
            self.solver.constraints_set(k, "lbx", np.array([lo, -c.w_max]))
            self.solver.constraints_set(k, "ubx", np.array([hi, c.w_max]))
        if not self.warm:
            for k in range(c.N + 1):
                self.solver.set(k, "x", np.r_[ref[k, :3], 0.0, 0.0] if k else x0)
            self.warm = True
        self.solver.set(0, "lbx", x0)
        self.solver.set(0, "ubx", x0)
        for k in range(c.N + 1):
            r = ref[k]
            cr, sr = np.cos(r[2]), np.sin(r[2])
            yref = np.array([cr * r[0] + sr * r[1], -sr * r[0] + cr * r[1], cr, sr, r[3], r[4]])
            stopping = abs(r[3]) < V_STOPPING
            if stopping != self._stopping[k]:
                self._stopping[k] = stopping
                self.solver.cost_set(k, "W", self._W_stop[k] if stopping else self._W[k])
            self.solver.set(k, "yref", np.r_[yref, 0.0, 0.0] if k < c.N else yref)
            self.solver.set(k, "p", np.r_[np.asarray(people[k], dtype=float).ravel(), cr, sr])
        status = self.solver.solve()
        xs = np.array([self.solver.get(k, "x") for k in range(c.N + 1)])
        if status != 0:
            self.warm = False
        return status, xs
