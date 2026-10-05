"""Joint-state estimator for a plant that applies each torque a fixed time after
the state it was computed from: an EKF on the measured joint positions with the
controller's own model, and a prediction to the moment the next torque starts.

See docs/implementation_notes.md#state_estimatorpy.
"""
import casadi as ca
import numpy as np

from core.dynamics import ManipulatorModel

RK4_STEP_S = 0.005


def _rk4(f, x, u, pm, duration):
    n = max(1, int(round(duration / RK4_STEP_S)))
    h = duration / n
    for _ in range(n):
        k1 = f(x, u, pm)
        k2 = f(x + h / 2 * k1, u, pm)
        k3 = f(x + h / 2 * k2, u, pm)
        k4 = f(x + h * k3, u, pm)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


class DelayEKF:
    """State [q, qdot]. Each tick the plant runs u_prev for `switch_s`, then u_new
    until the next state; the measurement is q alone (the driver's velocity comes
    from the same encoder reads). The mean is predicted with the full model, the
    covariance with q' = q + dt qdot (the model's Jacobian made an update ~7 ms)."""

    def __init__(self, m: ManipulatorModel, dt: float, switch_s: float, vel_noise, pos_noise: float):
        """vel_noise: the model's velocity error per tick, one value or one per joint."""
        f = ca.Function("f", [m.x, m.u, m.payload_mass], [m.xdot])
        nx, nu = m.nq + m.nv, m.nv
        x, u1, u2, pm = ca.SX.sym("x", nx), ca.SX.sym("u1", nu), ca.SX.sym("u2", nu), ca.SX.sym("pm")
        x_next = _rk4(f, _rk4(f, x, u1, pm, switch_s), u2, pm, dt - switch_s)
        self._tick = ca.Function("tick", [x, u1, u2, pm], [x_next])
        self._ahead = ca.Function("ahead", [x, u1, pm], [_rk4(f, x, u1, pm, switch_s)])
        self.nq = m.nq
        self.a = np.block([[np.eye(m.nq), dt * np.eye(m.nv)], [np.zeros((m.nv, m.nq)), np.eye(m.nv)]])
        self.q_noise = np.diag(np.r_[np.full(m.nq, 1e-6), np.broadcast_to(vel_noise, m.nv)] ** 2)
        self.r_noise = np.eye(m.nq) * pos_noise ** 2
        self.h = np.hstack([np.eye(m.nq), np.zeros((m.nq, m.nv))])
        self.x = None
        self.p = None

    def reset(self, q, qdot):
        self.x = np.concatenate([q, qdot]).astype(float)
        self.p = np.eye(len(self.x)) * 1e-4

    def update(self, q, u_prev, u_new, payload_mass):
        """Advance over the tick that ended at this measurement, then correct with q."""
        x_pred = np.array(self._tick(self.x, u_prev, u_new, payload_mass)).flatten()
        p = self.a @ self.p @ self.a.T + self.q_noise
        gain = p @ self.h.T @ np.linalg.inv(self.h @ p @ self.h.T + self.r_noise)
        self.x = x_pred + gain @ (np.asarray(q) - x_pred[:self.nq])
        self.p = (np.eye(len(self.x)) - gain @ self.h) @ p
        return self.x

    def ahead(self, u, payload_mass, q=None):
        """The state when the next torque starts, with u running until then; from the
        measured q with the estimated velocity if q is given."""
        x = self.x if q is None else np.concatenate([q, self.x[self.nq:]])
        return np.array(self._ahead(x, u, payload_mass)).flatten()
