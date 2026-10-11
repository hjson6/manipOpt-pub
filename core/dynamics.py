"""Symbolic arm dynamics for acados, from the MJCF via Pinocchio's CasADi
backend: q_ddot = M(q)^-1 (tau - h(q, q_dot)), a plain ODE for acados' IRK.

Built from the same panda_robot.xml as the MuJoCo plant, but through
Pinocchio's own parser, so the controller's model and the plant can differ.
Only kinematics and inertials are read, so the meshdir placeholder is fine.
"""
from dataclasses import dataclass

import casadi as ca
import numpy as np
import pinocchio as pin
import pinocchio.casadi as cpin


@dataclass
class ManipulatorModel:
    name: str
    nq: int  # number of joint positions
    nv: int  # = nq (fixed base)
    x: ca.SX  # state [q; q_dot]
    u: ca.SX  # joint torques
    xdot: ca.SX  # explicit ODE right-hand side f(x, u)
    link_frame_names: list  # frames for FK and collision proxies
    payload_mass: ca.SX  # online parameter: mass held at the tool
    model: pin.Model
    cmodel: cpin.Model
    cdata: cpin.Data


def load_manipulator(mjcf_path: str, link_frame_names: list[str], payload_frame: str = "tcp_site",
                     payload_offset: float = 0.05) -> ManipulatorModel:
    """CasADi ODE model of the arm from its MJCF.

    mjcf_path must be robot-only (Pinocchio's MJCF parser does not accept other
    top-level bodies). link_frame_names are body names used later for FK. A held
    payload's weight acts at payload_offset along payload_frame's z axis (below
    the TCP with the tool down); its mass is the symbol payload_mass. Its
    inertia is left out (small next to the arm's at these masses).
    """
    model = pin.buildModelFromMJCF(mjcf_path)
    data = model.createData()

    cmodel = cpin.Model(model)
    cdata = cmodel.createData()

    nq, nv = model.nq, model.nv
    q = ca.SX.sym("q", nq)
    qdot = ca.SX.sym("qdot", nv)
    tau = ca.SX.sym("tau", nv)

    payload_mass = ca.SX.sym("payload_mass", 1)
    frame_id = cmodel.getFrameId(payload_frame)
    cpin.computeJointJacobians(cmodel, cdata, q)
    cpin.updateFramePlacements(cmodel, cdata)
    jac = cpin.getFrameJacobian(cmodel, cdata, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    r = cdata.oMf[frame_id].rotation[:, 2] * payload_offset  # frame origin -> payload centre, world
    skew_r = ca.vertcat(ca.horzcat(0, -r[2], r[1]), ca.horzcat(r[2], 0, -r[0]), ca.horzcat(-r[1], r[0], 0))
    jac_point = jac[:3, :] - skew_r @ jac[3:, :]
    tau_payload = jac_point.T @ ca.vertcat(0, 0, -9.81 * payload_mass)

    # ABA ignores joint damping, so pass it in as a torque. Without it the model
    # thought the (damped) plant far livelier than it is.
    qddot = cpin.aba(cmodel, cdata, q, qdot, tau - cmodel.damping * qdot + tau_payload)

    x = ca.vertcat(q, qdot)
    xdot = ca.vertcat(qdot, qddot)

    return ManipulatorModel(
        name=model.name,
        nq=nq,
        nv=nv,
        x=x,
        u=tau,
        xdot=xdot,
        link_frame_names=link_frame_names,
        payload_mass=payload_mass,
        model=model,
        cmodel=cmodel,
        cdata=cdata,
    )


def forward_kinematics(m: ManipulatorModel, frame_name: str) -> ca.Function:
    """CasADi function q -> world position of frame_name."""
    q = ca.SX.sym("q", m.nq)
    frame_id = m.cmodel.getFrameId(frame_name)
    cpin.forwardKinematics(m.cmodel, m.cdata, q)
    cpin.updateFramePlacements(m.cmodel, m.cdata)
    p = m.cdata.oMf[frame_id].translation
    return ca.Function(f"fk_{frame_name}", [q], [p])


def forward_kinematics_axis(m: ManipulatorModel, frame_name: str,
                             local_axis=(0.0, 0.0, 1.0)) -> ca.Function:
    """CasADi function q -> world direction of local_axis (a unit vector in
    frame_name's frame). One axis leaves the rotation about it free; the OCP
    uses two to fix the orientation.
    """
    q = ca.SX.sym("q", m.nq)
    frame_id = m.cmodel.getFrameId(frame_name)
    cpin.forwardKinematics(m.cmodel, m.cdata, q)
    cpin.updateFramePlacements(m.cmodel, m.cdata)
    axis_world = m.cdata.oMf[frame_id].rotation @ ca.SX(list(local_axis))
    return ca.Function(f"fk_axis_{frame_name}", [q], [axis_world])
