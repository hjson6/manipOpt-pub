"""Plant/model mismatch for the simulated arm and boxes: the plant stops being the
controller's model (which is built from the unchanged panda_robot.xml). Joint
friction, damping and armature, link masses and centres of mass, and box
masses; for the mobile base, the wheel radii and track (odometry's taught
values) and the chassis mass. See docs/implementation_notes.md#plant_mismatchpy.
"""
import mujoco
import numpy as np

FRICTION_BIG_NM = (0.5, 1.5)  # joints 1-4
FRICTION_WRIST_NM = (0.2, 0.5)  # joints 5-7
DAMPING_SCALE = (0.7, 1.3)
# Rotor inertia is a datasheet value; the MPC chatters on joint 7 below ~0.75x (known_issues I8).
ARMATURE_SCALE = (0.85, 1.15)
LINK_MASS_SCALE = (0.9, 1.1)
LINK_COM_SHIFT_M = 0.005  # std per axis
BOX_MASS_KG = (0.2, 2.0)
WHEEL_RADIUS_SCALE = (0.995, 1.005)  # each wheel: tyre wear and load
WHEEL_TRACK_SHIFT_M = (-0.005, 0.005)  # effective contact line, each side
CHASSIS_MASS_SCALE = (0.9, 1.1)


def apply_plant_mismatch(model, rng, arm_joint_names, box_names, arm=True):
    """Change model in place; returns a one-line summary for the log. arm=False
    leaves the arm nominal (the oracle) but draws the same box masses.
    """
    for i, name in enumerate(arm_joint_names):
        dof = model.jnt_dofadr[model.joint(name).id]
        friction = rng.uniform(*(FRICTION_BIG_NM if i < 4 else FRICTION_WRIST_NM))
        damping, armature = rng.uniform(*DAMPING_SCALE), rng.uniform(*ARMATURE_SCALE)
        if arm:
            model.dof_frictionloss[dof] = friction
            model.dof_damping[dof] *= damping
            model.dof_armature[dof] *= armature
    for i in range(1, 8):
        b = model.body(f"link{i}").id
        s = rng.uniform(*LINK_MASS_SCALE)
        shift = rng.normal(0.0, LINK_COM_SHIFT_M, 3)
        if arm:
            model.body_mass[b] *= s
            model.body_inertia[b] *= s
            model.body_ipos[b] += shift
    masses = []
    for name in box_names:
        b = model.body(name).id
        mass = rng.uniform(*BOX_MASS_KG)
        model.body_inertia[b] *= mass / model.body_mass[b]
        model.body_mass[b] = mass
        masses.append(f"{name} {mass:.2f}")
    friction = ", ".join(f"{model.dof_frictionloss[model.jnt_dofadr[model.joint(n).id]]:.2f}"
                         for n in arm_joint_names)
    return f"joint friction (Nm) {friction}; box masses (kg) {', '.join(masses)}"


def apply_base_mismatch(model, rng, wheel_joint_names, base_body, worn=0.0):
    """The base's wheels and chassis; worn: the first wheel's radius this much smaller on top
    (plant_conditions.WORN_TYRE). Returns a one-line summary."""
    radii, track = [], 0.0
    for i, name in enumerate(wheel_joint_names):
        b = model.jnt_bodyid[model.joint(name).id]
        g = model.body_geomadr[b]
        s = rng.uniform(*WHEEL_RADIUS_SCALE) * (1.0 - worn if i == 0 else 1.0)
        model.geom_size[g][[0, 2]] *= s  # the tread: an ellipsoid
        mount = model.body_parentid[b]  # the suspension's body, on the chassis
        model.body_pos[mount][2] = model.geom_size[g][0]  # the chassis stays level
        model.body_pos[mount][1] += np.sign(model.body_pos[mount][1]) * rng.uniform(*WHEEL_TRACK_SHIFT_M)
        radii.append(model.geom_size[g][0])
        track += abs(model.body_pos[mount][1])
    b = model.body(base_body).id
    s = rng.uniform(*CHASSIS_MASS_SCALE)
    model.body_mass[b] *= s
    model.body_inertia[b] *= s
    return (f"wheel radii (mm) {', '.join(f'{1e3 * r:.2f}' for r in radii)}, track {1e3 * track:.1f} mm, "
            f"chassis {model.body_mass[b]:.0f} kg")


def refresh(model, data):
    """Recompute the model's derived constants after a change."""
    mujoco.mj_setConst(model, data)
