"""Plant/model mismatch for the simulated arm and boxes: the plant stops being the
controller's model (which is built from the unchanged panda_robot.xml). Joint
friction, damping and armature, link masses and centres of mass, and box
masses. See docs/implementation_notes.md#plant_mismatchpy.
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


def apply_plant_mismatch(model, rng, arm_joint_names, box_names):
    """Change model in place; returns a one-line summary for the log."""
    for i, name in enumerate(arm_joint_names):
        dof = model.jnt_dofadr[model.joint(name).id]
        model.dof_frictionloss[dof] = rng.uniform(*(FRICTION_BIG_NM if i < 4 else FRICTION_WRIST_NM))
        model.dof_damping[dof] *= rng.uniform(*DAMPING_SCALE)
        model.dof_armature[dof] *= rng.uniform(*ARMATURE_SCALE)
    for i in range(1, 8):
        b = model.body(f"link{i}").id
        s = rng.uniform(*LINK_MASS_SCALE)
        model.body_mass[b] *= s
        model.body_inertia[b] *= s
        model.body_ipos[b] += rng.normal(0.0, LINK_COM_SHIFT_M, 3)
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


def refresh(model, data):
    """Recompute the model's derived constants after a change."""
    mujoco.mj_setConst(model, data)
