"""Plant and offline side: the arm base's true pose in the room (MuJoCo's world),
for changing between the room and the arm frame. The method never calls it.

See docs/implementation_notes.md#framespy.
"""
import mujoco
import numpy as np

from pick_place_common.scene import ARM_IN_BASE, BASE_ARM_MOUNT, compose, invert

ARM_BASE_BODY = "link0"


def arm_base_pose(model, data):
    """link0's position and rotation in the room."""
    b = model.body(ARM_BASE_BODY).id
    return data.xpos[b].copy(), data.xmat[b].reshape(3, 3).copy()


def to_arm(pose, p, rot=None):
    """Room point p (and rotation rot) in the arm frame, pose = arm_base_pose()."""
    pos, r = pose
    p_arm = r.T @ (np.asarray(p, dtype=float) - pos)
    return p_arm if rot is None else (p_arm, r.T @ np.asarray(rot))


def to_room(pose, p, rot=None):
    """Arm-frame point p (and rotation rot) in the room."""
    pos, r = pose
    p_room = pos + r @ np.asarray(p, dtype=float)
    return p_room if rot is None else (p_room, r @ np.asarray(rot))


def yaw_of(rot):
    """Heading of a rotation's x axis about z."""
    return float(np.arctan2(rot[1, 0], rot[0, 0]))


def free_bodies_at_rest(model, data):
    """Put free bodies (boxes, the base) at their declared poses."""
    for b in range(1, model.nbody):
        if model.body_jntnum[b] != 1 or model.jnt_type[model.body_jntadr[b]] != mujoco.mjtJoint.mjJNT_FREE:
            continue
        a = model.jnt_qposadr[model.body_jntadr[b]]
        data.qpos[a:a + 3] = model.body_pos[b]
        data.qpos[a + 3:a + 7] = model.body_quat[b]


def arm_pose_from_base(base):
    """The arm base's planar pose (x, y, yaw) and height in the room from base_link's true
    pose [x, y, z, qw, qx, qy, qz]."""
    x, y, z, qw, qx, qy, qz = base
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return compose((x, y, yaw), ARM_IN_BASE), z + BASE_ARM_MOUNT[2]


def person_in_arm(sim, i):
    """sim.csv row i's person (x, y, z, yaw) in the arm frame, from the logged true base
    pose; runs from before the mobile base logged it in the arm frame already."""
    p = (sim["person_x"][i], sim["person_y"][i], sim["person_z"][i], sim["person_yaw"][i])
    if "base_x" not in sim or not np.isfinite(sim["base_x"][i]):
        return p
    arm, arm_z = arm_pose_from_base([sim[k][i] for k in BASE_COLUMNS])
    x, y, yaw = compose(invert(arm), (p[0], p[1], p[3]))
    return x, y, p[2] - arm_z, yaw


BASE_COLUMNS = ("base_x", "base_y", "base_z", "base_qw", "base_qx", "base_qy", "base_qz")


def in_arm(model, body):
    """True for link0 and the bodies below it (the arm, its tool, a welded box not)."""
    b0 = model.body(ARM_BASE_BODY).id
    while body > 0:
        if body == b0:
            return True
        body = model.body_parentid[body]
    return False
