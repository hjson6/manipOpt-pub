"""Offline helpers: the scene at rest (boxes at their declared poses) and the arm
at a TCP pose (tool straight down, heading psi) by damped least-squares IK, so the
wrist camera sees what it would at a scan.
"""
import sys
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tasks/pick_and_place/common"), str(REPO / "tasks/pick_and_place/mpc")]
from pick_place_common.mujoco_sim_node import ARM_JOINT_NAMES, HOME_KEYFRAME, TCP_SITE_NAME  # noqa: E402
from pick_place_common import frames  # noqa: E402
from pick_place_common.scene import tool_down_rot  # noqa: E402

POSTURE_GAIN = 0.05


def reset_scene(m, d):
    """The scene at rest; load the model with world="arm" (positions here are arm-frame)."""
    mujoco.mj_resetDataKeyframe(m, d, m.key(HOME_KEYFRAME).id)
    frames.free_bodies_at_rest(m, d)
    mujoco.mj_forward(m, d)


def place_tcp(m, d, pos, psi, iters=300):
    """Move the arm so the TCP is at pos with the tool down and heading psi; returns the
    remaining (position error m, rotation error rad)."""
    qa = np.array([m.joint(n).qposadr[0] for n in ARM_JOINT_NAMES])
    va = np.array([m.joint(n).dofadr[0] for n in ARM_JOINT_NAMES])
    lo, hi = m.jnt_range[[m.joint(n).id for n in ARM_JOINT_NAMES]].T
    home = m.key(HOME_KEYFRAME).qpos[qa].copy()
    home[0] = np.arctan2(pos[1], pos[0])
    home[6] = np.clip((home[0] - psi - np.radians(135) + np.pi) % (2 * np.pi) - np.pi, lo[6], hi[6])  # as task_node._ik_reachable
    d.qpos[qa] = home
    site = m.site(TCP_SITE_NAME).id
    target = tool_down_rot(psi)
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    for _ in range(iters):
        mujoco.mj_forward(m, d)
        r = d.site_xmat[site].reshape(3, 3)
        e = np.r_[np.asarray(pos) - d.site_xpos[site], 0.5 * sum(np.cross(r[:, i], target[:, i]) for i in range(3))]
        if np.linalg.norm(e) < 1e-6:
            break
        mujoco.mj_jacSite(m, d, jp, jr, site)
        j = np.vstack([jp, jr])[:, va]
        jinv = j.T @ np.linalg.inv(j @ j.T + 1e-4 * np.eye(6))
        dq = jinv @ e + (np.eye(7) - jinv @ j) @ (POSTURE_GAIN * (home - d.qpos[qa]))
        d.qpos[qa] = np.clip(d.qpos[qa] + dq, lo, hi)
    mujoco.mj_forward(m, d)
    r = d.site_xmat[site].reshape(3, 3)
    rot_err = np.arccos(np.clip((np.trace(target.T @ r) - 1) / 2, -1, 1))
    return float(np.linalg.norm(np.asarray(pos) - d.site_xpos[site])), float(rot_err)
