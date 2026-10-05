"""Plant side: planar safety-lidar scans by ray casting the scene (mj_multiRay)
from the scanners' sites on the moving base, with range noise and dropouts. Raw
ranges only, as a driver gives them; the detection is the method's
(perception/lidar_detection.py).
"""
import mujoco
import numpy as np

from pick_place_common import frames
from pick_place_common.scene import (
    LIDAR_DROPOUT, LIDAR_FOV_DEG, LIDAR_RANGE_M, LIDAR_RANGE_SIGMA_M, LIDAR_SCANNERS, LIDAR_STEP_DEG)

# Beam angles relative to the sector's centre.
BEAM_ANGLES = np.radians(np.arange(-LIDAR_FOV_DEG / 2, LIDAR_FOV_DEG / 2 + LIDAR_STEP_DEG / 2, LIDAR_STEP_DEG))
# Groups the rays see: not 2 (the arm's visual meshes).
RAY_GROUPS = np.array([1, 1, 0, 1, 1, 1], dtype=np.uint8)


def check_sites(model, data):
    """The lidar_* sites in the XML, seen from the arm, must be where scene.py's
    LIDAR_SCANNERS (the calibration) says."""
    arm = frames.arm_base_pose(model, data)
    for i, (x, y, z, yaw) in enumerate(LIDAR_SCANNERS):
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"lidar_{i}")
        if sid < 0:
            raise RuntimeError(f"no lidar_{i} site in the scene XML")
        p, r = frames.to_arm(arm, data.site_xpos[sid], data.site_xmat[sid].reshape(3, 3))
        dyaw = (frames.yaw_of(r) - yaw + np.pi) % (2 * np.pi) - np.pi
        if np.abs(p - (x, y, z)).max() > 1e-3 or abs(dyaw) > 1e-3:
            raise RuntimeError(f"lidar_{i} in the scene XML disagrees with scene.py's LIDAR_SCANNERS")


def scan(model, data, index, rng=None, hit_geoms=False):
    """Ranges (m) of scanner `index` over BEAM_ANGLES; nan where there is no return.
    The beams fan out in the site's x-y plane (it tilts with the base).
    hit_geoms: also each beam's geom id (-1: none), for offline checks."""
    sid = model.site(f"lidar_{index}").id
    rot = data.site_xmat[sid].reshape(3, 3)
    n = len(BEAM_ANGLES)
    vec = (np.column_stack([np.cos(BEAM_ANGLES), np.sin(BEAM_ANGLES), np.zeros(n)]) @ rot.T).ravel()
    geomid = np.zeros(n, dtype=np.int32)
    dist = np.zeros(n)
    own = int(model.site_bodyid[sid])  # the scanner's own chassis
    mujoco.mj_multiRay(model, data, data.site_xpos[sid].copy(), vec, RAY_GROUPS, 1, own, geomid, dist, None, n,
                       LIDAR_RANGE_M[1])
    r = np.where((geomid >= 0) & (dist >= LIDAR_RANGE_M[0]), dist, np.nan)
    if rng is not None:
        r = r + rng.normal(0.0, LIDAR_RANGE_SIGMA_M, n)
        r[rng.random(n) < LIDAR_DROPOUT] = np.nan
    return (r, geomid) if hit_geoms else r
