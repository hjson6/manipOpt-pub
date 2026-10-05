"""The method's own robot model (panda_robot.xml, robot only): its collision
geometry at given joint angles, as world point sets, for the ceiling camera's
kinematic self-filter. Forward kinematics only; nothing of the plant's scene.
"""
import mujoco
import numpy as np

from pick_place_common.mujoco_sim_node import MENAGERIE_PANDA_ASSETS_DIR, MESHDIR_PLACEHOLDER, MODELS_DIR

CYLINDER_SIDES = 12


def _primitive_points(m, g):
    """Local points whose convex hull is the primitive (a cylinder's two rims)."""
    t, size = m.geom_type[g], m.geom_size[g]
    if t == mujoco.mjtGeom.mjGEOM_CYLINDER:
        a = np.linspace(0.0, 2 * np.pi, CYLINDER_SIDES, endpoint=False)
        rim = np.column_stack([size[0] * np.cos(a), size[0] * np.sin(a)])
        return np.vstack([np.column_stack([rim, np.full(len(a), z)]) for z in (-size[1], size[1])])
    if t == mujoco.mjtGeom.mjGEOM_BOX:
        return np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * size
    raise ValueError(f"collision geom type {t} not handled")


class RobotGeometry:
    def __init__(self):
        text = (MODELS_DIR / "panda_robot.xml").read_text().replace(MESHDIR_PLACEHOLDER, MENAGERIE_PANDA_ASSETS_DIR)
        self.model = mujoco.MjModel.from_xml_string(text)
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.tcp_site = m.site("tcp_site").id
        self.geoms = []  # (geom id, local points)
        for g in range(m.ngeom):
            if not (m.geom_contype[g] or m.geom_conaffinity[g]):
                continue
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                k = m.geom_dataid[g]
                pts = m.mesh_vert[m.mesh_vertadr[k]:m.mesh_vertadr[k] + m.mesh_vertnum[k]]
            else:
                pts = _primitive_points(m, g)
            self.geoms.append((g, np.array(pts, dtype=float)))

    def update(self, q):
        self.data.qpos[:len(q)] = q
        mujoco.mj_kinematics(self.model, self.data)

    def point_sets(self):
        """World points of each collision geom at the last update."""
        d = self.data
        return [p @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g] for g, p in self.geoms]

    def tcp(self):
        """(position, rotation) of the TCP at the last update."""
        return self.data.site_xpos[self.tcp_site].copy(), self.data.site_xmat[self.tcp_site].reshape(3, 3).copy()
