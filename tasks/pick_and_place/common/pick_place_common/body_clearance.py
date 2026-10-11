"""Validation (plant side, on the truth): the smallest distance between the robot's
upper structure (the arm and a held box, above the pedestal) and any part of the people
(legs, torso, arms, head), the parts it is between, and how fast the robot's nearest
point closes on that person (the base's rigid motion). The chassis against the legs
is the leg-level scoring (nav_scoring.py); this is what the 2D scanners do not see.
"""
import mujoco
import numpy as np

from pick_place_common import frames

NEAR_M = 1.5  # people farther from the base (in plan) are not measured
CUTOFF_M = 3.0  # farther counts as nobody near


class BodyClearance:
    def __init__(self, m, people=tuple(f"person_{i}" for i in range(1, 7)), extra_bodies=(), base="base_link",
                 near_m=NEAR_M):
        """extra_bodies: names of bodies carried along (a held box); near_m: people
        farther from the base (in plan) are not measured."""
        self.m = m
        self.near_m = near_m
        self.base = m.body(base).id
        extra = {m.body(n).id for n in extra_bodies}
        self.robot_geoms = [g for g in range(m.ngeom)
                            if (m.geom_contype[g] or m.geom_conaffinity[g])
                            and (frames.in_arm(m, m.geom_bodyid[g]) or m.geom_bodyid[g] in extra)]
        self.people = []
        for name in people:
            b = m.body(name)
            self.people.append((name, b.id, list(range(b.geomadr[0], b.geomadr[0] + b.geomnum[0]))))
        self.fromto = np.zeros(6)

    def nearest(self, d):
        """(distance, robot part, person, robot point, person point) of the closest pair
        among the people within near_m of the base; distance inf if none."""
        best = (np.inf, "", "", None, None)
        base_xy = d.xpos[self.base][:2]
        for name, body, geoms in self.people:
            if np.hypot(*(d.xpos[body][:2] - base_xy)) > self.near_m:
                continue
            for pg in geoms:
                for rg in self.robot_geoms:
                    dist = mujoco.mj_geomDistance(self.m, d, rg, pg, CUTOFF_M, self.fromto)
                    if dist < min(best[0], CUTOFF_M):
                        best = (float(dist), self.m.body(self.m.geom_bodyid[rg]).name, name,
                                self.fromto[:3].copy(), self.fromto[3:].copy())
        return best

    def closing_speed(self, d, p_robot, p_person, v_world, w_world):
        """Speed (m/s) at which the robot's point p_robot, moving rigidly with the base
        (linear v_world, angular w_world at the base's origin), closes on p_person."""
        n = np.asarray(p_person) - np.asarray(p_robot)
        dist = float(np.linalg.norm(n))
        if dist < 1e-9:
            return 0.0
        v = np.asarray(v_world) + np.cross(w_world, np.asarray(p_robot) - d.xpos[self.base])
        return float(v @ n / dist)
