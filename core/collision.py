"""Collision constraints between sphere proxies on the arm's links and sphere
obstacles, one per (proxy, obstacle) pair at every shooting node:

    ||p_proxy(q) - p_obstacle|| - (r_proxy + r_obstacle) - margin >= 0


Proxy radii over-approximate the link. Obstacle position and radius are
acados online parameters, idle at a far-away placeholder, so an obstacle
appearing needs no code generation. The OCP makes the constraints soft.
"""
from dataclasses import dataclass

import casadi as ca


@dataclass
class SphereProxy:
    frame_name: str
    local_offset: tuple  # (x, y, z) from the frame origin, frame coords
    radius: float  # over-approximates the link


def proxy_world_center(fk_fn: ca.Function, q: ca.SX, offset) -> ca.SX:
    """World centre of a proxy sphere from its frame's FK."""
    return fk_fn(q) + ca.SX(offset)


def collision_margin_expr(p_proxy: ca.SX, p_obstacle: ca.SX,
                           r_proxy: float, r_obstacle_sym: ca.SX,
                           safety_margin: float) -> ca.SX:
    """Surface-to-surface clearance (>= 0: no collision). The obstacle radius is
    symbolic (an online parameter).
    """
    dist = ca.norm_2(p_proxy - p_obstacle)
    return dist - (r_proxy + r_obstacle_sym) - safety_margin


NO_OBSTACLE_POSITION = (1e4, 1e4, 1e4)  # idle placeholder, far away
NO_OBSTACLE_RADIUS = 0.0
