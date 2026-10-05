"""Unit tests for slam/ on a synthetic rectangular room (no simulator)."""
import numpy as np

from slam import icp
from slam.grid import OccupancyGrid
from slam.localizer import GridLocalizer
from slam.mapper import GraphSlam
from slam.pose_graph import PoseGraph, compose, relative
from slam.scan import merge_scan, transform

ROOM = (-3.0, 3.0, -2.0, 2.0)  # x0, x1, y0, y1 (inner faces)
ANGLES = np.radians(np.arange(-180.0, 180.0, 0.5))


def room_scan(pose, rng=None, sigma=0.0):
    """Points in the robot frame of a 360 deg scan from pose in the room, plus a pillar."""
    x, y, yaw = pose
    a = yaw + ANGLES
    d = np.column_stack([np.cos(a), np.sin(a)])
    with np.errstate(divide="ignore", invalid="ignore"):
        tx = np.where(d[:, 0] > 0, (ROOM[1] - x) / d[:, 0], (ROOM[0] - x) / d[:, 0])
        ty = np.where(d[:, 1] > 0, (ROOM[3] - y) / d[:, 1], (ROOM[2] - y) / d[:, 1])
    r = np.minimum(np.abs(tx), np.abs(ty))
    # a 0.4 m square pillar at (1.5, 1.0)
    for k in range(len(r)):
        for s in np.arange(0.05, r[k], 0.01):
            px, py = x + s * d[k, 0], y + s * d[k, 1]
            if abs(px - 1.5) < 0.2 and abs(py - 1.0) < 0.2:
                r[k] = s
                break
    if rng is not None:
        r = r + rng.normal(0.0, sigma, len(r))
    world = np.column_stack([x + r * d[:, 0], y + r * d[:, 1]])
    return relative_points(pose, world)


def relative_points(pose, world):
    c, s = np.cos(pose[2]), np.sin(pose[2])
    p = world - pose[:2]
    return np.column_stack([c * p[:, 0] + s * p[:, 1], -s * p[:, 0] + c * p[:, 1]])


def test_merge_scan_keeps_the_nearest_point_per_beam():
    pts = np.array([[1.0, 0.0], [2.0, 0.0], [0.0, 3.0]])
    r = merge_scan(pts, n_beams=360)
    assert np.isclose(r[180], 1.0) and np.isclose(r[270], 3.0)
    assert np.isnan(r[0])


def test_icp_recovers_a_known_motion():
    a, b = np.array([0.0, 0.0, 0.0]), np.array([0.25, -0.1, np.radians(8)])
    target = icp.Target(room_scan(a))
    pose, _info, rms, fit = icp.match(room_scan(b), target, (0.0, 0.0, 0.0))
    assert np.allclose(pose, relative(a, b), atol=[0.005, 0.005, np.radians(0.2)])
    assert fit > 0.8 and rms < 0.01


def test_pose_graph_closes_a_loop():
    rng = np.random.default_rng(0)
    truth = [np.array([2 * np.cos(t), 2 * np.sin(t), t + np.pi / 2]) for t in np.linspace(0, 2 * np.pi, 21)[:-1]]
    g = PoseGraph()
    est = truth[0].copy()
    g.add_node(est)
    info = np.diag([1e4, 1e4, 1e4])
    for i in range(1, len(truth)):
        z = relative(truth[i - 1], truth[i]) + rng.normal(0, [0.02, 0.02, 0.02])
        est = compose(est, z)
        g.add_node(est)
        g.add_edge(i - 1, i, z, info)
    drift = np.hypot(*(g.nodes[-1][:2] - truth[-1][:2]))
    g.add_edge(len(truth) - 1, 0, relative(truth[-1], truth[0]), info * 10, kind="loop")
    g.optimize()
    err = max(np.hypot(*(n[:2] - t[:2])) for n, t in zip(g.nodes, truth))
    assert err < drift and err < 0.1


def test_grid_save_and_load(tmp_path):
    pose = np.array([0.0, 0.0, 0.0])
    pts = room_scan(pose)
    grid = OccupancyGrid.around(transform(pose, pts), 0.05)
    for _ in range(3):
        grid.integrate(pose, pts, np.zeros_like(pts))
    grid.save(tmp_path / "m")
    back = OccupancyGrid.load(tmp_path / "m.yaml")
    assert np.array_equal(back.occupied, grid.occupied)
    assert np.allclose(back.occupied_points(), grid.occupied_points())


def test_mapping_then_localization_in_the_map():
    rng = np.random.default_rng(1)
    path = [np.array([-1.5 + 0.1 * k, -0.8, 0.0]) for k in range(30)]
    slam = GraphSlam()
    for p in path:
        pts = room_scan(p, rng, 0.01)
        slam.update(p - path[0] + rng.normal(0, [0.003, 0.003, 0.002]), pts, np.zeros_like(pts))
    est = slam.update(path[-1] - path[0], room_scan(path[-1], rng, 0.01), np.zeros((len(ANGLES), 2)))
    assert np.hypot(*(est[:2] - (path[-1] - path[0])[:2])) < 0.03
    grid = slam.map()
    start = relative(path[0], np.array([0.5, 0.5, np.radians(30)]))
    loc = GridLocalizer(grid, start + (0.05, -0.05, np.radians(2)))
    pose = loc.update(np.zeros(3), room_scan(np.array([0.5, 0.5, np.radians(30)]), rng, 0.01))
    assert np.hypot(*(pose[:2] - start[:2])) < 0.02 and abs(pose[2] - start[2]) < np.radians(0.5)
