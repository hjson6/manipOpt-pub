"""Unit tests for slam/base_ekf.py on synthetic motion (no simulator)."""
import numpy as np

from slam import base_ekf
from slam.base_ekf import B, BaseEKF, motion, motion_jacobian

R, TRACK, DT = 0.1, 0.5, 0.02
INFO = np.diag([1 / 0.004 ** 2, 1 / 0.004 ** 2, 1 / np.radians(0.1) ** 2])


class Robot:
    """The true motion and the sensors: wheel angles (exact radius and track), the gyro with a bias."""

    def __init__(self, bias=0.0, seed=0):
        self.pose = np.zeros(3)
        self.wheels = np.zeros(2)
        self.bias = bias
        self.rng = np.random.default_rng(seed)
        self.t = 0.0

    def step(self, v, w):
        x = motion(np.r_[self.pose, v, w, 0.0], DT)
        self.pose = x[:3]
        self.wheels += np.array([v - 0.5 * w * TRACK, v + 0.5 * w * TRACK]) * DT / R
        self.t += DT
        return self.t, self.wheels.copy(), w + self.bias + self.rng.normal(0.0, base_ekf.GYRO_NOISE)

    def match(self, sigma=0.002):
        pose = self.pose + self.rng.normal(0.0, sigma, 3) * (1.0, 1.0, 0.1)
        return lambda _prior: (pose, INFO, 0.9)


def test_jacobian_matches_numerical():
    for x in (np.array([1.0, 2.0, 0.3, 0.8, 0.5, 0.001]), np.array([0.0, 0.0, -2.9, -0.4, -1.2, 0.0])):
        F = motion_jacobian(x, DT)
        num = np.zeros((6, 6))
        for j in range(6):
            e = np.zeros(6)
            e[j] = 1e-6
            d = motion(x + e, DT) - motion(x - e, DT)
            d[2] = base_ekf.wrap(d[2])
            num[:, j] = d / 2e-6
        assert np.abs(F - num).max() < 1e-6


def test_bias_learnt_standing_still():
    robot, ekf = Robot(bias=0.004), BaseEKF((0.0, 0.0, 0.0), R, TRACK)
    for _ in range(500):
        ekf.tick(*robot.step(0.0, 0.0))
    assert abs(ekf.x[B] - 0.004) < 3e-4
    assert abs(ekf.pose[2]) < np.radians(0.05)


def test_bias_learnt_while_turning():
    robot, ekf = Robot(bias=-0.003), BaseEKF((0.0, 0.0, 0.0), R, TRACK)
    for k in range(1500):  # 30 s on a circle, a scan every third tick
        ekf.tick(*robot.step(0.3, 0.4))
        if k % 3 == 0:
            ekf.scan(robot.t, robot.match())
    assert abs(ekf.x[B] + 0.003) < 5e-4
    assert np.hypot(*(ekf.pose[:2] - robot.pose[:2])) < 0.01


def test_outlier_scan_rejected():
    robot, ekf = Robot(), BaseEKF((0.0, 0.0, 0.0), R, TRACK)
    for k in range(300):
        ekf.tick(*robot.step(0.5, 0.0))
        if k % 3 == 0:
            ekf.scan(robot.t, robot.match())
    before = ekf.pose
    wrong = robot.pose + (0.2, 0.0, 0.0)
    assert ekf.scan(robot.t, lambda _p: (wrong, INFO, 0.9)) == "rejected"
    assert np.allclose(ekf.pose, before)
    assert ekf.scan(robot.t, lambda _p: (robot.pose, INFO, 0.2)) == "poor"


def test_late_scan_equals_in_order():
    robot, ticks, truth = Robot(bias=0.002), [], []
    for k in range(200):
        ticks.append(robot.step(0.6, 0.3 * np.sin(0.05 * k)))
        truth.append(robot.pose.copy())
    scan_at = 150
    measured = truth[scan_at] + np.random.default_rng(1).normal(0.0, 0.003, 3) * (1.0, 1.0, 0.1)
    match = lambda _prior: (measured, INFO, 0.9)  # noqa: E731
    in_order, late = BaseEKF((0.0, 0.0, 0.0), R, TRACK), BaseEKF((0.0, 0.0, 0.0), R, TRACK)
    for k, tk in enumerate(ticks):
        in_order.tick(*tk)
        late.tick(*tk)
        if k == scan_at:
            assert in_order.scan(tk[0], match) == "accepted"
    assert late.scan(ticks[scan_at][0], match) == "accepted"
    assert np.abs(in_order.x - late.x).max() < 1e-9
    assert np.abs(in_order.P - late.P).max() < 1e-12


def test_skid_widens_the_covariance():
    robot, ekf = Robot(), BaseEKF((0.0, 0.0, 0.0), R, TRACK)
    for k in range(200):
        ekf.tick(*robot.step(0.8, 0.0))
        if k % 3 == 0:
            ekf.scan(robot.t, robot.match())
    sigma, skids = np.sqrt(ekf.P[0, 0]), ekf.counts["skids"]
    t, _wheels, gyro = robot.step(0.8, 0.0)
    ekf.tick(t, ekf.prev_wheels, gyro)  # the wheels locked: no arc this tick
    assert ekf.counts["skids"] == skids + 1
    assert np.sqrt(ekf.P[0, 0]) > 10 * sigma


def test_scans_followed_through_a_skid():
    robot, ekf = Robot(), BaseEKF((0.0, 0.0, 0.0), R, TRACK)
    for k in range(150):
        ekf.tick(*robot.step(min(0.6, 0.01 * k), 0.0))
        if k % 3 == 0:
            ekf.scan(robot.t, robot.match())
    locked, v, statuses = ekf.prev_wheels.copy(), 0.6, []
    for k in range(60):  # the wheels lock; the chassis slides to a stop at 1.5 m/s^2
        v = max(v - 1.5 * DT, 0.0)
        t, _wheels, gyro = robot.step(v, 0.0)
        ekf.tick(t, locked, gyro)
        if k % 3 == 0:
            statuses.append(ekf.scan(t, robot.match()))
    assert "rejected" not in statuses and "reset" not in statuses
    assert np.hypot(*(ekf.pose[:2] - robot.pose[:2])) < 0.01
