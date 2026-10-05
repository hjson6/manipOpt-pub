"""Lidar people detection on synthetic scans: legs as circles, a wall as background.
Run with: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest perception/test
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO)]
from perception.lidar_detection import BACKGROUND_SCANS, LidarPeopleDetector  # noqa: E402

POSE = (0.0, 0.0, 0.0, 0.0)
ANGLES = np.radians(np.arange(-135.0, 135.25, 0.5))
WALL_M = 6.0


def ranges(legs=(), noise=0.0, rng=None):
    """A circle of WALL_M round the scanner, and legs (x, y, r) in front of it."""
    r = np.full(len(ANGLES), WALL_M)
    d = np.column_stack([np.cos(ANGLES), np.sin(ANGLES)])
    for x, y, rad in legs:
        c = np.array([x, y])
        b = d @ c
        disc = b ** 2 - (c @ c - rad ** 2)
        hit = disc >= 0
        r = np.where(hit, np.minimum(r, b - np.sqrt(np.where(hit, disc, 0.0))), r)
    if noise:
        r = r + rng.normal(0.0, noise, len(r))
    return r


def learnt(noise=0.0, rng=None):
    det = LidarPeopleDetector([POSE], ANGLES)
    for _ in range(BACKGROUND_SCANS):
        det.update(0, ranges(noise=noise, rng=rng))
    assert det.ready
    return det


def test_two_legs_make_one_person_centred_between_them():
    det = learnt()
    det.update(0, ranges([(2.0, 0.15, 0.06), (2.0, -0.15, 0.06)]))
    (x, y, radius, legs), = det.people()
    assert legs == 2
    assert np.hypot(x - 2.0, y) < 0.02
    assert 0.15 < radius < 0.25


def test_empty_noisy_scans_give_no_people():
    rng = np.random.default_rng(0)
    det = learnt(0.02, rng)
    for _ in range(100):
        det.update(0, ranges(noise=0.02, rng=rng))
        assert det.people() == []


def test_two_people_apart_stay_apart():
    det = learnt()
    det.update(0, ranges([(2.0, 1.0, 0.06), (2.0, 0.8, 0.06), (2.0, -1.0, 0.06), (2.0, -0.8, 0.06)]))
    people = sorted(det.people(), key=lambda p: p[1])
    assert len(people) == 2
    assert np.hypot(people[0][0] - 2.0, people[0][1] + 0.9) < 0.03
    assert np.hypot(people[1][0] - 2.0, people[1][1] - 0.9) < 0.03
