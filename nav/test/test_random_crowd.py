"""Random crowds for training (mobile_scenarios.random_crowd): clear of the room's walls,
tables and fixtures, at most the scene's six people, nobody starting on the robot or on
each other, the same crowd from the same seed. Loads the stations scene (MuJoCo), no
solver."""
import numpy as np
import pytest

from pick_place_common.mobile_scenarios import (
    CLEAR_M, LINE_KEEP_M, PERSON_R, POINT_APART_M, RANDOM_SPEED, START_APART_M, START_FROM_HOME_M, WALL_KEEP_M, Crowd,
    gap_path, line_gap, random_crowd, room_obstacles)
from pick_place_common.mujoco_sim_node import load_scene_model
from pick_place_common.scene import BASE_HOME_POSE, ROOM_SIZE

SEEDS = range(300)
ROUNDING_M = 1e-3  # the points are kept to the millimetre


@pytest.fixture(scope="module")
def rects():
    return room_obstacles(load_scene_model(layout="stations"))


def crowds(rects, **kw):
    return [random_crowd(np.random.default_rng(seed), rects, **kw) for seed in SEEDS]


def test_paths_clear_of_the_room(rects):
    for specs in crowds(rects):
        for name, path, *_ in specs:
            walked = list(path) + list(path[:1]) if len(path) > 2 else list(path)
            assert gap_path(walked, rects)[0] >= PERSON_R + CLEAR_M - ROUNDING_M, (name, path)
            assert all(np.hypot(*np.subtract(a, b)) >= POINT_APART_M - ROUNDING_M
                       for a, b in zip(walked[:-1], walked[1:]) if len(walked) > 2), (name, path)


def test_walkers_stand_off_the_docking_lines(rects):
    for specs in crowds(rects):
        blockers = [s for s in specs if line_gap(s[1][0]) < LINE_KEEP_M - ROUNDING_M]
        assert len(blockers) <= 1
        for s in blockers:  # the one who stands in a docking line: drawn last, as dock_block
            assert s is specs[-1] and len(s[1]) == 2 and not s[5] and s[3] >= 15.0
        for s in specs:
            if len(s) > 6 or s in blockers:
                continue  # the one who steps in, the one in a docking line
            assert all(line_gap(q) >= LINE_KEEP_M - ROUNDING_M for q in s[1]), s


def test_people_and_their_starts(rects):
    for specs in crowds(rects):
        assert 1 <= len(specs) <= 6
        assert [s[0] for s in specs] == [f"person_{i + 1}" for i in range(len(specs))]
        starts = [np.array(s[1][0]) for s in specs]
        for i, p in enumerate(starts):
            assert np.hypot(p[0] - BASE_HOME_POSE[0], p[1] - BASE_HOME_POSE[1]) >= START_FROM_HOME_M - ROUNDING_M
            assert all(np.linalg.norm(p - q) >= START_APART_M - ROUNDING_M for q in starts[:i])
        assert all(RANDOM_SPEED[0] <= s[2] <= RANDOM_SPEED[1] for s in specs)


def test_at_most_n_max(rects):
    assert all(len(specs) <= 2 for specs in crowds(rects, n_max=2))


def test_the_same_seed_the_same_crowd(rects):
    assert random_crowd(np.random.default_rng(7), rects) == random_crowd(np.random.default_rng(7), rects)
    assert random_crowd(np.random.default_rng(7), rects) != random_crowd(np.random.default_rng(8), rects)


def test_every_kind_drawn(rects):
    specs = [s for c in crowds(rects) for s in c]
    step_in = sum(len(s) > 6 for c in crowds(rects) for s in c) / len(SEEDS)
    assert 0.12 < step_in < 0.28  # 20% of crowds
    assert 0.5 < np.mean([s[5] for s in specs if len(s) == 6]) < 0.75  # 60% give way, the dock blockers do not
    assert {len(s[1]) for s in specs if len(s) == 6} >= {2, 3, 4}


def test_a_crowd_walks_inside_the_room(rects):
    specs = random_crowd(np.random.default_rng(3), rects)
    crowd = Crowd(specs)
    first = np.array([p[:2] for p in crowd.step(0.0, BASE_HOME_POSE[:2])])
    for _ in range(3000):  # 60 s, the robot parked at home
        here = np.array([p[:2] for p in crowd.step(0.02, BASE_HOME_POSE[:2])])
        assert np.all(here >= WALL_KEEP_M - 1e-9) and np.all(here <= np.asarray(ROOM_SIZE) - WALL_KEEP_M + 1e-9)
    assert np.max(np.linalg.norm(here - first, axis=1)) > 0.5
