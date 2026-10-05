"""Unit tests for perception/map_people.py and people_tracker.py on a synthetic room."""
import numpy as np

from perception.map_people import MapPeopleDetector
from perception.people_tracker import PeopleTracker

ANGLES = np.radians(np.arange(-135.0, 135.0 + 0.25, 0.5))
MOUNTS = [(0.43, 0.31, 0.2, np.radians(45)), (-0.43, -0.31, 0.2, np.radians(-135))]
ROOM = (-3.0, 3.0, -2.0, 2.0)
LEG_R = 0.06


def wall_points(step=0.02):
    x0, x1, y0, y1 = ROOM
    xs, ys = np.arange(x0, x1, step), np.arange(y0, y1, step)
    return np.vstack([np.column_stack([xs, np.full_like(xs, y0)]), np.column_stack([xs, np.full_like(xs, y1)]),
                      np.column_stack([np.full_like(ys, x0), ys]), np.column_stack([np.full_like(ys, x1), ys])])


def scans(pose, legs=()):
    """Ranges per scanner from the robot at pose: the room's walls and leg circles."""
    c, s = np.cos(pose[2]), np.sin(pose[2])
    out = []
    for mx, my, _z, myaw in MOUNTS:
        o = np.array([pose[0] + c * mx - s * my, pose[1] + s * mx + c * my])
        a = pose[2] + myaw + ANGLES
        d = np.column_stack([np.cos(a), np.sin(a)])
        with np.errstate(divide="ignore", invalid="ignore"):
            tx = np.where(d[:, 0] > 0, (ROOM[1] - o[0]) / d[:, 0], (ROOM[0] - o[0]) / d[:, 0])
            ty = np.where(d[:, 1] > 0, (ROOM[3] - o[1]) / d[:, 1], (ROOM[2] - o[1]) / d[:, 1])
        r = np.minimum(np.abs(tx), np.abs(ty))
        for lx, ly in legs:
            v = np.array([lx, ly]) - o
            b = d @ v
            disc = b ** 2 - (v @ v - LEG_R ** 2)
            hit = (disc > 0) & (b > 0)
            r = np.where(hit, np.minimum(r, b - np.sqrt(np.maximum(disc, 0))), r)
        out.append(r)
    return out


def person_legs(x, y):
    return [(x, y + 0.1), (x, y - 0.1)]


def test_a_person_is_found_and_the_walls_are_not():
    det = MapPeopleDetector(wall_points(), MOUNTS, ANGLES)
    pose = np.array([-1.0, 0.0, 0.0])
    people = det.update(pose, scans(pose, person_legs(1.0, 0.5)))
    assert len(people) == 1
    assert np.hypot(people[0][0] - 1.0, people[0][1] - 0.5) < 0.08
    # a small localization error does not turn the walls into people
    assert det.update(pose + (0.03, -0.02, np.radians(0.5)), scans(pose)) == []


def test_track_velocity_and_no_ghost_after_a_turn():
    det = MapPeopleDetector(wall_points(), MOUNTS, ANGLES)
    trk = PeopleTracker()
    pose = np.array([-2.0, -1.0, 0.0])
    t = 0.0
    for k in range(30):  # walking +x at 1 m/s
        t = k / 15
        tracks = trk.update(t, det.update(pose, scans(pose, person_legs(-0.5 + t, 0.0))), det.seen_empty)
    assert len(tracks) == 1 and abs(tracks[0].x[2] - 1.0) < 0.15 and abs(tracks[0].x[3]) < 0.15
    x_turn = -0.5 + t
    for k in range(1, 15):  # turns to +y
        tracks = trk.update(t + k / 15, det.update(pose, scans(pose, person_legs(x_turn, k / 15))), det.seen_empty)
    assert len(tracks) == 1  # the old track did not walk on in +x
