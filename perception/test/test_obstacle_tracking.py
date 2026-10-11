import numpy as np

from perception import obstacle_tracking as ot


def _feed(tracker, positions, dt=0.1, t0=0.0, r=0.1):
    for k, p in enumerate(positions):
        tracker.update(t0 + k * dt, np.array([[p[0], p[1], 0.25, r]]))  # a carton-height top (< PERSON_MIN_HEIGHT_M)
    return tracker.tracks


def test_walker_is_person_and_keeps_identity():
    tk = ot.Tracker()
    tracks = _feed(tk, [(1.0 - 0.16 * k, -0.12) for k in range(15)])
    assert len(tracks) == 1 and tracks[0].id == 1
    assert tracks[0].label == ot.PERSON
    assert abs(tracks[0].speed - 1.6) < 0.2


def test_motionless_becomes_static_only_after_confirm_window():
    tk = ot.Tracker()
    tracks = _feed(tk, [(0.55, -0.12)] * 5)
    assert tracks[0].label == ot.PERSON  # unconfirmed: fail-safe
    tracks = _feed(tk, [(0.55, -0.12)] * 60, t0=0.5)
    assert tracks[0].label == ot.STATIC


def test_person_who_stops_stays_person():
    tk = ot.Tracker()
    _feed(tk, [(1.0 - 0.16 * k, -0.12) for k in range(6)])
    x_stop = 1.0 - 0.16 * 5
    tracks = _feed(tk, [(x_stop, -0.12)] * 100, t0=0.6)  # stands still for 10 s
    assert tracks[0].label == ot.PERSON


def test_tall_still_object_is_a_person():
    tk = ot.Tracker()
    for k in range(60):
        tk.update(k * 0.1, np.array([[0.55, -0.12, 1.45, 0.25, 400, 1.68]]))
    assert tk.tracks[0].label == ot.PERSON


def test_static_that_starts_moving_reverts_to_person():  # and then stays one
    tk = ot.Tracker()
    _feed(tk, [(0.55, -0.12)] * 40)
    assert tk.tracks[0].label == ot.STATIC
    _feed(tk, [(0.55 - 0.1 * k, -0.12) for k in range(1, 6)], t0=4.0)
    assert tk.tracks[0].label == ot.PERSON


def test_track_coasts_through_dropout_then_dies():
    tk = ot.Tracker()
    _feed(tk, [(0.55, -0.12)] * 5)
    tk.update(0.9, np.zeros((0, 4)))
    assert len(tk.tracks) == 1 and not tk.tracks[0].visible
    tk.update(2.5, np.zeros((0, 4)))
    assert tk.tracks == []


def test_protective_distance_grows_with_robot_speed_and_latency():
    r0, _ = ot.protective_distance(1.6, 0.0, 0.1, 3.0, 0.05)
    r1, ts = ot.protective_distance(1.6, 0.5, 0.1, 3.0, 0.05)
    assert r1 > r0 and abs(ts - 0.5 / 3.0) < 1e-9
    assert ot.protective_distance(1.6, -1.0, 0.1, 3.0, 0.05)[0] == r0  # moving away = stationary


def test_speed_scale_is_monotone_and_holds_below_threshold():
    req = 0.4
    gaps = np.linspace(0.0, 2.0, 50)
    s = [ot.speed_scale(g, req, 0.6) for g in gaps]
    assert all(b >= a for a, b in zip(s, s[1:]))
    assert ot.speed_scale(0.3, req, 0.6) == 0.0
    assert s[-1] == 1.0


def test_static_survives_occlusion_jitter_but_not_a_real_push():
    tk = ot.Tracker()
    _feed(tk, [(0.55, -0.12)] * 40)
    assert tk.tracks[0].label == ot.STATIC
    # arm passes over: centroid wanders +-6 cm for a few frames
    _feed(tk, [(0.55, -0.12), (0.55, -0.18), (0.55, -0.12), (0.55, -0.17), (0.55, -0.12)], t0=4.0)
    assert tk.tracks[0].label == ot.STATIC
    _feed(tk, [(0.55 - 0.1 * k, -0.12) for k in range(1, 9)], t0=4.5)
    assert tk.tracks[0].label == ot.PERSON


def test_static_reports_latched_estimate_and_survives_long_occlusion():
    tk = ot.Tracker()
    _feed(tk, [(0.55, -0.12)] * 40, r=0.09)
    tr = tk.tracks[0]
    assert tr.label == ot.STATIC and abs(tr.static_r - 0.09) < 1e-9
    _feed(tk, [(0.58, -0.16)] * 5, t0=4.0, r=0.04)  # occluded, shrunken, shifted blob
    assert np.allclose(tr.static_pos[:2], [0.55, -0.12]) and abs(tr.static_r - 0.09) < 1e-9
    tk.update(9.0, np.zeros((0, 4)))  # fully hidden for ~5 s: still remembered
    assert len(tk.tracks) == 1
    tk.update(20.0, np.zeros((0, 4)))
    assert tk.tracks == []
