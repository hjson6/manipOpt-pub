"""The tuned constants' file and bounds (nav/tuned.py) and the policy files' loader
(core/learned_policy.py): clamped to the envelope, refused when missing or altered."""
import json

import numpy as np
import pytest

from core.learned_policy import LearnedPolicy, PolicyFileError, save
from nav import tuned as nav_tuned


def policy_file(tmp_path, name="p", obs_dim=8, act_dim=2, version=1):
    rng = np.random.default_rng(0)
    dims = (obs_dim, 16, 16, act_dim)
    layers = [(rng.normal(0, 0.3, (b, a)), rng.normal(0, 0.1, b)) for a, b in zip(dims[:-1], dims[1:])]
    save(tmp_path / f"{name}.npz", layers, np.zeros(obs_dim), np.ones(obs_dim), 1e-8, 10.0, version, {"note": "test"})
    return tmp_path / f"{name}.npz"


def test_selector(tmp_path):
    assert nav_tuned.select("rules") is None
    with pytest.raises(PolicyFileError, match="one of"):
        nav_tuned.select("rl")
    with pytest.raises(PolicyFileError, match="tuned_v1.json not found"):
        nav_tuned.select("tuned", policy_dir=tmp_path)
    (tmp_path / "tuned_v1.json").write_text(json.dumps({"params": {
        "near_m": [0.1, 9.0, 2.0], "near_speed": [0.5, 0.9], "person_radius": 0.2, "w_lag": 5.0, "w_lat": 40.0,
        "w_heading": 4.0}}))
    with pytest.raises(PolicyFileError, match="base_safety"):
        nav_tuned.select("tuned", base_safety=0.6, policy_dir=tmp_path)
    p = nav_tuned.select("tuned", policy_dir=tmp_path)  # clamped: never bolder than the close setting, never faster
    assert p["near_m"] == pytest.approx((0.54, 2.0, 4.5)) and p["near_speed"] == pytest.approx((0.2, 0.5))
    assert p["person_radius"] == pytest.approx(0.39)


def test_learned_policy_runs_numpy_mlp(tmp_path):
    pol = LearnedPolicy(policy_file(tmp_path), feature_version=1)
    a = pol(np.ones(8))
    assert a.shape == (2,) and np.all(np.abs(a) <= 1.0) and np.array_equal(a, pol(np.ones(8)))


def test_policy_files_refused(tmp_path):
    with pytest.raises(PolicyFileError, match="not found"):
        LearnedPolicy(tmp_path / "nothing.npz")
    path = policy_file(tmp_path)
    man = path.with_suffix(".json")
    man.write_text(json.dumps({**json.loads(man.read_text()), "sha256": "0" * 64}))
    with pytest.raises(PolicyFileError, match="sha256"):
        LearnedPolicy(path)
    with pytest.raises(PolicyFileError, match="features"):
        LearnedPolicy(policy_file(tmp_path, "old", version=0), feature_version=1)
