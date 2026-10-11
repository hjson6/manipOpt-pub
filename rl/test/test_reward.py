"""The training reward (rl/reward.py) on whole drives: the time term is the drive's time,
a quiet room adds no closeness, hit or timeout; with people the terms are finite."""
from pathlib import Path

import numpy as np
import pytest

from pick_place_common.mobile_scenarios import CROWDS, walking
from pick_place_common.nav_world import DT, NavWorld
from rl.reward import DriveReward

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def at_repo(monkeypatch):
    monkeypatch.chdir(REPO)


def run(n_people, drives=2):
    w = NavWorld(REPO / "data" / "maps" / "stations", seed=0, crowd=walking(CROWDS["job"][:n_people]), arm="carry",
                 box=True, fusion="ekf", fast=True)
    out = []
    for i in range(drives):
        w.start(("pick", "place")[i % 2])
        rew = DriveReward()
        rew.start(w)
        total = 0.0
        while True:
            w.tick()
            total += rew.tick(w)
            if w.done or rew.hit:
                break
        score = w.finish()
        out.append((score, rew.finish(score), total, dict(rew.terms)))
    return out


def test_quiet_drive_costs_its_time():
    for score, ret, total, terms in run(0):
        assert terms["time"] == pytest.approx(-score.r["t"], abs=DT)
        assert terms["close"] == terms["hit"] == terms["timeout"] == 0.0
        assert ret == pytest.approx(total) and ret < 0.0


def test_crowd_terms_finite():
    for score, ret, total, terms in run(4):
        assert np.isfinite(ret) and ret <= terms["time"] + 1e-9
        assert terms["stops"] == pytest.approx(-2.0 * score.r["stop_onsets"])
