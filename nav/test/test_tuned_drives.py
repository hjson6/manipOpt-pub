"""The tuned constants in whole drives (NavWorld, as nav_sim): at today's values they
drive exactly as the rules (the job's crowd of four: the fixture's trajectory, tick for
tick); other weights drive differently. About a minute."""
import json
from pathlib import Path

import pytest

from nav import tuned as nav_tuned
from pick_place_common.mobile_scenarios import CROWDS, walking
from pick_place_common.nav_world import NavWorld

REPO = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
TODAY = {"near_m": (0.9, 1.8, 3.0), "near_speed": (0.2, 0.5), "person_radius": 0.45, "w_lag": 5.0, "w_lat": 40.0,
         "w_heading": 4.0}


def world():
    return NavWorld(REPO / "data" / "maps" / "stations", seed=0, crowd=walking(CROWDS["job"][:4]), arm="carry",
                    box=True, fusion="ekf")


def drive(w, n):
    for i in range(n):
        w.drive(("pick", "place")[i % 2])
    return w.digest.hexdigest()


@pytest.fixture(autouse=True)
def at_repo(monkeypatch):
    monkeypatch.chdir(REPO)  # the solver's build directory


def test_tuned_at_todays_constants_drives_as_the_rules():
    w = world()
    nav_tuned.apply_tuned(w.nav, TODAY)
    assert drive(w, 4) == json.loads((FIXTURES / "nav_sim_job4.json").read_text())["trajectory_sha256"]


def test_tuned_weights_change_the_drive():
    w = world()
    nav_tuned.apply_tuned(w.nav, {**TODAY, "w_lag": 10.0, "w_lat": 20.0})
    assert drive(w, 2) != drive(world(), 2)
