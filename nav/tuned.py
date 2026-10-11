"""The navigation's constants tuned by optimizer (nav_policy tuned; rl/tune_baseline.py):
where the base slows for people and how fast, the room the MPC keeps, the MPC's tracking
weights, within bounds: never bolder than the close setting (0.6 x today's distances),
never faster near people than today. The MPC and its constraints are today's; only these
numbers change. `rules` (the default) is today's code path, untouched. Framework-free.
"""
import json
from pathlib import Path

import numpy as np

from core.learned_policy import PolicyFileError
from nav.navigator import NEAR_M, NEAR_SPEED, PERSON_OWN_M, PERSON_RADIUS_M

MODES = ("rules", "tuned")
POLICY_DIR = Path(__file__).resolve().parents[1] / "policies" / "nav"
TUNED_DEFAULT = "tuned_v1"
CLOSE, CAUTIOUS = 0.6, 1.5  # the people distances' scale: the close and the cautious settings
TUNED_BOUNDS = {
    "near_m": (CLOSE * np.array(NEAR_M), CAUTIOUS * np.array(NEAR_M)),
    "near_speed": (np.array([0.05, 0.1]), np.array(NEAR_SPEED)),
    "person_radius": (PERSON_OWN_M + (PERSON_RADIUS_M - PERSON_OWN_M) * CLOSE,
                      PERSON_OWN_M + (PERSON_RADIUS_M - PERSON_OWN_M) * CAUTIOUS),
    "w_lag": (1.25, 20.0), "w_lat": (10.0, 160.0), "w_heading": (1.0, 16.0),
}


def clamp_tuned(params):
    """The tuned constants inside their bounds, distances and speeds in increasing order."""
    out = {}
    for key, (lo, hi) in TUNED_BOUNDS.items():
        v = np.asarray(params[key], dtype=float)
        v = np.clip(np.sort(v) if v.ndim else v, lo, hi)  # the bounds increase too: still in order
        out[key] = tuple(float(x) for x in v) if v.ndim else float(v)
    return out


def tuned_name(model):
    return model if model and model.startswith("tuned") else TUNED_DEFAULT


def load_tuned(name=TUNED_DEFAULT, policy_dir=POLICY_DIR):
    path = Path(policy_dir) / f"{name}.json"
    if not path.exists():
        raise PolicyFileError(f"nav_policy tuned: {path} not found (rl/tune_baseline.py makes it)")
    return clamp_tuned(json.loads(path.read_text())["params"])


def apply_tuned(nav, params):
    """Today's rules with the tuned constants: where the base slows and how fast, the room
    the MPC keeps, the MPC's tracking weights."""
    p = clamp_tuned(params)
    nav.near_m, nav.near_speed, nav.person_radius = p["near_m"], p["near_speed"], p["person_radius"]
    nav.mpc.set_weights(p["w_lag"], p["w_lat"], p["w_heading"])


def select(mode, model="", base_safety=1.0, policy_dir=POLICY_DIR):
    """The tuned constants for a nav_policy mode (None: the rules). Refuses (PolicyFileError)
    an unknown mode, a missing file, or tuned at a launch base_safety other than the
    default (its distances are absolute)."""
    if mode not in MODES:
        raise PolicyFileError(f"nav_policy '{mode}': one of {', '.join(MODES)}")
    if mode == "rules":
        return None
    if base_safety != 1.0:
        raise PolicyFileError(f"nav_policy tuned: base_safety must be 1.0 (its distances are absolute; got "
                              f"{base_safety:g})")
    return load_tuned(tuned_name(model), policy_dir)
