"""The tuned baseline of layer 1 (nav_policy tuned; handover_notes/rl/rl_crowd_nav_plan.md,
step 3): CMA-ES over today's navigation constants (where the base slows and how fast, the
room the MPC keeps, the MPC's tracking weights) inside the learned policy's envelope
(never bolder than the close setting, never faster near people; nav/tuned.py's TUNED_BOUNDS),
on the same objective as its reward (rl/reward.py), in random crowds on training seeds. If
a learned policy beats this, its gain is reading the situation, not better constants.
Each generation evaluates every candidate and today's constants on the same seeds; the
candidates kept are then run on the validation seeds against the rules, and the best is
written to policies/nav/<name>.json. The log: <out>/<name>.json (git-ignored).
usage: python rl/tune_baseline.py [--generations 25] [--seeds 8] [--drives 6] [--popsize 10] [--workers 12]
       [--name tuned_v1] [--out rl/runs/tune]
(in the training environment: source scripts/env_rl.sh)
"""
import argparse
import json
import os
import subprocess
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "tasks/pick_and_place/common")]
from nav.tuned import TUNED_BOUNDS, clamp_tuned  # noqa: E402
from nav.navigator import NEAR_M, NEAR_SPEED, PERSON_RADIUS_M  # noqa: E402

TODAY = {"near_m": NEAR_M, "near_speed": NEAR_SPEED, "person_radius": PERSON_RADIUS_M, "w_lag": 5.0, "w_lat": 40.0,
         "w_heading": 4.0}
LAYOUT = [("near_m", 3), ("near_speed", 2), ("person_radius", 1), ("w_lag", 1), ("w_lat", 1), ("w_heading", 1)]
LOG_SCALE = ("w_lag", "w_lat", "w_heading")
TRAIN_SEED0, VALIDATION_SEEDS = 1000, range(500, 510)
N_MAX = 6  # people in a random crowd, at most


def _span(key):
    lo, hi = (np.atleast_1d(np.asarray(b, dtype=float)) for b in TUNED_BOUNDS[key])
    return (np.log(lo), np.log(hi)) if key in LOG_SCALE else (lo, hi)


def encode(params):
    """Constants -> [0, 1]^9 (the weights on a log scale)."""
    out = []
    for key, _n in LAYOUT:
        v = np.atleast_1d(np.asarray(params[key], dtype=float))
        lo, hi = _span(key)
        out.extend(((np.log(v) if key in LOG_SCALE else v) - lo) / (hi - lo))
    return np.clip(np.array(out), 0.0, 1.0)


def decode(x):
    out, i = {}, 0
    for key, n in LAYOUT:
        lo, hi = _span(key)
        v = lo + np.asarray(x[i:i + n]) * (hi - lo)
        v = np.exp(v) if key in LOG_SCALE else v
        out[key] = tuple(v) if n > 1 else float(v[0])
        i += n
    return clamp_tuned(out)


def _init():
    os.chdir(REPO)  # the base MPC's build directory


def evaluate(job):
    """(constants or None for today's, seed, drives) -> the drives' mean return and metrics."""
    params, seed, drives = job
    from nav import tuned as nav_tuned
    from pick_place_common.mobile_scenarios import random_crowd
    from pick_place_common.nav_world import NavWorld
    from rl.reward import DriveReward
    w = NavWorld(REPO / "data" / "maps" / "stations", seed=seed, arm="carry", box=True, fusion="ekf", fast=True,
                 crowd=lambda rects: random_crowd(np.random.default_rng([seed, 7]), rects, N_MAX))
    if params is not None:
        nav_tuned.apply_tuned(w.nav, params)
    reward, rets, times, stops = DriveReward(), [], [], []
    for i in range(drives):
        w.start(("pick", "place")[i % 2])
        reward.start(w)
        while True:
            w.tick()
            reward.tick(w)
            if w.done or reward.hit:
                break
        score = w.finish()
        rets.append(reward.finish(score))
        times.append(score.r["t"])
        stops.append(score.r["stop_onsets_by"].get("route", 0))
    return {"ret": float(np.mean(rets)), "time": float(np.mean(times)), "route_stops": float(np.mean(stops)),
            "arrived": sum(d.r["state"] in ("docked", "parked") for d in w.drives),
            "hits": sum(d.r["intrusions"] + d.r["contacts"] for d in w.drives)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=25)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--drives", type=int, default=6)
    ap.add_argument("--popsize", type=int, default=10)
    ap.add_argument("--sigma", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--validation-drives", type=int, default=10)
    ap.add_argument("--name", default="tuned_v1")
    ap.add_argument("--out", default="rl/runs/tune")
    args = ap.parse_args()
    import cma
    t_wall = time.perf_counter()
    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / f"{args.name}.json"
    es = cma.CMAEvolutionStrategy(encode(TODAY), args.sigma,
                                  {"bounds": [0.0, 1.0], "popsize": args.popsize, "seed": 1, "verbose": -9})
    pool = get_context("spawn").Pool(args.workers, initializer=_init)
    log, kept = [], []
    for g in range(args.generations):
        xs = es.ask()
        seeds = [TRAIN_SEED0 + g * args.seeds + i for i in range(args.seeds)]
        cands = [decode(x) for x in xs] + [None]
        res = pool.map(evaluate, [(c, s, args.drives) for c in cands for s in seeds])
        ret = np.array([r["ret"] for r in res]).reshape(len(cands), len(seeds))
        es.tell(xs, list(-ret[:-1].mean(axis=1)))
        best = int(np.argmax(ret[:-1].mean(axis=1)))
        kept.append(cands[best])
        hits = sum(r["hits"] for r in res)
        log.append({"generation": g, "seeds": seeds, "rules": float(ret[-1].mean()),
                    "candidates": [{"params": c, "ret": float(m)} for c, m in zip(cands[:-1], ret[:-1].mean(axis=1))],
                    "mean": decode(es.mean), "sigma": float(es.sigma), "hits": hits})
        print(f"generation {g + 1}/{args.generations}: best {ret[best].mean():+.2f} against the rules "
              f"{ret[-1].mean():+.2f} on the same seeds (paired gain {(ret[best] - ret[-1]).mean():+.2f}); sigma "
              f"{es.sigma:.3f}; hits {hits}; {time.perf_counter() - t_wall:.0f} s", flush=True)
        log_path.write_text(json.dumps({"args": vars(args), "log": log}, indent=1) + "\n")
    # Validation: the rules, the search's mean, the last generations' bests.
    finals = {"rules": None, "mean": decode(es.mean)}
    finals.update({f"best_g{len(kept) - k}": kept[-k] for k in (1, 2, 3) if len(kept) >= k})
    names = list(finals)
    res = pool.map(evaluate, [(finals[n], s, args.validation_drives) for n in names for s in VALIDATION_SEEDS])
    pool.close()
    val = {}
    for i, n in enumerate(names):
        rs = res[i * len(VALIDATION_SEEDS):(i + 1) * len(VALIDATION_SEEDS)]
        val[n] = {k: float(np.mean([r[k] for r in rs])) for k in ("ret", "time", "route_stops")}
        val[n].update(arrived=int(sum(r["arrived"] for r in rs)), hits=int(sum(r["hits"] for r in rs)),
                      params=finals[n])
        print(f"validation {n}: return {val[n]['ret']:+.2f}, drive {val[n]['time']:.1f} s, route stops "
              f"{val[n]['route_stops']:.2f}, arrived {val[n]['arrived']}, hits {val[n]['hits']}")
    ok = [n for n in names if n != "rules" and val[n]["hits"] == 0]
    pick = max(ok, key=lambda n: val[n]["ret"]) if ok else None
    log_path.write_text(json.dumps({"args": vars(args), "log": log, "validation": val, "pick": pick}, indent=1) + "\n")
    if pick is None:
        print("no candidate without a hit on validation: nothing written")
        return
    commit = subprocess.run(["git", "describe", "--always", "--dirty"], cwd=REPO, capture_output=True,
                            text=True).stdout.strip()
    manifest = {"made_by": "rl/tune_baseline.py", "commit": commit, "picked": pick, "objective": "rl/reward.py",
                "crowds": f"random_crowd, up to {N_MAX} people", "generations": args.generations,
                "popsize": args.popsize,
                "training_seeds": f"{TRAIN_SEED0}-{TRAIN_SEED0 + args.generations * args.seeds - 1}",
                "drives_per_seed": args.drives, "validation_seeds": "500-509", "validation": val,
                "wall_s": round(time.perf_counter() - t_wall)}
    path = REPO / "policies" / "nav" / f"{args.name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"params": finals[pick], "manifest": manifest}, indent=1) + "\n")
    print(f"picked {pick}: {finals[pick]} -> {path}")


if __name__ == "__main__":
    main()
