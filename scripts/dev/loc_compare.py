"""Tables of nav_sim's localization scores (--json files): each run, each estimator.
usage: python loc_compare.py <dir or files...>
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    args = ap.parse_args()
    files = sorted(f for p in map(Path, args.paths) for f in (p.glob("*.json") if p.is_dir() else [p]))
    print("| run | drives | estimator | position p50 / p95 / max mm | yaw max deg | jump max mm | lost | NEES in band "
          "| docked lateral / along mm, yaw deg | ms per scan |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for f in files:
        d = json.load(open(f))
        drove = d["args"].get("fusion", "icp")
        for name, s in d["localization"].items():
            dock = (f"{d['dock_lateral_max']:.1f} / {d['dock_along_max']:.1f}, {d['dock_yaw_max']:.2f}"
                    if name == drove and "dock_lateral_max" in d else "")
            nees = f"{100 * s['nees_in_band']:.0f}%" if "nees_in_band" in s else ""
            lost = s.get("lost", 0) + s.get("rejected", 0)
            print(f"| {f.stem} | {d['arrived']}/{d['drives']} | {name}{' (drives)' if name == drove else ''} | "
                  f"{s['pos_p50']:.1f} / {s['pos_p95']:.1f} / {s['pos_max']:.1f} | {s['yaw_max']:.2f} | "
                  f"{s['jump_max']:.1f} | {lost} | {nees} | {dock} | {s.get('ms_mean', float('nan')):.2f} |")


if __name__ == "__main__":
    main()
