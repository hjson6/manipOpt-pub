"""Placement error budget (tight_packing_plan.md step 1), from launch.logs with the
validation lines ("placement aim" from task_node, "truth at release" and the settled
pose from mujoco_sim_node). Per set-down, arm frame, x/y mm:
  landing = settled box - aimed box = track + hand + release (exact)
  track:   true TCP - aimed TCP at release (the arm's tracking)
  hand:    the box's true offset from the TCP - the sensed one (pile-scan grasp)
  release: settled - at release (touch-down and letting go)
  yaw:     the box's yaw off square (deg), split into the tool's (TCP vs psi) and the
           hand's; "edge" = its sideways shift at the far end of the long side (mm)
  nbr:     the neighbours the packer used vs their truth at release
  size:    sensed footprint side - true side (mm, both sides, longest first)
  gap:     planned vs true facing gap to each neighbour planned within 10 mm (true sizes, settled poses)
usage: python place_budget.py <run_dir> [...] [--rows]
"""
import re
import sys
from pathlib import Path

import numpy as np

AIM = re.compile(r"placement aim: box ([-\d.]+)/([-\d.]+) half ([-\d.]+)/([-\d.]+), tcp ([-\d.]+)/([-\d.]+) "
                 r"psi ([-\d.]+), in hand [-\d.]+/[-\d.]+ \(mm, deg\); boxes ?(.*)$")
TRUTH = re.compile(r"truth at release (cbox_\d+): box ([-\d.]+)/([-\d.]+)/([-\d.]+) tcp ([-\d.]+)/([-\d.]+)/([-\d.]+) "
                   r"\(x/y mm, yaw deg\); boxes (.*)$")
SETTLED = re.compile(r"released (cbox_\d+) settled: .* at ([-\d.]+)/([-\d.]+)/([-\d.]+) \(arm frame\)")
SENSED = re.compile(r"grasped: sensed footprint (\d+) x (\d+) mm")
TRUE_SIZE = re.compile(r"picked up (cbox_\d+) .*true size (\d+) x (\d+) x \d+ mm")
PUSHED = re.compile(r"pushed (cbox_\d+):")
MATCH_MM = 30.0
GAP_NEAR_MM = 10.0


def wrap90(deg):
    return (deg + 45.0) % 90.0 - 45.0


def wrap360(deg):
    return (deg + 180.0) % 360.0 - 180.0


def parse(run):
    rows, aim, sensed, true_size, pending = [], None, None, {}, None
    for line in (Path(run) / "launch.log").read_text(errors="replace").splitlines():
        if m := SENSED.search(line):
            sensed = sorted((float(m[1]), float(m[2])), reverse=True)
        elif m := TRUE_SIZE.search(line):
            true_size[m[1]] = (float(m[2]), float(m[3]))
        elif m := AIM.search(line):
            nbrs = [tuple(map(float, b.split("/"))) for b in m[8].split()]
            aim = dict(box=np.array([float(m[1]), float(m[2])]), half=(float(m[3]), float(m[4])),
                       tcp=np.array([float(m[5]), float(m[6])]), psi=float(m[7]), nbrs=nbrs)
        elif (m := TRUTH.search(line)) and aim is not None:
            truth = {n: tuple(map(float, p.split("/"))) for n, p in (b.split("=") for b in m[8].split())}
            pending = dict(name=m[1], aim=aim, box=np.array([float(m[2]), float(m[3])]), yaw=float(m[4]),
                           tcp=np.array([float(m[5]), float(m[6])]), tcp_yaw=float(m[7]), truth=truth,
                           sensed=sensed, true_size=sorted(true_size[m[1]], reverse=True) if m[1] in true_size else None,
                           sizes=dict(true_size), pushed=False, run=Path(run).name)
            aim = None
        elif (m := SETTLED.search(line)) and pending is not None and m[1] == pending["name"]:
            pending["settled"] = np.array([float(m[2]), float(m[3])])
            pending["settled_yaw"] = float(m[4])
            rows.append(pending)
            pending = None
        elif (m := PUSHED.search(line)) and rows and rows[-1]["name"] == m[1]:
            rows[-1]["pushed"] = True
    return rows


def budget(r):
    a = r["aim"]
    track = r["tcp"] - a["tcp"]
    hand = (r["box"] - r["tcp"]) - (a["box"] - a["tcp"])
    release = r["settled"] - r["box"]
    landing = r["settled"] - a["box"]
    yaw = wrap90(r["settled_yaw"])
    tool_yaw = wrap360(r["tcp_yaw"] - a["psi"])
    edge = abs(np.radians(yaw)) * 2.0 * max(a["half"])
    nbr = []
    for bx, by, _hx, _hy in a["nbrs"]:
        d = [(np.hypot(bx - t[0], by - t[1]), t) for n, t in r["truth"].items() if n != r["name"]]
        dist, t = min(d, key=lambda v: v[0])
        if dist < MATCH_MM:
            nbr.append(np.array([bx - t[0], by - t[1]]))
    size = (np.array(r["sensed"]) - np.array(r["true_size"])) if r["sensed"] and r["true_size"] else None
    return dict(landing=landing, track=track, hand=hand, release=release, yaw=yaw, tool_yaw=tool_yaw,
                hand_yaw=wrap90(yaw - tool_yaw), edge=edge, nbr=nbr, size=size, gaps=gap_errors(r))


def true_half(name, yaw, sizes):
    sx, sy = sizes[name]
    return (sy / 2.0, sx / 2.0) if abs(wrap360(yaw)) % 180.0 > 45.0 and abs(wrap360(yaw)) % 180.0 < 135.0 \
        else (sx / 2.0, sy / 2.0)


def gap_errors(r):
    """(planned, true) facing gaps (mm) to the neighbours the box was planned within GAP_NEAR_MM of."""
    a, out = r["aim"], []
    if r["name"] not in r["sizes"]:
        return out
    bx, by = r["settled"]
    bhx, bhy = true_half(r["name"], r["settled_yaw"], r["sizes"])
    for nx, ny, nhx, nhy in a["nbrs"]:
        d = [(np.hypot(nx - t[0], ny - t[1]), n, t) for n, t in r["truth"].items() if n != r["name"]]
        dist, n, t = min(d, key=lambda v: v[0])
        if dist > MATCH_MM or n not in r["sizes"]:
            continue
        thx, thy = true_half(n, t[2], r["sizes"])
        for k, (c, h, nc, nh, tc, th, b, bh) in enumerate(((a["box"][0], a["half"][0], nx, nhx, t[0], thx, bx, bhx),
                                                           (a["box"][1], a["half"][1], ny, nhy, t[1], thy, by, bhy))):
            o = 1 - k  # facing along k: the footprints overlap along the other axis
            po = abs(a["box"][o] - (ny, nx)[k]) < a["half"][o] + (nhy, nhx)[k]
            planned = abs(c - nc) - h - nh
            if po and 0.0 <= planned < GAP_NEAR_MM:
                out.append((planned, abs(b - tc) - bh - th))
    return out


def stats(v):
    v = np.asarray(v, dtype=float)
    if v.size == 0:
        return "-"
    return (f"mean {np.mean(v):+5.1f} sd {np.std(v):4.1f} |median| {np.median(np.abs(v)):4.1f} "
            f"p95|.| {np.percentile(np.abs(v), 95):4.1f} (n {v.size})")


def main():
    runs = [a for a in sys.argv[1:] if not a.startswith("--")]
    rows = [r for run in runs for r in parse(run)]
    if not rows:
        sys.exit("no placements with the validation lines")
    bs = [budget(r) for r in rows]
    if "--rows" in sys.argv:
        for r, b in zip(rows, bs):
            print(f"{r['run']:<22} {r['name']} {'push' if r['pushed'] else '    '} "
                  f"landing {b['landing'][0]:+5.1f}/{b['landing'][1]:+5.1f} = track {b['track'][0]:+5.1f}/"
                  f"{b['track'][1]:+5.1f} + hand {b['hand'][0]:+5.1f}/{b['hand'][1]:+5.1f} + release "
                  f"{b['release'][0]:+4.1f}/{b['release'][1]:+4.1f}; yaw {b['yaw']:+5.2f} (tool "
                  f"{b['tool_yaw']:+5.2f}, hand {b['hand_yaw']:+5.2f}) edge {b['edge']:.1f}")
    print(f"{len(rows)} set-downs ({sum(r['pushed'] for r in rows)} then pushed), runs: {', '.join(runs)}")
    for k in ("landing", "track", "hand", "release"):
        for ax, name in ((0, "x"), (1, "y")):
            print(f"  {k:<8} {name}: {stats([b[k][ax] for b in bs])}")
    print(f"  yaw deg : {stats([b['yaw'] for b in bs])}")
    print(f"   tool   : {stats([b['tool_yaw'] for b in bs])}")
    print(f"   hand   : {stats([b['hand_yaw'] for b in bs])}")
    print(f"  edge mm : {stats([b['edge'] for b in bs])}")
    nb = [e for b in bs for e in b["nbr"]]
    for ax, name in ((0, "x"), (1, "y")):
        print(f"  nbr      {name}: {stats([e[ax] for e in nb])}")
    sz = [b["size"] for b in bs if b["size"] is not None]
    print(f"  size long : {stats([s[0] for s in sz])}")
    print(f"  size short: {stats([s[1] for s in sz])}")
    g =np.array([v for b in bs for v in b["gaps"]])
    if g.size:
        print(f"  box-box gap planned {np.median(g[:, 0]):.1f} mm median; true: median {np.median(g[:, 1]):.1f} "
              f"p5 {np.percentile(g[:, 1], 5):.1f} p95 {np.percentile(g[:, 1], 95):.1f}; true - planned: "
              f"{stats(g[:, 1] - g[:, 0])}")


if __name__ == "__main__":
    main()
