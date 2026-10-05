"""Offline check of pick_place_common.packing with the demo's boxes.

Each box is placed knowing the sizes of the next `known` boxes (standing in
for what the pile scan can see) and how many more are still unseen.
usage: python packing_sim.py [order, e.g. 3,0,5,1,6,4,7,2] [known counts, e.g. 0,2,7] [--draw]
"""
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tasks/pick_and_place/common"))
from pick_place_common.packing import plan_placement, touching  # noqa: E402

txt = (REPO / "tasks/pick_and_place/common/models/station_pick.xml").read_text()
SIZES = {int(n): [float(v) for v in s.split()] for n, s in re.findall(
    r'<body name="cbox_(\d)"[^>]*>\s*<freejoint/>\s*<inertial[^>]*/>\s*<geom type="box" size="([\d. ]+)"', txt)}
BOUNDS = ((-0.515, -0.085), (-0.565, -0.285))
CLEAR = 0.003
args = [a for a in sys.argv[1:] if a != "--draw"]
order = [int(v) for v in (args[0] if args else "3,0,5,1,6,4,7,2").split(",")]
knowns = [int(v) for v in (args[1] if len(args) > 1 else "0,1,2,3,7").split(",")]
(x0, x1), (y0, y1) = BOUNDS


def run(known):
    placed, names, missed = [], [], []
    for i, b in enumerate(order):
        hx, hy, _ = SIZES[b]
        upcoming = [tuple(SIZES[u][:2]) for u in order[i + 1:i + 1 + known]]
        p = plan_placement(hx, hy, placed, BOUNDS, CLEAR, upcoming=upcoming)
        if p is None:
            missed.append(b)
            continue
        placed.append((p[0], p[1], hx, hy))
        names.append(b)
    frac = sum(touching(bx, placed[:i] + placed[i + 1:], BOUNDS, CLEAR + 0.001) / (4 * (bx[2] + bx[3]))
               for i, bx in enumerate(placed)) / max(len(placed), 1)
    return placed, names, missed, frac


for known in knowns:
    placed, names, missed, frac = run(known)
    print(f"knows next {known}: on the floor {len(placed)}/{len(order)} (stack {missed}), "
          f"touching fraction {frac:.2f}")
    if "--draw" in sys.argv:
        W, H = 86, 56
        g = [["." for _ in range(W)] for _ in range(H)]
        for (cx, cy, hx, hy), b in zip(placed, names):
            for i in range(H):
                for j in range(W):
                    px, py = x0 + j * 0.005 + 0.0025, y0 + i * 0.005 + 0.0025
                    if abs(px - cx) < hx and abs(py - cy) < hy:
                        g[i][j] = str(b)
        for row in g[::-3]:
            print("   " + "".join(row))
