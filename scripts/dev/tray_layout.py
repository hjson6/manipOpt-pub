"""Final tray layout of a run (from its launch.log) and how much floor is
wasted in gaps no box could use.
usage: python tray_layout.py <run_dir> [...] [--draw]

wasted = free floor not covered by any placement of the smallest box
(60 x 60 mm plus clearance): slivers between boxes and walls.
"""
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
TXT = (REPO / "tasks/pick_and_place/common/models/cell_container.xml").read_text()
SIZES = {n: [float(v) for v in s.split()] for n, s in re.findall(
    r'<body name="(cbox_\d)"[^>]*>\s*<freejoint/>\s*<inertial[^>]*/>\s*<geom type="box" size="([\d. ]+)"', TXT)}
X0, X1, Y0, Y1 = -0.515, -0.085, -0.565, -0.285
G = 0.005
SMALL = 0.066  # smallest box side (60 mm) + clearances


def layout(run):
    log = (Path(run) / "launch.log").read_text(errors="replace")
    sel = re.findall(r"scanning destination\.\.\. selected \(([-\d.]+), ([-\d.]+)\) \(sensed surface ([-\d.]+)", log)
    names = re.findall(r"placed (cbox_\d)", log)
    return [(float(x), float(y), float(z), n) for (x, y, z), n in zip(sel, names)]


def analyse(boxes):
    nx, ny = int(round((X1 - X0) / G)), int(round((Y1 - Y0) / G))
    occ = np.zeros((ny, nx), bool)
    for x, y, z, n in boxes:
        if z > 0.01:
            continue  # stacked boxes do not cover floor
        hx, hy, _ = SIZES[n]
        c0, c1 = int(np.floor((x - hx - X0) / G)), int(np.ceil((x + hx - X0) / G))
        r0, r1 = int(np.floor((y - hy - Y0) / G)), int(np.ceil((y + hy - Y0) / G))
        occ[max(r0, 0):r1, max(c0, 0):c1] = True
    k = int(np.ceil(SMALL / G))
    usable = np.zeros_like(occ)
    for r in range(ny - k + 1):
        for c in range(nx - k + 1):
            if not occ[r:r + k, c:c + k].any():
                usable[r:r + k, c:c + k] = True
    free = ~occ
    wasted = free & ~usable
    return occ, usable, wasted, free.sum() * G * G, wasted.sum() * G * G


if __name__ == "__main__":
    tray = (X1 - X0) * (Y1 - Y0)
    for run in [a for a in sys.argv[1:] if a != "--draw"]:
        boxes = layout(run)
        occ, usable, wasted, free, waste = analyse(boxes)
        floor = sum(1 for b in boxes if b[2] < 0.01)
        print(f"{Path(run).name}: {floor} on the floor, {len(boxes) - floor} stacked; free floor "
              f"{100 * free / tray:.0f}% of tray, of which unusable gaps {100 * waste / tray:.1f}% of tray")
        if "--draw" in sys.argv:
            for r in range(occ.shape[0] - 1, -1, -3):
                print("   " + "".join("#" if occ[r, c] else ("x" if wasted[r, c] else ".")
                                      for c in range(occ.shape[1])))
