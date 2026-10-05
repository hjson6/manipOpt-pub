"""Offline comparison of floor-packing rules for the demo's boxes in the tray.

rows:     pick_place_common.packing (two rows along the long walls).
compact:  each box flush against walls/placed boxes, most of its perimeter
          touching them (pick_place_common.packing.plan_compact).
compact+rot: as compact, and each box may be turned 90 deg when it is placed.
All rules see the next `known` boxes (the pile scan's look-ahead).
Per rule: boxes on the floor, largest free rectangle left, group bounding area,
boxes needing a wall-clearance push (task_node's wrist model) and how many of
those pushes have room for the tool.
group+rot: as compact+rot, ranked by the group rectangle first (one block from the far
          corner). --land S: each box lands S (m, sd per axis) off its plan, as live.
--clear C: the planned gap (m; 0.001: boxes pushed flush after each placement).
--push-area A: group+rot counts a spot needing a wall push A m2 more group rectangle.
--strip: group+rot measures the block by its reach from the far short wall.
usage: python packing_compare.py [n_random_orders] [known] [--draw order] [--land 0.003] [--clear 0.005]
       [--push-area 0.005]
"""
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tasks/pick_and_place/common"))
from pick_place_common.packing import plan_compact, plan_placement  # noqa: E402

txt = (REPO / "tasks/pick_and_place/common/models/station_pick.xml").read_text()
SIZES = {int(n): tuple(float(v) for v in s.split()[:2]) for n, s in re.findall(
    r'<body name="cbox_(\d)"[^>]*>\s*<freejoint/>\s*<inertial[^>]*/>\s*<geom type="box" size="([\d. ]+)"', txt)}
BOUNDS = ((-0.515, -0.085), (-0.565, -0.285))
C = 0.004  # task_node.PLACE_CLEARANCE_M
LAND_SIGMA = 0.0  # each box lands this far off its plan (sd per axis), the next planned round it (--land S)
RES = 0.005
TOOL_R = 0.03
PUSH_GAP = 0.01
PUSH_MIN = 0.006
WRIST_MARGIN = 0.008
WRIST = (0.044, 0.044, 0.054, 0.088)  # link7 past the TCP along tool +x, -x, +y, -y
(X0, X1), (Y0, Y1) = BOUNDS


def overlaps(b, placed):
    cx, cy, hx, hy = b
    if cx - hx < X0 + C - 1e-9 or cx + hx > X1 - C + 1e-9 or cy - hy < Y0 + C - 1e-9 or cy + hy > Y1 - C + 1e-9:
        return True
    return any(abs(cx - px) < hx + phx + C - 1e-9 and abs(cy - py) < hy + phy + C - 1e-9
               for px, py, phx, phy in placed)


def bbox_area(placed):
    if not placed:
        return 0.0
    a = np.array(placed)
    return float((np.max(a[:, 0] + a[:, 2]) - np.min(a[:, 0] - a[:, 2])) *
                 (np.max(a[:, 1] + a[:, 3]) - np.min(a[:, 1] - a[:, 3])))


def largest_free_rect(placed):
    nx, ny = int(round((X1 - X0) / RES)), int(round((Y1 - Y0) / RES))
    occ = np.zeros((ny, nx), bool)
    for cx, cy, hx, hy in placed:
        c0, c1 = int((cx - hx - X0) / RES), int(np.ceil((cx + hx - X0) / RES))
        r0, r1 = int((cy - hy - Y0) / RES), int(np.ceil((cy + hy - Y0) / RES))
        occ[max(r0, 0):r1, max(c0, 0):c1] = True
    h = np.zeros(nx, int)
    best = (0, 0, 0)
    for r in range(ny):
        h = np.where(occ[r], 0, h + 1)
        stack = []
        for i in range(nx + 1):
            cur = h[i] if i < nx else 0
            start = i
            while stack and stack[-1][1] >= cur:
                s, hh = stack.pop()
                if hh * (i - s) > best[0]:
                    best = (hh * (i - s), hh, i - s)
                start = s
            stack.append((start, cur))
    return best[0] * RES * RES, best[1] * RES, best[2] * RES


def wrist_shift(cx, cy, turned):
    """TCP shift that keeps link7 inside the walls with the box set down flush.
    Heading at the tray: +90 deg unturned, 0 or 180 turned (the smaller shift)."""
    best = None
    for psi in ((np.pi / 2,) if not turned else (0.0, np.pi)):
        c, s = np.cos(psi), np.sin(psi)
        px, mx, py, my = WRIST
        corners = np.array([[px, py], [px, -my], [-mx, py], [-mx, -my]]) @ np.array([[c, s], [s, -c]])
        lo, hi = corners.min(axis=0), corners.max(axis=0)
        sh = [float(np.clip(v, w0 + WRIST_MARGIN - lo[k], w1 - WRIST_MARGIN - hi[k]) - v)
              for k, (v, (w0, w1)) in enumerate(zip((cx, cy), BOUNDS))]
        if best is None or np.hypot(*sh) < np.hypot(*best):
            best = sh
    return best


def push_report(box, placed_before, turned):
    """(needs a push, every needed push has room for the tool)."""
    cx, cy, hx, hy = box
    sh = wrist_shift(cx, cy, turned)
    need = [k for k in (0, 1) if abs(sh[k]) >= PUSH_MIN]
    ok = True
    half = (hx, hy)
    for k in need:
        tool = [cx + sh[0], cy + sh[1]]
        tool[k] -= np.sign(-sh[k]) * (half[k] + TOOL_R + PUSH_GAP)
        r = TOOL_R + 0.005
        if overlaps((tool[0], tool[1], r, r), placed_before):
            ok = False
    return bool(need), ok


RNG = np.random.default_rng(0)
STRIP = False  # --strip: group+rot grows the block across the tray's width first
PUSH_AREA = 0.0  # --push-area A: group+rot counts a spot needing a push A m2 more group rectangle


def needs_push(cx, cy, hx, hy, turned):
    sh = wrist_shift(cx, cy, turned)
    return max(abs(v) for v in sh) >= PUSH_MIN


def landed(s, placed):
    """Where a box planned at s lands: off by LAND_SIGMA per axis, never into a wall or a
    placed box (it stops against them)."""
    cx, cy, hx, hy = s
    for _ in range(20):
        q = (cx + RNG.normal(0, LAND_SIGMA), cy + RNG.normal(0, LAND_SIGMA), hx, hy)
        if not any(abs(q[0] - px) < hx + phx and abs(q[1] - py) < hy + phy for px, py, phx, phy in placed) \
                and X0 <= q[0] - hx and q[0] + hx <= X1 and Y0 <= q[1] - hy and q[1] + hy <= Y1:
            return q
    return s


def run(rule, order, known):
    placed, pushes, push_ok, missed = [], 0, 0, 0
    for i, b in enumerate(order):
        hx, hy = SIZES[b]
        upcoming = [SIZES[u] for u in order[i + 1:i + 1 + known]]
        if rule == "rows":
            p = plan_placement(hx, hy, placed, BOUNDS, C, upcoming=upcoming)
            s = None if p is None else (p[0], p[1], hx, hy)
        else:
            s = plan_compact(hx, hy, placed, BOUNDS, C, upcoming=upcoming, rotate=(rule != "compact"),
                             group_first=(rule == "group+rot"), needs_push=needs_push, push_area=PUSH_AREA,
                             strip=STRIP)
            if s is not None and LAND_SIGMA > 0:
                s = landed(s, placed)
        if s is None:
            missed += 1
            continue
        need, ok = push_report(s, placed, turned=abs(s[2] - hx) > 1e-9)
        pushes += need
        push_ok += need and ok
        placed.append(s)
    return dict(floor=len(placed), missed=missed, free=largest_free_rect(placed), bbox=bbox_area(placed),
                pushes=pushes, push_ok=push_ok, placed=placed)


def draw(order, known):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rules = ("rows", "compact", "compact+rot", "group+rot")
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, rule in zip(axs, rules):
        r = run(rule, order, known)
        ax.add_patch(plt.Rectangle((X0, Y0), X1 - X0, Y1 - Y0, fill=False, lw=2))
        for n, (cx, cy, hx, hy) in enumerate(r["placed"]):
            ax.add_patch(plt.Rectangle((cx - hx, cy - hy), 2 * hx, 2 * hy, fc="#d9a441", ec="k"))
            ax.text(cx, cy, str(n + 1), ha="center", va="center")
        a, h, w = r["free"]
        ax.set_title(f"{rule}: {r['floor']}/8 on floor, largest free {100 * w:.0f}x{100 * h:.0f} cm\n"
                     f"pushes {r['pushes']} ({r['pushes'] - r['push_ok']} without tool room)", fontsize=10)
        ax.set_xlim(X0 - 0.01, X1 + 0.01); ax.set_ylim(Y0 - 0.01, Y1 + 0.01); ax.set_aspect("equal")
        ax.set_xlabel("x (m)  [robot base is up and to the right]")
    fig.suptitle(f"pick order {order} (numbers = placement order)")
    out = REPO / "packing_compare.png"
    plt.tight_layout(); plt.savefig(out, dpi=110)
    print("saved", out)


def main():
    global LAND_SIGMA, C, PUSH_AREA, STRIP
    STRIP = "--strip" in sys.argv
    if "--push-area" in sys.argv:
        PUSH_AREA = float(sys.argv[sys.argv.index("--push-area") + 1])
    if "--land" in sys.argv:
        LAND_SIGMA = float(sys.argv[sys.argv.index("--land") + 1])
    if "--clear" in sys.argv:
        C = float(sys.argv[sys.argv.index("--clear") + 1])
    args = [a for i, a in enumerate(sys.argv[1:], 1)
            if not a.startswith("--") and sys.argv[i - 1] not in ("--land", "--clear", "--push-area")]
    n = int(args[0]) if args else 300
    known = int(args[1]) if len(args) > 1 else 3
    demo = [3, 5, 1, 6, 0, 4, 7, 2]
    if "--draw" in sys.argv:
        draw(demo, known)
        return
    rng = np.random.default_rng(0)
    orders = [demo] + [list(rng.permutation(8)) for _ in range(n)]
    print(f"box footprint total {sum(4 * hx * hy for hx, hy in SIZES.values()):.4f} m2, "
          f"tray {(X1 - X0) * (Y1 - Y0):.4f} m2; known={known}; {len(orders)} orders (first = demo)")
    for rule in ("rows", "compact", "compact+rot", "group+rot"):
        rs = [run(rule, o, known) for o in orders]
        d = rs[0]
        f = np.array([r["floor"] for r in rs])
        free = np.array([r["free"][0] for r in rs])
        print(f"{rule:12s} demo: floor {d['floor']}/8, largest free {1e4 * d['free'][0]:.0f} cm2 "
              f"({100 * d['free'][2]:.0f}x{100 * d['free'][1]:.0f}), pushes {d['pushes']} "
              f"(no room {d['pushes'] - d['push_ok']})")
        print(f"{'':12s} random: all 8 on floor {np.mean(f == 8):.0%}, mean floor {f.mean():.2f}, "
              f"largest free mean {1e4 * free.mean():.0f} cm2, group rectangle mean "
              f"{1e4 * np.mean([r['bbox'] for r in rs]):.0f} cm2, pushes mean "
              f"{np.mean([r['pushes'] for r in rs]):.2f}, no-room pushes mean "
              f"{np.mean([r['pushes'] - r['push_ok'] for r in rs]):.2f}")


if __name__ == "__main__":
    main()
