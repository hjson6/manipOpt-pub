"""The room's layout, checked and drawn (no ROS): the stations layout (tables, fixtures,
home, the docks and their docking lines, the mapping routes, every crowd's paths) and
the parked cell (its tables, the parked base backing out, its visiting and walking
person). Clearances: each table to its wall; the chassis turning on the spot at home,
at the docks and pre-dock poses, and at the routes' corners; routes and people's paths
to walls, tables and fixtures. Prints the tight ones; draws figures/mobile/room_layout.png.
--random N also draws N random crowds (mobile_scenarios.random_crowd, as nav_sim draws them
for seeds 1000 on) to figures/rl/random_crowds.png.
usage: python room_layout.py [--out path] [--random N] [--random-out path]
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tasks/pick_and_place/common"), str(REPO / "tasks/pick_and_place/mpc")]
from nav.navigator import PRE_DOCK_M, behind  # noqa: E402
from pick_place_common.mobile_scenarios import (  # noqa: E402
    CROSS, CROWDS, LOOP, PEOPLE, gap_path, gap_point, random_crowd, room_obstacles as obstacles)
from pick_place_common.mujoco_sim_node import load_scene_model  # noqa: E402
from pick_place_common.scene import (  # noqa: E402
    BASE_CHASSIS_HALF, BASE_HOME_POSE, BASE_PARK_POSE, CELL_POSE, PICK_DOCK_BASE, PLACE_DOCK_BASE, ROOM_SIZE, compose)
from pick_place_mpc import dynamic_obstacle_node as actor  # noqa: E402

TURN_R = float(np.hypot(*BASE_CHASSIS_HALF))  # the chassis's corners turning on the spot
PERSON_R = 0.30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "figures/mobile/room_layout.png"))
    ap.add_argument("--random", type=int, default=0)
    ap.add_argument("--random-out", default=str(REPO / "figures/rl/random_crowds.png"))
    args = ap.parse_args()
    report = []
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Stations layout.
    rects = obstacles(load_scene_model(layout="stations"))
    poses = {"home": BASE_HOME_POSE, "pick dock": PICK_DOCK_BASE, "place dock": PLACE_DOCK_BASE,
             "pick pre-dock": behind(PICK_DOCK_BASE, PRE_DOCK_M), "place pre-dock": behind(PLACE_DOCK_BASE, PRE_DOCK_M)}
    for name, (x0, x1, y0, y1) in rects:
        if "table_top" in name:
            report.append((f"{name} to its wall", min(x0, ROOM_SIZE[0] - x1, y0, ROOM_SIZE[1] - y1), 0.03))
    for name, pose in poses.items():
        if "dock" in name and "pre" not in name:
            continue  # docked beside its table on purpose
        g, what = gap_point(pose[:2], rects)
        report.append((f"chassis turning at {name} (to {what})", g - TURN_R, 0.05))
    for rname, route in (("LOOP", [BASE_HOME_POSE[:2]] + LOOP), ("CROSS", [BASE_HOME_POSE[:2]] + CROSS)):
        g, what = gap_path(route, rects)
        report.append((f"route {rname} (to {what})", g - BASE_CHASSIS_HALF[1], 0.15))
        for c in route[1:-1]:
            g, what = gap_point(c, rects)
            report.append((f"route {rname} corner {c} turning (to {what})", g - TURN_R, 0.05))
    for crowd, specs in CROWDS.items():
        for spec in specs:
            path = spec[1] + spec[1][:1] if len(spec[1]) > 2 else spec[1]
            g, what = gap_path(path, rects) if len(path) > 1 else gap_point(path[0], rects)
            report.append((f"crowd {crowd} {spec[0]} (to {what})", g - PERSON_R, 0.0))
    ax = axes[0]
    draw_room(ax, rects)
    for name, pose in poses.items():
        draw_chassis(ax, pose, "tab:blue" if "pre" not in name else "tab:cyan")
    for d in (PICK_DOCK_BASE, PLACE_DOCK_BASE):
        p = behind(d, PRE_DOCK_M)
        ax.plot([p[0], d[0]], [p[1], d[1]], "b--", lw=1)
    loop = np.array([BASE_HOME_POSE[:2]] + LOOP)
    ax.plot(loop[:, 0], loop[:, 1], "k:", lw=1, label="mapping loop")
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for k, spec in enumerate(CROWDS["job"]):
        path = np.array(spec[1] + spec[1][:1])
        ax.plot(path[:, 0], path[:, 1], "-o", ms=3, color=colors[k], label=spec[0])
    for crowd in ("dock_block", "step_in"):
        path = np.array(CROWDS[crowd][0][1])
        ax.plot(path[:, 0], path[:, 1], "-s", ms=4, color="gray", lw=2)
        ax.annotate(crowd, path[0], fontsize=7)
    ax.set_title("stations layout: docks, mapping loop, the job's crowd (grey: dock block, step in)")
    ax.legend(fontsize=6, loc="lower right")

    # The parked cell.
    rects_c = obstacles(load_scene_model(layout="cell"))
    back = compose(BASE_PARK_POSE, (-1.7, 0.0, 0.0))
    g, what = gap_point(back[:2], rects_c)
    report.append((f"cell: the base backed out 1.7 m, chassis rear (to {what})", g - BASE_CHASSIS_HALF[0], 0.05))
    entry = compose(CELL_POSE, (*actor.VISIT_ENTRY, 0.0))[:2]
    walk = [compose(CELL_POSE, (*p, 0.0))[:2] for p in (actor.WALK_START, actor.WALK_END)]
    for name, p in (("cell person's entry", entry), ("cell walk start", walk[0]), ("cell walk end", walk[1])):
        report.append((f"{name} to the walls", gap_point(p, [])[0] - PERSON_R, 0.0))
    for spec in PEOPLE:
        path = spec[1] + spec[1][:1] if len(spec[1]) > 2 else spec[1]
        g, what = gap_path(path, rects_c) if len(path) > 1 else gap_point(path[0], rects_c)
        report.append((f"cell layout's {spec[0]} (to {what})", g - PERSON_R, 0.0))
    ax = axes[1]
    draw_room(ax, rects_c)
    draw_chassis(ax, BASE_PARK_POSE, "tab:blue")
    draw_chassis(ax, back, "tab:cyan")
    ax.plot(*np.array([entry, compose(CELL_POSE, (*actor.VISIT_STAND, 0.0))[:2]]).T, "r-o", ms=3, label="visit")
    ax.plot(*np.array(walk).T, "m-o", ms=3, label="walk")
    ax.set_title("the parked cell: the base backing out 1.7 m, its person's visit and walk")
    ax.legend(fontsize=7, loc="lower right")

    bad = [r for r in report if r[1] < r[2]]
    for text, g, need in sorted(report, key=lambda r: r[1] - r[2])[:12]:
        print(f"{'TIGHT' if g < need else 'ok   '} {g:+.2f} m (need {need:+.2f}): {text}")
    print(f"{len(report)} clearances checked, {len(bad)} too tight")
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=100)
    print(f"wrote {args.out}")
    if args.random:
        draw_random(args.random, args.random_out)


def draw_random(n, out):
    """n random crowds on the stations layout, as nav_sim --people random draws them."""
    rects = obstacles(load_scene_model(layout="stations"))
    cols = min(n, 3)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)
    for ax, seed in zip(axes.ravel(), range(1000, 1000 + n)):
        draw_room(ax, rects)
        for pose in (BASE_HOME_POSE, PICK_DOCK_BASE, PLACE_DOCK_BASE):
            draw_chassis(ax, pose, "tab:blue")
        specs = random_crowd(np.random.default_rng([seed, 7]), rects)
        for i, (name, path, speed, dwell, _start, gives_way, *trigger) in enumerate(specs):
            pts = np.array(list(path) + (list(path[:1]) if len(path) > 2 else []))
            what = "steps in" if trigger else ("gives way" if gives_way else "does not give way")
            ax.plot(pts[:, 0], pts[:, 1], "-o", color=f"C{i + 1}", lw=1.5, ms=4,
                    label=f"{name}: {speed} m/s, stands {dwell:.0f} s, {what}")
            ax.plot(*path[0], "s", color=f"C{i + 1}", ms=9)
        ax.set_title(f"seed {seed}: {len(specs)} people (squares: where they start)")
        ax.legend(fontsize=7, loc="upper left")
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=80)
    print(f"wrote {out}")


def draw_room(ax, rects):
    ax.plot([0, ROOM_SIZE[0], ROOM_SIZE[0], 0, 0], [0, 0, ROOM_SIZE[1], ROOM_SIZE[1], 0], "k-", lw=2)
    for name, (x0, x1, y0, y1) in rects:
        ax.fill([x0, x1, x1, x0], [y0, y0, y1, y1], color="tan" if "shelf" in name or "pillar" in name else "gray",
                alpha=0.6)
    ax.set_aspect("equal")
    ax.set_xlim(-0.2, ROOM_SIZE[0] + 0.2)
    ax.set_ylim(-0.2, ROOM_SIZE[1] + 0.2)


def draw_chassis(ax, pose, color):
    c, s = np.cos(pose[2]), np.sin(pose[2])
    hx, hy = BASE_CHASSIS_HALF
    pts = np.array([[pose[0] + c * x - s * y, pose[1] + s * x + c * y] for x, y in
                    ((hx, hy), (hx, -hy), (-hx, -hy), (-hx, hy), (hx, hy))])
    ax.plot(pts[:, 0], pts[:, 1], color=color, lw=1.5)
    ax.add_patch(plt.Circle(pose[:2], TURN_R, fill=False, ls=":", color=color, lw=0.8))


if __name__ == "__main__":
    main()
