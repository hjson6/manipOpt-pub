"""True person-arm clearance in a recorded run (sim.csv: joint angles and the
person's pose), and holds without a person in the cell (launch.log).
  clearance  smallest distance between the arm's collision geoms and the person's
             geoms while the arm moves (|qdot| > 0.1 rad/s over 0.2 s) and the person is in
  holds      supervisor HOLD lines, and those with no person in the cell then
usage: python person_clearance.py <run_dir> [<run_dir> ...]
"""
import re
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_analyze import load  # noqa: E402
from pick_place_common import frames  # noqa: E402
from pick_place_common.mujoco_sim_node import ARM_JOINT_NAMES, load_scene_model  # noqa: E402

STRIDE = 5
PRESENT_Z = -2.0  # the person is parked far below the floor when absent

m = load_scene_model(world="arm")
d = mujoco.MjData(m)
qa = [m.joint(n).qposadr[0] for n in ARM_JOINT_NAMES]
pid = m.body_mocapid[m.body("person_obstacle").id]
person = m.body("person_obstacle")
person_geoms = range(person.geomadr[0], person.geomadr[0] + person.geomnum[0])
arm_geoms = [g for g in range(m.ngeom) if frames.in_arm(m, m.geom_bodyid[g]) and m.geom_contype[g]]


def clearance(run):
    s = load(Path(run) / "sim.csv")
    present = s["person_z"] > PRESENT_Z
    moving = np.convolve(s["qdot_norm"], np.ones(10) / 10, "same") > 0.1  # over 0.2 s: encoder noise is not motion
    best, fromto = (np.inf, None), np.zeros(6)
    for i in np.flatnonzero(present & moving)[::STRIDE]:
        d.qpos[qa] = [s[f"q{j}"][i] for j in range(1, 8)]
        x, y, z, yaw = frames.person_in_arm(s, i)
        d.mocap_pos[pid] = (x, y, z)
        d.mocap_quat[pid] = (np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2))
        mujoco.mj_forward(m, d)
        for a in arm_geoms:
            for b in person_geoms:
                dist = mujoco.mj_geomDistance(m, d, a, b, 2.0, fromto)
                if dist < best[0]:
                    best = (dist, f"{m.body(m.geom_bodyid[a]).name} / {m.geom(b).name}, t {s['sim_t'][i]:.1f} s")
    return best, s


def holds(run, s):
    text = (Path(run) / "launch.log").read_text()
    lines = re.findall(r"\[obstacle_supervisor_node\]: (HOLD.*)", text)
    stamps = [float(t) for t in re.findall(r"\[(\d+\.\d+)\] \[obstacle_supervisor_node\]: HOLD", text)]
    lonely = 0
    for t in stamps:
        i = int(np.clip(np.searchsorted(s["wall_t"], t), 0, len(s["wall_t"]) - 1))
        lo, hi = max(i - 25, 0), i + 25  # within half a second
        if not (s["person_z"][lo:hi] > PRESENT_Z).any():
            lonely += 1
    return len(lines), lonely


def main():
    for run in sys.argv[1:]:
        (dist, where), s = clearance(run)
        n, lonely = holds(run, s)
        print(f"{run}: min clearance while moving {1e3 * dist:.0f} mm ({where}); "
              f"holds {n}, without a person {lonely}")


if __name__ == "__main__":
    main()
