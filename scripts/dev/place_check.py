"""Per placement: TCP sideways offset from the target while the box is
below the tray wall tops (max, and at release), and heading error at
release (tool x-axis vs the nearest of +-90 deg).
usage: python place_check.py <run_dir> [...]"""
import re
import sys
from pathlib import Path

import numpy as np
import pinocchio as pin

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_analyze import load, MJCF  # noqa: E402

WALL_TOP_Z = 0.22
pm = pin.buildModelFromMJCF(MJCF)
pd = pm.createData()
fid = pm.getFrameId("tcp_site")
for run in sys.argv[1:]:
    log = open(Path(run) / "launch.log").read()
    s = load(Path(run) / "sim.csv")
    targets = re.findall(r"\[(\d+\.\d+)\] \[task_node\]: scanning destination\.\.\. selected \(([-\d.]+), ([-\d.]+)\) "
                         r"\(sensed surface ([-\d.]+)m, box height ([-\d.]+)m", log)
    placed = [float(t) for t in re.findall(r"\[(\d+\.\d+)\] \[mujoco_sim_node\]: placed", log)]
    rows = []
    for (t, x, y, surf, h), tp in zip(targets, placed):
        k = np.where((s["wall_t"] > float(t)) & (s["wall_t"] <= tp))[0]
        side, yaw = [], None
        for i in k:
            pin.framesForwardKinematics(pm, pd, np.array([s[f"q{j}"][i] for j in range(1, 8)]))
            p, R = pd.oMf[fid].translation, pd.oMf[fid].rotation
            if p[2] - float(h) < WALL_TOP_Z:  # box bottom below the wall tops
                side.append(np.hypot(p[0] - float(x), p[1] - float(y)))
            yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
        yaw_err = min(abs((yaw - a + 180) % 360 - 180) for a in (-90, 90))
        rows.append((max(side) * 1e3 if side else 0.0, side[-1] * 1e3 if side else 0.0, yaw_err))
    r = np.array(rows)
    print(f"{Path(run).name}: sideways max below wall top [mm] {np.round(r[:,0],1).tolist()} | at release "
          f"{np.round(r[:,1],1).tolist()} | yaw err [deg] {np.round(r[:,2],1).tolist()}")
