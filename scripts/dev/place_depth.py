"""How far below its placement target the TCP goes during each place
(box bottom below the sensed surface = penetration), per run.
usage: python place_depth.py <run_dir> [...]"""
import re
import sys
from pathlib import Path

import numpy as np
import pinocchio as pin

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_analyze import load, MJCF  # noqa: E402

pm = pin.buildModelFromMJCF(MJCF)
pd = pm.createData()
fid = pm.getFrameId("tcp_site")
for run in sys.argv[1:]:
    log = open(Path(run) / "launch.log").read()
    s = load(Path(run) / "sim.csv")
    sel = [(float(t), float(z) + 2 * 0) for t, z in []]
    targets = re.findall(r"\[(\d+\.\d+)\] \[task_node\]: scanning destination\.\.\. selected \(([-\d.]+), ([-\d.]+)\) "
                         r"\(sensed surface ([-\d.]+)m, box height ([-\d.]+)m", log)
    placed = [float(t) for t in re.findall(r"\[(\d+\.\d+)\] \[mujoco_sim_node\]: placed", log)]
    out = []
    for (t, x, y, surf, h), tp in zip(targets, placed):
        z_target = float(surf) + float(h)
        k = np.where((s["wall_t"] > float(t)) & (s["wall_t"] < tp + 0.3))[0]
        zs = []
        for i in k:
            pin.framesForwardKinematics(pm, pd, np.array([s[f"q{j}"][i] for j in range(1, 8)]))
            zs.append(pd.oMf[fid].translation[2])
        zs = np.array(zs)
        out.append((z_target - zs.min()) * 1e3)
    out = np.array(out)
    print(f"{Path(run).name}: below target per place [mm] {np.round(out, 1).tolist()}  max {out.max():.1f}")
