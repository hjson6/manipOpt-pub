"""Open the container scene in MuJoCo's viewer (no ROS 2, no acados).

Usage: python scripts/view_scene.py
"""
import os
import tempfile
import time
from pathlib import Path

import mujoco
import mujoco.viewer

MODELS_DIR = Path(__file__).resolve().parent.parent / "tasks/pick_and_place/common/models"
MESHDIR_PLACEHOLDER = "__MENAGERIE_PANDA_ASSETS_DIR__"
MENAGERIE_PANDA_ASSETS_DIR = os.environ.get(
    "MENAGERIE_PANDA_ASSETS_DIR",
    str(Path.home() / "mujoco_menagerie" / "franka_emika_panda" / "assets"),
)

# MuJoCo resolves <include> on disk, so both files get the meshdir
# substitution written to a temp dir.
with tempfile.TemporaryDirectory() as tmpdir:
    for fname in ("panda_scene_container.xml", "panda_robot.xml"):
        text = (MODELS_DIR / fname).read_text().replace(MESHDIR_PLACEHOLDER, MENAGERIE_PANDA_ASSETS_DIR)
        (Path(tmpdir) / fname).write_text(text)
    model = mujoco.MjModel.from_xml_path(str(Path(tmpdir) / "panda_scene_container.xml"))
data = mujoco.MjData(model)
home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
mujoco.mj_resetDataKeyframe(model, data, home_id)
mujoco.mj_forward(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()
        mujoco.mj_step(model, data)
        viewer.sync()
        dt_left = model.opt.timestep - (time.time() - step_start)
        if dt_left > 0:
            time.sleep(dt_left)
