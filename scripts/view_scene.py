"""Open the room scene (the cell, the mobile manipulator) in MuJoCo's viewer (no
ROS 2, no acados). Needs `source scripts/env.sh`.

Usage: python scripts/view_scene.py [cell_file]
"""
import sys
import time

import mujoco
import mujoco.viewer

from pick_place_common import frames
from pick_place_common.mujoco_sim_node import DEFAULT_CELL_FILE, HOME_KEYFRAME, load_scene_model

model = load_scene_model(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CELL_FILE)
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, model.key(HOME_KEYFRAME).id)
frames.free_bodies_at_rest(model, data)
mujoco.mj_forward(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()
        mujoco.mj_step(model, data)
        viewer.sync()
        dt_left = model.opt.timestep - (time.time() - step_start)
        if dt_left > 0:
            time.sleep(dt_left)
