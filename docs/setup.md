# Local setup (WSL2 Ubuntu)

Run all of this **locally**, inside WSL, not in a cloud/remote session.

Note on the plant: this project originally targeted Isaac Sim, but Isaac
Sim's Vulkan/RTX rendering pipeline is not supported under WSL2 (confirmed
via NVIDIA forums and multiple upstream issues — it fails at
`ERROR_INCOMPATIBLE_DRIVER` regardless of VRAM), and the compatibility
checker separately flagged this machine's 8.55GB VRAM as below Isaac Sim's
10GB minimum even on native Windows. MuJoCo replaces it as the plant: it
runs natively in WSL2 with no GPU/Vulkan dependency, has first-class Python
bindings, and is the standard substitute cited for exactly this situation
(no Isaac Sim-class GPU) in the robot-learning/control community — see
`docs/design_notes.md` "Why MuJoCo instead of Isaac Sim".

## 0. WSL networking mode: must be NAT, not mirrored

If you ever set `networkingMode=mirrored` in `%UserProfile%\.wslconfig`
(e.g. for an earlier Isaac-Sim-on-Windows / ROS2-on-WSL2 bridge, which this
project no longer needs), **revert it**. Mirrored mode breaks ROS2 DDS
participant discovery even between two processes on the same machine
(`ros2 topic list`, and actual cross-process `rclpy` pub/sub, both hang
indefinitely) despite raw UDP multicast working fine at the socket level —
confirmed by direct testing, not just a guess. `ROS_LOCALHOST_ONLY=1` and
switching RMW implementation do **not** fix it; only removing
`networkingMode=mirrored` (or setting it to `nat`, the default) does,
followed by `wsl --shutdown` from PowerShell (not from inside WSL) and
reopening WSL.

## 1. Conda environment (Python, CasADi/Pinocchio, MuJoCo, and ROS2)

ROS2 here comes from **RoboStack** (`robostack-staging` conda channel), not
apt. Two independent reasons:
- apt's `ros-humble-desktop` targets Ubuntu 22.04 ("jammy"); this machine is
  24.04 ("noble"), which apt has no Humble build for at all.
- Even switching to apt's `ros-jazzy-desktop` (the noble-targeted distro),
  `rclpy`'s compiled `_rclpy_pybind11` extension is built against the
  system Python (3.12) and cannot be imported from a separate conda
  environment running a different Python (conda envs and apt-installed ROS2
  do not share a Python ABI) — hence RoboStack, which packages ROS2 itself
  as conda packages so everything lives in one environment with one
  consistent Python version.
- RoboStack does not package Jazzy yet, so this uses `ros-humble-desktop`
  from RoboStack instead — its conda build is self-contained and does not
  depend on the host OS's own ROS apt repo, so the jammy/noble mismatch
  that rules out *apt's* Humble does not apply here.

```bash
conda config --set channel_priority strict   # RoboStack's own recommendation
conda env create -f environment.yml
conda activate manipopt
```

Sanity check:
```bash
python -c "import rclpy; print('rclpy ok')"
python -c "import casadi, pinocchio, mujoco; print('casadi/pinocchio/mujoco ok')"
ros2 pkg list | head
```

If you ran the apt-based ROS2 install steps from an earlier version of this
doc, that's harmless leftover (a repo config file, or nothing if
`ros-jazzy-desktop` failed to install) — just don't source
`/opt/ros/*/setup.bash` in the same shell as the `manipopt` conda env, since
mixing the two reintroduces the same Python-ABI conflict.

## 2. acados

acados is not on conda/pip in a form that includes prebuilt HPIPM/BLASFEO;
build from source:

```bash
git clone https://github.com/acados/acados.git ~/acados
cd ~/acados && git submodule update --recursive --init
mkdir build && cd build
cmake -DACADOS_WITH_QPOASES=ON ..
make install -j$(nproc)

# Python interface, into the same conda env:
conda activate manipopt
pip install -e ~/acados/interfaces/acados_template
export ACADOS_SOURCE_DIR=~/acados
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ACADOS_SOURCE_DIR/lib
```

Add the two `export` lines to `~/.bashrc` so they persist across shells.
Known trouble spot: if `make install` fails on BLASFEO target architecture
detection, set `-DBLASFEO_TARGET=GENERIC` and rebuild — slower BLAS but
removes CPU-specific build failures, fine for this problem size.

Sanity check:
```bash
python -c "import acados_template; print('ok')"
python /home/hojin/acados/examples/acados_python/getting_started/minimal_example_ocp.py
```
The second command exercises the full pipeline (CasADi model -> C codegen
-> compile -> solve) on acados' own pendulum example; a converging
iteration log and a saved `pendulum_ocp.png` confirms the whole toolchain
works, not just the Python import. On first run it downloads a small
templating binary (`tera_renderer`) automatically -- say yes if prompted.

## 3. Robot assets

Get a proper MJCF model of the target manipulator (Franka Panda) rather
than relying on MuJoCo's built-in URDF importer, which drops some
URDF-only features:

```bash
git clone https://github.com/google-deepmind/mujoco_menagerie.git ~/mujoco_menagerie
```

`models/panda_robot.xml` (`tasks/pick_and_place/common/models/`)
references those mesh assets via a placeholder meshdir, substituted at
load time from `$MENAGERIE_PANDA_ASSETS_DIR` (defaults to
`~/mujoco_menagerie/franka_emika_panda/assets`, so the clone above is
enough as-is; override the env var if you cloned elsewhere) -- see
`mujoco_sim_node.py`'s `MODELS_DIR`/`MENAGERIE_PANDA_ASSETS_DIR`. The
*same* file is also what `core/dynamics.py` loads via Pinocchio's MJCF
importer (`pin.buildModelFromMJCF`) to build the OCP's internal dynamics
model -- one file feeds both the plant and the controller's belief about
it, through two independent parsers (see `docs/design_notes.md` for why
that's deliberate, not accidental duplication). No separate URDF needed.

## 4. Build the ROS2 packages

```bash
cd ~/manipOpt   # or wherever you cloned this repo locally -- the local
                # directory name is decoupled from the GitHub repo name
                # (git clone <url> <local-dir-name>); scripts/env.sh
                # resolves paths relative to itself either way
colcon build --base-paths tasks --symlink-install
source install/setup.bash
```
`--base-paths tasks` points colcon at `tasks/` instead of scanning the
whole repo -- `core/` and `perception/` are plain Python packages, not
ROS2 ones, so there's nothing under them for colcon to build.
`--symlink-install` is what makes editing a `.py` file take effect
immediately, no rebuild, on the next `ros2 run`/`ros2 launch` (an XML
scene file is read straight from source either way, unaffected either
way -- see `mujoco_sim_node.py`'s `MODELS_DIR`).

## 5. Run it

In any new terminal tab, `source scripts/env.sh` once (activates the conda
env and sources the ROS2 workspace overlay) instead of retyping those
commands.

**All at once** (single terminal, one combined log):
```bash
bash scripts/container_pickplace.sh
```
This force-kills any leftover nodes from a previous run before launching
(see docs/design_notes.md's debugging journal, last entry, for why that
matters: a rare crash in the MuJoCo viewer's shutdown path can otherwise
leave a stuck process behind that fights the next run), then brings
everything up and waits for you to press ENTER before the arm actually
starts moving -- so you get a moment to see the viewer window, the boxes
sitting in their starting pile, and everything holding still, rather than
it lurching into motion mid-setup. Calling
`ros2 launch pick_place_mpc demo.launch.py` directly still works, minus
that safety net and the ENTER prompt -- in that case the arm just holds
indefinitely until something publishes to `/mpc/go`
(`ros2 topic pub --once /mpc/go std_msgs/msg/Empty '{}'` from another
terminal). Pass `render:=false visualize_pickup:=false` to either form
for a headless run (no MuJoCo viewer, no OpenCV dashboard windows).

**One terminal per node** (recommended while iterating): each `ros2 run`
call blocks its terminal forever (that's a running node, not a hang), so
open one tab per node, `source scripts/env.sh` in each, then:
```bash
ros2 run pick_place_common mujoco_sim_node   # opens a live viewer window by default
ros2 run pick_place_mpc task_node            # sensed pick/place decisions + reference generation
ros2 run pick_place_mpc mpc_controller       # solves the OCP and applies u0 directly
```
This lets you leave `mujoco_sim_node` (and its viewer window) running
continuously while editing and restarting just `mpc_controller` to iterate
on OCP tuning -- see docs/design_notes.md. `mpc_controller` also owns the
decel-to-stop fallback (there is no separate relay node for that -- see
docs/design_notes.md, "Real-time risk", for why keeping the hot loop to one hop
matters, on real hardware as much as here). It also sits gated on startup
here too (see above) -- publish to `/mpc/go` once you're ready for it to
start actually commanding the plant.

Everything (plant, task decision node, controller) runs as ROS2 nodes in
the same WSL2 environment — no cross-machine/cross-OS bridging needed,
unlike the Isaac Sim-on-Windows plan this replaced.
