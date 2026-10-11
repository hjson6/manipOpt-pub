# Source once per terminal: `source scripts/env.sh` (it changes this shell's
# environment, so running it does nothing). Paths are relative to this file.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate manipopt
# moveit_task_constructor is built from source in ~/mtc_ws (not in
# RoboStack); source that overlay before this repo's.
if [ -f "$HOME/mtc_ws/install/setup.bash" ]; then
  source "$HOME/mtc_ws/install/setup.bash"
fi
source "$REPO_ROOT/install/setup.bash"

# The ~/mtc_ws libraries have no RPATH to the conda prefix (RoboStack's own
# packages do), so without this mtc_executor_node cannot find librclcpp.so
# at startup.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# core/ and perception/ are imported from the repo root, not installed.
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"
