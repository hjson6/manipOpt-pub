#!/bin/bash
# Makes the training environment of the learned policies: a venv over the runtime's conda
# env (manipopt), so numpy, MuJoCo, casadi and acados are the runtime's own builds, with
# the training packages of requirements_rl.txt (torch from the CPU index) and the runtime's
# versions held as constraints. Activate with `source scripts/env_rl.sh`.
set -e
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate manipopt
python -m venv --system-site-packages "$REPO/.venv"
source "$REPO/.venv/bin/activate"
held=$(mktemp)
python -c "import numpy, scipy, matplotlib, mujoco, casadi
for m in (numpy, scipy, matplotlib, mujoco, casadi): print(f'{m.__name__}=={m.__version__}')" > "$held"
pip install -c "$held" --extra-index-url https://download.pytorch.org/whl/cpu -r "$REPO/requirements_rl.txt"
rm -f "$held"
python -c "import numpy, torch, stable_baselines3; print('numpy', numpy.__version__, '(the runtime\'s), torch', torch.__version__, 'sb3', stable_baselines3.__version__)"
