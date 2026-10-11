# The training environment of the learned policies: the runtime's (env.sh) with the
# training packages layered on in .venv (requirements_rl.txt; scripts/dev/make_rl_env.sh
# makes it). Source once per terminal: `source scripts/env_rl.sh`. Build with colcon from
# env.sh, not here (the venv's newer setuptools breaks --symlink-install).
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
source "$REPO_ROOT/.venv/bin/activate"
