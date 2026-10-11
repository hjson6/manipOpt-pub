#!/bin/bash
# The ground rule of the learned controllers: with them off, the robot runs exactly the
# MPC-only code of a reference commit (default: the remote's main). Two checks:
#  1. the files mobile_pickplace.sh runs live are those of the reference, byte for byte
#     (the navigator and the MPC are allowed changes that test 2 shows change nothing);
#  2. nav_sim at the reference and here drive the job's crowd of four the same, drive for drive.
# usage: bash scripts/dev/check_mpc_only.sh [ref] [drives]
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ref=${1:-origin/main}; drives=${2:-4}
cd "$REPO" || exit 2
LIVE=(scripts/mobile_pickplace.sh scripts/_run_scenario.sh scripts/env.sh
      tasks/pick_and_place/mpc/launch/mobile_job.launch.py tasks/pick_and_place/mpc/pick_place_mpc
      tasks/pick_and_place/mpc/launch/mobile.launch.py core perception slam
      tasks/pick_and_place/common/pick_place_common/mujoco_sim_node.py
      tasks/pick_and_place/common/pick_place_common/sim_sensors_node.py tasks/pick_and_place/common/models)
changed=$(git diff --name-only "$ref" -- "${LIVE[@]}" | grep -v "^core/learned_policy.py$")
if [ -n "$changed" ]; then echo "FAIL: live files differ from $ref:"; echo "$changed"; status=1; else echo "ok: live files as $ref"; fi
tmp=$(mktemp -d)
git worktree add -q "$tmp/ref" "$ref" || exit 2
ln -s "$REPO/data" "$tmp/ref/data"
source "$REPO/scripts/env.sh" >/dev/null 2>&1
args=(--people job --crowd 4 --arm carry --box --fusion ekf --seed 0 --drives "$drives")
python "$tmp/ref/scripts/dev/nav_sim.py" "${args[@]}" 2>&1 | grep "^drive" > "$tmp/ref.txt" &
python scripts/dev/nav_sim.py "${args[@]}" 2>&1 | grep "^drive" > "$tmp/here.txt"
wait
if [ -s "$tmp/here.txt" ] && diff "$tmp/ref.txt" "$tmp/here.txt"; then
  echo "ok: $(wc -l < "$tmp/here.txt") drives the same as $ref"
else
  echo "FAIL: the drives differ from $ref"; status=1
fi
git worktree remove --force "$tmp/ref"; rm -rf "$tmp"
exit ${status:-0}
