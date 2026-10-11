#!/usr/bin/env bash
# The container pick-and-place demo (MPC method, demo.launch.py): moves the
# pile's 8 boxes to the tray, then parks. See _run_scenario.sh.
set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/scripts/_run_scenario.sh"
run_scenario "pick_place_mpc" "demo.launch.py" "$@"
