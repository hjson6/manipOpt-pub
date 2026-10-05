#!/usr/bin/env bash
# The container demo with a workspace obstacle (demo_obstacle.launch.py).
# Default ("visit"): a person walks in, stands at the tray for 6 s and leaves;
# three visits, 15 s apart, the first 3 s after the first pick. The arm slows,
# waits and carries on. Extra arguments are launch arguments, e.g.
#   bash scripts/container_pickplace_obstacle.sh passes:=1 dwell_s:=10  # one longer visit
#   bash scripts/container_pickplace_obstacle.sh obstacle:=walk         # a person walking across the cell instead
#   bash scripts/container_pickplace_obstacle.sh obstacle:=static       # a carton on the transit (not validated on this layout)
#   bash scripts/container_pickplace_obstacle.sh obstacle:=none         # same stack, no obstacle (false-positive check)
# All arguments: see launch/demo_obstacle.launch.py.
set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/scripts/_run_scenario.sh"
run_scenario "pick_place_mpc" "demo_obstacle.launch.py" "$@"
