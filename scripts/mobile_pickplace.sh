#!/usr/bin/env bash
# The mobile pick and place (mobile_job.launch.py): the robot drives between the pile's
# table and the tray's, moving the 8 boxes, then home; people walk the room as the
# scenario says. Brings everything up, holds, and starts when you press ENTER
# (AUTO_START=1 bash scripts/mobile_pickplace.sh ... starts on its own once ready);
# Ctrl+C stops it (the monitors log their totals then). See _run_scenario.sh.
#   bash scripts/mobile_pickplace.sh                       # the job's crowd, two people (default)
#   bash scripts/mobile_pickplace.sh quiet                 # nobody
#   bash scripts/mobile_pickplace.sh walkers               # one walker (the walkers scenario's first)
#   bash scripts/mobile_pickplace.sh crowd people:=4       # the job's crowd of four
#   bash scripts/mobile_pickplace.sh dock_block            # someone standing in the pick dock's line
#   bash scripts/mobile_pickplace.sh step_in               # someone stepping into the route
#   bash scripts/mobile_pickplace.sh quiet max_boxes:=2    # extra launch arguments after the scenario
#   bash scripts/mobile_pickplace.sh crowd speed:=0.5      # the settings below, for this run
#   bash scripts/mobile_pickplace.sh crowd base_safety:=0.7 people_speed:=1.3
#   bash scripts/mobile_pickplace.sh quiet localization:=icp conditions:=dropout   # the scan matcher alone, scan gaps
# All arguments: see launch/mobile_job.launch.py.
set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
scenario=crowd     # quiet | walkers | crowd | dock_block | step_in
speed=1.0          # the base's top speed on routes, m/s
# Safety around people. The robot's stopping distances never scale; the margins on top do.
base_safety=1.0    # the base's distances to people x this (where it slows, the room it keeps, its stop
                   # margin; that never below 0.14 m): lower moves on closer, higher keeps more room
arm_safety=1.0     # the arm's distances to people x this (where it slows and holds; never below a
                   # 0.14 m reach allowance)
assumed_speed=1.6  # the walking speed the arm's safety assumes, m/s (ISO 13855: 1.6)
people_speed=1.0   # the people walk at this x their own speed (0.8-1.3 m/s)
localization=ekf   # ekf | icp: the filter (wheels, gyro and scans weighed) or the scan matcher alone
if [ $# -gt 0 ] && [[ "$1" != *":="* ]]; then
    scenario=$1
    shift
fi
case "$scenario" in
    quiet) people=(people:=0) ;;
    walkers) people=(people:=1 crowd:=walkers) ;;
    crowd) people=(people:=2 crowd:=job) ;;
    dock_block) people=(people:=1 crowd:=dock_block) ;;
    step_in) people=(people:=1 crowd:=step_in) ;;
    *) echo "unknown scenario '$scenario': quiet, walkers, crowd, dock_block or step_in" >&2; exit 2 ;;
esac
knobs=(speed base_safety arm_safety assumed_speed people_speed localization)
args=()
for arg in "$@"; do
    key=${arg%%:=*}
    if [[ "$arg" == *":="* && " ${knobs[*]} " == *" $key "* ]]; then
        printf -v "$key" '%s' "${arg#*:=}"
    else
        args+=("$arg")
    fi
done
settings=()
for key in "${knobs[@]}"; do
    settings+=("${key/#localization/fusion}:=${!key}")
done
echo "scenario $scenario; ${settings[*]}"
source "$REPO_ROOT/scripts/_run_scenario.sh"
run_scenario "pick_place_mpc" "mobile_job.launch.py" "${people[@]}" "${settings[@]}" "${args[@]}"
