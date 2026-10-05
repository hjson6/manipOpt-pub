#!/bin/bash
# One mobile job (mobile_job.launch.py): telemetry on, waits until the task is done
# (or stops, or a timeout), then stops everything.
# HEADLESS=1: no viewer or windows.
# usage: job_run.sh <tag> [timeout_s] [launch args...]
# output: $OUT_DIR/<tag>/{launch.log,*.csv}  (OUT_DIR default data/runs)
REPO=/home/hojin/manipOpt
tag=$1; tmo=${2:-2400}; shift 2 2>/dev/null || shift $#
out=${OUT_DIR:-$REPO/data/runs}/$tag
rm -rf "$out"; mkdir -p "$out"
bash "$REPO/scripts/dev/stop_all.sh"
source "$REPO/scripts/env.sh" >/dev/null 2>&1
export MANIPOPT_TELEMETRY_DIR=$out
[ -n "$HEADLESS" ] && set -- render:=false visualize_pickup:=false "$@"
setsid ros2 launch pick_place_mpc mobile_job.launch.py sup_csv_path:=$out/sup.csv "$@" > "$out/launch.log" 2>&1 &
pid=$!
t0=$(date +%s)
until grep -q "mpc_controller ready; waiting for /mpc/go to start" "$out/launch.log" 2>/dev/null; do
  sleep 1
  [ $(( $(date +%s) - t0 )) -ge 180 ] && { echo "never ready"; break; }
done
sleep 5  # the base's stack up too
# A single publish can go out before the controller is discovered: until it says it started.
until grep -q "received /mpc/go; starting operation" "$out/launch.log"; do
  ros2 topic pub --once /mpc/go std_msgs/msg/Empty '{}' > /dev/null
  sleep 2
  [ $(( $(date +%s) - t0 )) -ge 300 ] && { echo "never started"; break; }
done
while :; do
  sleep 3
  grep -q "job done\|; stopping\|\[task_node-[0-9]*\]: process has died" "$out/launch.log" && { sleep 3; break; }
  [ $(( $(date +%s) - t0 )) -ge "$tmo" ] && { echo "timeout"; break; }
  kill -0 $pid 2>/dev/null || { echo "launch ended"; break; }
done
kill -INT -- -$pid 2>/dev/null
sleep 6
bash "$REPO/scripts/dev/stop_all.sh"
echo "done $tag after $(( $(date +%s) - t0 ))s"
