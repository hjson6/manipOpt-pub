#!/bin/bash
# Headless plain run (no viewer, no windows; RENDER=true shows them) with per-tick telemetry, for quick checks
# between live batches.
# usage: smoke_run.sh <out_dir> <seconds> [launch args...]   (seed 2 unless overridden)
REPO=/home/hojin/manipOpt
out=$1; secs=$2; shift 2
rm -rf "$out"; mkdir -p "$out"
bash "$REPO/scripts/dev/stop_all.sh"
source "$REPO/scripts/env.sh" >/dev/null 2>&1
export MANIPOPT_TELEMETRY_DIR=$out
setsid ros2 launch pick_place_mpc demo.launch.py render:=${RENDER:-false} visualize_pickup:=${RENDER:-false} \
    mismatch_seed:=2 noise_seed:=2 "$@" > "$out/launch.log" 2>&1 < /dev/null &
pid=$!
for i in $(seq 1 90); do grep -q "mpc_controller ready" "$out/launch.log" && break; sleep 1; done
sleep 2
ros2 topic pub --once /mpc/go std_msgs/msg/Empty "{}" >/dev/null 2>&1
sleep "$secs"
kill -INT -- -$pid 2>/dev/null; sleep 4
bash "$REPO/scripts/dev/stop_all.sh"
echo "placed $(grep -c ': placed cbox' "$out/launch.log"), pushes $(grep -c 'pushed cbox' "$out/launch.log"), legs not reached $(grep -c 'not reached' "$out/launch.log")"
grep -o "stepping:.*\|real time: .*" "$out/launch.log" | tail -1
