#!/bin/bash
# One benchmark run in the normal configuration: runs the real demo script
# (viewer and dashboards on), presses ENTER for it, records per-tick
# telemetry (MANIPOPT_TELEMETRY_DIR) and stops when the task parks or after
# a timeout.
# usage: bench_run.sh <tag> <plain|obstacle> [timeout_s] [launch args...]
# output: $OUT_DIR/<tag>/{launch.log,sim.csv,mpc.csv}  (OUT_DIR default /tmp/manipopt_bench)
REPO=/home/hojin/manipOpt
tag=$1; kind=$2; tmo=${3:-260}; shift 3 2>/dev/null || shift $#
out=${OUT_DIR:-/tmp/manipopt_bench}/$tag
rm -rf "$out"; mkdir -p "$out"
bash "$REPO/scripts/dev/stop_all.sh"
script=$REPO/scripts/container_pickplace.sh
[ "$kind" = obstacle ] && script=$REPO/scripts/container_pickplace_obstacle.sh
export MANIPOPT_TELEMETRY_DIR=$out
echo | setsid bash "$script" "$@" > "$out/launch.log" 2>&1 &
pid=$!
t0=$(date +%s)
while :; do
  sleep 2
  grep -q "parked indefinitely\|destination blocked or full\|place failed" "$out/launch.log" && { sleep 2; break; }
  [ $(( $(date +%s) - t0 )) -ge "$tmo" ] && break
  kill -0 $pid 2>/dev/null || break
done
kill -INT -- -$pid 2>/dev/null
sleep 4
bash "$REPO/scripts/dev/stop_all.sh"
echo "done $tag after $(( $(date +%s) - t0 ))s"
