#!/bin/bash
# One mobile drive with SLAM (mobile.launch.py): telemetry on, waits for the scenario's
# "route done" (or a timeout), saves the map when mapping, stops.
# HEADLESS=1: no viewer.
# usage: mobile_run.sh <tag> <own|toolbox> <mapping|localization> <map> [timeout_s] [launch args...]
# output: $OUT_DIR/<tag>/{launch.log,sim.csv,slam.csv,localization.csv}  (OUT_DIR default data/runs)
REPO=/home/hojin/manipOpt
tag=$1; slam=$2; mode=$3; map=$4; tmo=${5:-300}; shift 5 2>/dev/null || shift $#
out=${OUT_DIR:-$REPO/data/runs}/$tag
rm -rf "$out"; mkdir -p "$out"
bash "$REPO/scripts/dev/stop_all.sh"
source "$REPO/scripts/env.sh" >/dev/null 2>&1
export MANIPOPT_TELEMETRY_DIR=$out
[ -n "$HEADLESS" ] && set -- render:=false "$@"
setsid ros2 launch pick_place_mpc mobile.launch.py slam:=$slam slam_mode:=$mode map:=$map "$@" > "$out/launch.log" 2>&1 &
pid=$!
t0=$(date +%s)
while :; do
  sleep 2
  grep -q "route done" "$out/launch.log" && break
  [ $(( $(date +%s) - t0 )) -ge "$tmo" ] && { echo "timeout"; break; }
  kill -0 $pid 2>/dev/null || { echo "launch ended"; break; }
done
if [ "$mode" = mapping ]; then
  if [ "$slam" = own ]; then
    timeout 60 ros2 service call /slam/save_map std_srvs/srv/Trigger >> "$out/launch.log" 2>&1
  else
    timeout 60 ros2 service call /slam_toolbox/serialize_map slam_toolbox/srv/SerializePoseGraph \
      "{filename: '$REPO/data/maps/$map'}" >> "$out/launch.log" 2>&1
    timeout 60 ros2 run nav2_map_server map_saver_cli -f "$REPO/data/maps/$map" >> "$out/launch.log" 2>&1
  fi
fi
kill -INT -- -$pid 2>/dev/null
sleep 6
bash "$REPO/scripts/dev/stop_all.sh"
echo "done $tag after $(( $(date +%s) - t0 ))s"
