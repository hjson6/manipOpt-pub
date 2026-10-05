#!/bin/bash
# usage: launch_headless.sh <tag> <extra launch args...>
cd /home/hojin/manipOpt; source scripts/env.sh >/dev/null 2>&1
S=${OUT_DIR:-/tmp/manipopt_runs}; mkdir -p $S
tag=$1; shift
nohup ros2 launch pick_place_mpc demo_obstacle.launch.py render:=false visualize_pickup:=true csv_path:=$S/$tag.csv "$@" > $S/$tag.log 2>&1 < /dev/null & disown
for i in $(seq 1 90); do grep -q "mpc_controller ready" $S/$tag.log && break; sleep 1; done
sleep 3
for i in 1 2 3 4 5; do
  ros2 topic pub --once /mpc/go std_msgs/msg/Empty "{}" >/dev/null 2>&1
  sleep 2; grep -q "received /mpc/go" $S/$tag.log && break
done
