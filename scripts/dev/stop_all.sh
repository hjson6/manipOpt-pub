#!/bin/bash
# kill every process of this workspace's install tree (all nodes, incl. supervisor/monitor/actor), slam_toolbox
# and the map saver + launch
ps aux | grep -E "manipOpt/install/|ros2 launch pick_place|slam_toolbox/lib/slam_toolbox|map_saver_cli" | grep -v grep \
  | awk '{print $2}' | xargs -r kill -9
sleep 2
for f in /dev/shm/fastrtps_*; do [ -e "$f" ] && rm -f "$f"; done 2>/dev/null
true
