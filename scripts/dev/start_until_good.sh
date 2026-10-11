#!/bin/bash
# usage: start_until_good.sh <tag> <args...>  (retries until the first move doesn't flake)
D=$(cd "$(dirname "$0")" && pwd); S=${OUT_DIR:-/tmp/manipopt_runs}; mkdir -p $S
tag=$1
for attempt in 1 2 3 4 5 6; do
  $D/stop_all.sh
  $D/launch_headless.sh $tag "${@:2}"
  source /home/hojin/manipOpt/scripts/env.sh >/dev/null 2>&1
  r=$(python $D/first_move_check.py 9 2>&1 | grep -E "FLAKE|OK")
  echo "attempt $attempt: $r"
  case "$r" in OK*) exit 0;; esac
done
exit 1
