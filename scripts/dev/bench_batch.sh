#!/bin/bash
# Acceptance batch: N consecutive launches of each demo script (viewer on),
# then one bench_analyze table per script.
# usage: bench_batch.sh <prefix> [N]
REPO=/home/hojin/manipOpt
prefix=$1; n=${2:-10}
export OUT_DIR=${OUT_DIR:-/tmp/manipopt_bench}
for kind in plain obstacle; do
  for i in $(seq 1 $n); do
    bash "$REPO/scripts/dev/bench_run.sh" "${prefix}_${kind}_$i" "$kind" 300
  done
done
source "$REPO/scripts/env.sh" >/dev/null 2>&1
for kind in plain obstacle; do
  python "$REPO/scripts/dev/bench_analyze.py" $(ls -d $OUT_DIR/${prefix}_${kind}_* | sort -V)
  for d in $(ls -d $OUT_DIR/${prefix}_${kind}_* | sort -V); do
    L=$d/launch.log
    echo "$(basename $d): status_fail=$(grep -c 'solve failed' $L) budget=$(grep -c 'missed budget' $L) HOLD=$(grep -c 'supervisor HOLD' $L) perception_down=$(grep -c 'perception down' $L) sup_silent=$(grep -c 'went silent' $L)"
  done
done
