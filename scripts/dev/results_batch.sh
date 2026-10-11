#!/bin/bash
# Full MPC baseline (handover_notes/baseline_benchmark_plan.md): plain, obstacle and
# oracle (arm mismatch off, same boxes) on test seeds 1-10, each seed run twice,
# then the summary. Runs already on disk are kept.
# usage: results_batch.sh [out_dir, default results/mpc] [seeds, default 1-10] [runs per seed, default 2]
REPO=/home/hojin/manipOpt
out=${1:-$REPO/results/mpc}; seeds=${2:-"1 2 3 4 5 6 7 8 9 10"}; reps=${3:-2}
export OUT_DIR=$out/runs
run() {  # tag kind args...
  [ -f "$OUT_DIR/$1/mpc.csv" ] && { echo "skip $1"; return; }
  bash "$REPO/scripts/dev/bench_run.sh" "$@"
}
for r in $(seq 1 "$reps"); do
  sfx=""; [ "$r" -gt 1 ] && sfx="_r$r"
  for s in $seeds; do
    run "plain_s$s$sfx" plain 900 mismatch_seed:=$s noise_seed:=$s
    run "obstacle_s$s$sfx" obstacle 900 mismatch_seed:=$s noise_seed:=$s
    run "oracle_s$s$sfx" plain 900 plant_mismatch:=false mismatch_seed:=$s noise_seed:=$s
  done
done
source "$REPO/scripts/env.sh" >/dev/null 2>&1
PYTHONPATH=$REPO:$REPO/tasks/pick_and_place/common:$PYTHONPATH python "$REPO/scripts/dev/results_summary.py" "$out"
