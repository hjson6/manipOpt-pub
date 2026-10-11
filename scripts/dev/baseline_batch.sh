#!/bin/bash
# Step 5 baseline: for each mismatch/noise seed, one plain and one obstacle run of
# the real demo scripts (viewer on), then the summary table.
# usage: baseline_batch.sh <prefix> [seeds, default "1 2 3 4 5"] [kinds, default "plain obstacle"]
REPO=/home/hojin/manipOpt
prefix=$1; seeds=${2:-"1 2 3 4 5"}; kinds=${3:-"plain obstacle"}
export OUT_DIR=${OUT_DIR:-/tmp/manipopt_bench}
for s in $seeds; do
  for kind in $kinds; do
    bash "$REPO/scripts/dev/bench_run.sh" "${prefix}_${kind}_s$s" "$kind" 900 mismatch_seed:=$s noise_seed:=$s
  done
done
source "$REPO/scripts/env.sh" >/dev/null 2>&1
python "$REPO/scripts/dev/baseline_summary.py" $(ls -d $OUT_DIR/${prefix}_* | sort -V)
