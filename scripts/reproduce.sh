#!/usr/bin/env bash
# The whole protocol in order, from the public archives to the Stage B test metrics.
# Finished steps are skipped, so the script can be re-run after an interruption.
#
#   bash scripts/reproduce.sh                   # Stage A, then Stage B with five seeds
#   LODO=1 bash scripts/reproduce.sh            # also the leave-one-dataset-out folds of Stage B
#   PY=/path/to/python SHARDS=2 bash scripts/reproduce.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
SEEDS=${SEEDS:-"42 43 44 45 46"}
SHARDS=${SHARDS:-1}          # tracking processes sharing GPU 0

$PY -m pytest -q tests/test_stage_a.py tests/test_stage_b.py
$PY scripts/download_data.py

# Stage A: index, track once, descriptors, split, gate
[ -f work/manifests/index.csv ] || $PY scripts/prepare_data.py scan
pids=()
for ((s = 0; s < SHARDS; s++)); do
  $PY scripts/prepare_data.py track --gpu 0 --shard "$s" --num-shards "$SHARDS" &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
$PY scripts/prepare_data.py windows
$PY scripts/prepare_data.py split
[ -f models/stage_a_gate.joblib ] || $PY scripts/train_stage_a.py
$PY scripts/eval_stage_a.py --split test

# Stage B: clips at the Stage A triggers, adaptation once, temporal encoder per seed
[ -f work/stage_b/benchmark.csv ] || $PY scripts/extract_stage_b_clips.py
[ -f runs/adapt/backbone.pt ] || $PY scripts/train_stage_b.py --phase adapt --out runs/adapt
for s in $SEEDS; do
  [ -f "runs/main/seed$s/model.pt" ] || \
    $PY scripts/train_stage_b.py --phase temporal --backbone runs/adapt/backbone.pt --seed "$s" --out "runs/main/seed$s"
done
$PY scripts/eval_stage_b.py --ckpt "runs/main/seed*/model.pt" --out runs/main/test
$PY scripts/eval_pipeline.py --stage-b "runs/main/seed*/model.pt" --out runs/main/pipeline_test

if [ "${LODO:-0}" = 1 ]; then
  for src in $(tail -n +2 work/stage_b/benchmark.csv | cut -d, -f2 | sort -u); do
    for s in $SEEDS; do
      [ -f "runs/lodo/$src/seed$s/model.pt" ] || \
        $PY scripts/train_stage_b.py --phase temporal --backbone imagenet --holdout "$src" --seed "$s" \
          --out "runs/lodo/$src/seed$s"
    done
    $PY scripts/eval_stage_b.py --ckpt "runs/lodo/$src/seed*/model.pt" --holdout "$src" --out "runs/lodo/$src/test"
  done
fi
