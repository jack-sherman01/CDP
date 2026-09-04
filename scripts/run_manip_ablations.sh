#!/usr/bin/env bash
# Modality-dropout (Days 22-23) and budget-sensitivity (Day 24) ablations,
# scoped to pick_egg/vector_lagrangian only (the headline pair, and the
# fastest scene) given time/compute realism -- not the full grid across
# every task/condition. Documented as partial coverage, same pattern as
# the multi-seed replication.
set -euo pipefail

source /data/heng/cdp/env.sh
conda activate oopsieverse_b1k
cd /data/heng/cdp/external/oopsieverse

STEPS=20000
LOG_DIR=/data/heng/cdp/logs/run_all_manip
mkdir -p "$LOG_DIR"

run() {
  local tag=$1; shift
  echo "=== [$(date -Is)] START $tag ==="
  OMNIGIBSON_HEADLESS=true PYTHONUNBUFFERED=1 python -u \
    /home/heng/work/CDP/scripts/train_ppo.py \
    --task_name pick_egg --condition vector_lagrangian --seed 0 --total_timesteps $STEPS "$@" \
    > "$LOG_DIR/${tag}.log" 2>&1
  echo "=== [$(date -Is)] DONE  $tag ===" | tee -a "$LOG_DIR/${tag}.log"
}

# Modality dropout: proposal's p=0.2
run vector_lagrangian_pick_egg_0_dropout0.2 --modality_dropout_p 0.2

# Budget sensitivity: low/med(default=30)/high
run vector_lagrangian_pick_egg_0_budget15.0 --budget 15.0
run vector_lagrangian_pick_egg_0_budget60.0 --budget 60.0

echo "=== MANIP ABLATIONS DONE ==="
