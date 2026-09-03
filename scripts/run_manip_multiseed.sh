#!/usr/bin/env bash
# Partial multi-seed replication for the manipulation-domain headline
# comparison (proposal calls for 5 seeds; full 5x replication of all 17
# runs is not feasible in this project's time budget given ~1.5-2.5hr/run
# wall-clock). Prioritizes the RQ1 headline pair (scalar_lagrangian vs.
# vector_lagrangian) on pick_egg (fastest scene) for seeds 1-2, giving n=3
# for the single most-cited comparison instead of leaving every manip
# result at n=1. Documented as PARTIAL multi-seed coverage, not full.
set -euo pipefail

source /data/heng/cdp/env.sh
conda activate oopsieverse_b1k
cd /data/heng/cdp/external/oopsieverse

STEPS=20000
LOG_DIR=/data/heng/cdp/logs/run_all_manip
mkdir -p "$LOG_DIR"

run() {
  local task=$1 cond=$2 seed=$3
  local tag="${cond}_${task}_${seed}"
  echo "=== [$(date -Is)] START $tag ==="
  OMNIGIBSON_HEADLESS=true PYTHONUNBUFFERED=1 python -u \
    /home/heng/work/CDP/scripts/train_ppo.py \
    --task_name "$task" --condition "$cond" --seed "$seed" --total_timesteps $STEPS \
    > "$LOG_DIR/${tag}.log" 2>&1
  echo "=== [$(date -Is)] DONE  $tag ===" | tee -a "$LOG_DIR/${tag}.log"
}

for seed in 1 2; do
  run pick_egg task_only "$seed"
  run pick_egg scalar_lagrangian "$seed"
  run pick_egg vector_lagrangian "$seed"
done

echo "=== MANIP MULTISEED DONE ==="
