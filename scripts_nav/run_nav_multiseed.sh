#!/usr/bin/env bash
# Multi-seed replication for the RQ1/RQ2 headline comparison (proposal
# calls for 3 seeds in the navigation domain; everything so far is seed 0
# only). Prioritizes scalar_lagrangian/vector_lagrangian (the actual RQ1/
# RQ2 comparison) over task_only/fixed_weight (secondary baselines) given
# limited compute -- seeds 1 and 2, all 4 single-exposure tasks + both
# joint-exposure composites.
set -euo pipefail
source /data/heng/miniconda3/etc/profile.d/conda.sh
conda activate cdp_nav
cd /home/heng/work/CDP

LOG_DIR=/data/heng/cdp/logs_nav/run_all_nav
mkdir -p "$LOG_DIR"
MAX_PARALLEL=${MAX_PARALLEL:-8}

run() {
  local task=$1 cond=$2 seed=$3
  local tag="${cond}_${task}_${seed}"
  echo "=== [$(date -Is)] START $tag ==="
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 python -u \
    /home/heng/work/CDP/scripts_nav/train_ppo_nav.py \
    --task_name "$task" --condition "$cond" --seed "$seed" \
    > "$LOG_DIR/${tag}.log" 2>&1
  echo "=== [$(date -Is)] DONE  $tag ===" | tee -a "$LOG_DIR/${tag}.log"
}

wait_if_full() {
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do
    wait -n
  done
}

for seed in 1 2; do
  for task in goal_hazards_only goal_vases_only button_gremlins_only button_wrong_button_only; do
    for cond in scalar_lagrangian vector_lagrangian; do
      wait_if_full
      run "$task" "$cond" "$seed" &
    done
  done
  for task in goal_joint button_joint; do
    wait_if_full
    run "$task" vector_lagrangian "$seed" &
  done
done

wait
echo "=== NAV MULTISEED DONE ==="
