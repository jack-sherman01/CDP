#!/usr/bin/env bash
# Remaining nav-domain runs after a mid-session interruption killed
# scripts_nav/retrain_lagrangian_fix.sh partway through (4/10 completed
# cleanly: scalar/vector_lagrangian on goal_hazards_only and
# button_wrong_button_only). The other 6 run_dirs' partial episodes were
# deleted (not scientifically wrong, just an incomplete run mixed with
# what would have been a from-scratch restart) before this script runs.
set -euo pipefail
source /data/heng/miniconda3/etc/profile.d/conda.sh
conda activate cdp_nav
cd /home/heng/work/CDP

LOG_DIR=/data/heng/cdp/logs_nav/run_all_nav
mkdir -p "$LOG_DIR"
MAX_PARALLEL=${MAX_PARALLEL:-6}

run() {
  local task=$1 cond=$2
  local tag="${cond}_${task}"
  echo "=== [$(date -Is)] START $tag (remaining) ==="
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 python -u \
    /home/heng/work/CDP/scripts_nav/train_ppo_nav.py \
    --task_name "$task" --condition "$cond" --seed 0 \
    > "$LOG_DIR/${tag}.log" 2>&1
  echo "=== [$(date -Is)] DONE  $tag (remaining) ===" | tee -a "$LOG_DIR/${tag}.log"
}

wait_if_full() {
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do
    wait -n
  done
}

for task in goal_vases_only button_gremlins_only; do
  for cond in scalar_lagrangian vector_lagrangian; do
    wait_if_full
    run "$task" "$cond" &
  done
done
for task in goal_joint button_joint; do
  wait_if_full
  run "$task" vector_lagrangian &
done

wait
echo "=== REMAINING NAV TRAINING DONE ==="
