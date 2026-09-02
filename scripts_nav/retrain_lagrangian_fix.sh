#!/usr/bin/env bash
# One-off: retrain the 10 nav runs that trained with the
# LagrangianUpdateCallback info-key bug (lambda stuck at 0.0 the whole run
# — see src/cdp/lagrangian_callback.py's "Bug history" docstring and
# private/CONTRIBUTIONS_LOG.md). task_only and fixed_weight runs are
# unaffected (neither uses the callback) and are NOT re-run here.
set -euo pipefail
source /data/heng/miniconda3/etc/profile.d/conda.sh
conda activate cdp_nav
cd /home/heng/work/CDP

STEPS=1000000
LOG_DIR=/data/heng/cdp/logs_nav/run_all_nav
mkdir -p "$LOG_DIR"
MAX_PARALLEL=${MAX_PARALLEL:-8}

run() {
  local task=$1 cond=$2
  local tag="${cond}_${task}"
  echo "=== [$(date -Is)] START $tag (retrain) ==="
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 python -u \
    /home/heng/work/CDP/scripts_nav/train_ppo_nav.py \
    --task_name "$task" --condition "$cond" --seed 0 --total_timesteps $STEPS \
    > "$LOG_DIR/${tag}.log" 2>&1
  echo "=== [$(date -Is)] DONE  $tag (retrain) ===" | tee -a "$LOG_DIR/${tag}.log"
}

wait_if_full() {
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do
    wait -n
  done
}

for task in goal_hazards_only goal_vases_only button_gremlins_only button_wrong_button_only; do
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
echo "=== RETRAIN DONE ==="
