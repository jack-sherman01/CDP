#!/usr/bin/env bash
# Training queue for the new manipulation tasks added 2026-09-05
# (open_drawer, open_single_door). wipe_counter excluded (particle-system
# bug) and shelve_item excluded (upstream reset()'s randomize-and-retry
# loop made one run take ~6.5 hours, 4-5x every other task) — see their
# TaskSpec comments in src/cdp/tasks.py. Full 4-condition sweep
# (task_only/scalar_lagrangian/vector_lagrangian/fixed_weight) at seed 0,
# same 20,000-step budget as the original 3 tasks. Sequential — one Isaac
# Sim process at a time.
set -euo pipefail

source /data/heng/cdp/env.sh
conda activate oopsieverse_b1k
cd /data/heng/cdp/external/oopsieverse

STEPS=20000
LOG_DIR=/data/heng/cdp/logs/run_all_manip
mkdir -p "$LOG_DIR"

run() {
  local task=$1 cond=$2
  local tag="${cond}_${task}"
  echo "=== [$(date -Is)] START $tag ==="
  OMNIGIBSON_HEADLESS=true PYTHONUNBUFFERED=1 python -u \
    /home/heng/work/CDP/scripts/train_ppo.py \
    --task_name "$task" --condition "$cond" --seed 0 --total_timesteps $STEPS \
    > "$LOG_DIR/${tag}.log" 2>&1
  echo "=== [$(date -Is)] DONE  $tag ===" | tee -a "$LOG_DIR/${tag}.log"
}

for task in open_drawer open_single_door; do
  for cond in task_only scalar_lagrangian vector_lagrangian fixed_weight; do
    run "$task" "$cond"
  done
done

echo "=== NEW TASKS TRAINING DONE ==="
