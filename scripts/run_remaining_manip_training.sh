#!/usr/bin/env bash
# Remaining manipulation-domain runs after the entry-14 PID gain fix
# (private/CONTRIBUTIONS_LOG.md). task_only is unaffected by the pivot
# entirely (never reads lambda) -- all 3 task_only checkpoints are the
# original pre-pivot runs (2026-08-30) and stay valid as-is, no rerun
# needed (a first pass at this script mistakenly re-ran task_only_
# pour_water and appended 33 stray episodes onto the pre-pivot summary.jsonl
# before being interrupted -- cleaned back to the original 51 rows).
# vector_lagrangian_pick_egg is already done (the validation run for the
# new gains, real production data, not a smoke test). Everything else
# either never ran or ran with the old, too-weak gains (quarantined in
# checkpoints/_STALE_WEAK_GAINS, runs/_STALE_WEAK_GAINS).
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

run pick_egg scalar_lagrangian
run add_firewood scalar_lagrangian
run pour_water scalar_lagrangian

run add_firewood vector_lagrangian
run pour_water vector_lagrangian

run pick_egg fixed_weight
run add_firewood fixed_weight
run pour_water fixed_weight

run fill_bowl vector_lagrangian
run food_in_microwave vector_lagrangian

echo "=== REMAINING MANIP TRAINING DONE ==="
