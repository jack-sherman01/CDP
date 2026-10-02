#!/usr/bin/env bash
# Phase 2 of the 100k-step rigor push, after the core 60-run campaign
# (entry 24-27, private/CONTRIBUTIONS_LOG.md) completed: 5-seed
# joint-exposure training on food_in_microwave (RQ2's upper bound, now at
# the realistic budget -- also directly tests whether more training steps
# make joint-exposure a stronger upper bound, per the "turn mixed findings
# into real science" plan), then zero-shot evaluation of the
# add_firewood-sourced checkpoints on food_in_microwave (RQ1 evidence from
# a second, thermal-dominant source task) and the joint-exposure
# checkpoints in-distribution (for RQ2's Delta_comp). Resumable like
# run_manip_100k_campaign.sh.
set -uo pipefail

source /data/heng/cdp/env.sh
conda activate oopsieverse_b1k
cd /data/heng/cdp/external/oopsieverse

STEPS=100000
DATA=/data/heng/cdp
REPO=/home/heng/work/CDP
LOG_DIR="$DATA/logs/run_manip_100k_campaign"
mkdir -p "$LOG_DIR"

train() {
  local task=$1 cond=$2 seed=$3 suffix=$4
  local tag="${cond}_${task}_${seed}${suffix}"
  local ckpt="$DATA/checkpoints/${tag}/final_model.zip"
  if [ -f "$ckpt" ]; then
    echo "=== [$(date -Is)] SKIP $tag (already done) ==="
    return
  fi
  echo "=== [$(date -Is)] START $tag ==="
  OMNIGIBSON_HEADLESS=true PYTHONUNBUFFERED=1 python -u \
    "$REPO/scripts/train_ppo.py" \
    --task_name "$task" --condition "$cond" --seed "$seed" --total_timesteps $STEPS \
    > "$LOG_DIR/${tag}.log" 2>&1
  if [ -f "$ckpt" ]; then
    echo "=== [$(date -Is)] DONE  $tag ===" | tee -a "$LOG_DIR/${tag}.log"
  else
    echo "=== [$(date -Is)] FAILED $tag ===" | tee -a "$LOG_DIR/${tag}.log"
  fi
}

ev() {
  local ckpt=$1 cond=$2 eval_task=$3 source_task=$4 run_dir=$5
  [ -f "$ckpt" ] || { echo "SKIP eval (no checkpoint): $ckpt"; return; }
  if [ -f "$run_dir/summary.jsonl" ]; then
    echo "=== [$(date -Is)] SKIP eval $run_dir (already done) ==="
    return
  fi
  echo "=== [$(date -Is)] eval: $(basename "$run_dir") ==="
  OMNIGIBSON_HEADLESS=true PYTHONUNBUFFERED=1 python -u \
    "$REPO/scripts/evaluate.py" \
    --checkpoint "$ckpt" --condition "$cond" --eval_task "$eval_task" \
    --source_task "$source_task" --n_episodes 20 --run_dir "$run_dir"
}

# ── Joint-exposure training: vector_lagrangian directly on food_in_microwave, 5 seeds ──
for seed in 0 1 2 3 4; do
  train food_in_microwave vector_lagrangian "$seed" "_joint"
done

# ── Joint-exposure in-distribution eval (RQ2's Delta_comp) ──
for seed in 0 1 2 3 4; do
  ev "$DATA/checkpoints/vector_lagrangian_food_in_microwave_${seed}_joint/final_model.zip" \
     vector_lagrangian food_in_microwave "food_in_microwave_joint_s${seed}" \
     "$DATA/runs/eval_vector_lagrangian_food_in_microwave_${seed}_joint_on_food_in_microwave"
done

# ── add_firewood-sourced zero-shot on food_in_microwave, 5 seeds x 4 conditions ──
for seed in 0 1 2 3 4; do
  for cond in task_only scalar_lagrangian vector_lagrangian fixed_weight; do
    ev "$DATA/checkpoints/${cond}_add_firewood_${seed}/final_model.zip" \
       "$cond" food_in_microwave "add_firewood_s${seed}" \
       "$DATA/runs/eval_${cond}_add_firewood_${seed}_on_food_in_microwave"
  done
done

echo "=== MANIP 100K PHASE 2 DONE ==="
