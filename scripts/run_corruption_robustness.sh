#!/usr/bin/env bash
# Damage-signal corruption robustness (Day 20 of the original plan,
# docs/PLAN.md). Pure eval-time -- no retraining needed, reuses the
# existing scalar_lagrangian/vector_lagrangian/fixed_weight pick_egg
# checkpoints (task_only excluded: no damage/health block to corrupt).
# pick_egg chosen for wall-clock reasons (fastest scene).
set -euo pipefail

source /data/heng/cdp/env.sh
conda activate oopsieverse_b1k
cd /data/heng/cdp/external/oopsieverse

DATA=/data/heng/cdp
REPO=/home/heng/work/CDP
N_EPISODES=${N_EPISODES:-10}

ev() {
  local cond=$1 kind=$2 param=$3
  local ckpt="$DATA/checkpoints/${cond}_pick_egg_0/final_model.zip"
  local tag="pick_egg_${kind}"
  [ -n "$param" ] && tag="${tag}${param}"
  local run_dir="$DATA/runs/eval_${cond}_pick_egg_0_on_${tag}"
  echo "=== $cond / $kind${param:+ param=$param} ==="
  local extra=()
  [ -n "$param" ] && extra=(--corruption_param "$param")
  OMNIGIBSON_HEADLESS=true PYTHONUNBUFFERED=1 python -u \
    "$REPO/scripts/evaluate.py" \
    --checkpoint "$ckpt" --condition "$cond" --eval_task pick_egg --source_task pick_egg \
    --n_episodes "$N_EPISODES" --corruption_kind "$kind" "${extra[@]}" --run_dir "$run_dir"
}

for cond in scalar_lagrangian vector_lagrangian fixed_weight; do
  ev "$cond" gaussian ""
  ev "$cond" modality_mask ""
  ev "$cond" held_constant ""
  ev "$cond" delay ""
  ev "$cond" scaling_error ""
done

echo "=== CORRUPTION ROBUSTNESS DONE ==="
