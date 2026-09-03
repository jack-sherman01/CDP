#!/usr/bin/env bash
# Evaluate the seed-1/seed-2 pick_egg checkpoints from
# run_manip_multiseed.sh: in-distribution + zero-shot on food_in_microwave.
set -euo pipefail

source /data/heng/cdp/env.sh
conda activate oopsieverse_b1k
cd /data/heng/cdp/external/oopsieverse

DATA=/data/heng/cdp
REPO=/home/heng/work/CDP
N_EPISODES=20

ev() {
  local ckpt=$1 cond=$2 eval_task=$3 source_task=$4 run_dir=$5
  [ -f "$ckpt" ] || { echo "SKIP (no checkpoint): $ckpt"; return; }
  OMNIGIBSON_HEADLESS=true PYTHONUNBUFFERED=1 python -u \
    "$REPO/scripts/evaluate.py" \
    --checkpoint "$ckpt" --condition "$cond" --eval_task "$eval_task" \
    --source_task "$source_task" --n_episodes "$N_EPISODES" --run_dir "$run_dir"
}

for seed in 1 2; do
  for cond in task_only scalar_lagrangian vector_lagrangian; do
    ckpt="$DATA/checkpoints/${cond}_pick_egg_${seed}/final_model.zip"
    echo "=== in-distribution: $cond / pick_egg / seed$seed ==="
    ev "$ckpt" "$cond" pick_egg "pick_egg_s${seed}" "$DATA/runs/eval_${cond}_pick_egg_${seed}_on_pick_egg"
    echo "=== zero-shot: $cond / pick_egg -> food_in_microwave / seed$seed ==="
    ev "$ckpt" "$cond" food_in_microwave "pick_egg_s${seed}" "$DATA/runs/eval_${cond}_pick_egg_${seed}_on_food_in_microwave"
  done
done

echo "=== MANIP MULTISEED EVAL DONE ==="
