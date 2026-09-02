#!/usr/bin/env bash
# Evaluation queue for the manipulation domain, run after
# run_all_manip_training.sh + run_remaining_manip_training.sh:
# (1) in-distribution eval of every single-exposure checkpoint on its own
# training task, (2) zero-shot eval of every single-exposure checkpoint on
# both composite tasks (food_in_microwave: FrankaPanda-matched, primary;
# fill_bowl: FrankaMounted-mismatched, known confound per docs/TASKS.md,
# generated for completeness but not the primary RQ1/RQ2 evidence), (3)
# in-distribution eval of the two joint-exposure checkpoints on their own
# composite task. Sequential -- one Isaac Sim process at a time.
set -euo pipefail

source /data/heng/cdp/env.sh
conda activate oopsieverse_b1k
cd /data/heng/cdp/external/oopsieverse

DATA=/data/heng/cdp
N_EPISODES=${N_EPISODES:-20}
CONDITIONS="task_only scalar_lagrangian vector_lagrangian fixed_weight"
TASKS="pick_egg add_firewood pour_water"

ev() {
  local ckpt=$1 cond=$2 eval_task=$3 source_task=$4 run_dir=$5
  [ -f "$ckpt" ] || { echo "SKIP (no checkpoint): $ckpt"; return; }
  OMNIGIBSON_HEADLESS=true PYTHONUNBUFFERED=1 python -u \
    /home/heng/work/CDP/scripts/evaluate.py \
    --checkpoint "$ckpt" --condition "$cond" --eval_task "$eval_task" \
    --source_task "$source_task" --n_episodes "$N_EPISODES" --run_dir "$run_dir"
}

for task in $TASKS; do
  for cond in $CONDITIONS; do
    ckpt="$DATA/checkpoints/${cond}_${task}_0/final_model.zip"
    echo "=== in-distribution: $cond / $task ==="
    ev "$ckpt" "$cond" "$task" "$task" "$DATA/runs/eval_${cond}_${task}_0_on_${task}"
    echo "=== zero-shot: $cond / $task -> food_in_microwave ==="
    ev "$ckpt" "$cond" "food_in_microwave" "$task" "$DATA/runs/eval_${cond}_${task}_0_on_food_in_microwave"
    echo "=== zero-shot: $cond / $task -> fill_bowl (known embodiment-mismatch confound) ==="
    ev "$ckpt" "$cond" "fill_bowl" "$task" "$DATA/runs/eval_${cond}_${task}_0_on_fill_bowl"
  done
done

echo "=== joint-exposure in-distribution ==="
ev "$DATA/checkpoints/vector_lagrangian_food_in_microwave_0_joint/final_model.zip" \
   vector_lagrangian food_in_microwave food_in_microwave_joint \
   "$DATA/runs/eval_vector_lagrangian_food_in_microwave_0_joint_on_food_in_microwave"
ev "$DATA/checkpoints/vector_lagrangian_fill_bowl_0_joint/final_model.zip" \
   vector_lagrangian fill_bowl fill_bowl_joint \
   "$DATA/runs/eval_vector_lagrangian_fill_bowl_0_joint_on_fill_bowl"

echo "=== ALL MANIP EVAL DONE ==="
