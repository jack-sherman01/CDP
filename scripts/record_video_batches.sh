#!/usr/bin/env bash
# Record MULTIPLE episodes per (task, condition) so the user can browse
# and pick the most compelling comparison themselves, rather than us
# auto-selecting one. Keeps every episode's video (tagged success/fail in
# the filename by scripts/evaluate.py / gym_env.py). Covers the 5
# in-distribution training tasks plus the food_in_microwave zero-shot
# scenario (pick_egg-sourced checkpoints, the RQ1 headline comparison).
set -euo pipefail

source /data/heng/cdp/env.sh
conda activate oopsieverse_b1k
cd /data/heng/cdp/external/oopsieverse

DATA=/data/heng/cdp
REPO=/home/heng/work/CDP
N_EPISODES=${N_EPISODES:-5}
CONDITIONS="task_only scalar_lagrangian vector_lagrangian fixed_weight"

record() {
  local ckpt=$1 cond=$2 eval_task=$3 source_task=$4 run_dir=$5
  [ -f "$ckpt" ] || { echo "SKIP (no checkpoint): $ckpt"; return; }
  echo "=== [$(date -Is)] recording $cond / $eval_task (source=$source_task) ==="
  OMNIGIBSON_HEADLESS=true PYTHONUNBUFFERED=1 python -u \
    "$REPO/scripts/evaluate.py" \
    --checkpoint "$ckpt" --condition "$cond" --eval_task "$eval_task" \
    --source_task "$source_task" --n_episodes "$N_EPISODES" --save_video --run_dir "$run_dir"
}

# In-distribution: all 5 training tasks
for task in pick_egg add_firewood pour_water open_drawer open_single_door; do
  for cond in $CONDITIONS; do
    ckpt="$DATA/checkpoints/${cond}_${task}_0/final_model.zip"
    record "$ckpt" "$cond" "$task" "$task" "$DATA/runs/video_batch_${cond}_${task}_0"
  done
done

# Zero-shot: food_in_microwave, from pick_egg-sourced checkpoints (headline RQ1 comparison)
for cond in $CONDITIONS; do
  ckpt="$DATA/checkpoints/${cond}_pick_egg_0/final_model.zip"
  record "$ckpt" "$cond" food_in_microwave pick_egg "$DATA/runs/video_batch_${cond}_pick_egg_0_on_food_in_microwave"
done

echo "=== VIDEO BATCH RECORDING DONE ==="
