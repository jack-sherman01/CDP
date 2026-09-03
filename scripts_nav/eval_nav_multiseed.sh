#!/usr/bin/env bash
# Evaluate the seed-1/seed-2 nav checkpoints from run_nav_multiseed.sh:
# in-distribution + zero-shot on the composite, plus joint-exposure
# in-distribution.
set -euo pipefail
source /data/heng/miniconda3/etc/profile.d/conda.sh
conda activate cdp_nav
cd /home/heng/work/CDP

DATA=/data/heng/cdp
N_EPISODES=20

declare -A DOMAIN_TASKS
DOMAIN_TASKS[goal]="goal_hazards_only goal_vases_only"
DOMAIN_TASKS[button]="button_gremlins_only button_wrong_button_only"
declare -A JOINT_TASK
JOINT_TASK[goal]="goal_joint"
JOINT_TASK[button]="button_joint"

ev() {
  local ckpt=$1 cond=$2 eval_task=$3 source_task=$4 run_dir=$5
  [ -f "$ckpt" ] || { echo "SKIP (no checkpoint): $ckpt"; return; }
  PYTHONUNBUFFERED=1 python -u scripts_nav/evaluate_nav.py \
    --checkpoint "$ckpt" --condition "$cond" --eval_task "$eval_task" \
    --source_task "$source_task" --n_episodes "$N_EPISODES" --run_dir "$run_dir"
}

for seed in 1 2; do
  for domain in goal button; do
    joint_task=${JOINT_TASK[$domain]}
    for task in ${DOMAIN_TASKS[$domain]}; do
      for cond in scalar_lagrangian vector_lagrangian; do
        ckpt="$DATA/checkpoints_nav/${cond}_${task}_${seed}/final_model.zip"
        echo "=== in-distribution: $cond / $task / seed$seed ==="
        ev "$ckpt" "$cond" "$task" "${task}_s${seed}" "$DATA/runs_nav/eval_${cond}_${task}_${seed}_on_${task}"
        echo "=== zero-shot: $cond / $task -> $joint_task / seed$seed ==="
        ev "$ckpt" "$cond" "$joint_task" "${task}_s${seed}" "$DATA/runs_nav/eval_${cond}_${task}_${seed}_on_${joint_task}"
      done
    done
  done
  for domain in goal button; do
    joint_task=${JOINT_TASK[$domain]}
    ckpt="$DATA/checkpoints_nav/vector_lagrangian_${joint_task}_${seed}_joint/final_model.zip"
    echo "=== joint-exposure in-distribution: vector_lagrangian / $joint_task / seed$seed ==="
    ev "$ckpt" vector_lagrangian "$joint_task" "${joint_task}_joint_s${seed}" \
      "$DATA/runs_nav/eval_vector_lagrangian_${joint_task}_${seed}_joint_on_${joint_task}"
  done
done

echo "=== NAV MULTISEED EVAL DONE ==="
