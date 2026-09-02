#!/usr/bin/env bash
# Evaluation queue for the Safety-Gymnasium domain, run after
# run_all_nav_training.sh: (1) in-distribution eval of every checkpoint on
# its own training task, (2) zero-shot eval of every single-exposure
# checkpoint on its domain's joint/composite task, (3) in-distribution eval
# of the joint-exposure checkpoints on the composite task (needed for RQ2's
# Delta_comp). Sequential (cheap enough; avoids contending for CPU with
# itself) — ~20 episodes x 1000 steps per combo.
set -euo pipefail

source /data/heng/miniconda3/etc/profile.d/conda.sh
conda activate cdp_nav
cd /home/heng/work/CDP

DATA=/data/heng/cdp
N_EPISODES=${N_EPISODES:-20}

declare -A DOMAIN_TASKS
DOMAIN_TASKS[goal]="goal_hazards_only goal_vases_only"
DOMAIN_TASKS[button]="button_gremlins_only button_wrong_button_only"

declare -A JOINT_TASK
JOINT_TASK[goal]="goal_joint"
JOINT_TASK[button]="button_joint"

CONDITIONS="task_only scalar_lagrangian vector_lagrangian fixed_weight"

ev() {
  local ckpt=$1 cond=$2 eval_task=$3 source_task=$4 run_dir=$5
  [ -f "$ckpt" ] || { echo "SKIP (no checkpoint): $ckpt"; return; }
  PYTHONUNBUFFERED=1 python -u scripts_nav/evaluate_nav.py \
    --checkpoint "$ckpt" --condition "$cond" --eval_task "$eval_task" \
    --source_task "$source_task" --n_episodes "$N_EPISODES" --run_dir "$run_dir"
}

# ── (1) In-distribution + (2) zero-shot on the domain's joint task ──
for domain in goal button; do
  joint_task=${JOINT_TASK[$domain]}
  for task in ${DOMAIN_TASKS[$domain]}; do
    for cond in $CONDITIONS; do
      ckpt="$DATA/checkpoints_nav/${cond}_${task}_0/final_model.zip"
      echo "=== in-distribution: $cond / $task ==="
      ev "$ckpt" "$cond" "$task" "$task" "$DATA/runs_nav/eval_${cond}_${task}_0_on_${task}"
      echo "=== zero-shot: $cond / $task -> $joint_task ==="
      ev "$ckpt" "$cond" "$joint_task" "$task" "$DATA/runs_nav/eval_${cond}_${task}_0_on_${joint_task}"
    done
  done
done

# ── (3) Joint-exposure checkpoints, in-distribution on the composite task ──
for domain in goal button; do
  joint_task=${JOINT_TASK[$domain]}
  ckpt="$DATA/checkpoints_nav/vector_lagrangian_${joint_task}_0_joint/final_model.zip"
  echo "=== joint-exposure in-distribution: vector_lagrangian / $joint_task ==="
  ev "$ckpt" vector_lagrangian "$joint_task" "${joint_task}_joint" \
    "$DATA/runs_nav/eval_vector_lagrangian_${joint_task}_0_joint_on_${joint_task}"
done

echo "=== ALL NAV EVAL DONE ==="
