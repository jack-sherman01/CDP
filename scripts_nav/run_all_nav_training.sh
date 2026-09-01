#!/usr/bin/env bash
# Training queue for the Safety-Gymnasium cross-domain validation
# (proposal.tex RQ4). Pure MuJoCo, cheap — unlike the manipulation domain's
# Isaac-Sim-backed queue, these run safely in PARALLEL (each capped to a
# few CPU threads so many fit on this machine's 128 cores).
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
  echo "=== [$(date -Is)] START $tag ==="
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 python -u \
    /home/heng/work/CDP/scripts_nav/train_ppo_nav.py \
    --task_name "$task" --condition "$cond" --seed 0 --total_timesteps $STEPS \
    > "$LOG_DIR/${tag}.log" 2>&1
  echo "=== [$(date -Is)] DONE  $tag ===" | tee -a "$LOG_DIR/${tag}.log"
}

jobs_started=0
wait_if_full() {
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do
    wait -n
  done
}

# ── Single-exposure: task_only / scalar_lagrangian / vector_lagrangian ──
for task in goal_hazards_only goal_vases_only button_gremlins_only button_wrong_button_only; do
  for cond in task_only scalar_lagrangian vector_lagrangian; do
    wait_if_full
    run "$task" "$cond" &
  done
done

# ── RQ3 ablation: fixed_weight ──
for task in goal_hazards_only goal_vases_only button_gremlins_only button_wrong_button_only; do
  wait_if_full
  run "$task" fixed_weight &
done

# ── RQ2 upper bound: vector_lagrangian trained directly on composite tasks ──
for task in goal_joint button_joint; do
  wait_if_full
  run "$task" vector_lagrangian &
done

wait
echo "=== ALL NAV TRAINING DONE ==="
