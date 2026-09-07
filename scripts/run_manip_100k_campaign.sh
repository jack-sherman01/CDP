#!/usr/bin/env bash
# The publication-grade re-run (2026-09-07): 100,000-step budget (vs. the
# original 20,000 used for wall-clock reasons — that budget left TCR ~0
# everywhere, a floor effect that undercuts every downstream comparison),
# 5 seeds (proposal's target for this domain), full 4-condition sweep,
# on the 3 core single-hazard tasks. This is a multi-week continuous
# background campaign — RESUMABLE: skips any (task, condition, seed) whose
# final_model.zip already exists, so a session interruption just needs a
# re-launch, not a restart from scratch. Sequential — confirmed via a live
# throughput test that 2 parallel Isaac Sim processes (even on separate
# GPUs, isolated caches) do NOT increase total throughput (each drops to
# ~half fps under contention) — see docs/DECISIONS.md.
set -uo pipefail  # not -e: one run's failure shouldn't kill the whole campaign

source /data/heng/cdp/env.sh
conda activate oopsieverse_b1k
cd /data/heng/cdp/external/oopsieverse

STEPS=100000
DATA=/data/heng/cdp
LOG_DIR="$DATA/logs/run_manip_100k_campaign"
mkdir -p "$LOG_DIR"

run() {
  local task=$1 cond=$2 seed=$3
  local tag="${cond}_${task}_${seed}"
  local ckpt="$DATA/checkpoints/${tag}/final_model.zip"
  if [ -f "$ckpt" ]; then
    echo "=== [$(date -Is)] SKIP $tag (already done) ==="
    return
  fi
  echo "=== [$(date -Is)] START $tag ==="
  OMNIGIBSON_HEADLESS=true PYTHONUNBUFFERED=1 python -u \
    /home/heng/work/CDP/scripts/train_ppo.py \
    --task_name "$task" --condition "$cond" --seed "$seed" --total_timesteps $STEPS \
    > "$LOG_DIR/${tag}.log" 2>&1
  if [ -f "$ckpt" ]; then
    echo "=== [$(date -Is)] DONE  $tag ===" | tee -a "$LOG_DIR/${tag}.log"
  else
    echo "=== [$(date -Is)] FAILED $tag (no checkpoint written, check log) ===" | tee -a "$LOG_DIR/${tag}.log"
  fi
}

# Ordered fastest-scene-first within each seed so partial progress is
# maximally useful early (pick_egg/pour_water ~ house_single_floor, faster;
# add_firewood ~ Rs_int, much slower).
for seed in 0 1 2 3 4; do
  for task in pick_egg pour_water add_firewood; do
    for cond in task_only scalar_lagrangian vector_lagrangian fixed_weight; do
      run "$task" "$cond" "$seed"
    done
  done
done

echo "=== MANIP 100K CAMPAIGN (core 3 tasks) DONE ==="
