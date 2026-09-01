#!/usr/bin/env bash
# Sequential training queue for the manipulation domain (proposal.tex
# Phases 1-3). MUST run sequentially, one Isaac Sim process at a time —
# see docs/DECISIONS.md / docs/DAILY_LOG.md: parallel Isaac Sim processes
# on this machine hit kvdb lock contention and time out.
#
# Uniform 20,000-step budget across every run in this queue (deliberately,
# to avoid the budget confound documented in
# private/CONTRIBUTIONS_LOG.md entry 7, where add_firewood's task_only ran
# at 40k and its scalar/vector ran at 20k).
set -euo pipefail

source /data/heng/cdp/env.sh
conda activate oopsieverse_b1k
cd /data/heng/cdp/external/oopsieverse

STEPS=20000
LOG_DIR=/data/heng/cdp/logs/run_all_manip
mkdir -p "$LOG_DIR"

run() {
  local task=$1 cond=$2 extra=${3:-}
  local tag="${cond}_${task}${extra:+_${extra// /}}"
  echo "=== [$(date -Is)] START $tag ==="
  OMNIGIBSON_HEADLESS=true PYTHONUNBUFFERED=1 python -u \
    /home/heng/work/CDP/scripts/train_ppo.py \
    --task_name "$task" --condition "$cond" --seed 0 --total_timesteps $STEPS $extra \
    > "$LOG_DIR/${tag}.log" 2>&1
  echo "=== [$(date -Is)] DONE  $tag ==="
}

# ── Single-exposure: task_only / scalar_lagrangian / vector_lagrangian ──
for task in pick_egg add_firewood pour_water; do
  for cond in task_only scalar_lagrangian vector_lagrangian; do
    run "$task" "$cond"
  done
done

# ── RQ3 ablation: fixed_weight (structured obs, constant lambda_m) ──
for task in pick_egg add_firewood pour_water; do
  run "$task" fixed_weight
done

# ── RQ2 upper bound: vector_lagrangian trained directly on composite tasks ──
for task in fill_bowl food_in_microwave; do
  run "$task" vector_lagrangian
done

echo "=== ALL MANIP TRAINING DONE ==="
