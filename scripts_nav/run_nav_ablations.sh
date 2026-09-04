#!/usr/bin/env bash
# Budget-sensitivity ablation (Day 24), scoped to goal_hazards_only/
# vector_lagrangian only (headline pair) given time realism. Modality
# dropout (Days 22-23) has no nav-domain equivalent yet -- src/cdp_nav has
# no CorruptedObservationWrapper analogue to src/cdp/corruption.py -- and
# is NOT run here; documented as a scope gap, not silently skipped.
set -euo pipefail
source /data/heng/miniconda3/etc/profile.d/conda.sh
conda activate cdp_nav
cd /home/heng/work/CDP

LOG_DIR=/data/heng/cdp/logs_nav/run_all_nav
mkdir -p "$LOG_DIR"

run() {
  local tag=$1; shift
  echo "=== [$(date -Is)] START $tag ==="
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 python -u \
    /home/heng/work/CDP/scripts_nav/train_ppo_nav.py \
    --task_name goal_hazards_only --condition vector_lagrangian --seed 0 "$@" \
    > "$LOG_DIR/${tag}.log" 2>&1
  echo "=== [$(date -Is)] DONE  $tag ===" | tee -a "$LOG_DIR/${tag}.log"
}

run vector_lagrangian_goal_hazards_only_0_budget12.5 --budget 12.5 &
run vector_lagrangian_goal_hazards_only_0_budget50.0 --budget 50.0 &
wait

echo "=== NAV ABLATIONS DONE ==="
