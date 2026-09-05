#!/usr/bin/env bash
# Closes entry 22's gap: RQ2's manipulation-domain finding (joint-exposure
# vector_lagrangian underperforming single-exposure zero-shot on
# food_in_microwave) was only ever checked at seed 0. Adds seeds 1-2 for
# the joint-exposure checkpoint itself, to pair against the already-
# existing 3-seed pick_egg-sourced zero-shot data
# (scripts/run_manip_multiseed.sh) for a real multi-seed Delta_comp.
# fill_bowl skipped -- known FrankaMounted-embodiment-mismatch confound
# (docs/TASKS.md), not informative regardless of seed count.
set -euo pipefail

source /data/heng/cdp/env.sh
conda activate oopsieverse_b1k
cd /data/heng/cdp/external/oopsieverse

STEPS=20000
LOG_DIR=/data/heng/cdp/logs/run_all_manip
mkdir -p "$LOG_DIR"

for seed in 1 2; do
  tag="vector_lagrangian_food_in_microwave_${seed}"
  echo "=== [$(date -Is)] START ${tag}_joint ==="
  OMNIGIBSON_HEADLESS=true PYTHONUNBUFFERED=1 python -u \
    /home/heng/work/CDP/scripts/train_ppo.py \
    --task_name food_in_microwave --condition vector_lagrangian --seed "$seed" --total_timesteps $STEPS \
    > "$LOG_DIR/${tag}_joint.log" 2>&1
  echo "=== [$(date -Is)] DONE  ${tag}_joint ===" | tee -a "$LOG_DIR/${tag}_joint.log"
done

echo "=== MANIP RQ2 MULTISEED TRAINING DONE ==="
