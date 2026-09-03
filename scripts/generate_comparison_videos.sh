#!/usr/bin/env bash
# Regenerate comparison videos with the new condition taxonomy (task_only/
# scalar_lagrangian/vector_lagrangian), superseding the pre-pivot videos in
# results/videos/ (see memory/project_video_deliverables). One episode per
# condition, --save_video, then stitched side-by-side via
# scripts/make_comparison_video.py.
set -euo pipefail

source /data/heng/cdp/env.sh
conda activate oopsieverse_b1k
cd /data/heng/cdp/external/oopsieverse

DATA=/data/heng/cdp
REPO=/home/heng/work/CDP
OUT_DIR="$REPO/results/videos"
mkdir -p "$OUT_DIR"

record() {
  local ckpt=$1 cond=$2 eval_task=$3 source_task=$4 run_dir=$5
  OMNIGIBSON_HEADLESS=true PYTHONUNBUFFERED=1 python -u \
    "$REPO/scripts/evaluate.py" \
    --checkpoint "$ckpt" --condition "$cond" --eval_task "$eval_task" \
    --source_task "$source_task" --n_episodes 1 --save_video --run_dir "$run_dir"
}

latest_video() {
  ls -t "$1"/videos/*.mp4 2>/dev/null | head -1
}

stitch() {
  local out=$1; shift
  python "$REPO/scripts/make_comparison_video.py" --videos "$@" --out "$out"
}

# ── In-distribution: pick_egg, add_firewood, pour_water ──
for task in pick_egg add_firewood pour_water; do
  labels=()
  for cond in task_only scalar_lagrangian vector_lagrangian; do
    run_dir="$DATA/runs/video_${cond}_${task}_0"
    record "$DATA/checkpoints/${cond}_${task}_0/final_model.zip" "$cond" "$task" "$task" "$run_dir"
    labels+=("${cond}=$(latest_video "$run_dir")")
  done
  stitch "$OUT_DIR/${task}_comparison.mp4" "${labels[@]}"
done

# ── Zero-shot: food_in_microwave (headline result) ──
labels=()
for cond in task_only scalar_lagrangian vector_lagrangian; do
  run_dir="$DATA/runs/video_${cond}_pick_egg_0_on_food_in_microwave"
  record "$DATA/checkpoints/${cond}_pick_egg_0/final_model.zip" "$cond" food_in_microwave pick_egg "$run_dir"
  labels+=("${cond}=$(latest_video "$run_dir")")
done
stitch "$OUT_DIR/food_in_microwave_zeroshot_comparison.mp4" "${labels[@]}"

echo "=== VIDEOS DONE ==="
ls -la "$OUT_DIR"
