#!/usr/bin/env python3
"""
RQ2: Delta_comp = STCR_joint-exposure - STCR_single-exposure, both evaluated
on the SAME held-out hazard-combination task (proposal.tex "Evaluation
Metrics and Statistical Protocol").

Takes two run_dirs already produced by scripts/evaluate.py (single-exposure
checkpoint's zero-shot eval on the composite task) and by a fresh
scripts/evaluate.py call for the joint-exposure checkpoint evaluated
in-distribution on that same composite task (i.e. --eval_task equal to the
task it was trained on, condition vector_lagrangian, checkpoint tagged
"_joint"). Pure Python, no simulator needed.

Usage:
    python scripts/compute_compositional_gap.py \
        --single_run_dir /data/heng/cdp/runs/eval_vector_lagrangian_pick_egg_0_on_food_in_microwave \
        --joint_run_dir /data/heng/cdp/runs/eval_vector_lagrangian_food_in_microwave_0_joint_on_food_in_microwave \
        --eval_task food_in_microwave
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cdp.eval import (
    load_summary, tcr, stcr, safety_gap, mean_field, median_field,
    compositional_gap, violation_rate,
)
from cdp.lagrangian import DEFAULT_BUDGET

MODALITIES = ("mechanical", "thermal", "electrical")


def summarize(rows, label, budget):
    print(f"  [{label}] n={len(rows)} TCR={tcr(rows):.3f} STCR={stcr(rows):.3f} "
          f"SafetyGap={safety_gap(rows):.3f} mean_dmg={mean_field(rows, 'total_damage'):.1f} "
          f"median_dmg={median_field(rows, 'total_damage'):.1f}")
    for m in MODALITIES:
        v = violation_rate(rows, m, budget)
        if v == v:  # not NaN
            print(f"      V_{m} (damage > budget={budget}) = {v:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--single_run_dir", required=True,
                     help="run_dir from evaluate.py: single-exposure checkpoint, zero-shot on the composite task")
    ap.add_argument("--joint_run_dir", required=True,
                     help="run_dir from evaluate.py: joint-exposure checkpoint, in-distribution on the same composite task")
    ap.add_argument("--eval_task", required=True)
    ap.add_argument("--budget", type=float, default=DEFAULT_BUDGET,
                     help="per-modality budget for V_m reporting (health points/episode)")
    ap.add_argument("--out", default=None, help="optional path to write a JSON summary")
    args = ap.parse_args()

    single_rows = load_summary(os.path.join(args.single_run_dir, "summary.jsonl"))
    joint_rows = load_summary(os.path.join(args.joint_run_dir, "summary.jsonl"))

    print(f"=== RQ2: compositional gap on {args.eval_task!r} ===")
    summarize(single_rows, "single-exposure (zero-shot)", args.budget)
    summarize(joint_rows, "joint-exposure (upper bound)", args.budget)

    delta = compositional_gap(single_rows, joint_rows)
    print(f"\n  Delta_comp = STCR_joint - STCR_single = "
          f"{stcr(joint_rows):.3f} - {stcr(single_rows):.3f} = {delta:+.3f}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({
                "eval_task": args.eval_task,
                "n_single": len(single_rows), "n_joint": len(joint_rows),
                "stcr_single": stcr(single_rows), "stcr_joint": stcr(joint_rows),
                "delta_comp": delta,
                "tcr_single": tcr(single_rows), "tcr_joint": tcr(joint_rows),
                "mean_damage_single": mean_field(single_rows, "total_damage"),
                "mean_damage_joint": mean_field(joint_rows, "total_damage"),
            }, f, indent=2)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
