#!/usr/bin/env python3
"""RQ2 for the Safety-Gymnasium domain — nav analogue of
scripts/compute_compositional_gap.py. See that file's docstring."""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cdp_nav.eval import (
    load_summary, tcr, stcr, safety_gap, mean_field, median_field,
    compositional_gap, violation_rate,
)
from cdp_nav.tasks import DOMAIN_MODALITIES, get_nav_task_spec


def summarize(rows, label, modalities, budget):
    print(f"  [{label}] n={len(rows)} TCR={tcr(rows):.3f} STCR={stcr(rows):.3f} "
          f"SafetyGap={safety_gap(rows):.3f} mean_cost={mean_field(rows, 'total_cost'):.1f} "
          f"median_cost={median_field(rows, 'total_cost'):.1f}")
    for m in modalities:
        v = violation_rate(rows, m, budget)
        if v == v:
            print(f"      V_{m} (cost > budget={budget}) = {v:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--single_run_dir", required=True)
    ap.add_argument("--joint_run_dir", required=True)
    ap.add_argument("--eval_task", required=True)
    ap.add_argument("--budget", type=float, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    task_spec = get_nav_task_spec(args.eval_task)
    modalities = DOMAIN_MODALITIES[task_spec.domain]
    budget = args.budget if args.budget is not None else task_spec.budget

    single_rows = load_summary(os.path.join(args.single_run_dir, "summary.jsonl"))
    joint_rows = load_summary(os.path.join(args.joint_run_dir, "summary.jsonl"))

    print(f"=== RQ2 (nav): compositional gap on {args.eval_task!r} ===")
    summarize(single_rows, "single-exposure (zero-shot)", modalities, budget)
    summarize(joint_rows, "joint-exposure (upper bound)", modalities, budget)

    delta = compositional_gap(single_rows, joint_rows)
    print(f"\n  Delta_comp = STCR_joint - STCR_single = "
          f"{stcr(joint_rows):.3f} - {stcr(single_rows):.3f} = {delta:+.3f}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({
                "eval_task": args.eval_task, "n_single": len(single_rows), "n_joint": len(joint_rows),
                "stcr_single": stcr(single_rows), "stcr_joint": stcr(joint_rows), "delta_comp": delta,
                "tcr_single": tcr(single_rows), "tcr_joint": tcr(joint_rows),
            }, f, indent=2)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
