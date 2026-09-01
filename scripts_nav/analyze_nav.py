#!/usr/bin/env python3
"""
Statistical analysis for the Safety-Gymnasium domain — nav analogue of
scripts/analyze.py. Aggregates summary.jsonl across runs_nav/, computes
TCR/STCR with binomial SE + 95% CI per (condition, task), and the RQ1
primary comparison (vector_lagrangian vs scalar_lagrangian STCR).

Usage:
    python scripts_nav/analyze_nav.py --runs_glob "/data/heng/cdp/runs_nav/*/summary.jsonl" \
        --out_dir results_nav/
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cdp_nav.eval import load_summary, group_by, tcr, stcr, safety_gap, mean_field, median_field

Z_95 = 1.959963984540054


def rate_stderr(successes: int, n: int) -> float:
    if n == 0:
        return float("nan")
    p = successes / n
    return math.sqrt(p * (1 - p) / n)


def load_all(runs_glob: str):
    rows = []
    for path in glob.glob(runs_glob):
        rows.extend(load_summary(path))
    return rows


def group_by2(rows, key1, key2):
    out = {}
    for r in rows:
        out.setdefault((r[key1], r[key2]), []).append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_glob", default="/data/heng/cdp/runs_nav/*/summary.jsonl")
    ap.add_argument("--out_dir", default=os.path.join(os.path.dirname(__file__), "..", "results_nav"))
    args = ap.parse_args()

    rows = load_all(args.runs_glob)
    if not rows:
        print(f"No summary.jsonl rows found under {args.runs_glob!r}")
        sys.exit(1)
    print(f"Loaded {len(rows)} episodes from {args.runs_glob!r}")
    os.makedirs(args.out_dir, exist_ok=True)

    table_rows = []
    for (condition, task_name), group_rows in group_by2(rows, "condition", "task_name").items():
        n = len(group_rows)
        n_success = sum(1 for r in group_rows if r["success"])
        n_safe = sum(1 for r in group_rows if r["successful_and_safe"])
        tcr_v, stcr_v = tcr(group_rows), stcr(group_rows)
        table_rows.append({
            "condition": condition, "task_name": task_name, "n_episodes": n,
            "TCR": tcr_v, "TCR_ci95": Z_95 * rate_stderr(n_success, n),
            "STCR": stcr_v, "STCR_ci95": Z_95 * rate_stderr(n_safe, n),
            "SafetyGap": safety_gap(group_rows),
            "mean_total_cost": mean_field(group_rows, "total_cost"),
            "median_total_cost": median_field(group_rows, "total_cost"),
        })

    out_csv = os.path.join(args.out_dir, "main_comparison_table_nav.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table_rows[0].keys()))
        writer.writeheader()
        writer.writerows(table_rows)
    print(f"Wrote {out_csv}")

    print("\n=== Main comparison table (nav) ===")
    for r in table_rows:
        print(f"  {r['condition']:18s} {r['task_name']:25s} n={r['n_episodes']:4d} "
              f"TCR={r['TCR']:.3f}±{r['TCR_ci95']:.3f}  STCR={r['STCR']:.3f}±{r['STCR_ci95']:.3f}  "
              f"SafetyGap={r['SafetyGap']:.3f}  cost={r['mean_total_cost']:.1f}")

    print("\n=== RQ1: STCR_vector_lagrangian - STCR_scalar_lagrangian ===")
    by_task = group_by(rows, "task_name")
    for task_name, task_rows in by_task.items():
        by_cond = group_by(task_rows, "condition")
        if "vector_lagrangian" in by_cond and "scalar_lagrangian" in by_cond:
            delta = stcr(by_cond["vector_lagrangian"]) - stcr(by_cond["scalar_lagrangian"])
            print(f"  {task_name:25s} delta_STCR = {delta:+.3f}")


if __name__ == "__main__":
    main()
