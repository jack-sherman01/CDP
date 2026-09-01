#!/usr/bin/env python3
"""
Statistical analysis (Day 21): aggregate summary.jsonl files across runs,
compute means/SE/95% CI per (condition, task), and emit the main
scalar-vs-structured comparison table + figures into results/.

Pure Python + numpy/matplotlib — no simulator needed, run with any env that
has them (e.g. the oopsieverse_b1k conda env, or system python3 with
`pip install numpy matplotlib`).

Usage:
    python scripts/analyze.py --runs_glob "/data/heng/cdp/runs/*/summary.jsonl" \
        --out_dir results/
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cdp.eval import load_summary, group_by, tcr, stcr, safety_gap, mean_field, median_field

Z_95 = 1.959963984540054  # two-sided 95% CI z-score


def stderr(vals):
    n = len(vals)
    if n < 2:
        return float("nan")
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    return math.sqrt(var / n)


def rate_stderr(successes: int, n: int) -> float:
    """Binomial SE for a proportion (TCR/STCR are proportions of episodes)."""
    if n == 0:
        return float("nan")
    p = successes / n
    return math.sqrt(p * (1 - p) / n)


def load_all(runs_glob: str):
    rows = []
    for path in glob.glob(runs_glob):
        rows.extend(load_summary(path))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_glob", default="/data/heng/cdp/runs/*/summary.jsonl")
    ap.add_argument("--out_dir", default=os.path.join(os.path.dirname(__file__), "..", "results"))
    args = ap.parse_args()

    rows = load_all(args.runs_glob)
    if not rows:
        print(f"No summary.jsonl rows found under {args.runs_glob!r}")
        sys.exit(1)
    print(f"Loaded {len(rows)} episodes from {args.runs_glob!r}")

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Per (condition, task) table with CIs ──────────────────────────
    table_rows = []
    for (condition, task_name), group_rows in group_by2(rows, "condition", "task_name").items():
        n = len(group_rows)
        n_success = sum(1 for r in group_rows if r["success"])
        n_safe = sum(1 for r in group_rows if r["successful_and_safe"])
        tcr_v, stcr_v = tcr(group_rows), stcr(group_rows)
        tcr_se, stcr_se = rate_stderr(n_success, n), rate_stderr(n_safe, n)
        dmg_vals = [r["total_damage"] for r in group_rows]
        table_rows.append({
            "condition": condition,
            "task_name": task_name,
            "n_episodes": n,
            "TCR": tcr_v, "TCR_se": tcr_se, "TCR_ci95": Z_95 * tcr_se,
            "STCR": stcr_v, "STCR_se": stcr_se, "STCR_ci95": Z_95 * stcr_se,
            "SafetyGap": safety_gap(group_rows),
            "mean_total_damage": mean_field(group_rows, "total_damage"),
            "median_total_damage": median_field(group_rows, "total_damage"),
            "damage_se": stderr(dmg_vals), "damage_ci95": Z_95 * stderr(dmg_vals),
        })

    out_csv = os.path.join(args.out_dir, "main_comparison_table.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table_rows[0].keys()))
        writer.writeheader()
        writer.writerows(table_rows)
    print(f"Wrote {out_csv}")

    print("\n=== Main comparison table ===")
    for r in table_rows:
        print(
            f"  {r['condition']:10s} {r['task_name']:15s} n={r['n_episodes']:4d} "
            f"TCR={r['TCR']:.3f}±{r['TCR_ci95']:.3f}  STCR={r['STCR']:.3f}±{r['STCR_ci95']:.3f}  "
            f"SafetyGap={r['SafetyGap']:.3f}  dmg={r['mean_total_damage']:.1f}±{r['damage_ci95']:.1f}"
        )

    # ── Primary comparison (RQ1): STCR_vector_lagrangian - STCR_scalar_lagrangian, per task ──
    print("\n=== RQ1: STCR_vector_lagrangian - STCR_scalar_lagrangian ===")
    by_task = group_by(rows, "task_name")
    for task_name, task_rows in by_task.items():
        by_cond = group_by(task_rows, "condition")
        if "vector_lagrangian" in by_cond and "scalar_lagrangian" in by_cond:
            delta = stcr(by_cond["vector_lagrangian"]) - stcr(by_cond["scalar_lagrangian"])
            print(f"  {task_name:15s} delta_STCR(vector_lagrangian - scalar_lagrangian) = {delta:+.3f}")

    # ── RQ2: Delta_comp = STCR_joint-exposure - STCR_single-exposure ──────
    # Same eval_task, vector_lagrangian trained joint (directly on the
    # composite task) vs. single (zero-shot from a single-hazard task);
    # exposure is read off the `experiment_id` convention train_ppo.py
    # writes ("_joint" suffix), which flows into `condition` only if the
    # caller tagged it — analyze.py instead disambiguates via `source_task`
    # recorded in the eval run_dir name (see scripts/evaluate.py), so this
    # comparison is done by scripts/compute_compositional_gap.py once both
    # single- and joint-exposure eval runs exist for the same eval_task.

    from cdp.eval import (
        plot_learning_curve, plot_completion_rates,
        plot_damage_by_modality, plot_min_health_per_episode,
    )
    plots_dir = os.path.join(args.out_dir, "figures")
    plot_learning_curve(rows, os.path.join(plots_dir, "learning_curve.png"))
    plot_completion_rates(rows, os.path.join(plots_dir, "completion_rates.png"))
    plot_damage_by_modality(rows, os.path.join(plots_dir, "damage_by_modality.png"))
    plot_min_health_per_episode(rows, os.path.join(plots_dir, "min_health.png"))
    print(f"\nWrote figures to {plots_dir}")


def group_by2(rows, key1, key2):
    out = {}
    for r in rows:
        out.setdefault((r[key1], r[key2]), []).append(r)
    return out


if __name__ == "__main__":
    main()
