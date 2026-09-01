#!/usr/bin/env python3
"""
Synthetic (no simulator needed) test of the Day-4 logging/eval pipeline.

Run with any Python that has numpy + matplotlib, e.g.:
    conda activate oopsieverse_b1k
    python /home/heng/work/CDP/tests/test_logger_eval.py
"""
from __future__ import annotations

import os
import shutil
import sys

import numpy as np

sys.path.insert(0, "/home/heng/work/CDP/src")

from cdp.logger import EpisodeLogger
from cdp.eval import (
    load_summary, tcr, stcr, safety_gap, zero_shot_gap,
    metrics_table, plot_learning_curve, plot_completion_rates,
    plot_damage_by_modality, plot_min_health_per_episode,
)

RUN_DIR = "/tmp/claude-1034/-home-heng-work-CDP/22108e7f-2286-4164-bb83-092d1b660465/scratchpad/test_run"


def make_episode(logger, condition, task, seed, rng, damaging: bool):
    logger.condition = condition
    logger.task_name = task
    logger.seed = seed
    logger.start_episode()
    n = 30
    min_health = 100.0
    for t in range(n):
        d_mech = float(rng.uniform(0, 5 if damaging else 0.5))
        d_therm = float(rng.uniform(0, 3 if damaging else 0.2))
        d_elec = float(rng.uniform(0, 2 if damaging else 0.1))
        min_health = max(0.0, min_health - (d_mech + d_therm + d_elec) * 0.1)
        logger.log_step(
            reward=1.0 - 0.01 * t,
            damage_by_modality={"mechanical": d_mech, "thermal": d_therm, "electrical": d_elec},
            min_object_health=min_health,
            robot_health=min_health,
            action=rng.uniform(-1, 1, size=7),
            terminated=(t == n - 1),
        )
    success = bool(rng.random() > (0.3 if damaging else 0.1))
    safe = success and min_health > 70.0
    return logger.end_episode(
        success=success, safe=safe,
        termination_reason="max_steps" if not success else "task_complete",
    )


def main():
    if os.path.exists(RUN_DIR):
        shutil.rmtree(RUN_DIR)
    rng = np.random.default_rng(0)

    logger = EpisodeLogger(run_dir=RUN_DIR, condition="task_only", task_name="pick_egg", seed=0)
    for cond, damaging in [
        ("task_only", True), ("scalar_lagrangian", True), ("vector_lagrangian", False),
    ]:
        for ep in range(15):
            make_episode(logger, cond, "pick_egg", ep, rng, damaging=damaging)

    rows = load_summary(os.path.join(RUN_DIR, "summary.jsonl"))
    assert len(rows) == 45, f"expected 45 rows, got {len(rows)}"

    table = metrics_table(rows, "condition")
    print("\n=== TEST[logger_eval] metrics_table ===")
    for cond, m in table.items():
        print(f"  {cond:10s} n={m['n_episodes']} TCR={m['tcr']:.2f} STCR={m['stcr']:.2f} "
              f"SafetyGap={m['safety_gap']:.2f} mean_dmg={m['mean_total_damage']:.2f}")

    seen = [r for r in rows if r["condition"] == "vector_lagrangian"]
    composite = [r for r in rows if r["condition"] == "task_only"]
    gap = zero_shot_gap(seen, composite)
    print(f"  zero_shot_gap(vector_lagrangian, task_only) = {gap:.3f}")

    plots_dir = os.path.join(RUN_DIR, "plots")
    plot_learning_curve(rows, os.path.join(plots_dir, "learning_curve.png"))
    plot_completion_rates(rows, os.path.join(plots_dir, "completion_rates.png"))
    plot_damage_by_modality(rows, os.path.join(plots_dir, "damage_by_modality.png"))
    plot_min_health_per_episode(rows, os.path.join(plots_dir, "min_health.png"))

    plot_files = os.listdir(plots_dir)
    all_ok = (
        len(rows) == 45
        and len(plot_files) == 4
        and all(os.path.getsize(os.path.join(plots_dir, f)) > 0 for f in plot_files)
    )
    print(f"\n=== TEST[logger_eval] {'PASS' if all_ok else 'FAIL'} (plots: {sorted(plot_files)}) ===")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
