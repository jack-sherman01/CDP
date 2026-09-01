#!/usr/bin/env python3
"""
Zero-shot / in-distribution evaluation for the Safety-Gymnasium domain —
nav analogue of scripts/evaluate.py. Cross-task use: a checkpoint trained
on `goal_hazards_only` (condition e.g. `vector_lagrangian`) can be evaluated
directly on `goal_joint` because NavObservationWrapper's vector mode
zero-pads missing hazard-type lidar slots to a fixed size (see
src/cdp_nav/obs_wrapper.py) — same role MAX_TASK_OBJECTS plays in the
manipulation domain.

Usage:
    conda activate cdp_nav
    PYTHONUNBUFFERED=1 python -u /home/heng/work/CDP/scripts_nav/evaluate_nav.py \
        --checkpoint /data/heng/cdp/checkpoints_nav/vector_lagrangian_goal_hazards_only_0/final_model.zip \
        --condition vector_lagrangian --eval_task goal_joint --n_episodes 20
"""
from __future__ import annotations

import argparse
import json
import os
import sys

CDP_SRC_ROOT = "/home/heng/work/CDP/src"
if CDP_SRC_ROOT not in sys.path:
    sys.path.insert(0, CDP_SRC_ROOT)

from stable_baselines3 import PPO

from cdp.lagrangian import LambdaState
from cdp_nav.gym_env import NavTaskEnv
from cdp_nav.tasks import DOMAIN_MODALITIES, NAV_COMPOSITE_EVAL_TASKS, NAV_TRAINING_TASKS, get_nav_task_spec

CDP_DATA_ROOT = os.environ.get("CDP_DATA_ROOT", "/data/heng/cdp")
ALL_NAV_TASK_NAMES = NAV_TRAINING_TASKS + NAV_COMPOSITE_EVAL_TASKS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--condition", required=True,
                     choices=["task_only", "scalar_lagrangian", "vector_lagrangian", "fixed_weight"])
    ap.add_argument("--eval_task", required=True, choices=ALL_NAV_TASK_NAMES)
    ap.add_argument("--source_task", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_episodes", type=int, default=20)
    ap.add_argument("--deterministic", action="store_true", default=True)
    ap.add_argument("--run_dir", default=None)
    args = ap.parse_args()

    source_tag = args.source_task or "unknown"
    run_dir = args.run_dir or os.path.join(
        CDP_DATA_ROOT, "runs_nav", f"eval_{args.condition}_{source_tag}_on_{args.eval_task}"
    )
    os.makedirs(run_dir, exist_ok=True)

    task_spec = get_nav_task_spec(args.eval_task)
    modalities = DOMAIN_MODALITIES[task_spec.domain]

    ckpt_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    lambda_final_path = os.path.join(ckpt_dir, "lambda_final.json")
    saved_lambda = None
    if os.path.exists(lambda_final_path):
        with open(lambda_final_path) as f:
            saved_lambda = json.load(f)

    lam_state = None
    fixed_lambda = None
    if args.condition == "scalar_lagrangian":
        lam_state = LambdaState(saved_lambda if saved_lambda is not None else 0.0)
    elif args.condition == "vector_lagrangian":
        lam_state = LambdaState(saved_lambda if saved_lambda is not None else {m: 0.0 for m in modalities})
    elif args.condition == "fixed_weight":
        fixed_lambda = saved_lambda if saved_lambda is not None else {m: 0.05 for m in modalities}

    env = NavTaskEnv(
        task_name=args.eval_task, condition=args.condition, seed=args.seed,
        run_dir=run_dir, lam_state=lam_state, fixed_lambda=fixed_lambda,
    )
    model = PPO.load(args.checkpoint)

    for ep in range(args.n_episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

    from cdp_nav.eval import load_summary, metrics_table

    rows = load_summary(os.path.join(run_dir, "summary.jsonl"))
    table = metrics_table(rows, "task_name")
    print(f"\n=== EVAL_NAV[{args.condition} ckpt={os.path.basename(args.checkpoint)} "
          f"trained_on={source_tag} evaluated_on={args.eval_task}] ===")
    for task_name, m in table.items():
        print(f"  {task_name:25s} n={m['n_episodes']} TCR={m['tcr']:.2f} STCR={m['stcr']:.2f} "
              f"SafetyGap={m['safety_gap']:.2f} mean_cost={m['mean_total_cost']:.2f}")
    sys.stdout.flush()
    env.close()


if __name__ == "__main__":
    main()
