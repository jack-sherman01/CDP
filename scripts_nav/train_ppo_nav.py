#!/usr/bin/env python3
"""
PPO training for one (nav task, condition, seed) on the Safety-Gymnasium
cross-domain validation (proposal.tex RQ4) — nav analogue of
scripts/train_ppo.py. Same architecture/hyperparameters and the SAME
`cdp.lagrangian`/`cdp.lagrangian_callback` PID-Lagrangian mechanism as the
manipulation domain (proposal's explicit "identical code reused across
domains" claim) — only the env/obs/reward glue differs (cdp_nav.*).

Cheap, pure-MuJoCo env: unlike scripts/train_ppo.py's Isaac-Sim-backed env
(one process at a time on this machine), several of these can run
concurrently.

Usage:
    conda activate cdp_nav
    PYTHONUNBUFFERED=1 python -u /home/heng/work/CDP/scripts_nav/train_ppo_nav.py \
        --task_name goal_hazards_only --condition vector_lagrangian --seed 0 \
        --total_timesteps 1000000
"""
from __future__ import annotations

import argparse
import json
import os
import sys

CDP_SRC_ROOT = "/home/heng/work/CDP/src"
if CDP_SRC_ROOT not in sys.path:
    sys.path.insert(0, CDP_SRC_ROOT)

import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from cdp.lagrangian import LambdaState, PIDLagrangianConfig, ScalarLagrangian, VectorLagrangian
from cdp.lagrangian_callback import LagrangianUpdateCallback
from cdp_nav.gym_env import NavTaskEnv
from cdp_nav.tasks import (
    DOMAIN_MODALITIES, NAV_COMPOSITE_EVAL_TASKS, NAV_TRAINING_TASKS, get_nav_task_spec,
)

CDP_DATA_ROOT = os.environ.get("CDP_DATA_ROOT", "/data/heng/cdp")
ALL_NAV_TASK_NAMES = NAV_TRAINING_TASKS + NAV_COMPOSITE_EVAL_TASKS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_name", required=True, choices=ALL_NAV_TASK_NAMES)
    ap.add_argument("--condition", required=True,
                     choices=["task_only", "scalar_lagrangian", "vector_lagrangian", "fixed_weight"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--total_timesteps", type=int, default=1_000_000)
    ap.add_argument("--n_steps", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--budget", type=float, default=None,
                     help="*_lagrangian only: per-modality episode cost budget "
                          "(default: NavTaskSpec.budget, Safety Gym benchmark convention = 25.0)")
    ap.add_argument("--k_p", type=float, default=1e-2)
    ap.add_argument("--k_i", type=float, default=1e-3)
    ap.add_argument("--k_d", type=float, default=1e-2)
    ap.add_argument("--fixed_lambda", type=float, default=0.05,
                     help="fixed_weight only: constant per-modality multiplier (RQ3 ablation).")
    ap.add_argument("--run_dir", default=None)
    args = ap.parse_args()

    task_spec = get_nav_task_spec(args.task_name)
    modalities = DOMAIN_MODALITIES[task_spec.domain]
    exposure = "joint" if args.task_name in NAV_COMPOSITE_EVAL_TASKS else "single"
    budget = args.budget if args.budget is not None else task_spec.budget

    experiment_id = f"{args.condition}_{args.task_name}_{args.seed}"
    if exposure == "joint":
        experiment_id += "_joint"
    if budget != task_spec.budget:
        experiment_id += f"_budget{budget}"
    if args.condition == "fixed_weight" and args.fixed_lambda != 0.05:
        experiment_id += f"_lambda{args.fixed_lambda}"
    run_dir = args.run_dir or os.path.join(CDP_DATA_ROOT, "runs_nav", experiment_id)
    os.makedirs(run_dir, exist_ok=True)

    pid_cfg = PIDLagrangianConfig(budget=budget, k_p=args.k_p, k_i=args.k_i, k_d=args.k_d)

    lagrangian = None
    lam_state = None
    fixed_lambda = None
    if args.condition == "scalar_lagrangian":
        lagrangian = ScalarLagrangian(pid_cfg, task_spec.max_episode_steps)
        lam_state = LambdaState(0.0)
    elif args.condition == "vector_lagrangian":
        lagrangian = VectorLagrangian({m: pid_cfg for m in modalities}, task_spec.max_episode_steps)
        lam_state = LambdaState({m: 0.0 for m in modalities})
    elif args.condition == "fixed_weight":
        fixed_lambda = {m: args.fixed_lambda for m in modalities}

    def make_env():
        env = NavTaskEnv(
            task_name=args.task_name, condition=args.condition, seed=args.seed,
            run_dir=run_dir, lam_state=lam_state, fixed_lambda=fixed_lambda,
        )
        return Monitor(env)

    vec_env = DummyVecEnv([make_env])

    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]), activation_fn=nn.Tanh)
    model = PPO(
        "MlpPolicy", vec_env, policy_kwargs=policy_kwargs,
        clip_range=0.2, gamma=0.99, gae_lambda=0.95, max_grad_norm=0.5,
        n_steps=args.n_steps, batch_size=args.batch_size, seed=args.seed, verbose=1,
        tensorboard_log=os.path.join(CDP_DATA_ROOT, "logs_nav", "tensorboard", experiment_id),
    )

    callback = (
        LagrangianUpdateCallback(
            lagrangian, lam_state, modalities=modalities, info_key="cost_by_modality", verbose=1,
        )
        if lagrangian is not None else None
    )
    model.learn(total_timesteps=args.total_timesteps, callback=callback)

    ckpt_dir = os.path.join(CDP_DATA_ROOT, "checkpoints_nav", experiment_id)
    os.makedirs(ckpt_dir, exist_ok=True)
    model.save(os.path.join(ckpt_dir, "final_model"))

    if lagrangian is not None:
        with open(os.path.join(ckpt_dir, "lagrangian_history.json"), "w") as f:
            json.dump(lagrangian.history, f)
        final_lambda = lagrangian.lam if args.condition == "scalar_lagrangian" else lagrangian.lam_vec
        with open(os.path.join(ckpt_dir, "lambda_final.json"), "w") as f:
            json.dump(final_lambda, f)
        print(f"[train_ppo_nav] final lambda: {final_lambda}")

    print(f"\n=== TRAIN_NAV[{experiment_id}] DONE — saved to {ckpt_dir}, logs in {run_dir} ===")
    sys.stdout.flush()
    vec_env.close()


if __name__ == "__main__":
    main()
