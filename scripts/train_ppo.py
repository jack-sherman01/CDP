#!/usr/bin/env python3
"""
PPO training (Days 5-6, 8-9, 10-12) for one (task, condition, seed).

Hyperparameters fixed per proposal.tex "Days 5-6":
  2-layer MLP, 256 units/layer, tanh; PPO clip 0.2; gamma 0.99;
  GAE lambda 0.95; grad-norm clip 0.5; Gaussian continuous actions.

Usage:
    conda activate oopsieverse_b1k
    cd /data/heng/cdp/external/oopsieverse
    OMNIGIBSON_HEADLESS=true PYTHONUNBUFFERED=1 python -u \
        /home/heng/work/CDP/scripts/train_ppo.py \
        --task_name pick_egg --condition task_only --seed 0 \
        --total_timesteps 200000
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("CARB_LOG_CHANNELS", "omni.physx.plugin=off")

OOPSIEVERSE_ROOT = "/data/heng/cdp/external/oopsieverse"
CDP_SRC_ROOT = "/home/heng/work/CDP/src"
for _p in (OOPSIEVERSE_ROOT, CDP_SRC_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import json

import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from cdp.corruption import CorruptionConfig
from cdp.gym_env import CDPTaskEnv
from cdp.lagrangian import (
    LambdaState, PIDLagrangianConfig, ScalarLagrangian, VectorLagrangian,
)
from cdp.lagrangian_callback import LagrangianUpdateCallback
from cdp.obs_wrapper import MODALITIES
from cdp.tasks import COMPOSITE_EVAL_TASKS, TRAINING_TASKS, get_task_spec

CDP_DATA_ROOT = os.environ.get("CDP_DATA_ROOT", "/data/heng/cdp")

# Every registered task is a legal --task_name: single-exposure conditions
# train on TRAINING_TASKS (one hazard modality each); the vector_lagrangian
# joint-exposure upper bound (proposal.tex "Joint-exposure") trains the same
# mechanism directly on a COMPOSITE_EVAL_TASKS task instead — the exposure
# regime is entirely a function of which task_name is passed, not a
# separate code path.
ALL_TASK_NAMES = TRAINING_TASKS + COMPOSITE_EVAL_TASKS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_name", required=True, choices=ALL_TASK_NAMES)
    ap.add_argument("--condition", required=True,
                     choices=["task_only", "scalar_lagrangian", "vector_lagrangian", "fixed_weight"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--total_timesteps", type=int, default=200_000)
    ap.add_argument("--n_steps", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--budget", type=float, default=30.0,
                     help="*_lagrangian only: health points/episode the PID controller enforces "
                          "(per proposal's b_m); Day-24-style ablation via --budget_low/med/high sweep.")
    # Proportional-dominant: a 20k-step run only yields ~10 PPO rollouts,
    # i.e. ~10 PID updates — not enough for slow integral accumulation to
    # reach a consequential lambda, so K_P must produce an immediately
    # meaningful penalty from the first update. See src/cdp/lagrangian.py's
    # "Gain history" docstring for the tuning failure these replace.
    ap.add_argument("--k_p", type=float, default=1.0)
    ap.add_argument("--k_i", type=float, default=0.02)
    ap.add_argument("--k_d", type=float, default=0.3)
    ap.add_argument("--fixed_lambda", type=float, default=0.05,
                     help="fixed_weight only: constant per-modality multiplier (RQ3 ablation).")
    ap.add_argument("--modality_dropout_p", type=float, default=0.0,
                     help="Days 22-23 ablation: P(each modality channel zeroed during training), "
                          "e.g. 0.2 per the proposal. 0 = disabled (default).")
    ap.add_argument("--run_dir", default=None)
    args = ap.parse_args()

    exposure = "joint" if args.task_name in COMPOSITE_EVAL_TASKS else "single"
    experiment_id = f"{args.condition}_{args.task_name}_{args.seed}"
    if exposure == "joint":
        experiment_id += "_joint"
    if args.modality_dropout_p > 0:
        experiment_id += f"_dropout{args.modality_dropout_p}"
    if args.budget != 30.0:
        experiment_id += f"_budget{args.budget}"
    if args.condition == "fixed_weight" and args.fixed_lambda != 0.05:
        experiment_id += f"_lambda{args.fixed_lambda}"
    run_dir = args.run_dir or os.path.join(CDP_DATA_ROOT, "runs", experiment_id)
    os.makedirs(run_dir, exist_ok=True)

    corruption = (
        CorruptionConfig(kind="modality_mask", modality_mask_p=args.modality_dropout_p)
        if args.modality_dropout_p > 0 else None
    )

    max_episode_steps = get_task_spec(args.task_name).max_episode_steps
    pid_cfg = PIDLagrangianConfig(budget=args.budget, k_p=args.k_p, k_i=args.k_i, k_d=args.k_d)

    lagrangian = None
    lam_state = None
    fixed_lambda = None
    if args.condition == "scalar_lagrangian":
        lagrangian = ScalarLagrangian(pid_cfg, max_episode_steps)
        lam_state = LambdaState(0.0)
    elif args.condition == "vector_lagrangian":
        lagrangian = VectorLagrangian({m: pid_cfg for m in MODALITIES}, max_episode_steps)
        lam_state = LambdaState({m: 0.0 for m in MODALITIES})
    elif args.condition == "fixed_weight":
        fixed_lambda = {m: args.fixed_lambda for m in MODALITIES}

    def make_env():
        env = CDPTaskEnv(
            task_name=args.task_name,
            condition=args.condition,
            seed=args.seed,
            run_dir=run_dir,
            lam_state=lam_state,
            fixed_lambda=fixed_lambda,
            corruption=corruption,
        )
        return Monitor(env)

    vec_env = DummyVecEnv([make_env])

    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]), activation_fn=nn.Tanh)
    model = PPO(
        "MlpPolicy",
        vec_env,
        policy_kwargs=policy_kwargs,
        clip_range=0.2,
        gamma=0.99,
        gae_lambda=0.95,
        max_grad_norm=0.5,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        seed=args.seed,
        verbose=1,
        tensorboard_log=os.path.join(CDP_DATA_ROOT, "logs", "tensorboard", experiment_id),
    )

    callback = (
        LagrangianUpdateCallback(lagrangian, lam_state, modalities=MODALITIES, verbose=1)
        if lagrangian is not None else None
    )
    model.learn(total_timesteps=args.total_timesteps, callback=callback)

    ckpt_dir = os.path.join(CDP_DATA_ROOT, "checkpoints", experiment_id)
    os.makedirs(ckpt_dir, exist_ok=True)
    model.save(os.path.join(ckpt_dir, "final_model"))

    if lagrangian is not None:
        # Reproducibility (proposal.tex Phase 4 checklist) + lets evaluate.py
        # log a meaningful reward value instead of an arbitrary placeholder.
        history = lagrangian.history
        with open(os.path.join(ckpt_dir, "lagrangian_history.json"), "w") as f:
            json.dump(history, f)
        final_lambda = lagrangian.lam if args.condition == "scalar_lagrangian" else lagrangian.lam_vec
        with open(os.path.join(ckpt_dir, "lambda_final.json"), "w") as f:
            json.dump(final_lambda, f)
        print(f"[train_ppo] final lambda: {final_lambda}")

    print(f"\n=== TRAIN[{experiment_id}] DONE — saved to {ckpt_dir}, logs in {run_dir} ===")
    sys.stdout.flush()
    vec_env.close()


if __name__ == "__main__":
    main()
