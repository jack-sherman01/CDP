#!/usr/bin/env python3
"""
Zero-shot / in-distribution evaluation (Days 17-18): load a trained PPO
checkpoint and roll it out (no further training) on a given task, logging
episodes the same way training does.

Cross-task use (the actual point of Days 17-18): a checkpoint trained on a
single-modality task (e.g. `pick_egg`, condition `vector_lagrangian`) can be
evaluated directly on a held-out composite task (`fill_bowl`,
`food_in_microwave`) because
`DamageObservationWrapper` pads every task's observation to the same fixed
`MAX_TASK_OBJECTS` slot count (see `cdp/tasks.py`) — the policy network
never sees a shape it wasn't trained on.

Usage:
    conda activate oopsieverse_b1k
    cd /data/heng/cdp/external/oopsieverse
    OMNIGIBSON_HEADLESS=true PYTHONUNBUFFERED=1 python -u \
        /home/heng/work/CDP/scripts/evaluate.py \
        --checkpoint /data/heng/cdp/checkpoints/vector_lagrangian_pick_egg_0/final_model.zip \
        --condition vector_lagrangian --eval_task fill_bowl --n_episodes 20 \
        --run_dir /data/heng/cdp/runs/eval_vector_lagrangian_pick_egg_0_on_fill_bowl
"""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("CARB_LOG_CHANNELS", "omni.physx.plugin=off")

OOPSIEVERSE_ROOT = "/data/heng/cdp/external/oopsieverse"
CDP_SRC_ROOT = "/home/heng/work/CDP/src"
for _p in (OOPSIEVERSE_ROOT, CDP_SRC_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from stable_baselines3 import PPO

from cdp.corruption import CorruptionConfig
from cdp.gym_env import CDPTaskEnv
from cdp.lagrangian import LambdaState
from cdp.obs_wrapper import MODALITIES
from cdp.tasks import TRAINING_TASKS, COMPOSITE_EVAL_TASKS

CORRUPTION_KINDS = ["none", "gaussian", "modality_mask", "delay", "held_constant", "scaling_error"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="path to SB3 .zip")
    ap.add_argument("--condition", required=True,
                     choices=["task_only", "scalar_lagrangian", "vector_lagrangian", "fixed_weight"],
                     help="must match the condition the checkpoint was trained with")
    ap.add_argument("--eval_task", required=True,
                     choices=list(TRAINING_TASKS) + list(COMPOSITE_EVAL_TASKS))
    ap.add_argument("--source_task", default=None,
                     help="task the checkpoint was trained on, for the summary row / "
                          "run_dir default only (not required for the env itself)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_episodes", type=int, default=20)
    ap.add_argument("--deterministic", action="store_true", default=True)
    ap.add_argument("--run_dir", default=None)
    ap.add_argument("--corruption_kind", default="none", choices=CORRUPTION_KINDS,
                     help="Day-20 damage-signal corruption robustness test")
    ap.add_argument("--corruption_param", type=float, default=None,
                     help="gaussian_sigma / modality_mask_p / delay_steps / scaling_factor, "
                          "depending on --corruption_kind (uses CorruptionConfig defaults if omitted)")
    ap.add_argument("--save_video", action="store_true",
                     help="render + save an mp4 (with a health-bar side panel) per episode, "
                          "to {run_dir}/videos/ — for the results-website deliverable "
                          "(see memory/project_video_deliverables). Slower per episode; "
                          "combine with a small --n_episodes.")
    args = ap.parse_args()

    source_tag = args.source_task or "unknown"
    tag = args.eval_task if args.corruption_kind == "none" else f"{args.eval_task}_{args.corruption_kind}"
    run_dir = args.run_dir or os.path.join(
        "/data/heng/cdp/runs", f"eval_{args.condition}_{source_tag}_on_{tag}"
    )
    os.makedirs(run_dir, exist_ok=True)

    corruption = None
    if args.corruption_kind != "none":
        kwargs = {}
        if args.corruption_param is not None:
            param_field = {
                "gaussian": "gaussian_sigma", "modality_mask": "modality_mask_p",
                "delay": "delay_steps", "scaling_error": "scaling_factor",
            }.get(args.corruption_kind)
            if param_field:
                kwargs[param_field] = args.corruption_param
        corruption = CorruptionConfig(kind=args.corruption_kind, **kwargs)

    # Eval never trains further, so the multiplier(s) are frozen at whatever
    # the checkpoint finished training with (falls back to 0 / DEFAULT_LAMBDA
    # if lambda_final.json wasn't saved, e.g. for pre-pivot checkpoints —
    # harmless since eval's reward number is descriptive only, not used to
    # update the policy). fixed_weight always needs an explicit value.
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
        lam_state = LambdaState(saved_lambda if saved_lambda is not None else {m: 0.0 for m in MODALITIES})
    elif args.condition == "fixed_weight":
        fixed_lambda = saved_lambda if saved_lambda is not None else {m: 0.05 for m in MODALITIES}

    env = CDPTaskEnv(
        task_name=args.eval_task,
        condition=args.condition,
        seed=args.seed,
        run_dir=run_dir,
        lam_state=lam_state,
        fixed_lambda=fixed_lambda,
        corruption=corruption,
        record_video=args.save_video,
    )
    model = PPO.load(args.checkpoint)

    for ep in range(args.n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

    from cdp.eval import load_summary, metrics_table

    rows = load_summary(os.path.join(run_dir, "summary.jsonl"))
    table = metrics_table(rows, "task_name")
    print(f"\n=== EVAL[{args.condition} ckpt={os.path.basename(args.checkpoint)} "
          f"trained_on={source_tag} evaluated_on={args.eval_task}] ===")
    for task_name, m in table.items():
        print(f"  {task_name:15s} n={m['n_episodes']} TCR={m['tcr']:.2f} STCR={m['stcr']:.2f} "
              f"SafetyGap={m['safety_gap']:.2f} mean_dmg={m['mean_total_damage']:.2f}")
    sys.stdout.flush()
    env.close()


if __name__ == "__main__":
    main()
