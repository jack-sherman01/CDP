#!/usr/bin/env python3
"""
Day-1 verification loop: instantiate a real BEHAVIOR-1K/OmniGibson task
through DamageSim and confirm the health obs + damage_info evaluators
actually run end to end.

Run with:
    conda activate oopsieverse_b1k
    cd <oopsieverse repo root>
    python /home/heng/work/CDP/scripts/verify_damagesim.py --task_name add_firewood
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys

os.environ.setdefault("CARB_LOG_CHANNELS", "omni.physx.plugin=off")

OOPSIEVERSE_ROOT = "/data/heng/cdp/external/oopsieverse"
if OOPSIEVERSE_ROOT not in sys.path:
    sys.path.insert(0, OOPSIEVERSE_ROOT)

import torch as th
import omnigibson as og
from omnigibson.macros import gm

from damagesim.omnigibson.damageable_env import OGDamageableEnvironment

TASK_CONFIG_PACKAGE = "oopsiebench.envs.behavior1k"


def build_env_config(task_cfg):
    scene_config = dict(task_cfg.scene_config)
    if "type" not in scene_config:
        scene_config["type"] = "InteractiveTraversableScene"
    return {
        "env": {
            "action_frequency": task_cfg.action_frequency,
            "rendering_frequency": task_cfg.rendering_frequency,
            "physics_frequency": task_cfg.physics_frequency,
        },
        "scene": scene_config,
        "robots": [dict(task_cfg.robot_config)],
        "objects": [dict(obj) for obj in task_cfg.task_objects.values()],
        "task": {"type": "DummyTask", "activity_name": task_cfg.task_name},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_name", required=True)
    ap.add_argument("--n_steps", type=int, default=10)
    args = ap.parse_args()

    mod = importlib.import_module(f"{TASK_CONFIG_PACKAGE}.{args.task_name}")
    task_cfg = mod.get_task_config()

    gm.USE_GPU_DYNAMICS = task_cfg.use_gpu_dynamics
    gm.ENABLE_TRANSITION_RULES = task_cfg.enable_transition_rules

    env_config = build_env_config(task_cfg)
    env = OGDamageableEnvironment(configs=env_config)

    env.reset()
    if hasattr(mod, "reset") and callable(mod.reset):
        mod.reset(env)
    env._reset_damage_tracking()
    for _ in range(5):
        og.sim.step()

    robot = env.robots[0]
    zero_action = th.zeros(robot.action_dim, dtype=th.float32)

    print(f"\n=== VERIFY[{args.task_name}] health_list_link_names ({len(env.health_list_link_names)}) ===")
    print(env.health_list_link_names)

    for i in range(args.n_steps):
        obs, reward, terminated, truncated, info = env.step(zero_action.clone())

    print(f"\n=== VERIFY[{args.task_name}] obs['health'] after {args.n_steps} steps ===")
    print(obs.get("health"))

    print(f"\n=== VERIFY[{args.task_name}] damage_info evaluator kinds seen ===")
    kinds = set()
    for obj_name, part_info in info.get("damage_info", {}).items():
        for part_name, ev_dict in part_info.items():
            for ev_name in ev_dict:
                kinds.add(ev_name)
    print(sorted(kinds))

    print(f"\n=== VERIFY[{args.task_name}] PASS ===")
    sys.stdout.flush()
    og.shutdown()


if __name__ == "__main__":
    main()
