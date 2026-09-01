#!/usr/bin/env python3
"""
Live verification of DamageObservationWrapper against a real BEHAVIOR-1K
task env. Not a pytest unit test (needs Isaac Sim + a GPU) — run directly:

    conda activate oopsieverse_b1k
    cd /data/heng/cdp/external/oopsieverse
    OMNIGIBSON_HEADLESS=true PYTHONUNBUFFERED=1 python -u \
        /home/heng/work/CDP/tests/test_obs_wrapper.py --task_name add_firewood
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys

os.environ.setdefault("CARB_LOG_CHANNELS", "omni.physx.plugin=off")

OOPSIEVERSE_ROOT = "/data/heng/cdp/external/oopsieverse"
CDP_SRC_ROOT = "/home/heng/work/CDP/src"
for p in (OOPSIEVERSE_ROOT, CDP_SRC_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch as th
import omnigibson as og
from omnigibson.macros import gm

from damagesim.omnigibson.damageable_env import OGDamageableEnvironment
from cdp.obs_wrapper import DamageObservationWrapper, MODALITIES
from cdp.tasks import get_task_spec

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
    ap.add_argument("--n_steps", type=int, default=8)
    args = ap.parse_args()

    mod = importlib.import_module(f"{TASK_CONFIG_PACKAGE}.{args.task_name}")
    task_cfg = mod.get_task_config()

    gm.USE_GPU_DYNAMICS = task_cfg.use_gpu_dynamics
    gm.ENABLE_TRANSITION_RULES = task_cfg.enable_transition_rules

    base_env = OGDamageableEnvironment(configs=build_env_config(task_cfg))
    base_env.reset()
    if hasattr(mod, "reset") and callable(mod.reset):
        mod.reset(base_env)
    base_env._reset_damage_tracking()
    for _ in range(5):
        og.sim.step()

    robot = base_env.robots[0]
    zero_action = th.zeros(robot.action_dim, dtype=th.float32)
    task_objects = get_task_spec(args.task_name).task_object_names

    results = {}
    for mode in ("task", "scalar", "vector"):
        wrapper = DamageObservationWrapper(base_env, mode=mode, task_object_names=task_objects)
        obs, info = wrapper.reset()
        shapes = [obs.shape]
        finite_ok = bool(np.all(np.isfinite(obs)))
        health_ok = True
        for _ in range(args.n_steps):
            obs, reward, terminated, truncated, info = wrapper.step(zero_action.clone())
            shapes.append(obs.shape)
            finite_ok = finite_ok and bool(np.all(np.isfinite(obs)))
            if mode == "vector":
                health_ok = health_ok and bool(np.all((wrapper.last_h_vec >= 0) & (wrapper.last_h_vec <= 100)))
        results[mode] = {
            "shape": obs.shape,
            "shapes_constant": len(set(shapes)) == 1,
            "finite": finite_ok,
            "health_in_range": health_ok,
        }

    print("\n=== TEST[obs_wrapper] results ===")
    for mode, r in results.items():
        print(f"  {mode:8s} shape={r['shape']} constant_shape={r['shapes_constant']} "
              f"finite={r['finite']} health_in_range={r['health_in_range']}")

    task_len = results["task"]["shape"][0]
    scalar_len = results["scalar"]["shape"][0]
    vector_len = results["vector"]["shape"][0]
    prefix_ok = (scalar_len == task_len + 2) and (vector_len == task_len + 2 * len(MODALITIES))
    print(f"  prefix-length relationship holds: {prefix_ok} "
          f"(task={task_len}, scalar={scalar_len}, vector={vector_len})")

    all_ok = prefix_ok and all(
        r["shapes_constant"] and r["finite"] and r["health_in_range"] for r in results.values()
    )
    print(f"\n=== TEST[obs_wrapper] {'PASS' if all_ok else 'FAIL'} ===")
    sys.stdout.flush()
    og.shutdown()
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
