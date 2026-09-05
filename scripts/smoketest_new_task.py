#!/usr/bin/env python3
"""One-off smoke test for a single new task (2026-09-05 task-diversity
expansion): construct CDPTaskEnv, run a handful of random steps, print
reward/damage/success so obvious breakage (missing object, wrong reward
mode wiring, etc.) is caught before spending training compute."""
import sys

OOPSIEVERSE_ROOT = "/data/heng/cdp/external/oopsieverse"
CDP_SRC_ROOT = "/home/heng/work/CDP/src"
for _p in (OOPSIEVERSE_ROOT, CDP_SRC_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cdp.gym_env import CDPTaskEnv
from cdp.lagrangian import LambdaState

task_name = sys.argv[1]
env = CDPTaskEnv(
    task_name=task_name, condition="vector_lagrangian", seed=0,
    lam_state=LambdaState({"mechanical": 0.1, "thermal": 0.0, "electrical": 0.0}),
)
print(f"[smoketest] {task_name} obs_space={env.observation_space.shape} action_space={env.action_space.shape}")
obs, info = env.reset()
print(f"[smoketest] reset OK, obs shape={obs.shape}")
total_r = 0.0
for i in range(15):
    a = env.action_space.sample()
    obs, r, term, trunc, info = env.step(a)
    total_r += r
    if term or trunc:
        print(f"[smoketest] episode ended at step {i}")
        break
print(f"[smoketest] {task_name} ran {i+1} steps, total_reward={total_r:.3f}, damage_by_modality={info.get('damage_by_modality')}")
env.close()
print(f"[smoketest] {task_name} PASS")
