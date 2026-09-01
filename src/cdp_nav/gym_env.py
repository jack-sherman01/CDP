"""
Gymnasium-compatible env for SB3 PPO training on the Safety-Gymnasium domain
— nav analogue of src/cdp/gym_env.py::CDPTaskEnv. Combines: NavObservation
Wrapper (obs mode), apply_cost_penalty (reward), the env's own `goal_met`
info flag (success), a per-modality cost budget (safety), and
NavEpisodeLogger.

Unlike CDPTaskEnv, `safety_gymnasium` envs are cheap (pure MuJoCo, no Isaac
Sim) — many of these can run concurrently on this machine, unlike the
Isaac-Sim-backed manipulation env which is limited to one process at a time.
"""
from __future__ import annotations

import os
from typing import Optional

import gymnasium as gym
import numpy as np
import safety_gymnasium as sg
from gymnasium import spaces

from cdp.lagrangian import LambdaState
from cdp_nav.custom_tasks import register_custom_tasks
from cdp_nav.logger import NavEpisodeLogger
from cdp_nav.obs_wrapper import NavObservationWrapper, ObsMode
from cdp_nav.reward import apply_cost_penalty, compute_cost_by_modality
from cdp_nav.tasks import DOMAIN_MODALITIES, NavTaskSpec, get_nav_task_spec

register_custom_tasks()

_OBS_MODE_BY_CONDITION = {
    "task_only": "task",
    "scalar_lagrangian": "scalar",
    "vector_lagrangian": "vector",
    "fixed_weight": "vector",
}


class NavTaskEnv(gym.Env):
    def __init__(
        self,
        task_name: str,
        condition: str,
        seed: int = 0,
        run_dir: Optional[str] = None,
        lam_state: Optional[LambdaState] = None,
        fixed_lambda=None,  # fixed_weight only: {modality: float}
    ):
        super().__init__()
        self.task_name = task_name
        self.condition = condition
        self.seed_value = seed
        self.spec_: NavTaskSpec = get_nav_task_spec(task_name)
        self.modalities = DOMAIN_MODALITIES[self.spec_.domain]

        if condition == "fixed_weight":
            assert fixed_lambda is not None, "fixed_weight needs fixed_lambda"
            self.lam_state = LambdaState(fixed_lambda)
        elif condition in ("scalar_lagrangian", "vector_lagrangian"):
            self.lam_state = lam_state if lam_state is not None else LambdaState(
                0.0 if condition == "scalar_lagrangian" else {m: 0.0 for m in self.modalities}
            )
        else:
            self.lam_state = LambdaState(None)

        raw_env = sg.make(
            self.spec_.env_id,
            config={"agent_name": self.spec_.agent_name, "observation_flatten": False},
        )
        obs_mode: ObsMode = _OBS_MODE_BY_CONDITION[condition]
        self._obs_wrapper = NavObservationWrapper(raw_env, mode=obs_mode, domain=self.spec_.domain)

        self.logger = (
            NavEpisodeLogger(
                run_dir=run_dir, condition=condition, task_name=task_name, seed=seed,
                modalities=self.modalities,
            )
            if run_dir else None
        )

        self.action_space = self._obs_wrapper.action_space
        probe_obs, _ = self._obs_wrapper.reset(seed=seed)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=probe_obs.shape, dtype=np.float32
        )
        self._step_count = 0
        self._goal_met_any = False

    def reset(self, *, seed=None, options=None):
        obs, info = self._obs_wrapper.reset(seed=seed if seed is not None else self.seed_value)
        self._step_count = 0
        self._goal_met_any = False
        if self.logger is not None:
            self.logger.start_episode()
        return obs, info

    def step(self, action):
        obs, r_env, cost, terminated, truncated, info = self._obs_wrapper.step(action)
        self._step_count += 1

        cost_by_modality = compute_cost_by_modality(info, self.spec_.cost_keys)
        reward = apply_cost_penalty(
            float(r_env), self.condition, self.modalities, cost_by_modality, lam=self.lam_state.value
        )
        info["cost_by_modality"] = cost_by_modality  # read by cdp.lagrangian_callback

        if info.get("goal_met"):
            self._goal_met_any = True
        truncated = bool(truncated or self._step_count >= self.spec_.max_episode_steps)
        terminated = bool(terminated)

        if self.logger is not None:
            self.logger.log_step(
                reward=reward, cost_by_modality=cost_by_modality, action=action,
                terminated=terminated, truncated=truncated,
            )
            if terminated or truncated:
                # per-modality episode-cumulative cost vs. budget (proposal's V_m)
                cum_cost = self.logger.cumulative_cost()
                safe = all(cum_cost[m] <= self.spec_.budget for m in self.modalities)
                self.logger.end_episode(
                    success=self._goal_met_any,
                    safe=bool(self._goal_met_any and safe),
                    termination_reason="max_steps" if truncated else (
                        "terminated" if terminated else "goal_met"
                    ),
                )

        return obs, reward, terminated, truncated, info

    def close(self):
        self._obs_wrapper.close()
