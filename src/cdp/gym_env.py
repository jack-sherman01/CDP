"""
Gymnasium-compatible env wrapping OGDamageableEnvironment for SB3 PPO
training (Days 5-6). Combines: DamageObservationWrapper (observation mode),
TaskRewardComputer + apply_damage_penalty (reward), the task's own
task_completion_check (success), our safety floors (docs/TASKS.md), and
EpisodeLogger (Day 4).
"""
from __future__ import annotations

import importlib
import os
from typing import Optional

import gymnasium as gym
import numpy as np
import omnigibson as og
import torch as th
from gymnasium import spaces
from omnigibson.macros import gm

from damagesim.omnigibson.damageable_env import OGDamageableEnvironment

from cdp.corruption import CorruptedObservationWrapper, CorruptionConfig
from cdp.lagrangian import LambdaState
from cdp.logger import EpisodeLogger
from cdp.obs_wrapper import MODALITIES, DamageObservationWrapper, ObsMode
from cdp.reward import (
    JointOpenRewardComputer, JointOpenRewardConfig, TaskRewardComputer, TaskRewardConfig,
    WipeRewardComputer, WipeRewardConfig, apply_damage_penalty,
)
from cdp.tasks import TaskSpec, get_task_spec, load_task_module

# condition -> observation structure (proposal.tex "Policy Representations"):
# task_only sees no hazard info; scalar_lagrangian sees one aggregate
# health/damage scalar; vector_lagrangian and fixed_weight both see the full
# per-modality vector (they differ only in how the reward penalty on top of
# that observation is computed, see src/cdp/reward.py).
_OBS_MODE_BY_CONDITION = {
    "task_only": "task",
    "scalar_lagrangian": "scalar",
    "vector_lagrangian": "vector",
    "fixed_weight": "vector",
}


def _capture_viewer_rgb() -> np.ndarray:
    """(H, W, 3) uint8 RGB frame from the viewer camera — same call
    `scripts/teleop_b1k.py::capture_rgb` uses, no external_sensors config
    needed."""
    obs, _ = og.sim.viewer_camera.get_obs()
    frame = obs["rgb"]
    frame = frame.cpu().numpy() if isinstance(frame, th.Tensor) else np.asarray(frame)
    if frame.shape[-1] == 4:
        frame = frame[:, :, :3]
    if frame.dtype != np.uint8:
        frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
    return frame


def _build_env_config(task_cfg):
    scene_config = dict(task_cfg.scene_config)
    if "type" not in scene_config:
        scene_config["type"] = "InteractiveTraversableScene"
    return {
        "env": {
            "action_frequency": task_cfg.action_frequency,
            "rendering_frequency": task_cfg.rendering_frequency,
            "physics_frequency": task_cfg.physics_frequency,
            "flatten_action_space": True,
        },
        "scene": scene_config,
        "robots": [dict(task_cfg.robot_config)],
        "objects": [dict(obj) for obj in task_cfg.task_objects.values()],
        "task": {"type": "DummyTask", "activity_name": task_cfg.task_name},
    }


class CDPTaskEnv(gym.Env):
    """
    One OmniGibson simulator instance per process (Isaac Sim can't be
    restarted within a process) — instantiate exactly one of these per
    training run.
    """

    def __init__(
        self,
        task_name: str,
        condition: str,  # "task_only" | "scalar_lagrangian" | "vector_lagrangian" | "fixed_weight"
        seed: int = 0,
        run_dir: Optional[str] = None,
        lam_state: Optional[LambdaState] = None,
        fixed_lambda=None,  # fixed_weight only: float or {modality: float}
        corruption: Optional[CorruptionConfig] = None,
        record_video: bool = False,
        video_dir: Optional[str] = None,
    ):
        super().__init__()
        self.task_name = task_name
        self.condition = condition
        self.seed_value = seed
        self.spec_ = get_task_spec(task_name)

        # Reward-side lambda source: *_lagrangian conditions read a shared,
        # externally-updated LambdaState (see cdp.lagrangian_callback); the
        # env owns a private, never-updated one when no training callback is
        # driving it (fixed_weight, task_only, or standalone/eval use)
        # so `step()` can always read `self.lam_state.value` uniformly.
        if condition == "fixed_weight":
            assert fixed_lambda is not None, "fixed_weight needs fixed_lambda"
            self.lam_state = LambdaState(fixed_lambda)
        elif condition in ("scalar_lagrangian", "vector_lagrangian"):
            self.lam_state = lam_state if lam_state is not None else LambdaState(
                0.0 if condition == "scalar_lagrangian" else {m: 0.0 for m in MODALITIES}
            )
        else:
            self.lam_state = LambdaState(None)  # task_only never reads it

        mod = load_task_module(task_name)
        task_cfg = mod.get_task_config()
        gm.USE_GPU_DYNAMICS = task_cfg.use_gpu_dynamics
        gm.ENABLE_TRANSITION_RULES = task_cfg.enable_transition_rules

        self._base_env = OGDamageableEnvironment(configs=_build_env_config(task_cfg))
        self._task_mod = mod
        self._task_cfg = task_cfg

        obs_mode: ObsMode = _OBS_MODE_BY_CONDITION[condition]
        self._obs_wrapper = DamageObservationWrapper(
            self._base_env, mode=obs_mode, task_object_names=self.spec_.task_object_names
        )
        self._obs_source = self._obs_wrapper
        if corruption is not None and corruption.kind != "none":
            assert obs_mode != "task", "corruption targets the damage/health block; task mode has none"
            self._obs_source = CorruptedObservationWrapper(self._obs_wrapper, corruption, seed=seed)
        self._reward_computer = self._build_reward_computer(mod)

        self.logger = (
            EpisodeLogger(run_dir=run_dir, condition=condition, task_name=task_name, seed=seed)
            if run_dir
            else None
        )

        # Video recording (eval-time only, see memory/project_video_deliverables
        # and docs/DAILY_LOG.md — never enabled during PPO training itself,
        # rendering would make the already multi-hour training budget worse).
        self.record_video = record_video
        self.video_dir = video_dir or (os.path.join(run_dir, "videos") if run_dir else None)
        if self.record_video:
            assert self.video_dir, "record_video=True needs run_dir or video_dir"
            os.makedirs(self.video_dir, exist_ok=True)
        self._video_frames: list = []
        self._video_health_history: dict = {}
        self._episode_idx = 0

        self.action_space = self._to_gym_box(self._base_env.action_space)
        # Probe one observation to fix the space shape.
        self._base_env.reset()
        if hasattr(mod, "reset") and callable(mod.reset):
            mod.reset(self._base_env)
        self._base_env._reset_damage_tracking()
        for _ in range(5):
            og.sim.step()
        probe_obs, _ = self._obs_source.reset()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=probe_obs.shape, dtype=np.float32
        )
        self._step_count = 0

    def _build_reward_computer(self, mod):
        """Dispatches on TaskSpec.reward_mode — "carry" (grasp-and-place,
        the original 3 tasks), "wipe" (wipe_counter), "joint_open"
        (open_drawer/open_single_door). See src/cdp/reward.py."""
        mode = self.spec_.reward_mode
        if mode == "carry":
            return TaskRewardComputer(
                self._base_env,
                TaskRewardConfig(
                    primary_object_name=self.spec_.primary_object_name,
                    goal_object_name=self.spec_.goal_object_name,
                    completion_check=mod.task_completion_check,
                ),
            )
        if mode == "wipe":
            return WipeRewardComputer(
                self._base_env,
                WipeRewardConfig(
                    sponge_object_name=self.spec_.primary_object_name,
                    dirt_state_attr=self.spec_.dirt_state_attr,
                    completion_check=mod.task_completion_check,
                ),
            )
        if mode == "joint_open":
            return JointOpenRewardComputer(
                self._base_env,
                JointOpenRewardConfig(
                    target_object_name=self.spec_.primary_object_name,
                    completion_check=mod.task_completion_check,
                ),
            )
        raise ValueError(f"unknown reward_mode {mode!r}")

    # IK arm controllers here use command_input_limits=None (raw, unbounded
    # deltas) — SB3 requires a finite Box, so unbounded dims get clipped to
    # a conservative per-step delta instead of the sim's true (infinite)
    # range. Already-finite dims (e.g. the gripper's [0, 1]) pass through
    # unchanged.
    _UNBOUNDED_CLIP = 0.2

    @classmethod
    def _to_gym_box(cls, og_space) -> spaces.Box:
        low = np.asarray(og_space.low, dtype=np.float32).reshape(-1)
        high = np.asarray(og_space.high, dtype=np.float32).reshape(-1)
        low = np.where(np.isfinite(low), low, -cls._UNBOUNDED_CLIP)
        high = np.where(np.isfinite(high), high, cls._UNBOUNDED_CLIP)
        return spaces.Box(low=low, high=high, dtype=np.float32)

    # ── Gym API ──────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        self._base_env.reset()
        if hasattr(self._task_mod, "reset") and callable(self._task_mod.reset):
            self._task_mod.reset(self._base_env)
        self._base_env._reset_damage_tracking()
        for _ in range(5):
            og.sim.step()
        obs, info = self._obs_source.reset()
        self._reward_computer.reset()
        self._step_count = 0
        if self.logger is not None:
            self.logger.start_episode()
        if self.record_video:
            if self._task_cfg.viewer_camera_pos is not None:
                og.sim.viewer_camera.set_position_orientation(
                    position=th.tensor(self._task_cfg.viewer_camera_pos, dtype=th.float32),
                    orientation=th.tensor(self._task_cfg.viewer_camera_orn, dtype=th.float32),
                )
            self._video_frames = [_capture_viewer_rgb()]
            self._video_health_history = {
                name: [100.0] for name in [*self.spec_.task_object_names, self._base_env.robots[0].name]
            }
        return obs, info

    def step(self, action):
        action_t = th.as_tensor(np.asarray(action, dtype=np.float32))
        obs, _env_reward, terminated, truncated, info = self._obs_source.step(action_t)
        self._step_count += 1

        r_task, success = self._reward_computer.compute()
        d_vec = self._obs_wrapper.last_d_vec
        d_scalar = float(d_vec.sum())
        reward = apply_damage_penalty(
            r_task, self.condition, d_scalar=d_scalar, d_vec=d_vec, lam=self.lam_state.value
        )
        # Read by cdp.lagrangian_callback.LagrangianUpdateCallback to drive
        # the PID multiplier(s); harmless for conditions that don't use it.
        info["damage_by_modality"] = {
            "mechanical": float(d_vec[0]), "thermal": float(d_vec[1]), "electrical": float(d_vec[2]),
        }

        terminated = bool(terminated or success)
        truncated = bool(truncated or self._step_count >= self.spec_.max_episode_steps)

        native_healths = None
        if self.logger is not None or self.record_video:
            native_healths = self._base_env.get_env_health() or {}
            robot_name = self._base_env.robots[0].name

        if self.record_video:
            self._video_frames.append(_capture_viewer_rgb())
            for name, history in self._video_health_history.items():
                obj_link_healths = [v for k, v in native_healths.items() if k.startswith(f"{name}@")]
                history.append(min(obj_link_healths) if obj_link_healths else 100.0)

        if self.logger is not None:
            min_obj_health = float(min(native_healths.values())) if native_healths else 100.0
            robot_health = float(
                min(
                    (v for k, v in native_healths.items() if k.startswith(f"{robot_name}@")),
                    default=100.0,
                )
            )
            self.logger.log_step(
                reward=reward,
                damage_by_modality=info["damage_by_modality"],
                min_object_health=min_obj_health,
                robot_health=robot_health,
                action=action,
                terminated=terminated,
                truncated=truncated,
            )
            if terminated or truncated:
                safe = bool(
                    success
                    and min_obj_health >= self.spec_.object_health_floor
                    and robot_health >= self.spec_.robot_health_floor
                )
                self.logger.end_episode(
                    success=success,
                    safe=safe,
                    termination_reason="success" if success else (
                        "max_steps" if truncated else "terminated"
                    ),
                )

        if self.record_video and (terminated or truncated):
            self._save_episode_video(success)

        return obs, reward, terminated, truncated, info

    def _save_episode_video(self, success: bool) -> None:
        from damagesim.utils.visualization import save_rgb_health_video_with_overlay

        tag = "success" if success else "fail"
        out_path = os.path.join(
            self.video_dir, f"{self.condition}_{self.task_name}_ep{self._episode_idx:04d}_{tag}"
        )
        health = {name: np.array(h, dtype=np.float32) for name, h in self._video_health_history.items()}
        try:
            save_rgb_health_video_with_overlay(
                out_path,
                np.stack(self._video_frames),
                target_objects=list(self._video_health_history.keys()),
                health=health,
                fps=max(1, int(self._task_cfg.action_frequency)),
            )
        except Exception as e:
            print(f"[CDPTaskEnv] WARNING: video save failed for {out_path}: {e}")
        self._episode_idx += 1

    def close(self):
        og.shutdown()
