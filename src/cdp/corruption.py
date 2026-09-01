"""
Damage-signal corruption tests (Day 20). Wraps a `DamageObservationWrapper`
and corrupts only the damage/health suffix of its observation (the last 2
dims in `scalar` mode, last 6 in `vector` mode) — the robot/task-state
prefix is left untouched, since corrupting it would confound "is the
policy robust to a noisy damage signal" with "is the policy robust to noisy
proprioception," a different question than the proposal asks.

Corruption types, matching proposal Day 20 exactly:
    gaussian        - additive N(0, sigma) noise
    modality_mask   - zero out a randomly chosen subset of modality channels
                       (vector mode only; no-op in scalar/task mode)
    delay           - replay the damage/health block from N steps ago
    held_constant   - freeze the damage/health block at its value from a
                       fixed step (e.g. episode start) for the rest of the episode
    scaling_error   - multiply the damage/health block by a fixed factor
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from cdp.obs_wrapper import DamageObservationWrapper, MODALITIES

CorruptionType = Literal["none", "gaussian", "modality_mask", "delay", "held_constant", "scaling_error"]


@dataclass
class CorruptionConfig:
    kind: CorruptionType = "none"
    gaussian_sigma: float = 5.0          # health-scale units (0-100)
    modality_mask_p: float = 0.2         # P(each modality channel zeroed), matches proposal's p=0.2
    delay_steps: int = 5
    scaling_factor: float = 2.0


class CorruptedObservationWrapper:
    """Drop-in replacement for `DamageObservationWrapper` with corruption applied."""

    def __init__(self, base: DamageObservationWrapper, config: CorruptionConfig, seed: int = 0):
        assert base.mode in ("scalar", "vector"), "corruption only applies to scalar/vector modes"
        self.base = base
        self.config = config
        self._rng = np.random.default_rng(seed)
        self._history: deque = deque(maxlen=max(1, config.delay_steps) + 1)
        self._held_value: Optional[np.ndarray] = None

    @property
    def _damage_block_size(self) -> int:
        return 2 if self.base.mode == "scalar" else 2 * len(MODALITIES)

    def reset(self, *args, **kwargs):
        obs, info = self.base.reset(*args, **kwargs)
        self._history.clear()
        self._held_value = None
        return self._corrupt(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.base.step(action)
        return self._corrupt(obs), reward, terminated, truncated, info

    def _corrupt(self, obs: np.ndarray) -> np.ndarray:
        n = self._damage_block_size
        prefix, block = obs[:-n], obs[-n:].copy()
        self._history.append(block.copy())

        kind = self.config.kind
        if kind == "none":
            pass
        elif kind == "gaussian":
            block = block + self._rng.normal(0, self.config.gaussian_sigma, size=block.shape).astype(np.float32)
        elif kind == "modality_mask":
            if self.base.mode == "vector":
                # block = [d_mech, d_therm, d_elec, h_mech, h_therm, h_elec]
                mask = self._rng.random(len(MODALITIES)) < self.config.modality_mask_p
                block[:3][mask] = 0.0
                block[3:][mask] = 0.0
        elif kind == "delay":
            # deque(maxlen=delay_steps+1): index 0 is the oldest kept entry,
            # i.e. exactly `delay_steps` steps ago once warmed up; before
            # that (episode start) it's just the oldest available so far.
            block = self._history[0]
        elif kind == "held_constant":
            if self._held_value is None:
                self._held_value = block.copy()
            block = self._held_value
        elif kind == "scaling_error":
            block = block * self.config.scaling_factor
        else:
            raise ValueError(f"unknown corruption kind {kind!r}")

        return np.concatenate([prefix, block]).astype(np.float32)
