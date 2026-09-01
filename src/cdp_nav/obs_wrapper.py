"""
Observation modes for the Safety-Gymnasium domain — the nav-domain analogue
of src/cdp/obs_wrapper.py, same task/scalar/vector split from proposal.tex's
"Policy Representations", adapted to what this domain actually exposes.

DamageSim has an explicit damage/health signal separate from the object's
geometric presence (an object's position is always observable; its
accumulated damage is the thing task/scalar/vector differ on). Safety-
Gymnasium has no such physically-embedded signal — "cost" is pure
bookkeeping the simulator does not expose through any sensor. The closest
physical analogue to "how much hazard information does the policy get" is
the per-hazard-type LIDAR channel (each ``{modality}_lidar`` key in the
un-flattened Dict observation, e.g. ``hazards_lidar``, ``vases_lidar``):
    task   : no hazard-type lidar at all (agent + goal_lidar only) --
             literally "no hazard information", matching the manipulation
             domain's task-only condition.
    scalar : + ONE merged lidar channel, elementwise max across whichever
             hazard-type lidars this task variant has -- the agent can
             sense "something's close in this direction" but not which
             type, the direct analogue of DamageSim's aggregated scalar
             health.
    vector : + ALL of the domain's hazard-type lidar channels, kept
             separate.

`vector` mode's per-modality slots are also what makes cross-task zero-shot
evaluation well-formed here, same role MAX_TASK_OBJECTS plays in
src/cdp/tasks.py: `SafetyCarButtonWrongButtonOnly1-v0`'s raw Dict
observation has no `gremlins_lidar` key at all (that hazard type isn't in
the scene) while `SafetyCarButtonCombo1-v0`'s does — zero-filling the
missing slot to a fixed-size block keeps the flattened observation shape
IDENTICAL across every task variant in a domain, so a policy trained on one
never sees a shape it wasn't trained on.
"""
from __future__ import annotations

from typing import Any, Dict, Literal, Tuple

import numpy as np

from cdp_nav.tasks import DOMAIN_MODALITIES

ObsMode = Literal["task", "scalar", "vector"]
LIDAR_BINS = 16  # safety_gymnasium's DEFAULT_LIDAR_CONF.num_bins


def _flatten_val(v) -> np.ndarray:
    return np.asarray(v, dtype=np.float32).reshape(-1)


class NavObservationWrapper:
    """Wraps a safety_gymnasium env made with `config={"observation_flatten":
    False}` (so `obs` is a Dict) and flattens it per `mode`, per the module
    docstring."""

    def __init__(self, env, mode: ObsMode, domain: str):
        self.env = env
        self.mode = mode
        self.domain = domain
        self.modalities: Tuple[str, ...] = DOMAIN_MODALITIES[domain]
        self._hazard_lidar_keys = {f"{m}_lidar" for m in self.modalities}

    def _flatten(self, dict_obs: Dict[str, Any]) -> np.ndarray:
        base_parts = [
            _flatten_val(v) for k, v in sorted(dict_obs.items())
            if k not in self._hazard_lidar_keys
        ]
        base = np.concatenate(base_parts) if base_parts else np.zeros(0, dtype=np.float32)
        if self.mode == "task":
            return base

        per_modality = []
        for m in self.modalities:
            key = f"{m}_lidar"
            per_modality.append(
                _flatten_val(dict_obs[key]) if key in dict_obs else np.zeros(LIDAR_BINS, dtype=np.float32)
            )

        if self.mode == "scalar":
            combined = np.maximum.reduce(per_modality) if per_modality else np.zeros(LIDAR_BINS, dtype=np.float32)
            return np.concatenate([base, combined])
        if self.mode == "vector":
            return np.concatenate([base, *per_modality])
        raise ValueError(f"unknown obs mode {self.mode!r}")

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._flatten(obs), info

    def step(self, action):
        obs, reward, cost, terminated, truncated, info = self.env.step(action)
        return self._flatten(obs), reward, cost, terminated, truncated, info

    @property
    def action_space(self):
        return self.env.action_space

    def close(self):
        self.env.close()
