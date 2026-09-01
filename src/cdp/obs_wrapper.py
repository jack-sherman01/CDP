"""
Low-dimensional observation wrapper for DamageSim/OmniGibson task envs.

Implements the three observation modes from ``private/proposal.tex``
(Day 3): ``task``, ``scalar``, ``vector``. See ``docs/OBSERVATIONS.md`` for
the design rationale — in particular, per-modality health (h^mech, h^therm,
h^fluid separately) is NOT tracked natively by DamageSim: the core
``DamageableMixin`` only keeps one aggregate scalar health per link,
decremented by every evaluator's damage combined. This wrapper reconstructs
per-modality health itself, at the same per-link granularity the sim uses,
by integrating each modality's own damage stream independently with the
same ``health -= damage, clip(0, 100)`` rule the sim applies to the
combined signal. Per-step per-modality *damage* (``d^m_t``), by contrast,
IS already exposed natively via ``info["damage_info"][obj][link][modality]
["damage"]`` — no reconstruction needed there.

Wraps an ``OGDamageableEnvironment``-like env: anything exposing
``.robots[0]``, ``.scene``, ``._get_all_objects()``, ``get_env_health()``,
and returning ``info["damage_info"]`` from ``step``.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Sequence, Tuple

import numpy as np
import torch as th

# "electrical" is DamageSim's name for what the proposal calls "fluid"
# damage (water-particle contact) — see docs/DECISIONS.md.
MODALITIES: Sequence[str] = ("mechanical", "thermal", "electrical")
ObsMode = Literal["task", "scalar", "vector"]


def _to_np(x) -> np.ndarray:
    if isinstance(x, th.Tensor):
        return x.detach().cpu().numpy().astype(np.float32).reshape(-1)
    return np.asarray(x, dtype=np.float32).reshape(-1)


class DamageObservationWrapper:
    """
    Builds a flat float32 vector observation from a DamageSim OG env.

    Parameters
    ----------
    env:
        The underlying ``OGDamageableEnvironment`` (or
        ``OGDamageableDataCollectionWrapper``-wrapped one).
    mode:
        ``"task"``    -> [robot_state, task-relevant object poses+distances]
        ``"scalar"``  -> task + [d_scalar_t, h_scalar_t]
        ``"vector"``  -> task + [d_mech_t, d_therm_t, d_elec_t,
                                   h_mech_t, h_therm_t, h_elec_t]
    task_object_names:
        Names of the task-relevant objects (subset of ``env`` objects) whose
        pose + relative-distance-to-eef features are included in every mode.
        Padded/truncated to ``cdp.tasks.MAX_TASK_OBJECTS`` slots so the
        observation shape is IDENTICAL across tasks — required for zero-shot
        cross-task evaluation (see ``cdp.tasks``'s module docstring note).

    Attributes exposed after each ``reset``/``step``
    --------------------------------------------------
    last_d_vec, last_h_vec : np.ndarray, shape (3,)
        Per-modality instantaneous damage / current health, in
        ``MODALITIES`` order — used by ``src/cdp/reward.py`` so the reward
        doesn't have to re-derive them from ``info["damage_info"]``.
    """

    def __init__(self, env, mode: ObsMode, task_object_names: List[str]):
        assert mode in ("task", "scalar", "vector"), mode
        from cdp.tasks import MAX_TASK_OBJECTS

        if len(task_object_names) > MAX_TASK_OBJECTS:
            raise ValueError(
                f"{len(task_object_names)} task objects > MAX_TASK_OBJECTS={MAX_TASK_OBJECTS}; "
                "bump MAX_TASK_OBJECTS in cdp/tasks.py (and retrain any existing checkpoints)."
            )
        self.env = env
        self.mode = mode
        self.task_object_names = list(task_object_names)
        self.max_task_objects = MAX_TASK_OBJECTS
        # {(object_name, link_name): {modality: health_float}} — our own
        # per-link, per-modality cumulative health (see module docstring).
        self._link_modality_health: Dict[Tuple[str, str], Dict[str, float]] = {}
        self.last_d_vec = np.zeros(len(MODALITIES), dtype=np.float32)
        self.last_h_vec = np.full(len(MODALITIES), 100.0, dtype=np.float32)

    # ── Gym-ish passthrough ──────────────────────────────────────────

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        self._link_modality_health = {}
        self.last_d_vec = np.zeros(len(MODALITIES), dtype=np.float32)
        self.last_h_vec = np.full(len(MODALITIES), 100.0, dtype=np.float32)
        return self._build_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._update_modality_health(info.get("damage_info", {}))
        return self._build_obs(), reward, terminated, truncated, info

    # ── Per-modality damage/health bookkeeping ───────────────────────

    def _update_modality_health(self, damage_info: Dict[str, Any]) -> None:
        """
        Recompute ``last_d_vec`` (this step's per-modality damage, summed
        over every tracked link) and update per-link cumulative health.
        """
        d_vec = np.zeros(len(MODALITIES), dtype=np.float32)
        # object_name -> modality -> min health over that object's links
        # (mirrors DamageableMixin.health's min-over-links convention).
        per_object_modality_min: Dict[str, Dict[str, float]] = {}

        for obj_name, per_link in damage_info.items():
            for link_name, per_evaluator in per_link.items():
                key = (obj_name, link_name)
                link_health = self._link_modality_health.setdefault(
                    key, {m: 100.0 for m in MODALITIES}
                )
                for i, modality in enumerate(MODALITIES):
                    leaf = per_evaluator.get(modality)
                    damage = float(leaf["damage"]) if leaf else 0.0
                    d_vec[i] += damage
                    link_health[modality] = max(0.0, link_health[modality] - damage)

                obj_mins = per_object_modality_min.setdefault(
                    obj_name, {m: 100.0 for m in MODALITIES}
                )
                for modality in MODALITIES:
                    obj_mins[modality] = min(obj_mins[modality], link_health[modality])

        self.last_d_vec = d_vec
        if per_object_modality_min:
            n = len(per_object_modality_min)
            self.last_h_vec = np.array(
                [
                    sum(obj[m] for obj in per_object_modality_min.values()) / n
                    for m in MODALITIES
                ],
                dtype=np.float32,
            )
        # else: no tracked objects reported this step — keep last_h_vec as-is.

    # ── Feature blocks ────────────────────────────────────────────────

    def _robot_state(self) -> np.ndarray:
        robot = self.env.robots[0]
        eef_pos = _to_np(robot.get_eef_position())
        eef_orn = _to_np(robot.get_eef_orientation())
        joint_pos = _to_np(robot.get_joint_positions())
        joint_vel = _to_np(robot.get_joint_velocities())
        return np.concatenate([eef_pos, eef_orn, joint_pos, joint_vel])

    def _object_poses_and_distances(self) -> np.ndarray:
        robot = self.env.robots[0]
        eef_pos = _to_np(robot.get_eef_position())
        feats = []
        for name in self.task_object_names[: self.max_task_objects]:
            obj = self.env.scene.object_registry("name", name)
            if obj is None:
                feats.append(np.zeros(10, dtype=np.float32))  # pos3+quat4+rel3
                continue
            pos, orn = obj.get_position_orientation()
            pos_np, orn_np = _to_np(pos), _to_np(orn)
            rel = eef_pos - pos_np
            feats.append(np.concatenate([pos_np, orn_np, rel]))
        # Zero-pad up to max_task_objects slots (see __init__ docstring —
        # keeps the observation shape identical across tasks).
        for _ in range(self.max_task_objects - len(feats)):
            feats.append(np.zeros(10, dtype=np.float32))
        return np.concatenate(feats)

    def _task_obs(self) -> np.ndarray:
        return np.concatenate([self._robot_state(), self._object_poses_and_distances()])

    def _build_obs(self) -> np.ndarray:
        task_obs = self._task_obs().astype(np.float32)
        if self.mode == "task":
            return task_obs

        if self.mode == "scalar":
            # Scalar baseline uses the sim's own undifferentiated signal:
            # d_scalar_t = sum of per-modality damage this step (Eq. in
            # proposal Sec. "Proposed Contribution"); h_scalar_t = mean of
            # DamageSim's native (already-aggregate) per-link health.
            d_scalar = float(self.last_d_vec.sum())
            native_healths = self.env.get_env_health() or {}
            h_scalar = float(np.mean(list(native_healths.values()))) if native_healths else 100.0
            return np.concatenate([task_obs, [d_scalar, h_scalar]]).astype(np.float32)

        # vector mode
        return np.concatenate([task_obs, self.last_d_vec, self.last_h_vec]).astype(np.float32)
