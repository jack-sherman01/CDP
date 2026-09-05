"""
Reward construction (Days 5-6, 8-9, 10-11 of the proposal).

The upstream task files only provide a boolean ``task_completion_check(env)``
(see docs/TASKS.md's Day-3 correction) — no reward shaping, no per-step
signal. Everything here is our own design, built on top of that boolean.

Task reward (shared across all three conditions — only the damage penalty
added on top differs):
    not grasping : r_task_t = -shaping_scale * dist(eef, primary_object)
                               + lift_scale * max(0, z_t - z_0)
    grasping     : r_task_t = -shaping_scale * dist(primary_object, goal_object)
                               (falls back to the not-grasping term if no
                                goal_object_name is configured for this task)
    always       : + grasp_bonus (while is_grasping(primary_object))
                    - time_penalty
                    + completion_bonus (once, on the step
                      task_completion_check first becomes True)

Revision 1 (Day 5-6, after the first live pick_egg run): with only the
distance-to-object term + a one-off sparse completion bonus, 20k timesteps
produced a clearly-improving reward/damage trend (reward -212 -> -131,
damage 139 -> ~100 over 68 episodes) but zero successes — a random-init
policy has no gradient at all between "near the object" and "lifted 8cm,"
a large gap to stumble into by exploration alone. Added continuous
height-gained shaping and a grasp bonus.

Revision 2 (Days 10-11, after the first live add_firewood run): that fix
was enough for pick_egg (whose goal is just "lift the object," so
dist(eef, object) is the right shaping the whole episode) but not for
"pick up X and carry it to Y" tasks. add_firewood's task-only run showed
reward improving (-220 -> -58) while damage kept *climbing* (146 -> 754)
and successes stayed at 0/136 — the policy learned to reach and grasp the
log (the only gradient available) but had zero incentive to then move it
toward the fireplace, and the always-on grasp_bonus actively fought
against ever letting go. Fixed by switching the distance term to
dist(primary_object, goal_object) once grasping (falls back to
dist(eef, primary_object) if the task has no goal_object_name, e.g.
pick_egg) — this is the first shaping term that actually rewards carrying
the object toward the goal, not just reaching it. `add_firewood`,
`fill_bowl`, `heat_saucepot` now set `goal_object_name` in
`src/cdp/tasks.py`; `pick_egg`/`pour_water` leave it unset since their
goal is "manipulate the object in place," not "carry it somewhere."

Damage penalty added on top (Revision 3, after the proposal.tex pivot to a
vector-valued CMDP with PID-Lagrangian multipliers — see src/cdp/lagrangian.py
for the controller and docs/DECISIONS.md for the pivot's rationale):
    task_only         : r_t = r_task_t
    scalar_lagrangian : r_t = r_task_t - lambda_t * d_scalar_t
                        (single multiplier, PID-updated on aggregate cost)
    vector_lagrangian : r_t = r_task_t - sum_m lambda_m,t * d_m,t
                        (independent multiplier per modality m, PID-updated
                        on that modality's own cost — this is the mechanism
                        RQ1/RQ3 are about, not just a structured observation)
    fixed_weight      : r_t = r_task_t - sum_m lambda_m * d_m,t
                        (RQ3 ablation: same per-modality structure as
                        vector_lagrangian but lambda_m is a constant set at
                        construction time, never PID-updated — isolates
                        "the policy sees separate channels" from "the
                        multipliers adapt to enforce a budget")
`lambda_t`/`lambda_m,t` for the two `*_lagrangian` conditions are supplied
externally each step via a `cdp.lagrangian.LambdaState` the training
callback updates once per PPO rollout (see
src/cdp/lagrangian_callback.py); this module only applies whatever value it
is handed, it does not compute it.

Superseded (kept for the historical record, docs/DAILY_LOG.md /
private/CONTRIBUTIONS_LOG.md): the original "scalar"/"vector" conditions
(pre-pivot) both used a single fixed lambda=0.05 for the whole run, with
"vector" scalarizing via an L1 sum of per-modality damage rather than
independent multipliers. Those 9 checkpoints are now the `fixed_weight`
ablation's informal precursor, not a `vector_lagrangian` result.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch as th

from cdp.obs_wrapper import MODALITIES

DEFAULT_SHAPING_SCALE = 1.0
DEFAULT_LIFT_SCALE = 20.0
DEFAULT_GRASP_BONUS = 0.5
DEFAULT_TIME_PENALTY = 0.01
DEFAULT_COMPLETION_BONUS = 10.0
DEFAULT_LAMBDA = 0.05


def _to_np(x) -> np.ndarray:
    if isinstance(x, th.Tensor):
        return x.detach().cpu().numpy().astype(np.float32).reshape(-1)
    return np.asarray(x, dtype=np.float32).reshape(-1)


@dataclass
class TaskRewardConfig:
    primary_object_name: str
    completion_check: Callable  # (env) -> bool
    goal_object_name: Optional[str] = None  # e.g. add_firewood's fireplace
    shaping_scale: float = DEFAULT_SHAPING_SCALE
    lift_scale: float = DEFAULT_LIFT_SCALE
    grasp_bonus: float = DEFAULT_GRASP_BONUS
    time_penalty: float = DEFAULT_TIME_PENALTY
    completion_bonus: float = DEFAULT_COMPLETION_BONUS


class TaskRewardComputer:
    """Stateful (tracks whether the episode already got its completion bonus,
    and the primary object's z position at reset, as the lift-shaping baseline)."""

    def __init__(self, env, config: TaskRewardConfig):
        self.env = env
        self.config = config
        self._already_completed = False
        self._initial_z: Optional[float] = None

    def reset(self) -> None:
        self._already_completed = False
        obj = self.env.scene.object_registry("name", self.config.primary_object_name)
        self._initial_z = float(obj.get_position_orientation()[0][2]) if obj is not None else None

    def compute(self) -> tuple[float, bool]:
        """Returns (r_task_t, success_this_step)."""
        robot = self.env.robots[0]
        eef_pos = _to_np(robot.get_eef_position())
        obj = self.env.scene.object_registry("name", self.config.primary_object_name)
        dist = 0.0
        lift_gain = 0.0
        is_grasping = False
        if obj is not None:
            obj_pos, _ = obj.get_position_orientation()
            obj_pos_np = _to_np(obj_pos)
            if self._initial_z is not None:
                lift_gain = max(0.0, float(obj_pos_np[2]) - self._initial_z)
            if hasattr(robot, "is_grasping"):
                try:
                    from omnigibson.controllers.controller_base import IsGraspingState
                    is_grasping = robot.is_grasping(candidate_obj=obj).value == IsGraspingState.TRUE
                except Exception:
                    is_grasping = False

            goal = (
                self.env.scene.object_registry("name", self.config.goal_object_name)
                if (is_grasping and self.config.goal_object_name) else None
            )
            if goal is not None:
                goal_pos, _ = goal.get_position_orientation()
                dist = float(np.linalg.norm(obj_pos_np - _to_np(goal_pos)))
            else:
                dist = float(np.linalg.norm(eef_pos - obj_pos_np))

        success_now = bool(self.config.completion_check(self.env))
        r = (
            -self.config.shaping_scale * dist
            + self.config.lift_scale * lift_gain
            + (self.config.grasp_bonus if is_grasping else 0.0)
            - self.config.time_penalty
        )
        if success_now and not self._already_completed:
            r += self.config.completion_bonus
            self._already_completed = True
        return r, (success_now or self._already_completed)


@dataclass
class WipeRewardConfig:
    """`wipe_counter`: no fixed carry-to-goal target — completion is "reduce
    a dirt-particle stain on a surface to zero, then lift off." Shaping:
    approach the sponge, then reward tracks fraction of dirt cleaned (0->1)
    continuously, mirroring pick_egg's height-gained shaping but for a
    cleaning progress signal instead of a lift height."""
    sponge_object_name: str
    dirt_state_attr: str  # env attribute name holding (dirt_system, group), set by the task's own reset() hook
    completion_check: Callable
    shaping_scale: float = DEFAULT_SHAPING_SCALE
    clean_scale: float = DEFAULT_LIFT_SCALE
    grasp_bonus: float = DEFAULT_GRASP_BONUS
    time_penalty: float = DEFAULT_TIME_PENALTY
    completion_bonus: float = DEFAULT_COMPLETION_BONUS


class WipeRewardComputer:
    def __init__(self, env, config: WipeRewardConfig):
        self.env = env
        self.config = config
        self._already_completed = False
        self._initial_dirt_count: int = 1

    def reset(self) -> None:
        self._already_completed = False
        dirt_system, group = getattr(self.env, self.config.dirt_state_attr)
        self._initial_dirt_count = max(1, int(dirt_system.num_group_particles(group=group)))

    def compute(self) -> tuple[float, bool]:
        robot = self.env.robots[0]
        eef_pos = _to_np(robot.get_eef_position())
        sponge = self.env.scene.object_registry("name", self.config.sponge_object_name)
        dist = 0.0
        is_grasping = False
        if sponge is not None:
            sponge_pos, _ = sponge.get_position_orientation()
            dist = float(np.linalg.norm(eef_pos - _to_np(sponge_pos)))
            if hasattr(robot, "is_grasping"):
                try:
                    from omnigibson.controllers.controller_base import IsGraspingState
                    is_grasping = robot.is_grasping(candidate_obj=sponge).value == IsGraspingState.TRUE
                except Exception:
                    is_grasping = False

        dirt_system, group = getattr(self.env, self.config.dirt_state_attr)
        remaining = int(dirt_system.num_group_particles(group=group))
        cleaned_fraction = 1.0 - (remaining / self._initial_dirt_count)

        success_now = bool(self.config.completion_check(self.env))
        r = (
            -self.config.shaping_scale * dist
            + self.config.clean_scale * cleaned_fraction
            + (self.config.grasp_bonus if is_grasping else 0.0)
            - self.config.time_penalty
        )
        if success_now and not self._already_completed:
            r += self.config.completion_bonus
            self._already_completed = True
        return r, (success_now or self._already_completed)


@dataclass
class JointOpenRewardConfig:
    """`open_drawer`/`open_single_door`: no object to carry — completion is
    "actuate a joint (drawer slide / door hinge) past 95% of its range."
    Shaping: approach the target object, then reward tracks the MAX joint-
    opening fraction gained across all of the object's joints (matches
    `task_completion_check`'s "any joint past threshold" semantics)."""
    target_object_name: str
    completion_check: Callable
    shaping_scale: float = DEFAULT_SHAPING_SCALE
    open_scale: float = DEFAULT_LIFT_SCALE
    time_penalty: float = DEFAULT_TIME_PENALTY
    completion_bonus: float = DEFAULT_COMPLETION_BONUS


class JointOpenRewardComputer:
    def __init__(self, env, config: JointOpenRewardConfig):
        self.env = env
        self.config = config
        self._already_completed = False
        self._initial_fraction = 0.0

    @staticmethod
    def _max_joint_fraction(obj) -> float:
        fracs = []
        for j in obj.joints.values():
            lo, hi = j.lower_limit, j.upper_limit
            if hi - lo <= 1e-6:
                continue
            fracs.append(float((j.get_state()[0] - lo) / (hi - lo)))
        return max(fracs) if fracs else 0.0

    def reset(self) -> None:
        self._already_completed = False
        obj = self.env.scene.object_registry("name", self.config.target_object_name)
        self._initial_fraction = self._max_joint_fraction(obj) if obj is not None else 0.0

    def compute(self) -> tuple[float, bool]:
        robot = self.env.robots[0]
        eef_pos = _to_np(robot.get_eef_position())
        obj = self.env.scene.object_registry("name", self.config.target_object_name)
        dist = 0.0
        open_gain = 0.0
        if obj is not None:
            obj_pos, _ = obj.get_position_orientation()
            dist = float(np.linalg.norm(eef_pos - _to_np(obj_pos)))
            frac = self._max_joint_fraction(obj)
            open_gain = max(0.0, frac - self._initial_fraction)

        success_now = bool(self.config.completion_check(self.env))
        r = (
            -self.config.shaping_scale * dist
            + self.config.open_scale * open_gain
            - self.config.time_penalty
        )
        if success_now and not self._already_completed:
            r += self.config.completion_bonus
            self._already_completed = True
        return r, (success_now or self._already_completed)


Condition = str  # "task_only" | "scalar_lagrangian" | "vector_lagrangian" | "fixed_weight"


def apply_damage_penalty(
    r_task: float,
    condition: Condition,
    d_scalar: Optional[float] = None,
    d_vec: Optional[np.ndarray] = None,
    lam=None,
) -> float:
    """`lam` shape depends on `condition`:
      - task_only: unused.
      - scalar_lagrangian: a float (the current PID-updated multiplier).
      - vector_lagrangian / fixed_weight: a per-modality vector — either a
        {modality: float} dict (as produced by VectorLagrangian.lam_vec /
        LambdaState.value) or an np.ndarray already ordered per
        cdp.obs_wrapper.MODALITIES.
    """
    if condition == "task_only":
        return r_task
    if condition == "scalar_lagrangian":
        assert d_scalar is not None and lam is not None
        return r_task - float(lam) * d_scalar
    if condition in ("vector_lagrangian", "fixed_weight"):
        assert d_vec is not None and lam is not None
        lam_vec = (
            np.asarray([lam[m] for m in MODALITIES], dtype=np.float32)
            if isinstance(lam, dict) else np.asarray(lam, dtype=np.float32)
        )
        return r_task - float(np.dot(lam_vec, d_vec))
    raise ValueError(f"unknown condition {condition!r}")
