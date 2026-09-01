"""
Unified per-step / per-episode / per-experiment logger (Day 4).

One JSON file per episode (full per-step traces) + one JSONL summary file
per experiment (one row per episode, used by eval/plotting).
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

MODALITIES = ("mechanical", "thermal", "electrical")


def _to_float(x) -> float:
    if hasattr(x, "item"):
        return float(x.item())
    return float(x)


@dataclass
class StepRecord:
    step: int
    reward: float
    total_damage: float
    damage_mechanical: float
    damage_thermal: float
    damage_electrical: float
    min_object_health: float
    robot_health: float
    action_magnitude: float
    terminated: bool
    truncated: bool


@dataclass
class EpisodeLogger:
    """
    Accumulates one episode's step records, then writes:
      - ``{run_dir}/episodes/{episode_id}.json``   (full per-step trace)
      - appends one row to ``{run_dir}/summary.jsonl``

    ``episode_id`` should encode ``{condition}_{task}_{seed}_{timestamp}``
    per ``docs/PLAN.md``'s reproducibility-record convention.
    """

    run_dir: str
    condition: str  # "task_only" | "scalar_lagrangian" | "vector_lagrangian" | "fixed_weight"
    task_name: str
    seed: int

    _steps: List[StepRecord] = field(default_factory=list, init=False)
    _episode_id: Optional[str] = field(default=None, init=False)
    _t0: float = field(default=0.0, init=False)

    def __post_init__(self):
        os.makedirs(os.path.join(self.run_dir, "episodes"), exist_ok=True)

    def start_episode(self) -> None:
        self._steps = []
        self._t0 = time.time()
        ts = time.strftime("%Y%m%dT%H%M%S")
        self._episode_id = f"{self.condition}_{self.task_name}_{self.seed}_{ts}_{len(self._steps)}"

    def log_step(
        self,
        *,
        reward: float,
        damage_by_modality: Dict[str, float],
        min_object_health: float,
        robot_health: float,
        action,
        terminated: bool = False,
        truncated: bool = False,
    ) -> None:
        assert self._episode_id is not None, "call start_episode() first"
        d_mech = _to_float(damage_by_modality.get("mechanical", 0.0))
        d_therm = _to_float(damage_by_modality.get("thermal", 0.0))
        d_elec = _to_float(damage_by_modality.get("electrical", 0.0))
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        self._steps.append(
            StepRecord(
                step=len(self._steps),
                reward=_to_float(reward),
                total_damage=d_mech + d_therm + d_elec,
                damage_mechanical=d_mech,
                damage_thermal=d_therm,
                damage_electrical=d_elec,
                min_object_health=_to_float(min_object_health),
                robot_health=_to_float(robot_health),
                action_magnitude=float(np.linalg.norm(action_arr)),
                terminated=bool(terminated),
                truncated=bool(truncated),
            )
        )

    def end_episode(
        self,
        *,
        success: bool,
        safe: bool,
        termination_reason: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Write the episode file + append a summary row. Returns the summary row."""
        assert self._episode_id is not None, "call start_episode() first"
        steps = self._steps
        episode_length = len(steps)
        total_reward = sum(s.reward for s in steps)
        total_damage = sum(s.total_damage for s in steps)
        damage_by_modality = {
            "mechanical": sum(s.damage_mechanical for s in steps),
            "thermal": sum(s.damage_thermal for s in steps),
            "electrical": sum(s.damage_electrical for s in steps),
        }
        min_health_episode = min((s.min_object_health for s in steps), default=100.0)
        final_robot_health = steps[-1].robot_health if steps else 100.0
        mean_action_magnitude = (
            sum(s.action_magnitude for s in steps) / episode_length if episode_length else 0.0
        )

        episode_record = {
            "episode_id": self._episode_id,
            "condition": self.condition,
            "task_name": self.task_name,
            "seed": self.seed,
            "wall_time_s": time.time() - self._t0,
            "steps": [s.__dict__ for s in steps],
        }
        with open(os.path.join(self.run_dir, "episodes", f"{self._episode_id}.json"), "w") as f:
            json.dump(episode_record, f)

        summary_row = {
            "episode_id": self._episode_id,
            "condition": self.condition,
            "task_name": self.task_name,
            "seed": self.seed,
            "success": bool(success),
            "safe": bool(safe),
            "successful_and_safe": bool(success and safe),
            "termination_reason": termination_reason,
            "episode_length": episode_length,
            "total_reward": total_reward,
            "total_damage": total_damage,
            "damage_mechanical": damage_by_modality["mechanical"],
            "damage_thermal": damage_by_modality["thermal"],
            "damage_electrical": damage_by_modality["electrical"],
            "min_object_health": min_health_episode,
            "final_robot_health": final_robot_health,
            "mean_action_magnitude": mean_action_magnitude,
        }
        if extra:
            summary_row.update(extra)

        with open(os.path.join(self.run_dir, "summary.jsonl"), "a") as f:
            f.write(json.dumps(summary_row) + "\n")

        self._episode_id = None
        self._steps = []
        return summary_row
