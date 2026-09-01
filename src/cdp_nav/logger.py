"""
Per-step/episode/experiment logger for the Safety-Gymnasium domain — nav
analogue of src/cdp/logger.py, generalized to an arbitrary modality list
(rather than the manipulation domain's hardcoded mechanical/thermal/
electrical) since this domain's two hazard-taxonomies (goal: hazards/vases;
button: gremlins/buttons) share no modality names with each other or with
DamageSim. Same on-disk shape otherwise (one JSON per episode + one summary
row per episode in `summary.jsonl`) so `scripts/analyze.py`-style tooling
generalizes with only field-name changes (see scripts_nav/analyze_nav.py).
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def _to_float(x) -> float:
    if hasattr(x, "item"):
        return float(x.item())
    return float(x)


@dataclass
class NavEpisodeLogger:
    run_dir: str
    condition: str
    task_name: str
    seed: int
    modalities: Sequence[str]

    _steps: List[dict] = field(default_factory=list, init=False)
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
        cost_by_modality: Dict[str, float],
        action,
        terminated: bool = False,
        truncated: bool = False,
    ) -> None:
        assert self._episode_id is not None, "call start_episode() first"
        costs = {m: _to_float(cost_by_modality.get(m, 0.0)) for m in self.modalities}
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        self._steps.append({
            "step": len(self._steps),
            "reward": _to_float(reward),
            "total_cost": sum(costs.values()),
            **{f"cost_{m}": v for m, v in costs.items()},
            "action_magnitude": float(np.linalg.norm(action_arr)),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        })

    def cumulative_cost(self) -> Dict[str, float]:
        """Per-modality cumulative cost so far this (still-open) episode —
        used by NavTaskEnv to judge safe-completion (V_m in proposal.tex)
        before calling end_episode()."""
        return {m: sum(s[f"cost_{m}"] for s in self._steps) for m in self.modalities}

    def end_episode(
        self,
        *,
        success: bool,
        safe: bool,
        termination_reason: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        assert self._episode_id is not None, "call start_episode() first"
        steps = self._steps
        episode_length = len(steps)
        total_reward = sum(s["reward"] for s in steps)
        total_cost = sum(s["total_cost"] for s in steps)
        cost_by_modality = {
            m: sum(s[f"cost_{m}"] for s in steps) for m in self.modalities
        }
        mean_action_magnitude = (
            sum(s["action_magnitude"] for s in steps) / episode_length if episode_length else 0.0
        )

        episode_record = {
            "episode_id": self._episode_id,
            "condition": self.condition,
            "task_name": self.task_name,
            "seed": self.seed,
            "wall_time_s": time.time() - self._t0,
            "steps": steps,
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
            "total_cost": total_cost,
            **{f"cost_{m}": v for m, v in cost_by_modality.items()},
            "mean_action_magnitude": mean_action_magnitude,
        }
        if extra:
            summary_row.update(extra)

        with open(os.path.join(self.run_dir, "summary.jsonl"), "a") as f:
            f.write(json.dumps(summary_row) + "\n")

        self._episode_id = None
        self._steps = []
        return summary_row
