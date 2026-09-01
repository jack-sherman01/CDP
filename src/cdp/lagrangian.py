"""
PID-Lagrangian dual-variable controller (Stooke, Achiam & Abbeel, ICML 2020),
adopted per the revised proposal.tex Sec. "Policy Representations" as the
adaptive constraint-enforcement mechanism behind the `scalar_lagrangian` and
`vector_lagrangian` conditions (replacing the old fixed-lambda-only reward
penalty in src/cdp/reward.py's earlier revision — that fixed-lambda scheme
survives only as the `fixed_weight` ablation, RQ3's isolation of "structured
observation" from "adaptive constraint enforcement").

One PID controller per constrained cost channel: `scalar_lagrangian` uses a
single instance on the aggregate per-step damage; `vector_lagrangian` uses
one independent instance per hazard modality (mechanical/thermal/electrical),
each with its own multiplier lambda_m and its own budget b_m — this
independence is the entire mechanism under test (proposal Sec. "Vector-
Constrained CMDP Formulation", Eq. vector-cmdp).

Per training rollout, each controller drives an inequality constraint
J_C <= b to zero via a PID controller on the constraint violation
e_k = J_C_k - b:
    I_k = clip(I_{k-1} + K_I * e_k, 0, I_max)
    lambda_k = clip(K_P * max(0, e_k) + I_k + K_D * max(0, e_k - e_{k-1}),
                     0, lambda_max)
following the P/I/D roles in Stooke et al. 2020 Sec. 3-4: the integral term
does the actual constraint-satisfaction work (drives the steady-state
multiplier), the proportional term reacts immediately to a currently-large
violation, and the derivative term damps oscillation by reacting to a
still-increasing violation (both P and D terms are zeroed when the
violation is improving, i.e. e_k <= e_{k-1} and e_k <= 0, matching the
paper's use of ReLU on the reactive terms so the controller never fights
a constraint that's already being satisfied).

We use mean per-step damage (health points lost per environment step,
averaged over the rollout) as the cost estimator J_C, and convert a
human-legible "health points allowed to be lost over one episode" budget
into a per-step budget via `budget / max_episode_steps`, so one budget
config is comparable across tasks with different episode lengths
(pick_egg: 300 steps; pour_water/fill_bowl: 400 steps).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

DEFAULT_BUDGET = 30.0  # health points/episode — matches object_health_floor's
                        # implicit "100 - 70 = 30 points of damage" headroom
                        # already used throughout src/cdp/tasks.py.
DEFAULT_K_P = 1e-2
DEFAULT_K_I = 1e-3
DEFAULT_K_D = 1e-2
DEFAULT_LAMBDA_MAX = 50.0
DEFAULT_INTEGRAL_MAX = 50.0


@dataclass
class PIDLagrangianConfig:
    budget: float = DEFAULT_BUDGET
    k_p: float = DEFAULT_K_P
    k_i: float = DEFAULT_K_I
    k_d: float = DEFAULT_K_D
    lambda_max: float = DEFAULT_LAMBDA_MAX
    integral_max: float = DEFAULT_INTEGRAL_MAX


class PIDLagrangian:
    """One constrained cost channel. `max_episode_steps` converts the
    human-facing episode budget into the per-step budget the controller
    actually tracks."""

    def __init__(self, config: PIDLagrangianConfig, max_episode_steps: int):
        self.config = config
        self.budget_step = config.budget / max(1, max_episode_steps)
        self.lam: float = 0.0
        self._integral = 0.0
        self._prev_error = 0.0
        self.history: List[dict] = []

    def update(self, mean_step_cost: float) -> float:
        error = mean_step_cost - self.budget_step
        self._integral = float(np.clip(
            self._integral + self.config.k_i * error, 0.0, self.config.integral_max
        ))
        # D-term only reacts while the violation is still growing (matches
        # Stooke et al.'s ReLU-gated reactive terms, see module docstring).
        derivative = max(0.0, error - self._prev_error)
        self.lam = float(np.clip(
            self.config.k_p * max(0.0, error) + self._integral + self.config.k_d * derivative,
            0.0, self.config.lambda_max,
        ))
        self._prev_error = error
        self.history.append({
            "mean_step_cost": mean_step_cost,
            "budget_step": self.budget_step,
            "error": error,
            "integral": self._integral,
            "lambda": self.lam,
        })
        return self.lam


class ScalarLagrangian:
    """`scalar_lagrangian` condition: one multiplier on the aggregate
    (summed-across-modality) per-step cost."""

    def __init__(self, config: PIDLagrangianConfig, max_episode_steps: int):
        self._pid = PIDLagrangian(config, max_episode_steps)

    def update(self, mean_costs: Dict[str, float]) -> float:
        return self._pid.update(sum(mean_costs.values()))

    @property
    def lam(self) -> float:
        return self._pid.lam

    @property
    def history(self) -> List[dict]:
        return self._pid.history


class VectorLagrangian:
    """`vector_lagrangian` condition: one independent multiplier per hazard
    modality — the mechanism the compositional-generalization hypothesis is
    actually about (proposal Sec. "Why scalarization is worse...")."""

    def __init__(self, configs_by_modality: Dict[str, PIDLagrangianConfig], max_episode_steps: int):
        self._pids = {
            m: PIDLagrangian(cfg, max_episode_steps) for m, cfg in configs_by_modality.items()
        }
        self.modalities = list(configs_by_modality.keys())

    def update(self, mean_costs: Dict[str, float]) -> Dict[str, float]:
        return {m: pid.update(mean_costs.get(m, 0.0)) for m, pid in self._pids.items()}

    @property
    def lam_vec(self) -> Dict[str, float]:
        return {m: pid.lam for m, pid in self._pids.items()}

    @property
    def history(self) -> Dict[str, List[dict]]:
        return {m: pid.history for m, pid in self._pids.items()}


class LambdaState:
    """Mutable holder shared between the training callback (writer, once per
    rollout) and CDPTaskEnv (reader, every step) — decouples "when the
    multiplier updates" (SB3 rollout boundary) from "when it's applied"
    (every env.step's reward), without threading SB3 internals through the
    env. Scalar conditions store a float in `.value`; vector conditions
    store a {modality: float} dict."""

    def __init__(self, initial):
        self.value = initial

    def set(self, value) -> None:
        self.value = value
