"""
Reward + cost-penalty combination for the Safety-Gymnasium domain — nav
analogue of src/cdp/reward.py's `apply_damage_penalty`. Deliberately a thin,
separate function (not a shared abstraction with the manipulation domain's
version) since the two domains' modality names/counts differ; the actual
shared mechanism (`cdp.lagrangian`'s PID controllers) is imported unchanged,
per proposal.tex's "identical code reused across domains" claim — only this
glue layer and the observation wrapper are domain-specific.

    task_only         : r_t = r_env_t                    (env's own dense reward)
    scalar_lagrangian : r_t = r_env_t - lambda_t * cost_scalar_t
    vector_lagrangian : r_t = r_env_t - sum_m lambda_m,t * cost_m,t
    fixed_weight      : r_t = r_env_t - sum_m lambda_m * cost_m,t  (constant lambda_m)
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

Condition = str  # "task_only" | "scalar_lagrangian" | "vector_lagrangian" | "fixed_weight"


def apply_cost_penalty(
    r_env: float,
    condition: Condition,
    modalities: Sequence[str],
    cost_by_modality: Dict[str, float],
    lam=None,
) -> float:
    if condition == "task_only":
        return r_env
    if condition == "scalar_lagrangian":
        assert lam is not None
        return r_env - float(lam) * sum(cost_by_modality.values())
    if condition in ("vector_lagrangian", "fixed_weight"):
        assert lam is not None
        penalty = sum(float(lam[m]) * cost_by_modality.get(m, 0.0) for m in modalities)
        return r_env - penalty
    raise ValueError(f"unknown condition {condition!r}")


def compute_cost_by_modality(info: dict, cost_keys: Dict[str, Sequence[str]]) -> Dict[str, float]:
    return {m: sum(float(info.get(k, 0.0)) for k in keys) for m, keys in cost_keys.items()}
