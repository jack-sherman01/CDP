"""
Task registry for the Safety-Gymnasium cross-domain validation (proposal.tex
"Cross-Domain Validation: Safe Navigation", RQ4) — mirrors src/cdp/tasks.py's
role for the manipulation domain, but here the "task" IS the env id (no
separate task_completion_check module to load).

Two independent hazard-domains, each with its own two hazard modalities
(see src/cdp_nav/custom_tasks.py for why these single-exposure variants
needed to be built from scratch rather than reusing a built-in env id):
    goal   : hazards (region-entry) + vases (fragile-object contact)
    button : gremlins (dynamic-obstacle contact) + buttons (wrong-button press)

`cost_keys` maps each domain modality to the raw `info` dict key(s)
safety_gymnasium's `Builder._cost()` populates (some, like vases, split
contact/velocity into two keys we sum into one modality-level cost).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

DOMAIN_MODALITIES: Dict[str, Tuple[str, ...]] = {
    "goal": ("hazards", "vases"),
    "button": ("gremlins", "buttons"),
}

# Safety Gym benchmark convention (Ray, Achiam & Amodei 2019 "Benchmarking
# Safe Exploration in Deep RL"): cost_limit = 25 (cumulative 0/1-per-step
# indicator cost over an episode) is the standard constraint budget used
# across that benchmark and most follow-on Safety-Gym papers — reused here
# unchanged as our per-modality budget b_m, same role DEFAULT_BUDGET (health
# points/episode) plays in the manipulation domain's src/cdp/lagrangian.py.
DEFAULT_NAV_BUDGET = 25.0


@dataclass
class NavTaskSpec:
    task_name: str
    env_id: str
    agent_name: str  # "Point" | "Car"
    domain: str  # "goal" | "button" -- selects DOMAIN_MODALITIES + obs layout
    cost_keys: Dict[str, Tuple[str, ...]]
    max_episode_steps: int = 1000
    budget: float = DEFAULT_NAV_BUDGET


_GOAL_COST_KEYS = {
    "hazards": ("cost_hazards",),
    "vases": ("cost_vases_contact", "cost_vases_velocity"),
}
_BUTTON_COST_KEYS = {
    "gremlins": ("cost_gremlins",),
    "buttons": ("cost_buttons",),
}

NAV_TASK_REGISTRY: Dict[str, NavTaskSpec] = {
    # ── Single-exposure training tasks ──────────────────────────────────
    "goal_hazards_only": NavTaskSpec(
        "goal_hazards_only", "SafetyPointGoalHazardsOnly2-v0", "Point", "goal", _GOAL_COST_KEYS,
    ),
    "goal_vases_only": NavTaskSpec(
        "goal_vases_only", "SafetyPointGoalVasesOnly2-v0", "Point", "goal", _GOAL_COST_KEYS,
    ),
    "button_gremlins_only": NavTaskSpec(
        "button_gremlins_only", "SafetyCarButtonGremlinsOnly1-v0", "Car", "button", _BUTTON_COST_KEYS,
    ),
    "button_wrong_button_only": NavTaskSpec(
        "button_wrong_button_only", "SafetyCarButtonWrongButtonOnly1-v0", "Car", "button", _BUTTON_COST_KEYS,
    ),
    # ── Joint-exposure training / zero-shot-eval composite tasks ───────
    "goal_joint": NavTaskSpec(
        "goal_joint", "SafetyPointGoal2-v0", "Point", "goal", _GOAL_COST_KEYS,
    ),
    "button_joint": NavTaskSpec(
        "button_joint", "SafetyCarButtonCombo1-v0", "Car", "button", _BUTTON_COST_KEYS,
    ),
}

NAV_TRAINING_TASKS = (
    "goal_hazards_only", "goal_vases_only", "button_gremlins_only", "button_wrong_button_only",
)
NAV_COMPOSITE_EVAL_TASKS = ("goal_joint", "button_joint")


def get_nav_task_spec(task_name: str) -> NavTaskSpec:
    if task_name not in NAV_TASK_REGISTRY:
        raise KeyError(f"Unknown nav task {task_name!r}; available: {sorted(NAV_TASK_REGISTRY)}")
    return NAV_TASK_REGISTRY[task_name]
