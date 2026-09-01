"""
Evaluation metrics + plots (Day 4), computed from the `summary.jsonl` rows
written by `src/cdp/logger.py`.

Metrics match `private/proposal.tex` exactly (post-pivot to the vector-CMDP /
PID-Lagrangian formulation, see docs/DECISIONS.md):
    TCR        = successful / total
    STCR       = successful_and_safe / total
    SafetyGap  = TCR - STCR
    Delta_comp = STCR_joint-exposure - STCR_single-exposure
                 (residual gap between single-exposure structured training
                 and the joint-exposure upper bound, both evaluated on the
                 SAME held-out-hazard-combination task, RQ2 — NOT the
                 seen-vs-unseen gap for one policy; that's `zero_shot_gap`
                 below, which is what our earlier pick_egg-on-food_in_
                 microwave headline result actually measured before this
                 pivot)
    V_m        = Pr[J^{C_m}(pi) > b_m]  (per-modality budget violation rate)
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence

MODALITIES = ("mechanical", "thermal", "electrical")


def load_summary(path: str) -> List[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def tcr(rows: Sequence[dict]) -> float:
    if not rows:
        return float("nan")
    return sum(1 for r in rows if r["success"]) / len(rows)


def stcr(rows: Sequence[dict]) -> float:
    if not rows:
        return float("nan")
    return sum(1 for r in rows if r["successful_and_safe"]) / len(rows)


def safety_gap(rows: Sequence[dict]) -> float:
    return tcr(rows) - stcr(rows)


def compositional_gap(single_exposure_rows: Sequence[dict], joint_exposure_rows: Sequence[dict]) -> float:
    """Delta_comp = STCR_joint - STCR_single-exposure (proposal RQ2): the
    residual gap between single-exposure vector_lagrangian evaluated
    zero-shot on a held-out composite task, and vector_lagrangian trained
    directly on that same composite task (joint-exposure upper bound).
    Both row sets must be the SAME eval_task, different training exposure."""
    return stcr(joint_exposure_rows) - stcr(single_exposure_rows)


def zero_shot_gap(seen_rows: Sequence[dict], composite_rows: Sequence[dict]) -> float:
    """STCR_seen - STCR_composite for one single-exposure policy: how much
    STCR drops moving from its own training task to an unseen combination.
    (Pre-pivot this module called this `compositional_gap`; kept under a new
    name since RQ2's Delta_comp now means something different — see the
    module docstring.)"""
    return stcr(seen_rows) - stcr(composite_rows)


def damage_increase(single_rows: Sequence[dict], composite_rows: Sequence[dict]) -> float:
    """Delta_damage = (D_composite - D_single) / D_single (proposal Day 19)."""
    d_single = mean_field(single_rows, "total_damage")
    d_composite = mean_field(composite_rows, "total_damage")
    if d_single == 0:
        return float("inf") if d_composite > 0 else 0.0
    return (d_composite - d_single) / d_single


def violation_rate(rows: Sequence[dict], modality: str, budget: float) -> float:
    """V_m = Pr[J^{C_m}(pi) > b_m]: fraction of episodes whose per-modality
    damage exceeded that modality's budget (proposal's Evaluation Metrics
    section). `budget` should be in the same "health points/episode" units
    as `cdp.lagrangian.PIDLagrangianConfig.budget`."""
    field = f"damage_{modality}"
    vals = [r[field] for r in rows if field in r]
    if not vals:
        return float("nan")
    return sum(1 for v in vals if v > budget) / len(vals)


def mean_field(rows: Sequence[dict], field: str) -> float:
    vals = [r[field] for r in rows if field in r]
    return sum(vals) / len(vals) if vals else float("nan")


def median_field(rows: Sequence[dict], field: str) -> float:
    """More outlier-robust than mean_field — early-training policies can
    produce rare catastrophic episodes (e.g. stuck in sustained high-force
    contact) that skew the mean by orders of magnitude; report both."""
    vals = sorted(r[field] for r in rows if field in r)
    n = len(vals)
    if n == 0:
        return float("nan")
    mid = n // 2
    return vals[mid] if n % 2 else (vals[mid - 1] + vals[mid]) / 2


def group_by(rows: Sequence[dict], key: str) -> Dict[str, List[dict]]:
    groups: Dict[str, List[dict]] = {}
    for r in rows:
        groups.setdefault(r[key], []).append(r)
    return groups


def metrics_table(rows: Sequence[dict], group_key: str = "condition") -> Dict[str, dict]:
    """{group_value: {tcr, stcr, safety_gap, n_episodes, mean_damage, ...}}"""
    out = {}
    for key, group_rows in group_by(rows, group_key).items():
        out[key] = {
            "n_episodes": len(group_rows),
            "tcr": tcr(group_rows),
            "stcr": stcr(group_rows),
            "safety_gap": safety_gap(group_rows),
            "mean_total_damage": mean_field(group_rows, "total_damage"),
            "median_total_damage": median_field(group_rows, "total_damage"),
            "mean_damage_mechanical": mean_field(group_rows, "damage_mechanical"),
            "mean_damage_thermal": mean_field(group_rows, "damage_thermal"),
            "mean_damage_electrical": mean_field(group_rows, "damage_electrical"),
            "mean_min_object_health": mean_field(group_rows, "min_object_health"),
        }
    return out


# ── Plots ──────────────────────────────────────────────────────────────

def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_learning_curve(rows: Sequence[dict], out_path: str, window: int = 20) -> None:
    """Rolling-mean total_reward vs. episode index, one line per condition."""
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(6, 4))
    for cond, group_rows in group_by(rows, "condition").items():
        rewards = [r["total_reward"] for r in group_rows]
        if len(rewards) >= window:
            roll = [
                sum(rewards[max(0, i - window):i + 1]) / len(rewards[max(0, i - window):i + 1])
                for i in range(len(rewards))
            ]
        else:
            roll = rewards
        ax.plot(roll, label=cond)
    ax.set_xlabel("episode")
    ax.set_ylabel(f"reward ({window}-ep rolling mean)")
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_completion_rates(rows: Sequence[dict], out_path: str) -> None:
    """Bar chart of TCR and STCR per condition."""
    plt = _mpl()
    table = metrics_table(rows, "condition")
    conds = list(table.keys())
    tcrs = [table[c]["tcr"] for c in conds]
    stcrs = [table[c]["stcr"] for c in conds]
    x = range(len(conds))
    fig, ax = plt.subplots(figsize=(6, 4))
    width = 0.35
    ax.bar([i - width / 2 for i in x], tcrs, width, label="TCR")
    ax.bar([i + width / 2 for i in x], stcrs, width, label="STCR")
    ax.set_xticks(list(x))
    ax.set_xticklabels(conds, rotation=20, ha="right")
    ax.set_ylabel("rate")
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_damage_by_modality(rows: Sequence[dict], out_path: str) -> None:
    plt = _mpl()
    table = metrics_table(rows, "condition")
    conds = list(table.keys())
    fig, ax = plt.subplots(figsize=(6, 4))
    bottom = [0.0] * len(conds)
    for m in MODALITIES:
        vals = [table[c][f"mean_damage_{m}"] for c in conds]
        ax.bar(conds, vals, bottom=bottom, label=m)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_ylabel("mean total damage")
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels(conds, rotation=20, ha="right")
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_min_health_per_episode(rows: Sequence[dict], out_path: str) -> None:
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(6, 4))
    for cond, group_rows in group_by(rows, "condition").items():
        vals = [r["min_object_health"] for r in group_rows]
        ax.plot(vals, label=cond, alpha=0.7)
    ax.set_xlabel("episode")
    ax.set_ylabel("min object health")
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
