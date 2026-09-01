"""
Evaluation metrics for the Safety-Gymnasium domain — nav analogue of
src/cdp/eval.py, same metric definitions (TCR/STCR/SafetyGap/Delta_comp/
zero_shot_gap/V_m) but reading `summary.jsonl` rows shaped by
cdp_nav.logger.NavEpisodeLogger (`total_cost`/`cost_{modality}` fields
instead of `total_damage`/`damage_{modality}`).
"""
from __future__ import annotations

import json
from typing import Dict, List, Sequence


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
    """Delta_comp = STCR_joint - STCR_single-exposure (RQ2), same eval task."""
    return stcr(joint_exposure_rows) - stcr(single_exposure_rows)


def zero_shot_gap(seen_rows: Sequence[dict], composite_rows: Sequence[dict]) -> float:
    return stcr(seen_rows) - stcr(composite_rows)


def mean_field(rows: Sequence[dict], field: str) -> float:
    vals = [r[field] for r in rows if field in r]
    return sum(vals) / len(vals) if vals else float("nan")


def median_field(rows: Sequence[dict], field: str) -> float:
    vals = sorted(r[field] for r in rows if field in r)
    n = len(vals)
    if n == 0:
        return float("nan")
    mid = n // 2
    return vals[mid] if n % 2 else (vals[mid - 1] + vals[mid]) / 2


def violation_rate(rows: Sequence[dict], modality: str, budget: float) -> float:
    """V_m = Pr[J^{C_m}(pi) > b_m]."""
    field = f"cost_{modality}"
    vals = [r[field] for r in rows if field in r]
    if not vals:
        return float("nan")
    return sum(1 for v in vals if v > budget) / len(vals)


def group_by(rows: Sequence[dict], key: str) -> Dict[str, List[dict]]:
    groups: Dict[str, List[dict]] = {}
    for r in rows:
        groups.setdefault(r[key], []).append(r)
    return groups


def metrics_table(rows: Sequence[dict], group_key: str = "condition") -> Dict[str, dict]:
    out = {}
    for key, group_rows in group_by(rows, group_key).items():
        out[key] = {
            "n_episodes": len(group_rows),
            "tcr": tcr(group_rows),
            "stcr": stcr(group_rows),
            "safety_gap": safety_gap(group_rows),
            "mean_total_cost": mean_field(group_rows, "total_cost"),
            "median_total_cost": median_field(group_rows, "total_cost"),
        }
    return out
