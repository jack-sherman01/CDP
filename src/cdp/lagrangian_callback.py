"""
SB3 training callback that drives the PID-Lagrangian multiplier update once
per rollout (see src/cdp/lagrangian.py for the controller itself).

CDPTaskEnv.step() returns per-modality damage in `info["damage_by_
modality"]`; NavTaskEnv (src/cdp_nav/gym_env.py) returns per-modality cost
in `info["cost_by_modality"]` — different key names since "damage" and
"cost" are genuinely different concepts in the two domains, not just a
naming inconsistency to paper over. This callback reads whichever key
`info_key` names out of SB3's `self.locals["infos"]` at every env step SB3
collects, accumulates a rollout-mean per modality, and at `_on_rollout_end`
feeds it to the Lagrangian controller and writes the new multiplier(s) into
the env's shared `LambdaState`. No changes to PPO's rollout collection are
needed — this only writes into a mutable object the env already reads from
every `step()` call.

**Bug history (2026-09-01, private/CONTRIBUTIONS_LOG.md):** this callback
originally hardcoded `"damage_by_modality"`. Reused as-is for the
navigation domain, it silently never found that key in NavTaskEnv's info
dicts (which uses `"cost_by_modality"`), so `_n_steps` stayed 0 forever and
`_on_rollout_end` returned immediately every rollout without ever calling
`self.lagrangian.update(...)` — every nav-domain `*_lagrangian` checkpoint
trained with this bug has `lambda` stuck at exactly 0.0 (verified: every
`lambda_final.json` under `/data/heng/cdp/checkpoints_nav/*_lagrangian_*`
reads 0.0), making `scalar_lagrangian`/`vector_lagrangian` behaviorally
identical to `task_only` for that whole batch. `task_only` and
`fixed_weight` runs are unaffected (neither uses this callback).
Manipulation-domain runs are unaffected (`CDPTaskEnv` always used
`"damage_by_modality"`, matching the callback's original hardcoded key).
Fixed by making the key configurable instead of assuming the manipulation
domain's name; affected nav runs need retraining.
"""
from __future__ import annotations

from typing import Dict, Sequence, Union

from stable_baselines3.common.callbacks import BaseCallback

from cdp.lagrangian import LambdaState, ScalarLagrangian, VectorLagrangian


class LagrangianUpdateCallback(BaseCallback):
    def __init__(
        self,
        lagrangian: Union[ScalarLagrangian, VectorLagrangian],
        lam_state: LambdaState,
        modalities: Sequence[str] = ("mechanical", "thermal", "electrical"),
        info_key: str = "damage_by_modality",
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.lagrangian = lagrangian
        self.lam_state = lam_state
        self.modalities = list(modalities)
        self.info_key = info_key
        self._cost_sums: Dict[str, float] = {m: 0.0 for m in self.modalities}
        self._n_steps = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            d = info.get(self.info_key)
            if d is None:
                continue
            for m in self.modalities:
                self._cost_sums[m] += float(d.get(m, 0.0))
            self._n_steps += 1
        return True

    def _on_rollout_end(self) -> None:
        if self._n_steps == 0:
            return
        mean_costs = {m: self._cost_sums[m] / self._n_steps for m in self.modalities}
        new_lam = self.lagrangian.update(mean_costs)
        self.lam_state.set(new_lam)
        if self.verbose:
            print(f"[LagrangianUpdateCallback] mean_costs={mean_costs} -> lambda={new_lam}")
        self._cost_sums = {m: 0.0 for m in self.modalities}
        self._n_steps = 0
