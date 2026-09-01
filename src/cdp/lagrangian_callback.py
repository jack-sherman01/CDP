"""
SB3 training callback that drives the PID-Lagrangian multiplier update once
per rollout (see src/cdp/lagrangian.py for the controller itself).

CDPTaskEnv.step() already returns per-modality damage in `info["damage_by_
modality"]` for the logger; this callback reads that same field out of SB3's
`self.locals["infos"]` at every env step SB3 collects, accumulates a
rollout-mean per modality, and at `_on_rollout_end` feeds it to the
Lagrangian controller and writes the new multiplier(s) into the env's
shared `LambdaState`. No changes to PPO's rollout collection are needed —
this only writes into a mutable object CDPTaskEnv already reads from every
`step()` call.
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
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.lagrangian = lagrangian
        self.lam_state = lam_state
        self.modalities = list(modalities)
        self._cost_sums: Dict[str, float] = {m: 0.0 for m in self.modalities}
        self._n_steps = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            d = info.get("damage_by_modality")
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
