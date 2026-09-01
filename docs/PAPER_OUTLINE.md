# Paper Outline & Reproducibility Checklist (Days 27-28)

Skeleton only — filled in once Days 5-13 (compressed plan) produce real
results. Section structure follows `private/proposal.tex`'s own framing so
the paper and the proposal stay traceable to each other.

## Sections

1. **Abstract** — restate the research question + the one-line finding
   once known (structured damage helps / doesn't help compositional
   generalization, and by how much on `STCR_vector - STCR_scalar`).
2. **Introduction** — motivate via the sink/laptop example already in
   `proposal.tex` Sec. "Research Question."
3. **Related work** — OOPSIEVERSE/DamageSim (base system), safe RL /
   constrained RL, compositional generalization in RL.
4. **Method** — damage vector formulation (`proposal.tex` Sec. "Proposed
   Contribution"), the three observation modes (`docs/OBSERVATIONS.md`),
   the three policy conditions.
5. **Experimental setup** — simulator/backend choice + why (`docs/
   DECISIONS.md`), tasks (`docs/TASKS.md`), PPO hyperparameters
   (`scripts/train_ppo.py`), evaluation metrics (`src/cdp/eval.py`).
6. **Results**
   - Single-modality training curves + final TCR/STCR (3 conditions x 3
     tasks).
   - Zero-shot composite generalization: `STCR_vector - STCR_scalar` on
     `fill_bowl` / `heat_saucepot` (the primary comparison, per
     `proposal.tex` Sec. "Days 17-18").
   - `Delta_comp` (seen vs. composite STCR gap) per condition.
   - Corruption robustness (Day 20).
   - Ablations: modality dropout (Days 22-23), damage-weight sensitivity
     (Day 24).
7. **Discussion** — where the hypothesis held / didn't, and why (tie back
   to specific failure modes seen in per-episode logs, not just aggregate
   numbers).
8. **Limitations** — explicitly: single simulator backend, low-dim state
   only (no vision), small task set, single seed per condition unless
   Day 21+ budget allowed more, BEHAVIOR-1K substitution rationale.
9. **Conclusion**
10. **Reproducibility checklist** (Day 27) — see below.

## Reproducibility checklist (Day 27)

To fill in once experiments are complete:

- [ ] Every reported number traces to a specific `run_dir` under
      `/data/heng/cdp/runs/<experiment_id>/`, keyed
      `{condition}_{task}_{seed}_{timestamp}` (never overwritten, per
      `docs/PLAN.md`).
- [ ] All hyperparameters used are the ones fixed in `scripts/train_ppo.py`
      (proposal Sec. "Days 5-6"); any deviation documented in
      `docs/DECISIONS.md`.
- [ ] Software/hardware versions recorded: OmniGibson version, Isaac Sim
      version, conda env (`oopsieverse_b1k`), Python 3.10, 2x RTX A5000,
      driver 535.230.02 (see `docs/DAILY_LOG.md` Day 1).
- [ ] `results/main_comparison_table.csv` (from `scripts/analyze.py`)
      matches the numbers quoted in the paper.
- [ ] Figures in `results/figures/` regenerate from `scripts/analyze.py`
      with no manual editing.
- [ ] Seeds used per condition/task listed explicitly (single-seed runs
      flagged as such — no implied multi-seed statistics where there's
      only one seed).
- [ ] Scope/substitution disclosure present in the paper itself, not just
      in `docs/DECISIONS.md`: BEHAVIOR-1K over RoboCasa, fixed damage-
      weight `lambda=0.05` default, the reward-shaping design in
      `src/cdp/reward.py` (task reward is our own construction — the base
      repo has no reward/termination logic, only `task_completion_check`).
