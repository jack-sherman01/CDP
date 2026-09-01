# CDP — 2-Week Execution Plan

Source of truth for scope/research design is `private/proposal.tex` (read-only,
never modified — see repo root note in `README.md`). This document is our
**derived, compressed execution plan**: the proposal's 4-week technical plan
(Days 1–28) mapped onto a hard 2-week (14 working day) budget, plus the
engineering decisions needed to make it feasible on this specific machine.

Start date: 2026-08-29. Target completion: 2026-09-12 (14 days).

## Compression mapping (28-day plan -> 14-day plan)

| New day | Original day(s) | Content |
|---|---|---|
| 1 | Day 1 | Simulator setup + verification loop |
| 2 | Day 2 | Task & object selection, task specs |
| 3 | Day 3–4 | Observation wrapper (task/scalar/vector modes) + logging infra |
| 4–5 | Day 5–7 | Task-only PPO baseline + Week-1 validation |
| 6 | Day 8–9 | Scalar-health baseline + sanity checks |
| 7 | Day 10–11 | Structured-damage (vector) representation |
| 8 | Day 12 | Train all 3 conditions on single-modality tasks |
| 9 | Day 13–14 | Diagnostics (masking, zeroing, correlations) |
| 10 | Day 15–16 | Composite task construction + scripted validation |
| 11 | Day 17–18 | Zero-shot evaluation on composite tasks |
| 12 | Day 19–20 | Compositional generalization analysis + corruption tests |
| 13 | Day 21–24 | Statistical analysis + modality-dropout + weight-sensitivity ablations |
| 14 | Day 27–28 | Reproducibility pass + paper draft |

Dropped from the 4-week plan to fit 2 weeks (explicitly marked optional in
`private/proposal.tex`, Sections "Experimental Conditions" and Days 25–26):
- The ensemble-uncertainty policy experiment. Revisit only if days 1–13
  finish early.

## Key engineering decision: simulator backend

`private/proposal.tex` states a preference for OmniGibson/BEHAVIOR-1K. We
initially substituted RoboCasa (MuJoCo) for disk/time reasons, then reversed
that after reading the actual DamageSim evaluator code: RoboCasa only has a
mechanical evaluator wired up in this codebase, while BEHAVIOR-1K/OmniGibson
has all three (mechanical/thermal/electrical-fluid). We use
**BEHAVIOR-1K/OmniGibson**. Full history and rationale in
`docs/DECISIONS.md` (see the two entries dated 2026-08-29).

## Repo/data layout

Large, regenerable, or machine-specific artifacts live outside the git repo,
on `/data/heng/cdp` (228GB free vs. 5.9GB free on the root disk, which is
shared across ~30 users on this machine):

```
/data/heng/cdp/
  external/oopsieverse/   # cloned OOPSIEVERSE codebase (pinned commit, see DECISIONS.md)
  miniconda -> /data/heng/miniconda3   # conda envs created by oopsieverse's install.py
  data/                   # downloaded demo datasets
  checkpoints/            # trained policy weights, per run
  logs/                   # per-episode logs + per-experiment summaries + tensorboard
  runs/                   # run configs actually used (resolved, with seeds) — reproducibility record
```

Everything above is the **experimental record** and must never be silently
overwritten: each run writes to `runs/<experiment_id>/` keyed by
`{condition}_{task}_{seed}_{timestamp}`.

Small, durable artifacts (final tables/figures/summary CSVs used in the
paper) are copied into `results/` in the git repo, so the paper's numbers are
version-controlled even though raw logs live on `/data`.

## Daily log

Progress, blockers, and decisions made along the way are appended to
`docs/DAILY_LOG.md` — one entry per working day, dated, never rewritten.

## 2026-09-01 — Proposal pivot: revised scope and plan

`private/proposal.tex` was substantially revised by the user (see
`docs/DECISIONS.md`'s 2026-09-01 entry for the full technical delta). The
proposal's own "Revised Technical Plan" now spans 8 weeks / 4 phases; this
section maps that onto what's actually executable from this environment.

**Phase 1-3 (manipulation domain, weeks 1-6 of the new plan) — in scope,
continuing autonomously:**
- PID-Lagrangian mechanism (`src/cdp/lagrangian.py`,
  `src/cdp/lagrangian_callback.py`) implemented and unit-tested.
- New condition taxonomy wired through `reward.py`/`gym_env.py`/
  `train_ppo.py`/`evaluate.py`: `task_only`, `scalar_lagrangian`,
  `vector_lagrangian`, `fixed_weight`.
- Joint-exposure training (train `vector_lagrangian` directly on a
  composite task) — no new code path, just `--task_name` drawn from
  `COMPOSITE_EVAL_TASKS` instead of `TRAINING_TASKS`.
- Re-run task_only / scalar_lagrangian / vector_lagrangian on all 3
  single-hazard tasks (pick_egg, add_firewood, pour_water), 1 seed to start
  (matching the budget-per-seed reality established in Week 1, see
  DAILY_LOG's wall-clock notes) — multi-seed expansion opportunistically as
  time allows, proposal calls for 5.
- Train vector_lagrangian joint-exposure on fill_bowl and
  food_in_microwave (the RQ2 upper bound).
- Train fixed_weight ablation on the 3 single-hazard tasks (RQ3).
- Zero-shot eval of all single-exposure checkpoints on the composite tasks;
  `scripts/compute_compositional_gap.py` for RQ2's Delta_comp against the
  joint-exposure runs.
- Corruption robustness (Day 20 in the old compressed plan), modality
  dropout, budget-sensitivity ablations — code already exists
  (`cdp/corruption.py`, `--modality_dropout_p`, `--budget`), needs runs.
- Cross-simulator (RoboCasa) generalization test — **not pursued**: per
  `docs/DECISIONS.md`'s 2026-08-29 reversal, RoboCasa only has a mechanical
  damage evaluator in this codebase, so a composite (multi-hazard) task has
  nothing to instantiate there. Flagged as a known gap vs. the proposal's
  "Primary Domain" section, not silently dropped.

**Phase 3 continued (Safety-Gymnasium cross-domain, RQ4) — in scope:** new,
independent codebase surface (`safety-gymnasium` pip package, Gymnasium-
based, no Isaac Sim dependency) — can be developed and run in parallel with
manipulation-domain training since it doesn't touch the Isaac Sim process
this machine can only run one instance of at a time. Reuses
`src/cdp/lagrangian.py` unchanged (proposal's explicit point: "the
constrained-optimization mechanism... [is] identical code reused across
domains").

**Phase 4 (real-robot validation, RQ5) — out of scope for autonomous
execution.** This environment has no physical robot, F/T sensor, or gripper
reachable from it. What *is* in scope: preparing the transfer/deployment
script and force-torque logging format so the user (or lab-present
personnel) can run the actual trials. The paper's real-robot section will
need to be filled in outside this session, or the paper scoped down to
simulation-only with this limitation stated explicitly — flagged here
rather than silently faked or dropped.
