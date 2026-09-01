# CDP
Compositional Generalization in Safe Robot Learning: does a policy trained
on individual hazard types (single-exposure) retain a comparable safety
advantage when those hazards co-occur zero-shot, vs. a policy trained
directly on the combination (joint-exposure)? Tested with a vector-valued
CMDP / independent per-hazard-type PID-Lagrangian multipliers, across two
unrelated domains: household manipulation (DamageSim/OopsieBench) and safe
navigation (Safety-Gymnasium).

Research proposal: `private/proposal.tex` (read-only, source of truth for
scope/design — never modified). Execution plan, decisions, and daily
progress: `docs/PLAN.md`, `docs/DECISIONS.md`, `docs/DAILY_LOG.md`. Task
specs: `docs/TASKS.md`. Large/machine-specific artifacts (conda envs,
datasets, checkpoints, run logs) live outside this repo on `/data/heng/cdp`
(manipulation) / `/data/heng/cdp/*_nav` (navigation) — see `docs/PLAN.md`
for the layout.

**Note (2026-09-01):** the proposal was substantially revised mid-project
(new title above, from the original "Compositional Damage-Aware Policies
for Safe Household Robot Manipulation" scope) — see `docs/DECISIONS.md`'s
2026-09-01 entry for the full technical delta. Real-robot validation
(proposal's RQ5) is out of scope for autonomous execution here (no
physical hardware reachable from this environment); everything else is
in scope and in progress.

## Code layout

**Manipulation domain** (`oopsieverse_b1k` conda env, Isaac Sim/OmniGibson
— one simulator process at a time on this machine):
- `src/cdp/` — observation wrapper, reward, PID-Lagrangian controller
  (`lagrangian.py`) + its SB3 training callback (`lagrangian_callback.py`),
  logger, eval metrics/plots, task registry, corruption tests, the
  SB3-compatible Gym env.
- `scripts/` — `train_ppo.py`, `evaluate.py`, `analyze.py`,
  `compute_compositional_gap.py`, `make_comparison_video.py`,
  `run_all_manip_training.sh`, `verify_damagesim.py`.
- `tests/` — live-sim verification scripts + a simulator-free
  logger/eval smoke test.
- `results/` — small, durable artifacts (final tables/figures) copied in
  from `/data/heng/cdp` for the paper; version-controlled.

**Navigation domain** (`cdp_nav` conda env, pure MuJoCo via
`safety-gymnasium` — cheap, runs many in parallel):
- `src/cdp_nav/` — single-exposure hazard-isolated task variants
  (`custom_tasks.py`), task registry, observation wrapper, reward/logger/
  eval (mirrors `src/cdp/`'s structure; reuses `cdp.lagrangian` unchanged).
- `scripts_nav/` — `train_ppo_nav.py`, `evaluate_nav.py`, `analyze_nav.py`,
  `compute_compositional_gap_nav.py`, `run_all_nav_training.sh`.
- `results_nav/` — same role as `results/`, for this domain.
