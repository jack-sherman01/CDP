# Engineering Decisions Log

Append-only. Each entry: date, decision, why, what it costs us.

## 2026-08-29 — Simulator backend: RoboCasa (MuJoCo), not BEHAVIOR-1K/OmniGibson

**Context.** `private/proposal.tex` says: "One simulator backend
(OmniGibson), supporting mechanical, thermal, and fluid damage... preferably
OmniGibson because it supports mechanical, thermal, and fluid damage
signals." The OOPSIEVERSE codebase (github.com/UT-Austin-RobIn/oopsieverse)
that the proposal's abstract commits us to using supports two backends
behind one DamageSim API: RoboCasa (MuJoCo) and BEHAVIOR-1K (OmniGibson /
NVIDIA Omniverse / Isaac Sim). DamageSim itself is explicitly
simulator-agnostic — mechanical, thermal, and fluid damage are available on
either backend.

**Constraint found.** This machine's root filesystem has 5.9GB free (3.5TB
disk, 100% full, shared across ~30 users). A separate `/data` mount has
228GB free and is the machine's established convention for large per-user
data (every other user on the box keeps large artifacts under
`/data/<username>/`, not `/home/<username>/`).

- BEHAVIOR-1K requires an Isaac Sim / Omniverse install, which alone is
  commonly 50-100GB+, plus BEHAVIOR dataset assets on top. Even redirected
  entirely to `/data`, this is a slow, historically fragile install
  (driver/Omniverse-launcher/licensing issues are a common source of
  multi-day delays) — a real risk to a 14-day budget where most of the time
  needs to go to training/evaluation, not simulator bring-up.
- RoboCasa runs on MuJoCo: install is minutes, footprint is a few GB,
  well-trodden with `robosuite`/`robomimic` tooling, and CPU/GPU-light
  enough to run many parallel low-dimensional-observation PPO jobs on this
  machine's 2x RTX A5000 (24GB each) without contention.

**Decision.** Use RoboCasa as the simulator backend for the whole project.
Install via `python install.py --new_env --robocasa` from the OOPSIEVERSE
repo, per its documented instructions.

**What this costs.** We deviate from the proposal's stated backend
preference. Nothing else changes: the damage-vector formulation, PPO
architecture, reward shaping, task structure, and evaluation metrics in
`private/proposal.tex` are all defined at the DamageSim API level, which is
identical across backends. This should be stated plainly in the paper's
"Scope and Feasibility" section as a deliberate, documented substitution
made for the same reason the proposal itself gives for restricting scope
(feasibility within a fixed time budget), not as an unexplained deviation.

**Confirmed with user:** 2026-08-29, user asked for the model's
recommendation and approved proceeding with RoboCasa.

**REVERSED same day — see next entry.** The RoboCasa decision above was made
from the OOPSIEVERSE website/docs, which describe DamageSim as
"simulator-agnostic" at the API level. That is true of the interface but not
of which evaluators are actually wired up per backend. Reading the real
evaluator code changed the conclusion — do not act on this entry alone.

## 2026-08-29 — REVERSAL: simulator backend must be BEHAVIOR-1K/OmniGibson, not RoboCasa

**What changed.** After the RoboCasa decision above, I read the actual
DamageSim evaluator code in the cloned repo
(`/data/heng/cdp/external/oopsieverse`, commit `151efcee2200e3ec1ad76524a5961aef15ce5f28`),
not just the docs site. Findings:

- `damagesim/robosuite/evaluators/__init__.py` (the RoboCasa/MuJoCo backend)
  registers **only** `"mechanical": RSMechanicalDamageEvaluator`. There is no
  thermal or fluid evaluator for RoboCasa.
- `damagesim/robosuite/params/damage_params.py` confirms every RoboCasa
  object/fixture/robot config lists `damage_evaluators: ["mechanical"]` —
  nothing else, across the whole file.
- The three-modality model (mechanical / thermal / **electrical**, not
  "fluid" — water damage is implemented as a form of electrical damage via
  water-particle contacts, see `damagesim/omnigibson/evaluators/electrical.py`)
  only exists on the BEHAVIOR-1K/OmniGibson backend
  (`damagesim/omnigibson/evaluators/{mechanical,thermal,electrical}.py`).
  The `pour_water` task (`oopsiebench/envs/behavior1k/pour_water.py`)
  explicitly requires `USE_GPU_DYNAMICS = True` for fluid-particle
  simulation and is BEHAVIOR-1K-only; there is no RoboCasa equivalent.

**Why this matters.** The paper's central research question — whether
factorizing damage by modality improves compositional generalization —
requires all three modalities to exist. On RoboCasa, two of three don't
exist in this codebase, so there is nothing to factorize; the whole
scalar-vs-vector comparison collapses to a 1-D signal either way.
Implementing new thermal/fluid evaluators for RoboCasa ourselves would
violate the proposal's own Day-2 constraint ("we will avoid introducing new
physical damage models during the project") — that would be new science,
not an engineering substitution.

**Decision (supersedes the RoboCasa entry above).** Use **BEHAVIOR-1K /
OmniGibson**, matching the proposal's original stated preference — it turns
out to not be just a preference but a hard requirement given what's
actually implemented. Install via
`python install.py --new_env --behavior1k` from the OOPSIEVERSE repo,
targeting `/data/heng` (228GB free) rather than the root disk.

**Cost / risk accepted.** Isaac Sim / Omniverse-based installs are large
(commonly 50-100GB+) and historically more fragile (driver, licensing,
Omniverse-launcher issues) than a MuJoCo install. This is a real risk to the
14-day budget — installation problems could burn 1-2 days that were meant
for training/evaluation. Accepted deliberately over the alternative
(mechanical-only scope on RoboCasa), because the compositional,
multi-modality story is the paper's actual contribution.

**Confirmed with user:** 2026-08-29, user chose "Use BEHAVIOR-1K/OmniGibson
(Recommended)" after being shown this finding and three options (go
OmniGibson, scope down to mechanical-only on RoboCasa, or try OmniGibson
with a fallback time-box).

## 2026-08-29 — Data/artifact layout: heavy state on `/data/heng/cdp`, not in the git repo

**Decision.** Conda envs, the cloned OOPSIEVERSE codebase, downloaded demo
datasets, checkpoints, and per-episode logs all live under
`/data/heng/cdp/` (see `docs/PLAN.md` for the layout), referenced from the
git repo via a `CDP_DATA_ROOT` environment variable (default:
`/data/heng/cdp`), not hardcoded absolute paths in tracked code. Only small,
durable summary artifacts (final CSVs/figures/tables for the paper) are
committed to `results/` in git.

**Why.** Same root-disk constraint as above — the git repo lives on the
5.9GB-free root disk and must stay small. This also keeps the git history
free of large binary/log churn.

## 2026-09-01 — Proposal pivot: PID-Lagrangian vector-CMDP, joint-exposure
upper bound, fixed-weight ablation, Safety-Gymnasium cross-domain, real robot

**What changed.** User replaced `private/proposal.tex` with a substantially
revised version, retitled "CDP: Compositional Generalization in Safe Robot
Learning." The core research question changed from "does a per-modality
damage signal beat an aggregate scalar penalty" to a specific,
literature-positioned claim: **does a policy trained on hazard types
individually (single-exposure) retain a comparable safety advantage when
those hazards co-occur zero-shot**, and how does that compare to a policy
trained directly on the combination (joint-exposure, matching P3O/
multi-constraint safe-RL's setting). Concretely, for the manipulation domain:

1. **Reward penalty → constrained MDP with adaptive multipliers.** The old
   `apply_damage_penalty` used one hand-fixed `lambda=0.05` for the whole
   run (`scalar`: aggregate cost; `vector`: L1-sum of per-modality cost,
   still effectively scalar). The new proposal's Eq. (vector-cmdp) requires
   an independent PID-Lagrangian multiplier per hazard modality
   (Stooke/Achiam/Abbeel, ICML 2020), each enforcing its own budget `b_m`.
   Implemented in `src/cdp/lagrangian.py` (`PIDLagrangian`,
   `ScalarLagrangian`, `VectorLagrangian`) + `src/cdp/lagrangian_callback.py`
   (SB3 callback updating the multiplier once per PPO rollout from the
   per-modality damage `CDPTaskEnv.step()` now returns in `info`).
2. **Condition taxonomy renamed and expanded**: `task_only` (unchanged),
   `scalar_lagrangian` (was `scalar`, now PID-adaptive), `vector_lagrangian`
   (was `vector`, now independent per-modality PID multipliers instead of a
   single fixed weight on the L1 sum), `fixed_weight` (NEW — the RQ3
   ablation: same per-modality structured observation/reward as
   `vector_lagrangian`, but constant non-adaptive `lambda_m`, isolating
   "structured observation" from "adaptive constraint enforcement").
3. **Joint-exposure training regime (NEW, RQ2 upper bound)**: train
   `vector_lagrangian` directly on a composite (multi-hazard) task instead
   of zero-shot transferring to it. No new code path needed —
   `scripts/train_ppo.py --task_name` now accepts `COMPOSITE_EVAL_TASKS`
   values (`fill_bowl`, `food_in_microwave`) in addition to
   `TRAINING_TASKS`; exposure is purely a function of which task_name is
   passed (tagged `_joint` in the experiment_id when it's a composite task).
4. **`Delta_comp` redefined**: was `STCR_seen - STCR_composite` for one
   policy (zero-shot generalization gap); now
   `STCR_joint-exposure - STCR_single-exposure` on the SAME composite task
   (the residual gap to the joint-exposure upper bound, RQ2). The old
   quantity is kept as `zero_shot_gap` in `src/cdp/eval.py` since it's still
   meaningful (and is what our pre-pivot pick_egg-on-food_in_microwave
   headline result actually measured) — just no longer what `Delta_comp`
   means in the current proposal. New `scripts/compute_compositional_gap.py`
   computes the new `Delta_comp` from two `evaluate.py` run_dirs.
5. **New metric `V_m = Pr[J^{C_m}(pi) > b_m]`** (per-modality budget
   violation rate) added to `src/cdp/eval.py`.
6. **Safety-Gymnasium cross-domain validation (NEW, RQ4)**: replicate the
   single-/joint-exposure/zero-shot protocol on SafetyPointGoal2/
   SafetyCarGoal2 (Hazards+Vases) and SafetyCarButton1 (Gremlins+Buttons),
   3 seeds, sanity-checked against P3O's published joint-exposure numbers.
   Entirely new domain/codebase surface, no DamageSim dependency — safe to
   develop in parallel with manipulation-domain work since it doesn't touch
   Isaac Sim.
7. **Real-robot validation (RQ5, mechanical channel only, Franka Panda + ATI
   F/T sensor + Robotiq gripper)**: **out of scope for autonomous execution
   in this environment** — no physical robot is reachable from here. This
   phase's code/protocol can be prepared (a transfer/deployment script,
   force-torque logging format) but the actual hardware trials need the
   user (or someone with lab access) to run them; flagged explicitly rather
   than silently dropped or faked.

**Old checkpoints/runs (9 combos, `{task_only,scalar,vector} x {pick_egg,
add_firewood,pour_water}`, seed 0) are NOT deleted** — per-user instruction,
all data/files are kept. They don't match the new taxonomy exactly (`scalar`
used a fixed, non-adaptive lambda; `vector` used an L1-sum single weight,
not independent per-modality weights) — they're closest in spirit to an
early, cruder version of the `fixed_weight` ablation and are documented as
such going forward, not presented as `scalar_lagrangian`/`vector_lagrangian`
results. New runs under the new condition names are needed for the
headline RQ1/RQ2 comparisons.

**Why.** User-directed: "I have updated the proposal.tex for you. you need
to read it and modify the code as needed." Following [[feedback_contributions_log]]
and [[project_video_deliverables]] memory conventions for keeping this
documented as it happens, and the standing "never stop until all tasks are
done in the proposal" instruction from earlier in the session.
