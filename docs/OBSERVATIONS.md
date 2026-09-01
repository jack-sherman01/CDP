# Observation Wrapper (Day 3)

Implementation: `src/cdp/obs_wrapper.py` (`DamageObservationWrapper`).

## Modes

Matches `private/proposal.tex` exactly:

- `task`: `[robot_state, object_poses_and_distances]`
- `scalar`: `task + [d_scalar_t, h_scalar_t]`
- `vector`: `task + [d_mech_t, d_therm_t, d_elec_t, h_mech_t, h_therm_t, h_elec_t]`

`electrical` is DamageSim's name for the proposal's "fluid" modality (see
`docs/DECISIONS.md`).

## Feature blocks

- **Robot state** (`s_t`): eef position (3), eef orientation quat (4),
  full joint positions (arm + gripper, N), full joint velocities (N).
  Gripper state is included implicitly — Franka's gripper joints are part
  of the robot's joint set, so no separate gripper feature block is needed.
- **Object poses + distances** (`p_t^i`): per task-relevant object — world
  position (3), world orientation quat (4), eef-relative position (3).
- **Damage/health**: see below.

## The health-tracking gap (important design decision)

DamageSim's core (`damagesim/core/damageable_mixin.py`,
`DamageableMixin.update_health`) keeps exactly **one aggregate scalar
health per link** — every evaluator's damage (mechanical + thermal +
electrical) is subtracted from the *same* number. There is no native
per-modality health signal, even though the proposal's math
(`h_{i,t} = h_{i,t-1} - d_{i,t}`, applied per-modality) assumes one.

Per-modality *damage* (`d^m_t`) IS native — `info["damage_info"][obj][link]
[modality]["damage"]` is reported separately per evaluator every step.

**Decision:** the wrapper reconstructs per-modality health itself, at the
same per-link granularity DamageSim uses internally, applying the exact
same rule the sim applies to the combined signal:
`link_health[modality] -= damage[modality]`, clipped to `[0, 100]`, reset
to 100 on episode reset. This is mathematically consistent with the
proposal's own definition — it's just tracking three independent scalars
per link instead of one — and requires no changes to DamageSim/OmniGibson
evaluator code (consistent with the proposal's "avoid introducing new
physical damage models" constraint, since this is bookkeeping on top of
existing per-modality damage values, not a new damage model).

Aggregation to environment level follows the Day 10-11 formulas in the
proposal:
- `d_vec_t[m] = sum over all tracked links of damage[m]` this step.
- `h_vec_t[m] = mean over N tracked objects of (min over that object's
  links of per-modality health)` — the per-object min mirrors
  `DamageableMixin.health`'s own min-over-links convention; the mean-over-
  objects matches the proposal's `h_t = (1/N) sum_i h_{i,t}`.
- `d_scalar_t = sum_m d_vec_t[m]` (proposal Eq. in "Proposed
  Contribution").
- `h_scalar_t = mean of DamageSim's own native per-link health`
  (`env.get_env_health()`) — deliberately the sim's pre-existing
  undifferentiated signal, not a re-derived one, since the scalar
  condition is supposed to represent "what you get without factorizing by
  modality."

## Normalization

Not yet implemented (per proposal: "normalize using statistics collected
from the training environments; clip damage values to a fixed range before
normalization"). Deferred to the PPO training script (Days 5-6), where
running statistics can be collected from actual rollouts rather than
guessed — collecting them here in the wrapper would mean guessing ranges
before any data exists.

## Verification

`tests/test_obs_wrapper.py` runs all three modes against a live
`add_firewood` env for a few steps and checks: shapes are constant across
steps, `task` ⊂ `scalar` ⊂ `vector` as prefixes, damage/health features are
finite and health stays in `[0, 100]`.
