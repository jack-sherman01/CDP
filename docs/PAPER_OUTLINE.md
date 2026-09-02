# Paper Outline & Reproducibility Checklist

Rewritten 2026-09-01 for the revised `private/proposal.tex`
("CDP: Compositional Generalization in Safe Robot Learning") — see
`docs/DECISIONS.md`'s 2026-09-01 entry for the full pivot rationale. Section
structure follows the proposal's own framing (RQ1-5) so the paper and the
proposal stay traceable to each other.

## Sections

1. **Abstract** — restate RQ1 (does single-exposure + per-hazard-type
   constraints generalize zero-shot to unseen combinations better than a
   scalar constraint) + the one-line finding once known, across both
   domains.
2. **Introduction** — motivate via the proposal's kitchen-robot example
   (fragile-object handling + hot-object avoidance trained separately,
   deployed where both co-occur); state explicitly how this differs from
   P3O's within-distribution multi-constraint setting.
3. **Related work** — DamageSim/OopsieBench, Safety-Gymnasium, P3O and
   other multi-constraint safe RL (RCPO, CUP, FOCOPS), PID-Lagrangian
   (Stooke/Achiam/Abbeel 2020), CMDP foundations (Altman 1999).
4. **Method**
   - Single-/joint-exposure/zero-shot-combination protocol
     (`private/proposal.tex` Sec. "Compositional Generalization Protocol").
   - Vector-CMDP formulation + PID-Lagrangian mechanism
     (`src/cdp/lagrangian.py`, shared verbatim across both domains — cite
     as evidence for RQ4's "identical mechanism, unrelated domains" claim).
   - Policy representations: task_only, scalar_lagrangian,
     vector_lagrangian (single- and joint-exposure), fixed_weight ablation.
5. **Experimental setup**
   - Manipulation domain: BEHAVIOR-1K/OmniGibson backend + why
     (`docs/DECISIONS.md`), tasks (`docs/TASKS.md`), PPO hyperparameters
     (`scripts/train_ppo.py`).
   - Navigation domain: Safety-Gymnasium, custom single-exposure hazard-
     isolated task variants (`src/cdp_nav/custom_tasks.py`,
     `docs/TASKS.md`'s Safety-Gymnasium section), same PPO hyperparameters
     (`scripts_nav/train_ppo_nav.py`).
   - Evaluation metrics: TCR, STCR, SafetyGap, `Delta_comp` (RQ2's
     joint-vs-single-exposure gap, redefined post-pivot — see
     `src/cdp/eval.py`'s module docstring), `V_m` (per-modality violation
     rate), `zero_shot_gap` (the pre-pivot seen-vs-composite quantity, kept
     as a secondary statistic).
6. **Results**
   - RQ1: `STCR_vector_lagrangian - STCR_scalar_lagrangian` on the held-out
     composite task, both domains (`scripts/analyze.py`,
     `scripts_nav/analyze_nav.py`).
   - RQ2: `Delta_comp` (joint-exposure upper bound vs. single-exposure
     zero-shot), both domains (`scripts/compute_compositional_gap.py`,
     `scripts_nav/compute_compositional_gap_nav.py`).
   - RQ3: `fixed_weight` vs. `vector_lagrangian` (isolates adaptive
     constraint enforcement from structured observation alone).
   - RQ4: side-by-side manipulation vs. navigation results — does the
     RQ1/RQ2 pattern replicate across unrelated hazard taxonomies?
     Navigation joint-exposure sanity-checked against P3O's published
     multi-constraint numbers before being trusted as an upper bound.
   - RQ5: real-robot mechanical-channel comparison (task_only vs.
     scalar_lagrangian vs. vector_lagrangian-single, contact force/torque)
     — **pending hardware trials, see Limitations**.
   - Corruption-signal robustness, modality-dropout ablation, budget-
     sensitivity ablation (manipulation domain; extend to navigation if
     time allows).
7. **Discussion** — where the hypothesis held / didn't, tied to specific
   per-episode failure modes (cross-modality compensation instances, not
   just aggregate STCR numbers).
8. **Limitations** — explicitly: low-dim state only (no vision), small
   task set per domain, seed count per comparison (flag any still-single-
   seed cell plainly), BEHAVIOR-1K-over-RoboCasa substitution rationale,
   the Button-domain's dropped native Hazards channel (`docs/TASKS.md`),
   and — most importantly — **RQ5's real-robot trials were not executed
   as part of this automated research pipeline** (no physical hardware
   reachable from the development environment); state whether they were
   completed separately before submission, and if not, scope the paper to
   RQ1-4 (simulation only) rather than imply hardware results that don't
   exist.
9. **Conclusion**
10. **Reproducibility checklist** — see below.

## Reproducibility checklist

To fill in once experiments are complete:

- [ ] Every reported number traces to a specific `run_dir` under
      `/data/heng/cdp/runs/<experiment_id>/` (manipulation) or
      `/data/heng/cdp/runs_nav/<experiment_id>/` (navigation), keyed
      `{condition}_{task}_{seed}[_joint]` (never overwritten).
- [ ] PID-Lagrangian hyperparameters (`K_P`, `K_I`, `K_D`, budget `b_m`)
      used for each reported run recorded — default
      `src/cdp/lagrangian.py::PIDLagrangianConfig` values unless a sweep
      cell, in which case the specific value is in the `experiment_id`
      suffix (e.g. `_budget15.0`).
- [ ] `lambda_final.json` / `lagrangian_history.json` (written per
      checkpoint dir by `train_ppo.py`/`train_ppo_nav.py`) available for
      every `*_lagrangian` run reported, so the multiplier trajectory is
      auditable, not just the final STCR number.
- [ ] Software/hardware versions recorded: OmniGibson/Isaac Sim version,
      `oopsieverse_b1k` conda env (manipulation), `safety-gymnasium==1.0.0`
      + `cdp_nav` conda env (navigation), Python 3.10, 2x RTX A5000, driver
      535.230.02 (see `docs/DAILY_LOG.md` Day 1).
- [ ] `results/main_comparison_table.csv` / `results_nav/
      main_comparison_table_nav.csv` (from `scripts/analyze.py` /
      `scripts_nav/analyze_nav.py`) match the numbers quoted in the paper.
- [ ] Figures regenerate from the analyze scripts with no manual editing.
- [ ] Seeds used per condition/task listed explicitly (single-seed runs
      flagged as such — no implied multi-seed statistics where there's
      only one seed).
- [ ] Scope/substitution disclosures present in the paper itself, not just
      in `docs/DECISIONS.md`: BEHAVIOR-1K over RoboCasa; the Button-domain
      dropping the native Hazards channel to match the proposal's stated
      2-hazard combo; RQ5 hardware-trial status (executed separately or
      explicitly out of scope for this submission).
- [ ] `scripts/run_all_manip_training.sh` / `scripts_nav/
      run_all_nav_training.sh` + `scripts_nav/run_all_nav_eval.sh`
      reproduce every checkpoint/eval run referenced in the paper from a
      clean checkout (modulo simulator install, which is documented
      separately in `docs/DECISIONS.md`/`docs/DAILY_LOG.md` Day 1).
