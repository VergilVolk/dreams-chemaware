# ChemAware listwise two-arm decisive experiment — pre-registration (2026-09-30)

Status: READY FOR FIRST RUN; NOT YET SUBMITTED. Parameters in Section 5 become
frozen at the first sbatch submission. This document supersedes the V16 union
submission plan.

## 1. Supersession and asset sealing

1. `run_GLM_chemaware_v16_union.sbatch` is RETIRED from submission. It will not
   be submitted as a main experiment. The file is kept unchanged because its
   run_2347164 outputs are the sealed chemical-relation asset used below.
2. The V16 corpus is SEALED as curriculum evidence:
   main arm 941 events / 626 queries / 10 direct current-winner corrections;
   contested arm 440 events / 314 queries / 5 winners; relation-level audit:
   1,202 admitted relations (602 uncontested + 600 subsignificant-recovered),
   482 contested relations, 5,232 rejected no-dominant. Integrity anchor is the
   SHA256 set of the run_2347164 report and ledgers.
3. Historical amendment ledger (disclosed, closed): launch gate 50→40
   (resource gate), mining caps 8/3→10/4 (single-source calibration). The
   1,000/500 coverage floor was NOT amended and never will be; the V16 route
   closed at 941 events as a mining-only result.

## 2. Verified structural diagnosis (code-anchored)

Each fact below was re-verified against source on 2026-09-30:

- `dreams/models/heads/heads.py` `ContrastiveHead.step` (L745–792): the loss
  consumes only anchor/positive/negative spectra; cosine at L779–781; fixed
  margin clamp at L789. Every chemical distinction collapses to the identity
  of the selected negative.
- `GLM_build_chemaware_v16_dynamic_triplets.py` (L492–543): main and contested
  relations are written to two disjoint pools.
- `run_GLM_chemaware_v16_union.sbatch` (L183): only the main pool reached the
  layered training pool; the contested pool had no downstream consumer, while
  contested counts participated in the launch narrative. Inconsistent by
  construction.
- Training optimizes a static pairwise triplet; evaluation scores candidates
  by `max_r cos(z_q, z_r)` over each candidate molecule's reference spectra
  (`chemaware_v2_triplet_eval_core.py` L127–130). Static winner, reference-count
  effects and molecule-max switching are untrained.
- The corrected candidate manifest
  (`chemaware_corrected_candidate_manifest_v1`) carries the full candidate-set
  structure needed to train the evaluated decision directly: per query, all
  same-adduct strict-10 ppm candidate molecules (positive identity first,
  median 3 molecules), each with its complete reference-spectrum list, all
  train-fold.
- Train/eval boundary: source ledgers are restricted at production to the
  frozen 4,032 formula-role-0/1 training queries
  (`build_chemaware_fragment_graph_candidate_source_ledger.py` L293–308);
  selection (role 2, 1,978 queries) and confirmation (role 3, 1,929 queries)
  are frozen evidence files with a replay-audited official baseline.

## 3. Hypotheses

- H1 (objective mismatch): part of the residual error comes from training a
  static pairwise triplet while the decision rule is a candidate-list
  molecule-max ranking. Training the listwise objective directly on the same
  queries recovers a measurable fraction. Measured by Arm 1 vs protected
  Phase-A on role 2 with role-3 confirmation.
- H2 (chemical increment): beyond H1, chemical evidence that marks specific
  false candidates adds independent signal. Measured by Arm 2 vs Arm 1
  (paired, same panels).
- H0 outcomes are admissible and pre-declared: if Arm 1 does not improve on
  Phase-A, the mismatch is not the bottleneck; if Arm 2 does not improve on
  Arm 1, the current chemical library carries no incremental training signal
  for the residual errors; both null ⇒ the spectrum-only observability ceiling
  argument is strengthened and the next structural move is candidate-side or
  local/multi-vector modelling, not more triplets.

## 4. Design

Both arms are identical in data, initialization, schedule and optimization;
they differ ONLY in the per-candidate margin vector.

- Data source: corrected candidate manifest, restricted to the frozen
  4,032-query training panel (verified against `train_triplet_evidence.npz`
  query registry; fail-closed on any intersection with role-2/role-3 panels).
- Group construction per training query: candidate molecule 0 = true identity,
  then negative molecules ranked by frozen Phase-A molecule-max score
  (deterministic: score desc, then molecule index asc), capped at
  `maximum_candidates_per_query` = 8; per molecule, reference spectra ranked
  by frozen Phase-A cosine to the query (desc, then row asc), capped at
  `maximum_reference_spectra_per_molecule` = 3. The query's own HDF5 row is
  excluded from its reference lists (training-only hygiene; the frozen
  evaluation panels are untouched); exclusion counts are reported.
- Objective (both arms), with embeddings z(·) from the shared live encoder
  (model output is L2-normalized; cosine = dot):
  `S_tau(q, c) = tau * ( logsumexp_{r in R(c)} cos(z(q), z(r)) / tau
  - log |R(c)| )`  (reference-count-corrected smooth maximum, tau = 0.1)
  `L(q) = softmax_cross_entropy( logits = (S_tau(q, ·) + margin(·)) / T,
  target = molecule 0 )`, T = 0.05.
- Arm 1 margins: identically zero.
- Arm 2 margins (false candidates only; molecule 0 always 0):
  - δ_main = 0.05 on every (query, false candidate) relation that the V16
    verdict admits to the main arm (unanimous or subsignificant-recovered);
  - δ_contested = 0.025 on contested relations (separate uncertainty layer,
    reported separately, never merged into main counts);
  - relations are recomputed with the exact frozen V16 verdict logic and the
    same source ledgers, then anchored: recomputed relation counts must equal
    the sealed run_2347164 audit (1,202 main / 482 contested) exactly, else
    the build fails closed.
- Validation slice: deterministic 1-in-10 of training queries
  (SHA-256 of query row, mod 10 == 0) for loss monitoring only; it never
  influences checkpoint selection.
- Initialization: protected Phase-A checkpoint
  `run_2345481/phasea_reproduction_gate/chemaware_phasea_max_boundary_step2000.ckpt`
  for BOTH arms (not the exploratory step-750). Architecture from
  `ssl_model_server.pt` via `load_base_model`, identical to the native chain.
- Optimization (both arms): Adam as configured by the native head, lr 2e-6,
  weight decay 0, batch = 1 query group per step, max 3,600 steps
  (approximately one pass over the non-validation training queries),
  checkpoint-mode fixed_steps with saves at 900/1800/2700/3600, seed 3407,
  fresh optimizer state (no restore) in both arms equally. Checkpoints are
  written in the same `e1_identity` structure the shared evaluator loads.
- Replay/mixing: none. Both arms train purely on the candidate-set listwise
  objective over the same 4,032-query panel. Forgetting is monitored by
  role-3 and GNPS transfer, which is their purpose.

## 5. Frozen parameter table

| parameter | value |
|---|---|
| tau (smooth-max) | 0.1 |
| T (candidate softmax) | 0.05 |
| δ_main | 0.05 |
| δ_contested | 0.025 |
| maximum_candidates_per_query | 8 |
| maximum_reference_spectra_per_molecule | 3 |
| batch (query groups) | 1; hard limit 32 encoded spectra per batch |
| steps | 3600 (saves 900/1800/2700/3600) |
| lr / weight decay | 2e-6 / 0 |
| seed | 3407 |
| init | Phase-A step-2000 (both arms) |
| optimizer | native Adam, fresh, both arms |

## 6. Arm-identity contracts (machine-checked)

The two training pools are ONE pool file containing both margin vectors;
Arm 1 and Arm 2 differ only by selecting the `arm1_margin` or `arm2_margin`
array. The builder, trainer and sbatch tests must verify: identical
`query_row`, `group_ptr`, `molecule_ref_ptr`, `ref_row`, `molecule_label`,
`val_query_mask`; `arm1_margin` all-zero; `arm2_margin` nonzero exactly on
relations counted in the anchored audit; identical trainer args except
`--arm`.

## 7. Leakage guards (fail-closed)

- Training panel must equal the frozen 4,032-query registry (count and set).
- Empty intersection with role-2 and role-3 query sets; formulas of training
  queries disjoint from both panels' formulas (formula-role contract).
- Molecule 0 label must be the unique positive in every group; margins
  restricted to molecules with label 0.
- No unknown/role-4 evidence file is read at any stage.
- Deterministic tie-breaking everywhere (score desc, index asc); the build is
  reproducible bit-for-bit from the same frozen inputs.

## 8. Evaluation and decision gates

Chain reuses the frozen machinery unchanged: encode + molecule-max ranking +
formula-cluster bootstrap paired summaries and official replay audit with
numerical-boundary exclusions. The shared step is selected by
`GLM_select_chemaware_listwise_shared_step.py` using Arm 1 only (maximize
corrected-2*introduced versus Phase-A, then Recall@1, MRR, earlier step).
Arm 2 is carried forward at the same step and never participates in selection,
preventing separate role-2 winner's-curse selection from masquerading as a
chemical increment.

- G1 (objective benefit, Arm 1 release): selected Arm-1 checkpoint vs
  `phaseA_2pp` on role 2: `delta_recall1 > 0` with formula-cluster bootstrap
  CI95 lower bound > 0, AND the same on role 3 (confirmation).
- G2 (chemical attribution): selected Arm 2 vs selected Arm 1, paired on the
  identical evaluated query sets: CI95 lower bound of `delta_recall1` > 0 on
  role 2 AND role 3. Only G2 licenses the phrase "chemical evidence added
  independent gain".
- G3 (protection): the formula-cluster CI95 lower bound of each arm versus
  Phase-A on role 3 must be at least −0.5 pp. An upper-bound check would be an
  invalid non-inferiority test.
- GNPS gold/silver 10 ppm transfer (identity- and formula-disjoint panels):
  evaluated for official, phaseA, arm1_selected, arm2_selected with paired
  CIs; descriptive evidence, not a release gate.
- Spectrum-free candidate baseline (MassSpecGym-in-the-Wild shortcut check):
  on role 2, role 3 and the GNPS panels, rank each query's candidates by
  reference count (desc; ties by IK14 lexicographic, truth-blind) and report
  recall@1 alongside the expected recall of random ranking. Context for
  interpreting absolute recall; not a gate.
- Honesty requirement: every report from this experiment carries the
  corrected-vs-introduced counts, both CI weightings, and the sealed-asset
  amendment ledger of Section 1.

## 9. Pre-registered predictions (honest, falsifiable)

- Arm 1 vs Phase-A: sign uncertain. The mechanism removes a real train/eval
  mismatch, but Phase-A's curriculum already approximated hard negatives.
  Expected |delta| within ±1.5 pp on role 2; the experiment's value is the
  attribution, not a promised gain.
- Arm 2 − Arm 1: expected small (chemical margins touch 1,202+482 relations
  concentrated on ~15% of training queries). A CI-exclusive-of-zero positive
  delta would be the first direct evidence that chemical evidence survives
  the compilation into gradients.
- No claim path exists from this experiment to +5 or +10 pp totals; the
  arithmetic of Section 2 in the 2026-09-30 verdict (≈57 additional net
  corrections needed on role 2 for +5 pp) stands.

## 10. No-amendment rule

No threshold, cap, gate, margin, temperature, step count or panel in this
document may be changed after the first sbatch submission of this
experiment. If a build-level failure requires a code fix, the fix must keep
Section 5 byte-identical and be declared in the run log. Negative results
are reported as negative results.

## 11. Deliverables

- `tasks/GLM_build_chemaware_listwise_two_arm_pools.py` (+ test)
- `tasks/GLM_train_chemaware_listwise.py` (+ test)
- `tasks/GLM_select_chemaware_listwise_shared_step.py` (+ test)
- `tasks/GLM_spectrum_free_candidate_baseline.py` (+ test)
- `tasks/run_GLM_chemaware_listwise_two_arm.sbatch` (+ static test)
- Run outputs under `data/validation/GLM_chemaware_listwise_two_arm/run_<id>/`
