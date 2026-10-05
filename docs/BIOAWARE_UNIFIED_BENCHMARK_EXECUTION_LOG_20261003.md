# BioAware unified benchmark execution log

Date: 2026-10-03
Scope: Track A local execution only. No B47 truth opened, no model fitted.

## What ran

1. **Asset readiness audit** — `tasks/audit_bioaware_benchmark_asset_readiness.py`
   → `data/validation/bioaware_benchmark_asset_readiness_20261003/report.json`.
   Findings: corrected graph fully local and schema-complete; GNPS identity
   main panel file missing; official DreaMS checkpoint and MassSpecGym v1.5
   resources server-only.

2. **Implementation gate replay** —
   `tasks/build_bioaware_tracka_corrected_replay_inputs.py`
   → `data/validation/bioaware_tracka_corrected_replay_inputs_20261003/`.
   Official molecule-max aggregation reproduced the frozen corrected-graph
   Recall@1 exactly (0.9287602 vs 0.928760); mean aggregation reproduced the
   documented degradation at −8.2864 pp.

3. **Metric-engine bug fix** — `exact_mcnemar_p` overflowed at large discordant
   totals (`2 ** total`); rewritten in log space with regression tests
   (small totals equal to the direct binomial sum; 3000/10000 returns 0.0
   instead of raising).

4. **First unified benchmark run (opened development)** —
   `data/validation/bioaware_tracka_corrected_benchmark_v1_20261003/`.
   83,619 queries, 6,220 formula clusters; official_dreams vs dreams_mean:
   −8.286 pp, 1,573/8,502, risk-net −15,431, McNemar p < 1e-300,
   formula-cluster CI [−9.60, −7.10] pp.

5. **Classical baselines on the sealed formula-disjoint panel** —
   `tasks/build_bioaware_tracka_gnps_classical_baselines.py` (modified cosine
   with greedy 0.01 Da matching; spectral-entropy similarity per Li et al.
   2021; molecule-max aggregation)
   → `data/validation/bioaware_tracka_gnps_classical_baselines_20261003/` and
   `data/validation/bioaware_tracka_gnps_classical_benchmark_v1_20261003/`.
   Result: spectral_entropy > modified_cosine by +0.228 pp (19/7,
   McNemar exact p = 0.0290, formula-cluster CI [+0.039, +0.423] pp) — an
   independent sealed-panel replication of the literature ordering, which
   also validates the unified pipeline end-to-end on external data.
   Absolute reference rows for future comparisons: modified cosine
   R@1 0.8688 (near 0.7782), spectral entropy R@1 0.8711 (near 0.7819).

6. **Recovery of the missing identity-disjoint main panel** —
   `tasks/replay_gnps_identity_disjoint_panel.py`
   → `data/validation/gnps_gold_silver_identity_panel_replay_20261003/`.
   The deterministic panel builder was rerun from the frozen manifest with
   the frozen seed. The control replay of the formula-disjoint panel matched
   the sealed artifact array-by-array, and all eight declared identity-panel
   statistics reproduced exactly (10,995 queries / 5,534 query identities /
   4,204 formulas / 22,881 candidate identities / 87,518 candidate molecules /
   175,171 candidate spectra / 19,726 positive pairs / 7,592 near queries).
   The sealed benchmark directory was not modified; the replay lives in its
   own directory and should be superseded by the server original when
   synchronized.

7. **Classical baselines on the recovered identity-disjoint panel** — inputs
   building executed after this log was written; the frozen evaluation output
   directory is
   `data/validation/bioaware_tracka_gnps_identity_classical_benchmark_v1_20261003/`.
   Result: modified cosine and spectral entropy are statistically equivalent
   on this panel (R@1 0.8560 vs 0.8559, 36/37, McNemar p = 1.0), in contrast
   to the significant entropy advantage on the formula-disjoint panel — the
   two sealed panels separate near-isomer difficulty exactly as designed.

8. **Server entry point** (mirrors the NOISE/ChemAware sbatch contract) —
   `tasks/run_bioaware_unified_benchmark_gnps_1gpu.sbatch`:
   required-file gates (sealed panels, `data/e1/official_embedding_slim.pt`,
   architecture checkpoint), `py_compile` plus three contract tests before
   compute, official DreaMS encoding via the frozen
   `encode_gnps_gold_silver_10ppm_checkpoint.py`, classical baselines and
   official molecule-max predictions via
   `tasks/build_bioaware_tracka_gnps_classical_baselines.py` and
   `tasks/build_bioaware_tracka_gnps_official_predictions.py` (unit-tested),
   then two sealed-external unified evaluations with `official_dreams` as the
   baseline method.  Outputs freeze under
   `data/validation/bioaware_unified_benchmark_gnps_run_<jobid>/` with the
   encoding report and input checksums.

## Adjudication (reviewer, end of day)

1. **REPEALED**: the hand-written classical scorers and their
   86.88%/87.11% results.  The `modified_cosine` lacked precursor-shifted
   matching and the `spectral_entropy` was not the pinned
   `ms_entropy==1.5.2` weighted implementation.  Frozen outputs are retained
   as retired artifacts only and must never be cited as baselines.
2. **SUSPENDED**: `run_bioaware_unified_benchmark_gnps_1gpu.sbatch`.  The
   formal run 2349091 already encodes official DreaMS and computes the
   standard classical scores on both sealed panels; re-encoding and
   re-scoring would duplicate compute.  (The reviewer fixed the
   pre-created-builder-directory defect and flagged that embeddings in
   `/tmp` are not failure-robust — both to be addressed if the sbatch is
   ever revived.)
3. **Source-cluster CI on single-source GNPS panels is a point estimate**
   and is never cross-source generalization evidence.

## Formal bundle unified evaluation (post-adjudication)

`tasks/build_bioaware_unified_inputs_from_score_bundle.py` converts the
run-2349091 `method_scores.npz` (10 methods, frozen per-reference scores,
aligned to each panel's `candidate_row`) through molecule-max aggregation
into the unified evaluator format.  Known-answer gate: the conversion
reproduced the reviewer table exactly (identity 85.36/87.37, formula
86.81/88.27 for official/WSE).  Frozen evaluations:
`data/validation/bioaware_unified_bundle_benchmark_{formula,identity}_v1_20261003/`.

Headline (vs official DreaMS, sealed external, formula-cluster CI):

| Method | identity ΔR@1 | formula ΔR@1 | note |
|---|---:|---:|---|
| weighted_spectral_entropy | +2.01 pp, p=1.3e-16 | +1.46 pp, p=3.8e-05 | strongest single scorer |
| p2b_noise_v1_frozen | +1.77 pp, p=2.7e-13 | +1.44 pp, p=2.9e-05 | frozen reranker |
| noise_v1 | +1.20 pp, p=8.8e-09 | +1.25 pp, p=3.1e-05 | |
| cosine_greedy / modified_cosine | −0.32/−0.33 pp, ns | +0.36/+0.34 pp, ns | statistically indistinguishable from DreaMS |

Unified-ruler observation not present in the original table: under the
project's own λ=2 risk criterion (`corrected − 2×introduced`), **no method is
net-positive against official DreaMS** (best: WSE identity −28), because every
method introduces substantial new errors while correcting others.  Positive
ΔR@1 with negative λ=2 risk-net is now visible per method and must accompany
any comparative claim.

Next: convert B30/B35 opened-development artifacts into the unified format as
Track C engineering baselines; B47 exact-event remains sealed Track C.

### Track C first row: B35 folded into the unified paired comparison

`tasks/build_bioaware_trackc_b35_unified_comparison.py` folds the frozen B35
per-query ledger (1,738 rotation rows → 1,631 physical queries, rank
consistency verified within each physical key) and runs the unified engine's
paired comparison.  Known-answer gate exact: ΔR@1 +2.8817 pp, 50/3,
McNemar p = 5.52e-12, formula-cluster CI [+0.0155, +0.0443] over 328 clusters
— every frozen B35 number reproduced.  Stratification confirms the protocol:
the negative intervention domain carries the entire +6.24 pp (753 queries)
while the 878 positive-abstention queries are exactly unchanged; all six
sources are positive (1.85–8.67 pp).  Frozen at
`data/validation/bioaware_trackc_b35_unified_20261003/` with the physical
query ledger.  Scope label: opened cross-fitted development; the B44 external
reversal stays in the limitation table.

### Track C unsealing path: preregistered B47 gate remediation

`docs/BIOAWARE_B47_TRACKC_GATE_REMEDIATION_PROTOCOL_20261003.md` freezes the
truth-blind repair before any result: operations O1 (per-candidate cap 25%,
weakest-advantage-first, deterministic ties), O2 (per-formula cap 25%), O3
(identity-quality screen at the candidate-pool 75th percentile of reference
multiplicity and catalogue degree), then the repaired authorization predicate
R1 (effective candidates ≥ 20, both largest fractions ≤ 25%), R2 (winner
medians ≤ pool medians), R3 (≥ 1,560 actions, the unchanged +3 pp arithmetic
gate), R4 (every study ≥ 25% share), R5 (null supremacy inherited from the
executed U3, never rerun).  Authorization failure keeps the U4 freeze at
`prospective-mechanistic`.

`tasks/audit_bioaware_b47_gate_remediation.py` implements it with U3
provenance verification (protocol-version mismatch fails closed) and is
covered by synthetic tests (diverse scenario authorizes; dominated scenario
refuses and cuts the hub; provenance corruption fails closed) —
`test_audit_bioaware_b47_gate_remediation.py` PASS.  It runs on the server
U3-v3 output directory; the R5 inheritance currently accepts several report
layouts and must be checked against the actual server `report.json` field
names on first execution.

## Post-hoc code review corrections (same day)

1. **MGF parsing unified.**  The classical-baseline builder originally carried
   a hand-written MGF reader.  It now reuses the frozen `iter_mgf` from the
   benchmark builder (the same reader the official encoder uses).  A
   verification rerun reproduced every frozen prediction score to within
   5e-15 (pure float rounding; rankings unchanged); the check directory was
   removed after comparison.
2. **sbatch/builder ownership contract verified.**  Both builders fail closed
   when their output directory already exists; the sbatch therefore creates
   only `$RUN_ROOT` and `$LOCAL_ROOT` and leaves all builder-owned
   directories to the builders themselves (commented in the script).
3. **Known minor inconsistencies (documented, not fixed in frozen outputs):**
   the local formula-panel run used the generic source label
   `gnps_gold_silver_10ppm` while server runs use per-panel labels; the
   corrected-graph manifest uses positional query indices rather than
   original dataset row identifiers; classical scoring is pure Python
   (~50 minutes for the identity panel locally); the evaluator leaves a
   `.partial.` staging directory behind when it fails mid-run.

## Correct interpretation

- All results so far are Track A baseline infrastructure: they establish the
  common ruler and its known-answer correctness, not any BioAware gain.
- The corrected-graph run is opened development and must not be used for
  public claims; the GNPS runs are sealed-external rows for classical methods
  only — official DreaMS rows on GNPS still require the server checkpoint.
- The entropy-over-cosine delta is a baseline-ordering sanity result, not a
  project contribution claim.

## Remaining gaps

1. Official DreaMS (and any new encoder) rows on the GNPS panels require the
   official checkpoint or server-side encoding.
2. MassSpecGym v1.5 official resources must be acquired and hashed before any
   public leaderboard claim.
3. Track C arms (frozen unary / catalogue-only / network / exact-event /
   structural nulls) remain pending the B47 gate remediation; B47 truth stays
   sealed.
