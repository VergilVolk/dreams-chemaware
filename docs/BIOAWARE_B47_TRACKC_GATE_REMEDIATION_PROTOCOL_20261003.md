# BioAware B47 Track-C gate remediation protocol

Date: 2026-10-03
Status: preregistered repair design. No B47 truth, phenotype or outcome has
been opened; no threshold below is derived from any performance result.

## Situation being repaired

U3-v3 executed truth-blind on the server. Per the adjudicated record:

- real candidate-specific events exceeded the degree-rewired and
  seed-context-permuted structural nulls (the null-supremacy requirement
  passed);
- the frozen authorization gate still refused to open the one-time ranking
  evaluation because of **evidence concentration** and **identity-quality**
  failures;
- U4 froze the primary action with those gates failing, which constrains any
  later evaluation of that exact freeze to a `prospective-mechanistic` label.

`pass_to_frozen_event_ranking_evaluation=false` remains binding. This protocol
defines the only permitted repair path.

## Non-negotiable prohibitions

1. B47 annotation truth, phenotype and outcome columns stay sealed.
2. No threshold may be tuned against any performance number.
3. U3 nulls are inherited as executed; they may not be rerun with different
   seeds to obtain a friendlier draw.
4. The repair operates only on the frozen U3 truth-blind tables
   (`report.json` summary, `query_event_opportunities.csv.gz`,
   `candidate_event_features.csv.gz`).

## Preregistered repair operations (fixed order)

- **O1 per-candidate cap.** At most `ceil(0.25 * N)` intervention actions per
  `event_top_candidate`, where `N` is the current action count. Removed
  actions are those with the smallest `event_advantage`; ties break
  lexicographically by `query_id` (deterministic).
- **O2 per-formula cap.** The same rule per `event_top_formula`.
- **O3 identity-quality screen.** Drop an action when its winner's
  `event_top_reference_spectra` exceeds the 75th percentile of the
  reference-multiplicity distribution of the full frozen candidate pool, or
  its `event_top_catalogue_degree` exceeds the 75th percentile of the pool's
  catalogue-degree distribution. The pool percentiles are computed from
  `candidate_event_features.csv.gz` over every candidate, truth-blind.

## Repaired authorization predicate

Let `A'` be the action set after O1–O3 and `N'` its size.

- **R1 concentration.** Effective (inverse-HHI) intervention candidate
  identities ≥ 20, largest single-candidate fraction ≤ 0.25, and largest
  single-formula fraction ≤ 0.25, all computed on `A'`.
- **R2 identity quality.** On `A'`: median winner reference multiplicity ≤
  pool median, and median winner catalogue degree ≤ pool median.
- **R3 arithmetic headroom.** `N' ≥ ceil(0.03 * 51,976) = 1,560` (the +3 pp
  decision gate from the U3 amendment, unchanged).
- **R4 per-study materiality.** Every study carries ≥ 25% of `N'`.
- **R5 null supremacy.** Inherited unchanged from the executed U3 report
  (real > every structural null, p ≤ 0.05, ≥ 10% lift, both studies).

Authorization = R1 ∧ R2 ∧ R3 ∧ R4 ∧ R5. If any fails, the U4 freeze retains
its `prospective-mechanistic` status and no ranking truth is opened; the
repaired ledger is still written for audit.

## Execution

```bash
python -u tasks/audit_bioaware_b47_gate_remediation.py \
  --u3-dir <server U3-v3 output directory> \
  --output data/validation/bioaware_b47_gate_remediation_<date>_v1
```

The script verifies the U3 provenance chain (protocol version and frozen
recipe hashes), refuses to overwrite existing output, applies O1–O3 in the
fixed order, evaluates R1–R5, and writes `report.json` plus
`repaired_action_ledger.csv.gz`.

## Claim boundary

This protocol repairs the *authorization gate*, not the evidence itself. Even
after authorization, the one-time evaluation is a single confirmatory opening
whose outcome may be negative; a passing repair does not claim BioAware gain.
