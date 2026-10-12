# Unified candidate energy: GNPS development implementation

Date: 2026-10-12

## Decision

The deployable algorithm is one continuous candidate-energy function, not a
vote, agreement gate, per-query router, or hard promotion cascade.  All
upstream assets remain frozen.  Only a small set of global evidence-family
coefficients is fitted on the consumed GNPS development collection.

The exact inputs are:

1. Noise V1 candidate score;
2. the confirmed ChemAware stage-1 encoder candidate score;
3. weighted spectral entropy (WSE), the anchor and strongest classical Top-1
   baseline;
4. the frozen P2b-on-Noise candidate score, used as the sole P2b family
   representative (its constituent channels are not added again);
5. candidate-keyed ChemAware V2 action evidence.  An acted query supplies a
   selected candidate and explicitly contrasted alternatives; an abstained
   query has zero availability, not a zero chemical score.

BioAware is absent because GNPS has no real event-level sample context.  It is
not imputed from catalogue membership or topology.

## Score definition

Every raw score is converted within query to centered, clipped rank-logit
evidence.  Let `e_WSE` be the anchor and `e_m` another available evidence
family.  The unified score is

`E(c|q) = e_WSE(c|q) + sum_m alpha_m * a_m(c,q) * [e_m(c|q)-e_WSE(c|q)]`.

The coefficients are global, non-negative, may be exactly zero, and have total
mass at most one.  Thus this is a continuous convex residual fusion around
WSE: correlated Noise/P2b/ChemAware scores cannot acquire unbounded repeated
weight, and no query-specific truth-blind feature is used to choose an expert.
The coefficients are estimated by listwise candidate cross-entropy plus a
fixed continuous WSE-anchor penalty.

This does not retrain Noise, ChemAware, P2b, or V2.  It learns only the evidence
composition.

## Leakage control that changes the scientific answer

GNPS is explicitly `development / consumed`; it is not an external test.
Identity and formula panels overlap heavily and are not counted as two
independent confirmations.

The fixed five-fold split hashes molecular formula.  For fold `k`, coefficient
fitting uses only identity-panel queries whose formulas are outside `k`.
Those coefficients predict both identity- and formula-panel queries in fold
`k`.  Therefore a formula-panel query cannot be predicted by a model fitted on
that query or on another query with its formula.  Formula-panel labels are
never supplied to optimization.  A contract test rewrites every formula-panel
label and requires predictions to remain bit-identical.

The completed ChemAware V2 action ledger is converted without loading the
panel truth array.  Candidate IDs and query IDs must align exactly before it is
accepted.

## Reported development results

For the full model, WSE, and each leave-one-family-out refit, both panels report:

- Recall at 1, 2, 3, 5, 10, and 20;
- MRR, mean rank, and median rank;
- macro-query AUROC/AUPRC and pooled candidate micro-AUROC/AUPRC;
- Top1-Top2 margins;
- the same complete block on the near-structure subset;
- corrected, introduced, and `corrected - 2*introduced` relative to WSE;
- formula-cluster bootstrap 95% interval for Recall@1 change.

The saved full model is trained on all identity-panel development formulas only
after cross-fitted reporting.  Enveda-180 is not read by this job.  The report
sets `eligible_for_enveda=true` only if both correlated GNPS strata have a
positive formula-cluster CI, positive risk-net, and no material micro-AUROC
loss versus WSE.  This is a single pre-external-test decision, not a claim that
the two GNPS strata are independent replications.

## Run

```bash
sbatch tasks/run_unified_candidate_energy_gnps.sbatch
```

The script requests one GPU and does not manually request CPU cores, matching
the current N26 scheduler policy.  It expects the already completed V2 transfer
at `gnps_noise_chemaware_v2_transfer_run_2358497`.  If an existing aligned
stage-1 embedding cache is supplied through
`GNPS_CHEMAWARE_STAGE1_EMBEDDINGS`, its paired JSON provenance report is also
required; otherwise the sealed stage-1 checkpoint is encoded once.

The output is
`data/validation/unified_candidate_energy_run_<jobid>/crossfit/report.json`.
Only after that report is interpreted and the single model is frozen should a
separate blind Enveda scoring job be written or submitted.
