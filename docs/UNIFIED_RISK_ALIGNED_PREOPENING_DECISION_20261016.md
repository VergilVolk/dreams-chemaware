# Unified algorithm pre-opening decision (2026-10-16)

## Verdict

Enveda-180 v3 component construction and score-blind scoring completed under
job `2372550`, but final label opening remains prohibited.  The external asset
is intact; the blocker is the decision function, not the benchmark.

The earlier candidate-energy freezer allowed a selected ablation to proceed
when its Recall@1 delta and formula-cluster interval were positive even though
`corrected - 2*introduced` was negative.  That contradicted the GNPS admission
contract.  The same workflow also lacked a frozen abstention threshold for the
unified score.  A post-opening selective curve is descriptive, not a
predeclared decision rule.

## Frozen external asset

Dataset role: Enveda-180 `sealed_external_test`, truth `frozen_unscored`.
Performance labels remain unopened.

- identity-disjoint: 25,000 queries;
- formula-disjoint: 9,435 queries;
- identity same-formula: 25,000 queries;
- formula same-formula: 6,833 queries;
- identity open-set: 25,000 queries, 17,755 match and 7,245 no-match;
- formula open-set: 9,435 queries, 7,184 match and 2,251 no-match;
- curated spectra: 99,112;
- exact consumed-identity overlap: 0;
- exact consumed-spectrum overlap: 0;
- unopened score-bundle SHA-256:
  `beb67dfcd5c3b3ed6428244f147040d11aaab07d5c3a10f1cc6ef87da1f4b97b`.

These counts are benchmark-construction results, not model performance.

## One confirmatory GNPS successor

Only one new development experiment is authorized before the abstention stage.
It is implemented in
`tasks/train_unified_candidate_energy_risk_confirmatory.py` and submitted by
`tasks/run_unified_candidate_energy_risk_confirmatory.sbatch`.

The evidence families are fixed before execution:

1. Noise V1;
2. weighted spectral entropy, the anchor;
3. frozen P2b-on-Noise, without its constituent channels;
4. sparse candidate-specific ChemAware V2 action evidence.

ChemAware Stage-1 remains a representation comparator because its removal was
the already selected development ablation.  BioAware remains unavailable
without event-level sample context.  RRF and raw P2b channels are forbidden.

The successor makes no module, seed, penalty, or threshold scan.  Its listwise
loss gives fixed weight 2 to GNPS development queries on which WSE is already
Top-1 correct, and weight 1 otherwise.  This aligns training with the frozen
utility `corrected - 2*introduced`.  Five molecular-formula folds generate OOF
predictions; formula-panel labels do not enter fitting.

Both GNPS panels are `development/consumed` and correlated.  They are two
robustness strata, not two external confirmations.

## Ranker gate

The fixed successor advances only if every item below passes on both GNPS
strata:

- Recall@1 delta versus WSE is positive;
- formula-cluster bootstrap 95% CI lower bound is strictly above zero;
- `corrected - 2*introduced` is strictly positive;
- MRR is nonnegative versus WSE;
- micro-candidate AUROC and AUPRC each decline by less than 0.001;
- near-structure Recall@1 delta and risk-net are nonnegative.

Failure ends this unified-ranker route.  It does not authorize another penalty
or module sweep.  The paper then uses the layered algorithm: WSE primary
ranking, Noise coordinate, P2b local evidence, ChemAware explanation, and
BioAware only where real event context exists.

## Abstention gate after a ranker pass

A ranker pass is not yet permission to open Enveda.  The next and only next
development stage must freeze one GNPS-derived margin threshold for the unified
model under a formula-disjoint calibration/evaluation design.  It must report
the complete risk-coverage curve and a fixed 5% target operating point on the
mixed match/no-match universe.  Enveda receives that threshold unchanged and
reports observed external risk and coverage; it is not automatically called
5% FDR under distribution shift.

Only after the ranker, threshold, primary panel, primary contrast, module
registry, candidate axes, and hashes are frozen may both predeclared Enveda
panels be opened once.  Failure after opening is a valid negative external
result and cannot trigger retuning on Enveda.

## Methodological basis

- Cross-validation criteria can themselves be overfit during model selection;
  therefore the successor is a single fixed procedure rather than another arm
  search: Cawley and Talbot, JMLR 2010,
  https://www.jmlr.org/papers/v11/cawley10a.html
- Abstention is evaluated as a risk-coverage trade-off rather than accuracy on
  answered cases alone: El-Yaniv and Wiener, JMLR 2010,
  https://jmlr.org/papers/v11/el-yaniv10a.html
- Risk calibration must be separated from external evaluation; conformal risk
  control provides the relevant finite-sample framework, subject to its
  exchangeability assumptions: Angelopoulos et al., arXiv:2208.02814,
  https://arxiv.org/abs/2208.02814
- WSE remains the primary classical baseline because spectral entropy has
  strong library-search evidence and must be defeated on the identical graph:
  Li et al., Nature Methods 2021,
  https://www.nature.com/articles/s41592-021-01331-z

## Separation from the biological method

Enveda contains reference standards, not biological samples.  It cannot
validate phenotype associations, event-level context, or the Metabolomic
Contrast Atlas.

The biological method continues independently on Pan-ReDU studies.  Its object
is a signed study-level response containing MS/MS evidence, effect direction,
effect magnitude, uncertainty, sample hierarchy, and provenance.  It must be
compared with spectrum-only retrieval, known-only analysis, MEMO/PALS-style
sample or family summaries, and shuffled-MS2 controls.  Its decisive result is
an independently supported cross-study biological relation that is recovered
at fixed selective risk because unresolved chemical entities are included.
Building another public kNN graph or reporting more annotations is not a pass.
