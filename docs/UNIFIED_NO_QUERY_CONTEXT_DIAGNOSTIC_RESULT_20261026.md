# Unified integrator no-query-context diagnostic: final result

## Decision

The bounded `54 -> 23` feature diagnostic succeeded as a transfer diagnosis,
but it did not promote the nonlinear head. Removing the 31 fields that are
constant across all candidates of a query left Enveda OOF ranking essentially
unchanged and significantly improved the fixed Enveda-to-GNPS transfer of the
nonlinear model. The reduced nonlinear model nevertheless did not outperform
the simpler learned Additive model on GNPS.

The current transferable decision layer is therefore the **23-feature learned
Additive arm** from run `2413878`. This is a fitted candidate ranker over the
full retained evidence set, not an unweighted average. The reduced nonlinear
arm remains an informative diagnostic and a strong Enveda-domain model; it is
not the deployment winner.

Neither Enveda nor GNPS is an untouched external test. Both are consumed
development resources, so this result is not a final SOTA claim.

## Scientific contract

Run: `data/validation/enveda_no_query_context_diagnostic_run_2413878`

- Enveda identity: 25,000 queries, 668,458 candidate molecules.
- Enveda formula: 9,435 queries, 58,757 candidate molecules.
- GNPS identity: 10,995 queries, 87,518 candidate molecules.
- GNPS formula: 5,261 queries, 24,401 candidate molecules.
- Enveda training used five global formula folds.
- GNPS was a fixed evaluation only: `training_or_selection_on_gnps=false`.
- Query keys and formula groups never crossed Enveda folds.
- All eight Enveda input hashes, all eight GNPS input hashes, candidate axes,
  labels, fold assignments, expert scores, hyperparameters and seed are
  identical to run `2411772`.
- The only intended model change was deletion of 31 query-constant columns,
  followed by refitting both the Additive and nonlinear arms.

The 31 deleted columns were three query-size fields, seven non-V2 expert
top-gap fields and 21 expert-pair top-1 agreement fields. The 23 retained
columns comprise 14 expert raw/percentile scores, three candidate-varying
reliability summaries and six continuous/availability-aware ChemAware V2
fields. The masked V2 top-gap was retained because it varies by candidate.

No new leakage or axis drift was found. The checkpoint SHA256 is
`7f1309e2a89ef3640d9b6871a821df6866f2528e629547075d5305ab6707d717`.

## Enveda OOF: source-domain performance was preserved

| Panel and method | Recall@1 | MRR | Near Recall@1 | Macro AUROC | Micro AUROC | Micro AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| identity, WSE | 73.028% | 83.322% | 70.908% | 96.853% | 96.922% | 56.591% |
| identity, Additive | 77.592% | 86.584% | 75.872% | 97.522% | 97.921% | 66.279% |
| identity, reduced nonlinear | **78.772%** | **87.298%** | **77.170%** | **97.649%** | **98.069%** | **67.147%** |
| formula, WSE | 85.056% | 91.603% | 80.931% | 95.087% | 94.672% | 71.694% |
| formula, Additive | 86.084% | 92.260% | 82.175% | 95.441% | 96.138% | **80.324%** |
| formula, reduced nonlinear | **86.953%** | **92.737%** | **83.272%** | **95.767%** | **96.179%** | 79.279% |

Across 30,343 unique queries with overlap counted once, Recall@1 was 75.398%
for WSE, 79.312% for Additive and 80.450% for reduced nonlinear. Reduced
nonlinear exceeded WSE by +5.052 pp, with formula-cluster 95% CI
`[+4.648,+5.449] pp`.

Relative to the old 54-feature model, however, reduced nonlinear changed
Recall@1 by only `+0.104/+0.053 pp` on the identity/formula panels; both paired
intervals crossed zero. The diagnostic therefore preserved, rather than
materially improved, source-domain query ranking.

The deletion did materially improve pooled micro metrics, especially AUPRC.
That is consistent with better cross-query score comparability or calibration,
not with a large change in within-query identity ordering.

## Fixed GNPS transfer: most of the nonlinear transfer deficit was recovered

| Method | Identity R@1, n=10,995 | Formula R@1, n=5,261 |
|---|---:|---:|
| Official fine-tuned DreaMS | 85.357% | 86.809% |
| WSE | 87.367% | 88.272% |
| GNPS nested-OOF B | **88.595%** | **89.850%** |
| 23-feature Additive | 88.249% | 89.527% |
| 23-feature nonlinear | 88.167% | 89.413% |

Reduced nonlinear versus the old 54-feature nonlinear:

- identity: `+0.400 pp`, 135 corrected / 91 introduced, formula-cluster CI
  approximately `[+0.128,+0.672] pp`, near Recall@1 `+0.566 pp`;
- formula: `+0.570 pp`, 79/49, CI approximately
  `[+0.148,+0.993] pp`, near Recall@1 `+1.121 pp`.

The change recovered about 81--82% of the old nonlinear-to-Additive transfer
gap. It supports the conclusion that this query-constant context block was a
transfer liability in the fixed training protocol. It does **not** identify a
unique causal mechanism: reduced dimensionality, regularization and refitting
can also contribute.

Reduced nonlinear still did not beat Additive:

- identity: `-0.082 pp`, 87/96, CI `[-0.339,+0.163] pp`;
- formula: `-0.114 pp`, 48/54, CI `[-0.522,+0.300] pp`.

Additive also had the safer WSE error exchange:

| Panel | Additive - WSE | Corrected / introduced | `corrected - 2*introduced` |
|---|---:|---:|---:|
| identity | +0.882 pp | 184 / 87 | +10 |
| formula | +1.255 pp | 98 / 32 | +34 |

For reduced nonlinear the corresponding values were `+0.800 pp`, 194/106 and
`-18` on identity, and `+1.140 pp`, 99/39 and `+21` on formula. Additive also
had substantially higher GNPS micro AUPRC than reduced nonlinear
(`84.475/87.395%` versus `78.420/80.868%`).

## Final architecture and next action

The unified architecture remains:

1. representation evidence from Noise V1, ChemAware Stage-1, Phase-A and the
   Noise-ChemAware FullMSG encoder;
2. local spectral evidence from WSE, frozen P2b-on-Noise and Noise-MSG;
3. continuous candidate-specific ChemAware V2 evidence against fixed nulls;
4. the 23-feature learned Additive candidate decision layer;
5. optional BioAware sample-context correction only where real event-level
   context exists.

No further fusion-head search is justified by this diagnostic. The next
high-value computation is to apply the fixed 23-feature Additive model on the
already-fixed biological candidate axis and report which entity assignments
and independently supported responses change. A genuinely new external
benchmark is still required before any SOTA or external-generalization claim.

## Archival notes

The run is numerically valid, with two non-scoring metadata limitations:

- `modeling/report.json` uses `reliability_context=false` to mean that the
  query-constant block was excluded, while the checkpoint stores
  `reliability_context=true` because inference first constructs the full
  feature vector before applying the saved feature-selection policy. The
  executed 23-column model is unambiguous, but the field names should not be
  interpreted as identical semantics.
- `SHA256SUMS.txt` covers the checkpoint, OOF ledger and both reports, but not
  the two GNPS candidate-score ledgers. Their independently computed hashes
  are `74b6e039ac5b8ff982eeaa1ebbf7922c28665c1c079481e95dab7e1f5df7edff`
  (identity) and
  `b5593f4a896a7e019987adb75e9be59eb199b734a409fb1cd9426d340d027a76`
  (formula).
