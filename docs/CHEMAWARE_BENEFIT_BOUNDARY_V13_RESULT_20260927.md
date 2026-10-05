# ChemAware V13 benefit-boundary result

Date: 2026-09-27  
Run: `data/validation/chemaware_benefit_boundary_native/run_2345648`  
Decision: `CLOSED_NEGATIVE_INCREMENT / RETAIN_PHASEA_2PP`

## Validity

This run is technically and scientifically evaluable.  It initialized from
the protected Phase-A checkpoint with SHA-256
`a8428329ca1d12bfe735f3f8b848ed020a8db18d0a654d4f62cdbeb006e61135`.
It did not open formula roles 3 or 4 during construction or selection.

The corrected builder installed exactly 1,327 unique active benefit-boundary
events in place of the 1,327 old Phase-A error/chemical events.  The resulting
pool retained all 3,605 Phase-A safety events, all 1,024 official replay
events, the original total of 5,956 train events, and the unchanged 3,709-event
validation pool.  Identity, uniqueness, cardinality, native sampler, loss,
optimizer, and held-role gates all passed.

The evidence and Phase-A geometry had the following support:

| quantity | value |
|---|---:|
| benefit-proven queries / formulas | 347 / 230 |
| active Phase-A queries / formulas | 259 / 163 |
| still-wrong Phase-A queries | 112 |
| already-correct active-margin queries | 147 |
| selected active reference-pair events | 1,327 |
| all active reference-pair combinations | 13,128 |
| effective balanced per-query cap | 7 |

## Frozen role-2 result

The protected Phase-A comparator had Recall@1 `0.9174684`, MRR `0.9535325`,
Recall@3 `0.9913924`, micro-AUC `0.9460788`, macro-AUC `0.9668947`, and mean
positive margin `0.3700935` on 1,975 role-2 queries.

| step | delta Recall@1 vs Phase A | corrected / introduced | risk utility | Recall@3 gate | micro-AUC gate | formula-cluster CI (pp) | decision |
|---:|---:|---:|---:|---|---|---:|---|
| 250 | +0.2025 pp | 13 / 9 | -5 | pass | fail | [-0.2619, +0.6868] | reject |
| 500 | -0.1013 pp | 13 / 15 | -17 | pass | fail | [-0.6546, +0.4253] | reject |
| 750 | +0.2025 pp | 12 / 8 | -4 | fail | fail | [-0.2501, +0.7082] | reject |
| 1000 | +0.2532 pp | 15 / 10 | -5 | fail | pass | [-0.2582, +0.7397] | reject |

No checkpoint had positive `corrected - 2 * introduced`; every confidence
interval crossed zero.  Later checkpoints also reduced Recall@3.  Step 1,000
had the largest point Recall@1 increment, but it is not an admissible gain and
must not be reported as a `+2.3797 pp` released embedding.  Role 3 correctly
remained unopened.

## Scientific diagnosis

The benefit evidence is not empty: three of four checkpoints had a small
positive Recall@1 point change, and MRR and macro-AUC also rose at those
checkpoints.  The failure is therefore more specific than “the chemical rule
has no signal.”  The signal-to-intervention map is unsafe.

1. **Reference expansion is not independent chemical supervision.**  The
   1,327 events arise from only 259 active queries and 163 formulas.  Repeating
   up to seven active positive/negative reference pairs causes the native
   uniform event sampler to overweight a small set of boundaries.
2. **Most active queries were no longer errors.**  Only 112 active queries
   were still wrong under Phase A; 147 were already correct and contributed
   only reference-level margin violations.  Treating those 147 as correction
   targets spends gradient on already-solved molecule decisions and creates a
   direct route to introduced errors.
3. **Global cardinality matching did not preserve query-level dose.**  Exact
   replacement of 1,327 events preserved total pool size but redistributed the
   old Phase-A error budget onto a much smaller support.  It therefore changed
   the effective query sampler even though the DataLoader itself was unchanged.
4. **The observed geometry confirms this failure.**  Mean positive margin
   fell from `0.37009` at Phase A to `0.35561` at step 250 and `0.34062` at
   step 1,000.  The small top-1 movement was purchased by erosion of the broad
   positive-reference geometry rather than a clean correction of false-winner
   boundaries.

## Frozen lesson and place in the global programme

V13 must not be rerun or tuned.  Query-position-matched surgical substitution
is a useful *mechanism control* for separating harmful current-correct events
from useful current-error events, but its limited support cannot be assumed to
deliver the target five-point gain and it is not the main successor.

The cumulative Phase-A, pair-expansion, V11, V12, and V13 evidence instead
requires a new triplet bank whose supervision unit is a cross-formula chemical
relation rather than a query-specific reference pair:

- retain the complete Phase-A safety and official replay populations;
- preserve the native DreaMS dataset, random one-positive/one-negative draw,
  shared encoder, triplet loss, and Adam;
- admit a chemical negative only when the same ordering transfers across
  independent training formulas and beats a margin-matched nonchemical null;
- use distinct false candidates, not additional references of the same false
  candidate, to increase candidate-decision coverage;
- cap each relation, formula, identity, and query before writing the uniform
  native event pool;
- reproduce Phase A from official and introduce the frozen relation bank on
  the same optimizer trajectory, so fresh-Adam and warm-start drift cannot
  explain the comparison;
- treat a small surgical-substitution arm as an attribution control, not as
  the primary performance method.

The global bottleneck is transferable candidate-ordering coverage under a
shared spectrum-only encoder.  It is not raw triplet count, reference-pair
multiplicity, optimizer continuity, or residual-error dose in isolation.
