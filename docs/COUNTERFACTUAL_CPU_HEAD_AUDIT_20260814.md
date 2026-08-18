# Counterfactual head CPU run: checkpoint audit

Checkpoint: `data/e1/counterfactual_formal/head/seed_20260813/best_counterfactual.pt`

## Bottom line

The head-only run did not collapse the official DreaMS space and produced a
small internal retrieval improvement.  The result is sufficient to continue
head-level validation, but not sufficient to unfreeze the DreaMS backbone or
claim that counterfactual peak supervision caused the gain.

## Full candidate retrieval (100 formula-isolated validation formulas)

| Metric | Official DreaMS | Trained head | Change |
|---|---:|---:|---:|
| Top-1 | 0.77121 | 0.77693 | +0.00572 |
| MRR | 0.87220 | 0.87538 | +0.00318 |
| Pairwise accuracy | 0.86694 | 0.87157 | +0.00463 |
| Hard-negative ROC-AUC | 0.74679 | 0.74823 | +0.00144 |
| Mean hard-negative margin | 0.16088 | 0.16037 | -0.00051 |

Formula-clustered bootstrap intervals:

- Top-1 change: `[0.00000, 0.01204]`.
- MRR change: `[0.00034, 0.00640]`.

The raw Top-1 comparison contains 10 fixes and 4 regressions.  After treating
absolute margins no larger than `1e-6` as numerical ties, it contains 6 fixes
and 2 regressions.  Thus the robust net gain is four queries, not six.

## Representation preservation

- The backbone in the checkpoint is bitwise identical to the official
  backbone.
- Mean cosine between trained and official embeddings: `0.99848`.
- First percentile: `0.99730`; minimum: `0.99328`.

There is no embedding collapse.  The trained head changes the local ordering
of a small number of near-boundary candidates while leaving the global space
almost unchanged.

## Why the logged 0.5747 is not the final result

The training evaluator defines a correct pair as `margin > 0`.  Thirteen of
395 validation pairs have an absolute margin no larger than `1e-6`.  Different
CPU matrix batch shapes can move these numerical ties across zero without
changing the mean margin.  This explains why the training log reports `0.5747`
while deterministic cached re-evaluation gives `0.5544` with the same mean
margin to approximately `1e-9`.

Using a `1e-6` tolerance, pairwise accuracy changes from `0.5392` to `0.5494`:
six robust fixes, two regressions, and a net correction of four pairs.  The
formula-bootstrap interval for the unthresholded accuracy change still touches
zero, so this is supportive rather than confirmatory evidence.

## What was and was not learned

Training loss decreases monotonically and the head improves some selected
hard rankings.  However, the counterfactual order metrics do not improve over
the saved epochs: identity-CF is `0.8846` at epoch 1 and `0.8814` at epoch 6;
confounder-CF stays at `0.8877`.  Therefore the current gain cannot yet be
attributed to chemical counterfactual supervision.  It may come from ordinary
identity triplet optimization or projection-head reweighting.

The residual failure pool also remains large.  In the selected validation
pairs, the robust changes correct four `fixed_oof` and two `residual_wrong`
cases, while most previously wrong cases remain wrong.

## Decision

Do not start `last1`/`last2` backbone unfreezing yet.  First run head-only
multi-seed and loss ablations with frozen hidden-state caching:

1. identity triplet + preservation;
2. add random peak masking;
3. add counterfactual peak loss;
4. compare robust margin, full-candidate MRR, fixes/regressions, and CF order.

Select checkpoints by full-candidate MRR plus a margin tolerance, not by raw
`margin > 0` accuracy.  Proceed to `last1` only if the counterfactual ablation
repeatedly improves over the identity-only control without increasing
regressions or damaging preservation.

## Claim boundary

This is one seed on internal formula-isolated validation.  Confirmation and
test sets remain untouched.  The result demonstrates feasibility of safe
head-level adaptation, not SOTA performance or validated chemical-mechanism
learning.
