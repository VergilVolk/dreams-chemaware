# Noise native Stage-3 calibrated repair

Date: 2026-09-27

## Decision

This is not a new Noise method and does not rebuild the Stage-3 triplets.  It
combines the two empirically supported parts of the existing route:

1. Stage-3's frozen multi-difficulty same-identity positive, closest
   different-identity negative, targeted action and matched-control relations;
2. Stage-2's safe late-continuation scale and one action opportunity per query.

The failed Stage-3 run remains immutable evidence.  Its zero-update audit is
the only relation-selection input.  Held ranks and correctness transitions are
forbidden during repair construction.

## What is unchanged

- Stage-1 targeted champion warm start;
- all original Stage-3 spectrum tensors and positive/negative row identities;
- easy, medium and hard positive definitions;
- registered A4, E10B, E11, E12B, N, P and V4 source families;
- matched targeted/control arms;
- native DreaMS `ContrastiveHead`, 1024-dimensional linear head,
  `SpectrumPreprocessor`, cosine triplet-margin loss and Adam optimizer;
- batch size 4, FP32, full shared encoder, margin 0.1 and one epoch;
- frozen formula split and full held evaluator.

## The two repairs

### 1. Evidence-calibrated relation choice

For every already-frozen Stage-3 relation, the zero-update audit contains the
targeted and matched-control geometry on the exact same positive and negative.
A relation can train only when both conditions hold before optimization:

```text
targeted positive similarity - control positive similarity > 1e-4
targeted triplet margin      - control triplet margin      > 1e-4
targeted triplet margin      - clean same-boundary margin  > 1e-4
```

The third condition restores the successful Stage-2 invariant on the exact
Stage-3 positive/negative rows.  The builder encodes only the required measured
clean/positive/negative rows under the exact Stage-1 warm start; it does not
re-mine a relation.  The `1e-4` floor is fixed above the observed numerical
replay error and is not tuned against held performance.

Among qualifying relations, the largest margin advantage is retained for each
query.  Exact query/positive/negative aliases are collapsed first.  No held
outcome, held rank, correction label or post-training result enters selection.

This rule does not ban hard positives.  A hard relation is retained whenever
it contains targeted-specific same-identity evidence.  Easy, medium and hard
relations and every registered source must remain represented in the final
bank or construction stops before training.

### 2. Query-level dose and late-continuation scale

The failed run exposed 10,146 actions from only 3,021 queries, so action-rich
queries received up to four optimizer opportunities.  The repair exposes:

```text
one frozen action relation per selected query
one original clean triplet per selected query
256 original action-free protection triplets
```

There is no synthetic query oversampling.  The learning rate is `1e-6`, the
same safe late-continuation scale used by Stage-2, rather than Stage-3's
restored from-base `5e-6`.

## Files

- selector: `tasks/repair_noise_dreams_native_multidifficulty_stage3.py`
- native trainer with registered repair curriculum:
  `tasks/train_noise_dreams_native_residual_stage2.py`
- regression test: `tasks/test_noise_dreams_native_stage3_calibrated_repair.py`
- two-GPU entrypoint:
  `tasks/run_noise_dreams_native_stage3_calibrated_repair_2gpu.sbatch`

## Promotion boundary

The repaired checkpoint is copied to the run's `checkpoint/` directory only
if the existing full evaluator establishes all registered Stage-1 and matched-
control improvements, formula-cluster intervals, risk-net guards, near guards
and official/mature-E8 baseline requirements.  Merely completing training is
not promotion.
