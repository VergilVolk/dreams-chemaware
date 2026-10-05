# Noise native multi-difficulty Stage-3 result and root-cause decision

Date: 2026-09-27
Run: `noise_dreams_native_multidifficulty_stage3_fold_0_run_2345317`

## Decision

Retain the Stage-1 targeted checkpoint.  Stage-3 is not a candidate model and
must not receive more epochs.  Its result is materially negative relative to
Stage-1 and even negative relative to official DreaMS.

This result does **not** show that the registered actions carry no information.
Targeted is `+0.23455 pp` above the matched control, equivalent to 43 net held
Top-1 decisions.  That point estimate is not significant, has negative
lambda-2 risk-net, and is far smaller than the common damage induced by the
large hard-triplet continuation.  The correct conclusion is therefore:

> action semantics retained a small favorable direction, but the current
> dose, difficulty distribution and equal-pressure native hinge moved the
> shared encoder much farther in a harmful common direction.

## Frozen result

| Comparison | Recall@1 delta | CI | corrected / introduced | lambda-2 risk-net |
|---|---:|---:|---:|---:|
| targeted vs Stage-1 | `-0.64910 pp` | `[-1.03020, -0.27854]` | `244 / 363` | `-482` |
| targeted vs matched control | `+0.23455 pp` | `[-0.15118, +0.63177]` | `353 / 310` | `-267` |
| targeted vs official | `-0.15273 pp` | baseline promotion failed | — | negative |
| targeted vs mature E8 | `-0.42001 pp` | baseline promotion failed | — | negative |

Near targeted vs Stage-1 was `141 / 174`, lambda-2 risk-net `-207`.  Near
targeted vs control was `160 / 150`, lambda-2 risk-net `-140`.  Nearly every
non-saturated registered metric regressed relative to Stage-1, including
Recall@1--5, MRR, macro AUROC/AUPRC, margins, mean rank, micro candidate
AUROC/AUPRC and both MassSpecGym pooled pairwise panels.

The summary code pairs all 18,333 held queries by query metadata and computes
rank differences directly.  The field name `absolute_stage2_gain` is a stale
report-key label, but its value is calculated from the Stage-3 targeted report;
it does not change the conclusion.

## Quantitative decomposition

Targeted lost 119 net Top-1 decisions relative to Stage-1:

```text
244 corrected - 363 introduced = -119
```

Targeted gained 43 net decisions relative to control:

```text
353 corrected - 310 introduced = +43
```

Therefore the matched control was approximately 162 net decisions below
Stage-1, while targeted action content recovered approximately 43 of that
common loss.  The decomposition concerns net correctness, not a claim that the
individual transition sets are identical.

This rules out two simplistic explanations:

1. **“The actions are pure garbage.”**  If targeted content had no favorable
   signal at all, a positive 43-decision point estimate over the matched
   control would not be expected.  It is weak and unsafe, not absent.
2. **“The injector dropped the signal.”**  There is no custom injector in this
   run.  Native shared-encoder gradients reached the model.  The failure is in
   the supervision distribution and optimization objective, not an intermediate
   transmission gate.

## Located causes

### 1. Action multiplicity became optimizer dose again

The run has 10,146 action events for 3,021 action queries: `3.36` action
updates per action query on average.  It has only 3,477 dynamic clean events
plus 256 protection events.  Thus action events comprise `73.10%` of the
one-pass ledger, and action-rich queries receive up to four separate optimizer
opportunities while another query receives one.

The query-disjoint scheduler prevents two actions from the same query sharing a
batch, but it does not equalize total query dose.  This is not a hidden code
crash; it is a scientific scheduling error for a shared embedding.  Distinct
actions should be coverage choices across passes, not simultaneous dose
multipliers within one pass.

### 2. The curriculum is dominated by the most severe tier

`5,709 / 10,146 = 56.27%` of action events are hard.  Their initial median
geometry is:

```text
same-identity positive similarity = 0.4463
different-identity negative similarity = 0.6514
margin = -0.1793
```

There are up to two hard actions per query, but at most one easy and one medium
action.  All tiers are mixed from the first and only pass; there is no measured
easy-to-hard progression.  Extremely violated triplets therefore dominate the
event stream immediately.

### 3. Native hinge gives equal pressure to unequal defects

For every active triplet,

```text
loss = max(0, 0.1 - s_positive + s_negative)
d loss / d s_positive = -1
d loss / d s_negative = +1
```

The median hard event can be deficient mainly because the positive reference is
very far, but it receives the same scalar attraction/repulsion balance as an
easy near-boundary event.  Adding harder events increased the number and
directional diversity of gradients, not their semantic calibration.

### 4. A late continuation reused the original `5e-6` learning rate with fresh
Adam state

Stage-3 starts from the already optimized Stage-1 weights, reconstructs a new
native Adam optimizer and uses `5e-6`.  Stage-2 used `1e-6` and produced a small
safe absolute improvement, while the much larger Stage-3 bank used the original
from-official rate.  The two arms degrading together is consistent with an
oversized late-continuation update.  This does not by itself prove that `1e-6`
is optimal; it identifies update scale as a causal variable that must not be
confounded with a new loss.

### 5. Action and clean relations are separated rather than jointly calibrated

Every action event is an action-anchor/positive/negative triplet.  Its clean
query appears as a separate event and, by construction, cannot share an
optimizer batch with another event from the same query.  This avoids repeated
candidate references inside a batch but provides no explicit per-query
calibration of action pressure against clean preservation.  Shared parameters
are the only bridge.

This must not be “fixed” by restoring the previously harmful direct
clean-to-action attraction.  The correct next test measures action-gradient
alignment with the clean triplet gradient and then controls query-level dose.

### 6. Ten thousand events still cover only 3,021 queries

The action supervision touches `4.63%` of the 65,286 outer-train queries.
Increasing row count mostly increased within-query severity and multiplicity,
not broad formula/identity coverage.  Stronger shared generalization cannot be
assumed from repeated hard views of a narrow set.

## Immediate next step

Run the frozen zero-update audit:

```bash
sbatch tasks/run_noise_dreams_native_stage3_zero_update.sbatch
```

It measures, without changing weights:

- true identity/formula/boundary effective sample sizes;
- measured-positive multiplicity;
- targeted/control margin and active loss by tier and source;
- exact pre-optimizer targeted-control and targeted-clean parameter-gradient
  direction;
- separate action-anchor, positive and negative embedding-gradient norms.

The audit decides between two next hypotheses:

1. **Gradients are directionally useful but too large/repeated:** retain native
   hinge first, use one action opportunity per query, rotate tiers across
   passes, return to the already demonstrated late-continuation scale, and keep
   a matched control.  This is the minimal dose/scale repair.
2. **Hard-tier gradients conflict with clean or are dominated by positive
   deficit:** do not merely lower the learning rate.  First exclude or defer the
   conflicting tier, then test multi-relation/adaptive weighting on the exact
   same selected relations.

No multi-positive loss, Circle loss or listwise loss is authorized before this
audit.  Stage-3 has shown that increasing difficulty and count without
query-level calibration is actively harmful.
