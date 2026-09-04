# E5: direct real-positive-guided shared-embedding pilot

## Entry evidence

- Existing P/N action union: 922 official errors.
- Existing-peak positive consensus adds 223 unique errors.
- Recurrent missing-peak transfer adds another 112 unique errors.
- Expanded action union: 1,257 / 23,876 = 5.26 percentage points of
  supervision-space headroom.
- Best fixed missing-peak action (`recurrent_union_mix`, dose 0.50) corrects
  294 and introduces 27 on the full graph, with positive formula-cluster and
  direction-specificity intervals.

These are frozen-encoder action outcomes, not weight gains.  E5 tests whether
one shared trainable DreaMS encoder transfers them to clean-spectrum inference.

## Shared-encoder contract

- Query, augmented query, positive references and negative references use the
  same trainable DreaMS model.
- Last Transformer block plus official projection head are trainable.
- Backbone learning rate 2e-6; head learning rate 1e-5; four fixed epochs.
- Dropout remains off during gradient training.
- P2b, downstream scores and post-outcome per-query action selection are
  forbidden.
- Guided actions are limited to official baseline errors with a
  positive-deficit screen, split by formula before training.
- Inference takes one clean spectrum and emits one new embedding.

## Paired pilot arms

All arms preserve the previously validated N curriculum and safety stream.

1. `none`: N-only paired baseline.
2. `transfer`: N plus recurrent real-positive missing-peak transfer.
3. `intensity`: N plus real-positive intensity/prevalence consensus.
4. `both`: N plus both guided positive-noise families.

Each guided action recipe is global and frozen: no query is selected because
the action happened to correct it.  The action branch receives a ranking loss;
the clean branch receives ranking, preservation and a stop-gradient transfer
floor from the privileged action margin.

An arm advances only when it passes every official-baseline safety gate and
its paired formula-cluster confidence interval versus N-only is above zero,
with more incremental corrections than introductions.

This one-fold stage is development only.  A selected arm must subsequently be
run across all formula folds and seeds before sealed evaluation.

## E5 pilot result (job 2326280)

Only array outputs 1--3 were downloaded locally; the exact paired array-0
N-only artifact is still required for the formal per-query comparison.  The
historical run with the same N-only optimizer/data configuration is shown only
as a diagnostic reference, not substituted for the missing paired artifact.

| arm | delta Recall@1 | near delta | corrected / introduced | risk net | preservation |
| --- | ---: | ---: | ---: | ---: | ---: |
| historical exact-config N-only | +0.5403 pp | -- | 38 / 6 | 26 | 0.995245 |
| transfer, weight 1.0 | +0.5403 pp | +0.7200 pp | 53 / 21 | 11 | 0.977831 |
| intensity, weight 1.0 | +0.5909 pp | +0.7477 pp | 54 / 19 | 16 | 0.978909 |
| both, weight 1.0 | +0.4896 pp | +0.6369 pp | 49 / 20 | 9 | 0.978882 |

All three guided arms have positive formula-cluster intervals versus official
DreaMS and learn a nonzero action advantage.  They nevertheless fail the
pre-registered preservation gate.  Intensity is the best guided arm; the joint
arm is worse than either single family and therefore provides no evidence of
synergy at equal weight.

The central diagnostic is optimizer dose.  In the N-only reference, global
gradient clipping is active in roughly 89--93% of steps and retains about
36--46% of the gradient norm.  With guided weight 1.0 it is active in 100% of
steps and retains only about 9--19%.  Thus the fixed action is learnable, but
the guided stream is injected too strongly and changes the shared embedding
far beyond the validated N-only operating regime.  This is an engineering
dose failure, not evidence that the action direction is absent.

## E5-B locked dose/safety screen

E5-B preserves the exact N curriculum, formula fold, seed, learning rates,
trainable last Transformer block and head.  It excludes transfer and joint
arms for this decision and scans intensity weights 0.10, 0.25 and 0.50 against
an exact N-only paired control.  A fifth arm holds intensity at 0.25 and raises
the safety ratio from 1 to 2.  This separates guided dose from safety coverage
without changing both at once.

The E5-B screen is still a one-fold development experiment.  It cannot be
called a 3--5 pp model result.  The 5.26 pp number above is an outcome-aware
union headroom bound across complementary actions, not the effect of any one
fixed strategy and not a trained shared encoder.

## E5-B observed dose response (job 2326319)

Four guided outputs (array indices 1--4) were downloaded locally.  Array index
0, the exact paired N-only control, is not present locally, so the historical
exact-configuration N-only run remains diagnostic only until the new control
artifact is recovered.

| intensity weight | safety ratio | delta Recall@1 | corrected / introduced | risk net | preservation |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.10 | 1 | +0.6078 pp | 43 / 7 | 29 | 0.992303 |
| 0.25 | 1 | +0.5572 pp | 44 / 11 | 22 | 0.988427 |
| 0.50 | 1 | +0.4390 pp | 45 / 19 | 7 | 0.984089 |
| 0.25 | 2 | +0.5572 pp | 45 / 12 | 21 | 0.988385 |

The monotone dose response is decisive: larger guided weight increases churn,
reduces risk-adjusted gain and progressively damages embedding preservation.
Weight 0.10 is the only retained guided dose, but it still fails the 0.995
preservation gate.

The safety-ratio arm also exposed an implementation-semantics error in the
experimental design.  `safety_ratio` changes the number and diversity of
safety examples, while `safety_loss` is a mean.  It therefore does not multiply
the safety gradient.  The trainer now has a separate `safety_stream_weight`;
the old parameter is explicitly documented as coverage-only.  E5-C repeats an
exact N-only control and scans explicit safety weights 2 and 4 around guided
weight 0.10, plus a lower guided dose 0.05.  No E5-B arm advances directly to
multifold evaluation.

## E5-C final fixed-policy decision (job 2326384)

| policy | guided weight | safety weight | delta Recall@1 | corrected / introduced | risk net | preservation | gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| N-only | 0 | 1 | +0.5403 pp | 38 / 6 | 26 | 0.995245 | pass |
| N-only | 0 | 2 | +0.5234 pp | 36 / 5 | 26 | 0.996796 | pass |
| intensity | 0.10 | 2 | +0.5740 pp | 41 / 7 | 27 | 0.994661 | fail preservation |
| intensity | 0.10 | 4 | +0.5065 pp | 36 / 6 | 24 | 0.996669 | pass |
| intensity | 0.05 | 2 | +0.5234 pp | 37 / 6 | 25 | 0.995812 | pass |

Explicit safety weighting behaves as intended, unlike `safety_ratio`.  It
restores preservation, but the safely regularized guided arms do not improve
on their matched N-only controls.  The 0.10/2 arm is only +0.0507 pp above the
N-only/2 point estimate and misses preservation by 0.000339; 0.10/4 is worse
than N-only/2, while 0.05/2 has identical Recall@1 and lower risk net.

Decision: terminate global fixed-intensity dose scanning.  This does not
discard the positive-guided action matrix.  It shows that averaging the same
action across every positive-deficit error dilutes a sparse, query-dependent
signal.  The next shared-embedding experiment must mine actions per training
query from the formula-disjoint training partition, while keeping the held
formula fold completely outside action-outcome selection.  The action selector
is training-only; clean-spectrum inference still uses one shared DreaMS
encoder and no downstream reranker.
