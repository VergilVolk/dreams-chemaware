# ChemAware rule-kernel direct transfer Phase A result

Date: 2026-09-05  
Formal Slurm array: `2331449`  
Scope: inner fold 3 only; outer fold 4 untouched

## Decision

The eight jobs completed four epochs and passed the absolute shared-encoder
safety gates.  Phase A proves that full-candidate continuation can improve the
official shared DreaMS encoder, but it does **not** prove a ChemAware-specific
increment.  The ordinary continuation arm and four chemistry arms have the
same aggregate Recall@1: 90.9279%, +0.4666 percentage points over the matching
official baseline.  Therefore the +0.4666 pp must be attributed to the common
continuation objective, not to the mass/rule teacher.

## Locked result

The matching official baseline is 90.4614% Recall@1 on 1,929 held identities.

| Arm | Recall@1 | Delta vs official | Corrected / introduced | MRR | Formula-cluster 95% CI | Preservation |
|---|---:|---:|---:|---:|---:|---:|
| none | 90.9279% | +0.4666 pp | 12 / 3 | 0.949075 | [+0.1267, +0.8306] pp | 0.998164 |
| mass | 90.9279% | +0.4666 pp | 12 / 3 | 0.949049 | [+0.1267, +0.8306] pp | 0.998346 |
| rule_response | 90.8761% | +0.4147 pp | 11 / 3 | 0.948711 | [+0.0849, +0.7075] pp | 0.998257 |
| rule_mass | 90.9279% | +0.4666 pp | 12 / 3 | 0.949057 | [+0.1267, +0.8306] pp | 0.998245 |
| rule_mass_shifted | 90.8761% | +0.4147 pp | 11 / 3 | 0.948704 | [+0.1128, +0.8130] pp | 0.998430 |
| mass_error | 90.9279% | +0.4666 pp | 12 / 3 | 0.948920 | [+0.1267, +0.8306] pp | 0.998461 |
| rule_mass_error | 90.9279% | +0.4666 pp | 12 / 3 | 0.948884 | [+0.1267, +0.8306] pp | 0.998324 |
| rule_mass_shifted_error | 90.7724% | +0.3110 pp | 9 / 3 | 0.948056 | [+0.0826, +0.7828] pp | 0.998506 |

All arms selected step 4,096.  Mean gradient-clipping fractions were
0.790-0.802, so the optimizer spent roughly 80% of its steps above the global
norm threshold before clipping.  Error-only teacher routing did not add an
aggregate retrieval gain.

The formula intervals in the table compare each arm with official DreaMS.
They are not pairwise chemistry-vs-none intervals.  Strict pairwise claims
require the eight `inner_per_query.npz` files from the server result directory;
the downloaded OUT files are sufficient for the aggregate attribution verdict
but not for query-paired bootstrap comparisons.

## Why the teacher can have frozen headroom but fail to transfer

The pair-first frozen `rule_mass` embedding remains a valid teacher-headroom
result: +1.244 pp over official and +0.363 pp over mass-only on the same held
inner set.  Phase A shows that the original full-distribution KL did not encode
that increment into the 1,024-dimensional student.

The original teacher score was

`(official_score + beta * chemistry_score) / (1 + beta)`.

The denominator is required to interpret the concatenated vector as a cosine,
but it is irrelevant to candidate ranking.  Feeding that compressed score to
a fixed-temperature softmax changes the target entropy.  Consequently the KL
gradient mixes a ranking-irrelevant global scale target with the desired
chemical residual.

A real official-model two-query audit quantified the remaining transfer
bottleneck.  The non-chemical objective had parameter-gradient norm 13.378;
the legacy normalized KL had norm 2.656 (19.9% of base).  Removing the score
compression increased the rule gradient to 3.210 (24.0% of base).  A strictly
positive margin-transfer objective had norm 2.216 and increased its cosine
with the base objective from 0.325 to 0.571.  The chemical direction therefore
exists, but it is both weaker than the common task gradient and repeatedly
subjected to global clipping.

## Phase B: rank-equivalent paired margin transfer

Phase B keeps the official model, split, candidate lists, schedule, learning
rates, trainable final block/head, and preservation constraints fixed.  It
changes only the chemical-transfer mechanism:

1. `rank_equivalent_kl` uses `official + beta * chemistry`, eliminating the
   ranking-irrelevant score compression.
2. `positive_margin_transfer` computes the teacher's increment in the
   true-vs-hardest-negative margin relative to official DreaMS.  Only strictly
   positive increments receive chemical weight; harmful/no-op rows receive
   zero corrective weight.
3. The target increment is clipped at 0.05 and transferred at alpha 0.25 or
   0.50.
4. An embedding-gradient norm controller targets a chemical gradient equal to
   0.5 times the non-chemical gradient, capped at a 4x loss multiplier.  This
   preserves the chemical/base ratio even when the combined gradient is
   globally clipped.
5. Mass-only and mass-shifted rules remain matched controls.  The deployed
   model is still one shared clean-spectrum encoder; all teachers and
   candidate lists are discarded at inference.

The direct entrypoint is:

```bash
sbatch tasks/run_chemaware_rule_margin_transfer_arm.sbatch
```

The eight Phase B arms are `none`, unbalanced rank-equivalent rule KL,
gradient-balanced rule KL, balanced shifted-rule KL, rule PMT alpha 0.25,
rule PMT alpha 0.50, shifted-rule PMT alpha 0.50, and mass-only PMT alpha 0.50.
The new objective passed deterministic unit contracts and a real
official-checkpoint CPU forward/backward smoke.  In that smoke, one of two
queries had a strictly positive teacher advantage and the adaptive scale was
1.908; checkpoint writing and evaluation also completed.

Phase B has not yet produced a retrieval result.  Advancement requires a
strictly positive paired formula-cluster increment over `none` and over its
matched shifted/mass control; a positive comparison with official alone is not
sufficient.

## Phase B final aggregate result: Slurm 2331468

All eight completed OUT files were downloaded locally.  They are sufficient
to reject Phase B because no target arm has a positive aggregate increment
over the matched `none` arm.  The per-query files would permit paired interval
estimation, but cannot reverse a non-positive primary point estimate.

| Arm | Recall@1 | Delta vs official | Corrected / introduced | Formula-cluster 95% CI | Clip fraction |
|---|---:|---:|---:|---:|---:|
| none | **90.9279%** | **+0.4666 pp** | **12 / 3** | **[+0.1267, +0.8306] pp** | 0.8020 |
| rule rank-equivalent KL, unbalanced | 90.9279% | +0.4666 pp | 12 / 3 | [+0.1267, +0.8306] pp | 0.8135 |
| rule rank-equivalent KL, ratio 0.5 | 90.8761% | +0.4147 pp | 12 / 4 | [+0.0715, +0.7989] pp | 0.8337 |
| shifted-rule rank-equivalent KL, ratio 0.5 | 90.8761% | +0.4147 pp | 11 / 3 | [+0.1128, +0.8130] pp | 0.8179 |
| rule PMT alpha 0.25, ratio 0.5 | 90.6687% | +0.2074 pp | 8 / 4 | [-0.0207, +0.6129] pp | 0.9241 |
| rule PMT alpha 0.50, ratio 0.5 | 90.7206% | +0.2592 pp | 9 / 4 | [-0.0092, +0.6248] pp | 0.9448 |
| shifted-rule PMT alpha 0.50, ratio 0.5 | 90.8243% | +0.3629 pp | 10 / 3 | [+0.0647, +0.6777] pp | 0.9412 |
| mass PMT alpha 0.50, ratio 0.5 | 90.7724% | +0.3110 pp | 9 / 3 | [+0.0482, +0.6618] pp | 0.9363 |

The unbalanced rank-equivalent KL exactly ties `none` in Recall@1 and has
slightly lower MRR (0.949066 versus 0.949075), while gradient balancing lowers
Recall@1 by 0.0518 pp.  Both target-rule
PMT arms fail the absolute CI and clipping gates, and the shifted-rule PMT is
better than the correct-rule PMT.  Rule PMT alpha 0.25 and 0.50 trail `none` by
0.2592 and 0.2074 pp, respectively.  This rejects Phase B, gradient ratio 0.5,
and also
reveals a deeper loss-design problem: the first PMT implementation reduces
each query to a scalar true-vs-hardest-negative margin.  It does not preserve
which candidate the chemical rule identifies as the confounder.  Hence even a
shifted rule can behave as a generic hard-query curriculum.  A successor must
transfer candidate-specific target-control margin differences on the same
query/candidate pairs; merely lowering the gradient ratio would not repair
that semantic collapse.
