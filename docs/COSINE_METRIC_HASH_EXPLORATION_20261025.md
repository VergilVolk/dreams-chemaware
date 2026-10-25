# Frozen-embedding metric and hashing exploration (2026-10-25)

## Question and boundary

This local experiment asks whether raw cosine is visibly underusing the frozen
Noise V1 and Noise-ChemAware FullMSG embeddings. It is a bounded side analysis;
it does not replace the main candidate-integration work.

No identity labels or biological responses were read while constructing the
scores. The fixed transformations were applied once to 23,482 embeddings and
20,374 precursor-compatible candidate pairs. Biological response results were
joined only afterwards on the frozen 38,628-query, 22,046-pair application
axis. The application is consumed development evidence and has no identity
truth.

## Fixed variants

1. exact raw cosine;
2. cosine after subtracting each study's embedding mean;
3. cosine after pooled per-dimension centering and variance scaling (a diagonal
   unsupervised Mahalanobis diagnostic);
4. symmetric cross-model cosine between Noise and FullMSG coordinates;
5. 256-bit sparse-random-hyperplane SimHash for FullMSG.

Raw cosine was reproduced to maximum absolute errors of `3.58e-7` (Noise) and
`4.17e-7` (FullMSG), confirming that the experiment uses the same score as the
production application run.

## Geometry result

There was no large hubness reduction from centering or diagonal whitening.
Top-1 candidate in-degree Gini remained approximately `0.40` for every exact
metric, and maximum in-degree remained between 27 and 34. Thus this application
does not show a strong geometric hubness failure that would make CSLS or a
similar local-density correction the immediate priority.

## Decision and response consequences

All numbers below compare a fixed variant to the corresponding raw cosine on
the same common-answer query axis.

| Variant | Exact Top-1 retained | Forward mutual net | Registered ranks improved / worsened | Registered Top-1 gained / lost |
|---|---:|---:|---:|---:|
| FullMSG study-centered | 95.23% | +7 | 14 / 8 | 0 / 1 |
| FullMSG diagonal-whitened | 96.17% | +9 | 10 / 8 | 0 / 0 |
| FullMSG SimHash-256 | 90.95% | +3 | 12 / 11 | 1 / 2 |
| Noise study-centered | 96.67% | +12 | 7 / 6 | 0 / 0 |
| Noise diagonal-whitened | 97.64% | +12 | 9 / 7 | 1 / 0 |
| Noise-FullMSG symmetric cross-space | 91.36% | -12 | 13 / 5 | 1 / 0 |

The single Noise-whitening Top-1 gain was an omeprazole/mix-3 repeated context.
The cross-space gain was one sertraline/mix-1 context. Neither is an independent
identity confirmation. FullMSG whitening preserved its five registered Top-1
contexts but did not add another one. Study centering removed one FullMSG
sertraline Top-1 context.

These results do not support replacing raw cosine with an unsupervised
centering or whitening rule in the main system. They do show that full-vector
geometry can change several hundred candidate decisions, so a **supervised**
low-capacity projection or pair head remains a meaningful mainline experiment.
It must be fitted only within the existing leakage-safe MassSpecGym development
protocol, with GNPS and application outcomes excluded from fitting.

## Hashing decision

FullMSG SimHash-256 achieved pair-score Pearson correlation `0.9491` with exact
cosine and mean absolute score error `0.0579`, while storing 256 bits rather
than 1,024 float32 values per embedding (16-fold vector-payload compression).
However, only 90.95% of exact-cosine Top-1 decisions were retained, and two
registered Top-1 response contexts were lost.

Therefore SimHash is not a replacement final score. Its scientifically valid
future role is first-stage approximate retrieval followed by exact embedding
or unified-model reranking. A large-library claim would need recall-at-shortlist,
latency, memory and final accuracy together; the present 22,046-pair axis is too
small for hashing to be a primary contribution.

## Mainline consequence

The side experiment stops here. Raw cosine remains the frozen inference
baseline. The mainline should train one low-capacity candidate-competition
model on the complete permitted MassSpecGym development data, comparing:

- scalar cosine/margin inputs;
- a shared learned projection followed by cosine;
- a capacity-matched full-embedding pair head.

The purpose is to test whether discriminative information exists beyond the
single cosine scalar, not to search more unsupervised normalizations.

## Reproducible artifacts

- Metric scorer: `tasks/score_crossstudy_identity_metric_variants.py`
- Truth-blind scores: `data/validation/msv000095331_identity_metric_variants_v1_20261025/`
- Frozen response join: `data/validation/msv000095331_entity_method_increment_metric_variants_v1_20261025/`

