# Full23 mega SOTA ladder and ten-million-pair benchmark status

Date: 2026-11-02

## Bottom line

Two distinct deliverables are now materialized locally:

1. an 18-scorer same-axis ladder with complete Top-k, rank,
   near-subset, ROC and precision--recall outputs; and
2. a score-blind seven-source GNPS2 benchmark preflight whose separate
   identity-verification axis contains 14,950,709 unique labelled spectrum
   pairs.

The completed ladder uses consumed GNPS development panels. Full23 v1.3.2 is
the Recall@1 leader on both panels, but this is development evidence rather
than unopened external SOTA confirmation. The 14.95-million-pair resource is
pre-exclusion scale capacity. Once the now-available official MassSpecGym
exclusion is applied with GNPS, MassBank and MSnLib exclusions, only 6,871
identity-disjoint and 2,218 formula-disjoint pairs remain. It is therefore a
large throughput/robustness stress axis, not an independent external accuracy
test. MoNA exclusion remains missing and can only shrink the strict axes.

## Denominators that must never be mixed

| Symbol | Meaning |
|---|---|
| `N_Q` | labelled query spectra |
| `N_I` | unique query IK14 identity clusters |
| `N_F` | unique query formula clusters |
| `N_E` | query-to-candidate-molecule edges |
| `N_P` | natural query-to-reference-spectrum comparisons in Top-k ranking |
| `N_pairs` | labelled pairs in the separate verification task |

Ten million `N_pairs` is not ten million queries and is not ten million
natural exact-mass Top-k comparisons.

## Completed 18-method common-domain ladder

MSBERT's official precursor domain removes 12 whole identity-panel queries and
8 whole formula-panel queries. No unsupported spectrum or incomplete
candidate block is imputed.

| Panel | N_Q | N_I | N_F | N_E | N_P | Near queries |
|---|---:|---:|---:|---:|---:|---:|
| Identity-disjoint common domain | 10,983 | 5,528 | 4,199 | 87,494 | 175,107 | 7,582 |
| Formula-disjoint common domain | 5,253 | 2,658 | 2,207 | 24,385 | 47,682 | 2,938 |

### Principal results

| Panel / method | R@1 | R@2 | R@3 | R@5 | R@10 | R@20 | MRR | Macro AUROC | Micro AUROC | Micro AP | Near R@1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Identity / Full23 v1.3.2 | **88.400%** | **94.009%** | **95.939%** | **97.505%** | **98.880%** | 99.754% | **0.9247** | **0.9497** | **0.9587** | **0.8445** | **83.870%** |
| Identity / WSE | 87.417% | 93.007% | 95.320% | 97.004% | 98.698% | 99.672% | 0.9168 | 0.9437 | 0.9423 | 0.7566 | 82.577% |
| Identity / official DreaMS | 85.387% | 92.315% | 95.010% | 97.014% | 98.762% | 99.763% | 0.9053 | 0.9377 | 0.9322 | 0.6817 | 79.979% |
| Identity / MSBERT official | 81.899% | 89.985% | 93.435% | 96.030% | 98.306% | 99.727% | 0.8812 | 0.9183 | 0.8947 | 0.5408 | 75.481% |
| Identity / SpecEmbedding official | 75.262% | 85.887% | 90.922% | 94.956% | 97.970% | 99.599% | 0.8374 | 0.8915 | 0.8563 | 0.4994 | 66.843% |
| Formula / Full23 v1.3.2 | **89.644%** | **95.755%** | **97.658%** | 98.972% | 99.695% | 99.962% | **0.9376** | **0.9470** | **0.9539** | **0.8744** | **82.369%** |
| Formula / WSE | 88.311% | 94.517% | 97.144% | 98.686% | 99.657% | 99.962% | 0.9281 | 0.9394 | 0.9285 | 0.7811 | 80.157% |
| Formula / official DreaMS | 86.865% | 94.384% | 97.068% | 98.667% | 99.581% | 99.981% | 0.9205 | 0.9325 | 0.9073 | 0.6802 | 78.080% |
| Formula / MSBERT official | 83.571% | 92.442% | 96.155% | 98.172% | 99.543% | **100.000%** | 0.8994 | 0.9124 | 0.8606 | 0.5569 | 73.315% |
| Formula / SpecEmbedding official | 79.288% | 89.987% | 94.384% | 97.544% | 99.315% | 99.943% | 0.8714 | 0.8851 | 0.8249 | 0.5407 | 66.814% |

The complete table contains weighted spectral entropy, official fine-tuned
DreaMS, Noise V1, greedy cosine, modified cosine, three P2b channels, frozen
P2b on official and Noise embeddings, MS2DeepScore 2.0, two Spec2Vec models,
raw entropy, denoising search, Full23 v1.3.2, SpecEmbedding and MSBERT.

Full23 leads the common-domain R@1 ladder. Relative to MSBERT its descriptive
R@1 differences are +6.501 and +6.073 percentage points; relative to
SpecEmbedding they are +13.138 and +10.356 percentage points. These public
checkpoints were not trained under an identical protocol, and GNPS is training
contaminated for several of them, so the result is a same-axis development
comparison rather than a contradiction of their published task results.

## Curves and reproducibility

Each panel directory contains:

- Recall@1/2/3/5/10/20 and Top-k curves;
- MRR, mean rank and median rank;
- near-subset Top-k and rank metrics;
- macro-query AUROC and average precision;
- micro-candidate AUROC and average precision;
- exact compressed ROC and PR coordinates;
- corrected/introduced ledgers and 10,000-resample cluster intervals;
- ladder, Top-k, ROC and PR figures in both PNG and PDF.

Curated output:
`deliverables/mega_sota_ladder/gnps_run18_msbert_common_20261102/`.

The detailed result record is
`docs/GNPS_PUBLIC_EMBEDDING_COMMON_DOMAIN_RESULT_20261102.md`.

## New GNPS2 multi-source benchmark

The strict seven-source inventory combines BMDMS-NP, PNNL lipid positive and
negative, MCE-DRUG, BERKELEY-LAB, and Birmingham UHPLC-MS positive and
negative. PNNL-negative has no usable identity truth and PNNL-positive has only
2/30,574 verifiable structures, so PNNL remains annotation-only. MCE-DRUG and
other quality-3 records are a separate Q3 stratum. The three newly downloaded
GNPS2 sources contain 39,230 spectra in total and are fixed by their source
hashes in `tasks/gnps2_standard_mega_sources_v2.json`.

### Natural 10-ppm Top-k capacity

| Stratum | N_Q | N_I | N_F | N_E | N_P |
|---|---:|---:|---:|---:|---:|
| Q1/Q2 primary | 19,703 | 1,117 | 477 | 62,617 | 1,127,864 |
| Q3 separate | 7,729 | 3,086 | 1,818 | 26,110 | 49,681 |
| Q1/Q2/Q3 expanded | 29,332 | 4,567 | 2,546 | 108,037 | 1,228,074 |

This honest Top-k task does not reach ten million comparisons. Even after the
three best unused GNPS2 additions, source aggregation cannot legitimately
manufacture a ten-million natural Top-k panel.

### Separate ten-million-pair verification axis

| Stratum | N_Q | N_I | N_F | Positive pairs | Negative pairs | N_pairs |
|---|---:|---:|---:|---:|---:|---:|
| Q1/Q2 primary | 48,272 | 2,619 | 1,914 | 346,544 | 8,092,592 | 8,439,136 |
| Q3 separate | 17,689 | 6,203 | 4,915 | 44,389 | 1,572,348 | 1,616,737 |
| Q1/Q2/Q3 expanded | 60,925 | 8,284 | 6,139 | 356,255 | 14,594,454 | **14,950,709** |

The materialized pair index is generated from the exact same score-blind
inventory and is checked for unique query/reference pairs, zero self-pairs,
disjoint query/reference axes, positive identity consistency, and negative
identity, polarity, adduct and 2-Da-window consistency.
It is 24,571,375 bytes with SHA-256
`dffce092d30890cf7ad6c4485b94077d6802f4b1e2763703274fb59c1e8e0462`.

The pre-exclusion axis supports ROC, PR, AUROC, average precision, sensitivity
at fixed false-positive rates and runtime scaling. Because the overwhelming
majority overlaps a consumed source once MassSpecGym is included, its accuracy
results must be labelled a known-overlap stress/development analysis. Top-k
remains reported only on the natural candidate graph above.

### Strict exclusion result

The official MassSpecGym split file is 582,897,368 bytes with SHA-256
`ccda2c4114d9b21413977df03376ca0fc097956a7fa304b861a3154a2b81e64f`.
After applying it together with the existing GNPS, MassBank and MSnLib
exclusions, the materialized score-blind axes contain:

| Axis | N_Q | Positive pairs | Negative pairs | N_pairs |
|---|---:|---:|---:|---:|
| Identity-disjoint | 461 | 2,173 | 4,698 | **6,871** |
| Formula-disjoint | 208 | 927 | 1,291 | **2,218** |

The independent integrity audit found no axis overlap, self-pair, duplicate,
label/type, identity, polarity/adduct or mass-window violation. The remaining
MoNA exclusion is unresolved, so these counts are not promoted as final
external denominators.

## Latest-method task boundaries

Reproducible experimental-spectrum identity methods belong in the same-axis
ladder. MS2DeepScore 2.0 and TransExION can be included only as auxiliary
structural-similarity scorers. MS2Query 1.5.4 must be evaluated as its native
whole-library system; its positive and negative assets are public. LSM-MS2 is
the same scientific identity-retrieval task but remains reported-only because
its mixed 1.8M library and checkpoint are not publicly reproducible. MSAlign
and MSAlign+ now have official MIT code and public pretrained archives, but
they, GLMR and MS2-SMILES AlignNet address spectrum-to-structure retrieval and
belong in a separate table rather than having their published Recall@k mixed
with experimental-spectrum identity Recall@k. The matching public MSAN release
remains provenance-pending and is not labelled official.

The machine-readable status is in `tasks/mega_sota_methods_v1.json`.

## Claim boundary and next opening

The current defensible claim is:

> Full23 v1.3.2 is benchmark-leading among reproducible same-axis methods
> actually evaluated on the consumed GNPS Gold/Silver 10-ppm development
> panels.

It is not yet defensible to claim field-wide or independently confirmed SOTA.
The GNPS2 14.95-million-pair resource remains
`GNPS2_STANDARD_MEGA_PREFLIGHT_BLOCKED_NOT_EXTERNAL`. The MassSpecGym-updated
strict build demonstrates that its apparent size was dominated by consumed
identity coverage: 6,871 identity-disjoint and 2,218 formula-disjoint pairs
remain before the still-missing MoNA exclusion. The 14.95-million axis may be
used for throughput and known-overlap robustness only; external transfer must
come from a genuinely post-freeze or prospective source.

The seven-source forecast and source hashes are bound to
`data/validation/gnps2_standard_mega_7source_preflight_v1_20261102/report.json`;
the earlier four-source v4 report remains a historical build and is not the
evidence source for the seven-source counts above. The strict exclusion
correction is recorded under
`data/validation/gnps2_standard_mega_7source_preflight_v3_massspecgym_20261102/`.
