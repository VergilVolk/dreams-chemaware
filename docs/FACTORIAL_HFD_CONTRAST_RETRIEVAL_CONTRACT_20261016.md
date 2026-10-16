# Factorial HFD cross-study response retrieval contract

Date: 2026-10-16  
Status: development experiment frozen before raw abundance, MS/MS response scores, or cross-study ranks are opened

## Scientific question

Can a structure-agnostic signed MS/MS response measure recover the same genotype-by-diet biological interaction in a separate experiment, rather than merely recover similar serum chemistry or a generic high-fat-diet main effect?

This is a focused first realization of the Metabolomic Contrast Atlas idea: the query object is a **study-level directed biological response**, not a spectrum, compound name, disease label, or pathway.

## Frozen evidence base

- MTBLS135: 24 serum samples, wild type/Hi-MYC x control/HFD, 6 animals per cell, 3 LC-MS/MS panels.
- MTBLS3316: 24 serum samples, wild type/Hi-MYC x control/HFD, 6 animals per cell, 4 LC-MS/MS panels.
- MTBLS2085: 19 serum samples, WT/NQO1-Tg x standard/HFD, 4/5/5/5 animals, 4 LC-MS/MS panels.
- Total: 3 studies, 11 panels, 67 animals, 244 panel-sample mzML files, 3,586,894,955 bytes (3.340556 GiB).

The official MetaboLights metadata and remote byte envelopes are frozen in `data/validation/factorial_hfd_wave_manifest_v5_20261016`. This is a **development** experiment. MTBLS135 and MTBLS3316 are separate experiments using the same Hi-MYC model and research lineage; they are not represented as independent-laboratory replication.

## Structure-agnostic response object

Within each analytical panel, recurrent MS2-supported precursor/RT anchors are defined without genotype or diet labels. Their MS1 peak heights are frozen before the sealed 2 x 2 design is opened.

Before response retrieval, anchors are conservatively grouped into ion families only when they coelute, have a predeclared isotope/adduct mass difference, show Spearman correlation of at least 0.95 across at least eight joint detections, and have log2 intensity-ratio MAD at most 0.35. One representative is selected by phenotype-blind detection prevalence with representative-MS2 TIC as a tie-break. The family-representative universe is primary; the uncollapsed anchor universe is a sensitivity analysis. These families control redundant-ion voting but are not claimed as confirmed metabolite identities.

For every anchor, the model is

`log2 intensity = intercept + genotype + HFD + genotype:HFD`,

with feature-wise HC3 covariance. Five estimands are retained:

1. HFD minus control in the reference genotype;
2. HFD minus control in the modified genotype;
3. modified minus reference under control diet;
4. modified minus reference under HFD;
5. genotype-by-HFD interaction.

The bounded signed weight is `beta / sqrt(beta^2 + SE^2)`, multiplied by the square root of detection prevalence. Positive and negative weights share one absolute-response denominator, so the up/down response mass is not forced to 50/50. Frozen spectral embeddings provide the chemical kernel; structure names are not required.

Panels receive equal weight in a study-level response. This prevents a method with more anchors from dominating the biological query.

## Predeclared primary test

Candidate universe for each query is all five response types in each other study: 10 study-response candidates.

Primary reciprocal queries:

- MTBLS135 `genotype_by_hfd` -> MTBLS3316 `genotype_by_hfd`;
- MTBLS3316 `genotype_by_hfd` -> MTBLS135 `genotype_by_hfd`.

Both directions must satisfy all of the following:

1. the predeclared counterpart is rank 1 by signed response score among all 10 candidates;
2. its score exceeds every wrong response type in the counterpart study;
3. its score exceeds the maximum null score under within-panel permutation of anchor response weights, with family-wise one-sided P < 0.05 across the two reciprocal tests;
4. rank 1 is retained in at least 80% of single-panel-deletion analyses across query and target studies.

Failure of either reciprocal query is a primary failure. A favorable result in only one direction is exploratory.

## Required baselines and falsifications

- **Chemistry-only:** effect-independent spectral composition. It cannot establish response retrieval.
- **Magnitude-only:** preserves response strength but removes direction. It tests whether direction is informative.
- **Mass/RT-only:** same frozen anchors and effects without learned fragment evidence.
- **Known-only:** only after a frozen, complete identity partition exists.
- **Unresolved-only and unresolved-MS2 shuffle:** only after the same identity partition; absence of an annotation is never automatically labelled unknown.

The method succeeds scientifically only if the signed response supplies information beyond chemistry-only and generic HFD similarity. Higher similarity caused solely by serum composition, instrument method, or shared research lineage is not a biological discovery.

## Role of MTBLS2085

NQO1-by-HFD is an out-of-family challenge. It is not frozen as a positive, a negative, or an opposite-direction truth. It may become a mechanistic result only with independent evidence. Until then it contributes wrong-response candidates and tests whether the method indiscriminately calls every HFD interaction similar.

## Claim boundary

A primary pass would establish a replicated-model proof of principle for retrieving a directed, structure-agnostic biological contrast across separate experiments. It would not yet establish cross-laboratory universality, a disease mechanism, or an unknown-metabolite discovery. Those require a larger response atlas, a frozen identity partition, and an independent source-family validation.
