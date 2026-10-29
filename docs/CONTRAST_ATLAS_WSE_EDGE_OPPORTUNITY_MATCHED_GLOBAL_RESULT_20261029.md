# System-wide edge-opportunity-matched WSE Contrast Atlas control

## Scope

This is a full-panel development analysis: 15 sources, 13 parent projects, 51 study-level contrasts, 238,412 response rows, 54,861 study-entities and 2,286 directed cross-source contrast candidates. It is not an IBD case study.

The full unresolved graph contained 32,328 reciprocal WSE identity edges versus 918 strict-known edges. Thirty-two response-blind controls therefore selected exactly 918 unresolved reciprocal edges in each selection, with exact matching across 105 source-pair, physicochemical-compatibility and response-availability strata. Each graph was evaluated separately with the same frozen 256 whole-profile permutations (seed 20261027) and query-level max-T correction over all 2,286 candidate contrasts.

These are matching-order sensitivity selections, not independent replicates: 850/918 edges occurred in all 32 graphs, and pairwise Jaccard overlap was 0.912-0.957 (median 0.935). Ranges below are therefore sensitivity ranges, not confidence intervals.

## Result

- Source-independent HFD/IBD primary directions (10 directions from five canonical relations): unsigned best-face Top-5 ranged **1-3**, median **2.0** across 32 resamples.
- None of those 10 source-independent directions passed the two-face-Bonferroni query-level max-T criterion in any resample: range **0-0**, median **0.0**. The primary signal therefore does not survive edge-opportunity matching under the corrected unsigned test.
- Shared-parent ChemProp exact-drug positive controls (24 directions): unsigned best-face Top-5 ranged **10-12**, median **12.0**. Similar-face Top-5 ranged **7-8** and opposite-face Top-5 ranged **4-6**; because response face was not registered, these are relevance diagnostics rather than signed response recovery. Two-face-Bonferroni max-T calls ranged **8-9**, median **8.0**. These remain consumed within-project controls, not external replication.
- Open similar-response calibrated calls fell from 49/51 in the complete unresolved graph to **30-32** (median **31.0**); opposite calls fell from 46/51 to **25-31** (median **28.0**).
- ChemProp remained a response hub after edge-count matching: median Top hits into its three accessions were **35.0/51** for similar and **38.0/51** for opposite retrieval.

The control removes the 35-fold reciprocal-edge-count advantage, but does not create a formal unknown-information PASS. Residual evidence-quality imbalance remains: median standardized mean differences were -0.597 for conservative WSE and -0.455 for minimum matched fragments. Endpoint reuse also remains different: the known graph uses 591 unique endpoints, while matched unresolved controls use 1,473-1,488. The strict-known versus unresolved status defines the two pools, although no molecule name or specific structure identifier value enters the matching distance. More importantly, the registered set contains zero signed entity-level truths, unregistered pairs are not negatives, and only five canonical primary relations are source-independent.

## Scientific verdict

The large-scale Atlas computation is now real and system-wide. It shows that broad unresolved evidence can support structured response retrieval even after edge-count matching, but current assets do not establish that those additional answers are biologically correct or specific. The present bottleneck is not graph size; it is independent response truth and same-axis scoring by Official DreaMS, Noise V1 and Full23.

Accordingly:

- large-scale opportunity-matched retrieval: **executed**;
- source-independent biological retrieval: **not passed**;
- unresolved-information increment: **not passed**;
- in-house algorithm increment on the same complete axis: **not yet evaluated**;
- sealed external biological validation: **absent**.

All inputs are development/consumed. Response max-T significance is not mechanism truth, identity FDR, specificity or no-match accuracy.
