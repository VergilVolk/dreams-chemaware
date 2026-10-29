# Parent-project-held-out system-wide Contrast Atlas result

## Scope

This is a full 15-source, 13 operational-dependence-group, 51-contrast analysis. Every query excludes all contrasts from its own parent-project group before ranking and query-level max-T. The candidate universe therefore contains 1,944 directed contrasts rather than 2,286; all 342 removed directions are between the three ChemProp accessions.

The 32 graphs are matching-order sensitivity selections and are highly overlapping. Their ranges are not confidence intervals. All inputs are development/consumed.

## Statistically selected cross-project hypothesis yield

- Two-face-Bonferroni calibrated answers among 51 queries: **15-21**, median **19.0**.
- Project-macro answer coverage across 13 operational groups: **0.282-0.442**, median **0.338**.
- Answers lost after removing same-parent candidates: **12-17**, median **14.5**.
- Answers gained after the smaller candidate universe was recalibrated: **0-0**, median **0.0**. These are not evidence of correctness.

## Stability and candidate-source concentration

- **15/51** queries were answered in all 32 overlapping sensitivity graphs, **27/51** were never answered and **9/51** varied.
- **25/51** queries selected the same candidate parent project in every graph; the median number of selected candidate projects per query was **2.0**.
- The dominant similar-response candidate project was **CHEMPROP_SHARED_PARENT_PROJECT** (median **13.0/51** top hits; exposure-adjusted ratio **0.76x**).
- The dominant opposite-response candidate project was **MSV000086360** (median **15.0/51** top hits; exposure-adjusted ratio **3.97x**).
- After candidate exposure was considered, **MSV000086360** was the strongest similar-response project hub (median **9.5** hits; **2.51x** exposure-adjusted), and **MSV000086360** was strongest for opposite responses (median **15.0**; **3.97x**).

These stable selections are not truth. They show that candidate-source concentration persists after removal of the shared ChemProp parent project. They do **not** identify its cause: measurement coverage, a broadly shared chemical response, erroneous identity edges, study-level scoring preferences and real but nonspecific biology remain competing explanations.

## Registered source-independent relations

The 24 ChemProp exact-drug directions are **not evaluable** by design and are not failures. The remaining primary set contains only 10 directions, five canonical pairs and two biological components (HFD and IBD).

- Unsigned best-face Top-5: **1-3**, median **2.0**.
- Two-face-Bonferroni query-level max-T calls: **0-0**, median **0.0**.

## Decision

The parent-project holdout is complete, but it does not establish biological accuracy: 1,926/1,944 candidate directions are unjudged, every registered response face is unspecified, and no sealed external biology panel exists. The valid endpoint is statistically selected cross-project hypothesis yield, not accuracy, FDR, specificity, biological discovery count or mechanism recovery.

This system-level result remains a gate for the same-axis Official DreaMS, Noise V1 and Full23 comparison; those methods have not yet been evaluated here.
