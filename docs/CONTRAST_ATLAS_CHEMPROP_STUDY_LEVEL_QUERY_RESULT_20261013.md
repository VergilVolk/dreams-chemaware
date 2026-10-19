# ChemProp study-level Contrast-to-Contrast query: formal negative result

Date: 2026-10-13  
Status: completed development result; do not use as external confirmation

## Question and evidence boundary

This experiment asks whether a real study-level chemical response can retrieve
the same drug across the ChemProp2 time-series (`MSV000094899`) and the two
ChemProp endpoint designs (`MSV000096724`, `MSV000093571`).  It replaces the
previous 36 replicate-level queries with 12 drug-level contrasts.  Each query
effect is the mean across three bacterial replicates; endpoint effects use two
replicates in the 23-drug batch and three in the 45-drug batch.

The estimand is

`(bacteria T2 - abiotic T2) - (bacteria T0 - abiotic T0)`.

The phenotype-blind cross-design feature graph was frozen before drug labels
were opened.  The same graph was scored as terminal chemical state, endpoint
difference, unsigned response, signed response, and signed response weighted by
replicate uncertainty.  Results are `development/consumed`, `formal=false`:
the ChemProp sources belong to one project, and this analysis follows prior
development on the same data.

Denominator: 12 drug queries, comprising three queries against three candidates
and nine queries against nine candidates.  The stratified random Recall@1 is
16.67%.

## Result

The biological response did not outperform the terminal chemical state.

| Frozen entity map | Evidence world | T2 state R@1 | signed response R@1 | signed + uncertainty R@1 |
|---|---|---:|---:|---:|
| mass 10 ppm | all | 8/12 (66.7%) | 5/12 (41.7%) | 2/12 (16.7%) |
| cosine top-1 >= 0.7 | all | 8/12 (66.7%) | 2/12 (16.7%) | 2/12 (16.7%) |
| WSE top-1 | all | 8/12 (66.7%) | 3/12 (25.0%) | 2/12 (16.7%) |
| WSE top-1 | library-supported same-key only | 6/12 (50.0%) | 1/12 (8.3%) | 3/12 (25.0%) |

For the all-entity WSE graph, uncertainty-aware response minus T2 state was
`-50.0 pp` Recall@1 with drug-cluster bootstrap CI `[-75.0, -25.0] pp`; the MRR
difference was `-0.3557`, CI `[-0.5486, -0.1612]`.  The same `8/12` versus
`2/12` ordering occurred under mass and cosine entity maps.  Weighting by the
replicate SE did not produce a stable benefit over the unweighted signed
response.

The all-entity world also did not beat the stringent library-supported
same-InChIKey world for the uncertainty-aware response: WSE gave `2/12` versus
`3/12`, and cosine gave `2/12` versus `4/12`.  These small denominators do not
support an unresolved-chemistry increment.

## Scientific interpretation

ChemProp is a real Contrast-to-Contrast experiment, because the ranking score
uses aligned study effects rather than spectrum similarity alone.  However, it
fails as the first positive control for the proposed Contrast Atlas.  Same-drug
retrieval is much stronger from the terminal chemical state than from the
double-difference microbial response.  This is consistent with drug-associated
chemical composition, including parent or related ions, carrying the dominant
cross-design signal; it does not establish that this is the sole cause.

Consequently:

1. Do not cite the old 36-query result as proof of transferable biological
   response.
2. Do not tune additional transport rules on these twelve drugs.
3. Retain ChemProp as a consumed negative/control shard showing why chemical
   state retrieval must not be confused with response retrieval.
4. Move the first positive test to an intervention replicated across genuinely
   different source families, with its relation fixed independently of the
   retrieval score.

Machine-readable outputs:

- `data/validation/chemprop_study_level_contrast_query_v1_20261013/report.json`
- `data/validation/chemprop_study_level_contrast_query_v1_20261013/retrieval_results.csv`

