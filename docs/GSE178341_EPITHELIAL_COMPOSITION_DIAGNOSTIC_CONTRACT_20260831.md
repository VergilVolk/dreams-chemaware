# GSE178341 epithelial-composition diagnostic contract (2026-08-31)

## Status and purpose

This is a **post-result diagnostic**, not a new confirmatory endpoint. It was
defined after the raw-UMI patient analysis showed mucinous-relative epithelial
AGR2 and SLC35A1 signals. Its sole purpose is to determine whether those signals
are adequately explained by a larger fraction of author-annotated goblet-lineage
epithelial cells.

## Frozen inputs and analysis unit

- Official GSE178341 cell metadata and author cluster labels.
- Frozen broad-epithelial patient pseudobulks from
  `nxpe1_mucinous_patient_pseudobulk_v1`.
- Inference unit: patient, never cell.
- Histologies: pure mucinous adenocarcinoma versus pure conventional
  adenocarcinoma, tumour samples only.
- Goblet-lineage definition, inherited from the earlier audit:
  `cE02`, `cE06`, `cE07`, `cE08`.

## Fixed diagnostics

1. Patient-level goblet-lineage fraction among all epithelial tumour cells.
2. Patient-level mature-goblet (`cE08`) fraction among all epithelial tumour
   cells.
3. Mucinous coefficient for `SECRETORY_COMPOSITE`, `SIALIC_BACKGROUND`, AGR2,
   SLC35A1, MUC2, SPDEF and NXPE1 in three nested patient-level models:
   histology only; histology plus logit goblet fraction; histology plus logit
   goblet fraction, right-colon status and MMR status.
4. Frozen six-case matched contrasts for goblet fraction.
5. Spearman association between goblet fraction and the fixed expression
   endpoints across patients.

All regressions use HC3 standard errors. Fractions are transformed as
`log((k+0.5)/(n-k+0.5))`. The diagnostic is considered evidence of a residual
cell-state component only when the composition-adjusted histology coefficient
keeps the original positive direction, its magnitude is at least 25% of the
unadjusted coefficient, and its HC3 95% interval excludes zero in the compact
histology + goblet-fraction model. This rule is descriptive and is not promoted
to a preregistered primary claim.

## Claim boundary

Cell-type fractions are obtained from dissociated single-cell samples and can be
affected by sampling and processing. Composition adjustment is therefore a
sensitivity analysis, not causal mediation. It cannot establish Neu5Ac source,
metabolic flux, enzyme activity, glycan destination or a therapeutic target.
