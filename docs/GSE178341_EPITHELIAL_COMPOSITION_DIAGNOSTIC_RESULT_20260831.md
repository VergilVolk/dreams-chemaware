# GSE178341 epithelial-composition diagnostic result (2026-08-31)

## Bottom line

The raw patient-level result is not explained by a single composition effect.
Mucinous tumours contain a larger mean fraction of author-annotated goblet-lineage
epithelial cells, but the six-patient uncertainty is large. After adjustment for
that fraction, AGR2 and SLC35A1 retain positive mucinous coefficients with HC3
intervals above zero, whereas MUC2, SPDEF and NXPE1 do not. The most defensible
interpretation is therefore a mixture of **goblet-lineage composition** and a
more selective **epithelial secretory-folding/Golgi-transport state**.

This analysis was explicitly frozen as a post-result diagnostic in
`docs/GSE178341_EPITHELIAL_COMPOSITION_DIAGNOSTIC_CONTRACT_20260831.md`; it is
not a new confirmatory endpoint.

## Patient-level composition

Using tumour epithelial cells only, the combined `cE02/cE06/cE07/cE08`
goblet-lineage fraction was `0.284` in six mucinous patients and `0.152` in 53
conventional-adenocarcinoma patients (difference `+0.132`; patient bootstrap
95% CI `-0.013 to +0.287`; permutation p=`0.0899`). In the frozen matched set,
four of six mucinous cases had a positive contrast and the mean contrast was
`+0.113` (exact sign-flip p=`0.406`). The mature `cE08` fraction differed by
only `+0.0050` (bootstrap interval `-0.0066 to +0.0186`). These data indicate a
plausible broad goblet-lineage enrichment, not a statistically resolved change
in mature goblet cells.

## Composition-adjusted state diagnostics

The compact patient-level model contains only histology and standardized logit
goblet fraction. The clinical sensitivity additionally includes right-colon and
MMR status. All standard errors are HC3.

| Endpoint | unadjusted mucinous beta | plus goblet fraction | plus goblet + right/MMR | diagnostic decision |
|---|---:|---:|---:|---|
| secretory composite | +0.917 | +0.369 (95% CI +0.147 to +0.591) | +0.316 (95% CI +0.116 to +0.517) | residual state present |
| AGR2 | +1.613 | +1.172 (+0.483 to +1.861) | +1.038 (+0.464 to +1.613) | residual state present |
| SLC35A1 | +0.833 | +0.676 (+0.216 to +1.135) | +0.643 (+0.129 to +1.157) | residual state present |
| sialic-background composite | +0.687 | +0.438 (-0.186 to +1.062) | +0.429 (-0.236 to +1.094) | unresolved |
| MUC2 | +2.117 | +0.627 (-0.263 to +1.518) | +0.493 (-0.450 to +1.437) | composition-sensitive |
| SPDEF | +1.956 | +0.977 (-0.232 to +2.186) | +0.862 (-0.351 to +2.075) | composition-sensitive |
| NXPE1 | +0.837 | +0.371 (-0.581 to +1.323) | +0.389 (-0.608 to +1.386) | no independent support |

Across all 59 patients, goblet fraction is strongly associated with the broad
secretory composite (Spearman rho=`0.900`) and MUC2 (rho=`0.815`), but less
strongly with SLC35A1 (rho=`0.441`). This quantitative separation explains why
MUC2/SPDEF attenuate after composition adjustment while AGR2/SLC35A1 retain a
substantial coefficient.

## Biological interpretation

Prior single-cell studies describe mucinous CRC cancer cells as goblet-like and
enriched for markers including MUC2 and FCGBP. The present patient-level audit
therefore treats goblet composition as an expected biological confounder rather
than an inconvenience. Its new contribution is narrower: within the limits of
dissociation-based cell fractions, the AGR2/SLC35A1 signal is not reducible to
the measured goblet-lineage fraction alone.

Together with the same-patient metabolite result, the evidence supports:

1. a larger extractable free Neu5Ac pool in Rmu;
2. a partly composition-driven mucin-secretory phenotype;
3. residual epithelial AGR2-mediated folding and SLC35A1-mediated Golgi
   CMP-Neu5Ac transport capacity;
4. no evidence that host NEU1/NEU3 release, NXPE1, or the entire sialic pathway
   is uniformly activated.

It does not establish the biochemical source of free Neu5Ac, transport flux,
enzyme activity, causal mediation or the destination glycan.

## Provenance

- Output: `data/external/GSE178341_mucinous_secretory_audit/epithelial_composition_diagnostic_v1/`
- Analysis: `tasks/audit_gse178341_epithelial_composition_v1.py`
- Validation: `tasks/validate_gse178341_epithelial_composition_v1.py`
- Literature context:
  - https://pmc.ncbi.nlm.nih.gov/articles/PMC9870908/
  - https://pmc.ncbi.nlm.nih.gov/articles/PMC10114614/
