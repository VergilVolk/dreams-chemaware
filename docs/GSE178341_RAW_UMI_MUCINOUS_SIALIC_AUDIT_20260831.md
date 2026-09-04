# GSE178341 raw-UMI patient audit: mucinous secretory and sialic context

## Outcome

The independent raw-count single-cell cohort supports a selective epithelial programme in mucinous colorectal cancer. It does **not** support a uniformly activated sialic-acid pathway or an independent NXPE1 driver.

All inference used patient pseudobulks. The 370,115 cells were never treated as biological replicates. The frozen cohort contained six pure mucinous and 53 pure conventional tumours.

## Primary NXPE1 audit

In broad tumour epithelium, NXPE1 was higher in mucinous CRC by `+0.837 log2(CPM+1)` (patient bootstrap 95% CI `+0.033 to +1.785`), but the patient-label permutation p was `0.152` and the fixed-panel BH q was `0.229`. In the right-colon/MMR-stratified sensitivity the effect remained positive (`+0.773`) but the bootstrap interval crossed zero and BH q was `0.318`.

The frozen 6-case/18-control match gave a mean contrast of `+1.049`, but only four of six mucinous patients were positive and the exact sign-flip p was `0.125`. HC3 adjustment for clinicotechnical covariates reduced the mucinous coefficient to `+0.647` (p=`0.313`); adding the fixed secretory programme reduced it further to `+0.242` (p=`0.706`).

**Decision:** NXPE1 is compatible with a secretory-carrier state but does not pass the independent primary-support gate. This agrees with the current-GDC result in which the mucinous NXPE1 coefficient disappears after adjustment for the distributed secretory programme.

## Fixed genes in tumour epithelium

Two genes show the cleanest independent patient-level signals:

| Gene | Mucinous minus conventional | patient bootstrap 95% CI | fixed-panel BH q | right/MMR BH q | Interpretation |
|---|---:|---:|---:|---:|---|
| AGR2 | +1.613 | +0.962 to +2.151 | 0.0068 | 0.0179 | replicated secretory-folding capacity |
| SLC35A1 | +0.833 | +0.462 to +1.212 | 0.0068 | 0.0179 | replicated Golgi CMP-sialic-acid transport capacity |
| SPDEF | +1.956 | +0.913 to +3.014 | 0.0538 | 0.128 | strong broad-epithelial direction, weaker stratified evidence |
| MUC2 | +2.117 | +1.033 to +3.188 | 0.141 | 0.223 | large direction but not multiplicity-confirmed |
| GNE | +0.968 | +0.502 to +1.433 | 0.141 | 0.223 | positive synthesis-capacity trend, not confirmed |
| NANS | +0.128 | -0.298 to +0.609 | 0.594 | 0.968 | no support |
| CMAS | +0.161 | -0.406 to +0.673 | 0.593 | 0.344 | no support |
| CASD1 | +0.066 | -0.219 to +0.391 | 0.832 | 0.968 | no O-acetylation-capacity support |
| SIAE | +0.200 | -0.095 to +0.484 | 0.475 | 0.776 | no support |

Within the predefined goblet-family clusters, AGR2 remains positive and multiplicity-controlled (all-tumour BH q=`0.0157`; right/MMR q=`0.0238`), whereas MUC2, TFF3 and SLC35A1 do not. This makes a simple “each goblet cell uniformly expresses more of every pathway member” model unlikely; cell-state and composition diagnostics remain secondary rather than confirmatory.

## Frozen cellular-source module audit

Seven compartment-module endpoints were frozen before reading raw-count outcomes.

| Compartment and module | Mucinous minus conventional z | bootstrap 95% CI | BH q across 7 | matched positive cases | Gate |
|---|---:|---:|---:|---:|---|
| Epithelial secretory carrier | +0.917 | +0.566 to +1.284 | 0.0627 | 5/6 | pass |
| Epithelial CMP-Neu5Ac capacity | +0.687 | +0.179 to +1.246 | 0.0627 | 5/6 | pass |
| Myeloid CMP-Neu5Ac capacity | +0.249 | -0.008 to +0.508 | 0.440 | 5/6 | fail |
| Epithelial NEU1/NEU3 release | -0.547 | -0.979 to -0.130 | 0.160 | 3/6 | fail |
| Myeloid NEU1/NEU3 release | -0.352 | -0.907 to +0.187 | 0.468 | 2/6 | fail |
| Epithelial salvage/catabolism | +0.266 | -0.152 to +0.640 | 0.468 | 4/6 | fail |
| Myeloid salvage/catabolism | -0.177 | -1.241 to +0.787 | 0.654 | 4/6 | fail |

The two passing modules retain positive effects in the right-colon/MMR-stratified sensitivity and after deleting any one mucinous matched case. The exact six-case sign-flip p values are `0.0625` and `0.09375`; this discreteness is reported rather than hidden.

## Integrated biological interpretation

The independent patient-level RNA result now supports the following constrained chain:

1. MTBLS13729: Level-1 free Neu5Ac rises in all ten Rmu pairs.
2. Same patients: CMP-Neu5Ac and UDP-GlcNAc do not rise proportionally.
3. Independent raw single-cell cohort: mucinous tumour epithelium has increased AGR2-mediated secretory folding and SLC35A1-mediated Golgi donor-transport capacity.
4. The full GNE/NANS/CMAS synthesis/activation route is not uniformly increased, and host epithelial/myeloid NEU1/NEU3 transcription does not explain the free pool.
5. Independent O-glycomics and MUC2 glycopeptide studies show core/linkage/carrier remodelling but do not replicate free Neu5Ac abundance.

The most defensible model is therefore **selective epithelial secretory/transport capacity coupled to a free-pool-to-donor/destination decoupling**, not global hypersialylation, increased flux, or a proven NXPE1/NEU1/NEU3 mechanism.

## Provenance and claim boundary

- Raw H5 SHA256: `f435bb2651ff5297d0c24a99daf58850ed67ae1ed6c5ef05fad48fa3f0186670`
- Primary result: `data/external/GSE178341_mucinous_secretory_audit/nxpe1_mucinous_patient_pseudobulk_v1/`
- Cellular-source result: `data/external/GSE178341_mucinous_secretory_audit/sialic_cell_source_patient_pseudobulk_v1/`
- Primary validator: `tasks/validate_gse178341_nxpe1_mucinous_v1.py` (PASS)
- Source validator: `tasks/validate_gse178341_sialic_cell_source_v1.py` (PASS)

These are independent patient-level host transcriptional contexts. They do not establish Neu5Ac biochemical source, microbial contribution, enzyme activity, glycan destination, metabolic flux, causal mediation, or therapeutic vulnerability.

