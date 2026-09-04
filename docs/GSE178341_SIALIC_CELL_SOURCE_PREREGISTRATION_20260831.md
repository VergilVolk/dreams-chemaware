# GSE178341 sialic-cell-source audit: frozen analysis contract

## Question

The MTBLS13729 result is a patient-paired increase in the **free Neu5Ac pool**. This audit asks a narrower, independently testable question: in mucinous versus conventional colorectal cancer, which cellular compartment carries transcriptional capacity that is compatible with that pool change?

This is a source-localisation analysis, not a flux analysis. RNA abundance cannot prove Neu5Ac synthesis, release, uptake, transport, glycan incorporation, or microbial contribution.

## Frozen cohort and unit of inference

- Dataset: GSE178341 / Human Colon Cancer Atlas raw 10x UMI matrix.
- Tumours only.
- Histology: pure `Adenocarcinoma;Mucinous` versus pure `Adenocarcinoma`.
- Inferential unit: patient PID, never cell.
- Expected cohort: 6 mucinous and 53 conventional tumours.
- Compartments: author-defined `Epi` and `Myeloid` only. These were chosen before reading raw-count outcomes because both have adequate cells in all six mucinous patients and represent the two main host-side source hypotheses.
- Minimum inclusion: 30 cells in the compartment and positive total UMI library.

## Frozen gene modules

1. **Epithelial secretory carrier:** `MUC2, TFF3, SPDEF, FCGBP, AGR2`.
2. **CMP-Neu5Ac synthesis/transport capacity:** `GNE, NANS, CMAS, SLC35A1` in Epi and Myeloid.
3. **Glycoconjugate release capacity:** `NEU1, NEU3` in Epi and Myeloid.
4. **Intracellular salvage/catabolism:** `SLC17A5, NPL` in Epi and Myeloid.

No additional gene or cell type may be promoted into the fixed panel after raw-count outcomes are inspected. Missing genes remain reported as missing.

## Estimands and statistics

- Sum raw UMI counts within each PID-compartment-gene pseudobulk.
- Transform as `log2(CPM + 1)` using the total UMI library for that PID-compartment.
- Standardise each gene within its compartment across the 59 patients, then average available genes to form a module score.
- Primary contrast: mean mucinous minus conventional module score.
- Uncertainty: 20,000 patient bootstrap resamples.
- P value: 100,000 patient-label permutations.
- Multiplicity: Benjamini-Hochberg across the seven fixed compartment-module endpoints.
- Sensitivity: right-colon subset with MMR-stratified permutation; fixed 6-case/18-control matching from the phenotype-blind metadata preflight; leave-one-mucinous-patient-out direction.

## Interpretation rules

- A module is called supported only when its bootstrap lower bound is above zero, BH q is below 0.10, and all six leave-one-mucinous-patient-out estimates retain the same positive sign.
- A positive epithelial secretory or CMP-Neu5Ac capacity result supports a host epithelial context but does not prove that it produced the measured free Neu5Ac.
- A positive myeloid release result supports a host immune-cell release context but does not prove extracellular Neu5Ac release.
- Negative RNA results do not exclude enzyme activity, microbial sialidases, protein-level regulation, or spatially restricted effects.
- Histology-by-compartment differences are not treated as causal mediation.

## Forbidden claims

- metabolic flux or enzyme activity;
- proof of Neu5Ac biochemical source;
- proof that free Neu5Ac enters MUC2 or any specified glycan linkage;
- microbial causation;
- confirmation from a significant-versus-nonsignificant contrast without a direct contrast test.

