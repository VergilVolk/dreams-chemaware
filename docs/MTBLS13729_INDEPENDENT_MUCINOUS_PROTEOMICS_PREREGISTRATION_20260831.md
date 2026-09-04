# Independent mucinous-CRC proteomics audit: frozen analysis contract

Date frozen: 2026-08-31, before inspecting group-level outcomes for the fixed panel.

## Question

Does an independent patient-level proteomics cohort support the specific biological axis suggested by the MTBLS13729 finding of increased free Neu5Ac in mucinous colorectal cancer: a secretory/mucin-handling program and/or a coherent sialic-acid biosynthesis/handling program?

This is an orthogonal protein-level validation. It is not an independent metabolite replication, does not validate flux, and does not establish enzyme activity.

## Cohort and unit of analysis

Source: Supplementary Table S2 (`Table2.XLSX`) from Xu et al., *Frontiers in Molecular Biosciences* (2023), DOI 10.3389/fmolb.2023.1150362.

- 15 mucinous colorectal adenocarcinoma patients (MC)
- 15 conventional adenocarcinoma-not-otherwise-specified patients (AC)
- 16 normal-colon patients (context only)
- The patient, not the protein or technical value, is the unit of analysis.

The primary comparison is MC versus AC. Normal colon is reported only to orient tumour-associated direction and is not used to rescue a failed MC-versus-AC result.

## Fixed proteins and modules

The panel was fixed from the pre-existing MTBLS13729/single-cell biological hypothesis before group outcomes were inspected.

Secretory/mucin module:

- AGR2
- MUC2
- TFF3
- FCGBP

Sialic-acid biosynthesis/handling module:

- GNE
- NANS
- CMAS
- SIAE

NXPE1, SPDEF, SLC35A1 and CASD1 are also prespecified but unavailable in this proteomics matrix and will be reported as unavailable rather than substituted post hoc.

## Primary statistics

1. Protein abundance is analysed as `log2(raw abundance)` after fail-closed confirmation that values are finite and strictly positive.
2. For each of the eight measured fixed proteins, the primary effect is the patient-level mean log2 difference, MC minus AC.
3. The primary p-value is a deterministic 200,000-draw label-permutation test preserving 15 versus 15 patients.
4. Benjamini-Hochberg correction is applied across exactly the eight measured fixed proteins.
5. A 10,000-resample percentile bootstrap provides the patient-level 95% confidence interval.
6. A fixed module score is the within-tumour z-score average of its four members. The two module tests use the same permutation/bootstrap scheme and BH correction across exactly two modules.

## Mandatory sensitivity and bias audits

- Mann-Whitney U test for each protein.
- HC3 robust linear model adjusting only for age and sex; no feature selection.
- Leave-one-MC-patient-out effect-sign stability.
- Per-protein minimum-value frequency by group, because repeated exact minima are treated as likely left-censoring/imputation evidence.
- A non-floor sensitivity is reported only when both tumour groups retain at least five patients; it is descriptive and cannot replace the primary analysis.
- Patient IDs in the matrix must match the proteomics rows in the patient-information sheet exactly.
- All fixed proteins, including null and unavailable proteins, remain in the report.

## Decision language

- `orthogonal support`: fixed module permutation q < 0.10, bootstrap CI excludes zero, and the direction is not reversed by HC3 adjustment.
- `protein-specific support`: protein permutation q < 0.10, bootstrap CI excludes zero, and the direction is stable in at least 14/15 leave-one-MC-out analyses.
- `directional only`: effect direction is compatible but the above gates are not met.
- `not supported`: effect is null/inconsistent or points in the opposite direction.

No result from this audit may be called a new metabolite, a flux change, or an independent replication of the Neu5Ac measurement.

