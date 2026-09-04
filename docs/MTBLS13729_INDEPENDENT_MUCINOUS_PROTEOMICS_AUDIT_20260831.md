# MTBLS13729 biological follow-up: independent mucinous-CRC proteomics audit

## Bottom line

The independent 15-MC versus 15-conventional-CRC proteomics cohort does **not** confirm either prespecified four-protein module at the frozen statistical threshold. It nevertheless provides a coherent but underpowered directional observation: AGR2, GNE and NANS are higher in mucinous CRC and keep the same direction in every leave-one-mucinous-patient-out analysis. This is supporting context, not a positive validation endpoint.

The result therefore sharpens the biological model rather than proving it:

- MTBLS13729 provides the primary Level-1 abundance result: free Neu5Ac is increased in all 10 Rmu pairs.
- The independent protein cohort is compatible with increased secretory folding capacity (AGR2) and increased upstream sialic-acid precursor synthesis (GNE/NANS), but its confidence intervals include zero.
- CMAS and SIAE are essentially unchanged, so these data do not support uniform activation of the complete pathway.
- The single-cell portal sensitivity previously showed AGR2 and SLC35A1 signals; raw-count patient pseudobulk remains the necessary primary transcriptomic test.

## Frozen design

The analysis contract was recorded before inspecting fixed-panel group outcomes in `docs/MTBLS13729_INDEPENDENT_MUCINOUS_PROTEOMICS_PREREGISTRATION_20260831.md`.

- Patient-level cohort: 15 mucinous adenocarcinoma (MC), 15 conventional adenocarcinoma NOS (AC), 16 normal colon.
- Primary contrast: MC versus AC.
- Fixed proteins measured: AGR2, MUC2, TFF3, FCGBP, GNE, NANS, CMAS, SIAE.
- Prespecified but unavailable: NXPE1, SPDEF, SLC35A1, CASD1.
- Primary inference: log2 abundance difference, 200,000 label permutations, 10,000 patient bootstraps, BH across eight proteins.
- Mandatory checks: Mann-Whitney, age/sex HC3 model, leave-one-MC-out direction, and exact-minimum-value audit.

## Results

| Protein | MC/AC fold | 95% bootstrap CI on log2 difference | permutation p | BH q | leave-one-MC-out direction | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| AGR2 | 1.86 | -0.30 to 2.02 | 0.161 | 0.643 | 15/15 | stable positive direction; not confirmed |
| MUC2 | 1.21 | -1.27 to 1.73 | 0.733 | 0.999 | 15/15 | weak positive; strong floor sensitivity |
| TFF3 | 1.03 | -0.70 to 0.86 | 0.999 | 0.999 | 13/15 | uninformative because 37/46 values equal the floor |
| FCGBP | 1.44 | -1.70 to 2.72 | 0.650 | 0.999 | 15/15 | positive but highly variable |
| GNE | 1.48 | -0.32 to 1.46 | 0.241 | 0.643 | 15/15 | stable positive direction; not confirmed |
| NANS | 1.42 | -0.28 to 1.31 | 0.240 | 0.643 | 15/15 | stable positive direction; not confirmed |
| CMAS | 0.92 | -1.03 to 0.78 | 0.799 | 0.999 | 10/15 | no positive support |
| SIAE | 1.04 | -0.70 to 0.76 | 0.891 | 0.999 | 10/15 | no positive support |

Age/sex HC3 adjustment preserves the positive direction for AGR2, GNE and NANS, but none is statistically confirmed. The strongest adjusted coefficient is AGR2 (+1.01 log2 units; normal-reference robust p=0.107).

The source article's AGR2 call can be reconciled rather than treated as a contradiction. On untransformed abundance, AGR2 has an arithmetic-mean MC/AC fold change of 2.60 and Welch p=0.0478, satisfying the article's exploratory `FC > 1.5, p < 0.05` rule and matching its supplementary `MC_AC_up` list. On the frozen log2/permutation analysis, however, p=0.161 and the patient bootstrap interval crosses zero. The robust conclusion is therefore **stable positive direction with scale-sensitive significance**, not confirmed protein-specific support.

The secretory/mucin module difference is +0.211 z-score units (bootstrap -0.317 to +0.738; permutation q=0.449). The sialic biosynthesis/handling module difference is +0.207 (bootstrap -0.292 to +0.684; q=0.449). Neither passes the frozen gate.

## Data-quality finding that changes interpretation

Repeated exact minima indicate likely left-censoring or imputation. TFF3 is the clearest failure: 37/46 values equal the exact minimum, including 25/30 tumour samples. It cannot serve as independent abundance evidence in this matrix. MUC2 and GNE also have substantial floor mass, so their raw mean differences must be interpreted with the rank and non-floor sensitivities.

This audit avoids the original article's exploratory `fold-change > 1.5 and p < 0.05` selection. All eight fixed proteins remain visible, including the null results.

## What the result contributes to the paper

It can support one restrained paragraph:

> In an independent patient-level proteomics cohort, the prespecified mucin-secretory and sialic-handling modules did not reach multiplicity-controlled significance. Nevertheless, AGR2, GNE and NANS showed stable positive MC-versus-conventional-CRC directions under leave-one-patient-out analysis, whereas CMAS and SIAE did not. Together with the Level-1 increase of free Neu5Ac, this pattern is compatible with selective precursor/secretory remodeling rather than uniform activation of the entire sialic-acid pathway.

It cannot support the claims that the pathway is confirmed, that flux is increased, or that these protein trends independently replicate the Neu5Ac measurement.

## Provenance

- Source workbook SHA256: `fafec320da25def890f430719d88d755f424b7bfa5fdd621f6a2911a9625ed28`
- Machine-readable result: `data/external/GSE178341_mucinous_secretory_audit/independent_proteomics_fixed_panel_v1/result.json`
- Validator: `tasks/validate_mucinous_crc_independent_proteomics_v1.py` (PASS)
- Source article: https://www.frontiersin.org/journals/molecular-biosciences/articles/10.3389/fmolb.2023.1150362/full
