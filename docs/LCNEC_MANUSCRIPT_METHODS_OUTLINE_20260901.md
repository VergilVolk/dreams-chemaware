# LCNEC manuscript Methods outline (frozen analysis contract)

## Study and acquisition scope

- Primary dataset: public HSST3n negative-mode raw arm of the paired LCNEC atlas.
- Biological design: 34 patients, one tumor and one matched adjacent tissue per patient (68 study
  injections).
- Analytical controls: nine pooled-QC, two blanks and six dilution-series injections.
- Primary inference unit: patient pair; pooled QC is used for analytical qualification and MS2 query
  construction, never as a biological replicate.

## Phenotype-blind analytical universe

Feature detection, precursor-RT grouping and QC/blank/dilution qualification were completed without
tumor labels. Starting from 1,138 precursor-RT families, 607 passed pooled-QC reproducibility, 675 passed
the blank criterion, 359 were dilution responsive and 263 passed all three. These 263 families define
the common denominator for source-table overlap, official DreaMS candidate coverage and full-workflow
evidence yield. No denominator is back-calculated for the source article because its supplement does not
report all detected features.

## Source positive controls and dark-module freezing

Source HSST3n rows were matched to the qualified universe using the frozen m/z-RT protocol. Forty-two
families overlapped the source table. Locally extracted effects were compared with source-reported
effects before author-unreported candidates were interpreted. Among source-table-absent families,
abundance robustness was evaluated under four frozen normalization schemes, followed by coelution-based
redundancy control, yielding 81 nonredundant dark modules. Biological labels were not used during
analytical qualification or spectral candidate generation.

## Spectral representation and candidate retrieval

One representative pooled-QC MS2 spectrum was selected per qualified precursor-RT family. The official
DreaMS checkpoint generated the spectral embedding. Candidate molecules were constrained by the frozen
20-ppm precursor protocol and ranked by official DreaMS similarity. A separately frozen P2b candidate
expert and predefined classical evidence (direct fragment matching, square-root cosine, entropy,
matched-peak and intensity coverage) were used as orthogonal ranking/consistency evidence. P2b is a
downstream candidate expert and is not described as an improved embedding.

The current application does not contain a full across-QC-query-spectrum repeatability experiment for
the four final candidates. Multiple reference spectra support the library candidates, but a single
representative QC query per family remains a limitation.

## Identity levels and rival structures

Candidate identity was frozen before pathway or phenotype interpretation. Formula agreement, direct
fragment support, DreaMS score, score margin, classical spectral metrics, number of reference spectra
and full-InChIKey multiplicity were retained. A separate rival audit recomputed molecular formulae from
top-five candidate SMILES and enumerated exact-formula structures in a frozen local HMDB table.

- ADP and ADP-ribose are connectivity-family hypotheses.
- Ascorbate and quinolinate are MSI Level-2 compound hypotheses.
- No priority is Level 1 and the number of new exact metabolite claims is zero.
- Absence of a same-formula library spectrum is treated as missing rival coverage, not uniqueness.

## Abundance analysis

Targeted MS1 extraction was performed for the frozen feature families in all 68 study injections.
Patient-specific tumor-to-adjacent log2 effects were calculated under the frozen normalization schemes.
Primary evidence includes paired effect size, concordant-pair count, two-sided paired tests,
Benjamini-Hochberg correction where applicable and leave-one-patient sign stability. Static abundance is
not interpreted as metabolic flux or enzyme activity.

## Patient-level covariation

For the four priority hypotheses, all six Spearman correlations among patient-specific paired effects
were tested. The frozen gate required absolute rho >=0.35, BH q<0.10 across six tests, a patient-bootstrap
95% interval excluding zero and leave-one-patient sign stability. This analysis tests a shared
patient-level module; it does not validate identity or mechanism.

## Recorded technical-confounding audit

The public acquisition workbook was inspected before defining covariates. It contained sample code,
tumor/adjacent group, tissue amount and injection order for all 34 pairs, but no clinical stage, smoking,
sex or tumor-purity variables. Four technical predictors were fixed: log2 tumor-to-normal tissue-amount
ratio, mean paired injection number, signed tumor-minus-normal injection number and absolute paired
injection gap. Each was related to each of the four frozen patient-specific priority effects by Spearman
correlation (16 tests total). The common gate required |rho|>=0.35, BH q<0.10 across 16 tests, a
5,000-resample patient-bootstrap interval excluding zero and leave-one-patient sign stability. This is a
discovery-cohort sensitivity analysis, not independent replication; a null result cannot exclude
unrecorded technical or clinical confounding.

## Objective smoking-exposure sensitivity audit

The source-paper supplement was inspected separately from the public acquisition overview. Table S4
provided tumor-tissue cotinine concentrations and a source-defined cotinine smoking classification for
all 34 patients; Table S1 provided age, sex, BMI and stage. Before computing candidate associations, the
four frozen endpoints and three analysis arms were preregistered. The primary arm compared paired
tumor-minus-normal log2 effects between cotinine-classified smokers and non-smokers by Welch test with
BH correction across four endpoints. The secondary arm used Spearman correlation with log2 tumor
cotinine. The adjusted sensitivity used OLS with HC3 covariance and cotinine smoking, standardized age,
sex, standardized BMI and late stage. A potential smoking-sensitivity call required q<0.10 in all three
arms with concordant direction. Bootstrap intervals and leave-one-patient analyses assessed stability.
Cotinine is an exposure proxy measured in tumor tissue; this audit cannot validate identity or mechanism,
and a null result cannot prove absence of smoking confounding.

## BioAware reaction context

Spectral hypotheses were locked before one-hop Rhea context was queried. Reaction evidence could label a
candidate as a nonhub context anchor or trigger abstention, but could not change identity. High-degree
currency metabolites were excluded from pathway-specific interpretation; ADP was abstained under this
rule. Reaction membership does not determine reaction direction, flux, enzyme activity or causal source.

## Independent proteomic context

Before opening the independent patient matrix, 22 proteins were frozen across de-novo-NAD/quinolinate,
ADP-ribose/PARP and redox handling. The deposited matrix comprised 103 tumor/NAT pairs (80 pure and 23
combined LCNEC). The primary endpoint was paired pure-LCNEC tumor-minus-NAT abundance with two-sided
Wilcoxon testing, BH correction across all 22 frozen proteins, patient-bootstrap uncertainty and
direction-stability gates. Missing proteins were reported and were not replaced after outcome review.
The combined-versus-pure contrast was secondary. Protein evidence supplies pathway context only.

## External LCNEC transcriptomic and genomic-stratum context

The same frozen 22-gene panel and three axes were evaluated in the George et al. LCNEC cohort
(66 tumors; no matched normal tissue). Log2(RSEM+1) values were standardized gene-wise. The author
transcriptomic subtype analysis was retained as secondary context because subtype construction partly
used the same expression matrix. Its primary axis statistic was a Euclidean multivariate pseudo-F with
10,000 label permutations and BH correction across three axes; the fixed gate additionally required
R2>=0.10, a stage-stratified permutation p<0.05 and no BH-significant dispersion difference.

For an expression-independent contrast, clean genomic strata were fixed before inspecting the axis
outcomes: STK11/KEAP1-altered tumors required an STK11 or KEAP1 event and no RB1 event; RB1-altered
tumors required an RB1 event and no STK11 or KEAP1 event. Missing, overlapping and neither-group tumors
were excluded. This yielded 22 versus 17 tumors. The same multivariate, stage-stratified and dispersion
gates were applied. Single-gene Mann-Whitney tests were secondary and BH-corrected across all 22 frozen
genes. These analyses test external LCNEC pathway-context heterogeneity only; they contain no
tumor-adjacent contrast or metabolite measurement.

## Same-denominator performance reporting

The following are reported separately:

- source-table feature overlap: 42/263;
- official DreaMS constrained candidate coverage: 158/263;
- DreaMS-P2b candidate agreement: 136/263;
- full multi-evidence retention: 66/263; and
- source-positive-control structural concordance: 17/19 on the uniquely resolvable subset.

Only the last quantity is an identity-concordance proxy, and it remains limited by its small,
source-Level-2-biased denominator. The other quantities are coverage or evidence yield.

## Reproducibility and claim audit

Every terminal analysis writes a machine-readable report, tables and source hashes. The manuscript
package is rebuilt from those frozen artifacts and validated by hash. A reverse claim audit currently
tests 19 prohibited or required interpretations, including annotation-rate denominator mixing,
same-formula uniqueness, independent metabolite replication, patient-module confirmation, recorded
technical confounding and flux.
