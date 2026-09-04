# LCNEC manuscript-ready Results draft (frozen evidence; updated 2026-09-01)

## Working title

Algorithm-enabled recovery of previously unreported nucleotide/NAD-related and antioxidant metabolite signals from paired LCNEC tissues

## Result 0 — Annotation recovery was benchmarked with non-interchangeable denominators

The source article reported 1,052 metabolites across all platforms (73 MSI Level 1, 935 Level 2 and 46
Level 3 statistical rows after removal of one extra note row). Its supplement does not report a common
detected-feature denominator, so an author annotation rate cannot be reconstructed and must not be
estimated from our feature universe. Within the source HSST3n arm, 97 main-table rows plus three
HSST3n-2HG rows were reported.

Our independently reconstructed QC/blank/dilution-qualified HSST3n universe contained 263
precursor–retention-time families. Forty-two (16.0%) matched the source HSST3n table and 221 (84.0%)
were source-table-absent analytical headroom. We then ran the official DreaMS and the frozen full
candidate workflow on all 263 representative QC spectra, establishing a genuinely shared denominator:
official DreaMS returned a 20-ppm constrained library candidate for 158/263 (60.1%), DreaMS and P2b
agreed on 136/263 (51.7%), and 66/263 (25.1%) survived the high/moderate multi-evidence gate. As an
internal positive control, 38/42 source-matched families had a DreaMS candidate and 31/42 survived the
full evidence gate. Among the 221 source-table-absent families, these counts were 120 and 35,
respectively. Because the source table contains names but not structures, we additionally resolved only
exact normalized source names that mapped uniquely to one IK14 in a frozen local HMDB library, without
manual synonym rescue. Nineteen of 42 source-matched families were evaluable. Official DreaMS and the
full workflow were each concordant with 17/19 source structures (89.5%; Wilson 95% CI 68.6–97.1%). The
full gate retained both discordant cases: N-acetylserine was ranked as glutamate, and cis-aconitate as
dehydroascorbate. Both pairs share a molecular formula, exposing a persistent same-formula-isomer failure
that multi-evidence confidence did not eliminate. This conservative subset is a source-concordance proxy,
not global accuracy, because 23 names were unresolved and most source identities are MSI Level 2.

In the separately frozen 81-module abundance-selected dark universe,
official DreaMS produced candidates for 51/81 (63.0%); 45/81 (55.6%) also agreed with the frozen P2b
candidate expert; 22/81 (27.2%) survived the full multi-evidence consistency gate; 12/81 (14.8%)
reproduced source-reported cross-platform biology; nine were author-unreported and four were retained as
priority hypotheses. These are successive coverage and evidence-calibration counts, not interchangeable
accuracy or annotation-rate estimates.

## Result 1 — A phenotype-blind raw-data workflow exposed reproducible analytical headroom

We reprocessed the public HSST3n raw-data arm of the 34-patient paired pulmonary large-cell
neuroendocrine carcinoma (LCNEC) atlas. The acquisition contained 68 study injections, nine pooled-QC
injections, two blanks and six dilution-series injections, providing 133,925 study and 17,727 pooled-QC
MS2 spectra. Of 1,138 precursor–retention-time families, 607 were reproducible in pooled QC, 675 were
blank-clean, 359 were dilution-responsive and 263 passed all three analytical criteria. Forty-two
families matched the published HSST3n atlas, whereas 221 were absent from that table and therefore
represented analytical headroom rather than automatically novel metabolites.

Targeted extraction from the raw MS1 data recovered the published direction and magnitude of the known
features: locally estimated effects correlated strongly with the author-reported effects (Spearman
rho=0.943), with 90.5% direction concordance. After four frozen normalization schemes and coelution-based
redundancy control, 81 nonredundant dark-feature modules remained for identity-blind annotation.

## Result 2 — The annotation workflow reproduced cross-platform biology before nominating new hypotheses

All 81 modules had pooled-QC MS2 evidence and were evaluated by the official DreaMS embedding, the
frozen candidate-ranking expert and classical peak-matching evidence under a fixed precursor-mass
protocol. Twenty-two features, representing 21 connectivity hypotheses, passed the cross-window and
cross-model consistency criteria. Twelve hypotheses overlapped metabolites independently reported by
the authors on another LC-MS platform. All 12 reproduced the published direction, their effect sizes
were strongly correlated (Spearman rho=0.902), and 10 were same-direction findings with author-reported
FDR<0.05. This cross-platform recovery was treated as an orthogonal positive control, not as discovery.

## Result 3 — Four author-unreported hypotheses survived orthogonal structure and patient-level checks

Nine consistency-filtered hypotheses were absent from the published HSST3n table. Four were prioritized before
the final structure and patient-level audits: an ADP family, an ADP-ribose family, ascorbate and
quinolinate. Their precursor masses matched the corresponding formulas within 1.51 ppm. Direct
query-to-library comparisons supported the assignments with 13, 20, 15 and five matched fragments,
respectively; query intensity coverage ranged from 0.846 to 0.973 and square-root cosine similarity from
0.890 to 0.975.

Across the 34 matched patient pairs, per-mass tumor-to-adjacent mean log2 fold changes were +2.400 for
the ADP family, +1.556 for the ADP-ribose family, +5.407 for ascorbate and +2.047 for quinolinate. The
directions were concordant in 33/34, 31/34, 32/34 and 28/34 pairs, respectively. All four effects retained
their direction in every leave-one-patient-pair-out analysis, and all paired Wilcoxon P values were below
2.83e-6. None showed a predefined coeluting C13, Na-H, chloride, formate or acetate mass-spacing flag in
the quality-passed feature ledger. This screen reduces obvious ion-form ambiguity but does not replace
authentic-standard retention time.

The ADP and ADP-ribose results are therefore reported as connectivity-family hypotheses because the
library evidence does not uniquely identify a full InChIKey. Ascorbate and quinolinate are reported as
compound-level Level-2 hypotheses. None is reported as Level 1.

An explicit same-formula rival audit reinforced this boundary: three of the four priorities had an
observed same-formula rival among the reported top-five candidates. The ADP-family top candidate exceeded
the strongest observed same-formula rival (adenosine 3',5'-diphosphate) by only 0.032 in official DreaMS
score. Ascorbate exceeded D-glucuronolactone by 0.305, whereas quinolinate exceeded 3-nitrobenzoate by
0.061. ADP-ribose had no same-formula rival spectrum in the top-five library output, but the local HMDB
formula subset contained two distinct structures; absence of a reference spectrum is not chemical
uniqueness. Thus all four remain non-unique without matched chromatography or another orthogonal
identity measurement.

We then tested whether “author-unreported” survived comparison with the source atlas beyond the HSST3n
arm. Across all 1,054 reported identity rows, no priority had an exact normalized alias or a uniquely
resolved exact-name-to-HMDB structure match. Neutral mass could be reconstructed from the recorded
adduct for 1,050 rows; none was within 5 ppm of any of the four priority formulas. The four unsupported
rows were one N-Lac-Met entry and three derivatized 2-hydroxyglutarate entries, not plausible aliases of
the priorities. Thus all four are source-atlas-unreported under the frozen exact-name/structure/mass
resolver, not merely absent from HSST3n. This still does not establish chemical novelty: the exact-name
HMDB resolver left 970 source names unresolved, and the candidate identities themselves retain the
same-formula and authentic-retention-time limitations described above.

## Result 4 — The abundance pattern supports pool redistribution, while BioAware prevents pathway overclaiming

The consistency-filtered hypotheses formed two descriptive abundance patterns. AMP, GMP, ADP, ADP-ribose and
UDP-HexNAc-related signals increased, whereas free guanosine and guanine decreased. Glutathione,
glutathione disulfide and ascorbate increased, whereas ophthalmate decreased. Together, these data
support redistribution of phosphorylated nucleotide/NAD-related pools and expansion of measured
antioxidant pools in LCNEC tissue. Static tissue abundance cannot establish ATP energy charge, pathway
flux, enzyme activity or causal redox adaptation.

To prevent circular presentation, every entry in the abundance map was labeled by evidence source:
`[R]` denotes a metabolite already present in the source atlas and reproduced in the raw HSST3n arm,
`[N]` denotes one of the four author-unreported priority hypotheses, and `[H]` denotes another Level-2
or connectivity-family hypothesis. The four axes and their directions were then frozen in a descriptive
coherence ledger (14 entries: nine `[R]`, four `[N]`, one `[H]`). Because this ledger was assembled after
metabolite review, it is not an independent pathway-enrichment test and supplies no additional P value.

BioAware was used only after spectral annotation to provide reaction-network context and to enforce
abstention. ADP occurred in 881 Rhea reactions and was classified as a currency hub; it was therefore
forbidden from activating a pathway-specific interpretation. ADP-ribose, ascorbate and quinolinate were
retained as nonhub context anchors. Reaction membership was not used to change the spectral identity and
does not establish reaction direction, flux or enzyme activity.

## Result 5 — A pre-registered independent protein panel supports three pathway contexts while resolving histology-specific direction

Before opening the patient-level matrix from an independent 107-patient LCNEC proteogenomic study, we
froze 22 proteins spanning quinolinate/de-novo-NAD metabolism, ADP-ribose turnover and ascorbate/redox
handling. The deposited matrix contained 103 tumor/NAT pairs, including 80 pure and 23 combined LCNEC.
Eighteen proteins were measured; TDO2, NMNAT2, SLC23A1 and SLC23A2 were reported missing and were not
replaced. In the primary pure-LCNEC paired endpoint, 13 proteins passed the fixed 22-test BH and
direction-stability gates. PARP1 (+1.319 tumor-minus-NAT log2, q=1.79e-13) and PARP2 (+0.868,
q=3.10e-12) were coherently increased, independently supporting a PARP-associated context for the
local ADP-ribose-family signal. QPRT (-0.853, q=8.13e-10), HAAO (-1.103, q=2.38e-13), IDO1
(-1.063, q=1.15e-8), KYNU and NADSYN1 decreased, whereas NMNAT3 increased. This mixed pattern supports
NAD-pathway redistribution and makes lower QPRT a testable quinolinate-utilization hypothesis, but it
does not establish pathway flux. The redox panel was also mixed: GSR, G6PD, TKT and TALDO1 decreased,
whereas TXNRD1 increased, supporting compensatory antioxidant-pool remodeling rather than uniform
pentose-phosphate activation in pure LCNEC.

As a secondary positive control, the combined-versus-pure tumor comparison reproduced the source
article's histology-dependent pentose-phosphate program: G6PD (+0.402 median log2, q=3.995e-4), TKT
(+0.719, q=2.57e-4) and TALDO1 (+0.258, q=0.00189) were higher in combined LCNEC, with PGD directionally
higher (q=0.0513). Thus the independent cohort strengthens pathway context and exposes subtype-specific
direction, but it does not measure or validate the metabolite identities, abundances, fluxes or enzymatic
sources reported above.

In a post-primary exploratory audit, all 46 pairwise patient-effect correlations within the three frozen
protein axes were tested together. Twelve passed |rho|>=0.30, BH q<0.05, bootstrap and leave-one-patient
stability gates; all 12 belonged to the redox axis. G6PD covaried with TKT (rho=0.440), TALDO1
(rho=0.428), PGD (rho=0.520) and TXNRD1 (rho=0.566). Thus the mixed redox mean directions coexist with
coordinated patient-to-patient variation, consistent with heterogeneous compensatory remodeling rather
than independent noise. PARP1--PARP2 was suggestive but did not pass the exploratory gate (rho=0.271,
BH q=0.05004), and no de-novo-NAD pair passed. This post-primary result is hypothesis-generating and is
not metabolite replication.

## Result 5A — Frozen axes align with expression-independent genomic strata in a second external LCNEC cohort

We next evaluated the same 22 genes and three pathway-context axes in a separate published cohort of
66 LCNEC tumors with RSEM expression and genomic annotations. Author transcriptomic subtype labels were
analyzed only as a secondary, potentially circular context because those labels partly derive from the
same expression matrix. Under the fixed multivariate gate, only the quinolinate/de-novo-NAD axis passed
the complete subtype test (BH q=0.0003, R2=0.150, stage-stratified p=0.0002, no dispersion signal).

The stronger primary external contrast therefore used expression-independent genomic definitions:
22 tumors with STK11 or KEAP1 alteration and no RB1 event versus 17 tumors with RB1 alteration and no
STK11/KEAP1 event. All three frozen axes passed the preregistered multivariate gate. Genomic stratum
explained 11.1% of the quinolinate/de-novo-NAD expression structure (BH q=0.0003;
stage-stratified p=0.0002), 10.4% of ADP-ribose turnover (q=0.0023; stage p=0.0006) and 13.7% of
ascorbate/redox handling (q=0.0023; stage p=0.0050). All three dispersion diagnostics remained above
the fixed BH threshold. At the single-gene level, only four of 22 genes passed joint BH correction:
NMNAT1 and NMNAT3 were lower in the STK11/KEAP1-altered group than in the RB1-altered group (median
log2-RSEM differences -0.573 and -1.619), PARP1 was lower (-0.855), and TKT was higher (+1.423).

A frozen post-primary leave-one-gene-out audit separated multi-gene structure from single-gene
dependence. The redox axis passed all eight omissions. The de-novo-NAD axis passed 8/9 omissions; after
NMNAT3 removal it remained statistically different (BH q=0.0079 and stage p=0.0080) but fell below the
predefined effect-size gate (R2=0.0716). The ADP-ribose axis passed 4/5 omissions; removing PARP1 reduced
the contrast to R2=0.0486 and BH q=0.119. Thus the external redox result is distributed across multiple
genes, whereas the NAD and ADP-ribose external contrasts are substantially anchored by NMNAT3 and PARP1.

This external result independently supports genomic-stratum heterogeneity of the same pathway contexts
that were frozen from the metabolite and protein analyses. It does not reproduce tumor-versus-normal
metabolite or protein direction because the cohort contains tumor tissue only, and it cannot validate
metabolite identity, flux, prognosis or therapeutic dependency.

## Result 6 — Multi-cohort triangulation resolves four candidates into distinct validation roles

Integrating the frozen metabolite, source-atlas, BioAware and independent-protein ledgers did not make
the four author-unreported hypotheses equivalent. The ADP-ribose connectivity family provided the
cleanest cross-omics mechanism context: its paired abundance increased by +1.556 log2 (31/34 concordant
pairs), while PARP1 and PARP2 increased in the independent pure-LCNEC cohort. This prioritizes an
ADP-ribose/PARP-turnover hypothesis but neither resolves the ADP-ribose isomer nor proves PARP-derived
flux. Quinolinate increased by +2.047 log2 while QPRT and several upstream de-novo-NAD proteins decreased
and NMNAT3 increased independently. The combined evidence therefore makes a quinolinate-utilization
bottleneck or NAD-redistribution model directly testable and ranks quinolinic acid first for authentic-
standard confirmation; it does not establish pathway direction or flux.

Ascorbate showed the largest paired effect (+5.407 log2; 32/34 concordant pairs), embedded in a source-
reproduced antioxidant-pool pattern. Its independent protein context was mixed (GSR/G6PD/TKT/TALDO1
decreased, TXNRD1 increased), supporting compensatory redox remodeling rather than a uniform pathway-
activation claim. ADP showed a strong family-level paired increase (+2.400 log2; 33/34 concordant pairs)
and a source-reproduced phosphorylated-nucleotide context, but BioAware correctly abstained because ADP
is a high-degree currency hub and the frozen protein panel supplied no ADP-specific bridge. Accordingly,
ADP is retained as a nucleotide-pool sentinel rather than an exact metabolite or pathway-specific result.
Across all four candidates, the number of new exact metabolite claims remains zero.

We additionally tested whether the four priority effects behaved as one coordinated patient-level
module. Across the 34 paired patients, none of the six pairwise Spearman associations passed the frozen
joint gate requiring |rho|>=0.35, BH q<0.10, a cluster-bootstrap interval excluding zero and
leave-one-patient sign stability. ADP and the ADP-ribose family showed the strongest association
(rho=0.373, BH q=0.101, bootstrap 95% CI 0.008-0.663), but narrowly missed the multiplicity threshold.
The ADP-ribose--quinolinate association was similar in magnitude (rho=0.365, q=0.101) but its bootstrap
interval crossed zero. The abundance axes are therefore coherent at the group-direction level; the data
do not establish a shared patient-level metabolic module.

We next tested whether the four patient-specific effects tracked technical variables recorded in the
public acquisition workbook. Tissue amount and injection order were complete for all 34 pairs, whereas
clinical stage, smoking, sex and tumor purity were not available. Across 16 fixed Spearman tests
(four priorities by tissue-mass ratio, mean injection position, signed tumor--normal injection order and
absolute paired injection gap), none passed the joint gate after BH correction, patient bootstrap and
leave-one-patient analysis. The minimum BH q value was 0.378. The largest association was between the
quinolinate effect and signed tumor--normal injection order (rho=-0.387, q=0.378); it is retained as a
sensitivity warning, not as evidence of technical confounding. Tumor was injected after normal in 17
pairs and before normal in 17. Thus, no recorded technical factor explained a frozen priority effect,
although unmeasured technical or clinical confounding remains possible.

Because the public acquisition overview lacked clinical fields, we separately inspected the source-paper
supplement and recovered an objective smoking-exposure proxy for every patient. Table S4 identified 11
cotinine-classified smokers and 23 non-smokers, with quantitative tumor-tissue cotinine available
for all 34 pairs; Table S1 supplied age, sex, BMI and stage. Under the preregistered three-arm audit, none
of the four priorities passed the joint smoking-sensitivity gate. The smallest BH q values were 0.657 for
the smoker-versus-non-smoker comparison, 0.800 for continuous log2 cotinine and 0.896 for the age/sex/BMI/
stage-adjusted HC3 model. Thus, the four effects showed no strong evidence of being driven by objective
smoking exposure in this cohort. This null sensitivity result does not establish absence of smoking
confounding and does not strengthen metabolite identity.

## Figure legends

### Extended Data Figure 1. Same-universe annotation recovery and source-table-absent evidence funnel

All analytical percentages use the same phenotype-blind universe of 263 QC/blank/dilution-qualified
precursor–retention-time families. Source-table overlap is a reconstructed m/z–RT recovery metric,
official DreaMS is constrained candidate coverage, DreaMS–P2b agreement is model concordance, and the
full gate is Level-2/connectivity-family evidence yield; none is annotation accuracy. The second panel
tracks the 221 source-table-absent families through candidate, agreement, evidence, abundance-robust and
priority stages.

### Figure 1. Phenotype-blind recovery and cross-platform validation of LCNEC dark metabolites

Analytical filtering of raw HSST3n precursor–retention-time families, known-feature positive-control
recovery and cross-platform comparison of 12 consistency-filtered hypotheses. The diagonal denotes equal
effect magnitude; colors encode direction. Reported rho is Spearman correlation across the 12 unique
connectivity hypotheses.

### Figure 2. Patient-level abundance consistency of four author-unreported hypotheses

Per-mass tumor-to-adjacent log2 fold changes for all 34 patients. Points are patient pairs, horizontal
lines show the mean, and the dashed line denotes no change. Annotation labels explicitly distinguish
compound-level Level-2 from connectivity-family hypotheses.

### Figure 3. Abundance evidence map with network-context abstention

Measured abundance directions for phosphorylated nucleotide/NAD-related, free nucleoside/base and
antioxidant-pool hypotheses. `[R]` marks source-atlas metabolites reproduced across platforms, `[N]`
marks author-unreported priority hypotheses, and `[H]` marks another Level-2 or family hypothesis.
The BioAware inset supplies reaction-network context only; ADP is marked as a currency-hub abstention.
Box placement does not represent reaction direction, flux or causality.

### Figure 4. Full pooled-QC query/library mirror spectra for the four priority hypotheses

Full mirror spectra for the pooled-QC query and selected library reference. Matched fragments are
highlighted and unmatched peaks remain visible. Precursor formula error, DreaMS similarity, direct
fragment count, intensity coverage and classical spectrum similarity are shown for each hypothesis.

### Figure 5. Pre-registered independent LCNEC proteogenomic context panel

All 22 frozen proteins are shown without post-outcome replacement. Panel A reports pure-LCNEC paired
tumor-minus-NAT effects for 80 patients with patient-bootstrap 95% intervals; filled points additionally
pass the pre-registered 22-test BH and direction-stability gates. Panel B reports the exploratory
combined-minus-pure tumor contrast and recovers the source article's combined-LCNEC pentose-phosphate
context. Crosses denote proteins not quantified in the deposited matrix. Protein abundance is pathway
context, not metabolite replication, identity confirmation, enzyme activity, flux or causality.

### Figure 6. Multi-cohort evidence triangulation and explicit claim boundaries

The four priority hypotheses are compared across paired abundance, spectral/formula evidence,
source-atlas axis reproduction, phenotype-blind BioAware context and the pre-registered independent
protein panel. Dark-blue cells denote direct or strong evidence, pale-blue cells contextual evidence,
gray cells unavailable or abstained evidence, and pink cells an explicit identity limitation. The matrix
is a synthesis of frozen outputs rather than a new enrichment test. Independent proteins provide pathway
context only; every metabolite remains MSI Level 2 or a connectivity-family hypothesis.

### Figure 7. Three-cohort triangulation of abundance, protein context and genomic heterogeneity

Panel A shows paired tumor-to-adjacent abundance effects for the four Level-2/connectivity-family
hypotheses in 34 LCNEC patients. Panel B shows the 13 proteins passing the frozen primary gate in 80
independent pure-LCNEC tumor/NAT pairs. Panel C shows the four genes passing BH correction in the clean
external genomic contrast (22 STK11/KEAP1-altered versus 17 RB1-altered tumors); all three frozen axes
passed the multivariate genomic-stratum gate. The three panels answer different questions and must not
be interpreted as cross-platform measurements of the same molecular quantity or as evidence of flux.

### Extended Data Figure 2. Patient-level covariation of the four priority abundance effects

Spearman correlations were calculated from the 34 patient-specific tumor-to-adjacent log2 effects for
the ADP family, ADP-ribose family, ascorbate and quinolinate. Six pairwise tests were corrected together
by Benjamini-Hochberg; uncertainty was assessed by patient bootstrap and sign stability by leave-one-
patient-out analysis. No pair passed the frozen four-part gate. The figure is an internal module
diagnostic and does not validate identity, reaction direction, flux or causality.

### Extended Data Figure 3. Exploratory patient-level structure within the independent protein axes

Heatmaps show all within-axis Spearman correlations of paired tumor-minus-NAT protein effects in 80
pure-LCNEC patients. Stars denote the exploratory four-part gate after BH correction across all 46
within-axis pairs. All passing pairs occurred in the redox axis. The analysis was specified after the
primary fixed-panel result and is therefore contextual, not a new confirmatory endpoint.

### Extended Data Figure 4. Technical-confounding audit of the four frozen priority effects

Spearman correlations relate each patient-specific tumor-to-adjacent effect to the tumor/normal tissue-
amount ratio, mean injection position, signed tumor-minus-normal injection order and absolute paired
injection gap. Sixteen tests were corrected together by Benjamini-Hochberg; the fixed gate additionally
required |rho|>=0.35, a patient-bootstrap interval excluding zero and leave-one-patient sign stability.
No association passed. The public workbook did not contain stage, smoking, sex or tumor-purity metadata,
so this figure addresses recorded technical variables only.

### Extended Data Figure 5. Objective smoking-exposure sensitivity audit

The left panel shows smoker-minus-non-smoker differences in each patient-specific tumor-to-adjacent
effect using the source supplement's cotinine classification; the right panel shows Spearman correlation
with log2 tumor cotinine. Error bars are patient-bootstrap 95% intervals. Primary, continuous and
age/sex/BMI/stage-adjusted analyses were frozen before outcome calculation, and none of the four endpoints
passed the joint gate after BH correction. Tumor cotinine is an exposure proxy, not identity, flux or
causal-mechanism validation; a null association cannot exclude residual smoking confounding.

## Frozen claim boundary

The manuscript may claim algorithm-enabled Level-2/connectivity-family hypothesis generation in paired
LCNEC tissue, anchored by a separately demonstrated cross-platform-reproduced abundance program. It may
not claim that the four author-unreported hypotheses themselves were cross-platform or independently
replicated, nor claim authentic-standard-confirmed identity, ATP energy charge,
metabolic flux, enzyme activity, causal tumor dependency, clinical biomarker performance or a uniquely
resolved stereoisomer for connectivity-family assignments.

`Author-unreported` means absent from all source-atlas identity rows under the frozen exact-name/unique-
structure and supported-adduct 5-ppm neutral-mass audit. It does not mean a
chemically novel metabolite or a first report in cancer. The current claim set requires no authentic
standard because it is explicitly limited to Level-2/connectivity-family hypotheses. Same-method
authentic standards remain mandatory for any Level-1 upgrade; if only two can be prioritized,
quinolinic acid provides the most pathway-specific test and ascorbic acid provides the largest paired
effect.

## Machine-readable sources

- `data/validation/lcnec_hsst3n_manuscript_readiness/readiness_report.json`
- `data/validation/lcnec_hsst3n_manuscript_supplement/supplement_manifest.json`
- `data/validation/lcnec_hsst3n_manuscript_figures/figure_report.json`
- `data/validation/lcnec_hsst3n_annotation_benchmark_v1/report.json`
- `data/validation/lcnec_hsst3n_identity_claim_defense_v1/report.json`
- `data/external/LCNEC_proteogenomic_2026/fixed_panel_patient_audit_v1/report.json`
- `data/validation/lcnec_hsst3n_multicohort_triangulation_v1/report.json`
- `data/validation/lcnec_hsst3n_source_positive_control_identity_v1/report.json`
- `data/validation/lcnec_hsst3n_priority_patient_covariation_v1/report.json`
- `data/validation/lcnec_hsst3n_priority_formula_rivals_v1/report.json`
- `data/validation/lcnec_hsst3n_priority_technical_confounding_v1/report.json`
- `data/validation/lcnec_hsst3n_priority_smoking_confounding_v1/report.json`
- `data/validation/lcnec_hsst3n_priority_global_source_novelty_v1/report.json`

## External primary sources

- LCNEC metabolomics atlas: https://www.sciencedirect.com/science/article/pii/S0753332226003604
- Public raw LC-MS data: https://zenodo.org/records/19005638
- Independent LCNEC proteogenomics: https://pubmed.ncbi.nlm.nih.gov/42585338/
- Prior PARP1 evidence in SCLC/LCNEC: https://doi.org/10.1158/2159-8290.CD-12-0112
- Quinolinic acid and prospective lung-cancer risk: https://doi.org/10.1002/ijc.32555
- Quinolinic acid/QPRT mechanism in glioma: https://doi.org/10.1158/0008-5472.CAN-12-3831
