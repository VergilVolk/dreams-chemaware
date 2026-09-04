# LCNEC manuscript-ready Discussion draft (evidence frozen 2026-09-01)

## Principal finding

Reanalysis of the public HSST3n arm of a paired LCNEC metabolomics atlas recovered a reproducible
analytical space substantially larger than the source table and converted part of that space into
explicitly bounded biological hypotheses. The central result is not that every source-table-absent ion
has been identified. Rather, a phenotype-blind analytical filter, DreaMS-based candidate retrieval,
classical fragment evidence, a frozen candidate expert and paired-abundance tests produced four
author-unreported Level-2 or connectivity-family priorities from 221 source-table-absent families. The
same workflow first recovered 12 source-reported cross-platform signals with complete directional
concordance. This positive-control-first design separates analytical headroom from biological discovery
and prevents annotation coverage from being presented as annotation accuracy.

The author-unreported designation was also checked against all 1,054 reported identity rows across the
source atlas, rather than inferred only from HSST3n-table absence. No priority matched by exact normalized
name, uniquely resolved structure or supported-adduct neutral mass within 5 ppm; 1,050 source rows were
mass-reconstructable. This materially strengthens the source-study increment while remaining narrower
than chemical novelty, because the source-name structure resolver was incomplete and the four candidate
identities still lack authentic-retention-time confirmation.

## What the algorithm adds beyond the source analysis and official DreaMS

The source article reported metabolite names but did not provide a common detected-feature denominator.
Consequently, its global annotation rate cannot be reconstructed. On our shared universe of 263
QC/blank/dilution-qualified precursor-RT families, source-table overlap was 42/263, official DreaMS
candidate coverage was 158/263 and the full multi-evidence workflow retained 66/263. These quantities
measure different stages and must not be compared as if all were accuracies. The clearest identity
benchmark available within the source data was a conservative subset of 19 source-matched names that
mapped uniquely to one local-HMDB IK14. Official DreaMS and the full workflow were concordant for 17/19.
The two same-formula isomer errors were the only discordant cases, and both survived the full evidence gate. The
workflow therefore increases usable evidence and biological reach, but does not solve the fundamental
isomer problem of tandem mass spectrometry.

This differs from the DreaMS Atlas strategy, which organizes repository-scale spectra into a molecular
network. Here, the learned representation is embedded in a study-level evidence ladder: analytical
quality, source-positive controls, paired effects, orthogonal spectral evidence, network abstention and
an independent frozen protein panel. The contribution is thus not a smaller imitation of a global
atlas. It is a falsifiable framework for deciding which dark features can enter a disease-mechanism
hypothesis and which must remain unknown.

## Biological interpretation

The four priorities have distinct roles. The ADP-ribose connectivity family has the strongest
cross-omics context because its paired abundance increase coincides with independent PARP1 and PARP2
increases in pure LCNEC. This supports follow-up of PARP-associated nucleotide turnover, but PARP
expression in high-grade neuroendocrine lung cancer is already established and is not itself novel. The
incremental observation is the author-unreported metabolite-family signal linked to that established
protein context.

Quinolinate is the highest-value chemical confirmation target. Its local increase, together with
independently lower QPRT and a mixed de-novo-NAD protein pattern, motivates a quinolinate-utilization or
NAD-redistribution hypothesis. It does not establish increased de-novo-NAD flux: upstream IDO1, KYNU and
HAAO were not coherently increased, the samples are static tissue snapshots and the metabolite identity
remains Level 2. Ascorbate showed the largest paired abundance effect, but the independent redox proteins
were directionally mixed. The defensible interpretation is compensatory antioxidant-pool remodeling,
not uniform pentose-phosphate activation. ADP remains a family-level nucleotide-pool sentinel because it
is a high-degree reaction-network currency metabolite and lacks a specific independent bridge.

The independent redox pattern nevertheless contained patient-level structure. In an explicitly
post-primary all-pairs audit, 12 of 46 within-axis protein-effect correlations passed a common
exploratory gate, and all 12 occurred in the redox axis. PPP enzymes, GSR and TXNRD1 therefore varied
coordinately across patients despite their mixed mean shifts. This supports heterogeneous compensatory
remodeling rather than unrelated protein noise, but it remains exploratory. PARP1--PARP2 narrowly missed
the common gate and the de-novo-NAD axis did not show comparable within-axis coordination.

A second external LCNEC cohort adds a different kind of support. Genomic groups were defined without
using expression outcomes, and all three frozen axes differed across the clean STK11/KEAP1-altered and
RB1-altered strata after stage-restricted sensitivity analysis. RB1-altered tumors had higher NMNAT1,
NMNAT3 and PARP1, whereas STK11/KEAP1-altered tumors had higher TKT at the single-gene multiplicity
threshold. This provides a plausible basis for the mixed protein directions: the candidate contexts are
not expected to behave as one uniform LCNEC program across genomic subtypes. The analysis remains tumor-
only and cross-sectional, however. It supports external pathway-context heterogeneity, not replication
of the local tumor-to-adjacent metabolite directions.

Leave-one-gene-out sensitivity further limits the interpretation. The redox contrast survived removal
of every individual gene, whereas the NAD contrast fell below the effect-size gate without NMNAT3 and
the ADP-ribose contrast failed without PARP1. Accordingly, external redox heterogeneity is multi-gene;
the other two external contexts are best described as NMNAT3- and PARP1-anchored rather than uniformly
coherent pathway programs.

The four effects should not be collapsed into one patient-level module. None of six pairwise effect
correlations passed the frozen joint covariation gate after multiplicity correction. ADP and ADP-ribose
were suggestively correlated, but the BH q value was 0.101. The biological synthesis therefore operates
at the level of reproducible group directions and independently supported contexts, not a confirmed
co-regulated patient program.

The recorded acquisition variables do not provide an obvious alternative explanation. None of 16 fixed
relationships between the four priority effects and tissue amount or injection order passed the common
technical-confounding gate. Quinolinate showed the largest unadjusted relationship with signed paired
injection order, but its BH q value was 0.378. This supports retaining quinolinate for validation while
explicitly balancing injection order in the next experiment. The public acquisition overview contains no
stage, smoking, sex or tumor-purity fields, but the source-paper supplement did permit a separate objective
smoking audit. None of the four effects passed the preregistered joint gate across cotinine-classified,
continuous-cotinine and age/sex/BMI/stage-adjusted analyses. This reduces concern about a large smoking-
stratification artifact, but it does not prove the cohort is free of smoking or other clinical confounding.

## Position relative to current computational metabolomics

Recent methods illustrate three complementary routes beyond single-spectrum library search. DreaMS
learns general spectral representations and enables repository-scale molecular networking. MIST encodes
fragment-formula priors and demonstrates disease-cohort candidate generation. NetID uses global
peak-relationship optimization and validates new metabolites with isotope tracing. DeepMet and related
structure-generative workflows expand the candidate structure universe, but their strongest chemical
claims still rely on authentic standards and orthogonal measurements. Our study does not compete with
de-novo structure generation. It addresses a different translational gap: how to carry foundation-model
candidates into a paired human-tissue study without turning spectral similarity, pathway membership or
external protein abundance into false identity confirmation.

The two retained same-formula errors are particularly informative for future method development. They
show that agreement between a representation model, a downstream candidate expert and conventional peak
matching can remain correlated rather than independent. Noise/ChemAware embedding development should
therefore be evaluated against these concrete isomer transitions and other locked real errors, not only
against candidate coverage or randomly perturbed spectra.

## Why these results are publishable without claiming Level 1

Metabolomics reporting standards permit putative Level-2 annotations when the evidence and limitations
are explicit. A computational biology application can therefore report these four rows as prioritized
hypotheses if it does not call them confirmed metabolites. The paper's strongest current chain is:

1. an auditable, phenotype-blind raw-data universe;
2. recovery of source biology before nomination of source-table-absent features;
3. same-denominator comparison of source overlap, DreaMS coverage and full evidence yield;
4. direct spectral and formula evidence for four prioritized hypotheses;
5. stable paired abundance effects;
6. phenotype-blind reaction-network abstention;
7. a pre-registered independent protein-context test; and
8. an external expression-independent genomic-stratum audit of the same frozen axes.

This chain supports algorithm-enabled biological hypothesis generation. It does not support four newly
identified exact metabolites, a clinical biomarker, metabolic flux or a therapeutic dependency.

## Limitations and decisive validation experiments

The largest remaining weakness is chemical identity. Same-method authentic-standard RT and MS/MS are
required for Level 1. If only two compounds can be tested, quinolinic acid should be first because it
tests the most pathway-specific model, followed by ascorbic acid because it has the largest local effect.
The ADP and ADP-ribose findings require isomer-aware chromatographic or MSn work before exact structures
can be claimed.

An independent LCNEC metabolomics cohort would test abundance replication, which neither the protein
nor tumor-only transcriptomic cohort can provide. Stable-isotope tracing or perturbation would be needed to distinguish accumulation from
production, impaired utilization or compartment redistribution. Those experiments are not prerequisites
for the current Level-2 computational application, but they are prerequisites for claims of flux,
enzymatic mechanism or dependency.

## Frozen novelty statement

The defensible novelty is a positive-control-calibrated application of a mass-spectral foundation model
and orthogonal evidence stack that recovers source-table-absent LCNEC metabolite hypotheses, exposes its
own same-formula failure mode, and links selected hypotheses to independently tested protein and
genomic-stratum contexts.
It is not the first observation of PARP biology, oxidative stress or quinolinate in cancer, and it does
not claim discovery of chemically novel compounds.

## Primary methodological comparators

- DreaMS: https://doi.org/10.1038/s41587-025-02663-3
- MIST: https://doi.org/10.1038/s42256-023-00708-3
- NetID: https://doi.org/10.1038/s41592-021-01303-3
- DeepMet: https://doi.org/10.1038/s41586-025-09969-x
- BUDDY: https://doi.org/10.1038/s41592-023-01850-x
