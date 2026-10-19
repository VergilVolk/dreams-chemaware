# TT7 event-level structure-agnostic response result

Date: 2026-10-14  
Evidence role: development / consumed, same study and same instrument  
Primary status: T1--T3 PASS in both polarities and both preprocessing worlds; unresolved-MS2 positive-mode sufficiency PASS  
Chemical interpretation status: T4 capacity FAIL

## Scientific question

Can a study-level response representation recover a prespecified genetic
epistasis relation from non-targeted LC--MS features without requiring feature
structures, and does the recovered response support a pre-frozen fragment-level
biochemical interpretation?

This is an event-level genetic perturbation qualification experiment for the
biology method. The later positive-mode spectral partition tests whether
strictly unresolved MS2-linked entities retain the genetic result. It is not
cross-study validation, a powered superiority test against known-only, or
evidence that the unified Noise/ChemAware/P2b algorithm improves the result.

## Truth-blind table construction

The MSV000096214 positive- and negative-mode raw files were processed without
reading genotype labels. Gap filling used 5 ppm, a +/-15 s file-scale RT window
and the frozen per-file MS1 Q90 floor. The final blank-filtered matrices contain:

- positive mode: 4,562 features;
- negative mode: 9,751 features;
- 49 genotype-labelled study samples in the effect stage; pooled QCs remained
  available to the table audit.

All pre-frozen G1/G2/G4/G6/G7 table gates passed in both polarities. Only after
that PASS were genotype labels opened.

The last six negative-mode files reused the completed partial matrix through
`SJ-Mutant-Flavo-7-NEG`. An exact loop-order optimization changed gap filling
from repeated full-scan traversal per target to one scan traversal updating all
targets within the same RT window. On 235 real targets, the old and new
implementations had identical missingness and maximum intensity difference
zero; runtime was 18.64 versus 0.44 seconds (42.4-fold for that check).

## T1--T3 primary result

The primary world is per-sample median-centered log2 abundance. Raw log2 is the
frozen sensitivity world. Three independent `tt7` alleles and their three
paired `tt7 x tt4` double mutants were evaluated.

### T1: double-mutant resemblance

For every double mutant, correlation with `tt4` exceeded correlation with the
mean `tt7` response.

| polarity | double | rho versus tt4 | rho versus tt7 mean | delta | feature-label permutation p |
|---|---|---:|---:|---:|---:|
| positive | `tt7_115 x tt4_11` | 0.8712 | 0.3758 | +0.4954 | 0.00050 |
| positive | `tt7_8 x tt4_11` | 0.9060 | 0.3313 | +0.5747 | 0.00050 |
| positive | `tt7_9 x tt4_11` | 0.9239 | 0.3697 | +0.5541 | 0.00050 |
| negative | `tt7_115 x tt4_11` | 0.7380 | 0.3936 | +0.3444 | 0.00050 |
| negative | `tt7_8 x tt4_11` | 0.7626 | 0.3635 | +0.3991 | 0.00050 |
| negative | `tt7_9 x tt4_11` | 0.7838 | 0.4276 | +0.3561 | 0.00050 |

The corresponding raw-log2 deltas remained positive in all six tests.

### T2: exact held-out genotype-cluster test

In each polarity and in both preprocessing worlds, the three true `tt7`
alleles were the top assignment among all 20 genotype-cluster assignments:

- observed statistic: 1.0;
- observed rank: 1/20;
- exact right-tail p: 0.05;
- minimum attainable p under this design: 0.05.

The p value is therefore at the design resolution boundary and must not be
reported as stronger than 0.05.

### T3: topology consistency

All three paired epistasis deltas were positive in both polarities. In the
primary world they were +0.490/+0.568/+0.580 in positive mode and
+0.347/+0.408/+0.376 in negative mode. Median pairwise correlation among the
three `tt7` alleles was 0.902 in positive mode and 0.854 in negative mode;
their similarity to `tt3` was substantially smaller (0.308 and 0.178).

Both polarities passed T1, T2 and T3 in the primary and raw-log2 sensitivity
worlds. The effect report therefore legitimately opened T4.

## T4 fragment-level interpretation did not have a valid control comparison

The first T4 output is invalid and retained only as implementation history. It
treated raw zero intensities as valid for the Mann--Whitney eligibility mask;
log2 then converted those zeros to NaN, and the BH cumulative minimum propagated
NaN to the entire FDR vector. Version 2 defines eligibility after log2 and maps
any non-finite p value to one; no threshold was changed.

Corrected T4 v2 found:

| polarity | F | raw-log2 F | matched-control pool | F linked to MS2 | F with frozen glycosidic loss |
|---|---:|---:|---:|---:|---:|
| positive | 1,079 | 386 | 0 | 503 | 171 |
| negative | 0 | 0 | 20 | 0 | 0 |

The positive mode has many candidate F features but no unchanged controls that
meet the pre-frozen all-genotype stability rule. The negative mode has controls
but no F set. Consequently neither polarity can form the F-versus-C Fisher
table. Glycosidic-loss enrichment and the +15.9949 Da branch comparison are
not testable. The count of 171 spectra with a listed neutral loss is descriptive
only and does not establish flavonoid identity or pathway mechanism.

Formal T4 decision: capacity FAIL; do not relax F/C definitions post hoc.

## Unresolved MS2 entities retain the prespecified genetic topology

A separate truth-blind capacity audit linked cached MS2 precursor observations
to the final consensus features using the frozen RT warps, 10 ppm, +/-6 s and
one deterministic nearest-feature assignment per MS2 event. It did not read
genotype labels or effect estimates.

The two polarities had sharply different usable capacity:

| polarity | consensus features | cached MS2 events | features with >=1 MS2 having >=6 raw peaks | pooled-QC linked | decision |
|---|---:|---:|---:|---:|---|
| positive | 4,562 | 584,132 | 2,004 (43.93%) | 952 | proceed |
| negative | 9,751 | 578,493 | 36 (0.37%) | 14 | capacity fail |

The negative result was not rescued by changing tolerances. In the first
negative raw file, only 306/11,078 MS2 precursors had any 10-ppm match to its
2,387 detected MS1 features and none also matched within 6 s. Negative-mode
MS2 and quantitative feature coordinates therefore do not support this test.

Positive-mode extraction rescanned the 52 nonblank files once. Exact precursor,
RT and raw-peak-count checkpoint checks passed for the selected spectrum
ordinals. Of the 2,004 linked features, 1,998 representative spectra retained
at least six peaks after the frozen cleaning contract; 947 representatives came
from pooled QC and 1,051 from nonblank files selected without genotype labels.

Against the already-consumed GNPS Gold/Silver 10-ppm library, using WSE >=0.70
and at least six matched peaks:

- 10/1,998 had strict same-library support;
- 1,988/1,998 were unresolved under this contract.

"Unresolved" here means no match under this specific positive-mode library and
threshold. It is not evidence that a feature is a novel molecule; unsupported
adducts and incomplete public-library coverage are included.

The spectral partition was frozen before genotype labels were reopened. The
unchanged T1--T3 tests were then run on fixed worlds:

| world | features | primary median-centered log2 | raw-log2 sensitivity |
|---|---:|---|---|
| all cleaned MS2-linked | 1,998 | PASS | PASS |
| strict library-supported | 10 | abstain, minimum 30 | abstain |
| unresolved | 1,988 | PASS | PASS |
| pooled-QC representative only | 947 | PASS | PASS |
| pooled-QC representative and unresolved | 939 | PASS | PASS |

In the unresolved primary world, the three double-mutant T1 deltas were
+0.5120, +0.6092 and +0.6100. Every conditional feature-label permutation had
`p=0.00050`, and every conditional bootstrap interval was strictly positive
([0.4576, 0.5481], [0.5556, 0.6471], [0.5610, 0.6532]). T2 again placed the
true three-allele assignment at rank 1/20 with exact `p=0.05`, the minimum
attainable value. The three T3 paired epistasis deltas were +0.4990, +0.6132
and +0.6410. Raw-log2 sensitivity retained all gates.

Because choosing a representative from biological files could in principle
inherit DDA abundance selection, a post-result sensitivity restricted the
analysis to features whose selected spectrum came from pooled QC. In the 939
pooled-QC unresolved features, primary T1 deltas remained +0.4189, +0.5348 and
+0.5376; all three permutation p values were 0.00050 and all three conditional
bootstrap intervals were strictly positive ([0.3561, 0.4732], [0.4553,
0.5788], [0.4737, 0.5936]). T2 remained rank 1/20 with exact p=0.05 and all
T3 paired deltas remained positive. Raw-log2 sensitivity also passed. This
control was added after the first unresolved result and is therefore labelled
a strengthening sensitivity, not an independent preregistered confirmation.

Formal decision: unresolved positive-mode MS2 entities are sufficient to
recover the prespecified TT7 genetic topology in this same-study positive
control. Because strict known-only contains only 10 features, this experiment
does not estimate superiority or incremental gain over known-only. The correct
claim is unresolved-information sufficiency, not unknown-information increment.
Acquisition order was genotype-blocked, so study-wide run-order confounding
remains a limitation even though the pooled-QC spectral-selection sensitivity
passed.

## Scientific meaning for the general method

This is the strongest real event-level result so far for the biology branch:
a response object constructed without structures recovers a predeclared
genetic epistasis topology across three independent alleles, three double
mutants, two polarities and two preprocessing worlds.

It establishes that non-targeted response patterns can encode falsifiable
biological relations before metabolite identification, and the positive-mode
result is carried by a large fixed set of MS2-linked entities unresolved by a
strict public-library contract. It does not establish that unresolved entities
outperform a sufficiently large known-only panel, that the relation transfers
across studies or instruments, or that fragment chemistry or the unified
spectral algorithm explains the signal. Those are separate gates.

The result should therefore serve as an event-level positive control for the
Metabolomic Contrast Atlas method, not as a TT7-specific redefinition of that
method. The next high-value experiment is to transport a response relation with
equally strong prior truth across independent sources and measure whether
spectral relationship evidence adds calibrated correct retrieval beyond
mass-only and known-only controls.

## Machine-readable evidence

- `data/external/msv000096214/consensus/positive/report.json`
- `data/external/msv000096214/consensus/negative/report.json`
- `data/validation/msv000096214_table_audit_20261014_v4/report.json`
- `data/validation/msv000096214_effect_stage_20261014_v1/report.json`
- `data/validation/msv000096214_posthoc_t4_20261014_v2/report.json`
- `data/validation/msv000096214_ms2_linkage_capacity_20261014_v1/report.json`
- `data/validation/msv000096214_positive_ms2_partition_20261014_v1/report.json`
- `data/validation/msv000096214_unresolved_response_increment_20261014_v2/report.json`
