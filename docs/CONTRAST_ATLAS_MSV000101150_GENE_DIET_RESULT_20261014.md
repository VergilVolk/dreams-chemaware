# MSV000101150 structure-agnostic gene-by-diet response result

Date: 2026-10-14  
Status: dual-polarity development result; entity-level cross-polarity concordance failed  
Evidence role: development / consumed; not sealed external validation

## Decision

Both analytical polarities provide a real factorial application of the
response-object method in this study. Pooled-QC-defined, structure-independent
chemical spaces detect a TMEM-120 genotype-by-HFD interaction, survive explicit
acquisition-order adjustment and leave-one-sample-out analysis, and retain the
interaction after effect-blind WSE spectral-family collapse. Large pooled-QC-MS2
subsets retain the global response despite receiving no strict support from the
applicable frozen references.

This is a genuine biological-method result, but its exact boundary matters:

- it establishes within-study structure-unresolved response **sufficiency**;
- it does not establish superiority over known-only, because known-only has no
  eligible negative-mode denominator and correctly abstains;
- it does not identify the four leading entities, establish a lipid mechanism,
  prove cross-study transfer, or measure a Noise/ChemAware/P2b increment;
- positive polarity is a same-sample analytical replication of the global
  interaction, not an independent biological cohort;
- effect-blind neutral-mass/RT correspondences show attractive raw cross-polarity
  concordance, but it does not exceed a synchronized paired-sample null and is
  therefore a formally failed entity-level replication claim.

## Frozen data and leakage boundary

Dataset: MSV000101150, negative Q Exactive arm. The 15 / 15 RAW files total
1,120,490,267 bytes: 12 biological samples in a balanced 2 genotype x 2 diet x
3 biological-replicate design and three pooled QCs. Every file was converted to
mzML. All files and all results are development / consumed.

Chemical coordinate discovery, EIC rules, QC reproducibility and MS2 linkage
were fixed without reading the genotype-by-diet effects. The three pooled QCs
alone froze OpenMS `noise_threshold_int=3200`; they yielded 10,024, 10,314 and
10,419 detected features (median 10,314; denominator 3 QCs). No lower threshold
was opened.

The effect-blind feature chain was:

| Stage | Denominator | Retained | Status |
|---|---:|---:|---|
| Features detected in at least 2 / 3 pooled QCs | 7,673 | 7,673 | effect-blind |
| Same coordinates after common 5-ppm, +/-15-s EIC; support in at least 2 QCs; QC CV <=30% | 7,673 | 2,477 | effect-blind |
| Primary complete-case biological matrix | 2,477 | 1,445 observed in 12 / 12 samples | labels then opened |
| QC-MS2 representative with at least six cleaned peaks | 2,477 | 1,660 | effect-blind |
| QC-MS2 representatives also complete in 12 / 12 samples | 1,660 | 1,173 | labels then opened |

No blank was deposited, so no blank subtraction or contaminant exclusion is
claimed. Median centering makes this a relative-composition analysis; it cannot
measure total lipid-pool expansion.

## Primary factorial result

The prespecified response was

`(hj50_HFD - hj50_normal) - (N2_HFD - N2_normal)`.

On 1,445 / 2,477 QC-qualified features complete in all 12 biological samples,
the global mean interaction t-squared was 4.26477255. A Freedman--Lane test
under the additive genotype-plus-diet null, using fixed seed 20261014 and 4,999
permutations, gave **p=0.0032**.

Feature-level results show why a global response object is preferable to a list
of nominal peaks:

- parametric BH FDR <=0.05: 48 / 1,445;
- pointwise permutation BH FDR <=0.05: 0 / 1,445;
- max-T family-wise p <=0.05: 4 / 1,445.

Thus the global interaction and four exceptionally strong coordinates are
supported, while a longer list of 48 should not be presented as 48 confirmed
metabolites.

The four max-T features were all QC-MS2 linked and lacked even a 10-ppm mass
candidate in the frozen negative reference:

| Feature | m/z | RT (s) | N2 HFD effect (log2) | hj50 HFD effect (log2) | Interaction | max-T p |
|---|---:|---:|---:|---:|---:|---:|
| NEG_F000498 | 932.564652 | 548.107 | -1.1168 | -0.5294 | +0.5874 | 0.0086 |
| NEG_F000749 | 869.556093 | 527.377 | -1.6264 | -0.0665 | +1.5598 | 0.0124 |
| NEG_F000684 | 736.529217 | 610.762 | -0.6006 | -0.0477 | +0.5530 | 0.0242 |
| NEG_F000935 | 909.551219 | 492.591 | -0.6435 | +0.4059 | +1.0493 | 0.0326 |

All four therefore express the same descriptive motif: an HFD-associated decline
in N2 is attenuated or reversed in `hj50`. This coherence is useful for selecting
falsifiable follow-up targets, but does not establish that the four ions are TAGs,
independent molecules, or members of one pathway.

## Acquisition-order sensitivity

The mzML timestamps put pooled QCs at run ranks 0, 7 and 14. The four biological
cells were interleaved rather than acquired as four intact blocks. The global
interaction remained significant in post-hoc robustness models:

| Model | Global mean t2 | Permutation p | Effect correlation with primary | Sign agreement |
|---|---:|---:|---:|---:|
| Primary replay | 4.2648 | 0.0032 | 1.0000 | 1.0000 |
| + linear run order | 4.8287 | 0.0064 | 0.9936 | 0.9737 |
| + linear and quadratic run order | 5.4301 | 0.0126 | 0.9922 | 0.9696 |

These models reduce concern about smooth acquisition drift. They were specified
after the primary result and are robustness analyses, not a new confirmatory
endpoint; they cannot remove unrecorded extraction batches or all technical
confounding.

No single animal file explains the result. In 12 leave-one-biological-sample-out
fits on the unchanged 1,445-feature universe, every 11-sample model retained a
global permutation p<=0.05 (1,999 permutations per fit; maximum p=0.0240).
Interaction-effect correlations with the 12-sample estimate were all at least
0.9089 and sign agreement was at least 0.8865. This is a same-experiment sample
influence analysis, not an independent cohort replication.

## Reference-supported versus unresolved worlds

The frozen GNPS Gold/Silver benchmark used for earlier positive-mode work is
positive-only. It supplied 0 negative-polarity candidates for 1,660 / 1,660
queries and was therefore rejected as an inapplicable partition rather than
used to manufacture an unknown-space win.

The polarity-applicable fallback was the local `unified_v3` negative GNPS-source
reference: 29,564 spectra, including 29,562 with InChIKey and 29,564 with an
explicit negative adduct. At the unchanged 10-ppm, 0.02-Da, WSE>=0.70 and at
least-six-matched-peak contract:

- 204 / 1,660 QC-MS2 queries had at least one 10-ppm candidate;
- 0 / 1,660 passed strict spectral support;
- the best candidate reached WSE 0.699095 with only five matched peaks, so the
  threshold was not relaxed post hoc.

Consequently known-only has 0 complete features and abstains below the frozen
minimum of 30. The 1,173 complete QC-MS2-linked, unresolved-against-this-reference
features retain the global interaction: mean t-squared 4.47591886,
Freedman--Lane 4,999-permutation **p=0.0032**.

This is unresolved-space sufficiency, not an estimable unknown-over-known
increment. "Unresolved" means no match under these references and thresholds;
it is not proof of a novel molecular structure.

### WSE spectral-family redundancy stress test

One biological change can generate correlated ions or repeated spectra. A
post-hoc, effect-blind collapse therefore connected complete QC-MS2 features
only when their warped RT differed by at most 15 seconds and their WSE was at
least 0.645, the pre-existing development/consumed GNPS cross-formula q99.9
threshold. One representative per connected component was chosen by lowest QC
CV without consulting its effect.

- input: 1,173 complete QC-MS2 features;
- graph: 213 WSE edges, 999 spectral families, 113 multi-member families,
  largest family 7;
- one-representative result: 999 features, global mean t-squared 4.38452327,
  4,999-permutation **p=0.0042**.

The interaction is therefore not explained solely by counting repeated highly
similar spectra. This is a redundancy sensitivity, not evidence that each WSE
component is one molecule or one biochemical pathway.

## Positive-polarity unchanged-protocol result

The positive Q Exactive arm contains the same 12 biological samples and three
pooled QCs in a separate analytical polarity: 15 / 15 RAW files totaling
1,756,221,148 bytes. QC-only calibration again selected the highest predeclared
threshold, `noise_threshold_int=3200`; the three QCs yielded 18,875, 19,386 and
18,907 features (median 18,907; denominator 3 QCs).

The effect-blind chain produced 13,835 coordinates detected in at least 2 / 3
QCs, 3,756 after the common 5-ppm, +/-15-s EIC and QC-CV<=30% contract, and
2,545 observed in all 12 biological samples. On those 2,545 features, the
unchanged interaction had global mean t-squared 3.95292466 and a 4,999-permutation
Freedman--Lane **p=0.0034**. Feature-level counts were 46 / 2,545 by parametric
BH FDR, 0 / 2,545 by pointwise permutation BH FDR, and 3 / 2,545 by max-T FWER.

| Feature | m/z | RT (s) | N2 HFD effect (log2) | hj50 HFD effect (log2) | Interaction | max-T p |
|---|---:|---:|---:|---:|---:|---:|
| POS_F000197 | 936.799775 | 1002.460 | +0.1004 | +0.6705 | +0.5701 | 0.0412 |
| POS_F001388 | 702.543442 | 623.383 | +1.5247 | +2.2754 | +0.7507 | 0.0088 |
| POS_F001677 | 208.181657 | 986.062 | +0.6807 | +0.1547 | -0.5261 | 0.0292 |

Run-order robustness remained positive: linear-order p=0.0050 and
linear-plus-quadratic p=0.0120, with interaction-effect correlations 0.9917 and
0.9914 to the primary estimate and sign agreement 0.9603 and 0.9599. All 12
leave-one-biological-sample-out fits retained p<=0.05 (1,999 permutations each;
maximum p=0.0125), with minimum effect correlation 0.9465 and minimum sign
agreement 0.9073. These are post-hoc robustness analyses in the same experiment.

Of 3,756 QC-qualified coordinates, 2,048 had a pooled-QC MS2 representative
with at least six cleaned peaks, 1,799 were linked in at least two QCs, and
1,568 were complete in all 12 samples. Against the positive frozen GNPS
Gold/Silver reference, 423 / 2,048 had at least one 10-ppm candidate but 0 / 2,048
passed the unchanged joint WSE>=0.70 and at-least-six-matched-peak rule. Thus
known-only again abstains; the 1,568 unresolved-against-this-reference features
retain the interaction (mean t-squared 4.22018022, p=0.0034). This is sufficiency,
not an unknown-over-known comparison or a claim that no public library could
identify the entities.

The positive WSE redundancy stress test was deliberately aggressive: 1,568
complete QC-MS2 features, 1,885 edges, 937 connected families, 122 multi-member
families and a largest transitive component of 286. One effect-blind representative
per family still retained the global interaction (937 features; mean t-squared
4.26758904; p=0.0034). The large component also warns that WSE-connected families
must not be interpreted as single molecules or pathways.

## Cross-polarity entity-level audit: failed

To distinguish two marginally significant chemical spaces from replication of
the same response entities, complete features were paired without effects by
mutual-nearest neutral mass and RT. The frozen rule used 10 ppm, 15 seconds and
explicit common adducts. It produced 486 mutual pairs from 1,102 candidate pairs;
median absolute mass error was 1.169 ppm and median absolute RT difference was
0.661 seconds.

The raw paired effects looked strong: Pearson r=0.6644, Spearman rho=0.7093 and
direction agreement 77.78%. However, a stricter null applied the same
Freedman--Lane residual-row permutation to both polarities, preserving the fact
that the same 12 animals and extracts generated both measurements. The observed
values did not exceed that paired-sample null: Pearson p=0.2318, Spearman
p=0.1430 and sign-agreement p=0.2292 (4,999 permutations; seed 20261014).

The entity-level cross-polarity claim therefore **fails**. The raw concordance
is explainable by paired-sample covariance. The defensible result is two
polarity-specific global responses in the same experiment, not independent or
entity-resolved replication.

An immediate post-result capacity probe also tested whether the positive-mode
response could be transported into the already-consumed MSV000088023 mouse
plasma HFD candidate panel under the unchanged 10-ppm, 0.02-Da, WSE>=0.70,
at-least-six-peak reciprocal contract. Among 1,568 complete query spectra and
212 candidate features, only four reciprocal edges were available, below the
frozen minimum of 12. The cross-study query therefore abstains; no observed rank
is admissible and no threshold was relaxed. This closes the cheap retrospective
reuse route rather than generating another underpowered retrieval result.

## Scientific next gate

Further retrospective matching or threshold adjustment has low value. The next
experiment must use independently collected N2 and `hj50` animals under normal
and HFD conditions, with blinded targeted re-acquisition of the four negative
and three positive max-T coordinates plus predeclared null coordinates. It must
measure absolute or internal-standard-normalized responses where possible and
test the interaction before opening identities. Only after response replication
should authentic standards or orthogonal fragmentation be used to promote an
entity to structural or mechanistic interpretation.

For the broader Contrast Atlas method, this dataset is now a positive within-study
response-object example and a negative warning about same-sample cross-view
validation. The next public-data gate remains a genuinely independent study-level
contrast retrieval, not another analysis of these same 12 samples.

## Machine-readable evidence

- `data/external/msv000101150/qc_feature_matrix/NEG/report.json`
- `data/validation/msv000101150_gene_diet_interaction_20261014_v1/report.json`
- `data/validation/msv000101150_run_order_sensitivity_20261014_v1/report.json`
- `data/validation/msv000101150_leave_one_sample_out_20261014_v1/report.json`
- `data/validation/msv000101150_neg_qc_ms2_partition_20261014_v1/report.json`
- `data/validation/msv000101150_unified_neg_partition_20261014_v1/report.json`
- `data/validation/msv000101150_unresolved_interaction_20261014_v1/report.json`
- `data/validation/msv000101150_spectral_family_redundancy_20261014_v1/report.json`
- `data/external/msv000101150/qc_feature_matrix/POS/report.json`
- `data/validation/msv000101150_pos_gene_diet_interaction_20261014_v1/report.json`
- `data/validation/msv000101150_pos_run_order_sensitivity_20261014_v1/report.json`
- `data/validation/msv000101150_pos_leave_one_sample_out_20261014_v1/report.json`
- `data/validation/msv000101150_pos_qc_ms2_partition_20261014_v1/report.json`
- `data/validation/msv000101150_pos_unresolved_interaction_20261014_v2/report.json`
- `data/validation/msv000101150_pos_spectral_family_redundancy_20261014_v1/report.json`
- `data/validation/msv000101150_cross_polarity_response_concordance_20261014_v2/report.json`
