# Repository-scale Metabolomic Contrast Atlas execution

Date: 2026-10-14  
Status: active implementation; no biological performance claim yet

## Scientific objective

The method indexes **study-level chemical responses**, not isolated spectra and
not disease-specific marker lists.  A query is an entire LC-MS/MS contrast with
signed within-study effects and uncertainty.  Its answer is a ranked or
abstained set of independently generated public contrasts showing concordant or
inverted chemistry.

`Unbiased` has one narrow, falsifiable meaning: every QC-qualified quantitative
entity can enter before its structure, pathway, expected disease role, or final
cross-study truth is inspected.  It does not erase ionisation, chromatography,
DDA, cohort, or repository-selection bias.

## Scale decision

No further method development will be centred on a single small accession.  The
development unit is a frozen repository-scale universe compiled from the full
Pan-ReDU metadata dump and linked, where available, to the DreaMS/MassIVE
spectral universe.  Small studies may later appear as individual held-out
shards, but cannot define the method or its parameters.

The first compute wave must contain multiple accessions per biological contrast
family and must be selected from metadata only.  Species and sampled body part
are fixed in the conservative comparison stratum; instrument, polarity, and
chromatography are explicit transfer axes.  Dataset accession alone does not
prove source independence: publication/project duplication must be adjudicated
before a sealed result.

Pan-ReDU `UniqueSubjectID`/`SubjectIdentifierAsRecorded` values are qualification
metadata, not proof of independent persons or animals.  Some longitudinal or
multi-site studies encode visit, body site, or aliquot in these fields.  Before
effect estimation, every selected accession must therefore reconstruct the
true biological-unit and repeated-measure hierarchy from its deposited design;
the metadata registry may not supply the statistical denominator.

## Two input tiers

1. **Quantitative response tier (primary).** A sample-by-feature MS1 abundance
   matrix, biological-unit identifiers, design labels, and phenotype-blind
   feature-to-MS/MS links are all available.  Only this tier may support a
   biological response claim.
2. **Spectral-distribution tier (diagnostic).** Only per-sample acquired MS/MS
   is available.  It may test chemical-space coverage and acquisition transfer,
   but DDA scan frequency is not treated as metabolite abundance and cannot
   establish a biological contrast.

This prevents repository scale from being purchased by silently changing the
scientific endpoint.

## Scalable response representation

For entity `i` in contrast `d`, estimate an internal effect and uncertainty
from biological samples, then bound the response:

\[
r_{di}=\frac{\hat\beta_{di}}
 {\sqrt{\hat\beta_{di}^{2}+SE_{di}^{2}}}.
\]

Let `phi(z_i)` be a frozen, unit-normalised Noise/DreaMS spectral embedding (or
a later frozen qualified unified kernel feature map), and let `q_di` contain
only observable measurement-quality terms.  Store two compact kernel-mean
sketches per contrast:

\[
v_d^+=\frac{\sum_i q_{di}[r_{di}]_+\phi(z_i)}
                 {\sum_i q_{di}[r_{di}]_+},\qquad
v_d^-=\frac{\sum_i q_{di}[-r_{di}]_+\phi(z_i)}
                 {\sum_i q_{di}[-r_{di}]_+}.
\]

Comparison exposes all four terms (`+:+`, `-:-`, `+:-`, `-:+`).  The signed
score rewards concordant directions and penalises inverted directions; the
unsigned score is retained as a control.  This avoids exact feature matching
while never upgrading spectral proximity to molecular identity.

The implementation is:

- `tasks/build_pan_redu_contrast_universe.py`: streams the full repository dump
  and freezes outcome-blind eligible contrast panels;
- `tasks/build_signed_spectral_measure_sketches.py`: estimates entity responses
  and builds all/known/unresolved response sketches;
- `tasks/query_signed_spectral_measure_sketches.py`: performs pre-truth,
  cross-source retrieval and retains the four directional components.

## Required repository-scale test

The first claim-bearing evaluation must satisfy all conditions below.

1. At least three independent source families for every evaluated biological
   contrast family; sources are split before any retrieval score is opened.
2. At least twenty quantitative-tier studies overall, spanning more than one
   biological family and more than one instrument family.
3. The same frozen representation and thresholds for all studies; no
   accession-specific rescue or disease-specific feature selection.
4. Candidate universe fixed across `all`, `known_only`,
   `measurement_matched_known`, `unresolved_MS1_only`,
   `unresolved_shuffled_MS2`, strongest frozen spectral baseline, and the full
   method.
5. Primary endpoint is correct retrieval coverage at a frozen selective-risk
   ceiling.  Recall, MRR, direction accuracy, abstention, and source-family
   cluster confidence intervals are also reported.
6. `known + unresolved` must exceed the strongest eligible control with a
   source-family clustered CI lower bound strictly above zero, without a higher
   wrong-link rate.  The added correct links must be listed and independently
   auditable.

Metadata labels create candidate positives, not truth.  Exact repeated
interventions, independent cohort replication, or another predeclared
biological relationship must establish truth; “no known relation” is not a
negative label.

## Stop rules

- If fewer than twenty primary-tier studies can be recovered, the repository
  supports a resource/readiness paper, not the claimed general biological
  method.
- If signed response does not outperform chemical composition and unsigned
  response, the system is not a Contrast Atlas.
- If unresolved information does not add correct retrieval at matched risk,
  the method remains a useful response index but not an expansion of
  non-targeted metabolomics.
- If results are driven by one source, one ion family, or one acquisition
  platform, the generalisation claim fails.

## Current claim boundary

All previously opened GNPS, ChemProp, HFD/obesity, TT7, IBD, and IsoNet analyses
remain development/consumed evidence or negative controls.  None is promoted by
this scale-up.  The repository registry is metadata-only; the response-sketch
code passing a synthetic direction test establishes implementation semantics,
not biological validity.

## Executed repository census (2026-10-14)

Source: complete Pan-ReDU dump, `939,434,264` bytes.  This is a
`development / metadata-only / abundance-unopened / cross-study-truth-unopened`
census.

| Qualification tier | Pan-ReDU rows | accessions with biological rows | candidate panels | repeated species-body strata | unit caveat |
|---|---:|---:|---:|---:|---|
| broad, >=5 reported IDs/level | 1,123,912 | 6,292 | 1,107 | 96 | 973 panels use file fallback |
| >=3 reported MS2 files/level | 1,123,912 | 6,292 | 192 | 32 | 68 panels use file fallback |
| reported subject-ID required | 1,123,912 | 6,292 | 124 | 22 | IDs still do not prove biological independence |

The systematic first-wave candidate uses every subject-ID/MS2-qualified stratum
repeated in at least two accessions.  Before cohort adjudication it contains:

- 63 contrast panels in 22 species-body strata;
- 31 accessions collapsed to 29 preliminary source families after publication
  and exact-design duplicate checks;
- 6,950 **reported-unit appearances**, not independent biological samples;
- 8,565 unique metadata file records (`8,052 mzML`, `513 mzXML`) before
  within-accession collection-mirror adjudication;
- 8,190 files with reported MS2 and 26,020,047 reported MS2 spectra.

The unfiltered repositories behind 29/31 matched MassIVE accessions total about
2,425.56 GB, so whole-accession download is rejected.  The selected-file
manifest is the acquisition boundary.  All 8,565 selected paths have an
unambiguous deposited size and total 465,568,959,305 bytes (433.59 GiB).
Collapsing the confirmed HNRC accession mirror and 373 exact-size
`peak`/`ccms_peak` collection aliases removes 557 duplicate paths and
18,081,803,782 bytes, leaving 8,008 files and 447,487,155,523 bytes
(416.75 GiB) for acquisition.  The within-accession correction was made before
raw abundance was opened; it saves 10.79 GiB relative to the earlier manifest.
Different ion modes from the same cohort are retained as
measurements but remain one source family.  Twenty-eight accessions expose 90 small
candidate design assets totalling 35,705,062 bytes; three currently expose no
such asset through the inventory query.

The first unit-hierarchy audit found 45/63 panels with one file per reported
identifier, 18/63 with repeated files, and 4/63 in which the same reported ID
crosses contrast levels.  Those four fail closed.  No panel is called
statistically ready until its deposited design establishes the actual
person/animal, repeated-measure, batch, and covariate hierarchy.

All 90 candidate design assets were downloaded and byte-verified.  They map all
selected files for 28 accessions, but most are ReDU-normalised tables rather
than independent evidence for the unit hierarchy.  Native, non-standard unit-ID
columns currently recover 4,201/10,867 contrast-file appearances and make 16/63
panels topology-complete across five preliminary source families.  “Topology
complete” does not mean analysis-ready: model covariates, batches, contrast
coding, cohort independence, and repeated-measure handling remain mandatory.

## Processed quantitative shortcut across the full subject-strict registry

The search was expanded from the 31-accession repeated-stratum wave to all 49
accessions in the subject-ID-required registry.  This remains
`development / abundance-unopened-for-selection / cross-study-truth-unopened`.

- 63 path candidates (2,715,917,328 bytes) were found across the 49 accessions;
- 9 accessions expose a compact quantitative-table plus MGF pair;
- all 9 best pairs pass an exact schema join, with 100% MGF-to-quant feature-ID
  overlap;
- the 9 pairs contain 38,895 MS/MS-linked features, 5,107 quantitative columns,
  and 4,014 columns that join the strict Pan-ReDU file universe;
- they support 18 metadata contrast panels, but only 8 currently have a
  topology-complete native biological-unit map.

Only one comparison stratum appears in two topology-complete source families:
the generic mouse-feces `healthy` versus `unhealthy (NOS)` label.  Native design
review rejects it as cross-study truth.  MSV000082608 is a longitudinal
sleep-deprivation/social-defeat/treatment factorial study; MSV000082650 is a
different longitudinal ASD/CDR/treatment factorial study.  ReDU's generic
health label erases those interventions.  Therefore the processed shortcut has
**zero claim-bearing cross-study truth pairs** after semantic adjudication.

This is not a reason to tune a two-study score.  It is evidence that the
primary repository-scale experiment must use design-native perturbations and
the 416.75-GiB post-deduplication raw acquisition wave, rather than ontology
labels alone.

The corresponding assets are:

- `data/validation/pan_redu_subject_strict_quant_pair_audit_20261014/`;
- `data/validation/pan_redu_subject_strict_processed_readiness_20261014/`;
- `tasks/build_pan_redu_processed_contrast_readiness.py`.

## Invalidated semantic shortcut

An exploratory configuration paired Turek's explicit sleep-deprivation factor
with Lowry's `ATTRIBUTE_asd` before the latter abbreviation was adjudicated.
That configuration is now fail-closed and its output is marked `INVALIDATED`.
Its score is neither positive nor negative evidence for response transport and
must not be cited.  No timepoint or parameter search is allowed to rescue it.

## First design-native cross-source control

A predeclared human-plasma biological-sex control used two independent source
families, 206 biological units (HNRC 184; ONR Wright 22 after excluding one
unit with inconsistent sex labels), and 4,282 MS/MS-linked entities treated as
structure-unresolved.  Direction was fixed as male minus female.

The transparent fragment-bin response score exceeded both an unconstrained
entity-to-spectrum shuffle and a 5-Da precursor-mass-stratified shuffle
(`p=1/2001` for each).  Those are insufficient controls.  When biological-unit
sex labels were independently permuted within each cohort and every feature
effect was re-estimated, the result failed (`500` permutations,
`p=0.22555`, `z=0.6021`).  The formal decision is therefore
`FAIL_BIOLOGICAL_UNIT_LABEL_NULL`.

This establishes a new mandatory gate: every claimed cross-study response must
pass an outer biological-unit label null.  Randomising entity-to-spectrum links
alone can make a high-dimensional response look convincing while failing to
show that the observed phenotype labels carry reproducible chemistry.  The
failed control will not be tuned or promoted.

## Cost-efficient raw tranche

The full raw boundary remains 8,008 files / 416.75 GiB.  To reach repository
scale sooner without redefining the universe, deposited quantitative/MS2 pairs
are reused for 9 accessions (7 source families), and new raw processing is
selected by lowest GiB per candidate panel among the remaining sources.

The first raw tranche contains 20 additional source families, 21 accessions,
42 candidate contrasts in 23 acquisition panels, 3,845 open-format files, and
157,279,532,029 bytes (146.48 GiB).  Combined with the processed sources this
creates a 27-source development program while preserving the ability to add
the rest of the frozen 416.75-GiB universe unchanged.  Selection used only
metadata, deposited sizes, and processed-pair schema availability; no abundance
effect or retrieval score entered the decision.

Execution assets:

- `data/validation/pan_redu_high_value_raw_tranche_20261014/`;
- `tasks/select_pan_redu_high_value_raw_tranche.py`;
- `tasks/run_pan_redu_high_value_raw_tranche.sbatch`.

Before raw acquisition, the tranche is split into a phenotype-blind
quantification tree and a separately sealed contrast-membership tree.  The 23
blind panels contain 3,845 files and 2,859 panel-local biological units.  The
generic builder admits mzML and mzXML, counts recurrence by biological unit
rather than file, freezes representative MS2 spectra, then quantifies each
anchor from MS1 with resumable per-file caches.  A real mzML smoke test found
6,206 eligible MS2 spectra and recovered MS1 signal for all 25 tested anchors;
this verifies implementation only and is not biological evidence.

Execution assets:

- `data/validation/pan_redu_high_value_raw_panel_jobs_20261014/`;
- `tasks/prepare_pan_redu_raw_panel_jobs.py`;
- `tasks/build_pan_redu_ms2_anchored_panel.py`;
- `tasks/run_pan_redu_ms2_anchored_panels.sbatch`.

No contrast label is read during anchor creation or quantification.  Blank
files are absent from the cost-efficient tranche, so blank-sensitive entities
must be excluded downstream or revisited with targeted blank acquisition.
Study effects will be estimated only after each abundance matrix is frozen,
with repeated files clustered by the hashed biological-unit identifier.
The frozen effect stage uses positive-only log2 values, performs sample-median
normalisation across the complete phenotype-blind panel, then opens contrast
membership and takes the median across repeated files within each biological
unit before estimating `level_b - level_a` and its Welch uncertainty.  It does
not silently turn files into replicates.  Three of the 42 candidate contrasts
already fail closed because a reported subject identifier crosses biological-
sex levels; this is a metadata/design failure, not a score to optimise.  After
those exclusions, 32 contrasts fall into 12 strata represented by at least two
preliminary source families.  The largest topology-qualified calibration
strata are human fecal sex (4 sources), human fecal early-versus-middle
adulthood (4), human urine sex (3 sources across 4 accessions), and human urine
later-versus-middle adulthood (3 sources across 4 accessions).  These are
systematic transfer controls, not yet independent biological truth:
confounding, cohort overlap, medication, diet, collection and batch structure
still require design-native adjudication before any biological claim.

Effect-stage assets:

- `tasks/estimate_pan_redu_panel_contrasts.py`;
- `tasks/run_pan_redu_panel_effects.sbatch`.

## Frozen spectral bridge and response index (2026-10-16)

The raw-panel chain now ends in a study-level response index rather than an
MS/MS neighbourhood.  Each panel's representative spectra are encoded without
reading contrast membership, effects, ontology endpoints or cross-study truth.
The resulting vectors are joined to the already-frozen `level_b - level_a`
effects by an exact `panel_id + anchor_id` key.  The response index stores
positive and negative measures separately, so concordant and inverted biology
cannot cancel invisibly.

The first execution uses official DreaMS only as a frozen spectral baseline.
The same anchor-keyed interface will accept the final qualified Noise/ChemAware/
P2b representation without changing raw quantification, biological effects or
candidate contrasts.  This separation is essential: an algorithm comparison
must not redefine the biological question after effects have been opened.

An absent identity partition is recorded as `unpartitioned`, never as
`unresolved`.  Known-only and unresolved worlds therefore abstain until an
explicit frozen identity table is supplied.  In every world both the query and
candidate require at least 12 eligible chemical entities and a non-zero signed
response measure.  Zero-evidence candidates receive no rank.

Execution assets:

- `tasks/encode_pan_redu_panel_spectra.py`;
- `tasks/assemble_pan_redu_signed_response_index.py`;
- `tasks/query_signed_spectral_measure_sketches.py`;
- `tasks/run_pan_redu_panel_embeddings.sbatch`;
- `tasks/run_pan_redu_response_index.sbatch`.

The frozen server order is download, phenotype-blind panel construction,
effect estimation, phenotype-blind spectral encoding, response-index assembly,
and finally pre-truth cross-source retrieval.  Retrieval ranks are not opened
as biological discoveries.  First, each repeated stratum needs design-native
adjudication of comparability and independence; then signed response must beat
composition/unsigned response, mass-only, known-only, measurement-matched
known and shuffled-MS2 controls at matched selective risk.  Only links that
survive this sequence can support the unknown-information increment claim.
