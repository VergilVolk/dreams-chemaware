# Three-source unresolved-response replication for the Metabolomic Contrast Atlas

Date: 2026-10-19

Role: biology-method development

Evidence status: all sources are public, outcome-open, development/consumed; none is unopened external confirmation

## A necessary calibration experiment, not the whole method

This is not a contest to recognize that three datasets all carry the label `COVID-19`. It is not a two-candidate study matcher, and it does not treat disease-name agreement as biological truth.

The claim-bearing unit is an identity-calibrated unresolved ion entity: phenotype-blind MS/MS evidence and precursor/adduct compatibility link features before case-control effects are opened; effects are then estimated independently within each study with direction and uncertainty. Two sources may nominate the entity. Its linkage rule, response channel, direction, aggregation rule, and multiplicity procedure are frozen before the third source is opened. The third source either reproduces that entity-level response, contradicts it, lacks measurement support, or yields no match. Spectrally related analogues are reported on a separate hypothesis track and never counted as replication of the same entity.

The primary falsifiable question is:

> Can identity-calibrated unresolved MS/MS entities discovered in two independent plasma studies reproduce in a third independent plasma study, and can they add a correct conclusion that a measurement-matched known-only analysis cannot provide?

This is one minimum calibration that tests whether public biological responses can become accumulable knowledge units. It cannot by itself establish a general Atlas method because all three studies concern one disease and all outcomes are already public. Whole-study retrieval remains a downstream use, not the truth definition.

## Three independent public sources

| Source | Design and biological denominator | Chemical substrate | Current qualification |
|---|---|---|---|
| `MSV000085505` | Brazil; 33 hospitalized COVID and 13 healthy plasma files | 1,031 quantified features with exactly corresponding 1,031 MGF spectra | Response blocked by subject/treatment crosswalk; not signed truth |
| `MTBLS1866` | Italy; 83 non-critical COVID and 26 healthy positive-DDA plasma files | about 1.11 million MS2 spectra in the primary file universe | All-feature matrix or frozen raw reprocessing still required; not signed truth |
| `MSV000092887` | Austria; 39 COVID and 39 age/sex-matched control plasma samples in 39 pairs | 3,041 HILIC-positive features; the paper reports 1,982 with MS2 and 175 reference-spectrum hits | Real paired response object built; feature-level raw-MS2 linkage pending; not signed truth |

The sources differ by cohort, institution and acquisition platform. That supports source independence, but it does not guarantee a common molecular response. Heterogeneity is an estimand, not an inconvenience to hide.

`MTBLS3852` is excluded because it is a targeted full-MS panel of 51 metabolites. `MTBLS2542` is deferred because raw DDA exists but the deposited positive MAFs do not expose a direct all-feature, unknown-MS2 response object. These decisions prevent a convenient but biased known-metabolite panel from defining an allegedly unbiased method.

## First new real response object

`MSV000092887` now contributes a real study-level response object rather than a dataset description:

- 3,041 features and 129 mzTab assays;
- 78 biological assays arranged as 39 COVID-control matched pairs;
- 2,598 features pass the prespecified abundance-response support threshold;
- 804 pass the prespecified differential-detection threshold;
- 3,020 pass at least one channel;
- 175 features carry reference-spectrum evidence;
- 2,866 lack a reference-spectrum link, but are **not** automatically called MS2-bearing unknowns;
- the publication reports 1,982 MS2-bearing features in aggregate, implying 1,807 MS2-bearing features without a reference-spectrum match; raw mzXML linkage is still required to identify those 1,807 feature by feature.

Two response channels remain separate. Abundance is the mean within-pair log2(COVID/control) among pairs detected in both groups, with standard error across observed pair differences; zeros are never imputed as abundance. Detection is the Haldane-corrected log odds of COVID-only versus control-only paired detections, with its own standard error. Combining the channels before cross-source calibration is forbidden.

The build also uncovered and froze one deposited metadata correction. Biological unit `C43`, pair 24, was assigned metabolomics assay `21FP`, which was already used by `C31`. The same row reports lipidomics assay `24FL`, the pair field is 24, and the mzTab inventory contains `24FP`; therefore only this row is corrected from `21FP` to `24FP`. This is explicit provenance, not general metadata repair.

Artifacts:

- builder: `tasks/build_msv000092887_paired_response_object.py`;
- tests: `tasks/test_build_msv000092887_paired_response_object.py`;
- response: `data/validation/contrast_atlas_msv000092887_paired_response_v1_20261019/paired_response_object.tsv.gz`;
- assay ledger and report: the same artifact directory.

Primary public records: [Scientific Data study](https://www.nature.com/articles/s41597-024-03822-y) and [MassIVE accession](https://massive.ucsd.edu/ProteoSAFe/dataset.jsp?accession=MSV000092887).

## Frozen leave-one-source-out logic

There are three rotations. In each, two sources are discovery sources and the third is untouched validation:

1. `MSV000085505 + MTBLS1866 -> MSV000092887`;
2. `MSV000085505 + MSV000092887 -> MTBLS1866`;
3. `MTBLS1866 + MSV000092887 -> MSV000085505`.

Identity edges may use precursor/neutral-mass and adduct compatibility, fragment evidence, acquisition metadata, within-source ion-family evidence, and a frozen algorithm score. They may not use case-control labels, beta, standard errors, p-values or candidate condition names. Exact planar identities calibrate whether cross-source entity linking works; they are not biological-response truth. Cross-mass spectral neighbours belong to the separate chemical-relation track.

After the graph is frozen, the two discovery sources nominate same-direction or explicitly heterogeneous entity responses. The held-out source is then opened once. An entity is counted as replicated only under the frozen direction, confidence-interval and multiplicity rules. No match and inadequate measurement support remain visible denominators rather than being removed.

## Controls that distinguish science from more coverage

Every conclusion must be compared on the same measurement and candidate universe against:

1. measurement-matched known-only responses;
2. an identity-field-only world that retains entities but removes spectral similarity;
3. precursor-mass-only linking;
4. raw cosine or WSE and the frozen unified algorithm on the same graph;
5. within-source response reassignment;
6. unresolved-MS2 shuffling that preserves mass and measurement support;
7. strict-identity positive calibration;
8. source-wise effects, heterogeneity and leave-one-ion-family sensitivity.

The unknown-information claim passes only if at least one held-out, identity-calibrated unresolved entity response survives frozen multiplicity control, is not recoverable by measurement-matched known-only, weakens under unresolved-MS2 shuffling, and is not driven by one source or redundant ions. Structure need not be asserted. The biological output must instead state the exact studies, response directions, uncertainty, evidence level, and a falsifiable follow-up.

If no unresolved family replicates in any rotation, or known-only/identity-field/mass-only produces the same result, the central biological-method claim fails. That negative result must not be rescued by retuning a disease-specific score.

## What this does and does not establish

The completed `MSV000092887` build proves that one independent matched study can be represented as 3,041 directed, uncertainty-aware feature responses without requiring structure and without converting missingness into abundance. It does not yet prove a cross-study relation, unknown-MS2 increment, biological mechanism, or external validation.

The immediate high-value task is phenotype-blind raw-MS2 linkage for this 838.6 MB HILIC-positive collection, followed by recovery of the two small blockers in the other sources. No larger Atlas construction is justified before the three rotations yield an actual held-out chemical-response result.

That acquisition universe is now frozen in `data/validation/msv000092887_hilic_positive_mzxml_manifest_v2_20261019`: 129 mzXML files and 838,574,158 bytes, including 78 biological files and 51 blank/QC files. The linkage-facing manifest contains paths, sizes and URLs but physically omits biological unit, pair and group columns. Blank/QC files are retained to measure acquisition-domain and false-link behavior. The publication's alignment tolerances (0.05 min retention time and 0.005 Da MS1) are frozen as initial audit values, not tunable success parameters. No file has yet been marked downloaded, and the manifest does not claim that any feature has MS2 evidence.

Machine-readable contract: `tasks/contrast_atlas_covid_three_source_replication_v1.json`.

## Raw-MS2 linkage execution path prepared on 2026-10-22

The phenotype-blind linkage implementation is now executable rather than only specified:

- `tasks/stage_msv000092887_hilic_mzxml.py` stages the exact 129-file, 838,574,158-byte acquisition universe with resume support, per-file size/SHA-256 evidence, three download workers, and fallback across the official MassIVE, GNPS and Proteomics ProteoSAFe front ends without changing the frozen MassIVE file key. This fallback is acquisition robustness only; it does not alter study membership or evidence.
- `tasks/link_msv000092887_hilic_ms2.py` initially reads only feature ID, precursor m/z, retention time and deposited reference-spectrum status. It does not load effect estimates or decode the `C`/`F` assay suffixes while links are made. A spectrum is assigned only when exactly one feature lies within the fixed 0.005-Da and 3-s window; multi-feature matches are retained in the audit table but abstained. Biological- or pooled-QC-supported unique links are usable, while blank-only links are excluded.
- The linker freezes the full spectrum-link table, feature membership and representative MGF before reopening the response object. Its source-local partition distinguishes deposited library support, unresolved MS2 evidence, blank-only evidence and absence of a unique raw-MS2 link. `unresolved_source_local` is not a cross-source unknown identity and cannot yet support an unknown-information increment claim.
- `tasks/run_msv000092887_hilic_ms2_link.sbatch` is the single CPU execution route. Results are first written to a job-specific directory and moved into the frozen result path only after denominator assertions pass, so an interrupted batch job cannot masquerade as a completed result.

A one-file development smoke test first used biological assay `10CP` without reading its phenotype. The 6,548,777-byte file had 7,045 MS2 spectra, all with precursor metadata. Under the frozen window, 783 spectra reached the response feature universe: 776 had a unique feature candidate and 7 had multiple candidates and were abstained.

The complete phenotype-blind linkage has now been executed on the frozen 129-file universe:

- 129/129 files and 838,574,158 bytes were staged; 78 biological, 11 pooled-QC and 40 blank files were all retained;
- 921,688 raw MS2 spectra were parsed;
- 81,761 spectra entered the 0.005-Da/3-s feature window; 81,037 had exactly one feature candidate and 724 ambiguous spectra were abstained;
- 1,714/3,041 features had at least one unique raw-MS2 link, but 98 were blank-only and therefore excluded;
- 1,616 features had usable biological or pooled-QC MS2: 155 with deposited reference-spectrum support and 1,461 source-local unresolved;
- among the pre-frozen response channels, abundance contained 1,411 usable-MS2 features (146 library-supported, 1,265 unresolved), while differential detection contained 386 (23 library-supported, 363 unresolved); 187 entities passed both channels, which remain separate.

The 1,461 unresolved entities are feature-level MS2-supported responses, not 1,461 unique metabolites. Raw precursor/RT linkage does not establish chemical identity; source-local absence of a reference-spectrum hit is not yet a harmonized cross-source unknown partition. The public outcome is open, so this is development/consumed evidence.

Artifacts:

- raw linkage: `data/validation/contrast_atlas_msv000092887_raw_ms2_link_v1_20261022`;
- channel-preserving source object: `data/validation/contrast_atlas_msv000092887_open_source_v1_20261022`;
- executable preparation: `tasks/prepare_msv000092887_open_response_source.py`.

Only the continuous paired-abundance channel enters the current open collection. The differential-detection log-odds channel is frozen beside it and cannot be numerically collapsed into abundance before cross-source channel calibration. This decision prevents a larger feature denominator from masquerading as stronger evidence.
