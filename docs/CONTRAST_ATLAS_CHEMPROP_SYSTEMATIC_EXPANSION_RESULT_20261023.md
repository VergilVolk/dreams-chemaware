# Metabolomic Contrast Atlas: ChemProp systematic expansion result (2026-10-23)

## Verdict

This expansion produces a real coverage-dependent development signal, but it
does **not** establish an unknown-information increment or qualify a general
or external biological-discovery claim.

The frozen collection now contains 15 studies, 51 directed study contrasts,
238,412 study-contrast response rows and 54,861 study-level chemical entities.
The strict common GNPS control labels 979 entities as library-supported and
53,882 as unresolved under the fixed policy.  `Unresolved` does not mean a
novel molecule.

The previous 12-study local-alignment result was not rescued by silently
changing its graph, threshold or candidate universe.  Instead, three consumed
ChemProp accessions added 24 contrasts and 12 exact same-drug cross-accession
relations.  Each query competed against 39 external-source contrast
candidates.  These are same-project development controls, not cross-laboratory
confirmation.

## Frozen data and leakage boundary

- `MSV000094899`: 8,055 entities, 12 drug responses, 96,660 response rows.
- `MSV000096724`: 4,429 entities, 3 shared-drug responses, 13,287 rows.
- `MSV000093571`: 4,130 entities, 9 shared-drug responses, 37,170 rows.
- The source reports contain 36, 6 and 27 biological units, respectively;
  consequently the endpoint panels remain small-sample development controls.
- Added total: 16,614 study entities and 147,117 response rows.
- Corrected IBD Plate3 input remains 662 estimable entities; the old 746-row
  plate-confounded version was not reintroduced.
- Chemical graphs and identity worlds were frozen without beta, SE, contrast
  relations or retrieval truth.  Registered relations were opened only after
  observed and 256 within-study, within-identity-stratum whole-profile null
  scores were frozen.
- MassSpecGym/GNPS and all three ChemProp panels are consumed development data.
  No sealed or external biology panel was opened.

The registered set has 21 canonical relations in 15 connected components:
five earlier HFD/IBD relations, four post-hoc obesity relations, and twelve
exact-drug ChemProp relations.  The 15 components are not 15 independent
projects: twelve share the same broader ChemProp project.

## Exact-drug development result

The original entity-mean local response method gave the following results on
the 12 exact same-drug relations (39 candidates/query):

| Evidence world | Composition Top-1 | Similar-response Top-1 / Top-5 | Median similar-response rank | Minimum-resolution response-null p | Unsigned Top-1 / median rank |
|---|---:|---:|---:|---:|---:|
| all | 9/12 | 1/12 / 7/12 | 4.5 | 10/12 | 1/12 / 24.5 |
| strict known | 9/12 | 1/12 / 4/12 | 13.5 | 2/12 | 0/12 / 20.0 |
| unresolved | 9/12 | 1/12 / 7/12 | 4.5 | 10/12 | 1/12 / 25.0 |

The minimum empirical p value is 1/257 = 0.003891.  Ten unresolved relations
reached that resolution for the same-direction score, versus two known-only
relations.  Eight relations were unresolved-only at that threshold and none
were known-only.  This is a useful development signal that unresolved spectral
evidence contains response-specific information.

It is **not yet a qualified unknown-information increment**, for three reasons:

1. Same response direction was not frozen as signed truth in the contract;
   these data had already been consumed by earlier ChemProp work.
2. Unresolved and strict-known worlds have very different entity counts and
   chemical coverage.  The paired similar-rank comparison was not conclusive
   (unresolved better/equal/worse in 7/1/4 drugs; one-sided paired Wilcoxon
   p=0.1245).
3. Retrieval is not solved: same-direction Top-1 is 1/12, and direction-agnostic
   unsigned ranking remains poor despite 10/12 response-specific same-direction
   signals.

Thus this experiment supports a narrower statement: response information can
survive in unresolved local chemical evidence after candidate-specific null
calibration, but the current scoring rule does not yet turn it into reliable
open-candidate retrieval.

## Response-measure kernel test

A parameter-free response-energy normalization was tested after the original
result was observed.  It treats each contrast as a signed measure over the
frozen chemical graph and divides by complete-world response energy, so missing
chemical support remains in the denominator.

It did not improve the exact-drug result:

- all: same-direction Top-1 1/12, Top-5 6/12, median rank 5.0;
- unresolved: Top-1 1/12, Top-5 7/12, median rank 4.0;
- unsigned all: Top-1 1/12, median rank 25.0.

This variant is retained as a transparent development negative/sensitivity,
not promoted as the new primary method.

## Scientific decision

The sparse one-neighbour fragment graph is no longer allowed to support a
claim that Contrast Atlas has solved open biological response retrieval.  It
does, however, provide the first systematic non-zero response-specific signal
on twelve exact intervention controls, concentrated in the unresolved world.

## Count-matched unresolved control

The required entity-count control was completed with 16 independently frozen
resamples and 256 response-profile permutations per resample.  Within every
study, unresolved entities were sampled without replacement to equal that
study's strict-known count.  Each resample therefore contained 979 entities in
total, exactly matching the full strict-known world.  All 105 cross-study
spectral graphs were rebuilt inside each sampled universe before effects or
relations were opened; this was not a post-hoc deletion from the full graph.

For the twelve exact-drug relations:

- minimum-resolution same-direction p count: median 0/12, range 0--1/12;
- same-direction Top-5: median 2/12, range 0--6/12;
- median same-direction rank: 12.25 across resamples;
- unsigned Top-1: four occurrences across 16 x 12 relation evaluations.

Thus the full unresolved 10/12 signal does not survive entity-count matching.
It depends strongly on the much broader unresolved chemical coverage.  This
does not prove that the full signal is mere random noise: additional relevant
chemistry can only be observed when coverage is retained.  It does prove that
the current known-versus-unresolved comparison cannot distinguish biologically
useful additional coverage from a generic entity-count/coverage advantage.
Consequently, **unknown increment remains unqualified**.

The next useful experiment is no longer another aggregate ranker adjustment.
It is a systematic edge-level replication analysis that separates
mass-compatible same-entity evidence from non-identical chemical relations and
asks whether any unresolved support has significant effects on both sides of
an exact cross-accession perturbation.  Any scoring rule frozen after this
consumed panel must still be tested on a new independent perturbation family
before a biological discovery claim.

## Edge-level response replication

That systematic analysis was completed for all twelve exact-drug relations.
The frozen unresolved graph contained 10,314 unique spectral edges and 68,118
relation-edge evaluations.  An edge counted as response support only when both
endpoints passed within-contrast feature-level BH FDR 0.05 and had the same
effect direction.  No edge was selected using its effect.  The null reassigned
whole entity response profiles within each study 256 times.

Across all one-direction or mutual edges, all 12/12 drug relations had at
least one dual-FDR same-direction edge, and 8/12 reached the minimum empirical
p=1/257.  The more selective mutual-top1 analysis gave:

- 1,421 relation-edge supports across the twelve drugs;
- 12/12 drugs with at least one mutual support;
- 9/12 drugs at empirical p=1/257;
- 1,270 supports on the precursor-compatible 10-ppm identity-hypothesis track;
- 151 supports on the non-mass-compatible chemical-relation track;
- 9/12 minimum-p results on the identity-hypothesis track, but only 1/12 on
  the chemical-relation-only track.

This is the first non-zero, entity-resolved result worth carrying forward.
Without requiring a structure name, the method recovers cross-accession
response replication for same-mass, mutually similar spectra.  It still does
not prove exact molecular identity: isomers, adduct/in-source ambiguity and
ion-family redundancy remain.  It is also development evidence within one
broader ChemProp project.

Just as importantly, the ambitious relation-layer claim is currently mostly
unsupported.  Different-mass chemical neighbours did not show systematic
response replication beyond the null: only one of twelve drugs reached the
minimum permutation resolution on the mutual relation-only track.  Such edges
remain hypotheses and their beta/SE values must not be meta-analysed as if they
belonged to one molecule.

The next independent experiment should therefore freeze two separate outputs:
(1) structure-agnostic same-entity response replication, which now has a real
development signal, and (2) chemical-relation response association, which has
not yet qualified.  The former must be tested across a genuinely independent
project or laboratory before it can anchor a paper claim.

## Frozen cross-project stress tests

The exact same edge-level rule was then applied without retuning to the five
previously registered HFD/IBD relations in independent accessions.

For HFD, the two sex-stratified target relations contained 32 and 30 dual-FDR
same-direction edges, but these did not exceed whole-profile reassignment:
all-edge empirical p values were 0.0856 and 0.2840; mutual-top1 p values were
0.0973 and 0.5525.  Neither identity-hypothesis nor relation-only tracks passed.

For the three near-exact IBD relations, 4,765 frozen relation-edge evaluations
produced zero dual-FDR same-direction edges.  The sources differ in age,
subtype and treatment exposure, so this is not evidence that IBD biology lacks
shared chemistry.  It is evidence that the current frozen operator cannot
recover it under these data and comparability boundaries.

Therefore the ChemProp entity-replication signal has **not generalized**.  The
method currently supports a consumed, same-project positive control and two
cross-project negative stress tests.  No general Contrast Atlas success claim
is qualified.  Further threshold tuning on HFD or IBD is prohibited; the next
confirmation panel must contain a genuinely repeated perturbation with
compatible biospecimen/estimand and adequate independent sampling.

## Reproducible artifacts

- Source preparation: `tasks/prepare_chemprop_contrast_atlas_sources.py`
- Frozen-base append: `tasks/append_contrast_atlas_frozen_collection.py`
- Extension contract: `tasks/contrast_atlas_open_collection_chemprop_extension_v1.json`
- Collection: `data/validation/contrast_atlas_open_response_collection_v9_frozenbase_chemprop_20261023`
- Common identity: `data/validation/contrast_atlas_open_v9_frozenbase_common_identity_20261023`
- Frozen graph/alignment: `data/validation/contrast_atlas_sparse_local_alignment_v3_chemprop_20261023`
- Entity-mean calibration: `data/validation/contrast_atlas_sparse_local_response_calibration_v4_entitymean_chemprop_20261023`
- Response-energy sensitivity: `data/validation/contrast_atlas_sparse_local_response_calibration_v1_responseenergy_chemprop_20261023`
- Count-matched unresolved control: `data/validation/contrast_atlas_count_matched_unresolved_v1_chemprop_20261023`
- Edge-level replication: `data/validation/contrast_atlas_unresolved_edge_replication_v3_chemprop_20261023`
- HFD stress test: `data/validation/contrast_atlas_unresolved_edge_replication_v2_hfd_20261023`
- IBD stress test: `data/validation/contrast_atlas_unresolved_edge_replication_v2_ibd_20261023`
