# Metabolomic Contrast Atlas: ChemProp systematic expansion result (2026-10-23)

## Verdict

This expansion produces a real development signal, but it does **not** yet
qualify a general or external biological-discovery claim.

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

The next decisive control is a count/coverage-matched unresolved world.  It
must determine whether the 10/12 versus 2/12 difference reflects genuinely
additional chemical response information or merely thousands more measured
entities.  Any scoring rule frozen after this consumed panel must then be
tested on a new independent perturbation family before a biological discovery
claim.

## Reproducible artifacts

- Source preparation: `tasks/prepare_chemprop_contrast_atlas_sources.py`
- Frozen-base append: `tasks/append_contrast_atlas_frozen_collection.py`
- Extension contract: `tasks/contrast_atlas_open_collection_chemprop_extension_v1.json`
- Collection: `data/validation/contrast_atlas_open_response_collection_v9_frozenbase_chemprop_20261023`
- Common identity: `data/validation/contrast_atlas_open_v9_frozenbase_common_identity_20261023`
- Frozen graph/alignment: `data/validation/contrast_atlas_sparse_local_alignment_v3_chemprop_20261023`
- Entity-mean calibration: `data/validation/contrast_atlas_sparse_local_response_calibration_v4_entitymean_chemprop_20261023`
- Response-energy sensitivity: `data/validation/contrast_atlas_sparse_local_response_calibration_v1_responseenergy_chemprop_20261023`

