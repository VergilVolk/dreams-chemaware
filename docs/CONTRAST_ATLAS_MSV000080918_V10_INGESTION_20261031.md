# MSV000080918 response-source ingestion into the development Contrast Atlas

Date: 2026-10-31

## Outcome

The qualified MSV000080918 / MSV000080558 source family has been added to a
new **development/consumed** response collection. All 238,412 pre-existing
response rows, all source rows, and all registered relations remain unchanged.
The derived `source_disjoint_candidate_contrasts` field in the 51 prior
contrast rows is recomputed and increases by two because two new contrast
candidates now exist; all other prior contrast fields remain unchanged.

The v10 development collection now contains:

| Item | v9 base | v10 development |
|---|---:|---:|
| Source families | 15 | 16 |
| Study-level contrasts | 51 | 53 |
| Response rows | 238,412 | 238,863 |
| Unique study entities | 54,861 | 55,127 |
| Registered directional relations | 42 | 42 |

No relation was registered for the new source. IgA and CD8 are two contrasts
from one consumed parent study, not an independent cross-study truth pair.

## What entered the identity-compatible response table

The original response analysis has 421 response-eligible features with at
least one spectrum candidate. The current Atlas collection represents one
response entity with one spectrum, so ambiguous feature--spectrum links cannot
be inserted by selecting a convenient candidate.

Only the reciprocal-unique subset entered the main table:

| Contrast | Role | Rows | Library-supported | Unresolved |
|---|---|---:|---:|---:|
| IgA-KO minus intact | primary development positive control | 237 | 16 | 221 |
| CD8-KO minus intact | sensitivity-only development control | 214 | 19 | 195 |
| Union | one source family | 266 entities / 451 rows | 19 entities | 247 entities |

Feature IDs are whitespace-canonicalized before becoming global entity IDs.
Every exported entity maps to one distinct consensus cluster. The extracted
MGF preserves the original cluster scan and records the canonical feature ID.

The adapter also writes `response_evidence.tsv.gz`, which retains p, q, the
original BH family size, analysis role, leave-one-date-out direction
diagnostics, and the source spectral-evidence state. These fields are evidence
annotations; none was used to select significant rows. All 451 response rows
entered irrespective of p or q.

The 155 response-eligible features with ambiguous or nonreciprocal spectral
links remain in the complete source result and are not silently assigned a
representative spectrum.

## Frozen-base protection

The historical source contracts cannot all be replayed from the present local
asset set. The append path therefore supports a standalone, checksum-validated
staging collection containing exactly the new study. It then:

1. verifies the frozen v9 collection and all four base tables;
2. verifies the standalone new-source collection;
3. selects only the study ID registered in the extension contract;
4. mechanically appends those rows and recomputes only the derived
   source-disjoint candidate count for every contrast;
5. rejects duplicate studies, contrasts, response entities, and relations.

The final report records `staging_mode=standalone_new_sources` and confirms
that all 238,412 base response rows are preserved. This operation reads no
candidate score, retrieval rank, response-relation outcome, or model
prediction.

## What this does and does not enable

This source adds a substantially larger, cage-clustered public response object
to the Atlas. It does not by itself establish a biological discovery.

It now enables a future same-denominator experiment asking whether:

- the IgA primary response receives a correct public-study answer from
  unresolved evidence that known-only lacks;
- the CD8 sensitivity response behaves consistently without being promoted to
  primary evidence;
- a legal spectrum model changes a feature connection and consequently changes
  a research-level answer.

That experiment cannot yet be claimed. On the frozen 291,330-pair v5 identity
axis, the current strict score-package status is:

- current-contract-valid packages: `0/4` for WSE, Official, Noise, Full23;
- complete reusable all-axis score source: WSE only, `291,330/291,330`, still
  requiring repackaging into the current three-table contract;
- Official, Noise, and Full23 complete-axis delivery: `0/291,330` each.

Historical 41,082-edge embedding reweights remain fixed-topology diagnostics
and cannot be used as model-specific Atlas retrieval results.

## Claim boundary

Allowed:

> A new development/consumed public source family contributes two qualified,
> cage-clustered study-level response objects and 266 one-spectrum response
> entities to a frozen-base-preserving Contrast Atlas extension.

Not allowed:

- external or sealed validation;
- causal IgA/CD8 mechanism;
- metabolite identity from reciprocal uniqueness or a GNPS LibraryID alone;
- unknown-information increment;
- WSE, Official, Noise, Full23, ChemAware, BioAware, or unified-model increment;
- a new cross-study biological relation.

## Artifacts

- Source adapter: `tasks/prepare_msv000080918_atlas_source.py`
- Source package: `data/validation/contrast_atlas_msv000080918_atlas_source_v5_20261031`
- Standalone staging: `data/validation/contrast_atlas_open_response_collection_v10_staged_sourceonly_msv000080918_v5_20261031`
- Frozen-base extension: `data/validation/contrast_atlas_open_response_collection_v10_frozenbase_msv000080918_v7_20261031`
- Extension contract: `tasks/contrast_atlas_open_collection_msv000080918_extension_v1.json`

Primary biological source: [Shalapour et al., Nature 2017](https://doi.org/10.1038/nature24302).
The direct identity-alignment comparator is
[X-Align, Chemical Science 2026](https://doi.org/10.1039/d6sc04644d).
