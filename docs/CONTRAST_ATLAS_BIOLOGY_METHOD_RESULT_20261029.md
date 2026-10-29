# Contrast Atlas biology method: direct baselines and current decision

Date: 2026-10-29  
Role: biology application  
Evidence status: development/consumed; no external biological validation

## Fixed scientific question

Given a directed disease or intervention contrast, can unresolved MS/MS entities
add reliable retrieval of related or opposite responses in other studies, and
does the in-house spectral algorithm add value beyond the same evidence scored
by existing methods?

The two increments are separate:

1. **unknown-information increment:** unresolved evidence must beat known-only,
   measurement-matched known and shuffled-unresolved controls;
2. **algorithm increment:** methods are replaced on exactly the same entities,
   candidates, effects and response operator.

A biological success additionally requires independent evidence that did not
construct the match, tune the method or define the queried response.

## Result 1: the m/z 595.348 IBD IDs are two feature pairs, not aliases

The apparent conflict between adult features `1041`/`1286` and paediatric
features `16870`/`28796` is resolved.  Each study contains two distinct
chromatographic features near the same precursor mass:

| Source | Entity | m/z | RT (min) | Legacy annotation |
|---|---|---:|---:|---|
| adult IBD200 Plate3 | `1041` | 595.348004 | 4.1555 | Stercobilin, Bronze hypothesis |
| adult IBD200 Plate3 | `1286` | 595.347698 | 5.3594 | none |
| paediatric VEO-IBD | `16870` | 595.348427 | 3.5154 | Stercobilin, same Bronze library spectrum |
| paediatric VEO-IBD | `28796` | 595.347529 | 5.8192 | none |

The response-blind cross-study diagnostic sqrt-cosine matrix at 0.02 Da was:

| Adult → paediatric | `16870` | `28796` |
|---|---:|---:|
| `1041` | 0.5369 | 0.2150 |
| `1286` | 0.1803 | 0.8451 |

Thus `1041↔16870` and `1286↔28796` are two separate identity hypotheses.
They cannot be combined into one three-study replication chain.  The first is
still unresolved under the current common Gold/Silver policy despite the
legacy Bronze name; the second has no structure name.  Neither is independently
validated.

Reproducible output:

- `data/validation/contrast_atlas_ibd_mz595_entity_resolution_v2_20261029/`
- `tasks/resolve_contrast_atlas_ibd_mz595_entities.py`

## Result 2: direct study-response competitors now run on the real collection

Three strict-known baselines were run on the frozen 15-source, 51-contrast
development collection:

- continuous full-InChIKey response cosine using
  `beta / sqrt(beta^2 + SE^2)`;
- z-thresholded iDMET-like signed overlap;
- iDMET+-like same/opposite response enrichment.

These are task-level competitors, not exact reproductions of external software.
One feature per study/full InChIKey was chosen response-blind by WSE, matched
peaks and entity ID.  Every source-disjoint target remained in the denominator;
insufficient known evidence produced abstention.

| Denominator | Count |
|---|---:|
| source families | 15 |
| study-level contrasts | 51 |
| response rows | 238,412 |
| strict-known study identity representatives | 734 |
| unique full InChIKeys | 429 |
| directed source-disjoint query-target pairs | 2,286 |
| registered directed relations | 42 |
| primary-eligible registered relations | 34 |
| consumed exact-drug directions | 24 |

Availability and descriptive registered-relevance ranks were:

| Baseline | Available candidate pairs | Primary Top-5 | Exact-drug Top-5 |
|---|---:|---:|---:|
| continuous strict-known | 1,814 / 2,286 | 2 / 28 available | 0 / 24 |
| iDMET-like signed overlap | 516 / 2,286 | 6 / 20 available | 6 / 20 available |
| iDMET+-like ORA | 1,218 / 2,286 | 10 / 28 available | 9 / 24 |

The 42 registered relations have no signed entity-level truth.  These are
relevance ranks, not same/opposite accuracy and not biological confirmation.
All sources and relations are development/consumed.

Reproducible output:

- `data/validation/contrast_atlas_known_response_baselines_v2_20261029/`
- `tasks/run_contrast_atlas_known_response_baselines.py`

## Result 3: current examples separate unknown gain from algorithm gain

### IBD is not an unknown-required flagship result

For adult IBD200 → paediatric VEO-IBD, the strict-known continuous baseline
already ranks the paediatric contrast first.  In reverse, the paediatric query
ranks adult IBD second on the similar face and third by absolute response.
The unresolved m/z 595 evidence may add chemical detail, but it did not create
an otherwise unavailable study relation.  IBD is retained as an entity
disambiguation and evidence-accumulation example, not the primary unknown-gain
claim.

### Cilnidipine is an unknown-information positive control, not an algorithm win

Among the 24 consumed exact-drug directions, cilnidipine is the only current
two-direction screen hit for unresolved information:

- strict-known direct baselines rank the two directions 14 and 15 by ORA;
- Stage-1 on known fixed relations ranks them 13 and 22;
- the all/unresolved relation evidence ranks both directions first with
  query max-T `p = 0.003891`.

However fragment cosine, Official DreaMS and Stage-1 all rank cilnidipine first.
It therefore supports the hypothesis that unresolved signals contain useful
response information, but it does **not** show an in-house algorithm increment.
It also belongs to one broader consumed ChemProp project and is not
cross-laboratory validation.

Across all 24 directions, the frozen screen found:

| Screen class | Directions |
|---|---:|
| unknown-information candidate | 2 |
| calibrated Stage-1-over-fragment algorithm candidate | 0 |
| joint unknown + algorithm candidate | 0 |

Reproducible output:

- `data/validation/contrast_atlas_development_candidate_screen_v2_20261029/`
- `tasks/summarize_contrast_atlas_development_candidates.py`

## Legal algorithm boundary

| Track | Question | Eligible algorithm use |
|---|---|---|
| identity | Are two observations the same entity? | complete precursor/adduct-compatible blocks; WSE, Official, Noise V1 and Full23 |
| relation | Do distinct entities support a related response pattern? | a separately calibrated chemical-relation scorer; effects are never merged |

Full23 is currently eligible only on the identity track.  Its closed-set
identity ranking cannot be transferred to cross-mass chemical relations.
The current relation graph is fragment-admitted, so replacing edge weights can
claim only fixed-graph reweighting, not relation recall.

The frozen executable contract is
`tasks/contrast_atlas_biology_evaluation_contract_v1.json`.

## Current verdict and next decisive computation

The Atlas direction remains scientifically coherent, but the full application
claim is not yet achieved.  Current evidence establishes a direct known-only
baseline and one consumed unknown-information positive control.  It does not
yet establish a joint algorithm-to-biology increment or independent biological
confirmation.

The next decisive computation is the identity-track 4-world × 4-method matrix
on the already frozen complete candidate blocks:

- worlds: known-only, measurement-matched known, unresolved-shuffled MS/MS,
  all;
- methods: WSE, Official DreaMS, Noise V1, Full23 v1;
- same complete query blocks, including zero- and one-candidate abstentions;
- identical response aggregation and target-contrast universe.

The relation track is evaluated separately with fragment/WSE/Official/Noise and
a validated relation scorer.  Full23 is excluded.  Only a case that survives
the strongest control, changes a study-level answer, and is confirmed by new
independent evidence can become the flagship biology result.
