# Contrast Atlas global biological panorama: corrected verdict (2026-10-29)

## Outcome

The Atlas is now a system-wide data product rather than a single IBD feature or a two-study retrieval example.  Its current release contains 15 sources, 13 **operational** dependence groups, 51 contrasts, 54,861 study-entities and 238,412 entity-by-contrast response estimates.  It also exposes all 2,286 source-disjoint directed contrast candidates and preserves the study design, estimand, inferential unit and claim boundary of every contrast.

This scale is real, but biological validity is not yet established.  The current WSE identity track produces many statistically selected response hypotheses; their anatomy shows strong source concentration and domain mismatch.  Those observations justify a global evidence-adjudication stage.  They do not identify a single failure mechanism and do not make the negative result independently publishable.

All inputs are `development/consumed`.  No sealed biological validation set was opened.

## What the Atlas contains

| Layer | Current denominator | Scientific meaning |
|---|---:|---|
| Sources | 15 | Public or locally reconstructed analysis sources |
| Operational dependence groups | 13 | Known shared parent projects are merged; complete independence is not proven |
| Study-level contrasts | 51 | 16 observational human contrasts and 35 intervention/controlled contrasts |
| Response estimates | 238,412 | Study-internal beta/SE records; raw beta scales are heterogeneous |
| Study-entities | 54,861 | 979 common-policy library-supported and 53,882 unresolved entities |
| Model-independent identity candidates | 291,330 directed pairs | Same-precursor, 10-ppm physicochemical candidates frozen before response access |
| Source-disjoint contrast directions | 2,286 | Complete retrieval candidate universe |
| WSE reciprocal identity hypotheses | 34,219 all; 918 known; 32,328 unresolved | Observed-ion identity hypotheses, not confirmed same molecules |

The 51 contrast-level `independent_units` values must not be summed.  Contrasts reuse participants, organisms and experimental batches.  The source-level unit counts also mix people, animals, cultures, tissues and response replicates and therefore are not a single sample-size denominator.

The common identity policy is mandatory.  Source-native identity labels disagree with the frozen common policy for 1,418 study-entities and cannot be used to define the global known/unresolved comparison.

## What the complete WSE identity track actually says

After combining similar and opposite faces with a two-face Bonferroni rule:

| Diagnostic view | Statistically selected queries / 51 |
|---|---:|
| All entities | 49 |
| Common known-only | 19 |
| Unresolved-only | 49 |

These are hypothesis yields, not correct biological links.  The all-world answers have the following global anatomy:

| Property | Count |
|---|---:|
| Same operational parent project | 23 / 49 |
| Cross-parent-project hypotheses | 26 / 49 |
| Target the shared ChemProp parent project | 36 / 49 |
| Cross-parent and same condition family | 2 / 26 |
| Cross-parent and same species plus biospecimen | 0 / 26 |
| Cross-parent, cross-domain exploratory | 24 / 26 |

Therefore, the current complete graph does not yet support a claim of general cross-study biological retrieval.  It also does not prove that the response operator is intrinsically wrong.  Coverage, broad chemical responses, identity-edge error, study-level score behaviour and genuine but nonspecific biology remain competing explanations.

Forty-three queries have an all-world selection reproduced by unresolved-only but not by known-only.  This is a prioritisation property, not proof that unresolved information is correct or uniquely necessary.  Only three all-world selections are corroborated by the corresponding known-only selection.

## Complete application value and matched controls answer different questions

The complete graph asks how many hypotheses the available system can produce at its full coverage.  Opportunity-matched graphs ask how much of the apparent unresolved advantage remains after equalising edge opportunity.

The 32 opportunity-matched graphs are highly overlapping sensitivity selections: 850/918 edges occur in every graph.  They are not 32 biological replications.  Their HFD/IBD registered primary directions yield 0/10 two-face max-T passes in every graph, but the registered set contains only five canonical cross-project relations in two biological components and has no signed entity-level truth.

The parent-project holdout yields 15--21 statistically selected cross-project hypotheses per sensitivity graph (median 19), with 1,926/1,944 candidate directions unjudged.  It demonstrates that source concentration persists after removing same-parent candidates.  It does not establish a project-hub failure mechanism or biological specificity.

## Identity and chemical-relation tracks remain distinct

The current complete release evaluates only same-precursor identity hypotheses.  A legacy exploratory graph contains 23,989 different-mass reciprocal fragment-cosine edges, but it was selected under an older graph contract.  It can be exposed as a separate chemical-relation hypothesis layer; it cannot merge beta/SE across entities and cannot be combined with the current identity scores as if both meant molecular identity.

A valid Atlas must eventually return two different statements:

1. **Identity evidence:** whether an observed ion may represent the same chemical entity across studies, with calibrated identity risk.
2. **Relation evidence:** whether different chemical entities support a reproducible response pattern, without calling them the same molecule.

Neither layer is allowed to borrow the response being evaluated to choose its spectral connection.

## The algorithm comparison must start from the complete candidate axis

The current `apply_algorithm_edge_scores` route fixes WSE reciprocal edges and clips replacement scores to `[0,1]`.  It is only a fixed-topology sensitivity analysis.  It cannot evaluate Official DreaMS, Noise or Full23 fairly because those methods cannot select a candidate rejected by WSE, remove a WSE edge or express their own abstention.

The correct comparison freezes all 291,330 directed candidate pairs.  Each method scores every complete candidate block, forms its own directional decision and then applies the same predeclared reciprocal rule.  Raw ranking score and response evidence weight are separate:

- the primary cross-method topology analysis assigns unit weight to every accepted reciprocal edge;
- confidence weighting is permitted only with a frozen identity calibrator that never reads Atlas beta/SE, phenotypes or registered relations;
- raw cosine, logit, percentile and arbitrary sigmoid output are not identity probabilities.

## The next biological deliverable is global, not a selected feature story

The immediate release must contain four linked ledgers:

1. a 51-row contrast catalogue with biological design and inferential-unit boundaries;
2. a complete contrast-connection table containing every candidate, both response faces, all three evidence worlds and the registration boundary;
3. a complete entity-edge contribution table showing every beta/SE pair that contributes to each identity or relation hypothesis;
4. an adjudication queue that separates same-parent controls, domain-matched cross-parent tests, registered audits and cross-domain exploratory hypotheses.

Only after the whole queue is visible should individual evidence cards be selected.  A card is explanatory follow-up to a global analysis, not a substitute for it.

This release now materialises that evidence layer rather than leaving it as a plan:

| Traceable ledger | Rows | Conservation check |
|---|---:|---:|
| Study-internal response effects | 238,412 | Common identity policy covers all 54,861 study-entities |
| WSE identity-hypothesis edges | 67,465 | 34,219 all / 918 known / 32,328 unresolved |
| Identity edge × contrast contributions | 1,315,906 | Maximum strength reconstruction error `3.41e-13` |
| Identity connection/world rows | 6,858 directed / 3,429 undirected | Edge counts, weighted counts and dual-q counts reproduce the frozen score table |
| Legacy different-mass relation edges | 23,989 | Kept in a separate exploratory namespace |
| Legacy relation edge × contrast contributions | 315,584 | Maximum strength reconstruction error `4.26e-14` |

The full ledger explicitly represents missing entity-by-contrast effects as `effect_observed=False` with zero response weight.  It does not silently delete the spectral edge.  This matters because the 238,412-row response table is not rectangular within multi-contrast studies.

The full contribution files are reproducible data products and are not themselves biological validation.  Their purpose is to make every selected study-level link auditable across **all** supporting signals, rather than explaining a global result with a Top-10 subset that often accounts for only a small fraction of total response strength.

The compact panorama, edge tables, connection-strength tables and reports are versioned with the code.  The deterministic full contribution matrices are retained as local generated artifacts because the compressed identity matrix is about 136 MB and the legacy relation matrix about 35 MB; their SHA-256 values and conservation summaries are frozen in the reports, and the committed builder regenerates them from existing frozen inputs.

## Success criteria

The Atlas becomes a convincing biological application only when the same frozen analysis shows all three quantities separately:

- **unresolved-information increment:** known plus unresolved provides additional independently supported responses after controlling coverage and match opportunity;
- **algorithm increment:** the in-house method changes identity or relation decisions on the same complete axis and increases supported responses relative to WSE/Official/Noise, without looser entry rules;
- **biological confirmation:** at least one added or corrected research-level relation is supported by evidence not used in connection selection or parameter tuning.

Until then, the correct description is: **a large, traceable response-hypothesis Atlas under active biological adjudication**, not a validated biological knowledge graph and not a failed small benchmark.
