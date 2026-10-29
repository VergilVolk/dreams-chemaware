# Contrast Atlas biological panorama v1

## What is now visible

This release organises all **15 sources, 13 operational dependence groups and 51 study-level contrasts** into one traceable biological panorama.  It contains 238,412 entity-by-contrast response rows from 54,861 study-entities, all 2,286 source-disjoint directed contrast candidates, and 2,881 top supporting entity-pair records with beta, SE, direction and spectral evidence.

The contrast-level unit count is not additive across contrasts: several contrasts reuse participants, organisms or experimental batches.  Every row retains its design, estimand, biological unit, covariates, causal boundary and dependence group.

## Complete-system view versus explanatory controls

The complete WSE identity graph yields **49/51** all-world two-face-calibrated answers.  Known-only yields **19/51** and unresolved-only yields **49/51**.  These are discovery yields, not numbers of correct biological links.

The anatomy of those 49 statistical answers is itself a global diagnostic: **23** select a contrast from the same operational parent project, **26** are cross-parent hypotheses, and **0** of the cross-parent answers preserve both species and biospecimen.  **36/49** select the shared ChemProp parent project.  These facts show that biological validity is unresolved; they do not identify whether the concentration is caused by coverage, chemistry, edge error, scoring or real nonspecific biology.

Opportunity-matched and parent-project-held-out experiments remain explanatory controls.  They do not replace this complete-system view, and their heavy edge overlap is not treated as independent biological replication.

## What each proposed connection contains

Every face-specific connection records the two study designs, parent projects, species, biospecimens, estimands, response face, evidence count, enrichment score, max-T value and registration status.  Its support ledger then identifies the observed-ion pairs, WSE score, precursor agreement, beta/SE on both sides, diagnostic q values and weighted contribution.  Identity hypotheses never merge beta/SE across different observed ions.

Current cross-project questions include:

| Query contrast | Face | Candidate contrast | Support edges | Status |
|---|---|---|---:|---|
| `MSV000083077::obesity_low_depression` | opposite | `MSV000084112::obesity_with_diabetes` | 721 | registered_cross_project_audit |
| `MSV000084112::obesity_with_diabetes` | opposite | `MSV000083077::obesity_low_depression` | 721 | registered_cross_project_audit |
| `MSV000090788::lesion_vs_nonlesion_paired` | opposite | `MSV000086360::bsubtilis_3610_vs_control_stem` | 341 | reciprocal_cross_project_adjudication |
| `MSV000090788::nonlesion_vs_healthy` | similar | `MSV000086360::bsubtilis_3610_vs_control_stem` | 341 | reciprocal_cross_project_adjudication |
| `MSV000086360::bsubtilis_3610_vs_control_stem` | similar | `MSV000090788::nonlesion_vs_healthy` | 341 | reciprocal_cross_project_adjudication |
| `MSV000086360::bsubtilis_3610_vs_control_stem` | opposite | `MSV000090788::lesion_vs_nonlesion_paired` | 341 | reciprocal_cross_project_adjudication |
| `IBD200_GROUP::adult_all_ibd_vs_healthy` | similar | `MSV000094899::drug_cilnidipine` | 97 | unresolved_supported_cross_project_adjudication |
| `MSV000083077::depression_nonobese` | similar | `MSV000094899::drug_telmisartan` | 649 | unresolved_supported_cross_project_adjudication |

These rows are an adjudication queue, not discoveries.  A candidate can reflect biology, drug exposure, biospecimen, coverage, identity error or a project-level response hub; those explanations must be examined using the attached signal-level evidence.

## Complete evidence materialisation

The separate connection evidence release expands this display layer into **1,315,906 identity edge-by-contrast contribution rows** spanning all **67,465** world-specific WSE identity-hypothesis edges.  Its reconstructed same/opposite strengths agree with the frozen retrieval table to at most `3.41e-13`.  Thus the biological interpretation no longer depends on the 2,881 top-support display rows.

An exploratory legacy different-mass relation namespace separately contains **23,989** relation edges and **315,584** edge-by-contrast contribution rows.  These are distinct-entity relation hypotheses and never molecular identity evidence.

## Scientific boundary

This is the first complete **presentation and evidence-traceability layer** for the existing response collection.  It does not yet establish biological validity, unknown-specific increment or Full23 increment.  It also covers only the same-precursor identity-hypothesis track: a valid different-mass chemical-relation layer remains absent.

The next scientific operation is not another null model.  It is systematic adjudication of the cross-project queue, followed by the same complete-candidate analysis with model-specific Official, Noise and Full23 connection decisions and score calibration.
