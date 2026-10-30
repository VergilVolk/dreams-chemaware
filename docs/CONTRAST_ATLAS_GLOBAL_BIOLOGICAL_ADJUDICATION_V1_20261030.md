# Contrast Atlas global biological adjudication v1

## Outcome and denominator

This release evaluates the complete **51-contrast, 15-source, development/consumed** Atlas at two explicitly separate evidence layers and expands every connection into all supporting entity-response contributions. It contains 6,858 connection/world diagnostics, 306 query/world/layer answer records, 54,861 entity library-evidence records and 67,465 method-edge annotations. No selected hypothesis is labelled correct or false.

| Evidence layer | Selected / 51 | Same parent | Cross parent | Cross-parent same species+matrix | Targets ChemProp parent |
|---|---:|---:|---:|---:|---:|
| legacy_different_mass_relation_hypothesis_exploratory | 38/51 | 6 | 32 | 0 | 23 |
| same_precursor_identity_hypothesis | 49/51 | 23 | 26 | 0 | 36 |

The identity layer is a same-precursor WSE identity-hypothesis graph. The relation layer is the older different-mass mutual fragment-cosine graph and remains exploratory. Their agreement is a prioritisation signal only: 38 queries are answered by both, but only 2 select the same target and response face.

## Global identity-track result

- The all-world identity track selects a statistical hypothesis for **49/51** queries, representing **44 unique undirected links**; **45/49** choose the opposite-response face.
- **23** answers remain within one operational parent project and **26** cross parent projects. Among cross-parent answers, **21/26** mismatch species, biospecimen and condition family simultaneously; **0** preserve both species and biospecimen.
- Candidate-source concentration is severe: parent-project HHI is **0.561** (effective parent projects **1.78**), and 36/49 answers target the shared ChemProp parent.
- The median cross-parent Top-1 and Top-10 contribution fractions are **2.256%** and **19.119%**. Median effective edge count is **91.3**. Thus most answers are distributed rather than one-edge artefacts, but this does not make them biologically valid.
- Only **3.66%** of cross-parent selected-face strength comes from edges significant at study-internal q<=0.05 at both endpoints.

## What the unresolved evidence actually contributes

Across the 49 selected identity answers, selected-face strength is composed of:

| Endpoint evidence class | Strength fraction | Interpretation |
|---|---:|---|
| exact full InChIKey at both endpoints | 2.506% | shared qualified library candidate; not MSI level 1 |
| discordant qualified identities | 0.313% | explicit identity conflict; never merge effects |
| one named endpoint only | 1.035% | named anchor only; never propagate the name |
| unresolved at both endpoints | 96.145% | structure-agnostic evidence; correctness unproven |

The all-world and unresolved-world target plus face agree for **46/49** answered queries; the corresponding all-versus-known agreement is only **3/49**. This proves numerical dependence on unresolved evidence, not unknown-information validity or increment.

Qualified library support is not equivalent to exact cross-study identity. Of the 1,785 double-supported method-edge rows across the three worlds, 417 have discordant InChIKeys. The released edge annotation therefore blocks name propagation and effect merging for those edges.

## Prespecified relation audit

The 42 registered directions are all unsigned and development/consumed. Among **34** eligible-primary directions, the identity track selects the registered target for only **2/34**:

- exact-drug cross-design: **2/24**;
- HFD intervention family: **0/4**;
- IBD condition family: **0/6**.

The registered target itself ranks Top-1 on its better face for **3/34** directions and Top-5 for **20/34** (median best-face rank **5.0**). Although **28/34** target-face tests have query-maxT p<=0.05 before the final two-face choice, competing candidates dominate the open answer. Complete absence of spectral edges is therefore not the sole explanation; source attraction, response nonspecificity and candidate discrimination remain unresolved alternatives.

The two exact-drug hits are the reciprocal cilnidipine pair inside the consumed ChemProp parent project; they are a technical control, not external biological confirmation. The identity track also selects 2/8 post-hoc obesity directions. Those are the only same-species, same-condition cross-parent reciprocal candidates, but use different biofluids and have no dual-q supporting edge; they remain hypotheses.

## Released Atlas tables

- `identity_global_answer_adjudication.tsv`: every identity query, including two abstentions.
- `relation_global_answer_adjudication.tsv`: every exploratory different-mass relation query, including thirteen abstentions.
- `identity_selected_link_evidence.tsv.gz`: 44 unique selected links with complete contribution, concentration, missingness, library evidence and relation-axis diagnostics.
- `registered_target_audit.tsv`: all 42 registered directions, including every miss.
- `source_concentration.tsv`: candidate-source indegree, same/cross-parent counts and concentration.
- `entity_library_evidence.tsv.gz` and `identity_method_edge_library_annotation.tsv.gz`: normalised endpoint and edge evidence without copying names into the 1.32-million-row contribution matrix.

## What is now auditable

For every contrast pair, evidence world and response face, the connection table reports total strength, contributing edges, dual-q support, effective edge count, missing-effect fraction, Top-1/5/10 strength concentration and the known-known/mixed/unresolved-unresolved contribution fractions.  The all-world queue retains every query, including abstentions, and separates same-parent controls, registered unsigned audits, domain-matched tests, condition-only matches, cross-track convergence and cross-domain hypotheses.

This is the Atlas-wide interpretation layer that was previously missing. It prevents a Top-10 display, one IBD feature or one drug example from standing in for the global result.

## Scientific decision

The Atlas is now globally traceable, but **biological validity is not established**. The immediate P0 is to explain why the prespecified HFD and IBD targets are 0/10 and why exact-drug recovery is only 2/24 before promoting unregistered cross-domain links. The post-hoc obesity reciprocal pair is the first cross-parent candidate for independent adjudication; the B. subtilis-to-AD reciprocal pair and the ChemProp hub are bias diagnostics, not discoveries.

## Scientific boundary

All inputs are development/consumed. The collection contains no signed entity-level truth and no sealed biological validation panel. Source concentration, contribution concentration and track convergence do not establish mechanism or correctness. Official, Noise and Full23 still require complete-axis method-specific score packages on all 291,330 directed pairs before algorithm increment can be evaluated; the old 41,082-row fixed-WSE-edge packages are not admissible substitutes.
