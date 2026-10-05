# BioAware B47-U1c seed denominator result and U2 route

Date: 2026-09-21  
Scope: truth-blind B47 context readiness; no annotation truth, phenotype, event score, model fit, P2b, or shared-embedding claim.

## Result

The seed shortage is not caused by insufficient spectrum confidence or insufficient cross-sample consensus.

| Frozen stage | Query events | Candidate identities | Interpretation |
|---|---:|---:|---|
| all queries | 51,976 | 4,300 | external query denominator |
| unique Top-1 | 51,781 | 4,300 | score ties do not cause identity loss |
| score >=0.80 and margin >=0.05 | 5,983 | 379 | still above the 200-identity gate |
| absolute gate plus feature consensus | 5,480 | 238 | still above the gate |
| Rhea-covered, noncurrency, degree <=250 | 3,603 | 127 | first stage below 200 |
| sample-candidate collapse | 3,243 | 127 | removes repeated events, not identities |

The candidate universe contains 10,578 identities, including 865 with a Rhea edge and 861 passing the Rhea degree/noncurrency screen. Therefore Rhea is not globally too small; it fails to cover enough of the 238 identities selected by the frozen spectral-consensus policy.

Top-1 identity concentration is not the primary explanation: all queries contain 4,300 Top-1 identities, the ten most frequent identities account for 3.85% of events, and the HHI is 0.001084.

## Scientific decision

The fixed route is `RHEA_COVERAGE_OR_HUB_BOTTLENECK`.

Consequences:

1. Stop tuning the max/mean spectrum unary. U1 already showed only +0.0682 pp with unsafe 131/74 transitions, and U1c proves the identity denominator survives both the absolute and consensus gates.
2. Do not lower score, margin, consensus, or 200-identity gates.
3. Do not score reaction events yet. The current 127 identities do not satisfy the frozen readiness contract.
4. Audit the 111 consensus identities lost at graph eligibility. Separate exact Rhea absence, Rhea currency/hub exclusion, strict-KEGG exact-edge recovery, EMRN-only expansion, and no-catalogue coverage.
5. EMRN-expanded edges cannot be relabelled as exact biochemical events. They may quantify headroom only.

## U2 contract

`tasks/run_bioaware_b47_u2_catalog_coverage.sbatch` performs the coverage audit using the frozen U1c ledger, Rhea, strict MetDNA2 KEGG, and MetDNA2 EMRN resources.

U2 may authorize only a strict-KEGG currency/hub curation stage. It cannot authorize event scoring, annotation claims, reranking, or embedding training. Exact-event work remains blocked until at least 200 spectrally qualified identities have source-labelled exact edges and all added identities pass a frozen currency/hub screen.

## Claim boundary

U1c establishes a denominator bottleneck, not BioAware performance. It does not show that KEGG, Rhea, EMRN, sample context, or a reaction model improves identification.
