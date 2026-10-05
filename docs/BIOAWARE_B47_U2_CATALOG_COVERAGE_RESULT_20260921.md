# BioAware B47-U2 catalogue coverage result

Date: 2026-09-21  
Scope: truth-blind catalogue coverage headroom only. No annotation truth, phenotype, reaction-event score, model fitting, reranking, P2b, or shared-embedding claim.

## Frozen denominator

U2 starts from the 238 candidate identities that already passed the frozen spectrum-confidence and feature-consensus gates in U1c. These identities represent 394 consensus features and 5,480 query events across ST001122 and ST003356.

## Coverage result

| Coverage definition | Identities | Fraction of 238 | Shortfall to 200 |
|---|---:|---:|---:|
| Rhea covered | 129 | 54.2% | 71 |
| Rhea safe after noncurrency/degree screen | 127 | 53.4% | 73 |
| Rhea-safe or strict-KEGG-safe provisional | 141 | 59.2% | 59 |
| Rhea/KEGG/EMRN edge headroom | 149 | 62.6% | 51 |
| No catalogue edge | 89 | 37.4% | — |

Strict KEGG contributes only 14 additional provisional identities beyond Rhea-safe. The broad EMRN union reaches 149 identities, but EMRN is a reaction-neighbour expansion rather than a source-labelled exact reaction event and is not eligible to close the exact-event gate. The union adds eight identities beyond Rhea+strict-KEGG; six are clean EMRN-only coverage classes and two overlap identities already labelled as Rhea-blocked.

The safety filter is not the main bottleneck: only two unique Rhea-covered identities are removed between 129 covered and 127 safe. The dominant loss is absence of a catalogue edge for 89 of 238 consensus identities.

The pattern is present in both sources:

| Study | Consensus identities | Rhea safe | Rhea+strict KEGG provisional | Broad edge headroom |
|---|---:|---:|---:|---:|
| ST001122 | 175 | 98 | 106 | 113 |
| ST003356 | 140 | 81 | 90 | 94 |

Study identity counts overlap and must not be summed. Similar coverage fractions across both studies argue against a single-source ingestion failure.

## Scientific decision

The fixed decision is `CATALOGUE_COVERAGE_UNDERPOWERED`.

1. B47 exact-event scoring, reranking, and embedding work remain blocked.
2. Do not lower the 200-identity gate, spectrum thresholds, consensus requirement, or currency/hub safety screen.
3. Do not treat EMRN, pathway co-membership, mass-difference neighbours, or static catalogue membership as exact events.
4. Do not reopen B44--B46-style static catalogue/topology scoring: those questions already have closed negative evidence and U2 measures coverage, not performance.
5. U2 does **not** establish that 127 safe seed identities are statistically underpowered for a candidate-specific event model. Those identities generate 3,243 sample-collapsed seed events, and one seed identity can support many independent query--candidate competitions. The next admissible stage is therefore a truth-blind event-yield preflight, not an immediate search for 59 replacement identities.
6. A broader exact catalogue or independent seed source becomes necessary only if the event-yield preflight shows inadequate candidate-specific query/error opportunity. It is not a prerequisite for constructing the event table.

## Claim boundary

U2 proves limited seed-identity diversity under the old 200-identity gate, not inadequate event-level power. It is neither a BioAware accuracy failure nor a BioAware gain. It does not establish that Rhea, KEGG, EMRN, reaction context, reranking, or shared-embedding training improves annotation.

Canonical server result: `data/validation/bioaware_b47_u2_catalog_coverage_20260921_v1/report.json`.
