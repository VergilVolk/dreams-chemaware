# Multi-encoder identity application result (2026-10-24)

## Scope

The recovered `run_2404230` was evaluated on the frozen MSV000095331 plus
ChemProp application handoff. This is consumed development evidence, not an
external validation. Identity hypotheses were selected without biological
response labels; the frozen response endpoint was joined only afterwards.

Fixed application denominators were 38,628 directional queries and 22,046
same-observed-precursor 10-ppm candidate pairs. The encoder delivery covered
18,274 of those pairs. Missing low-peak spectra remained explicit abstentions;
they were not removed from the denominator or imputed.

## Entity decisions

| Method | Selected directions | Forward mutual Top-1 | Forward mutual net vs Noise | Forward Top-1 changes vs Noise |
|---|---:|---:|---:|---:|
| WSE | 9,182 | 2,402 | -79 | 598 |
| Noise V1 | 9,163 | 2,481 | 0 | 0 |
| ChemAware Stage-1 | 9,163 | 2,481 | 0 | 469 |
| ChemAware Phase-A | 9,163 | 2,499 | +18 | 433 |
| Noise-ChemAware FullMSG | 9,163 | 2,493 | +12 | 432 |
| Untrained query-minmax mean | 9,163 | 2,476 | -5 | 294 |

Across all common answered directions, Stage-1, Phase-A and FullMSG changed
1,037, 1,003 and 1,077 Top-1 hypotheses relative to Noise, respectively. These
are real decision differences, but the application panel has no identity truth,
so they must not be called corrections.

The untrained mean did not dominate the specialists. It produced 24 exact
multi-candidate score ties and lost five forward mutual directions relative to
Noise. This rejects unconditional score averaging as the final integration
rule on this panel.

## Registered response endpoint

Thirty prespecified response contexts were evaluated. Five microbial strata
per mixture are repeated contexts rather than independent studies.

| Method | Registered Top-1 contexts | Rank improved / worsened vs Noise | Top-1 gained / lost vs Noise |
|---|---:|---:|---:|
| WSE | 2 | 10 / 14 | 0 / 0 |
| Noise V1 | 2 | reference | reference |
| ChemAware Stage-1 | 2 | 7 / 15 | 0 / 0 |
| ChemAware Phase-A | 4 | 9 / 7 | 2 / 0 |
| Noise-ChemAware FullMSG | 5 | 10 / 8 | 3 / 0 |
| Untrained query-minmax mean | 1 | 7 / 13 | 0 / 1 |

All three FullMSG Top-1 gains were sertraline/mix-1 contexts. They therefore
show a localized application signal, not broad transfer across drugs. Phase-A
gained two Top-1 contexts, spanning mix-1/sertraline and
mix-3/omeprazole. Loratadine/mix-4 remained difficult for every encoder.

## Algorithm decision

The application result supports keeping the encoder outputs as distinct
experts. It does not support simple averaging, and it does not yet select a
deployable router because biological responses were never allowed to choose
the spectral method.

The highest-value next training experiment is one candidate-competition model
trained only on the fixed MassSpecGym training partition. It should see the
raw specialist scores, their margins/disagreements, WSE and the existing local
spectral evidence, while retaining explicit missingness. The MassSpecGym held
partition is the development endpoint; GNPS remains a consumed evaluation
resource and cannot be presented as untouched confirmation. Application
responses remain evaluation-only.

This recovered application run contains WSE plus four encoder similarities. It
does **not** contain B, P2b, ChemAware V2, Noise MSG/RRF or BioAware. It must not
be described as the completed grand unified algorithm.

## Reproducible artifacts

- Application scoring: `data/validation/msv000095331_integrated_multiencoder_identity_run_2404230/identity_scoring/`
- Fixed-axis response join: `data/validation/msv000095331_entity_method_increment_multiencoder_run_2404230/`
- Deterministic summary: `data/validation/msv000095331_multiencoder_increment_summary_v1_20261024/`
- Summary entry point: `tasks/summarize_msv000095331_multiencoder_increment.py`

The deterministic summary exports method coverage, pairwise Top-1 agreement,
all changed identity hypotheses versus Noise, and prespecified response-rank
deltas. Its report records all input and output SHA-256 values.

## Entity attribution of the response gains (2026-10-25)

The five Top-1 gains were decomposed after scoring into the mutual identity
hypotheses that changed relative to Noise V1. This is an evaluation-only
analysis: biological responses were not used to fit, tune or select an
encoder. The decomposition exactly reproduces the change in each response
numerator.

| Encoder and registered response | Noise rank -> encoder rank | Gained / removed edges | Helpful / harmful changed edges | Net numerator change | Largest helpful edge share |
|---|---:|---:|---:|---:|---:|
| Phase-A, mix-1 full SynCom -> sertraline (MSV000093571) | 4 -> 1 | 113 / 112 | 120 / 105 | +6.251 | 1.44% |
| Phase-A, mix-3 SynCom-red -> omeprazole (MSV000093571) | 2 -> 1 | 120 / 117 | 126 / 111 | +9.395 | 1.22% |
| FullMSG, mix-1 group-4 slow -> sertraline (MSV000094899) | 8 -> 1 | 173 / 173 | 190 / 156 | +10.959 | 1.22% |
| FullMSG, mix-1 group-3 medium-slow -> sertraline (MSV000093571) | 2 -> 1 | 96 / 97 | 95 / 98 | +0.106 | 1.88% |
| FullMSG, mix-1 full SynCom -> sertraline (MSV000093571) | 4 -> 1 | 104 / 103 | 113 / 94 | +13.838 | 1.51% |

Across these five contexts, 1,208 identity hypotheses changed: 644 moved the
registered response numerator in the helpful direction and 564 opposed it.
The net gain therefore did not come from accepting every new edge. It combined
355 newly added concordant edges with removal of 289 discordant edges.

The structure-agnostic contribution is concrete but not yet externally
validated. Of the 644 helpful changes, 612 (95.0%) connected two unresolved
entities and 32 touched at least one library-supported entity. Across all
changed edges, 1,142/1,208 (94.5%) were unresolved on both ends. No single
helpful edge explained more than 1.88% of the positive contribution in any
gain context, so the registered gains are distributed response evidence rather
than one dominant annotated peak.

One prespecified omeprazole-product query (`MSV000095331` entity `11178`) is a
particularly useful orthogonal follow-up. Phase-A removed its Noise-selected
link to `MSV000093571` entity `3375` and added a link to entity `4908`; those
two changes contributed +0.324 and +0.540 to the response numerator,
respectively. Both candidate links remain unresolved identity hypotheses.
They are therefore a falsifiable target for retention-time, adduct/isotope and
targeted-MS/MS review, not a confirmed molecular correction.

The follow-up table deliberately includes the six largest helpful and six
largest harmful changed edges per gain context. This prevents the application
story from being assembled only from favorable examples.

Reproducible attribution artifacts:

- `tasks/attribute_msv000095331_multiencoder_response_gains.py`
- `data/validation/msv000095331_multiencoder_response_gain_attribution_v3_20261025/`

### Biological interpretation boundary

These results establish that encoder choice changes which unresolved entities
carry a study-level response and that this can change prespecified cross-study
retrieval. They do not yet establish the molecular identities of those edges,
independent-study replication, or a broad drug-mechanism claim. The five
microbial strata are repeated contexts in one application, not five independent
studies. This remains consumed development evidence.
