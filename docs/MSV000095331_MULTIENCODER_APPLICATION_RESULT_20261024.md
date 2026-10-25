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

## Frozen-readout robustness (2026-10-25)

Four label-free readouts of the same frozen FullMSG representation were used as
perturbations, not selected using the response outcome: raw cosine,
study-centered cosine, pooled diagonal-whitened cosine and SimHash-256. The
purpose was to determine whether the raw-FullMSG biological gains and entity
actions were tied to one exact cosine calculation.

The three raw-FullMSG Top-1 gains had unequal but nonzero readout robustness:

| Raw-FullMSG gain | FullMSG readouts retaining Top-1 |
|---|---:|
| mix-1 full SynCom -> sertraline, MSV000093571 | 4/4 |
| mix-1 group-4 slow -> sertraline, MSV000094899 | 3/4 |
| mix-1 group-3 medium-slow -> sertraline, MSV000093571 | 2/4 |

The 746 changed edge-context instances corresponded to 567 unique identity
hypotheses. Of those unique hypotheses, 351 (61.9%) retained the same FullMSG
add/remove action under all four readouts. At the context-instance level,
470/746 (63.0%) actions were supported by all four readouts, including 254/398
helpful actions and 216/348 harmful actions. Thus a substantial portion of the
entity changes is not a raw-cosine numerical accident, although the readouts
share one representation and are not independent confirmations.

The 567-pair registry separates 270 pairs that helped in every context in
which they appeared, 225 that harmed in every such context, and 72 whose
response contribution changed sign between microbial contexts. The latter are
especially important falsification cases: a stable spectral identity action
does not guarantee a stable biological contribution.

No metric is promoted from this analysis. Raw cosine remains the frozen
baseline, consistent with the label-free metric experiment. The robustness
registry is used only to prioritize orthogonal examination of identity
hypotheses, and includes favorable and unfavorable cases.

Additional reproducible artifacts:

- `tasks/build_msv000095331_metric_robust_entity_followup.py`
- `data/validation/msv000095331_metric_robust_entity_followup_v2_20261025/`

## Orthogonal-validation packet and identity-evidence conflict (2026-10-25)

A balanced validation packet was frozen from the 351 unique identity actions
supported by all four FullMSG readouts. Selection took at most two pairs from
each observed combination of identity-evidence class and response-contribution
class. It therefore includes helpful, harmful and context-dependent cases
rather than only the largest favorable effects.

The packet contains 14 cross-study identity hypotheses, 25 unique spectra and
24 response-context instances across eight populated strata. It exports the
original MS/MS peaks, precursor and feature metadata, retention time where
available, response beta/SE contributions, source-native library matches and
the independent common Gold/Silver library candidates.

This packet exposed a concrete limitation of a binary `library_supported`
field. `MSV000095331` entities `2369` and `3062` both have m/z approximately
182.081 and both matched the same MassBank tyrosine spectrum in the source
workflow, but their within-study retention times are 1.255 and 1.529 minutes.
Both compete for `MSV000094899` entity `1459`. FullMSG adding the `2369 -> 1459`
edge harmed the registered response, whereas removing `3062 -> 1459` helped it.
Moreover, the source workflow calls candidate `1459` L-tyrosine, while the
frozen common Gold/Silver search assigns its best qualifying candidate to a
different planar InChIKey. This is one explicit cross-library structural
conflict, not an identity truth.

The consequence is methodological but bounded: response retrieval must retain
the provenance and disagreement of identity evidence. `library_supported`
remains useful for constructing matched controls, but it cannot be interpreted
as a unique molecular structure or used by itself to adjudicate whether an
encoder changed an entity correctly. Multiple same-mass, same-library-name
features within one study also cannot be silently collapsed or forced through
a one-to-one cross-study mapping without ion/RT evidence.

The frozen packet is ready for orthogonal review; it is not fed back into model
training or metric selection.

- Packet builder: `tasks/build_msv000095331_fullmsg_validation_packet.py`
- Final packet: `data/validation/msv000095331_fullmsg_validation_packet_v5_20261025/`

## Context-robust biological endpoint (2026-10-25)

The five microbial backgrounds within each drug mixture are related strata,
not independent confirmations. To prevent the three FullMSG context-specific
Top-1 gains from being counted as three biological replications, the frozen
response scores were aggregated with equal weight across the five backgrounds.
This yields six evaluable mixture-by-candidate-study units: three registered
mixtures in each of two studies containing the corresponding drug response.

| Method | Target Top-1 units, mean aggregation | Median target rank | Mean target rank | Improved / worsened / unchanged vs Noise |
|---|---:|---:|---:|---:|
| WSE | 1/6 | 4.5 | 5.00 | 1 / 2 / 3 |
| Noise V1 | 2/6 | 4.0 | 4.83 | reference |
| ChemAware Stage-1 | 1/6 | 4.5 | 5.33 | 0 / 2 / 4 |
| ChemAware Phase-A | 1/6 | 4.0 | 4.83 | 1 / 1 / 4 |
| Noise-ChemAware FullMSG | 1/6 | 4.0 | 4.83 | 1 / 1 / 4 |
| Simple normalized mean | 1/6 | 4.0 | 5.00 | 0 / 1 / 5 |

FullMSG therefore does not show a context-robust aggregate advantage over
Noise on this consumed panel. Its three raw-context Top-1 gains remain useful
evidence that the representation can change biologically relevant entity
support in particular microbial settings, but they do not survive as a broad
retrieval improvement after the related contexts are collapsed. Specifically,
FullMSG improves the loratadine target rank from 11 to 10 in `MSV000094899`,
but worsens the omeprazole target rank from 1 to 2 in the same study; the other
four mean-aggregated target ranks are unchanged from Noise.

The median-across-context sensitivity also gives every method 1/6 target
Top-1 units. With only six descriptive development units, no significance or
general-transfer claim is made. The correct current conclusion is localized
entity-level complementarity plus a prioritized validation set, not a proven
overall biological retrieval gain.

- Aggregation script: `tasks/summarize_msv000095331_context_robust_retrieval.py`
- Frozen result: `data/validation/msv000095331_context_robust_retrieval_v1_20261025/`
