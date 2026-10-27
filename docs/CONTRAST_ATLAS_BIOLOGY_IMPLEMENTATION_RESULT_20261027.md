# Contrast Atlas biology implementation result — 2026-10-27

## Verdict

This delivery implements the requested study-design registry, a model-independent complete candidate axis, an entity-axis change ledger, and a concrete IBD decision-block handoff. It does **not** establish a flagship biological discovery yet.

The strongest current IBD hypothesis is an unresolved stercobilin-like m/z 595.348 response observed in three consumed fecal studies. The strongest method-control case is a same-mass deoxycholic-acid–5-ASA identity competition. Both are now frozen as complete candidate blocks for WSE, Noise and Full23 v1 scoring. None of those complete-block algorithm scores existed locally at the time of this delivery. Score cells are therefore empty and availability is represented in separate status fields; the targeted algorithm handoff contains no score columns at all.

All datasets in this result are `development_consumed`. There is no signed entity-level truth and no unopened biological validation panel.

## 1. Design eligibility: what the 51 contrasts actually represent

The frozen input is `contrast_atlas_open_response_collection_v9_frozenbase_chemprop_20261023`, not the staged 238,496-row collection that retained the invalid adult-IBD effect.

| Quantity | Frozen denominator | Interpretation |
|---|---:|---|
| Analytic sources | 15 | Not necessarily independent projects |
| Study-level contrast rows | 51 | Not 51 independent validations |
| Response rows | 238,412 | All have finite beta and positive SE |
| Common identity entities | 54,861 | 979 library-supported; 53,882 unresolved |
| Directional registered relevance rows | 42 | Collapse to 21 canonical pair edges |
| Canonical primary pair edges | 17 | Collapse to 14 connected components |
| Cross-project primary pair edges | 5 | Collapse to only two components: HFD and IBD |
| ChemProp registered directions | 24 | Twelve drugs nested in one broader project |
| Registered signed entity truth | 0 | Similar/opposite accuracy is not currently a truth endpoint |
| External-validation contrasts | 0 | Entire collection remains consumed development |

Identity eligibility now uses the common GNPS Gold/Silver partition. Source-native labels differed for 1,418 entities and would have inflated the known subset from 979 to 1,249. Both common and source-native counts are retained in the eligibility table, but method comparisons must use the common partition.

Every contrast now points to an exact effect input. Response-directory sources point to `<root>/<contrast>/response_object.tsv.gz`; the adult IBD row points to the corrected Plate3 36-case/19-control response rather than the withdrawn 158-case/19-control artifact.

Acquisition conditions remain conservative: the frozen central collection does not harmonize instrument, polarity and chromatography across all 15 sources. They therefore cannot silently serve as matched variables. A future manuscript table may add source-specific values only with explicit evidence and per-field status.

Outputs:

- `data/validation/contrast_atlas_contrast_eligibility_v3_20261027/contrast_eligibility.tsv`
- `data/validation/contrast_atlas_contrast_eligibility_v3_20261027/registered_relation_independence.tsv`
- `data/validation/contrast_atlas_contrast_eligibility_v3_20261027/report.json`

## 2. New model-independent identity-candidate axis

Candidate admission uses no model score, beta, SE, phenotype or registered relation. The response table is read only to recover the frozen raw-spectrum-to-entity lineage. It admits all source-disjoint spectra within 10 ppm observed precursor mass after excluding explicit charge, polarity or adduct conflicts. Missing ion chemistry is retained and labelled; it is not silently treated as confirmed compatibility.

| Quantity | Result |
|---|---:|
| Entities | 54,861 |
| Query × target-source blocks | 768,054 |
| No-candidate abstention blocks | 628,925 |
| Singleton blocks | 81,871 |
| Multi-candidate blocks | 57,258 |
| Maximum candidates in one block | 44 |
| Directed mass-window pairs before explicit conflicts | 297,370 |
| Explicit conflicts excluded | 6,040 |
| Directed eligible pairs | 291,330 |
| Canonical eligible pairs | 145,665 |

This is an observed-precursor identity **hypothesis axis**, not a confirmed molecular-identity graph. No adduct metadata are available in the frozen representative spectra, so cross-adduct neutral-mass identity is not claimed. Isomers, isotopes, in-source fragments and ion-family redundancy remain possible.

The essential advance over the legacy graph is experimental, not rhetorical: every nonempty query block is now complete before scoring. Full23 query-local percentiles and gaps can therefore be computed honestly rather than pretending a mutual-Top1 edge is a singleton query. Every block has a stable `query_block_id`, a zero-based contiguous `query_block_index`, and half-open candidate offsets. Every candidate has a direction-specific `directed_pair_id` and a block-local index, so all 768,054 blocks—including zero- and one-candidate blocks—can be processed without changing the denominator.

Outputs:

- `data/validation/contrast_atlas_model_independent_identity_axis_v5_20261027/observed_ion_registry.tsv.gz`
- `data/validation/contrast_atlas_model_independent_identity_axis_v5_20261027/directed_complete_candidate_pairs.tsv.gz`
- `data/validation/contrast_atlas_model_independent_identity_axis_v5_20261027/query_target_source_blocks.tsv.gz`
- `data/validation/contrast_atlas_model_independent_identity_axis_v5_20261027/report.json`

## 3. Old/new entity-axis ledger

The comparison against the legacy all-world fixed graph gives:

| Axis status | Canonical pairs |
|---|---:|
| Legacy all-world unique edges | 20,485 |
| New model-independent identity candidates | 145,665 |
| Union | 157,704 |
| Retained identity candidates | 8,446 |
| Gained complete-block candidates | 137,219 |
| Lost after explicit ion conflict | 42 |
| Legacy nonmass relation-only edges | 11,997 |

These labels describe candidate-axis membership, not correct/incorrect model decisions. The 11,997 nonmass legacy edges remain chemical-relation hypotheses and are not called lost identities.

Old fragment cosine, Official DreaMS and Stage-1 scores are available only on the old graph. WSE, Noise, Full23 v1 and experimental v1.1 are absent on the complete axis. The ledger records that absence with blank numeric score cells plus explicit availability fields rather than writing textual placeholders into score columns or copying another method's value. It reserves separate left-to-right and right-to-left score fields and IDs because block-normalized rankings are query-direction dependent.

Outputs:

- `data/validation/contrast_atlas_entity_change_ledger_v7_20261027/entity_change_ledger.tsv.gz`
- `data/validation/contrast_atlas_entity_change_ledger_v7_20261027/registered_response_evidence.tsv.gz`
- `data/validation/contrast_atlas_entity_change_ledger_v7_20261027/report.json`

## 4. IBD decision blocks: a hypothesis and a falsifiable control

### 4.1 Stercobilin-like m/z 595.348 hypothesis

| Entity | Common identity | beta ± SE | Within-contrast BH q | Boundary |
|---|---|---:|---:|---|
| `IBD200_GROUP::1041` | unresolved | -1.964 ± 0.601 | 0.0799 | Plate3 adult IBD, 36 vs 19 |
| `MSV000097610::16870` | unresolved | -4.033 ± 1.063 | 0.0153 | pediatric VEO-IBD, 21 vs 27 |
| `MSV000096564::13749` | unresolved | -6.360 ± 1.999 | 0.2286 | baseline Crohn, 11 vs 11 |

What is real:

- three source families, all human feces, all effects point downward;
- adult Plate3 restriction preserves the sign, so the historical perfect plate confounding does not alone create it;
- pediatric and the second adult spectrum form a legacy mutual-Top1 edge at 0.672 ppm with fragment cosine 0.94856;
- the new axis admits all three pairwise links, including adult-Plate3 ↔ pediatric, which the old mutual graph missed.

What is not established:

- all three are unresolved under the current common Gold/Silver policy; the earlier Stercobilin label came from a legacy Bronze hypothesis;
- only the pediatric entity passes within-contrast BH q<0.05;
- the three beta scales differ, so inverse-variance pooling is invalid;
- adult healthy ages are missing and medication, antibiotics, activity and diet are not jointly adjusted;
- all three studies were already consumed during development.

Therefore the valid label is **stercobilin-like three-study response hypothesis**, not identified Stercobilin and not an IBD mechanism.

### 4.2 m/z 528.332 deoxycholic-acid–5-ASA identity disambiguation

The pediatric query `MSV000097610::37125` has four eligible adult candidates in its complete block. Two key same-mass candidates are:

- expected source-annotation identity candidate `IBD200_GROUP::1547`, 0.303 ppm, beta 0.648 ± 0.215;
- different-InChIKey same-mass competitor `IBD200_GROUP::1527`, 0.379 ppm, beta 0.061 ± 0.082.

The expected pair shares the source-reviewed full InChIKey `QLCGWJXUZVMSLH-SUMDJKQASA-N`; the competitor is a chenodeoxy-5-ASA structure. This makes the four-candidate block a direct test of whether the integrated scorer resolves a biologically consequential isomer competition instead of merely finding a mass match.

It is not a disease-mechanism result. A reproducible read-only recomputation gives adult `1547` a current-5-ASA effect of 1.084 ± 0.312 (31 labelled yes versus 25 no), but all 56 labelled samples are UC because the medication field is missing for CD. The available third 5-ASA study also failed its frozen confirmation gate. Its role is therefore a medication-exposure/identity-topology control.

Each manual identity or medication statement is now linked to its source path, record key or decompressed line and SHA-256 in the candidate registry. This includes the two legacy Bronze stercobilin annotations, the current common-identity shards, and the quantification, metadata and implementation used for the adult medication-effect recomputation.

Outputs:

- `data/validation/contrast_atlas_ibd_decision_blocks_v5_20261027/complete_block_algorithm_handoff.tsv.gz` — 24 rows, nine directed blocks, no model preselection and no response/adjudication/score columns;
- `data/validation/contrast_atlas_ibd_decision_blocks_v5_20261027/pair_adjudication_evidence.tsv.gz` — response effects, registered roles and legacy-edge evidence kept outside the algorithm input;
- `data/validation/contrast_atlas_ibd_decision_blocks_v5_20261027/query_target_block_manifest.tsv`;
- `data/validation/contrast_atlas_ibd_decision_blocks_v5_20261027/candidate_response_evidence.tsv`;
- `data/validation/contrast_atlas_ibd_decision_blocks_v5_20261027/report.json`.

## 5. What the algorithm handoff must return

Primary join keys:

`query_block_id` and direction-specific `directed_pair_id`

The explicit four-field key (`query_study_id + query_entity_id + candidate_study_id + candidate_entity_id`) remains available for human audit.

For WSE, Noise and Full23 v1, return raw score, within-block rank, block size, evidence availability and model/artifact hash. Score every row in the block. Do not use beta, SE, response sign or candidate-set label during scoring. v1.1 is experimental and must remain a separate later column.

The 24-row targeted handoff is a fast candidate-adjudication experiment selected after seeing IBD responses. It cannot estimate unbiased system-wide gain. A method-level claim still requires scoring the full 291,330 directed-pair axis and comparing correct/incorrect/abstain with a truth source that is independent of response selection.

## 6. Current paper boundary

What can be written now:

> A model-independent complete candidate axis exposes many cross-study identity hypotheses excluded by a legacy mutual spectral graph and supplies falsifiable entity-level decision blocks for integrated scoring.

What cannot be written now:

- Full23 improved the IBD application;
- unresolved entities provide an independent biological increment;
- the stercobilin-like group is structurally identified;
- IBD response is externally replicated or mechanistically explained;
- 51 contrasts or 17 primary edges are independent validations.

The highest-value next computation is fixed: score the frozen 24-row IBD handoff immediately for candidate adjudication and the 291,330-row full axis for unbiased accounting. No further disease expansion, threshold tuning or Atlas renaming is needed before those scores arrive.

## Verification

`python -m unittest tasks.test_contrast_atlas_biology_delivery -v`

Result: 5/5 tests passed after final regeneration. The tests enforce the common identity denominator, 15/51/238,412 eligibility boundary, score-blind admission, contiguous block offsets, response/adjudication-blind algorithm handoff, empty undelivered score cells and the directional score schema.
