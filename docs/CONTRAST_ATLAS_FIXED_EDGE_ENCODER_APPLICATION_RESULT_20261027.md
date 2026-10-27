# Contrast Atlas fixed-edge encoder application result — 2026-10-27

## Verdict

This is the first end-to-end use of frozen DreaMS-family representations in the
15-study Metabolomic Contrast Atlas open response task.  The result supports a
narrow but real conclusion:

> Reweighting already-frozen local chemical-relation edges with a learned
> spectrum representation improves cross-design response retrieval relative to
> a coverage-matched fragment-cosine baseline.  The gain is concentrated in the
> nonmass chemical-relation track.  ChemAware Stage-1 does **not** outperform
> official DreaMS overall on this panel.

This experiment does not yet establish an unresolved-entity increment, an HFD
retrieval success, a new molecular identity, or an externally confirmed
biological discovery.  All sources and registered relations are
development/consumed.

## Frozen application task

| Item | Denominator / status |
|---|---:|
| Independent studies / source families | 15 / 15 |
| Study-level directed contrasts | 51 |
| Response rows | 238,412 |
| Unique study entities | 54,861 |
| Frozen directional spectral edges | 294,445 |
| Canonical mutual edge rows | 41,082 |
| `all`-world unique edges | 20,485 |
| Same-precursor 10-ppm identity-hypothesis rows | 17,093 |
| Nonmass chemical-relation-hypothesis rows | 23,989 |
| Directed query-candidate contrast pairs | 2,286 |
| Candidate contrasts per query | 39–50 |
| Whole-profile response permutations | 256 |
| Data role | development / consumed |

The edge set, response effects, candidate universe, permutations and registered
relations were identical across methods.  A method could only replace the
strength on the exact frozen join key:

`left_study_id, right_study_id, evidence_world, left_entity_id, right_entity_id`

No method was allowed to add or remove an edge.  Registered biological
relations were read only after the complete score ledger had been written and
hashed.  Thus this experiment evaluates edge weighting, not candidate recall.

## Encoder coverage

The two encoder runs had identical application coverage:

| Method | Edge rows scored | Edge rows unavailable | `all` unique edges scored |
|---|---:|---:|---:|
| official DreaMS | 38,475 / 41,082 | 2,607 | 19,181 / 20,485 (93.64%) |
| ChemAware Stage-1 | 38,475 / 41,082 | 2,607 | 19,181 / 20,485 (93.64%) |

The remaining 306 unique endpoints had fewer than three usable fragment peaks
or were absent from the study MGF.  Missing model evidence was set to zero in
both runs.  The coverage-matched control likewise retained the original
fragment cosine only on these same available edges and set the rest to zero.

Checkpoint provenance:

- official DreaMS slim SHA-256:
  `e6ae44c8e05c954d403222dd2ef410be24c5fb41b4a0f6e0783b4863da410f97`
- ChemAware Stage-1 slim SHA-256:
  `dd2646152d585d02d21eb39e3b8c0faa255df26596ecd4ca9e9da8f3321f44e2`
- Frozen edge handoff SHA-256:
  `9b6eae168eefab7451d29603ac3090bf285f8d2d1e6b40a746baec3f454a2590`

The encoders did not read phenotype, response or registered-relation fields.

## Primary consumed positive control: exact drug cross-design retrieval

The primary diagnostic is the 24 directional relations formed by 12 undirected
drug-pair units.  The table below uses the `all` world and the nonmass chemical
relation track.

| Edge weight | Top-1 | Top-5 | Median rank | query max-T <= 0.05 |
|---|---:|---:|---:|---:|
| Original fragment cosine | 4/24 | 9/24 | 9.5 | 4/24 |
| Coverage-matched fragment cosine | 3/24 | 10/24 | 7.5 | 4/24 |
| official DreaMS cosine | 2/24 | 13/24 | 4.5 | 4/24 |
| ChemAware Stage-1 cosine | 4/24 | 13/24 | 4.5 | 4/24 |

Paired at the 12 undirected drug-pair units, relative to the
coverage-matched fragment baseline:

| Method | Better / worse / tied units | Median rank delta | One-sided sign p | One-sided Wilcoxon p |
|---|---:|---:|---:|---:|
| official DreaMS | 10 / 1 / 1 | -1.50 | 0.005859 | 0.001465 |
| ChemAware Stage-1 | 10 / 1 / 1 | -1.75 | 0.005859 | 0.004395 |

The direct ChemAware Stage-1 versus official DreaMS comparison was 5 / 5 / 2,
median rank delta 0, sign p = 0.623047 and Wilcoxon p = 0.744141.  Stage-1's
higher Top-1 count is therefore not evidence of an overall ChemAware-specific
gain.  Because the 12 drugs share the same small set of source studies, these
unit-level p values are development diagnostics, not source-independent
generalization tests.

The same improvement was not present in the combined or same-precursor tracks.
This localization is scientifically useful: the learned representation adds
most in weighting relations between distinct-mass observations, whereas the
same-precursor identity-hypothesis track is already close to saturation on the
consumed drug control.

## IBD case: distributed cross-study response evidence

For adult IBD (`IBD200_GROUP`) queried against pediatric VEO-IBD
(`MSV000097610`), both learned encoders returned the prespecified VEO-IBD
contrast as the Top-1 combined-track candidate among 50 candidates:

| Method | Rank | Query max-T p | Mutual edges | Positive-weight edges |
|---|---:|---:|---:|---:|
| official DreaMS | 1 | 0.003891 | 154 | 118 |
| ChemAware Stage-1 | 1 | 0.011673 | 154 | 119 |

In the reverse direction, adult IBD was rank 4 for both methods but remained a
calibrated candidate (official p = 0.015564; Stage-1 p = 0.007782).  This is a
prespecified near-exact condition-family control, not a signed mechanistic
truth.

For Stage-1, representative same-precursor hypotheses contributing to the adult
to pediatric match were:

| Adult entity | Pediatric entity | Precursor m/z | ppm | Stage-1 cosine | Adult beta | Pediatric beta |
|---|---|---:|---:|---:|---:|---:|
| `IBD200_GROUP::1767` | `MSV000097610::41087` | 512.33715 / 512.3380 | 1.659 | 0.8030 | +0.533 | +0.943 |
| `IBD200_GROUP::1286` | `MSV000097610::28796` | 595.34770 / 595.3475 | 0.336 | 0.8899 | -0.649 | -1.444 |
| `IBD200_GROUP::928` | `MSV000097610::13473` | 305.18563 / 305.1859 | 0.885 | 0.8593 | -0.467 | -1.796 |

None of these individual pairs is claimed as a confirmed molecular identity.
The adult-to-pediatric call had zero dual-FDR-significant individual edges; it
is an aggregate, whole-profile-permutation-calibrated response pattern.  This is
the intended method capability, but it still requires independent molecular or
prospective biological validation.

## Unresolved-entity boundary

The `all` world did not show a robust rank advantage over `known` at the 12
undirected exact-drug units:

| Method / track | Better / worse / tied | Median all-minus-known rank | Sign p | Wilcoxon p |
|---|---:|---:|---:|---:|
| official / nonmass relation | 7 / 5 / 0 | -1.25 | 0.387207 | 0.195312 |
| Stage-1 / nonmass relation | 7 / 5 / 0 | -1.50 | 0.387207 | 0.190186 |
| official / combined | 7 / 4 / 1 | -2.00 | 0.274414 | 0.076172 |
| Stage-1 / combined | 8 / 4 / 0 | -1.75 | 0.193848 | 0.106689 |

The adult-to-pediatric IBD result is supported separately by known and
unresolved subsets in one direction, while the reverse direction becomes
max-T significant only in the combined `all` world.  This is useful conditional
evidence that unresolved spectra can participate in a response result, but it
does not establish population-level unresolved information gain.  Entity-count
and measurement-opportunity differences remain possible explanations.

## Negative and unresolved results

- Neither encoder produced a calibrated HFD relation; ChemAware Stage-1 had
  0/4 Top-5 nonmass directional relations, while official DreaMS had 2/4 but
  still 0/4 max-T passes.
- ChemAware Stage-1 did not outperform official DreaMS as a whole.
- The experiment reweighted a fragment-cosine-admitted graph and therefore did
  not test whether the learned encoders recall new candidate relations.
- Stage-1 and official DreaMS changed 111/918 open top-hit decisions relative to
  one another.  Those changes are unadjudicated hypotheses, not corrections.
- No structure, pathway or mechanism is inferred from a nonmass relation edge;
  distinct entities never have beta or SE values merged.
- This run does not include Noise V1, P2b, ChemAware V2, the complete integrated
  ranker or BioAware.  Their future contribution must be evaluated on this same
  fixed handoff rather than inferred from the Stage-1 result.

## Scientific conclusion and next gate

The application now demonstrates more than an annotation-count change: learned
spectrum representations alter local chemical evidence and improve a real,
open-universe study-response retrieval task on a consumed exact-intervention
control.  The improvement is strongest precisely on the relation track where
different chemical observations must not be collapsed into one identity.

However, the current result supports **learned local relation weighting**, not
yet a ChemAware-specific effect and not yet the central unbiased-discovery
claim.  The next high-value test is to pass the frozen full integrated algorithm
through the exact same edge handoff, then require an independent study in which
`all` supplies a correct calibrated answer that matched `known` evidence cannot
supply.  Until that gate passes, unresolved-entity benefit remains a hypothesis.

## Reproducible outputs

- Fixed Stage-1 edge scores:
  `data/validation/contrast_atlas_fixed_edge_embedding_scores_stage1_v2_20261027`
- Fixed official DreaMS edge scores:
  `data/validation/contrast_atlas_fixed_edge_embedding_scores_official_v1_20261027`
- Coverage-matched fragment run:
  `data/validation/contrast_atlas_open_local_edge_enrichment_fragment_matched_stage1_v1_20261027`
- official DreaMS response run:
  `data/validation/contrast_atlas_open_local_edge_enrichment_official_v1_20261027`
- ChemAware Stage-1 response run:
  `data/validation/contrast_atlas_open_local_edge_enrichment_stage1_v1_20261027`
- Final four-method comparison:
  `data/validation/contrast_atlas_fixed_edge_algorithm_comparison_stage1_official_v3_20261027`
