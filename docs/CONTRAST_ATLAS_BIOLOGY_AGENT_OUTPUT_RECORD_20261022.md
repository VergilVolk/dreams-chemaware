# Contrast Atlas biology agent output record — 2026-10-22

## Repository snapshot

- Branch: `codex/contrast-atlas-panredu-20261016`
- Last completed scientific commit before this record: `409d33b` (`Measure structure-free cross-adduct rank stability`)
- Scope: biology-application work only. Pre-existing modified and untracked files outside the paths listed here were not staged, rewritten, or deleted.
- Evidence role: all numerical results below are `development_consumed` or `development_consumed_posthoc_diagnostic`; none is an unopened external validation result.

## Fixed scientific objective

The application objective remains Metabolomic Contrast Atlas: use study-level, directed, quantitative responses from public untargeted LC-MS/MS studies as searchable evidence, while retaining structurally unresolved entities. Candidate identity and local chemical relation are separate scientific objects. Only identity-compatible evidence may accumulate effects as the same entity; related spectra may support response-pattern hypotheses but may not be relabelled as the same molecule.

The immediate algorithm–biology handoff asks one fixed question: on the same query/candidate axis and with the same response model, does replacing WSE by B or the integrated algorithm add reliable chemical evidence and thereby add correct, independently supported response information?

## Corrected IBD Plate 3 application base

- Full denominator: 662 corrected Plate 3 response entities from 55 participants (36 IBD and 19 healthy).
- The earlier 177-person result is historical and must not be used.
- The original protonated candidate axis contained 6,813 candidate edges and allowed WSE to answer 354 queries, but it is not safe for strict biological identity interpretation.

The adduct audit (`contrast_atlas_ibd200_candidate_adduct_audit_v1_20261022`) found:

| Quantity | Result |
|---|---:|
| Queries | 662 |
| Queries with explicit ion and neutral mass | 58 |
| Queries with unresolved adduct | 604 |
| Original candidate edges | 6,813 |
| Explicitly neutral-mass compatible edges | 151 |
| Explicitly incompatible edges | 366 |
| Adduct-unresolved edges | 6,296 |
| Original WSE Top-1 compatible / incompatible / unresolved | 7 / 25 / 322 |

Status: `ORIGINAL_PROTONATED_CANDIDATE_UNIVERSE_NOT_APPLICATION_SAFE`. This invalidates strict biological use of that candidate universe; it does not invalidate spectrum-only evaluation under its stated protonated-spectrum contract.

## Adduct-aware fixed application axis

The replacement application slice is `contrast_atlas_ibd200_plate3_adduct_aware_slice_v2_20261022`; its audit is `contrast_atlas_ibd200_plate3_adduct_aware_candidate_audit_v2b_20261022`.

| Quantity | Result |
|---|---:|
| Full query denominator | 662 |
| Explicit ion plus neutral-mass queries | 58 |
| Adduct-unresolved abstentions | 604 |
| Queries with candidates | 45 |
| WSE-answerable queries | 38 |
| Candidate structures / edges | 509 / 509 |
| Explicit compatible / incompatible / unresolved candidate edges | 509 / 0 / 0 |
| Candidate library-spectrum edges | 865 |
| Cross-adduct candidate structures / queries | 353 / 34 |

Status: `ADDUCT_AWARE_CANDIDATE_UNIVERSE_VERIFIED` on this development slice. The exact WSE score-axis artifact has SHA-256 `f7f836ee659bf5cb713f5ab71dac8fb28bb9999455e18eacef3201f2d6b88b1a`.

## Algorithm-to-public-response result

The fixed evaluation is `contrast_atlas_ibd200_adduct_aware_wse_public_response_links_v2_20261022`.

- Fixed denominators: 662 queries, 509 candidate edges, two prespecified independent IBD target studies, 228 public response anchors representing 210 structures.
- Thirteen candidate edges and ten queries had a public-response opportunity.
- WSE answered 38 queries and selected a public anchor for nine: two library-supported queries and seven unresolved queries.
- These yielded 22 held-out-fold response links. Descriptive direction concordance was 0.5455 overall and 0.5000 among unresolved queries.
- The strict gate required Plate 3 BH `q <= 0.05` across all 662 entities, at least one public target with `|z| >= 1.96`, both target studies represented, and concordant held-out directions.
- Number of unresolved candidates passing the complete gate: **0**.

This is a valid negative/development result, not a failed software run. The earlier direction-only bile-acid-like lead did not pass because the full Plate 3 effect was beta `-0.0444`, SE `0.0669`, and was not significant.

## BioAware candidate-context results

The first candidate-specific context analysis used 6,813 edges, 40 strict anchor entities and 22 anchor structures. Its specificity audit had 31 paired fold edges, 18 queries and six independent structure clusters. Median real-minus-wrong context was `+0.08035` (22 positive, nine negative); the exact one-sided structure-cluster sign-flip p-value was `0.140625`.

On the stricter adduct-aware axis, the audit had four paired policy-match fold edges, two queries and two structure clusters. Median real-minus-wrong context was `+0.29396` (three positive, one negative); exact p=`0.5`.

Both statuses are `REAL_CONTEXT_TREND_NOT_QUALIFIED`. The context route is real and phenotype-blind, but neither panel qualifies BioAware ranking or biological-response gain.

## Cross-adduct evaluations

### Policy-anchor consistency, not exact identity truth

The initial “hidden truth” wording was withdrawn after the frozen policy anchors disagreed with qualified author-library InChIKeys. The valid report is `contrast_atlas_ibd200_cross_adduct_policy_anchor_wse_v2_20261022`.

- Two ion families, four nonprotonated queries and three adduct classes.
- Both policy anchors had qualified author annotations; exact InChIKey agreement was 0/2.
- WSE unique policy-anchor Top-1: 1/4; anchor in the top tie: 3/4.
- Deterministic Top-1 was 3/4, but arbitrary candidate-ID tie breaking is not credited.
- Optimistic/conservative MRR: `0.7917 / 0.4583`.

Status: `DEVELOPMENT_CONSUMED_POLICY_ANCHOR_CONSISTENCY_ONLY`. This cannot support a chemical-identity accuracy claim.

Real-versus-wrong BioAware context on the same policy anchors produced seven of seven positive paired query-fold deltas, median `+0.29618`, and coverage-matched Top-1 of 2/2 versus 1/2. Only two ion-family clusters existed; exact p=`0.25`. Status: `REAL_CONTEXT_POLICY_ANCHOR_SIGNAL_NOT_QUALIFIED`.

### Structure-free rank stability

The direct, truth-free application endpoint is `contrast_atlas_ibd200_cross_adduct_rank_stability_wse_v1_20261022`.

| Quantity | WSE result |
|---|---:|
| Ion families | 6 |
| Nonprotonated queries | 11 |
| Score pairs available | 11/11 |
| Top-tie overlap | 8/11 |
| Mean top-tie Jaccard | 0.6061 |
| Unique Top-1 comparable / concordant | 6 / 4 |
| Rank correlations available | 5 |
| Median Spearman rank correlation | 0.8290 |

This measures within-ion-family candidate-rank stability relative to the method's own protonated member. It makes no structure-correctness, biological-response, or unknown-information-increment claim. B and the integrated algorithm should be compared on this exact axis without changing eligibility.

## Superseded or withdrawn outputs

1. The original 6,813-edge protonated application candidate universe is unsafe for strict biological identity use.
2. The former unresolved entity 1957 / norethandrolone-like response bridge is withdrawn: the query was annotated `[M+Na]+` with neutral mass about 280.24026, while the candidate had been treated as `[M+H]+` with exact mass about 302.22458.
3. The direction-only bile-acid-like entity 1413 is not a response discovery because it failed the prespecified full Plate 3 response gate.
4. Cross-adduct “hidden structure truth” was downgraded to policy-anchor consistency because the anchors conflicted with author annotations.
5. Neither candidate-context result qualifies BioAware.

## Systematic public-response collection

The frozen open collection currently contains 12 studies, 12 source families, 27 contrasts, 629 biological units, 91,295 response rows and 38,247 unique entities. Strict common-library support covers 752 entities (1.966%); 37,495 are unresolved.

The sparse-fragment baseline evaluated 648 fixed source-disjoint rows per world (1,944 total) against ten registered design relations. It supplied no signed truth, and neither the all-entity nor unresolved-only world showed a registered-relation Top-1 gain; the same-source unresolved HFD specificity control also failed. These are negative development findings.

Nine official DreaMS embedding shards were completed and committed before this snapshot: `IBD200_GROUP`, `MSV000088023`, `MSV000092887`, `MSV000097426`, `MSV000084794`, `MSV000090788`, `MSV000096564`, `MSV000086360`, and `MSV000090804`. The `MSV000084112` shard had no completion report at this snapshot and is not committed. Full 12-source embedding analysis is therefore incomplete and must not be reported as a completed result.

## Algorithm handoff contract

The biology side is ready to accept B and integrated-system score bundles on the frozen 509-edge axis. Each bundle must preserve `query_ptr`, `query_ids`, and `candidate_ids` exactly. No method receives broader chemistry eligibility or a different response endpoint.

Primary evaluators:

- `tasks/evaluate_ibd200_cross_adduct_rank_stability.py`
- `tasks/evaluate_ibd200_cross_adduct_hidden_truth.py` (despite the historical filename, current interpretation is policy-anchor consistency only)
- `tasks/evaluate_ibd200_algorithm_public_response_links.py`

Expected score specifications follow the existing interface, for example `B=<bundle>:B_score` and `integrated=<bundle>:integrated_score`. The joint deliverable is not “more annotations”; it is a traceable table from algorithm-specific added/removed candidate evidence to held-out response support, errors, abstentions and any independently qualified new response answer.

## Verification

The following current biology-application test set was run on 2026-10-22:

```text
tasks/test_audit_ibd200_candidate_context_specificity.py
tasks/test_audit_ibd200_candidate_adduct_compatibility.py
tasks/test_build_ibd200_plate3_algorithm_application_slice.py
tasks/test_build_ibd200_plate3_adduct_aware_slice.py
tasks/test_build_ibd200_candidate_context_evidence.py
tasks/test_evaluate_ibd200_cross_adduct_rank_stability.py
tasks/test_evaluate_ibd200_cross_adduct_hidden_truth.py
tasks/test_evaluate_ibd200_algorithm_public_response_links.py
```

Result: **16 passed in 44.29 seconds**.

## Commits covered by this output record

```text
dd0d7d1 Add fourth frozen open-response embedding shard
c2503f0 Build real IBD candidate-specific context evidence
6f09b24 Audit real versus wrong IBD candidate context
92c6db2 Add sixth frozen open-response embedding shard
896b8c5 Trace IBD algorithm choices into public responses
9e6919b Add seventh frozen open-response embedding shard
86b099d Enforce adduct-compatible IBD biology links
6fcac9d Rebuild IBD application on an adduct-aware candidate axis
12b8dfe Add real cross-adduct ion-family positive control
71b76fa Test real BioAware context on cross-adduct truth
29b1d9c Add ninth frozen open-response embedding shard
dec363e Downgrade cross-adduct labels to conflicting policy anchors
409d33b Measure structure-free cross-adduct rank stability
```

## Current claim boundary and unfinished work

- No unresolved IBD biological candidate currently passes the complete independent response gate.
- BioAware has directional evidence but is not qualified.
- Cross-adduct stability is not identity accuracy.
- Public-response and IBD panels are development/consumed; no external biological validation is claimed.
- B and integrated-system bundles have not yet been delivered on the frozen 509-edge axis, so their biological increment is not yet measured.
- The full 12-source official embedding run and MTBLS1866 feature extraction remain incomplete. Partial caches are not committed as results.

The next valid result is therefore fixed in advance: run WSE, B, the new spectral system, and the real-context integrated system on the same 509-edge axis; report evidence exchanges and their downstream held-out response consequences; then require any proposed unresolved discovery to pass independent identity and biological confirmation.
