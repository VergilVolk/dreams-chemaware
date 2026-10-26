# Unified ranker biology application result (2026-10-26)

## Verdict

The latest unified ranker is usable on the frozen MSV000095331 application
axis, and it produces materially different entity decisions. It does **not**
yet provide a positive context-robust biological increment over Noise V1.

The correct result code is:

`LOCAL_RANK_CHANGES_WITHOUT_CONTEXT_ROBUST_TOP1_OR_EVIDENCE_GAIN`

This is consumed development evidence, not external validation.

## Frozen application contract

- 38,628 directional entity queries per method;
- 22,046 fixed 10-ppm identity candidate pairs;
- 20 MSV000095331 response contrasts;
- 24 candidate response contrasts;
- 30 prespecified target-response contexts;
- five microbial backgrounds per mixture are related strata, not independent
  studies.

The selected MassSpecGym model was frozen before application responses were
opened. Application scoring used WSE, frozen P2b and all four encoder caches.
ChemAware V2 was explicitly unavailable because these cross-study candidates
have no structure objects; its six fields remained at the registered missing
zero. BioAware events were not used.

## Entity-decision coverage

| Method | Forward selected hypotheses | Forward mutual Top-1 |
|---|---:|---:|
| Noise V1 | 5,319 | 2,481 |
| FullMSG raw cosine | 5,319 | 2,493 |
| Unified ranker | 5,160 | 1,999 |

Relative to Noise, the unified ranker changed 334 forward Top-1 decisions,
gained 170 mutual edges and lost 652, for a net loss of 482 mutual identity
hypotheses. The unified result is therefore substantially more selective on
this application domain; this must not be confused with higher identity
accuracy because no complete identity truth exists here.

## Registered biological-response result

| Method | Raw-context target Top-1 | Median target rank | Mean target rank |
|---|---:|---:|---:|
| WSE | 2/30 | 10.5 | 11.13 |
| Noise V1 | 2/30 | 10.5 | 11.03 |
| Frozen P2b | 2/30 | 11.5 | 11.33 |
| FullMSG raw cosine | 5/30 | 11.0 | 10.90 |
| Unified ranker | 3/30 | 10.5 | 10.07 |

Against Noise, the unified ranker improved 15 target ranks, worsened 12 and
left three unchanged. However, the target response score itself increased in
only 11 contexts and decreased in 19; its mean change was `-0.000996`.
Matched-entity coverage decreased in all 30 contexts, by 137.2 pairs on
average.

Thus the lower mean rank cannot be interpreted as uniformly stronger target
response evidence. It combines some useful reordering with broad loss of
matched support.

## The sole new raw-context Top-1

The only Top-1 gained over Noise was:

- query: `drug_mix_1_group_3_medium_slow`;
- candidate study: `MSV000093571`;
- target: `drug_sertraline`;
- rank: `2 -> 1`;
- target response score: `0.012695 -> 0.010830`;
- matched entity pairs: `568 -> 469`.

The target became first because competing candidates were suppressed more,
not because its evidence increased. The change involved 211 identity edges:
104 favored and 107 opposed the target, with a net response-numerator change
of `-5.811`. Of these changes, 194 joined two unresolved entities, and none
was one of the prespecified published transformation-product queries.

This is a genuine ranking change but not a recovered biological-response
increment.

## Context-robust aggregation

Collapsing the five microbial backgrounds with equal weight gives six
mixture-by-candidate-study units.

| Method | Aggregate target Top-1 | Median target rank | Mean target rank |
|---|---:|---:|---:|
| Noise V1 | 2/6 | 4.0 | 4.83 |
| FullMSG raw cosine | 1/6 | 4.0 | 4.83 |
| Unified ranker | 1/6 | 4.5 | 4.50 |

The unified ranker improved two aggregate units, worsened one and left three
unchanged relative to Noise. Its average rank is slightly lower, but the
context-robust Top-1 count is lower and the median target rank is worse. With
six descriptive units, no significance or general-transfer claim is made.

## Scientific consequence

The latest delivery is useful in three bounded ways:

1. it proves the frozen unified model can be applied to real unknown-rich
   cross-study spectra without biological-label leakage;
2. it supplies a distinct, more selective set of entity hypotheses for
   follow-up;
3. it falsifies the stronger claim that this version already recovers more
   robust biological response information than Noise or the raw FullMSG
   representation.

The application should therefore retain the unified ranker as a comparison
arm, not promote it as the biology winner. The next algorithm version should
enter the same frozen candidate and response axis. No new application task or
data-specific threshold is warranted from this result.

## Reproducible artifacts

- Compact verdict builder:
  `tasks/build_msv000095331_unified_biology_verdict.py`
- Compact verdict:
  `data/validation/msv000095331_unified_biology_verdict_v1_20261026/`
- Unified scoring:
  `data/validation/msv000095331_unified_ranker_identity_local_20261026/`
- Frozen response run:
  `data/validation/msv000095331_entity_method_increment_unified_local_20261026/`
- Context aggregation:
  `data/validation/msv000095331_context_robust_unified_local_20261026/`
- Edge attribution:
  `data/validation/msv000095331_unified_response_gain_attribution_local_20261026/`

Two deployment blockers were fixed before this run: empty spectra are now
represented as valid empty peak matrices rather than crashing WSE/P2b, and
nonpositive/nonfinite precursor masses are retained as zero-candidate queries
rather than entering a zero-mass identity window.
