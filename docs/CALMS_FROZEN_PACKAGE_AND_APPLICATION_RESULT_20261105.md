# CALMS frozen package and first fixed-axis application result

Date: 2026-11-05

## Frozen release

The public method is **CALMS**: **Context-Aware Learning for Metabolite
Annotation from Mass Spectra**.  The frozen public implementation is CALMS
v1.0 (internal implementation Full23 v1.3.2):

```text
candidate-evidence parent + 0.9 * frozen embedding correction
```

The executable package is `deploy/calms_v1`.  It contains both selected
checkpoints and the exact runtime rather than reimplementing their mathematics.
The documentation-complete portable archive is
`deliverables/CALMS_v1.0_20261105.zip` (2,386,163 bytes; SHA-256
`7305d38b74dad4edf74c1313e5a3989c72df1cfffd030781336a11dce2be6deb`).
It retains a synthetic end-to-end CLI example in addition to the frozen
scorer. The CPU golden replay, real CLI smoke, missing-evidence fallback and
batch-invariance tests pass. This documentation-only refresh does not change
either checkpoint or the scoring equation.

This release scores a pre-aligned candidate/evidence axis.  It is not presented
as an undocumented raw-MGF-to-every-feature pipeline.  It is a closed-library
annotation engine without calibrated open-set rejection.  BioAware remains a
separate conditional extension that requires real sample-event context.

## Application protocol

The frozen algorithm was applied, without response access during spectral
scoring, to the existing MSV000095331 plus three ChemProp-study identity axis.

| Quantity | Denominator |
|---|---:|
| Directional entity queries | 38,628 |
| Queries with no mass-compatible candidate | 27,815 |
| Singleton queries | 6,477 |
| Multi-candidate queries | 4,336 |
| Candidate pairs | 22,046 |
| CALMS-scoreable candidate pairs | 16,916 |
| Queries receiving the embedding correction | 3,324 |

The axis is same-observed-precursor, 10-ppm identity hypotheses.  It is
development/consumed application evidence, not external confirmation and not
identity proof.  ChemAware V2 is unavailable because this axis has no valid
candidate-structure evidence; BioAware was not fabricated.

## All-method application comparison

The biological endpoint uses independently estimated response effects after
the spectral decisions are frozen; raw spectral score scale never enters the
response score.  The table below reports the 30 registered positive-control
contexts.

| Method | Top-1 | Top-3 | Top-5 | Top-10 | Mean rank | Median rank | Forward selected | Mutual top-1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **CALMS** | 2 | **7** | **10** | 15 | **10.30** | 10.5 | 5,288 | 2,099 |
| CALMS parent | 2 | 4 | 8 | 15 | 10.73 | 10.5 | 5,288 | 2,088 |
| Noise-ChemAware FullMSG encoder | **5** | **9** | 9 | 14 | 10.90 | 11.0 | 5,319 | 2,493 |
| ChemAware Phase-A | 4 | 8 | 8 | **16** | 10.97 | 10.0 | 5,319 | **2,499** |
| Noise-MSG reranker | 3 | 4 | **11** | 14 | 10.40 | 11.5 | 5,288 | 2,098 |
| Noise V1 | 2 | 4 | 7 | 15 | 11.03 | 10.5 | 5,319 | 2,481 |
| WSE | 2 | 6 | 7 | 15 | 11.13 | 10.5 | **5,491** | 2,402 |
| Official DreaMS | 1 | 5 | 7 | 15 | 11.53 | 10.5 | 5,319 | 2,469 |
| P2b-on-Noise | 1 | 6 | 7 | 14 | 11.53 | 12.0 | 5,288 | 2,097 |
| Simple query-wise mean | 1 | 5 | 8 | 13 | 11.27 | 11.5 | 5,288 | 2,095 |
| Previous unified scorer | 2 | 4 | 8 | 15 | 10.73 | 10.5 | 5,288 | 2,088 |

CALMS has the best mean rank among the 11 methods and improves Top-3 and Top-5
over WSE, Noise V1 and the previous unified scorer.  It is not the Top-1 or
coverage winner.  The result therefore supports improved rank recovery under
selective evidence availability, not a claim that every application identity
is more accurate.

The 30 contexts are not 30 independent experiments: they are three drug
mixtures by five related microbial strata by two candidate studies, i.e. at
most six drug-by-study units.  Both CALMS Top-1 contexts are related sertraline
contexts in the same candidate study and must not be counted as independent
discoveries.

## Coverage and embedding-correction controls

To separate scoring from abstention, WSE was restricted to the exact 16,916
candidate rows on which CALMS is available.  The frozen scalar parent was also
replayed on that same axis.

| Same-coverage method | Top-1 | Top-3 | Top-5 | Top-10 | Mean rank | Median rank |
|---|---:|---:|---:|---:|---:|---:|
| **CALMS** | 2 | 7 | 10 | 15 | **10.30** | **10.5** |
| CALMS parent | 2 | 4 | 8 | 15 | 10.73 | 10.5 |
| WSE at CALMS coverage | 2 | 8 | 8 | 12 | 11.30 | 12.5 |

Against coverage-matched WSE, CALMS improves the rank in 17/30 contexts, ties
in 8 and worsens in 5 (mean rank delta -1.00).  Against its own frozen parent,
the embedding correction improves 11/30, ties 14 and worsens 5 (mean rank delta
-0.43).  Thus the mean-rank and Top-5 result is not explained only by lower
coverage, and the embedding correction contributes application-level rank
changes.  Because contexts are correlated and lack complete identity truth,
these are descriptive paired results, not confirmatory significance tests.

## Error-exchange boundary

Relative to unrestricted WSE over all 38,628 directions, CALMS retains 6,991
decisions, changes 431, gains 1,161, loses 1,760 and leaves 28,285 unresolved
under both methods.  It gains 806 mutual top-1 directions and loses 1,412.
`gained` means a computational identity hypothesis became available; it does
not mean a newly proven correct identity.

## Reproducible artifacts

- Frozen application scores:
  `data/validation/calms_v1_msv000095331_fixed_axis_20261103`
- All-method result:
  `data/validation/calms_v1_msv000095331_all_methods_increment_20261104`
- Coverage/parent controls:
  `data/validation/calms_v1_msv000095331_coverage_controls_20261105`
- Score-free representation adapter:
  `tasks/build_calms_single_reference_representation.py`
- CALMS score export:
  `tasks/export_calms_application_scores.py`
- Coverage/parent control export:
  `tasks/export_calms_application_controls.py`

## External-confirmation status

No new external performance claim is made here.  MultiMS2 is prepared as the
highest-value cross-condition benchmark, but its contamination exclusion must
remain blocked until the exact historical positive and negative MoNA source
files are restored or a new exclusion registry is explicitly defined and
rebuilt.  A current MoNA snapshot cannot silently replace those historical
files.
