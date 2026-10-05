# BioAware B44 external failure root-cause audit

Date: 2026-09-13  
Status: post-hoc read-only audit of the consumed B44 panel; the sealed decision is unchanged.

## Decision

BioAware-Catalogue v1 is not externally confirmed and must not be deployed as a
general candidate reranker.  Its B42 gain was conditional on catalogue
ascertainment: truth candidates were much more likely than wrong candidates to
be present in strict KEGG or currency-filtered Rhea.  Independent MassBank B44
reversed that relation, so the same frozen action promoted well-catalogued wrong
isomers.

This is not an implementation-replay failure.  The downloaded B44 query and
candidate ledgers match the sealed SHA256 values, every query has exactly one
truth, candidate counts reconcile, the official checkpoint is unchanged, and
the eight corrections minus thirteen introductions exactly reproduce the five
lost Top-1 decisions.

## Frozen-gate transport

| Quantity | B42 development OOF | B44 MassBank |
|---|---:|---:|
| Truth candidate mapped to at least one catalogue | 89.65% | 19.37% |
| Wrong candidate mapped to at least one catalogue | 53.38% | 24.31% |
| Truth-minus-wrong mapping advantage | +36.27 pp | -4.94 pp |
| Intervention rate | 12.91% | 3.14% |
| Corrected / introduced | 65 / 5 | 8 / 13 |
| Fraction of baseline errors corrected | 22.26% | 4.65% |
| Harm among baseline-correct queries | 0.88% | 1.80% |
| Precision among non-neutral interventions | 92.86% | 38.10% |

The shift is stronger in the B44 negative-ion primary panel: 18.84% of truths
but 30.76% of wrong candidates are mapped.

## B44 mechanism decomposition

Every one of the 28 interventions promoted a candidate with a higher catalogue
membership count than the DreaMS baseline candidate.

- Seven of eight corrections are unmapped-baseline to mapped-truth promotions.
- Eleven of thirteen new errors are correct-unmapped-truth to mapped-wrong
  promotions.
- Fully mapped degree competition contributes one correction and two new errors.

The hidden condition is visible in coverage strata:

| B44 stratum | Queries | Corrected / introduced | Recall@1 delta | Formula-cluster 95% CI |
|---|---:|---:|---:|---:|
| Mixed coverage; truth mapped | 118 | 8 / 1 | +5.93 pp | [+1.56, +11.01] pp |
| Mixed coverage; truth unmapped | 122 | 0 / 11 | -9.02 pp | [-14.29, -4.29] pp |
| All candidates mapped | 55 | 0 / 1 | -1.82 pp | [-5.66, 0.00] pp |
| No candidates mapped | 598 | 0 / 0 | 0.00 pp | [0.00, 0.00] pp |

Thus the historical approximately six-point gain is reproducible when the truth
is catalogue-covered, but reverses when that unobservable condition fails.  It
is a conditional database prior, not a general reaction-network signal.

The failure is structurally identifiable.  For C9H12O the same proposed isomer
and the same model probability (0.655426) correct one query and damage another.
For C9H10O2 the same proposal/probability pair (0.729959) produces one
correction, one new error and one neutral move.  Candidate-static topology
cannot distinguish those spectra once it dominates a small spectral gap.

## Why threshold tuning cannot repair v1

The median proposal probability is 0.648 for corrections and 0.665 for new
errors.  A post-hoc diagnostic over the existing margin and probability family
found no non-empty stricter gate with positive unweighted or lambda=2 risk net;
at probability 0.80 the model stops intervening entirely.  B44 is consumed and
must not be used to select a replacement threshold.

## Scientific correction

Source-, identity- and formula-isolated B42 evaluation did not remove a shared
dataset-construction bias: confirmed biological standards and seed metabolites
are preferentially catalogued, while enumerated same-formula decoys are not.
Leave-source-out validation can preserve that bias across all six opened
sources.  B44 is valuable precisely because its independent formula and
identity distribution breaks the correlation.

B44 falsifies the portability of static catalogue membership/degree.  It does
not test or falsify sample-specific reaction evidence, co-observation,
co-abundance, ion-family evidence, or a context-conditioned representation.

## Next hard gate

1. Archive Catalogue v1 as a failed external candidate-static prior.  Do not
   refit or tune it on B44.
2. Catalogue membership becomes a missingness/coverage mask, never a positive
   candidate score on its own.
3. Before building a new external expert, test a coverage-neutral action on a
   development-only universe: candidate and baseline must share catalogue
   membership status, model fitting uses only similarly matched truth-negative
   pairs, and topology must beat formula-preserving topology permutations under
   nested leave-source-out validation.
4. A true BioAware successor must use query-specific sample context and must
   beat degree-preserving, sample-shuffled and within-formula null graphs.
5. Static database metadata must not be distilled into a shared spectrum-only
   encoder.  Any representation update using biological context must remain
   context-conditioned and must revert to DreaMS when context is absent.

## Reproducible audit entry point

`tasks/audit_bioaware_b44_external_failure.py` reads the sealed B44 result and
the frozen B42 ledgers, verifies their provenance, and writes a separate
post-hoc audit directory.  It does not refit a model or alter B44.
