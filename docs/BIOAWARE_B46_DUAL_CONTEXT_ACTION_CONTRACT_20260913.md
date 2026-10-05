# BioAware B46 dual-context action contract

## Question

After B44 and B45 rejected static catalogue coverage/degree as a portable
BioAware mechanism, can a candidate-specific conjunction of two observed data
layers produce a material corrective action?

The two layers are:

1. reaction-transform spectral concordance: the candidate--seed molecular mass
   change must explain the corresponding query--seed fragment transformation
   better than an opposite-sign transformation;
2. sample-matrix concordance: the candidate--seed abundance relationship must
   exceed matched non-neighbour controls.

B46 is an opened-development action screen, not an external test and not an
embedding result.

## Frozen action

- Inputs are the already frozen B2 rotation-level reaction-transform ledger,
  B3 rotation-level matched coabundance ledger, B2 candidate graph and B42
  independent catalogue signatures.
- No B44 row or result is read.
- Within each query and held-seed rotation, each evidence variable is converted
  to a candidate percentile rank. This avoids adding incomparable raw scales.
- Real dual evidence is the harmonic mean of reaction-transform-specificity
  rank and positive coabundance-excess rank, multiplied by joint availability.
- Candidate scores are means over all frozen seed rotations. At least two
  rotations must contain both evidence layers.
- A candidate can replace DreaMS Top-1 only when it has the exact same
  `(independent_member_count, independent_member_intersection)` signature and
  a strictly larger, unique dual-evidence score.
- Primary deployment guard is the historical, outcome-independent DreaMS
  Top1--Top2 margin `<= 0.05`.

## Strong controls

The identical action is replayed after replacing one layer at a time:

- wrong-sign spectral transformation in place of the chemically directed
  transformation;
- matched-random coabundance in place of real candidate--seed coabundance.

These controls use the same rows, coverage restriction, aggregation, baseline
margin, tie rule and evaluation denominator.

## Advancement rule

The real action advances only if all conditions hold:

- at least 800 queries and 150 baseline errors are evaluated;
- at least 100 errors have a coverage-neutral dual-evidence opportunity;
- Recall@1 gain is at least 3 percentage points;
- corrected identities are at least 25;
- corrected is greater than twice introduced;
- identity- and formula-cluster 95% CI lower bounds exceed zero;
- every biological source is nonnegative;
- real evidence beats both wrong-sign and random-coabundance actions with a
  positive formula-cluster CI lower bound.

Failure prevents context-adapter or shared-embedding training from these
targets.

## Claim boundary

The internal four-source contexts use held-seed rotations and are not
prospective unknown annotation. A passing B46 result would establish a
candidate action suitable for a new independent sample-context benchmark; it
would not by itself establish biological causality, external performance,
shared-embedding improvement or SOTA.
