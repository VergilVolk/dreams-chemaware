# Noise-final: real-positive-guided action matrix

## Why this stage exists

The frozen P/N action union covers 922 of 1,805 official errors, which is only
3.86 percentage points of supervision-space headroom.  The remaining 883
errors are dominated by positive-deficit mechanisms (795 queries).  Therefore
another negative-peak deletion sweep cannot supply the 272 additional unique
corrections required even for a nominal five-point upper bound.

This stage expands the **noise-finetuning action space** on the positive arm.
It is not P2b, not a reranker and not an embedding-after-the-fact expert.

## Frozen action matrix

Each query uses up to three highest-official-similarity real spectra from its
own positive molecule.  Only peaks already present in the query may change.
No synthetic peak is inserted.

1. `matched_intensity_transport`: interpolate the intensity of matched peaks
   toward the median intensity observed in the real positive references.
2. `prevalence_attenuation`: attenuate each query peak according to how often
   it is observed across the positive references.
3. `consensus_projection`: jointly apply positive-reference intensity and
   prevalence evidence.

Every family is evaluated at doses 0.25, 0.50, 0.75 and 1.00.  The exact same
family and dose are also applied toward the current hardest wrong molecule.
That wrong-direction intervention is a direction-matched specificity control,
not a training target.

## Evaluation contract

- Full 23,876-query P3-disjoint strict-10ppm candidate graph.
- Frozen official DreaMS encoder and frozen official candidate embeddings.
- Exact corrected and introduced counts for every fixed cell.
- Formula-cluster bootstrap confidence intervals.
- Near-candidate safety report.
- Positive-guided versus hardest-wrong-guided specificity interval.
- Fresh clean forward is reproduced first. Numerical rank-boundary mismatches
  are reported and excluded from action effects.
- P2b is forbidden.

A fixed cell may advance only if its formula-cluster Recall@1 interval is
positive, corrected > introduced, corrected - 2*introduced > 0, near Recall@1
does not decrease, and the positive-versus-wrong specificity interval is
positive.

The union across actions is outcome-aware and is reported only as an upper
bound on supervision coverage.  It must add at least 272 new errors beyond the
frozen P/N union to reach five-point headroom; 350 is the desired buffer before
capacity training.

## What follows if the matrix passes

Passing fixed cells are cross-fitted by formula and converted into real
positive-guided augmented views.  A single shared DreaMS encoder is then
fine-tuned on P (positive-deficit rescue), N (validated negative corrections)
and S (clean preservation).  Inference remains one clean spectrum to one new
embedding; no candidate graph, positive reference or downstream expert is
required at inference.
