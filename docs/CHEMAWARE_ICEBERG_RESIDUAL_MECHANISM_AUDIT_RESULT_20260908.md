# ChemAware ICEBERG candidate-residual mechanism audit result

Date: 2026-09-08

## Scope

This document closes only the old ICEBERG candidate-score residual Phase A at
`data/validation/chemaware_iceberg_residual_shared_phase_a/run_2332765`.
The audit loaded four already-trained checkpoints and re-encoded 32,779 spectra
per checkpoint.  It performed no retraining and updated no weights.

It is not an audit of the frozen qualified observed-peak action
`conflict_attenuate / strength=0.75 / top_k=3`.  Consequently, the result below
cannot erase the retained E1 action evidence and cannot decide the new
qualified-action delta-transfer experiment.

## Measured mechanism

| arm | correct-target Huber | correct-target cosine | active corrected | observed boundary change | median transfer ratio | target reached | safety introduced |
|---|---:|---:|---:|---:|---:|---:|---:|
| clean duplicate | 0.049069 | 0.718541 | 11 / 272 | +0.038984 | 0.370762 | 23.529% | 2 / 1,024 |
| correct residual | 0.047424 | 0.796255 | 10 / 272 | +0.037831 | 0.363333 | 23.162% | 2 / 1,024 |
| structure control | 0.049761 | 0.675743 | 11 / 272 | +0.038325 | 0.367493 | 23.529% | 1 / 1,024 |
| peak control | 0.049198 | 0.716977 | 10 / 272 | +0.038428 | 0.374559 | 23.529% | 1 / 1,024 |

The correct arm has a small continuous target-fit signature: its target Huber
is about 3.35% below clean continuation and its mean target cosine is the
largest of the four arms.  That signature does not reach the decision layer.
The correct arm corrects fewer active queries than clean or structure control,
has a smaller mean observed boundary change than clean, reaches no more of the
target, and introduces no fewer safety errors than clean.

All optimizer steps in all four arms were gradient clipped.  The old global
gradient-ratio plus clip implementation therefore changed the requested
chemical dose before AdamW, but clipping alone is not the causal diagnosis:
even the resulting direction-level advantage did not exceed ordinary
continuation at the held decision boundary.

The reported mean transfer ratios between 322 and 390 are not scientifically
interpretable because some target boundary changes are arbitrarily close to
zero.  The robust quantity is the median, approximately 0.36--0.37 in every
arm; it shows no correct-arm advantage.

## Decision

The old candidate-residual route is closed.  It must not be rerun, tuned by
learning rate, or expanded to additional folds.  The valid retained statement
is only that the trained model weakly aligns with the residual target in a
continuous representation; there is no causal clean-embedding improvement
over matched controls.

The next experiment uses a different source and a different target: the frozen
official DreaMS candidate-score displacement caused by the already-qualified
top-3 observed-peak action.  Its correct arm must beat an identical-schedule
clean duplicate before candidate-swapped and peak-permuted arms are allowed to
consume GPU time.  Until that matched comparison passes, no ChemAware
shared-embedding increment is claimed.
