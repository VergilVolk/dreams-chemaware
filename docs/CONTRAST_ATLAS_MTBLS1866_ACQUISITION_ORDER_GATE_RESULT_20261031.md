# MTBLS1866 acquisition-order qualification result

Date: 2026-10-31

Evidence role: **development / consumed source qualification**

## Decision

`MTBLS1866` does not qualify as an independent positive biological-validation
source for the current COVID-plasma Contrast Atlas analysis.  The 109-file
positive-DDA universe remains useful as a phenotype-blind technical asset, but
the COVID-versus-healthy response and the registered design-relevance link to
`MSV000092887` are not activated.

The phenotype-blind measurement object was not completed, the response
estimator was not run, and no cross-source retrieval result was produced.  The
existing resumable cache remains 24/109 samples from the earlier processing
wave; this gate did not create a frozen measurement output.

This is a source-qualification failure, not a negative COVID biology result.

## Frozen denominator and observed acquisition structure

The official primary universe contains 109 unique biological files: 83
non-critical COVID and 26 healthy plasma samples.  Acquisition timestamps were
recovered from 109/109 mzML headers.

| Quantity | Result |
|---|---:|
| COVID-after-healthy acquisition-rank AUC | 0.301668 |
| Reverse orientation AUC | 0.698332 |
| Mann-Whitney two-sided p | 0.002370 |
| Observed adjacent group transitions | 14 |
| Permutation-null transition median | 40 |
| Null 2.5%-97.5% interval | 32-46 |
| 100,000-permutation p for transitions <=14 | 0.000010 |
| Acquisition days | 10 |
| Days containing both groups | 3 |

Only three dates contain both groups.  On 2020-04-20 the denominator is 21
COVID versus 3 healthy, on 2020-04-21 it is 15 versus 1, and on 2020-05-05 it
is 9 versus 9.  The latter two mixed days are themselves strongly ordered
rather than interleaved.  Seven dates contain only one group.

Consequently, a day fixed effect would discard most acquisition days and still
leave weak or order-confounded within-day support.  A continuous injection-order
covariate cannot identify nonlinear drift without interleaved QC support.  A
post-hoc restricted or nearest-neighbor analysis would change the estimand and
does not rescue this source as positive truth.

## Contract and sequence boundary

The frozen contract already excluded a source from positive biological truth
when near-separation lacked a defensible batch adjustment or interleaved QC.
The normal contract sequence would freeze the full phenotype-blind measurement
object before opening labels.  Here the design ledger was opened early only to
terminate an ineligible, hours-long processing wave.  No feature parameter was
changed, and no abundance effect, detection effect, identity result, registered
relation, or retrieval score was calculated.

The partially completed per-sample feature caches may be reused for a future
technical stress test.  They cannot support a COVID response claim.

## Claim boundary

Allowed:

> MTBLS1866 failed the prespecified technical-confounding qualification because
> disease group was strongly structured by acquisition date and order.  It was
> therefore excluded from independent COVID biological validation before any
> response was estimated.

Not allowed:

- MTBLS1866 disproves a COVID metabolic response;
- MTBLS1866 validates or invalidates the Atlas retrieval method;
- unresolved MS/MS information or the unified algorithm was tested here;
- a day-restricted post-hoc contrast is an independent replication.

Machine result:
`data/validation/contrast_atlas_mtbls1866_acquisition_order_gate_v1_20261031/`.
