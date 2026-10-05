# ChemAware V11 continuous-Adam online-packet result

## Verdict

Run `2345454` completed without a runtime or evaluation failure.  It is a
scientifically valid negative incremental result: the exact protected Phase-A
checkpoint was reproduced, online re-mining reduced optimization-graph errors,
but no online checkpoint safely improved frozen formula role 2.  Role 3 stayed
unopened and Phase A remains the qualified champion.

## Reproduction anchor

The in-run step-2,000 checkpoint exactly recovered the protected result:

- Recall@1 `0.9174683544`, or `+2.1266 pp` over official;
- 54 official errors corrected and 12 errors introduced;
- formula-cluster CI `[+1.2413,+3.0362] pp`;
- Recall@3 `0.9913924051`, MRR `0.9535324675`;
- checkpoint SHA-256
  `a8428329ca1d12bfe735f3f8b848ed020a8db18d0a654d4f62cdbeb006e61135`.

Thus this run tests the online continuation rather than a drifting Phase-A
reproduction.

## Role-2 results relative to Phase A

| step | Recall@1 | delta | corrected / introduced | risk utility | CI | decision |
|---:|---:|---:|---:|---:|---:|---|
| 2,250 | 0.917975 | +0.0506 pp | 8 / 7 | -6 | [-0.3158,+0.4467] pp | fail |
| 2,500 | 0.917975 | +0.0506 pp | 10 / 9 | -8 | [-0.4115,+0.5139] pp | fail |
| 2,750 | 0.916456 | -0.1013 pp | 8 / 10 | -12 | crosses zero | fail |
| 3,000 | 0.916962 | -0.0506 pp | 12 / 13 | -14 | crosses zero | fail |

Steps 2,250 and 2,500 gained one net top-1 query but each lost one Recall@3
query, had negative `corrected-2*introduced`, and had a formula-cluster CI that
crossed zero.  Later steps degraded Recall@1.  Retaining Phase A was correct.

## What worked

- one Adam instance continued from step 0 through 3,000;
- Adam state ranges exactly matched global steps at every re-mine;
- training current errors fell `151 -> 142 -> 135 -> 133`;
- current winner switches remained observable (`821 -> 785` versus frozen
  official candidates);
- all identity, query uniqueness, role isolation and replay gates passed.

This proves online current-winner mining is operational and optimizes its
training objective.  The failure is transfer, not gradient reachability or
implementation.

## Root cause: query balancing destroyed the successful role dose

Phase A's 5,956 events have a measured role distribution:

- safe: 3,605 / 5,956 = **60.53%**;
- official-error: 777 / 5,956 = **13.05%**;
- chemical-error: 550 / 5,956 = **9.23%**;
- native replay: 1,024 / 5,956 = **17.19%**.

V11 sampled 4,032 focused queries uniformly and allocated about 80% of steps
to that stream.  At step 2,000 only 151/4,032 queries were errors.  Across the
1,000 online steps this yields only about 119 error-query examples, versus
about 522 official-error examples under the successful Phase-A role dose.

Chemistry was diluted more severely.  Only 59 step-2,000 error packets carried
chemistry, 44 with a distinct chemical slot.  Because that slot was averaged
inside a width-three packet, V11 supplied only tens of effective chemical
triplets, versus about 369 chemical examples implied by the Phase-A 9.23%
event fraction.  Most optimizer exposure instead went to current-correct
safety packets, many already at zero loss.

Therefore the correct lesson is not that online re-mining failed.  It is that
global query-uniform sampling is the wrong interpretation of query balancing
when the scientifically important roles are rare.  Query equality must hold
*within role*, while the successful inter-role dose remains fixed.

## V12 decision

Keep the same official initialization, exact Phase A, single live Adam,
250-step re-mining, native identity labels and native single-negative triplet.
Replace V11's global query-uniform packet stream with four homogeneous streams
whose optimizer-step budget is derived directly from the frozen Phase-A event
counts:

- safe 60.53%;
- current winner error 13.05%;
- same-query active chemical 9.23%;
- official replay 17.19%.

Each stream samples queries uniformly internally.  This restores the proven
correction/chemistry dose without returning to identity broadcast or event
multiplicity, and it removes duplicate padding: every optimizer batch again
uses the mature DreaMS one-positive/one-negative triplet.
