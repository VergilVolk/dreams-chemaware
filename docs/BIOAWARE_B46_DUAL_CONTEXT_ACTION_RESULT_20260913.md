# BioAware B46 dual-context action result

## Decision

The fixed coverage-neutral conjunction of reaction-transform spectral evidence
and matched coabundance evidence failed. It must not be used as a reranker,
context-adapter target, or shared-embedding supervision signal.

## Reproduction correction

The first local execution inferred baseline correctness from candidate identity
equality. One frozen DreaMS query had a tied Top-1 whose deterministic
representative identity equalled the truth; the official protocol counts that
tie as wrong. The implementation was corrected so a no-op replays the stored
baseline outcome and only an actual unique promotion is evaluated by identity.

The corrected B46 baseline is `0.7924263674614306`, exactly matching the frozen
B2 report across all 1,426 queries. The correction changed only the baseline
ledger count, not the action delta, corrections or harms.

## Corrected result

| Arm | Recall@1 delta | Corrected / introduced | Risk net (lambda=2) |
|---|---:|---:|---:|
| Real transform + real coabundance | **-0.7714 pp** | **10 / 21** | -32 |
| Wrong-sign transform + real coabundance | -0.5610 pp | 13 / 21 | -29 |
| Real transform + random coabundance | 0.0000 pp | 17 / 17 | -17 |

For the real dual action:

- formula-cluster 95% CI: `[-1.5526, -0.0654] pp`;
- identity-cluster 95% CI: `[-1.7012, +0.1426] pp`;
- 49/1,426 queries were changed;
- only 7 distinct truth identities were corrected;
- BV2cell, mouse brain and mouse liver were negative; only NIST plasma had a
  small positive point estimate.

Real dual evidence minus wrong-sign control was `-0.2104 pp` with a formula CI
crossing zero. Real dual evidence minus random-coabundance control was
`-0.7714 pp`, with formula and identity CI upper bounds below zero.

The experiment had adequate gross opportunity: 144 coverage-neutral baseline
errors were present. Failure is therefore not explained by having fewer than
the pre-registered 100 opportunities.

## Scientific interpretation

1. Two individually plausible but weak aggregate evidence layers do not become
   candidate-specific merely by taking their conjunction.
2. The B2 and B3 ledgers align by query, candidate and held-seed rotation, but
   each value is still an aggregate over potentially different seed neighbours
   and reaction records. B46 therefore does not test an exact
   candidate--seed--reaction event.
3. Positive coabundance is not universally expected for a biochemical
   substrate/product pair, and the existing matched-random excess does not
   recover reaction direction or enzyme state. Treating positive excess as a
   universal vote creates more harms than corrections.
4. The four internal biological sources use held-identity seed rotations. They
   are useful mechanism screens but are not prospective unknown sample
   annotation.

## Frozen boundary and next step

Do not tune the B46 margin, minimum rotations, rank aggregator or neural model
on this outcome. Do not distil B46 into a clean-spectrum encoder.

A genuine BioAware successor now requires a benchmark where:

- seed metabolites are measured independently of query truth;
- MS1 features, sample-level abundance vectors and MS/MS candidates are linked;
- reaction direction or reaction class is explicit when used;
- truth identities are provided independently and withheld from context
  construction;
- degree/coverage, wrong-edge, seed-permutation and non-neighbour controls are
  all available.

The immediate work item is therefore an external sample-context benchmark
readiness and acquisition stage, not another model fit on the current rotation
ledgers.

## Artifacts

- `docs/BIOAWARE_B46_DUAL_CONTEXT_ACTION_CONTRACT_20260913.md`
- `tasks/audit_bioaware_b46_dual_context_action.py`
- `data/validation/bioaware_b46_dual_context_action_local_20260913_v2/report.json`
- the three arm-specific transition ledgers in the same output directory.

## Claim limit

B46 is an opened-development held-seed-rotation action screen. It is not an
external performance result, prospective unknown annotation, a biological
causality test, an embedding result, or SOTA evidence.
