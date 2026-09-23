# ChemAware pair-expanded native triplets

Date: 2026-09-22

## Decision

The 7,688-event stage-1 pool and the 7,357-event reference-aligned pool still
compress many valid spectrum-level constraints into one randomly sampled
molecule event.  The next experiment therefore expands unique active
positive-reference/negative-reference pairs while preserving the native DreaMS
dataset, `ContrastiveHead`, cosine hinge, Adam optimizer and the confirmed
`+1.8144 pp` initialization.

## Capacity bottleneck

On formula roles 0--1:

- 4,032 training queries and 2,518 formulas;
- 12,755 legal query-false-molecule boundaries in the complete candidate graph;
- at most 8,663 candidate boundaries when retaining three candidates per query;
- at most 10,716 candidate boundaries when retaining five candidates per query;
- 44,370 active positive-negative spectrum pairs among the first five
  candidates and their top two negative references;
- 11,388 active events after retaining at most two complementary positive
  references per negative reference, before a per-query cap.

The bottleneck is therefore not a lack of identity-valid spectrum pairs.  It is
the previous molecule-level representation: one event stored many positives
and negatives, while native DreaMS sampled only one of each when the event was
visited.  At short training horizons, many active pairs were never consumed.

## Pair-expanded construction

For each query:

1. retain up to five false candidate molecules;
2. reserve the two hardest candidates independently of chemistry;
3. reserve up to two distinct ChemAware candidates with nonzero expected
   native hinge;
4. fill any remaining candidate slots by checkpoint hardness;
5. inspect the two highest-scoring negative reference spectra per candidate;
6. for each active negative reference, emit:
   - a boundary positive: the highest-similarity positive that still violates
     the margin;
   - a hard positive: the lowest-similarity active same-identity reference;
7. retain at most 12 active events per query, filling candidate diversity
   before a second reference event from the same candidate;
8. if a query has no active pair, retain exactly one zero-loss safety sentinel.

Every event has a unique key:

`(query, positive reference, negative molecule, negative reference)`.

Thus the increase is not produced by copying a triplet.  Positive and negative
identity edges remain checked against the HDF5 source.

## Local full-graph audit

The following audit uses the complete local official-embedding cache.  It is a
curriculum audit, not a retrieval-performance result.  The server job must
recompute the bank from the frozen `+1.8144 pp` checkpoint.

| measure | stage-1 successful pool | pair-expanded pool | ratio |
|---|---:|---:|---:|
| total native events | 7,688 | 12,333 | 1.60x |
| active events | 3,046 | 10,188 | 3.34x |
| zero-activation events | 4,642 | 2,145 sentinels | 0.46x |
| mean activation probability | 0.1944 | 0.8261 | 4.25x |
| mean native hinge | 0.02720 | 0.12430 | 4.57x |
| queries | 4,032 | 4,032 | 1.00x |
| formulas | 2,518 | 2,518 | 1.00x |
| retained candidate boundaries | 7,688 | 6,134 | 0.80x |
| explicit chemical events | not activation-filtered | 2,322 | -- |

The lower candidate-boundary count is intentional: a candidate remains only
when it contributes an active spectrum pair, except for one sentinel per safe
query.  Effective gradient constraints, rather than molecule names, are the
quantity being expanded.

Correct and null arms have essentially identical total budgets
(`12,330--12,334` events locally).  On role 2 the two-chemical-slot design has
`74--82` correct-only chemical candidates per null comparison.  Chemical-slot
Jaccard is `0.664--0.692`; shared hard slots keep the majority of the
curriculum identical across causal controls.

## Risk controls

Explicit hard-positive selection can overweight atypical replicate spectra.
The implementation therefore does not select two arbitrary lowest-scoring
positives.  It combines one boundary positive with one hard positive, caps a
query at 12 active events, and uses role-2 checkpoint selection at
500-step intervals.  Role 3 is evaluated only after role 2 advances beyond the
stage-1 checkpoint under all retrieval and safety metrics.

Before training, the stage-1 checkpoint cache must show at least a 1.25x gain over
the old pool in all three quantities:

- active event count;
- mean activation probability;
- mean native hinge.

Failure stops before optimization.

### 2026-09-23 server-threshold correction

The first server construction produced 9,957 unique spectrum triplets, 7,346
active triplets, 5,389 candidate boundaries, 1,516 chemical spectrum events,
and complete 4,032-query/2,518-formula coverage.  Construction was valid, but
it was incorrectly rejected by absolute gates copied from the local official
cache (`6,000/12,000/800`).  Those counts are checkpoint-dependent and cannot
be transported to the stronger stage-1 geometry.

The repaired contract uses only conservative structural floors (4,500
candidate events, 8,000 spectrum events, 300 chemical candidates and 1,000
chemical spectrum events).  The scientific quality decision is made afterward
by comparing old and new pools in the exact same server checkpoint cache and
requiring a 1.25x gain in active events, activation probability and mean hinge.

## Optimization contract

Unchanged:

- learning rate `5e-6`;
- margin `0.1`;
- batch size `4`;
- weight decay `0`;
- full backbone trainable;
- native DreaMS preprocessing, model, loss and Adam optimizer.

Checkpoints are evaluated at 500, 1,000, 1,500, 2,000, 2,500 and 3,000 steps.
Three matched semantic-null arms are trained at the selected step count only if
the correct arm beats stage 1 on role 2.

## Execution

```bash
sbatch tasks/run_chemaware_pair_expanded_native.sbatch
```

The job requests exactly one GPU and contains no manual memory request.

## Claim boundary

Current evidence supports only a stronger, larger, identity-valid and
checkpoint-active triplet curriculum.  It does not yet establish a retrieval
gain beyond `+1.8144 pp`; that claim requires the server role-2 selection and
role-3 paired evaluation.
