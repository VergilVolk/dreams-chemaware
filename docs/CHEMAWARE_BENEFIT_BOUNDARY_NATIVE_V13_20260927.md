# ChemAware V13: benefit-proven native triplet construction

> **Closed negative incremental result (run 2345648).**  The constructor and
> all preservation contracts passed, but no role-2 checkpoint safely exceeded
> the protected Phase-A embedding.  Retain Phase A and do not resubmit V13.
> See `CHEMAWARE_BENEFIT_BOUNDARY_V13_RESULT_20260927.md`.

## Frozen scope

V13 changes only the ChemAware triplet constructor.  It does not modify the
DreaMS dataset, uniform shuffled DataLoader, native one-positive/one-negative
draw, shared encoder, projection head, cosine triplet-margin loss, Adam, or the
protected official replay events.

The default initialization is the protected Phase-A shared embedding with
`+2.1266 pp` role-2 Recall@1.  V13 is therefore an incremental fine-tune, not a
mandatory restart from the original checkpoint.

## Corrected triplet semantics

The prior action-hard constructor selected false candidates that ranked highly
under a chemical action and treated them as negatives.  That identifies a hard
candidate, but it does not mean the action chemically rejects that candidate.

V13 instead uses the strongest already-established action event.  On a frozen
training query where official retrieval is wrong, the correct-arm action must
promote the known true candidate to rank one.  It must also beat each of the
three matched semantic nulls on at least two directionally oriented metrics.
This event proves the desired candidate ordering.  The ordinary identity-valid
DreaMS triplet is then built against the actual false winner under the chosen
initialization geometry:

`(query spectrum, true-candidate reference, current false-winner reference)`.

Chemistry decides which decision boundary is qualified.  It never supplies a
continuous target to the encoder and never changes the triplet identity label.

## Geometry audit and corrected replacement contract

The first local capacity audit used the official embedding cache.  The formal
job correctly rebuilt geometry from the protected Phase-A checkpoint and
showed that the original audit was not a valid count forecast: Phase A had
already moved many reference-level hinges.  This is a geometry change, not a
failure of the benefit evidence.

| quantity | result |
|---|---:|
| benefit-proven queries | 347 |
| independent formulas | 230 |
| official-geometry active triplets, cap 4/query | 1,189 |
| Phase-A-geometry active triplets, cap 4/query | 882 |
| Phase-A active benefit queries / formulas | 259 / 163 |
| Phase-A active reference combinations before cap | 13,128 |
| preserved Phase-A safety events | 3,605 |
| preserved official DreaMS replay events | 1,024 |
| old Phase-A error/chemical events replaced | 1,327 |
| required replacement target | exactly 1,327 |

The corrected constructor no longer removes 1,327 Phase-A error/chemical
events and replaces them with only 882 events, which unintentionally changes
pool size and sampling dose.  It now traverses active pairs in balanced query
layers, prioritizes current Phase-A errors within each layer, and stops only
after exactly 1,327 unique active events have been installed.  The per-query
cap is not guessed in advance: the constructor chooses the smallest cap that
can restore the budget, subject to a hard ceiling of 16.  Thus the final pool
cardinality, 3,605 Phase-A safety events, and 1,024 official replay events must
all remain unchanged.

Formula coverage is audited at the two distinct stages that actually exist:
at least 180 formulas must carry a counterfactually benefit-proven action, and
at least 160 must expose an active Phase-A native-loss boundary.  The former
cannot be replaced by the latter: 230 formulas have qualifying evidence, while
only 163 retain a positive reference-level hinge in Phase-A geometry.

## Formal execution

```bash
sbatch tasks/run_chemaware_benefit_boundary_native.sbatch
```

The job requests exactly one GPU and no manual memory.  It encodes the
protected Phase-A checkpoint, reconstructs the triplets in that geometry,
then runs the unchanged native DreaMS fine-tuner for at most 1,000 steps with
250-step checkpoints.  Frozen role 2 compares every checkpoint directly with
Phase A.  Role 3 remains unopened unless a checkpoint safely exceeds Phase A.
