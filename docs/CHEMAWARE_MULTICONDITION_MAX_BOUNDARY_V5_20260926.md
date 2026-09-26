# ChemAware multi-condition max-boundary native curriculum V5

Date: 2026-09-26

## Decision

The next performance experiment retains the frozen `+2.1266 pp` Phase-A
mechanism and expands independent retrieval boundaries, not arbitrary spectrum
pairs. It continues to use the unmodified DreaMS shared encoder,
`ContrastiveSpectraDataset`, `ContrastiveHead`, cosine triplet hinge, Adam,
learning rate `5e-6`, margin `0.1`, batch size `4`, and full-backbone update.

The only changes are:

1. add a bounded number of instrument/collision-energy-diverse query anchors;
2. admit an extra anchor only when it is wrong, near the retrieval boundary,
   or exposes a new hardest false identity;
3. retain exact singleton max-positive/max-negative reference pairs;
4. transfer a frozen ChemAware false identity to an extra condition only when
   that identity is present in the new query's candidate graph and its exact
   max-reference hinge is active;
5. use stratified identity-equal event sampling; and
6. evaluate every role-2/3 query in addition to the one-query-per-identity
   checkpoint-selection panels.

## Why Phase A improved and why it plateaued

On the 1,975-query formula-role-2 panel, the released stage-1 model and Phase A
both corrected 54 official top-1 errors. Phase A reduced introduced errors from
24 to 12, increasing net corrections from 30 to 42 and Recall@1 improvement
from `+1.5190 pp` to `+2.1266 pp`. Its main demonstrated advance was therefore
protection, not broader correction coverage.

The retrieval decision for candidate molecule `M` is

`s(q,M) = max_(r in R(M)) cosine(z_q, z_r)`.

Phase A freezes the positive and false references that define this maximum and
optimizes the native DreaMS hinge

`[0.1 - max_positive(q) + max_negative(q)]_+`.

This removes the reference-sampling mismatch in the successful stage-1 pool.
However, Phase A still uses one query anchor per training identity: 4,032
anchors from 33,812 eligible spectra. A molecule that is safe in one
instrument/collision-energy condition can remain wrong in another condition,
and its hardest false identity can switch. Iterative remine of the same anchors
did not improve Phase A, so stale negatives on the original anchors are not the
remaining primary bottleneck.

## Five-point error budget

On role 2, official DreaMS has 205 errors. A five-percentage-point improvement
requires 99 net top-1 corrections. Phase A supplies 42, leaving 57 additional
net corrections. If introduced errors remain at 12, 57 of the remaining 151
official errors must be corrected (`37.75%`).

The method therefore has two simultaneous requirements:

- expand distinct error-boundary coverage; and
- preserve the Phase-A introduced-error suppression.

Raw pair expansion is rejected. The earlier 9,957-event pair-expanded run
increased active hinges but failed the role-2 safety gate. More active pairs on
the same anchors are not equivalent to more independent error boundaries.

## Frozen construction

Formula roles 0 and 1 are the only training roles. For each of their 4,032
identities:

1. retain the exact frozen Phase-A base-anchor events;
2. preselect at most six additional query spectra, prioritizing distinct
   instrument/collision-energy bins and embedding diversity;
3. retain at most two extra anchors, prioritizing official errors, margin at or
   below `0.1`, distinct experimental condition, and a distinct hardest false
   identity;
4. use the exact best same-identity reference and top false reference;
5. include at most one transferred ChemAware false identity on an extra error,
   and only with a positive native hinge;
6. append unmodified official DreaMS replay at 20% of stored events.

The event sampler has three frozen strata:

- 25% error boundaries, identity-equal within the stratum;
- 55% safety boundaries, identity-equal within the stratum;
- 20% official DreaMS replay, identity-equal within the stratum.

Thus spectra-rich identities cannot dominate any stratum, while the new error
anchors retain enough optimizer exposure to target the missing corrections.

## Executed local full-graph construction audit

The real 83,619-query manifest and official embedding cache produced:

| Quantity | Phase-A base | V5 selected |
|---|---:|---:|
| training identities | 4,032 | 4,032 |
| query anchors | 4,032 | 6,602 |
| official-error anchors | 427 | 1,398 |
| distinct identity-to-hard-negative boundaries | 427 | 1,028 |
| condition-diverse extra anchors | -- | 2,456 |
| extra anchors with a new hardest identity | -- | 1,452 |
| focused events | 4,932 | 9,024 |
| official DreaMS replay | 1,024 | 2,256 |
| total native events | 5,956 | 11,280 |

The audit checked 113,703 positive and 108,780 negative identity edges. All
focused events are singleton max-reference boundaries, all sampling weights are
positive and normalized, and formula roles 2, 3 and 4 were untouched.

These are curriculum and coverage results, not retrieval-performance results.

## Selection and evaluation

Checkpoints are saved at 500, 1,000, 1,500, 2,000, 2,500 and 3,000 optimizer
steps. Formula role 2 selects against the checksum-verified protected Phase-A
checkpoint, not merely against official DreaMS. Advancement requires:

- strictly positive Recall@1 and MRR relative to Phase A;
- strictly positive formula-cluster Recall@1 confidence-interval lower bound;
- `corrected - 2 * introduced > 0` relative to Phase A;
- nonnegative Recall@3, micro-AUC and macro-AUC.

Only an advancing checkpoint reaches formula role 3. The secondary full-role
evaluation then reports all 16,706 role-2 and 16,903 role-3 queries in a single
embedding pass. Identity-equal metrics are primary; query-micro metrics measure
cross-condition robustness. Formula role 4 remains inaccessible.

No checkpoint is deleted automatically, including negative runs.

If a role-2-advancing checkpoint reaches role 3, the checkpoint, selection
ledger, triplet report, role-2 and role-3 evaluations, and full-role evaluation
are atomically hard-linked or copied into `protected_artifact/` with both
triplet pools, the training report, an exact source-code snapshot, a manifest,
and recursive `SHA256SUMS`. A role-3 failure is still preserved, but its
manifest is explicitly labelled role-2-only rather than a confirmed release
candidate.

## Execution

The only server command is:

```bash
sbatch tasks/run_chemaware_multicondition_max_boundary.sbatch
```

The job requests exactly one GPU and no manual memory. It automatically locates
and verifies the newest protected Phase-A artifact. If that artifact does not
exist yet, it stops before training and instructs the operator to finish
`tasks/run_chemaware_phasea_2pp_protect.sbatch` first.

## Claim boundary

Before the GPU result, the allowed claim is limited to:

> A full-graph construction audit expanded condition-diverse, exact
> max-reference error coverage while preserving the successful Phase-A base
> curriculum, identity validity, stratified identity-equal sampling, and
> untouched role-2/3/4 evaluation formulas.

No retrieval gain beyond `+2.1266 pp`, role-3 confirmation, five-point gain or
chemical causal attribution is claimed before the frozen evaluations complete.
