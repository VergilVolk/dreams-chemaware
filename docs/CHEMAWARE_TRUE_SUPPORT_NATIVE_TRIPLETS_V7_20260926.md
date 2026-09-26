# ChemAware true-support native Triplets V7

Date: 2026-09-26

## Decision

V7 changes only Triplet construction. It retains the protected Phase-A
shared embedding, the DreaMS `ContrastiveSpectraDataset`, `ContrastiveHead`,
cosine triplet hinge, Adam optimizer, full-backbone update, margin `0.1`, and
the standard spectrum-only deployment interface. It adds no reranker, teacher
distillation, auxiliary loss, adapter, or candidate-side inference input.

The V6 continuation failed because its scientific unit was still a generic
current retrieval error. The 929 added main events were chosen by the closest
false molecule under Phase A; the 346 action-bank events only transferred
false identities selected by an older weak rule. That older rule called a
candidate “specific” when any one of four features exceeded the median of the
three nulls. It did not require the true candidate to receive the chemical
support, did not require superiority to every null, and replayed active
Phase-A triplets after resetting Adam. Consequently the first update direction
already produced more churn than net correction.

## Direct true-candidate criterion

For a labelled training query, let `t` be its true candidate, `f_c,m(t)` a
candidate feature under correct rule content, and `f_k,m(t)` the same feature
under matched null `k`. Define

`D_m(t) = f_c,m(t) - max_k f_k,m(t)`.

The broad tier requires `D_m(t) > 0` for at least two of these four rule
features:

- candidate rule maximum;
- candidate rule top-two mean;
- candidate-minus-baseline rule maximum; and
- candidate-minus-baseline rule top-two mean.

The strict tier additionally requires `D_m(t) > 0` for all three action-grid
support features and both action-advantage features. Thus correct chemistry
must support the known true candidate more strongly than every one of three
content-permuted controls. A high-scoring false candidate is no longer
mistaken for a corrective chemical action.

Only formula roles 0 and 1 are inspected for construction. On the frozen
training evidence this gives:

| criterion | queries | identities | formulas |
|---|---:|---:|---:|
| broad direct true support | 347 | 347 | 230 |
| strict direct true support | 118 | 118 | 103 |

For scale only, the identical predeclared criterion has 161 eligible official
errors on role 2 and 153 on role 3. These are upper-bound coverage counts, not
trained-model results and not tuning inputs.

## Native corrective and protection Triplets

Every eligible query is rescored under the checksum-verified Phase-A
embedding. If it remains wrong, V7 emits only

`(query, current max true reference, current max false reference)`.

Up to the two currently active references of that same hardest false molecule
are retained. No stale official negative and no chemically unsupported extra
condition is inserted.

Before retaining a correction, V7 must find a Phase-A-current correct guard
for the true identity or the false counterparty identity. The preferred guard
has a gap just above `0.1`, hence zero native hinge at construction and becomes
active only after adverse drift. If no such guard exists, the closest positive
near-boundary guard is used. If neither endpoint has a correct guard, the
correction is rejected.

The frozen raw sampling masses are:

| stratum | mass |
|---|---:|
| strict true-support correction | 45% |
| broad true-support correction | 20% |
| already-correct chemistry-supported preservation | 20% |
| paired endpoint sentinel | 15% |

Mass is identity-equal inside each present stratum. An absent stratum receives
exactly zero and the remaining predeclared masses are renormalized. Unlike V6,
there is no Phase-A base-pool replay that continues pushing an already selected
checkpoint after optimizer reset.

## Executed local construction audit

The full role-0/1 graph was executed against the official embedding cache as
an engineering stand-in because the protected Phase-A weights are held on the
server. This does not establish retrieval performance. It produced:

| quantity | count |
|---|---:|
| paired correction queries | 280 |
| strict correction events | 171 |
| broad correction events | 351 |
| unique endpoint sentinels | 368 |
| inactive zero-loss sentinels | 245 |
| active near-boundary sentinels | 123 |
| queries rejected without a guard | 67 |
| total singleton Triplets | 890 |

All 890 positive and 890 negative identity edges passed. Roles 2--4 were not
opened. Two independent constructions were array-identical with training-pool
hash

`2674d1ee12b2b94350691a1dc04d3e5e870bfe1509aa1648e047a4a1207829f2`.

Formal server counts will differ because current errors and correct guards are
computed from Phase A rather than the engineering stand-in.

## Optimization and stopping

Training starts from the protected `+2.1266 pp` Phase-A checkpoint. It uses
learning rate `1e-6`, batch size 4, and at most 400 optimizer steps. Eight
checkpoints at steps 50 through 400 are evaluated on frozen formula role 2
against Phase A. Advancement requires all of:

- positive Recall@1 and MRR;
- strictly positive formula-cluster Recall@1 confidence-interval lower bound;
- `corrected - 2 * introduced > 0`;
- nonnegative Recall@3, micro-AUC, and macro-AUC.

Only an advancing checkpoint reaches independent role 3. Role 4 remains
inaccessible. A role-2 advance is protected even if role 3 subsequently
fails, but it is labelled role-2-only.

## Execution

The sole server command is:

```bash
sbatch tasks/run_chemaware_true_support_native.sbatch
```

The job requests exactly one GPU and no manual memory. It verifies the
protected Phase-A checksums, reuses a checksum-matched V6 Phase-A cache when
available, runs CPU contracts before training, and never deletes a checkpoint.

## Claim boundary

The established result remains the protected `+2.1266 pp` role-2 Phase-A
embedding and the historical `+1.8144 pp` role-3-confirmed embedding. V7 is a
completed, deterministic Triplet method and local construction audit. It is
not a demonstrated performance gain until the frozen server evaluation
finishes.
