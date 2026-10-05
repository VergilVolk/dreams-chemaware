# ChemAware continuous-Adam online query-packet fine-tuning

> **Closed negative incremental result.** Run `2345454` completed and retained
> Phase A.  The historical global-query-uniform scheduler is superseded by the
> Phase-A-role-budget V12 entry.  See
> `docs/CHEMAWARE_ONLINE_PACKET_V11_RESULT_20260927.md`.

## Decision

The protected Phase-A role-2 result (`+2.1266 pp`, 54 corrected / 12
introduced) remains the comparator.  The next experiment does not start a
residual optimizer from that checkpoint.  It starts once from official DreaMS,
executes the exact Phase-A curriculum for 2,000 steps, and then retains the
same in-memory Adam object while candidate boundaries are re-mined.

This directly removes the unresolved fresh-Adam confound shared by V6--V10.
It also changes the unit of optimization from a uniformly sampled event to one
equally represented training query.

## Training object

For query `q`, the current shared embedding defines a molecule score

`S(q,c) = max_(r in R(c)) cos(z_q, z_r)`.

At steps 2,000, 2,250, 2,500 and 2,750, every formula-role-0/1 training query
and all of its candidate references are re-encoded.  Each query contributes
one width-three packet:

1. the current highest-scoring false molecule and its max reference;
2. one active semi-hard molecule closest to the middle of the native margin
   band, when available;
3. one active ChemAware candidate observed for the same query and candidate,
   prioritizing counterfactually specific evidence, when available.

Missing slots repeat the current winner.  Repetition leaves the mean native
triplet loss unchanged.  A currently correct query receives only its closest
false boundary as a safety packet.  Chemistry is never broadcast to another
query or identity.

The model still evaluates

`mean_n [0.1 + cos(z_q,z_n) - cos(z_q,z_p)]_+`

through the repository's `ContrastiveHead.step`.  There is no teacher target,
reranker, custom encoder, custom loss or custom optimizer.  Focused packets and
the 1,024 official DreaMS replay events use separate homogeneous batches.
Replay retains native random one-positive/one-negative selection, while the
number of replay batches preserves the frozen replay fraction.

## Optimizer continuity contract

- Phase A: native Lightning trainer, Adam, LR `5e-6`, batch size 4, 2,000
  steps from official DreaMS.
- Online phase: `optimizer = trainer.optimizers[0]`; no checkpoint load and no
  second optimizer construction.
- Adam first and second moments must report step range `(2000,2000)` at the
  handoff and match the global step after every online interval.
- Only the learning rate changes to `2e-6` for the 1,000-step online phase.
- Saved checkpoints are weights-only; optimizer continuity is asserted during
  the live process rather than reconstructed from a residual checkpoint.

## Local full-graph preflight

The deterministic official-geometry preflight completed on all 4,032 training
queries:

- 5,056 events: one packet per query plus exactly 1,024 native replay events;
- 428 recomputed current errors and 3,604 current-correct safety queries;
- 574 queries with an active same-query ChemAware candidate;
- 230 error packets carrying ChemAware evidence, including 224 with a chemical
  candidate distinct from the current winner: 156 counterfactually specific
  and 68 action-only; another 11 have a chemistry-tagged winner;
- 150 error packets with a distinct semi-hard candidate;
- 51,148 positive and 56,385 negative identity edges checked with no violation.

The recomputed official ranks differ from the frozen ledger at exactly three
queries.  All three are inside the existing float32 tie interval (score gaps
`0`, `5.96e-8`, and `0`); the numerical replay audit therefore excludes them
and finds no unexplained mismatch.  The observed 428-versus-ledger-427 error
count is consequently a tie convention, not a newly discovered biological or
retrieval discrepancy.

The serialized packet pool was reproduced byte-identically in two independent
runs (SHA-256
`96579ea24c5ed23dea267c53da9346618ed4bc27a9e515d9d8b5981e2eba2645`).
This preflight updates no weights; formal packets are rebuilt from the live
step-2,000 embedding inside the one-optimizer job.

## Evaluation and stopping

Role 2 compares steps 2,250/2,500/2,750/3,000 against the in-run step-2,000
Phase-A checkpoint.  Before checkpoint selection, step 2,000 must exactly
reproduce the protected 54 corrected / 12 introduced transitions and
`+2.1266 pp`; otherwise the job fails closed and role 3 is not opened.  An
online checkpoint advances only if it has:

- strictly positive Recall@1 and MRR;
- `corrected - 2 * introduced > 0`;
- nonnegative Recall@3, micro-AUC and macro-AUC;
- a strictly positive formula-cluster Recall@1 CI.

If no checkpoint passes, role 3 stays unopened and Phase A remains the result.
If one passes, exactly the selected checkpoint is evaluated once on frozen
formula role 3.  No checkpoint is automatically deleted.

This first job is the performance gate for the complete online packet method;
it does **not** by itself attribute an increment uniquely to the chemical slot,
because online winner refresh, query balancing, semi-hard selection and
ChemAware selection change together.  Only after a safe role-2 pass should a
second, fixed-budget matched-null chemical-slot control be run against the
frozen winning schedule.  Running that control before the performance gate
would double GPU cost without first establishing that the corrected training
objective is useful.

## Historical entry point

The filename below now launches the corrected V12 role-budget scheduler; it no
longer reproduces V11's global-query-uniform schedule:

```bash
sbatch tasks/run_chemaware_online_packet_native.sbatch
```
