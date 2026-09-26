# ChemAware Phase-A residual-consensus native continuation V6

Date: 2026-09-26

## Decision

V5 is a controlled negative result, not an engineering failure. Its best
checkpoint retained `+2.0759 pp` versus official DreaMS but remained one net
top-1 query below protected Phase A (`+2.1266 pp`). Direct pairing against
Phase A showed 27 corrected and 28 introduced decisions. The expanded
multi-condition curriculum therefore contains corrective information, but
retraining from official DreaMS spends its update budget reconstructing Phase A
and causes excessive boundary churn.

V6 combines the two previously separated requirements:

1. initialize weights directly from the checksum-verified Phase-A checkpoint;
2. encode every training-role query and candidate reference under Phase A;
3. mine residual triplets from Phase-A's current errors rather than official
   DreaMS errors;
4. prioritize false identities that recur across experimental conditions;
5. replay the exact successful Phase-A pool as the dominant protection stratum;
6. add active correct-condition sentinels to constrain overshoot; and
7. use only the native DreaMS cosine triplet loss, model and Adam optimizer.

## Scientific target

For query spectrum `q`, true reference set `P(q)` and false molecule reference
set `N_c(q)`, retrieval uses

`score(q,c) = max_(r in N_c(q)) cosine(z_q, z_r)`.

At the protected Phase-A embedding `z_A`, V6 finds

`p*(q) = argmax_(p in P(q)) cosine(z_A(q), z_A(p))`

and the highest-scoring false reference `n*(q)`. A residual error exists only
when

`cosine(z_A(q), z_A(n*)) >= cosine(z_A(q), z_A(p*))`.

Every new corrective event is the native DreaMS triplet `(q,p*,n*)`. There is
no candidate-score distillation and no custom loss. A false identity receives
the consensus tier when it is the current hardest false identity in at least
two distinct instrument/collision-energy conditions for the same true
identity. This favors a repeated, spectrum-observable error over a singleton
condition accident.

For an identity that has both a selected Phase-A error and a Phase-A-correct
condition with gap in `(0,0.1]`, V6 stores the correct condition's exact
max-positive/max-negative pair as an active sentinel. It produces native hinge
gradient before that condition crosses the top-1 boundary.

The stored event sampler freezes the following expected masses:

| stratum | mass | within-stratum rule |
|---|---:|---|
| exact Phase-A pool | 70% | original uniform event replay |
| cross-condition residual | 18% | identity-equal |
| isolated residual | 7% | identity-equal |
| protection sentinel | 5% | identity-equal |

Thus 75% of expected updates are protection or exact Phase-A replay, while the
25% residual dose concentrates on errors that Phase A actually makes. These
values are frozen before role-2 evaluation and are not selected by role-2
performance.

## Optimization and evaluation

Training starts from the protected Phase-A checkpoint, uses learning rate
`1e-6`, batch size 4, native margin `0.1`, and at most 800 optimizer steps.
Checkpoints are saved every 100 steps. This is a short residual continuation,
not a second full fine-tuning run.

Formula roles remain:

- roles 0 and 1: triplet construction and training only;
- role 2: checkpoint selection against protected Phase A;
- role 3: independent confirmation only after a role-2 pass;
- role 4: inaccessible.

Role-2 advancement requires all of the following relative to Phase A:

- positive Recall@1 and MRR;
- strictly positive formula-cluster Recall@1 CI lower bound;
- `corrected - 2 * introduced > 0`;
- nonnegative Recall@3, micro-AUC and macro-AUC.

No pass means all artifacts are retained as negative evidence and role 3 is not
opened. A pass triggers role-3 confirmation, full role-2/3 evaluation, and
atomic artifact protection.

## Executed local construction audit

The full local graph was executed with official embeddings only as an
engineering stand-in for the unavailable local Phase-A checkpoint. This audit
does not make a performance claim. It established that the construction is
nonempty and executable:

| quantity | count |
|---|---:|
| exact Phase-A prefix events | 5,956 |
| Phase-A-current stand-in errors inspected | 1,881 |
| selected residual queries | 1,103 |
| cross-condition consensus queries | 561 |
| isolated residual queries | 542 |
| condition-diverse residual queries | 982 |
| chemical transfer events | 478 |
| protection sentinels | 331 |
| total events | 7,868 |

The Phase-A prefix was exactly unchanged, all new events were singleton
max-reference boundaries, 53,960 positive and 51,133 negative identity edges
passed, and roles 2--4 were untouched. Two independent builds were array-equal;
the engineering-stand-in training pool semantic hash was
`2cdbe7c79bc1b9d888affd2c4582bf3789d0bf5475e4a5ba95ed3e6ff0b384cb`.

The formal server run replaces the stand-in with a freshly encoded,
checksum-matched Phase-A role-0/1 cache. Counts may therefore differ and must
be read from that run's frozen report.

## Execution

```bash
sbatch tasks/run_chemaware_phasea_residual_consensus.sbatch
```

The job requests one GPU and no manual memory. It automatically resolves and
verifies the newest protected `+2.1266 pp` artifact. No checkpoint is deleted.

## Claim boundary

Before the server evaluation, V6 is a completed and audited method
implementation, not a demonstrated improvement. The established shared
embedding result remains Phase A at `+2.1266 pp` on role 2 and the historical
role-3-confirmed release at `+1.8144 pp`. A `+5 pp` claim requires the frozen
evaluation to report it; it cannot be inferred from triplet coverage.
