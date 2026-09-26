# ChemAware dense true-support native Triplets V8

Date: 2026-09-26

## Decision

V7 is a valid negative result.  Its formal Phase-A construction contained only
96 independent correction queries and 177 correction events, while a custom
replacement sampler assigned 45% of all sampling mass to 8 strict identities.
The resulting continuation increased sampled margins but reduced role-2
Recall@1 relative to the protected `+2.1266 pp` Phase-A embedding.

V8 removes the replacement sampler and changes the scientific unit from one
representative query per supported identity to a unique, currently active
retrieval boundary.  The hard construction gate is at least 1,000 distinct
corrective `(anchor, max-positive, active-negative)` signatures.  Duplicating
or resampling an existing signature cannot satisfy this gate.

## Construction

Only formula roles 0 and 1 are inspected.  The frozen counterfactual chemical
gate is unchanged:

`D_m(t) = f_correct,m(t) - max_k f_null-k,m(t)`.

The known training identity is eligible when its true candidate beats all
three matched content nulls on at least two of the four rule metrics.  The
strict tier additionally requires all support and action-advantage metrics to
beat every null.

For every eligible identity, all of its role-0/1 query spectra are rescored
under the checksum-verified protected Phase-A embedding.  A spectrum produces
corrections only if it is currently wrong.  V8 then retains at most six
condition/false-identity-diverse error anchors.  For every retained anchor it
uses:

- the current maximum-scoring true reference;
- up to four distinct false candidate molecules with positive native hinge;
- up to three currently active references per false candidate; and
- singleton native DreaMS events only.

Thus every correction satisfies

`0.1 + cosine(q, negative) - cosine(q, max-positive) > 0`

at construction.  Current-correct boundaries are retained only when their gap
is in `(0, 0.1]`, so every safety sentinel is also active rather than a
zero-loss placeholder.  Exactly 1,024 unmodified official DreaMS triplets are
added for global replay.

There is no `sampling_weight` array.  Training uses the unmodified native
DreaMS shuffled DataLoader without replacement, `ContrastiveSpectraDataset`,
`ContrastiveHead`, cosine triplet margin, and Adam.

## Executed local full-graph engineering audit

The protected Phase-A weights are server-only, so the local execution uses the
official embedding cache strictly as an engineering stand-in.  It is not a
performance result.

| Quantity | V8 local audit |
|---|---:|
| chemically supported identities | 347 |
| supported query spectra available | 1,862 |
| current-error query spectra | 973 |
| retained independent error queries | 913 |
| distinct identity-to-false-identity boundaries | 867 |
| strict correction triplets | 982 |
| broad correction triplets | 3,443 |
| unique active correction triplets | **4,425** |
| active safety sentinels | 208 |
| untouched native DreaMS replay | 1,024 |
| total triplets | 5,657 |

All 51,749 positive and 48,922 negative identity edges passed.  Two complete
constructions were array-identical.  The ordered array semantic hash was
`9684c3b8de443fbf3a2ea215ff04ce4a793cc9f43bfe2725d3c9d6c6580d7d19`.

Formal server construction must independently pass all of:

- at least 1,000 unique active correction triplets;
- at least 200 independent correction queries;
- at least 200 distinct identity-to-false-identity boundaries;
- both strict and broad correction strata nonempty;
- active safety sentinels present;
- exact 1,024-event DreaMS replay;
- no custom sampling weights; and
- untouched formula roles 2, 3 and 4.

Failure of any construction gate stops before training.

## Training and evaluation

Training initializes from the protected Phase-A checkpoint and uses the native
DreaMS runtime at learning rate `1e-6`, batch size `4`, margin `0.1`, and at
most 1,500 optimizer steps.  Six checkpoints at 250-step intervals are
evaluated on frozen formula role 2 against Phase A.  Only a checkpoint with
positive Recall@1/MRR, positive formula-cluster CI lower bound, positive
`corrected - 2 * introduced`, and nonnegative Recall@3/micro-AUC/macro-AUC may
reach independent role 3.  Formula role 4 remains untouched.

The sole server command is:

```bash
sbatch tasks/run_chemaware_dense_true_support_native.sbatch
```

The job requests exactly one GPU and does not specify memory manually.

## Claim boundary

The established result remains the protected `+2.1266 pp` role-2 Phase-A
embedding and the historical `+1.8144 pp` role-3-confirmed embedding.  V8 has
passed construction, identity, hardness, determinism, and execution-contract
audits locally; retrieval improvement is not claimed before frozen server
evaluation.
