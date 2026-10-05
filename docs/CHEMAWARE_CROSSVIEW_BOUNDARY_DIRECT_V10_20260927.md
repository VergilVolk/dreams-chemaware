# ChemAware cross-view boundary direct fine-tuning V10

Date: 2026-09-27

## Objective

Continue the protected Phase-A `+2.1266 pp` shared embedding using only native
DreaMS spectrum triplets.  This route does not distil a teacher, fit an
auxiliary head, add an inference-time chemical input, or train a reranker.

## Why V9 stopped

The formal Phase-A V9 preflight found 151 current errors but only 23 independent
queries / 23 formulas with a same-query, same-current-false-candidate proof.
Those proofs emitted 58 correction events, below the frozen `100 events / 50
queries / 40 formulas` coverage gate.  Training correctly did not start.

## V10 evidence unit

For held spectrum `q`, true candidate `c+`, current Phase-A hardest false
candidate `c-`, action arm `a`, and other spectra `r` of the same known
training identity, candidate utilities are aligned by molecular identity:

`U_a(q,c) = Q25({u_a(r,c): r != q})`.

The candidate-pair boundary and chemical specificity are:

`B_a(q) = U_a(q,c+) - U_a(q,c-)`

`S(q) = B_correct(q) - max_a!=correct B_a(q)`.

A correction triplet is emitted only when the primary true-vs-false boundary
clears the frozen absolute action threshold, `S(q) > 0`, and the current
Phase-A false reference is active under the native margin `0.1`.  Applying the
threshold to the pair boundary is necessary because an official-baseline true
candidate has the defined no-action utility zero; evidence then comes from the
matched false candidate having sufficiently negative utility.

Candidate action tables omit the official top candidate by construction.  V10
represents that candidate by the exact abstention action, whose utility is zero
under every arm.  This is not imputation: zero is the policy's defined
no-action reference.  It restores valid baseline-vs-challenger boundaries
without inventing candidate evidence.

## Leakage control

- Formula role 0 is mined only by a policy fit on role 1.
- Formula role 1 is mined only by a policy fit on role 0.
- The source role uses a deterministic formula-level two-fold split for
  nuisance out-of-fold residual targets.
- A held spectrum never supplies utility to its own proof.
- Candidate slots are never aligned across queries; molecular identity is the
  only alignment key.
- Evidence is never broadcast from one spectrum to all spectra of an identity.
- Formula roles 2, 3 and 4 are untouched during triplet construction.

Ground-truth identity is used only in the training split to define the positive
candidate and ordinary triplet labels, exactly as supervised DreaMS triplet
training requires.

## Direct training object

The saved pool contains only:

`anchor_idx, positive_ptr, positive_idx, negative_ptr, negative_idx,
source_query, negative_candidate, source_tag, curriculum_role`.

It contains no chemical score, utility, teacher target, sampling weight or
candidate input.  After mining, training is the existing
`train_chemaware_dreams_native.py` pipeline initialized from protected Phase A:

- unchanged `ContrastiveSpectraDataset`;
- unchanged `ContrastiveHead`;
- cosine triplet margin `0.1`;
- Adam, LR `1e-6`;
- uniform shuffled loader;
- 500 updates, checkpoints every 100 updates.

Role 2 selects only a checkpoint that safely beats Phase A with a positive
formula-cluster CI.  Only then is role 3 evaluated.  Role 4 remains untouched.

## Executed local end-to-end smoke

The official cache was used only as an explicitly marked engineering stand-in.
Both cross-fit directions ran through action scoring, policy fitting,
leave-one-spectrum-out boundary proof, triplet serialization and identity-edge
audit.

| Quantity | Smoke result |
|---|---:|
| held repeat queries | 176 |
| held identities | 64 |
| current errors | 20 |
| errors with true and false context | 20 |
| proven correction queries | 4 |
| correction events | 9 |
| safety events | 27 |
| official replay events | 8 |
| total native events | 44 |
| positive / negative edges checked | 160 / 242 |

The smoke is an engineering pass, not a performance result.  Its low strict
retention reinforces the requirement that the formal full-data coverage gate
must run before any neural-network update.

## Server entrypoint

```bash
sbatch tasks/run_chemaware_crossview_boundary_direct.sbatch
```

The job requests exactly one GPU and no manual memory.  It first constructs the
full formula-crossfit triplet pool.  If the frozen coverage gate fails, it exits
successfully before training and reports that no weights were updated.
