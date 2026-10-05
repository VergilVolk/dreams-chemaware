# ChemAware Fragment-Graph Native Route: Contract and Exploratory Result

## Status

`EXPLORATORY_SIGNAL_RETAINED / FORMAL_RELEASE_BLOCKED_PENDING_HARDENED_REBUILD`

This document covers the fragment-graph seed route and the subsequent
fragment-graph/MassBank union continuation. It does not upgrade either run to
a protected result.

## Scientific intervention

The permitted intervention is triplet membership only. Chemical evidence may
select a truth-blind `(query, true candidate, false candidate)` relation. The
training example remains the native DreaMS tuple `(q, p, n)`, with the native
`ContrastiveSpectraDataset`, `ContrastiveHead`, cosine triplet-margin loss and
Adam optimizer. Chemical scores are neither regression targets nor inference
features, and no teacher/distillation loss is used.

For the hardened route, one distinct false candidate can contribute at most
one event. That event contains one highest-quality active true reference and
one hardest active false reference. Extra spectra of the same false candidate
are not additional chemical evidence. Event mass is capped per query, formula
and identity.

## Exploratory artifacts and observed signal

- Source construction: `run_2346306`.
- Fragment continuation: `run_2346408_resume_run_2346306`.
- Exploratory step-750 checkpoint: role-2 Recall@1 was reported as 92.00%,
  approximately +2.3797 percentage points versus official and +0.2532 points
  versus the protected Phase-A checkpoint. The latter paired confidence
  interval crossed zero; this is an exploratory signal, not a confirmed gain.
- Union continuation: `run_2346441`. Step-1000 was reported as 92.0506%
  Recall@1 on role 2, approximately +2.4304 points versus official and +0.0506
  points versus the exploratory step-750 seed. It is not a formal candidate.

The server-side source ledgers and run artifacts were not all present locally
when this document was written. The numbers above therefore remain
provenance-qualified server observations until their exact reports and hashes
are copied into the protected local artifact set.

## Why the old union chain is not releasable

The former chain admitted C-tier single-source positive deltas without using
the per-relation matched-null scores, multiplied one false candidate through
several reference combinations, reset Adam at each continuation, allowed the
formula-role argument to act only as a label, and confirmed only against the
exploratory seed. These faults can inflate training mass and invalidate the
meaning of a green release decision even though the exploratory numerical
signal itself remains useful.

## Hardened release contract

1. The policy arrays `(query, formula, identity, baseline_rank)` must exactly
   equal the canonical frozen role-2 or role-3 evidence selected by the stated
   formula role.
2. Backbone checkpoint loading is strict. Any missing or unexpected parameter
   aborts before encoding.
3. Every admitted candidate relation must dominate its own matched controls
   under `pairwise_dominance`; source-family qualification alone is not enough.
4. One relation yields one singleton `(p, n)` event. No reference
   multiplicity is counted as independent chemical supervision.
5. Caps apply per query, formula and identity. At least 1,000 distinct chemical
   events and 500 distinct chemical queries are required before GPU training.
6. The embedding cache report must match the exact checkpoint, manifest, row
   array and embedding array hashes.
7. The continuation restores the single native Adam state from the same
   checkpoint that supplies the model weights, and proves that all registered
   Adam step counters advance.
8. Role-3 release must pass the paired seed gates and be non-regressive versus
   protected Phase A on Recall@1, Recall@3, MRR, micro-AUC and macro-AUC.
9. Numerical replay exclusions are capped by an absolute count, reported as
   exact manifest queries and hashed. Candidate boundary ties are reported,
   not silently removed.
10. Training remains one GPU, fixed-step checkpointing, and no manual Slurm
    memory request.

## Current decision

Keep the existing step-750 and union step-1000 artifacts as exploratory
starting points. Do not publish, freeze as champion, or use their old green
gates. Rebuild the candidate corpus with the hardened independent-relation
contract. If it cannot reach the frozen 1,000-event/500-query coverage floor,
stop before training and expand genuinely independent chemical sources rather
than repeating references.
