# ChemAware specific-replay native triplets V2

## Frozen motivation

The pair-expanded role-2 run is a qualified negative result for model
selection, not a zero-signal result.  Relative to the confirmed stage-1
checkpoint, step 2000 improved Recall@1 by 0.759 pp (38 corrected, 23
introduced), but its formula-cluster CI crossed zero, Recall@3 fell by 0.152
pp, and `corrected - 2 * introduced = -8`.  No checkpoint passed the frozen
role-2 gate, so role 3 and the null arms were correctly not consumed.

The source pool contained 9,957 events, but only 1,516 were chemically selected.
Generic primary, secondary, and fallback events therefore supplied most update
directions.  The result identifies gradient selectivity and preservation—not
raw triplet count—as the next bottleneck.

## Method

For query `q`, the unchanged native DreaMS loss is

`max(0, margin - cosine(q, positive) + cosine(q, negative))`.

Only the empirical event distribution changes:

1. Retain exactly one stage-1 primary-boundary event `s_q` for every training
   query.  This is the safety replay background.
2. Consider active correct-arm events tagged `CHEMICAL_HARD`.
3. For candidate `c`, compute its null agreement as the number of three frozen
   content-permuted arms that also select `c` chemically.  Retain `c` only when
   null agreement is at most one.
4. Cap retained chemical events at three per query and order them by lower null
   agreement, native hinge, and activation.
5. For every null arm, keep the exact correct-arm query/event schedule and the
   exact same safety event.  Replace each correct chemical event by an active
   null event with a different candidate.  If an exact matched budget is not
   feasible, prune the lowest-priority correct event before training.

Before optimization, the distinct chemical events must comprise at least 6%
of all stored events and at least 15% of events with nonzero native hinge.  The
second threshold is the relevant gradient-purity gate; inactive, already-safe
replay rows cannot make a chemically weak curriculum appear strong.

Thus correct and null arms differ only in the chemically selected
candidate-reference event.  Model architecture, preprocessing, optimizer,
triplet loss, initialization, seed, update count, and safety replay remain
identical.

## Frozen execution and decision

- Initialization: confirmed stage-1 `best.ckpt` (+1.8144 pp on the historical
  role-3 development panel).
- Optimizer/loss: native DreaMS `ContrastiveHead`, Adam, cosine triplet margin.
- Residual continuation: LR `3e-6`, checkpoints every 500 steps through 3,000.
- Role 2 chooses a checkpoint only if Recall@1 and MRR improve, Recall@3,
  micro-AUC and macro-AUC do not fall, and `corrected > 2 * introduced`.
- Null arms run only after the correct arm passes role 2.
- Role 3 runs once at the selected update budget for the correct arm and all
  three exact-schedule nulls.
- Formula role 4 remains inaccessible.

The experiment may establish a larger incremental gain, but no gain is claimed
before the role-2 and role-3 reports exist.
