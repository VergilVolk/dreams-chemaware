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
2. Re-open the frozen 22-slot candidate evidence as a **discovery bank only**
   and retain at most two directionally ranked chemical candidates per query.
   A candidate selected by the correct arm because it is *rejected more than
   under the three content-permuted arms* receives priority. An already-hard
   candidate retains both its safety and chemical tags. Generic fallback rows
   in the discovery bank never enter the optimizer curriculum.
3. For candidate `c`, compute its null agreement as the number of three frozen
   content-permuted arms that also select `c` chemically.  Before training,
   scan agreement limits 0, 1, and 2 and freeze the strictest level that reaches
   all coverage and active-gradient gates.  Agreement 3 is forbidden because it
   would admit candidates shared by every null and provide no counterfactual
   chemical specificity.
4. Cap retained chemical events at two per query and order them by lower null
   agreement, native hinge, and activation. The final curriculum keeps exactly
   one checkpoint-hard safety boundary for all 4,032 queries; it is not the
   earlier 9,957-event pair-expanded training pool.
5. For every null arm, keep the exact correct-arm query/event schedule and the
   exact same safety event.  Replace each correct chemical event by an active
   null event with a different candidate.  If an exact matched budget is not
   feasible, prune the lowest-priority correct event before training. Also
   require mean hinge-gap <=0.05 and per-event hinge-gap <=0.25 after matching;
   geometry outliers are removed before any GPU optimization.

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
- Residual continuation: LR `3e-6`, checkpoints at 250/500/750/1,000 steps.
  The 1,000-step cap is below one full pass through the approximately 4.8k
  event residual pool and explicitly follows the pair-expanded failure: more
  active pairs and 2,000--3,000 steps increased introduced errors.
- Role 2 chooses a checkpoint only if Recall@1 and MRR improve, Recall@3,
  micro-AUC and macro-AUC do not fall, and `corrected > 2 * introduced`.
- Null arms run only after the correct arm passes role 2.
- Role 3 runs once at the selected update budget for the correct arm and all
  three exact-schedule nulls.
- Formula role 4 remains inaccessible.

The experiment may establish a larger incremental gain, but no gain is claimed
before the role-2 and role-3 reports exist.

## 2026-09-24 local dose audit

The official-embedding cache was used only as a conservative, local structural
proxy; these are not retrieval results and the server must rebuild the pool in
the stage-1 checkpoint geometry.  With the same directional candidate bank and
the same correct-vs-three-null geometry constraints:

| chemical-event cap per query | selected null agreement | chemical events | chemical queries | active chemical fraction |
|---:|---:|---:|---:|---:|
| 1 | 1 | 295 | 295 | 13.52% |
| 2 | 1 | 522 | 296 | 21.67% |
| 4 | 1 | 759 | 297 | 28.68% |

Cap 1 misses the predeclared 15% active-gradient floor. Cap 4 adds 237 events
over cap 2 while covering only one additional query. Cap 2 is therefore the
minimum sufficient residual dose. This explicitly rejects the failed strategy
of treating more active triplets as intrinsically better.
