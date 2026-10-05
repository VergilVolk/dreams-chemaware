# Noise E4 multi-action direct injection: systemic audit and repair

Date: 2026-09-08  
Scope: noise-only direct fine-tuning of the shared DreaMS encoder. No distillation,
teacher embedding target, BioAware, ChemAware, downstream expert, or biological
analysis is introduced here.

## 1. Immutable result being explained

The completed two-arm job `2333052` is an engineering success but not the desired
scientific result. On 18,333 formula-held corrected-graph queries, targeted
one-best E4 changed Recall@1 by only `+0.152730 pp` relative to its E8
initialization (`67` corrected, `39` introduced, risk-net with lambda 2 = `-11`).
The shuffled arm was essentially tied on exact Top-1. The targeted arm also moved
MassSpecGym 10-ppm pooled pairwise AUROC by `-0.242018 pp` and `[M+H]+` pooled
pairwise AUROC by `-0.255802 pp` relative to E8.

The approximately `+0.635 pp` historical E4 result belongs to an older graph and
protocol and must not be substituted for the corrected-graph result. Likewise,
the `4.93--5.33 pp` values are action-space union headroom: they prove that raw
actions can correct that many clean errors when evaluated as actions, not that a
trained clean-only encoder has already inherited the same gain.

## 2. Root cause: the good actions were present in the ledger but mostly absent
from the optimizer

The routed ledger contained 32,127 strict corrective Top-1 action rows from
`N_mature`, `P_guided_original`, `E10B`, `E11`, `E12B`, `A4_exact`, and
`V4_gradient_path`. The one-best selector sorted by raw action margin and kept
one row per query, leaving 3,482 robust rows after the numerical floor. Thus
28,645 of 32,127 strict rows, or `89.16%`, did not enter training.

This was not an absence of good E4/E8/E10/E11/E12/A4/V4 actions. It was a
training-interface loss introduced after routing:

1. `drop_duplicates(query_index)` converted a multi-path action panel into a
   single maximum-margin path.
2. Maximum raw action margin favored already easy/far actions rather than actions
   that provide complementary transferable boundaries.
3. Of the 3,482 selected rows, 2,973 (`85.38%`) were P-like sources
   (`P_guided_original`, `E10B`, `E11`, `E12B`), so N/A4/V4 evidence was diluted.
4. At epoch 4, action-margin pass was about `99.97%`, while clean-margin pass was
   only about `56.90%`. The ungated action-rank term therefore kept rewarding
   views that were already correct instead of reserving rank pressure for the
   unresolved clean boundary.
5. Every optimizer step was gradient-clipped. The observed clip scale was about
   `0.155--0.173`, so loss components were being combined largely by their
   clipped direction rather than their intended pre-clip dose.
6. The held table used per-query matrix-vector scoring, while the registered
   metric panel used chunked `einsum`. A one-query fp32 boundary reversal made
   the two reported Recall@1 deltas disagree. This was an evaluator defect, not
   a biological or modeling effect.
7. The old action replay expanded each action into clean, action, positive, and
   negative spectra. Full multi-action replay would redundantly encode roughly
   450k spectra and could again consume hours before the first optimizer step.

The central diagnosis is therefore: action discovery retained large headroom,
but one-best selection, source-family dose imbalance, already-satisfied rank
pressure, clipping, and duplicate replay together formed the injection
bottleneck.

## 3. Repaired direct-training contract

The new arm is `multi_action_balanced`; it keeps the E8 initialization, shared
encoder, historical E4 symmetric direct objective, final-block/head scope,
learning rates, four epochs, four-action microbatches, clean rank, preservation,
and safety stream. It changes only the damaged injection interface:

- retain every strict corrective action above the frozen `5e-6` numerical margin
  floor; no one-best query collapse;
- schedule a no-replacement prefix containing every action before any recycling;
- preserve at least the historical 16 physical contacts per identity, while an
  identity with more than 16 unique actions is never truncated;
- solve weights from actual schedule multiplicity so every identity has total
  effective dose 16 and every `(source, family)` available for that identity
  receives an equal share;
- keep four physical safety examples per four-action batch and restore matched
  safety dose by a scalar, preventing a high action weight from creating an OOM
  safety forward;
- set action-rank gradient to zero only after the live action clears the frozen
  `0.05` E4 margin; clean rank, action-clean consistency, clean margin floor, and
  preservation remain active;
- apply the preregistered static `0.16` pre-clip scale obtained from job 2333052's
  training-gradient diagnostics, never from held performance;
- encode all action spectra exactly once for replay, while reusing the already
  computed current-E8 query/candidate embeddings to verify every stored winning
  boundary;
- use per-query matrix-vector products for both held ranks and the registered
  complete metric panel, and fail if their Recall@1 differs by even one query.

This remains direct fine-tuning. Raw action spectra and their exact candidate
boundaries enter the ordinary shared-encoder loss. There is no teacher embedding
or teacher-margin regression target.

## 4. Scientific gate

No `4--5 pp` encoder claim is made before the server run. Promotion requires all
of the following on the same frozen formula-held query/candidate graph:

- Recall@1 improvement at least 4 pp versus E8 and strict improvement versus the
  source/family-matched shuffled-action arm;
- Recall@1/2/3/5/10/20, MRR, mean/median rank, macro-query AUROC/AUPRC,
  micro-candidate AUROC/AUPRC, positive-vs-best-negative margin, and Top1-Top2 gap
  all nonnegative in their registered direction;
- corrected greater than introduced, positive lambda-2 risk-net, and nonnegative
  near-subset metrics;
- strictly positive multiplicity-corrected formula-cluster confidence intervals
  versus both E8 and shuffled control;
- MassSpecGym all-adduct and `[M+H]+` 10-ppm pooled pairwise AUROC/AUPRC reported
  as supplementary real-task metrics, explicitly not mislabeled as exact NIST20
  paper replication;
- held-table and registered exact Recall@1 equality;
- preservation gate and complete provenance hashes.

## 5. Server entry point

Upload the changed task files listed in the handoff, then run only:

```bash
sbatch tasks/run_noise_e4_multiaction_balanced_2gpu.sbatch
```

The job requests exactly two GPUs and contains no manual memory request. Targeted
and shuffled arms run concurrently, fail fast on an implementation error, and
retry only a status-137 arm alone after both workers exit. The final directory is
published atomically only after both model validators and the paired summary pass.
