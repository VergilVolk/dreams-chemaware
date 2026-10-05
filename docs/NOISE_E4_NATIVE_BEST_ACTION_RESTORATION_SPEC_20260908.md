# Noise E4-native best-action restoration

Date: 2026-09-08

## Corrected diagnosis

The prior result was not erased because the spectrum actions stopped working.
Two different quantities were collapsed into one claim and the subsequent V5
training run also replaced the mature optimization kernel.

- E12-B measured a real held-action union gain of `+4.929934 pp` on 5,923
  queries, with formula-cluster CI `[+3.642973, +6.530802] pp`.  It applied the
  best observed raw-spectrum action independently to each held query.  It was
  not itself a newly trained clean-input encoder.
- The corrected-graph V5 ledger retained at least `+5.334988 pp` strict Top-1
  action capacity in the 65,286-query outer-train geometry.
- The V5 bounded trainer used only 512 corrective queries for one epoch and
  substituted a new multi-branch margin-transfer/calibration objective for the
  historical E4 direct objective.  Its `+0.024414 pp` versus initial E8 result
  therefore rejects that rewritten training implementation; it does not erase
  the raw-action result.
- The strongest previously established formula-held clean shared-encoder gain
  remains the E4-A multi-fold result near `+0.635 pp`.  The purpose of this run
  is to close the gap between that encoder result and the 4--5 pp action
  capacity without changing the scientific route.

## Restoration boundary

The new job does not call `train_noise_corrected_routed_direct.py` and does not
rerun V4 mining.  It reuses the immutable ledger already preserved by job
2332784, but feeds its exact action tensors into
`train_noise_final_e4a_direct_augmentation.py`.

The training objective is the mature E4 objective, unchanged in coefficients:

```text
1.00 * clean shared rank
+ 1.00 * real action-view shared rank
+ 0.25 * symmetric clean/action consistency
+ 2.00 * clean margin floor
+ 5.00 * clean/reference preservation
```

The same trainable DreaMS encoder processes clean queries, action spectra,
positive references and negative references.  Inference consumes only the
ordinary clean spectrum.  There is no teacher embedding, teacher margin,
distillation head, P2b reranker or P3 consumption.

The historical optimization schedule is frozen:

- E8 initialization;
- `symmetric/shared` gradients;
- four epochs and four action views per identity budget;
- four actions per optimizer microbatch;
- last Transformer block plus official projection head;
- backbone LR `2e-6`, head LR `1e-5`;
- weight decay `1e-4`, full fp32, gradient clip `1.0`.

## Lossless action interface

Only outer-train rows satisfying all three conditions can enter the positive
action stream:

1. current E8 clean rank is not Top-1;
2. routed supervision is corrective;
3. the materialized action itself is Top-1.

This removes merely margin-improving but still-wrong actions from the positive
target stream.  Among qualifying actions, exactly one action is selected per
query: the action with the largest full-candidate-graph action margin, with a
stable source/family/recipe/action-id tie break.  This restores the semantics
of the historical best-action union instead of averaging many unequal actions
for one query.  The qualifying bank includes actions from
`N_mature`, `P_guided_original`, `E10B`, `E11`, `E12B`, `A4_exact` and
`V4_gradient_path`.

After selection every winning action receives unit E4 weight.  Routing score,
teacher advantage and prior mechanism-balancing coefficients are not used as
loss targets or continuous weights.

Before optimization, the current E8 checkpoint must reproduce every selected
row's stored clean rank/margin and targeted-action margin within fp32 tolerance.
The action's exact winning positive spectrum and exact hardest negative
spectrum are forcibly retained in the minibatch.  Current-E8 top clean
references are also retained.  Consequently the E4 loss sees the same candidate
boundary that made the raw action successful; it cannot silently turn a rich
action into a different local task.

Every selected action is exposed once before any action is recycled.  Only
then are actions repeated to give every identity the historical `4 epochs x 4
views` total exposure.  Identities with more than 16 unique winning actions are
not truncated; this is the only permitted departure from the equal historical
budget.

## Preservation repair for warm start

The old E4 implementation was written for official-checkpoint initialization.
With an E8 warm start, protecting only official-correct queries leaves queries
newly corrected by E8 unprotected.  Materialized mode therefore recomputes all
83,619 corrected-graph queries with the exact E8 checkpoint before the first
optimizer step.  E4's existing margin-floor and preservation losses are then
anchored to current E8, and every current-E8-correct outer-train query is
eligible for the safety stream.  This changes the stale baseline, not the loss
formula.

## Two-GPU causal test

The Slurm allocation runs simultaneously:

- GPU 0: exact query-matched best action tensors;
- GPU 1: source/family/exact-recipe-matched cross-query shuffled action tensors.

Both arms share the initial checkpoint, selected rows, reference construction,
identity budget, optimizer schedule and held formula fold.  Only the action view
changes.  Shuffled donors are drawn only from the already selected best-action
union, never from weaker or harmful ledger rows.  This directly tests whether
the restored gain comes from query-specific action content rather than generic
extra training.

The one server command is:

```bash
sbatch tasks/run_noise_e4_native_best_actions_2gpu.sbatch
```

The job requests exactly two GPUs and does not request memory manually.  All
Python compilation and tests execute inside the Slurm job, not on the login
node.  By default it reuses:

```text
data/validation/noise_corrected_best_v5_canary_fold_0_run_2332784/ledger
```

Set `SOURCE_RUN` only if the same completed artifact was stored under another
job-specific directory.

## Job 2332915 implementation failure and repair

Job 2332915 did not reach an optimizer step and therefore produced no
scientific result.  Its two workers failed for separate implementation reasons:

- the original evaluator expanded the query embedding once for each of
  6,220,661 spectrum edges before taking an `einsum`.  At 1,024 fp32 dimensions,
  each expanded operand is 23.73 GiB and the two operands approach 47.46 GiB
  per worker before result/workspace overhead.  The simultaneous arms crossed
  the CPU cgroup limit and one worker exited 137 with seven recorded OOM kills;
- that same global `einsum` was not the per-query matrix-vector scorer used by
  the V4 action router.  At strict near-zero ties, this changed fp32 reduction
  order and triggered the clean-rank replay exception in the other worker.

The repair keeps the action union, E4 loss and training schedule unchanged.
Initial, replay and final retrieval are now scored in bounded complete-query
blocks with `score_candidate_boundary`, exactly as in the router.  Replay
reports the number of rank mismatches, the subset explained by near-zero ties,
the unexplained count and margin error; any unexplained mismatch or margin
drift above `5e-4` still fails before optimization.

Two additional duplicated allocations were removed: the full official cache is
released after the reachable 87,848-row array is materialized, the loaded CPU
checkpoint state is released after `load_state_dict`, and the immutable E8
reference array is aliased rather than copied.  The job still requests two GPUs
and no explicit memory.  It first runs both arms concurrently; if and only if a
worker exits 137, it automatically retries that killed arm alone after both
workers have exited, without deleting or overwriting an existing result.

A subsequent retry exposed a separate CUDA batching bug in the pre-optimizer
targeted-action replay.  `eval_batch_size=256` had been passed as though it were
an action minibatch size.  Each action expands to clean, action, four positive
and eight negative spectra, so the call sent about 3,584 spectra into one
Graphormer forward and requested 100.22 GiB on a 31.74-GiB GPU.  Replay is now
bound to the frozen E4 `batch_actions=4` geometry, at most about 56 spectra for
the current layout.  An independent 64-spectrum hard guard checks every replay
forward before it reaches CUDA, and the validator requires the recorded replay
batch to agree with the training configuration.  This changes neither selected
actions nor the loss; it corrects only an audit batching-unit error.

Job 2333030 then completed the bounded replay and separated two candidate-row
semantics that the implementation had incorrectly conflated.  The current-E8
full-graph clean replay was exact for all 3,483 selected queries (zero rank
mismatches; maximum margin error `1.19e-7`), and the targeted action replay had
maximum margin error `1.37e-6`.  However, only 3,482 actions remained strictly
positive in the four-action training batch, and the separately reported clean
margin error was large.  This was not action loss: the action-active spectrum
row had replaced, rather than supplemented, the clean-active spectrum row of
the same candidate molecule.  Consequently the action boundary replayed while
the clean boundary inside the minibatch did not.

Materialized examples now retain the union of clean-active and action-active
positive/negative spectrum rows.  Because the scalar rank loss takes a maximum
over those rows, this union reproduces both full-graph margins without changing
their values; it can add at most one positive and one negative spectrum to the
current layout, keeping four examples within the 64-spectrum CUDA guard.  A
stored best action with margin at or below `5e-6` is also excluded as a
machine-precision boundary rather than counted as robust corrective content.
The job preflight reports the before/after count and recomputes the surviving
training-geometry headroom, which must remain at least 4 pp.

## Simultaneous evaluation and strict gate

Both arms are evaluated on all 18,333 fold-0 held queries from the corrected
83,619-query development graph.  The report includes:

- Recall@1/2/3/5/10/20;
- MRR, mean rank and median rank;
- macro-query AUROC/AUPRC;
- micro-candidate AUROC/AUPRC;
- positive-vs-best-negative margin and raw/signed Top1--Top2 gap;
- corrected, introduced and `corrected - 2 * introduced` risk-net;
- the same retrieval and outcome panel on the near subset;
- Bonferroni-adjusted formula-cluster paired Recall@1 CIs;
- MassSpecGym all-adduct and `[M+H]+` strict-10-ppm pooled pairwise
  AUROC/AUPRC.

The pooled MassSpecGym result is not labelled an exact reproduction of the
NIST20 paper value near 0.85.

Promotion requires all of the following, simultaneously:

- at least `+4.0 pp` Recall@1 versus initial E8;
- targeted beats shuffled on Recall@1;
- formula-cluster CI lower bounds are strictly positive versus both E8 and
  shuffled;
- positive risk-net versus both comparators;
- no registered metric direction regresses versus either comparator.

The action ledger makes a 4--5 pp result plausible; it cannot mathematically
guarantee that the clean encoder will realize it.  The output may be called a
4--5 pp shared-embedding result only when the generated summary sets
`four_pp_full_metric_gate_passed=true`.
