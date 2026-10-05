# Noise native Stage-4: complete Stage-1 actions plus broad measured positives

## Current executable correction: query-calibrated residual v5

The executable no longer trains every residual action row as an independent
optimizer dose.  That v4 design preserved the seven mature sources but violated
the already established Stage-3 lesson that action multiplicity must not become
query multiplicity.  The current builder therefore applies the following
composition before any optimizer step:

- all seven registered sources (`A4_exact`, `E10B`, `E11`, `E12B`,
  `N_mature`, `P_guided_original`, `V4_gradient_path`) enter candidate
  construction;
- the old Stage-1 positive is excluded and each action receives new measured
  same-identity positives plus easy/medium/semi-hard different-identity
  negatives;
- invalid positive-negative cross-combinations are removed so every pair that
  the native dynamic sampler can draw is hinge-active, bounded away from the
  destructive extreme tail, and better than both the registered control and
  clean same-boundary geometry;
- exactly one residual action is selected per query, accompanied by exactly one
  clean preservation event; 256 action-free protection queries are retained;
- the continuation uses the empirically safer native Adam `1e-6` scale, not the
  failed late-stage `5e-6` scale;
- training is adjudicated on both the corrected MassSpecGym held graph and the
  sealed GNPS Gold/Silver identity- and formula-disjoint 10-ppm panels. No
  network download is part of the job.

The sections below retain the v3/v4 history so the failed constructions are not
silently rewritten; wherever they conflict with this section, v5 is current.

## Post-run correction after run 2345708

Run `2345708` exposed a construction error in the original Stage-4 question.
The `25,736` Noise events replayed the same Stage-1 action/positive/negative
boundaries from a checkpoint that had already learned them. The `7,839` newly
added `broad-positive` events were action-free clean-anchor triplets. Therefore
the experiment did **not** broaden the relation between a Noise action and
same-identity measured positives. It doubled the old action dose and added a
generic hardest-negative continuation. The resulting `+0.08182 pp` Recall@1
increment and `-6.00153 pp` MassSpecGym 10-ppm pooled AUROC change reject that
construction.

The corrected builder version is
`noise_native_action_positive_residual_v4`. It starts from the evaluated
Stage-1 targeted checkpoint weights but, contrary to the earlier text here,
the executed shared trainer set `optimizer_restore = None` and therefore used
a fresh Adam state. The generated training report records
`warm_start_adam_state_restored: false`. This historical Stage-4 artifact must
not be described as an exact Adam continuation. Its triplet changes were:

- the Stage-1 exact positive row is forbidden from every new action event;
- every representable action is the anchor of a new positive pool containing
  its original clean query and, when available, additional same-identity
  measured spectra;
- negative membership contains easy/medium/hard molecule representatives and
  avoids the Stage-1 exact negative row whenever another row exists;
- the former 7,839 action-free hardest-negative events are removed;
- only four action-free clean events remain for native final-batch padding;
- the run is not padded back to 8,394 optimizer steps with unrelated triplets.

This repair changes triplet membership and resulting one-pass length only. It
does not change the DreaMS preprocessor, shared encoder, native head, cosine
triplet margin, Adam hyperparameters, batch size, FP32 precision, or formula
split. The earlier sections below describe the rejected v3 design and are kept
as an error record rather than a current specification.

## Fixed factual baseline

- The continuation starts from the evaluated Stage-1 targeted checkpoint from
  run `2344820`.
- Stage-1 targeted minus its matched control is `+1.26548 pp` Recall@1 with a
  positive formula-cluster CI. This is the causal action-content contrast.
- Stage-1 targeted minus official DreaMS is `+0.49637 pp`. This is the absolute
  shared-encoder gain. The two values must not be interchanged.
- The requested `5 pp` remains a promotion threshold, not a guaranteed result.

## Only permitted intervention: triplet membership

The training implementation preserves every effective Stage-1 action exactly
once. Each action keeps its original positive row, hard-negative row, target or
same-query control tensor, and the original clean-boundary fallback for an
unrepresentable all-zero action view. There is no post-hoc one-action-per-query
thinning.

The remaining slots in the fixed Stage-1 event budget are real measured
triplets selected across outer-train formulas:

```text
anchor   = measured clean query
positive = measured spectrum of the same molecule
negative = most similar measured different-molecule candidate
```

Positive difficulty is requested deterministically as 25% easy, 50% median and
25% semi-hard. The semi-hard tier uses the 25th percentile rather than the
absolute farthest replicate. If a requested positive has zero native hinge
gradient against the fixed most-similar negative, selection falls back within
that query to another measured positive tier and records both requested and
realized tiers. Measured triplets must have initial margin in `[-0.1, 0.1)`.
All action triplets retain their Stage-1 exact boundaries even when their margin
lies outside that measured-stream band.

## Frozen native DreaMS runtime

- native `ContrastiveSpectraDataset`, one positive and one negative;
- native `ContrastiveHead` and 1024-dimensional linear head;
- native `SpectrumPreprocessor`, 100 peaks and precursor intensity `1.1`;
- native cosine triplet-margin hinge with margin `0.1`;
- native `torch.optim.Adam`, learning rate `5e-6`, weight decay `0`;
- batch size `4`, FP32, backbone unfrozen at epoch zero;
- one pass, `33,575` base events and exactly `8,394` optimizer steps;
- Stage-1 model tensors and its single Adam state are restored exactly;
- targeted and control arms use identical relations, ordering, seed and dose.

No custom loss, distillation target, action-rank loss, optimizer projection,
adapter, parameter weighting or learning-rate change is admitted.

## AUC is co-primary and numerical

The old Stage-1 report is not assumed to contain AUC. The Stage-4 sbatch first
re-evaluates the actual Stage-1 champion with the same frozen evaluator used for
both new arms. The summary writes candidate, reference, numerical delta and
delta in percentage points for:

- macro-query AUROC and AUPRC, full and near subset;
- micro-candidate AUROC and AUPRC;
- MassSpecGym same-adduct strict-10-ppm pooled pairwise AUROC and AUPRC;
- `[M+H]+` strict-10-ppm pooled pairwise AUROC and AUPRC.

Paired formula-cluster CIs are additionally computed for full and near
macro-query AUROC/AUPRC. Promotion requires every registered AUC/AUPRC number to
improve against Stage-1, matched control, official DreaMS and mature E8, plus
positive macro-AUC formula-cluster lower bounds against Stage-1 and control.
Recall@1/2/3/5/10/20, MRR, mean/median rank, margins, signed Top1-Top2 gap,
corrected/introduced, lambda-2 risk-net, near metrics and their registered CIs
remain mandatory.

## Server entry and storage boundary

Before submission, run the read-only inventory and capture active jobs:

```bash
bash tasks/audit_server_storage_readonly.sh data/validation > server_storage_inventory.tsv
squeue -u "$USER" -o '%.18i %.30j %.2t %.10M %.6D %R' > server_active_jobs.txt
```

Do not delete run `2344820`, the corrected graph, official slim checkpoint,
mature-E8 comparator, MassSpecGym HDF5, source manifest, reports, held-per-query
tables or hashes. Exact deletion paths must be chosen only after reading the
server inventory; recursive globs are forbidden.

After cleanup, the sole entry point is:

```bash
sbatch tasks/run_noise_dreams_native_broad_positive_stage4_2gpu.sbatch
```
