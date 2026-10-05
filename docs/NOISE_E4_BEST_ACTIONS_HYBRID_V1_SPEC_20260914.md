# Noise E4 Best-Actions Hybrid V1

## Decision

This is not a rerun of pure E4 and it is not a continuation of V11.

The only approved training composition is:

1. the complete frozen seven-source strict corrective action panel;
2. the historical E4 shared-encoder direct objective and safety objective;
3. official DreaMS initialization;
4. ActionInjectorV1 as the sole optimizer-boundary replacement.

The stored historical E4 checkpoint is an evaluation comparator. It is not
retrained in this job.

## Frozen action supplier

The source ledger is
`data/validation/noise_corrected_best_v5_canary_fold_0_run_2332784/ledger`.
Its required hashes are:

- `report.json`: `246e7e871e6669fec9f2c62330a69bf9eb563a1b97b72b89732cecd682844349`;
- `training_actions.csv.gz`: `93f0785a69b5e323490a0b543059fa213fabb6ff848697831efba5b3fed667aa`;
- `action_spectra.npz`: `6d57615aebbfb7bd6327bb7edb143c6837d2586221c61aa2ff45e833f5e1512a`.

After the frozen `5e-6` numerical margin floor, the supplier must contain
exactly 32,114 action views for 3,482 queries and all seven sources:

- `N_mature`;
- `P_guided_original`;
- `E10B`;
- `E11`;
- `E12B`;
- `A4_exact`;
- `V4_gradient_path`.

`N_mature` carries the mature candidate-gradient and role-confounder action
content underlying the E4 route. E4 itself is the training/scientific kernel,
not an eighth independent spectrum-action table; E8 is frozen selection
provenance only. Thus this panel does not omit the E4 action content and does
not duplicate it under a second source label.

No one-best or per-query `drop_duplicates` compression is legal. Every action
ID must appear once before any recycling. Action membership was selected in
the frozen E8 geometry, but E8 is not an initialization or a teacher target.
At official initialization the action replay is a measured diagnostic, not a
false requirement that the stored E8 clean/action ranks remain identical.

## Historical E4 scientific kernel

The golden reference is `tasks/train_noise_e4_faithful_v1.py`. The hybrid
must retain:

- one shared encoder for clean query, action query, positives and negatives;
- clean groupwise rank loss, weight 1;
- action-view groupwise rank loss, weight 1, with no satisfied-action gate;
- symmetric live clean/action consistency, weight 0.25, with neither side detached;
- clean official-margin floor, weight 2;
- clean/reference-spectrum preservation against the frozen official cache, weight 5;
- historical E4's target budget of 16 action exposures per identity across four
  epochs; the complete supplier uses `max(unique_actions, 16)` and partitions
  that sequence across the four epochs, so identities with more than 16 unique
  actions are not truncated;
- action batch 4, with the E4 base cap of top-4 positive spectra and top-8
  negative molecules; the complete supplier may append the exact forced
  positive/negative spectrum needed to preserve a stored winning action
  boundary, while the groupwise-max loss and every coefficient remain E4;
- last transformer block plus official projection head trainable;
- backbone LR `2e-6`, head LR `1e-5`, weight decay `1e-4` on the head;
- full FP32, dropout disabled and global gradient clip 1.

The complete-panel supplier necessarily changes the set and coverage order of
action views. It does not change the mathematical E4 loss or its coefficients.

Before optimization, the targeted arm must pass an independent semantic audit:
the gradient of `action rank + 0.25 * live symmetric consistency` must have
positive median cosine with the separately constructed
`clean rank + 2 * official-margin floor` gradient in both the head and the
backbone. Preservation is excluded from both sides and remains a separate
protective component. The shuffled arm records the same diagnostic but is only
required to have finite, nonzero gradients; a negative control is not required
to be beneficial.

## Injector V1 boundary

The action loss is backpropagated first. Its complete gradient is captured.
The unchanged E4 safety loss is then backpropagated. The bridge performs the
single global clip and exactly one real AdamW step, then invokes the frozen
Injector V1 reconciliation. Required every-step gates are:

- optimizer-space action-attributable fraction `0.25`, maximum absolute error `2e-6`;
- protective component retention at least `0.90` in head and backbone;
- update norm ratio no more than `1.50`;
- first-moment reconstruction relative error no more than `1e-6`;
- no second optimizer step and no action tensor/loss mutation.

## Explicitly removed from the failed predecessor

The following are forbidden in this hybrid:

- E8 warm start;
- `0.16` static pre-clip loss scaling;
- satisfied-action rank-gradient gating;
- source/family corrective-weight redistribution;
- one-best action compression;
- teacher embeddings, teacher margins or distillation;
- PMT, P2b, P3, BioAware or ChemAware branches;
- candidate-boundary replacement losses or refreshed negatives.

## Causal and performance evaluation

The two allocated GPUs run one targeted arm and one source/family/recipe-matched
shuffled action-view control with the same seed. The stored historical E4
checkpoint for that seed is evaluated, not retrained.

On the complete corrected held-formula development graph, report for official,
stored E4, targeted hybrid and shuffled control:

- Recall@1/2/3/5/10/20;
- MRR and mean/median rank;
- macro-query AUROC/AUPRC;
- micro-candidate AUROC/AUPRC;
- positive-versus-best-negative margin and Top1-Top2 gaps;
- corrected, introduced and risk-net with lambda 2;
- near-subset metrics;
- formula-cluster paired confidence intervals;
- MassSpecGym 10-ppm pooled pairwise AUROC/AUPRC, including the `[M+H]+` subset.

The MassSpecGym pooled metric is not an exact reproduction of the paper's
NIST20 `0.85` result.

Success requires both preservation and gain: the targeted hybrid must not lose
the stored E4 signal, must beat the matched shuffled control with a positive
formula-cluster lower bound, and must achieve an observed Recall@1 gain of at
least 4 percentage points over official. The 5.33 pp action-space headroom is
not a promised encoder gain.

The report distinguishes non-regression from strict comprehensive success.
The strict full-panel gate additionally requires every registered retrieval,
near-subset, macro/micro, 10-ppm pooled, margin/gap and mean/median-rank metric
to be strictly better than both official DreaMS and the stored historical E4.
The only exception is a metric whose reference is already at its mathematical
boundary (for example median rank 1 or recall 1); it must remain at that
boundary without regression. All other ties are failures of the full-panel
gate.

## Server rule

All Python tests, preflight, training and evaluation execute inside the Slurm
allocation. The login node only runs `sbatch`. The job requests exactly two
GPUs and does not specify memory.

The official-initialization zero-change audit is score based, not a raw rank-
mismatch quota. A fresh FP32 forward must keep every held graph edge within
`5e-4` of the frozen official cache and mean embedding cosine at least
`0.9999`. For every rank mismatch, the audit identifies each negative-vs-
positive comparison whose Boolean contribution to strict rank changed. Every
such comparison must lie within `2 * 5e-4` of a tie in the old or new score
path; checking only the single best-negative margin is invalid because a lower
negative can change rank 2 to rank 3 while the best negative remains far above
the positive. Any changed comparison outside this boundary fails before the
first optimizer step. The number and first 64 details of tie flips are reported
rather than silently discarded.
