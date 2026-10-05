# Noise E4 complete best-action reuse through Injector V1

Date: 2026-09-13
Status: implemented; local integration gates passed; server result pending

## Decision

This run does not introduce a new action selector, teacher, distillation loss,
margin-transfer objective or training route.  It composes two already existing
and independently audited assets:

1. the E4 symmetric/shared direct fine-tuning implementation in
   `train_noise_final_e4a_direct_augmentation.py`; and
2. the frozen optimizer-boundary `ActionInjectorV1` implementation.

V11's one-maximum-margin-action-per-query supplier is forbidden.  The run uses
the existing `multi_action_balanced` panel and therefore retains every strict
Top-1 corrective view above the frozen `5e-6` numerical floor.

## Reused action assets

The immutable job-2332784 ledger supplies the same seven mature sources:

- `N_mature`;
- `P_guided_original`;
- `E10B`;
- `E11`;
- `E12B`;
- `A4_exact`;
- `V4_gradient_path`.

E4 is the direct-training kernel and E8 is the initialization checkpoint, not
two omitted spectrum-action tables.  Historical E13 reused the E12B action
tensors, so retaining `E12B` retains that action content without duplicating it
under a second label.

The compute-node preflight requires the complete strict panel, requires more
rows than the one-best panel, requires repeated query indices, requires all
seven sources after selection, and verifies that no fold-0 formula is consumed.
Every action is physically exposed before recycling.  Total effective dose is
the historical 16 per identity, balanced across available source/family blocks.
Routing scores and stored margins are not continuous loss weights or embedding
targets.

## Reused E4 kernel

The action and schedule implementation remains:

```text
E8 initialization
4 epochs x 4 views per identity
4 physical actions per microbatch
1.00 clean shared rank
+ 1.00 real action-view shared rank
+ 0.25 symmetric live clean/action consistency
+ 2.00 clean margin floor
+ 5.00 clean/reference preservation
shared trainable candidate references
last Transformer block + official projection head
backbone LR 2e-6; head LR 1e-5; fp32
```

The existing multi-action repair remains intact: `0.16` static pre-clip scale,
action-rank pressure gated after the live action clears margin `0.05`, complete
first exposure, and identity/source/family-balanced effective dose.  Neither
the action tensors nor the E4 loss are reconstructed in a new trainer.

## Injector boundary

After the unchanged E4 action backward, the bridge snapshots the complete E4
gradient.  The unchanged E4 safety backward is then accumulated.  The bridge
passes the following same-state, same-clip ledgers to Injector V1:

- combined: E4 action plus E4 safety;
- noncorrective: E4 safety;
- protective: the same E4 safety reference.

Injector V1 applies its frozen `0.25` optimizer-space action fraction, hard
`0.90` protective-component floor and `1.50` update-norm cap independently to
head and backbone.  It executes one real AdamW step, materializes the composed
update, reconciles only the first moment and preserves the actual combined-
gradient second moment and step number.  Every optimizer step is audited; no
sampled tail is allowed.

This optimizer boundary changes effective delivery but does not change action
selection, action spectra, E4 loss terms, E8 initialization or inference.

## Causal server run

Two GPUs run the complete four-epoch panels concurrently:

- GPU 0: exact query-matched mature actions;
- GPU 1: source/family/exact-recipe-matched shuffled actions drawn from the
  same selected union.

Both use the identical E4 schedule, safety stream, initial E8 checkpoint and
Injector V1 thresholds.  The job runs only on allocated compute resources,
requests exactly two GPUs, makes no manual memory request, validates a frozen
source snapshot, and publishes atomically.

Submit from the server repository root:

```bash
sbatch tasks/run_noise_e4_best_actions_injector_v1_2gpu.sbatch
```

## Evaluation and claim boundary

The existing full E4 evaluator reports Recall@1/2/3/5/10/20, MRR, mean/median
rank, macro-query AUROC/AUPRC, micro-candidate AUROC/AUPRC, margins, Top1-Top2
gaps, corrected/introduced/risk-net, near-subset metrics, formula-cluster paired
CIs, and all-adduct plus `[M+H]+` 10-ppm pooled pairwise AUROC/AUPRC.  The pooled
metric is not an exact NIST20-paper replication.

The historical 4.93--5.33 pp action-space headroom is not predeclared as an
encoder result.  Promotion requires at least +4 pp clean-input Recall@1 versus
the exact initial E8 checkpoint, targeted superiority over matched shuffle,
strictly positive multiplicity-corrected formula-cluster intervals, positive
risk-net and no registered metric regression.
