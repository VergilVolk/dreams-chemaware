# Noise native residual Stage-2 contract (2026-09-26)

## Frozen starting point

The immutable starting champion is
`noise_dreams_hard_positive_native_fold_0_run_2344820`.
Its targeted arm improves held Recall@1 by `+0.49637 pp` versus official DreaMS,
`+0.22910 pp` versus mature E8, and `+1.26548 pp` versus the registered
same-query control.  The causal formula-cluster interval is strictly positive.
This checkpoint, its evaluation, and its Stage-1 triplets are never overwritten.

The registered `5 pp` value is a promotion target.  It is not a forecast or a
claim about an unrun Stage-2 model.

## Scientific intervention

Stage-2 is not a new embedding architecture and does not replay the complete
Stage-1 action pool.  The frozen targeted champion re-encodes only outer-train
candidate-graph spectra and the Stage-1 targeted/control action views.

For each action query, at most one action survives when all conditions hold:

1. its targeted current margin remains below the native `0.1` hinge margin;
2. on the targeted action's exact current positive/negative boundary, targeted
   margin exceeds the registered same-query control by at least `5e-6`;
3. targeted margin exceeds the clean query on that same boundary by at least
   `5e-6`.

The selected action triplet is paired with that query's current clean
max-positive/max-negative triplet when the clean boundary is active.  A maximum
of 256 currently correct but margin-active action-free queries provide clean
protection and final-batch fillers.  One query can therefore contribute at most
one action event and one clean event; those events may never share an optimizer
batch.  No action multiplicity is converted into dose.

The matched control is used only in action selection and as the matched control
training arm.  Its residual is never negated, subtracted, or injected as a
gradient target.  Outer-held ranks and outcomes are unavailable to the builder.

## Native runtime and bounded continuation

- warm start: Stage-1 targeted final checkpoint;
- model/head: repository `ContrastiveHead`, including the 1024-D linear head;
- data: repository `ContrastiveSpectraDataset`, dynamically drawing one member
  from each registered positive and negative list (lists are singleton here);
- preprocessing: `SpectrumPreprocessor`, 100 peaks, precursor intensity 1.1;
- loss: native cosine triplet-margin hinge, margin 0.1;
- optimizer: native Adam, weight decay 0;
- learning rate: `1e-6` continuation rate;
- precision/batch/backbone: FP32, batch 4, backbone unfrozen at epoch 0;
- budget: one query-disjoint pass, at most 2,000 optimizer steps.

Targeted and control arms use the same warm start, exact query/positive/negative
relations, schedule, seed, and optimizer dose.  They differ only in targeted
versus registered same-query control action spectra.

## Promotion and retention

Stage-2 replaces the champion only if all registered gates pass:

- Recall@1 and MRR exceed Stage-1;
- Stage-2 versus Stage-1 Recall@1 formula-cluster CI is strictly positive;
- corrected minus twice introduced is positive;
- targeted beats control with a strictly positive formula-cluster CI and
  positive causal risk net;
- every registered full/near retrieval, AUC/AUPRC and margin direction is
  noninferior to Stage-1, official DreaMS, and mature E8;
- absolute Recall@1 gains versus both official and mature E8 reach `5 pp`;
- baseline formula CIs and risk nets are positive.

Failure of any gate preserves Stage-1 as the selected checkpoint.  It does not
turn an engineering completion into a scientific success.

## Submission

From the repository root on the server:

```bash
sbatch tasks/run_noise_dreams_native_residual_stage2_2gpu.sbatch
```

The job requests exactly two GPUs and no manual memory amount.  Construction,
tests, training, evaluation and summarization all run inside the allocation.
Pair-level `held_metric_evidence.npz` is deliberately omitted in this Stage-2
job because it duplicates the already computed metric inputs and exceeded the
shared filesystem quota in run 2344904.  The complete registered metric report
and paired 18,333-query ledger are still written. Full Lightning checkpoints
remain node-local; only a promoted targeted model is exported in compact
weights-only form.
