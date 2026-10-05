# Noise native Stage-5: action-mined negative residual

## Frozen starting point

- Warm start: Stage-1 targeted champion only.
- Model/runtime: repository-native `ContrastiveHead`, `ContrastiveSpectraDataset`, `SpectrumPreprocessor`, cosine triplet-margin loss and `torch.optim.Adam`.
- Frozen settings: learning rate `5e-6`, weight decay `0`, margin `0.1`, batch size `4`, FP32, backbone unfrozen from epoch 0, one epoch.
- The intervention is triplet membership only.

## Why this differs from failed Stage-4

Stage-4 continued training on action spectra as anchors and again pulled them toward positive spectra. It changed a relation that Stage-1 had already learned and reduced both internal and GNPS transfer performance. Stage-5 does not encode an action tensor during training.

For every Stage-1 clean query there is exactly one optimizer event:

```text
anchor   = measured clean query spectrum
positive = dynamic measured same-identity Stage-1 positive pool
negative = dynamic measured different-identity candidate rows
```

For action queries, registered targeted or matched-control action views only rank the negative candidate molecules. The selected model input remains a measured spectrum. Actions are first collapsed within source and then sources are averaged, so repeated actions do not increase a query's dose.

## Matched causal arms

Targeted and control have identical:

- query set and event count;
- clean anchor;
- positive memberships;
- negative pool cardinality;
- optimizer schedule and all training parameters.

They differ only in action-mined negative membership. Each chosen negative is required to be hinge-active against every dynamically sampled positive at the Stage-1 checkpoint. Queries with no paired hinge-active residual remain unchanged in both arms.

## Evaluation and promotion

The same job evaluates Stage-1, targeted and control on:

- the corrected MassSpecGym graph, including Recall@1/2/3/5/10/20, MRR, rank, macro/micro AUROC/AUPRC, margins, corrected/introduced/risk-net, near subset and formula-cluster paired CIs;
- frozen GNPS Gold/Silver identity-disjoint and formula-disjoint panels, including retrieval, macro/micro and pooled 10-ppm AUROC/AUPRC.

The candidate is stored as a promoted checkpoint only if the existing strict internal and GNPS gate passes, including the absolute 5 pp target. A failed candidate leaves metric ledgers but does not consume shared quota with another checkpoint.

## Entry point

```bash
sbatch tasks/run_noise_dreams_native_negative_residual_stage5_2gpu.sbatch
```

No login-node Python execution and no manual memory request are required.
