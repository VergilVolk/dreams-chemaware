# Noise native multi-difficulty Stage-3

## Purpose

Stage-2 used a noisy action anchor and a current hard negative, but selected the
most similar same-identity positive. It was therefore hard on the negative side
and comparatively easy on the positive side. Its matched targeted-control gain
was only `+0.02182 pp`; that result does not support promotion.

Stage-3 changes only triplet membership and the action anchor. The DreaMS model,
preprocessor, loss, optimizer and checkpoint format remain native.

## Registered triplet semantics

For each non-outer training query and each native-representable registered
Noise action:

- anchor: the targeted noisy spectrum of the query;
- positive candidates: independent measured spectra with the same 14-character
  identity, excluding the clean query row;
- negative candidates: measured spectra from every different-identity candidate
  in the query graph;
- hard negative: the different-identity spectrum with maximum current cosine
  similarity to the noisy anchor;
- easy positive: the same-identity spectrum with maximum current similarity;
- medium positive: the median-similarity same-identity spectrum;
- hard positive: the same-identity spectrum with minimum current similarity.

The Stage-1 targeted champion computes all similarities. No held outcome,
corrected/introduced label, or outer-fold formula is used in construction.

At most four distinct action views are selected per query: one easy, one
medium, and up to two hard. Every selected triplet must be active under the
native margin `0.1`. The artifact must contain at least 10,000 action triplets,
including at least 1,000 easy, 1,000 medium and 4,000 hard triplets, and must
retain every registered source family. Hard-positive triplets must comprise at
least 35% of the action ledger. A dynamic
clean-anchor event is retained
for every selected query, with all same-identity positives and its 16 currently
hardest different-identity negatives. Action-free clean sentinels are used only
for final-batch completion.

## Native training contract

- initialization: exact evaluated Stage-1 targeted checkpoint;
- `dreams.models.heads.heads.ContrastiveHead` and its 1024-dimensional linear
  projection head;
- `ContrastiveSpectraDataset`; registered action events contain one exact
  tier-specific positive/negative relation, while clean events dynamically
  sample one positive and one negative from their multi-member pools;
- `SpectrumPreprocessor`, 100 peaks, precursor intensity 1.1;
- cosine triplet-margin loss, margin 0.1;
- native `torch.optim.Adam`, learning rate `5e-6`, weight decay 0;
- batch size 4, FP32, backbone unfrozen at epoch 0;
- one query-disjoint pass; every semantic action appears exactly once;
- targeted and matched-control arms run on two allocated GPUs.

This is deliberately a one-pass continuation, not another 301-epoch run. The
larger and harder ledger supplies dose through distinct semantic triplets rather
than repeated epochs.

## Evaluation and decision

Both arms are evaluated on the same frozen corrected held graph. The existing
evaluator reports Recall@1/2/3/5/10/20, MRR, ranks, macro/micro AUROC and AUPRC,
margins, Top1-Top2 gap, corrected/introduced/risk-net, near subsets and
formula-cluster paired confidence intervals.

The Stage-1 checkpoint is retained unless every registered promotion gate
passes, including positive Stage-3-vs-Stage-1 and targeted-vs-control evidence,
near safety, all-metric noninferiority, and at least `+5 pp` versus both official
DreaMS and mature E8. Five pp is a hard decision threshold, not a promised
result.

## Submission

```bash
sbatch tasks/run_noise_dreams_native_multidifficulty_stage3_2gpu.sbatch
```

The script requests exactly two GPUs and does not manually request memory. Full
Lightning checkpoints remain node-local; only a promoted targeted slim
checkpoint is copied to shared storage.
