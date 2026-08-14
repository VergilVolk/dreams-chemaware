# E1 - DreaMS identity hard-triplet baseline

## Scope

E1 answers one question only: can the original self-supervised DreaMS embedding
be improved on difficult same-molecule retrieval using the contrastive procedure
described in the DreaMS paper?

E1 does not use MCES, chemical rules, masked-spectrum reconstruction, embedding
preservation, random negatives, or the old T1 near-isomer triplets.

## Frozen protocol

- Dataset: `MassSpecGym_MurckoHist_split.hdf5`
- Optimization fold: `train`
- Development evaluation fold: `val`
- Train/val 14-character InChIKey overlap: zero
- Adduct: `[M+H]+`
- Anchor-positive: same 14-character InChIKey, different spectrum and different
  tolerance-quantized peak hash
- Anchor-negative: different 14-character InChIKey and absolute precursor m/z
  difference no greater than 0.05 Da
- Backbone initialization: `ssl_model_server.pt`
- Embedding head: trainable linear 1024-to-1024 layer
- Trainable parameters: complete DreaMS backbone and embedding head
- Input: precursor token plus 100 highest-intensity fragment peaks
- Loss: `max(0, 0.1 - cos(anchor, positive) + cos(anchor, negative))`
- Optimizer: Adam, learning rate `5e-6`, no weight decay
- Formal seeds: 42, 43, 44

The validation fold is a development benchmark because the local HDF5 file has
no independent test fold. It must not be described as a final held-out test.

## Candidate-pool audit

| Pool | `[M+H]+` spectra | Eligible anchors | Median positives | Median hard negatives |
|---|---:|---:|---:|---:|
| Train | 156,568 | 145,512 | 22 | 128 |
| Validation | 38,669 | 34,171 | 24 | 51 |

An independent random audit of 10,000 sampled training triplets found zero
same/different-label errors and zero negatives beyond 0.05 Da.

## Build commands

```bash
python tasks/build_e1_triplet_pool.py --fold train
python tasks/build_e1_triplet_pool.py --fold val
```

## Formal training

Submit the three-seed array job:

```bash
sbatch run_e1_identity.sbatch
```

Each array task trains one seed and then evaluates its best checkpoint with the
same 10-ppm MassSpecGym validation protocol used by E0.

## Metrics

Training diagnostics:

- identity loss
- positive and negative cosine means
- cosine separation
- triplet accuracy
- margin satisfaction and violation rates
- pairwise embedding cosine mean and standard deviation
- mean per-dimension embedding standard deviation

Formal E1-versus-E0 metrics:

- pooled 10-ppm ROC-AUC (paper-compatible primary metric)
- query-macro ROC-AUC (project robustness co-primary metric)
- average precision
- positive-negative cosine separation
- Recall@1/5/10 and MRR as secondary metrics only

## Decision gate

E1 passes only if all of the following hold:

1. All three seeds improve pooled 10-ppm ROC-AUC directionally over E0.
2. The paired, query-clustered bootstrap confidence interval of the mean AUC
   improvement excludes zero.
3. Query-macro AUC does not materially decline.
4. Training observes both satisfied and violated triplets; loss is not identically
   zero and does not remain fully violated.
5. Pairwise cosine variance and per-dimension variance show no embedding collapse.

If E1 fails, sampling, preprocessing, checkpoint loading, and the training loop
must be corrected before E2 starts.
