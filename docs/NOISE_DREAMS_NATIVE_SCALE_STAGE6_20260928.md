# Noise native Stage-6: incremental two-sided hard-triplet scale

## Decision

Stage-5 is closed as a negative-only residual. It changed negative membership
for only 474 queries, did not beat its matched control, and reduced the
MassSpecGym `[M+H]+` pooled 10-ppm AUROC by 6.624 pp relative to Stage-1. The
retained checkpoint remains the Stage-1 targeted champion.

Stage-6 tests a different and substantially larger hypothesis: the useful
Stage-1 relation should be expanded into a full noisy triplet. The anchor and
positive are independently perturbed spectra of the same identity; the
negative is a separately perturbed spectrum of a different identity and is
retained only when that nuisance view is measurably more confusable. The
limitation was not a shortage of minor negative choices; it was that genuinely
difficult, identity-preserving relations covered only a small fraction of the
outer-train library.

## Triplet corpus

Stage-6 is built only from outer-train formulas. The Stage-1 formula-disjoint
validation pool is reused bitwise.

The train pool contains three streams in one native one-pass corpus:

1. Every effective Stage-1 action triplet and action tensor is retained as an
   exact prefix, including all seven registered sources and clean-boundary
   fallbacks.
2. Every role-safe outer-train query receives one measured clean-preservation
   event with at most eight same-identity positives above the registered
   identity floor and at most three hard, formula-safe measured negatives.
3. Every qualifying outer-train query receives exactly two broad two-sided
   hard events using different noisy anchors and different negative molecules.
   Three deterministic acquisition-noise views are proposed: 10/20/30% peak
   dropout, 15/30/40% intensity jitter, and 2/5/8 tiny candidate-independent
   background peaks. The precursor and retained fragment m/z values never
   move. Every saved view must round-trip exactly through the unmodified DreaMS
   `SpectrumPreprocessor`.

All nuisance views are source-local and role-blind: their generator receives
only the source spectrum row, severity, replicate and global seed. It never
receives `anchor`/`positive`/`negative`, the query index, candidate rank, label,
action source or correction outcome.
The frozen Stage-1 checkpoint is then used only to mine outer-train relations.
A saved event must satisfy all of the following:

- noisy anchor and noisy positive each reduce same-identity similarity by at
  least 0.01 relative to the clean pair;
- noisy anchor, positive and negative each remain at least as self-consistent
  as the empirical outer-train same-identity q05 floor and retain the source
  base peak;
- the noisy negative raises false-match similarity by at least 0.01 for both
  the noisy and clean anchor;
- both targeted and anchor-ablation triplets remain hinge-active below margin
  0.1 and above the empirical outer-train clean-margin q01 floor.

The checkpoint is not used as a teacher, target, or distillation loss.
Positive and negative source rows are independently permuted once per query
before assignment to noise severity. Consequently easy noise is not
systematically coupled to the hardest source pair, nor hard noise to the
easiest source pair. The post-selection easy/medium/hard labels are descriptive
realized-margin strata, not independent scientific gates.

Every newly added positive and negative role is also formula-audited. Its
source formula must be outside outer fold 0 and outside the Stage-1 validation
formula set. The inherited Stage-1 checkpoint and exact Stage-1 prefix predate
this role-level audit, so Stage-6 reports that legacy scope explicitly and does
not relabel the complete continuation as a newly strict formula-blind training
run. The paired incremental contrast remains valid because both arms inherit
the same Stage-1 state and prefix.

The registered minimum is 20,000 **new broad two-sided hard exposures** from at
least 10,000 independent queries. It is not a total-row count and not a count
of possible pool combinations. Every qualifying query contributes exactly two
different views using different negative identities. Clean preservation and
Stage-1 replay do not count. Difficulty strata and diversity are reported, not
used as arbitrary rejection gates.

## Causal control

Targeted and control start from the identical Stage-1 targeted champion and
its native Adam state. Both replay the identical Stage-1 targeted triplets.
They also have the same query, noisy positive, noisy negative, event count,
schedule and seed. Their only difference in the new broad stream is:

```text
targeted = noisy anchor + noisy same-ID positive + noisy hard negative
control  = clean anchor + the exact same noisy positive + noisy negative
```

The original Stage-1 targeted-versus-registered-control result remains prior
evidence and is not mixed into this new targeted-versus-control contrast.

## Native runtime

- initialization: exact Stage-1 targeted champion tensors and native Adam
  state;
- dataset: native `ContrastiveSpectraDataset`, one positive and one negative
  draw; every new broad event has exactly one audited noisy positive and one
  audited noisy negative, while the preserved Stage-1/clean pools retain their
  original dynamic memberships;
- model: native `ContrastiveHead` and official 1024-dimensional linear head;
- preprocessor: 100 peaks, precursor intensity 1.1, FP32;
- loss: native cosine triplet-margin hinge, margin 0.1;
- optimizer: native Adam, learning rate `5e-6`, weight decay 0;
- batch size 4, backbone unfrozen at epoch 0, exactly one query-disjoint pass;
- no teacher target, custom loss, adapter, optimizer projection, replay, or
  manual parameter weighting.

The trainable model does not restart from official DreaMS. Both arms continue
the exact Stage-1 champion and retain its complete targeted triplet stream.
Full clean preservation, bounded positive similarity, a semihard margin floor,
and the complete numerical AUC/AUPRC gate protect against the global-geometry
forgetting exposed by Stage-5.

The saved `train_pool.npz` and `action_spectra.npz` are an appendable library:
the complete Stage-1 library is an exact prefix and Stage-6 records explicit
anchor/positive/negative role codes. A later successful stage must append to
this frozen prefix rather than reconstructing earlier triplets.

## Evidence basis

DreaMS itself uses random peak masking in pretraining and end-to-end triplet
fine-tuning for spectral similarity. MS2DeepScore used more than 100,000
spectra and acquisition-like peak removal, intensity jitter, and low-intensity
peak addition. These motivate the nuisance operators and data scale; they do
not establish a result for Stage-6.

- DreaMS: <https://pmc.ncbi.nlm.nih.gov/articles/PMC13090125/>
- MS2DeepScore: <https://pmc.ncbi.nlm.nih.gov/articles/PMC8556919/>

## Evaluation and entry point

The same job evaluates Stage-1, targeted, and control on the complete corrected
MassSpecGym graph and both GNPS Gold/Silver identity- and formula-disjoint
panels. It reports every registered retrieval metric, macro/micro AUROC/AUPRC,
pooled 10-ppm AUROC/AUPRC, margins, risk-net, near subset, and formula-cluster
paired confidence intervals. A checkpoint is copied to the run directory only
if the internal and GNPS promotion gate passes.

```bash
sbatch tasks/run_noise_dreams_native_scale_stage6_2gpu.sbatch
```

The 5 pp endpoint remains a promotion threshold, not a guaranteed result.
