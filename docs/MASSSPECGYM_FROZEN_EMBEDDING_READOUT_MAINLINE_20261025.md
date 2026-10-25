# Unified candidate ranker mainline

## Question

Can one candidate-level ranker use the strongest existing spectral, robust-
representation and chemical evidence without either reducing every encoder to
one cosine or counting correlated rerankers more than once?

This is the main integrated algorithm experiment, not a separate cosine study.
It trains one capacity-matched control and two complete-integration
realizations on the same complete MassSpecGym corrected graph:

1. a capacity-controlled scalar-evidence **non-promotable control**;
2. the same evidence plus a learned shared projection and cosine for every
   frozen encoder;
3. the same evidence plus a symmetric full-embedding pair head.

Every realization receives WSE, Noise V1, the frozen P2b composite and
ChemAware V2 continuous correct-versus-null evidence. The two representation
realizations additionally read the complete frozen embeddings from Noise V1,
ChemAware Stage-1, ChemAware Phase-A and
the Noise-initialized FullMSG ChemAware continuation. WSE and the single frozen
P2b composite enter once rather than as both composites and subchannels.
ChemAware V2 enters with an availability flag; unavailable candidates retain
an exact zero residual through normalization. The ordinary continuation is an
attribution control, not a production expert. Historical MSG/RRF scores are
not added again: FullMSG is represented by its independently trained encoder,
while retired RRF remains excluded.

## Leakage boundary

The frozen encoders trained on stable formula folds 1--4.  The readout heads
therefore also train only on those 65,286 queries and are selected once on the
complete 18,333-query fold 0.  Predictions on folds 1--4 are deliberately not
called OOF, because the encoders saw those formula families.  After selecting
the better of the two full-embedding arms by held Recall@1 then MRR, that arm
is refit on the complete development graph for subsequent fixed GNPS
evaluation. The scalar arm cannot be selected as the final model. It is kept
only to test whether reading the complete embeddings adds value beyond the
same scalar evidence and comparable head capacity. If neither full arm beats
it, the result is reported as a failed representation-integration increment;
the scalar control is not relabeled as a unified multi-encoder algorithm.

GNPS and application-response outcomes are not accepted as inputs by model
fitting or arm selection.  The experiment reports Recall@1/2/3/5/10, MRR, ranks, near
performance, molecule-level micro AUROC/AUPRC, corrected/introduced exchanges,
and formula-cluster paired intervals on the 18,333-query held split.

## Computational choices

All candidate molecules remain in held evaluation.  Training uses the unique
positive plus at most six label-known hard negatives nominated by independent
evidence families.  Each encoder chooses its reference spectrum by its own
frozen raw cosine before head fitting.  Thus reference multiplicity does not
become accidental training dose, while each representation retains its own
molecule-max retrieval semantics.

The single server entry point is:

```bash
sbatch tasks/run_massspecgym_embedding_readouts_2gpu.sbatch
```

It requires exactly Noise V1, ChemAware Stage-1, ChemAware Phase-A and the
Noise-initialized FullMSG continuation; a missing or substituted encoder is a
hard input error. It reuses the existing six-million-edge WSE/P2b evidence and aligned V2 cache
when present. If only the aligned V2 cache is absent, the same job regenerates
that evidence from the existing policy and token assets; it does not ask for a
second submission. It encodes four frozen checkpoints with two GPUs,
trains the registered control and two integration realizations, and emits one
fixed full-integration candidate. Only after that model is saved and refitted on the full development
graph does the same allocation encode and score the fixed GNPS panels. The
GNPS evaluator has no optimizer, threshold, router or selection branch, so its
outputs cannot alter the chosen algorithm.

The selected model is accepted directly by
`tasks/score_crossstudy_identity_multiencoder.py --unified-model ...`. That
application path reproduces frozen P2b, keeps zero- and one-candidate queries,
and returns the biology handoff columns `candidate_pair_id`, `method_name`,
`method_score`, and `evidence_available`. Candidate structures are absent on
that current cross-study axis, so V2 is explicitly unavailable there rather
than replaced by a fabricated chemical value. A legal BioAware event scorer
can therefore be joined on the same candidate-pair identifier without changing
the spectral candidate universe or using response outcomes during ranking.
