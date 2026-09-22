# ChemAware reference-aligned native triplets

Date: 2026-09-22

## 1. Frozen starting point

The starting checkpoint is the confirmed ChemAware native-triplet model:

- role-3 Recall@1: `0.9046137895 -> 0.9227579057`;
- improvement: `+1.8144116122 pp`;
- formula-cluster bootstrap 95% CI: `[+0.8155, +2.8703] pp`;
- corrected/introduced at rank 1: `57/22`.

The new experiment does not replace this result and does not change the DreaMS
encoder, projection head, optimizer or cosine triplet loss.  It changes only
the triplet curriculum initialized from the frozen best checkpoint.

## 2. What was wrong with the successful pool

The stage-1 training pool contains 7,688 unique query-negative-molecule events,
4,032 queries and 2,518 formulas.  A molecule event stores all available
negative reference spectra, but native `ContrastiveSpectraDataset` samples only
one negative spectrum when that event is visited.

Using the complete local official-embedding cache, the old pool has:

| audit | old successful pool |
|---|---:|
| molecule events | 7,688 |
| events with any non-zero spectrum-pair hinge | 3,046 |
| events with zero hinge for every stored reference pair | 4,642 |
| mean positive hinge activation probability | 0.1944 |
| mean native hinge | 0.02720 |
| candidate-max margin violations | 1,206 |

The candidate graph itself contains 12,755 legal false-molecule boundaries.
The first three false molecules per query provide at most 8,663 distinct
candidate boundaries.  Therefore the problem is not solved by duplicating the
existing 7,688 events.  The missing quantity is the number of distinct,
currently active spectrum-level constraints.

There is also an objective mismatch.  Deployment scores a molecule by

`s(q,m) = max_{r in R(m)} cos(z_q, z_r)`,

whereas the old training event samples `r` uniformly from `R(m)`.  For a
candidate with many references, the reference responsible for the deployed
maximum can be sampled too rarely to shape the embedding.

## 3. Reference-aligned curriculum

For each query, the builder first computes checkpoint-specific cosine scores
for all positive and false-candidate reference spectra.  It then constructs:

1. the hardest false molecule under the current checkpoint;
2. a second arm-independent hard molecule;
3. one distinct ChemAware candidate that is close to the hard boundary and has
   at least 5% positive-reference hinge activation, or a hard fallback.

For each selected molecule, the two highest-scoring negative reference spectra
are considered explicitly.  For negative reference `r_n`, its activation is

`A(q,r_n) = mean_{r_p in P(q)} 1[0.1 + cos(q,r_n) - cos(q,r_p) > 0]`.

Events with `A>0` are retained, with at most four active spectrum events per
query.  Candidate diversity is filled before a second reference from an
already represented molecule.  If a query has no active event, exactly one
nearest-negative safety sentinel is retained.  Thus safe queries remain able
to constrain drift without consuming several guaranteed-zero events.

All positive references remain available to the native DreaMS random positive
sampler.  Only the negative curriculum is reference-aligned.

## 4. Local full-graph structural result

This is a construction audit using the official local embedding cache, not a
performance result and not the stage-1 best-checkpoint cache.  The server job
must recompute the pool from the `+1.8144 pp` checkpoint.

| audit | old pool | reference-aligned pool |
|---|---:|---:|
| total spectrum/native events | 7,688 molecule events | 7,357 spectrum events |
| active events | 3,046 molecule events | 5,212 spectrum events |
| mean activation probability | 0.1944 | 0.4651 |
| mean native hinge | 0.02720 | 0.07261 |
| zero-activation events | 4,642 | 2,145 safety sentinels only |
| unique queries | 4,032 | 4,032 |
| unique formulas | 2,518 | 2,518 |
| retained candidate boundaries | 7,688 | 5,592 |
| chemical spectrum events | not activation-filtered | 890 |

Relative to the old pool, active-event count rises by 71.1%, mean activation
probability rises by 2.39x, and mean native hinge rises by 2.67x, while all
queries and formulas remain represented.  These are curriculum-quality
statistics only; they do not establish a retrieval improvement.

Correct and null arms have the same 7,357-event training budget in the local
audit.  On role 2, correct-vs-null overall candidate Jaccard is about
`0.950--0.953`, because the two base hard slots are deliberately shared.  The
chemical-slot Jaccard is lower (`0.563--0.583`), with `65--69` correct-only and
the same number of null-only chemical candidates per null arm.  This is the
specific contrast to be tested after the correct arm passes the role-2 gate.

## 5. DreaMS parameter space and current decision

The repository's contrastive fine-tuning example explicitly uses `lr=5e-6`,
`batch_size=4`, one positive, one negative, margin `0.1`, Adam, no delayed
backbone unfreezing and 100 peaks.  The generic parser default for margin is
`0.2`, but the concrete contrastive recipe overrides it to `0.1`.

There is real optimization space:

| knob | plausible test | current decision |
|---|---|---|
| training steps | 500/1,000/1,500/2,000 | test now; role 2 selects |
| learning rate | `2e-6`, `5e-6`, `1e-5` | keep `5e-6` until triplet effect is known |
| margin | `0.05`, `0.1`, `0.2` | keep `0.1`; changing it confounds triplet quality |
| negatives sampled per event | 1 versus 2 | defer; current explicit reference events use 1 |
| batch size | 4 versus 8/16 | keep 4; batch changes optimizer dose |
| weight decay | 0 versus small positive | keep 0 |
| backbone freezing/PEFT | several variants | reject for this experiment |
| dynamic remine frequency | one-shot versus iterative | defer until one-shot passes |

The highest-value safe sequence is therefore:

1. run the reference-aligned pool with all successful DreaMS parameters fixed;
2. select only the optimizer step on role 2;
3. if and only if it beats stage 1 under all safety metrics, run three matched
   chemistry-null curricula at the identical step budget;
4. evaluate role 3 once;
5. only after a positive result run a small learning-rate/margin ablation.

## 6. Execution and gates

The only server command is:

```bash
sbatch tasks/run_chemaware_reference_aligned_native.sbatch
```

The job requests exactly one GPU and does not request memory manually.  It
rebuilds the embedding cache from the stage-1 best checkpoint, constructs the
pool, trains 500/1,000/1,500/2,000-step checkpoints, selects on role 2, and
runs matched nulls plus role 3 only after the correct arm advances.

The selected checkpoint must beat the frozen stage-1 model in Recall@1, have
`corrected - 2 * introduced > 0`, and not reduce MRR, Recall@3, micro-AUC or
macro-AUC.  Failure keeps the `+1.8144 pp` checkpoint unchanged.

## 7. Claim boundary

Allowed before the GPU result:

> The reference-aligned builder raises the density and expected activity of
> native DreaMS triplet constraints in a complete local structural audit while
> preserving query/formula coverage and matched chemistry-null budgets.

Not allowed before the GPU result:

- the new curriculum improves retrieval;
- it exceeds `+1.8144 pp`;
- correct chemistry independently causes the gain;
- the role-3 result is an untouched outer result.
