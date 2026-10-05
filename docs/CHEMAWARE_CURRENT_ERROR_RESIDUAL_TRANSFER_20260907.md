# ChemAware current-error residual transfer

## Scientific correction

The failed routes treated three different questions as though they were one:

1. whether a relation is chemically meaningful;
2. whether that relation corrects an error currently made by official DreaMS;
3. whether a loss can write the corrective relation into one shared spectrum
   encoder without moving unrelated margins.

Generic chemical validity and generic action-to-embedding gradient reachability
do not answer questions 2 or 3.  The new route therefore fixes the current
DreaMS decision boundary before constructing the training target.

For query `q`, let `p` be the true molecule and let `n*(q)` be the negative
molecule with the largest frozen official-DreaMS score.  Let `c`, `c_struct`
and `c_peak` be candidate-centred negative ICEBERG distances for the correct,
candidate-swapped and peak-permuted arms.  A chemical action is active only if:

- official DreaMS is wrong on `q`;
- correct ICEBERG ranks `p` first over the complete candidate list;
- `c_p - c_n* > 0`;
- `c_p - c_n*` is strictly larger than the same fixed-boundary delta under
  both controls.

All other chemical weights are exactly zero.  The controls use the same active
query membership and the same training schedule; they are not separately
selected on their own outcome.

## Frozen action evidence

`tasks/build_chemaware_iceberg_corrective_residual_ledger.py` rebuilds the
strict ledger from the existing 700-query ICEBERG teacher:

- 619 official DreaMS errors;
- 348 errors rescued by correct ICEBERG;
- 272 actions retained by the current-boundary plus two-control gate;
- 244 formula clusters;
- 76 ICEBERG rescues rejected from chemical training;
- all inactive payloads are exact zero.

The frozen score-space utility audit over 2,048 development queries gives:

| Residual dose | Corrected | Introduced | Global Recall@1 delta |
|---:|---:|---:|---:|
| 0.25 | 56 | 0 | +2.7344 pp |
| 0.50 | 102 | 0 | +4.9805 pp |

At dose 0.50, candidate-swapped and peak-permuted controls give only +0.0977
pp and +0.1465 pp Recall@1.  The correct arm also has positive frozen MRR,
macro-AUC and micro-AUC deltas.  These numbers establish corrective action
headroom only; they are not a trained-embedding result and do not establish
external generalization.

## Injection objective

For a query with `M_q` candidates, define

`H_q = I - 11^T / M_q`,

where `s_q(theta)` is the current shared-embedding molecule-score vector,
`b_q` is the frozen official-DreaMS vector and `c_q` is the frozen chemical
residual.  The direct objective is

`L_chem(q) = mean_m Huber(H_q(s_q(theta)-b_q) - alpha*c_q)`.

Candidate centring removes the ranking-irrelevant common offset.  Among all
vectors representing the same pairwise score differences, the centred vector
has minimum L2 norm.  Huber limits outlier pressure without replacing the full
candidate residual by the previously failed single scalar margin.

The implementation additionally enforces:

- `alpha=0` is a literal absent chemical objective;
- one query has equal loss mass regardless of candidate-list length;
- formula clusters have equal action mass;
- every candidate molecule includes its frozen official-DreaMS top reference,
  so random reference sampling cannot change the audited boundary;
- one final Transformer block and the official projection head are updated;
- the same encoder processes queries and references;
- baseline-correct safety queries receive listwise continuation, official
  margin floors and embedding preservation;
- the final epoch is fixed, with no arm-specific checkpoint selection.

## Causal Phase A

The only submission entry is:

```bash
sbatch tasks/run_chemaware_iceberg_residual_shared_phase_a.sbatch
```

It requests exactly one GPU, specifies no manual memory, and sequentially runs
four matched arms: clean duplicate, correct residual at `alpha=0.50`,
candidate-swapped residual at `alpha=0.50`, and peak-permuted residual at
`alpha=0.50`.  The job rebuilds and audits its run-specific 272-action ledger
before constructing DreaMS, then evaluates one fixed fold-3 development cohort
on Recall@1/5/10/20/50, MRR, macro-AUC, micro-AUC, corrected/introduced errors,
embedding preservation and absolute margin motion.  Fold 4 remains reserved.

No individual arm is release eligible.  The automatic summary requires the
correct arm to beat the clean duplicate and both chemical controls with a
positive formula-cluster paired interval, while its absolute primary metrics
improve and its Recall@5/10/20/50 do not fall.  A Phase-A pass is still a
development result, not a globally untouched confirmation.
