# Noise 5-pp target and AUC co-primary corrected ledger

Date: 2026-09-27

## Decision

The project has already established a real `1--2 pp`-scale effect, but that
phrase has referred to different comparators.  These quantities must never be
collapsed into one headline number:

| Evidence | Recall@1 delta | Correct interpretation |
|---|---:|---|
| Noise native Stage-1 targeted vs registered matched control | `+1.26548 pp` | causal action-content contrast; formula-cluster CI `[+0.87586,+1.65425] pp` |
| Noise native Stage-1 targeted vs official DreaMS | `+0.49637 pp` | absolute trained shared-encoder gain |
| Noise native Stage-2 vs official DreaMS | approximately `+0.6000 pp` | strongest absolute Noise point estimate, but only approximately `+0.1036 pp` beyond Stage-1 and not a promoted champion |
| Historical E4-A on the old cohort | mean `+0.6362 pp` | multi-fold mechanism evidence; not a current corrected-graph claim |
| ChemAware native triplet best vs its official baseline | `+1.81441 pp` | a separate shared-encoder experiment whose runtime lesson is reusable; not a Noise result |

Therefore it is correct to say that Noise has demonstrated a strictly positive
`+1.265 pp` targeted-over-control signal.  It is not correct to say that the
current Noise checkpoint is already `+1.265 pp` or `+1.8 pp` above official
DreaMS.  The current absolute Noise scale is approximately `+0.5--0.6 pp`.

### Exact evaluation population behind `+1.26548 pp`

The value is not a training-set score and not a MoNA score. It was measured on
the corrected MassSpecGym retrieval graph
`noise_corrected_candidate_graph_v1_20260906/candidate_graph.npz`, whose spectra
come from `MassSpecGym_MurckoHist_split.hdf5`. The evaluation uses outer formula
fold `0` from seed `20260825`: `18,333` held queries in `1,249` formula
clusters. Targeted and same-query control use the identical query rows,
candidate molecules, positive/negative labels, schedule, seed and optimizer
dose. Targeted corrected `455` control errors and introduced `223` errors, so
`(455 - 223) / 18,333 * 100 = 1.26548 pp`. This is a matched causal contrast on
the held MassSpecGym fold, not an absolute gain over official DreaMS.

The `5 pp` objective means an absolute improvement of the deployable shared
clean-spectrum encoder against the frozen official and mature-E8 comparators.
On 18,333 held queries, `5 pp` is about 917 net Top-1 decisions.  The present
approximately `+0.6000 pp` Noise point estimate is about 110 net decisions, so
the remaining absolute gap is approximately `4.40 pp`, or about 807 additional
net decisions.  This does not erase the `+1.265 pp` causal evidence; it shows
that useful action content is currently accompanied by a large common
continuation/control loss and insufficient coverage.

## AUC is a co-primary endpoint

Recall@1 and AUC answer different questions and both must be reported.  A
checkpoint cannot be promoted on Recall@1 alone.  Every frozen evaluation must
publish numerical values and deltas for:

1. full-graph macro-query AUROC and AUPRC;
2. near-subset macro-query AUROC and AUPRC;
3. micro-candidate AUROC and AUPRC;
4. MassSpecGym 10-ppm pooled pairwise AUROC and AUPRC;
5. MassSpecGym `[M+H]+` 10-ppm pooled pairwise AUROC and AUPRC.

The DreaMS paper's approximately `0.85` is pooled ROC-AUC on its NIST20 pair
ledger.  The MassSpecGym 10-ppm pooled pairwise AUROC uses the same mathematical
kind of statistic but is not an exact reproduction of the NIST20 value.  The
two labels must remain separate.

The Stage-1 arm evaluation reports contain the complete numerical candidate,
control, official and mature-E8 metric blocks, but its final summary previously
retained only Boolean AUC direction checks.  Stage-2/Stage-3 summaries had the
same visibility defect.  Both summary paths now emit
`auc_metric_comparisons`, containing candidate, reference, absolute delta,
percentage-point delta and an improvement flag for every AUC and AUPRC panel.
Boolean direction checks remain guards but are no longer the only visible AUC
result.  This reporting repair does not modify or rerun a checkpoint.

## Promotion contract toward 5 pp

The next Noise method must continue from the Stage-1 champion and preserve its
demonstrated targeted-over-control advantage.  It must not restart from
official DreaMS or a weaker invented injector.  The registered intervention is
restricted to triplet contents and their outcome-blind ordering.  The DreaMS
preprocessor, shared encoder, native contrastive head, cosine triplet-margin
loss, Adam optimizer and registered runtime parameters remain unchanged.

The promotion decision is conjunctive:

- absolute Recall@1 gain approaches or reaches `+5 pp` against both frozen
  baselines;
- Recall@1 and MRR formula-cluster paired intervals are strictly positive;
- macro-query, micro-candidate and both pooled pairwise AUROC values improve
  numerically, with AUPRC reported beside each AUROC;
- positive-vs-best-negative margin and Top1--Top2 gap do not regress;
- corrected minus twice introduced is positive overall and in the near subset;
- Recall@2/3/5/10/20 and mean/median rank satisfy their registered directions;
- targeted beats a same-query, same-dose matched control, so common native
  continuation cannot be mistaken for Noise-specific gain.

The next data intervention should broaden identity-preserving noisy-positive
coverage while retaining hard negatives and equal query dose.  Existing action
tensors cannot all be relabelled as positives: prior replay showed that the
clean-to-action relation was frequently harmful.  A broad positive curriculum
must first establish identity retention and targeted-over-control value at the
tier-by-source level, then train through the unchanged native DreaMS path.

## Claim boundary

Current confirmed statements are:

- Noise action semantics have a significant `+1.26548 pp` causal advantage
  over their registered matched control in Stage-1.
- The best reliable absolute Noise shared-encoder gain is approximately
  `+0.5--0.6 pp` versus official DreaMS on the corrected graph.
- ChemAware separately demonstrated `+1.81441 pp`; this motivates the native
  triplet runtime and short-course curriculum but cannot be counted as Noise
  performance.
- A `5 pp` Noise checkpoint has not yet been obtained and must not be inferred
  from action-space oracle/headroom numbers.
