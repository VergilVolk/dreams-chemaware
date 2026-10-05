# Noise native triplet and apparent 2-pp ceiling audit

Date: 2026-09-27

## 1. Decision first

The current Stage-3 run must finish.  Nothing in this audit justifies killing or
mutating it.  Its frozen question is useful: does a one-pass native DreaMS
continuation benefit from a much larger set of severity-stratified, valid
same-identity hard positives?

However, Stage-3 is **not** already a complete answer to the remaining transfer
problem.  It repairs positive difficulty and action-event scale, but it retains
three important bottlenecks of the native DreaMS objective:

1. one sampled positive and one sampled negative define each update;
2. every active cosine hinge triplet gives equal-amplitude pressure to the
   positive and negative similarity, irrespective of which side is actually
   deficient;
3. a pairwise spectrum triplet is trained while the endpoint is a molecule-max,
   full-candidate ranking decision.

The apparent `~2 pp` scale is not a universal optimizer ceiling.  Three numbers
that have repeatedly been conflated must stay separate:

| Quantity | Value | Meaning |
|---|---:|---|
| reliable Noise native Stage-1 held gain vs official | `+0.49637 pp` | trained shared clean-spectrum encoder |
| Noise Stage-2 held gain vs official | about `+0.6000 pp` | trained shared encoder; only about `+0.1036 pp` beyond Stage-1 and that incremental CI did not establish a new champion |
| old mature-N corrected-full-graph coverage ceiling | `+1.735 pp` | if every covered official error were corrected with zero introductions; not a checkpoint result |
| ChemAware development result | about `+1.8 pp` | different supervision and experiment; its runtime is reusable, not its performance number |
| historical Noise action-rich conditional/oracle gains | `3.35--5.33 pp` | selected action-space capacity, not deployable clean-encoder performance |

The scientific conclusion is therefore narrower and more actionable: current
Noise supervision is sparse and conditional, and the native triplet objective
uses that supervision inefficiently.  More epochs or merely more extremely hard
triplets cannot by themselves bridge this gap.

## 2. Evidence hierarchy

### 2.1 Results that qualify as shared-encoder evidence

- Historical E4-A on the old cohort: five folds by three seeds, mean
  Recall@1 `+0.6362 pp`, MRR `+0.4246 pp`, near Recall@1 `+0.5300 pp`.
  It is useful mechanism evidence but not a current corrected-graph claim.
- Native Stage-1 run `2344820`: Recall@1 `+0.49637 pp` versus official,
  `+0.22910 pp` versus mature E8, and `+1.26548 pp` versus the registered
  same-query control; its formula-cluster causal interval was positive.
- Native Stage-2: approximately `+0.6000 pp` versus official and `+0.3327 pp`
  versus mature E8, but only approximately `+0.1036 pp` beyond Stage-1.  On
  18,333 held queries this is about 19 additional net decisions, so it is not a
  large independent advance.

### 2.2 Results that do not qualify as shared-encoder gains

- `3.346--5.33 pp` action-space numbers expose the answer through selected
  action views and/or outcome-aware choices.
- `+1.735 pp` is a query-specific zero-risk correction ceiling from the old
  mature-N coverage (`1,451 / 5,957` official errors on 83,619 corrected-graph
  queries).
- The local frozen shared-map `+3.808 pp` is a representation/selection proxy,
  not raw shared encoder weights and was not stable across folds/seeds.
- ChemAware's development gain is not evidence for a Noise triplet ledger.

## 3. Stage-3 factual audit

The attached Stage-3 builder output is internally coherent:

| Item | Observed |
|---|---:|
| action triplets | `10,146` |
| action queries | `3,021` |
| easy / medium / hard | `2,050 / 2,387 / 5,709` |
| hard fraction | `56.268%` |
| clean dynamic events | `3,477` |
| protection events | `256` |
| total events | `13,879` |
| action event fraction | `73.103%` |
| action-query coverage of 65,286 outer-train queries | `4.627%` |

Median initial margins (`s_positive - s_negative`) are `+0.0447` for easy,
`-0.0408` for medium and `-0.1793` for hard.  All are active under the native
margin `0.1`.  The hard tier is therefore genuinely hard in the frozen Stage-1
geometry; it is not a label attached to easy pairs.

The source distribution is nevertheless concentrated: E11 contributes
`30.85%`, E12B `24.10%`, and E10B `20.70%`; together these three supply
`75.65%` of action events.  Event count is not the same as independent chemical
coverage because several views can share a query, identity, formula, positive
row or negative molecule.

### 3.1 A small but real construction discrepancy

`clean_by_query` is populated before action selection for every action-capable
query, whereas the action-query set is defined after selection.  Consequently
there are `3,477 - 3,021 = 456` clean-only query events beyond the selected
action queries.  They are common to targeted and control arms and consume only
`3.286%` of all events, so they do not invalidate the causal comparison.  They
do slightly dilute action dose and contradict a literal reading of “one clean
event per selected action query.”  The completed Stage-3 run should be reported
with the observed dose, not retroactively changed.

### 3.2 Warm start is not the problem

Although the builder prints `initialization_kind = official_embedding`, this is
the package format returned by `load_base_model`, not evidence that training
restarted from official DreaMS.  The Stage-3 SBATCH passes the Stage-1 targeted
checkpoint as `--warm-start-checkpoint`; the residual trainer reconstructs the
native `ContrastiveHead` from it and checks every backbone and head tensor for
exact equality before optimization.  This must still be confirmed in each arm's
training `report.json`, but the source path is scientifically correct.

## 4. Why discrimination remains insufficient

### 4.1 Supervision covers too little of the task

Only `3,021 / 65,286 = 4.63%` of outer-train queries receive Stage-3 action
supervision.  The older corrected-full-graph audit is even more revealing: the
old mature N bank touched only `24.36%` of official errors.  If learned changes
remain query-local, `+1.735 pp` is the mathematical maximum even under perfect,
zero-risk correction.  Surpassing it requires genuine formula/identity-level
generalization to queries not carrying an action—not merely injecting covered
actions more strongly.

On the 18,333-query held fold, a `+5 pp` gain means about **917 net Top-1
decisions**.  That is roughly `74.6%` of official errors and `81.6%` of mature
E8 residual errors.  Stage-1 corresponds to about 91 net decisions and Stage-2
to about 110.  Thus the requested endpoint is about an order of magnitude above
the present trained effect; it cannot be honestly guaranteed from the current
coverage.

### 4.2 Ten thousand rows are not ten thousand independent constraints

Stage-3 caps four action events per query, so its 10,146 action rows reduce to
3,021 query units.  Rows can also repeat formula, identity, positive reference,
negative molecule and source mechanism.  The current report does not publish:

- unique identities and formulas;
- unique `(query, positive molecule, negative molecule)` boundaries;
- multiplicity distribution at identity/formula/negative-molecule level;
- source-by-formula effective sample size;
- how often easy/medium/hard tiers use the same positive or negative row.

Without these, `10,146` proves computational dose, not broad representation
coverage.  These counts are required from the frozen artifact before any claim
that data scale is now sufficient.

### 4.3 The native hinge cannot distinguish positive deficit from negative excess

The repository computes

```text
L = max(0, 0.1 - s_positive + s_negative).
```

For every active triplet, `dL/ds_positive = -1` and
`dL/ds_negative = +1` before cosine geometry.  Therefore a hard example with
`s_positive = 0.45, s_negative = 0.65` does **not** receive a larger scalar
weight than an easy active example with `s_positive = 0.83,
s_negative = 0.79`; and the loss cannot decide that the former mostly needs
positive recovery while another case mostly needs negative repulsion.

This is precisely the limitation analyzed by Circle Loss: objectives that only
reduce `s_negative - s_positive` enforce equal penalty amplitudes and have an
ambiguous convergence target.  Stage-3 fixes membership but not this objective
geometry.

### 4.4 One positive and one negative throw away the main value of the new ledger

The native dataset dynamically samples `n_pos_samples=1` and
`n_neg_samples=1`.  For an action event the pools are singletons, so there is no
dynamic choice at all.  For a clean event, only one member from each pool
contributes per pass.  Consequently:

- easy, medium and hard positive relations belonging to one identity do not
  interact in a single loss;
- alternatives within the same positive molecule are not jointly protected;
- competing negative molecules are not compared together;
- the loss cannot learn a cluster around an identity or distribute pressure
  across the current candidate list.

Supervised Contrastive Learning and Multi-Similarity Loss show, in other
domains, why multiple positives and negatives plus explicit pair weighting can
use class/identity labels more efficiently than one triplet.  They are design
evidence, not proof that an unmodified import will improve MS/MS.

### 4.5 Training and evaluation optimize different objects

Training sees one spectrum-level positive and one spectrum-level negative.
Final retrieval compares a query against all candidate spectra and then takes a
maximum over references belonging to each molecule before ranking molecules.
The negative that determines Top-1 can switch as the encoder moves.  A frozen
single negative can stop being decision-relevant after a few updates, while a
new competitor receives no gradient.

This is not primarily “stale action path”—historical E9 found little benefit
from re-mining the peak action itself.  It is a **candidate-boundary staleness**
and aggregation mismatch.  The correct refresh target is the live molecule-max
candidate boundary, not the action recipe.

Smooth-AP demonstrates the general benefit of optimizing a differentiable
ranking surrogate for retrieval.  For this project, a generic AP loss still
does not exactly match the endpoint: it must be adapted to molecule-max scores
and query-wise Top-1/MRR, with the full candidate subset kept identical across
target and control.

### 4.6 Hardest-only mining can lower signal-to-noise

The current hard tier deliberately takes the least similar same-identity
positive and closest different-identity negative.  This is scientifically
valuable, but the hardest cases can be dominated by annotation ambiguity,
different acquisition regimes, poor spectra or cosine noise.  Sampling Matters
shows that extremely close hard negatives can yield high-variance gradient
directions, while random negatives are often too easy; FaceNet therefore used
semi-hard mining and an increasing-difficulty curriculum.

Stage-3 has a mixture of three levels, which is better than hardest-only, but
all levels enter one pass without a measured curriculum or confidence weight.
The next audit must determine whether the hard tier's gradient is consistent
across formulas and with the clean stream, rather than assuming negative margin
means high-quality supervision.

### 4.7 “Same identity” is necessary but not sufficient for a good view

Every positive is IK14-matched, which prevents an obvious label error.  It does
not prove that every severe action view preserves the information needed for
the downstream retrieval task.  InfoMin's primary result is that useful views
remove nuisance information while retaining task-relevant information; more
augmentation is not monotonically better.

For MS/MS, the hard-positive tail must therefore be stratified by acquisition
condition, adduct, precursor agreement, peak support and reference quality.
Foreign-fragment content is not automatically invalid—the action is a
counterfactual spectrum—but extreme views need evidence that they teach
identity invariance instead of erasing identity-bearing fragments.

### 4.8 The model/data ratio and metric saturation matter, but are secondary

The run updates approximately 117M parameters from 13,879 events in one pass.
This raises variance and forgetting risk, but historical LR scans and partial
unfreezing do not support “just tune LR/layers” as the main solution.  The more
direct defects are supervision coverage and objective mismatch.

Recall@10 and Recall@20 are already approximately saturated on the frozen
candidate graph, so a literal requirement that every metric rise by 4--5 pp is
mathematically impossible.  The meaningful primary endpoints are Recall@1,
MRR, margin, risk-net and controlled-FDR coverage; saturated metrics should be
noninferiority guards.

## 5. Literature-constrained next method

No paper below proves a gain on this exact MS/MS graph.  They constrain which
hypotheses are credible enough to test:

| Primary source | What it supports here | What it does not support |
|---|---|---|
| DreaMS, Nature Biotechnology | native shared encoder and contrastive spectrum fine-tuning are valid foundations | that one-positive/one-negative hinge is optimal for Noise |
| CLERMS, Analytical Chemistry | multi-negative InfoNCE-style contrastive learning is viable for MS/MS embeddings | direct superiority on our frozen MassSpecGym graph |
| FaceNet, CVPR 2015 | triplet selection and difficulty curriculum are central | blindly selecting the single hardest relation |
| Sampling Matters, ICCV 2017 | sampling controls gradient variance; extremes can be noisy | its image-distance distribution transfers unchanged to spectra |
| Supervised Contrastive Learning, NeurIPS 2020 | many positives/negatives can exploit identity labels more efficiently than triplets | an automatic MS/MS improvement |
| Circle Loss, CVPR 2020 | separately weight under-optimized positive and negative similarities | that published hyperparameters are appropriate here |
| Multi-Similarity Loss, CVPR 2019 | joint mining and weighting of several pair relations | that image benchmark gains predict ours |
| Smooth-AP, ECCV 2020 | ranking-aligned differentiable objectives can outperform pairwise metric losses | direct molecule-max Top-1 compatibility without adaptation |
| InfoMin, NeurIPS 2020 | augmentation/view severity needs a task-information sweet spot | that minimum similarity is the best positive |
| Metric Learning Reality Check, ECCV 2020 | loss comparisons require identical backbones, splits and tuning budgets | any new loss should be accepted from its name or publication |

## 6. Highest-value next experiment: multi-relation native continuation

The next experiment should not be another invented injector and should not
replace the successful native DreaMS foundation.  It should isolate one missing
capability at a time while preserving the same warm start, shared encoder,
preprocessor, Adam optimizer, target/control action tensors, formula split and
evaluation graph.

### 6.1 Zero-update audit before another long run

Using the frozen Stage-3 artifacts and warm start, materialize:

1. unique query, IK14, formula, positive row, negative molecule and boundary
   counts, plus multiplicity distributions;
2. targeted and control `s_positive`, `s_negative`, active fraction and loss by
   tier, source, formula and action-clean similarity;
3. pre-optimizer parameter-gradient norm and direction for each tier/source,
   separately for positive attraction and negative repulsion;
4. cosine between each action stratum and the clean continuation gradient;
5. metadata/quality enrichment of the extreme hard-positive tail;
6. live candidate switching after the current Stage-3 checkpoint, without
   updating weights.

The targeted-control difference must be measured before an injector or
optimizer statistic.  A different targeted loss magnitude is a treatment
mediator, not automatically a confound; it should be reported, not normalized
away after seeing the result.

### 6.2 Minimal causal ladder

Do not run a broad hyperparameter sweep.  Use a small, pre-registered ladder:

1. **T0 native Stage-3**: the current run, retained as the reference.
2. **T1 multi-relation hinge**: same selected rows and one-pass query dose, but
   each anchor sees all available selected positives and the current top-M
   negative molecules; aggregate per query so multiplicity does not increase
   dose.  This tests information usage without changing the basic loss family.
3. **T2 adaptive pair weighting**: on the same T1 relations, use Circle- or
   Multi-Similarity-style weighting so positive deficit and negative excess can
   receive different pressure.  Hyperparameters are frozen on inner formulas,
   never on the held fold.
4. **T3 molecule-max listwise boundary** only if T1/T2 establish an independent
   target-over-control gain: aggregate candidate references by differentiable
   log-sum-exp/max and optimize the true-positive molecule against the current
   competing molecule list.  Keep native triplet continuation as a stabilizer,
   not a competing reinvention.

Every treatment has its registered same-query control arm.  A clean
continuation arm is required whenever the training budget differs from T0.
This factorial structure separates “more continuation”, “more relations” and
“correct action semantics.”

### 6.3 Difficulty policy

Use a measured easy-to-hard curriculum, not an arbitrary epoch count:

- begin with easy + medium relations;
- introduce hard relations only after their pre-update gradient direction is
  not dominated by formula-specific noise and their label/quality audit passes;
- retain a mixture rather than converging to hardest-only mining;
- refresh negative **candidate molecules** at bounded checkpoints, while the
  action spectra and formula split remain frozen.

The curriculum boundary must be selected on outer-train formulas and then
frozen.  The present Stage-3 result remains the unbiased reference; it must not
be reinterpreted after seeing held outcomes.

## 7. Promotion and stopping rules

A candidate is not promoted merely because it beats official DreaMS.  It must:

1. beat both the warm start and its matched same-query control on Recall@1 and
   MRR with a strictly positive formula-cluster paired interval;
2. have positive `corrected - 2 * introduced` overall and on the near subset;
3. preserve all registered Recall@k, AUROC/AUPRC, margins and rank summaries as
   noninferiority guards, recognizing saturated metrics cannot gain 5 pp;
4. show that target-control gain is present across more than one source family
   and not carried by a few formulas or identities;
5. repeat directionally on a second seed before expansion across folds;
6. report MassSpecGym 10-ppm pooled pairwise AUROC separately from the NIST20
   paper protocol and never call it an exact reproduction of the paper's 0.85.

Stop a branch if it improves training triplet margins without improving the
matched held retrieval endpoint, if the gain disappears against clean
continuation, or if hard-tier gains are offset by near-subset introductions.

## 8. What should and should not happen next

Immediately after Stage-3 finishes:

- recover both full evaluation reports and the paired summary before judging;
- preserve the targeted and control checkpoints even if the promotion gate
  fails;
- run the zero-update relation/gradient audit above on the frozen artifacts;
- do not add epochs to Stage-3 post hoc;
- do not claim `5 pp` from the 5.33-pp training action headroom;
- do not replace valid hard positives with arbitrary synthetic noise;
- do not jump directly to a fashionable loss without the T0/T1/T2 causal
  ladder.

The most defensible route to a larger gain is therefore: broaden independent
chemical/error coverage, use multiple same-identity positives and multiple live
candidate molecules per query, weight positive and negative deficiencies
separately, and finally align the training aggregate with molecule-level
retrieval.  This preserves the proven DreaMS encoder and the best Noise actions
while addressing the specific information discarded by the present triplet
interface.

## 9. Primary references

- DreaMS: https://www.nature.com/articles/s41587-025-02663-3
- MassSpecGym: https://proceedings.neurips.cc/paper_files/paper/2024/hash/c6c31413d5c53b7d1c343c1498734b0f-Abstract-Datasets_and_Benchmarks_Track.html
- CLERMS: https://pubs.acs.org/doi/10.1021/acs.analchem.3c00260
- FaceNet: https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Schroff_FaceNet_A_Unified_2015_CVPR_paper.pdf
- Sampling Matters: https://openaccess.thecvf.com/content_ICCV_2017/papers/Wu_Sampling_Matters_in_ICCV_2017_paper.pdf
- Supervised Contrastive Learning: https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html
- Circle Loss: https://openaccess.thecvf.com/content_CVPR_2020/papers/Sun_Circle_Loss_A_Unified_Perspective_of_Pair_Similarity_Optimization_CVPR_2020_paper.pdf
- Multi-Similarity Loss: https://openaccess.thecvf.com/content_CVPR_2019/html/Wang_Multi-Similarity_Loss_With_General_Pair_Weighting_for_Deep_Metric_Learning_CVPR_2019_paper.html
- Smooth-AP: https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123540647.pdf
- InfoMin: https://proceedings.neurips.cc/paper/2020/file/4c2e5eaae9152079b9e95845750bb9ab-Paper.pdf
- A Metric Learning Reality Check: https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700681.pdf
