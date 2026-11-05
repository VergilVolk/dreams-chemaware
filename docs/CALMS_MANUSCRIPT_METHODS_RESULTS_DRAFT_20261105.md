# CALMS manuscript-ready Methods and Results draft

Date: 2026-11-05
Frozen method: **CALMS v1.0** (internal implementation Full23 v1.3.2)
Expanded name: **Context-Aware Learning for Metabolite Annotation from Mass
Spectra**

This document is a manuscript drafting source, not a new experiment. Every
number below is taken from a frozen report assembled in
`deliverables/calms_v1_manuscript_tables_20261105`. No model was fitted and no
bootstrap was rerun to produce this draft.

## Methods

### Task definition and candidate axis

CALMS addresses closed-library metabolite candidate annotation. For each query
MS/MS spectrum, the caller supplies a mass-compatible candidate block and
aligned evidence for every query-candidate pair. CALMS returns a candidate
score and a ranked structural hypothesis. The released scorer does not create
the candidate universe, compute upstream evidence from raw MGF files, or
provide calibrated no-match rejection.

In CALMS v1.0, “context-aware” refers to within-query candidate-relative
statistics, reference-set aggregation, cross-encoder interactions and the
availability of candidate-specific chemistry. It does not mean that
biological-sample networks or BioAware events are inputs to the frozen scorer.
BioAware remains outside CALMS v1.0; any future sample-event extension requires
a separate versioned interface and evaluation with genuine sample-level
events.

The model has two frozen layers:

\[
s_{\mathrm{CALMS}}(q,c)
=s_{\mathrm{parent}}(q,c)
+0.9\,r_{\mathrm{emb}}(q,c).
\]

The candidate-evidence parent integrates representation similarities,
fragmentation matching and candidate chemistry. The embedding correction reads
the complete learned representations and can adjust the parent only when all
required representation inputs are available.

### Candidate-evidence parent

The parent receives seven aligned candidate-level score families:

1. Noise V1 representation similarity;
2. weighted spectral entropy (WSE);
3. the frozen P2b-on-Noise composite;
4. the frozen Noise-MSG composite;
5. ChemAware Stage-1 representation similarity;
6. ChemAware Phase-A representation similarity; and
7. Noise-ChemAware FullMSG representation similarity.

P2b and MSG each enter once as their frozen composite; their constituent
channels are not duplicated as independent votes. For every score family, the
parent uses the raw candidate score and its within-query percentile. Three
additional summaries describe agreement among the seven families: the
fraction voting for the same Top-1 candidate, the mean candidate percentile
and the percentile standard deviation.

When candidate structures and valid chemical evidence are available,
ChemAware V2 contributes six fields: one binary availability indicator and
five continuous real-versus-null summaries (correct-minus-zero utility,
correct-minus-worst-null utility, null spread, the percentile of the chemical
delta and the chemical-delta Top-1 gap). These fields distinguish chemical
support from the absence of chemical evidence; an unavailable V2 observation
follows the frozen no-op contract and is not treated as negative evidence.

For ordinary fields, normalization uses query-balanced weighted means and
standard deviations estimated within the training fold. The V2 availability
field remains binary. The other V2 fields are not centered and are scaled only
by their weighted root-mean-square, so an unavailable zero remains an exact
no-op. The resulting 23 fields are scored by a 24-parameter affine model,

\[
s_{\mathrm{parent}}(q,c)=b+\sum_{j=1}^{23}w_j\tilde{x}_j(q,c).
\]

Query-constant candidate-count, Top-gap and pairwise expert-agreement fields
were excluded because they cannot directly change a within-query ordering in
an additive readout. The selected parent was fit on the consumed Enveda-180
meta-development panels using five formula-grouped folds. Cross-panel
occurrences of one unique query remained indivisible and were weighted to a
total mass of one. Five global formula folds produced OOF development
predictions; the deployed affine parent was then refit on all Enveda-180
meta-development data with the same fixed settings rather than formed as a
fold ensemble. Training started from WSE with its weight set to one and all
other weights and the bias set to zero. The loss was query-balanced listwise
cross-entropy plus a 0.01 squared penalty on the score residual relative to
WSE. Optimization used 20 epochs, batches of 256 unique queries, learning rate
0.001, weight decay 0.0001 and seed 20261026, without a hyperparameter scan.
The additive parent was ultimately frozen after cross-source development
diagnostics; this was model development, not external confirmation.

### Full-embedding correction

The second layer uses the complete 1,024-dimensional embeddings from Noise V1,
ChemAware Stage-1, ChemAware Phase-A and Noise-ChemAware FullMSG. For one
encoder and one query-candidate pair, CALMS normalizes the query embedding and
up to three candidate reference embeddings and constructs

\[
[\lvert z_q-z_r\rvert,\;z_q\odot z_r].
\]

A learned 2,048-to-64 pair map is applied before reference aggregation. An
attention-weighted mean and standard deviation summarize the reference set,
and a 128-to-64 map produces one token per encoder. Four encoder tokens are
combined by four-head self-attention and a feed-forward block. Their flattened
256-dimensional representation is mapped through a 128-unit residual head.
This design exposes dimension-wise differences and multiplicative interactions
that are lost when each embedding pair is compressed to a single cosine.

The raw residual is centered within each query and bounded as

\[
r_{\mathrm{emb}}=\rho\tanh(r_{\mathrm{raw}}/\rho),\qquad
\rho=0.6255066395.
\]

The bound was calculated from Enveda formula-OOF parent margins without GNPS
access. The embedding readout contains 625,345 trainable parameters and was
trained against the five-fold formula-OOF parent scores, not in-sample scores
from the all-data parent refit. The fixed training used six epochs, 48 query
groups per batch, AdamW, learning rate 0.0002, weight decay 0.0001 and seed
20261031. Its loss was listwise cross-entropy plus 0.5 times a hardest-negative
softplus term (margin 0.10) and a 0.01 penalty on the query-centered raw
residual. Correct parent decisions were not upweighted (anchor-correct
multiplier 1.0). The final multiplier 0.9 was selected after inspection of
consumed GNPS predictions and is therefore disclosed as post-result
development calibration.

### Missing evidence and inference behavior

Every non-empty query requires all seven finite parent score families. A query
that violates this parent contract is retained but not scored. Zero-candidate
and singleton queries are legal. ChemAware V2 is optional under its explicit
real/unavailable contract. All four embedding-bank files, their manifest and
their hashes are required scorer inputs; a missing bank fails closed. When all
banks are present but a particular query has incomplete or invalid aligned
positions or reference rows, CALMS applies no embedding correction to that
query and retains its parent scores. Singleton queries are likewise scored by
the parent only; the correction is applied only to a block with at least two
candidates and complete representation evidence. This technical fallback is
not open-set rejection. BioAware is not part of CALMS v1.0; it remains a
conditional downstream extension requiring genuine sample-level event
context.

### Development datasets

Enveda-180 was consumed as meta-development data. Its two primary panels
contained 25,000 identity-disjoint queries (8,454 formula clusters) and 9,435
formula-disjoint queries (4,426 formula clusters). Enveda labels were used to
fit the parent and embedding readout and to estimate formula-OOF development
performance.

GNPS Gold/Silver was also consumed during development, comparison and final
residual-amplitude selection. The native panels contained 10,995
identity-disjoint queries (4,204 formula clusters) and 5,261 formula-disjoint
queries (2,210 formula clusters). The panels overlap: 5,237 of 5,261 formula
queries also occur in the identity panel, so their counts and error exchanges
must not be added.

The panel names describe each benchmark's own construction and must not be
interpreted as a global zero-overlap guarantee against every formula later
used to train CALMS. A post-freeze exact-set audit found that the 4,426 formulas
in the Enveda formula-disjoint training panel had zero overlap with either
GNPS panel, whereas the full 10,179-formula Enveda training union shared 1,046
query formulas with GNPS identity and 339 with GNPS formula. All overlap came
from the Enveda identity-disjoint training panel, whose design permits seen
formulas. This is not by itself evidence of query-identity or candidate-label
leakage, but it reinforces that GNPS is consumed development evidence rather
than an external formula-disjoint confirmation of the final refit. Input
hashes, exact-set counts and interpretation limits are frozen in
`docs/CALMS_ENVEDA_GNPS_QUERY_FORMULA_OVERLAP_AUDIT_20261105.md`.

For the 18-method comparison, all methods were restricted to the strict common
availability axis: 10,983 identity queries (4,199 formula clusters) and 5,253
formula queries (2,207 formula clusters). Every method therefore saw identical
query and candidate blocks. The native-axis mechanism analysis used the full
10,995/5,261 panels and is reported separately.

### Evaluation and statistics

Primary ranking endpoints were Recall@1, Recall@3, Recall@5 and mean reciprocal
rank (MRR). Candidate discrimination was summarized by micro candidate AUROC
and average precision. Paired Top-1 exchanges were recorded as corrected and
introduced queries.

On the common axis, Recall@1 intervals were formula-cluster bootstrap intervals
familywise-adjusted over 20 endpoints within one CALMS-versus-one-comparator
analysis. They are not simultaneous over all 17 comparators or both panels.
Native-axis global, near-subset and selective-ranking intervals used 10,000
formula-cluster bootstrap resamples (seed 20261105) and are descriptive unless
otherwise stated. Parent strict-refit and final-embedding intervals were read
from the corresponding frozen 5,000-resample reports.

Selective ranking ordered queries by the frozen Top1-Top2 CALMS score gap.
Each method selected its own highest-confidence prefix at matched coverage.
Because every evaluated query contained the true candidate, these analyses are
closed-set and do not estimate open-set false acceptance, FDR, a calibrated
match probability or a deployable abstention threshold.

## Results

### CALMS improved same-axis closed-set ranking

On the strict 18-method common axis, CALMS achieved Recall@1/3/5 of
88.400%/95.939%/97.505% on identity-disjoint queries (n=10,983) and
89.644%/97.658%/98.972% on formula-disjoint queries (n=5,253). MRR was 0.9247
and 0.9376, micro AUROC was 0.9587 and 0.9539, and micro average precision was
0.8445 and 0.8744, respectively.

Relative to fine-tuned DreaMS, Recall@1 increased by 3.014 percentage points
(pp; within-comparator familywise CI 2.121-3.870) on identity queries and 2.779
pp (1.512-4.083) on formula queries. These changes corresponded to 20.6% and
21.2% reductions in Top-1 error. Relative to WSE, gains were 0.983 pp
(0.432-1.541) and 1.333 pp (0.596-2.165). On this exact-identity axis, CALMS
also exceeded the evaluated MS2DeepScore 2.x checkpoint by 9.242 pp
(8.029-10.548) and 8.110 pp (6.370-9.908). MS2DeepScore 2.x is optimized for
structural similarity, so this is a same-axis exact-identity comparison rather
than a judgment of its native task.

### Improvements concentrated in difficult candidate competitions

On the native full panels, CALMS increased WSE Recall@1 from 87.367% to 88.367%
on identity queries (+1.000 pp; descriptive CI 0.637-1.361; 223 corrected and
113 introduced) and from 88.272% to 89.603% on formula queries (+1.331 pp;
0.812-1.874; 116 corrected and 46 introduced).

Among predefined near-candidate queries, gains over WSE were 1.317 pp
(0.810-1.825) across 7,592 identity queries and 2.208 pp (1.317-3.112) across
2,944 formula queries. The near label uses truth and structure metadata and is
therefore a post-freeze mechanism diagnostic, not an observable deployment
router.

### High-coverage selective ranking retained the advantage

At 90% retained coverage, CALMS Recall@1 was 93.108% on identity queries and
93.874% on formula queries. These high-confidence prefixes exceeded the
corresponding WSE prefixes by 0.677 pp (paired functional CI 0.327-0.979) and
0.634 pp (0.131-1.124).

Descriptive AURC decreased from 0.01956 for WSE to 0.01791 for CALMS on the
identity panel and from 0.01730 to 0.01563 on the formula panel, relative
reductions of 8.47% and 9.62%. However, margin-based correctness AUROC was
0.915 for CALMS versus 0.921 for WSE on identity and 0.912 versus 0.923 on
formula. The selective-accuracy improvement therefore reflects a stronger base
ranking rather than demonstrated superiority of the CALMS margin as a
calibrated confidence score.

### Strict-refit parent ablations supported grouped evidence contributions

Compared with a learned additive model using only DreaMS and WSE, the complete
candidate-evidence parent increased Recall@1 by 2.760 pp (CI 2.390-3.122) and
0.827 pp (0.410-1.246) on the Enveda identity/formula panels and by 0.809 pp
(0.495-1.117) and 1.083 pp (0.632-1.561) after fixed transfer to the consumed
GNPS identity/formula panels.

Removing continuous ChemAware V2 evidence reduced Recall@1 by 0.748, 0.435,
0.628 and 0.912 pp across the same four panels; all four intervals excluded
zero. This was the most consistent grouped contribution. Removing P2b+MSG
changed Recall@1 by +0.384, -0.095, +0.227 and +0.361 pp, respectively; the
Enveda formula interval crossed zero. Removing the Stage-1/Phase-A/FullMSG
scalar representation group changed Recall@1 by +1.300, +0.011, +0.045 and
+0.133 pp; only Enveda identity resolved a clear independent contribution.
These are grouped parent-layer refits and do not isolate individual-module
causal effects.

### The final full-embedding correction had a small unresolved increment

The final correction increased native GNPS Recall@1 over the parent by 0.118 pp
(CI -0.130 to 0.374; 89 corrected/76 introduced) on identity and 0.076 pp
(-0.281 to 0.440; 40/36) on formula. Both intervals crossed zero. The full
embedding layer nevertheless changed rankings in the fixed-axis application,
but its independent accuracy increment is a positive development point
estimate rather than a confirmed effect. Parent strict-refit contributions and
the final embedding increment answer different questions and must not be
added.

### Fixed-axis application demonstrated executable model replacement

CALMS was applied without response access during spectral scoring to 38,628
directional entity queries on the existing MSV000095331 plus three-study
application axis. The axis contained 22,046 candidate pairs, of which 16,916
were CALMS-scoreable. These pairs formed 8,583 scored query blocks; 3,324
query blocks received the full embedding correction. The application endpoint
used 22,014 forward MSV000095331-to-candidate-study queries, whereas 38,628 is
the complete bidirectional axis; these denominators are not interchangeable.

Across 30 registered positive-control contexts, CALMS achieved Top-3/Top-5 of
7/10 and the best mean rank among 11 compared methods (10.30). Against WSE at
the exact CALMS coverage, CALMS improved 17 contexts, tied 8 and worsened 5
(mean rank change -1.00). Against its parent, the embedding correction improved
11, tied 14 and worsened 5 (mean rank change -0.43). The 30 contexts represent
at most six drug-by-study units and lack complete identity truth; these are
descriptive application results, not independent biological discoveries or
external accuracy confirmation.

## Code and data availability draft

The frozen CALMS v1.0 implementation is distributed under the repository MIT
License. The standalone archive is `CALMS_v1.0_20261105.zip` (SHA-256
`7305d38b74dad4edf74c1313e5a3989c72df1cfffd030781336a11dce2be6deb`).
It contains both frozen checkpoints, the exact runtime, input templates, a
package-integrity test and a retained synthetic CLI smoke example.

Compact supporting evidence is archived as
`CALMS_v1.0_evidence_bundle_20261105.zip` (SHA-256
`f6995e2acaa746d3a808d066e6129b71e1d692c8fe429c96b12bb2cc1a431648`).
It contains the executable release, compact frozen reports, strict-refit
ablation evidence, manuscript tables, figures and their generating scripts.
From the extracted evidence-bundle root, the main tables are reproduced with:

```bash
python tasks/build_calms_manuscript_main_tables.py \
  --evidence-root . \
  --output reproduced_tables
```

The compact bundle omits four large per-query correction ledgers, original raw
libraries, complete candidate-axis construction inputs and full method-score
caches. It therefore reproduces the published compact tables from frozen
reports but does not independently rebuild every score and bootstrap from raw
MGF files.

## Internal submission boundary

The following claims are not supported by the current frozen evidence and must
not appear as completed results:

1. untouched external confirmation or field-wide SOTA;
2. calibrated open-set/no-match rejection, FDR control or match probability;
3. a general raw-MGF-to-all-evidence software workflow;
4. independent confirmation that the final embedding correction improves
   accuracy; or
5. BioAware benefit without genuine sample-level event context and an
   independent biological endpoint.

GNPS and Enveda are consumed development resources. A genuinely unused source
with a frozen contamination-exclusion protocol is required before an external
confirmation claim.
