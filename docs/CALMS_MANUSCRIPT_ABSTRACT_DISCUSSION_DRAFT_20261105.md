# CALMS manuscript-ready Abstract and Discussion draft

Date: 2026-11-05

Frozen method: **CALMS v1.0** (internal implementation Full23 v1.3.2)

Expanded name: **Context-Aware Learning for Metabolite Annotation from Mass
Spectra**

This is a writing artifact, not a new analysis. Numerical statements are tied
to the frozen tables and reports distributed with the CALMS v1.0 evidence
bundle. Literature citations and journal-specific formatting remain to be
inserted.

## Abstract

Untargeted liquid chromatography-tandem mass spectrometry detects thousands of
molecular features, yet reliable metabolite annotation remains limited by
acquisition-dependent spectral variation, competition among structurally
similar candidates and incomplete use of fragmentation chemistry. We developed
CALMS (Context-Aware Learning for Metabolite Annotation from Mass Spectra), a
frozen two-layer scorer for caller-supplied, mass-compatible candidates in a
closed-library setting. Here, context denotes candidate-relative statistics,
reference-set aggregation, cross-encoder interactions and chemistry
availability; it does not denote sample-level metabolic-network or BioAware
context. A low-capacity candidate-evidence
model combines condition-robust spectral representations, fragmentation
matching, local spectral rerankers and availability-aware chemical evidence. A
bounded residual layer then reads complete 1,024-dimensional representations
from four frozen encoder checkpoints. On strict common-availability axes
of the consumed GNPS Gold/Silver development resource, CALMS achieved
Recall@1/3/5 of 88.400%/95.939%/97.505% for 10,983 identity-disjoint queries
and 89.644%/97.658%/98.972% for 5,253 formula-disjoint queries. Micro candidate
AUROC and average precision were 0.9587/0.8445 and 0.9539/0.8744,
respectively. Relative to fine-tuned DreaMS, Recall@1 increased by 3.014
percentage points (pp; within-comparator familywise interval, 2.121-3.870) and
2.779 pp (1.512-4.083), reducing Top-1 error by 20.6% and 21.2%. Relative to
weighted spectral entropy, gains were 0.983 pp (0.432-1.541) and 1.333 pp
(0.596-2.165). Strict-refit ablations identified continuous ChemAware V2
evidence as the most consistent grouped contribution. The final embedding
correction had smaller positive point estimates of 0.118 and 0.076 pp over its
parent, with intervals crossing zero. A fixed-axis metabolomics application
showed executable changes in entity and response rankings, but its correlated,
incompletely identified contexts remain descriptive. Because Enveda and GNPS
were consumed during development and calibration, the results establish a
reproducible development-stage improvement in closed-set annotation, not
untouched external confirmation, calibrated open-set identification or a
field-wide state-of-the-art claim.

## Discussion

CALMS addresses a practical limitation of current MS/MS annotation systems: no
single representation or spectral-similarity measure is uniformly reliable
under changing acquisition conditions and close candidate competition. The
method therefore separates annotation into two complementary operations. A
low-capacity parent first combines aligned representation, fragmentation and
candidate-chemistry evidence on a shared candidate axis. A bounded second
layer reads complete embeddings from four frozen encoder checkpoints and
can adjust the parent when dimension-level representation information is
available. This division preserves the transferability and interpretability of
candidate-level evidence while permitting information discarded by scalar
cosine similarities to influence difficult decisions.

### Improvement extended beyond Top-1 accuracy

On the strict 18-method common-availability axes, CALMS improved multiple
ranking endpoints. Recall@1 reached 88.400% and 89.644% on the identity- and
formula-disjoint panels, while Recall@5 reached 97.505% and 98.972%. MRR was
0.9247 and 0.9376. Micro candidate AUROC values of 0.9587 and 0.9539, together
with average precision values of 0.8445 and 0.8744, show that the improvement
was not limited to the first-ranked candidate. These pooled candidate metrics
must nevertheless remain distinct from query-level Top-1 accuracy: high
candidate discrimination does not guarantee the correct ordering inside every
candidate block.

Fine-tuned DreaMS provides the clearest shared-representation baseline on this
exact axis. CALMS increased Recall@1 by 3.014 pp on identity-disjoint queries
and 2.779 pp on formula-disjoint queries, reducing the remaining Top-1 error by
approximately one fifth. CALMS also exceeded the strong WSE baseline by 0.983
and 1.333 pp. Positive formula-cluster intervals make an explanation based on
only a few formula clusters unlikely. These intervals do not, however, absorb
all uncertainty introduced by earlier development decisions or the post-result
selection of the final embedding-residual scale.

### Candidate chemistry was the most stable grouped contribution

The complete candidate-evidence parent exceeded a capacity-matched learned
additive model containing only DreaMS and WSE by 2.760 and 0.827 pp on the two
consumed Enveda panels and by 0.809 and 1.083 pp after fixed transfer to the
consumed GNPS panels. Removing continuous ChemAware V2 evidence reduced
Recall@1 by 0.748, 0.435, 0.628 and 0.912 pp across those four panels, with all
four intervals excluding zero. ChemAware V2 was therefore the most consistent
grouped contribution under the frozen strict-refit protocol. The result
supports availability-aware chemistry: missing chemical evidence is encoded
as unavailable rather than as evidence against a candidate. It remains a
grouped model comparison, not a causal decomposition of every V2 field or a
guarantee of the same magnitude in a new source.

Other evidence groups were less uniform. Removing P2b and MSG produced
positive full-minus-ablation differences on three panels, while the Enveda
formula result was unresolved. Removing the Stage-1, Phase-A and FullMSG
scalar representation group produced a clear loss only on Enveda identity;
the other intervals crossed zero. The trained representations remain integral
inputs to the complete system, but their independent scalar contribution is
distribution-dependent. Historical single-module improvements should not be
expected to add linearly after integration, and coefficient magnitude alone
does not establish module value.

### The complete-embedding layer remains an unresolved increment

Relative to the frozen parent, the complete-embedding correction increased
native GNPS Recall@1 by 0.118 pp on identity-disjoint queries and 0.076 pp on
formula-disjoint queries. Both formula-cluster intervals crossed zero. The
layer changed 165 and 76 Top-1 decisions, respectively, so it is operational
rather than numerically inert, but its independent accuracy advantage is not
confirmed. Its final multiplier of 0.9 was chosen after inspection of consumed
GNPS predictions. The embedding layer should therefore be described as a
frozen component with an unresolved incremental development effect. Its point
estimates must not be added to the parent strict-refit contributions, because
the analyses answer different questions.

### Gains arose in difficult candidate competitions

Every observed CALMS-versus-WSE Top-1 exchange occurred in WSE's lowest
Top1-Top2 gap quartile. Within predefined near-candidate subsets, CALMS
improved Recall@1 by 1.317 pp on the identity panel and 2.208 pp on the formula
panel. These observations are consistent with the intended role of
condition-robust representations and candidate chemistry in resolving close
alternatives. They do not define a deployable router: the near label uses
truth and structural metadata, and gap quartiles are relative ranks rather
than calibrated probabilities.

At 90% retained coverage, CALMS achieved Recall@1 of 93.108% and 93.874%,
exceeding matched-coverage WSE by 0.677 and 0.634 pp. CALMS also had lower
descriptive areas under the risk-coverage curve. Its score-gap correctness
AUROC was nevertheless slightly lower than WSE on both panels. The selective
advantage therefore appears to arise mainly from stronger base ranking, not a
better calibrated confidence score. Because every evaluated query contained
the correct candidate, these analyses do not establish no-match detection,
false-acceptance control, FDR or a deployable abstention threshold.

### Public-method comparisons are task-specific

CALMS had higher exact-identity Recall@1 than every evaluated public checkpoint
and spectral-search baseline on the common GNPS candidate axis. It exceeded
the evaluated MS2DeepScore 2.x checkpoint by 9.242 and 8.110 pp. However,
MS2DeepScore 2.x was developed primarily for structural-similarity estimation,
and the compared methods differ in training data, preprocessing domains and
native objectives. The comparison establishes same-axis performance for the
tested implementations; it is not a universal ranking of methods on their
native tasks or evidence for a field-wide state-of-the-art claim.

### The application establishes deployment, not biological confirmation

Frozen CALMS was applied without response labels during spectral scoring to a
complete bidirectional axis of 38,628 directional query blocks and 22,046
same-observed-precursor, 10-ppm candidate pairs. Of these, 16,916 pairs formed
8,583 CALMS-scored query blocks, and 3,324 query blocks received the full
embedding correction. The downstream endpoint used 22,014 forward
MSV000095331-to-candidate-study queries; this is distinct from the complete
bidirectional denominator.

Across 30 registered positive-control contexts, CALMS achieved Top-3 and Top-5
recovery in 7 and 10 contexts and the best mean rank among 11 compared methods
(10.30). At identical candidate coverage, CALMS improved 17 contexts relative
to WSE, tied 8 and worsened 5, for a mean rank change of -1.00 (lower is
better). Relative to its parent, the embedding correction improved 11, tied 14
and worsened 5, with mean rank change -0.43. Matched-coverage WSE achieved
Top-3/Top-5 counts of 8/8, so the supported application claim is improved mean
rank and Top-5 recovery, not uniform dominance at every Top-k endpoint.

These 30 contexts correspond to at most six independent drug-by-study units;
the repeated microbial strata are correlated. Complete identity truth,
authentic-standard confirmation and orthogonal structural validation are not
available. Candidate structures required by ChemAware V2 were unavailable on
this axis, and BioAware was not invoked without genuine sample-event context.
The application demonstrates executable model replacement and changes in
downstream rankings, not newly verified metabolite identities or an
independent biological discovery.

### Validation priorities and conclusion

Both Enveda-180 and GNPS were consumed during model fitting, diagnosis or
residual-amplitude selection. A genuinely unused source screened under a
frozen contamination-exclusion protocol is required to test cross-source
generalization. Open-set evaluation must add realistic no-match queries and
independently calibrate acceptance or error control. The portable release
scores caller-provided aligned candidate evidence and embeddings; it is not a
raw-MGF-to-annotation workflow. Sample-context evidence should enter only when
genuine candidate-specific biological events are present and the evaluation
endpoint was not also used to construct that context.

Within these boundaries, CALMS shows that condition-robust representations,
fragmentation matching and continuous candidate chemistry can be combined in
a compact, reproducible system that improves closed-set candidate ranking
across two large consumed development resources. The strongest supported
contribution is the integrated candidate-evidence parent, particularly its
continuous chemical evidence. The complete-embedding correction is
scientifically motivated and operational but awaits independent confirmation.

## Non-negotiable reporting boundaries

1. Enveda and GNPS are consumed development resources; GNPS also informed the
   final residual scale of 0.9.
2. The two GNPS panels overlap substantially. Their denominators and error
   exchanges must not be added.
3. Main-table values of +3.014/+2.779 pp use the strict common axis; native-axis
   values of +3.010/+2.794 pp answer a different question.
4. Main-table intervals are familywise-adjusted within one CALMS-versus-one
   comparator analysis, not simultaneously over all comparators and all prior
   development choices.
5. The V2 result is a grouped parent-layer strict refit, not a complete causal
   ablation of final CALMS.
6. The embedding increment is unresolved and its final scale is post-result
   development calibration.
7. The application does not use V2 or BioAware and has at most six independent
   drug-by-study units.
8. The MS2DeepScore 2.x difference is a same-axis exact-identity comparison,
   not a comparison on its native structural-similarity task.
9. Current evidence does not support untouched external confirmation,
   calibrated open-set performance or a field-wide SOTA claim.
