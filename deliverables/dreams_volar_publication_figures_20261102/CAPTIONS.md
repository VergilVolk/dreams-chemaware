# DreaMS-VOLAR manuscript figure captions

## Figure 1 | DreaMS-VOLAR integrates complementary evidence before candidate ranking

DreaMS-VOLAR (Vector, Orthogonal and Local Evidence with an Additive Residual)
starts from a query spectrum and a precursor-compatible candidate set. Four
condition-robust representation streams provide global vector evidence; WSE,
P2b-on-Noise and Noise-MSG provide local fragmentation evidence; and ChemAware
V2 supplies candidate-specific real-versus-matched-null chemical evidence. A
learned additive parent is complemented by a query-centred full-embedding
residual. The public DreaMS-VOLAR v1.0 implementation is frozen internally as
Full23 v1.3.2.

## Figure 2 | Public-method benchmark on a shared candidate axis

Recall@1 ladders, Top-k curves and candidate-level ROC curves compare
DreaMS-VOLAR with weighted spectral entropy, official fine-tuned DreaMS,
entropy and denoising search, Spec2Vec, MS2DeepScore 2.x, MSBERT and
SpecEmbedding. All methods use the same GNPS candidate axis: 10,983
identity-disjoint and 5,253 formula-disjoint queries. Thin grey ROC curves show
all 18 evaluated scoring streams. MS2DeepScore 2.x is an auxiliary comparator
because its native learning objective is structural similarity rather than
exact identity.

## Figure 3 | Precision-recall and decision changes

Candidate-level precision-recall curves are paired with query-level decision
flows relative to WSE. Flow width is the exact fraction of shared queries.
Green paths are WSE errors corrected by DreaMS-VOLAR, red paths are newly
introduced errors, blue paths retain a correct Top-1 decision, and grey paths
remain unresolved.

## Figure 4 | Evidence contribution and paired error exchange

Bubble size and labels show the Full23-v1 strict-refit Recall@1 advantage over
each evidence-family ablation on consumed Enveda and GNPS development panels.
The butterfly plots show DreaMS-VOLAR Top-1 errors corrected (right) and correct
decisions lost (left) relative to public comparison methods on the shared GNPS
domain. Internal module streams are intentionally separated from external/public
method comparisons.

## Figure S1 | Multi-metric public-method rank profile

Within-panel ranks across Recall@1/3/5, MRR, macro- and micro-AUROC,
micro-AUPRC and near-subset Recall@1. Rank 1 denotes the best result among the
nine reader-facing public methods.

Scientific boundary: GNPS and Enveda are consumed development resources. These
figures support same-axis method development comparisons, not an unopened
external SOTA claim.
