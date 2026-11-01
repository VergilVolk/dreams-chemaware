# Full23 v1.3 formal development selection — 2026-11-01

## Decision

`no_count_query_centered` is designated **Full23 v1.3**, the formal unified
candidate-ranking development algorithm. Full23 v1 remains its frozen parent;
the failed uncentred v1.2 experiment remains negative development evidence.

The primary selection endpoint is Recall@1. A candidate advances when its
Recall@1 point estimate exceeds Full23 v1 on both registered GNPS development
panels and on the query-balanced unique-query endpoint. Formula-cluster
intervals, corrected/introduced exchanges, near metrics and the historical
`corrected - 2*introduced` quantity remain visible diagnostics, but none can
override a positive primary ranking endpoint.

This policy was corrected after the result was observed. It must be described
as a post-result development choice, not as preregistered confirmation. GNPS is
already consumed development data, so v1.3 still requires genuinely new data
for an external claim.

## Architecture

Full23 v1.3 does not replace or retrain the upstream assets. It composes:

1. Full23 v1 candidate-level evidence: Noise V1, WSE, P2b-on-Noise,
   Noise-MSG, ChemAware Stage-1, ChemAware Phase-A, Noise-ChemAware FullMSG,
   and ChemAware V2 continuous real-versus-null evidence when available.
2. A bounded tensor-set residual reading the complete embeddings from Noise
   V1, Stage-1, Phase-A and Noise-ChemAware FullMSG.
3. Removal of the reference-count shortcut.
4. Query-centering of the learned raw residual before bounded addition to the
   Full23 v1 score.

The final score is:

`Full23 v1 score + bounded query-centred full-embedding residual`.

## Consumed GNPS development result

| Endpoint | Full23 v1 | Full23 v1.3 | Change |
|---|---:|---:|---:|
| Identity-disjoint Recall@1, 10,995 queries | 88.2492% | **88.2947%** | **+0.0455 pp** |
| Formula-disjoint Recall@1, 5,261 queries | 89.5267% | **89.6408%** | **+0.1140 pp** |
| Query-balanced Top-1, 11,019 unique queries | 88.2839% | **88.3429%** | **+0.0590 pp** |

Identity corrected/introduced was 92/87; formula was 52/46. These exchanges
explain why the old lambda=2 statistic is negative even though the actual
number of correct Top-1 decisions increased on both panels. The lambda=2
quantity is therefore retained only as a user-specific asymmetric-utility
description and is no longer a general ranking gate.

Other supporting changes were directionally positive: MRR, macro AUROC, micro
AUROC, micro AUPRC and near Recall@1 increased on both panels. The formal claim
remains limited to consumed development performance.

## Release

The portable decision package is `deploy/unified_additive_full23_v13`. It
contains the original Full23 v1 parent checkpoint, the v1.3 representation
readout checkpoint, exact runtime sources and the unmodified source reports.
