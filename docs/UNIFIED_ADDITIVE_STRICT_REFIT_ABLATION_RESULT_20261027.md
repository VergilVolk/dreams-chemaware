# Unified Additive strict-refit contribution result

**Date:** 2026-10-27

**Server run:** `2415768`

**Result directory:**
`data/validation/enveda_unified_additive_ablation_run_2415768`

**Scientific role:** consumed development evidence; not a final external or
field-wide SOTA test.

## Decision

The fixed 23-feature Additive is retained as the current transferable unified
ranking candidate.  This run establishes that the full self-developed evidence
package cannot be reduced to a learned combination of official fine-tuned
DreaMS and weighted spectral entropy (WSE).  It also separates the evidence
families that carry portable incremental value from those whose contribution is
source-dependent.

No model selection is made from this matrix.  The full checkpoint and its
Enveda OOF predictions are reused exactly from run `2413878`; only the four
declared reduced arms are refit.  GNPS receives those frozen Enveda-trained
models without parameter fitting in this evaluation.

## What the full model contains

Noise V1 was not omitted.  It is a core expert in the full system and also
provides the encoder on which the frozen P2b score is defined.  The complete
candidate-level input is:

- Noise V1, WSE, P2b-on-Noise and the frozen Noise-MSG composite;
- ChemAware Stage-1, ChemAware Phase-A and Noise-ChemAware FullMSG encoder
  similarities;
- continuous ChemAware V2 evidence relative to its fixed chemical nulls;
- query-local percentiles and three reliability summaries recomputed from the
  experts retained by each arm.

The strict matrix did not contain a `without_noise_v1` arm.  Therefore this run
does not estimate the marginal effect of Noise alone.  The direct frozen GNPS
comparison already established Noise V1 over official fine-tuned DreaMS by
`+1.2005 pp` on identity and approximately `+1.25 pp` on formula.  The present
matrix asks what the complete Noise-centred system and its evidence families add
beyond stronger learned comparators.

## Evaluation scale and construction

The two panel names are overlapping views, not independent datasets.  Their
query counts must not be added naively.

| Resource and role | Panel | Queries | Query identities | Formula clusters used for paired CI | Candidate-molecule occurrences |
|---|---|---:|---:|---:|---:|
| Enveda-180, fusion OOF development | identity-disjoint | 25,000 | 24,544 | 8,454 | 668,458 |
| Enveda-180, fusion OOF development | formula-disjoint | 9,435 | 8,924 | 4,426 | 58,757 |
| GNPS Gold/Silver, fixed consumed transfer | identity-disjoint | 10,995 | 5,534 | 4,204 | 87,518 |
| GNPS Gold/Silver, fixed consumed transfer | formula-disjoint | 5,261 | 2,662 | 2,210 | 24,401 |

There are 30,343 unique Enveda queries and 11,019 unique GNPS queries after
giving a query present in both panels total weight one.  Enveda was constructed
from a 1,175,202-spectrum source release; GNPS retained 329,607 clean spectra
from 2,091,446 scanned records.

### Was this random sampling?

Not in the sense of a new random draw from all metabolites or instruments.

- Enveda identity queries are ordered by a fixed SHA-256 hash of
  `seed=20261008`, identity and adduct, before any model score exists.  The
  25,000-query ceiling was reached, so this panel is a deterministic,
  model-blind pseudo-random subset of the eligible identity/adduct units.  The
  pre-cap eligible count was not persisted, so the exact sampling fraction is
  unknown; only that at least 25,000 queries were eligible is established.
- Enveda formula-disjoint produced 9,435 queries, below the 25,000 ceiling.
  Under the builder's single pass this is the complete set that survived the
  frozen leakage, cross-condition and candidate-competition rules, rather than
  a further capped sample.
- GNPS is not sampled at the query level.  The builder enumerates every eligible
  Gold/Silver `[M+H]+` identity in sorted order and retains at most two
  high-quality, independent-acquisition queries per identity.  The fixed hash is
  used only if a query has more than 200 negative identities; the observed
  maximum is 52, so the candidate cap did not bind.  Before the two-query cap,
  27,320/11,949 query spectra were eligible on identity/formula; the final
  10,995/5,261 queries still cover all 5,534/2,662 eligible identities.

This construction is appropriate for a model-blind candidate-retrieval
benchmark, but it is not population-representative sampling.  Enveda is tied to
its 60 eV query versus 20 eV reference design and primary validated adducts;
GNPS deliberately favours curated positive-mode Gold/Silver spectra.  Neither
resource alone establishes performance across arbitrary instruments, ion modes,
collision programmes or real cohort spectra.  Both are closed-set panels: the
correct identity must be present and at least one competing identity must exist.

## Absolute performance

| Dataset and panel | Official fine-tuned DreaMS | WSE | Full 23-feature Additive |
|---|---:|---:|---:|
| Enveda identity, n=25,000 | 53.100% | 73.028% | **77.592%** |
| Enveda formula, n=9,435 | 67.207% | 85.056% | **86.084%** |
| GNPS identity, n=10,995 | 85.357% | 87.367% | **88.249%** |
| GNPS formula, n=5,261 | 86.809% | 88.272% | **89.527%** |

The query-balanced Enveda result is `75.398% -> 79.312%` relative to WSE,
`+3.914 pp`.  On fixed Enveda-to-GNPS transfer, Full exceeds WSE by
`+0.882/+1.255 pp` and official fine-tuned DreaMS by `+2.892/+2.718 pp` on
identity/formula.  The latter corresponds to approximately 20% fewer Top-1
errors than official DreaMS on both GNPS panels.

## Strict full-minus-refit contribution matrix

All entries are percentage-point changes in Recall@1.  Positive values favour
the frozen Full model.  Confidence intervals are formula-cluster paired 95%
intervals.

| Reduced refit | Enveda identity | Enveda formula | GNPS identity | GNPS formula |
|---|---:|---:|---:|---:|
| official DreaMS + WSE | **+2.760** `[+2.390,+3.122]` | **+0.827** `[+0.410,+1.246]` | **+0.810** `[+0.495,+1.117]` | **+1.083** `[+0.633,+1.561]` |
| without Stage-1/Phase-A/FullMSG representations | **+1.300** `[+1.004,+1.590]` | +0.011 `[-0.328,+0.337]` | +0.045 `[-0.138,+0.225]` | +0.133 `[-0.151,+0.417]` |
| without ChemAware V2 | **+0.748** `[+0.473,+1.033]` | **+0.435** `[+0.151,+0.721]` | **+0.628** `[+0.377,+0.879]` | **+0.912** `[+0.544,+1.269]` |
| without P2b/MSG | **+0.384** `[+0.200,+0.577]` | -0.095 `[-0.368,+0.187]` | **+0.227** `[+0.027,+0.443]` | **+0.361** `[+0.076,+0.672]` |

Query-balanced Full-minus-refit results are:

| Reduced refit | Enveda, 30,343 unique queries | GNPS, 11,019 unique queries |
|---|---:|---:|
| official DreaMS + WSE | **+2.419 pp** `[+2.109,+2.733]` | **+0.781 pp** `[+0.484,+1.099]` |
| without representation group | **+1.102 pp** `[+0.852,+1.347]` | -0.005 pp `[-0.191,+0.180]` |
| without V2 | **+0.700 pp** `[+0.475,+0.926]` | **+0.590 pp** `[+0.346,+0.840]` |
| without P2b/MSG | **+0.287 pp** `[+0.123,+0.461]` | **+0.213 pp** `[+0.018,+0.412]` |

## Scientific interpretation

1. **The unified evidence package adds information beyond off-the-shelf
   baselines.**  Full beats the capacity-matched official-DreaMS-plus-WSE
   learned combination in both resources and all four panels, with every
   formula-cluster CI lower bound above zero.
2. **Noise V1 is the transferable representation backbone, not a missing
   module.**  It is retained in every group ablation and underlies P2b.  Its
   already-established direct gain and the performance of the retained
   Noise/WSE/P2b/MSG/V2 core show that the portable system is Noise-centred.
3. **ChemAware V2 is the most consistent independent incremental family.**  It
   is the only declared group with positive, supported Top-1 contribution in
   both resources and both panels.  On GNPS, setting all V2 evidence to the
   registered unavailable state in the frozen Full checkpoint loses
   `0.600/0.874 pp`; an application without legal candidate chemistry is a
   partial deployment, not the fully active model.
4. **The local P2b/MSG family has a small but portable contribution.**  GNPS
   Top-1 gains are `+0.227/+0.361 pp`; micro AUPRC gains are approximately
   `+0.028/+0.026`.  This is a family result and does not separately promote
   MSG or revive retired RRF.
5. **The three ChemAware-derived encoder similarities are source-dependent.**
   They add Enveda identity/joint performance but no supported GNPS Top-1
   increment and no Enveda formula increment.  Their historical representation
   results remain valid, but a universal additional ranking gain is not
   established by this scalar integration.

## Is the benchmark large enough?

For the primary overall effects, yes.  The uncertainty is already small enough
to distinguish the observed changes:

- Full versus WSE on Enveda has cluster-CI widths of approximately
  `[+4.138,+4.999] pp` and `[+0.572,+1.497] pp`.
- Fixed Full versus WSE on GNPS has intervals `[+0.577,+1.208] pp` and
  `[+0.783,+1.750] pp`.

Thus 0.8--1.3 pp transfer gains are resolved on thousands of independent
formula clusters.  Increasing the same Enveda hash sample from 25,000 to a much
larger number would narrow sampling error, but it would not repair source,
instrument, adduct or collision-condition dependence.  The more important next
evidence is a genuinely new collection, not another larger slice of the same
release.

The benchmark is not sufficient for every smaller claim.  It cannot resolve a
portable ChemAware-representation increment of only approximately 0.05--0.13
pp, and it does not automatically provide adequately powered instrument-,
source-, adduct- or chemistry-stratified conclusions.  Those claims require
their own effective cluster counts and a new-source confirmation.

## Final boundary and next action

The result supports a publishable algorithm-development statement: specialized
Noise-centred representation, local spectral reranking and continuous
candidate-chemical evidence form a learned low-capacity system that improves
candidate identity ranking across two large development resources.

It does **not** yet support a final SOTA/generalization statement because both
Enveda and GNPS have been consumed by project decisions.  Full remains frozen;
the next high-value work is to apply this exact model to the fixed biological
candidate axis with legitimate V2 evidence wherever structures exist, and to
reserve a genuinely new external collection for one final comparison.  No
additional fusion-head search is justified by this result.
