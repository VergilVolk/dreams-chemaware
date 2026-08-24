# Deployment-clean modules (`deploy/`)

This directory is the **deployable subset** of the repo: the frozen, validated
pieces that are safe to import into a production annotation pipeline, kept
separate from the experiment scripts under `tasks/`.

## What is deployable now

| Module | Where | Status |
|---|---|---|
| DreaMS encoder + M0–M9 annotation platform | `annotation/` | ✅ deployable (embed → retrieve → confidence → FDR → calibrate → diff → pathway → darkmatter) |
| **P2b frozen local rank fusion** | `deploy/p2b_rank_fusion.py` | ✅ deployable score function (frozen weights) |

The full retrieval-plus-rerank path is:

```
query MS2
  -> DreaMS embedding + strict-10 ppm / same-adduct candidate generation   [annotation]
  -> P2b frozen rank fusion over that candidate group                        [deploy]
  -> confidence / FDR / calibration                                         [annotation]
```

## The frozen P2b scorer

```text
score = 0.10 * dreams_similarity
      + 0.00 * sqrt_cosine
      + 0.10 * entropy_similarity
      + 0.80 * neutral_loss_sqrt_cosine
```

`normalization = "absolute"`, `min_support = 1`, `min_advantage = 0.0`.
Features (in order): `dreams_similarity`, `sqrt_cosine`, `entropy_similarity`,
`neutral_loss_sqrt_cosine`.  Candidate molecules are scored by the per-molecule
maximum of their spectrum-pair scores (no cross-spectrum feature mixing).

### Evidence (one-shot, sealed P3 — `docs/P2B_RANK_FUSION_FORMAL_RECORD_20260823.md`)

| Panel | n | DreaMS Recall@1 | P2b Recall@1 | Δ | verdict |
|---|---:|---:|---:|---:|---|
| main-real-pristine | 3,000 | 0.8793 | 0.8900 | **+1.07 pp** | CI [+0.24, +1.89] pp, McNemar p=0.0101 → **significant** |
| isomer-real-pristine | 1,989 | 0.7949 | 0.7959 | +0.10 pp | CI crosses 0, p=0.93 → not significant |
| **near-core-real-pristine** | 496 | 0.4879 | 0.4456 | **−4.23 pp** | CI entirely negative, p=0.0099 → **significantly worse** |
| nearmid-real-pristine | 661 | 0.5446 | 0.5144 | −3.03 pp | p=0.033 → worse |

## Hard boundary: do NOT run P2b standalone on near-isomer candidate sets

The frozen P2b **hurts** the MCES 0–2 near-isomer candidate sets (−4.23 pp,
significant).  The "lock down" near-safety firewall that falls back to DreaMS
on such sets is **not yet deployable**:

- the current firewall (`tasks/audit_g8r_p3_candidate_ambiguity_router.py`)
  detects near pairs from a cached MCES relation table — i.e. it consumes
  **structure labels** and is explicitly marked *diagnostic-only / consumed-P3*;
- a deployable gate must rely only on inference-time information (DreaMS-vs-P2b
  rank conflict, Top-1/Top-2 margin, peak/neutral-loss evidence, explanation
  confidence) and must be re-developed and re-locked on a fresh split.

Until that gate is built and re-validated, the safe production posture is:
**use DreaMS retrieval as the default, and only enable P2b fusion when the
candidate group is known not to contain near-isomer ambiguity** (or gate it
behind an out-of-band structure check, which is then a structure-aware service,
not a label-free model).

## Honest framing

The validated claim is narrow and specific:

> "On the sealed 3,000-query main panel, the frozen P2b local rank fusion raised
> Recall@1 by 1.07 pp (formula-cluster bootstrap CI [+0.24, +1.89] pp,
> McNemar p=0.010), with the gain driven by neutral-loss evidence.  It is not a
> new DreaMS checkpoint, does not fix near-isomer retrieval, and is not SOTA."

Do **not** claim an overall "better than DreaMS" encoder, a near-isomer
solution, or a fully-gated production system from this package.

## Rebuild / verification

The frozen config provenance (SHA256 of cache, selection report, frozen
artifact, HDF5, sealed P3 lock summary) is recorded in `deploy/p2b_config.json`
and `deploy/p2b_rank_fusion.py` (`FROZEN_CONFIG`).  Any change to the weights,
normalization, or gate requires a new development split and a new sealed test.
