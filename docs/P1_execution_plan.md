# P1: Rule-Decodable Chemical Metric Fine-Tuning — Execution Plan

**Date**: 2026-08-06
**Status**: Plan recorded, awaiting E0 start
**Prerequisite**: Task 0 complete — global Rule-Jaccard alignment is dead as primary loss

---

## Core Decision

> Rule Jaccard is a useful coarse-grained chemical semantic signal,
> but NOT a continuous structural distance ruler.

Pearson r ≈ -0.17 to -0.19 is directionally correct but too weak for cosine regression.
Δ Jaccard (0.113–0.155) and 1.31–1.54× discrimination ratio show rules DO distinguish broad regions.

**Rules change role, not get abandoned:**

| Original design | Decision |
|---|---|
| Rule-Jaccard as global embedding distance | **REMOVED** from primary loss |
| Rules as chemical concept supervision | **KEPT & UPGRADED** — linearly decodable directions |
| Rules for hard sample mining | **STRENGTHENED** — high-Jaccard + high-MCES = best hard negatives |
| Rules for interpretability | **KEPT** |
| 335 main rules | Core interpretable set |
| 3,486 rules (incl. MassBank) | Hierarchical auxiliary (needs filtering) |
| Module 1 / Module 2 architecture | **KEPT** — only training logic changes |

---

## New Loss Function

```
L_total = L_identity + λ_m·L_MCES-rank + λ_r·L_rule-decode + λ_p·L_preserve
```

Initial weights: λ_m = 0.3, λ_r = 0.1, λ_p = 0.05

### L_identity — Identity Discrimination

Triplet loss keeping same/different retrieval capability:

```
L_identity = max(0, m - s(a, p) + s(a, n))
```

- **a**: anchor spectrum
- **p**: another spectrum of same 14-char InChIKey
- **n**: mass-close but different molecule
- **s**: cosine similarity of normalized embeddings

Hard negative priority:
1. Same molecular formula, different InChIKey
2. Precursor m/z difference < 0.05 Da
3. High rule-Jaccard but high MCES
4. Currently misclassified as similar by model

### L_MCES-rank — MCES Ordinal Preservation

Given anchor a, if MCES(a, p) + δ < MCES(a, n), require s(a, p) > s(a, n):

```
L_MCES-rank = max(0, m - s(a, p) + s(a, n))
```

- Positive = "relatively more similar structure" (not necessarily same molecule)
- Only use MCES gaps large enough to avoid noise:
  - **Strong**: gap ≥ 4
  - **Medium**: gap 2–3
  - **Gap 0–1**: excluded from training (evaluation only)

### L_rule-decode — Rule-Decodable Chemical Semantics

Linear probe from embedding → rule match vector:

```
R̂ = σ(Wz + b)
L_rule-decode = MaskedWeightedBCE(R̂, R)
```

Three-state labels:
- **1**: rule definitively matched
- **0**: rule testable under current spectrum conditions but NOT matched
- **unknown**: insufficient spectral quality/peak range/metadata → excluded from loss

Class weight: w_k = 1 / sqrt(freq(k) + ε)

### L_preserve — Original Embedding Protection

```
L_preserve = 1 - cos(z, z₀)
```

where z₀ = frozen original DreaMS embedding.

---

## Experimental Sequence

### E0: Original DreaMS Fixed Baseline

**Goal**: Establish common zero point for all subsequent experiments.

Same data, same split. Record:
- 10 ppm difficult retrieval AUC
- Recall@1/5/10
- Cosine vs MCES Spearman/Pearson
- MCES binned correlation
- Isomer retrieval
- 335-rule and 3,486-rule linear probe AUPRC

**Stop condition**: Must complete before any training. This is the reference.

---

### E1: DreaMS-Style Difficult Triplet

**Goal**: Establish credible fine-tuning baseline with identity loss only.

```
L = L_identity
```

- Negatives must be mass-near, NOT random different molecules
- Confirm data pipeline and training loop work
- Rule out "training itself is broken before adding rules"

**Pass criteria**:
- Difficult retrieval significantly better than E0 zero-shot
- Effective triplets observed during training (non-zero loss, margin violations)
- No embedding collapse (check pairwise cosine distribution)
- Three random seeds directionally consistent

**Fail → fix sampling & training pipeline first.**

---

### E2: Add MCES Ordinal Preservation

**Goal**: Check if structure ordering can be improved before adding rules.

```
L = L_identity + 0.3·L_MCES-rank
```

**Pass criteria** (vs E1):
- MCES Spearman shows stable positive improvement
- Bootstrap CI of Δρ does not cross zero
- Difficult retrieval not significantly degraded
- Local ordering in MCES 1–5 region improved, not just global number shift

**Fail → check MCES triplet construction; do NOT proceed to rule training.**

---

### E3: Add 335-Rule Decodable Supervision

**Goal**: First true ChemRule-DreaMS model — can rules inject chemical semantics without hurting retrieval?

```
L = L_identity + 0.3·L_MCES-rank + 0.1·L_rule-decode + 0.05·L_preserve
```

**Pass criteria** (vs E2):
- Rule macro AUPRC significantly improved
- At least one batch of well-supported rules stably linearly decodable
- 10 ppm retrieval AUC drop ≤ ~0.005
- MCES ordering not significantly degraded
- Embedding drift within controllable range

**If rule decode improves but retrieval drops**: lower λ_r, or escalate to dual-subspace design.

**If rule decode improves, retrieval unchanged**: E3 is still valuable — embedding gained extra chemical semantics.

---

### E4: Rule-Structure Conflict Sample Mining

**Goal**: Use rules to find the hardest samples, let structure decide the geometry.

Add these hard samples:
- High Jaccard + high MCES: shared fragmentation but different skeleton
- High Jaccard + same formula + different InChIKey
- Low Jaccard + low MCES: structurally close but rule observation differs
- Current model high similarity + rule evidence clearly conflicts

**Rules only discover hard samples; they do NOT decide final distance.**

**Pass criteria** (vs E3):
- High-Jaccard/high-MCES hard negative misclassification reduced
- Isomer retrieval improved
- Normal samples not significantly degraded

**This is likely the key experiment determining whether Module 1 is truly innovative.**

---

### E5: Add Filtered MassBank Rules

**Goal**: Add empirical fragmentation patterns after filtering for quality.

Filter criteria (training-set statistics only):
- Covers enough independent molecules (not just many spectra of few molecules)
- Reproducible across instruments/collision conditions
- Not a single-molecule memorized pattern
- Predictable on held-out molecules
- Clear peak or mass-diff evidence returnable

Two heads:
```
L_rule = L_mechanistic_335 + η·L_MassBank
```
η initial = 0.2–0.3

335 = "mechanistic rules"; MassBank = "empirical fragmentation patterns" (named honestly).

**Pass criteria** (vs E4):
- More stably decodable concepts, OR better hard-sample performance, OR higher explanation coverage
- If only training-set rule AUC rises, no held-out improvement → **memory/leakage → ABANDON extended rule set.**

---

### E6: Old Jaccard Global Alignment (Negative Control)

**Goal**: Scientific control — prove the granularity mismatch analysis is correct.

```
L = L_identity + λ·L_Jaccard-align
```

**Not a candidate model.** It validates the architecture decision if it causes:
- Structure ordering degradation
- High-Jaccard different skeletons wrongly pulled together
- Isomer discrimination degradation

---

## Batch Sampling Ratios

Per-batch composition:
- 35%: same-molecule positives + mass-near negatives
- 25%: MCES-ordered structural triplets (gap ≥ 2)
- 20%: same-formula isomers
- 20%: rule-structure conflict samples

Rule multi-label supervision computed on all individual spectra in batch — no pair construction needed.

---

## Model Unfreezing Strategy

### Stage 1: Projection Head Warmup
- Freeze DreaMS
- Train 1024-dim residual projection: `z = Norm(h + α·P(h))`
- Simultaneously train temporary rule head
- 1–2 epochs

### Stage 2: Unfreeze Top Layers
- Unfreeze Transformer layers 6–7
- Unfreeze final LayerNorm and precursor aggregation path
- Freeze layers 1–5
- Projection LR ≈ 5–10× upper Transformer LR
- Gradient clipping
- Evaluate 4 embedding properties per epoch

### Stage 3: End-to-End Control
- If top-layers-only is stable, run small-scale end-to-end control
- DreaMS paper uses end-to-end; freezing bottom 5 layers is parameter-efficient, not necessarily optimal

---

## Task 0 Supplementary Analysis (Lightweight, Non-blocking)

Before/during E0, complete:
1. Spectrum coverage of 335 and 3,486 rules
2. Independent molecule support per rule (not spectrum support)
3. Per-category correlation: NL / CF / functional group / rearrangement / MassBank empirical
4. Δ Jaccard within same-formula isomer subset
5. Partial correlation controlling precursor mass, peak count, collision conditions
6. Manual inspection of high-Jaccard/high-MCES conflict samples
7. MassBank rule provenance check — any overlap with test set data source?

**Goal**: Determine which rules suit concept supervision, which suit hard mining, which suit explanation display only.

---

## Execution Order

```
1. [ ] Finalize Task 0 conclusions (FINAL_ANALYSIS.md updated)
2. [ ] Task 0 supplementary analysis (7 items above)
3. [ ] E0: Original DreaMS fixed baseline
4. [ ] E1: Difficult triplet fine-tuning
5. [ ] E2: + MCES ordinal preservation
6. [ ] E3: + 335-rule decodable supervision  ← First ChemRule-DreaMS
7. [ ] E2 vs E3 comparison: did rules add chemical semantics without hurting retrieval?
8. [ ] E4: + Rule-structure conflict sample mining
9. [ ] E5: + Filtered MassBank rules (decide whether to include)
10. [ ] E6: Old Jaccard alignment as negative control
11. [ ] Module 1 stable → Module 2 peak evidence & faithfulness experiments
```

---

## Questions Log

Questions to resolve during execution (ask user, don't assume):

- [ ] E0: Which DreaMS checkpoint? Which data split? Where is the frozen model?
- [ ] E0: What is the "10 ppm difficult retrieval" eval protocol exactly?
- [ ] E1: What margin value for L_identity? Default triplet margin?
- [ ] E2: MCES computation at training scale — precompute or online? Caching strategy?
- [ ] E3: How to determine "unknown" vs "0" for 3-state rule labels?
- [ ] E3: Rule head dimension = 335? Or per-category heads?
- [ ] E3: α initial value for residual projection?
- [ ] E4: Conflict sample mining — online (per epoch) or offline (precomputed pool)?
- [ ] E5: MassBank rule filtering thresholds — molecule support minimum? Reproducibility metric?
- [ ] Training infrastructure: Which GPU(s)? Mixed precision? Effective batch size?
