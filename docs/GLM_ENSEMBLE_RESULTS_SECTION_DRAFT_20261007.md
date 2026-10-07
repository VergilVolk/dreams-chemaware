# Results-section draft: complementarity, its inaccessibility, and confidence gating (2026-10-07)

> 论文材料草稿。所有数字来自冻结工件（run15 15 方法梯队 + 真值盲套件），双源验证通过。
> 本文是 `GLM_ENSEMBLE_COMPLEMENTARITY_RETRACTION_AND_TRUTHBLIND_RESULT_20261007.md` 的投稿语言版。

---

## Draft text (English)

**Inter-method complementarity on a model-blind GNPS benchmark is real but not deployable by confidence routing; confidence instead supports abstention gating.**

We benchmarked fifteen spectral-similarity methods on two sealed, model-blind GNPS Gold/Silver panels (identity-disjoint: 10,995 queries; formula-disjoint: 5,261 queries; zero overlap to the MSG/MoNA training corpora of the learned methods; 10 ppm candidate graph, [M+H]+). The ladder spans classical measures (greedy/modified cosine, entropy similarity, weighted spectral entropy), public learned embeddings (official DreaMS, MS2DeepScore 2.x, Spec2Vec 2020, Spec2Vec 2026 retrained), a denoising-search pipeline, neutral-loss rerankers, and our noise-curriculum encoder variants.

*The metric split.* No single method dominates. Weighted spectral entropy (WSE) leads top-1 identification on both panels (87.37% / 88.27%), statistically tied with our best variant (87.13% / 88.25%; paired Δ CI [−0.11, +0.57] / [−0.49, +0.51]), while our encoders lead pairwise discrimination (pooled 10 ppm AUROC 0.9416 identity / 0.9299 formula) and formula-panel MRR. The 2026-retrained Spec2Vec *underperforms* its 2020 predecessor on top-1 (86.08% / 87.25%) despite a marginal AUROC gain — further evidence that discrimination and identification dissociate.

*Oracle headroom.* The union of all fifteen methods' top-1 answers reaches 93.25% / 94.58% — a +5.88 / +6.31 pp oracle headroom over the best single method, concentrated on the near-structure subset. On this recoverable set our encoder has the largest among-winner membership (287 of 647 identity recoverable queries; 161 of 332 on formula), ahead of MS2DeepScore (253/133), official DreaMS (249/133), and Spec2Vec 2020 (230/127); the most strictly unique wins belong to MS2DeepScore (65/24 queries where it alone is correct), which is net-harmful overall (introduced far exceeds corrected) — most headroom is shared wins, consistent with the error correlation below.

*Truth-blind selection and fusion capture none of it.* Six unsupervised, deployment-computable strategies — per-method selection by raw top1−top2 gap, by batch-standardized or batch-percentile gap, within-query z-score fusion, Borda fusion, and reciprocal-rank fusion — all yield point estimates within ±0.25 pp of the best single method with bootstrap CIs crossing zero. A supervised per-query router (gradient-boosted per-method correctness models over gap, top-1 score, cross-method top-1 agreement, consensus, candidate count, and precursor mass; leak-free: training queries' spectra and structures never appear in test), evaluated under three disjoint splits, reaches at most +0.61 pp (CI [+0.15, +1.06]) — roughly 10% of the oracle headroom. Learned weighted score fusion transfers +0.50 pp in one direction (CI [0.00, +1.01]) and is insignificant in the others. A label-permutation control (5 permutations; identical pipeline, shuffled training labels) collapses the router to −0.91 / −0.54 pp — actively harmful — confirming the small true gain is label-driven rather than a pipeline artifact.

*Why: within the headroom, confidence carries no winner signal.* Within each method, the top1−top2 gap is well calibrated (AUC 0.87–0.93 for predicting the method's own top-1 correctness; the top gap quintile exceeds 99.3% accuracy). But correctness is strongly correlated across methods (mean pairwise φ = 0.75–0.76): methods fail together. On queries where the best single method fails but some other method succeeds — exactly the oracle headroom — a within-query control shows winners are indistinguishable from losers by any gap measure (winner-vs-loser AUC 0.449/0.437 on raw gap, 0.485/0.476 on within-query gap rank, versus 0.905/0.908 for the global calibration contrast), and the gap-argmax selection hits only 31% of these queries. The headroom is thus concentrated where method self-confidence stops discriminating, which the selection/fusion experiments confirm empirically. We report this as a diagnostic of the benchmark's error structure, not as an independent discovery claim.

*What confidence is for: gating.* The same calibrated gap supports selective prediction: accepting the top-c fraction most confident queries yields 99.4–99.5% accuracy at 60% coverage and ≈99.6–99.7% at 50% for WSE and our encoder alike, versus 87–88% at full coverage. The abstained band (bottom gap quintile) has ≈50% accuracy — a coin flip — and is the natural routing target for orthogonal evidence (MSⁿ, chemistry-aware reranking, or manual curation). We release the gate as a training-free tool consuming any method's per-query margins.

*Two evaluation traps.* During this analysis we identified and retracted two misleading intermediate results, which we report as methodological warnings: (i) any confidence signal derived from the true positive's score (e.g., positive-vs-best-negative margin) leaks the answer key and reproduces the oracle bound; (ii) panels overlapping in query spectra silently convert cross-panel training into memorization — our identity- and formula-disjoint GNPS panels share 5,237 of the formula panel's 5,261 queries, so all cross-panel supervised numbers must deduplicate by query spectrum and structure.

## Table/figure plan

| Element | Source artifact |
|---|---|
| Table 1 — 15-method ladder (R@1, R@k, MRR, near R@1, pooled/micro AUROC, both panels) | `deliverables/GLM_gnps_article_ladder/run15/ladder_full.csv` + `verification_report.json` |
| Fig 1 — metric split scatter (R@1 × pooled AUROC, 15 methods, both panels) | `deliverables/figures/GLM_gnps_metric_split.{svg,pdf,png}` |
| Fig 2 — gating story: (A) risk-coverage; (B) winner-gap-percentile histogram | `deliverables/figures/GLM_gating_story.{svg,pdf,png}` |
| Table 2 — truth-blind strategies (6 unsupervised + router splits + learned fusion, Δ vs best single with CI) | `deliverables/GLM_gnps_article_ladder/run15/truthblind_ensemble.json`, `router_v3.json` |
| Table 2b — U1 vs EACH of the 15 methods (paired CI; significantly better than 11-12/15, never significantly worse) | `run15/u1_vs_each_method.json` |
| Table 3 — gap calibration (quintile accuracy + AUC per method) | `run15/gap_calibration.json` |
| Suppl — failure-band structure (winners' percentiles, φ, winner counts) | `run15/risk_coverage.json` |

## Methods addendum (for the same section)

**Benchmark.** GNPS Gold/Silver ([M+H]+, strict 10 ppm) candidate graphs were sealed before any model scoring; the identity-disjoint and formula-disjoint panels share no query identity or formula with the MassSpecGym/MoNA training corpora of the learned methods. Per-method molecule-level rankings follow the frozen evaluator semantics: candidate-pair scores are max-pooled per molecule and the unique positive molecule (always block-first) must strictly exceed every negative (ties count against the positive, float64). All per-method R@1 values were dual-source verified against the frozen per-query tables (exact match), and one full number chain (U1 selection, WSE baseline, oracle) was re-derived through an independent path (spot audit, PASS).

**Truth-blind protocol.** Selection and fusion signals use only deployment-computable quantities: per-method top1−top2 gaps, batch percentiles thereof, top-1 scores, cross-method top-1 agreement, modal-candidate consensus, candidate counts, and precursor m/z. The supervised router is a per-method gradient-boosted classifier (fixed a-priori hyperparameters: 300 iterations, learning rate 0.06, min 40 samples/leaf, L2 1.0) over these features, deployed as argmax predicted correctness. Leak-free splits remove all test-panel query spectra and structures from training (the two panels share 5,237/5,261 query spectra, so naive cross-panel training is memorization). Uncertainties are paired-by-query percentile bootstrap CIs (10,000 resamples); the router additionally carries a 5-permutation label-shuffle control.

**Gating.** Coverage-accuracy operating points sort queries by within-method gap (descending) and report cumulative top-1 accuracy among the top-c fraction; thresholds and CIs are emitted by the released tool (`GLM_confidence_gate.py`, resampling the selected population).

## Server queue notes (not in the paper)

- MoNA polarity external validation: stage-1 sbatch + scorer ready (`GLM_mona_15method_stage1.sbatch`), blocked on asset sync (public models up, MoNA MGFs already server-side, sealed panels upload).
- Task-vector 7-arm internal gate sbatch ready; grand router 2349964 log pull pending; B47 remediation pending — all server-side.

## Claim discipline

- May claim: metric split; oracle headroom existence + structure; negative result for truth-blind routing (with within-query-controlled diagnostics); risk-coverage gating; our encoders' discrimination crowns + parity on identification; noise_v1's largest among-winner membership on the recoverable set (not unique-wins; those belong to MS2DeepScore, net-harmful overall).
- Must not claim: any ensemble/routing reaching 92–94%; "fully SOTA"; leaked S1/router-v2 numbers (retracted, see audit-trail doc).
