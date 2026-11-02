# SpecEmbedding official checkpoint on the frozen GNPS same-axis panels

## Scope and evidence boundary

This is a **17-method, closed-set, consumed-development comparison**. It is not an external confirmation and it does not establish field-wide SOTA. The GNPS Gold/Silver panels were already used during method development. The public SpecEmbedding checkpoint may also have GNPS-related training exposure, and its published objective is not identical to this 10-ppm identity-ranking task. The result therefore answers only: how does the authors' released checkpoint rank the same frozen candidates under the same evaluator?

The exact authors' public demo checkpoint was used (SHA-256 `0ca0aa002a0d061a95410f7a4055e82c7fcb428d0ba04b5714ac3a4e7f0f5cca`), with its official top-99-peak plus precursor-token preprocessing. All 52,871 unique rows required by the two panels were encoded; no row was imputed or dropped.

## Same-axis result

| Panel / method | Queries | Candidate edges | R@1 | R@2 | R@3 | R@5 | R@10 | R@20 | MRR | Macro AUROC | Micro AUROC | Micro AP | Near R@1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Identity-disjoint / SpecEmbedding | 10,995 | 87,518 | 75.252% | 85.903% | 90.932% | 94.961% | 97.972% | 99.600% | 0.8374 | 0.8913 | 0.8560 | 0.4991 | 66.834% |
| Identity-disjoint / Full23 v1.3.2 | 10,995 | 87,518 | **88.367%** | **94.015%** | **95.944%** | **97.508%** | **98.881%** | **99.754%** | **0.9245** | **0.9493** | **0.9585** | **0.8441** | **83.825%** |
| Formula-disjoint / SpecEmbedding | 5,261 | 24,401 | 79.282% | 90.002% | 94.393% | 97.548% | 99.316% | 99.943% | 0.8714 | 0.8849 | 0.8248 | 0.5406 | 66.814% |
| Formula-disjoint / Full23 v1.3.2 | 5,261 | 24,401 | **89.603%** | **95.761%** | **97.662%** | **98.974%** | **99.696%** | 99.962% | **0.9374** | **0.9465** | **0.9538** | **0.8742** | **82.303%** |

SpecEmbedding ranks 17th of 17 by R@1 on both frozen panels. Relative to SpecEmbedding, Full23 v1.3.2 is higher by **+13.115 percentage points** on identity-disjoint and **+10.321 percentage points** on formula-disjoint. These are within-panel descriptive differences, not external-effect estimates.

## Reproducible artifacts

- Embedding cache: `data/validation/sota_identity_method_assets_20261102/specembedding_gnps_embeddings.npz` (SHA-256 `a03b8185d4d0b262e4acacbd5f2e2b20635480f7036dad09bf5a43133fae7a3a`)
- Identity score cache: `data/validation/sota_identity_method_assets_20261102/specembedding_identity_scores.npz` (SHA-256 `062d6121ca11359df3e3355008dd790b270b03a0d45063a9b45039c6a2ccf2e8`)
- Formula score cache: `data/validation/sota_identity_method_assets_20261102/specembedding_formula_scores.npz` (SHA-256 `e2b8fe2841aea1b0291aaf8af4f1f71874ad556d13919f9b8f70e3def374a537`)
- Identity evaluation and exact curves: `data/validation/gnps_run17_mega_evaluation_v1_20261102/identity`
- Formula evaluation and exact curves: `data/validation/gnps_run17_mega_evaluation_v1_20261102/formula`
- Versioned reports and publication figures: `deliverables/mega_sota_ladder/gnps_run17_specembedding_20261102`
- Each evaluation directory contains `summary_metrics.csv`, per-query ranks, paired comparisons, exact ROC/PR curve tables, and PNG/PDF ladder, Top-k, ROC, and PR figures.

MSBERT is intentionally excluded from this 17-method table until its official-domain common panel is complete. Its released implementation rejects precursor m/z at or above 1000; the later 18-method comparison must therefore use the strict common-domain panels rather than fabricating full coverage.
