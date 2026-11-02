# Public embedding checkpoints on the strict GNPS common domain

## Decision boundary

This is an **18-method closed-set comparison on consumed GNPS development panels**. It is not an unopened external benchmark, an open-set result, or a field-wide SOTA claim. SpecEmbedding and MSBERT were evaluated with their authors' released checkpoints and official preprocessing, but their original training data/objectives are not identical to this 10-ppm identity-ranking protocol.

MSBERT's official loader drops an entire spectrum when precursor m/z is at least 1000. No peak was clipped and no embedding was fabricated. A query was retained only when its query spectrum and every candidate reference spectrum in its complete candidate block were supported. The existing 16-method score matrices were then sliced exactly by `source_molecule_index`; scores were not recomputed or imputed.

## Common-domain denominators

| Panel | Original N_Q / N_E / N_P | Retained N_Q / N_E / N_P | Dropped queries | Retained queries |
|---|---:|---:|---:|---:|
| Identity-disjoint | 10,995 / 87,518 / 175,171 | **10,983 / 87,494 / 175,107** | 12 | 99.8909% |
| Formula-disjoint | 5,261 / 24,401 / 47,724 | **5,253 / 24,385 / 47,682** | 8 | 99.8479% |

All 20 dropped panel-query records (12 plus 8; panels may overlap) were outside the official MSBERT precursor domain. There were zero additional drops caused by an incomplete candidate block.

## Same-axis result

| Panel / method | R@1 | R@2 | R@3 | R@5 | R@10 | R@20 | MRR | Macro AUROC | Micro AUROC | Micro AP | Near R@1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Identity / Full23 v1.3.2 | **88.400%** | **94.009%** | **95.939%** | **97.505%** | **98.880%** | **99.754%** | **0.9247** | **0.9497** | **0.9587** | **0.8445** | **83.870%** |
| Identity / WSE | 87.417% | 93.007% | 95.320% | 97.004% | 98.698% | 99.672% | 0.9168 | 0.9437 | 0.9423 | 0.7566 | 82.577% |
| Identity / MSBERT | 81.899% | 89.985% | 93.435% | 96.030% | 98.306% | 99.727% | 0.8812 | 0.9183 | 0.8947 | 0.5408 | 75.481% |
| Identity / SpecEmbedding | 75.262% | 85.887% | 90.922% | 94.956% | 97.970% | 99.599% | 0.8374 | 0.8915 | 0.8563 | 0.4994 | 66.843% |
| Formula / Full23 v1.3.2 | **89.644%** | **95.755%** | **97.658%** | **98.972%** | **99.695%** | 99.962% | **0.9376** | **0.9470** | **0.9539** | **0.8744** | **82.369%** |
| Formula / WSE | 88.311% | 94.517% | 97.144% | 98.686% | 99.657% | 99.962% | 0.9281 | 0.9394 | 0.9285 | 0.7811 | 80.157% |
| Formula / MSBERT | 83.571% | 92.442% | 96.155% | 98.172% | 99.543% | **100.000%** | 0.8994 | 0.9124 | 0.8606 | 0.5569 | 73.315% |
| Formula / SpecEmbedding | 79.288% | 89.987% | 94.384% | 97.544% | 99.315% | 99.943% | 0.8714 | 0.8851 | 0.8249 | 0.5407 | 66.814% |

Full23 v1.3.2 ranks first by R@1 on both common panels. Relative to MSBERT, its descriptive R@1 advantage is **+6.501 percentage points** on identity-disjoint and **+6.073 percentage points** on formula-disjoint. Relative to SpecEmbedding, the corresponding differences are **+13.138** and **+10.356 percentage points**. MSBERT ranks 16th of 18 by R@1 on each panel; SpecEmbedding ranks 18th.

## Reproducible artifacts

- MSBERT embeddings: `data/validation/sota_identity_method_assets_20261102/msbert_gnps_embeddings.npz`, SHA-256 `59f31e0b906a5b3d3d701ef2848b703d91a7a6d3b5c70d27c910eeefbb05943a`
- Identity common panel: `data/validation/sota_identity_method_assets_20261102/panel_identity_msbert_common_domain.npz`, SHA-256 `1277537fdc9f5ff8133cd8a6c447d276983141025e2c6f0ead3c322b177566bb`
- Formula common panel: `data/validation/sota_identity_method_assets_20261102/panel_formula_msbert_common_domain.npz`, SHA-256 `514254b3d9bf09e0c442c00efdd63db58a4527abed9c8dd6b8e1e9ad207e4ba1`
- Merged 18-method caches: `data/validation/gnps_run18_common_domain_evaluation_inputs_20261102`
- Identity report, per-query ranks, exact curves, and figures: `data/validation/gnps_run18_common_domain_evaluation_v1_20261102/identity` (report SHA-256 `9d2b033a76227b1afe04e7f4e261e91255f0e9f811383d68774a3e88b2e95a51`)
- Formula report, per-query ranks, exact curves, and figures: `data/validation/gnps_run18_common_domain_evaluation_v1_20261102/formula` (report SHA-256 `f4dc42760401040c0534e0972fdc2602b7e71c0f57ecb2a3b4eefc814862e96c`)

Each evaluation directory contains the full Top-k table, macro and micro metrics, paired formula/identity-cluster intervals, exact ROC/PR curve CSVs, and PNG/PDF ladder, Top-k, ROC, and PR figures.
