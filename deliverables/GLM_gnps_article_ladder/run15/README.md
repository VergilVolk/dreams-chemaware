# run15 交付目录——工件地图（2026-10-07）

15 方法 GNPS 模型盲基准（identity-disjoint 10,995 q / formula-disjoint 5,261 q）的全部冻结产物。
数据源：`data/validation/GLM_gnps_article_benchmark_s2v26/run15/`（bundle + evaluation）。

## 主张 → 工件对照

| 主张 | 工件 | 关键数字 |
|---|---|---|
| 15 方法梯队（双源验证） | `ladder_full.csv` + `verification_report.json` | WSE R@1 87.37/88.27；我方 pooled AUROC 0.9416/0.9299 |
| metric split（判别≠检索） | `../../figures/GLM_gnps_metric_split.{svg,pdf,png}` | R@1 冠军 ≠ AUROC 冠军，双面板成立 |
| spec2vec_2026 低于 2020 版 | `ladder_full.csv` 行 `spec2vec_2026_retrained` | 86.08/87.25 vs 86.29/87.44 |
| oracle 互补性存在 | `truthblind_ensemble.json` `_oracle_any` | 93.25/94.58 = +5.88/+6.31pp |
| 无监督 truth-blind 全灭 | `truthblind_ensemble.json` U1–U6 | 全部 CI 跨零，≤±0.25pp |
| U1 vs 每个单方法（配对CI） | `u1_vs_each_method.json` | 显著胜 11-12/15，对顶部集群 n.s.，从不显著落败 |
| 贡献结构（口径修正） | `audit_unique_wins.json` | 胜者成员数第一：noise_v1（287/161）；唯一命中第一：ms2deepscore（65/24，net为负） |
| 泄露清除 router ≈10% 空间 | `router_v3.json` | B +0.61 [+0.15,+1.06]、C +0.51 [+0.09,+0.95]、A +0.23 n.s. |
| 谱特征通道关闭 | `spectrum_router.json` | 谱特征S≈0；S+C vs C：+0.09/0/−0.03pp=无增益 |
| router 信号为真（非伪影） | `router_permutation.json` | 置换 −0.91/−0.54 vs 真实 +0.23/+0.61 |
| gap 校准好（方法内） | `gap_calibration.json` | AUC 0.86–0.93；Q5 ≈99.3–100% |
| 赢家不自知（诊断，含对照审核） | `audit_winner_confidence.json` + `risk_coverage.json` | R集内赢家vs输家 gap AUC 0.449/0.437（vs 全集校准0.905）；φ 0.762/0.748 |
| 门控（幸存部署物） | `risk_coverage_ci.json` + `../run_local/gate_noise_v1_*.json` + `tasks/GLM_confidence_gate.py` | WSE c60 99.41 [99.23,99.59] / 99.52 [99.27,99.75] |
| 门控故事图 | `../../figures/GLM_gating_story.{svg,pdf,png}` | risk-coverage + 赢家百分位直方图 |
| 论文结果章节草稿 | `docs/GLM_ENSEMBLE_RESULTS_SECTION_DRAFT_20261007.md` | 英文成稿语言 + 表图计划 |
| 撤回与审计痕迹 | `docs/GLM_ENSEMBLE_COMPLEMENTARITY_RETRACTION_AND_TRUTHBLIND_RESULT_20261007.md` + `../run_local/s1_validation.json`（RETRACTED 标记） | 两次泄露的完整机制 |
| SOTA 审计 | `docs/GLM_SOTA_AUDIT_20261007.md` | 判别双冠；检索平手（CI 跨零） |

## 14 方法版（历史保留）

`../run_local/` 为 14 方法版（spec2vec_2026 之前），其 `truthblind_ensemble.json`/`router_v3.json`/`gap_calibration.json`/`risk_coverage.json` 同样有效且结论一致；`s1_validation.json`/`complementarity_mining.json` 含 RETRACTED 标记。

## 边界

- 集成/路由不得宣称超过最佳单方法（除 router 的 ~+0.5pp 显著增量，2/3 分割）。
- MoNA 极性外部验证本地受阻（谱本体+checkpoint 在服务器），见撤回文档 §9。
