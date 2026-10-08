# 项目文档总索引（2026-10-08 更新）

**用途：** 一页找到所有文档。按时间倒序、按研究线分组。

## 一、谱图坐标代谢组学（当前主线）

| 文档 | 内容 | 状态 |
|---|---|---|
| **GLM_SPECTRAL_COORDINATE_COMPLETE_DISCOVERY_CHAIN_20261008.md** | **P0→P2 完整发现链（最重要）** | ✅ 最新 |
| SPECTRAL_COORDINATE_METABOLOMICS_P0_RESULT_20261007.md | P0-L 坐标映射结果（6 门全过） | ✅ |
| SPECTRAL_COORDINATE_METABOLOMICS_P0_CONTRACT_20261007.md | P0 预注册合同 | 冻结 |
| LCNEC_P2_SPECTRAL_COORDINATE_RUN_2353437_AUDIT.md | P2 v1 审计（实现失败记录） | 已处置 |

**核心发现：**
- P0-L：坐标映射可行（跨仪器 R@1 84%+）
- P0-G：绝对分数 ≠ 置信度（AUC 0.507）→ **方法论发现**
- P0-M：边际 = 置信度（AUC 0.848）→ **部署级发现**
- P2：LCNEC 91% 暗物质（margin < 0.1）→ **精确量化**

## 二、算法整合（15 方法天梯 + 统一算法）

| 文档 | 内容 |
|---|---|
| GLM_GNPS_ARTICLE_LADDER_20261007.md | 15 方法天梯（最新版） |
| GLM_SOTA_AUDIT_20261007.md | SOTA 审计 |
| GLM_ALGORITHM_INTEGRATION_PACKAGE_20261005.md | 算法整合包 |
| GLM_ENSEMBLE_COMPLEMENTARITY_RETRACTION_AND_TRUTHBLIND_RESULT_20261007.md | 融合撤回 + 真值盲重审 |
| GLM_ENSEMBLE_RESULTS_SECTION_DRAFT_20261007.md | 融合结果论文草稿 |
| UNIFIED_ALGORITHM_POST_GATE1_ADVERSARIAL_AUDIT_20261007.md | 统一算法对抗审计 |
| GLM_UNIFIED_ALGORITHM_PHASE1_HANDOFF_20261007.md | Phase-1 交接文档 |

## 三、LCNEC 生物学应用

| 文档 | 内容 |
|---|---|
| GLM_LCNEC_DARK_THREE_TIER_CLAIMS_20261007.md | 30 暗模块三级主张（blank 污染审计） |
| GLM_LCNEC_FLUORINATED_HITS_MEDICATION_CROSSCHECK_SPEC_20261006.md | 含氟命中用药史核查规格 |
| PHENOTYPE_CONDITIONED_CHEMICAL_TRANSFORMATION_PROGRAM_20261006.md | 化学转换程序合同（Stage-1d PASS） |
| GLM_GLOBAL_REMODELING_STAGE2_CONTRACT_20261007.md | Stage-2 冻结合同 |
| GLM_REMODELING_METHOD_DEVELOPMENT_REPORT_20261007.md | 重塑方法开发报告 |

## 四、方法学探索历史（按时间倒序）

| 文档 | 尝试 | 结局 |
|---|---|---|
| ATLAS_ANCHORED_GLOBAL_CHEMICAL_SPACE_ASSOCIATION_RESET_20261006.md | 化学空间关联 | 被坐标方法替代 |
| SPECTRUM_ANCHORED_PERTURBATION_RESPONSE_IDENTITY_20261005.md | 扰动响应指纹 | 数据墙阻断 |
| SPECTRUM_ANCHORED_PERTURBATION_EVIDENCE_GATE_REVIEW_20261005.md | 证据门审查 | 三门前置 |
| URSEA_FINAL_METHOD_SPEC_20261005.md | URSEA 统计框架 | 降级为 QC 附件 |
| URSEA_ADVERSARIAL_REVIEW_AND_METHOD_REDESIGN_20261005.md | URSEA 对抗审查 | 概念股诊断 |
| URSEA_NOVELTY_AND_FEASIBILITY_AUDIT_20261005.md | URSEA 新颖性审计 | |
| CLAIM_IDENTIFIABILITY_GUIDED_UNTARGETED_METABOLOMICS_20261005.md | 可辨识主张分层 | 降级为 QC |
| CORE_METHOD_REDESIGN_CANDIDATE_DIFFERENTIAL_EVIDENCE_20261005.md | 候选差分证据 | Gate-1 否证 |
| GRAND_FUSION_INTEGRATION_DECISION_20261004.md | 三层融合决策 | MSG 融合外测负 |

## 五、项目总账与审计

| 文档 | 内容 |
|---|---|
| GLM_PROJECT_OVERVIEW_20260930.md | 全课题统筹总览（9/30） |
| GLM_POSITIVE_RESULTS_AND_ALGORITHM_DOSSIER_20260930.md | 正向结果与算法案卷 |
| GLM_TRACK2_FINAL_REPORT_20261006.md | 任务 2 终版覆命 |
| GLM_TRACK2_STATUS_AND_CENSUS_20261005.md | 任务 2 状态与普查 |
| GLM_TRACK2_SHIFT2_ADDENDUM_20261005.md | 值守班增补（含暗特征反向搜索） |

## 六、关键数据工件路径

| 工件 | 路径 |
|---|---|
| GNPS 基准（52,871 谱） | data/validation/gnps_gold_silver_10ppm_benchmark_v1/ |
| 15 方法分数包 | data/validation/GLM_gnps_article_benchmark_s2v26/run15/bundle/ |
| P0-L 坐标映射报告 | data/validation/GLM_p0_spectral_coordinate_mapping/run_2353434/ |
| P0-G 弃权测试 | data/validation/GLM_p0g/ |
| P0-M 边际弃权 | data/validation/GLM_p0m/ |
| P2 v2 LCNEC 坐标投影 | data/validation/lcnec_p2v2/run_2354362/ |
| Stage-1 重塑结果 | data/validation/lcnec_global_remodeling_stage1/run_2352292/ |
| Stage-1d 患者bootstrap | data/validation/lcnec_global_remodeling_stage1d/run_2353147/ |
| LCNEC 暗模块全量重扫 | data/validation/GLM_track2_census/lcnec_dark_full_rescan.json |
| 天梯终版 | deliverables/GLM_gnps_article_ladder/ |

## 七、参考论文（docs/papers/）

- Quartet metabolite reference materials (Genome Biology 2024)
- Schuhknecht metabolic map (Nature Biotechnology 2025)
- 其余 10 篇见 docs/papers/ 目录
