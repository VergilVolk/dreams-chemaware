# 项目文档总索引（2026-10-12 更新）

**用途：** 一页找到所有文档。按时间倒序、按研究线分组。

> **2026-10-12 生物学方法定位正式冻结：** [GLM_BIOLOGY_METHOD_POSITION_DECLARATION_20261012.md](./GLM_BIOLOGY_METHOD_POSITION_DECLARATION_20261012.md) 确立本项目为"面向未知化学空间的无偏代谢组学发现范式"（Unbiased, structure-agnostic metabolomics discovery），冻结五条原则、三种入口、已封存路线和下一步优先级。后续所有实验设计必须与此对齐。

> **2026-10-11 生物学应用二次纠偏：** `UNBIASED_STRUCTURE_AGNOSTIC_DISCOVERY_CORE_20261011.md` 将总地基恢复为无需标签的可观测化学表示，并并列支持无标签样本、有设计响应和单谱查询。`BIOLOGY_APPLICATION_METHOD_SCIENTIFIC_DECISION_20261011.md` 中的 response transport 降为有监督算子，不再是唯一主方向。

> **2026-10-11 首个真实 M1 结果及纠偏：** `CHEMPROP2_RESPONSE_REPRODUCIBILITY_M1_RESULT_20261011.md` 在 MSV000094899 的 12 drugs x 3 replicate、36-query development/consumed 评测中确认 unresolved MS1 response 高于坐标打乱，但检索未使用 MS/MS 峰或 embedding；unknown 相对 library-supported 的增量不耐受 MQScore 0.70。删除含母药节点的 22 个 FBMN 成分后 R@1 仍为 38.89%，但这仍不是谱学算法增量证据。

> **2026-10-11 ChemProp2 两项敏感性：** `audit_chemprop2_parent_drug_dependence.py` 删除含母药节点的 FBMN 成分；`audit_chemprop2_matched_unresolved.py` 在 MQScore 0.80 下对总体强度、检测率、QC CV、缺失和 MS2 峰数作 283-vs-283 匹配。matched unresolved R@1 41.67%、MRR 0.5379，但 score 0.70 身份阈值失败仍使 unknown 独立增量保持不通过。

> **2026-10-11 ChemProp 跨设计 M1b：** `CHEMPROP_CROSS_DESIGN_M1B_RESULT_20261011.md` 的正式 v4 在共同 `10 ppm ∩ endpoint QC ∩ nonempty MS/MS` 图上完成 36-query development/consumed 对照。随机期望 16.67%；mass 与 WSE 均为 7/36，cosine 0.70 为 10/36、相对 mass +8.33 pp 但药物簇 CI `[-2.78,+19.44] pp`。全部方法 FAIL，relation transport 不启动，ChemProp 路线封存。

> **2026-10-12 ST003325 事件级扰动裁决：** `ST003325_FEATURE_SPECTRUM_READINESS_RESULT_20261012.md` 确认 3,080 个 QC 合格 features、804 个 replicated-MS2 entity links 和 760 个独立 spectral units；但 `ST003325_STRUCTURE_HIDDEN_PERTURBATION_RESULT_20261012.md` 的冻结跨编辑机制任务仅 `10/27 = 37.04%`，CE/KO/PTC 均未超过 50%，2,000 次 clone-label permutation `p=0.7126`。正式 FAIL；不在此终点继续叠加 Noise/P2b/公共层。

> **2026-10-12 遗传反事实新入口：** `MSV000096214_READINESS_RESULT_20261012.md` 在未打开 outcome 的条件下确认 49 个完整正负离子样本、10 genotype、98/98 研究文件含 1,095,667 MS2，及 66,958 pooled-QC MS2。`MSV000096214_GENETIC_CHEMICAL_RESPONSE_CONTRACT_20261012.md` 冻结三项真值：tt7 等位留出、路径特异性、tt7×tt4 双突变上位/救援；Atlas 不是方法前提。

> **2026-10-12 遗传反事实首个正控：** `MSV000096214_STRUCTURE_HIDDEN_MS1_RESULT_20261012.md` 在作者处理的 920-feature、49-sample development/consumed 表中隔离全部注释后，真实 tt7 集合在 20 种 genotype-cluster 分配中唯一第一，12 个 pairwise AUROC 最差为 1.00、精确 `p=0.05`，3/3 双突变按预注册方向塌缩到 tt4；五个 QC/极性世界均复现。该结果只授权进入 raw truth-blind 重建，不是 MS/MS 或未知实体增量。`GLM_MSV000096214_STRUCTURE_HIDDEN_CONTRACT_20261012.md` v1.3 已修复 double-log 与 90 次伪重复推断。

## 一、谱图坐标代谢组学（当前主线）

| 文档 | 内容 | 状态 |
|---|---|---|
| **ALGORITHM_INTEGRATION_AND_BIOLOGY_CURRENT_STATE_20261009.md** | **算法整合、生物学边界、撤回项与下一步的权威状态页** | ✅ 当前入口 |
| GLM_SPECTRAL_COORDINATE_COMPLETE_DISCOVERY_CHAIN_20261008.md | P0→P2 历史发现链；P0-G 与 P2 数字已被后续实现修复部分推翻 | ⚠️ 不可单独引用 |
| SPECTRAL_COORDINATE_METABOLOMICS_P0_RESULT_20261007.md | P0-L 坐标映射结果（6 门全过） | ✅ |
| SPECTRAL_COORDINATE_METABOLOMICS_P0_CONTRACT_20261007.md | P0 预注册合同 | 冻结 |
| LCNEC_P2_SPECTRAL_COORDINATE_RUN_2353437_AUDIT.md | P2 v1 审计（实现失败记录） | 已处置 |

**当前裁决：**
- P0-L：GNPS consumed-development 上支持跨仪器坐标可行；不是独立外测
- P0-G：旧 no-match 未移除阳性，旧 AUC 失效；以修复后的 entity-recovery FDR-coverage 为准
- P0-M：margin AUC 约 0.848 是开发线索，但 selective risk 尚不足以称部署级
- P2：旧 entropy 实现有关键错误，23 medium/high 与 91% dark 均待重算

## 二、算法整合（15 方法天梯 + 统一算法）

| 文档 | 内容 |
|---|---|
| **ALGORITHM_INTEGRATION_AND_BIOLOGY_CURRENT_STATE_20261009.md** | **所有模块的当前准入、数据角色、泄漏状态和生物学边界** |
| **NOISE_CHEMAWARE_SINGLE_ENCODER_MERGING_RESEARCH_20261009.md** | **Noise→ChemAware 负结果、MassSpecGym 停用、Fisher/TIES 单编码器合并调研与 GNPS→Enveda 执行方案** |
| UNIFIED_RECENT_BENCHMARK_POSITIVE_LEDGER_20261009.md | 近期 benchmark 正/零/负证据账本 |
| GLM_GNPS_ARTICLE_LADDER_20261007.md | 15 方法天梯（最新版） |
| GLM_SOTA_AUDIT_20261007.md | SOTA 审计 |
| GLM_ALGORITHM_INTEGRATION_PACKAGE_20261005.md | 算法整合包 |
| GLM_ENSEMBLE_COMPLEMENTARITY_RETRACTION_AND_TRUTHBLIND_RESULT_20261007.md | 融合撤回 + 真值盲重审 |
| GLM_ENSEMBLE_RESULTS_SECTION_DRAFT_20261007.md | 融合结果论文草稿 |
| UNIFIED_ALGORITHM_POST_GATE1_ADVERSARIAL_AUDIT_20261007.md | 统一算法对抗审计 |
| UNIFIED_ALGORITHM_DATA_LEAKAGE_LEDGER_20261009.md | 统一算法泄漏总账、Enveda fail-closed 构建与一次性开启边界 |
| GLM_UNIFIED_ALGORITHM_PHASE1_HANDOFF_20261007.md | Phase-1 交接文档 |

## 三、LCNEC 生物学应用

| 文档 | 内容 |
|---|---|
| **BIOLOGY_APPLICATION_FRONTIER_RESET_20261010.md** | **生物学应用主线重置：实体效应迁移、known+unknown 增量和前瞻功能三角验证** |
| **LCNEC_ENTITY_EFFECT_TRANSPORT_P0_CONTRACT_20261010.md** | **LCNEC 同患者跨平台实体效应迁移：冻结顺序、联合检验和 fail-closed 实现** |
| **LCNEC_ENTITY_EFFECT_TRANSPORT_P0_READINESS_RESULT_20261010.md** | **LIPn 表型盲实体门通过；HSST3n→LIPn 仅 6/263 有 5 ppm 候选，全局运输 headroom 失败** |
| GLM_LCNEC_DARK_THREE_TIER_CLAIMS_20261007.md | 30 暗模块三级主张（blank 污染审计） |
| GLM_LCNEC_FLUORINATED_HITS_MEDICATION_CROSSCHECK_SPEC_20261006.md | 含氟命中用药史核查规格 |
| PHENOTYPE_CONDITIONED_CHEMICAL_TRANSFORMATION_PROGRAM_20261006.md | 化学转换程序合同（Stage-1d PASS） |
| GLM_GLOBAL_REMODELING_STAGE2_CONTRACT_20261007.md | Stage-2 冻结合同 |
| GLM_REMODELING_METHOD_DEVELOPMENT_REPORT_20261007.md | 重塑方法开发报告 |

## 三-B、通用生物学方法与事件级扰动

| 文档 | 内容 | 状态 |
|---|---|---|
| **UNBIASED_STRUCTURE_AGNOSTIC_DISCOVERY_CORE_20261011.md** | **结构非依赖总纲；Atlas/response transport 均非唯一主线** | 当前方法入口 |
| BIOLOGY_APPLICATION_METHOD_SCIENTIFIC_DECISION_20261011.md | 三类科学问题、强基线与否定条件 | 当前边界 |
| PUBLIC_SPECTRAL_EVIDENCE_SPACE_CORE_20261011.md | 公共谱学证据的可选层；失败可从生产方法删除 | 可选开发层 |
| ST003325_FEATURE_SPECTRUM_READINESS_RESULT_20261012.md | 3,080 QC features、804 entity links、760 spectral units | readiness PASS |
| ST003325_STRUCTURE_HIDDEN_PERTURBATION_CONTRACT_20261012.md | clone/edit-class 留出与 plate-confounding 边界 | 冻结 |
| **ST003325_STRUCTURE_HIDDEN_PERTURBATION_RESULT_20261012.md** | **MS1-all 10/27，三 edit class 均未过，停止谱学追加** | **正式负结果** |
| MSV000096214_READINESS_RESULT_20261012.md | 49 paired samples、10 genotype、1,095,667 study MS2 | readiness PASS |
| **MSV000096214_GENETIC_CHEMICAL_RESPONSE_CONTRACT_20261012.md** | **等位留出、路径特异性、双突变救援；注释后置** | **冻结主实验** |
| **MSV000096214_STRUCTURE_HIDDEN_MS1_RESULT_20261012.md** | **作者处理表上的遗传反事实正控；20 种簇分配精确 p=0.05** | **development 正控 PASS** |
| **GLM_MSV000096214_STRUCTURE_HIDDEN_CONTRACT_20261012.md** | **raw mzXML 表型盲重建；v1.3 修复 double-log 和伪重复** | **冻结 raw 主线** |

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
| 公共谱学证据空间核心 | docs/PUBLIC_SPECTRAL_EVIDENCE_SPACE_CORE_20261011.md |
| LCNEC 公共证据 M0 冻结合同 | docs/LCNEC_PUBLIC_EVIDENCE_SPACE_M0_CONTRACT_20261011.md |
| 公共证据证书引擎 | tasks/build_public_spectral_evidence_certificates.py |
| 公共证据样本投影 | tasks/project_public_evidence_certificates_to_samples.py |
| 公共证据匹配置乱 | tasks/shuffle_public_evidence_certificates.py |
| Stage-1d 患者bootstrap | data/validation/lcnec_global_remodeling_stage1d/run_2353147/ |
| LCNEC 暗模块全量重扫 | data/validation/GLM_track2_census/lcnec_dark_full_rescan.json |
| 天梯终版 | deliverables/GLM_gnps_article_ladder/ |

## 七、参考论文（docs/papers/）

- Quartet metabolite reference materials (Genome Biology 2024)
- Schuhknecht metabolic map (Nature Biotechnology 2025)
- 其余 10 篇见 docs/papers/ 目录
