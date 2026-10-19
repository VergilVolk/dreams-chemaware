# Metabolomic Contrast Atlas：方法空位、统计核心与首轮实证路线

日期：2026-10-12  
状态：开发方法；所有真实数据均为 `development/consumed`，当前没有外部确认  
总纲：`METABOLOMIC_CONTRAST_ATLAS_MASTER_CONTRACT_20261012.md`

## 1. 文献复核后的准确空位

不能把下列既有能力重新包装成本文创新：

| 已有方法 | 已经实现 | 对本项目的约束 |
|---|---|---|
| DreaMS Atlas | 2.01 亿公共 MS/MS 的表示与分子网络 | 更换 embedding 或扩大谱图数不是生物学方法 |
| MASST / StructureMASST | 谱图、结构、亚结构、类似物到公共文件和元数据的全局检索 | “未知谱出现在哪里”不是 contrast retrieval |
| Pan-ReDU | 跨仓库元数据协调、文件发现与再分析入口 | 数据发现与 metadata harmonization 不是效应整合；其论文也明确跨数据集 abundance feature 分析仍不可直接进行 |
| MEMO | 无需 RT 对齐的样本级 MS/MS 向量与跨批次样本比较 | 样本聚类、PCoA、UMAP 不是创新终点 |
| PALS | 通路、GNPS molecular family 和 Mass2Motif 的条件活性评分 | 单研究的 feature-set 活性不等于公共研究响应检索 |
| MS2LDA 2.0 | 可扩展 Mass2Motif 发现、MAG 注释建议与 MotifDB 检索 | motif 只能作为证据原子，不能自动充当实体或功能真值 |
| PAIRUP-MS | 未知 feature 跨数据集匹配、遗传关联一致性与通路重建 | 跨数据集未知峰对应和“未知也能做关联”已被证明 |
| FBMN-STATS | feature table 清洗、单/多变量统计并回填网络 | “上传 feature table、做差异、画网络”已经成熟 |
| TidyMass2 | 未注释 feature 的网络模块解释和来源推断 | “让更多未知峰可解释”不是足够的新主张 |
| MetaboKG | 公共元数据、谱图、workflow、注释与 provenance 的知识图谱 | 知识图谱和可追踪查询本身不是方法创新 |

仍值得正面检验的空位是：

> 在不同研究只观测到部分重叠化学空间、无法充分建立精确实体同一、且 MS/MS 获取强烈非随机的情况下，能否把各研究内部冻结的效应检索为相似、相反或不可比较的研究级响应；并证明未解析化学证据相对 known-only 和 measurement-matched controls 有独立增量？

这是一项待证伪的统计与化学推断能力，不是数据库规模声明。

## 2. 核心对象不是样本向量，而是可观测域受限的 contrast

对研究 `d`、独立化学单位 `i`，先在研究内以合适设计估计：

`E_di = (beta_di, SE_di, detectability_di, acquisition_di, identity_state_di)`。

原始强度不跨研究合并。adduct、isotope、in-source fragment 和同一色谱实体必须在效应估计前折叠为 independent unit；生物学分母是 subject、clone、line 或独立培养，而不是 injection、feature 或 MS2 scan。

表型盲且在 query/target 标签之外冻结的谱学证据把 independent units 映射到 chemical evidence atom `k`。atom 可以是严格实体或校准碎裂模式，但两者不得混写。每项研究形成 `C_dk` 及传播后的 `SE(C_dk)`。

## 3. 观测交集上的对称检索

查询 `q` 与目标 `t` 只在共同合格 atom 集 `Omega_qt` 比较。缺失不是零效应，也不是反向效应。

每对研究必须同时报告：

- `coverage_q = |Omega_qt| / |Omega_q|`；
- `coverage_t = |Omega_qt| / |Omega_t|`；
- strict-entity 与 shared-pattern 的独立数量；
- source、instrument、mode、chromatography 和 DDA 获取差异；
- known-only、unresolved-only 与 all 三个证据世界。

任一侧覆盖不足即弃权。只按查询侧 coverage 会错误奖励目标端只检测到少量共同模式的研究，已在原型 v2 中禁止。

原型目前使用传播不确定性加权的 signed cosine；正式方法不得直接采用固定 `0.3` 阈值。候选统计量需在开发来源中按 source/instrument/ion-mode/feature-count 匹配的 contrast-label permutation 建立经验 null，并输出：

- association sign 与 calibrated tail probability；
- local FDR 或 selective risk；
- risk--coverage curve；
- 对 independent unit、biological replicate 和 source family 分层 bootstrap 的区间。

模式共用实体导致相关性时，不得把 pattern 数当独立样本量。正式实现应以 independent unit block 重采样，或估计 atom 相关矩阵后做 shrinkage。

## 4. 三层结果必须保持分离

1. **strict entity replication**：回答同一分析实体的效应是否复现；要求一对一、表型盲、端点存在、身份概率过门。
2. **shared chemical response**：回答不同实体集合是否携带相似的碎裂化学变化；控制 motif 常见度、单峰支撑、DDA propensity 和 source dominance。
3. **study contrast association**：在前两层的资格与可观测边界上回答研究响应是否相似/相反；不升级为同一分子、同一反应、通量或因果机制。

三个分数不做强制正权融合。某层失败只撤回该层主张。

## 5. 真正的主终点

### 5.1 有完备真值的扰动检索

以完整 source family 留出查询一个干预，相关目标在揭盲前由同一干预、相同已知靶点/MoA 或预先登记的正交遗传关系定义。报告：

- selective Recall@K、MRR/nDCG 与 no-match accuracy；
- 在固定错误率下的可回答 query 比例；
- similar、opposite、indeterminate、abstain 的完整混淆；
- biological perturbation 聚类 CI，而非 feature-level CI。

### 5.2 疾病到干预的开放真值发现

疾病 query 没有完整 relevance truth，不能用 post-hoc 文献命中计算 AUROC。只能在 consumed 数据中生成冻结候选，在未参与选择的新扰动研究或前瞻实验中验证有限个预注册预测，并同时验证强 decoy 和反向预测。

### 5.3 未解析化学信息的必要性

主增量不是 `all > known-only` 的裸差，而是：

`Delta_unknown = performance(all) - max(known_only, measurement_matched_known, unresolved_shuffled_MS2)`。

要求 source/instrument 留出均同号、cluster CI 下界严格大于零，且风险不升高。只因 all 世界 feature 更多而提高即失败。

## 6. 必须正面对照的六类世界

1. MS1-only：m/z、RT、检测率、强度、missingness；
2. sample fingerprint：MEMO/CSCS/Qemistree 的适用版本；
3. feature set：PALS、MS2LDA/Mass2Motif、FBMN family；
4. strict entity：质量、cosine、WSE、official DreaMS、Noise、资格内 P2b；
5. prior/network：PAIRUP-MS 与 TidyMass2/FFMA 的适用版本；
6. full method 与负控：matched-known、MS2 permutation、effect permutation、source/instrument-only predictor。

算法整合侧提供的冻结模型只能作为第 4 类证据。Noise、ChemAware、P2b 或 BioAware 任一模块若未通过自己的资格门，不得因 Atlas 应用需要而宣布 active。

## 7. 仓库真实证据给出的停止门

- ChemProp 跨设计：36 queries，随机期望 16.67%；同图结果 mass 和 WSE 均 7/36，cosine 10/36，相对 mass 的药物簇 CI 跨零。实体层 FAIL，relation transport 已封存。
- ST003325：结构隐藏跨编辑机制仅 10/27，clone-label permutation `p=0.7126`。正式负 shard，不再追加谱学模块救阳性。
- LCNEC：跨平台表型盲候选只有 6/263（2.28%），`pass_to_global_transport=false`。可用于检验弃权，不能证明广覆盖。
- MSV000096214/TT7：作者处理表的结构隐藏 MS1 遗传正控通过，但数据为 `development/consumed`；raw truth-blind 重建和 MS/MS 增量尚未裁决。
- MTBLS13729：可作为疾病 query 候选，但 Rmu-vs-RN abundance 不代表 Rmu-vs-Rtu interaction、通量或酶活。

这些结果说明：第一版小 Atlas 必须以“知道什么时候不能比较”为主要能力之一。不能用扩大数据库、放宽 ppm 或事后机制解释覆盖失败。

## 8. 首轮可执行实验

### A. 合同与负结果重放

运行 machine schema、同/反向合成 smoke，并把 ChemProp、ST003325、LCNEC 转换成统一 contrast record，确认失败和弃权能原样重现。目标不是阳性，而是证明统一方法没有抹去既有负证据。

### B. 单来源内部正控

完成 MSV000096214 raw truth-blind feature table。先只验证结构隐藏 MS1 contrast 是否重现作者表中的 allele/pathway/epistasis 关系；再在冻结 MS2 linkage 后比较 MS1-only、WSE、official、Noise 和资格内 P2b。任何模型调参均在该 consumed shard 内完成。

### C. 两个独立 source family 的小 Atlas

在下载前登记第二来源，要求 raw MS1、实际 MS2、样本级 design、blank/QC、独立 biological units 和可构造真值/decoy。阈值、null、endpoint 和停止门在打开其 outcome 前冻结。ST003940 与 ST003325 同属 `JAX_MorPhiC_PR001844`，不能算第二来源。

### D. 疾病端旗舰链

仅在 A--C 通过后，以 MTBLS13729 或其他高质量配对疾病研究为 query，预注册有限个 perturbation links；在全新来源验证。若 known-only 已能得到相同结论，未知化学主张失败，即使整体检索仍可用。

## 9. 当前原型的准确资格

`tasks/contrast_atlas_core.py`、`tasks/contrast_atlas_schema_v1.json` 和 `data/validation/contrast_atlas_core_smoke_20261012_v2` 已实现：

- 输入资格与 effect-column leakage 拒绝；
- strict link 一对一与真实端点检查；
- known/all/unresolved 三世界；
- 双侧 coverage；
- strict entity 与 shared pattern 分层；
- 跨 source 默认门与证据不足弃权；
- 输入 SHA-256 与参数登记。

合成 smoke 的同向、反向和稀疏目标行为符合合同；这只证明软件语义，不是方法准确率。尚未实现的决定性部分是 source/instrument-matched 外层 null、cluster uncertainty、measurement-matched known control、完整 real-shard effect adapters 和 source-held-out benchmark。

2026-10-13 更新：`calibrate_contrast_atlas_pattern_null.py` 已实现保持研究效应/SE边际和覆盖不变的 target pattern-label permutation、双向 empirical p 与 query 内 BH-FDR；它是内层 chemical-alignment null，不能替代 source/instrument-matched 外层 null。`audit_contrast_atlas_shard_admission.py` 已重放 ChemProp、ST003325 和 LCNEC，三个历史停止门全部保留。MSV000096214 raw response 若通过，已有 gated record builder 可产生带 Welch SE 的 `development/consumed` contrast records。

## 10. 主要参考边界

- DreaMS Atlas: https://www.nature.com/articles/s41587-025-02663-3
- StructureMASST: https://www.nature.com/articles/s41587-026-03082-8
- Pan-ReDU: https://pmc.ncbi.nlm.nih.gov/articles/PMC12103507/
- PAIRUP-MS: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006734
- PALS: https://pmc.ncbi.nlm.nih.gov/articles/PMC7916825/
- MEMO: https://pmc.ncbi.nlm.nih.gov/articles/PMC9580960/
- MS2LDA 2.0: https://www.nature.com/articles/s41467-026-75038-0
- FBMN-STATS: https://www.nature.com/articles/s41596-024-01046-3
- TidyMass2: https://www.nature.com/articles/s41467-026-68464-7
- MetaboKG preprint: https://arxiv.org/abs/2605.24706
