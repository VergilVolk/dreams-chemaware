# ChemProp2 M1：MS2-qualified features 的 MS1 响应同来源重复可辨识性

日期：2026-10-11  
状态：**development / consumed；响应对象可进入跨设计 M1b，unknown 独立增量未通过**

## 1. 本实验回答什么

本实验不是算法候选排序 benchmark，也不是跨研究验证。它只回答一个更早的地基问题：

> 在不先要求精确结构的情况下，微生物依赖、时间分辨的 MS1 feature 响应能否在独立生物学重复之间复现；具有非空 MS/MS 但未获 GNPS 库支持的 features 是否含有超过坐标打乱的重复信息？

**关键纠偏：** 本实验检索没有使用 MS/MS 碎片峰、谱学 embedding、WSE、Noise 或 P2b。MGF 只用于要求 feature 有至少三个碎片峰，库表只用于分层；相似度完全来自天然 row ID 对齐的 MS1 abundance trajectories。因此它是 MS1-only 基线，不是谱学算法增量实验。

数据为 ChemProp2 `MSV000094899`，GNPS2 task `1b5b94b4191d4223a5f57afb2aaaf0b0`。全部结果已经用于方法选择，因此从本次起登记为 development/consumed。

## 2. 真实分母与输入

- MZmine 定量 features：8,055；
- 可解析研究注射：699；其中 12 个药物各有 9 个时间点、Bac/noBac 两臂、3 个重复，另有 DMSO+Bac、Medium+noBac controls；
- pooled-QC、blank 和非空共识 MS/MS 过滤后：3,538 个分析实体；
- GNPS library-supported（主敏感性阈值 MQScore >= 0.80）：283；
- unresolved：3,255；
- 检索分母：12 drugs x 3 leave-one-replicate queries = 36 queries；每个查询在 12 个药物原型中排序。

这里的 `library-supported` 不是 authentic-standard confirmed；`unresolved` 也不是已证明的新分子，只表示在给定库匹配阈值下没有支持。

冻结输入哈希：

| 资产 | bytes | SHA-256 |
|---|---:|---|
| quantification.csv | 33,365,286 | `15131ea1623e07f12ea9adfee78208cf2a8326a0cc799ed6669b7ef3dba37ab5` |
| consensus.mgf | 6,979,017 | `49ab4ca3815e43c8110b263cc753cd858ed32a2eaf40b7a1fd67667c284ace7d` |
| network_singletons.graphml | 13,217,850 | `c125c073375360bd918db80b804027bb5ce05aa12f8cd62d3d91f43e79ef2c37` |
| library_matches.tsv | 349,100 | `d2f239733310dc97e81ae26de5ecd0cfb4bef546dbc3a92b6a7171c67344dc7c` |
| metadata.tsv | 41,670 | `df07dcbd338de0481e860a572b722b9e5357b0c33b1a6d0bac8d27ead5b74b33` |

## 3. 响应定义

每个 drug x replicate x time 的响应为：

`(drug Bac - drug noBac) - T0 baseline - [(DMSO Bac - Medium noBac) - control T0]`

强度使用 feature-specific pseudocount 后的 log2，并以 sample median centering 作为主归一化。对 T1-T8 做梯形 AUC；沉积数据中少量 T7 文件缺失，要求每个 profile 至少有 6 个 later times。检索时，每个 query replicate 只与另外两个 replicate 构成的 drug prototype 比较。

该 estimand 同时控制 abiotic drug change、无药培养基、vehicle/time drift 和 T0 差异，但仍可能由 drug-derived biotransformation products 主导。它不是一般宿主机制响应，也不支持反应方向或通量。

## 4. 主结果

主分析为 median-centered trajectory AUC：

| feature universe | queries | R@1 | MRR | median rank |
|---|---:|---:|---:|---:|
| all QC + MS/MS | 36 | 38.89% | 0.5363 | 3.0 |
| library-supported, score >= 0.80 | 36 | 25.00% | 0.4270 | 4.0 |
| unresolved, score < 0.80/no hit | 36 | 41.67% | 0.5562 | 2.5 |

以 12 个 drug 为聚类单位做 10,000 次 bootstrap：

- unresolved - library-supported R@1：`+16.67 pp`，95% CI `[+2.78,+30.56] pp`；
- unresolved - library-supported MRR：`+0.1292`，95% CI `[+0.0370,+0.2221]`；
- all - library-supported MRR：`+0.1093`，95% CI `[+0.0153,+0.2056]`。

但效应不均匀：trajectory 主分析中 12 个药物只有 5 个在 unresolved universe 达到三次查询至少一次 Top-1；erythromycin、montelukast 和 simvastatin 等仍失败。不能把 pooled 阳性写成对所有扰动通用。

endpoint 响应更容易找回同一药物（unresolved R@1 72.22%，MRR 0.7860），但这是观察后的次级结果，且更可能由残留 parent/product 信号驱动；不作为主门。

## 5. 两个关键负对照

### 5.1 特征数量匹配

从 3,255 个 unresolved 中重复 200 次随机抽取 283 个，与 library-supported 数量一致：

- R@1 mean 40.49%，95% empirical interval `[30.49%,50.00%]`；
- MRR mean 0.5395，95% empirical interval `[0.4550,0.6206]`。

因此 score 0.80 下的 unresolved 优势并非仅由 3,255 对 283 的维度差解释。

### 5.2 坐标独立打乱

对每个 replicate 独立打乱 3,255 个 unresolved feature 坐标，保持每个 profile 的边际值分布：

- null R@1 mean 10.35%，接近 12-way chance 8.33%；
- null MRR mean 0.2826；
- observed unresolved 的 R@1 与 MRR 均为 empirical `p = 1/201 = 0.00498`。

所以真实 feature correspondence 携带可复现的 drug-specific response 信息；结果不是仅由每个 profile 的效应分布形状产生。

## 6. 库阈值敏感性推翻了“unknown 已有独立增量”

| GNPS MQScore 定义 | supported / unresolved | supported MRR | unresolved MRR | delta MRR 95% CI | 严格增量门 |
|---:|---:|---:|---:|---:|---|
| 0.70 | 324 / 3,214 | 0.5206 | 0.5631 | `[-0.0390,+0.1260]` | FAIL |
| 0.80 | 283 / 3,255 | 0.4270 | 0.5562 | `[+0.0370,+0.2221]` | PASS |
| 0.90 | 203 / 3,335 | 0.3918 | 0.5432 | `[+0.0270,+0.2710]` | PASS |

在三个阈值下，unresolved 都显著优于坐标打乱；但相对 library-supported 的增量在 0.70 时消失，数量匹配 unknown 的下分位也未超过 supported。由于 0.80 不是独立身份真值，不能据此选取有利阈值。

正式裁决：

- `unresolved_response_signal_robust_to_library_threshold = true`；
- `pass_response_object_to_cross_design_m1b = true`；
- `unknown_increment_robust_to_library_threshold = false`；
- `pass_unknown_increment_claim = false`。

## 7. 母药与药物相关网络敏感性

MQScore >= 0.80 时，56 个分析 features 是直接母药库节点，69 个是广义药物相关库节点；22 个含母药节点的 FBMN 成分覆盖 410 个分析 features。

| 排除 | features | R@1 | MRR |
|---|---:|---:|---:|
| 无 | 3,538 | 38.89% | 0.5363 |
| 直接母药节点 | 3,482 | 38.89% | 0.5418 |
| 广义药物相关节点 | 3,469 | 38.89% | 0.5412 |
| 含母药节点的全部 FBMN 成分 | 3,128 | 38.89% | 0.5295 |

unresolved-only 在成分排除后保留 2,917 features，R@1 `38.89%`，MRR `0.5277`；相对原 unresolved 的 MRR 差值 `-0.0285`，drug-cluster 95% CI `[-0.0818,+0.0222]`。所以结果不完全由已标出的母药/网络邻域解释，但仍可能含未标注药物转化物。整成分删除是保守敏感性上界，不是可靠的 adduct/isotope/产物去卷积。

结果：`data/validation/chemprop2_parent_drug_dependence_v2_20261011/report.json`；实现：`tasks/audit_chemprop2_parent_drug_dependence.py`。

## 8. 质量与响应幅度匹配敏感性

在 MQScore >= 0.80 的 development 定义下，用一对一最优匹配从 3,255 个 unresolved 中选择 283 个。质量匹配变量为总体强度、QC 检测率、QC CV、研究样本检测率和 MS2 峰数；所有标准化均值差绝对值 <= 0.045。第二个更保守的匹配再加入 response magnitude；全部绝对值 <= 0.052。

| universe | features | R@1 | MRR | 相对 supported 的 MRR delta 95% CI |
|---|---:|---:|---:|---:|
| library-supported | 283 | 25.00% | 0.4270 | reference |
| quality-matched unresolved | 283 | 41.67% | 0.5379 | `[+0.0016,+0.2177]` |
| quality + response-matched unresolved | 283 | 41.67% | 0.5605 | `[+0.0304,+0.2393]` |

因此在 score 0.80 内，集合大小、总体强度、检测率、QC CV、缺失和 MS2 峰数不足以解释差异。response magnitude 是生物学效应的一部分，匹配它可能过度控制，故只作保守敏感性。原表没有独立峰形真值，本分析不能声称已匹配所有 chromatographic peak quality。

这不推翻第 6 节的阈值失败：MQScore 0.70 时 unknown 相对 supported 的优势仍不稳健，且 supported 不是 authentic-standard truth。故 `pass_unknown_increment_claim` 仍为 false。

结果：`data/validation/chemprop2_matched_unresolved_20261011/report.json`；实现：`tasks/audit_chemprop2_matched_unresolved.py`。

## 9. 泄漏与主张边界

- 数据集：MSV000094899；36 queries；development/consumed；
- 同一实验室、同一仪器、同一 MZmine feature table；不是跨研究、跨 source 或跨 instrument；
- drug label 只用于检索真值，没有进入 feature QC 或响应权重；但归一化和 feature scaling 使用了同一完整无标签矩阵；
- exact feature row ID 已天然对齐，本轮没有测试谱学 response transport；坐标置乱不证明 MS/MS 化学信息的贡献；
- GNPS library match 是推定身份，不是 Level-1 known truth；
- 同药物检索可能主要反映药物及其转化产物，不等于相同 MoA 或一般生物学机制；
- 本结果不证明 Noise、P2b、ChemAware 或统一排序器的应用增量。

## 10. 下一实验

1. 用 50-drug 两时点数据中的共享药物做跨实验设计复现，保持本轮参数冻结；
2. 当前结果正式作为 MS1-only 基线；跨设计时移除 natural feature ID，比较 m/z/RT、WSE/official DreaMS/Noise、冻结 P2b 和 relation transport，显式允许 no-match；
3. 区分 parent/drug-derived subnetwork 与非直接连接的 endogenous/community response；前者支持 biotransformation specificity，后者才可能支持更广机制；
4. known/unresolved 改用 authentic standard、作者确认等级或冻结身份状态，不再用一个可调 GNPS score 作为最终真值；
5. 只有跨设计通过后才寻找有重复 target/MoA 且有 raw LC-MS/MS 的第二来源。S-BSST1724 官方存档只有 processed fold-change workbook，不能承担这一角色。

## 11. 实现与结果

- `tasks/analyze_chemprop2_response_reproducibility_m1.py`
- `tasks/adjudicate_chemprop2_response_m1_sensitivity.py`
- `data/validation/chemprop2_response_m1_v2_20261011/report.json`
- `data/validation/chemprop2_response_m1_sensitivity_20261011.json`

本轮最重要的进展不是一个漂亮百分比，而是证明：大量具有非空 MS/MS、但当前未获库支持的 MS1 features，其方向性 abundance response 在生物学重复间不是随机；同时也证明，是否优于“已知”取决于身份定义。它尚未证明 MS/MS 表示、跨研究运输或 unknown 的独立应用增量。
