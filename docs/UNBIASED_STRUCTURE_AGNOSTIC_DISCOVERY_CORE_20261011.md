# 面向未知化学空间的通用发现核心：纠偏后的研究设计

> **REPOSITIONED 2026-10-12：** 本文的可观测化学表示、三种入口和证据分解保留为 Contrast Atlas 的底层能力；它不再取代以 study-level contrast signatures 为核心的两层 Atlas。总纲见 `METABOLOMIC_CONTRAST_ATLAS_MASTER_CONTRACT_20261012.md`。

日期：2026-10-11  
状态：方法总纲；其中只有 ChemProp2 的同表 MS1 响应基线已有 development/consumed 结果，其余均为待证伪假设

## 0. 科学定位

目标不是再造一个药物响应检索器，也不是把 Atlas、最优运输或未知网络改名为新范式。目标是检验：

> **在不以结构身份、谱库名单、通路或特定疾病假说决定分析对象的前提下，LC-MS/MS 的已知与未知可观测化学信息，能否在无标签探索、有设计的表型分析和单谱查询三类入口中，产生超过 MS1、谱库匹配与现有谱学方法的可复现增量？**

`unbiased` 只表示发现入口不以身份/通路筛选；不否认提取、电离、DDA、批次、样本和公共数据偏差。`structure-agnostic` 表示结构不是实体参与分析的前置条件；不表示谱图相似等于结构、反应或功能相同。

ChemProp2 响应检索是第一个有监督正控。`entity transport` 与 `relation transport` 是跨研究算子之一，不再是唯一主方向。

## 1. 两个基础对象

### 1.1 样本的可观测化学表示

对样本 `s`：

\[
\mu_s=\sum_{i\in\mathcal O_s} a_{si}\,q_{si}\,\delta_{\phi(S_i),h_i}.
\]

- `O_s`：通过同一表型盲 QC 的 ion families / entities；
- `a_si`：研究内可解释的相对定量，不假定跨研究原始强度可比；
- `q_si`：blank、QC、缺失、峰形、嵌合和 MS2 获取倾向形成的可靠性；
- `phi(S_i)`：冻结谱学证据；可分别来自 WSE、official DreaMS、Noise 和合格局部证据；
- `h_i`：m/z、RT、离子家族、仪器、采集条件、身份等级和来源审计字段。

这个对象不需要疾病或扰动标签。它也不预设一个全局 kNN 连通分量就是稳定化学家族。

### 1.2 研究设计导出的响应

若研究有配对、组别、时间、剂量或扰动设计：

\[
R_{d,c}=\mathcal C_c(\mu_{s_1},\ldots,\mu_{s_n}),
\]

其中 `C_c` 是预注册的效应估计算子，输出方向、效应、不确定性和测量可靠性。ChemProp2 当前只验证了这个对象的一种窄特例，而且检索实际使用的是天然 feature ID 对齐的 MS1 abundance response。

公式本身不是创新。潜在创新只能来自：谱学证据如何被校准为不同主张等级，如何允许 no-match/弃权，以及未知信息是否在跨研究和真实生物学终点上产生条件增量。

## 2. 三种平等的研究入口

### 2.1 无标签样本集合

输出不是 UMAP 图片，而是可检验的：

1. 跨批次/采集条件的同一样本或生物重复检索；
2. 在不使用评价标签训练表示时，对独立样本属性、来源或组成的恢复；
3. 已知化学类别或标准混合物比例的恢复；
4. unresolved-only 及其加入 known 后的条件增量；
5. 对 batch、TIC、MS2 数量、缺失率和仪器的可预测性，作为混杂负终点。

必须比较 MS1 feature histogram、m/z/RT、FBMN、MEMO、Qemistree/MS2LDA/PALS 的适用版本、WSE、official DreaMS、Noise 及谱图置乱。若完整方法只恢复仪器而不能恢复独立样本信息，则该入口失败。

### 2.2 有监督研究设计

可处理疾病、组织、药物、基因扰动、时间和剂量。响应运输只属于本入口。必须比较：

- MS1-only 自然对齐；
- m/z/RT 或传统 feature matching；
- 严格 `entity evidence`，只支持同一实体/不可分辨集合；
- `relation evidence`，只支持局部化学关系，不支持身份或机制；
- known-only、质量匹配 known、unresolved-only、known+unresolved；
- frozen WSE/official DreaMS/Noise/P2b 的合格组合及置乱谱图。

主终点是 held-out 个体、研究、来源或干预上的增量，不是同一 feature table 的重复检索。

### 2.3 单个或一组未知 MS/MS 查询

即使没有样本标签，用户也应能获得分层结果：

1. 校准的同实体候选及 no-match；
2. 局部碎裂/化学关系证据；
3. 公共研究中的出现背景与测量条件；
4. 已冻结、可复现的研究响应证据；
5. 每项结论的分母、来源、身份等级和禁止外推边界。

公共出现共现不能写成生物学功能，谱学邻近不能写成反应，单研究差异不能写成机制。

## 3. 公共数据不是网站，而是原生证据层

公共层的最小单位不是“更大的谱库”，而是带来源和资格的证据记录：

`entity observation -> spectral evidence -> sample/study context -> design-specific response -> replication status`。

第一阶段不建大型网页，也不一次性吞下所有公共数据。先建立三个可审计的小型 shard：

1. 无标签/弱标签组成 shard；
2. 有配对或扰动设计的 response shard；
3. 有跨仪器身份真值的 calibration shard。

所有用于方法选择的 shard 登记为 development/consumed；任何最终外测必须在表示、匹配、阈值和终点冻结后打开一次。公共记录只在真实 feature-spectrum-sample linkage 可验证时进入，只有 processed fold-change 表而无 MS/MS 联结的数据不能裁决谱学增量。

## 4. 证据分解，而非模块堆叠

统一生物学方法不强迫所有算法模块正权，也不重复计票。每个候选关系输出分开的证据：

- `E_id`：同实体校准概率或区间，含 dustbin/no-match；
- `E_rel`：有限分辨率的局部化学关系；
- `E_obs`：样本/研究中的观测和 QC；
- `E_resp`：设计特异响应及不确定性；
- `E_rep`：独立研究复现状态。

Noise 是条件稳健表示候选；WSE 是独立强基线；P2b 是局部判别候选。ChemAware 只按其真实 checkpoint、适用域和证据等级进入消融。BioAware B44-B46 不得复活；没有事件级上下文时不加入。

模型必须允许整个证据族归零，并报告 leave-one-family-out。谱学算法若不超过同一数据上的 MS1-only、m/z/RT 和 WSE，就没有应用增量。

## 5. ChemProp2 的正确位置与新敏感性结果

### 5.1 当前确实成立

- 数据集：MSV000094899；development/consumed；12 drugs，36 leave-one-replicate queries；
- 分析集：3,538 个具有非空 MS/MS 的 QC 合格 features；
- unresolved 3,255 个；R@1 `15/36 = 41.67%`，MRR `0.5562`，高于独立坐标置乱；
- unknown 相对 library-supported 的优势不耐受 MQScore 0.70，因此 unknown 独立增量失败。

### 5.2 必须纠正的解释

检索没有使用 MS/MS 峰、embedding、WSE、Noise 或 P2b。MGF 只用于判断 feature 是否有至少三个碎片峰；检索坐标是天然 row ID 对齐的 MS1 abundance trajectory。因此：

- 它是 `MS1-response baseline in an MS2-qualified universe`；
- 坐标置乱证明 feature correspondence 携带响应，不证明碎裂化学贡献；
- 它不支持跨研究 entity/relation transport；
- 它不证明本项目统一谱学算法带来应用增量。

### 5.3 母药/药物相关敏感性

在 MQScore >= 0.80 下：

| 排除规则 | 保留 features | R@1 | MRR |
|---|---:|---:|---:|
| 不排除 | 3,538 | 38.89% | 0.5363 |
| 直接母药库节点 | 3,482 | 38.89% | 0.5418 |
| 广义药物相关库节点 | 3,469 | 38.89% | 0.5412 |
| 含母药节点的全部 FBMN 成分 | 3,128 | 38.89% | 0.5295 |

unresolved 在成分排除后保留 2,917 个，R@1 `14/36 = 38.89%`，MRR `0.5277`；相对未排除的 MRR 差值 `-0.0285`，drug-cluster bootstrap 95% CI `[-0.0818,+0.0222]`。

这表明结果不完全由已标出的母药及其 FBMN 邻域决定。它仍未完整排除未标注药物产物；删除整个网络成分会误删谱学相似的内源分子，是敏感性上界而非化学去卷积。

质量匹配的 283 个 unresolved features（总体强度、QC 检测率/CV、研究样本检测率、MS2 峰数；协变量绝对 SMD <= 0.045）仍得到 R@1 `41.67%`、MRR `0.5379`，相对 283 个 library-supported 的 MRR 差值 95% CI `[+0.0016,+0.2177]`。加入 response magnitude 的保守匹配结果为 R@1 `41.67%`、MRR `0.5605`。这只排除 score 0.80 下若干测量质量解释，不修复 score 0.70 的阈值失败，也不构成 MS/MS 增量。

## 6. 三场景可证伪开发矩阵

### A. 药物时间序列：机制正控，不定义总方法

1. 冻结当前 MS1-only 结果；
2. 保留已完成的检测率、QC CV、强度、缺失率、响应幅度和 MS2 峰数匹配；若取得 chromatographic peak-shape 真值再补峰形匹配；
3. 将 50-drug endpoint 与 12-drug time-series 分开处理，移除自然 row-ID 对齐；
4. 比较 m/z/RT、WSE、official DreaMS、Noise、P2b 与 entity/relation evidence；
5. 报告 coverage、no-match、错误匹配和对 MS1-only 的严格增量。

只有第 3-5 步通过，才能说谱学表示支持跨设计推断。

### B. 疾病/组织：真实表型，但不让单队列定义规则

MTBLS13729 可作为 consumed development pressure test，不是外部确认。预先分开 Rmu-vs-RN 丰度与 Rmu-vs-Rtu interaction；在个体/批次留出下比较 known-only、unresolved-only 和 known+unresolved。若未知增量只能在全体样本拟合或由一个候选峰故事支撑，则失败。

最终至少需要第二个独立来源，且不能用同研究的不同文件或 raw/mzML 副本冒充独立队列。

### C. 无标签样本集合：通用性挑战

先冻结一个不使用评价标签构建的样本化学表示。主任务至少包含技术条件变化下的同一样本检索和独立来源/组成恢复；负任务是仪器、批次、TIC 与 MS2 数量预测。不得用 UMAP 视觉判定。若 MS1 或 MEMO 已达到相同结果，完整方法不晋级。

## 7. 三层科学裁决

| 科学层次 | 当前状态 | 晋级要求 |
|---|---|---|
| 未获库支持信号携带可重复响应 | ChemProp2 同表 MS1 开发结果为正 | 质量匹配、药物来源敏感性和跨阈值边界完整 |
| 谱学证据支持跨设计/跨研究推断 | 未证明 | 无自然 feature ID，超过 MS1/mzRT/WSE，允许 no-match，外层 source/instrument 留出 |
| 通用 unbiased、structure-agnostic 方法 | 未证明 | 药物、疾病和无标签三入口共享同一冻结核心，至少两个独立来源，同号且有真实条件增量 |

任一上层主张都不能由下层结果自动继承。

## 8. 当前实施顺序

1. 完成 ChemProp2 匹配质量对照与跨设计无天然 ID 基准；
2. 同时冻结一个无标签任务合同和一个疾病任务合同，禁止为结果改规则；
3. 在三个入口复用同一谱学证据接口与资格等级；
4. 只把在 development 中超过最强适用基线的版本冻结；
5. 再寻找一次性独立验证和前瞻生物学三角验证。

当前不建大 Atlas，不命名“新范式”，不把 response transport 作为唯一主线。进展只由实测的条件增量、覆盖/错误、跨来源迁移和可证伪生物学结果决定。
