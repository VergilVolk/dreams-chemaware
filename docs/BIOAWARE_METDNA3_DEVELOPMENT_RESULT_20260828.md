# BioAware × MetDNA3 外部开发结果与下一阶段冻结方案（2026-08-28）

## 1. 当前结论

BioAware v1 尚未形成可冻结的外部注释增益，RP 面板继续封存。失败不是单一阈值问题，而是两个先后独立的瓶颈：

1. **反应拓扑与种子覆盖不足**：22 个官方 DreaMS Top-1 错误中，至少 14 个首先缺少可用真候选路径；
2. **现有 raw-MS2 边缺少候选特异性**：在同时存在真、假路径的错误实例中，当前 MetDNA3 `scoreReverse` 边大多持平或支持错误候选。

因此，下一阶段不得继续扫网络权重、DreaMS margin 或 raw-MS2 阈值。必须补齐 MetDNA3 的“MS1 特征预映射 → 逐层反应传播 → raw-MS2 验证 feature-edge”结构，再重新做公式隔离开发。

## 2. 已锁定的外部开发协议

- 数据：MetDNA3 NIST urine HILIC Level-1 子集；
- query：117 张无身份歧义谱图，91 个 IK14；
- 候选：strict-10 ppm、同加合物；
- 基线：官方共享 DreaMS encoder，候选分子按最大 cosine 聚合；
- 划分：10 次固定 30/70 身份轮换，真身份从对应 seed 中剔除；
- 禁止：P2b、表型标签、RP/外部测试集、按结果删除规则；
- tie：计为错误。

官方 DreaMS 在该任务上为 **95/117 = 81.20% Recall@1**，共 22 个错误、17 个分子式。

## 3. 逐步结果

### 3.1 Rhea 一跳 BioAware

| 方法 | rotation instances | corrected | introduced | ΔRecall@1 | 结论 |
|---|---:|---:|---:|---:|---|
| 原始路径累加 | 819 | 4 | 7 | -0.366 pp | 失败 |
| dependency-corrected hyperedge | 819 | 4 | 2 | +0.244 pp | 方向略正，但公式 bootstrap CI 跨 0 |
| formula-group OOF safe gate | 819 | 4 | 7 | -0.366 pp | 失败，不得冻结 |

全开发集事后最优配置能达到 6/1，但公式 OOF 失败，不能作为泛化结果，也不能用于开启 RP。

### 3.2 MetDNA3 当前 raw-MS2 数据层

已按照当前 `MrnAnnoAlgo3` 源码校正为：较小 precursor 谱作为 reference、`scoreReverse`、25 ppm、intensity exponent 1、m/z exponent 0。

- 640 条可评价路径：truth 419，wrong 221；
- 全部 46 个真/假成对实例：truth > wrong 为 41.3%，公式 CI 跨 0；
- 22 个 baseline-error rotation instances：truth > wrong 仅 13.6%；
- 错误实例 mean(truth - wrong) = -0.09697，formula-cluster 95% CI = [-0.1832, -0.0053]。

这说明 raw-MS2 相似度在总体路径上有富集趋势，但在我们真正要修复的 DreaMS 错误上形成显著反向证据。它不能直接作为候选身份加分项。

### 3.3 22 个错误的正式分解

unique query 与 rotation instance 已完全分开报告。22 个 unique error query 的主瓶颈为：

| 主瓶颈 | query 数 | 含义 |
|---|---:|---|
| 真候选不在 Rhea | 11 | 图本体无法产生真路径 |
| 真候选在 Rhea，但没有合格 Level-1 邻居 | 2 | 有节点、无种子可达性 |
| 有网络路径，但种子无 raw MS2 | 1 | 数据层不可评价 |
| 有真 raw 路径，但无错误候选 raw 路径 | 2 | 能做覆盖证明，不能做成对判别 |
| raw MS2 支持错误候选或 tie | 5 | 数据层候选特异性失败 |
| 存在正向救援 headroom | 1 | 当前唯一可直接利用的错误 |

对应的 154 个错误 rotation instances 中：

- truth network path：32；
- wrong network path：53；
- truth raw path：29；
- wrong raw path：50；
- 真/假 raw 成对：22；
- network 可将 truth 排第一：16；
- 实际 dependency-corrected 修正：4。

所以“9 个错误有 truth network path”与“8 个错误有 truth raw path”并不矛盾：前者是网络可达，后者还要求路径种子存在已匹配 MS2。

### 3.4 公开 MetDNA2 KEGG reaction-pair 增量

从仓库内官方 MetDNA2 `reaction_pair_network$version1` 提取：

- 7,639 个序列化节点、9,603 条边；
- 映射后 5,741 个 IK14、7,353 条唯一边；
- 对错误 query，Rhea truth-path 覆盖从 9/22 提高到 Rhea∪KEGG 的 12/22；
- 新覆盖 3 个错误 query；
- 但 7 个错误 raw-MS2 成对实例中 truth > wrong 为 0，平均差为 0。

结论：KEGG version1 能补少量覆盖，但不能单独提供新的候选区分证据，不能直接进入重排器。

### 3.5 MetDNA2 eMRN 逐层 headroom

从 `reaction_pair_network$version2` 提取并保留每条边的最早出现步数：

- step0：7,353 条已知边；
- step1-step8：新增 243,830 条预测边；
- 最终 154,820 个映射化合物、251,183 条唯一边。

然而从 step0 到 step8：

- 117 个 query 的 truth 一跳 Level-1 种子覆盖始终为 43；
- 22 个错误的 truth 一跳覆盖始终为 8；
- 每一层新增 error truth coverage 都是 0。

原因是预测边主要连接扩展代谢物，而这些节点并非 Level-1 seed。**一跳 seed propagation 无法利用 eMRN 的规模；必须先把观测 MS1 feature 映射到这些节点，并进行受控递归传播。**

### 3.6 eMRN 质量索引与 mass-only 预映射

已从官方 `cpd_emrn` 与 `lib_adduct_nl` 固化：

- 165,504 个有效质量条目、161,374 个 IK14；
- 正离子 11 个、负离子 8 个默认注释加合物；
- 完整保留每个化合物的最早 reaction step；
- m/z 计算严格为 `(monoisotopic_mass × nmol + delta_mz) / |charge|`。

用论文固定的 15 ppm、完全不使用身份构建候选后：

- 751 个 Level-1 峰：step0 truth recall 52.7%，step1 58.5%，step2 59.1%；
- step1 相对 step0 增加 43 个 Level-1 truth，但默认加合物候选中位数从 5 增至 8；
- 117-query DreaMS benchmark：step0 为 61/117，step1 为 63/117；
- 22 个 DreaMS 错误：step0 与 step1 均为 11/22，预测层没有直接补回新的错误真值。

这否定的是“直接把 eMRN 扩展化合物塞入候选表”，不是递归 BioAware。真正尚未检验的变量是：样本中其他稳定 MS1 feature 能否充当观测到的中间反应节点。因此 F0 被拆成两个独立阶段：mass-only 候选覆盖已完成；全 MS1 技术重复共识与递归桥 headroom 正在执行。

### 3.7 全 MS1 feature 图与递归 headroom

16 个 HILIC mzML 已用同一 OpenMS 配方处理。阈值只使用技术重复一致性审计，不使用身份或 DreaMS 结果；正式主分析固定沿用 10,000，1,000 与 100,000 为敏感性轴。

- 技术重复配对 feature：104,988；
- 跨 acquisition-window 去重后的稳定 MS1 节点：39,077；
- Level-1 feature 恢复：645/751；
- DreaMS query feature 恢复：105/117；
- 所有递归传播只允许落到实际观测且质量匹配的 feature candidate；
- seed/query 身份隔离，真/假路径距离 tie 计为不能救援。

“严格救援”进一步要求在同一 query 的 7 个 held-out seed 轮换中至少 4 次真路径优于错路径。三个 MS1 阈值共同复现：

| 网络与深度 | 稳健 query | 独立 IK14 | 解释 |
|---|---:|---:|---|
| step0 一跳 | 0 | 0 | 原一跳路径不稳定 |
| step0 两跳 | 5 | 3 | 递归中间 feature 产生新空间 |
| step0 三跳 | 6 | 4 | 再增加 1 个身份 |
| step1 三跳 | 三阈值交集 6 | 4 | 预测层无稳健独立增量 |

因此只允许 **step0 已知 reaction-pair、两跳为主、三跳为消融** 进入候选特异路径打分。step1 预测 eMRN 暂停；不能因为 10,000 阈值下曾出现第 5 个身份，就忽略它在 100,000 敏感性下消失。

## 4. 被否定的实现方式

以下方案停止：

1. 在当前 Rhea 一跳图上继续扫描 network weight/margin/advantage；
2. 把 DreaMS query-seed cosine 当作 data-layer edge；
3. 把 MetDNA3 reverse dot 直接作为 candidate identity 加分；
4. 把 eMRN step8 全图直接并入一跳传播；
5. 使用单条 strongest path/max aggregation 决定候选；
6. 因开发集全量最优 6/1 而开启 RP。

## 5. 下一阶段：真正的两层 BioAware v3

### F0：MS1 feature 预映射（不可省略）

- 从 16 个 HILIC mzML 的 MS1 扫描构建跨技术重复的 feature 共识；
- 只使用 m/z、RT、同位素/加合物一致性和重复出现，不使用身份或表型；
- 将 feature 枚举到 eMRN 节点，step0 与 step1 分开；
- 输出 feature→candidate 的多解集合，不在此阶段做身份确认。

**F0 门**：Level-1 truth feature 的候选召回率必须高；错误候选数、公式歧义和 step1 膨胀必须逐层报告。若 step1 只扩大候选而不增加 truth recall，则停止预测层。

### F1：受控递归 reaction propagation

- 初版只允许 step0；step1 为单独消融；step2+ 全部关闭；
- 每个新节点必须对应一个实际观测 MS1 feature；
- 每条传播边需要完整记录 seed、feature、reaction/prediction rule 和路径深度；
- 高度数/货币代谢物、共享路径和缺依赖物路径降权或弃权；
- 不允许纯图扩散激活未观测节点。

### F2：raw MS2 只验证 feature-edge

- 严格复用当前 MetDNA3 `scoreReverse`；
- raw MS2 判断两条 feature 是否具有可传播的谱学关系，不直接声明某个候选身份；
- 候选区分来自 candidate-specific reaction path、多 seed 共识、依赖完整性和唯一证据；
- 同一条谱学边同时支持真/假候选时标记为冲突，不加分；
- 禁止 strongest-path max；使用路径去重后的多证据共识与 conflict abstention。

### F3：开发与冻结门

仍在 consumed HILIC development 上做 formula-group OOF：

1. corrected > introduced；
2. corrected - 2×introduced > 0；
3. formula-cluster bootstrap CI lower > 0；
4. 每个公式 fold 的 ΔRecall@1 非负；
5. degree-preserving decoy 不得复现增益；
6. step1 必须相对 step0 有独立增益，不能仅增加覆盖率；
7. 只有全部通过后冻结工件，并一次性开启 RP。

## 6. 当前可发表价值与边界

目前可以成立的是一个负责任的方法学发现：

> 代谢反应网络并不会因为规模更大而自动改善谱库注释。若没有观测 MS1 feature 对扩展节点的预映射、候选特异的反应路径以及 raw-MS2 冲突处理，网络扩张只增加不可用节点或错误支持。

这仍不是性能结果。论文级 BioAware 结论必须等待 F0-F3 的公式隔离增益和冻结 RP 复核。

## 7. 主要产物

- `data/validation/bioaware_metdna3_failure_decomposition_v1/`
- `data/reference/metdna2_kegg_network_20260828/`
- `data/validation/bioaware_metdna3_kegg_extension_v1/`
- `data/reference/metdna2_emrn_network_20260828/`
- `data/validation/bioaware_metdna3_emrn_headroom_v1.json`
- `data/reference/metdna2_emrn_mass_adduct_20260828/`
- `data/validation/bioaware_metdna3_ms1_premapping_v1/`
- `data/validation/bioaware_metdna3_ms1_feature_pilot_v1/`
- `data/validation/bioaware_metdna3_recursive_headroom_v1/`
- `data/validation/bioaware_metdna3_recursive_sensitivity_v2.json`

方法依据：MetDNA3 的两层 sequential mapping 与扩展 MRN 见其 [Nature Communications 论文](https://www.nature.com/articles/s41467-025-63536-6)；当前数据层实现由 [MrnAnnoAlgo3 官方源码](https://github.com/ZhuMetLab/MrnAnnoAlgo3) 对齐；KEGG reaction-pair 与 eMRN 资源来自仓库内官方 MetDNA2 数据，原始 MetDNA 思路见 [Nature Communications 论文](https://www.nature.com/articles/s41467-019-09550-x) 与 [官方源码](https://github.com/ZhuMSLab/MetDNA)。
## F2-B raw-MS2 edge validation

The next stage does not add network weight directly.  It reconstructs known
step-0 candidate paths through observed stable MS1 nodes and requires raw MS2
support at every path edge.  The primary path depth is two; depth three is an
ablation.  Intermediate spectra are selected without identities or outcomes,
and no score threshold is selected in this audit.  The purpose is to test
whether raw edge evidence retains the 47 truth-advantage error rotations while
rejecting the 64 stronger-wrong-path risk rotations found by the structural
path audit.

Outputs are written to
`data/validation/bioaware_metdna3_feature_ms2_cache_v1/` and
`data/validation/bioaware_metdna3_candidate_edge_ms2_v1/`.

The corrected v2 edge audit is stored in
`data/validation/bioaware_metdna3_candidate_edge_ms2_v2/`.  Under the fixed
structural-plus-raw-edge rule, the preregistered depth-two analysis makes no
query-level changes.  The depth-three development ablation changes Recall@1
from 95/117 to 98/117 (+2.56 percentage points), correcting three queries and
introducing none.  Those three queries represent only two independent
identities, and the formula-cluster confidence interval includes zero; this is
therefore a mechanism-development signal rather than a validated performance
claim.  The frozen decision report is in
`data/validation/bioaware_metdna3_candidate_edge_decision_v1/`.
