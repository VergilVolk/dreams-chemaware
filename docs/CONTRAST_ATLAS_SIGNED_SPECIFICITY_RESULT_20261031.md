# Contrast Atlas 有向特异性响应检索结果（2026-10-31）

## 结论先行

本轮不是一次新的 Atlas 规划，而是对既定核心命题的实际检验：一项研究中带方向、效应量和不确定性的化学响应，能否在其他独立研究中被特异检索，并在没有合格对应时弃权。

结论为：**当前基于 WSE 身份边的研究级响应端点，在现有 `development/consumed` 控件上未获得进入算法—生物闭环的资格。**

- 新的单一有符号算子消除了旧方法的双 response-face 选择伪影，并显著降低调用量；这本身不证明新 p 值已经严格校准；
- 在排除已知共享母项目的暂定分组、使用 999 次端点匹配置换后，五个算子均未得到一条同时命中登记相关性控件并通过当前校准的答案；
- 在移除登记目标的弃权压力检验中，所有算子仍产生较多 post-removal calls；这些调用表示未能弃权，不等于已证明的生物学错误；
- 12 种 exact-drug 只显示同一 ChemProp 母项目内部的有限算子信号，不能充当跨项目验证；
- 本轮没有证明 unresolved 信息增量、Full23 增量或生物学发现。

因此，旧的 `51/53 similar`、`49/53 opposite` 不能再解释为公共响应联系丰富；当前 WSE 响应端点也不能作为算法—生物应用闭环的合格终点。这一裁决不否定 Contrast Atlas 总科学问题，也不能外推为所有可能的跨研究响应算子均失败。

## 1. 为什么必须重做响应算子

上一版全局算子把相似和相反证据分别作为非负总量累加，并分别寻找最大值。诊断表明：

- 53 个查询中，51 个得到 calibrated similar call，49 个得到 calibrated opposite call；
- 49 个查询同时在两面通过，而且通常指向不同候选；
- Top-hit 统计量与连边数的 Spearman 相关分别约为 `0.840/0.812`；
- 三个 ChemProp accession 吸收了 `38/53` 个 similar 和 `39/53` 个 opposite Top hit；
- 旧 null 从研究中所有实体抽 donor，而实际谱学边端点具有不同的响应强度和可匹配性。

所以旧结果主要检验了证据量和来源结构，不是响应特异性。

## 2. 本轮冻结的最低复杂度算子

每个实体—contrast 的响应权重为：

\[
w_{sec}=\frac{\beta_{sec}}{\sqrt{\beta_{sec}^{2}+SE_{sec}^{2}}}.
\]

在冻结的身份边上，只计算一个带符号、能量归一化的响应余弦：

\[
T_{sc,td}=
\frac{\sum_{(e,f)\in E_{st}}a_{ef}w_{sec}w_{tfd}}
{\sqrt{\sum a_{ef}w_{sec}^{2}}\sqrt{\sum a_{ef}w_{tfd}^{2}}}.
\]

`T>0` 表示相似，`T<0` 表示相反；不再分别优化两个 response faces。比较五个本轮固定的低容量 development arms：

1. 原始 signed cosine；
2. 仅保留两端研究内 BH `q≤0.05` 响应边的 dual-FDR sensitivity arm；
3. 候选来源内 contrast centering；
4. 来源对双中心化；
5. 实体在本研究多个 contrasts 中的平均响应残差化。

残差均值只使用真实观测，缺失保持缺失；singleton study 不伪造背景。零模型固定谱学图，只在实际图端点内、相同 identity world 和固定 100-Da 前体区间中置换完整多-contrast profile。

必须说明一项实现边界：端点置换按 source-pair 独立生成；同一 query profile 在不同候选来源中不是一次同步重排。当前 p 值来自把这些 source-pair-specific null 拼接后的 blockwise max-null calibration，不是严格的联合 Westfall–Young FWER。它只在单个 `method × evidence world` 内比较候选，也没有跨 5 个 arms、3 个 worlds 和 53 个查询进行全局多重性校正。所有 p 值只能作为 development 量。

## 3. 数据与独立性边界

本轮输入为：

- 16 个 study source IDs；
- 14 个暂定 parent-project groups；其中 ChemProp 与 Shalapour 的共享母项目已显式合并，其余多数 accession 暂按独立来源处理，尚不是完整核实的项目独立性注册表；
- 53 个研究级 contrasts；
- 238,863 条研究内实体效应；
- 68,586 条 world-specific WSE 身份边；
- 32,220 条 parent-project-disjoint 定向候选分数；
- 999 次冻结端点匹配置换。

全部研究、contrasts 和登记关系均为 `development/consumed`，不是封存外部验证。

登记证据必须分层解释：

- 12 种 exact drug 形成 24 个定向关系，但均来自同一 ChemProp 母项目，只是内部算子正控；
- HFD 和 IBD 合计形成 10 个跨母项目登记方向，但本质上只有两个关系组件，而且没有 signed entity-level truth；
- 未登记关系不是自动阴性，显著未登记答案也不是自动发现。

## 4. 同一母项目内部 exact-drug 正控

以下排名只在指定目标来源的 contrasts 内计算，不把其他来源候选混入分母：

| 响应算子 | 可评价方向 | Top-1 | Top-3 | 中位秩 | MRR |
|---|---:|---:|---:|---:|---:|
| signed cosine | 24 | 4 | 13 | 3.0 | 0.403 |
| dual-FDR signed cosine | 24 | 5 | 8 | 5.0 | 0.369 |
| entity-profile residual | 24 | 5 | 11 | 4.0 | 0.392 |
| candidate-source centered | 24 | 2 | 10 | 4.0 | 0.311 |
| source-pair double centered | 24 | 2 | 9 | 4.0 | 0.311 |

按每个目标来源实际包含的 3、9 或 12 个候选均匀随机选择，24 个方向的期望为 Top-1 `3.0`、Top-3 `9.0`、MRR `0.324`。因此最好的点估计只略高于机会水平，不能称为稳定药物特异信号。24 个方向还是 12 种药物的双向计数，也没有跨实验室或跨母项目独立性。

## 5. 来源留出与登记目标移除：没有算子通过资格门

在保留不同 accession、但尚未排除同一母项目的诊断中：

| 响应算子 | 53 个查询中的 development calls | 命中登记目标的调用 | 登记查询中未命中登记目标的调用 | 无登记控件查询的调用 | 移除登记目标后的 post-removal calls |
|---|---:|---:|---:|---:|---:|
| signed cosine | 18 | 4 | 12 | 2 | 16/30 |
| dual-FDR signed cosine | 8 | 2 | 4 | 2 | 6/30 |
| entity-profile residual | 15 | 1 | 13 | 1 | 13/30 |
| candidate-source centered | 23 | 0 | 21 | 2 | 21/30 |
| source-pair double centered | 19 | 1 | 16 | 2 | 17/30 |

`registered-target-removal` 使用完整冻结分数账本，删去该查询全部登记目标后重新选择最佳候选，并重新应用至少 3 条响应边的支持门。相对“删除候选以后重新估计 null”，沿用完整候选全集的 blockwise max-null 没有放宽阈值；但未登记关系并非真阴性，所以 post-removal call 只能解释为弃权压力检验失败，不能用于估计生物学假阳性率。没有一个算子同时提供有说服力的登记目标覆盖与弃权行为。

## 6. 已知共享母项目排除：当前登记关系上的可支持主张为零

在折叠已知共享母项目、把其余 accession 暂定为独立来源并运行 999 次置换后：

| 响应算子 | 53 个查询中的 development calls | 可评价登记方向 | 登记 Top-1 | 登记 Top-5 | 登记关系 blockwise max-null `p≤0.05` | 登记相关性 Top-1 且通过当前校准 |
|---|---:|---:|---:|---:|---:|---:|
| signed cosine | 3 | 10 | 0 | 3 | 0 | **0** |
| dual-FDR signed cosine | 2 | 4 | 1 | 2 | 0 | **0** |
| entity-profile residual | 1 | 0 | 0 | 0 | 0 | **0** |
| candidate-source centered | 4 | 2 | 1 | 1 | 0 | **0** |
| source-pair double centered | 4 | 0 | 0 | 0 | 0 | **0** |

最后一列要求登记目标同时为 Top-1 且通过当前 blockwise max-null 校准。五个算子全部为零。这里的 10 个方向只有 HFD 和 IBD 两个关系组件，而且只登记“相关性”，没有 signed entity-level truth；结果只能说明现有控件没有建立合格正向证据，不能据此宣判所有跨研究响应检索都失败。

排除已知共享母项目后仍出现少量 development calls，例如：

- AD lesion–nonlesion → IgA-KO 粪便响应；
- pediatric VEO-IBD → male HFD；
- COVID → *B. subtilis* stem contrast；
- lanthanide limitation → sertraline response。

这些联系没有登记真值或独立确认，部分跨越物种、组织和实验背景。它们只能保留为 development 探索假说；`p≤0.05` 只表示在某一 arm/world 的已实现 product-null 下越过阈值，不表示全局校准、关系正确或机制成立。IgA–AD 芳香族氨基酸样候选的有限化学背景来自[此前独立证据审查](CONTRAST_ATLAS_V10_WSE_SYSTEMATIC_BIOLOGICAL_RESULT_20261031.md)，不来自本轮 v7/summary；其身份、跨组织关系和机制均未验证，不能作为 Atlas 成功案例。

## 7. 本轮没有建立的两项增量

### 7.1 unresolved 增量没有建立

known-only 世界更稀疏，unresolved 世界拥有更多可匹配机会。本轮没有完成 measurement/opportunity-matched known 与 shuffled-unresolved-MS2 控制。因此不能把 unresolved 排名或调用更多解释为未知信息的特殊价值。

### 7.2 统一算法增量没有建立

本轮图由 WSE 身份边产生，没有运行 Official DreaMS、Noise、Full23 或合法 BioAware 上下文。因此本结果既不支持也不否定统一算法；它只说明把更好的身份模型直接接入当前未获资格的响应终点，不能自动形成生物应用证明。

## 8. 科学裁决与当前路线停止条件

### 8.1 立即停止的工作

- 不再围绕当前 WSE 响应端点微调阈值、中心化形式或 null 次数；
- 不再把更多公共研究接入同一个未通过特异性门的算子；
- 不再把内部 exact-drug 排名、未登记显著关系或候选网络规模写成 Atlas 成功；
- 在 response endpoint 合格前，不用 Full23 替换 WSE 后的分数变化声称算法具有生物学增量。

### 8.2 保留的总科学问题

Contrast Atlas 总目标不因该负结果撤销：仍然研究带方向、效应量和不确定性的已知及未命名化学响应，能否成为可检索、可累积、可独立验证的公共知识单元。当前没有获得资格的是 WSE 身份边上的响应端点与现有验证资产，不是这个科学问题本身。

### 8.3 下一项真正有价值的实验

现有 53 contrasts 的原始数量不是主要瓶颈；缺的是跨母项目、定义一致的 signed truth。下一步的最高性价比路线不是继续无选择扩来源，而是获得或构造至少一组满足以下条件的独立响应复现单元：

1. 同一明确干预或疾病对比，在两个以上独立母项目中有可比样本类型和时间尺度；
2. 一个项目用于冻结检索，另一个项目完全不参与算子选择；
3. 有实体级隐藏身份或独立定量证据，可判定具体回答，而不只依靠疾病名称；
4. 预先包含 no-match 和 sibling-axis hard decoys；
5. 先证明 response operator 能正确回答和弃权，再分别检验 unresolved 与统一算法增量。

统一算法仍可先在身份连接层评价 corrected、introduced、gained 和 lost；但在出现合格研究级 endpoint 前，这些只能证明化学对应改善，不能冒充 Atlas 生物学成功。

## 9. 可复现产物

- 实现：`tasks/run_contrast_atlas_signed_specificity_retrieval.py`
- 科学汇总：`tasks/summarize_contrast_atlas_signed_specificity_result.py`
- 母项目映射：`tasks/contrast_atlas_parent_projects_v2_msv000080918.tsv`
- 999 次来源留出结果：`data/validation/contrast_atlas_signed_specificity_wse_v8_source_disjoint_999perm_20261031/`
- 999 次母项目留出结果：`data/validation/contrast_atlas_signed_specificity_wse_v7_parent_disjoint_999perm_20261031/`
- 决策汇总：`data/validation/contrast_atlas_signed_specificity_scientific_result_v1_20261031/`

所有完整分数、Top hits、登记关系排名、决策表和 SHA-256 均随结果目录保存。测试覆盖有符号方向、分数边界、缺失保持、端点分层置换的确定性以及 singleton abstention。
