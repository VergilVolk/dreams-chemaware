# Metabolomic Contrast Atlas：MSV000095331 开放 51 对比真实结果

> **已被 2026-10-23 端点交互重分析取代，不得引用为正式结果。** 本文使用的
> `(微生物 T72-T0)-(BHI T72-T0)` 没有减去无药微生物培养变化，并以正态近似处理
> 3 个响应重复；它只能作为失效历史。现行结果见
> `CONTRAST_ATLAS_MSV000095331_ENDPOINT_INTERACTION_RESULT_20261023.md`。

## 结论

本轮不是数据审计，也不是又一个两候选匹配实验。我们从公开药物—肠道微生物实验
MSV000095331 恢复了真实定量响应，并将其作为查询，在已有 Contrast Atlas 的 15 项
研究、51 个研究级对比中进行开放检索。

结果首次支持一个有限但真实的方法学信号：去除每个候选共有的时间/培养响应后，
结构未解析 MS/MS 响应在开放检索中的表现高于 known-only。透明 cosine 只能稳定找回
sertraline；WSE 在 51 候选中将 pooled omeprazole 提到第 3，并在保留局部群落响应时
将 loratadine 从第 12 提到第 4。但 Top-1 仍只有 1/3，故当前证据支持继续用冻结的
B/完整统一算法替换谱学层，不支持宣称 Contrast Atlas 已经完成。

## 数据与分母

- 新查询项目：MSV000095331；120 个主分析样本，4 个已发表药物混合物，4 个 72 h
  微生物群落，每个实验单元 3 个重复。
- 新项目响应实体：7,515 个具有非空 MS/MS 且在相关实验单元中实际检出的实体；
  其中 40 个通过当前公共库策略支持，7,475 个未解析。
- 查询：4 个跨群落汇总响应，加 16 个群落特异响应。
- 候选：15 项公开研究、51 个研究级对比；每个查询均保留同一完整候选集合。
- 预设重叠：mix 1–sertraline、mix 3–omeprazole、mix 4–loratadine；mix 2 无预设
  重叠，用作 no-match 压力测试。
- 数据角色：开发。MSV000095331 对本方法是新项目，但并非封存外测，也不是独立实验室
  确认。

## 固定计算对象

每个新项目响应为：

`(微生物群落 endpoint - T0) - (BHI T72 - BHI T0)`。

跨群落汇总对四个 72 h 群落等权；局部分析保留四个群落，不先平均效应。

透明谱学基线使用 0.1 Da 碎片箱、平方根强度、谱内 L2 归一化和 cosine。每个新项目—
候选项目对仅保留 cosine >= 0.70 的 mutual top-1 边。所有边均在读取 beta、SE、
处理名称和预设重叠关系之前构建。

响应权重为 `beta / sqrt(beta^2 + SE^2)`。直接相似分数受到候选自身通用时间/培养响应
严重影响，因此增加不使用药物名或真值的响应特异性分数：

`specificity(q,c) = signed_alignment(q,c) - mean_q signed_alignment(q,c)`。

其含义是：候选对当前处理的化学响应匹配，是否超过它对同一查询面板其他处理的共有
匹配。该算子是在未中心化 v1 暴露出 generic-response bias 后于开发期引入，不能作为
封存确认结果。

## 真实结果

### 24 个同领域药物响应候选

| 证据世界 | 跨群落最佳真关系中位名次 | 跨群落 MRR | Top-1 查询 |
|---|---:|---:|---:|
| all | 2 / 24 | 0.567 | 1 / 3 |
| known-only | 6 / 24 | 0.444 | 1 / 3 |
| unresolved-only | 2 / 24 | 0.537 | 1 / 3 |

### 15 项研究、51 个开放候选

| 证据世界 | 跨群落最佳真关系中位名次 | 跨群落 MRR | Top-1 查询 |
|---|---:|---:|---:|
| all | 4 / 51 | 0.450 | 1 / 3 |
| known-only | 7 / 51 | 0.131 | 0 / 3 |
| unresolved-only | 4 / 51 | 0.432 | 1 / 3 |

unresolved-only 的具体最佳真关系名次为：

- sertraline：1 / 51；四个群落的特异性贡献均为正；
- loratadine：4 / 51；
- omeprazole：22 / 51，失败。

未进行响应特异性校正时，所有查询反复被 simvastatin 等通用强响应候选吸引；51 候选
下 all-world 最佳真关系中位名次为 8，Top-1 为 0/3。这证明只累加同向边不能区分
生物处理特异性。

### WSE 强基线

WSE 使用 0.02 Da 碎片容差、至少 6 个匹配碎片、相似度至少 0.70，并重新构建双向
mutual top-1 图；样本、响应、候选和下游算子均不改变。

| 51 候选 WSE | 最佳真关系中位名次 | MRR | Top-1 查询 |
|---|---:|---:|---:|
| pooled all | 2 / 51 | 0.515 | 1 / 3 |
| pooled known-only | 15 / 51 | 0.097 | 0 / 3 |
| pooled unresolved-only | 3 / 51 | 0.472 | 1 / 3 |
| 四群落 all | 4 / 51 | 0.464 | 1 / 3 |
| 四群落 known-only | 17 / 51 | 0.073 | 0 / 3 |
| 四群落 unresolved-only | 4 / 51 | 0.472 | 1 / 3 |

WSE pooled unresolved-only 的最佳真关系名次为 sertraline 1、omeprazole 3、
loratadine 12。保留四个群落后分别为 1、6、4：局部响应牺牲部分 omeprazole 名次，
但将 loratadine 从 12 提到 4；三条最佳真关系的特异性贡献均在四个群落中为正。

在相同 51 候选任务上，WSE 相对稀疏 cosine 将 pooled unresolved MRR 从 0.405
提高到 0.472，并将中位名次从 7 提到 3。最明确的算法—生物学交换是 omeprazole：
稀疏 cosine 排第 14，WSE 排第 3。

## 这一步真正增加了什么

1. 生物学响应而不是谱图出现记录成为检索对象。输入包含方向、效应和不确定性。
2. 候选集合从两选一扩大到 15 项研究、51 个实际生物学对比，并允许 no-match。
3. 已知和未解析信号分别计算。当前开发结果显示 unknown 响应有增量，但尚未做
   measurement-matched known 控制，因此不能解释为正式因果增量。
4. 四个微生物群落不先平均，直接检验局部响应是否跨群落复现。它们是同一实验中的
   重复生物背景，不是四项独立研究。
5. 同前体 10 ppm 边与质量不兼容的化学关系边分开。前者仍只是身份假说；后者不能将
   beta/SE 合并成同一分子效应。

## 未通过的部分

- WSE 的 3 个预设药物仍只有 sertraline 达到 Top-1；loratadine 依赖局部群落聚合才到
  第 4，不能称为稳定准确检索。
- no-match 查询仍会得到高分候选，当前没有足够独立 no-match 查询校准弃权阈值。
- known 与 unresolved 的测量数、覆盖和化学类别不匹配，当前差值不是正式 unknown
  information increment。
- 未进行离子/加合物家族折叠，支持数是谱学响应边数，不是代谢物数。
- 当前使用透明稀疏 cosine，而不是最终 Noise/双 encoder/多重排器统一算法。

## 下一项决定性实验

保持 120 个样本、20 个查询响应、51 个候选、响应特异性算子和全部分母不变，仅继续
替换谱学证据：B、完整谱学系统、带合法事件级上下文的系统；WSE 结果已经完成。主要终点为预设 3 个
重叠查询的最佳真关系名次/MRR、no-match 弃权，以及新增 unresolved 边实际改变的研究级
回答。任何方法不得获得额外候选或放宽效应纳入条件。

只有新算法在相同任务中稳定改善 omeprazole、保持 sertraline/loratadine，并在更多独立
查询中校准错误回答风险，才足以把本结果提升为论文主张。

## 可复跑入口

- 响应恢复：`tasks/prepare_msv000095331_drug_microbiome_responses.py`
- 开放检索：`tasks/analyze_msv000095331_chemprop_open_retrieval.py`
- 24 候选结果：
  `data/validation/msv000095331_chemprop_open_retrieval_localrep_v3_20261023/`
- 51 候选结果：
  `data/validation/msv000095331_contrast_atlas_open51_localrep_v1_20261023/`
- 51 候选 WSE 结果：
  `data/validation/msv000095331_contrast_atlas_open51_wse_localrep_v1_20261023/`

公开来源：MassIVE MSV000095331；GNPS2 task
`401db749414546aeb37222c752294861`；论文 DOI
`10.1038/s41467-025-65993-5`。
