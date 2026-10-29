# Metabolomic Contrast Atlas：完整 WSE 身份轴的全局响应检索结果（2026-10-29）

## 结论

本轮已经完成真正的系统级分析，不再以 IBD 的一个实体或一个小面板代替 Atlas：

- 15 个分析来源、13 个更高层 parent-project family；
- 51 个研究级 contrast；
- 238,412 条效应记录、54,861 个 study-entity；
- 291,330 条完整的 10-ppm 身份候选对；
- 2,286 个跨来源有向 contrast 候选；
- 256 次研究内、身份层内的整实体多-contrast response-profile 重排；
- 每个查询在完整 39–50 个候选上实施 query-level max-T。

结果没有支持“Atlas 已经成功”或“unknown 增量已经成立”。它揭示了更重要的系统性问题：**完整 WSE 身份轴能够检出大量非随机局部响应耦合，但当前开放检索主要受到证据机会量和 parent-project 响应枢纽支配，尚不能稳定找回真正跨项目的生物学关系。**

所有数据均为 `development/consumed`；本轮没有打开新的封存生物学外测。

## 1. 计算合同

候选轴在读取 `beta/SE`、表型和 registered relation 之前冻结。每个 query-entity × target-study 区块分别在三个诊断世界内重排：

| 世界 | 定义 |
|---|---|
| `all` | 全部物理化学兼容候选 |
| `known` | 两端均为共同严格库策略支持的实体 |
| `unresolved` | 两端均为该策略下的 unresolved entity |

每个世界独立选择唯一、正 WSE 且至少一个匹配碎片的 Top-1；WSE 完全并列时弃权。随后仅保留 reciprocal Top-1 作为同前体身份假说。不同实体的 `beta/SE` 从不合并，也不称为同一分子的 meta-analysis。

响应权重为

\[
w_i=\frac{\beta_i}{\sqrt{\beta_i^2+SE_i^2}}.
\]

相同与相反方向的局部响应能量分开累计。候选特异 null 来自研究内、共同 identity-status 层内的整条多-contrast response profile 重排；开放选择风险由 query-level max-T 控制。

完整身份轴只有 10-ppm 同前体候选，因此本轮**不评价不同质量的化学关系轨**。

## 2. 证据机会高度不平衡

| 项目 | 数量 |
|---|---:|
| strict-known entities | 979 / 54,861 |
| unresolved entities | 53,882 / 54,861 |
| all-world reciprocal identity hypotheses | 34,219 |
| known-world reciprocal identity hypotheses | 918 |
| unresolved-world reciprocal identity hypotheses | 32,328 |
| unresolved / known 边机会比 | 35.22× |

all-world 的 34,219 条边由以下部分组成：

- known–known：867；
- known–unresolved：1,419；
- unresolved–unresolved：31,933。

因此，`all > known` 不能直接解释为 unknown 化学信息更有价值。它首先可能是可用实体和可连接边数量显著增加的结果。三个世界是必要诊断，但不是正式的 unknown-increment 四世界裁决。

## 3. 开放检索暴露出明显的 parent-project 枢纽偏倚

all-world 身份轨的 51 个查询中：

- similar：49/51 通过 query-level max-T；
- opposite：47/51 通过 query-level max-T。

这些高比例只表明最优候选超出了指定的 response-profile 重排 null，不能说明它是正确的生物关系。

最直接的反证来自候选来源集中度：

| 指标 | 结果 |
|---|---:|
| ChemProp 三来源在 2,286 候选行中的占比 | 990/2,286 = 43.31% |
| all-world similar Top-1 指向 ChemProp | 38/51 = 74.51% |
| Top-hit / candidate-exposure 比 | 1.72× |
| 非 ChemProp 查询指向 ChemProp | 14/27 |
| ChemProp 查询仍指向同一 parent project | 24/24 |
| 单一最大候选 hub | `drug_cilnidipine`, 11/51 queries |

三项 ChemProp accession 属于一个更大的 parent project，不能当作三个独立生物学来源。上述结果说明当前 null 能排除随机响应归属，却不能排除大型项目、药物时间响应和广泛共享化学程序形成的检索枢纽。

所以，49/51 的 max-T call 不是“Atlas 找到了 49 条正确联系”，而是一个**全局特异性不足的警报**。

## 4. Registered relation 必须按真正独立单元解释

42 个有向关系对应 21 个 canonical relation，不能当作 42 个独立重复：

- 12 个 canonical exact-drug 关系：同一 ChemProp parent project 内的开发期正控；
- 2 个 canonical HFD 关系：跨项目预设 exact-intervention；
- 3 个 canonical IBD 关系：跨项目 near-exact condition；
- 4 个 canonical obesity 关系：post-hoc，只作描述。

### 4.1 共享项目 exact-drug 正控

all-world：

- directional Top-1：3/24；Top-5：18/24；
- canonical 任一方向 Top-5：11/12；
- canonical 两方向均 Top-5：7/12；
- canonical 两方向均通过 max-T：12/12。

unresolved-world 的结果接近：任一方向 Top-5 11/12、两方向均 Top-5 6/12。known-world 仅为任一方向 6/12、两方向均 3/12。

这证明大规模 unresolved 边能够恢复同一 parent project 内的响应正控，但边机会差 35.22×，而且这些正控已经被开发反复使用。因此它们不能单独证明 unknown 特异增量，更不能当外部验证。

### 4.2 真正跨项目的 HFD / IBD

HFD 的 2 个 canonical 关系：

- all-world 任一方向 Top-5：0/2；
- unresolved-world 任一方向 Top-5：0/2；
- 通过 max-T：0/2。

IBD 的 3 个 canonical 关系：

- all-world 任一方向 Top-1：0/3，任一方向 Top-5：1/3；
- known-world 任一方向 Top-1：1/3，两方向均 Top-1：1/3；
- unresolved-world 任一方向 Top-1：0/3，任一方向 Top-5：2/3，但两方向均 Top-5：0/3。

因此，当前五个 source-independent canonical 关系没有显示 all/unresolved 相对 known 的一致改善。尤其 IBD 的严格 known 证据反而恢复了一项双向 Top-1，不能把 IBD 再包装成 unknown-required flagship。

## 5. 与直接 response-level 基线的关系

现有 strict-known response 基线在相同 2,286 候选分母上报告：

| 方法 | primary directional Top-1 | Top-5 |
|---|---:|---:|
| continuous full-InChIKey cosine | 1/34 | 2/34 |
| iDMET-like signed overlap | 1/34 | 6/34 |
| iDMET+-like ORA | 2/34 | 10/34 |
| WSE identity, known-world | 5/34 | 11/34 |
| WSE identity, all-world | 3/34 | 20/34 |

这是描述性上下文，不是公平的胜负表：strict-known response baseline 使用确认的 full-InChIKey 对齐；WSE 图使用未确认的同前体 reciprocal identity hypothesis。all-world 的 Top-5 增加也与大幅增加的边机会纠缠。

旧 fragment-admitted graph 与本轮 complete WSE identity graph 的候选准入不同，因此也不能把两次结果差直接写成 WSE 的算法增量。

## 6. 冻结裁决

| 问题 | 当前裁决 |
|---|---|
| 是否完成大规模研究级检索？ | 是：15 sources / 51 contrasts / 2,286 directions |
| 是否证明 source-independent 生物检索成功？ | 否 |
| 是否证明 unresolved 信息有独立增量？ | 否；机会量未匹配，shuffled-MS2 控制未完成 |
| 是否检验了自研统一算法增量？ | 否；Official / Noise V1 / Full23 尚未在同一 291,330 对完整轴交付 |
| 是否有外部生物验证？ | 否 |
| 是否可报告 no-match accuracy/FDR/AUROC？ | 否；未注册候选不是阴性，signed entity truth 为 0 |

这不是“什么都没做”。相反，它把此前被少数案例掩盖的核心失败定位清楚：**当前响应算子能够发现大量结构化共享响应，却不能可靠区分真正相关的研究与项目级响应 hub。** 下一步优化应针对这个系统性问题，而不是继续寻找一个漂亮实体截图。

## 7. 下一步只做三件高性价比工作

1. **机会匹配的 unresolved 控制。** 在每个 study 内按前体质量、charge/polarity、contrast 覆盖和目标来源候选机会匹配 known 实体，多种子重抽样；明确称 opportunity-matched，而不是完整 measurement-matched。
2. **parent-project holdout。** 将三项 ChemProp accession 视为同一项目；对 ChemProp 查询禁止同项目候选，并单独报告 27 个非 ChemProp 查询的来源集中度。该分析回答跨项目迁移，而非同项目批次恢复。
3. **同轴算法替换。** 算法端在同一 291,330 条完整候选轴返回 Official、Noise V1、Full23 的逐对分数。生物端原样复用本轮 2,286 分母、profile-null 和 max-T，不改变响应算子或阈值。

只有当 all/unresolved 同时超过 strict-known、机会匹配 unresolved 和 shuffled-response 控制，并且 Full23 在同一候选轴提高五个 source-independent canonical 单元或新的封存关系，才允许把“unknown 信息增量”和“统一算法生物增量”写成正结论。

## 8. 可复现产物

- WSE 完整轴适配器：`tasks/build_contrast_atlas_complete_identity_wse_alignment.py`
- 响应检索内核：`tasks/run_contrast_atlas_open_local_edge_enrichment.py`
- 全局总结器：`tasks/summarize_contrast_atlas_complete_identity_wse_global.py`
- 冻结 WSE 图：`data/validation/contrast_atlas_complete_identity_wse_alignment_v3_20261029/`
- 2,286 方向结果：`data/validation/contrast_atlas_complete_identity_wse_open_retrieval_v3_20261029/`
- 系统级摘要：`data/validation/contrast_atlas_complete_identity_wse_global_summary_v1_20261029/`
- 新增与相关回归测试：6 passed。

