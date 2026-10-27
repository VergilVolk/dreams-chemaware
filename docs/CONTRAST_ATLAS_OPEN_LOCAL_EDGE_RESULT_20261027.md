# Metabolomic Contrast Atlas：开放局部响应检索结果（2026-10-27）

## 结论

本轮完成了真正的多研究开放响应检索，而不是又一项两数据集匹配实验。方法以**研究级、有方向且带不确定性的响应**为查询对象，在完整跨来源候选全集中分别返回相似响应、相反响应和无校准匹配，并把同前体身份假设与非同质量化学关系严格分开。

结果是**方法实现成立、当前生物学检索性能仍不合格**：局部响应证据广泛存在，但原始 0.1-Da fragment cosine 图不能把这种证据稳定转化为正确的开放研究关系。该负裁决冻结，不再在这批 consumed 数据上调阈值。

全部数据均为 `development/consumed`；没有打开新的封存生物学测试集。

## 实际完成的计算

最终产物为 `data/validation/contrast_atlas_open_local_edge_enrichment_v5_20261027`。

| 项目 | 分母 |
|---|---:|
| 独立 study/source family | 15 / 15 |
| 研究级对比 | 51 |
| 响应记录 | 238,412 |
| study-level 化学实体 | 54,861 |
| 冻结定向谱学边 | 294,445 |
| canonical mutual edges | 41,082 |
| 10-ppm 同前体身份假设边 | 17,093 |
| 非同质量化学关系边 | 23,989 |
| 定向 query-candidate 对 | 2,286 |
| world × track 候选得分行 | 20,574 |
| 每个查询的跨来源候选 | 39–50 |
| 整谱响应重排 | 256 次 |

没有删除零边或不可回答候选；它们仍在固定分母中并显式弃权。

## 算子

每个实体的连续响应权重为

\[
w_i=\frac{\beta_i}{\sqrt{\beta_i^2+SE_i^2}}.
\]

对于冻结的互为近邻谱学边，分别累积两端同号与异号的局部响应能量。不同实体的 `beta/SE` 从不合并，也不被称为同一分子的 meta-analysis。

背景分布通过研究内、共同身份分层的**整条多对比实体响应谱重排**获得。候选先得到自身置换标准化分数，再以每个查询在完整候选全集上的 max-T 分布控制“挑最好候选”的选择风险。

两个证据轨独立输出：

- `same_precursor_10ppm_identity_hypothesis`：互为近邻且前体差在 10 ppm 内；只是身份假设，不能排除异构体、加合物和离子家族冗余。
- `nonmass_chemical_relation_hypothesis`：两端前体均有效但差异大于 10 ppm；只支持不同观测实体之间的响应关系，绝不冒充同一分子。

三个 ChemProp 来源仅有 2–3 个生物重复，而冻结汇总表不足以重建精确 Welch 自由度。因此正常近似 feature FDR 边数仅作诊断，**不参与排名或回答**；正式分数使用连续收缩效应及整谱置换。

## 决定性结果

### 1. 同药开发正控：信号存在，开放检索仍未解决

12 个同药、跨 accession 的 canonical 正控，每次查询面对 39 个外来源对比。`unresolved + 10-ppm identity hypothesis` 的结果为：

- similar Top-1：`1/12`；
- similar Top-5：`7/12`；
- median similar rank：`4.5`；
- 正确候选的 query-level max-T `p <= 0.05`：`9/12`。

这组数字揭示了关键差别：`9/12` 表明正确同药对中存在超出研究内随机响应归属的局部证据，但只有 `1/12` 成为 Top-1，说明许多错误药物对也具有更强的共享响应。检出 response-associated signal 不等于完成 biological retrieval。

它没有超过此前 entity-mean 局部方法的整体结果（同样为 `1/12` Top-1、`7/12` Top-5、median rank `4.5`），因此不能宣称新算子改善了开放检索。

非同质量关系轨更弱：同药正控为 `2/12` Top-1、`4/12` Top-5，只有 `2/12` 正确候选通过 query-level max-T。关系层尚未取得普适证据。

### 2. 跨项目压力测试失败

- HFD exact-intervention family：`0/2` similar Top-5，`0/2` 通过 query-level max-T；
- IBD near-exact condition family：`0/3` similar Top-1，`1/3` Top-5，`1/3` 通过 query-level max-T；
- post-hoc obesity family 不具 primary 资格，也没有形成可晋级结果。

所以 ChemProp 内部信号没有推广为跨项目 Contrast Atlas 成功。

### 3. 开放调用不能被误写成发现

在 unresolved 身份假设轨上，`45/51` 查询的最佳 similar 候选通过 max-T；但已登记关系的正确检索率仍很低，而且许多非 ChemProp 查询被药物或植物对比排到首位。max-T 排除的是“随机重派实体响应谱”这一零模型，并不证明候选之间存在正确生物学关系。

因此所有开放 Top hit 都带 `adjudication_status=unadjudicated_hypothesis`。`no_calibrated_match` 也只表示在当前有限功效规则下没有候选通过，不表示生物学上不存在关系。

## 一个真实但边界严格的实体链

唯一的同药 Top-1 为：

`MSV000094899 / ketoconazole` → `MSV000093571 / ketoconazole`

- 候选数：39；
- unresolved 10-ppm identity-hypothesis similar rank：1；
- enrichment z：25.589；
- query-level max-T p：`1/257 = 0.003891`。

主要局部边包括：

| m/z（query → candidate） | cosine | beta（query → candidate） |
|---|---:|---:|
| 268.1040 → 268.1041 | 0.9916 | -5.596 → -4.914 |
| 204.1245 → 204.1245 | 0.9880 | +2.187 → +2.804 |
| 417.1180 → 417.1183 | 0.9793 | -7.859 → -5.114 |

这些实体在共同严格库策略下属于 unresolved，故 known-only 不能提供同等实体链。它们证明结构名称不是参与研究级响应复现的前提；但该案例来自同一个更大的 ChemProp 项目且已被开发使用，不能当作外部生物学发现。

## 最新统一算法交付是否可用

最新 23-feature Additive 确实是有价值的算法资产，但当前交付是“查询谱图 → 结构候选库”的候选排序接口，不是可直接替换任意 study-spectrum ↔ study-spectrum 边的 Atlas 接口。本地现有的完整多研究图仍使用原始 fragment cosine。

为避免算法端重新生成另一张候选图，本轮已输出：

`frozen_mutual_edge_algorithm_handoff.tsv.gz`

它固定 41,082 条边及以下连接键：

`left_study_id, right_study_id, evidence_world, left_entity_id, right_entity_id`

算法端下一次只需在这些键上交付 WSE、Noise、B 和统一 Additive 的谱学证据，不得新增、删除或按生物学结果筛选边。生物端随后原样复算 2,286 个候选对；这样才能回答“更好的算法是否把 9/12 的局部信号转成更多正确 Top-1”，而不是比较两套不同候选宇宙。

Atlas 响应聚合、置换和开放查询已在本机完成。多 encoder 编码不适合继续在本机硬跑：历史 official-only 编码 17,358 个实体已耗时 8,939.5 秒，而当前需要 54,861 个实体和多个 encoder。高性价比方案是服务器只做一次 phenotype-blind 编码/边评分，本机继续完成全部响应分析。

## 冻结裁决与下一步

1. 保留本轮方法、完整分母、逐候选分数、Top hit 和逐实体贡献证据。
2. 不再在这 15 个 consumed 来源上调整阈值、邻居数或响应权重来救 Top-1。
3. 下一次唯一允许的同面板开发比较，是在**同一 41,082 条边**上替换冻结算法分数。
4. 若统一算法仍不能提高正确关系的 Top-1/Top-5，而只增加 max-T 显著候选，则算法—生物闭环不成立。
5. 即便同药正控改善，最终论文主张仍需新的独立重复扰动或前瞻验证；当前没有外部封存生物学成功结果。

## 可复现产物

- 实现：`tasks/run_contrast_atlas_open_local_edge_enrichment.py`
- 测试：`tasks/test_run_contrast_atlas_open_local_edge_enrichment.py`（3 passed）
- 主报告：`data/validation/contrast_atlas_open_local_edge_enrichment_v5_20261027/report.json`
- 固定候选得分：`open_local_edge_candidate_scores.tsv.gz`
- 开放查询输出：`open_local_edge_top_hits.tsv`
- 关系盲后审计：`registered_relation_ranks.tsv`
- Top hit 实体证据：`top_hit_contributing_edges.tsv.gz`
- 算法交接边：`frozen_mutual_edge_algorithm_handoff.tsv.gz`

