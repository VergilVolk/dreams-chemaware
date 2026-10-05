# BioAware B42–B43：独立目录拓扑与增益机制审计

日期：2026-09-13  
状态：本地正式协议完成；全部统计使用 10,000 次 identity/formula cluster bootstrap；尚未进行新的未开封外部测试。

## 1. 必须先纠正的历史语义

B37/B41 中的历史字段 `network_member` 和 `known_log_degree` 不是 Rhea 特征。它们来自 MetDNA2 EMRN 的 `minimum_step == 0` 图，即 strict KEGG 派生图；部分早期构建器还将度数限制在当时可观测的 MS1 子图。

因此：

- B37 的 `+5.93 pp` 应称为 **strict-KEGG-derived catalogue topology prior**；
- 历史 B41 把该字段命名为 `rhea_topology_replay` 是科学语义错误；
- B41 已停止继续执行，独立 Rhea 检验由 B42 取代；
- 该纠正不改变 B37 的数值，但改变其科学解释。

B42 对历史 B37 转移进行了 860/860 精确回放。当前静态 strict KEGG 重建仅 6 个候选行与历史字段不同（候选行差异率 0.000896），原因是历史构建器存在来源特异的可观测子图限制；因此历史字段只用于精确回放，静态重建作为独立臂报告。

## 2. B42：真正独立的 Rhea–KEGG 检验

Rhea 图由 78,843 条 participant 记录重新构建：去除 currency metabolites，排除参与物超过 8 个的反应，再形成无向化合物图。得到 9,963 个成员、22,481 条边。strict KEGG 有 5,741 个成员、7,353 条边。

两图并不等价：

- member Jaccard：0.3503；
- edge Jaccard：0.1776。

因此 Rhea 与 strict KEGG 可以作为相对独立的目录拓扑来源。EMRN 包含 strict KEGG step-0 图，只能视为扩展资源，不能算独立复现。

### 六来源 nested leave-source-out 结果

| 臂 | Recall@1 Δ | corrected / introduced | λ=2 risk net | formula-cluster 95% CI |
|---|---:|---:|---:|---:|
| strict KEGG 历史精确回放 | +5.930 pp | 55 / 4 | 47 | [+3.356, +8.782] pp |
| strict KEGG 静态重建 | +5.698 pp | 54 / 5 | 44 | [+3.198, +8.508] pp |
| 独立 Rhea 拓扑 | +6.047 pp | 57 / 5 | 47 | [+3.513, +8.848] pp |
| EMRN 扩展拓扑 | +5.698 pp | 56 / 7 | 42 | [+3.205, +8.484] pp |
| KEGG+Rhea 共识 | +6.279 pp | 55 / 1 | 53 | [+3.879, +8.924] pp |
| KEGG+Rhea+EMRN | +6.047 pp | 62 / 10 | 42 | [+3.115, +9.347] pp |

独立 Rhea 在六个来源上相对 DreaMS 均不为负。说明“已知代谢物目录拓扑可作为候选先验”可以从 strict KEGG 迁移到独立构建的 Rhea，而不是单一资源的偶然现象。

但是不能声称多目录融合显著优于单目录：

- Rhea vs strict KEGG：+0.116 pp，29/28 query 相互翻转，formula CI [-2.241, +2.593] pp；
- KEGG+Rhea vs strict KEGG：+0.349 pp，21/18，formula CI [-1.329, +2.155] pp；
- 共识相对 strict KEGG 在 Mouse brain 来源下降 6.11 pp，未通过逐来源非负门。

结论：**跨资源可迁移成立；多资源增量未成立。**

## 3. B43：约 6 pp 到底来自哪里

B43 不重训模型、不重选门，只对冻结 B42 OOF 转移按候选目录覆盖关系分层。

### strict KEGG

- 真值在目录、DreaMS 错候选不在目录：101 queries，+45.54 pp，46/0；
- 真值和 DreaMS 候选都在目录：571 queries，+1.05 pp，9/3，formula CI 跨 0；
- 全部候选都在目录：191 queries，+2.09 pp，5/1，formula CI 下界为 0。

strict KEGG 的主要收益来自成员不对称；目录内度数排序证据尚不显著。

### 独立 Rhea

- 真值在目录、DreaMS 错候选不在目录：73 queries，+43.84 pp，32/0；
- 真值和 DreaMS 候选都在目录：671 queries，**+2.98 pp，25/5**，formula CI **[+0.69, +5.69] pp**；
- 全部候选都在目录：246 queries，+3.25 pp，9/1，formula CI 约 [-0.01, +7.83] pp。

Rhea 不仅提供“是否收录”信息；在双方均收录时，静态拓扑度数仍提供显著候选区分信号。

### KEGG+Rhea 共识

- 真值在任一目录、DreaMS 错候选均不在：72 queries，+44.44 pp，32/0；
- 真值和 DreaMS 候选都在至少一个目录：699 queries，**+3.15 pp，23/1**，formula CI **[+1.10, +5.64] pp**；
- 全部候选都在至少一个目录：274 queries，**+4.01 pp，11/0**，formula CI **[+0.82, +7.67] pp**。

因此 B42 的约 6 pp 不能被归结为单纯目录成员偏置。约一半修正来自成员不对称，剩余部分包含可重复的目录内拓扑区分信号。

## 4. 现在已经解决和仍未解决的问题

已经解决：

1. 找到一个工程上可执行、风险较低的候选先验；
2. 在六来源 nested leave-source-out 中达到约 +6 pp；
3. 用独立 Rhea 与 strict KEGG 证明其不是单一目录偶然性；
4. 证明 Rhea/共识在 mapped competition 中仍有约 +3 pp，而不是只有成员收录偏置；
5. 证明历史 B37 的语义并非 Rhea，并完成纠正。

仍未解决：

1. 所有六来源都已经打开，尚不是新的盲测；
2. 该先验使用候选身份查图，属于 embedding 后的候选专家；
3. 它没有使用同一样本观测代谢物，不是样本生物上下文；
4. 它没有证明真实反应边或多跳传播优于静态拓扑；B40 已显示朴素 Rhea 扩散显著伤害困难上下文；
5. 它不能无损地蒸馏为只看原始谱图的共享 embedding，因为目录成员和度数不是谱图的确定函数；
6. 尚不能称 SOTA、前瞻性注释提升或生物机制发现。

## 5. 冻结决策

### 保留

- 将 **BioAware-Catalogue v1** 作为独立候选专家保留；
- 首选 KEGG+Rhea 共识，因为当前 OOF 中新增错误最少（1 个），但不得宣称显著优于单独 strict KEGG；
- 输出必须保留“spectral score、目录成员、目录度数、触发门、最终切换”的逐 query 审计路径。

### 停止

- 不再把 B37/B41 称为 Rhea 结果；
- 不再调 B40 的 hop/decay 试图挽救朴素扩散；
- 不把目录先验直接蒸馏进共享 spectrum-only embedding；
- 不把 EMRN 当成 strict KEGG 的独立复现。

### 下一步唯一高性价比路线

1. 冻结一个可部署的 BioAware-Catalogue v1 工件；
2. 在模型、特征、门和阈值全部冻结后，建立一个此前未打开的外部候选检索面板；
3. 同时报告全体、mapped competition、all-candidates-mapped、unseen formula、来源分层与 introduced-error 审计；
4. 只有该盲测通过，才讨论把目录专家与后续真正的 sample-context adapter 组合；
5. 真正的生物上下文路线必须使用 truth 从 seed 中移除的样本局部观测网络，并证明真实图优于 degree-preserving/null graph。当前数据不满足时，应先补队列，而不是继续发明传播函数。

## 6. 工件

- B42 结果：`data/validation/bioaware_b42_independent_catalog_topology_localcheck_20260913_v2/`
- B43 结果：`data/validation/bioaware_b43_catalog_gain_mechanism_localcheck_20260913_v1/`
- B42 入口：`tasks/audit_bioaware_b42_independent_catalog_topology.py`
- B43 入口：`tasks/audit_bioaware_b43_catalog_gain_mechanism.py`

