# BioAware B38：显式反应路径特异性与上下文表示准入协议

## 一、B37 后不可再混淆的事实

B37 在冻结六域 860-query 开发协议中复现并拆解了既有 BioAware 增益：

- `network_member`：Recall@1 `+4.77 pp`，42 corrected / 1 introduced；
- `network_member + known_log_degree`：Recall@1 `+5.93 pp`，55 / 4；
- catalog observability：仅 `+0.47 pp`；
- 现有聚合反应特征未证明 reaction-specific 增量；
- 55 个 topology 修正中 46 个来自 truth 属于网络而 DreaMS 错候选不属于网络；4 个新增错误主要由高 degree 错候选造成。

因此，当前可部署的工程模块是 catalog topology calibrator，而不是已经成立的反应传播算法。它依赖候选数据库，不是谱图内禀属性，不得作为 clean-spectrum shared embedding 的教师。

## 二、B38 的唯一主问题

在同一候选图、同一外层留域、同一模型容量下，保留完整的
`seed -> reaction -> candidate` 路径身份、方向、化学计量、完整度、谱学一致性和样本共丰度后，真实反应路径能否相对 `DreaMS + catalog topology` 再稳定提升至少 `3 pp`，并显著优于结构/机会匹配空模型？

若不能，停止“Rhea 路径提高检索”的主张；保留 catalog 模块，并转向 feature/ion-family 全局一致性优化。若能，才允许训练上下文表示。

## 三、B38-M0：显式路径事件账本

每一行必须是一个不可再聚合的事件：

`(source, polarity, query_id, candidate_id, seed_query_id, seed_identity, reaction_id, direction)`。

必须保留：

- Rhea/KEGG 反应身份和左右侧方向；
- seed 分数、seed degree、candidate degree；
- 非 currency 反应物/产物集合、化学计量和两侧完整度；
- candidate specificity 与 dependency group；
- actual-edge 与匹配非邻居的谱学差值，包括直接峰、中性丢失、modified cosine；
- actual-edge 与匹配非邻居的共丰度差值；
- 缺失类型：`no_path`、`path_unobservable`、`complete_nonspecific`、`complete_specific`。

禁止用 truth、corrected/introduced、外层结果选择事件、特征或缺失填充值。禁止把缺失统一填 0。

### M0 准入门

- 完整复现 B37 的 860 query、候选集合、DreaMS 分数和 baseline rank；
- 每个 query 恰有一个 truth，仅用于最终评价；
- 至少 90% 的可检验真实边拥有 3 个相同 source/polarity/adduct、degree/质量/元素/结构匹配的非邻居对照；
- 路径证据在总体至少 50% query 内候选间变化，且每个来源至少 30%；
- 输出每个来源的四类缺失比例，不允许静默丢弃；
- 所有外层 held-out identity、formula、reaction component 均与训练拟合隔离。

M0 不读取 embedding outcome，不拟合模型，不报告性能提升。

## 四、B38-M1：反应特异性因果压力测试

### 固定主基线

`DreaMS + network_member + capped_degree`。degree cap、门控阈值和所有超参数只在内层训练域选择。

同时保留两个工程工作点：

1. conservative：`network_member`；
2. high-recall：`network_member + capped_degree`。

### 固定真实路径证据

1. 方向支持且 hyperedge 两侧完整；
2. 对候选具有特异性，而不是同时支持大量同分子式候选；
3. actual-edge 谱学一致性减去 matched non-edge 谱学一致性；
4. actual-edge 共丰度减去 matched non-edge 共丰度；
5. 依赖校正后的多 seed 独立见证；
6. 反应类型与预期质量/元素变换一致。

### 必须击败的空模型

- degree-preserving reaction rewiring；
- source/polarity/seed-score/degree 分层内 seed identity permutation；
- 质量、元素、结构、degree、谱图数量匹配的非反应邻居；
- path availability / path count / network degree 的 opportunity-only 模型。

### M1 主终点

相对 `catalog topology` 的 outer leave-source-out Recall@1 增量，而不是相对裸 DreaMS 的总增量。

### M1 准入门

- 增量 Recall@1 >= `+3.0 pp`；
- identity-cluster 与 formula-cluster 多重校正 CI 下界均 > 0；
- corrected identity > `2 * introduced identity`；
- 至少 25 个 corrected identity、净增至少 15、覆盖至少 30 个 formula；
- 所有来源 risk-net 非负，至少 5/6 来源 Recall@1 为正，最差来源不低于 `-1 pp`；
- 真实路径效果超过 rewiring 和 seed-shuffle 分布的 95 百分位；
- matched-control 完整覆盖 >=90%。

未通过时，不得进入 shared embedding，也不得继续调 gate 掩盖失败。

## 五、B38-M2：候选上下文集合模型（仅在 M1 通过后）

模型输入不是十个路径汇总标量，而是每个候选的无序路径事件集合。使用小型、容量受控的 DeepSets 或 attention pooling：

`h_c = Pool({phi(path_event_j)})`

`score(q,c) = spectral_score(q,c) + catalog_score(c) + gate(q,c) * bounded_residual(h_c)`

要求：

- residual 有界，证据缺失时严格退化为 catalog baseline；
- 外层留 biological source；内层同时隔离 identity、formula、reaction component；
- 固定容量对照：聚合标量模型、无 reaction ID 模型、无 direction 模型、无 data-layer evidence 模型；
- 不把外层性能用于选模型、阈值或温度。

这一步的创新点不是“给 DreaMS 加网络分”，而是学习哪一组方向一致、实验可见、候选特异的反应路径可以安全消解同分子式候选。

## 六、B38-M3：表示注入的正确边界

### 6.1 首选：上下文候选 embedding

若增益依赖 seed、样本共丰度或候选图，正确形式是：

`z_c_ctx = normalize(z_c + alpha_c * A(h_c))`

查询谱图仍使用 DreaMS 表示；候选表示由当前样本上下文调整。证据不足时 `alpha_c = 0`。

### 6.2 shared clean-spectrum embedding

只有当 M1/M2 的有效动作可以由单张谱图稳定预测，且在候选库、seed 集合和数据库改变时方向保持，才允许把该部分转成共享 encoder 的排序/对比损失。catalog membership、degree 和样本特异路径不得蒸馏进 clean-spectrum encoder。

## 七、与前沿方法的关系

- MetDNA/KGMN 的关键是种子传播与多层数据网络约束，不是孤立的路径计数；
- MetDNA3 的关键是知识层与数据层预映射后保持两层拓扑一致；
- NetID 的关键是全局一致性优化和 target-decoy/FDR，而不是逐候选局部加分。

B38 必须吸收这三点：显式路径、知识与实验双层一致、全局冲突/空模型控制。单纯提高 catalog membership 的结果只能作为工程先验基线。

## 八、近期交付顺序

1. B38-M0 显式路径事件与 coverage/null 可构造性报告；
2. B38-M1 固定路径证据和三类空模型的 nested LOSO 结果；
3. 若 M1 通过，冻结 catalog + path 两个模块并建立新盲测；
4. 之后才训练 context-conditioned candidate embedding；
5. shared clean-spectrum embedding 作为“谱图可恢复分量”的后续独立问题。

任何阶段都不预先承诺 3--5 pp；`3 pp` 是进入下一阶段的科学门槛，不是通过调参必须做出的数字。
