# BioAware 科学问题重置与唯一允许的下一步（2026-09-04）

## 1. 本次纠错

下列两条临时路线已经撤销，不得提交到服务器，也不得作为 BioAware 证据：

1. `R0 candidate-teacher validity` 使用了已经废弃的 595-query 负离子候选表。该表包含未经结构—离子质量完整性过滤的 MoNA 记录，并且把冻结专家的风险门控改成了全量重排。它检验的不是已经得到 +3.47 pp 的 BioAware v2。
2. `R1 multilayer-teacher validity` 把已经打开的外部评估结果重新用作训练样本，并把样本特异网络压成静态候选向量。它既不是独立外部验证，也不是样本内联合注释，更不是共享 embedding 微调。

两条路线的数值只记录为失效实现的诊断，不得进入论文、模型选择或后续教师构造。

## 2. 当前已经成立的结果

当前唯一可辩护的 BioAware 性能结果仍然是化学完整性过滤后的负离子候选专家：

- 固定协议：548 个 Level-1 `[M-H]-` 查询；结构验证过的 MoNA `[M-H]-` 参考库；10 ppm；每个 IK14 聚合；并列算错。
- `full_bioaware`：相对官方 DreaMS 提升 +3.47 pp，19 corrected / 0 introduced；formula-cluster 95% CI 为正。
- 高召回消融 `full_no_edge_gate`：+4.01 pp，23 / 1，但不是主部署配方。
- 候选内网络特征置换：经验 `p=0.0099`。

这是已经打开的四个生物来源上的严格开发证据，不是独立外部验证、不是 SOTA 证明，也不改变 DreaMS embedding。

## 3. BioAware 实际包含两个不同科学问题

### 3.1 样本上下文候选专家

问题是：给定一个真实 LC-MS 样本中的 MS1 特征、MS2 谱图和少量高置信种子，反应网络、离子家族和峰相关结构能否在固定错误发现率下增加正确注释覆盖率？

该问题的推理输入必须包含整个样本或批次的特征图。它的输出是候选后验概率或样本上下文候选 embedding；它不是只由单张谱图决定的通用 embedding。

图中至少需要四类因子：

- 谱图一元证据：冻结 DreaMS 候选分数；
- 离子/同位素/加合物约束：同一中性分子的多个观测峰不得重复计数；
- 数据层边：共洗脱、跨样本相关和原始 MS2 边；
- 生化超边：带方向、化学计量和 currency-metabolite 处理的 Rhea 反应。

必须有显式 `unknown/abstain` 状态，并用 target-decoy 或候选置换校准 FDR。主要终点不是单独 Recall@1，而是固定 FDR 下的正确注释数、覆盖率、corrected/introduced 以及 no-context fallback。

### 3.2 反应感知但身份保持的共享谱图 embedding

问题是：在推理时只输入一张谱图时，是否能让共享 encoder 同时保持分子身份判别，并额外解码与谱图化学变化一致的反应类型？

这里必须满足：

- 只有同一分子的跨条件谱图是 retrieval positive；
- 反应邻居是不同分子，永远不是 identity positive；
- Rhea 关系只能作为辅助 relation-decoding 监督；
- 任何辅助梯度不得降低同分子检索、MCES-near margin 或 preservation；
- 样本共现、疾病标签、候选真值和网络路径在单谱推理时不可用。

因此“样本上下文 embedding”和“通用单谱 embedding”不能混称。前者需要样本图作为推理输入；后者只能学习可由谱图本身支持的反应变换表征。

## 4. 现有 B0 为什么仍不够严谨

旧 B0 同时做了对照构造、公式社区切分、PCA 和逻辑回归。其失败不能解释为没有生物信号，原因包括：

1. 非邻边候选只来自其他真实反应的 target，而不是完整合格分子宇宙；
2. 全局 Hungarian 匹配没有保证逐组化学 caliper，导致 Tanimoto、质量差和重原子数严重失衡；
3. formula-community 切分后部分 fold 为空或极小；
4. “是否存在谱学关系信号”和“某个低容量分类器能否学到”被混成同一个检验；
5. degree-preserving topology null 与 chemically matched non-edge control 被混为一类对照。

## 5. 唯一允许的下一项实验：B0-M0 配对可辨识性预检验

在训练任何 BioAware head 或 adapter 之前，先完成一个不训练模型的配对实验。

### 5.1 实验单位

每个单位固定一个源分子 `s`：

- 1 个 Rhea 真实反应靶点 `t+`；
- 1 个严格匹配、未记录为 `s` 的一步邻居、且自身有 Rhea 覆盖的主对照 `t-`；
- `t+` 和 `t-` 必须来自同一个 P3-disjoint、真实谱图、化学结构有效的分子宇宙。

每条无向 Rhea 边以两个 source-fixed 方向分别构造条件对照；后续统计按原始无向边、身份和分子式聚类，不能把两个方向当作完全独立重复。额外近邻对照只用于敏感性分析，不以牺牲主分析的反应边覆盖为代价。

### 5.2 对照构造

主对照通过全局受约束边重连产生：固定所有 source，并在真实 Rhea target 多重集合内做一对一置换；已知一步邻边和 self-edge 被禁止。完整置换天然保持 source 与 target 的网络度数、谱图数和采集覆盖分布，再在冻结的化学 caliper 内最小化变换差异。更广的完整分子宇宙非邻边仅作为后续敏感性分析；不得在观察 embedding 结果后改变 caliper。

最低匹配变量：

- source 固定；
- Morgan Tanimoto；
- `|mass(t)-mass(s)|`；
- 重原子数差；
- C/N/O/P/S/卤素元素变化；
- 环数、形式电荷、HBD/HBA；
- target 的非 currency Rhea degree；
- target 的真实谱图数和离子模式可观测性。

主对照是 chemically matched non-edge；degree-preserving edge rewiring 是独立的 topology null，二者分别报告。

### 5.3 首个终点

不拟合分类器，直接计算每个匹配组：

`delta_g = relation_evidence(s,t+) - mean relation_evidence(s,t-)`。

其中 relation evidence 至少分开报告：

- 官方 DreaMS prototype cosine；
- peak/neutral-loss transformation evidence；
- 两者组合只能作为预注册的次要终点。

按 source identity 和 source formula 分别 cluster bootstrap；再做组内标签置换。不能用全局 pair AUC 替代组内效应。

### 5.4 预检验硬门

- 至少 500 个完整匹配组、300 个 source identities、200 个 source formulas；
- 主分析每组固定 1:1，缺少严格对照则整组删除；
- 所有连续协变量总体 SMD <= 0.10；
- degree bin、spectrum-count bin、稀有元素变化、电荷和观测条件必须组内精确匹配；其余变量必须满足冻结 caliper，组内 P90 作为诊断报告而非重复的硬阈值；
- 不得有 P3 identity overlap；
- 不使用 phenotype、候选真值结果、P2b 或当前 BioAware 的预测作为特征。

若匹配门失败，结论只能是“当前数据不能识别反应特异信号”，不得训练 B1；若匹配门通过但 DreaMS/峰级条件增量失败，停止通用单谱 BioAware embedding，继续样本上下文专家；若通过，才进入 relation-decoding adapter。

## 6. B0-M0 之后的决策树

1. **配对不可平衡**：扩充谱库或反应覆盖；不训练。
2. **化学匹配后仅峰级变换证据有效，DreaMS 无增量**：BioAware 放在样本图/候选专家层，不能声称共享 embedding 已有反应信息。
3. **DreaMS 有稳定组内增量**：训练关系类型 readout；先冻结 encoder 证明可读性。
4. **冻结 readout 通过**：才允许零初始化、低秩、受限 adapter；主任务仍是 identity retrieval，辅助梯度冲突时丢弃。
5. **独立真实队列**：冻结 548-query 专家和任何 adapter，在从未打开的 Level-1、原始样本级 MS1/MS2 队列一次性验证。

## 7. 近期最高性价比排序

1. 保存并冻结当前负离子 BioAware v2，不再在四个已打开来源调参。
2. 完成 B0-M0 配对可辨识性预检验，先修对照，不先训模型。
3. 并行寻找新的带 Level-1 真值和样本级 MS1/MS2 的外部队列，用于专家一次性验证。
4. MTBLS13729 只用于无真值的生物学发现和上下文图可运行性，不用于证明候选身份准确率或 SOTA。

## 8. 论文主张边界

MetDNA、KGMN 和 NetID 已经证明网络传播与全局一致性有用。我们的可检验增量只能是：

> 以冻结 foundation spectral evidence 为一元项，在身份隔离、化学完整性、原始 MS2 边验证和显式 abstention 下学习候选特异的生物网络增量；并进一步检验反应类型是否能以不损害身份检索的方式成为共享谱图表示的可解码辅助因子。

没有新的未见队列和固定 FDR 覆盖率结果前，不称 SOTA。
