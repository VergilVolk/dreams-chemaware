# BioAware B39：来源分层的反应动作识别与表示准入合同

## 1. 结论边界

B37 在六个已开放开发域上建立了一个稳定的候选目录先验：

- `network_member + known_log_degree` 相对裸 DreaMS 为 `+5.9302 pp`；
- 55 个修正中 46 个来自 truth 是网络成员、DreaMS 错候选不是网络成员；
- 4 个新增错误中 3 个由高 degree 错候选造成。

该结果是候选数据库校准，不是同一样本反应传播，也不是共享谱图 embedding 的改进。

B38-M1 否定的是当前实现：

> 在混合 seed 语义、混合 KEGG/Rhea 方向、缺少统一实验层对照的条件下，把一步路径压缩成候选级存在比例，不能在 catalog topology 之上提供增量。

它没有否定“方向明确、候选特异、实验层可见的反应事件”本身。B39 只检验后一个更窄的命题。

## 2. B38 暴露的五个结构性问题

### 2.1 六域并不共享同一个上下文定义

- 内部四域：truth identity 决定其 held-out rotation；每个 identity 有七个合成 seed contexts。
- ST001154：seed 来自实际样本，但 162/162 个 query truth 本身也在对应样本的 seed 表中；评价时利用 truth 将其移除，因此它是 sample-local leave-one-seed-out 反事实恢复，不是真实未知物部署。
- KGMN-200STD：seed 是同一标准混合物的 hidden-seed repeat。

三者不能被平均成同一种“seed context”。内部 rotation 适合开发防泄漏机制，KGMN 适合标准混合物压力测试；ST 保留真实样本内上下文结构，但 query truth 是人为从已知 seed 中隐藏的。当前六域中没有一个严格的 prospective unknown-annotation 队列。

### 2.2 B38 的主要特征丢失了事件身份

`mean(has_direct_path over contexts)` 同时丢失：

- 哪个 seed 支持候选；
- 哪个 reaction 支持候选；
- 路径方向是否可信；
- 一个事件支持一个候选还是多个候选；
- 反应两侧是否被实验观测；
- 真实边是否比匹配非邻居具有更强谱学或共丰度证据。

因此，B38 不是“路径集合模型的负结果”，而是“路径存在率标量的负结果”。

### 2.3 知识层和实验层没有在同一个事件上对齐

B38 负离子事件中约 88% 为 `direction_unknown`。内部 B3/B9 缓存是候选/rotation 级汇总，不能证明某条具体 Rhea/KEGG 事件获得实验支持；ST/KGMN 又没有对应的 B3/B9 controls。把这些列共同输入模型会把“有一条知识边”和“这条边在样本中被观测”偷换为同一件事。

### 2.4 当前空模型只保持节点 degree，不够强

反应重连还必须匹配：

- reaction component；
- hyperedge arity 与两侧非 currency 参与物数量；
- direction class；
- seed degree、seed score 与 seed visibility；
- candidate degree 与候选组覆盖率；
- source、polarity、adduct；
- 元素变化或反应家族。

否则真实边与重连边的差异可能仍是 opportunity 差异。

### 2.5 独立选 gate 混合了“证据增量”和“门控变化”

B38 每个 arm 单独在内层选择 gate。最终 Top-1 差异同时包含：模型拟合、概率校准、proposal 改变和 gate 改变。B39 必须同时报告：

1. 不经 gate 的候选排序增量；
2. 所有 arm 共用的、只依赖 DreaMS 边界状态的固定风险层增量；
3. 只有前两项通过后才报告的 nested end-to-end 增量。

## 3. 三个不可再混合的估计目标

### 3.1 Catalog prior

估计候选身份在当前参考数据库和代谢知识库中的先验可靠性：

`P(y=1 | DreaMS score, catalog membership, capped degree)`。

它是 B37 的工程模块。候选库变化时必须重新校准，不能注入 clean-spectrum encoder。

### 3.2 Sample-context reaction increment

主估计量为同一来源内：

`Delta_ctx = Accuracy(topology + admissible reaction events) - Accuracy(topology)`。

其中 admissible event 必须在推理时可观测，且不依赖 truth 或评价 outcome。不同来源先分别估计，再用来源级随机效应或保守最小效应汇总；禁止先拼接样本再拟合一个“跨域平均上下文”。

### 3.3 Spectrum-recoverable embedding increment

只有当一个 context 动作在改变 seed 集合、候选库和来源后仍可由 query/candidate 的谱图信号预测，才允许检验其是否能进入共享 clean-spectrum encoder。否则它只能进入 context-conditioned candidate representation。

## 4. 来源分层

| 层 | 上下文语义 | 当前数据 | 可以支持的结论 |
|---|---|---|---|
| L：sample-local LOSO | 同一样本的高置信 seed 与 feature network，但人为隐藏 query truth seed | ST001154 | 最接近部署的反事实恢复；不是 prospective unknown |
| H：hidden-standard | 标准混合物中隐藏一部分已知身份 | KGMN-200STD | 化学真值压力测试，不等同生物样本 |
| S：synthetic-rotation | 在来源内人为划分 seed/held-out identity | BV2cell、Mouse brain、Mouse liver、NIST plasma | 动作发现与消融，不作为同样本外部复现 |

主报告不得把 L/H/S 的 query 合并计算一个 Recall@1。可以分别报告，并对预先定义的同类来源做 meta-analysis。任何“真实样本部署增益”必须另有 query truth 未被种子构建流程使用的 prospective 或预先封存队列。

## 5. B39 的原子事件

每一行是：

`(source, context_type, sample_or_rotation, query, candidate, seed_feature, seed_identity, reaction, oriented_side)`。

必须保留以下字段，缺失值必须有明确类型，禁止统一填 0：

1. **seed 可靠性**：谱库/标准匹配等级、seed score、是否同一样本、可见次数；
2. **知识边**：Rhea/KEGG、reaction id、左右侧、方向 class、反应家族；
3. **hyperedge 完整度**：两侧非 currency 参与物、化学计量、缺失共底物/共产物签名；
4. **候选特异性**：同一 seed-reaction 事件支持多少竞争候选；
5. **谱学实验层**：query-seed 的 direct、neutral-loss、modified-cosine 真实边分数及匹配非邻居分数；
6. **样本实验层**：feature-feature 共丰度/峰相关真实边分数及匹配非邻居分数；
7. **机会/伤害变量**：candidate degree、seed degree、network membership、候选组大小、baseline margin；
8. **空模型映射**：matched non-neighbour id、seed-permutation id、degree/component/arity/direction-matched rewire id。

truth identity 只允许用于：

- 评价时确认唯一正确候选；
- 构造训练/评价隔离；
- 排除“truth 已经作为 seed 被显式给出”的平凡 query。

truth 不得决定部署 seed context，不得选择事件、阈值、缺失填充值或 action cell。

## 6. 预注册动作矩阵

不做所有二元因素的指数穷举。先冻结 12 个有明确科学含义的 cell：

### A. 核心可执行 cell

1. Rhea direction-supported + candidate-specific + spectral excess positive；
2. Rhea direction-supported + candidate-specific + coabundance excess positive；
3. Rhea direction-supported + candidate-specific + 两个实验层均正；
4. Rhea bidirectional + candidate-specific + spectral excess positive；
5. Rhea bidirectional + candidate-specific + coabundance excess positive；
6. Rhea bidirectional + candidate-specific + 两个实验层均正。

### B. 机制消融 cell

7. 去掉 direction 条件，保留 candidate specificity 与双实验层；
8. 去掉 candidate specificity，保留 direction 与双实验层；
9. 只保留知识边，不要求实验层支持；
10. 只保留实验层相似，不要求真实反应边。

### C. 负对照 cell

11. direction-conflicted / same-side identity-nontransformative 事件；
12. component/arity/direction/degree 匹配的 rewired event。

`direction_unknown` 与 KEGG unknown-direction 先单独报告 coverage，不进入主要 corrective cell。它们只有在核心 cell 已建立增量后才可作为 coverage 扩展，而不能用于稀释主检验。

## 7. 两阶段评价，禁止再次把 gate 与证据混在一起

### 7.1 阶段一：模型无关、固定动作

对每个 cell，预先冻结单调支持函数：

`support = robust_max_over_independent_dependency_groups(min(edge_specificity, data_layer_excess))`。

dependency group 以 `seed identity + reaction id + missing coproduct signature` 定义，避免一个物理事件因多个 Rhea record 被重复计数。

对每个 query：

- action score 在候选内必须变化；
- 最高 action score 必须唯一；
- 只在 DreaMS 的预注册低边界风险层评价；
- 所有 cell 使用完全相同的 query 风险层；
- 报告 candidate pairwise accuracy、truth margin、proposal precision、corrected、introduced 和 `corrected - 2*introduced`。

风险层只允许由 `DreaMS top1-top2 margin + candidate count + source` 在 training sources 上冻结，不能读取 reaction cell 的 outcome。

### 7.2 阶段二：容量匹配的 nested model

只有阶段一通过的 cell 才进入小模型。每个真实 cell 必须与对应 rewire、seed permutation 和 matched non-neighbour 使用：

- 相同特征维度；
- 相同模型容量；
- 相同 fold 与 random seed；
- 相同固定风险层；
- 额外报告各自 nested gate 的端到端结果，但它不是主要机制估计量。

## 8. Null 设计

### 8.1 Matched non-neighbour

在同 source/polarity/adduct/context 内匹配 seed：formula mass、元素向量、heavy atoms、peak count、network degree、spectra count 与 seed score。真实邻居全集均不得作为 control。

### 8.2 Seed permutation

只在同一个实际样本或同一个 rotation 内置换，保持 seed 数量、seed score 分布、degree 分布与可观测率。禁止跨样本置换。

### 8.3 Reaction rewiring

在同一 reaction component 和 direction class 内执行 hyperedge-preserving swap，匹配两侧 arity、非 currency 参与物数、candidate/seed degree、元素变化和来源覆盖。仅 degree-preserving 不足以作为主空模型。

### 8.4 Catalog opportunity

所有模型必须包含 B37 的 `network_member + capped_degree`；reaction effect 永远报告为其上的增量。

## 9. 准入门

一个 cell 进入 context representation 需要同时满足：

1. 相对 topology 的 Recall@1 增量至少 `+3 pp`，或在预注册困难子集至少 `+5 pp` 且总体不降；
2. `corrected > 2 * introduced`；
3. identity-cluster 与 formula-cluster CI 下界均大于 0；
4. 相对三类 null 的 multiplicity-corrected p 小于等于 0.05；
5. 在至少两个独立 sample-local 来源方向一致；其中 ST 类 LOSO 只能作为反事实恢复，至少一个来源必须是预先封存的 prospective/independent-target 队列后才能形成部署主张；
6. topology-correct query 的伤害率不超过 0.5%；
7. 未知方向、hub 和证据缺失分层均显式报告。

当前只有一个 sample-local LOSO 来源，而且其 query truth 全部来自原 seed 表。B39 可以完成动作发现和 leave-one-seed-out 机制验证，但在增加独立 target 队列前，不能完成“真实未知注释增益”的最终主张。

## 10. 表示学习的正确落点

### 10.1 首选：上下文候选表示

`z_c_ctx = normalize(z_c + alpha(q,c,O,G) * A(event_set(q,c,O,G)))`

- `z_c` 是通用 DreaMS 候选谱图 embedding；
- `O` 是当前样本可观测 seed/feature 集；
- `event_set` 是通过 B39 的原子事件集合；
- 无 admissible event 时 `alpha=0`，严格退化为通用 DreaMS；
- 不把反应邻居当同分子正例，不无条件拉近不同代谢物。

这是真正的 BioAware context embedding，但不是 candidate-independent clean-spectrum embedding。

### 10.2 Shared clean-spectrum encoder

若一个动作必须读取 sample seed、candidate membership 或 reaction graph，它在信息论上不能由单张原始谱图唯一恢复。只有 B39 进一步证明某个反应动作对应稳定的 query/candidate 谱学变换，并在 seed 集合和候选库变化时仍保持方向，才可把该“谱图可恢复分量”作为 auxiliary ranking loss 微调共享 encoder。

## 11. 最近执行顺序

1. B39-M0：重建原子事件 ledger；不读取 outcome，不拟合模型；
2. B39-M1：按 R/H/S 分层报告 coverage、方向、特异性、实验层与 null 可构造性；
3. B39-M2：只跑 12 个冻结 cell 的模型无关固定动作；
4. B39-M3：只对通过 cell 做容量匹配 nested model；
5. 增加至少一个 query truth 未参与 seed 构建的独立 target 队列，并一次性确认；
6. 之后才训练 context-conditioned candidate representation；
7. shared clean-spectrum embedding 作为谱图可恢复分量的独立后续课题。

## 12. 停止规则

- 若真实 cell 不优于 matched non-neighbour/rewire/seed permutation，停止反应特异主张；
- 若只有 catalog membership/degree 有效，保留 B37 为候选先验，不改名为 reaction propagation；
- 若只在 synthetic rotation 有效而 sample-local LOSO/independent-target 无效，不进入部署；
- 若 fixed-action 不过门，不允许用更复杂模型或更宽 gate 掩盖；
- B39 任何结果都不得被描述为独立盲测或 SOTA，直到新增 real-sample 队列封存并验证。

## 13. 与代表方法的差异

KGMN 联合知识反应网络、MS2 相似网络和全局峰相关网络；MetDNA3 在知识层与数据层之间先做 MS1/RT/MS2 约束映射，再保持两层拓扑一致；NetID 通过全局网络优化与 target-decoy/FDR 控制冲突。B39 的目标不是复刻其覆盖扩张，而是补足一个更窄、可证伪的问题：

> 在强 DreaMS 谱学基线和 catalog prior 之后，哪类原子反应事件仍能提供跨来源、超过匹配空模型的候选特异增量，并以 abstention-safe 的方式转成上下文候选表示。

这只有在上述准入门实际通过后才构成方法学创新。
