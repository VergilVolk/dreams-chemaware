# ChemAware 系统性批判、文献反证与研究范式重置

日期：2026-09-12

## 0. 结论先行

ChemAware 不是“没有成果”，但此前确实没有把成果组织成一个可证伪、可累积的科学问题。最严重的错误不是某个学习率、缓存、loss 或动作脚本写错，而是长期把下面四个不同问题混在了一起：

1. **化学信息是否有候选判别力**：知道真实结构和候选结构后，化学教师能不能把真候选与同分子式异构体分开；
2. **化学信息能否形成有用的候选分数修正**：冻结模型上是否真实纠错，并严格胜过匹配伪动作；
3. **这种候选修正能否由同一个共享谱图 embedding 表示**：目标分数变化是否属于共享点积几何可实现的集合；
4. **这种可表示修正能否仅由干净谱图预测并泛化**：训练时使用的结构特权信息，部署时能否由 spectrum-only encoder 恢复。

现有强证据主要成立在第 1、2 层；`rule_mass` 是少数直接落在第 3 层的开发结果；真正的 shared-embedding 化学因果增量仍未在严格对照下成立。我们过去反复从第 2 层直接跳到第 4 层，失败后又回去修改动作细节。这就是“钻牛角尖”的核心。

新的科学对象不再是“找到一个更强的峰编辑动作”，而是：

> 从候选条件化化学教师给出的纠错场中，分离出**对当前检索决策有用、可由共享点积几何表示、并且能够由干净谱图观测和泛化**的那一部分；无法满足三项条件的剩余信号，必须诚实地留在候选条件模型，而不能强塞进共享 embedding。

本文先冻结既有证据，再系统批判旧路线；随后提出一版看似合理的“机制动作升级”路线，用一手文献和数学约束反驳它；最后给出新的统一框架和最低成本验证顺序。

## 1. 冻结证据账本：到底已经证明了什么

### 1.1 已成立的正证据

| 证据 | 已观测结果 | 它证明什么 | 它没有证明什么 |
|---|---:|---|---|
| ICEBERG 结构条件教师 | 700 个错误富集 query 上，correct R@1 `0.5886`，candidate-swapped `0.1329`，peak-permuted `0.2986` | 候选结构条件碎裂预测包含强判别信息 | 不是 spectrum-only embedding，也不是泛化性能 |
| 旧 E1 冻结峰动作 | formula-disjoint confirmation：3 corrected / 2 introduced，R@1 `+0.7519 pp`，配对 margin 相对控制为正 | 已观察峰的特定方向动作有冻结纠错能力 | 风险效用 `3-2×2=-1`；不能直接迁移为 embedding 结论 |
| 分层经验 meta-action | `fr_NH0 -> C2H6N, m/z 44.0495`；44 个确认公式、67 个正 identity；effect `0.11157`，95% CI `[0.04512, 0.18408]`，BH q `0.00980` | 至少一条 parent-predicate/fragment 关联跨多个采集域复现 | effect 不是 Recall pp；关联不是完整碎裂机制，也不是可部署模型 |
| 正确规则矩阵教师 | fold-2 confirmation 中 correct R@1 `0.12057`，structure-swapped `0.09220`，peak-permuted `0.09701`；两项公式聚类 CI 均为正 | OOF 聚合规则矩阵确有结构特异的候选判别信息 | 低绝对 R@1；非零系数不能解释为逐条机制规则；未改变 DreaMS |
| `rule_mass` 显式共享核 | 1,929-query held inner：39 corrected / 15 introduced，R@1 `+1.2442 pp`；绝对公式 CI `[0.6090, 1.9507] pp`；胜过 mass 与 shifted-rule | 干净谱图、同一特征映射、候选无关的显式核可以产生真实 shared-embedding 开发增量 | 仍是单次 inner development；没有 outer/external 复现；rule-mass 未显著胜过 rule-response |
| 非化学训练骨架 | error-conditioned full-candidate adapter `+0.3629 pp`（9/2）；另一 matched continuation `+0.4666 pp`（12/3） | 完整候选、错误定向训练本身能够改善 shared embedding | 这些增量不能归因于化学 |
| 旧 direct shared proof | 180-query inner-all `+0.5556 pp`（1/0），49-query selected `+2.0408 pp` | 动作训练能够改变干净 embedding 并翻转局部排序 | 无因果控制、CI 下界为 0，preservation/clip 失败，不是严格 ChemAware 增量 |
| 严格 ICEBERG residual 冻结 headroom | 2,048-query dev，272 个严格动作；dose 0.25 为 56/0、`+2.7344 pp`；dose 0.50 为 102/0、`+4.9805 pp`；dose 1.0 为 156/0、`+7.6172 pp` | 候选分数空间中存在足够大的结构条件纠错上限 | 这是 oracle-routed frozen score residual；不是共享 embedding，不代表可学习性 |
| 新增 R0 shared-Gram 切空间审计 | dose 0.50、ridge 0.25 的自由逐谱投影：full 2,048 graph 为 85 corrected / 3 introduced，净 `+4.0039 pp`；correct residual 能量解释 `95.30%`；structure control `-0.3906 pp`，peak control `+0.0488 pp` | 旧 correct residual 在这张稀疏开发图上存在共享单位点积几何解，且其几何效用保留化学特异性 | 每张谱拥有独立 oracle 位移，ridge 为事后诊断网格；没有证明 `u=h(x)`、没有更新 encoder，也不是正式性能 |

### 1.2 已成立的负证据

| 路线 | 结果 | 严格结论 |
|---|---:|---|
| rule-selected clean-boundary adapter | 训练 pair loss 下降约 63%，fold-3 R@1 变化 `0`；correct-control margin `-4.85e-5`，CI 跨 0 | loss 可拟合不等于决策纠错；样本大多远离当前边界 |
| 旧 ICEBERG candidate-residual 训练 | correct 10/272，clean 11/272；四臂 median transfer ratio 均约 `0.36–0.37`；所有 step clipping | 正确 residual 的连续拟合略好，但没有超过普通 continuation 的决策收益；该路线关闭 |
| A2 action-gradient transfer | 三条路由均未胜过 candidate-swapped/peak-permuted；correct-control CI 跨 0 | 梯度可达不等于化学特异；不能再靠 LR、epoch、PCGrad 或 cap 调出来 |
| A2 pair 路由 | correct 与 candidate-swapped 同 pair 16/24；与 peak-permuted 同 pair 19/24 | 动作在 reference-pair 选择处发生结构性信息坍缩 |
| ChemBERTa cross-modal shared feature | selection 有 `+0.1004 pp` 信号；576-query confirmation 1 corrected / 1 introduced，净 `0` | 分子 embedding 教师没有形成公式隔离的 spectrum-only shared 增量 |
| learned PSD observation feature | confirmation R@1 `-0.5208 pp`；uniform 同样，mass-only 反而 `+0.3472 pp` | “化学加权观测通道”未胜过简单质量特征；PSD 形式正确不等于权重正确 |
| direct prior 全图开发 | 50,397 queries 中仅 258 active，最多 2/0，约 `+0.0040 pp`，CI 下界为 0 | 直接先验覆盖和效应远不足以支持微调 |
| Morgan/structure probe | 正确标签相对置乱平均高 `6.69 pp`，但 `corrected-2×introduced` 为负 | 结构信息可解码，但直接按结构相似排序会破坏正式检索 |
| MolFormer teacher | correct 与 same-formula mismatch 的 adapted embedding cosine `0.999981` | 当前教师/权重机制没有提供身份特异的可迁移方向 |
| multiview probe | 化学增量梯度只有 query-only 的 54.9%，方向 cosine `0.9860` | 简单增加 reference view 主要是重复和稀释，不是新信息 |

### 1.3 尚未产生结果的对象

- `qualified-action delta transfer` 目前只是新的实现和合同，不能继承 E1 的 `+0.7519 pp`，也不能继承 residual oracle 的 `+4.9805 pp`；
- B1、B1J 的 preflight 只证明动作可构造；其 discovery/confirmation 失败不能被描述为性能收益；
- 一条经验规则和 117 个 bank-selected 动作都属于**训练信号候选**，不是 shared embedding 成果；
- 当前没有严格证据支持“ChemAware 已经让官方 shared embedding 提升 3–5 pp”。

R0 已经完成，但它是**几何上限审计**而非训练结果。其输入直接读取已被关闭的 ICEBERG residual ledger，用来定位旧路线为何失败；它不重新开放旧 Huber 训练，也不替代 qualified-action 的独立验证。

### 1.4 3–5 pp 目标的算术约束

在 2,048-query dev 上，`+4 pp` 需要至少 82 个净纠正。dose 0.50 的 frozen oracle 只有 102 个纠正，因此若不扩展覆盖，训练必须保留至少 `82/102 = 80.4%` 的 oracle 纠错且几乎不引错。旧 residual 训练只有 10 个 active correction，甚至低于 clean continuation 的 11 个。

这说明当前瓶颈不是“再把 transfer ratio 从 0.36 调到 0.40”就够了。要达到 4 pp，至少必须发生一件根本变化：

1. 化学动作覆盖显著扩展；
2. 可迁移比例出现数量级提高；
3. 使用更适合候选条件信号的部署模型；
4. 或三者共同发生。

`rule_mass` 的 39/15 等于 24 个净纠正；同一 1,929-query 分母上 4 pp 约需 77 个净纠正，也就是现有净效用的约 3.2 倍。这个差距不能用不同 cohort 的 pp 相加掩盖。

## 2. 之前真正取得的“极大成果”是什么

成果不只是某个 pp，而是已经排除了大量伪路径，并建立了后续研究所需的基础设施：

1. **严格候选图和冻结基线**：query、候选分子、多参考谱 max、10 ppm、公式折和官方 embedding 已经固定；
2. **可复算的结构条件教师**：ICEBERG correct、candidate-swapped、peak-permuted 三臂已形成同一候选图上的 score ledger；
3. **纠错动作与安全动作分离**：非纠错样本化学权重严格为 0，伪动作不能负权进入优化器；
4. **从“规则数量”转向效应估计**：formula-disjoint、domain-stratified、sign-flip、bootstrap、BH 校正已经进入规则筛选；
5. **从 loss 下降转向决策审计**：corrected、introduced、margin、MRR、R@K、macro/micro AUC 和 preservation 已成为必报项；
6. **动作源、注入、部署边界开始分离**：teacher、frozen action、candidate score、shared embedding 不再允许互相冒充；
7. **已有一个真正部署兼容的正例**：`rule_mass` 说明 spectrum-only、同函数、显式特征核这条几何类型不是空想。

因此不能从头再来。新路线必须复用这些账本、规则效应、冻结教师和评估合同，但要停止复用旧的概念错误。

## 3. 对旧研究范式的系统性批判

### 3.1 把四种不同对象都叫“化学规则”

此前规则库混合了：

- 化学机制：某类结构可能经过某类断键或重排；
- 观测关联：某 parent predicate 下某个 m/z/neutral loss 更常出现；
- 模型干预：把 top-k 峰强度乘以 0.75；
- 决策动作：提高真候选相对当前 hard negative 的 margin。

这四者没有一一对应关系。一个统计关联不自动给出峰编辑方向；一个峰编辑有效也不意味着它复现真实碎裂机制；一个候选 margin 变好更不意味着共享 embedding 能表示它。

### 3.2 动作空间过窄，但“动作更复杂”不是充分修复

旧动作主要是已有峰上的 top-k 强度 boost/attenuate，忽略了：

- 峰的出现/消失与检出限；
- 峰—峰 neutral-loss、互补离子和碎裂级联；
- 同位素、加合物、氢迁移和电荷保留；
- collision energy、fragmentation method、instrument 和 polarity 的条件分布；
- 一个峰对应多个候选碎片的解释不确定性；
- 多个独立碎裂路径汇聚到同一 m/z 的多重解释。

但是，若继续沿用原注入目标，复杂动作只会制造更复杂的候选特权向量，仍然不能解决可表示性和可观测性。

### 3.3 使用真实结构选择动作，随后要求无结构模型复现

ICEBERG residual、parent predicate 和 candidate-swapped 控制都使用了训练期候选结构。部署时 shared encoder 只看到单张干净谱图。只有化学教师输出中能够由干净谱图条件期望预测的部分才可迁移：

\[
d_{\mathrm{obs}}(x)=\mathbb E[d^*(x,c,\text{structure})\mid x].
\]

剩余部分是不可观测的 privileged residual。过去主要检查梯度是否到达参数，却没有先估计 `Var(E[d*|x]) / Var(d*)`，因而把“教师知道”误当成“学生能知道”。

### 3.4 忽略共享点积 embedding 的几何约束

设所有谱图的单位 embedding 构成矩阵 `Z`，共享相似度矩阵为：

\[
K=ZZ^\top.
\]

因此任何完整 shared-embedding 相似度都必须是：

- 对称；
- 半正定（PSD）；
- 对角为 1（归一化时）；
- 秩不超过 embedding 维数。

在基线 `Z0` 附近，令每张谱的微小位移为 `U`，一阶可实现变化只有：

\[
\Delta K = Z_0U^\top + UZ_0^\top,
\qquad z_i^\top u_i=0.
\]

而过去的 teacher target 是逐 query、逐 candidate、candidate-centered、经 molecule-wise max 后得到的方向性 score residual。它不天然是对称 PSD Gram 变化，也不保证落在上式的切空间中。当前 action-delta loss 只让 live query 对 frozen references 拟合局部 score displacement；这可以证明一个 query 行上的局部可达性，却没有证明当同一 encoder 同时作用于所有 query/reference 后存在全局一致解。

必须强调：这不是说所有稀疏边目标都必然不可实现。候选较少时，单独移动一个 query 往往可以拟合多项局部分数。新增 R0 已经对旧 residual 完成首次测量：1,314 条候选边、1,430 个谱图节点中只有 22 条 reciprocal/duplicate unordered edges；系统高度欠约束，所以自由逐谱切空间确实能承载大部分 target。问题因此从“是否存在几何解”进一步收缩为：

1. 共享函数 `u_i=h(x_i)` 能保留自由解的多少；
2. 对未参与 target 的安全 query，reference 复用会引入多少副作用；
3. correct 相对伪动作的优势能否跨公式泛化，而不是记住 272 个 oracle identity；
4. PEFT 的函数类和优化器能实现该可观测解的多少。

R0 在事后 ridge 0.25 下已把归一化、reference 复用和 full 2,048 graph side effect 纳入有限步评价：correct 为 85/3、净 `+4.0039 pp`，mean/min preservation 为 `0.99809/0.95847`；两个控制均远弱。这否定了“旧 target 主要因为 shared 几何装不下而失败”的初始猜想。当前第一嫌疑已经变为**干净谱图可观测性和共享函数学习**，其次才是优化实现。

### 3.5 把连续 target fit 当作决策收益

旧 residual 中 correct arm 的 Huber 和 target cosine 略优，但 corrected 反而少于 clean。rule-boundary 实验 loss 降 63%，R@1 不动。根因是 surrogate 与最终决策不一致：

- R@1 只由当前真候选与最强负候选的符号决定；
- 全候选 score 向量的平均误差可被大量不决定 Top-1 的维度主导；
- reference-wise max 会导致活跃参考谱发生离散切换；
- 远离边界的容易样本可以贡献很大、很稳定、但无效的梯度。

### 3.6 把“正确教师胜过置乱”误当作“正确教师胜过 clean continuation”

化学特异性需要两个独立问题：

1. correct 是否胜过 matched pseudo-chemistry；
2. correct 是否胜过完全不看化学、但具有相同样本、优化预算和错误聚焦的 clean continuation。

旧 residual 在连续拟合上胜过部分伪教师，但没有胜过 clean；因此不能归因于化学。两个门缺一不可。

### 3.7 选择偏差与 oracle enrichment 被低估

272 个动作是从 619 个官方错误、348 个 correct-teacher rescue 中再筛选出的 strict-positive 子集。它非常适合证明 oracle headroom，却天然高估一般样本上的可学习效应。正确的层级应是：

- source discovery；
- frozen action confirmation；
- representability confirmation；
- clean observability confirmation；
- shared model confirmation；
- outer/external generalization。

同一层反复选择和评估会把幸运波动包装成动作能力。

### 3.8 有效样本量远小于 query 数

83,619 个 query 不等于 83,619 个独立化学证据。公式、identity、重复谱、采集域和候选参考高度聚类。对某条规则，真正的有效样本可能只有几十个公式；对 13.6M 可训练参数的 adapter/PEFT，这远远不足。必须以 formula/identity/domain cluster 为统计单位，而不是用谱图数制造信心。

### 3.9 对照设计曾经改变了研究对象

早期 pseudo-action 可能重新选择 hard negative 或 reference pair，导致 correct/control 不再比较同一边界。后续修复了容量匹配，但 A2 又证明多数动作在 pair selection 后变成同一 pair。教训是：对照必须同时固定：

- query；
- truth 与同一 hard negative；
- reference 谱；
- 修改峰位置、数量和绝对 dose；
- optimizer schedule；
- selection ledger。

如果某个表示步骤使 correct/control 坍缩为同一对象，该表示本身就失败，不应再训练。

### 3.10 局部 Jacobian 被赋予了过多因果含义

输入 Jacobian 只描述当前模型在一个连续邻域内的敏感方向。Transformer、归一化和 reference max 都会产生非线性或分支切换；一个一阶同向峰不等于有限化学干预后仍同向，更不等于跨样本可泛化。Jacobian 最多用于动作排序或切空间审计，不能为化学机制背书。

### 3.11 clipping 与 AdamW 掩盖了请求剂量

旧 residual 四臂所有 step 都被裁剪。global gradient ratio、clipping 和 AdamW 的坐标自适应共同改变了化学 dose；不同 target 的数值强度不再对应不同参数步长。即使取消 clipping，也只修复优化器可解释性，不能修复目标错位。

### 3.12 规则库缺少明确的证据层和不确定性

规则记录需要至少拆成：

- `mechanistic_hypothesis`：可能的断键、重排、fragment/neutral loss；
- `observation_effect`：域内效应、CI、FDR、覆盖；
- `applicability_domain`：adduct、polarity、fragmentation method、instrument、CE；
- `ambiguity`：同质量候选碎片数、解释熵；
- `deployment_visibility`：推理时是否只凭谱图可计算；
- `action_status`：diagnostic、teacher-only、shared-eligible、candidate-only。

没有这些字段，“规则库更大”只会提高误触发和多重检验负担。

### 3.13 单峰关联没有表达组合证据

异构体判别通常依赖“某峰出现且某 neutral loss 缺失”“两个峰的相对强度关系”“一组互补碎片共同支持某局部结构”，而不是一个 m/z 的独立主效应。当前 parent-predicate × channel 的单变量矩阵无法表达：

- AND/OR/NOT 组合；
- 竞争解释；
- 峰组共现；
- 路径一致性；
- 证据之间的冗余与协同。

### 3.14 评估协议虽已加强，但仍有三类缺口

1. 不同 cohort 的 pp 仍容易被口头相加；
2. inner fold 被多次用于方向选择，不能继续承担正式确认；
3. 需要 MassSpecGym v1.5 风格的候选 canonicalization、预训练污染、shortcut 和 metric implementation 审计。

### 3.15 工程失败不是科学失败，但曾消耗了科学判断力

缺模块、缓存漏行、`None` 指标、失配 CLI、role matrix 和路径错误已经被合同测试覆盖。真正的教训是：工程 preflight 应尽早、便宜、fail-closed；但不能因为工程终于跑通，就默认科学目标正确。`PASS: contracts` 只证明可运行，不能证明有化学增益。

## 4. 全部教训压缩成十二条不可再违反的规则

1. teacher headroom、frozen action、reranker、kernel 和 learned shared embedding 必须分栏报告。
2. 化学 source specificity、shared-geometry representability、clean-spectrum observability、optimization realization 必须逐级过门。
3. 训练前先问目标是否能由部署模型表示，而不是先问梯度能否到达。
4. 训练时可使用候选结构；部署只看谱图时，必须测量特权信号的 spectrum-only 可预测部分。
5. shared embedding 的目标必须尊重对称、PSD、单位范数和低秩约束。
6. 任何 chemical arm 都必须同时胜过 matched pseudo-chemistry 和 equal-budget clean continuation。
7. 规则是带适用域和不确定性的证据，不是结构真理，也不是天然动作。
8. 不把远离当前错误边界的 loss 下降解释为检索能力。
9. 不把 formula/fingerprint/fragment 预测更准解释为 R@1 必然更高。
10. 不用增加 epoch、LR、top-k 或 GPU 数量修复结构性信息丢失。
11. 任何 3–5 pp 目标都先做覆盖 × 精度 × 可迁移率的算术可行性审计。
12. 只在前一层有正证据时进入下一层；失败路线封存，不换名字重跑。

## 5. 第一版方向：机制化动作升级（随后将被反驳）

在阅读文献之前，最自然的改进是把 top-3 峰强度动作升级成“碎裂事件动作”：

1. 用 parent predicate、候选结构和 ICEBERG/FIORA 生成断键与 fragment DAG；
2. 将单峰编辑扩展为 fragment 出现/消失、neutral loss、峰组共现、氢迁移、相对强度和 CE 轨迹；
3. 用 FLARE 式 peak-to-atom 对齐找到可解释局部证据；
4. 在完整候选集上计算 correct 与 matched pseudo 的 score displacement；
5. 把这一更精确的 action delta 蒸馏进 DreaMS PEFT。

这条路线比当前 top-k 动作更有化学细节，也能开发动作库潜能。若只看“动作不够具体”，它几乎是标准答案。

## 6. 一手文献调研：哪些经验可以迁移，哪些不能

### 6.1 DreaMS：共享谱图 embedding 的原始成功来自部署一致的谱图对监督

[DreaMS](https://pmc.ncbi.nlm.nih.gov/articles/PMC13090125/) 使用同分子不同谱作为正例、近质量异分子谱作为负例，并以同一个谱图 encoder 和 cosine 进行对比微调。它的训练对象和部署对象一致。这解释了为什么 DreaMS 原始 triplet 虽简单，却比我们的候选特权 residual 更容易迁移。

可迁移经验：正负边界应由完整检索任务产生；query/reference 必须共享编码路径。

不能偷换的结论：DreaMS 的成功不表示任意候选分数教师都能压进同一个 1,024 维谱图点积空间。

### 6.2 MIST 与 CSI:FingerID：化学知识应进入表示，但目标仍可能错位

[MIST](https://www.nature.com/articles/s42256-023-00708-3) 把峰表示成化学式，隐式编码 neutral loss，并增加子结构预测；[CSI:FingerID](https://www.pnas.org/doi/10.1073/pnas.1509788112) 使用 fragmentation tree、多种谱图/树核和分子指纹预测。它们共同说明，比单峰强度更结构化的干净谱图特征确有价值。

但 2026 年的理论工作 [Small molecule retrieval from tandem mass spectrometry: what are we optimizing for?](https://arxiv.org/abs/2602.16507) 证明并实证：fingerprint/vector similarity 目标与 HR@1 存在内在 regret，尤其在近重复候选和宽 similarity band 下。因而“化学属性预测更准”不是我们的终点，最终仍须直接评价固定候选集排序。

### 6.3 ICEBERG 与 FIORA：机制化碎裂教师很强，但属于 structure-to-spectrum

[ICEBERG](https://pmc.ncbi.nlm.nih.gov/articles/PMC12154671/) 生成多步 fragmentation DAG，再预测 fragment 强度和氢迁移；[FIORA](https://www.nature.com/articles/s41467-025-57422-4) 以局部断键邻域预测单步 fragment，并显式使用 CE、instrument 等协变量。它们支持：

- 动作应以碎裂事件而非任意峰缩放为单位；
- CE、adduct、instrument/polarity 不能混池；
- fragment 解释存在多路径和氢迁移不确定性。

但它们输入候选分子结构。它们可作为候选条件教师或 reranker，却不自动提供 clean-spectrum shared embedding 的可观测表示。FIORA 的单步限制、多解释冲突和域依赖也禁止把预测 fragment 当作无误标签。

### 6.4 FLARE：局部对齐的成功恰好反驳“全部池化进一个向量”

[FLARE](https://pmc.ncbi.nlm.nih.gov/articles/PMC12873900/) 的核心是双向 peak-to-atom / atom-to-peak MaxSim，而不是先把两侧全部池化成一个全局向量再点积。它证明局部、互相一致的证据可以显著改善 spectra-to-molecule retrieval。

但 MaxSim 是候选结构条件、集合到集合、非简单 pooled dot product 的打分器。把 FLARE 当成“给 shared spectrum embedding 加一个局部 auxiliary”会丢掉其真正的 scoring mechanism。它更支持 hybrid/local scorer，而不是证明旧蒸馏路线正确。

### 6.5 SpecBridge 与 MSAlign：高性能来自跨模态候选几何，而非 spectrum-to-spectrum 约束

[SpecBridge](https://arxiv.org/abs/2601.17204) 把谱图投影到冻结 ChemBERTa 分子空间，推理时与候选分子 embedding 比较；[MSAlign](https://arxiv.org/abs/2605.19752) 对冻结 DreaMS/ChemBERTa 使用 candidate-based contrastive objective，并强调 split 在 leakage 与 domain shift 间的权衡。

它们支持轻量投影、候选对比和结构教师；但正式 scorer 都读取候选分子表示。若我们部署时禁止候选结构输入，它们的性能不能作为 shared spectrum embedding 的先验承诺。

### 6.6 谱图相似性文献：显式特征核是被低估的主线

[Spec2Vec](https://pmc.ncbi.nlm.nih.gov/articles/PMC7909622/) 将 fragment 和 neutral loss 当作词并汇聚为谱图向量；[MS2DeepScore](https://pmc.ncbi.nlm.nih.gov/articles/PMC8556919/) 用 Siamese encoder 让谱图 embedding 的 cosine 拟合结构相似；CSI:FingerID 更早使用多个 fragmentation-tree kernel。这些方法说明“可计算的 spectrum-only witness feature -> shared kernel”是成熟而高效的几何形式。

这与本项目 `rule_mass +1.2442 pp` 的开发结果一致：最可靠的 ChemAware shared 信号目前不是复杂蒸馏，而是部署时可见、同函数计算的显式谱学关系。

### 6.7 Kernel 几何：shared dot product 不是任意相似度容器

[Learning the Kernel Matrix with Semidefinite Programming](https://www.jmlr.org/papers/v5/lanckriet04a.html) 明确指出，欧氏 embedding 的内积信息构成对称 PSD kernel matrix。[Learning Kernels from Indefinite Similarities](https://pages.stat.wisc.edu/~wahba/stat860public/pdf1/chen.gupta.recht.09.pdf) 则专门处理实际相似度可能不定、需要学习 PSD surrogate 的问题。

因此我们不能只验证 teacher residual 有效；还必须验证它能被投影为低失真的 shared PSD correction。过去没有做这一层审计。

### 6.8 Privileged information：蒸馏是方法，不是可迁移保证

[Unifying distillation and privileged information](https://arxiv.org/abs/1511.03643) 给出了训练时额外信息通过 teacher/student 迁移的统一框架。但它没有保证任意结构特权输出都能由缺少结构输入的学生恢复。本项目必须实证估计可观测条件期望，而不能用“这是知识蒸馏”替代可识别性证明。

### 6.9 MassSpecGym v1.5：强结果也可能来自协议错误

[MassSpecGym in the Wild](https://arxiv.org/abs/2606.19624) 在使用 MassSpecGym 报告结果的论文中发现广泛的数据泄漏、shortcut、实现 bug 和指标漂移，并发布 v1.5 修正。ChemAware 的最终验证必须锁定 candidate canonicalization、预训练污染、split、候选集和指标实现；内部 inner 结果不能升级为 benchmark 主张。

## 7. 对第一版“机制动作升级”路线的反驳

文献没有支持“动作更细 -> 用原 loss 蒸馏 -> shared embedding 必然更强”。恰恰相反，它暴露了六个问题：

1. **任务类型偷换**：FLARE、SpecBridge、MSAlign、ICEBERG 和 FIORA 的主要优势都依赖候选结构或局部 cross-modal scorer；我们的部署目标是 spectrum-to-spectrum shared embedding。
2. **表示形式偷换**：FLARE MaxSim 和 fragmentation graph score 不一定能写成一个 pooled spectrum vector 的点积。
3. **目标函数偷换**：fragment/fingerprint 预测更准可能与 HR@1 冲突；更逼真的谱并不保证当前候选边界正确。
4. **可观测性未解决**：更机制化的 teacher 可能包含更多学生在单张谱中无法恢复的结构信息，反而扩大 privileged gap。
5. **域错配更严重**：CE、fragmentation method、adduct 和 instrument 元数据不完整时，精细模型会更自信地利用错误条件。
6. **计算成本先于识别问题**：若目标大部分不在 shared geometry 或 clean-observable 子空间，增加 fragmenter、动作数和 GPU 只会更昂贵地拟合不可部署分量。

所以第一版路线被否决。机制动作库仍应发展，但其角色应是**产生带不确定性的候选纠错证据**，而不是直接成为微调 target。

## 8. 新的一般框架：可表示性优先的化学见证迁移

暂称 **Representability-First Chemical Witness Transfer（R-CWT）**。它不是又一个 peak action，而是把 ChemAware 从“编辑谱图”改造成“分解可部署化学纠错量”的研究框架。

### 8.1 四个连续上限

对 query `q` 的候选集 `Cq`，官方分数为：

\[
s_0(q,c)=\max_{r\in R_c}\langle z_0(x_q),z_0(x_r)\rangle.
\]

化学教师产生候选纠错场 `d*(q,c)`。随后依次定义：

1. **效用上限 `H_chem`**：直接在候选分数上加入 `d*` 能纠正多少；
2. **几何上限 `H_geom`**：`d*` 投影到共享单位点积 embedding 的切空间后还能保留多少；
3. **可观测上限 `H_obs`**：把自由逐谱位移限制为由干净谱图函数 `h(x)` 产生后还能保留多少；
4. **优化实现 `H_opt`**：实际 PEFT/adapter 能实现 `H_obs` 的多少。

过去的 `+4.9805 pp` 属于 `H_chem`；旧 residual 训练属于 `H_opt`。中间两层没有被独立量化，因此无法知道损失发生在哪里。

### 8.2 第一阶段：共享 Gram 切空间投影

对当前参与检索的谱图，求自由位移 `U`：

\[
U^*=\arg\min_U
\sum_{(i,j)\in E}w_{ij}
\left[z_i^\top u_j+u_i^\top z_j-d^*_{ij}\right]^2
+\lambda\sum_i\|u_i\|^2,
\quad z_i^\top u_i=0.
\]

该问题只需冻结 embedding 和稀疏边，可先用线性代数求解，不更新 DreaMS。必须同时报告：

- target energy 中可解释比例；
- 对当前 Top-1 边界的保留比例；
- reciprocal edge 冲突与 reference-max switch；
- correct 相对 candidate-swapped、peak-permuted、mass-shifted 的增量；
- 位移范数和最坏 preservation 上限。

若 `H_geom` 很低，结论不是“化学动作无效”，而是该动作应留在 candidate-conditioned scorer；禁止进入 shared-embedding 训练。

### 8.3 第二阶段：干净谱图可观测投影

自由的 `u_i` 仍会记住 query identity。下一步只允许由部署时可见的谱学见证函数产生：

\[
u_i=h_\phi(\psi(x_i,d_i)),
\]

其中 `ψ` 不读取候选结构，包括：

- peak m/z 与强度的多分辨率基；
- precursor-relative neutral loss；
- peak-pair mass difference 与互补关系；
- isotope/adduct-compatible features；
- instrument/CE/polarity 的可用元数据及 missingness；
- 由 OOF 规则得到的可靠性、适用域和不确定性，而非真实 parent predicate。

训练与评估按 formula-held cross-fit；correct 与每个伪教师共享完全相同的输入、模型容量和 selection。`H_obs/H_geom` 才是真正的 privileged-to-clean transfer ratio。

### 8.4 第三阶段：把可观测纠错编译成 PSD 化学见证核

优先学习一个部署一致的显式特征映射 `phi_chem(x)`，构造：

\[
z_{new}(x)=\operatorname{normalize}
\left[
z_0(x),\sqrt{\beta}\,\phi_{chem}(x)
\right].
\]

其点积天然得到 PSD shared kernel。`phi_chem` 不再是 120 个独立峰通道，而是低秩的**化学见证**：它表示可由谱图观测的 fragment/neutral-loss/峰组关系，并通过 OOF chemical teacher 只确定哪些关系在当前异构体边界上有用。

与旧 `rule_mass` 的区别是：

- 旧核使用固定 mass/rule-response 通道；
- 新核先从 candidate-conditional teacher 中提取决策 residual，再做 shared-geometry 和 clean-observability 投影；
- 支持组合见证、适用域、不确定性和低秩交互；
- 权重直接由完整候选排序效用选择，而不是由规则主效应或结构相似性选择。

### 8.5 第四阶段：只有显式核通过后才蒸馏进 1,024 维 DreaMS

如果扩展 embedding 的冻结核在未触碰确认折的设置下成立，再把其 pairwise score correction 蒸馏到 DreaMS PEFT：

\[
L=L_{clean\_listwise}
+\lambda\,L_{pairwise\_kernel}
+\gamma\,L_{preserve}.
\]

`L_pairwise_kernel` 只作用于当前边界和安全回放，并直接拟合经过两次投影后的 deployable score correction；不再拟合原始 ICEBERG score、不再编码 action view、不再把规则命中当身份标签。

若扩展核有效而 1,024 维蒸馏失败，仍保留扩展 shared embedding；不能把压缩失败解释为化学方法失败。

### 8.6 动作库在新框架中的正确位置

动作库仍然重要，但每个动作必须输出一个带证据的对象：

```text
source mechanism hypothesis
  -> candidate-relative score effect and uncertainty
  -> matched pseudo-action specificity
  -> shared-geometry representability
  -> clean-spectrum observability
  -> deployment class: shared / hybrid / candidate-only / rejected
```

动作类型可以扩展为：

- fragment presence/absence；
- neutral-loss path；
- complementary fragment pair；
- relative-intensity ordering；
- multi-peak conjunction；
- CE response shape；
- adduct/polarity consistency；
- fragmentation-graph path；
- teacher ensemble disagreement/uncertainty。

但它们不再被手工转成“把第 k 个峰乘 0.75”。动作是候选纠错证据的来源，最终能否进入 shared embedding 由两次投影决定。

## 9. 下一步实验：最低成本、最高信息增益顺序

### R0：不训练的 shared-geometry 可表示性审计（已完成）

输入直接复用：

- official embedding cache；
- 2,048-query corrective residual ledger；
- 272 strict actions；
- correct / structure-swapped / peak-permuted；
- 完整 candidate/reference graph。

实现位于 `tasks/audit_chemaware_shared_gram_representability.py` 和 `tasks/chemaware_shared_gram_representability_core.py`，结果位于 `data/validation/chemaware_shared_gram_representability_v1/report.json`。它没有运行 116M backbone，也没有更新权重。

主要结果：

- 272 active queries、244 formula clusters、1,314 candidate edges、1,430 unique spectrum nodes；
- dose 0.50 的 correct oracle 为 102 corrections；
- ridge 0.25 的线性 shared-tangent 投影保留 92 个 active corrections；有限归一化后为 84 个；
- 扩展到完整 2,048-query ICEBERG graph 后为 85 corrected / 3 introduced，净 `+4.0039 pp`；
- correct target energy explained `95.30%`，target/projected cosine `0.99037`；
- mean/min preservation `0.99809/0.95847`；
- structure-swapped 在同 ridge 下为 1/9、`-0.3906 pp`；peak-permuted 为 3/2、`+0.0488 pp`。

这是一项强定位结果：**化学 target 的几何上限足够，旧训练失败不能再主要归咎于共享点积表达能力。**但自由更新直接按 oracle target 为每张谱分配不同位移，仍然可能记住 identity；ridge 0.25 也是看完开发网格后的诊断点。下一步不能把这个 4.0039 pp 报成模型性能，而要测 `H_geom -> H_obs`。

R0 的长期晋级条件仍不是一个拍脑袋绝对阈值，而是同时满足：

1. correct 的 projected frozen risk utility 为正；
2. correct 显著胜过全部伪动作；
3. projected residual 仍保留足以覆盖目标 pp 的净纠错数；
4. 位移范数/preservation 不要求异常大的几何扭曲。

### R1：公式隔离的 clean observability 审计（当前第一优先级）

只拟合 R0 的可表示分量；比较四组输入：

1. mass-only；
2. rule-response；
3. fragment + neutral-loss + peak-pair witness；
4. 上述特征的 domain-conditioned 低秩组合。

所有组使用相同 formula folds、ridge/low-rank 预算和候选边界。必须胜过 mass-only、uniform、shifted-mass、predicate/teacher permutation。

### R2：冻结扩展 shared kernel

在尚未用于选择的 cohort 上扫描极小的 `beta` 网格；报告 R@1/5/10/20/50、MRR、macro/micro AUC、corrected/introduced、near、公式聚类 CI。先证明扩展 kernel，再讨论神经网络。

### R3：单 GPU、最小 PEFT 压缩

只有 R2 通过才运行。正确化学臂先与 equal-budget clean continuation 配对；未胜过 clean 立即停止，不运行伪教师矩阵和多 seed。任何脚本必须一张 GPU、无手工内存请求、原子输出目录和服务器 preflight。

### R4：由证据决定部署形态

| 结果 | 科学判断 | 下一步 |
|---|---|---|
| `H_chem` 高，`H_geom` 低 | 化学纠错有效但不适合共享点积 | candidate-conditioned reranker/hybrid |
| `H_geom` 高，`H_obs` 低 | 几何可表示但单谱不可识别 | 需要候选输入、多谱/CE 或额外实验信息 |
| `H_obs` 高，R2 高，R3 低 | 显式共享核有效，神经压缩失败 | 部署扩展 embedding 或改压缩容量 |
| R2、R3 均高且胜过控制 | 化学先验成功进入 shared embedding | formula-disjoint 多 seed、outer/external |

这个裁决表不会预先强迫“必须 embedding”或“必须 reranker”；它用证据决定哪一种部署形态与化学信号相容。

## 10. 立即停止和立即保留的事项

### 停止

- 继续调整旧 candidate-residual Huber、global gradient ratio、clip、LR 或 epoch；
- 仅凭更复杂的 ICEBERG/FIORA 动作直接启动全局微调；
- 把 action teacher、rule effect 或 probe separation 报成 embedding pp；
- 在 inner fold 上继续多轮选择后声称 confirmation；
- 把 contract PASS、preflight PASS 或 loss 下降当作能力提升。

### 保留并前移

- 272-action residual ledger 及三个匹配控制；
- E1 已确认动作和完整逐 action 证据；
- domain-stratified empirical rule registry；
- `rule_mass` 显式共享核及 mass/shifted 控制；
- nonchemical full-candidate continuation；
- formula/identity/domain splits、完整 R@K/AUC 和 provenance contracts；
- qualified action delta 代码作为对照性实现资产，但在 R0 前不扩大训练。

## 11. 最终研究判断

此前的工作证明了三件重要事情：化学教师有信息、少量动作能纠错、干净谱学关系能改善共享相似度。失败发生在我们试图把这三句话直接连成“更强动作 -> 微调 shared embedding -> 3–5 pp”。中间缺了两个必要定理：目标对 shared geometry 是否可表示，以及特权化学信息是否可由干净谱图观测。

下一阶段最强路线不是继续堆动作，也不是立即转成 reranker。`H_chem -> H_geom` 已经首次跑通并保留了净 4.0039 pp 的自由几何上限；现在应集中完成 `H_geom -> H_obs` 的公式隔离可观测性审计，再决定 `H_obs -> H_opt` 的模型。这样既能最大程度开发已有动作，又能在信号不适合 spectrum-only 函数时及时转向正确的模型形式，避免再次用昂贵训练回答一个可以用公式隔离审计先回答的问题。
