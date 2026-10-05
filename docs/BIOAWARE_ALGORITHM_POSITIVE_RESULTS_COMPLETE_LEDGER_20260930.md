# BioAware 算法与全部正向结果总账（2026-09-30）

## 0. 文档范围与一句话结论

本文只整理 **BioAware 算法**，不讨论生物学课题结论，也不混入 Noise、ChemAware 或 P2b 的结果。覆盖范围包括：

1. 冻结 DreaMS embedding 之后的 BioAware 候选重排器；
2. 最新的 B30/B35 风险受控重排结果及六个生物来源的逐来源结果；
3. BioAware 监督直接进入共享 DreaMS encoder 的 B4/B6/B7/B32 结果；
4. B47 真值盲、样本局部、精确反应事件层的最新结果；
5. 所有数值为正、但因协议、泛化或安全门不足而不能升级为主性能主张的结果。

当前最严谨的总判断是：

> BioAware 已经建立了一个有效的、低干预的候选风险重排框架。在六来源打开开发协议中，B30 将 DreaMS Recall@1 从 66.05% 提高到 71.98%（+5.93 pp，54 修正/3 新增）；按物理查询去重后提高 +6.24 pp（50/3）。在 1,631-query 真实谱库协议中，整体 Recall@1 提高 +2.88 pp，MRR 提高 +1.47 pp，macro-query AUROC 提高 +1.35 pp。该增益在所有六个留一来源上均非负，但仍属于打开的多队列开发证据，不是独立外部 SOTA。共享 encoder 方向已证明直接梯度可以进入最后一层 Transformer 和官方投影头，并能在监督动作上实现大规模边界翻转；但完整隔离评估尚未复现重排器的 5–6 pp，因此目前不能说 BioAware 已经产出显著优于官方 DreaMS 的新 embedding。B47 则首次在两个外部研究中建立了真值盲的候选—种子—精确反应事件证据，其真实事件产率显著超过结构化空模型，但尚未进入揭盲排序评价。

---

## 1. 三类结果必须严格分开

| 层级 | 推理时改变什么 | 训练/输入信息 | 当前最好结果 | 当前资格 |
|---|---|---|---:|---|
| BioAware 重排器 | 改候选顺序，不改 embedding | DreaMS 候选分数、目录/拓扑/网络机会特征、跨来源风险记忆 | B30 +5.93 pp；B35 整体 +2.88 pp | 强开发证据，非独立外部验证 |
| BioAware 共享 encoder | 同一模型重编码 query 和 reference，改变 embedding space | 真候选与错误候选的直接 margin/listwise 梯度、安全边界、clean preservation | B32 监督动作 36/59 跨界；B7 held fold +0.92 pp | 可实现性成立，泛化性能未成立 |
| B47 精确事件层 | 生成样本局部候选特异反应证据 | 真值盲谱学种子、Rhea 反应事件、结构化空模型 | 4,996 个候选特异 query，2,239 个潜在干预机会 | 事件信号成立，尚无 accuracy 结果 |

其中：

- “动作提升”是冻结 embedding 后选择候选的提升，不等于 encoder 提升。
- “动作跨界”表示训练过的特定真/错候选 margin 改善，不等于 held-out Recall@1 提升。
- “潜在干预机会”只表示事件证据与 DreaMS 基线发生候选特异分歧，不等于修正。

---

## 2. BioAware 重排器究竟是什么算法

### 2.1 候选与谱学基线

每个 query 在固定质量与加合物协议中形成候选分子组。候选分子谱学分数为其所有参考谱与 query 的官方 DreaMS cosine 最大值：

\[
s_{\mathrm{DreaMS}}(q,c)=\max_{r\in R(c)}\cos(E(q),E(r)).
\]

所有比较必须使用相同 query、候选集合、参考谱聚合和并列规则；负候选与真值并列时按失败计算。

### 2.2 候选证据块

历史上构建过的候选证据包括：

- DreaMS Top1/Top2 分差及候选内相对谱学分数；
- 候选是否被反应/代谢目录覆盖、候选网络度数、质量窗内目录覆盖率；
- 身份隔离种子到候选的路径覆盖、最短深度、种子支持数；
- 原始 MS2 step-0 边完整性、bottleneck 与 step-1 增量证据；
- 参考谱数量、候选数量和跨来源候选历史；
- 低 margin 门、唯一 proposal 门、冲突弃权与回退 DreaMS。

候选专家使用候选组内 pairwise/浅层非线性排序，而不是把“有反应边”直接当作身份标签。其基本原则是：

> DreaMS 负责候选生成和高置信回退；BioAware 只在低置信候选组中提出一个候选特异动作，并在证据不足、冲突或风险过高时弃权。

### 2.3 B17：跨动作的嵌套裁决器

B17 在每个留一来源外层折中，使用外层训练来源选择 B12 的目录机会动作或 B16 的浅层非线性动作；测试来源不参与分支选择。B17 的打开开发结果为：

- 860 个 evaluation rows；
- Recall@1 +5.814 pp；
- 57 corrected / 7 introduced；
- \(C-2I=43\)；
- 28 个 corrected identities、26 个 corrected formulas；
- identity/formula cluster CI 下界均为正；
- 所有外层来源均非负。

### 2.4 B30：跨来源 candidate-sink 风险记忆

B30 不重新发明 proposal，而是在 B17 之上加入留一来源的候选风险记忆：

1. 对当前外层来源完全不可见；
2. 仅统计其他来源中相同 proposal candidate 的物理查询动作；
3. 候选至少在两个来源出现、至少积累三个物理动作，且历史 \(corrected-2\times introduced\le0\) 时，标记为 annotation sink；
4. 对 sink candidate 撤销 B17 动作并回退 DreaMS；
5. 未见候选不假设安全或有害，继续使用 B17。

这解决了高覆盖、高度数、glucose-like 候选反复吸走不同 query 的工程风险。它是候选身份级的跨来源可靠性记忆，不是反应机理模型，也不保证未见候选泛化。

### 2.5 B35：把动作扩展成完整排名

B30 本质是选择性 Top-1 路由器。B35 的完整排名定义为：

- 干预时，把冻结 B30 选中的候选提升为唯一第 1；
- 其他候选之间的 DreaMS 相对顺序不变；
- 不干预时，整个 DreaMS 排名不变；
- 正离子模式显式弃权。

因此 B35 可以严格计算 Recall@k、MRR 和 query-AUROC，但它仍不是对所有候选输出跨 query 可比较概率的通用概率排序器。

---

## 3. 重排器正向结果全表

### 3.1 化学完整性修正后的 v2 四来源结果

旧 595-query v1 使用了不纯的 MoNA negative 候选库，正式结果已由化学完整性修复后的 548-query v2 取代。v2 使用经结构和理论 `[M-H]-` 质量联合验证的候选库。

| 配方/协议 | n | DreaMS R@1 | BioAware R@1 | ΔR@1 | corrected/introduced | 统计/解释 |
|---|---:|---:|---:|---:|---:|---|
| `full_bioaware`，leave-source-out + identity/formula purge | 548 | 0.70255 | 0.73723 | **+3.47 pp** | **19/0** | formula CI +1.11 至 +6.29 pp；四来源均正 |
| `full_no_edge_gate`，高召回消融 | 548 | 0.70255 | 0.74270 | **+4.01 pp** | **23/1** | formula CI +1.51 至 +6.94 pp |

候选内联合置换网络特征后，主模型 null 平均仅 +0.044 pp，95% 上界 +0.365 pp；观测 +3.47 pp 的经验单侧 \(p=0.0099\)。

同一 v2 候选图的八单元消融揭示：

| 证据臂 | ΔR@1 | corrected/introduced | 结论 |
|---|---:|---:|---|
| spectral-only | 0.00 pp | 0/0 | DreaMS 单调再校准不能改序 |
| mass-membership only | +2.19 pp | 13/1 | 目录收录先验有明显信号 |
| known topology | +2.74 pp | 18/3 | 拓扑有收益也有伤害 |
| raw step-0 only | +2.01 pp | 16/5 | 单独使用不够安全 |
| network-only | +3.47 pp | 22/3 | 网络/目录证据块贡献最大 |
| spectral + known topology | +2.92 pp | 16/0 | 更保守的组合 |
| `full_bioaware`（八单元） | +3.10 pp | 17/0 | 高精度基线 |
| `full_no_edge_gate`（八单元） | +4.01 pp | 24/2 | 高召回但有代价 |

### 3.2 B9–B30 动作发现阶段的全部有价值正结果

这些数值全部来自已打开的多来源开发数据，价值是比较动作和安全层，不是独立确认。

| 阶段 | 核心策略 | ΔR@1 | corrected/introduced | 风险净收益 `C-2I` | 裁决 |
|---|---|---:|---:|---:|---|
| B11 | Full16 目录交互、nested OOF | +5.657 pp | 33/2 | 29 | 正向但身份覆盖窄 |
| B12 | 六域目录机会动作、domain LOSO | +5.814 pp | 58/8 | 42 | 宽覆盖动作基线 |
| B13 | 固定 hub-degree veto | +5.581 pp | 56/8 | 40 | 未优于 B12 |
| B14 | 反应可用性/强度 recipes | +5.698 pp | 57/8 | 41 | 路径参与但无独立增益 |
| B16 | 浅层非线性 pairwise action | +4.767 pp | 44/3 | 38 | 精度高、覆盖较低 |
| B17 | B12/B16 嵌套仲裁 | +5.814 pp | 57/7 | 43 | B30 前最安全路由器 |
| B18 | query-context nonlinear action | +4.884 pp | 45/3 | 39 | 未优于 B17 |
| B21 | mass-coverage fallback | +5.814 pp | 57/7 | 43 | 完全回退为 B17 |
| B22 | 九动作 cross-fit selector | +5.814 pp | 57/7 | 43 | 合理选择 B17 |
| B23 | catalog/reaction/coabundance consensus | +5.349 pp | 58/12 | 34 | 多 1 修正却多 5 伤害 |
| B29 | action-risk linear router | +5.698 pp | 55/6 | 43 | 少 1 伤害也少 2 修正 |
| **B30** | **leave-one-source-out candidate-sink veto** | **+5.930 pp** | **54/3** | **48** | 当前最好打开动作 |

### 3.3 B30 总体和物理查询结果

| 口径 | n | baseline R@1 | BioAware R@1 | ΔR@1 | corrected/introduced | CI |
|---|---:|---:|---:|---:|---:|---|
| row-weighted nested OOF | 860 | 0.66047 | 0.71977 | **+5.930 pp** | **54/3** | formula +3.313 至 +8.811 pp；identity +3.362 至 +8.734 pp |
| 物理 query 去重 | 753 | — | — | **+6.242 pp** | **50/3** | 所有来源非负 |

其他关键数字：

- row-weighted 干预率 11.40%（98/860）；
- corrected identities 27，corrected formulas 25；
- introduced identities/formulas 均为 3；
- 物理查询风险净收益 \(50-2\times3=44\)；
- B30 严格优于 B17 的 Recall@1、introduced 和风险净收益。

### 3.4 六个生物来源/组织的稳定正向结果

这是此前“不同生物组织数据集提升幅度不同”的完整结果。这里报告 row-weighted 外层折；每一折的 sink 历史只来自其余五个来源。

| 留出来源 | n | DreaMS R@1 | B30 R@1 | ΔR@1 | corrected/introduced | 干预数 |
|---|---:|---:|---:|---:|---:|---:|
| BV2cell | 95 | 0.78947 | 0.84211 | **+5.263 pp** | 5/0 | 10 |
| Mouse brain | 131 | 0.69466 | 0.76336 | **+6.870 pp** | 10/1 | 19 |
| Mouse liver | 176 | 0.68182 | 0.73864 | **+5.682 pp** | 10/0 | 16 |
| NIST plasma | 146 | 0.67808 | 0.73288 | **+5.479 pp** | 10/2 | 20 |
| ST001154 same-formula 10 ppm | 150 | 0.38000 | 0.46667 | **+8.667 pp** | 13/0 | 24 |
| KGMN200STD hidden-seed（row） | 162 | 0.77778 | 0.81481 | **+3.704 pp** | 6/0 | 9 |

KGMN 的 162 rows 含重复 seed-mask 评估。按 55 个物理 query 去重后是 2/0、+3.636 pp。不能把 162 rows 当作 162 个独立实验单位。

这些逐来源结果说明：

1. 提升不由单一组织驱动；
2. 困难的同分子式面板收益最大；
3. 不同来源的增益幅度不同，符合基线难度、候选结构和可干预错误数不同；
4. 但六个来源均参与了动作开发/交叉拟合，因此“全折正向”仍不是新数据库外部复现。

### 3.5 B35 真实谱库检索协议

B35 使用 259,176 张参考谱、30,984 个分子身份；query-local 采用 strict 10 ppm、same-adduct、每候选分子对所有参考谱取最大官方 DreaMS cosine，负候选并列领先。

#### 全部 1,631 个物理查询

| 指标 | 官方 DreaMS | DreaMS + B30 | 变化 |
|---|---:|---:|---:|
| Recall@1 | 0.75414 | 0.78296 | **+2.882 pp** |
| Recall@2 | 0.88289 | 0.88473 | +0.184 pp |
| Recall@5 | 0.98529 | 0.98590 | +0.061 pp |
| Recall@10 | 0.99755 | 0.99755 | 0 |
| Recall@20 | 1.00000 | 1.00000 | 0 |
| MRR | 0.85236 | 0.86707 | **+1.471 pp** |
| macro-query AUROC | 0.85475 | 0.86820 | **+1.345 pp** |

Recall@1 为 50 corrected / 3 introduced，McNemar exact \(p=5.52\times10^{-12}\)。公式簇 95% CI 为 +1.553 至 +4.388 pp，身份簇 CI 为 +1.623 至 +4.320 pp。92/1,631（5.64%）查询发生干预。

#### 753 个合格负离子物理查询

| 指标 | 官方 DreaMS | DreaMS + B30 | 变化 |
|---|---:|---:|---:|
| Recall@1 | 0.64409 | 0.70651 | **+6.242 pp** |
| MRR | 0.78133 | 0.81321 | **+3.187 pp** |
| macro-query AUROC | 0.78457 | 0.81370 | **+2.913 pp** |

该子集为 50/3。另有 878 个正离子查询严格弃权，R@1 保持 0.84852、0 corrected/0 introduced。整体 +2.88 pp 小于负离子 +6.24 pp，原因正是正离子适用域不成立时不改排。

B35 的价值是证明 B30 不只是一个孤立的 top1 计数脚本，而能嵌入完整真实谱库排名并同时改善 R@1、MRR 和 query-AUROC。它仍是打开的多队列开发 benchmark，不是独立外部确认。

---

## 4. 重排收益的真实来源：重要归因修正

### 4.1 B36 决定性消融

在相同 860-query 六来源协议、相同浅层 pairwise HGB、相同 nested source holdout 和门控下：

| 证据臂 | ΔR@1 | corrected/introduced | `C-2I` |
|---|---:|---:|---:|
| spectral-only | 0.000 pp | 0/0 | 0 |
| spectral + catalog | **+5.581 pp** | **50/2** | **46** |
| spectral + reaction | +1.395 pp | 19/7 | 5 |
| spectral + catalog + reaction | +5.000 pp | 47/4 | 39 |

关键配对结果：

- full 相对 catalog-only 为 -0.581 pp，identity CI [-1.537, +0.355] pp，formula CI [-1.517, +0.346] pp；
- reaction-only 相对 spectral-only 虽为 +1.395 pp，但 identity/formula CI 均跨 0；
- 真实反应上下文未超过 within-query 和 catalog-stratum 错配对照。

因此当前 5–6 pp 主结果的正确表述是：

> 它主要来自“谱学边界 + 目录覆盖/候选熟悉度 + 风险门控/跨来源 sink veto”的候选专家，而不是已经被证明的候选特异 Rhea 反应推理。

这不使 B30/B35 失去工程价值，但限制了“BioAware 生化网络算法”的机理主张。

### 4.2 独立目录/拓扑的历史正数与后续降级

在同一打开六来源协议上，B42/B43 曾获得：

| 配方 | ΔR@1 | corrected/introduced |
|---|---:|---:|
| strict KEGG replay | +5.930 pp | 55/4 |
| strict KEGG static | +5.698 pp | 54/5 |
| Rhea static | +6.047 pp | 57/5 |
| EMRN static | +5.698 pp | 56/7 |
| KEGG + Rhea consensus | **+6.279 pp** | **55/1** |
| 三目录 union | +6.047 pp | 62/10 |

更严格的 mapped-competition 子集仍为正：strict KEGG truth/wrong 都映射时 +2.98 pp（25/5）；任一目录双方都映射时 +3.15 pp（23/1）；所有候选均映射时 +4.01 pp（11/0）。

但这些正数不能作为当前外部主结果，因为：

- B44 MassBank 一次性外部评估反向为 -0.56 pp；
- B45 在覆盖度、目录度数和参考谱数匹配后，拓扑增量只剩 +0.233 pp（4/2），CI 跨 0；
- 因此静态目录结果主要说明跨库可观测性/收录先验，不证明可迁移的反应拓扑机制。

这些结果应保留为算法演进与反事实证据，但不能与 B30/B35 主性能数字并列声称“跨数据库稳定”。

---

## 5. 共享 embedding / encoder 微调：算法与全部正向结果

### 5.1 训练定义

BioAware encoder 路线不是在 embedding 后拟合一个头，而是让 query、truth reference 和 wrong reference 通过同一个 DreaMS encoder：

\[
z_q=E_\theta(x_q),\qquad z_+=E_\theta(x_+),\qquad z_-=E_\theta(x_-).
\]

直接优化真候选相对错误候选的 margin/listwise 目标，同时对官方正确查询施加单边安全 margin floor，并用 clean preservation 限制表示漂移。工程合同为：

- 同一共享 encoder 同时处理 query/reference；
- dropout 关闭；
- 只解冻 7 层 Transformer 中最后一层（block 6）和官方 projection head；
- 不拉近“反应相邻但不同身份”的分子；
- 不使用 P2b、phenotype 或 truth-at-inference；
- candidate/network 信息只用于训练时选择 true/wrong 监督对；
- 推理时只输入干净谱图并输出新 embedding。

### 5.2 B4/B6 五折 OOF：第一次完整泛化审计

548-query、164 identities、136 formulas 的打开队列五个 truth-formula folds：

| arm | baseline R@1 | 新 R@1 | ΔR@1 | corrected/introduced | ΔMRR | formula CI | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| `full_bioaware_safe` | 0.70255 | 0.69891 | -0.365 pp | 4/6 | -0.291 pp | [-1.501, +0.735] pp | 失败 |
| `full_no_edge_high_recall` | 0.70255 | 0.70438 | **+0.182 pp** | **5/4** | **+0.109 pp** | [-0.928, +1.257] pp | 正向但不显著，风险净收益 -3 |

高召回 arm 的正向 fold 包括：

- fold 3：+2.353 pp，2/0；
- fold 4：+1.418 pp，2/0；
- Mouse brain HILIC：+1.333 pp，1/0；
- Mouse liver HILIC：+1.020 pp，1/0；
- Mouse liver RPLC：+2.564 pp，2/0。

但 fold 1、2 退化，formula CI 跨 0，且总体 `C-2I=-3`。因此该结果只能证明“有可传递的局部正梯度”，不能证明新 embedding 整体更好。

### 5.3 B7 单折直接图动作注入

B7 的动作路由器本身在 548-query 开发图上为：

- baseline R@1 0.70255；
- action R@1 0.75091；
- +4.836 pp；
- 191 corrected / 85 introduced（outer rows），风险净收益 21；
- 27 corrected identities、25 formulas。

这只是训练动作上限，不是 encoder 结果。真正 held fold 的共享 encoder 结果为：

| 指标 | baseline | shared encoder | 变化 |
|---|---:|---:|---:|
| Recall@1 | 0.65138 | 0.66055 | **+0.917 pp** |
| MRR | 0.77245 | 0.77092 | -0.153 pp |
| corrected/introduced | — | 3/2 | `C-2I=-1` |
| clean preservation | — | 0.98845 | — |

这是共享 encoder 在 held fold 上的明确正 Recall@1 信号，但 MRR 与风险净收益不通过，只能作为可行性证据。

### 5.4 B32：当前最强的直接梯度可实现性结果

B32 把两套动作账本去重并补齐已知伤害：

- 63 corrective rows，折叠为 59 个物理谱；
- 31 个 identities、28 个 formulas；
- 14 个 safety rows；
- 3 个 sink-conflicting corrections 的 corrective weight 固定为 0。

同一共享 query/reference encoder 中，仅训练最后一层 Transformer 和官方 projection head。结果：

- 59 个 corrective action 的平均 margin 从 **-0.03300** 提高到 **+0.00844**；
- **36/59** 个原本错误的 pairwise 边界被跨越；
- **0/14** 已知 harm pairs 翻转；
- query 最低 preservation **0.9962**；
- reference 最低 preservation **0.9979**；
- 96/96 个 optimizer steps 触发全局 norm cap 1.0，原始 gradient norm 约 4.3–14.3。

B32 严格证明：动作方向可以不经蒸馏，直接通过共享 encoder 注入，并在很小 clean drift 下实现。它没有计算完整候选图 Recall@1，因此不能写成 embedding 性能提升。

### 5.5 共享 encoder 的当前结论

正向事实：

1. 直接梯度可到达最后一层 backbone 与官方 head；
2. B32 在监督动作上实现 36/59 跨界且 14/14 safety 不翻转；
3. B7 held fold 的 R@1 为 +0.917 pp；
4. B4 高召回五折 OOF 为 +0.182 pp，MRR 同向。

尚未成立：

1. 没有完整 formula/source-isolated 多折结果达到 3–5 pp；
2. B4 formula CI 跨 0，B7 MRR 与风险净收益失败；
3. B32 没有完整候选图评估；
4. 因此不能把 B30 的 +5.93 pp 写成 encoder 已经学到 +5.93 pp。

---

## 6. B47：最近的真值盲精确事件算法

### 6.1 为什么 B47 与旧静态目录重排不同

B47 不再把“候选存在于 Rhea/KEGG”直接当分数，而是在每个外部样本中构建原子事件：

\[
(q,c,s,r),
\]

其中 \(q\) 是 query，\(c\) 是候选，\(s\) 是同一样本独立高置信种子，\(r\) 是连接候选和种子的精确 Rhea reaction transform。约束包括：

- truth 和 phenotype 全程关闭；
- query 自身不能作为 seed；
- 不变计量参与物被删除；
- currency/hub 受控；
- 候选—种子关系必须满足公式质量对、加合物对和 reaction transform signature；
- 与 degree-rewired network、seed-context permutation 和 candidate-identity diagnostic 比较。

### 6.2 B47 U3 正向事件结果

外部真值盲图规模：51,976 queries、216,793 candidate rows、3,243 seed events、127 seed identities。

真实 Rhea 事件结果：

- 19,602 atomic events；
- 18,012 eligible events；
- 5,386 event-supported queries；
- **4,996 candidate-specific queries**；
- **2,239 potential ranking-change opportunities**，占全部 query 的 4.31%；
- 154 个 supported candidate identities；
- 329 个 reactions used；
- formula-mass pair fraction 1.000；
- reaction-transform signature match 1.000；
- supported-adduct pair fraction 1.000。

分研究：

| 研究 | query | candidate-specific query | potential intervention |
|---|---:|---:|---:|
| ST001122 | 33,829 | 3,316 | 1,510 |
| ST003356 | 18,147 | 1,680 | 729 |

结构化空模型：

- 真实 candidate-specific queries 4,996；
- degree-rewired 20 次最大 91；
- seed-context permutation 20 次最大 4,072；
- 真实相对最坏结构 null lift **+22.69%**；
- ST001122 lift +23.18%，ST003356 lift +19.91%；
- 两类结构 null 的总体及逐来源经验单侧 \(p=1/21=0.047619\)。

这证明样本局部、候选特异的精确反应事件结构显著多于仅由网络度数或 seed 上下文频率产生的机会。

### 6.3 为什么 B47 尚未成为最新重排性能

U3 没有打开 annotation truth，也没有计算 corrected/introduced。以下集中度门失败：

- effective seed identities 19.32（门为 50）；
- effective intervention candidate identities 32.56（门为 50）；
- seed identities used 86（门为 100）；
- intervention candidate formulas 94（门为 100）；
- 最大 seed event 占比 11.78%（门为 10%）；
- 最大 intervention candidate/formula 占比 8.53%（candidate 门为 5%）。

因此 `pass_to_frozen_event_ranking_evaluation=false`。2,239 是可评价机会，不是 2,239 个修正，更不是 +4.31 pp 性能。

### 6.4 B47 谱学 unary 的小正向结果

在 corrected 83,619-query development graph 上，U1 对 molecule-max 加入极小 mean/top2-mean shrinkage：

- R@1 0.928760 → 0.929442，**+0.0682 pp**；
- formula CI [+0.0249, +0.1135] pp；
- near R@1 +0.1348 pp；
- MRR +0.0398 pp；
- 131 corrected / 74 introduced，\(C-2I=-17\)；
- fold 0 轻微为负。

这是大图上的微小谱学聚合信号，但安全门失败，不能用于 B47。修复候选顺序泄漏后的 U1b v2 是严格 0/0、R@1 0，因此不构成额外正向性能。

---

## 7. 支持性正结果：必须保留但不能升级

| 结果 | 正向数值 | 为什么不能作为主结论 |
|---|---:|---|
| 早期 21-query hyperedge-completeness 事后规则 | 1/0，表面 +4.76 pp | 规则由同批错误启发，属于 post-hoc mechanism fit |
| MetDNA3 HILIC depth-3 消融 | +2.56 pp，3/0 | 仅两个独立身份，formula CI 下界为 0 |
| dependency-corrected 一跳融合 | +0.244 pp，4/2 | CI 跨 0 |
| B36 reaction-only | +1.395 pp，19/7 | identity/formula CI 跨 0，跨来源不一致 |
| B42/B43 static catalog | +2.98 至 +6.28 pp | 后续外部反向、覆盖中和后趋近 0 |
| B47 U1 spectrum unary | +0.0682 pp | 131/74、风险净收益负、折间不全非负 |
| B4 high-recall shared encoder | +0.182 pp，5/4 | formula CI 跨 0，风险净收益负 |
| B7 shared encoder held fold | +0.917 pp，3/2 | 单折，MRR 下降，风险净收益负 |
| B32 shared encoder action canary | 36/59 跨界，0/14 harm flip | 仅监督动作，无完整检索 R@1 |
| B47 U3 exact events | 4,996 candidate-specific，2,239 opportunities | 未打开真值，未计算 accuracy |

旧 v1 595-query 负离子结果（identity-purged +3.36 pp；source/formula-purged +3.03 pp）已被化学完整性审计取代，只能作为历史。正式引用应使用 v2 的 548-query +3.47/+4.01 pp。

---

## 8. 当前可以和不可以怎样表述

### 8.1 可以表述

1. BioAware 构建了 DreaMS 之后的候选特异、风险受控重排器。
2. 在六来源 opened nested OOF 中，B30 为 +5.93 pp、54/3，所有来源非负。
3. 在 1,631-query 真实谱库协议中，整体 R@1 +2.88 pp、MRR +1.47 pp、macro-query AUROC +1.35 pp；合格负离子子集 R@1 +6.24 pp。
4. 增益主要发生在 Top-1 局部次序修正，而非扩大 Top-5/10/20 检索覆盖。
5. B32 证明 BioAware 动作可直接注入共享 DreaMS encoder，并在高 preservation 下实现监督 margin 跨界。
6. B47 证明真值盲、样本局部的精确 Rhea 事件产率显著超过网络重连和 seed-context 空模型。

### 8.2 不可以表述

1. 不可说 BioAware 已经在独立数据库稳定超过 DreaMS 或达到 SOTA。
2. 不可说当前 5–6 pp 由真实反应拓扑造成；B36 不支持该归因。
3. 不可说共享 encoder 已经提高 5–6 pp；当前 held-out encoder 正向结果不足 1 pp 且未过完整门。
4. 不可把 B47 的 2,239 个 opportunities 写成 corrected 或 4.31 pp gain。
5. 不可说正离子模式有效；B35 在正离子上是明确弃权。
6. 不可把 KGMN 重复 seed-mask rows 当作独立物理查询。

---

## 9. 当前算法价值与最硬的剩余缺口

### 已经解决的问题

- 证明强谱学 embedding 仍存在可被候选先验修正的 Top-1 局部错误；
- 建立 corrected/introduced、\(C-2I\)、formula/identity cluster CI 和逐来源非退化的风险评价体系；
- 将跨来源 annotation sink 变成显式安全层，显著减少新增错误；
- 建立完整真实谱库 Recall@k、MRR、query-AUROC 评价；
- 证明动作可直接到达共享 encoder，而不是只能通过蒸馏头；
- 建立真值盲、样本局部、候选特异的精确反应事件及结构化空模型。

### 尚未解决的问题

- B30/B35 的 5–6 pp 主要是目录覆盖/候选熟悉度和风险记忆，未证明是生化反应推理；
- 已知 candidate sink 不能泛化到未见候选；
- opened six-source 结果尚缺新数据库一次性外部确认；
- 共享 encoder 尚未把动作层 5–6 pp 转移为稳定、显著、低伤害的 clean embedding 增益；
- B47 事件证据集中度过高，尚不能安全揭盲评价；
- 正离子适用域尚未建立。

因此截至 2026-09-30，BioAware 的最硬成果不是“已经完成端到端生物网络 embedding”，而是两个互补节点：

1. **一个在打开多来源与真实谱库协议上显著、低伤害的候选风险重排器；**
2. **一个已通过结构化空模型的真值盲精确事件层，以及已证明可直接接收其监督的共享 encoder 工程桥。**

两者之间的最后科学缺口，是在不依赖候选身份记忆的情况下，把 B47 的候选特异事件变成可外推的动作，并在 formula/source-isolated 的完整候选图上证明共享 encoder 的 R@1、MRR 与风险净收益同时为正。

---

## 10. 主要可追溯工件

- `data/validation/bioaware_metdna3_external_negative_source_loso_v2_full_bioaware/`
- `data/validation/bioaware_metdna3_external_negative_source_loso_v2_full_no_edge_gate/`
- `data/validation/bioaware_b30_cross_source_sink_veto_localcheck_20260907_v1/report.json`
- `data/validation/bioaware_b35_real_library_retrieval_formal_local_20260912/report.json`
- `data/validation/bioaware_b36_reaction_specificity_2336381/report.json`
- `data/validation/bioaware_b4_direct_shared_embedding_v1_20260905/runs/summary.json`
- `data/validation/bioaware_b7_direct_graph_injection_2331964_resume_2331981/`
- `data/validation/bioaware_b32_direct_shared_embedding_canary_2332639/`
- `data/validation/bioaware_b47_u1_spectrum_unary_dev_20260921_v1/report.json`
- `data/validation/bioaware_b47_u3_event_yield_20260922_v3/report.json`（服务器工件；本地证据来自正式完成日志）

对应方法与裁决文档：

- `docs/BIOAWARE_NEGATIVE_NETWORK_EXPERT_V2_CHEMICAL_INTEGRITY_20260901.md`
- `docs/BIOAWARE_B9_TO_B15_ACTION_MINING_RESULT_20260907.md`
- `docs/BIOAWARE_B35_REAL_LIBRARY_RETRIEVAL_BENCHMARK_20260912.md`
- `docs/BIOAWARE_B36_REACTION_SPECIFICITY_ABLATION_CONTRACT_20260913.md`
- `docs/BIOAWARE_B42_B43_INDEPENDENT_CATALOG_TOPOLOGY_RESULT_20260913.md`
- `docs/BIOAWARE_B45_COVERAGE_NEUTRAL_TOPOLOGY_RESULT_20260913.md`
- `docs/BIOAWARE_B47_U1_SPECTRUM_UNARY_RESULT_20260921.md`
- `docs/BIOAWARE_B47_U3_EVENT_YIELD_PROTOCOL_AMENDMENT_20260921.md`

