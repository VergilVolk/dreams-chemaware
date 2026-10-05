# BioAware 前沿证据审计与下一代算法决策（2026-09-14）

## 执行结论

当前最重要的决定不是放弃约 `+5–6 pp` 的结果，也不是把它直接蒸馏进 DreaMS，而是把它放回正确的位置：

1. **B30/B37 是有效的开发期候选风险先验**。在 860 个已开放负离子查询上，B30 达到 `+5.930 pp`、54 corrected / 3 introduced，说明候选目录覆盖和历史“错误汇聚点”确实能纠正一批 DreaMS 错排。
2. **它尚不是反应网络算法的有效性证据**。B36–B40 的控制实验表明，显式 Rhea 路径、事件摘要和多跳扩散没有在目录先验之上增加可靠收益；B45 在匹配目录覆盖/度数后仅约 `+0.23 pp` 且区间跨零。
3. **它也不能直接变成只看 clean spectrum 的通用 embedding**。目录成员、网络度、候选历史和样本内邻居依赖候选库与样本上下文；它们不是谱图本身的确定函数。同一张谱在两个候选库或两个生物样本中应得到不同的后验排序，但通用 `E(x)` 只能输出同一个向量。
4. **外部证据已经否定“把目录捷径继续调强”**。B44 在独立 MassBank 上从内部正增益反转为 `-0.56 pp`（8/13）；B45、B46 进一步否定了覆盖中性的静态拓扑与聚合上下文。它们是冻结负证据，不得再调门、换模型或蒸馏。
5. **最高性价比的下一步不是另一个特征 sweep，也不是立即训练 GNN**。应在已经冻结的 B47 真实样本候选图上，先构建候选—种子—反应—离子形态的精确事件图，用全局一致性和结构化空模型证明“生物上下文在目录先验之外有信息”，再决定是否训练上下文 adapter。

因此，下一代方法的正确定位是：

> **BioAware-ECSR（Event-Conditioned Structured Retrieval）**：以 DreaMS 为谱学 unary score，以离子形态、反应变换和样本内证据为稀疏 event factors，通过全局约束与风险拒答完成候选排序；只有当 event factors 在独立、覆盖中性的验证中胜过目录先验与结构化随机图，才将其转移到 context-conditioned representation。

这不是对现有 +5.9 pp 的否定，而是把它从“伪装成生物机制的结果”降为一个必须超越的强工程阳性对照。

## 一、项目证据：此前的 +5.9 pp 究竟是什么

### 1.1 能保留的事实

冻结的 B30 开发结果为：

- 860 个负离子、六来源、nested leave-one-source-out 查询；
- Recall@1 `0.6605 -> 0.7198`，即 `+5.930 pp`；
- 54 corrected、3 introduced，风险净收益 `54 - 2×3 = 48`；
- 六个来源方向均为正，formula/identity cluster CI 下界大于零；
- B30 通过跨来源历史识别被反复错误提升的候选 identity，并在其风险净收益非正时回退 DreaMS。

这是一个真实而有工程价值的结果。它证明：**当候选库和开发域稳定时，候选可观测性及跨来源失败历史可以充当低伤害的选择性路由信号。**

### 1.2 不能保留的解释

后续消融已经给出清晰反证：

- B9 的反应特异谱学变换约为 `-0.07 ~ +0.07 pp`；
- B36 中 `spectral + catalogue` 约 `+5.58 pp`，`spectral + reaction` 仅约 `+1.40 pp` 且 cluster CI 跨零，full 还低于 catalogue-only；
- B37 中大多数修正是 truth 在目录内、DreaMS 错候选不在目录内；
- B38 的直接路径和更丰富事件摘要相对 topology 反而下降；
- B39 的六个预注册一步 Rhea 原子动作均未超过 topology；
- B40 的四跳扩散和样本局部传播显著破坏 topology 收益；
- B44 在独立 MassBank 上为 `-0.56 pp`；
- B45 在目录覆盖与度数匹配后为约 `+0.23 pp`，CI 跨零；
- B46 的 dual-context 聚合为 `-0.77 pp`，10 corrected / 21 introduced。

所以 +5.9 pp 最准确的名称是：

> **strict-KEGG/catalogue-aware candidate risk routing**，而不是 reaction propagation、biochemical reasoning 或 BioAware embedding。

### 1.3 为什么不能直接注入通用 clean-spectrum embedding

设谱图为 `x`，候选库为 `C`，样本内已观测种子为 `O`。通用 DreaMS 表示为：

`z = E(x)`。

但当前 +5.9 pp 信号依赖：

`catalogue_membership(c), degree(c), history(c), O, C`。

存在两个上下文 `(O1,C1)` 与 `(O2,C2)`，同一谱图 `x` 的合理候选顺序可能不同。任何只计算 `E(x)` 的模型都无法同时表达这两个后验。把上下文标签硬蒸馏进 `E(x)`，得到的往往是数据集/目录偏好，而不是可迁移的谱学几何；B44 正是这一点的外部反例。

因此必须区分：

- **通用谱学表示** `z_spec = E(x)`：只描述谱图；
- **候选结构表示** `z_mol(c)`：描述候选结构；
- **样本上下文表示** `h_S(c)`：描述候选在当前样本中的离子、反应和共现证据；
- **上下文条件分数或表示**：只在注释时组合上述信息。

## 二、前沿方法真正提供的启发

### 2.1 MetDNA3：知识网络必须被实验网络反复约束

[MetDNA3（Nature Communications, 2025）](https://www.nature.com/articles/s41467-025-63536-6)不是把“网络距离”直接加到分数上。它先以 MS1 将实验特征映射到知识网络，再用反应关系和 MS2 相似性构造知识层—数据层一致的双层拓扑；传播时，候选代谢物邻居和实验 feature 邻居必须在两层同时匹配。作者报告，这种预映射将一个示例 MRN 压缩到约 0.4% 的节点和 2.3% 的边，并通过受约束的双层传播提高覆盖和正确率。

对本项目的直接启发是：

- 网络边不能脱离真实 feature 使用；
- “候选属于网络”不等于“当前样本支持该候选”；
- 每次传播必须有数据层和知识层的交叉证据；
- 预测扩展的反应边只能用于候选传播，不能冒充新反应发现。

### 2.2 KGMN：反应网络、MS2 网络和峰相关网络是三个不同证据层

[KGMN（Nature Communications, 2022）](https://www.nature.com/articles/s41467-022-34537-6)联合知识型代谢反应网络、知识引导的 MS2 相似性网络和全局 peak correlation。关键不是“多加几个分数”，而是让 MS1、RT、MS2、生化变换和峰间关系相互约束。

我们的 B46 将不同 seed、不同反应记录压成 query-candidate 聚合分数，丢失了事件身份；它不等价于 KGMN 的多层网络。这解释了为何“看似有两个上下文证据”仍可能是两个互不相干事件的伪交集。

### 2.3 NetID：局部贪心传播应让位于全局一致性

[NetID（Nature Methods, 2021；官方代码）](https://github.com/LiChenPU/NetID)同时利用质量、RT、MS/MS、加合物、同位素、源内碎片和生化质量变换，以全局优化选择一组相互一致的 peak assignments。它的核心价值不是某个单特征更强，而是避免多个局部正确动作在全局上互相冲突。

对 BioAware 的启发是：不能逐 query 独立地“看见一个种子就加分”；同一样本内的候选分配必须共同满足：

- 一个离子家族尽量对应一个母体分子；
- 加合物/同位素/源内碎片关系不能重复计为多个代谢物；
- 一个高连接候选不能无成本吸走许多 feature；
- 同一事件的反应方向、质量变化和实验 feature 必须一致。

### 2.4 IIMN：先解决离子身份，再谈代谢物网络

[Ion Identity Molecular Networking](https://pmc.ncbi.nlm.nih.gov/articles/PMC8219731/)通过 RT、峰形相关、跨样本强度相关和已知质量差将同一分子的不同离子形式合并。论文明确指出，最常见的 `[M+H]+` 与 `[M+Na]+` 往往不能仅靠 MS2 相似性连接。

B47 当前冻结候选图明确写着 query adduct unknown，而 `[M+H]+` 与 `[M+Na]+` 在同一质量窗竞争。这意味着当前最大的低成本缺口不是更深的 Rhea GNN，而是**ion-family resolution**。如果不先解决，网络可能把一个母体的多个离子当作多个种子或多个被注释事件，造成假支持和重复证据。

### 2.5 BAM：反应先验应成为“带类型的结构变换”，而非邻接标签

[BAM 官方实现](https://github.com/HassounLab/BAM)调用 PROXIMAL2/位点预测，将已知锚点结构通过具体 biotransformation rule 生成候选产物。其可借鉴之处是：反应证据应包含作用位点、原子变化和产物结构，而不是只有 `one_hop=1`。

对我们而言，Rhea 边至少应展开为：

`seed structure -> typed atom transformation -> candidate structure`

并检查 precursor mass residual、候选结构差异和可解释碎片变化是否一致。否则 degree/path 仍然主要是数据库覆盖信号。

### 2.6 JESTR、MSAlign、FLARE：候选辨识的前沿是条件化的谱—结构对齐

- [JESTR](https://pmc.ncbi.nlm.nih.gov/articles/PMC11601792/)学习 spectrum–molecule joint space，并在训练后段加入同分子式难候选正则；论文同时报告这种正则在 MassSpecGym 上并不稳定，提示候选策略与 split/domain shift 决定收益。
- [MSAlign（2026）](https://arxiv.org/abs/2605.19752)冻结 DreaMS 与 ChemBERTa，仅训练轻量投影，用 candidate-based contrastive objective 对齐谱图和分子。
- [FLARE（2026）](https://pubmed.ncbi.nlm.nih.gov/41659479/)不只比较全局向量，而以双向 peak–atom late interaction 累积局部对应，兼顾性能与可解释性。

这些工作提示：如果目标是改善结构检索，单一 spectrum–spectrum cosine 不足。BioAware 更合理的谱学基座是候选条件化的 `S_spec(x,c)`，其上再加入事件级生物证据；不是把“候选在 KEGG 中”写进谱图向量。

### 2.7 可靠性是独立任务，不应靠宽门控掩盖

[2026 年 conformal molecular retrieval](https://pubmed.ncbi.nlm.nih.gov/42113637/)为每张谱构造具有目标覆盖率的候选集合，并强调 calibration/test shift 会扩大集合。BioAware 的输出因此应包含：

- 候选排序；
- 校准的不确定性/候选集合；
- 不满足证据门时明确 abstain。

不能只报告“介入子集准确率”，也不能通过缩窄 gate 把总体 embedding 提升与选择性预测混在一起。

### 2.8 2026 年的评估教训：漂亮数字必须先排除 shortcut

[MassSpecGym in the Wild（2026）](https://openreview.net/pdf/232c7367849f16949a8a87ff30e963b877be55de.pdf)系统审计了分子检索工作中的泄漏、候选捷径、实现与评估偏差。它与我们的 B44 经验一致：一个内部大增益如果来自 canonicalization、目录覆盖、候选生成或身份记忆，不能当成可迁移的谱学/生物学性能。

因此，BioAware 的创新不应是“比别人再多一个图网络”，而应是：**把事件级生化证据、离子身份与候选谱—结构对应纳入一个严格抗捷径的结构化检索协议。**

## 三、下一代算法：BioAware-ECSR

### 3.1 变量与证据

对真实样本中的 query feature `q` 和候选分子 `c`，定义：

- `u_spec(q,c)`：DreaMS 或谱—结构模型提供的谱学 unary；
- `u_catalog(c)`：目录覆盖，只作显式对照/缺失 mask，不当生物机制；
- `f_ion(q,c)`：adduct、isotope、multimer、in-source fragment 与 ion-family 一致性；
- `f_event(q,c,s,r)`：候选 `c` 与样本内 seed `s` 经反应 `r` 相连的精确事件证据；
- `f_transform(q,c,s,r)`：反应原子/质量变换与 precursor、MS2 局部变化的一致性；
- `f_sample(q,s)`：RT、峰形、跨样本丰度或同一样本存在性；
- `f_conflict`：一个 ion family 被重复分配、hub 候选吸附过多 query、方向/质量冲突等惩罚。

一个最小可解释能量模型为：

`Score(y) = Σ_q u_spec(q,y_q) + Σ_event g_event f_event + Σ_family f_ion - Σ_conflict f_conflict`

其中 `y_q` 是 query 的候选分配，`g_event` 必须由 event completeness 与校准可靠性门控。模型优化的是整张样本 feature 图的联合赋值，而不是独立 query 的逐个加分。

### 3.2 “事件完整”必须满足什么

一条可用于模型的 BioAware event 至少包含：

`(query_feature, candidate, seed_feature, seed_compound, reaction_id, direction)`

并具有下列证据中的预注册组合：

1. query 与 seed 是不同 ion family；
2. seed 的谱学置信度经外层校准，且 leave-query-out、leave-truth-identity-out；
3. 反应方向与结构变换可定义；
4. 理论质量变化与两个 feature 的中性质量变化匹配；
5. transformed-fragment 或 peak–atom 局部证据支持该变化；
6. RT/峰形/跨样本丰度证据可用时方向一致；
7. 候选组内该事件对 truth-like candidate 有区分度，而非所有候选共享同一分数。

仅有 `candidate in network`、degree、path count 或几跳距离不构成 event。

### 3.3 两种表示，不再强迫一个向量承担所有职责

建议最终输出拆成：

1. **通用谱图表示**：`z_spec = E_DreaMS(x)`，保留跨样本、候选无关的谱学语义；
2. **候选条件表示**：`z_c^S = z_mol(c) + α(q,c,S) A(z_mol(c), h_S(c))`；
3. **最终分数**：`S(q,c|S) = S_spec(z_spec,z_c^S) + structured_event_score`。

当上下文不存在、冲突或分布外时，`α -> 0`，严格回退谱学基线。这样既避免污染通用 DreaMS embedding，也允许同一谱图在不同样本上下文中得到不同候选后验。

若后续一定要微调 shared spectral encoder，合理目标不是把反应邻居谱图无条件拉近，而是训练**关系条件变换**：

`score_r(seed,candidate) = -||T_r(z_seed) - z_candidate||²`

其中 `T_r` 由反应类型、方向和结构变换条件化；负样本必须是质量、目录覆盖、度数、结构相似性和样本可观测性匹配的非邻居。只有 event model 已经独立胜过这些 matched controls 后，这种微调才有合法教师。

## 四、B47：下一步应做什么

B47 已冻结 51,976 个真实样本 query events、10,578 个全局候选 identity；每 query 候选分子中位数 3，参考谱中位数 20，且当前 candidate graph 未打开 truth。这个规模足以做正式的事件模型，但 seed 生成前还有三个阻塞审计。

### B47-M1A：复用已有冻结 seed 产物，不覆盖、不重跑

当前服务器提示 `Refusing to overwrite frozen result` 是正确的 fail-closed 行为。下一步只应对现有 seed 目录运行 validator，并下载 `report.json` 与 seed table 哈希。若文件不完整，创建显式 `v2` 路径；不得删除或覆盖 v1。

### B47-M1B：校准种子，而不是只用固定 cosine 阈值

现有设计以 max reference cosine、固定 score/margin、跨样本 modal consistency 选 seed，仍需在 truth-blind 阶段冻结以下对照：

- `max` vs top-k robust aggregation / multiplicity-adjusted aggregation；
- 候选 reference spectrum 数量对 max score 的影响；
- source/instrument/candidate-count 分层的 null score；
- seed 候选集合或 conformal coverage，而非只给硬 identity；
- 对高目录度候选与 reference-rich 候选进行单独风险标记。

这一步不需要打开 query truth；可用 reference self-retrieval、decoy 和 held-reference calibration 完成。

### B47-M1C：先构建 ion-family graph

使用现有 feature table 的：

- RT 共洗脱；
- 峰形相关（如果原始 MS1 扫描可恢复）；
- 跨样本强度相关；
- adduct/isotope/multimer/in-source fragment 质量差；
- polarity-specific adduct rules。

把 feature-level seed 折叠为 molecule-event-level seed，并保存从每个原始 feature 到 ion family 的可追溯映射。没有这层，任何“同一样本支持”都可能重复计数。

### B47-M2：构建精确事件表与结构化空模型

对每个 `(q,c)` 枚举可用 seed，保存完整 event tuple 与原始证据；同时冻结至少四个 null：

1. degree-preserving reaction rewiring；
2. same-formula/degree-matched seed permutation；
3. wrong-direction reaction；
4. mass-matched wrong transformation。

真实 graph 必须在不看 query truth 的条件下完成构建；truth 打开后只计算结果，不再改 event、门或权重。

### B47-M3：只比较两个最小模型

- **Positive-control baseline**：DreaMS + 冻结 catalogue/topology prior + B30-style abstention；
- **ECSR**：DreaMS + ion-family + exact event factors + global assignment + abstention。

不要在第一次 truth evaluation 前扫几十个模型。固定一个线性/广义加性 ECSR 与一个小型非线性 ECSR，所有超参数在外层 source/formula OOF 或 reference calibration 中完成。

### B47-M4：一次性揭盲门

ECSR 只有同时满足以下条件才可进入 context adapter：

- 相对裸 DreaMS Recall@1 `>= +3 pp`；
- 相对 catalogue-only positive control 的增量 CI 下界 `> 0`；
- formula-cluster 与 source-cluster CI 下界均 `> 0`；
- `corrected > 2 × introduced`；
- 至少 50 个 corrected truth identities；
- real event graph 胜过全部结构化 null；
- 在不同 candidate count、reference multiplicity、adduct/ion-family、source strata 不出现显著反向；
- 同时报告 Recall@1/5/10、MRR、macro query AUC、coverage/abstention 与校准误差。

若只胜过 DreaMS、不能胜过 catalogue-only，则它仍是目录先验；若胜过 catalogue-only 但不胜 null，则它仍是图结构/度数捷径；只有两者都过，才是可用于 embedding 的生物上下文动作。

## 五、什么暂时不做

1. 不再调 B44、B45、B46；它们已消耗或已给出冻结负结果。
2. 不把 B30 candidate identity history 蒸馏进 clean-spectrum encoder。
3. 不把反应邻居直接拉近；相邻代谢物是不同结构，必须由 typed transformation 条件化。
4. 不用 degree、membership、path count 冒充反应证据。
5. 不在 ion-family 和 event provenance 未完成前做多跳 diffusion/GNN。
6. 不用 outcome-aware action 或 query truth 选 seed、事件或阈值。
7. 不以“内部 +5.9 pp”承诺外部 +5 pp；它是强阳性对照与工程基线，不是外部可迁移下界。
8. 不把 selective coverage 的高准确率写成总体 embedding 提升。

## 六、创新性与论文边界

如果 ECSR 通过 B47 门，方法学创新可以严谨表述为：

> 在冻结谱学检索与样本级代谢特征之间建立候选特异、事件可追溯的结构化推断层；通过 ion-family 去冗余、typed biotransformation、局部谱—结构一致性、全局分配与校准拒答，区分可迁移的生物上下文证据和目录覆盖捷径；并仅将通过结构化空模型验证的 event signal 转移到 context-conditioned representation。

这与 MetDNA3/KGMN 的区别不应声称为“首次使用代谢网络”，而应落在：

- 以强谱图 foundation model 为 unary；
- 候选级 typed event 而非递归硬传播；
- 显式处理 ion family、候选冲突和 annotation sink；
- 以 coverage-neutral/null-controlled 协议识别捷径；
- 将通用 spectrum embedding 与样本条件 candidate representation 分开；
- 输出可校准候选集合和 abstention。

在 B47 揭盲前，最准确的论文状态仍是：

- B30/B37：内部开发上的强 catalogue-risk module；
- B44–B46：外部与机制负证据；
- B47-ECSR：前沿对齐、可证伪的下一代算法合同，尚无性能结果。

## 七、最高性价比执行顺序

1. **验证并封存服务器现有 B47 seed v1**，不重跑；获得 report、table、hash。
2. **做 reference multiplicity/seed calibration 审计**，成本低且可能直接消除 max-cosine 偏差。
3. **构建 ion-family graph**，这是 IIMN/KGMN/NetID 都表明不可跳过的物理层。
4. **构建 exact event table + 四类 null**，保持 truth blind。
5. **冻结 catalogue positive control 与 ECSR 两个模型**。
6. **一次性揭盲**；只有 ECSR 超过 catalogue-only 与全部 null 才开始上下文表示学习。

这条路线保留了已获得的 +5.9 pp 工程资产，同时停止把它误当反应机制；它也避免在尚无可迁移事件标签时再次浪费 GPU 训练 116M encoder。

