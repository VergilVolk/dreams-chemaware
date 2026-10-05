# URSEA 对抗性审稿审计与方法学重构

日期：2026-10-05  
状态：方法设计冻结前的拒稿级审计；不得把本文中的 proposed 方法写成已经验证的结果

## 0. 裁决

当前 p-value 版 URSEA **不足以稳定构成方法学创新**。它解决了真实问题，也比硬 argmax 严谨，但仍可被概括为：

> 用已有谱图模型产生跨队列相似度，用 target/decoy 或经验零分布变成匹配 p 值，再用 intersection–union、Bonferroni/BY 和既有 replicability 方法控制错误。

这些组件分别已有成熟先例。若论文的数学贡献只有 `max(p_match,p_assoc)` 和加权 Bonferroni，计算方法审稿人有充分理由评价为“正确但平凡”；组学审稿人则会追问为什么谱图匹配等同于同一分析物、为什么跨平台同患者算生物复现、以及没有标准品时到底得到了什么生物学结论。

建议保留现有 p-value 路线作为 **safe baseline**，把主方法重构为：

> **潜在实体对应下的联合复现 e-value 推断**：针对“发现关联成立、跨队列实体相同、验证关联同向成立”的复合命题构造候选对 e-value；再在未知候选对应上聚合为实体 e-value，并在任意实体相依下控制复现 FDR；同时对所有合理全局链接给出结论稳定性证书。

这不是发明 e-value、record linkage 或 partial conjunction，而是定义一个此前各方向都没有直接解决的推断对象：**当被检验的生物学对象本身需要通过有误差的 LC–MS/MS 链接才能确定时，如何对“同一未知分析物跨队列复现”做有限样本、候选多重性和跨实体相依同时受控的推断。**

是否最终成立，取决于严格基准能否证明：它相对硬链接和 p-value/BY 基线不仅更保守，而且在相同错误率下保留或提高净功效。

---

## 1. 严苛计算方法审稿人的拒稿意见

### 1.1 “不确定性感知”不是新方法

概率 record linkage 已长期传播链接不确定性；稳健匹配推断已经考虑所有“同样合理”的匹配方案；conformal link prediction 已产生覆盖真答案的候选集合。仅把链接概率乘入代谢组学分析，不足以支持首创性。

拒稿措辞可能是：

> The manuscript combines an off-the-shelf representation model with standard probabilistic linkage and multiple-testing corrections. The uncertainty-aware terminology overstates the methodological contribution.

### 1.2 现有定理过于接近加权 Bonferroni

`min_j p_j/w_j` 在权重和不超过 1、条件超均匀时有效，是标准并集界。真正困难的不是三行证明，而是现实中权重、匹配 p 值、候选生成和 DDA 缺失共同破坏条件超均匀。若论文把简单结论当主定理、把关键条件留给经验图，会显得避重就轻。

### 1.3 匹配 p 值的零分布未被识别

质谱 target–decoy 的核心难点不是“有多少 decoy”，而是错误 target 与 decoy 是否可交换。代谢物空间没有像蛋白质序列那样天然的反向 decoy；同分子式异构体、常见骨架、库内谱数和仪器域都会使错误 target 比人工 decoy 更像 query。已有研究明确显示谱图匹配 FDR 必须按项目调整，且 entrapment 可以揭露名义 FDR 失控。

因此，下列做法不能单独证明 `p^M` 有效：

- 随机打乱峰；
- 随机选质量窗口内分子；
- 将库外身份全部当同质负例；
- 只报告总体 reliability diagram 或 ECE；
- 在同一个数据集上训练打分器并估计 null tail。

### 1.4 候选生成失败被藏在条件分析里

如果真实对应根本不在 `J_i`，方法仍可能从大量错误候选中找到一个有关联的实体。主假设必须包含“无真对应”分支，性能必须以所有可测实体为分母，同时报告 candidate recall、abstention 和 no-match calibration。只在“真值已经进入候选集”的条件下报告准确率，会严重高估部署性能。

### 1.5 一对一图假设不符合 LC–MS/MS 观测机制

同一中性分子可产生多个加合物、同位素、原位碎裂离子和不同色谱峰；一个 DDA isolation window 也可产生嵌合谱。反过来，不同异构体可能无法被现有谱图区分。因此跨队列对应不是普通双射，而是带 no-match、many-to-one、one-to-many 和 chimeric 状态的部分对应图。若算法只做 Hungarian matching，数学整洁但科学对象错误。

### 1.6 BY 控制虽安全但可能把方法变成零发现机器

BY 在高度相关的大规模实体上通常保守。若完整方法只通过牺牲绝大多数功效换取控制，而 hard match 在经验上本来没有明显膨胀，则没有实用价值。必须同时展示错误率—功效前沿，而不是只报告“受控”。

### 1.7 现有 benchmark 不能直接支撑队列链接

GNPS identity-disjoint 10,995 查询是谱库检索真值，不是完整的跨队列 feature correspondence：它缺少 RT 漂移、MS1 缺失、ion-family、DDA 选择、基质效应和真实 no-match 比例。86,390 对真实跨仪器同分子谱适合训练或压力测试谱图不变性，但来自 `[M+H]+` 训练折，不能同时作为最终外部证明。

### 1.8 计算复杂度与可复算性尚未定义

候选图可能包含数十万边；如果再枚举匹配、置换表型、bootstrap 公式簇和多层校准，成本可能超过普通实验室能力。方法必须给出候选阻断复杂度、内存上界、确定性随机种子、缓存边界和单 GPU/CPU 运行时间，而不是只给概念流程。

---

## 2. 严苛组学与生物学审稿人的拒稿意见

### 2.1 “稳定谱图实体”可能只是稳定污染物或观测伪影

跨队列重复出现不自动等于内源代谢物。稳定实体也可能是：

- 溶剂、塑化剂、柱流失物或试剂背景；
- 常见药物、饮食或环境暴露；
- 高丰度物质的加合物、同位素或源内碎片；
- 共洗脱嵌合谱；
- 批次或采集顺序驱动的信号。

必须在统计复现之前完成 blank、dilution、pooled-QC、批次、注射顺序、峰形和离子家族审计。否则“跨队列复现”只是重复的技术背景。

### 2.2 同患者跨平台不是独立生物学复现

LCNEC 的方向 12/12 和效应排序相关性是很强的技术一致性证据，但相同患者共享真实生物效应和所有患者级混杂，不能证明跨人群复现。它可以验证链接层和平台稳健性，不能成为 replicability 主终点。

### 2.3 精确结构没有被验证

谱图实体复现允许声明“同一个或不可区分的一组分析物与表型相关”，不能声明具体异构体、通路通量或酶活。ChemAware/V2/P2b 的一致候选仍是模型证据；独立蛋白组和转录组只是机制背景。论文必须把 entity、chemical family、exact structure 三层永久分开。

### 2.4 独立队列可能并不具有可比表型

疾病名称相同不等于可直接复现：组织部位、分期、治疗、吸烟、空腹状态、采样时间、保存方法和平台覆盖均可能不同。必须预先定义可迁移 estimand，而不是事后找一个方向相同的队列。

### 2.5 DDA 缺失与丰度相关

差异最强的峰更容易触发 MS/MS，因此“有谱图并可链接”的资格本身可能受表型影响。pooled-QC/迭代 DDA 是正式确认层；研究样本 DDA 只能进入选择模型充分审计后的探索层。仅置换病例标签未必恢复采集机制下的交换性。

### 2.6 生物学收益必须超过“不知道名字也能做差异分析”

如果方法最后只报告若干未知 feature 在两个队列同向变化，审稿人会问：普通 m/z/RT alignment 或 PAIRUP-MS 是否已经能做到？真正的生物学增益必须是至少一项：

- 在 m/z/RT 无法区分的密集候选中，MS/MS 证据阻止错误复现；
- 将不同平台上原本无法对齐的未知实体纳入正式复现；
- 对 replicated entity 给出稳定的化学类别或诊断碎片解释；
- 明确弃权，从而避免错误机制故事。

---

## 3. 通用审稿人的拒稿意见

### 3.1 论文可能同时讲太多故事

Noise、P2b、ChemAware、V2、BioAware、LCNEC、MTBLS13729、跨队列链接和复现统计若全部并列，主贡献会消失。每个组件都“有一点用”不是系统方法。主文必须只围绕一个可证伪问题：**链接不确定时，跨队列复现结论能否仍然可靠且有功效。**

### 3.2 “新范式”“定律”“1+1>10”容易被认为营销化

在预注册外部实验完成前，只能称 proposed framework 和 prespecified risk surface，不能称定律或范式。所谓 forced-linkage error law 必须先证明跨数据集形状稳定，并提供置信区间；否则只是给一张经验曲线起了大名字。

### 3.3 复杂方法必须有简单失败示例

论文需要一个读者一眼看懂的反例：两个质量相同的异构体，硬 argmax 在验证队列选中了具有关联但身份错误的候选；普通 association replication 显著，而联合证据方法拒绝或弃权。没有这个示例，理论与生物学问题无法连接。

### 3.4 负结果和停止条件必须写在主文

BioAware B44–B46 的负结果、LCNEC 没有独立代谢组复现、标准品未完成、以及 86,390 对资产属于训练折，都不能藏在补充材料。诚实边界本身能提高可信度。

---

## 4. 与相邻工作的真实边界

| 方向 | 已有能力 | 本项目不能声称 | 仍可能留下的空位 |
|---|---|---|---|
| PAIRUP-MS | 用 m/z 和相关/插补结构匹配未知信号并增加跨数据集关联 | 首次跨队列研究未知峰 | 没有以 MS/MS 身份证据和复现证据构造同一复合零假设 |
| Feature correspondence | 用 m/z、RT、强度寻找跨 LC–MS 数据集对应，并用多种正交策略验证 | 首次做 feature matching | 不提供“身份正确且生物学复现”的联合错误保证 |
| 谱图 target–decoy/FDR | 对谱库匹配估计项目特异 FDR | 首次给谱图相似度统计显著性 | 不处理跨队列表型复现与候选对应不确定性 |
| Probabilistic record linkage | 对链接后回归传播链接不确定性 | 首次传播链接误差 | 未针对 LC–MS/MS 复合身份—关联命题、DDA 缺失和谱图诱饵建立有限样本程序 |
| Robust matching inference | 在所有近似等价匹配上优化最坏/最好检验结果 | 首次考虑匹配选择敏感性 | 不是跨队列未知分析物复现，也没有质谱校准层 |
| Conformal link prediction | 输出覆盖真答案的候选集合 | 首次产生受控候选集 | 通常控制 coverage，不直接控制复现发现中的错误比例 |
| Partial conjunction / replicability r-values | 控制同一、已知 hypothesis 在多研究中的复现错误 | 首次做组学复现统计 | 默认各研究的 hypothesis 身份已经对齐 |
| p-filter | 同时控制 feature 和 group 等多层 FDR | 首次多层 FDR | 分组通常已知，不处理潜在实体对应图 |
| e-BH | 在任意 e-value 相依下控制 FDR | 首次在任意依赖下控 FDR | 尚未针对未知 LC–MS/MS 对应构造身份—发现—验证联合 e-value |

关键文献：

- PAIRUP-MS: https://doi.org/10.1371/journal.pcbi.1006734
- Feature correspondence: https://doi.org/10.1021/acs.analchem.1c03592
- Metabolomics spectral-match FDR: https://doi.org/10.1038/s41467-017-01318-5
- Robust matching inference: https://doi.org/10.1287/ijds.2022.0020
- Replicability review: https://doi.org/10.1214/23-STS892
- p-filter: https://doi.org/10.1111/rssb.12218
- e-BH: https://doi.org/10.1111/rssb.12489
- Conformal link-prediction example: https://aclanthology.org/2025.naacl-long.32/
- MS entrapment assessment: https://doi.org/10.1038/s41592-025-02719-x

这一文献边界意味着：创新不能落在任何单独组件上，只能落在新的复合推断对象与相应证据构造上。

---

## 5. 建议的主方法：潜在实体对应下的联合复现 e-value

### 5.1 被检验的命题

对发现队列实体 `i` 和验证队列候选 `j`，定义三个命题：

- `A^D_i`：实体 `i` 在发现队列与表型关联；
- `M_ij`：`i` 与 `j` 是同一中性分析物，或在预先定义的分辨率下不可区分的同一谱图实体；
- `A^V_{j,d}`：`j` 在验证队列沿发现队列冻结方向 `d` 关联。

候选对的目标备择是假设三者同时成立：

\[
H_{1,ij}=A^D_i\cap M_{ij}\cap A^V_{j,d}.
\]

因此零假设是并集：

\[
H_{0,ij}=\neg A^D_i\cup\neg M_{ij}\cup\neg A^V_{j,d}.
\]

发现实体层面的零假设为所有候选对均不成立：

\[
H_{0,i}=\bigcap_{j\in J_i}H_{0,ij}.
\]

这一定义显式包含 no-match、错误身份、发现假阳性和验证失败四类情况。

### 5.2 三类基础 e-value

分别构造：

- `E^D_i`：发现关联 e-value；
- `E^M_ij`：同一实体的匹配 e-value；
- `E^V_ij`：验证队列同向关联 e-value。

e-value 的要求是在对应零假设下期望不超过 1，而不是“越接近 0 越显著”。基础实现可先从有效 p 值通过预先冻结的 p-to-e calibrator 得到，例如

\[
E=(1-\kappa)p^{-\kappa},\qquad 0<\kappa<1,
\]

其中 `κ` 必须在外部开发集冻结。更有功效的版本可以使用独立样本上的混合似然比，但不能在测试集上调备择分布。

### 5.3 候选对联合证据

定义

\[
E^{pair}_{ij}=\min(E^D_i,E^M_{ij},E^V_{ij}).
\]

**定理 1。** 在 `H_{0,ij}` 下，至少一个子零假设成立。若对应基础 e-value 有效，则

\[
\mathbb E_0[E^{pair}_{ij}]
\le \mathbb E_0[E^{null}_{ij}]\le1.
\]

因此 `E^{pair}_{ij}` 是“发现关联、身份相同、验证同向关联”交命题的有效 e-value；不需要三路证据相互独立。

这一步比 `max(p^D,p^M,p^V)` 的 p-value IUT 本身并不更“原创”，但为下一步候选聚合和任意依赖 FDR 奠定了合法证据代数。

### 5.4 未知候选对应的实体证据

选择在零假设下与候选对证据独立，或满足条件 e-validity 的非负权重 `a_ij`，且 `sum_j a_ij<=1`。最安全主分析使用 `a_ij=1/|J_i|`。定义

\[
E^{entity}_i=\sum_{j\in J_i}a_{ij}E^{pair}_{ij}.
\]

**定理 2。** 在 `H_{0,i}` 下，每个候选对零假设都成立，因此

\[
\mathbb E_0[E^{entity}_i]
\le\sum_ja_{ij}\le1.
\]

故它是“至少存在一个正确链接且关联复现候选”的有效实体 e-value。候选间可以任意相关。

注意：不能让 `a_ij` 直接追随同一个 `E^M_ij`，否则可能把全部权重交给最大的零假设证据。若要使用谱图概率分配权重，必须通过独立拆分/外部数据证明条件 e-validity，或把整个自适应聚合器在实体级 entrapment 上重新校准为 e-value。

### 5.5 跨实体任意依赖下的复现 FDR

对全部预先定义的发现实体应用 e-BH。e-BH 已知在 e-values 任意依赖下无需 BY 的调和修正即可控制 FDR。共享碎片、共洗脱、同一 ion family 和共同患者引起的跨实体相关，因此不再迫使主分析使用极保守 BY。

这一点构成方案相对当前 p-value 版最实质的算法收益：

- 复合零假设通过 `min` 证据构造处理；
- 候选对应不确定性通过 e-value mixture 处理；
- 跨实体依赖通过 e-BH 处理；
- 三层均保留有限样本、无独立性假设的清晰边界。

### 5.6 关键算法补强：交叉拟合谱图下注

如果主方法永远使用等权，虽然安全，但可能因候选数多而失去全部功效；如果直接让权重追随 Noise/P2b 链接分数，又会产生权重—匹配证据依赖。利用项目现有的多条件实测谱、技术重复和 pooled-QC，可以构造 **cross-fitted spectral betting**：

1. 对具有多张独立 MS/MS 扫描的实体，按采集事件而不是按峰随机拆成 allocation view `A_r` 和 evidence view `B_r`；禁止把同一张谱的峰拆成两半冒充独立。
2. 只用 `A_r` 的 official DreaMS、Noise、P2b 和质量信息生成候选权重 `a^{(r)}_{ij}`，并归一化使权重和不超过 1。
3. 只用 `B_r` 和独立 entrapment/calibration 构造 `E^{M,(r)}_{ij}`；关联 e-values 始终来自冻结后的发现/验证表型模型。
4. 构造

\[
E^{entity,(r)}_i=\sum_j a^{(r)}_{ij}
\min(E^D_i,E^{M,(r)}_{ij},E^V_{ij}).
\]

5. 交换扫描分割并得到多个 split-specific entity e-values，最后取算术平均：

\[
\bar E^{entity}_i=\frac1R\sum_{r=1}^{R}E^{entity,(r)}_i.
\]

**定理 3。** 若 evidence view 的匹配 e-value 给定 allocation view 条件有效，则每个 `E^{entity,(r)}` 的零假设期望不超过 1；算术平均仍是 e-value，因为期望具有线性性，不要求不同 split 相互独立。

这一步带来三个项目特异的贡献：

- 合法使用谱图模型进行自适应候选资源分配，而不是把模型概率误当检验概率；
- 多条件实测谱同时贡献跨条件稳健性和推断有效性；
- 自动暴露单谱实体的证据不足：只有一张谱时使用等权安全路线，或按预注册质量门弃权，绝不通过峰级伪拆分制造功效。

必须设置 leakage audit：同一原始扫描、同一 consensus 谱的复制、同一 collision-energy 聚合谱不得跨 allocation/evidence；库内参考谱也要按 molecule identity 分组拆分。若实际重复谱数量不足以支持扫描级拆分，本算法降为次级方法，不得用相关峰拆分替代。

### 5.7 近似有效性传播

若某一基础 e-value 在其零假设下仅满足

\[
\mathbb E_0[E]\le1+\epsilon,
\]

则 pair 和 entity e-value 的最坏膨胀同样不超过相应 `1+epsilon_max` 上界。主分析可以把所有 entity e-value 除以预先估计的安全因子 `c>=1+epsilon_max` 后进入 e-BH。该安全因子必须来自独立 entrapment/置换层的上置信界，而非测试数据上的平均 ECE。

---

## 6. 第二项方法学补强：全局链接稳定性证书

e-value 主方法回答平均错误控制，但仍可能出现某一个结论对匹配算法任意细节极敏感。为此增加一个不替代主检验的 **linkage stability certificate**。

### 6.1 合理链接集合

构造满足以下约束的全局部分匹配集合 `C_tau`：

- 中性质量、离子模式和 RT 映射在冻结容差内；
- ion-family 允许多离子对应一个中性实体；
- 显式允许 no-match；
- 嵌合谱可标记为 ambiguous，不强制一对一；
- 总链接代价不超过最优值加 `tau`，或每条边进入 conformal/entrapment 合格集合。

### 6.2 结论稳定性

对每个通过 e-BH 的实体，计算在所有 `Pi in C_tau` 中：

- 是否始终存在同向验证候选；
- 效应方向是否稳定；
- 最差验证效应和最大 p/r-value；
- 候选结构类别是否稳定。

由此分为：

1. **certified**：所有合理链接均支持同向复现；
2. **link-sensitive**：至少一个合理链接改变结论；
3. **unresolved/no-match**：合理集合包含无链接状态。

计算上使用 min-cost flow/MILP 求最坏情况，不枚举全部匹配。稳健匹配推断已有先例，因此创新不应写成“首次考虑匹配不确定性”，而应写成其在 LC–MS/MS 潜在实体复现中的专门化和与联合 e-value FDR 的结合。

---

## 7. 匹配证据必须如何校准

### 7.1 分数输入

主链接分数只允许使用表型盲、跨队列均可观测的信息：

- neutral mass / adduct compatibility；
- Noise 跨仪器谱图表示；
- official DreaMS 作为冻结基线；
- P2b 的局部碎片、中性丢失和质量差证据；
- 峰数、解释强度、前体残留、chimericity 和 OOD；
- 表型盲 RT anchor mapping。

ChemAware、V2 和 conditional-null energy 首先属于结构候选解释层。只有在独立 benchmark 证明它们能够判断“两个实测谱是否来自同一分析物”，且不借助共享数据库候选造成循环证据后，才可作为链接消融；否则不得进入 `E^M`。

BioAware 在 B47 prospective gate 通过前完全不进入匹配或复现统计。

### 7.2 三层负对照

`E^M` 不能依赖单一 decoy。至少使用：

1. **matched wrong-identity decoys**：相同质量区间、模式、峰数、仪器域和库覆盖；
2. **hard structural entrapments**：同分子式/近结构但明确不同身份；
3. **cohort-native shifted links**：在真实候选图内保持度数和质量分布的错配。

校准器必须在 identity/formula/source 分组隔离下训练，在目标项目用 pooled-QC/技术重复正锚与 cohort-native entrapment 做表型盲再校准。若三类 null tail 不一致，取最坏层或弃权，不能平均掩盖失效。

### 7.3 OOD 与 no-match

每个实体必须允许 `J_i` 为空或全部弃权。正式报告：

- candidate recall；
- conformal/entrapment coverage；
- no-match rate；
- abstention rate；
- 在有真匹配和无真匹配两种分母下的错误率。

---

## 8. 真正的 money figure：强制链接风险面，而不是一条夸张曲线

“强制链接制造假复现”在验证前只能称 **forced-linkage risk surface**，不称定律。预先定义：

\[
FLR(\tau,m,d,q)=
P(\text{wrong identity and declared replication}
\mid \tau,m,domain=d,quality=q).
\]

主图建议四个面板：

### A. 身份风险面

在 GNPS identity-disjoint 10,995 次检索和独立保留的跨仪器重复谱上，展示阈值、候选数、质量和域漂移对 wrong-link rate 的影响。86,390 对训练折只能用于开发或内部压力测试，最终图必须另有未消费身份/分子式/来源隔离面板。

### B. Null calibration

在 cohort-native entrapment 和表型置换下，画名义 FDR 与实测 FDP/FDR。比较：

- m/z/RT hard argmax；
- PAIRUP-MS 式点匹配；
- DreaMS/Noise argmax；
- p-value IUT + Bonferroni/BY；
- proposed entity e-value + e-BH。

### C. 错误率—功效前沿

在保留真实候选图、相关结构、DDA 缺失和 effect heterogeneity 的半合成数据中注入发现/验证效应，横轴错误复现率，纵轴正确复现实体数。若 proposed 方法只向左移动但严重向下坠落，不构成成功。

### D. 真实技术与生物学覆盖

LCNEC 同患者跨平台只展示技术链接稳定性；独立受试者、相同 estimand 的队列才展示生物复现。没有后者时，论文只能作方法与技术验证稿，不能以疾病复现为主标题。

核心指标不是 AUROC，而是：

- false replicated links per 1,000 tested entities；
- empirical FDR/FDP upper confidence bound；
- true replicated entities at fixed 5% FDR；
- abstention/no-match；
- formula/identity cluster bootstrap CI；
- 运行时间与显存/内存。

---

## 9. 仓库各组件的正确位置

| 组件 | 主方法角色 | 禁止用途 |
|---|---|---|
| official DreaMS | 冻结谱图表示基线 | 与不同协议数字拼接 |
| Noise Stage-1/E6 | 跨仪器同实体链接分数候选 | 用训练折 86,390 对当最终外部证明 |
| Noise 两个重排器 | 作为匹配/检索消融，检查是否改善同一实体证据 | 未校准就直接当概率 |
| P2b | 提供局部碎片、中性丢失、峰间质量差证据 | 把局部规则命中当独立结构确认 |
| ChemAware / V2 / conditional-null energy | replicated entity 之后的结构候选排序与类别稳定性 | 通过共享候选数据库反向定义跨队列身份 |
| BioAware | B47 通过后作为 prospective context；当前仅解释层/负对照 | 把 B42/B43 开发增益写成普适样本生化证据 |
| LCNEC | 同患者跨平台技术复现、真实缺失与质量压力测试 | 独立生物学复现、Level-1 身份或通量 |
| MTBLS13729/8090 | 迁移边界和阴性/反向案例 | 与 LCNEC 拼成同疾病复现 |

“所有组件都用上”不等于所有组件进入同一个分数。最强整合是证据分工和防止循环：链接组件负责实体对应，联合统计负责复现，结构重排器负责复现后的解释，BioAware 负责通过外部门后的样本上下文。

---

## 10. 必须完成的 benchmark 合同

### Benchmark A：基础证据有效性

- discovery/validation association p-to-e 在置换下均值不超过 1；
- match p/e 在三类 entrapment、identity/formula/source 隔离下有效；
- 报告下尾或上尾置信界，不只报 ECE；
- 校准、模型选择和最终测试身份完全隔离。

### Benchmark B：候选与图结构

- 显式 no-match；
- many-to-one ion family；
- chimeric spectrum；
- 候选数和质量分层；
- candidate recall 与错误率同分母报告。

### Benchmark C：联合推断

- 三类零假设分别模拟：发现无关联、身份错误、验证无关联；
- 两类组合零假设：身份错误但另一个实体真有关联；身份正确但方向不复现；
- 任意实体相关下 e-BH FDR；
- 与 p-IUT + BY、Holm、普通 BH 和置换 FDR 比较。

### Benchmark D：真实迁移

- LCNEC 技术复现；
- 至少一个独立受试者、相同样本类型和兼容表型的生物学队列；
- pooled-QC/technical-repeat anchor recalibration；
- m/z/RT-only、PAIRUP-MS、feature correspondence、official DreaMS、Noise、Noise+P2b 和完整方法公平同分母比较。

### Benchmark E：解释层

- 只对通过复现门的实体运行结构融合；
- 报告 family-level 稳定性和 exact-structure 不确定性；
- 用来源已知正对照检查方法不会把同分子式异构体一致误注释；
- BioAware 不得改变正式复现发现集。

---

## 11. 论文级成功与停止条件

### 必须同时满足

1. entity e-value 在全部预注册 null 层的经验期望上置信界不超过安全因子；
2. e-BH 在相关、缺失和 no-match 条件下控制实体复现 FDR；
3. 相同经验错误率下，完整方法相对 hard Noise argmax 和 p-IUT/BY 的正确复现实体数 CI 下界大于 0；
4. 增益不能由更宽质量窗口、更多候选或数据库覆盖解释；
5. LCNEC 技术复现增益超过 m/z/RT-only；
6. 至少一个独立生物学队列提供同 estimand 复现，或主动把文章降级为方法/技术验证稿；
7. exact structure 与 entity replication 永久分栏。

### 任一出现即停止主张

- entrapment 显示 match e-value 失控；
- 只有训练折或谱库检索阳性，队列链接无增益；
- e-BH 相对 BY 没有实际功效优势；
- 增益只来自含真值的数据依赖权重；
- 真实数据阳性仅存在于同患者跨平台；
- 只有候选结构故事，没有受控的实体复现；
- BioAware 未过 B47 却进入主统计；
- 标准品缺失却写成精确结构或通路机制。

---

## 12. 当前最值得推进的最小版本

不要立刻重写所有流水线。先做一个能杀死或保住核心方法的最小实验：

1. 冻结 GNPS identity-disjoint 与一个未消费跨仪器测试层；
2. 为 official DreaMS、Noise、Noise+P2b 产生同一候选图；
3. 构造三类 match null/entrapment，并得到有效 `p^M`；
4. 用固定 `κ` 把 `p^M,p^D,p^V` 转为基础 e-values；
5. 实现 `min` pair e-value、等权 entity e-value 和 e-BH；
6. 在半合成表型上同时测 identity error、replication FDR 和 power；
7. 与 hard argmax、现有 p-IUT/BY 比较。

这一轮不需要标准品、不需要新多模态模型，也不需要重新训练全部 encoder。若它不能在同一错误率下胜过保守 p-value 基线，整个“方法学新范式”应停止；若它通过，再进入队列锚点重校准和 LCNEC/独立队列验证。

## 13. 可发表主张的最终收口

通过全部门后，可以主张：

> We introduce a finite-sample framework for replicability analysis when the tested molecular entities are not pre-aligned across untargeted LC–MS/MS studies. The method constructs conjunction e-values for discovery association, spectral identity and directional validation, aggregates over ambiguous candidate correspondences, and controls the false discovery rate of replicated spectral entities under arbitrary cross-feature dependence.

不能主张：

- 首次分析未知代谢物；
- 首次传播链接不确定性；
- 首次做跨队列 feature matching；
- 无标准品确认精确结构；
- 当前已经建立新范式。

真正的新意是：**把潜在实体对应本身纳入复现假设，并给出从候选对、候选集合到全实体发现集的统一证据代数和错误保证。**
