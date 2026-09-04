# BioAware 策略严格审计与重构方案

**审计日期：2026-08-30**  
**审计对象：** BioAware v1/v2、MetDNA3 对齐数据层、candidate evidence ledger、G3-v1/v2/v3 融合器、MTBLS13729 全局峰图与生物学闭环。  
**目的：** 区分真实进展、实现漏洞、科学假设错配和下一代可发表算法，不再用 headroom 或少量开发修正替代真实性能。

---

## 1. 最终裁决

BioAware 当前**还不是完善算法，也没有形成可主张的性能提升**。它已经建立了有价值的数据资产、负结果体系、候选证据账本、表型盲峰图和谨慎弃权框架，但核心排序器仍存在两类根本问题：

1. **工程逻辑不闭合：** G3-v2 把权重为零的证据族计入 gate 支持数；
2. **统计独立性不成立：** reported-reaction、reported-plus-predicted-reaction 和 SMN 高度相关，其中前两者还是嵌套关系，却被当成多张独立票。

因此，原 `2修正/0新增、+1.71 pp` 只能保留为历史开发输出：

- 修复“零权重仍计票”后：`1修正/0新增，+0.85 pp`；
- 再把嵌套/相关的网络证据合并为一个网络族后：`0修正/0新增`。

这不是说 BioAware 思想无效，而是说明当前正结果主要来自 gate 设计漏洞和相关证据重复计数，尚未证明生物网络提供了可迁移的独立注释信息。

---

## 2. 当前 BioAware 实际在做什么

当前系统是一个 **DreaMS embedding 之后的候选重排专家**，不是新 embedding，也不改变 DreaMS 权重。其推理链为：

1. 用官方 DreaMS 对 query 与同分子式/质量候选的参考谱打分；
2. 为每个候选附加 decoder、规则、反应网络、结构网络和 RT 证据；
3. 在每个 query 的候选组内把各证据 min-max 到 `[0,1]`；
4. 学习非负线性权重；
5. 当 DreaMS margin 小、融合优势足够且“支持证据族”达到阈值时覆盖 Top-1，否则弃权。

这个框架的优点是可解释、可以弃权、没有使用表型标签，也没有把 P2b 混入 BioAware。但它仍然是**逐 query 的局部线性重排**，不是 NetID 意义上的全局网络优化，也不是 MetDNA3 意义上的数据层—知识层递归传播。

---

## 3. 已确认的有效资产

以下内容应保留，不因当前融合失败而否定：

### 3.1 数据与评价纪律

- 117-query HILIC 开发任务具有91个身份、88个分子式和466个候选；
- 官方 DreaMS 基线固定为 `95/117=0.8120`，22个 Top-1 错误；
- formula-group nested OOF、truth identity leave-out、RP 封存、P2b 禁用、表型禁用均是正确纪律；
- 负结果和工件哈希完整保留，避免反复发明同一失败策略。

### 3.2 失败分解

22个官方错误中：

- 11个 query 的真值不在当前 Rhea 图中；
- 只有9个存在真值网络路径；
- 6个具有 paired raw-MS2 证据；
- 5个 raw-MS2/网络证据偏向错误候选或打平。

这说明覆盖率、错误传播和候选特异性确实是核心问题，而不是单纯权重没调好。

### 3.3 生物学基础设施

- MS1 全局峰图、离子家族、加合物/同位素去冗余具有直接应用价值；
- MTBLS13729 的修饰鸟苷样模块、长链酰基肉碱类别和多胺乙酰化轴来自冻结注释后的患者内定量，不依赖 BioAware 排名成功；
- BioAware 目前对生物学部分的最可靠贡献是**去重、证据分层和冲突保留**，不是身份升级。

---

## 4. 关键工程问题

### 4.1 零权重证据仍然为 gate 开门

`develop_bioaware_rank_consensus_fusion.py` 中，融合权重由训练折拟合；但 `support_count` 只检查“proposed 候选的 family feature 是否高于 baseline”，没有检查该 family 的权重是否大于零。

后果：

- 两个原始修正案例都把 `decoder` 计为支持，但 decoder 在对应折的权重均为 `0`；
- 第一个修正还把权重为 `0` 的 known-reaction 计为支持；
- 该 query 的 gate 要求至少3族支持，真正有非零贡献的只有 predicted-reaction 与 RT 两族，因此不应干预。

严格重放后，原2个修正只剩1个。

**必须修复：** 支持族应按 `weight × candidate_delta` 的实际贡献计数，并设最小 log-likelihood/score contribution，而不是只看原始方向。

### 4.2 嵌套网络证据被重复计票

candidate ledger 中：

- `known_reaction` 来自 reported reactions；
- `predicted_reaction` 来自 reported + predicted reactions；
- 后者包含前者，不是独立证据。

候选级相关性实测：

- known vs predicted：约 `0.95`；
- known vs SMN：约 `0.64`；
- predicted vs SMN：约 `0.63`。

因此“known + predicted + SMN 三票一致”不能解释为三种生物证据共识。合并为一个 network family 后，当前 OOF 不再产生净修正。

**必须修复：** predicted evidence 只能建模为相对 known network 的**增量残差**，或者全部置于一个共享 network latent factor 中。

### 4.3 候选组内 min-max 会放大极弱差异

每个 query 内将一个 family 的最小候选映射为0、最大候选映射为1。两个候选即使只差极小数值，也会变成完整的 `0 vs 1` 投票。

这会丢失：

- 证据绝对强度；
- 测量不确定度；
- 缺失证据与真实负证据的差别；
- 不同 query 之间的可比性。

**必须修复：** 每类证据应在外部训练数据上校准成 likelihood ratio、posterior odds 或分位数可靠度，不能在测试 query 内现算极差归一化。

### 4.4 family 内 `max` 是未经校准的 OR 运算

规则 family 取 `rule_jaccard_idf` 与 `sparse_rule_overlap` 的归一化最大值。候选只要在任一噪声指标上偶然高，就获得整族高票。网络类指标也存在相似的“最优路径”偏差。

**必须修复：** 多测量应通过预训练校准器、显式缺失模型或保守下界组合；不能用测试候选中的最大值代替证据概率。

### 4.5 缺失、图外和反证被统一填为0

以下状态具有完全不同语义：

- 候选不在 Rhea/HMDB；
- 有候选节点但无合格 seed；
- 有反应边但实验 feature 未观测；
- 有实验邻居但 MS2 不支持；
- 明确存在反向证据。

当前 ledger 主要将它们压成0。模型因此无法区分“未知”与“不支持”。

**必须修复：** 每个 evidence family 至少拆成 `available / support / conflict / uncertainty` 四个量。

### 4.6 非负线性加和不能表达条件依赖

反应网络只有在 seed 可靠、必要参与物可观测、候选质量兼容且 data-layer edge 成立时才应生效。当前线性模型默认各证据可独立相加，不能表达：

- `network × seed confidence`；
- `reaction completeness × observed neighbor`；
- `RT × chemical class/adduct`；
- `rule evidence × collision energy/instrument`；
- 网络冲突导致的否决。

这不是增加隐藏层就能自动解决的问题，首先需要正确的因子结构。

---

## 5. 当前策略逐项裁决

| 策略 | 当前结果 | 审计判断 | 后续处理 |
|---|---:|---|---|
| 一跳 Rhea 加分 | 21例中0修正/1新增 | 反应不完整、currency/高阶节点、共同底物缺失 | 终止直接覆盖；仅保留 typed hyperedge |
| expanded Rhea | 1/1 | 覆盖增加同时错误传播 | 不再以扩图数量为目标 |
| two-layer feature graph | 0/0 | 安全但覆盖不足 | 保留数据层，重建跨层因子 |
| reported raw-MS2 path | 小量正向 | 有方向但覆盖窄 | 保留为 network factor 的观测组成 |
| predicted MRN | 5修正/6新增 | 非独立、假边风险高 | 只能作为 residual/低权重候选生成 |
| SMN | 4修正/8新增 | 密图降低特异性 | 禁止单独覆盖；只做候选生成或类级先验 |
| RT | 0修正/3新增 | hard-error 子集上极弱 | 改为 adduct/class 条件化似然；缺失时弃权 |
| embedding→Morgan decoder | 8修正/46新增，权重全折为0 | 有结构 headroom，但校准与泛化不足 | 不作为投票；重做概率指纹/候选似然 |
| mass-rule overlap | 5–6修正/40–54新增 | 重编码谱峰，非唯一结构证据 | 只作候选特异碎裂似然的一部分 |
| G3-v2 rank consensus | 原2/0；严格审计后最多1/0；依赖去重后0/0 | 当前性能主张不成立 | 归档并修复，不开封 RP |
| G3-v3 peer context | 0/0 | transductive 且不是真正全局分配 | 停止同公式 peer 加分 |

---

## 6. 评价协议仍不充分

### 6.1 任务规模太小

117个 query 只有22个错误。一个 query 就是 `0.85 pp`。原 `+1.71 pp` 实际只对应2个修正，公式簇 CI 下界为0。

10个额外公式拆分种子的诊断中，原逻辑始终修正2个，但有4个种子会新增1个错误。说明方向不是完全随机，却无法证明安全性稳定。

### 6.2 仅 formula split 不等于结构和数据集外推

formula isolation 可以阻断相同分子式泄漏，但不能阻断：

- 相似 Murcko scaffold；
- 同一化学类别；
- 同一实验数据集/仪器；
- 同一参考库覆盖模式。

下一版至少需要 formula、scaffold、dataset/instrument 三种分组评估。

### 6.3 baseline 使用每身份最大参考谱相似度

候选分数取该身份所有参考谱的 maximum cosine。候选拥有的参考谱数量从1到数百不等，极值统计会带来库大小偏差。该协议可以模拟“有多谱参考库的检索”，但不能直接代表单谱身份判别，也必须报告按参考谱数量分层的表现。

### 6.4 缺少注释率—FDR 主指标

BioAware 的价值本应是“在可控错误率下扩大可注释覆盖”，而当前主要优化 Recall@1。下一版必须报告：

- coverage-risk/selective accuracy 曲线；
- target-decoy estimated FDR；
- 在固定1%、5%、10% FDR下的可注释数；
- calibration、Brier/ECE；
- corrected/introduced 及其化学类别分层。

---

## 7. MTBLS13729 生物学侧的潜在循环性

“phenotype labels not used”是必要条件，但不是完全独立。全局峰图使用全部患者的丰度相关性；即使不读取 tumor/normal 标签，强烈的组织差异本身也可能驱动两峰相关，继而影响离子家族合并，再用于报告同一批患者的差异。

当前全局峰图还有两个具体风险：

1. union-family 使用 `accepted` 全样本相关边，而不是要求 `split_replicated`；
2. 通过传递闭包合并家族，少量错误边可能串联多个真实分子。

因此“表型盲”不能等同于“无循环”。

**必须补充：**

- 对患者内 tumor/normal 均值或配对效应做残差化后再计算相关；
- leave-patient-pair-out 与患者半拆分重复建图；
- ion family 只用质量差、同位素包络、色谱峰形和加合物化学一致性作为强边；
- across-sample abundance 只作辅助，不能单独触发合并；
- 对每个最终家族报告所有边，不允许无审计的长链传递合并。

这不会推翻现有修饰鸟苷和酰基肉碱结果，但会决定“6峰折叠为5家族”等去重结论是否足够稳固。

---

## 8. 下一代 BioAware 应该是什么

### 8.1 从“多票加分器”改为“有未知状态的概率因子图”

对每个实验 feature `i` 建立候选身份变量 `y_i`，候选中必须包含 `unknown/abstain`。主要因子为：

1. **谱学 unary：** DreaMS 与候选参考谱的校准 likelihood；
2. **结构 unary：** 概率分子指纹/候选碎裂似然；
3. **MS1 unary：** 精确质量、同位素、adduct、RT/CCS 条件似然；
4. **ion-family factor：** 同一中性分子的多离子形态一致性；
5. **reaction hyperedge factor：** 只有在反应类型、方向、必要参与物和实验 feature 映射共同成立时激活；
6. **data-layer edge factor：** MS2 transformation、共洗脱和残差化 abundance consistency；
7. **conflict factor：** 明确惩罚网络、RT、碎裂之间的冲突。

所有因子输出可校准的 log-likelihood ratio，而不是候选组内0/1票。

### 8.2 真正的全局推断

首版优先使用可审计的 ILP/因子图，而不是立即上 GNN：

- 每个 feature 最多选择一个候选或 unknown；
- 同一 ion family 的成员必须映射为同一中性分子及兼容 adduct；
- reaction edge 只有两端候选与观测 transformation 同时一致时得分；
- 重复峰、同位素、源内碎片不能重复计为独立代谢物；
- 网络先验不能单独把低谱学候选升级为 Level 2。

当这一版稳定后，GNN 只用于学习 factor strength 或候选生成，不负责无约束地传播身份。

### 8.3 证据依赖建模

建议将证据分为三个互相较独立的大组：

- `spectral/fragmentation`：DreaMS、规则、指纹 decoder；
- `physicochemical`：m/z、isotope、adduct、RT、CCS；
- `biological-context`：MRN hyperedge、observed feature neighbors、tissue prior。

组内先建联合校准器，组间再进入因子图，禁止把同组衍生指标伪装成多票共识。

---

## 9. 紧急执行顺序

### P0：修复现有结论，暂停开封测试集

1. 修复 gate：只有正的实际加权贡献才能计入支持；
2. `predicted = reported + predicted` 改成 `predicted_increment`，不得与 reported 重复投票；
3. known/predicted/SMN 合并或用共享 latent network factor；
4. 在同一117-query数据上只做审计重放，不再声称性能；
5. RP 与外部 benchmark 继续封存。

### P1：建立外部校准证据

1. 对 DreaMS margin 按候选数、参考谱数、仪器和化学类别校准；
2. decoder 改为 probabilistic fingerprint likelihood，并做 scaffold/dataset isolation；
3. RT 改为 class/adduct/domain-conditional prediction interval；
4. 规则证据要求候选结构可产生该碎片/中性丢失，不能只比较两张谱的 motif overlap；
5. 为每族训练 availability/support/conflict/uncertainty 四通道。

### P2：实现最小全局因子图

先只接三种高可信 factor：

1. DreaMS calibrated unary；
2. ion-family hard/strong constraints；
3. complete observed reaction hyperedge。

在这个最小系统明确优于 DreaMS 后，再逐一加入 decoder、RT、SMN 和预测反应边。每次只加一种并做外部消融。

### P3：重新锁定正式 benchmark

正式性能结论至少要求：

- 数千 query、数百 baseline errors；
- dataset、formula、scaffold 三层隔离；
- DreaMS、MetDNA3、NetID式 global optimization 和候选结构工具的同协议比较；
- fixed-FDR annotation coverage；
- target-decoy、随机网络和错误 seed 传播压力测试；
- 所有模型与阈值冻结后一次性外部评估。

### P4：生物学验证去循环

1. 患者半拆分/LOPO 建图稳定性；
2. residualized abundance correlation；
3. 峰形与同位素/adduct 化学审计；
4. 冻结家族后才做 Rmu/RN 和 Rmu/Rtu 统计；
5. 关键候选仍按 Level 2/Level 3 表述，不因网络支持升级身份。

---

## 10. 论文定位

当前最诚实且最有潜力的创新点不是“BioAware 已提高10pp”，而是：

> DreaMS 提供强谱学 unary；BioAware 将 ion-family、候选特异碎裂、实验 feature 网络和完整反应 hyperedge 作为具有显式依赖关系的因子，在允许 unknown/abstain 的全局分配中控制错误传播，并把注释结果直接连接到患者级定量证据。

这比简单 Rhea 加分、同公式 peer 投票或无限扩图更有方法学价值。但必须由真正的全局推断、固定 FDR 和外部 benchmark 支撑。

---

## 11. 对照方法学依据

- MetDNA/MetDNA3 的关键不是“用了代谢网络”，而是 data-layer 与 knowledge-layer 的预映射、MS1/反应/MS2顺序约束、网络特异性和 decoy FDR；其研究也明确显示网络过密会降低特异性。
- NetID 的关键是把节点与边候选放入全局线性规划，在一致性约束下联合选择，而不是逐 query 加一项网络分数。
- CSI:FingerID/MSNovelist 类结构工具的关键是预测候选结构指纹概率；当前 kernel-ridge decoder 只能视为早期 headroom 探针。

参考：

- MetDNA3: https://www.nature.com/articles/s41467-025-63536-6
- NetID: https://pmc.ncbi.nlm.nih.gov/articles/PMC8733904/
- MetDNA: https://www.nature.com/articles/s41467-019-09550-x
- CSI:FingerID: https://www.pnas.org/doi/10.1073/pnas.1509788112

---

## 12. 一句话结论

BioAware 的数据基础和科学问题是成立的，但当前排序器把相关证据当独立票、把零权重证据计入 gate，并且没有真正做全局网络推断；原 `+1.71 pp` 必须撤回为历史开发输出。下一步应停止扩图和调线性权重，先修复证据依赖，再用有 unknown 状态的全局因子图，在大规模外部任务上以固定 FDR 的注释覆盖率证明价值。
