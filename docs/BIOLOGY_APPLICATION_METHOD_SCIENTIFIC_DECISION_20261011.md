# 生物学应用方法科学决策：从可观测化学空间到多场景发现

日期：2026-10-11  
状态：研究方向裁决；不是阳性结果，不改变算法 coder 的训练、冻结或外测计划

## 0. 纠偏裁决

本文保留 response transport 的全部技术合同，但撤回其“唯一主方向”地位。新的总纲见 `UNBIASED_STRUCTURE_AGNOSTIC_DISCOVERY_CORE_20261011.md`。首要研究问题改为：

> **在不以结构、通路或疾病名单筛选分析对象的前提下，已知与未知 LC-MS/MS 信息能否在无标签探索、有监督响应和单谱查询中产生超过 MS1、谱库匹配和现有谱学方法的可复现增量？**

**不确定性感知的谱学响应运输只是有监督研究算子之一。** 它比较的不是两个样本的 embedding 均值，也不要求不同研究中的每个未知峰具有相同身份；它把一项对照实验表示成带方向、效应量和测量不确定性的化学响应测度，并在两个严格区分的层级比较响应：

1. `entity transport`：只运输被校准为同一实体或不可分辨实体集合的质量；
2. `relation transport`：运输具有局部碎裂/化学关系证据的质量，但只支持“化学响应程序相似”，永不冒充同一实体。

总方法必须先支持无需 contrast 的样本化学表示，再在有研究设计时派生 response。ChemProp2 M1 只验证了后者的一种同表 MS1 特例；它没有使用 MS/MS 峰或 embedding，不能作为 response transport 或统一谱学算法已经有效的证据。

精确实体效应复现保留为备选方向；疾病未知实体到功能干预的三角验证是主方法通过后的生物学兑现，不是当前方法地基。Contrast Atlas、静态网络扩散、通路投影和单疾病差异分析均不再作为独立研究方向。

## 1. `unbiased` 的可检验定义

这里的 `unbiased` 不表示 LC-MS 没有电离、提取、DDA、样本选择或数据库偏差。它只表示发现阶段满足以下可审计性质：

- 实体进入统计总体不以结构身份、谱库命中、已知通路或目标疾病分子名单为条件；
- known 与 unresolved 使用同一 QC、离子家族去冗余和效应估计流程；
- 精确结构只改变解释等级，不决定一个 QC 合格实体能否贡献生物学响应；
- 表型标签不能进入跨研究谱学对应、候选生成或谱学权重校准；
- 结果必须报告 DDA 获取倾向、可测覆盖和弃权，不能把“未采到 MS/MS”当无生物学效应。

因此可以检验的是 `identity-unfiltered discovery` 和 `structure-agnostic inference`，不能宣称“无测量偏差的代谢组学”。

## 2. 三个候选科学问题

### 2.1 有监督算子：不同分子覆盖下能否迁移真实化学响应？

**科学假设。** 同一机制或控制节点会引起一个带方向的局部化学空间变形。两个研究即使没有检测到完全相同的全部分子，只要都采样到该变形的一部分，仍应比无关扰动更相似。

**现有缺口。** PAIRUP-MS 与 GromovMatcher解决跨数据集 feature 对应；Qemistree、MS2LDA、MEMO/PALS 与分子网络组织结构或样本差异；药理代谢图谱用代谢响应推断药物作用。它们没有共同回答：在不强制实体同一、又不把普通 embedding 接近当功能相同的情况下，未知 MS/MS 响应是否对 held-out 生物学机制有独立预测增量。

**不可替代的项目资产。** Noise 提供条件稳健表征；WSE 是强而独立的谱学基线；冻结 P2b 提供局部判别证据。ChemAware 仅在其资格范围内进入开发消融，不预设生产增益；BioAware 静态目录/拓扑不得进入，只有真实 event-level context 通过 B47 后才可作外部条件层。

**决定性实验。** 对多干预 LC-MS/MS 数据构造响应对象，以药物、靶点、来源和仪器为分组单位交叉拟合；查询一个 held-out 响应，检索相同靶点/MoA 或已知正交干预。比较 known-only、matched known-only、MS1-only unresolved、m/z/RT、cosine、WSE、official DreaMS、Noise、Noise+P2b、Qemistree/MS2LDA/MEMO/PALS 适用基线及完整方法。

**否定结果。** unknown 增量在任一关键来源/仪器留出中消失；增量可被 feature 数量、MS1 强度、缺失率、QC CV、MS2 获取率或细胞死亡解释；relation transport 只在同一实验批次内有效；或标签置换产生同等检索性能。出现这些结果时，停止“通用未知化学响应”主张。

### 2.2 备选方向：同一未知实体的跨队列效应能否被受控复现？

**科学假设。** 在存在真实实体重叠的研究中，表型盲谱学链接与效应复现可以作为一个联合命题，在固定错误率下比 m/z/RT 和 hard argmax 找回更多可靠未知实体。

**现有缺口。** 现有 feature alignment、record linkage 和 replicability inference 分别成熟；剩余价值只在 LC-MS/MS 的 no-match、多离子、嵌合谱和 DDA 选择下联合控制错误，并证明真实净功效。

**现有基础。** URSEA 与 `analyze_entity_effect_transport.py` 已提供安全基线；LCNEC 证明了停止门有效，但 HSST3n 到 LIPn 的 5-ppm 候选覆盖仅 `6/263 = 2.28%`，不能承担广覆盖主结论。

**决定性实验。** 隐藏名称的跨仪器身份真值 + 独立生物学队列；报告 link-FDR–coverage、no-match、错误复现/1000 entities，以及 known+unknown 对 known-only 的净复现增量。

**否定结果。** 可接受 link-FDR 下覆盖不足；只有同患者跨平台而无独立生物学复现；或完整联合推断相对保守基线没有功效增量。此时它只保留为质量控制工具。

### 2.3 下游兑现：疾病未知响应能否预测可验证干预？

**科学假设。** 一个在独立患者中复现的 unresolved entity 或小型响应模块，可以从扰动响应库中指向正交干预，并通过时间/剂量、示踪和前瞻表型检验。

**现有缺口。** Reverse metabolomics 已完成 chemistry-first 的公共检索和功能实验；IsoNet 与 isotope-tracing identification 已利用示踪研究未知反应/身份。可争取的增量是从无偏临床发现端出发，经已验证的响应运输提出功能预测，而不是先选化学家族再搜索疾病。

**决定性实验。** 预注册一个实体/模块、一个方向、两个机制不同但指向同一轴的干预、一个区分竞争来源的 tracer 和一个独立表型；验证同一实体按预测移动且表型改善。

**否定结果。** 只有静态丰度或公共出现位置；干预只改变实体不改变表型；示踪只有单次掺入却无法区分来源；必须先知道精确结构才能提出所有干预。该方向当前不独立启动，只能在主方向产生合格候选后进入。

## 3. 主方法的计算对象

### 3.1 研究内响应不是显著峰清单

对研究 `s` 的预注册对比 `c`，每个表型盲构建的 ion family / entity `i` 有：

\[
R_{sci}=(\hat\beta_{sci},\;SE_{sci},\;r_{sci},\;z_{si},\;h_{si}).
\]

- `beta` 与 `SE`：批次和预设协变量调整后的标准化效应及不确定性；
- `r`：由 pooled-QC、blank、稀释、缺失、峰形和 MS2 获取倾向定义的测量可靠性；
- `z`：冻结谱学表示，不由表型训练；
- `h`：身份状态、离子家族、仪器和采集条件等审计字段。

不先按 `p < 0.05` 截断。正、负方向分别形成响应测度：

\[
\mu_{sc}^{+}=\sum_i w_{sci}^{+}\delta_{z_{si}},\qquad
\mu_{sc}^{-}=\sum_i w_{sci}^{-}\delta_{z_{si}},
\]

其中权重来自预先冻结的效应收缩与测量可靠性，例如 `r * E[(theta-delta)_+ | data]`；`delta` 是最小相关效应。这样大样本微小效应不会自动支配，也不会用显著性筛选制造不稳定集合。

### 3.2 两种运输语义永久分离

对研究 A 的实体 `i` 与研究 B 的实体 `j`：

- `C_id(i,j)` 由 open-set 同实体证据构成，包含 dustbin/no-match；只有它能支持同一实体迁移；
- `C_rel(i,j)` 由局部碎裂、峰差/中性丢失及谱学邻域构成；它只支持有限分辨率的化学关系，不能支持身份、反应方向、通量或酶活。

使用带 dustbin 的 partial/unbalanced transport，而非强制质量守恒：

\[
T_C(\mu,\nu)=\min_{\pi\ge0}
\langle \pi,C\rangle+
\lambda_A D(\pi\mathbf 1\|\mu)+
\lambda_B D(\pi^T\mathbf 1\|\nu).
\]

同向响应分别比较 `positive-positive` 与 `negative-negative`；逆转响应比较 `positive-negative` 与 `negative-positive`。未运输质量必须报告为 coverage/abstention，不能由归一化隐藏。

### 3.3 不是“embedding + OT”的关键约束

单纯在 embedding 上做最优运输不是足够创新，也很容易把平台覆盖当生物学。主方法必须同时满足：

1. 运输单位是带方向和效应不确定性的研究内对照响应，不是原始峰云；
2. identity 与 chemical-relation 两种 cost 分开报告，主张等级不同；
3. partial mass、dustbin 与 acquisition propensity 显式进入；
4. cost 的校准完全表型盲，并在 identity/formula/source/instrument 隔离下完成；
5. 生物学终点是 held-out 干预/靶点回收及 unknown 的条件增量，不是内部聚类轮廓；
6. null 必须保持质量、强度、缺失、候选度数、仪器和 MS2 获取倾向，只打乱生物学响应或局部谱学对应。

如果这些约束没有产生超越 WSE/known-only/现有方法的结果，就不应给普通 OT 另起名字。

## 4. unknown 增量的正式分解

对每个 held-out 查询同时冻结六个世界：

1. `known-only`：confirmed + 预先规定的 library-supported entities；
2. `matched-known`：与 unresolved 在数量、强度、缺失、QC CV、MS2 信息量和采集倾向上匹配；
3. `known + unresolved MS1-only`：只给 m/z/RT 与效应；
4. `known + unresolved shuffled-MS2`：保持覆盖和度数但破坏化学关系；
5. `known + unresolved strongest existing spectral baseline`；
6. `known + unresolved full response transport`。

主估计量不是总体准确率，而是：

\[
\Delta_{unknown}=Metric_6-\max(Metric_1,\ldots,Metric_5),
\]

其中 `Metric` 至少包括 target/MoA retrieval MRR、Top-k、macro-AUROC、校准和弃权后的 coverage。置信区间按 target、drug family、source 与 instrument 聚类；任何用于选择方法的公共数据一律登记为 development/consumed。

还需报告 unknown-only。若 all-feature 阳性但 unknown-only 完全无信息，增益可能只是已知信号的重复表示；若 unknown-only 有信号但加入 known 后无增量，则可称未知实体携带信息，不能称改善实际发现。

## 5. 数据路线由假设决定

### 5.1 第一开发层

- **ChemProp/ChemProp2**：`MSV000096724`、`MSV000093571`、`MSV000094899`；50-drug endpoint 和 12-drug 9-time-point 数据有 raw LC-MS/MS、abiotic/vehicle controls 和时间方向。用途是开发 response construction、方向/null 和 drug holdout，不是最终跨来源确认。
- **IsoNet**：`MSV000097764`；用于检验 relation transport 是否保留 isotopologue/reaction truth。它不是 MoA 检索外测，也不能证明疾病生物学。
- **MorPhiC PR001844**：ST003325 与 ST003940 包含多个基因扰动和 DHEA-S 条件，可测试 gene/condition holdout；二者属于同一项目/来源，不能被计为两个独立外部来源。ST003940 当前逐样本扰动标签与完整 archive 展开不足，暂不进入正式结果。
- **MSV000086206 / MSV000083387**：E. coli Nissle WT/KO 可作小型遗传扰动正对照；需先确认两 accession 的独立性和 raw/mzML 重复表示，不能把文件数当样本数。

### 5.2 扩展开发层

- **S-BSST1724 human metabolic map**：已有 1,520 个药物、2,269 个 putative metabolites 和 MoA 结果，是机制标签与 known-response 强基线。2026-10-11 对 BioStudies API 的实际核查只发现一个 `NBT-RA62370A-Z_NonPooledFoldChanges.xlsx`（137,274,520 bytes），没有 raw/mzML/mzXML 或逐 feature MS/MS 文件。因此它当前只能裁决已知/推定离子响应检索，不能裁决本方法的未知谱学增量；除非作者仓库另有可验证的谱图—feature 联结，否则不得把它写成 LC-MS/MS 外测。
- **ToxCast/HeCaToS 等剂量/时间毒理数据**：只有同时存在合格 MS/MS、表型和对照设计时使用；剂量本身不替代独立机制真值。

### 5.3 最终验证层

最终层必须在方法、超参数、known/unknown 定义和主终点冻结后只打开一次；至少留出一个完整 biological source 与一个 instrument family。若现实数据不能支持这一隔离，主张自动降为单来源开发结果。Enveda-180 是候选排序算法的封存外测，不是本生物学方法的功能真值，二者不得混用。

## 6. 最小决定性实验

第一轮按资格顺序做一个最小研究程序，不建大 Atlas：

1. **M0 标签与分母资格审计。** 先冻结每个数据集的 drug、target、MoA、source、instrument、time/dose、replicate 与 controls；每个评价类必须在训练与测试各有独立干预实例。没有重复正类的 endpoint 只允许同干预复现，不得计算 MoA retrieval。
2. **M1 响应对象可复现性。** 在 ChemProp2 12-drug 时间序列中表型盲构建 ion family 和冻结 MS/MS 原型，以时间模型估计方向、效应与不确定性；用 50-drug endpoint 中的共享药物检验跨两时点/多时点设计能否找回同一干预，并区分其他药物。这里回答“响应对象是否存在”，不回答跨来源通用机制。
3. **M2 多干预机制增量。** 只有 M0 找到每类足够重复、且带 raw LC-MS/MS 的多干预资源后，才按 drug、target 和 chemical family 外层留出；内层只调运输正则、dustbin 与效应收缩。查询 held-out response，评价 target/MoA 回收，并控制 vehicle、abiotic、总离子量、OD/生长和 time-only response。
4. **M3 跨来源裁决。** 至少一个完整 biological source 和一个 instrument family 从模型选择中封存。随机样本折、同实验不同文件和 raw/mzML 副本均不能充当跨来源验证。
5. 每层比较第 4 节六个世界与 ChemProp2/FBMN、MEMO、Qemistree、MS2LDA、cosine/WSE/DreaMS/Noise 的适用版本。
6. M1 通过门是共享干预相对其他干预的 retrieval 增量 CI 下界大于 0；它只允许进入 M2。主方法通过门是 M2 的 `Delta_unknown` target/drug-family 聚类 95% CI 下界严格大于 0，且 M3 同号、错误机制检索不增加、unknown-only 高于 matched-shuffle。

M1 失败即停止主方向。M1 通过但找不到合格 M2/M3 数据，只能报告单来源响应复现可行性，不能宣布通用方法；M2 或 M3 失败则转入备选的实体效应复现，或承认现有谱学表示未提供通用生物学增量。

## 7. 现有仓库资产的边界

- 算法 coder 冻结的 production-eligible channel 才能进入正式主分析；生物学 coder 不重训、不改阈值、不从生物学标签反向选择 checkpoint。
- ChemAware Stage-1/Phase-A/V2 可作开发消融和化学解释，但在独立资格门前不写成已验证生产增量。
- P2b composite 与其原始通道不能重复计票；near-core 风险区必须允许归零/弃权。
- BioAware B44-B46 静态目录、拓扑、扩散和聚合上下文是冻结负结果；B47 前不进入 response cost。
- MTBLS13729 是结论边界与候选生成资产：Rmu-vs-RN 丰度与 Rmu-vs-Rtu 交互必须分开，静态丰度不等于 flux/酶活。它不裁决主方法。
- LCNEC 是低覆盖反例和技术校准资产，不再承担广覆盖跨平台运输或独立患者复制。
- 既有 ST003325/ST003940 元数据审计保留为数据记录，但停止继续扩建通用资产编译器，直到主实验明确需要。

## 8. 明确否决的路线

1. **Contrast Atlas 作为主方法**：它是结果载体，不是科学贡献；没有 held-out 机制增量时只是数据库工程。
2. **单疾病 unknown 网络**：无法证明通用性，也容易把相关、共享碎片和批次写成机制。
3. **静态通路/目录投影**：TidyMass2 等已有强竞争，且本项目 B44-B46 已给出负证据。
4. **硬性跨研究一对一对齐**：真实 overlap 低时会制造假复现；只能作为基线。
5. **embedding centroid/平均余弦**：不能区分谱图身份、局部化学关系、平台覆盖与生物学功能。
6. **把更多未知峰等同于更多发现**：必须在 matched known-only、shuffled-MS2 和独立功能真值下证明增量。
7. **用术语宣布新范式**：在最小决定性实验前，所有名称均为内部工作描述。

## 9. 最终推荐（已纠偏）

**总方法地基：** 无需标签的可观测化学表示，支持组成、异质性、公共证据和单谱查询，并在有设计时派生响应。

**有监督算子：** 带方向、效应不确定性和弃权的谱学响应运输；必须在无自然 feature ID 的跨设计/跨研究任务中超过 MS1-only 和谱学强基线。

**并行证伪入口：** 一个疾病/组织任务和一个无 contrast 的样本集合任务，共享同一冻结证据核心，不为数据集重写规则。

**下游生物学目标：** 总方法通过多场景门后，从疾病数据选择一个仍未定结构但在独立扰动中有预注册响应的实体/模块，完成干预—示踪—表型三角验证。

现在的研究进展指标只有：严格留出性能、unknown 条件增量、错误/弃权、跨来源迁移和真实生物学验证。transport 参数本身不是进展指标。

## 10. 主要相邻工作与证据来源

1. DreaMS: https://www.nature.com/articles/s41587-025-02663-3
2. PAIRUP-MS: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006734
3. GromovMatcher: https://elifesciences.org/articles/91597
4. Qemistree: https://www.nature.com/articles/s41589-020-00677-3
5. PALS: https://pmc.ncbi.nlm.nih.gov/articles/PMC7916825/
6. MEMO: https://www.frontiersin.org/journals/bioinformatics/articles/10.3389/fbinf.2022.842964/full
7. MS2LDA 2.0: https://www.nature.com/articles/s41467-026-75038-0
8. Reverse metabolomics: https://www.nature.com/articles/s41586-023-06906-8
9. Human metabolic map of pharmacological perturbations: https://www.nature.com/articles/s41587-024-02524-5
10. ChemProp2 functional metabolomics: https://pmc.ncbi.nlm.nih.gov/articles/PMC12889426/
11. IsoNet: https://www.nature.com/articles/s41467-025-60258-7
12. TidyMass2: https://www.nature.com/articles/s41467-026-68464-7
13. Metabolomics reporting and measurement guidance: https://www.nature.com/articles/s41592-021-01197-1
