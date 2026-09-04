# MTBLS13729 三轴生物学机制对标与论文闭环（2026-08-30）

## 1. 当前冻结结论

本项目已经得到三个可复核的 **Rmu-associated abundance programs**，但还没有得到一个可以称为通量或酶因果的机制：

1. 修饰鸟苷/嘌呤周转轴；
2. 乙酰化多胺轴；
3. 长链酰基肉碱/肉碱穿梭轴。

三条轴在同一批患者中并不形成一个稳定的单一相关模块，因此正文应将它们写成并列的代谢池表型，而不是强行串成一条因果链。主终点始终是患者内配对的 `Rmu vs RN`；`Rmu-RN` 与 `Rtu-RN` 的差异是次要交互终点，当前不能称作已确认的黏液型特异性。

## 2. 本地发现强度

| 轴 | 本地锚点 | Rmu vs RN | 谱学证据 | 当前身份边界 |
|---|---|---:|---|---|
| 修饰鸟苷 | feature 1597 / 3019 | `+3.721 / +2.401 log2`；有值患者全部同向 | 核糖丢失、跨加合物一致；MassBank 最优异构体差距极小 | methyl-/dimethyl-guanosine isomer family；位置异构体未定 |
| 嘌呤伴随 | feature 4966 | `+2.440 log2`；10/10 同向 | 重复碎片和患者内共变 | C7H9N5O nitrogenous-heterocycle/purine-like family |
| 乙酰化多胺 | feature 1717 | `+3.009 log2`；9/9 同向 | 73 张峰界内 MS2，`m/z 100.0759` 为 73/73 基峰；跨色谱差值相关 | N1,N8-diacetylspermidine-like；标准品未确认 |
| 长链酰基肉碱 | feature 3222 | `+1.776 log2`；8/10 同向 | 30 张峰解析共识 MS2；85.0281/60.0808 类别碎片 | long-chain/C20:4-acylcarnitine-like；双键位置和同法 RT 未定 |
| LCAC 类别 | 表型盲长链类别分数 | PQN `+1.382 log2`，精确 p=`0.00586`；链级去冗余后 `+1.368 log2` | 每个 Rmu 患者对至少 16、通常 20 个 LCAC 特征 | 类别级静态丰度重塑；不是单峰偶然，也不是通量 |

## 3. 三轴逐项机制对标

| 轴 | 最强外部正证 | 直接反证或竞争解释 | 仍缺的决定性证据 | 现在允许写什么 |
|---|---|---|---|---|
| 修饰鸟苷/嘌呤 | 2026 METTL1 CRC 工作使用蛋白组、scRNA、TRAC-seq/RNC-seq、催化失活突变、体内转移和 CCND3 rescue，证明 RNA m7G 可驱动 CRC 进展；TCGA 配对 CRC 支持一般嘌呤轴增强 | OEP00006137 的 Level-1 N2,N2-dimethylguanosine 在 MSI/MSS 中不升反降；METTL1 论文测的是 RNA 修饰与翻译，不是游离 methylguanosine 离子来源 | m7G/m2G/Gm/m2²G 同法标准；RNA m7G 定量；核酸降解/回收来源示踪或 writer/turnover 扰动 | Rmu 中修饰鸟苷样离子家族和嘌呤伴随轴积累；与 CRC RNA-m7G/嘌呤生物学相容，但来源与酶未定 |
| 乙酰化多胺 | 酸性 pH–SAT1–N1-acetylspermidine 研究完成非靶向筛选、定量、SAT1 KD/OE、细胞内外代谢物、免疫和血管生成表型；CRC 组织与尿液研究支持 N1,N12-diacetylspermine 增加 | 2013 CRC 组织质谱中 N1,N12-diacetylspermine 增加，但 N1,N8-diacetylspermidine并未显著增加；不同乙酰化位置不能互换 | N1,N8 标准的 RT/MS2/spike-in；与 N1-acetylspermidine、N1,N12-diacetylspermine 等异构/近邻标准并测；SAT1/HDAC10 轴的样本证据 | 存在强的 acetylated-polyamine/N1,N8-diacetylspermidine-like Rmu 丰度轴；精确分子和 SAT1 因果未定 |
| 长链酰基肉碱 | 本地 LCAC 类别广度；ATF6 人体 251 配对处理队列有 17 个显著 C20-C24 特征；Gut 2026 在 152+28 人体队列及小鼠示踪/类器官中证明外源长链 PUFA 输入；Oncogene 2026 在 1,257 人和 AOM/DSS 模型中建立 carnitine/acetylcarnitine–CPT1A 功能链 | MTBLS8090 35 对泛 CRC 未复现 LCAC 普遍升高；ATF6 数据中游离 AA 不显著且同链不同 RT 可反向；Oncogene 显示 carnitine 升高也可伴随 CPT1A 上调 | C20:4/C16:0/C18:0/C18:1 标准组合；若有资源再做同位素内标、游离肉碱/乙酰肉碱与 LCAC 比值；真正区分机制需要 tracer/OCR/酶扰动 | Rmu 中 broad LCAC accumulation 和 `carnitine-shuttle imbalance`；输入增强、利用瓶颈、不完全氧化和组织组成均是竞争假说 |

## 4. 标杆机制论文实际完成的五道门

| 门 | 高水平论文常见做法 | 本项目状态 | 最小升级动作 |
|---|---|---|---|
| 结构终证 | 同法标准 RT、MS2、多能量、spike-in；异构体并测 | 三个主轴均未过 Level 1 | 优先 N1,N8-diacetylspermidine；其次 m7G/m2²G 与 C20:4 LCAC 组合 |
| 来源/通量 | `13C/15N/2H` tracer、MIDA、时间过程 | 无 | 现实条件下只能明确标为未完成；不得由静态丰度推断 |
| 候选酶 | 遗传和药理扰动，最好方向一致 | 无本队列扰动 | 只能用外部 METTL1/SAT1/CPT1A 文献提出可检验假说 |
| 表型 | 细胞、类器官、动物、空间/细胞来源 | 有公开转录/蛋白背景，无本队列表型实验 | 继续做公开队列与空间定位，但必须称 mechanism-supporting context |
| rescue | 下游代谢物、基因回补、底物剥夺或受体激动剂 | 无 | 不得写治疗靶点已验证 |

## 4.1 本队列临床分层敏感性

新增的精确置换审计使用四条预定义轴：修饰鸟苷模块、feature 4966、feature 1717 和 feature 3222。

1. Rmu 内有 6 例 dMMR、4 例 pMMR；四条轴在 dMMR 中均值均较高，但最小名义 p=`0.103`，同一对比内 BH 最小 q=`0.324`。因此 dMMR 可能放大代谢表型，但不能解释为已证实的驱动因素。
2. 限定 pMMR 后，Rmu 相对 Rtu 的四条轴均值仍全部为正；feature 3222 差异最大约 `+1.53 log2`，精确 p=`0.0929`、四轴 BH q=`0.20`。这说明主信号并非显然完全由 MMR 构成造成，但仍不足以建立组织学独立效应。
3. BRAF+ 只有 2 例，不做显著性推断；10 个 Rmu 全部位于右侧，位置与组织学完全混杂，无法从本队列分离。

可复核产物：`data/mtbls13729/clinical_axis_sensitivity_v1/clinical_axis_comparisons.csv` 和 `report.json`。

## 5. 对标论文的共同设计规律

1. 发现队列与验证队列分离；非靶向发现随后转入更大 targeted quantitative 队列。
2. 结构验证和生物学机制是两条独立证据链；谱库相似度高不等于机制，网络邻近也不等于身份。
3. 患者组织只负责回答“代谢池是否改变”；示踪负责“碳/氮从哪里来”；扰动和 rescue 负责“谁驱动、是否影响表型”。
4. 空间或单细胞层用于拆分肿瘤细胞、免疫、基质和坏死区，避免 bulk 组织组成冒充细胞内重编程。
5. 阴性外部队列和竞争机制必须保留。它们缩小主张范围，同时使论文更可信。

## 6. 当前最强论文主线

> A DreaMS-enabled, evidence-calibrated reanalysis of paired colorectal tissues recovered reproducible ion families and class-level abundance programs that were unresolved in the original m/z–RT-only table. The Rmu discovery subgroup showed high-amplitude modified-guanosine/purine, acetylated-polyamine and long-chain-acylcarnitine programs. Raw fragmentation, ion-family concordance, paired targeted-EIC requantification and orthogonal human datasets support these programs, while external counterexamples reveal cohort, isomer and flux dependence. The resulting contribution is a rigorously bounded mechanism-generating atlas rather than a claim of causal metabolic flux.

这条主线的创新不是“第一次发现 CRC 有肉碱或多胺”，而是：

1. 用改进后的谱学表示与候选专家从原始 DDA 中恢复原表未解析的离子家族；
2. 把注释置信度、离子家族去冗余和患者内重定量连成一条可审计链；
3. 同时展示正证、反证和竞争机制，避免从静态丰度跳到通量；
4. 给出最小标准品集合，使关键候选可以低成本升级。

## 7. 近期优先级

### P0：无需新湿实验

1. 冻结三轴证据矩阵和所有允许/禁止措辞。
2. 补一张本地 LCAC 类别图，展示患者配对广度、链级去冗余和 3222 锚点；服务器类别产物需要同步到本地以完成一键复核。
3. 将 feature 1717 与公开 N1,N12-diacetylspermine 的正证、N1,N8-diacetylspermidine 的阴性组织证据并列写入结果。
4. 修改仍将 3222 单向解释为 FAO-utilization bottleneck 的活动文档和图注。
5. 临床分层敏感性已完成；后续公开队列应优先寻找同时具有组织学、MMR/MSI、CMS 和原始代谢谱的数据，不能继续用本队列内部切分替代外部复现。

### P1：若只能购买少量标准

1. N1,N8-diacetylspermidine，并加入 N1-acetylspermidine/N1,N12-diacetylspermine 反证组合；
2. m7G、m2²G；若预算允许加 m2G/Gm；
3. C20:4、C18:1、C18:0、C16:0 acylcarnitine 组合；
4. 每个候选执行同法 RT、多碰撞能 MS2 和样本 spike-in，不能只跑纯标准谱。

## 8. 主要来源

- METTL1–m7G tRNA–CRC/CRLM: https://pubmed.ncbi.nlm.nih.gov/41627602/
- METTL1–MACC1/SDCCAG8 mRNA m7G: https://pubmed.ncbi.nlm.nih.gov/42464401/
- Acidic pH–SAT1–acetylspermidine: https://pmc.ncbi.nlm.nih.gov/articles/PMC10563787/
- CRC tissue N1,N12-diacetylspermine: https://pubmed.ncbi.nlm.nih.gov/23443255/
- ATF6–LCFA–microbiota: https://pmc.ncbi.nlm.nih.gov/articles/PMC12460170/
- Extrinsic lipids in CRC: https://pubmed.ncbi.nlm.nih.gov/41856524/
- Carnitine/acetylcarnitine–CPT1A: https://www.nature.com/articles/s41388-026-03835-4
- Integrated untargeted/targeted/scRNA/spatial CRC lipidomics: https://pubmed.ncbi.nlm.nih.gov/39658263/
- CRC AHCY mechanism benchmark: https://pubmed.ncbi.nlm.nih.gov/37580540/

## 9. 声明边界

本文档综合的是静态丰度、谱学、外部表达/蛋白及已发表机制实验。不同层级证据不合并 p 值，也不互相替代。当前项目不声称 MSI Level 1 结构、代谢通量、酶活、治疗靶点或已经建立的黏液型特异机制。
