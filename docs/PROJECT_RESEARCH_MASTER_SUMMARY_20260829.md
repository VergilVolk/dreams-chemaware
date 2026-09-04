# DreaMS/ChemAware/BioAware/生物学应用：全项目科研总账与统一故事

**版本日期：2026-08-29**  
**用途：** 汇总从项目启动至今的算法、解释性、生化网络与生物学应用成果；统一所有指标口径；为论文结构、阶段答辩和后续三条 agent 路线提供共同事实底稿。  
**原则：** 正结果、负结果、工程纠错与尚未完成项全部保留。任何“百分点提升”必须同时写明任务、数据划分、是否改变模型权重及是否为盲测。
**本次增补：** 将原先被压缩为“平台准备”的候选协议修复、统一谱库、真实三路应用、E6证据层提升、P2b覆盖—质量边界、MS1冻结定量和离子家族去冗余提升为正式成果条目。

> **2026-09-01 注释—生物学统一对照增补：** 已把作者原生注释、作者RPLC坐标在当前MS1目标中的恢复、官方DreaMS、E6共享embedding、冻结P2b、三路共识与三路并集放到分母安全的同一审计中。作者原生为345/9,766（3.53%）或345/6,054个MS2-bearing feature（5.70%）；当前16,953个RPLC重定量target上，官方DreaMS/E6/P2b候选覆盖分别为20.16%/20.21%/21.16%，三路稳定共识为12.75%。E6相对官方只增加9个候选，但Level2a-supported增加22个；P2b增加171个候选但Level2a-supported减少24个。完整边界与生物学归因见 [`MTBLS13729_ANNOTATION_TO_BIOLOGY_BENCHMARK_20260901.md`](./MTBLS13729_ANNOTATION_TO_BIOLOGY_BENCHMARK_20260901.md)。

> **2026-09-01 生物学证据校准增补：** 18-candidate ledger 已拆为9个source remap、3个同队列正交Level-1 recovery、5个source-table-absent family hypothesis和1个主动降级control；没有一个允许写成新标准确认精确代谢物。早期6个DreaMS优先feature中仅hypoxanthine、tryptophan和carnitine作为已知身份背景节点保留；L-kynurenine、malic acid和3-phenyllactic-acid vote因来源已知、targeted-EIC不稳或身份冲突被过滤。当前Package A主现象冻结为`mucinous-relative free-Neu5Ac pool expansion with pool-to-donor/destination decoupling`，而不是C20:4酰基肉碱或修饰鸟苷单轴。投稿级结果包见`data/mtbls13729/manuscript_evidence_package_v3/`。

---

## 0. 先给结论：我们现在到底做成了什么

本项目已经形成了一条完整而可证伪的科研链：

> **DreaMS 残余错误并非均匀随机，而是集中在近异构体、跨采集条件正例不足以及候选相关峰证据误用；峰级动作具有可重复的方向性，部分方向已经能够通过直接噪声微调迁移进同一个共享 embedding；embedding 后的正交谱学专家能进一步提高常规候选检索，但在最难 near-core 上存在明确安全边界；化学双重映射已经证明 embedding 中存在可解码概念及其峰级输入来源；生化网络专家已经完成严谨工程闭环，但尚未获得显著外部准确率增益；在表型盲冻结注释后，MTBLS13729 已产生 C20:4 酰基肉碱、核苷样和多胺代谢的真实生物学候选。**

### 0.1 当前最重要的真实数字

| 模块 | 证据层级 | 基线与结果 | 当前裁决 |
|---|---|---|---|
| **直接噪声微调共享 embedding（E4-A）** | 5 formula folds × 3 seeds 的开发图 OOF；真实权重 | Recall@1 平均 **+0.635 pp**；near **+0.522 pp**；每 seed 约 186 修正/35 新增；15/15 fold 的公式簇 CI 下界 >0 | 当前最强、最稳定的共享 embedding 结果；尚未做新的外部/封存总裁决 |
| **E8 成熟共享 embedding** | 单个开发 formula fold；真实权重 | Recall@1 **+0.574 pp**；near +0.609 pp；38/4；preservation 0.99527 | 证明共享 query/reference 更新和多步 curriculum 有效；单折，不替代 E4-A 多折结论 |
| **P2b rank fusion** | embedding 后冻结候选专家 | 开发 nested formula OOF **+3.91 pp**；封存 P3 main **+1.07 pp**，CI 为正；P3 near-core **−4.23 pp** | 常规检索保底模块成立；不是新 embedding；不能全局宣称 SOTA |
| **RAW reranker v1** | embedding 后候选重排 | 开发 +4.35 pp；冻结原协议 Test-A +0.45 pp、Test-B +0.73 pp，均不显著 | 开发增益未稳定泛化；只作历史基线/消融 |
| **MTBLS13729 三路真实应用** | 无结构真值的描述性应用审计 | E6 在 pos_rp 仅新增9个分配，但 Level2a-supported +21、总Level2证据 +37；P2b新增171个分配（约+4.79 pp覆盖），但Level2证据下降 | E6偏向证据稳定，P2b偏向覆盖扩张；变化不能称纠正 |
| **BioAware v1/v2** | embedding 后生化网络专家 | MTBLS1905 evaluation-only 1修正/0新增，CI 下界0；可部署0变化；MTBLS13729 v1 0/1，v2 two-layer 0/0 | 工程与解释链成立，显著准确率增益尚未成立 |
| **化学双重映射** | 冻结表示解释 | 谱图概念 macro-AUPRC 0.659（基线0.200）；结构环境 AUPRC 0.240（基线0.0447）；175条跨层桥；2个因子通过定向删峰→embedding位移 | 解释性骨架成立；尚无因子证明控制检索 Top-1 |
| **MTBLS13729 生物学应用** | 表型盲冻结注释 + 配对MS1定量 | 6个正离子丰度feature过探索性 FDR10、去冗余后最多5个离子家族，3个feature过FDR5；C20:4候选在 Rmu/RN 升高2.34–3.39倍 | 真实生物学发现候选；结构仍为 Level 2，不能声称通量或具体酶 |

### 2.1 2026-08-29 生物学闭环新增成果

BioAware 的反应网络没有直接覆盖八个核心候选，也没有获得外部准确率增益，因此未被用来
强行提升身份。其表型盲全局峰图却完成了两个关键离子家族合并：`1597/7489` 为
methylguanosine isomer family，`3019/8481` 为 dimethylguanosine isomer family。

完全去冗余后，修饰鸟苷模块在 10 对 Rmu/RN 中 10/10 同向升高：raw `+2.953 log2`
（7.74倍），三档全局 PQN 为 `+2.852–2.860 log2`（7.22–7.26倍），每档精确 sign-flip
`p=0.001953`，留一患者后方向不变。该模块与 purine-like `4966` 高度相关，却与
N1,N8-diacetylspermidine-like 和 C20:4 acylcarnitine-like 不相关，支持三条相对独立的
代谢轴，而非一个强行拼接的通路。Rmu/Rtu 模块效应差约 `+2.0 log2`、探索性 `p≈0.02`，
但尚非亚型特异性确认。完整闭环见
[`MTBLS13729_BIOAWARE_BIOLOGY_CLOSURE_20260829.md`](./MTBLS13729_BIOAWARE_BIOLOGY_CLOSURE_20260829.md)。

进一步的表型盲技术匹配负对照从完整 feature 空间冻结 2,000 个随机双家族模块，并对 95 个
唯一背景 feature 从原始 mzML 做相同定向 EIC。1,412 个随机模块拥有全部10对患者值；真实
模块效应超过这些随机模块中的全部，四种归一化经验单侧 `p=0.000708`。但随机面板完整率
70.6% 未达到预设75%覆盖门，因此保留综合 gate=false：可写强背景特异性证据，不能写成
外部确认。

### 0.2 “最终提高多少”必须这样回答

目前**不存在一个可把上述数字直接相加的最终系统分数**。原因是：

1. E4-A 的 +0.635 pp 是共享权重在开发公式折上的结果；
2. P2b 的 +1.07 pp 是以官方 DreaMS 为基线、在封存 P3 main 上的下游专家结果；
3. BioAware 的网络结果来自另一外部任务；
4. MTBLS13729 缺少大规模标准品真值，评价的是冻结候选与丰度发现，不是检索准确率。

因此当前可分别声称：

- **模型权重本身：** 多折多种子开发证据为 **+0.635 pp overall / +0.522 pp near**；
- **冻结下游专家：** P3 main 上 **+1.07 pp**，但 near-core **−4.23 pp**；
- **动作空间潜力：** 历史全图 P/N/positive action union 已超过名义 5 pp 头寸；在更严格的成熟 E8 held-fold 几何中，E11 后 outcome-aware 总头寸为 **+4.457 pp**。二者都是上限，不是模型成绩；
- **整个新系统的最终增益：** **尚未在同一个新的、未查看过的基准上测量**。在完成“新 embedding + P2b 安全路由 + BioAware 弃权门”的冻结组合评估前，不能给出一个合计数字，也不能声称全面 SOTA。

### 0.3 这项工作已经形成的七类实质成果

此前版本过度按“路线”叙述，容易让大量已经完成的工作看起来只是准备步骤。按可交付成果重新归纳，项目实际上已经形成七类独立资产：

1. **严格评价资产：** 建立了公式隔离、IK14 隔离、候选按分子聚合、平分计错、公式簇 bootstrap、McNemar、开发/封存分离和哈希封存的一整套检索评价协议；P3 把常规检索和 near-core 压力测试分开，成功抓住了开发集虚高与子群退化。
2. **错误机制资产：** 在 23,876 个真实 query 上得到 1,805 个官方错误的全图谱，并拆为 positive deficit、negative excess、共享主峰、条件漂移和比较边界等机制；峰删除与匹配随机对照使“哪些峰参与错误”从相关性观察进入可证伪的反事实实验。
3. **共享 embedding 资产：** 噪声动作矩阵、敏感性审计、梯度兼容性和共享编码器训练已闭环；E4-A 在 5 个 formula fold × 3 个 seed 上稳定获得约 +0.635 pp，而不是只训练一个后处理头。
4. **谱学专家资产：** RAW reranker 暴露了开发—盲测落差；P2b 进一步形成可冻结、可复现的 rank-fusion 工件，在 P3 main 上获得 +1.07 pp，同时明确 near-core 不安全区。
5. **化学解释资产：** 3,486 条规则、峰 token 因子、结构环境探针、175 条跨层桥和定向删峰忠实性共同构成“embedding 因子—化学概念—具体峰”的初版双重映射，而不是仅有 PCA 图或相关性热图。
6. **真实注释平台资产：** 修复了候选生成顺序错误，构建正/负离子统一谱库、质量约束检索、FDR/校准、Schymanski 分层、三路注释比较、MS1–MS2 连接和冻结 EIC 重定量；可以对数十万真实谱图端到端运行并保留逐条审计记录。
7. **生物学发现资产：** 在没有 pooled QC、标准品和新湿实验的约束下，完成表型盲冻结注释、患者内配对、多归一化敏感性、离子家族去冗余和外部类别复核；形成 C20:4 酰基肉碱、核苷样未知离子家族和多胺代谢的分层候选。

这些成果不能被压缩成一个“模型只提高 0.6 pp”的句子。0.6 pp 回答的是共享权重性能；其余成果分别回答错误为何发生、峰级证据能否定位、候选专家能否补充、化学意义能否解释，以及真实非靶向数据能否形成生物学发现闭环。

---

## 1. 统一证据语言：防止再次偷换概念

本项目以后统一使用五类标签：

| 标签 | 含义 | 可以声称什么 |
|---|---|---|
| **W：weight result** | 原始 clean MS/MS 经同一新编码器得到新 embedding | 可以讨论“模型权重/表示空间改善” |
| **X：post-embedding expert** | 冻结 embedding 后使用候选谱、网络或规则重排 | 可以讨论“系统性能”，不能说 embedding 改善 |
| **A：action/oracle/headroom** | 看过动作结果后逐 query 选择动作或 no-op | 只能讨论理论容量和数据中是否有梯度 |
| **D：development/OOF** | 开发集、交叉验证或已消耗 held fold | 可做模型选择与机制结论，不能当最终盲测 |
| **B：sealed/external blind** | 工件、阈值全部冻结后的一次性测试 | 才能支持最终泛化主张 |

最常见的错误是把 A 当 W、把 X 当 W、把 D 当 B，或把不同协议的百分点相加。本总账中的每个结果均按这五类重新归位。

---

## 2. 数据、候选协议与评价基础设施

### 2.1 MassSpecGym 严格候选任务

经过多轮纠错后，主训练/开发错误图固定为：

- 23,876 个真实 query；
- 2,522 个 IK14；
- 1,082 个分子式；
- strict 10 ppm、同 adduct、候选按分子聚合；
- 官方 DreaMS Recall@1 = 0.924401；
- 1,805 个官方 Top-1 错误；其中 1,446 个为 near 子群；
- formula folds 固定，P3 query identity overlap = 0；
- 平分计错，所有 corrected/introduced 均与同一 clean forward 对账。

这套协议修复了早期至少四个会改变结论的问题：按 IK14 排序截取前1万导致实际只有516个分子、训练/评估“同分子”定义不一致、同 anchor 与非同 anchor margin 混算、候选正谱重复和 rank tie 口径不一致。

### 2.2 P3 锁集

P3 封存面板包括：

- main-real-pristine：3,000 query；
- isomer-real-pristine：1,989；
- near-core-real-pristine：496；
- near+mid：661；
- exposed extension：851；
- sim-to-real secondary：609。

P3 的价值不只是一个分数，而是把常规主任务与 near-core 压力测试分开。P2b 正是在 main 上正向、near-core 上负向，证明单一总体均值会掩盖真实安全边界。

### 2.3 外部生物数据

- **MTBLS13729：** 240 个沉积 mzML，4 panel × 60 样本；MS1 全扫真实存在，可做定量。项目已从原始 mzML 完成逐样本 MS1 feature 提取、跨样本共识、目标重定量和 MS1–MS2 连接；`neg_rp` 冻结3,798个重定量目标，`pos_rp` 冻结13,155个目标，最后分别形成62与555个三路注释支持的表型盲定量目标。无 pooled QC、blank 和外部参考，故使用患者内配对与多归一化敏感性分析；Rmu 仅10对，定位为发现队列。
- **MTBLS1905：** 22位 HNSCC 患者，18位有完整 C/E/N 三联；QC-DDA 共8,601张MS2；构建36谱/18目标的外部已知身份小面板。
- **MTBLS8090：** 35对 CRC 肿瘤/癌旁，用于 LCAC 类别外部复核。

### 2.4 从 embedding 到可交付注释的平台

项目还完成了一个容易被算法指标掩盖、但对真实非靶向代谢组不可缺的工程层：

- precursor m/z/ppm 硬约束后的候选检索；
- 目标—诱饵 q-value/FDR 链路与 Platt/isotonic `P(correct)` 校准；
- Schymanski 分级，平台诚实封顶 Level 2a/3，永不自动输出需要标准品的Level1或需要分子式工具的Level4；
- 患者配对差异、通路接口和暗谱聚类/候选 lead；
- 全部中间产物、参数、消融和报告落盘。

其中一个决定性工程纠错是：旧实现曾先在全谱库做 global cosine Top-k，再用 precursor ppm 过滤；这会让真正位于质量窗口内的候选在进入质量过滤前就被全局高相似谱挤出。正式实现改为**先建立 precursor-mass 候选图，再仅在窗口内排序**。在同一 `neg_rp` smoke、20 ppm、cosine≥0.7 条件下，可信谱数由 **929 增至 1,023**，覆盖率由 **13.43% 增至 14.79%**，新增94张谱，唯一 InChIKey 由87增至89。这是检索协议修复带来的真实覆盖提升，不是调低阈值。

统一参考库也已从小型试点扩展为真实应用规模：正离子 **265,011** 张、负离子 **29,564** 张，合计覆盖约 **207,787** 个唯一 InChIKey；同时剔除619条 MassSpecGym 和129,929条 GNPS 前体信息不一致记录，避免错误 precursor 元数据污染候选图。在 MTBLS13729 上，管线实际处理 `neg_rp` 374,232 张与 `pos_rp` 419,676 张 MS2，并从 MS1-linked 谱中分别选择24,846与86,646张进入统一注释协议。

真实试点的 confident 注释覆盖约5.9%，其余约94.1%是“暗物质谱”。平台对它们做聚类和候选lead，而不是把94.1%包装成已鉴定结构。FDR链路已经验证，但正式q-value要求1:1全量诱饵，不能用小诱饵子集代替。规则证据只用于Schymanski语义升级；13,770个Top-1中实际仅翻转1个Level，说明规则命中不等于置信度提高。

### 2.5 不只是“跑通”：已经建立的复现与防错基础设施

- 模型、候选图、HDF5、参考谱库、训练缓存、测试面板均保存 SHA256；冻结评估只加载工件，不在测试时重新拟合。
- query/reference 使用同一共享编码器、同一 HDF5 行顺序和同一质量候选协议；缺行、无正例、文件数不符或顺序不符均 fail-closed。
- 每个 query 保留候选数、rank、Top-1、是否干预、模型分差和迁移结果；真实应用无结构真值时只使用 `retained/changed/abstained`，禁止使用 `corrected/introduced`。
- 统计上同时报告点估计、公式簇配对 bootstrap、McNemar 和 near/mid/main 分层，避免把重复谱当独立样本或用总体均值掩盖 near-core 失败。

这些基础设施曾真实抓住训练池只有516个分子、A/B测试面板重叠、错误候选协议、dropout train/eval 混杂、并列规则不一致、开发集 +4.35 pp 不泛化等问题。它们不是附属工程，而是当前所有可信结论成立的前提。

---

## 3. 从“DreaMS 很强”到“错误可定位”：基础发现

### 3.1 官方微调确实大幅改善了原始 SSL

在项目自建的严格10 ppm MassSpecGym队列中：

| 表征 | pooled AUC | Top-1 |
|---|---:|---:|
| 原始 SSL 完整 embedding | 0.615 | 0.559 |
| 官方微调完整 embedding | **0.799** | **0.763** |

因此项目不是从一个弱基线开始。后续困难来自官方模型已经接近高性能区间，剩余错误高度局部化。

### 3.2 残余错误集中在真实近结构邻域

早期 P0 审计得到 2,109 个 Top-1 错误、Recall@1 0.9003。错误显著富集于：

- 相同/相近分子式；
- Morgan 高相似；
- MCES 0–2 的 near 候选；
- 高规则重叠；
- 候选数多的局部邻域。

正式残差审计进一步表明：官方微调改善了同分子式局部结构关系，却牺牲了部分全局结构连续性。因此 Pearson、AUC、Top-1 回答不同问题，不能用一个指标替代另一个。

### 3.3 峰级因果证据成立

在 discovery/confirmation/test 隔离协议中：

- confirmation 删除全部混淆峰，错误纠正率 28.1%；匹配随机删峰仅7.6%；
- 删除身份特异峰会进一步降低正确排序 margin；
- 冻结8项峰机制面板在一次性测试中：定向删除纠正10.74%，匹配随机4.52%；
- 公式平衡 margin 净改善0.0305，bootstrap 95% CI [0.0229, 0.0386]。

这建立了全项目最重要的科学起点：DreaMS 一部分近异构体错误确实来自可重复的峰证据利用偏差。但动作依赖已知正/错候选，仍不是部署策略。

### 3.4 真实错误总图谱

23,876-query atlas 将1,805个错误拆为：

| 错误机制 | 数量 |
|---|---:|
| positive deficit only | 1,242 |
| positive deficit + negative excess | 197 |
| negative excess only | 188 |
| comparative boundary | 178 |

辅助屏幕中：1,439个存在 positive deficit，385个存在 negative excess，984个有 shared-major-peak，1,032个有 neutral-loss convergence，563个有跨条件正例问题，783个可由 RAW 证据救援。这个分解解释了为什么“只删错误峰”不可能解决全部问题：绝大多数残余错误还需要恢复正例证据。

---

## 4. 化学规则库：没有成为距离标签，但成为了证据语言

### 4.1 规则库规模与工程完成度

正式 G0 缓存覆盖25,275张可达谱、3,472个身份，共3,486条规则：

- 335条核心机制/经验规则；
- 3,151条 MassBank 记录衍生规则；
- NL 293、CF 3,174、ISO 8、NR 1、EE 1、HR 9；
- 35条零谱支持，113条少于10身份支持，53条覆盖超过50%身份。

同时修正了旧规则语义：核心 NL 使用 precursor−fragment；ISO/HR 使用峰对差；MassBank 前体/中性质量偏移单独处理。两谱均无规则时 Jaccard 记缺失，不能记为1。

### 4.2 规则作为错误检测特征：有中等信号

- 不同分子 error-detection AUC：核心335规则0.612、全3,486规则 **0.647**；
- 同分子一错一对配对 AUC：全规则 **0.599**；
- 规则权重头把等权 AUC 0.6177 提到0.6382（+0.0205 AUC），但置信区间高度重叠；
- 增益主要来自粗类别重加权（NL上调、MassBank CF下调），连续元特征贡献很小。

这些结果证明规则与错误风险相关，但信号不足以直接定义正负样本或 embedding 距离。

### 4.3 规则噪声与直接规则微调没有成立

- 规则侧遮峰后错误样本通常保留“错误方向”的规则证据；
- 最大正向单规则 margin 只有0.052，覆盖至多2个错误分子；
- 强度抖动 consistency triplet 在官方头上起始即满足，loss≈0，三seed变化约1e−4，无有效梯度；
- 匹配因果删除在更严格的错误/正确对照上没有错误特异性，正式 eligible intervention = 0。

因此，“规则重叠直接监督 ChemAware 距离”被否定；规则的正确位置被固定为：峰证据注释、概念监督、样本分层、冲突解释和候选特异碎裂 likelihood。

---

## 5. 化学可解释性双重映射：已完成的闭环与未完成的终点

目标链为：

`global/peak embedding → 化学概念或局部结构环境 → 具体峰/质量差 → 定向删峰忠实性`

### 5.1 全局 embedding 可解码

| 任务 | 结果 |
|---|---:|
| 266个谱图碎裂概念 | test macro-AUPRC **0.659**，流行率基线0.200；254/266达到各自基线≥2倍 |
| 469个数据驱动局部结构环境 | macro-AUPRC **0.240**，基线0.0447；396/469达到≥2倍 |
| 结构环境—谱图概念桥 | 29,484候选中1,260跨 discovery/confirmation 复现；再经方向余弦99百分位保留175条 |

这证明官方 DreaMS 中存在可线性读取的实验碎裂概念和局部结构信息，不只是黑盒相似度。

### 5.2 峰 token 因子

- 10个稳定峰因子；
- 8个复现因子—峰质量；
- 2个复现因子—结构环境；
- factor117：m/z 91.06、`ccc` OR 9.38；
- factor176：m/z 67.06、`ccc` OR 9.41；
- 定向删峰相对匹配随机导致额外全局 embedding 位移：factor117 CI [0.0021,0.0056]，factor176 [0.0033,0.0080]。

但两者的严格10 ppm检索 margin CI 均跨0。因此已经证明“这些峰是 embedding 因子的真实输入来源”，尚未证明“这些因子控制检索判断”。

### 5.3 阴性探索同样构成进展

- PCA/varimax：0/32方向完成结构+谱峰+混杂全门；
- precursor SAE：发现5个跨seed稳定方向，S0277与芳香甲基环境相关，但峰证据未复现；
- 多层 Crosscoder：共享稀疏子空间可泛化，但0个单因子达到跨seed稳定门；
- KPGT contextual bond token 受全局描述符和虚拟节点影响，不作为无先验局部碎裂发现器。

这些阴性结果把解释路线从“给潜变量起名字”推进到严格的独立复现与干预忠实性标准。

---

## 6. 噪声微调：完整方法学演进

### 6.1 第一阶段：早期反事实头证明可训练，但规模小

- 只训练官方1024×1024 projection head、冻结7层主干；
- 内部1049-query/100 formula任务中约 +0.57 pp；原始计数10 fixes/4 regressions，严格 tie 重算后为6个稳健修正/2个回归、净4；
- preservation 0.99848；Top-1 bootstrap 下界为0；
- 旧低置信峰重排可达 +0.88 pp，但规则本身相对 RAW 只增 +0.14 pp，CI跨0。

它证明局部顺序可以安全改变，却暴露了高 preservation 权重、偏小样本、预计算峰集合和错误候选覆盖不足。

### 6.2 跨条件“拉近正例”路线为什么失败

真实跨条件对数量很大，但简单把正例拉近导致全局空间收缩：

- M3 head-only：cross cosine +0.028，但 negative +0.060，margin −0.033；
- M4 解冻最后1层：cross +0.050，negative +0.097，margin −0.048；
- 加绝对 ceiling 后仍为 cross +0.037、negative +0.064、margin −0.027；
- train negative 可以下降而 validation negative 上升，说明负例守卫记住训练负例、没有泛化。

随后 G5/G6/G7 的正式全量门也失败：官方 macro-AUC 0.8676，G5/G6/G7 分别0.8216/0.8354/0.8365；Recall@1由0.9301降至0.9201/0.9228/0.9223。根因是训练池实际仅516个分子、合成噪声替代真实跨条件正例、无关in-batch negatives和主峰随机删除。

### 6.3 G8R/M1b：把失败机制拆清

G8R M1：

- 跨条件正例 cosine +0.0292；
- hard-negative cosine +0.0303；
- macro-AUC −0.00094；Recall@1不变；4/4；preservation 0.99465。

M1b 改为完整局部三元组后，宏观检索略升，但同-anchor margin仍压缩。P0三seed证明 dropout 是真实混杂：关掉 dropout 后 margin 从 −0.0056变为+0.0004，但 pairwise accuracy仍不改善。结论不是“线性head数学上不可能”，而是当前冻结backbone+线性head+局部损失没有解决 near 排序。

### 6.4 从盲目删峰到正交动作矩阵

#### S1a 单峰矩阵

| 动作 | 最佳/代表结果 | 科学含义 |
|---|---|---|
| candidate-gradient 50% | 138修正/113新增，净+25 | 有纠错信号，但全局执行风险高 |
| candidate-gradient 100% | 288/560，净−272 | 强梯度不等于安全增强 |
| role-confounder 100% | 99/18，净+81；near净+60 | 最干净的固定负臂动作 |
| role-identity 100% | 43/503，净−460 | 身份峰确实承载正证据，负对照成立 |

#### S2 动态多步矩阵

- candidate-gradient 50% × 3步：153/69，净+84，广覆盖主动作；
- role-confounder 100% × 3步：37/1，净+36，高精度补充；
- candidate-gradient 100% 全路径净负，正式禁用。

完整重算纠正了旧S1c汇总漏掉 r2–r5 的 bug：S1c完整单峰上限为553个错误，S1c+S2联合663个，即 +2.7768 pp；near可恢复574/1446。

#### S3A 六步动作

- candidate-gradient a=0.50 的累计净修正从 step1 +22 增至 step6 **+98**；
- role-confounder a=1.00 高精度，step5约24修正/0新增；
- role-unmatched只产生小幅净增；
- role-shared 尤其 a=1.00 从 step1净−370恶化至step6净−874，明确禁用。

1,718个独特新增错误中，99.91%发生在同分子式候选间，78.84%流向 MCES 0–2 near 候选。新增错误的实质是削弱本来正确的异构体特异证据。

S1c+S2+S3A outcome oracle 为799个错误，即 +3.346 pp。注意：这是动作空间上限，不是权重提升。

### 6.5 A4 全峰精确扫描与非线性动作教师

A4 对1,805个错误和3,193个匹配正确对照评估825,152个精确干预变体：

| 剂量 | 可恢复错误 | 受损正确对照 | 梯度—真实效应 Spearman |
|---:|---:|---:|---:|
| 25% | 138 | 183 | 0.794 |
| 50% | 289 | 375 | 0.732 |
| 75% | 461 | 652 | 0.652 |
| 100% | 738 | 1,078 | 0.374 |

A4精确动作可恢复776个，历史+A4 union 920，即 **+3.853 pp**。其中只有411个最佳动作位于梯度Top-1，说明“找最大梯度”会漏掉大量真实动作。

非线性动作教师在 formula OOF 中：

- correction ROC-AUC 0.849、AUPRC 0.132（阳性率1.742%，7.59×）；
- harm ROC-AUC 0.853、AUPRC 0.122（阳性率1.308%，9.34×）；
- margin Spearman 0.540；
- 40%覆盖：182修正/39新增，风险净收益104。

但它相对保守 confounder-only 的增量不显著，只新增19个历史未覆盖修正，未过80个新动作门。正确结论是“动作收益/风险可学习”，不是“模型已经提升”。

### 6.6 正例证据教师与动作头寸

C1 使用支持谱与评价正谱严格互斥的80,250个样本：

- accuracy 0.5664→0.5912，**+2.47 pp**；
- near +2.34 pp；
- 2,382修正/396新增；risk net 1,590；
- formula-cluster CI 全正。

这是强的正例教师上限，但仍是身份监督教师，不是可部署 shared embedding。

后续全图头寸逐步扩展：

1. 冻结 P/N union：922/1,805，+3.86 pp；
2. positive-guided intensity/consensus 新增223个独特错误，union 1,145；最佳固定 `consensus_projection@0.75` 为456/187、净269，固定直接增益+1.13 pp；
3. recurrent peak transfer 再新增112个，union 1,257，超过名义5 pp容量；最佳 `recurrent_union_mix@0.50` 为294/27、净267、固定直接增益+1.12 pp。

以上仍是冻结编码器动作结果/上限。它们证明训练监督空间足够，不保证学生能实现全部动作。

### 6.7 E1–E4 训练前证据链：为什么不是拍脑袋组合动作

在最终共享训练前，项目完成了三道不可省略的定量门：

1. **E1 真实采集变化标定：** 57,274个允许训练谱中构建3,412个 identity-adduct重复谱组、2,588个身份、1,957个分子式；1,738组跨条件；402,458个共识峰簇、1,211,271对匹配峰、104,952个可靠pairwise变体。噪声剂量由真实重复谱估计，不再用任意20%/30%删峰。
2. **E2 冻结动作矩阵：** 44个预注册cell（28 corrective、12 robustness、4 negative controls）；303,638个编码变体。18个cell过初门，经公式簇max-T多重性、匹配层级和阈值敏感性审计后保留14个，避免从大量动作中事后挑赢家。
3. **E3 梯度兼容性：** 14个cell聚为4个机制family、27,735条action rows、1,104身份、408分子式；16个family pair兼容、0个显式冲突。它只证明期望更新方向可以联合，不证明共享模型一定实现。

E4-M0随后冻结17,310个训练action target：acquisition-positive-gradient 7,851、candidate-gradient 7,206、role-confounder 2,253。早期E4-M1/M1b和F1 v1–v3表明“让学生追逐stop-gradient动作embedding”并不能自动提升clean检索，最终才转为E4-A的直接共享排序/噪声课程。这个过程解释了为什么E4-A是经过证据筛选后的结果，而不是线性叠加几个看起来正向的动作。

### 6.8 真正迁移进共享 embedding 的结果

#### E4-A：当前最可靠的权重结果

固定模型是一个共享的 query/reference DreaMS encoder：解冻最后 **1/7 Transformer block** 和官方 projection head，4 epochs；candidate-gradient使用50%衰减step3–6，role-confounder使用100%删除step1–5，identity-equal且每identity最多4 views。优化器扫描选择 backbone LR 2e−6、head LR 1e−5、clip1：单折 +0.540 pp、38/6、preservation0.99524。更激进 clip2 可达+0.591 pp但 preservation0.99453，故拒绝。

5 folds × 3 seeds：

| seed | overall | near | corrected/introduced | risk net | 最低fold preservation |
|---|---:|---:|---:|---:|---:|
| 20260828 | +0.611 pp | +0.508 pp | 183/37 | 109 | 0.995156 |
| 20260829 | +0.662 pp | +0.537 pp | 191/33 | 125 | 0.995140 |
| 20260830 | +0.632 pp | +0.522 pp | 185/34 | 117 | 0.995245 |

均值 **+0.635 pp overall / +0.522 pp near**；15/15 formula fold CI 下界>0。该结果确认噪声动作可以改变并改善统一 embedding，不是下游专家。

#### E5–E7：固定 guided 分支增量很小

- E5 N-only 单折 +0.5403 pp，38/6；
- intensity guided 最好点估计 +0.6078 pp，但相对 N-only 增量CI跨0；更强剂量降低 preservation；
- outcome-mined per-query 动作严重过拟合，最好仅 +0.0507 pp，显著差于 fixed control；
- recurrent transfer 权重扫描相对 fixed control最多多3个修正，但 preservation失败，未过多折门。

这说明“动作 oracle 很大”与“把动作标签直接喂给学生”之间存在可学习性鸿沟。

#### E8–E11：共享几何中的机制结论

- E8 `curriculum+symmetric+shared` 单折：+0.574 pp，near+0.609，38/4，MRR+0.386，preservation0.99527；
- stop-gradient action target无增益；冻结 official reference anchors有害；terminal-only有害；因此共享 query/reference 更新和多步 curriculum不可缺；
- E9 在线重挖与冻结动作近似，动作陈旧不是主瓶颈；
- E9B成熟N臂相对E8完整任务最多 +0.675 pp oracle；
- E10 `recurrent_union_mix@0.50` 在成熟几何上固定动作 +0.878 pp，59/7，CI全正；
- E10-B扩展P臂联合总 oracle +3.799 pp；
- E11参考多样性再加39个独特修正，成熟E8 held-fold总 oracle **+4.457 pp**，CI [+3.250,+5.862]，点估计仍低于5 pp。

这里的 +4.457 pp 是“成熟几何下还有多少可恢复错误”的上限，不能替代 E4-A 的真实 +0.635 pp 权重结果。

### 6.9 噪声线的最终科学认识

1. **有用噪声不是随机噪声。** 它是候选条件化的峰强度/缺失/混淆操作，并必须与匹配随机动作比较。
2. **negative excess 与 positive deficit 必须分臂。** 前者可通过 confounder/candidate-gradient 降低错误证据；后者需要真实同身份跨条件谱恢复稳定证据。
3. **强 shared-peak 删除是主要新增错误来源。** 大梯度、shared角色和100%删除均不能直接当标签。
4. **教师头寸大、学生增益小的原因已经定位：** 动作依赖候选和答案；共享编码器推理时只有一张 clean 谱，必须把条件化局部动作压缩成全局可泛化几何；大量动作在不同 query 上互相冲突，preservation 又限制可移动幅度。
5. **路线并未失败。** +0.635 pp 的多折多种子权重提升已成立；下一步不是继续扫学习率，而是扩展可跨身份复现的正例动作族、按机制分专家训练后再蒸馏进同一 encoder，并在新的盲测上裁决。

---

## 7. Embedding 后专家：从简单重排到 P2b

### 7.1 早期低置信重排器

在3,529条公式隔离困难确认查询上：

- 官方 DreaMS Top-1 0.8541；
- RAW+规则 0.8620，+0.79 pp；
- 低置信门控 RAW+规则 0.8629，**+0.88 pp**，公式bootstrap CI [0.60,2.68] pp；
- 规则面板相对RAW只多+0.14 pp，CI跨0。

这是最早的模型级正增益，但数据已用于开发，后续被更严格的 RAW/P2b 系列取代。

### 7.2 RAW reranker v1：开发大涨、盲测缩水

- g8r_val：0.8081→0.8516，+4.35 pp；44/17；coverage46.5%；
- 风险门控 P1：29/7，coverage8.06%，Recall0.8435；简单 disagreement-only 与学习门控风险净收益相同，说明门控没有学到超越“不一致”的新因素；
- 冻结原协议 Test-A：+0.45 pp，59/50，CI跨0，p=.444；
- Test-B：+0.73 pp，77/64，CI跨0，p=.312；
- A/B还重叠1,427个IK14，不能算两次独立复现。

教训：开发集 +4.35 pp 不是可迁移性能；必须候选协议对齐、工件冻结、公式隔离和一次性测试。

### 7.3 P2/P2b listwise rank fusion

P2 构建5,037-query、2,522身份、1,082公式、2,094 near 的完整候选组，单个 RAW 特征显示明显 headroom：neutral-loss sqrt cosine Recall@1 0.8972，高于官方0.8606。

P2b 冻结为一个简单但可审计的绝对 rank fusion：

`0.10 × DreaMS + 0.10 × entropy + 0.80 × neutral-loss sqrt cosine`

开发 nested formula OOF：

- Recall@1 0.8606→0.8997，**+3.91 pp**；
- MRR +2.50 pp；
- near +5.83 pp；
- 280/83，净197；
- formula bootstrap CI [+2.91,+4.87] pp；五个 outer fold overall/near均非负。

封存 P3：

| 面板 | DreaMS | P2b | Δ | corrected/introduced | 裁决 |
|---|---:|---:|---:|---:|---|
| main, n=3000 | 0.8793 | 0.8900 | **+1.07 pp** | 89/57 | CI正，McNemar p=.0101 |
| near-core, n=496 | 0.4879 | 0.4456 | **−4.23 pp** | 20/41 | CI全负，明确失败 |

P2b 的价值是常规候选上的强保底和中性丢失证据，不是最难异构体解决方案。最终系统必须在 near/高歧义子群回退官方 embedding 或使用新安全路由。

### 7.4 MTBLS13729 真实样本三路应用：首次把模型、专家和生物学管线接通

项目已在同一 precursor 候选图、同一 MS1–MS2 feature 连接和同一 feature 汇总规则下，完成：

1. 官方 DreaMS；
2. 实验性 E6 `fixed-v2-sw2` shared embedding；
3. 冻结 P2b 下游候选专家；

三路真实应用对照。这里的 E6 是**真实改变共享权重**的模型，但当前应用工件仅为 seed 20260828、formula fold 0 的单折实验模型；它不是 E4-A 多折多种子的最终外部模型。P2b 使用官方 embedding 后的 rank fusion，二者不可混称。

| panel | 有质量候选的 feature | 官方已分配 | E6已分配 | P2b已分配 | 官方/E6/P2b Level2a-supported |
|---|---:|---:|---:|---:|---:|
| neg_rp | 346 | 345 | 345 | 345 | 30 / 31 / 29 |
| pos_rp | 3,571 | 3,072 | 3,081 | 3,243 | 224 / 245 / 201 |

这组结果揭示了两个不同且有价值的行为：

- **E6 是保守的证据改善器。** 在 `pos_rp` 中只比官方多分配9个 feature，但 Level2a-supported 增加21个，总 Level2 证据 feature 增加37个；feature 层身份多数保留，证据层升级170、降级29，净升级141。它的主要价值不是大规模改名，而是使相同候选获得更稳定的跨样本谱学支持。
- **P2b 是覆盖扩张器。** `pos_rp` 分配由3,072增至3,243，增加171个、约 **+4.79 pp** 候选覆盖；但 Level2a-supported 由224降至201，总 Level2证据由517降至457。它能提供候选 lead，却不能把覆盖增加自动写成高置信注释增加。

三路 feature 共识为 `neg_rp` 256个、`pos_rp` 1,906个，合计 **2,162** 个。该共识可作为优先人工复核和下游 MS1 定量的稳定队列，但仍不是结构真值。真实应用没有标准品真值，因此所有模型差异只称“变化”或“新增候选”，绝不称“修正”。

C20:4 锚点同时展示了 E6 的实际作用：三路系统给出同一候选身份，官方 DreaMS 的最大/中位谱相似度约为0.8505/0.8091，E6 提高到0.8568/0.8166；它没有为了追求新名字而翻转候选，而是提高了19张支持谱之间的证据一致性。这是当前“共享 embedding 改进如何进入真实生物学工作流”的最直接实例。

---

## 8. BioAware：生化网络从“合理故事”变成可审计专家

### 8.1 v1 工程成果

BioAware v1 是 embedding 后专家，不改 DreaMS 权重：

- Rhea 17,656反应、78,843 participant rows、10,152 IK14；
- compound→reaction→compound 显式超图；
- leave-query-out、leave-truth-out、货币代谢物/高度数过滤；
- 度数保持 rewiring、冲突弃权、无证据回退 DreaMS；
- 每次干预保存 seed、reaction、contribution 路径。

这套工程首先保证网络不因数据库度数或真值泄漏产生假增益。

### 8.2 MTBLS1905 外部小面板

- 36 query、18目标、358候选；官方 DreaMS Top-1 0.750；
- evaluation-only leave-target-out：0.750→0.778，1修正/0新增；公式CI 0至0.0968，10个rewiring均0；门失败；
- 可部署自动种子只有1条、0条有效路径，DreaMS/BioAware均0.750。

结论：真实网络提供过一个合理修正，但独立种子饥饿使部署版本没有作用。

### 8.3 MTBLS13729 v1/v2

- v1：DreaMS 20/21，BioAware 19/21，0修正/1新增；新增是Rhea转氨路径把GABA改成2-氨基丁酸；
- expanded Rhea-only：1/1，净0；
- two-layer feature graph：只有2/21 query有路径，0/0、0干预；
- post-hoc hyperedge 完整反应侧规则可得1/0、21/21（+4.76 pp），但由已看错误启发，只是机制拟合，不能报正式性能。

这一负结果非常关键：生化邻接“合理”不等于当前 feature 的身份。网络必须与实验 feature graph、共同底物、共检出和候选特异证据同时成立。

### 8.4 MetDNA3/NetID 式重设计

在117-query外部 HILIC development benchmark：

- DreaMS 95/117=81.20%，22错误；
- 原始路径累加4/7，−0.366 pp；
- dependency-corrected hyperedge 4/2，+0.244 pp，CI跨0；
- formula OOF safe gate 4/7，失败；
- raw MS2 对 baseline error 的 truth>wrong仅13.6%，平均差−0.09697，CI全负；
- reported known MRN + observed MS1 + depth3 raw-MS2：3/0，+2.56 pp，但只2个身份、CI跨0；
- SMN 4/8、−3.42 pp；RT 0/3、−2.56 pp；reported+predicted eMRN 5/6、−0.85 pp。

旧证据实际修正并集7；包括SMN/RT truth-headroom的乐观并集11，低于+10 pp所需12。随后公式隔离指纹解码器和335核心规则 likelihood 对未解决错误新增5个候选头寸，使 consumed-development 实际可触达并集达到13，首次数学上超过12；但单独使用都会引入大量错误。

数据层本身也已实质扩展：16个HILIC mzML用统一OpenMS配方产生全MS1稳定feature图，约104,988个稳定feature-pair并归并为39,077个节点；可恢复645/751个Level-1结构、覆盖105/117 benchmark query。Mass-only eMRN的Level-1 truth覆盖从step0的52.7%升至step1的58.5%、step2的59.1%，但benchmark只从61/117升到63/117，证明“候选覆盖增加”不等于“候选排序变好”。递归raw-MS2 bridge在2-hop/3-hop仅覆盖少量独立身份，支持把后续重点放在候选特异全局一致性而不是继续堆网络深度。

G3-v1 nested formula OOF 完全弃权0/0，定位到原始证据量纲不一致；G3-v2 已改为候选组内归一化、机制族只计一票、至少两个独立证据族共识和风险弃权。**当前仍无可报告的冻结外部性能增益。**

### 8.5 BioAware 当前创新边界

真正可能形成论文创新的不是“网络扩散”，而是：

> **DreaMS-guided、evidence-factorized、risk-controlled global metabolite assignment**：把 DreaMS unary、真实 feature graph、reaction hyperedge、raw-MS2、RT/isotope/adduct、规则/峰token冲突作为独立因子，联合分配并允许弃权。

当前完成了网络工程、负对照、失败分解、候选头寸和量纲纠错；显著外部增益、全局分配和冻结验证尚未完成。

---

## 9. 生物学应用：从原论文模糊方向到冻结候选

### 9.1 MTBLS13729：为什么从鞘脂转向长链酰基肉碱

原作者的Rmu鞘脂线只有4个明确条目。患者内重算仅 sphingosine 名义p=.016，四条BH q=.064；Rmu/Rtu效应差异均不显著。因此鞘脂降为次线。

局部峰解析修复了固定EIC窗口混合相邻色谱峰的问题。长链酰基肉碱类别：

- 20个独立特征的PQN类别分数：Rmu-RN +1.35 log2，p=.0059；Rtu-RN +0.18，p=.617；交互+1.16，p=.0308；
- 漂移敏感性：Rmu +1.37，p=.0039；交互+1.17，p=.0311；
- 合并为17个链假设后：Rmu +1.32，p=.0078；交互降为p=.0694；
- 2,000个m/z/RT/检出率匹配随机面板中，真实效应超过全部随机，经验p≈.0005。

因此可写的是“Rmu相对癌旁的LCAC类别积累”，不可写“已确认黏液型特异”或“FAO通量下降”。

### 9.2 外部复核给出的边界

MTBLS8090 35对CRC中，24个预冻结LCAC条目的患者类中位数为−0.196 log2FC，Wilcoxon p=.273、置换p=.429。它不复现“泛CRC LCAC上升”，但无Rmu标签，不能否定Rmu特定现象。这个阴性结果阻止了过度泛化。

### 9.3 2026-08-29 冻结生物学结果

注释在看表型前冻结；官方 DreaMS、实验性 E6 `fixed-v2-sw2` 单折 shared embedding、冻结 P2b 三路对照；统一5 ppm EIC、12s apex，主终点为10对Rmu/RN。E6 承担共享表示的实验性复核，P2b-only 只作为正交候选队列，不承担主要身份结论。

| 面板 | 冻结目标 | 中位检出率 | 方向一致 | 名义候选 |
|---|---:|---:|---:|---:|
| neg_rp | 62 | 96.67% | 30 | 0 |
| pos_rp | 555 | 94.92% | 365 | 8 |

正离子8个候选中6个跨三种归一化过探索性FDR10，3个过FDR5。最重要的A层+FDR10候选：

1. feature3019，m/z312.130679，核苷相关候选，最小|log2FC|1.874，最大q=.0283；
2. feature1717，m/z230.185931，与 N1,N8-diacetylspermidine 的精确质量/分子式一致，最小|log2FC|2.448，最大q=.0872；2026-08-30 逆向审计确认当前接受的实验 MS2 桥接数为0，故不得再称 Level 2；
3. feature3222，m/z448.339463，C20:4 arachidonoylcarnitine-like，最小|log2FC|1.228，最大q=.0925。

完整8候选及其边界为：

| feature | m/z | 冻结候选/类别 | 证据层 | 最大q | 处理 |
|---:|---:|---|---|---:|---|
| 4966 | 180.087969 | 黄嘌呤样 | C | .0167 | FDR5，但保持异构体歧义 |
| 3019 | 312.130679 | Guanosine_Acetaldehyde样 | A | .0283 | FDR5，优先人工复核 |
| 1597 | 298.114285 | 核苷样未知物 | D | .0458 | FDR5，不使用数据库精确名称 |
| 7489 | 320.097232 | 核苷样异构体 | C | .0652 | Nelarabine名称不作内源身份 |
| 1717 | 230.185931 | N1,N8-diacetylspermidine精确质量/分子式候选 | 旧A标签已降级：无accepted实验MS2 | .0872 | FDR10丰度候选；标准品+MS2优先 |
| 3222 | 448.339463 | C20:4 arachidonoylcarnitine-like | A | .0925 | FDR10，最稳单分子锚点 |
| 3180 | 373.211422 | 含氯聚醚月桂酸酯样 | C | .1589 | 未过FDR10，污染/误注释优先 |
| 16425 | 502.287374 | LPE-like | A | .1614 | 未过FDR10，保留名义候选 |

#### 9.3.1 离子家族去冗余：六个显著 feature 不等于六个代谢物

表型盲全局峰网络确认 `feature 1597` 与 `feature 7489` 属于同一离子家族：二者符合 `[M+H]+/[M+Na]+` 质量关系，RT 相差2.96 s，质量残差0.0010 Da，跨样本丰度 Pearson 相关0.730，并在奇偶样本分半后均复现。因此二者必须合并报告为**一个核苷样未知离子家族**，不能分别计作两个代谢物，更不能把 Nelarabine 当作内源身份。

由此，当前6个通过主终点 FDR10 的丰度 feature 最多对应 **5个独立离子家族**。这是 feature-level FDR 后的描述性去冗余，不是重新计算的 family-level FDR。`feature 4966` 与 `1597/7489` 没有形成质量—RT峰网络边，只能在更宽泛的核苷/嘌呤模块中并列讨论，不能合并成同一离子家族。

### 9.4 C20:4 当前最稳锚点

- feature3222，59/59正离子样本检出；
- 三路系统给同一 InChIKey，19张谱/样本提供支持；官方最大/中位相似度0.8505/0.8091，E6为0.8568/0.8166；
- raw：log2FC+1.760，p=.00204，q=.0266；
- PQN：+1.228，p=.01099，q=.0925；
- PQN+paired drift：+1.256，p=.00977，q=.0861；
- 对应2.34–3.39倍升高；
- Rmu/Rtu interaction p=.0126–.0221但q=.60–.72。

身份仍为 Level 2。它支持“酰基肉碱稳态积累/脂肪酸利用异常假设”，不证明 β-氧化通量、CPT1A/CPT2/SLC25A20或具体因果酶。

### 9.5 MTBLS1905：重要但尚未完成的新发现线

外部已知目标面板：

- DreaMS Top-1 27/36=75.0%，Top-5 94.4%；
- 经典高分辨峰匹配 Top-1 31/36=86.1%，Top-5 36/36；
- 两者Top-1一致的23/36全部正确；
- 13个分歧中，DreaMS独占正确4，经典独占正确8。

DreaMS错误集中在1-methyladenosine/N6-methyladenosine、asparagine/glycylglycine、guanosine/crotonoside、pantothenate、cystathionine、guanine和kynurenine；DreaMS又能挽救经典方法在N6-methyladenosine、deoxyadenosine、carnosine和一张kynurenine谱上的错误。

该队列已经成为极性生物代谢物错误的真实外部压力测试和互补证据案例，但尚无达到预设门槛的“新增非S1实体 + 配对FDR<.05”结果，不能写成已完成的新生物学发现。

---

## 10. 把所有工作串成一篇完整科研故事

### 10.1 科学问题

DreaMS 在大规模预训练后具有强检索能力，但在非靶向代谢组学最重要的局部任务上仍面临三个瓶颈：

1. 同质量/同分子式近异构体共享主峰，错误候选被拉得过近；
2. 同分子跨仪器/碰撞能谱图缺失稳定正证据，被推得过远；
3. embedding 相似度缺少峰级化学解释和队列生化上下文，难以安全转为生物学结论。

### 10.2 我们的答案

1. **错误图谱：** 用 strict候选组、formula isolation和真实错误转换定位错误，而非盲目全局优化；
2. **因果峰动作：** 用目标动作与匹配随机动作证明哪些峰操作真正改变排序；
3. **定向噪声微调：** 把 negative-excess与positive-deficit分臂，训练同一共享编码器，已得到稳定+0.635 pp；
4. **正交候选专家：** P2b利用中性丢失等原始谱学信息，在P3 main增加+1.07 pp，但对near-core显式回退；
5. **双重映射：** 将全局概念、局部结构环境和具体峰连接，并用定向删峰验证输入忠实性；
6. **BioAware：** 将反应网络从“故事”改造成带leave-out、rewiring、实验图约束与弃权的证据专家；
7. **生物应用：** 表型盲冻结注释后再做患者内定量，产生C20:4、核苷样和多胺候选，同时明确Level2与静态丰度边界。

### 10.3 方法学创新的正确表述

单个组件都不是凭空首次出现：噪声增强、候选重排、代谢网络传播和线性探针均有前例。项目的创新潜力在联合设计：

> **error-conditioned, chemically auditable spectral representation learning**：以真实候选错误定义训练压力，以匹配反事实控制选择峰级动作，将安全动作蒸馏进共享 embedding；再用峰级双重映射解释表示变化，并用风险控制的正交谱学/生化专家在必要时补充而非覆盖模型。

要把“潜力”升级为论文主张，还需在一个新锁定外部基准上验证完整冻结系统，并证明双重映射至少有一批概念同时通过可解码、峰定位、忠实性和错误修正四门。

---

## 11. 当前能写进论文的结论、不能写的结论

### 11.1 已有充分证据

- DreaMS残余错误集中在近结构和采集条件漂移，不是均匀随机；
- 混淆峰删除与身份峰删除产生方向相反的因果效应；
- candidate-gradient与role-confounder动作具有跨分子式可重复信号；
- 固定预注册噪声动作可稳定改善共享 embedding（多折多seed +0.635 pp）；
- P2b在P3 main显著改善+1.07 pp，同时near-core显著退化，边界明确；
- embedding中存在可解码谱图概念和结构环境；两个峰因子完成输入忠实性验证；
- MTBLS13729的LCAC类别及C20:4候选在配对定量中稳健升高；
- BioAware全套工程、负对照、失败归因与两层证据框架已经建立。

### 11.2 尚不可写

- “新模型全面超过DreaMS/SOTA”；
- “3.85/4.46/5 pp动作上限已经被学生模型学到”；
- “P2b开发+3.91 pp就是最终泛化性能”；
- “规则库直接改善了embedding距离”；
- “双重映射已经解释Top-1决策”；
- “BioAware显著提高外部注释准确率”；
- “八个候选均为已确认代谢物”；
- “已证明Rmu特异、代谢通量下降或具体酶改变”。

---

## 12. 最短的论文完成路线

### 12.1 算法主线

1. 冻结当前最优 shared encoder 训练配方；
2. 完成剩余正例动作族的跨身份复现和机制分层蒸馏，不再扫无依据学习率；
3. 在未查看的新外部检索集一次性评价：official、new embedding、P2b-safe、combined；
4. 预注册 near-core回退/弃权，不允许main收益掩盖near退化；
5. 报告 corrected/introduced、formula/scaffold CI、跨仪器/CE、near/mid、生物分子类别和尾部preservation。

### 12.2 解释性主线

1. 对175条桥梁完成概念级目标删峰 vs 强度/mz匹配随机删峰；
2. 在真实修正/新增错误上验证概念方向是否与模型变化一致；
3. 只把同时通过可解码、峰定位、忠实性、检索方向四门的概念写入主文；其余进补充材料。

### 12.3 BioAware主线

1. 完成候选组内归一化、机制族共识和全局分配基线；
2. 在更大 identity-known 样本矩阵上冻结验证；
3. 网络只在至少两个正交证据族共同支持时干预，否则输出ambiguity/abstain；
4. 若外部准确率仍不增，BioAware降级为路径解释器，不影响算法主线诚实性。

### 12.4 生物学主线

1. 人工复核3019/1717/3222的MS2、诊断峰、加合物与异构体；
2. 将1597/7489合并为一个核苷样未知离子家族；4966保持独立黄嘌呤样候选，只在通路模块层面并列讨论，不使用不合理药物名；
3. 复核LCAC类别是否与C20:4单峰一致；
4. 原论文345个注释的全量对照已经完成；后续按“新离子家族、同名异峰、已知通路解释修正、身份降级控制”四类固化到主文和补充表；
5. 无标准品时坚持Level2和“丰度重编程假设”，不写通量因果。

---

## 13. 研究工件与文档覆盖索引

### 13.1 错误图谱、因果峰和早期模型

- [P0_failure_case_audit.md](./P0_failure_case_audit.md)
- [DREAMS_CAUSAL_RESIDUAL_AUDIT_DECISION_20260815.md](./DREAMS_CAUSAL_RESIDUAL_AUDIT_DECISION_20260815.md)
- [PEAK_EVIDENCE_STAGE_DECISION.md](./PEAK_EVIDENCE_STAGE_DECISION.md)
- [COUNTERFACTUAL_PEAK_FINETUNE_STAGE_REPORT.md](./COUNTERFACTUAL_PEAK_FINETUNE_STAGE_REPORT.md)
- [COUNTERFACTUAL_CPU_HEAD_AUDIT_20260814.md](./COUNTERFACTUAL_CPU_HEAD_AUDIT_20260814.md)
- [NOISE_FINETUNE_RED_TEAM_DECISION_20260821.md](./NOISE_FINETUNE_RED_TEAM_DECISION_20260821.md)
- [G8R_M1_POSTMORTEM_AND_M1B_DECISION_20260822.md](./G8R_M1_POSTMORTEM_AND_M1B_DECISION_20260822.md)

### 13.2 噪声微调正式演进

- [NOISE_V3_G1_RESULT_20260824.md](./NOISE_V3_G1_RESULT_20260824.md)
- [NOISE_V3_S1A_RESULT_20260824.md](./NOISE_V3_S1A_RESULT_20260824.md)
- [NOISE_V3_S1B_HEADROOM_RESULT_20260824.md](./NOISE_V3_S1B_HEADROOM_RESULT_20260824.md)
- [NOISE_V3_S1C_RESULT_20260824.md](./NOISE_V3_S1C_RESULT_20260824.md)
- [NOISE_V3_S2_RESULT_20260824.md](./NOISE_V3_S2_RESULT_20260824.md)
- [NOISE_V3_S3A_FORMAL_DECISION_20260824.md](./NOISE_V3_S3A_FORMAL_DECISION_20260824.md)
- [NOISE_V3_A4_FORMAL_RESULT_AND_A4B_DECISION_20260825.md](./NOISE_V3_A4_FORMAL_RESULT_AND_A4B_DECISION_20260825.md)
- [NOISE_V3_A4_ACTION_TEACHER_FORMAL_RESULT_20260825.md](./NOISE_V3_A4_ACTION_TEACHER_FORMAL_RESULT_20260825.md)
- [NOISE_V3_C1_CROSSFIT_TEACHER_RESULT_20260825.md](./NOISE_V3_C1_CROSSFIT_TEACHER_RESULT_20260825.md)
- [NOISE_E4A_OPTIMIZER_SCAN_RESULT_20260827.md](./NOISE_E4A_OPTIMIZER_SCAN_RESULT_20260827.md)
- [NOISE_E4A_HIGHLR_MULTIFOLD_RESULT_20260827.md](./NOISE_E4A_HIGHLR_MULTIFOLD_RESULT_20260827.md)
- [NOISE_FINAL_E5_GUIDED_SHARED_EMBEDDING_20260828.md](./NOISE_FINAL_E5_GUIDED_SHARED_EMBEDDING_20260828.md)
- [NOISE_FINAL_E6_OUTCOME_MINED_DIRECT_20260828.md](./NOISE_FINAL_E6_OUTCOME_MINED_DIRECT_20260828.md)
- [NOISE_FINETUNE_ACTIVE_CONTEXT_20260826.md](./NOISE_FINETUNE_ACTIVE_CONTEXT_20260826.md)

### 13.3 下游专家

- [RAW_RERANKER_AND_P0_DROPOUT_RESULTS_20260822.md](./RAW_RERANKER_AND_P0_DROPOUT_RESULTS_20260822.md)
- [RAW_RERANKER_NEAR_METRICS_RECONCILIATION_20260822.md](./RAW_RERANKER_NEAR_METRICS_RECONCILIATION_20260822.md)
- [RAW_RERANKER_V1_FINAL_VERDICT_20260822.md](./RAW_RERANKER_V1_FINAL_VERDICT_20260822.md)
- [g8r_p2_listwise_design_20260823.md](./g8r_p2_listwise_design_20260823.md)
- [P2B_RANK_FUSION_FORMAL_RECORD_20260823.md](./P2B_RANK_FUSION_FORMAL_RECORD_20260823.md)

### 13.4 规则、解释与双重映射

- [P1_noise_rule_preexperiment_decision.md](./P1_noise_rule_preexperiment_decision.md)
- [WEIGHTED_RULE_NOISE_TRAINING_PLAN_20260816.md](./WEIGHTED_RULE_NOISE_TRAINING_PLAN_20260816.md)
- [embedding_factor_discovery_pilot.md](./embedding_factor_discovery_pilot.md)
- [multilevel_factor_pilot.md](./multilevel_factor_pilot.md)
- [fragmentation_factor_pilot_v1.md](./fragmentation_factor_pilot_v1.md)
- [implicit_chemical_factor_full_experiment.md](./implicit_chemical_factor_full_experiment.md)
- [DOUBLE_MAPPING_STATUS_20260816.md](./DOUBLE_MAPPING_STATUS_20260816.md)
- [CHEMAWARE_FINETUNE_RESET_20260824.md](./CHEMAWARE_FINETUNE_RESET_20260824.md)

### 13.5 BioAware

- [BIOAWARE_V1_IMPLEMENTATION_AND_PILOT_RESULT_20260827.md](./BIOAWARE_V1_IMPLEMENTATION_AND_PILOT_RESULT_20260827.md)
- [BIOAWARE_MTBL13729_V1_FAILURE_AUDIT_20260828.md](./BIOAWARE_MTBL13729_V1_FAILURE_AUDIT_20260828.md)
- [BIOAWARE_V2_0_FEATURE_GRAPH_EXECUTION_20260828.md](./BIOAWARE_V2_0_FEATURE_GRAPH_EXECUTION_20260828.md)
- [BIOAWARE_METDNA3_DEVELOPMENT_RESULT_20260828.md](./BIOAWARE_METDNA3_DEVELOPMENT_RESULT_20260828.md)
- [BIOAWARE_10PP_NETWORK_HEADROOM_PLAN_20260828.md](./BIOAWARE_10PP_NETWORK_HEADROOM_PLAN_20260828.md)
- [BIOAWARE_LITERATURE_GROUNDED_REDESIGN_20260828.md](./BIOAWARE_LITERATURE_GROUNDED_REDESIGN_20260828.md)

### 13.6 生物学应用

- [MTBLS13729_BIOLOGY_PIPELINE_STATUS_20260820.md](./MTBLS13729_BIOLOGY_PIPELINE_STATUS_20260820.md)
- [MTBLS13729_ACYLCARNITINE_BIOLOGY_DECISION_20260820.md](./MTBLS13729_ACYLCARNITINE_BIOLOGY_DECISION_20260820.md)
- [EXTERNAL_BIOLOGY_REPLICATION_LEDGER_20260821.md](./EXTERNAL_BIOLOGY_REPLICATION_LEDGER_20260821.md)
- [MTBLS13729_FROZEN_P2B_APPLICATION_PROTOCOL_20260828.md](./MTBLS13729_FROZEN_P2B_APPLICATION_PROTOCOL_20260828.md)
- [MTBLS13729_FROZEN_BIOLOGY_RESULT_20260829.md](./MTBLS13729_FROZEN_BIOLOGY_RESULT_20260829.md)
- [MTBLS1905_HNSCC_APPLICATION_DECISION_20260821.md](./MTBLS1905_HNSCC_APPLICATION_DECISION_20260821.md)
- [MTBLS1905_HIGH_VALUE_BIOLOGY_EXECUTION_20260821.md](./MTBLS1905_HIGH_VALUE_BIOLOGY_EXECUTION_20260821.md)
- [MTBLS13729_MODIFIED_GUANOSINE_EXTERNAL_VALIDATION_20260830.md](./MTBLS13729_MODIFIED_GUANOSINE_EXTERNAL_VALIDATION_20260830.md)

### 13.7 2026-08-30 生物学新增闭环

- 对 `1597/7489/3019/8481` 完成峰解析 EIC 内的原始 MS2 审计，共链接 93 张谱；质子化主离子分别有 42 和 32 张 MS2，`132.042 Da` 核糖丢失支持率为 71.4% 和 100%。
- 两个 `[M+H]+/[M+Na]+` 家族的质量残差分别为 `0.001003/0.000465 Da`，并复现共同核糖丢失；这把身份证据推进到修饰鸟苷离子家族，但仍不区分位置异构体。
- 直接重分析独立 OEP00006137 40 对 Level-1 组织矩阵。四个唯一修饰鸟苷峰组成的模块在 MSI 中 `-0.778 log2`（15/20 下降，Wilcoxon `p=0.024`），MSS 中无显著变化；因此正式否定“泛 CRC 普遍升高”，把主线收敛为黏液型/异构体/分子背景依赖的 RNA 周转重编程候选。
- 已进一步下载并严格校验 OEP00006137 的 180 个可用 RPLC mzXML 原始归档。冻结坐标的 `5 ppm / ±15 s` 重提取中，M296T181、M296T200、M312T210 与作者补充矩阵的逐样本 Spearman 分别为 `0.993/0.901/0.994`，并复现 MSI-H 下降方向；这把外部证据从“重算作者表格”推进为“独立原始 MS1 重提取”。
- 7-methylguanosine 对 RT 漂移敏感：30 s 半窗恢复 12 个主分析零峰并保持 MSI-H 负方向，但宽窗会合并邻近的 m/z 296.1 峰，故只作为预注册敏感性证据。公开 pooled-QC mzXML 实测无 MS2，不能据此升级位置异构体身份。
- 对冻结的 HILIC methionine/guanosine/SAM/SAH 轴完成原始重提取。SAH 与作者矩阵 Spearman `0.997`，在 MSI-H 和 MSS 中分别为 `+2.403/+2.202 log2` 且 Wilcoxon 显著；methionine、guanosine 数值高度复现但无稳定肿瘤方向。SAM 原始峰仅检出 3 例，禁止计算 raw SAM/SAH ratio。
- SAH 与三个原始可复现修饰鸟苷峰的患者级相关不显著，故把 SAH 保留为并列的一碳/甲基供体产物池异常，不写成修饰鸟苷变化的上游因果。MSS 中 methionine 相关仅为探索性结果。
- MTBLS13729 本地 HILIC+ 的 SAH `[M+H]+` 精确质量全 RT 搜索未得到跨样本可复现峰簇（最高仅 2/60），因此明确禁止把外部 SAH 机制迁回本地 Rmu 队列。
- 外部矩阵同时显示 SAM/SAH 轴强烈依赖 MSS/MSI 背景，为一碳代谢异质性提供旁证，但不构成通量或甲基转移酶因果。
- Rmu 患者内，修饰鸟苷模块与独立 purine-like feature 4966 高度共变（raw Spearman `rho=0.903, p=0.000344`，三档 PQN `rho=0.879`），支持“修饰核苷/嘌呤周转轴”；其与多胺和酰基肉碱轴相关不显著，故保留为三条并列重编程轴，不构造事后总分。
- 2026-08-30 新增独立黏液型 CRC 组织蛋白组重分析：固定嘌呤合成/回收面板在 LMC/RMC/LNMC/RNMC 相对正常组织的中位 log2 比值均为正（`+0.42/+0.32/+0.25/+0.28`）；固定肉碱穿梭/长链 FAO 面板在 LMC/RMC 为 `-0.29/-0.43`，且 LMC/LNMC 与 RMC/RNMC 的 12/12 可检蛋白方向均为负。它支持“修饰核苷/嘌呤周转增强”和肉碱穿梭失衡中的利用受限/不完全氧化分支。源研究为池化 TMT，不能当作患者级显著性、通量或酶因果证据；2026 carnitine/acetylcarnitine–CPT1A 工作同时保留输入/利用增强这一竞争分支。
- 修饰鸟苷标准品优先级已收敛：最低两标为 m7G 与 m2²G；四标方案再加 m2G 与 Gm。原因是公开标准碎裂中 m7G/m2G 共享 `298→166`，Gm 偏 `298→152`，m2²G 为 `312→180`，正好对应本地 feature 1597 与 3019 的主要和次要碎片矛盾。完整执行门见 `docs/MTBLS13729_BIOLOGY_CLOSURE_AND_MINIMAL_VALIDATION_20260830.md`。
- 已完成高匹配外部资源 `GSE236696` 的本地重分析与逆向审计：6 对黏液型结直肠腺癌肿瘤/匹配癌旁单细胞转录组的 36/36 原始文件通过 GEO 字节核对，患者而非细胞作为统计单位。保守上皮门下，嘌呤轴 6/6 上升（均值 `+0.634`，两侧 `p=0.03125`），FAO 轴原计分 6/6 下降（`-0.567`，`p=0.03125`），但两个共同主终点的两侧 Holm 校正均为 `0.0625`。全基因表达匹配的 20,000 个随机轴审计进一步显示嘌呤效应幅度经验 `p=0.0113`、FAO `p=0.00020`，而修饰核苷 `p=0.275`；因此嘌呤/FAO 的基因集信号保留，修饰核苷转录轴降级。该门是基于公开 marker 的组成敏感性分析，不等同于源研究 Seurat cluster/CNV 恶性细胞标注。
- TCGA COADREAD 的 32 对肿瘤/正常样本进一步显示嘌呤轴升高（29/32，BH `q=3.34e-8`）和 FAO 轴下降（32/32，BH `q=1.16e-9`）是一般 CRC 程序；42 例黏液型与 329 例常规型肿瘤的协变量调整比较不支持两轴在黏液型中特异增强。高覆盖 MSI 敏感性模型（364/371 完整）中 FAO 差异接近零（beta `+0.010`, p=`0.911`），嘌呤在黏液型中反而较低（beta `-0.232`, p=`0.0091`）。因此现阶段不得使用“黏液型特异重编程已证实”，只能写“本地 Rmu 丰度现象与一般 CRC 程序及一组黏液型方向性证据一致”。
- 上述独立单细胞验证的完整统计设计、敏感性分析、反例和论文表述边界已单独固化在 `docs/MTBLS13729_GSE236696_EPITHELIAL_VALIDATION_20260830.md`。
- 2026-08-30 进一步完成乙酰化多胺机制轴的患者级单细胞和空间逆向审计。GSE236696 的 6 对黏液型病例中，冻结 broad epithelial gate 下多胺乙酰化/分解、酸性应答和趋化程序均为 6/6 上升；多胺乙酰化/分解变化与酸性应答变化的精确置换相关为 `rho=0.886, p=0.0333`。三种上皮门和表达/检出率匹配随机基因集复核后，酸性和趋化最稳，多胺乙酰化/分解为中等支持，多胺合成不稳。
- 源研究单病例空间数据 GSE236697 已从 12 个官方文件重建：肿瘤 3,481 个 QC spot、癌旁 1,725 个。肿瘤中酸性/趋化描述性升高、FAO 描述性降低；趋化与单核/巨噬 marker 的深度校正相关为 `0.273`。但多胺乙酰化/分解与酸性或趋化的深度校正空间相关仅 `0.010/0.040`，故正式否定“已形成空间连续因果链”的写法。
- 当前乙酰化多胺主线的正确强度是：feature 1717 为 Rmu 中强升高的 N1,N8-diacetylspermidine-like 候选，患者配对转录背景支持多胺乙酰化与酸性程序，但标准品身份、SAT1 因果和髓系募集机制仍未确认。完整裁决见 `docs/MTBLS13729_ACETYLATED_POLYAMINE_MECHANISM_AUDIT_20260830.md`。

### 2026-08-30 黏液型风险转录组新增审计

- GSE281917（140 例 MuC）与 GSE281918（119 例 NMuC）的 histology 与 GEO platform 完全重合，禁止把跨系列差异当独立 subtype replication，也禁止用 ComBat 声称解决完全共线混杂；
- GSE281917 内 MuC23 风险分数与 purine synthesis/salvage（partial-rank rho=-0.498）和 modified-nucleoside processing（rho=-0.310）负相关，与 polyamine acetylation/catabolism（rho=+0.262）正相关；
- 加入六类 broad-lineage marker 后，GSE 中仅 purine 轴保留（rho=-0.254，q=0.0139）；MuC23 与 fibroblast/endothelial 分数高度相关；
- TCGA 42 例黏液型病例中 purine 临床校正方向复现（rho=-0.508），但组成校正后不显著；polyamine 只形成次级方向性信号；
- 已生成本地代谢物丰度—GSE281917—TCGA 的风险背景综合图，明确把 direct abundance、risk-associated transcript context 和 causality 三层分开：`data/mtbls13729/mucinous_risk_context_figure_v1/`；
- 已建立与 AHCY、SORD、Gut lipidomics、ATF6–LCFA–microbiota 等机制论文逐项对标的 readiness scorecard：`docs/MTBLS13729_MECHANISM_READINESS_SCORECARD_20260830.md`。当前项目达到 evidence-calibrated clinical discovery / mechanism-supporting 层，未达到 isotope-tracing/perturbation/rescue 的 causal metabolism 层；
- 机器可审计的跨队列证据矩阵已扩展为 27 行，新增 GSE281917 和 TCGA MuC23 风险背景，同时保留“风险关联不是代谢物复现”的字段级边界：`data/mtbls13729/mechanism_evidence_matrix_v1/`；
- 已对 OmicsDI 冻结的 88 条 CRC metabolomics 记录做数据库级队列检索；53 个 MetaboLights 样本表和 15 个 Metabolomics Workbench factor tables 均完成审计。只有 MTBLS13729 自身含 10 个编码 `Rmu` 样本，没有找到第二个公开患者级 mucinous tissue metabolomics 队列。结果与边界见 `docs/MTBLS13729_EXTERNAL_MUCINOUS_METABOLOMICS_SEARCH_AUDIT_20260830.md`；
- 最高表述是 `risk-associated bulk transcript state`，不是独立预后因子、肿瘤细胞自主机制、代谢通量或候选酶因果。完整结果见 `docs/MTBLS13729_MUCINOUS_RISK_TRANSCRIPTOMIC_AUDIT_20260830.md`。

### 13.8 2026-08-30 原论文增量审计与投稿主线收敛

- 已通过 Figshare/ACS 补充材料完整获取并审计原论文 Table S4：345 条 UHPLC 注释，其中 Level 1 为157条、Level 2为188条；Rmu-vs-normal nominal 差异条目93条。原作者已覆盖24条 carnitine、13条 purine/nucleoside、6条 polyamine 和15条 LysoPE 上下文。
- 八个 DreaMS 候选与原作者表的严格 m/z+RT 匹配均为0。该结果证明它们不是原表同一 chromatographic feature，但不等同于新分子身份。
- 原作者已列 `N1,N8-Diacetylspermidine`（HILIC, Level 2），因此feature1717不能包装为首次发现该名字。其真实增量是：独立 positive-RP feature、73张峰界内MS2、跨色谱rank-1相关以及新的Rmu患者内强升高。
- 原作者已建立广泛 carnitine program，并在Rmu-vs-normal中报告11种carnitine。feature3222的增量仅是long-chain/C20:4-like类别锚点，以及结合外部FAO轴偏低所提出的“利用瓶颈”替代解释；不能声称首次发现肉碱代谢或已经证明FAO通量下降。
- 原作者表没有 methylguanosine/dimethylguanosine 离子家族。features1597/7489和3019/8481的峰界内MS2、132.042 Da核糖丢失与跨加合物一致性构成当前最强注释新增。
- 投稿准备度矩阵正式锁定：P0为1597、1717、3019；P1为3222、4966；7489仅作加合物支撑；3180与16425排除出主机制。主生物学结构为“修饰鸟苷/嘌呤共变主轴 + 乙酰化多胺平行轴 + 长链acylcarnitine平行轴”，不构造单一因果链。
- 论文级综合图已经生成：DreaMS候选效应、原作者已知carnitine program、独立 pooled mucinous proteomics、六对 epithelial pseudobulk 四层并列展示；标题和图注明确 static abundance/transcript/protein 不建立通量或黏液型特异性。
- 完整对账见 `docs/MTBLS13729_ORIGINAL_PAPER_DELTA_AND_NOVELTY_20260830.md`；论文主图、终点、标准品优先级和红线见 `docs/MTBLS13729_BIOLOGY_MANUSCRIPT_BLUEPRINT_20260830.md`。

### 13.9 2026-08-30 跨队列机制证据矩阵与最终生物学边界

- 新建 19 行、6 数据源的机制证据矩阵，其中14行独立于MTBLS13729发现队列。矩阵逐项分开 abundance、identity、pathway context、subtype replication 和 causality，并为每项写明允许与禁止结论。
- ST001087 的 formula-level N2,N2-dimethylguanosine 和 N1,N12-diacetylspermine 为正方向；OEP00006137 的 Level-1 N2,N2-dimethylguanosine 在 MSI/MSS 中为负方向。该正反证组合正式把修饰鸟苷主线定义为 species/isomer/context-dependent，而非泛CRC统一升高。
- TCGA 32对肿瘤–正常强支持 modified-nucleoside/purine 上升和FAO下降是一般CRC程序；42例黏液型与329例常规型的调整比较不支持 modified-nucleoside 或FAO在黏液型中特异增强。
- 当前最强机制模型为：Rmu中的高幅度修饰鸟苷/嘌呤、多胺乙酰化和长链acylcarnitine代谢池表型，嵌在更广泛CRC程序中；acylcarnitine积累支持肉碱穿梭失衡，但输入增强、利用瓶颈、不完全氧化和组织组成仍为竞争解释，不能证明FAO通量方向。
- 跨队列证据主图已输出为 `data/mtbls13729/crosscohort_mechanism_figure_v1/crosscohort_mechanism_evidence.png/.pdf`；矩阵为 `data/mtbls13729/mechanism_evidence_matrix_v1/mechanism_evidence_matrix.csv`。完整阐释见 `docs/MTBLS13729_CROSSCOHORT_MECHANISM_SYNTHESIS_20260830.md`。
- 已对账“有原始MS2”与“无accepted身份桥”的概念差异。feature1717有73张峰界内MS2、45个样本，m/z100.0759在100%谱中出现，并与使用 authentic standard 的N1,N8-diacetylspermidine `230.2→100.0` MRM转换一致；但没有同法标准RT/完整谱镜像，故升级为强diacetylspermidine-like候选，仍不称MSI Level 1/2精确身份。
- 已冻结 MassBank authentic-standard 谱对照：feature1597 对 7-methylguanosine 与 N2-methylguanosine 的最佳 sqrt-cosine 为0.6712与0.6667，分数差仅0.00450。结论是 methylguanosine family 相容、位置异构体不可辨；不得把1597直接写成m7G。
- 已逐谱读取五个主候选的碰撞能元数据：200张峰界内MS2均为30 eV，故证据是跨50个样本/30位患者的30-eV复现，而不是跨碰撞能复现。诊断离子支持为1597 30/42、3019 32/32、1717 73/73、3222 23/30、4966 16/23。

### 13.10 2026-08-30 MTBLS7387 外部人体脂质复算与最新机制标杆

- 已直接复算 Nature Metabolism 2025 ATF6 研究的 Fig. 3 人体来源矩阵：251 对完整 CRC 肿瘤–癌旁样本、186 个脂肪酸 feature；论文展示的 9 组配对 t 检验/BH-FDR 在来源表舍入允许范围内全部复现。
- 全 panel 有 56 个 FDR<0.05（38 升、18 降，52 个同时通过 Wilcoxon FDR）；C20–C24 有 17 个 FDR<0.05（14 升、3 降，16 个同时通过 Wilcoxon FDR）。这形成对“CRC 长链脂质重塑背景”的真实大样本人群支持。
- 该复现没有直接验证 feature3222：free arachidonic acid 不显著（`+0.0865 log2`, FDR `0.243`），两个 hydroxy-C20:4 峰显著升高；外部是 free/hydroxy FA，本地是 acylcarnitine-like ion。允许通路背景联系，禁止身份、通量和亚型因果迁移。
- 样本透明度已单独审计：论文 259 人、MetaboLights 258 tumour+258 adjacent、Fig.3 处理矩阵251对。标识对账提示净少7对并存在`315Tu2/315u`疑似别名，公开文件未给排除原因；复算一律称251-pair processed cohort。
- 最新对标加入 Gut 2026：152例发现+28例独立验证、Apc小鼠口服稳定同位素、无菌模型、CD36/CPT1A抑制和5例患者类器官。它将“静态长链脂质变化”升级为外源摄取与功能链，是3222轴真正需要补齐的机制层级。
- 2026 Oncogene 的1,257人功能代谢组又给出竞争性机制：carnitine/acetylcarnitine在高脂饮食AOM/DSS模型中可伴随CPT1A上调，CPT1A silencing或β-hydroxybutyrate–FXR干预可抑制表型。它不验证C20:4-LCAC，但足以禁止把3222积累单向写成“FAO利用下降”；当前冻结术语改为`carnitine-shuttle imbalance`与竞争性通量假说。
- 新图：`data/mtbls13729/mtbls7387_lcfa_context_figure_v1/mtbls13729_mtbls7387_lcfa_context.png/.pdf`；复算：`data/external/mtbls7387_paired_lcfa_replication_v1/`；样本审计：`data/external/mtbls7387_processed_pair_attrition_v1/`。

### 13.11 2026-08-30 三轴机制硬门与临床分层敏感性

- 已将修饰鸟苷/嘌呤、乙酰化多胺和长链酰基肉碱三轴逐项对标到结构终证、来源/通量、候选酶、表型和 rescue 五道机制门。完整矩阵见 `docs/MTBLS13729_THREE_AXIS_MECHANISM_BENCHMARK_20260830.md`。
- 2026 METTL1–m7G CRC/CRLM 研究证明 RNA m7G 可通过翻译调控、体内转移和 CCND3 rescue 形成因果链，但它没有证明本地游离 methylguanosine 离子来源；因此只能作机制背景，不能给 feature1597 命名或归因。
- 多胺外部证据必须按异构体分流：CRC 组织和尿液研究强支持 N1,N12-diacetylspermine，而既往组织质谱没有显示 N1,N8-diacetylspermidine 显著增加。它增强多胺轴合理性，同时提高 feature1717 标准品和异构体反证的优先级。
- 临床敏感性审计显示，Rmu 内四条轴在 dMMR 中均值均较高，但最小名义 p=`0.103`、BH 最小 q=`0.324`；不能把发现归因于 dMMR。限定 pMMR 后，Rmu 相对 Rtu 四轴均值仍全部为正，feature3222 差异约 `+1.53 log2`，精确 p=`0.0929`、四轴 BH q=`0.20`。这支持“并非显然完全由 MMR 构成造成”，但不建立组织学独立效应。
- BRAF+ 只有2例，禁止显著性推断；10个Rmu全部右侧，位置与组织学完全混杂。可复核产物为 `data/mtbls13729/clinical_axis_sensitivity_v1/`。
- 跨队列主图已重绘，明确 feature3222 为8/10同向、外部代谢物方向异质以及肉碱穿梭竞争机制，不再使用单向FAO瓶颈图注。

### 13.12 2026-08-30 扩展候选、四模块收敛与身份纠错

- 全空间 EIC、扩展峰界 DDA、原论文 source table 和同样本跨面板数据已合并为 15 候选冻结总账：9 个 source-identity remap、5 个强谱学家族候选、1 个降级控制；
- 5 个 same-mode RPLC 重映射峰的质量误差为 0.28–2.06 ppm，RT 偏差约 1.5–6.6 秒；myristoylcarnitine、isoleucine、phenylalanine 对应原论文 Level 1，N1-acetylspermine 和 methylthioadenosine 对应 Level 2；这些是原论文标准身份的再映射，不是本项目重新注射标准；
- carnitine、hypoxanthine、tryptophan 和 feature1717 获得同样本跨色谱/极性复核；taurine 谱学身份虽强，但跨面板生物学丰度不一致，正式降级；
- feature722 的弱 DreaMS synephrine 投票被原论文 Level-1 phenylalanine 的 m/z/RT 证据推翻，形成“深度模型增加候选覆盖、证据层级负责最终纠错”的真实案例；
- 四个 phenotype-selected 模块全部方向稳定：乙酰化多胺–MTA `+3.52 log2`（10/10）、嘌呤/修饰核苷 `+2.36`（9/9）、长链酰基肉碱 `+1.66`（10/10）、大中性氨基酸 `+0.94`（10/10）；四者均通过 leave-one-feature-out 方向稳定；
- 患者级协调只在 amino-acid–purine/nucleoside 上出现 rho `0.833`，六组比较 BH q `0.053`；acylcarnitine 与 purine/polyamine 近乎不相关。因此最新主线是四个并行 abundance programs，不是一条统一因果链；
- 新总账：`data/mtbls13729/integrated_biology_ledger_v1/`；新图：`data/mtbls13729/convergent_biology_figure_v1/`；完整结果：`docs/MTBLS13729_INTEGRATED_BIOLOGY_RESULT_20260830.md`。
- 新颖性分层已完成：15 个核心候选中 9 个是原论文身份重映射，1 个是原名存在但新增正交峰证据，4 个是原 identity table 未列出的算法候选家族，1 个因丰度正交失败被降级；source-linked 10 个节点与原表 Rmu 效应 Spearman `rho=0.830`，明确属于同队列技术一致性而非独立复现。可复核表在 `data/mtbls13729/biology_novelty_audit_v1/`。

### 13.13 2026-08-30 脯氨酸与 Neu5Ac 的正交找回及跨队列分化

- 正相 RPLC 新找回 proline feature345、glutamate feature374 和 Neu5Ac feature703；Rmu–RN 分别为 `+1.299/+0.715/+1.975 log2`，均 10/10 同向。三者分别映射到原论文其他色谱/极性的 Level-1 节点，不是新分子身份，但形成同样本正交 recovery；
- 跨面板患者配对变化 Spearman 分别为 `0.814/0.849/0.959`，对应 source feature 均为候选列表 rank 1。feature301 的 proline sodium-like 解释因质量/竞争库冲突被排除，feature1695 的 leucine-like 解释因患者配对丰度不一致被排除；
- TCGA 32 对显示 proline-synthesis 32/32 升高（BH `q=3.73e-9`），独立 pooled mucinous proteomics 中 ALDH18A1/PYCR1/PYCR2/OAT 在 LMC/RMC 均 4/4 升高；但 TCGA 黏液型相对常规型 proline axis 较低，故冻结为一般 CRC 程序，不称黏液型特异；
- GSE236696 上皮 pseudobulk 中 proline axis 5/6 上升、PYCR1 5/6 上升，但 20,000 次表达匹配随机集审计未通过（幅度经验 `p=0.228`）。单细胞只作方向背景，不作为主显著性证据；
- Neu5Ac 丰度增加与 TCGA 的转录方向呈分层：一般 CRC 中 sialic synthesis/transport、remodeling 和 mucin-sialylation 多数下降；黏液型相对常规型中 GNE/NANS/SLC35A1、ST3GAL4、ST6GALNAC1/2 和 MUC2/SPDEF 程序相对富集。最高表述为 `mucinous-relative sialic/mucin-glycan program`，禁止写全局 hypersialylation；
- 单病例 GSE236697 空间数据支持 tumour/goblet 区域 secretory-mucin program 和 CAF/collagen context，但不支持 sialic/proline 轴在 spot 分布中普遍升高；反证已纳入主结果；
- 新综合图为 `data/mtbls13729/proline_sialic_summary_figure_v1/proline_sialic_crosscohort_summary.png/.pdf`；完整边界和对标见 `docs/MTBLS13729_PROLINE_SIALIC_CROSSCOHORT_RESULT_20260830.md`。

### 13.14 2026-08-30 原文叙事审计与 TCGA 组织组成敏感性

- 对原文正文逐轴审计确认：Neu5Ac 与长链肉碱已被原作者明确点名；proline/glutamate 仅存在于 pathway/family context；acetylated-polyamine 未在正文展开。故 Neu5Ac/肉碱不能包装成首次发现，其增量是另一 LC 面板 raw-MS2/丰度桥和对 flux 叙事的纠偏；修饰鸟苷离子家族与乙酰化多胺仍是更强的新叙事节点；
- 原文在静态 abundance 数据上使用了 sphingolipid flux、increased sialic-acid conjugation 与 activated carnitine shuttle/FAO 等强表述。本项目冻结术语改为 abundance programs、selective remodeling 和 competing flux hypotheses；
- 新增 TCGA 371 例原发肿瘤的 broad-lineage sensitivity。每个代谢轴的 lineage score 都剔除与该轴重叠的基因，避免机械过校正。sialic synthesis/transport 在 clinical+lineage 模型中为 beta `+0.480`、BH q `2.64e-8`，MSI 完整病例中 beta `+0.448`、p `1.45e-7`；secretory-mucin program 为 beta `+0.922`、q `4.27e-11`；
- mucin-sialylation 轴 lineage 校正后衰减55%至 beta `+0.113`、q `0.1067`，提示 bulk 组成敏感；proline-synthesis 保留负方向但为边界 q `0.0842`；glutamate-supply 较低保持 beta `-0.296`、q `0.00235`；
- 由此，当前最稳黏液型相对转录背景是 sialic precursor synthesis/transport + secretory mucin，而非统一的 global hypersialylation；lineage scores 仍只是粗粒度组成代理，不建立细胞来源或通量；
- 新工件：`data/mtbls13729/source_narrative_audit_v1/`、`data/external/TCGA_COADREAD_Xena_20260830/proline_sialic_lineage_sensitivity_v1/`；实现：`tasks/audit_mtbls13729_source_narrative_v1.py` 与 `tasks/audit_tcga_proline_sialic_lineage_sensitivity_v1.py`。
- 统计分母已重新对账：6个FDR10/3个FDR05属于冻结的555个 phenotype-blind positive-RP annotation targets；完整13,155-target discovery-matrix审计没有全空间FDR10 feature，只有132个nominal exact-gate feature。后续不得把候选面板FDR写成全非靶向空间确认。

### 13.15 模块匹配背景与竞争机制树（2026-08-30）

- 在 1,018 个非候选正相 feature 中，按 m/z、RT、检测率、ion-family size 和可选 MS2-support 建立三套 phenotype-blind 匹配背景；每套每模块 50,000 次抽样且禁止离子家族重复；
- acetylated-polyamine–MTA、purine/modified-guanosine、long-chain acylcarnitine、expanded amino-acid 和 Neu5Ac 五个模块在三套匹配定义中均保持描述性上尾比例 `<0.05`；
- 该结果只排除简单 acquisition/background 作为全部解释，不修正同队列 post-selection，不能作为 confirmatory p 值或独立复现；
- 机制解释已拆成 5 轴 14 个竞争生成过程，详见 `docs/MTBLS13729_COMPETING_MECHANISM_TREES_20260830.md` 与 `data/mtbls13729/competing_mechanism_trees_v1/`；
- 论文必须并列 writer-vs-turnover、host-vs-biofilm、FA entry-vs-incomplete oxidation、free Neu5Ac-vs-glycan linkage 等分支，不得用 bulk transcript context 直接选定其中一条。

### 13.16 2026-08-30 亚型主轴收敛与机制论文完成度审计

- 五模块 exact-permutation interaction 审计把 `Rmu-RN` 主效应与 `(Rmu-RN)-(Rtu-RN)` 亚型敏感性严格分开。只有 Neu5Ac 在 raw 与 PQN 两套归一化下通过五模块 BH 校正：交互差分别为 `+2.209/+2.142 log2`，q=`0.00179/0.00162`；
- 17 个正相冻结候选的逐节点复核得到相同结论：feature703 是唯一兼具充分覆盖、source-Level-1 正交身份桥、Rmu 升高与候选面板 subtype q<`0.01` 的节点；
- claim scorecard 将18个节点收敛为1个 `PRIMARY_SUBTYPE_ANCHOR`、9个 `GENERAL_TUMOUR_SUPPORT`、1个 `FAMILY_VALIDATION_PRIORITY`、5个 `LOW_COVERAGE_IDENTITY_VALIDATION`、1个 `CONTEXT_ONLY` 与1个 `NEGATIVE_CONTROL`；
- 完整13,155-target正相空间仍为0个 exact-FDR10 feature。故当前主轴是 discovery-level 的 `mucinous-relative Neu5Ac/mucin-glycan remodeling`，不是全空间确认、global hypersialylation、glycan linkage、flux 或酶因果；
- 机制论文完成度总账已由12门升级为16门：配对终点、raw-data复核、亚型发现、丰度协议对账已通过；外部结构糖组和TCGA分支解耦为带边界支持；MS2同法身份、独立代谢组复制、同一样本糖链去向与因果干预仍未通过；BioAware v1 为0修正/1新增的明确负结果。当前工件见 `data/mtbls13729/mechanism_paper_completion_audit_v2_final/`；
- 当前可投稿定位冻结为 **algorithm-enabled, evidence-calibrated biological discovery**。若强化生物学主轴，最短补强是 Neu5Ac 同法标准/spike-in、linkage-aware glycan readout 与一份独立患者级丰度复现；若这些不可获得，标题与摘要不得使用 causal mechanism 或 flux reprogramming。
- 2022 CRC PGC-LC-MS/MS O-glycomics 补充表提供了新的独立结构层支持：按权威 Table S2，两个 MUC 病例 T2/T3 的 core-2 和 sialyl-Lewis X/A 在 11 个 AC/MUC 原发肿瘤中均排名第1/第2，α2-6 sialylation 则排名倒数第2/第1；T2-C2、T3-C3 的 core-2、sLeX/A 和 core-2+α2-3 均同向上升，α2-6 均大幅下降。该结果把主轴收窄为 `core-2/sLeX/A expansion with α2-6 loss`，但只有2例MUC，属于 independent structural support，不是 free Neu5Ac abundance replication。字段级审计见 `docs/MTBLS13729_EXTERNAL_OGLYCOMICS_MUCINOUS_AUDIT_20260831.md`。
- Neu5Ac 丰度协议已完成最终对账：锁定 targeted-EIC 在 Rmu 为10/10正向、均值 `+1.935 log2`；`log2(EIC+1)` 跨面板审计为10/10、`+1.975 log2`；早期 discovery peak-picker matrix 因 P24 缺失为9/9、`+1.881 log2`。主结果采用前者，后者只作缺失阈值敏感性；禁止跨协议拼接 n、均值和显著性。三层整合主图及源数据见 `data/mtbls13729/neu5ac_glycan_publication_figure_v1/`。
- 新增独立患者组织空间背景：Jain 等 2024 年 372 对 CRC–匹配正常黏膜的官方补充表将 Neu5Ac 定为 HILIC(-) Level 1（m/z `308.0980`、RT `355.7 s`），并显示正常黏膜 cecum-to-rectum 斜率 `+0.349`、p`<0.001`，肿瘤斜率衰减为 `+0.088`、p=`0.091`。该研究没有 mucinous 字段，故只支持 disease-dependent spatial context 和 location-aware analysis，不构成 Rmu 独立丰度复制。其补充分期单元格合计374、与方法报告372不一致，已显式冻结为源文件数据质量边界。审计见 `docs/MTBLS13729_EXTERNAL_NEU5AC_BIOGEOGRAPHY_AUDIT_20260831.md` 与 `data/external/CRC_metabolic_biogeography_PMC11438248_20260831/neu5ac_biogeography_audit_v1/`。
- 进一步冻结了上述研究公开 Dash 回调中的 Neu5Ac 绘图值：normal/tumour 各371个、七个亚部位均固定53个、无患者或配对键；网页回归的标准化系数为 `0.391/0.179`，不能复现补充表的 `0.349/0.088`，尤其 tumour 网页回归显著而正式补充p=`0.091`。故网页只作方向性可视化背景，正式外部统计继续以补充材料为准。审计见 `docs/MTBLS13729_EXTERNAL_NEU5AC_DASH_REPRODUCIBILITY_AUDIT_20260831.md`。
- 新增外部sialyltransferase–histology背景：2026年整合TCGA、Sidra-LUMC、CPTAC-2的公开补充表中，980例有histology，Sialyl-High占mucinous `85/154`、non-mucinous `238/826`；重算OR=`3.043`、95% CI `[2.142,4.325]`、Fisher p=`6.58e-10`。该score只含20个sialyltransferase genes，不测Neu5Ac/糖链/通量，并与本地TCGA分析部分重叠；因此只作为外部转录背景，不构成独立丰度复现。审计见 `docs/MTBLS13729_EXTERNAL_SIALYLOME_MUCINOUS_AUDIT_20260831.md`。

### 13.17 2026-08-31 Neu5Ac 主轴收敛为 hybrid mucin glycome

- 在 TCGA COAD/READ 的42例黏液型与329例常规型原发肿瘤中，按独立 O-glycomics 生物合成层次冻结分支并调整临床变量、非重叠 broad-lineage proxies 与 MSI：Neu5Ac donor supply/transport beta `+0.480`、q=`3.30e-8`，secretory-mucin beta `+0.922`、q=`5.34e-11`，core-3/Sda mucosal-lineage beta `+0.879`、q=`1.76e-8`；
- 相反，core-2/sLeX transcript composite 接近零，alpha2-3 O-glycan sialylation beta `-0.439`、q=`0.0093`，ST6GAL1 beta `-0.742`、q=`8.50e-5`；ST6GALNAC1 与 GCNT3 分别显著正向。该结果反对一条统一的高唾液酸化转录程序；
- 外部 MUC O-glycomics 解决了参照系表面矛盾：两例MUC的core-3在肿瘤间排名最高，但相对各自正常组织仍下降约34%；同时core-2、sLeX/A和core-2+alpha2-3配对升高、alpha2-6下降。因此core-3表示黏膜分泌谱系的相对保留，不是肿瘤转化中的绝对增加；
- 当前最小充分模型冻结为 `hybrid mucin glycome`：扩大/重分配的 free Neu5Ac pool 和 donor/secretory-mucin program，与 core-3/Sda 相对保留、core-2/sLeX TACA 获得及 alpha2-6 丢失共存。论文创新点是 donor–carrier–core–linkage 解耦，不是首次发现 Neu5Ac 或 CRC sialylation；
- 算法—生物学归属已单独冻结：feature703 是原论文 Level-1 Neu5Ac 在正相 RPLC 的正交找回，不是 E6/P2b 新身份；E6 的真实应用增量偏向证据稳定性，P2b 偏向候选覆盖。三路 feature-level 表尚未同步到本地，因此重点候选的逐模型归属不得从汇总日志反推。详见 `docs/MTBLS13729_ALGORITHM_TO_BIOLOGY_INCREMENT_AUDIT_20260831.md`；
- 详细审计见 `docs/MTBLS13729_NEU5AC_HYBRID_GLYCOME_AUDIT_20260831.md` 和 `data/external/TCGA_COADREAD_Xena_20260830/glycan_branching_v2/`。最短升级仍为同法 Neu5Ac 标准/spike-in/内标加 linkage-aware O-glycomics；因果升级另需 CMP-Neu5Ac、MUC2 glycopeptide、示踪、扰动和 rescue。
- 新主图 `data/mtbls13729/neu5ac_glycan_publication_figure_v2_final/neu5ac_hybrid_glycome_figure_v2.png/.pdf` 已通过视觉复核；源表和哈希与图同目录，旧 v1 保留为审计历史，不再作为优先主图。

### 13.18 2026-08-31 同患者free Neu5Ac—activated donor解耦

- 直接重建原始HILIC(-)补充表的10位Rmu患者配对值：Level-1 free Neu5Ac在10/10患者中升高，平均`+2.249 log2`，bootstrap 95% CI `[1.641,2.866]`；Level-2 CMP-Neu5Ac和Level-1 UDP-GlcNAc分别为`+0.556/+0.327 log2`，均不显著；
- 两个预设的患者内差值均通过：free Neu5Ac相对CMP-Neu5Ac与UDP-GlcNAc分别高`+1.693/+1.922 log2`，bootstrap下界均大于零，Holm-Wilcoxon p均`0.0273`；这不是用“一个显著、一个不显著”推导差异，而是直接检验患者内变化差；
- 因而当前主轴新增了same-patient `free-pool-to-activated-donor decoupling`证据，反对把GNE/NANS/SLC35A1转录富集直接等同于CMP-Neu5Ac扩增；但它仍是同队列静态丰度，CMP-Neu5Ac为Level 2，不能区分合成、回收、释放、摄取、转运或glycan incorporation；
- 完成度总账升级为20门的`data/mtbls13729/mechanism_paper_completion_audit_v6_final/`；Package A保持可投稿，独立Rmu丰度复现、同法标准、同一样本glycan destination与因果实验仍缺失。审计见`docs/MTBLS13729_SIALIC_DONOR_DECOUPLING_AUDIT_20260831.md`。

### 13.19 2026-08-31 free Neu5Ac来源机制的预定义分支裁决

- 在同一TCGA资源上预先冻结de novo supply、CMP activation/transport、NEU1/NEU3 release和CASD1−SIAE O-acetyl protection四轴，并统一在15个gene/axis outcome内做BH；模型沿用临床、六类非重叠lineage proxy和MSI校正；
- 32对一般CRC tumour-normal中NEU1/NEU3 release轴显著升高（平均`+0.854 z`、BH q=`9.02e-7`），但42个mucinous相对329个conventional中显著降低（lineage beta=`-0.691`、q=`5.58e-6`；MSI beta=`-0.654`、q=`1.53e-5`）。因此mucinous free Neu5Ac不能由NEU1/NEU3转录上调简单解释；
- CMP activation/transport在mucinous中相对升高（lineage beta=`+0.449`、q=`1.61e-4`），主要由SLC35A1而非CMAS驱动，但实测CMP-Neu5Ac不升，形成明确的RNA-capacity与metabolite-pool失配；
- CASD1−SIAE signed O-acetyl balance未通过统一校正；该阶段使用的旧表达矩阵尚未覆盖NXPE1，因此当时只保留为未决机制。此数据缺口后来已由current-GDC补齐，最终裁决以13.23为准：NXPE1的mucinous相对效应由分布式secretory-mucin carrier state解释，并不构成独立酶驱动或O-acetylation通量证据；
- 完成度总账升级为21门的`data/mtbls13729/mechanism_paper_completion_audit_v7_final/`。详细审计见`docs/MTBLS13729_SIALIC_POOL_MECHANISM_DISCRIMINATION_20260831.md`。

### 13.20 2026-08-31 mono-O-acetyl-Neu5Ac-like精确质量负结果

- 对negative-HILIC的60个原始mzML在`m/z 350.109269 [M-H]-`实施表型盲全RT发现；早期
  complete-linkage会把RT漂移峰带错误切成14簇，已废弃。修正为每样本等权总体共识峰并强制
  18秒峰间距后，只冻结4.29和5.55分钟两个独立峰，分别由50/60与54/60样本支持；
- 两个峰分别有47和56张RT分层DDA MS2，`m/z 87.0088`见于47/47和54/56张谱，说明存在稳定
  精确质量/碎片家族；但精确质量和该碎片不能区分4/7/8/9-O-acetyl-Neu5Ac，身份仍为`-like`；
- 两个峰的Rmu完整配对效应均不稳定，BH q均`0.930`；floor敏感性仅4/10和5/10为正，患者变化
  与Level-1 free Neu5Ac的rho为`0.170/-0.067`。因此不支持bulk mono-O-acetyl-Neu5Ac-like pool
  随free Neu5Ac同步升高；
- 该负结果只反对简单统一pool，不排除glycan-bound、空间/细胞型特异O-acetylation或
  NXPE1/CASD1/SIAE蛋白活性。完成度总账升级为22门的
  `data/mtbls13729/mechanism_paper_completion_audit_v8_final/`；完整审计与图见
  `docs/MTBLS13729_OACETYL_NEU5AC_LIKE_AUDIT_20260831.md`和
  `data/mtbls13729/oacetyl_neu5ac_like_figure_v1/`。

### 13.21 2026-08-31 PXD055865 MUC2载体层与O-acetyl标准谱库边界

- 已下载并逐表审计Nature Communications 2026 MUC2空间糖肽研究PXD055865的
  Supplementary Data 2与source spectra；三块黏液癌标本实际来自两位独立患者，Colon1a/1b
  属于同一患者，另有一份healthy colon，后续禁止写成三位独立患者；
- 去重后Colon1a/1b/2分别有439/390/451条MUC2糖肽，healthy colon为21条；公开表和source
  spectra确认存在sialylated、O-acetyl-Neu5Ac及putative O-acetyl-GalNAc MUC2证据。但发现
  深度、切区和标本数不均，鉴定数只能作存在性审计，绝不能解释为肿瘤-正常丰度；
- 该资源不测free Neu5Ac，因此不是MTBLS13729丰度复制；它补足的是carrier/destination层：
  MUC2 glycoform具有空间异质，肿瘤可呈低唾液酸化/非唾液酸化载体与O-acetyl destination
  重排，方向上支持free-pool与carrier-specific destination解耦；
- 标准谱库审计显示HMDB0000794仅有预测谱，MassBank/MoNA按精确名称/分子式未找到可直接复用
  的实验记录；公开IM-MS合成标准研究表明4/7/8/9-O-acetyl位置及linkage需要标准品加CCS，
  普通exact mass与常规MS2不足以定位；
- 因而主论文最小采购优先级保持为普通Neu5Ac同法RT/MS2/spike-in/同位素内标；若专门追踪
  O-acetyl位置，至少成对购买4-O与9-O标准，必要时转IM-MS/CCS。O-acetyl-like游离峰当前是
  机制反证，不应抢占主阳性验证预算；
- 完成度总账升级为23门的`data/mtbls13729/mechanism_paper_completion_audit_v9_final/`；字段级
  审计见`docs/MTBLS13729_PXD055865_MUC2_GLYCOPEPTIDE_AUDIT_20260831.md`，标准/谱库缺口见
  `docs/MTBLS13729_OACETYL_NEU5AC_STANDARD_RESOURCE_AUDIT_20260831.md`。
- 新增可投稿Extended Data候选图`data/mtbls13729/pool_carrier_boundary_figure_v1/`，把本地
  free Neu5Ac/CMP-Neu5Ac/UDP-GlcNAc患者内变化、两个O-acetyl-like游离峰阴性结果和外部
  MUC2鉴定存在性分为三个独立panel，图内直接写明PXD055865不是丰度比较。

### 13.22 2026-08-31 PXD055865完整Source Data与图示定量边界

- Nature Communications页面实际提供8个补充文件，现已全部下载并审计；MOESM8解压后只有258张MALDI PNG，覆盖主图1/2及8个补图目录，没有数值矩阵，因此只能证明图像来源可追溯，不能重算患者级丰度；
- MOESM1补图20、22–25的图内normalized-level标签已人工转录并结构化。单个健康结肠的AcNeu5Ac/Neu5Ac、Ac2Neu5Ac/Neu5Ac、Ac3Neu5Ac/Neu5Ac图示比值分别为`11.91/44.33/182.17`；两个CRC患者的4个肿瘤区域中，前两项范围为`0.0094–1.0000`和`0.0042–2.2798`，显示很强的患者内外异质性；
- Colon1a/1b来自同一患者，Colon2是第二位患者，健康结肠只有1位供者；NL是图示强度标尺而非校准浓度/统一峰面积，禁止做群体显著性或称独立丰度复制；
- 新证据强化而非推翻`hybrid mucin glycome`：free Neu5Ac pool、activated donor、MUC2 carrier、O-acetyl destination与core/linkage必须分层解释。PRIDE另公开约3–4GB关键RAW，可进一步统一提取fingerprint-ion XIC，但仍不能克服只有2位肿瘤患者的样本量边界；
- 详细审计见`docs/MTBLS13729_PXD055865_SOURCE_DATA_QUANTITATIVE_AUDIT_20260831.md`与`data/external/PXD055865_2026_MUC2/source_data_audit_v1/`。

### 13.23 2026-08-31 NXPE1与pool–carrier–O-acetyl destination最终裁决

- 使用current-GDC STAR TPM和FPKM-UQ，在原先严格锁定的371例TCGA COAD/READ原发肿瘤
  （42 mucinous、329 conventional）中复核NXPE1；TPM临床+lineage模型beta=`+0.621`、
  p=`0.000369`，加入MSI后beta=`+0.530`、p=`0.00134`，FPKM-UQ方向和显著性一致；
- 加入预定义secretory-mucin程序`MUC2/TFF3/SPDEF/FCGBP/AGR2`后，NXPE1效应降为
  beta=`+0.064`、p=`0.734`（再加MSI为`-0.048`, p=`0.782`）。故NXPE1不是已证独立驱动，
  而是secretory-mucin carrier-linked O-acetylation-capacity marker；
- 在current-GDC 50对一般CRC肿瘤/正常中NXPE1为47/50肿瘤较低，TPM平均差`-2.709 log2`；
  因此“一般CRC相对正常下降”与“mucinous相对conventional保留/富集”可同时成立；
- secretory程序结论通过完整leave-one-out复核：TPM和FPKM-UQ中删除任意一个
  `MUC2/TFF3/SPDEF/FCGBP/AGR2`标志后，NXPE1 mucinous系数仍全部不显著；所有双标志调整也
  不显著，说明衰减来自分布式secretory carrier state，而非单个MUC2共线性；该结果仍是
  covariate sensitivity，不能写成因果中介；
- GSE236696六对黏液型CRC的保守上皮patient-level pseudobulk中，NXPE1为6/6肿瘤低于癌旁，
  平均`-1.084 log2`、bootstrap 95% CI `[-1.707,-0.484]`、精确双侧`p=0.0625`。该基因低计数，
  且公开feature index在12个样本中均缺失MUC2，因此只作“一般肿瘤相对正常下降”的方向支撑，
  不能验证mucinous-vs-conventional，也不能把MUC2技术缺失写成零表达；
- 文献机制表述已纠偏：2025两项原始研究分别支持free Neu5Ac和CMP-Neu5Ac acceptor context，
  禁止把本地Level-1 free Neu5Ac直接指定为体内NXPE1底物；
- 原始补充材料全表审计未发现命名的mono/di/tri-O-acetyl-Neu5Ac或对应精确质量行；本地
  negative-HILIC虽有两个`m/z 350.109269`稳定峰和103张RT分层MS2，但二者均不随Rmu或free
  Neu5Ac同步，继续作为位置未定的`O-acetyl-Neu5Ac-like`机制反证；
- 当前最小充分模型最终收敛为四层解耦：free pool expansion、activated donor non-expansion、
  secretory MUC2/NXPE1 carrier state、core/linkage/O-acetyl destination remodelling。创新点是
  pool–carrier–destination decoupling，不是global hypersialylation或O-acetylation flux；
- 完整审计见`docs/MTBLS13729_NXPE1_POOL_CARRIER_OACETYL_MECHANISM_AUDIT_20260831.md`，
  机器结果见`data/external/TCGA_COADREAD_Xena_20260830/nxpe1_free_donor_v3_secretory/`。

### 13.24 2026-08-31 独立黏液型CRC蛋白组固定面板审计

- 从GSE178341黏液型重分析论文的患者级补充蛋白矩阵中冻结15例mucinous、15例conventional
  CRC和16例normal，主比较严格设为MC-vs-AC；预设8个可测蛋白为
  `AGR2/MUC2/TFF3/FCGBP/GNE/NANS/CMAS/SIAE`，另有`NXPE1/SPDEF/SLC35A1/CASD1`
  在矩阵中不可测，未以替代蛋白补位；
- 两个预设模块均未确认：secretory/mucin与sialic biosynthesis/handling的MC-vs-AC差异均约
  `+0.21 z`，bootstrap区间跨零，模块BH q均`0.449`；因此禁止写成独立通路复现；
- AGR2、GNE、NANS分别为`+0.897/+0.565/+0.502 log2`，且15/15 leave-one-MC-out方向为正，
  但患者bootstrap区间跨零且8蛋白BH q均`0.643`。CMAS/SIAE无正支持，符合选择性而非整条通路
  统一激活的模型；
- 原论文AGR2的`MC_AC_up`可在未转换原始尺度复现：算术均值FC=`2.597`、Welch p=`0.0478`；
  冻结log2+置换口径为p=`0.161`。故只能称方向稳定、显著性尺度敏感；
- 蛋白矩阵有明显左删失/填充值：TFF3为37/46全表最小值，MUC2非floor敏感性甚至反向，
  不能把这些蛋白的均值差包装成确认；机器结果和审计见
  `data/external/GSE178341_mucinous_secretory_audit/independent_proteomics_fixed_panel_v1/`与
  `docs/MTBLS13729_INDEPENDENT_MUCINOUS_PROTEOMICS_AUDIT_20260831.md`；
- GSE178341原始10x UMI患者级pseudobulk被保留为更高优先级的转录细胞来源裁决；分析前已冻结
  `docs/GSE178341_NXPE1_MUCINOUS_PREREGISTRATION_20260831.md`和
  `docs/GSE178341_SIALIC_CELL_SOURCE_PREREGISTRATION_20260831.md`，不得在看到raw outcome后
  增删基因或细胞类型。

### 13.25 2026-08-31 GSE178341原始UMI患者级细胞来源裁决

- 官方1.203GB 10x H5已按NCBI公布长度和SHA256
  `f435bb2651ff5297d0c24a99daf58850ed67ae1ed6c5ef05fad48fa3f0186670`完整校验；370,115个细胞
  只用于构建患者pseudobulk，统计单位严格为6例pure mucinous与53例pure conventional患者；
- broad tumour epithelium固定12基因面板中，AGR2与SLC35A1分别为
  `+1.613/+0.833 log2(CPM+1)`，全队列BH q均`0.0068`，右侧结肠/MMR分层BH q均`0.0179`；
  SPDEF全队列q=`0.0538`但分层q=`0.128`，GNE只有趋势，NANS/CMAS/CASD1/SIAE无支持；
- 七个预注册cell-source端点只有`Epi|secretory_carrier`和`Epi|cmp_neu5ac_capacity`通过：
  效应`+0.917/+0.687 z`，bootstrap下界大于零，BH q均`0.0627`，冻结匹配均5/6为正且
  leave-one-mucinous-out保持正；Myeloid CMP capacity未通过，上皮/髓系NEU1+NEU3 release均
  无正支持；
- NXPE1 broad-epithelial效应`+0.837`但BH q=`0.229`，冻结匹配4/6为正、exact p=`0.125`；
  加secretory composite后HC3 mucinous beta=`+0.242`、p=`0.706`。这与current-GDC结果一致：
  NXPE1是分布式secretory carrier state的伴随标志，不是已证独立驱动；
- 机制主线因此进一步收敛为：Level-1 free Neu5Ac pool扩张与独立患者层的上皮secretory folding
  和Golgi donor-transport capacity共存，但实测CMP-Neu5Ac不扩张，host NEU1/NEU3 RNA不升，
  glycan destination又呈core/linkage/carrier异质。这是pool–capacity–destination decoupling，
  不是global hypersialylation或flux；
- 机器结果见`data/external/GSE178341_mucinous_secretory_audit/nxpe1_mucinous_patient_pseudobulk_v1/`
  与`data/external/GSE178341_mucinous_secretory_audit/sialic_cell_source_patient_pseudobulk_v1/`，
  两个fail-closed validator均PASS；完整解释见
  `docs/GSE178341_RAW_UMI_MUCINOUS_SIALIC_AUDIT_20260831.md`；
- 完成度总账升级为25门的`data/mtbls13729/mechanism_paper_completion_audit_v10_final/`：raw transcript
  记为`PASS_CONTEXT`，独立proteomics固定面板记为`NEGATIVE_RESULT`，Package A仍ready，
  Package B/C和独立mucinous Neu5Ac abundance replication仍缺失。

### 13.26 2026-08-31 上皮组成诊断与统一生物学主图 v3

- 在看到raw-UMI结果后另行冻结了post-result诊断合同，不把它伪装成预注册确认分析；统计单位
  继续严格为患者，goblet-lineage固定为作者标注`cE02/cE06/cE07/cE08`；
- 6例mucinous的goblet-lineage epithelial fraction均值为`0.284`，53例conventional为`0.152`，
  差值`+0.132`，但患者bootstrap 95% CI `[-0.013,+0.287]`、匹配仅4/6为正，成熟`cE08`
  仅差`+0.0050`；故只能称广义goblet组成可能富集，不能称成熟goblet显著扩增；
- 加入logit goblet fraction后，MUC2、SPDEF、NXPE1区间均跨零；AGR2仍为`+1.172`
  （95% CI `[+0.483,+1.861]`），SLC35A1仍为`+0.676`（`[+0.216,+1.135]`），并在加入
  right-colon/MMR后保持正区间。因此raw转录信号不是单一组成效应，而是“部分组成富集 +
  选择性secretory-folding/Golgi-transport state”；
- 上述调整不是因果中介，细胞比例还受解离和取样影响；不能据此推断Neu5Ac来源、运输通量、
  酶活或glycan destination；
- 新统一主图`data/mtbls13729/neu5ac_evidence_convergence_figure_v3_final/`已经把同患者
  free/CMP/UDP丰度、独立raw-UMI组成调整、独立蛋白组阴性背景和未解决硬缺口放进同一张图；
  图内箭头均为虚线证据汇合，不是因果路径；
- 组成机器结果见`data/external/GSE178341_mucinous_secretory_audit/epithelial_composition_diagnostic_v1/`，
  fail-closed validator已PASS；完整解释见
  `docs/GSE178341_EPITHELIAL_COMPOSITION_DIAGNOSTIC_RESULT_20260831.md`。

### 13.27 2026-08-31 LCNEC 暗代谢物主结果：主生物学方向迁移

- 对2026年LCNEC代谢组图谱的公开HSST3n平台完成全链路复核：34对tumor/adjacent tissue，
  68个study、9个pooled QC、2个blank和6个dilution；study/QC分别有133,925/17,727张MS2；
- 263个precursor-RT families通过QC/blank/dilution门，42个与作者HSST3n表匹配、221个作者
  表外；原始MS1 targeted EIC、四归一化和共洗脱去冗余后冻结81个dark modules；42个作者
  已知阳性对照的本地效应与作者beta Spearman rho=`0.943`、方向一致率`90.5%`；
- 81/81模块已有pooled-QC MS2，并按10/20/50 ppm的冻结m/z约束协议完成official DreaMS、
  frozen P2b与classical spectral evidence注释；22个feature通过一致性门，12个与作者另一平台
  重叠且方向`12/12`一致、效应rho=`0.902`，其中10个在作者表中同向FDR<0.05；
- 4个作者表外优先候选全部通过5 ppm exact-formula与direct-fragment门：ADP family、
  ADP-ribose family、ascorbate和quinolinate。34对患者per-mg平均log2FC依次为
  `+2.400/+1.556/+5.407/+2.047`，同向患者`33/34、31/34、32/34、28/34`，全部
  leave-one-pair-out方向稳定且Wilcoxon p<=`2.83e-6`；四者均为singleton module，且在全部
  quality-passed features中没有命中±5 s、0.01 Da容差的C13/Na-H/chloride/formate/acetate
  共洗脱替代解释；该阴性门不代替标准品；
- BioAware只作context/abstention：ADP关联881条Rhea reactions，作为currency hub禁止激活特异
  通路；ADP-ribose、ascorbate和quinolinate为3个非hub锚点。网络没有修改谱学身份，也没有
  使用表型；reaction membership不代表反应方向、flux或酶活；
- 成组丰度模式为：AMP/GMP/ADP/ADP-ribose/UDP-HexNAc上升而guanosine/guanine下降；
  GSH/GSSG/ascorbate上升而ophthalmate下降。允许写pool redistribution与expanded antioxidant
  pools，禁止写ATP energy charge、PARP/CD38/NUDT5/QPRT活性或causal redox adaptation；
- LCNEC在cohort size、注射级QC、跨平台效应复现和作者表外机制候选数上已经明显超过
  MTBLS13729，因此升级为当前主生物学论文候选；MTBLS13729作为冻结保底和迁移验证保留。
  两者仍都缺authentic-standard RT与因果扰动；
- 完整结果与图见`docs/LCNEC_DARK_METABOLITE_BIOLOGY_RESULT_20260831.md`、
  `data/validation/lcnec_hsst3n_annotation_biology/`、
  `data/validation/lcnec_hsst3n_priority_structure/`、
  `data/validation/lcnec_hsst3n_priority_pair_consistency/`、
  `data/validation/lcnec_hsst3n_bioaware_context/`和
  `data/validation/lcnec_hsst3n_manuscript_figures/`。
- 新增fail-closed论文就绪总门：
  `data/validation/lcnec_hsst3n_manuscript_readiness/readiness_report.json`。它对全部冻结上游工件
  逐项校验并保存SHA256；当前裁决为algorithm-enabled Level-2 biology manuscript ready，
  但Level-1 identity、独立代谢物复制和causal metabolism均明确未ready，避免把完整计算闭环
  偷换成标准品或因果闭环。
- 已冻结六张补充表（81模块成员、21身份假说、12跨平台复现、9作者表外假说、4优先候选
  综合证据、136条患者-候选效应）及全部输入/输出哈希，见
  `data/validation/lcnec_hsst3n_manuscript_supplement/`；可直接落稿的英文Results与图注见
  `docs/LCNEC_MANUSCRIPT_RESULTS_DRAFT_20260831.md`。

### 13.28 2026-09-01 LCNEC 注释率口径、身份防线与稿件证据包 v2

- 对作者`article_mmc7.xlsx`逐表复核：主统计表有1,054条有效代谢物行，其中MSI Level
  1/2/3分别为73/935/46；作者正文声明的1,052是全平台代谢物数。补充表没有提供共同的
  detected-feature denominator，因此作者“注释率”不可计算，严禁拿1,052除以我们重建的
  HSST3n feature数；
- 在独立重建的263个QC/blank/dilution-qualified HSST3n families中，42个匹配作者HSST3n表、
  221个为source-table-absent analytical headroom；在另行冻结的81个dark modules中，official
  DreaMS有候选51个、DreaMS-P2b一致45个、多证据保留22个、跨平台复现12个、作者表外9个、
  优先候选4个。这些是不同阶段的coverage/evidence-calibration计数，不是连续准确率；
- 已生成并视觉审计注释恢复漏斗图与机器账本：
  `data/validation/lcnec_hsst3n_annotation_benchmark_v1/`；图中显式标出分母不可互换；
- 已冻结4候选身份声明防线：ADP/ADP-ribose继续写connectivity-family，ascorbate/quinolinate
  写MSI Level-2；精确新代谢物声明数为0。BioAware仅保留ADP-ribose/ascorbate/quinolinate
  三个非hub context anchors，并对ADP currency hub弃权；
- 当前Level-2/家族声明不以标准品为前提，但任何Level-1升级必须有同法RT/MS2；若只购两种，
  最优先quinolinic acid（机制区分度最高）与ascorbic acid（患者配对效应最大）。审稿回答矩阵
  见`data/validation/lcnec_hsst3n_identity_claim_defense_v1/`；
- 17个正文、图、表和审稿防线文件已封成哈希校验的
  `data/validation/lcnec_hsst3n_manuscript_evidence_package_v2/`，validator PASS；
- 在打开独立107对LCNEC proteogenomic患者级结果前，已经冻结QPRT/NUDT5/PPP/redox小面板、
  pure-vs-combined与KEAP1分层、缺失不替换和BH校正规则，见
  `docs/LCNEC_INDEPENDENT_PROTEOGENOMIC_FIXED_PANEL_PREREGISTRATION_20260901.md`。该资源只能补
  protein/pathway context，仍不能冒充代谢物丰度复制。

### 13.29 2026-09-01 LCNEC 机制图证据分层与独立蛋白组正文预审计

- 修订`abundance_evidence_map.pdf`，用`[R]/[N]/[H]`分别标记原作者图谱代谢物的跨平台复现、
  作者未报告的四个优先假说和其他Level-2/家族假说；颜色只编码丰度方向。图内同时写明
  Level-2、非通量、非酶活和非因果边界，避免把复现阳性对照与新候选混为一谈；
- 冻结14条描述性机制成员，形成四个内部一致的丰度轴：free purine pool depletion（2/2）、
  phosphorylated nucleotide/sugar accumulation（5/5）、tryptophan-quinolinate-NAD context（3/3）
  和antioxidant pool remodeling（4/4）。其中9条为`[R]`、4条为`[N]`、1条为`[H]`；该轴在看过
  代谢物后组装，因此明确标记`formal=false`，不产生新的独立P值或pathway-enrichment声明；
- 机器账本和fail-closed验证见`data/validation/lcnec_hsst3n_mechanism_coherence_v1/`；更新后的
  稿件证据包现含20个哈希核验文件，validator PASS；
- 独立Science Advances 107对LCNEC蛋白组正文在冻结面板之后完成文本预审计：固定22个蛋白中，
  正文明确提及`IDO1`和PPP的`G6PD/PGD/TKT/TALDO1`；作者报告combined LCNEC with NSCLC及
  KEAP1-mutant亚组PPP增强。该结果仅作为二级蛋白背景，不能替代即将对患者级处理矩阵进行的
  pure-LCNEC tumor-vs-NAT固定面板检验，更不能充当代谢物复制、身份确认或flux证据；
- 正文预审计工件见`data/external/LCNEC_proteogenomic_2026/text_context_audit_v1/`。患者级数据仍按
  预注册规则等待下载、哈希与结构审计后一次性计算。

---

## 14. 最终一句话版本

> 我们已经把 DreaMS 的局部错误推进为可定位、可反事实验证并可部分迁移进共享 embedding 的峰级机制；下游冻结候选专家构成检索保底但仍有 near-core 安全边界。生物学主线现已迁移到34对LCNEC：公开原始数据中81个冻结dark modules经全量谱学注释后得到12个跨平台正交复现和4个作者表外优先Level-2/连接性家族候选，支持phosphorylated-nucleotide/NAD-related pool redistribution与expanded antioxidant pools；BioAware同时保留3个非hub context anchors并对ADP hub主动弃权。MTBLS13729的hybrid mucin glycome作为冻结保底保留。项目已具备“算法方法 + 可复核暗代谢物发现”的投稿骨架，但标准品RT、独立LCNEC代谢组复制与causal tracing/perturbation/rescue仍是硬缺口。

### 13.30 2026-09-01 独立107例LCNEC蛋白组固定面板一次性检验

- Zenodo 20922299 的 `LCNEC_2026-SA.rar` 已通过有限Range恢复为完整130,002,347字节文件，MD5
  `1c3cb3dd041b6b23ccb5a84f25cd7714`与官方记录一致；`SuppData5.xlsx` ZIP完整性通过，矩阵为
  8,142蛋白×206样本，即103对tumor/NAT。
- 在打开患者级蛋白矩阵前已冻结22蛋白、纯LCNEC配对tumor-minus-NAT、双侧Wilcoxon、全22项BH、
  三项方向稳定门以及缺失不替换规则。矩阵实际含80对pure和23对combined LCNEC；18/22蛋白可测，
  TDO2、NMNAT2、SLC23A1、SLC23A2缺失且未替换。
- 主终点有13个蛋白通过q<0.10和方向稳定门。最干净的独立机制支持是PARP1
  `+1.319 log2, q=1.79e-13`和PARP2 `+0.868, q=3.10e-12`同向上升，为本地ADP-ribose家族
  积累提供PARP-associated context，但不证明其确切异构体或酶来源。
- quinolinate/de-novo-NAD轴呈重排而非单向变化：QPRT `-0.853, q=8.13e-10`、HAAO
  `-1.103, q=2.38e-13`、IDO1 `-1.063, q=1.15e-8`及KYNU/NADSYN1下降，而NMNAT3上升。
  这允许把QPRT降低写成quinolinate利用受限的可检验假说，但上游下降和NMNAT3反向意味着不能宣称
  已证明线性通量瓶颈。
- redox轴同样为混合补偿：GSR/G6PD/TKT/TALDO1下降、TXNRD1上升；结合本地ascorbate/GSH/GSSG
  上升和ophthalmate下降，正确术语是antioxidant-pool remodeling，不能写纯LCNEC统一PPP激活。
- 探索性combined-vs-pure tumor对照精确复现独立论文的组织学背景：G6PD `+0.402, q=3.995e-4`、
  TKT `+0.719, q=2.57e-4`、TALDO1 `+0.258, q=0.00189`，PGD同向且q=0.0513。这既验证了
  样本标签/分析方向，也说明combined的PPP激活不能外推成pure LCNEC tumor-vs-NAT结论。
- 工件见`data/external/LCNEC_proteogenomic_2026/fixed_panel_patient_audit_v1/`，完整解释见
  `docs/LCNEC_INDEPENDENT_PROTEOGENOMIC_RESULT_20260901.md`。该独立证据仍只是蛋白上下文，不是
  代谢物复制、Level-1身份、酶活、flux、因果或治疗脆弱性证据。

### 13.31 2026-09-01 LCNEC 263家族同分母注释比较

- 为回答“作者普通方法、官方DreaMS、完整工具”能否直接比较，已对全部263个phenotype-blind
  QC/blank/dilution-qualified precursor-RT families逐一抽取代表QC MS2，并统一运行官方DreaMS、冻结
  P2b和完整证据门。此前只在81个dark modules运行的结果不再承担全分析宇宙比较。
- 同一263家族分母上，来源HSST3n表m/z-RT重叠为42/263（16.0%）；官方DreaMS 20-ppm候选覆盖
  158/263（60.1%）；DreaMS-P2b top candidate一致136/263（51.7%）；完整high/moderate证据保留
  66/263（25.1%），其中high-consistency 57/263（21.7%）。这些分别是来源表重叠、候选覆盖、
  模型一致和Level-2/family证据产率，不是准确率。
- 42个source-matched正对照中，DreaMS有候选38（90.5%），完整证据保留31（73.8%）；221个
  source-table-absent families中，DreaMS有候选120（54.3%），DreaMS-P2b一致101（45.7%），完整
  证据保留35（15.8%），最终9个通过生物学稳健筛选、4个进入优先机制候选。
- 原文未给detected-feature denominator，因此作者“注释率”仍不可重建；42/263只能称为在我们
  重建分析宇宙中的source-table feature overlap。无标准品truth时，158/263和66/263同样不得称为
  注释准确率。
- 工件见`data/validation/lcnec_hsst3n_same_universe_comparison_v1/`；全263查询的官方DreaMS/P2b
  工件见`data/validation/lcnec_hsst3n_all_qc_annotation_v1/`。

### 13.32 2026-09-01 LCNEC 多队列机制三角化与主张反向审计

- 已把4个优先候选按“本地34对配对丰度—谱学/分子式—来源图谱轴—BioAware—独立蛋白组—
  身份边界”统一成冻结三角化矩阵，而不是继续把所有阳性证据等权堆叠。ADP-ribose/PARP是当前
  最干净的跨组学机制背景；quinolinate是最具机制区分度的标准品优先项；ascorbate效应量最大但
  独立redox蛋白为混合补偿；ADP因currency-hub和缺乏特异蛋白桥接，只保留为家族级核苷酸池哨兵。
- ADP-ribose家族本地`+1.556 log2, 31/34`与独立PARP1/2上升共同支持PARP-turnover随访假说，
  但不确认ADP-ribose异构体或PARP来源通量；quinolinate本地`+2.047 log2, 28/34`与QPRT下降、
  NMNAT3上升形成“利用瓶颈/重分配”可检验模型，但上游IDO1/KYNU/HAAO下降阻止线性通量叙事。
- 已执行14项反向主张审计并全部通过：42/263不是作者注释率、158/263不是DreaMS准确率、66/263
  不是完整工具准确率、author-unreported不是化学新颖性、独立蛋白不是代谢物复制、静态丰度不是
  flux/酶活，4个精确新代谢物声明仍为0。
- 工件见`data/validation/lcnec_hsst3n_multicohort_triangulation_v1/`与
  `data/validation/lcnec_hsst3n_biology_claim_audit_v1/`；正文新增Result 6和Figure 6，明确四个候选
  的不同论文角色和验证优先级。

### 13.33 2026-09-01 LCNEC 来源正对照身份一致性与真实失败病例

- 为避免继续只报coverage，新增保守身份一致性基准：仅使用42个source-matched families中，作者
  名称经精确归一化后可在冻结本地HMDB唯一解析到IK14的19个；未做人工同义词救援，23个未解析
  名称从身份分母排除。
- official DreaMS与完整工具均为17/19结构一致（89.5%，Wilson 95%CI 68.6–97.1%）；这只是来源
  正对照结构一致性代理，不是全局准确率，因为样本小、排除了未解析名称，且多数来源身份本身为
  MSI Level 2。
- 完整多证据门没有纠正或拦截2个discordant cases，反而都以high-consistency保留：
  N-acetylserine→glutamate，cis-aconitate→dehydroascorbate。两例均为同分子式异构体混淆，直接说明
  DreaMS/P2b/峰匹配证据联合仍不能消除结构异构体错误。
- 这一负结果已写入Result 0和主张审计；它强化而不是削弱当前Level-2边界，也为后续噪声微调、
  ChemAware和标准品优先级提供了真实生物学应用错误。工件见
  `data/validation/lcnec_hsst3n_source_positive_control_identity_v1/`。

### 13.34 2026-09-01 LCNEC 四优先候选的患者级协变边界

- 已在34个配对患者的肿瘤-癌旁log2效应上，对ADP family、ADP-ribose family、
  ascorbate和quinolinate的6个成对组合执行统一Spearman、BH6、患者bootstrap和
  leave-one-patient符号稳定审计。预注册门为`|rho|>=0.35`、`q<0.10`、bootstrap CI不跨0
  且LOO符号稳定；最终0/6通过。
- ADP--ADP-ribose是最强的提示性关系（rho=0.373，BH q=0.101，95%CI 0.008--0.663），
  但它略越过多重校正门，因此不能写成已确认的患者级模块。ADP-ribose--quinolinate
  rho=0.365、q=0.101，且bootstrap CI跨0，同样不通过。
- 正确的结论是：四轴的总体丰度方向和多组学上下文可用于建立分层假说，但现有
  34对数据不支持它们构成同一个统计上锁定的患者级代谢模块。这是必须主动报告的负边界，
  不是丢弃单候选。工件见`data/validation/lcnec_hsst3n_priority_patient_covariation_v1/`。

### 13.35 2026-09-01 LCNEC 投稿准备度与Discussion收口

- 已新建13项机器可读readiness scorecard。当前已就绪的是：固定分析宇宙、来源生物学
  正对照、同分母算法比较、4个Level-2/连接性家族假说、独立蛋白上下文和BioAware主动
  弃权。未就绪的是：Level-1身份、独立代谢物复制、flux/酶机制和临床预测。
- 已完成可投稿Discussion初稿`docs/LCNEC_MANUSCRIPT_DISCUSSION_DRAFT_20260901.md`。新颖性收窄为：
  一个经正对照校准的基础模型-正交证据阶梯，能从人类配对组织的dark features中提出
  边界清晰的生物学假说，同时暴露并主动报告同分子式异构体失败。不宣称PARP、氧化
  应激或肿瘤quinolinate本身是首次发现。
- 当前最高价值验证顺序冻结为：quinolinic acid同方法标准品＞ascorbic acid同方法
  标准品＞独立LCNEC代谢组。工件见`data/validation/lcnec_hsst3n_biology_readiness_scorecard_v1/`。

### 13.36 2026-09-01 LCNEC 四优先候选的同分子式竞争结构审计

- 已对冻结top-5谱库候选重算SMILES分子式，并与本地HMDB精确分子式集合对账。
  3/4优先项在top-5内就有同分子式竞争者；余下ADP-ribose虽然没有第二张同式谱，
  但本地HMDB同式集合有2个不同结构，因此不能将“无对手谱”当作化学唯一。
- ADP家族对adenosine 3',5'-diphosphate的official DreaMS领先仅0.032，支持继续报告
  connectivity family。Ascorbate对D-glucuronolactone领先0.305，是四者中谱学分离最强的；
  quinolinate对3-nitrobenzoate领先0.061，仍是实质的同式异构体风险。
- 因此排名能为标准品排序，不能证明化学唯一。工件见
  `data/validation/lcnec_hsst3n_priority_formula_rivals_v1/`；该结果已写入Result 3和第19项主张审计。

### 13.37 2026-09-01 独立LCNEC蛋白轴内患者级结构（探索性）

- 已对独立80对pure-LCNEC的3个冻结蛋白轴执行轴内所有两两组合，共46对，统一使用
  BH46、5000次患者bootstrap和LOO符号稳定门。12对通过，全部位于redox轴。
- G6PD与PGD/TKT/TALDO1/TXNRD1的rho分别约0.520/0.440/0.428/0.566，且多个GSR、PGD、
  TXNRD1组合通过。这说明pure-LCNEC中“均值方向混合”与“患者间协调变化”可同时
  存在，更符合异质性代偿性redox remodeling，而不是各蛋白无关噪声。
- PARP1--PARP2 rho=0.271、BH q=0.05004，因为未过`|rho|>=0.30`和严格q门，仅作提示性；
  de-novo-NAD轴无配对通过，因此继续写“redistribution”而非协调通量程序。
- 该分析在主固定面板结果后提出，明确`formal=false`，不作新确证终点。工件见
  `data/external/LCNEC_proteogenomic_2026/protein_axis_covariation_exploratory_v1/`。

### 13.38 2026-09-01 LCNEC 四优先候选的记录技术混杂审计

- 公开overview工作簿的HSST3n表对34对研究样本完整提供组织取材量与进样号；不含stage、smoking、
  sex或tumor purity，因此不得虚构临床亚组分析。
- 已固定4个技术因子：tumor/normal组织量log2比、配对平均进样位置、tumor-minus-normal有符号进样差、
  配对进样绝对间隔；与4个冻结候选患者效应组成16项Spearman检验，统一BH16、5000次患者bootstrap
  和LOO符号稳定门。0/16通过，最低BH q=0.378；肿瘤晚于/早于癌旁进样恰为17/17。
- 最大关系为quinolinate效应对有符号进样差`rho=-0.387, q=0.378`。它只作为下一次标准品/独立队列
  平衡进样顺序的敏感性警告，不能称技术混杂已证实。阴性审计也不能排除未记录技术因素或临床混杂。
- 工件见`data/validation/lcnec_hsst3n_priority_technical_confounding_v1/`；已纳入结果、方法、Discussion
  和投稿证据包的Extended Data Figure 4。

### 13.39 2026-09-01 外部LCNEC转录组的表达独立基因组分层验证

- 已取得并哈希冻结George等LCNEC队列的66例肿瘤RSEM表达和基因组注释。作者表达亚型仅作为
  次级上下文，因为亚型本身部分由同一表达矩阵定义；在完整固定门下只有
  quinolinate/de-novo-NAD轴通过，未将其包装为独立验证。
- 更严格的主外部对照完全由基因组事件定义：STK11或KEAP1改变且RB1无事件22例，对比RB1改变且
  STK11/KEAP1无事件17例。三个预冻结轴全部通过BH、R2>=0.10、分期层内置换和离散度门：
  quinolinate/de-novo-NAD R2=0.111、ADP-ribose turnover R2=0.104、ascorbate/redox R2=0.137。
- 22个冻结基因中只有NMNAT1、NMNAT3、PARP1、TKT通过单基因BH22。RB1改变组的NMNAT1、
  NMNAT3和PARP1更高，STK11/KEAP1改变组TKT更高，支持本地混合蛋白方向来自LCNEC基因组
  异质性的解释，而不是所有患者共享一条统一代谢程序。
- 预冻结的逐基因删除审计进一步区分强弱：redox轴8/8 omission通过；NAD轴8/9通过，删NMNAT3后
  统计仍显著但R2降至0.0716、未过0.08效应门；ADP-ribose轴4/5通过，删PARP1后R2=0.0486、
  BH q=0.119。因此redox可称多基因上下文，NAD与ADP-ribose必须分别写成NMNAT3/PARP1锚定。
- 该队列无匹配癌旁、无代谢物，故只提供表达独立的通路上下文，不能复制肿瘤/正常代谢物方向、
  验证身份或证明flux/因果。验证器为`tasks/validate_lcnec_george2018_external_axes.py`；工件见
  `data/external/LCNEC_George2018_transcriptome/frozen_axis_genomic_audit_v1/`。
- 已生成三队列综合图`data/validation/lcnec_hsst3n_three_cohort_mechanism_v1/`，将34对代谢物丰度、
  80对pure-LCNEC蛋白上下文和39例外部基因组分层并列展示，并明确三者不可互换。

### 13.40 2026-09-01 LCNEC 客观吸烟暴露敏感性审计

- 公开采集overview确实没有临床字段，但源论文补充表`article_mmc7.xlsx`的Table S4完整给出34例
  肿瘤组织cotinine定量与cotinine吸烟分类，Table S1给出age、sex、BMI和stage。因此此前“无法审计
  smoking”的边界已被更完整的源补充材料纠正，且没有借用候选结果选择暴露口径。
- 结局计算前冻结四个候选和三条分析臂：cotinine smoker/non-smoker Welch比较、log2 cotinine
  Spearman连续关联、以及`effect ~ cotinine + age + sex + BMI + stage`的HC3调整模型；每条臂各自
  BH4，并要求三者q<0.10且方向一致才判潜在smoking sensitivity。
- 34例中11例cotinine-classified smoker、23例non-smoker；四个候选0/4通过联合门。三条臂的最小
  BH q分别为0.657、0.800和0.896。该结果排除了“明显由吸烟分层驱动”的信号，但阴性敏感性审计
  不能证明完全不存在吸烟或其他临床混杂，更不能提高代谢物身份等级。
- 工件见`data/validation/lcnec_hsst3n_priority_smoking_confounding_v1/`，预注册为
  `data/validation/lcnec_hsst3n_priority_smoking_confounding_preregistration_v1.json`；已经纳入正文、
  Methods、Discussion、14项readiness scorecard和70文件投稿证据包。
