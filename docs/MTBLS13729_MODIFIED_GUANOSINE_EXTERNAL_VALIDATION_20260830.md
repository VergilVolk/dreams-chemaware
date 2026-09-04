# MTBLS13729 修饰鸟苷模块：原始 MS2、外部 Level-1 队列与机制边界

## 结论先行

当前可以站住的生物学发现是：**MTBLS13729 的黏液型结直肠癌组织中存在一个强、患者内一致、经离子家族折叠后的修饰鸟苷样丰度模块；原始 MS2 支持“核糖核苷/修饰鸟苷类别”，但尚不能确定具体位置异构体。**

这项发现不应写成“CRC 普遍上调修饰鸟苷”。对独立的 40 对 MSS/MSI 结直肠癌 Level-1 代谢组补充矩阵进行配对重分析后，四个唯一测量峰组成的修饰鸟苷模块在 MSI 中为 `-0.778 log2`（15/20 下降，Wilcoxon `p=0.024`），在 MSS 中为 `-0.122 log2`（不显著）。因此，更合理的研究命题是：

> 修饰鸟苷代谢在结直肠癌中具有组织学和分子背景依赖性；MTBLS13729 的强升高可能与黏液型状态、特定位置异构体或局部 RNA 周转/甲基供体环境有关，而不是泛 CRC 标志物。

## 1. 本地原始 MS2 审计

审计采用每个样本已解析的 EIC 峰界，只接受落在该峰内、前体质量误差 `<=5 ppm` 的原始 DDA MS2。未用表型选择谱图，也未用 BioAware/Rhea 提升身份等级。

| feature | 当前离子角色 | MS2 数 / 样本数 | 132.042 Da 核糖中性丢失 | 主要苷元离子 | 结论 |
|---|---|---:|---:|---:|---|
| 1597 | methylguanosine-like `[M+H]+` | 42 / 30 | 30/42 (71.4%) | 166.0725 | 强类别证据 |
| 7489 | 同一家族 `[M+Na]+` | 3 / 3 | 3/3 | 188.0552 | 低覆盖的加合物支持 |
| 3019 | dimethylguanosine-like `[M+H]+` | 32 / 32 | 32/32 | 180.0885 | 强类别证据 |
| 8481 | 同一家族 `[M+Na]+` | 16 / 16 | 16/16 | 202.0705 | 加合物支持；否定旧低分匹配 |

交叉加合物质量关系也成立：

- `1597/7489` 的 `[M+Na]-[M+H]` 残差为 `0.001003 Da`；
- `3019/8481` 的残差为 `0.000465 Da`；
- 两个家族在质子化和钠加合物 MS2 中均复现 `132.042 Da` 的核糖丢失。

MS2 证据分布在 Rmu、Rtu 和匹配正常组织中，而非只出现在肿瘤组，因此它支持离子身份/类别，不像由单一表型组触发的采集伪影。`P06-Ltu` 缺少峰解析 EIC，当前审计覆盖 59/60 个正离子文件；这不影响 Rmu 的 10 对主终点。

旧表中 feature 8481 的 `ADRENALINE BITARTRATE` 命中 cosine 仅 `0.5355`，且与精确加合物关系和核糖丢失相冲突，不能作为身份依据。

### 身份上限

上述证据支持“methylguanosine-like / dimethylguanosine-like ion family”。它不能区分 `1-methyl-`、`2-methyl-`、`2'-O-methyl-`、`3'-O-methyl-`、`7-methyl-` 等位置异构体，也不能升级到 MSI Level 1。标准品共洗脱和同碰撞能 MS2 仍是位置异构体确认的必要条件。

## 2. MTBLS13729 主效应

完全按两个加合物家族折叠后，Rmu 10 对患者的原始强度模块为：

- 10/10 肿瘤高于匹配癌旁；
- 平均 `+2.953 log2`，即 `7.74x`；
- 精确 sign-flip `p=0.001953`；
- 三档表型盲 PQN 后仍为 `+2.852` 至 `+2.860 log2`；
- Rtu 为 6/8 正向、均值约 `+0.91 log2`，远弱于 Rmu，但 Rmu-vs-Rtu 仅为探索性比较。

严格技术匹配随机模块中，真实模块超过全部 1,412 个完整可比模块，条件经验 `p=1/1413=0.000708`。由于预注册的随机面板完整率门只达到 70.6%（目标 75%），该结果必须写成“条件于完整可比面板的强特异性证据”，不能改门后包装为正式通过。

### 2.1 患者内多轴结构

修饰鸟苷模块与未纳入该模块的独立 purine-like feature 4966 在 10 个 Rmu 患者中的变化高度一致：raw Spearman `rho=0.903, p=0.000344`，患者 bootstrap 95% CI `[0.590, 1.000]`；三档 PQN 下 `rho=0.879`，均保持显著。这支持一个比单峰更完整的**嘌呤/修饰核苷周转轴**，但 feature 4966 仍只有异构体家族身份，相关性也不证明因果。

相反，二乙酰亚精胺样 feature 1717 和 C20:4 酰基肉碱样 feature 3222 与修饰鸟苷模块的患者内相关不显著。当前更合理的结构不是把全部候选强行塞入一条通路，而是三个并列的代谢重编程轴：

1. 修饰鸟苷/嘌呤周转轴；
2. 多胺乙酰化轴；
3. 长链酰基肉碱/脂肪酸氧化压力轴。

三轴在 Rmu 中方向性都很强，但患者间幅度相对独立。这正适合后续做分层表型，而不适合在当前 n=10 下构造一个事后加权的“总分”。

## 3. 独立 40 对 Level-1 组织队列重分析

外部队列来自一项 40 对结直肠癌肿瘤/癌旁组织的 MSI/MSS 代谢组研究（[原始研究](https://pmc.ncbi.nlm.nih.gov/articles/PMC12090577/)），补充矩阵包含 595 个 Level-1 代谢物、20 对 MSI 和 20 对 MSS。我们只重分析原作者补充矩阵，不重新选择代谢物。

重要的表格身份审计：

- `1-Methylguanosine` 与 `2'-O-Methylguanosine` 是同一个 `M296T181` 峰的两个候选名，丰度向量完全相同；
- `2-Methylguanosine` 与 `3'-O-Methylguanosine` 同理，是同一个 `M296T200` 峰；
- 因而外部模块只按四个唯一测量峰计算，禁止把替代异构体名称当作独立复现。

| Level-1 峰 | MSI mean log2FC | MSS mean log2FC | 方向说明 |
|---|---:|---:|---|
| 1-methylguanosine peak | -0.370 | -0.088 | 无显著升高 |
| 2-methylguanosine peak | -0.691 | -0.025 | MSI 偏下降 |
| N2,N2-dimethylguanosine | -0.672 | -0.203 | MSI 偏下降 |
| 7-methylguanosine | -1.380 | -0.172 | MSI 15/20 下降，Wilcoxon `p=0.0266` |
| 四唯一峰模块 | **-0.778** | **-0.122** | MSI 15/20 下降，Wilcoxon `p=0.0240` |

这不是 MTBLS13729 的“失败复现”，因为两个数据集的组织学分层不同，测得的位置异构体也未证明相同。它严格否定的是“泛 CRC 普遍升高”的解释，并提高了“黏液型或异构体特异”的研究价值。

外部矩阵还显示甲基供体环境明显依赖分子背景：SAM 在 MSS 中平均 `+2.122 log2`，而 SAH 在 MSI/MSS 中均大幅升高；这支持一碳代谢/甲基化环境存在异质性，但静态丰度不能推出甲基化通量或具体甲基转移酶活性。

### 3.1 外部原始 mzXML 独立重提取

已从 NODE/OEP00006137 冻结并校验四个组织 RPLC/HILIC 实验的公开文件清单。本轮先完成与四个修饰鸟苷峰直接对应的 RPLC 正、负离子数据：目录声明 182 个 RPLC 对象，其中 180 个可通过文件长度、MD5 和 tar/mzXML 结构校验；`ZZ_MSI-H_T12` 的 RPLC 正离子对象与 `ZZ_MSS_N20` 的 RPLC 负离子对象在 NODE 公共下载端持续返回全零/非 gzip 内容，已按不可用对象排除并保留审计记录。

在完全冻结补充表的 `m/z/RT` 坐标后，以 `5 ppm / ±15 s`、不看表型的方式重新积分原始 centroided MS1 EIC：

| 唯一峰 | 原始数据可检出样本 | 与补充矩阵 Spearman | MSI-H 原始重提取 mean log2FC | 判读 |
|---|---:|---:|---:|---|
| M296T181 | 79/79 | 0.9931 | -0.370 | 逐样本高度复现 |
| M296T200 | 79/79 | 0.9011 | -0.517 | 方向和排序复现 |
| M312T210 | 79/79 | 0.9942 | -0.869 | Wilcoxon `p=0.0401`，强复现 |
| M298T55 | 65/79 | 0.9147 | -0.226（complete-case） | 受早洗脱 RT 漂移/左删失影响 |

M298T55 的异常已进一步拆解，而不是用一种填补法强行收口：把质量窗从 5 ppm 放宽到 10 ppm 几乎不增加检出；把 RT 半窗从 15 s 放宽到 30 s 则把检出从 65 提高到 77，并恢复 12 个主分析零峰。恢复样本的峰顶 RT 为约 78–84 s，说明问题主要是部分运行的早段色谱漂移，而非质量偏差。宽窗下 MSI-H complete-case 为 `-0.698 log2`（13/18 下降），全局半最小值删失敏感性为 `-1.168 log2`（14/19 下降）；方向与补充矩阵 `-1.380 log2` 一致，但显著性不足。由于宽 RT 窗会直接合并两个相邻的 m/z 296.1 峰，宽窗只能作为敏感性证据，不能替代主分析或被事后选为“最佳结果”。

公开存档中的 11 个 pooled-QC RPLC 文件均只有 MS1，实测 ddMS2 扫描数为 0；这与正文所述 QC-ddMS2 采集不一致。因此外部原始数据可以独立复核丰度和峰可观测性，却不能独立复核作者的实验谱库匹配，也不能解决位置异构体。

### 3.2 HILIC 甲基供体/嘌呤轴的原始数据复核

为避免只围绕结果峰事后扩故事，HILIC 扩展严格限制为原先已经冻结的四个目标：methionine、guanosine、SAM 和 SAH。180 个 HILIC 归档可用；两个 HILIC 正离子正常组织对象在公共端稳定损坏并按缺失记录。

- Methionine：78 个生物样本检出，与补充矩阵 Spearman `0.9990`；MSI-H/MSS 均无稳定配对方向。
- Guanosine：80/80 检出，Spearman `0.9864`；MSI-H/MSS 均无稳定配对方向。
- SAH：64 个样本直接检出，Spearman `0.9972`。MSI-H complete-case `+2.403 log2`（12/13 上升，Wilcoxon `p=0.00049`），MSS `+2.202 log2`（10/12 上升，`p=0.00928`）。全局半最小值/最小值删失敏感性仍为强正向，因此 SAH 升高是原始数据可复现结果。
- SAM：固定坐标下仅 3 个生物样本检出；放宽质量或 RT 窗仍不能复现补充矩阵。故不计算 raw SAM/SAH ratio，也不把作者表中的 MSS SAM 升高当作本次原始数据确认结果。

患者级探索性耦合审计采用三个原始可复现 RPLC 修饰鸟苷峰组成的固定模块，不纳入 RT 左删失的 M298T55。SAH 与该模块的相关在 MSI-H (`rho=0.259, n=12`) 和 MSS (`rho=0.371, n=12`) 均不显著、bootstrap CI 跨 0。因此 SAH 只能作为**并列的甲基供体产物池异常**，不能被解释为修饰鸟苷变化的患者级驱动。MSS 中 methionine 与模块有探索性相关 (`rho=0.561, p=0.019, n=17`)，但样本小且多重比较未校正，不进入主结论。

同时在 MTBLS13729 的 60 个 HILIC+ 文件中，按 SAH `[M+H]+` 精确质量 `m/z 385.1289 ±5 ppm` 做了全 RT、表型盲峰簇搜索。最高可复现 RT 簇仅覆盖 2/60 样本，三组配对均 `n=0`。因此本地数据**没有可用的 SAH-like MS1 峰证据**；不得把外部 SAH 结果迁移为本地 Rmu 机制，也不得用缺失峰计算本地 SAM/SAH 比值。

## 4. 文献三角验证

当前文献支持“CRC 中 RNA 修饰与修饰核苷异常”，但不支持我们把本地峰直接命名为某一位置异构体：

1. 靶向血清核苷研究显示 m2G 和 2'-O-methylguanosine 在 CRC 中下降，而部分甲基腺苷升高，说明不同修饰位点和样本基质可呈相反方向（[PubMed 37441164](https://pubmed.ncbi.nlm.nih.gov/37441164/)）。
2. 前瞻性队列在 CRC 风险研究中观察到 1-methylguanosine 和 N2,N2-dimethylguanosine 信号，支持修饰鸟苷与 CRC 相关，但不能替代组织内配对因果（[prospective metabolomics study](https://pmc.ncbi.nlm.nih.gov/articles/PMC11884856/)）。
3. CRC 组织和模型研究证明 METTL1 介导的 tRNA m7G 修饰可促进翻译重编程、进展与肝转移；这给出可检验机制轴，但并未证明游离 7-methylguanosine 就是 METTL1 通量的直接读数（[PubMed 41627602](https://pubmed.ncbi.nlm.nih.gov/41627602/)）。
4. CRC 组织中 METTL1 调控的 m7G-tRNA 衍生小 RNA 与 5-FU 耐药相关，进一步支持 tRNA 修饰/裂解轴具有生物学合理性（[PubMed 39153969](https://pubmed.ncbi.nlm.nih.gov/39153969/)）。
5. 最近的 tRNA 修饰研究在 70 对 CRC 组织中证明另一种 tRNA 修饰（A-to-I）及 ADAT2 轴显著重编程，说明“tRNA 修饰—翻译重编程”是活跃且可实验验证的 CRC 机制范式（[PubMed 41845367](https://pubmed.ncbi.nlm.nih.gov/41845367/)）。
6. 早期 47 例患者内配对结直肠腺癌研究已经观察到肿瘤 SAH 显著升高，并由此降低 SAM/SAH 甲基化比值；这为我们原始数据复现的 SAH 升高提供独立人组织证据，但不能替代本队列中无法原始复现的 SAM（[PubMed 17375119](https://pubmed.ncbi.nlm.nih.gov/17375119/)）。
7. 2023 年代谢分型研究把 AHCY/adenosylhomocysteinase 识别为 CRC 的可干预代谢节点，说明 SAH/AHCY 轴具有机制优先级；仍需 AHCY 表达、同位素示踪或干预才能把我们的静态 SAH 池变化写成通量（[PubMed 37580540](https://pubmed.ncbi.nlm.nih.gov/37580540/)）。

这些研究只能建立机制候选优先级。没有 RNA 修饰定量、转录组/蛋白组或干预实验时，当前文章应写“RNA 周转/修饰相关代谢模块”，不能写“METTL1 导致”或“tRNA 降解通量增加”。

## 5. BioAware 的正确位置

BioAware v2 的表型盲峰图可用于：

- 合并同一中性分子的不同加合物；
- 检查共洗脱、跨样本相关和双层证据；
- 把候选组织成可审计的模块。

但在 21-query 伪真值集上，两层网络专家为 `0 corrected / 0 introduced`，没有证明身份排序增益。Rhea 也没有直接覆盖本地核心候选。因此 BioAware 当前是**证据组织和去冗余工具**，不是身份真值生成器，也不是本地生物学发现的统计来源。

## 5.1 独立黏液型组织蛋白组的正交支撑（2026-08-30）

我们进一步重分析了独立的黏液型结直肠癌组织 TMT 蛋白组（29 名患者；LMC/LNMC/RMC/RNMC 加正常组织通道）。固定的嘌呤合成/回收面板中 9 个可检蛋白在四类肿瘤相对正常组织的中位 log2 比值均为正（`+0.42/+0.32/+0.25/+0.28`），HPRT1、PNP、GMPS、IMPDH1、IMPDH2、GDA 构成方向一致的嘌呤需求/周转证据。修饰核苷加工面板也整体偏正，但 METTL1 本身仅表现为轻度且位置依赖的变化，不能据此指定具体 writer。

该证据把本地主结论从“一个大效应未知峰”推进为“修饰核苷与嘌呤周转共同改变的候选轴”，但仍不证明修饰鸟苷增加由某个甲基转移酶造成。源研究为组池化 TMT，故只能作方向性正交支撑，不能称为逐患者统计复现。完整表格、热图与身份升级计划见 `docs/MTBLS13729_BIOLOGY_CLOSURE_AND_MINIMAL_VALIDATION_20260830.md`。

公开标准方法进一步收紧了异构体判断：m7G/m2G 可产生 `298→166`，Gm 更偏向 `298→152`，m2²G 对应 `312→180`。本地 feature 1597 因主体 166 离子且部分谱见 152，必须用 m7G、m2G、Gm 同法保留时间区分；feature 3019 与 m2²G 高度相容，但仍需标准品排除其他二甲基鸟苷位置异构体。

## 6. 当前最有价值的论文命题

建议把生物学主线从“发现一个新代谢物”升级为：

> A phenotype-blind spectral and ion-family consolidation workflow reveals a strongly elevated modified-guanosine module in mucinous colorectal tumors, while independent Level-1 tissue metabolomics demonstrates context-dependent rather than universal CRC behavior.

中文口径：

> 表型盲谱学与离子家族整合揭示黏液型结直肠癌中显著升高的修饰鸟苷模块；独立 Level-1 组织队列显示该模块具有背景依赖性，而非泛 CRC 同向标志物。

创新点不是“又做一个差异代谢物表”，而是：

1. DreaMS/P2b 扩大候选谱学证据；
2. BioAware 峰图解决加合物冗余和证据传播边界；
3. 原始 MS2 与定向 EIC 把算法候选落实为可审计离子家族；
4. 独立 Level-1 队列充当反证式外部验证，暴露亚型/异构体依赖性；
5. 最终形成“算法发现—峰级证据—配对定量—外部反证—机制假设”的闭环。

## 7. 下一步优先级

### P0：立即完成，无需新湿实验

1. 对 `1597/3019` 的原始 MS2 建立诊断碎片表，核对公开 mzCloud/MassBank/HMDB 标准谱；只在能区分位置异构体时升级名称。
2. OEP00006137 的 RPLC 原始 MS1 重提取已完成；下一步只需对 HILIC 的 SAM/SAH/鸟苷轴做冻结扩展。因公开 QC 文件无 MS2，不再把“外部原始 MS2 复核位置异构体”列为当前可执行目标。
3. 在 MTBLS13729 内把修饰鸟苷模块与 C20:4 酰基肉碱、二乙酰亚精胺三条轴做患者内相关和共变模块分析，检验是否形成黏液型联合代谢状态，而非孤立峰。
4. 对原论文 345 条注释逐一对账：哪些是原文已有、哪些由本工具新增、哪些只是身份重排、哪些因证据不足降级。

### P1：可选的最低成本确认

购买最少量标准品时，优先顺序应为：

1. N2,N2-dimethylguanosine；
2. 7-methylguanosine 与 1-/2-methylguanosine 中至少两个可分离异构体；
3. 同条件 RT + MS2 共洗脱，而不是只比谱库 cosine。

### P2：机制层，只能作为后续合作

- RNA 修饰 LC-MS 或 tRNA 修饰测序；
- METTL1/WDR4/TRMT 相关表达；
- tRNA 片段与修饰核苷联合；
- 这些数据缺失时，不写通量、酶活性或因果。

## 8. 工件

- 原始 MS2 审计：[modified_guanosine_ms2_report.json](../data/mtbls13729/modified_guanosine_ms2_audit_v1/modified_guanosine_ms2_report.json)
- 外部 Level-1 重分析：[report.json](../data/external/OEP00006137_support/modified_guanosine_reanalysis/report.json)
- 综合图：[modified_guanosine_biology_evidence_20260830.png](../data/mtbls13729/modified_guanosine_biology_evidence_20260830.png)
- 患者内三轴分析：[report.json](../data/mtbls13729/biology_axes_analysis_v1/report.json)
- 外部原始 MS1 重提取：[summary.json](../data/external/OEP00006137_raw/modified_guanosine_raw_reextraction_v1/summary.json)
- 外部 RT/删失敏感性分析：[summary.json](../data/external/OEP00006137_raw/modified_guanosine_raw_sensitivity_v1/summary.json)
- 外部 HILIC 甲基/嘌呤轴原始复核：[summary.json](../data/external/OEP00006137_raw/hilic_methyl_purine_summary_v1/summary.json)
- 外部患者级跨轴耦合审计：[summary.json](../data/external/OEP00006137_raw/methyl_purine_coupling_v1/summary.json)
- 本地 SAH 精确质量负向试点：[summary.json](../data/mtbls13729/sah_exact_mass_pilot_v1/summary.json)
- 跨队列原始数据主图：[modified_guanosine_raw_crosscohort_20260830.png](../data/mtbls13729/modified_guanosine_raw_crosscohort_20260830.png)
- OEP00006137 公开文件下载审计：[download_rp_only_report.json](../data/external/OEP00006137_raw/download_rp_only_report.json)
- 本地完整生物学闭环：[MTBLS13729_BIOAWARE_BIOLOGY_CLOSURE_20260829.md](./MTBLS13729_BIOAWARE_BIOLOGY_CLOSURE_20260829.md)
