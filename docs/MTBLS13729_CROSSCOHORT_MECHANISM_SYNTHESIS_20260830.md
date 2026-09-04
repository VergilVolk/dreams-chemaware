# MTBLS13729 跨队列生物学机制综合与证据边界（2026-08-30）

## 一、当前最强、可防守的生物学结论

MTBLS13729 的 10 对 Rmu–RN 组织中出现三个**并行的高幅度代谢池表型**：

1. 修饰鸟苷/嘌呤周转轴：feature 1597、3019、4966 分别约为 `+3.72/+2.40/+2.44 log2`；
2. 乙酰化多胺轴：feature 1717 约为 `+3.01 log2`；
3. 长链酰基肉碱轴：feature 3222 约为 `+1.78 log2`。

这些结果来自患者内配对丰度，并在留一患者和多种归一化下保持方向；它们不是通量测量。三个轴之间也不应被串成一条单一因果链：修饰鸟苷模块与 purine-like 4966 强共变，但与 1717/3222 的患者内相关不显著。

## 二、跨队列综合后的机制模型

当前最符合全部正、负证据的模型是：

> Rmu 发现亚组表现出修饰核苷/嘌呤周转和多胺乙酰化相关代谢池的积累，同时出现长链 acylcarnitine 积累。独立 CRC 转录组、单细胞和蛋白组支持更广泛的核苷/嘌呤重塑与部分 FAO 能力降低背景；但 2026 年人群/高脂饮食 CRC 研究也表明 carnitine/acetylcarnitine 升高可以伴随 CPT1A 上调和促肿瘤代谢。因此 feature 3222 目前支持的是“carnitine-shuttle imbalance”，其来源可能包括输入增加、利用瓶颈、不完全氧化或组织组成变化，静态丰度不能在这些竞争机制之间裁决。

这是一项**机制支持型解释**，不是因果机制终证。

## 三、外部证据如何改变结论

### 1. 支持的部分

- ST001087 的 17 对 CRC 组织中，N2,N2-dimethylguanosine 的 formula-level 信号平均 `+1.317 log2`，N1,N12-diacetylspermine 平均 `+0.763 log2`；这是家族/通路方向支持，注释为 FindByFormula 且检出稀疏。
- GSE236696 的 6 对黏液型 CRC 上皮 pseudobulk 中，purine 轴 `6/6` 上升，FAO 轴 `6/6` 下降；多胺乙酰化/分解轴在冻结 broad gate 下 `6/6` 上升。它们是转录背景，不是代谢物复现或通量。
- TCGA COADREAD 的 32 对肿瘤–正常显示 modified-nucleoside、purine 和 FAO 轴分别约 `+0.806/+0.779/-1.669`，说明这些主要是一般 CRC 程序。
- 独立 pooled mucinous CRC 蛋白组中，LMC–normal 的 modified-nucleoside、purine、FAO 轴分别约 `+0.33/+0.42/-0.29`；但 pooled 设计不能做患者级推断。

### 2. 必须保留的反证

- OEP00006137 的 Level-1 N2,N2-dimethylguanosine 在 MSI/MSS 中分别约 `-0.672/-0.203 log2`，与 ST001087 相反；因此不能声称“修饰鸟苷在所有 CRC 中普遍升高”。
- TCGA 的协变量调整 mucinous–conventional 比较不支持 modified-nucleoside 或 FAO 轴在黏液型中特异增强；purine 轴在黏液型中反而较低。
- MTBLS8090 不支持一般 CRC 的 N1-acetylspermidine 或长链 acylcarnitine 统一上升。
- GSE236697 只有一个空间病例，且多胺乙酰化与酸性/趋化的空间偏相关很弱；它不能构成空间因果链。

## 四、身份与机制必须分开

| 轴 | 当前身份强度 | 当前允许结论 | 禁止结论 |
|---|---|---|---|
| 1597/7489 | methylguanosine ion family；跨加合物与核糖丢失支持 | 新增修饰鸟苷离子家族 | 具体 m7G/m2G/Gm 已确认 |
| 3019/8481 | dimethylguanosine ion family；位置异构体未分 | 新增 dimethylguanosine 家族 | N2,N2 或 1,7 异构体已确认 |
| 4966 | C7H9N5O purine-like family | 与修饰鸟苷模块共变的独立 purine companion | 精确数据库结构或单一反应已确认 |
| 1717 | 73 张峰界内 MS2；m/z 100.0759 在 100% 谱中出现，并与 authentic-standard `230.2→100.0` 文献转换一致；尚无同法 RT/完整镜像 | 强 diacetylspermidine-like 候选丰度轴 | MSI Level 1/2 精确身份、SAT1 因果、分泌/免疫因果 |
| 3222 | long-chain/C20:4 acylcarnitine-like | 长链 acylcarnitine 类积累 | C20:4 位置异构体、FAO 通量升高/降低已证明 |

## 五、相对原论文的真实增量

原论文已报告广泛 carnitine program 和 N1,N8-diacetylspermidine 名称，因此这些不能包装为首次发现。我们的主要新增是：

1. 原作者 345 条注释表中没有 methylguanosine/dimethylguanosine 离子家族；DreaMS 辅助重分析从原始峰界 MS2、核糖丢失和跨加合物一致性中找回了两个修饰鸟苷家族；
2. 将候选从单一谱图注释推进到患者内配对丰度、ion-family 去冗余和反证式外部复核；
3. 将 acylcarnitine 升高从“FAO 增强”的直觉改写为与外部 FAO 轴降低兼容的利用瓶颈假说；
4. 明确量化了算法能增加什么证据，以及不能替代标准品、通量和扰动的边界。

## 六、高水平机制论文对照后的证据缺口

近期高水平工作的一般闭环是：非靶向发现 → 标准品/targeted LC-MS/MS → 细胞或空间来源 → 基因/药理扰动 → isotope tracing/flux → phenotype/rescue → 独立队列。典型例子包括 CRC AHCY 研究、Gut 的脂质炎症研究、SORD–糖驱动 CRC 转移研究和空间单细胞同位素示踪。

我们目前到达：**非靶向重分析 + 原始 MS2/EIC + 患者配对丰度 + ion-family + 独立代谢组反证 + 单细胞/TCGA/蛋白组背景**。尚缺：

1. authentic-standard RT + 同碰撞能 MS2 + spike-in；
2. 独立带 mucinous/conventional 标签的组织代谢组；
3. isotope tracing；
4. METTL1/TRMT、SAT1 或 CPT/FAO 关键节点的遗传/药理扰动；
5. phenotype rescue 或体内验证。

因此，无湿实验时应把论文定位为**算法驱动、证据校准的临床非靶向代谢组重分析与机制假说生成**；不能将“causal mechanism”作为标题主承诺。

## 七、当前最有性价比的下一步

1. 标准品 P0：m7G、m2²G、N1,N8-diacetylspermidine；若能扩充，再加 m2G/Gm 和 C20:4/C18:1/C18:0/C16:0 acylcarnitine 面板。
2. 每个标准执行同法 RT、MS2 镜像和样本 spike-in 共洗脱；只做谱库 cosine 不足以升级身份。
3. 在没有新湿实验前，优先完成所有候选的 raw-spectrum evidence panel、患者配对森林图、跨队列正反证图和机器可审计证据矩阵。
4. 继续寻找独立 mucinous CRC metabolomics；截至 2026-08-30 已检索的公开资源中，尚未识别出比当前 MTBLS13729 更合适且可下载、可患者级重分析的独立组织队列。

## 八、固化产物

- 证据矩阵：`data/mtbls13729/mechanism_evidence_matrix_v1/mechanism_evidence_matrix.csv`
- 机器报告：`data/mtbls13729/mechanism_evidence_matrix_v1/report.json`
- 跨队列主图：`data/mtbls13729/crosscohort_mechanism_figure_v1/crosscohort_mechanism_evidence.png`
- 矢量图：`data/mtbls13729/crosscohort_mechanism_figure_v1/crosscohort_mechanism_evidence.pdf`
- 标准诊断转换对账：`data/mtbls13729/fragmentation_standard_consistency_v1/fragmentation_standard_consistency.csv`

## 九、主要对标来源

- CRC AHCY mechanism study: https://pmc.ncbi.nlm.nih.gov/articles/PMC10447251/
- Integrated lipidomics/scRNA/spatial CRC study: https://pmc.ncbi.nlm.nih.gov/articles/PMC11885055/
- SORD and sugar-driven CRC metastasis: https://doi.org/10.1038/s42255-025-01368-w
- Spatial single-cell isotope tracing: https://www.nature.com/articles/s42255-024-01118-4
- Spatially guided bulk/MSI metabolomics: https://pmc.ncbi.nlm.nih.gov/articles/PMC13328631/
- Reporting/annotation guide: https://www.nature.com/articles/s41592-021-01197-1

## 十、公开 authentic 谱与碰撞能元数据的新增审计

### 10.1 feature 1597 的公开标准谱只能确认家族，不能确认位置异构体

已从 MassBank 官方 API 冻结 7-methylguanosine 与 N2-methylguanosine 在 10/20/40 eV 下的六张 authentic-standard 谱，并与 feature 1597 的峰界内复现碎片共识比较。最佳 sqrt-cosine 分别为 `0.6712` 与 `0.6667`，最佳分数差仅 `0.00450`。因此公开标准谱支持 methylguanosine family，但在当前跨仪器、稀疏共识协议下不能分辨 m7G 与 m2G。feature 1597 不得命名为确定的 7-methylguanosine；同法 RT、完整谱镜像和样本 spike-in 是不可替代的升级条件。

### 10.2 本地 raw-MS2 复现强，但没有碰撞能梯度

对五个主候选的 200 张峰界内 MS2 逐谱读取 mzML 元数据后，碰撞能记录完整，但全部为 `30 eV`。因此本地数据能够提供跨样本/跨患者复现，却不能提供跨碰撞能复现：1597 的 m/z 166.0725 在 30/42 谱中出现；3019 的 180.0886 为 32/32；1717 的 100.0759 为 73/73；3222 的 85.0281 为 23/30；4966 的 110.0347 为 16/23。这里的正确表述是“30 eV 下跨样本复现的诊断离子”，而不是“碰撞能系列稳定”。

新增可复核产物：

- `data/mtbls13729/massbank_isomer_spectral_audit_v1/report.json`
- `data/mtbls13729/candidate_ce_recurrence_v4/report.json`
- `data/mtbls13729/structural_evidence_figure_v1/structural_evidence.png`
- `data/mtbls13729/structural_evidence_figure_v1/structural_evidence.pdf`

## 十一、MTBLS7387 251 对人体组织复算：对 3222 的帮助与边界

### 11.1 这是独立的大样本人群证据，但不是同一代谢物复现

对 ATF6 论文公开 Fig. 3 来源矩阵的正式复算得到 `251` 对完整 CRC 肿瘤–癌旁样本和 `186` 个脂肪酸特征。论文配对 t 检验 + 全 panel BH-FDR 的 9 个公开 p/q 值均在来源表舍入允许的 0.5% 相对误差内复现。

- 全 panel：56 个 FDR<0.05，38 升、18 降；52/56 同时通过 paired Wilcoxon FDR<0.05；
- C20–C24：59 个特征，17 个 FDR<0.05，14 升、3 降；16/17 同时通过 Wilcoxon FDR<0.05；
- paper-standard validated 的 C22:4/C22:5/C22:6 均在显著长链集合中；
- free arachidonic acid：`+0.0865 log2`，FDR `0.243`，不显著；
- hydroxy-C20:4 RT 5.50：`+0.683 log2`，FDR `9.58e-4`；RT 5.43：`+0.414 log2`，FDR `0.0418`。

这支持“CRC 中存在可复现的长链脂肪酸重塑背景”。但外部对象是 free/hydroxy fatty acids，本地 feature 3222 是 long-chain acylcarnitine-like ion；二者不是同一分析物类别，所以不构成直接身份复现。

### 11.2 异质性本身是机制信息

同一链长、不同 RT 的 feature 可发生相反变化：hydroxy-C20:2 RT 5.88 为 `+1.136 log2`，RT 6.30 为 `-1.469 log2`。Early CRC 的 99 对只有 5 个显著 C20–C24 特征，late CRC 的 152 对有 15 个。它们说明脂质程序存在分子种/色谱峰和人群异质性，反对把一个类别总分直接写成统一通量。

### 11.3 样本流失透明度

论文文字报告 259 人；MetaboLights 样本表为 258 tumour + 258 adjacent；Fig. 3 处理矩阵为 502 行、251 完整配对。原始 identifier 对账得到 8 个 metadata-only 标识和 1 个 processed-only 标识，其中 `315Tu2/315u` 很可能是拼写别名，净差 7 对。公开文件没有给出这 7 对的排除原因，也不能解释论文到 deposition 的 1 人差异。我们的所有复算结果严格称为 `251-pair processed cohort`。

### 11.4 机制论文对标后的新解释

ATF6 研究使用标准品、类器官 D3-FA elongation、FASN 抑制、无菌/FMT 和微生物生长/功能实验，建立了 ATF6–LCFA–microbiota 因果链。2026 Gut 研究又在 152 例发现、28 例验证人群上定量总脂肪酸，并用 Apc 小鼠口服稳定同位素、无菌模型、CD36/CPT1A 抑制和 5 例患者类器官证明外源脂肪酸摄取与功能。

这些论文说明我们的 3222 轴下一步应问“脂肪酸来自哪里、进入哪一代谢池、是否被氧化”，而不是仅继续扩大静态 acylcarnitine 名单。2026 年 Oncogene 研究进一步提供了反向机制可能：1,257 人发现/验证中 carnitine、acetylcarnitine 与 β-hydroxybutyrate 构成诊断面板，高脂饮食 AOM/DSS 模型中 carnitine/acetylcarnitine 与 CPT1A 上调同向，CPT1A silencing 或 β-hydroxybutyrate–FXR 干预抑制相关表型。由于它研究的是 carnitine/acetylcarnitine 而非 C20:4 acylcarnitine，也不能直接迁移到 3222；但它正式否定“acylcarnitine 积累必然代表 FAO 利用下降”的单向解释。

现阶段允许写“independently supported long-chain lipid context”和“competing carnitine-shuttle imbalance hypotheses”；禁止把 `FAO-utilization bottleneck` 写成已证明事实，也禁止写 ATF6 causality、extrinsic FA flux、mucinous specificity 或 feature 3222 exact identity。

工件：

- `data/external/mtbls7387_paired_lcfa_replication_v1/paired_fatty_acid_results.csv.gz`
- `data/external/mtbls7387_paired_lcfa_replication_v1/paired_fatty_acid_age_subgroups.csv.gz`
- `data/external/mtbls7387_processed_pair_attrition_v1/report.json`
- `data/mtbls13729/mtbls7387_lcfa_context_figure_v1/mtbls13729_mtbls7387_lcfa_context.png`

## 十二、黏液型风险转录背景：新增支持与反证

GSE281917/281918 的 MuC 与 NMuC 标签和测序平台完全共线，因此禁止把两个 GEO series 的直接差异当作亚型复现，也不能用 ComBat 消除不可辨识的混杂。主分析仅在 GSE281917 的 140 例 MuC 内部检验冻结代谢轴与 MuC23 风险分数的关系。

- 临床协变量校正后，MuC23 与 modified-nucleoside、purine 轴负相关，与 polyamine-acetylation 轴正相关；
- MuC23 与 fibroblast/endothelial marker scores 高度相关；加入六类 broad-lineage marker 后，仅 purine 轴保留（rho `-0.254`，95% CI `[-0.464,-0.086]`，BH q `0.0139`）；
- 在 42 例 TCGA 黏液型病例中，purine 的 clinical-adjusted 方向复现，但 broad-lineage adjustment 后 CI 跨 0；因此 composition-independent primary replication 未通过；
- polyamine 在 TCGA 出现次级方向性信号，但样本小、属于 secondary endpoint，不能提升为主机制。

这部分的允许结论是 **risk-associated bulk transcript state**。它不是代谢物复现、独立预后验证、细胞自主机制、通量或酶因果证据。综合图见：

- `data/mtbls13729/mucinous_risk_context_figure_v1/mtbls13729_mucinous_risk_context.png`
- `data/mtbls13729/mucinous_risk_context_figure_v1/mtbls13729_mucinous_risk_context.pdf`

## 十三、项目相对顶级机制论文的最终定位

与 AHCY、SORD、ATF6–LCFA–microbiota 和 Gut targeted lipidomics 工作逐项对照后，项目已完成 clinical discovery、原始 MS2/EIC、患者配对丰度、ion-family、跨队列正反证和 bulk composition sensitivity；尚缺 authentic-standard terminal identification、独立黏液型代谢组、isotope tracing、酶扰动、phenotype/rescue 和体内验证。

因此当前最强、最诚实的定位是“算法驱动、证据校准的临床非靶向代谢组重分析与机制假说生成”，而不是 causal metabolism paper。完整 scorecard：`docs/MTBLS13729_MECHANISM_READINESS_SCORECARD_20260830.md`。

## 十四、独立黏液型代谢组的系统检索裁决

OmicsDI 冻结检索得到 88 条 CRC metabolomics 记录。已逐一审计 53 个 MetaboLights 的 public ISA sample tables 和 15 个 Metabolomics Workbench 的 factor tables，均无静默失败。唯一含 mucinous 患者级编码的是 MTBLS13729 自身的 10 个 `Rmu` 样本；其他公开条目没有可用于患者级复算的 mucinous histology 字段。OEX MSI/MSS 组织队列、普通 CRC tissue cohorts 和 GNPS 镜像可用于一般 CRC 背景与反证，但不是黏液型复现。

因此“尚未找到可复算的独立黏液型组织代谢组”现在有数据库级审计支持；它仍不是“此类队列绝对不存在”的证明。完整记录：`docs/MTBLS13729_EXTERNAL_MUCINOUS_METABOLOMICS_SEARCH_AUDIT_20260830.md`。
