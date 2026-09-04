# MTBLS13729 生物学机制论文 readiness scorecard（2026-08-30）

## 结论先行

当前项目已经形成一篇**算法驱动、证据校准的临床非靶向代谢组重分析论文**所需的大部分计算证据，但尚未达到“代谢机制已被因果证明”的证据层级。最强贡献不是把三个代谢轴硬串成一条通路，而是：

1. 用 DreaMS/P2b 辅助从原始 DDA MS2 中找回原论文注释表遗漏的修饰鸟苷离子家族；
2. 把候选推进到患者内配对丰度、ion-family 去冗余、原始谱学复现和跨队列正反证；
3. 同时保留 long-chain acylcarnitine、乙酰化多胺和修饰核苷的竞争机制，不把静态 abundance 偷换成 flux；
4. 量化算法证据在哪些环节能提高注释和生物学解释，哪些环节仍必须由标准品、示踪和扰动完成。

在无新增湿实验的现实条件下，文章最合理的定位是：

> **DreaMS-enabled, evidence-calibrated reanalysis of paired colorectal tissue metabolomics reveals modified-guanosine ion families and parallel nucleotide, polyamine and carnitine-shuttle abundance programs in a mucinous discovery subgroup.**

不得把标题主承诺写成 “causal mucinous metabolic reprogramming”。

## 一、顶级机制研究实际完成了哪些层级

符号：`✓` 已直接完成；`△` 间接或部分完成；`—` 未完成/不适用。

| 研究 | 人体/疾病发现 | 标准品或 targeted 定量 | 独立人群 | 细胞/空间来源 | 同位素来源/通量 | 基因或药理扰动 | phenotype/rescue | 体内验证 | 核心机制强度 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| CRC AHCY, Nature Metabolism 2023 | ✓ | ✓ | △ | ✓ | ✓ | ✓ | ✓ | ✓ | APC–AHCY–methionine cycle 与肿瘤生长形成完整链 |
| SORD–sugar CRC metastasis, Nature Metabolism 2025 | ✓ | ✓ | △ | △ | ✓ | ✓ | ✓（LbNOX/mevalonate） | ✓ | SORD–NAD redox–glycolysis/mevalonate–转移链 |
| CRC unresolved lipid inflammation, Gut 2025 | ✓（40 对） | ✓（81 对） | ✓ | ✓（scRNA/spatial） | — | △（转录/受体证据） | — | — | targeted lipid mediator + cell-source mechanism-supporting study |
| ATF6–LCFA–microbiota, Nature Metabolism 2025 | ✓（251 对处理队列） | ✓ | △ | ✓（类器官/宿主–微生物） | ✓（D3-FA） | ✓ | ✓ | ✓（germ-free/FMT） | 宿主 ATF6–LCFA–微生物功能因果链 |
| Hepatocyte–CRC crosstalk, 2025 | △（模型系统） | ✓ | — | ✓（2D/3D） | ✓ | △ | △ | — | 多组学和示踪支持微环境燃料切换 |
| Spatial multi-omics CRLM, 2025 | ✓ | △ | △ | ✓（spatial/scRNA/IF） | — | — | — | — | 空间来源强，代谢因果弱于 tracing/perturbation 论文 |
| 本项目 MTBLS13729 | ✓（30 对；Rmu 10 对发现） | △（原始 MS2/EIC；无同法标准） | △（多队列正反证；无独立黏液型代谢组） | △（GSE/TCGA bulk/pseudobulk） | — | — | — | — | 高质量 evidence-calibrated discovery；尚非因果机制 |

## 二、按论文审稿门逐项审计本项目

| 证据门 | 当前状态 | 已有工件 | 仍缺什么 | 对主文的影响 |
|---|---|---|---|---|
| 临床设计 | 已完成 | 患者内 tumour–normal 配对；Rmu/Rtu 分层；临床敏感性分析 | Rmu 只有 10 对 | 定位为 discovery subgroup，不宣称确认性人群结论 |
| MS1 定量 | 已完成但无经典 QC | 240 mzML 的 MS1 重提取、targeted EIC、三种归一化、留一患者 | pooled QC/blank 不存在 | 必须报告缺失与替代 QC，不能伪造 QC-CV |
| MS2 注释增量 | 已完成 | DreaMS/P2b 三路审计、峰界内原始 DDA、ion family | 精确异构体标准 | 可声称新增 ion family，不可声称精确位置异构体 |
| feature-space 统计 | 部分完成 | 冻结候选 panel、多归一化、效应量、选择后边界 | 全特征 selective-inference 仍需统一整理 | 候选统计称 discovery-priority，不称全谱 FDR-confirmed |
| 外部代谢组 | 已完成正反证但非同亚型 | ST001087、OEP00006137、MTBLS7387、MTBLS8090 | 独立带 mucinous 标签的患者级组织代谢组 | 支持通路背景和异质性，不支持黏液型特异复现 |
| 外部转录/蛋白 | 已完成 | GSE236696/697、TCGA、GSE281917、mucinous proteomics | 真正恶性细胞标签和代谢物同样本联合测量 | 只称 pathway/risk context，不称代谢物或 flux replication |
| 组成混杂 | 已主动审计 | GSE281917 broad-lineage adjustment；TCGA 定向复算 | 单细胞代谢组或 microdissection | 组成敏感信号必须降级；purine 风险关联也不能称 cell-autonomous |
| 身份终证 | 未完成 | MassBank authentic spectra；30 eV 诊断离子复现 | 同法 RT + 多 CE MS2 + spike-in；必要时合成异构体 | 1597/3019/1717/3222 均保持 family-like 名称 |
| 通量和来源 | 未完成 | 只有静态 abundance 和转录背景 | ^13C/^15N/^2H tracing 或至少公开同位素资源中的同化合物验证 | 禁止写 flux、利用率或来源方向 |
| 酶和表型因果 | 未完成 | METTL1/TRMT、SAT1、CPT/FAO 是候选解释 | genetic/pharmacologic perturbation、rescue、organoid/in vivo | 禁止写 therapeutic target 或 enzyme causality |

## 三、三个生物学轴分别达到什么水平

### 1. 修饰鸟苷/嘌呤周转轴：当前最有原创性的算法发现

**已成立：**

- feature 1597/7489 构成 methylguanosine ion family，3019/8481 构成 dimethylguanosine ion family；
- 原始峰界 MS2 支持核糖丢失和跨加合物一致性；
- 1597、3019、4966 在 Rmu–RN 中效应大、方向稳定；
- 原作者 m/z–RT 注释表没有系统识别这些修饰鸟苷家族；
- 外部代谢组和转录组同时提供正、负证据，说明该程序具有队列依赖异质性。

**尚未成立：** m7G、m2G、Gm、m2²G 的具体位置异构体；RNA/tRNA 来源；METTL1/TRMT 因果；核苷周转通量。

**最小升级实验：** m7G、m2G/Gm、m2²G 标准的同法 RT/MS2/spike-in。若只能做一个轴，这是最高优先级，因为它最能把“算法新增 ion family”升级为命名代谢物发现。

### 2. 乙酰化多胺轴：谱学最强，但原论文已有名称

**已成立：** feature 1717 在 73 张峰界内 MS2 中稳定出现 m/z 100.0759，并有明显 Rmu 配对丰度升高；与 published authentic transition 相容。

**边界：** 原论文已经给出 N1,N8-diacetylspermidine 名称，因此不能包装成首次命名；当前没有同法标准 RT，且 GSE/TCGA 风险关联对组成敏感。

**最小升级实验：** N1,N8-diacetylspermidine authentic standard；这是最有把握升级到精确身份的一项。

### 3. 长链 acylcarnitine 轴：临床幅度稳，但机制解释最不唯一

**已成立：** feature 3222 是 long-chain/C20:4-acylcarnitine-like，Rmu–RN 平均约 +1.78 log2；长链 acylcarnitine 类总分也有配对方向性。

**竞争解释：** 脂肪酸输入增加、CPT1A/运输增强、下游利用瓶颈、不完全氧化、组织组成差异都可以造成 acylcarnitine 积累。MTBLS7387 支持长链脂肪酸重塑，但不是同一分析物复现；FAO 转录轴下降与 carnitine 增加也不是充分的 flux 证据。

**最小升级实验：** C20:4/C18:1/C18:0/C16:0 acylcarnitine 标准 + isotope internal standards。即使身份升级，仍不能省略 palmitate tracing/OCR 才能讲利用机制。

## 四、GSE281917/TCGA 风险分析对主线的真实贡献

GSE281917 的 140 例黏液型 CRC 中，MuC23 高风险与较低 purine/modified-nucleoside 分数及较高 polyamine-acetylation 分数相关。加入 broad-lineage marker 后，仅 purine 关联保留；MuC23 与 fibroblast/endothelial 分数高度相关。TCGA 42 例黏液型病例中 clinical-adjusted 方向复现，但 purine 在组成校正后不显著。

因此，这部分的价值是：

1. 把本地代谢发现放入一个**风险相关 bulk state**；
2. 证明必须处理组织组成，避免把 stromal-rich signature 误写为肿瘤细胞代谢；
3. 为未来 microdissection/single-cell/空间验证预先定义方向。

它不能证明：本地代谢物决定 MuC23 风险、嘌呤通量下降、细胞自主代谢重编程或独立预后价值。

综合图：`data/mtbls13729/mucinous_risk_context_figure_v1/mtbls13729_mucinous_risk_context.png`。

## 五、在不能做湿实验时，最强计算闭环应补齐什么

### P0：投稿前不可缺的计算证据

1. 把全部 discovery-priority features 放回全特征背景，统一报告选择路径、有效检验数和 selective-inference 边界；
2. 对 1597/3019/1717/3222 生成完整 raw EIC + precursor/isotope/adduct + consensus MS2 + competing-isomer panel；
3. 将 original DreaMS、P2b、BioAware 的每个候选变化拆成 corrected/introduced/abstained，并标注是否改变了最终生物学结论；
4. 对所有外部队列统一使用“direct metabolite / orthogonal transcript / risk context / adversarial evidence”标签；
5. 把 negative controls 放进主文图或扩展图，而不是只放成功候选。

### P1：如果只能购买极少量标准

1. N1,N8-diacetylspermidine；
2. m7G + m2G/Gm + m2²G 组成的修饰鸟苷小面板；
3. C20:4 + C18:1 + C18:0 + C16:0 acylcarnitine。

每个标准必须完成同法 RT、与样本同碰撞能的镜像 MS2 和 spike-in 共洗脱；只有数据库 cosine 不能升级 MSI 等级。

### P2：真正进入机制论文所需的新实验

- 修饰鸟苷：writer/turnover 节点扰动 + RNA/游离核苷来源追踪；
- 多胺：SAT1 gain/loss、细胞内外定量、髓系/血管表型和 rescue；
- acylcarnitine：palmitate tracing、OCR、CPT1/CACT/CPT2 或下游 beta-oxidation 扰动。

## 六、允许和禁止的最终论文句式

### 允许

> DreaMS-enabled reanalysis recovered modified-guanosine ion families that were not represented in the original m/z–RT annotation table. Together with a diacetyl-polyamine-like feature and a long-chain acylcarnitine-like feature, these signals define three parallel abundance programs in the Rmu discovery subgroup. Orthogonal cohorts support context-dependent nucleotide, polyamine and lipid remodeling, while exact identities, mucinous specificity and flux remain unresolved.

### 禁止

> We proved mucinous-specific METTL1/SAT1/FAO reprogramming and identified a causal therapeutic pathway.

## 七、主要对标来源

- AHCY CRC mechanism: https://pubmed.ncbi.nlm.nih.gov/37580540/
- SORD and sugar-driven CRC metastasis: https://www.nature.com/articles/s42255-025-01368-w
- Integrated lipidomics/scRNA/spatial CRC study: https://pubmed.ncbi.nlm.nih.gov/39658263/
- Hepatocyte–CRC metabolic crosstalk: https://pubmed.ncbi.nlm.nih.gov/39713297/
- Spatial multi-omics CRLM: https://pubmed.ncbi.nlm.nih.gov/40340245/
- Original MTBLS13729 study: https://pubmed.ncbi.nlm.nih.gov/42366730/
- ATF6–LCFA–microbiota: https://www.nature.com/articles/s42255-025-01350-6

## 八、当前裁决

项目不是“还没有生物学结果”，而是已经完成了**临床发现与机制假说生成**的大部分计算闭环。真正尚未完成的是身份终证和因果机制。下一轮工作必须围绕三件事推进：

1. 提高精确身份而不是再堆候选名；
2. 找独立黏液型代谢组或提供透明的“未找到”系统检索记录；
3. 让算法提升与生物学结论逐候选对账，证明新算法确实改变了注释覆盖、证据等级或发现优先级，而不只是改善抽象 benchmark。

## 九、扩展复核后的 readiness 更新

新增原始数据工作把“候选是否真实存在”推进了一步，但没有改变因果门：

- 15 个候选进入冻结证据总账，其中 9 个可映射到原论文 Level-1/2 source identity，5 个是强谱学家族候选，1 个为降级控制；
- 5 个 same-mode RPLC 重映射峰的误差为 0.28–2.06 ppm、RT 误差约 1.5–6.6 s；
- carnitine、hypoxanthine、tryptophan 和 feature 1717 获得同样本跨色谱/极性正交支持；taurine 未通过这一门；
- 四个模块均为 9/9 或 10/10 同向并通过留一节点方向稳定，但它们是后验模块，不是独立验证；
- 模块相关分析否定“一条统一代谢链”，只保留 amino-acid–purine/nucleoside 的边缘耦合假说。

因此 readiness 的提升发生在 **结构/技术正交与生物学收敛** 两层；独立人群、标准品新确认、示踪、扰动、rescue 和体内因果仍未完成。主文可更强地写 `convergent abundance programs`，仍不可写 `causal metabolic reprogramming`。

## 十、脯氨酸与唾液酸新增轴的 readiness 裁决

### 脯氨酸/P5C–基质轴

- **身份与丰度：强。** proline/glutamate 均有原论文 Level-1 source anchor、独立正相峰、同样本跨面板配对耦合和 10/10 Rmu 丰度方向；
- **一般 CRC 复现：强。** TCGA 32/32 proline-synthesis 上升，独立 pooled proteomics 的四个合成酶全部正向；
- **黏液型特异：不成立。** TCGA 调整比较中 proline-synthesis 在黏液型相对较低；
- **细胞来源：中低。** 单细胞上皮 PYCR1 方向支持，但轴未超过表达匹配随机集；
- **通量/因果：未完成。** 不能由 proline pool 推断合成或氧化通量。

### Neu5Ac–mucin glycan 轴

- **身份与丰度：强。** feature703 与 source Level-1 Neu5Ac 的同样本与患者配对相关均约 0.959，Rmu 10/10 升高；
- **一般 CRC 程序：方向复杂。** TCGA paired transcript axes 多数下降；
- **黏液型相对程序：强转录支持。** GNE/NANS/SLC35A1、ST3GAL4、ST6GALNAC1/2 和 MUC2/SPDEF 在黏液型相对富集；
- **空间/蛋白：异质。** secretory-mucin 与 goblet 空间定位清楚，sialic enzymes 的左右/细胞区室方向不统一；
- **糖链与因果：未完成。** 游离 Neu5Ac 不等于 glycan linkage 或 cell-surface hypersialylation，需 glycomics/glycoproteomics 或 perturbation。

新增轴提高了文章的“身份找回 + 一般 CRC/黏液型相对程序分层”价值，但没有改变整篇论文仍属于 mechanism-supporting discovery、而非 causal metabolism 的总裁决。

## 十一、Hybrid mucin glycome 分支审计后的 readiness 更新（2026-08-31）

新增 TCGA 分支模型与外部患者 O-glycomics 对账后，以下两项由“方向背景”升级为通过：

- **分支解耦通过：** donor supply/transport、secretory-mucin carrier、core/linkage 分支并非统一
  上升；ST6GAL1 与 alpha2-6 方向为负，足以排除简单 global hypersialylation 叙事；
- **相对保留与绝对损失对账通过：** core-3/Sda 在黏液型肿瘤间相对较高，但在 MUC tumour-normal
  配对中仍下降，两个参照系已经被明确分开。

仍未通过的决定性门为：

- 同一样本中 free Neu5Ac/ManNAc/CMP-Neu5Ac 与具体 O-glycan 结构的耦合；
- MTBLS13729同一样本或独立人群中的MUC2/其他载体glycopeptide与空间定位；PXD055865只提供
  2位独立黏液癌患者的外部载体背景，不能替代这一门；
- 独立患者级 free-Neu5Ac 丰度复现；
- isotope incorporation、节点扰动和 rescue。

因此当前可以使用 `hybrid mucin glycome` 和 `donor–carrier–core–linkage decoupling`，但不能使用
`causal sialylation mechanism`、`increased glycan incorporation` 或 `ST6GAL1-mediated immune escape`。

## 十二、PXD055865载体审计后的readiness更新（2026-08-31）

- **外部carrier/destination层升级为PASS_WITH_LIMITATION。** PXD055865的人工复核糖肽和
  source spectra确认MUC2上存在sialylated、O-acetyl-Neu5Ac与putative O-acetyl-GalNAc结构，
  并直接展示空间glycoform异质性；
- **独立性被严格降至2位患者。** Colon1a/1b来自同一患者，三块标本不等于三次独立人群重复；
- **鉴定覆盖与丰度完全分开。** 不同标本切区、谱深和表格行数不等，任何661-vs-21或单标本
  糖肽数量比较都不作为丰度结果；
- **仍未通过same-sample destination门。** PXD055865不测free Neu5Ac，也不是MTBLS13729样本，
  因此只能与本地free-pool/donor解耦作方向相容的三角验证；
- **位置异构体仍未解决。** HMDB仅有预测谱，MassBank/MoNA无精确名称/分子式实验记录；
  4/7/8/9-O-acetyl-Neu5Ac需要成对标准，必要时IM-MS/CCS。

这一更新提高了Package A的载体层完整性，但不使Package B或C自动通过。完整总账以
`data/mtbls13729/mechanism_paper_completion_audit_v9_final/`为准。
