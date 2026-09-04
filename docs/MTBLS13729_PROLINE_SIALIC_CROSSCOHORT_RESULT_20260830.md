# MTBLS13729 生物学新增结果：脯氨酸合成与唾液酸池的跨队列分化

## 结论先行

这轮扩展没有得到一条可以直接写成因果的统一通路，而是得到两个方向不同、证据层级也不同的程序：

1. **脯氨酸/P5C–胶原背景是稳健的一般 CRC 程序。** 正相 RPLC 中找回的 proline feature 345 在 10 个 Rmu 患者中全部升高；它与原论文负相 HILIC 的 Level-1 proline 在 59 个同样本中的丰度、组织内残差和患者配对变化均高度一致。TCGA 32 对 CRC 肿瘤–癌旁的 proline-synthesis 轴 32/32 上升，独立黏液型组织蛋白组的 4 个可检合成酶也全部高于正常组织。单细胞上皮中 PYCR1 是最清楚的候选节点，但 6 人轴级信号没有通过表达匹配随机集特异性门，因此单细胞只能作方向性背景。
2. **Neu5Ac 不是“全局唾液酸化增强”的简单证据。** feature 703 与原论文 Level-1 N-acetylneuraminic acid 的跨面板相关极强，且 Rmu 10/10 升高；但 TCGA 32 对一般 CRC 的 sialic-synthesis/transport 与多数 mucin-sialylation 轴整体下降。相反，在 42 个黏液型与 329 个常规型肿瘤的调整比较中，GNE/NANS/SLC35A1、ST3GAL4、ST6GALNAC1/2 与 MUC2/SPDEF 程序相对富集。正确解释是**黏液型相对糖组/唾液酸池重塑**，不是已经证明游离 Neu5Ac 来源、糖链连接方式或净表面唾液酸化方向。

这两个程序共同支持一个更精确的生物学模型：

> Rmu 组织同时呈现一般 CRC 的脯氨酸合成/基质适应程序，以及黏液型背景下选择性的唾液酸前体、转运与黏蛋白糖基化程序；静态游离代谢物丰度和转录/蛋白背景相互补充，但不等于通量或因果。

## 一、三个新增代谢物身份节点

| 正相 RPLC feature | 候选身份 | Rmu–RN mean log2FC | 方向 | 95% bootstrap CI | 正交身份链 |
|---:|---|---:|---:|---:|---|
| 345 | Proline | +1.299 | 10/10 | [1.051, 1.559] | 94 张 MS2、59 样本；MassSpecGym median cosine 0.9988；映射原论文 Level-1 HMDB0000162 |
| 374 | Glutamic acid | +0.715 | 10/10 | [0.423, 1.049] | 102 张 MS2、59 样本；跨面板配对 rho 0.849；映射原论文 Level-1 HMDB0000148 |
| 703 | N-acetylneuraminic acid (Neu5Ac) | +1.975 | 10/10 | [1.331, 2.607] | 33 张 MS2、33 样本；median cosine 0.905；映射原论文 Level-1 HMDB0000230 |

这些是**同队列、另一色谱/极性中的正交找回**，不是新患者队列，也不是本项目重新注射标准。身份强度来自原论文 Level-1 节点、同样本跨面板丰度耦合与独立 MS2 的共同支持。

### 跨面板对账

| feature | 样本级 Spearman | 组织内残差 Spearman | 患者配对 delta Spearman | 源 feature 排名 |
|---:|---:|---:|---:|---:|
| 345 proline | 0.887 | 0.872 | 0.814 | 1/70 |
| 374 glutamate | 0.744 | 0.683 | 0.849 | 1/85 |
| 703 Neu5Ac | 0.959 | 0.925 | 0.959 | 1/70 |

feature 301 与 proline 共洗脱，但其 `[M+Na]+` 质量误差和竞争谱库候选不满足身份门；feature 1695 虽有 leucine/isoleucine-like MS2，却未复现原论文 leucine 的患者配对变化。这两个反例已排除，不进入主结论。

## 二、脯氨酸程序：支持什么，反证什么

### 2.1 本地代谢物

- proline +1.299 log2，10/10 Rmu 患者升高；
- glutamate +0.715 log2，10/10 升高；
- 与 isoleucine、phenylalanine、tryptophan 合并后的扩展氨基酸池 +0.969 log2，10/10，同一节点留一后方向稳定；
- 该模块为候选发现后的描述性收敛，不是预注册 pathway p 值。

### 2.2 TCGA 大队列

- 32 对肿瘤–癌旁中，proline-synthesis 轴 32/32 上升，均值 +0.952 standardized units，Wilcoxon BH q=3.73e-9；
- PYCR1 32/32 上升，PYCRL 31/32，PYCR2 28/32；
- collagen/proline-context 轴 28/32 上升，BH q=1.33e-6；
- 42 个黏液型相对 329 个常规型的 proline-synthesis 调整 beta=-0.249，BH q=0.0130。故它是一般 CRC 程序，不能包装为黏液型特异增强。

### 2.3 GSE236696 患者级单细胞

- 保守 marker-gated epithelial pseudobulk 中 proline-synthesis 5/6 上升，均值 +0.353；
- PYCR1 5/6 上升，均值 +1.674；
- 但轴的双侧精确 p=0.125、单侧 p=0.0625；全基因表达/检出率/变异度匹配的 20,000 随机轴中，幅度经验 p=0.228、5/6 同向经验 p=0.253。

所以单细胞结果只允许写“上皮方向与 PYCR1 一致”，不能写“上皮特异脯氨酸程序得到显著验证”。

### 2.4 独立黏液型组织蛋白组

源研究为 pooled TMT，只能做组级描述：

- LMC vs normal：ALDH18A1/PYCR1/PYCR2/OAT 4/4 为正，中位 +0.485 log2；
- RMC vs normal：4/4 为正，中位 +0.330；
- PYCR1 分别 +0.62/+0.45；
- PRODH/ALDH4A1 也升高，提示可能是 proline/P5C cycle 或高周转，而不是只凭 proline pool 判断合成通量。

### 2.5 机制边界

CRC 的 PYCR 文献已用人体组织、细胞与小鼠证明 mitochondrial PYCR 对生存/增殖重要，但补充外源 proline 或 nucleotides 不能简单 rescue PYCR depletion。因此我们的静态 proline 升高不能被直接写成“蛋白合成原料增加”。更合理的竞争解释包括 redox、proline/P5C cycling、线粒体功能和胶原/基质适应。

## 三、Neu5Ac/黏液糖组程序：为什么不能简单叫高唾液酸化

### 3.1 本地代谢物身份与丰度

- feature 703 在 Rmu 10/10 上升，mean +1.975 log2；
- 与原论文负 HILIC Level-1 N-acetylneuraminic acid 在 59 个同样本中 rho=0.959，患者配对变化 rho=0.959；
- 因此“游离/可提取 Neu5Ac pool 增加”成立；其来源和下游去向未定。

### 3.2 一般 CRC 与黏液型相对效应方向相反

TCGA 32 对肿瘤–癌旁：

- sialic-acid synthesis/transport -0.576，27/32 下降，BH q=1.58e-5；
- sialic-acid remodeling -0.587，26/32 下降，BH q=1.33e-6；
- mucin-sialylation -0.308，24/32 下降，BH q=0.00131。

TCGA 黏液型–常规型调整比较：

- synthesis/transport beta +0.392，BH q=2.27e-4；
- mucin-sialylation beta +0.252，BH q=0.00426；
- secretory-mucin program beta +0.919，BH q=9.59e-12；
- gene-level 中 GNE、NANS、SLC35A1、ST3GAL4、ST6GALNAC1/2、MUC2、SPDEF、FCGBP 均为相对正向并通过跨 42 个目标基因的 BH 门。

这支持的是“mucinous-relative program”。它可能意味着黏液型肿瘤相对常规型保留或重建特定糖组能力，但不能从 bulk RNA 推出游离 Neu5Ac 的来源或组织表面糖链总量。

### 3.3 反证和复杂性必须进入正文

- GSE236696 的 epithelial synthesis/transport 轴接近零，mucin-sialylation 仅 3/6 上升；
- 单病例空间转录组中 secretory-mucin program 明显富集于 tumour/goblet 区域，但 sialic axes 在 spot 分布中不升；
- 独立 pooled proteomics 的左、右黏液型方向不同，sialic synthesis/transport 分别约 -0.06/-0.45，只有部分转移酶呈局部正向；
- complete CMAS loss 在 MC38 模型中反而可增加肿瘤生长并促进 CD8 T-cell apoptosis，说明“去唾液酸一定抗癌”是错误的；
- 正常结肠 MUC2 存在复杂的 core-3、STn 和 9-O-acetylation 环境，游离 Neu5Ac 不能替代 glycoproteomics/glycomics。

## 四、这轮结果对论文主线的改写

### 允许的主线

> Evidence-calibrated reanalysis recovered three source-anchored metabolites in an orthogonal LC-MS panel. Proline and glutamate extend a broadly conserved CRC proline/P5C–matrix program, whereas an increased free Neu5Ac pool is embedded in a mucinous-relative but internally heterogeneous sialic/mucin-glycan transcriptional context.

### 禁止的主线

> Mucinous CRC is globally hypersialylated because Neu5Ac is elevated, and increased proline proves increased proline flux.

## 五、和高水平非靶向机制论文相比还缺什么

Nature Metabolism 的 CRC–AHCY 工作不是只做 untargeted feature 和 pathway enrichment，而是连续完成：标准/RT/MS2 身份、组织空间定位、同位素来源、候选酶转录、类器官抑制、遗传模型和体内表型。近期高水平工作同样把 abundance、flux、细胞来源、perturbation 与 rescue 分开。

本项目已完成：

- 患者内配对原始 MS1 定量；
- 跨面板 Level-1 identity bridge；
- 多归一化与反例审计；
- TCGA 配对、黏液型调整模型、单细胞患者级 pseudobulk、单病例空间和独立 pooled 蛋白组；
- 明确区分一般 CRC 与黏液型相对程序。

本项目仍缺：

- 同法 authentic-standard RT/MS2/spike-in；
- glycan linkage / glycoproteomics；
- ^13C/^15N tracing 或直接 flux；
- PYCR/GNE/CMAS/ST6GALNAC perturbation 和 rescue；
- 独立带患者级 mucinous 标签的组织 metabolomics。

因此当前定位是**算法驱动的临床发现与机制支持论文**，不是已经完成因果链的代谢机制论文。

## 六、最小升级优先级

1. **Proline/glutamate 标准不是最高优先级**：两者已有原论文 Level-1 节点和很强跨面板桥，继续购买标准带来的信息增量较小。
2. **Neu5Ac 同法标准 + spike-in**：可把正相 feature 703 从“source-anchored Level-2 recovery”提升为本方法 Level-1，并确认是否存在共洗脱异构/降解峰。
3. **若能做 glycomics**：优先测 STn/core-3、α2-3/α2-6 和 9-O-acetylated sialic structures；这是判断 feature 703 生物学去向的最短实验。
4. **若只能做计算**：保留双轴分化，不再扩张“唾液酸化”结论；将新增注释率、身份纠错案例和跨组学分层作为方法学贡献。

## 七、可复核产物

- `data/mtbls13729/integrated_biology_ledger_v2/`
- `data/mtbls13729/proline_orthogonal_audit_v1/`
- `data/mtbls13729/expanded_crosspanel_audit_v1/`
- `data/external/GSE236696/proline_sialic_by_lineage_v1/`
- `data/external/GSE236696/proline_sialic_robustness_v1/`
- `data/external/GSE236696/proline_genomewide_matched_null_v1/`
- `data/external/GSE236697/spatial_proline_sialic_v1/`
- `data/external/TCGA_COADREAD_Xena_20260830/proline_sialic_axes_v1/`
- `data/external/TCGA_COADREAD_Xena_20260830/proline_sialic_gene_audit_v1/`
- `data/external/mucinous_crc_proteomics_2021/proline_sialic_reanalysis_v1/`
- `data/mtbls13729/proline_sialic_summary_figure_v1/`

## 八、主要对标来源

- CRC PYCR mechanism: https://pubmed.ncbi.nlm.nih.gov/35130302/
- CRC AHCY multimodal mechanism benchmark: https://www.nature.com/articles/s42255-023-00857-0
- CMAS/desialylation CRC counterexample: https://pubmed.ncbi.nlm.nih.gov/30578646/
- ST6GAL1–LGALS3BP CRC multi-omics mechanism: https://pubmed.ncbi.nlm.nih.gov/39937175/
- Defined human mucin O-glycans: https://www.nature.com/articles/s41467-021-24366-4
- Mucinous CRC collagen–integrin polarity mechanism: https://www.nature.com/articles/s41467-026-75127-0
- Static metabolomics versus isotope tracing boundary: https://www.nature.com/articles/s44324-024-00017-2

