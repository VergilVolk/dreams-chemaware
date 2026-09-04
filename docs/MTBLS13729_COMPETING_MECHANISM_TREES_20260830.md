# MTBLS13729 竞争机制树与最短验证闭环（2026-08-30）

## 结论先行

当前数据已经支持五个可复核的 **abundance programs**，但没有任何一条静态丰度轴能够唯一决定上游机制。最严谨、也最有论文价值的做法，不是选一个酶把故事讲满，而是为每条发现保留互相竞争的生成过程，并预先规定能把它们区分开的最短实验。

机器可读版本位于：

- `data/mtbls13729/competing_mechanism_trees_v1/competing_mechanism_hypotheses_v1.csv`
- `data/mtbls13729/competing_mechanism_trees_v1/competing_mechanism_hypotheses_v1.md`
- `data/mtbls13729/competing_mechanism_trees_v1/report.json`

该总账冻结了 5 个轴、14 个竞争机制。它不是新的统计发现，而是防止把“丰度—身份—通量—酶—表型”五个层级偷换成同一个结论。

## 一、修饰鸟苷：当前最有原创性的算法发现，但来源最容易误判

### 本地已经成立

- feature 1597/7489 与 3019/8481 分别形成 methylguanosine 和 dimethylguanosine 的离子家族；
- 原始峰界 MS2 支持核糖丢失和跨加合物一致性；
- Rmu–RN 患者内丰度方向稳定，并与 purine-like feature 4966 构成候选模块；
- 原论文 identity table 没有系统列出这些离子家族，因此它们是本项目最强的 source-table-absent narrative increment；
- authentic MassBank 谱只能支持 family compatibility：1597 对 m7G 与 m2G 的最佳谱相似度几乎相同，位置异构体仍不可分。

### 三种尚不能区分的生成机制

1. **writer 增强：** METTL1/WDR4 介导的 m7G deposition 增强，扩大 methylated-RNA pool，继而增加游离核苷；
2. **RNA turnover 增强：** 已有 methylated RNA 更快降解，writer 活性不变，游离修饰核苷仍会上升；
3. **释放/组成改变：** 坏死、分泌、细胞组成或清除差异改变 bulk tissue 中可提取的游离核苷。

METTL1 的 CRC 因果研究完成了 proteomics/scRNA、tRNA m7G、催化失活突变、TRAC-seq/RNC-seq、细胞和动物表型以及 CCND3 rescue，但这只能证明 `METTL1–RNA m7G` 在 CRC 中有生物学意义，不能证明本地游离离子就是 m7G，更不能证明其升高来自 METTL1。相反，`13C-dynamods` 直接说明 modified-ribonucleoside abundance 可同时受 methylation deposition、RNA transcription 和 RNA decay 控制。

### 最短闭环

1. 同法标准先分开 `m7G / m2G / Gm / m2²G`；
2. 同一样本分别定量游离核苷、消化后的 tRNA/rRNA/mRNA 核苷；
3. 若进入机制阶段，做 `[13C-methyl]-methionine` pulse–chase 或 NAIL-MS；
4. METTL1 wild-type、knockout 和 catalytic-dead rescue 后，同时看 RNA modification 与 free-nucleoside pool。

当前允许写：`modified-guanosine positional-isomer families and an RNA-turnover/modification hypothesis`。

当前禁止写：`METTL1-driven modified-guanosine flux`。

主要依据：

- https://pubmed.ncbi.nlm.nih.gov/41627602/
- https://pmc.ncbi.nlm.nih.gov/articles/PMC8567201/
- https://www.nature.com/articles/s41467-020-20576-4

## 二、乙酰化多胺：谱学证据强，但精确异构体和生物来源都未锁定

### 本地已经成立

- feature 1717 有 73 张峰界内 MS2，m/z 100.0759 稳定为 base peak；
- RP–HILIC 同样本、组织内和患者配对差值均高度相关；
- 乙酰化多胺–MTA 模块在 Rmu 中具有高幅度、方向一致的患者级效应；
- 原论文正文没有展开该轴，因此它是比 Neu5Ac/肉碱更强的 narrative increment。

### 三种尚不能区分的生成机制

1. **宿主 SAT1/SSAT 乙酰化和外排增强；**
2. **结构名错误：** 本地峰可能更接近 CRC 文献证据更强的 N1,N12-diacetylspermine 或其他 mono/diacetyl isomer，而不是 N1,N8-diacetylspermidine；
3. **微生物贡献：** 结肠 biofilm 与宿主肿瘤可共同产生 tissue DiAcSpm pool，不能默认是癌细胞自主代谢。

CRC 原发组织的直接证据主要支持 N1,N12-diacetylspermine 与 N1-acetylspermidine；biofilm 研究中，抗生素清除 biofilm 后 tissue N1,N12-DiAcSpm 降至 biofilm-negative 水平。这个事实使“精确异构体标准 + host/microbiome source”成为必须并列的两道门。

### 最短闭环

1. 同法跑 `N1,N8-diacetylspermidine / N1,N12-diacetylspermine / N1-与N8-monoacetylspermidine`；
2. 标准、样本和 spike-in 的 RT 与多 CE MS2 同时通过；
3. 再测 SAT1 蛋白/活性、细胞内外比例；
4. 若要声称肿瘤细胞来源，补 biofilm FISH/16S 或空间共定位。

当前允许写：`acetylated-polyamine family compatible with enhanced acetylation/export`。

当前禁止写：`N1,N8-diacetylspermidine is confirmed and produced by tumour-cell SAT1`。

主要依据：

- https://pubmed.ncbi.nlm.nih.gov/23443255/
- https://pubmed.ncbi.nlm.nih.gov/6692383/
- https://pubmed.ncbi.nlm.nih.gov/25959674/
- https://pubmed.ncbi.nlm.nih.gov/34603448/

## 三、长链酰基肉碱：身份可升级，通量方向不能由丰度决定

### 本地已经成立

- feature 3222 为 long-chain/C20:4-acylcarnitine-like，Rmu–RN 中稳定升高；
- 原论文已明确描述 carnitine program，本项目的增量是更细粒度的长链锚点、raw-MS2 复核和机制纠偏；
- MTBLS7387 独立人体数据支持 CRC long-chain-lipid remodeling，但不复现 3222 的精确结构；
- acylcarnitine 与本地 purine/polyamine/amino-acid 模块的患者级协调不足，不支持统一上游链。

### 三种尚不能区分的生成机制

1. **输入增强：** CPT1A-dependent fatty-acid entry 增加，形成更多 LCAC；
2. **利用瓶颈：** CACT/CPT2 或 downstream beta-oxidation 受限，导致 LCAC 积累；
3. **底物/组成改变：** 脂肪酸供应、坏死、免疫或基质比例改变 bulk pool。

无论是“FAO 激活”还是“FAO 抑制”，都能产生 acylcarnitine 升高。只有 isotopologue fate、链长产物谱、acyl-CoA/TCA readout 和呼吸测量能区分。

### 最短闭环

1. `C20:4/C16:0/C18:0/C18:1 acylcarnitine` 标准与同位素内标；
2. chain-length-resolved targeted panel；
3. `[U-13C16]palmitate` 进入 LCAC、短链产物和 TCA 的时间轨迹；
4. OCR 与 CPT1A、CACT、CPT2/下游节点分开扰动。

当前允许写：`carnitine-shuttle imbalance with competing entry-versus-utilization hypotheses`。

当前禁止写：`FAO is activated/inhibited`。

主要依据：

- https://www.nature.com/articles/s41388-026-03835-4
- https://www.nature.com/articles/s41419-020-02936-6
- https://www.nature.com/articles/s41467-025-63243-2

## 四、Neu5Ac–mucin：目前最稳的是前体/转运与分泌黏蛋白背景，不是全局高唾液酸化

### 本地与外部已经成立

- feature 703 是原论文 Level-1 Neu5Ac 在正相面板的同队列正交恢复，Rmu 10/10 升高；
- TCGA 中 mucinous-relative `GNE/NANS/SLC35A1` synthesis/transport 轴在 clinical + six-lineage model 中 beta `+0.480`、BH q `2.64e-8`；
- secretory-mucin program beta `+0.922`、q `4.27e-11`；
- mucin-sialylation transferase 轴经组成敏感性校正后衰减到 beta `+0.113`、q `0.1067`。

这组结果支持选择性的 precursor/transport + secretory-mucin context，却明确反对把所有 sialyltransferase、所有 linkage 和 surface hypersialylation 合并成同一个方向。

### 三种尚不能区分的生成机制

1. GNE/NANS synthesis 与 SLC35A1 transport 扩大 Neu5Ac/CMP-Neu5Ac donor pool；
2. 特定 transferase 和 carrier protein 的 linkage-specific remodeling；
3. sialidase/lysosome 介导的 glycan turnover 释放更多 free Neu5Ac。

### 最短闭环

1. 同法定量 Neu5Ac、ManNAc 和 CMP-Neu5Ac；
2. linkage-aware N/O-glycomics 与 intact glycopeptides；
3. MAL-II/SNA 等正交 lectin readout；
4. sialyltransferase 与 sialidase 分开扰动，并把 free pool 与 conjugated pool 放在同一时间轴。

当前允许写：`selective mucinous-relative sialic precursor/transport and secretory-mucin remodeling`。

当前禁止写：`global hypersialylation`。

主要依据：

- https://www.nature.com/articles/s41598-024-79893-z
- https://www.nature.com/articles/s41598-022-26521-3

## 五、proline/glutamate：是一般 CRC 程序和基质背景，不是当前最强的亚型机制

- proline/glutamate 有 source-Level-1 身份桥、正相正交恢复和 10/10 Rmu 丰度方向；
- TCGA paired tumour–normal 和 pooled proteomics 支持 general CRC proline/P5C synthesis；
- mucinous-versus-conventional 的 lineage-adjusted proline synthesis 为 beta `-0.179`、q `0.0842`，并不支持黏液型特异增强；
- collagen/proline context 在 lineage adjustment 后明显衰减，bulk tissue matrix source 不能忽略。

要区分 tumour-cell redox/biomass synthesis 与 CAF/collagen turnover，需要 glutamine/proline tracing、hydroxyproline/collagen readout和空间/细胞来源；继续叠加 bulk RNA cohort 已不能跨过这道门。

## 六、验证优先级重新裁决

### P0：最能改变论文结论的身份门

1. `m7G / m2G / Gm / m2²G`：把最有原创性的 ion-family 发现升级为具体结构；
2. `N1,N8-DiAcSpd / N1,N12-DiAcSpm / monoacetylspermidines`：避免把文献更强的相邻异构体错命名；
3. `C20:4/C16:0/C18:0/C18:1 acylcarnitines`：确定 3222 及类群组成。

### P1：最小功能去向门

1. free nucleoside 与 RNA-digest nucleoside 配对；
2. free Neu5Ac、CMP-Neu5Ac 与 linkage-aware glycan 配对；
3. intracellular/extracellular acetyl-polyamine 配对；
4. chain-resolved acylcarnitine 与 acyl-CoA/short-chain products 配对。

### P2：真正进入因果机制

1. isotope time course；
2. genetic/pharmacologic perturbation；
3. phenotype and rescue；
4. spatial/cell-source localization；
5. independent patient validation。

## 七、当前文章的真实最高结论

> DreaMS-enabled, evidence-calibrated reanalysis recovered modified-guanosine and acetylated-polyamine ion families and orthogonally recovered proline, glutamate and Neu5Ac in paired colorectal tissues. The findings define parallel Rmu-associated abundance programs and a mucinous-relative sialic precursor/secretory-mucin context. Competing mechanism analysis shows that RNA writing versus turnover, host versus microbial polyamine production, fatty-acid entry versus incomplete oxidation, and free versus glycan-bound sialic acid remain experimentally distinguishable alternatives rather than settled causal mechanisms.

这个结论比“多组学都支持某通路”更严格，也更接近高水平审稿的逻辑：每个主张都有证据上限、反解释和一项可以推翻它的实验。
