# MTBLS13729 生物学成果整合与机制论文边界（2026-08-30）

## 结论

> **2026-08-30 v2 更新（优先于下文早期 15-candidate 版本）**：冻结候选总账现为 18 个节点。新增 feature 345 proline、374 glutamic acid 和 703 Neu5Ac，均为原论文 Level-1 身份在正相 RPLC 中的同队列正交找回，不是新分子，也不是独立队列复现。完整证据矩阵见 `data/mtbls13729/mechanism_evidence_matrix_v2/`。

当前最扎实的生物学结果不是单个“神奇代谢物”，而是 Rmu 发现亚组中四组方向一致、留一代谢物后仍稳定的丰度程序：

| 模块 | 节点 | 平均模块 log2FC | 同向患者 | bootstrap 95% CI | 当前含义 |
|---|---|---:|---:|---:|---|
| 乙酰化多胺–MTA | N1-acetylspermine、N1,N8-diacetylspermidine-like、MTA | +3.52 | 10/10 | 2.38–4.91 | 最强幅度与多节点收敛 |
| 嘌呤/修饰核苷池 | hypoxanthine、methylguanosine family、dimethylguanosine family | +2.36 | 9/9 | 1.60–3.13 | 最有算法新增价值的结构家族 |
| 长链酰基肉碱 | myristoyl-、palmitoyl-、C20:4-acylcarnitine-like | +1.66 | 10/10 | 1.07–2.27 | 支持 carnitine-shuttle imbalance，但不判定 FAO 通量方向 |
| 大中性氨基酸池 | isoleucine、phenylalanine、tryptophan | +0.94 | 10/10 | 0.58–1.32 | 支持氨基酸池积累；不等于 kynurenine 通路激活 |

这些是 phenotype-aware 候选筛选后的描述性收敛结果，不是全 feature-space FDR 确认，也不是独立队列复现。

## 零、原论文效应对账与真正的新颖性

15 个冻结候选不能统一称为“新代谢物”。逐项对账后的构成为：

| 类型 | 数量 | 可写贡献 |
|---|---:|---|
| 原论文身份的重提取/重映射 | 9 | 身份与原始数据重新连接；不是新身份 |
| 原论文身份存在，但本次获得不同色谱峰或正交支撑 | 1 | feature 1717 的独立 pos-RP 乙酰化多胺信号 |
| 原论文 identity table 未列出的算法新增家族 | 4 | palmitoylcarnitine、C20:4-acylcarnitine-like、methylguanosine family、dimethylguanosine family |
| 谱学身份相容但生物学被降级 | 1 | taurine；身份强不能替代跨面板丰度失败 |

10 个可与原论文具体身份行连接的节点中，9 个在原表 Rmu-vs-normal 达到 nominal `p<0.05`，6 个在原表 all-cancer-vs-normal 达到 FDR `<0.05`。本次重提取与原表 Rmu 效应方向 10/10 一致，Spearman `rho=0.830`（`p=0.00294`），Pearson `r=0.887`（`p=0.000624`）。这是同一队列的技术一致性与管线复核，不能称为独立生物学复现。

因此，当前最强新颖性不是“重新发现原论文已经命名的分子”，而是：

1. 把原论文身份、原始 DDA、跨面板峰和配对重定量连接成可审计证据链；
2. 找到 4 个原 identity table 未列出的谱学家族，其中修饰鸟苷家族最有结构与机制增量；
3. 用 source Level-1 证据推翻 feature 722 的弱 synephrine 投票，并用丰度正交证据降级 taurine，证明算法输出必须接受证据校准。

机器可复核审计位于 `data/mtbls13729/biology_novelty_audit_v1/`。

## 一、身份证据经过重新分层

### A 层：可映射到原论文标准品/高等级注释表的节点

以下不是“我们重新做了标准品”，而是将原论文 source table 的身份重新映射到我们从原始 mzML 重提取的峰：

| feature | 身份 | source MSI | 同模式 m/z/RT 对齐或正交复核 | Rmu–RN |
|---:|---|---|---|---:|
| 347 | Myristoylcarnitine | Level 1 | pos-RP；0.28 ppm；RT −2.86 s | +1.96；10/10 |
| 457 | N1-Acetylspermine | Level 2 | pos-RP；2.06 ppm；RT −1.46 s | +3.40；10/10 |
| 494 | Methylthioadenosine | Level 2 | pos-RP；−1.24 ppm；RT −2.50 s | +2.34；9/10 |
| 83 | Isoleucine | Level 1 | pos-RP；1.58 ppm；RT +6.64 s | +0.99；9/9 |
| 722 | Phenylalanine | Level 1 | pos-RP；1.58 ppm；RT −3.19 s | +0.77；9/10 |
| 398 | Carnitine | Level 1 | pos-RP 对 pos-HILIC；59 样本，配对差值 rho 0.856 | +0.91；9/10 |
| 73 | Hypoxanthine | Level 1 | pos-RP 对 neg-RP；配对差值 rho 0.621 | +1.06；9/10 |
| 732 | Tryptophan | Level 1 | pos-RP 对 neg-RP；配对差值 rho 0.466 | +1.13；10/10 |
| 9900175 | Sphingosine | Level 1 | 原论文 pos-HILIC 峰原始 EIC 重提取 | +1.80；9/10 |

其中 feature 722 是一个关键纠错案例：DreaMS 的弱 Level-3 谱库投票指向 synephrine，但原论文 Level-1 phenylalanine 与重提取峰的前体和 RT 高度吻合。论文应将它写成“模型候选必须由色谱/标准证据校准”，而不是把 DreaMS 投票当最终真值。

### B 层：强谱学家族候选

- feature 150：59 张谱、58 个样本，DreaMS palmitoylcarnitine consensus，median similarity 0.916，agreement 1.0；无原论文同一 exact row。
- feature 3222：long-chain/C20:4-acylcarnitine-like；类别碎片稳定，但双键位置、立体结构和精确异构体未解析。
- feature 1717：acetylated-polyamine/N1,N8-diacetylspermidine-like；RP–HILIC 跨样本 rho 0.860、组织内 rho 0.756、配对差值 rho 0.719，HILIC 85 个注释特征中 rank 1；源 MAF 无可靠 MS2 分数且 m/z 偏差较大，因此保留 `-like`。
- feature 1597/3019：methyl-/dimethylguanosine positional-isomer families；核糖丢失和跨加合物一致，但位置异构体不可分。

### C 层：降级或隔离

- feature 428 taurine：DreaMS 谱学很强（72 张支持谱、median similarity 0.912、agreement 0.986），但 neg-RP 对 neg-HILIC 的组织内 rho 仅 0.049，配对差值 rho −0.143；因此身份相容不等于 Rmu 生物学信号得到跨面板复现。
- feature 722 的 synephrine 投票被 phenylalanine source identity 推翻。
- feature 725 的 pyridoxine、301 的 trigonelline、1070 的 skatole只保留探索级，不进入核心机制。
- 4 个 m/z 843–862 高质量峰没有峰界 MS2，隔离为 MS1-only 候选。

## 二、模块之间不能被强行串成单链

患者级模块协调结果：

| 模块对 | Spearman rho | 6 对比较 BH q | 裁决 |
|---|---:|---:|---|
| 氨基酸–嘌呤/修饰核苷 | 0.833 | 0.053 | 边缘探索信号；值得外部复核 |
| 多胺–氨基酸 | 0.612 | 0.208 | 不足以建立共同调控 |
| 多胺–酰基肉碱 | 0.055 | 0.948 | 基本独立 |
| 酰基肉碱–嘌呤 | −0.033 | 0.948 | 基本独立 |

因此当前主文应使用 `parallel abundance programs`。可以提出“氨基酸–核苷池耦合”作为次级假说，但不能写成代谢流或共同上游酶机制。

## 三、最有价值的三个机制方向

### 1. 乙酰化多胺–MTA 周转

这是当前幅度最大且多节点最完整的方向：N1-acetylspermine、N1,N8-diacetylspermidine-like 和 methylthioadenosine 同时升高。它支持“多胺乙酰化/输出与甲硫氨酸–腺苷循环相关 pool accumulation”的假说，但不能从静态丰度判定 SAT1、SMS/SRM、MTAP 或 AHCY 的反应方向。

真正的机制升级需要：

1. N1/N8/N1,N8 乙酰化多胺标准区分；
2. 同位素标记 ornithine/arginine/methionine；
3. SAT1 或 MTAP/AHCY 扰动；
4. 细胞内外定量和 rescue；
5. 免疫/基质来源定位。

### 2. 长链酰基肉碱积累与利用歧义

myristoyl、palmitoyl、C20:4-like 三个节点和 free carnitine 同向升高。最安全的解释是 `carnitine-shuttle imbalance`。可能原因至少包括脂肪酸输入增加、CPT1A entry 增强、CACT/CPT2/下游 beta-oxidation 瓶颈、不完全氧化和组织组成差异。

2026 Oncogene 的 carnitine/acetylcarnitine–CPT1A 研究在 1,257 人发现/验证、400 人独立验证、AOM/DSS、高脂饮食、细胞/动物处理、CPT1A silencing 和 FXR/BHB rescue 后才提出因果机制。它提示 carnitine pool 上升也可能对应 FAO entry 增强，而不是我们可以凭 abundance 直接写“FAO 被抑制”。来源：https://www.nature.com/articles/s41388-026-03835-4

### 3. 嘌呤/修饰核苷与大中性氨基酸耦合

hypoxanthine、methylguanosine family、dimethylguanosine family 同向升高；tryptophan、phenylalanine、isoleucine 也一致升高，且两个模块在 9 个完整患者中的 rho 为 0.833。该结果可以提出核苷周转与氨基酸 pool coupling，但不能直接指定 METTL1、RNA turnover、SLC7A5 或 kynurenine–AhR。

尤其需要避免：本数据只有 tryptophan 上升，没有 kynurenine/xanthurenate 的可靠身份和比值，因此不得写“IDO/Kyn pathway activation”。2025 年的 SLC7A5–Kyn/XANA–AhR 工作使用 targeted metabolomics、isotope tracing、SLC7A5 knockdown、scRNA、conditioned medium 和外源 Kyn/XANA rescue 才建立免疫机制。来源：https://pubmed.ncbi.nlm.nih.gov/42563490/

## 四、对标高水平非靶向机制论文：它们实际做了什么

| 标杆 | 发现层 | 身份/定量层 | 因果层 | 对本项目的硬要求 |
|---|---|---|---|---|
| CRC AHCY, Nature Metabolism 2023 | GEMM + untargeted LC-MS + MSI | targeted MS 与体内同位素 | organoid、shRNA/药理、13C5-methionine、in vivo | bulk 差异只是入口，机制来自定位、示踪和扰动；https://www.nature.com/articles/s42255-023-00857-0 |
| FAP multiomics, Nature Cancer 2024 | 93 份 normal/benign/dysplasia；转录、蛋白、非靶向代谢、脂质 | 跨组学同一通路收敛 | 主要为 progression context，非完整单代谢物因果 | 即使多组学显著，也把结论停在早期通路事件；https://www.nature.com/articles/s43018-024-00831-z |
| Carnitine/CPT1A, Oncogene 2026 | 大人群发现与独立验证 | 候选面板定量 | diet/mouse/cell、knockdown、agonist 与 metabolite rescue | 我们的 acylcarnitine 只能叫机制假说；https://www.nature.com/articles/s41388-026-03835-4 |
| DKC1–sphingolipid, Nature Communications 2026 | patient tissue + omics | lipid pathway readouts | DKC1 perturbation、therapy-resistance phenotype | 单个 sphingosine 上升不能升级为 broad sphingolipid biosynthesis mechanism；https://www.nature.com/articles/s41467-026-72800-2 |
| SLC7A5–Trp, 2025 | 临床/组学定位 | targeted Kyn/XANA + isotope tracing | knockdown、scRNA、conditioned medium、rescue | tryptophan pool 不能替代下游产物和 tracer；https://pubmed.ncbi.nlm.nih.gov/42563490/ |

高水平机制论文的共同模板不是“通路富集 + 网络图”，而是：

> 非靶向发现 → 结构/定量确认 → 独立复现 → 细胞/空间来源 → 同位素来源/通量 → 基因或药理扰动 → 代谢物/基因 rescue → 动物或临床外推。

## 五、现在可发表的最强主线

### 推荐标题强度

> DreaMS-enabled evidence-calibrated reanalysis reveals convergent polyamine, modified-nucleoside, amino-acid and acylcarnitine abundance programs in paired mucinous colorectal tissues

### 推荐核心结论

> Reanalysis of raw paired-tissue LC–MS/MS recovered four directionally stable abundance programs in the Rmu discovery subgroup. Published source assignments and orthogonal chromatographic panels strengthened several named nodes, while raw DDA consensus recovered additional positional-isomer families. The strongest convergence involved acetylated polyamines and methylthioadenosine, followed by modified nucleosides, long-chain acylcarnitines and large neutral amino acids. These programs motivate testable turnover and substrate-utilization hypotheses, but do not establish mucinous specificity, metabolic flux or enzyme causality.

### 不能写

- “证明了黏液型 CRC 特异性代谢重编程”；
- “证明 SAT1/METTL1/CPT1A/DKC1 被激活或抑制”；
- “tryptophan 上升证明 kynurenine–AhR 激活”；
- “acylcarnitine 上升证明 FAO 上升或下降”；
- “DreaMS top-1 就是结构真值”；
- “模块符号检验是全 feature-space FDR”。

## 六、现实条件下的完成优先级

### 不需要新湿实验、现在就能完成

1. 将 15 候选冻结总账、4 模块、患者级模块矩阵和失败候选放入主文/补充表；
2. 用原始 EIC 与 representative MS2 做候选证据图；
3. 逐候选对账 original annotation、DreaMS、P2b/BioAware 与最终证据身份，量化 corrected/introduced/abstained；
4. 把 GSE236696、TCGA、FAP/CRC 外部研究严格标为 pathway context，而不是代谢物复现；
5. 把“未找到独立 mucinous metabolomics cohort”写入系统检索审计。

### 若可购买少量标准

P0：N1,N8-diacetylspermidine、N1-acetylspermine；m7G/m2G/Gm/m2²G 小面板；C20:4/C16:0/C18:0/C18:1 acylcarnitine。

每个标准至少完成同法 RT、多 CE MS2、样本+standard spike-in 共洗脱。若不能做这些，保持 source remap/family-like 身份，不伪造新 MSI Level 1。

## 七、可复核工件

- `data/mtbls13729/integrated_biology_ledger_v1/integrated_candidate_ledger.csv`
- `data/mtbls13729/integrated_biology_ledger_v1/integrated_module_summary.csv`
- `data/mtbls13729/biology_novelty_audit_v1/candidate_novelty_layers.csv`
- `data/mtbls13729/biology_novelty_audit_v1/report.json`
- `data/mtbls13729/convergent_metabolic_modules_v1/`
- `data/mtbls13729/module_coordination_v1/`
- `data/mtbls13729/named_candidate_crosspanel_audit_v1/`
- `data/mtbls13729/expanded_candidate_dreams_consensus_v1/`
- `data/mtbls13729/convergent_biology_figure_v1/mtbls13729_convergent_biology.png`
- `tasks/build_mtbls13729_integrated_biology_ledger.py`
- `tasks/analyze_mtbls13729_convergent_metabolic_modules.py`
- `tasks/analyze_mtbls13729_module_coordination.py`
- `tasks/plot_mtbls13729_convergent_biology.py`

## 八、总边界

这是一个证据校准的临床发现闭环：身份、谱学、跨面板技术正交和配对丰度分别记录。它没有独立 Rmu 代谢组复现、同位素通量、酶扰动、rescue 或体内功能，因此不能包装为已完成的因果机制论文。当前最有把握的文章形态是“算法方法 + 可复核的真实生物学应用 + 明确的机制假说和失败边界”。

## 九、v2 生物学整合：两个层级、六个轴

### 9.1 新增的三个源身份正交恢复

| feature | 身份 | Rmu–RN mean log2FC | 同向 | 跨面板 paired rho | 身份结论 |
|---:|---|---:|---:|---:|---|
| 345 | proline | +1.299 | 10/10 | 0.814 | 原论文 Level-1 proline 的正相 RPLC 正交恢复 |
| 374 | glutamic acid | +0.715 | 10/10 | 0.849 | 原论文 Level-1 glutamate 的正相 RPLC 正交恢复 |
| 703 | Neu5Ac | +1.975 | 10/10 | 0.959 | 原论文 Level-1 Neu5Ac 的正相 RPLC 正交恢复 |

feature 301、1695、725 和 458 分别因质量/谱库冲突、患者配对不一致或位置异构体歧义被排除。失败候选不是附带噪声，而是说明证据门确实能够拒绝看似合理的身份。

### 9.2 更新后的生物学结构

当前结果应分为两个层级，而不是继续扩张为一条代谢链：

1. **一般 CRC 的 biomass/redox/matrix 程序**：proline、glutamate、其他大中性氨基酸以及 TCGA 的 PYCR/collagen 背景支持 proline/P5C–matrix 适应；TCGA 黏液型相对常规型的 proline 轴反而较低，因此不是 Rmu 特异机制。
2. **黏液型相对的 glycan/mucin 程序**：游离/可提取 Neu5Ac pool 在本地 Rmu 升高；TCGA 黏液型相对常规型富集 GNE/NANS/SLC35A1、ST3GAL4、ST6GALNAC1/2 和 MUC2/SPDEF，但一般 CRC 配对分析中的多个 sialic axes 下降。最高结论是选择性 glycome remodeling，不是 global hypersialylation。

原有 modified-guanosine、acetylated-polyamine、purine 和 long-chain acylcarnitine 继续作为平行 abundance programs。患者级协调不足以把六个轴画成一个上游调控网络。

新五模块患者级协调审计进一步支持这一点：Neu5Ac 与 expanded amino-acid pool 的 rho 为 `0.164`，与 purine/modified-nucleoside pool 为 `-0.033`，两者 BH q 均为 `0.948`；expanded amino-acid 与 purine 模块 rho 为 `0.833`，但在 10 组模块比较后 BH q 为 `0.0775`。所以 Neu5Ac/mucin-glycan 与 proline/P5C/purine 不能被画成同一条患者级耦合链。工件见 `data/mtbls13729/module_coordination_v2/`。

### 9.3 原论文增量的正确数字

- 原论文表：345 条注释，其中 Level 1 为 157、Level 2 为 188；
- 当前选择性证据面板：18 个候选；
- 其中 9 个是 source-identity remap，3 个是 orthogonal Level-1 recovery，5 个是 source-table-absent family candidate，1 个是 downgraded/control；
- **不能**用 `18/345` 或新增候选数除以全特征数宣称全局注释率提升，因为候选选择、分母和任务不同。

可审核增量是：已知身份获得新的色谱/极性与原始 MS2 证据、五个原表无精确身份的离子家族候选，以及对错误候选的主动降级。详见 `data/mtbls13729/original_paper_delta_v2/`。

### 9.4 跨队列证据和反证必须成对报告

- TCGA 32 对支持一般 CRC proline-synthesis 上升，但不支持黏液型相对增强；
- GSE236696 上皮 proline 轴为 5/6 正向，但 20,000 个表达匹配随机轴审计未通过，因此只能作方向背景；
- 独立黏液型 pooled proteomics 的 proline-synthesis 蛋白方向一致，但没有患者级统计；
- 单病例空间数据支持 secretory-mucin/goblet 和 collagen/CAF context，却不支持 tumour-wide proline 或 sialic-axis 升高；
- 因此当前证据达到 mechanism-supporting discovery，不达到 causal metabolic mechanism。

### 9.5 更新工件

- `data/mtbls13729/integrated_biology_ledger_v2/`
- `data/mtbls13729/mechanism_evidence_matrix_v2/`
- `data/mtbls13729/original_paper_delta_v2/`
- `data/mtbls13729/proline_sialic_summary_figure_v1/`

### 9.6 原文叙事与组织组成敏感性校正

- 原论文正文已经明确点名 Neu5Ac 和长链肉碱，proline/glutamate 只在 pathway/family context 中出现，acetylated-polyamine 未在正文展开；因此 Neu5Ac/肉碱的新增价值是正交 raw-MS2 recovery 与机制措辞纠偏，修饰鸟苷离子家族和乙酰化多胺是更强的叙事增量；
- 原论文把静态 pool size 直接写成 sphingolipid flux、sialic-acid conjugation 和 activated carnitine shuttle/FAO，本项目统一降级为 competing mechanistic hypotheses；
- TCGA broad-lineage sensitivity 中，sialic synthesis/transport 在临床+六类 lineage proxy 校正后仍为 beta `+0.480`、BH q `2.64e-8`，secretory-mucin program 为 `+0.922`、q `4.27e-11`；mucin-sialylation 轴衰减至 `+0.113`、q `0.1067`；
- proline-synthesis 的黏液型相对效应保留负方向但降至边界（beta `-0.179`, q `0.0842`），所以 proline/P5C 仍归入 general CRC context，而不是确认的 mucinous-specific axis；
- lineage proxy 不等于真实细胞比例，composition-adjusted 结果属于敏感性分析，不是新的独立验证。
- 6个FDR10/3个FDR05只针对冻结的555个 phenotype-blind positive-RP annotation targets；完整13,155-target discovery matrix没有FDR10 feature。候选面板FDR与全空间FDR必须分栏报告。

对应机器可审计工件：`data/mtbls13729/source_narrative_audit_v1/` 与 `data/external/TCGA_COADREAD_Xena_20260830/proline_sialic_lineage_sensitivity_v1/`。

### 9.7 同队列匹配背景与竞争机制更新

五个冻结丰度模块均在三种 outcome-blind 匹配定义下超过 m/z、RT、检测率和 ion-family-size 相近的背景。主 acquisition-only 匹配中，模块平均效应与背景均值分别为：polyamine–MTA `2.582 vs 0.155`，purine/modified-guanosine `1.414 vs 0.126`，long-chain acylcarnitine `1.588 vs 0.064`，expanded amino-acid `0.914 vs 0.321`，Neu5Ac `1.833 vs 0.185`。这支持分子程序并非简单 acquisition background，但由于候选来自同一队列，经验 tail area 仅作 post-selection descriptive robustness，不作独立统计确认。

同时，5 个轴、14 个竞争机制已冻结在 `data/mtbls13729/competing_mechanism_trees_v1/`。其核心是：modified-guanosine 的 writer 与 RNA turnover、polyamine 的 host 与 biofilm source、acylcarnitine 的 entry 与 utilization bottleneck、Neu5Ac 的 free pool 与 linkage-specific glycan destination 均不能由当前静态 abundance 唯一决定。完整说明见 `docs/MTBLS13729_COMPETING_MECHANISM_TREES_20260830.md`。

### 9.8 亚型敏感性裁决：主轴收敛到 Neu5Ac

新的五模块 exact-permutation 审计把 `Rmu-RN` 与 `(Rmu-RN)-(Rtu-RN)` 分开。五个模块都在 Rmu 内升高，但只有 Neu5Ac 的亚型交互在 raw 与 PQN 两种归一化下通过五模块 BH 校正：差值分别为 `+2.209` 和 `+2.142 log2`，q=`0.00179` 和 `0.00162`。多胺–MTA、修饰鸟苷/嘌呤、长链 acylcarnitine 和 expanded amino-acid 的跨归一化最大亚型 q 分别为 `0.181`、`0.125`、`0.125` 和 `0.647`，应降为 general tumour 或低覆盖探索程序。

17 个正相冻结候选的逐节点复核得到相同结论：feature 703 Neu5Ac 是唯一候选面板亚型 BH q<`0.01` 的充分覆盖身份锚；1597、3019 和 3222 仅有 `2/2/4` 个严格有效 Rmu 患者，不能承担亚型主结论。整合身份、覆盖、候选面板统计、全13,155-target exact FDR、匹配背景和外部证据后的 claim scorecard 将18个节点分为：1个 `PRIMARY_SUBTYPE_ANCHOR`、9个 `GENERAL_TUMOUR_SUPPORT`、1个 `FAMILY_VALIDATION_PRIORITY`、5个 `LOW_COVERAGE_IDENTITY_VALIDATION`、1个 `CONTEXT_ONLY` 与1个 `NEGATIVE_CONTROL`。没有候选通过全空间 exact-FDR10。

因此当前最强故事不再是多个轴共同定义 Rmu，而是：**Neu5Ac/mucin-glycan remodeling 是唯一有同队列亚型敏感性证据的主轴；其余轴提供并行肿瘤代谢背景与待验证离子家族。** 独立 CRC N-glycomic 与 MUC2 spatial glycopeptide 研究支持黏液型糖链结构异质性，但均不构成 feature 703 的独立患者级复现。
- `docs/MTBLS13729_PROLINE_SIALIC_CROSSCOHORT_RESULT_20260830.md`
