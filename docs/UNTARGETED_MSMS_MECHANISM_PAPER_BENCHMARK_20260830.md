# 非靶向 LC–MS/MS 从算法到机制论文的对标框架（2026-08-30）

## 一、先区分三种论文，不能互相借结论

### A. 注释算法论文

核心问题是“候选排序/注释覆盖和准确率是否优于现有方法”。最低证据包括：

1. 身份、分子式、骨架等层面的严格无泄漏划分；
2. 与谱库、CFM-ID/SIRIUS/MetDNA/NetID 等基线在同一候选协议比较；
3. Top-k、校准、错误率、seen/unseen formula/scaffold 和仪器迁移；
4. target–decoy、合成 decoy 或锁定外部测试；
5. 标准品或独立谱库上的结构级真值；
6. 对错误传播、同分异构体和候选覆盖边界做压力测试；
7. 实际样本应用只能证明可用性，不能替代算法基准。

MetDNA 用标准谱种子和反应对递归传播，但其论文仍以 200 个加标标准、被遮蔽标准和外部谱库验证正确/异构/错误率；正确结构的 top-3 比例约 70–80%，不是“传播出来就是身份”。NetID 用 m/z、RT、MS2、同位素/加合物和生化/非生化边做全局优化，同时用 target–decoy、人工 gold standards、标准品和同位素实验验证新代谢物。MetDNA3 对网络传播候选仍合成标准并匹配 RT 与 MS2。

### B. 非靶向生物学发现论文

核心问题是“哪些可复核代谢变化与表型相关”。最低证据包括：

1. 设计与主要终点预先明确，病例/批次/配对关系正确进入模型；
2. pooled QC、blank、漂移和离子抑制控制，或在数据缺失时诚实使用替代方案并报告限制；
3. feature-level 多重校正和效应量，而非只挑 nominal p；
4. ion family 去冗余，避免把同一分子的同位素/加合物/源内碎片计成多个发现；
5. 重要候选的原始 EIC、峰界、DDA/DIA 采集覆盖、谱图纯度与批次稳定性；
6. MSI 等级和具体结构名称严格匹配证据；
7. 独立队列、正交色谱/极性、空间或细胞类型证据至少一项；
8. 通路富集必须基于可靠身份或 feature-level network 方法，并报告背景集与不确定性。

Nature Methods 的报告指南强调标准品、RT/MS、回收、基质效应、离子抑制、手工歧义复核和正确 MSI 分级。DeepMet 的实际发现率也揭示同一边界：67 个购买/合成候选中只有 25 个同时匹配组织峰的 RT 和 MS/MS（约 37%；更严格口径约 31%），说明先进生成模型的 top-1 仍只是实验候选。

### C. 机制/治疗靶点论文

核心问题是“某个代谢反应或分子是否因果性改变表型”。仅有静态组织丰度不够。高水平闭环通常包含：

1. 结构终证：标准品同法 RT + MS2；必要时 spike-in 共洗脱、NMR 或合成异构体反证；
2. 代谢来源：13C/15N/2H 示踪，证明原子从哪个前体进入候选；
3. 候选酶：基因敲低/敲除和药理抑制至少一种，最好两者方向一致；
4. 生化反应：纯化酶、细胞裂解液或 targeted metabolomics 证明底物/产物变化；
5. 表型：细胞/类器官/动物中的增殖、存活、分化、药敏或转移；
6. rescue：补加下游代谢物、回补基因或去除上游底物挽救表型；
7. 空间/细胞定位：说明变化发生在肿瘤细胞、基质、免疫或坏死区，而不是 bulk composition；
8. 独立人群或临床关联；
9. 若声称通量，必须有同位素动力学或稳态 flux 模型，不能由 abundance 推断。

CRC 的 AHCY 工作是典型标杆：GEMM + LC–MS/MS + DESI/MALDI 空间定位发现 methionine-cycle 变化；MS/MS 和体内示踪支持离子；TCGA 和人组织支持 AHCY；药物抑制和 shRNA 均降低类器官生长；13C5-methionine 追踪显示 SAH/trimethyllysine/cystathionine 改变；体内抑制降低肿瘤负荷。这里“机制”来自扰动、示踪和表型，不来自单次 untargeted 差异表。

## 二、代表性论文的证据链

| 工作 | 计算发现 | 结构/注释验证 | 生物学验证 | 我们应借鉴的硬门 |
|---|---|---|---|---|
| MetDNA, Nat Commun 2019 | 标准谱种子 + 反应对递归传播 | 200 标准加标、外部谱库、被遮蔽标准；报告 correct/isomer/error | 通路定量展示 | 网络邻近不是身份；必须测错误率和传播错误 |
| NetID, Nat Methods 2021 | 质量/RT/MS2/同位素/加合物/生化边全局优化 | target–decoy FDR、人工 gold standards、标准品、13C | 发现 thiamine derivatives、N-glucosyl-taurine | 图优化要有 decoy FDR 和正交真值 |
| MetDNA3, Nat Commun 2025 | knowledge + data 两层网络 | 新 γ-Glu-Thr-Gly、N-glycolyltaurine 用合成标准 RT/MS2 验证 | 应用展示 | 两层网络提高候选质量，但最终仍回到标准品 |
| DeepMet, Nature 2026 | 语言模型提出未识别哺乳动物代谢物 | 67 个候选购买/合成，25 个 RT+MS2 成功；失败也完整报告 | 饮食、抗生素、组织分布、动物 isotope tracing | 必须把 prediction-to-standard success rate 当主结果 |
| NetID + isotope identification, Nat Methods 2025 | 多底物 isotope reference material 做未知峰识别 | 同位素来源正交于 MS2，发现 TMGL | 生物来源与结构候选连接 | 同位素可解决 MS2 难分的异构体，但不能替代标准 |
| CRC AHCY, Nat Metab 2023 | 多模态 untargeted/MSI 定位 APC 相关 methionine cycle | MS/MS + in vivo isotope tracing | 药理 + shRNA + 类器官 + 体内肿瘤 + 13C5-methionine | 机制论文必须有 target perturbation + flux/readout + phenotype |
| CRC unresolved inflammation, Gut 2025 | 40 对 untargeted lipidomics 提出 AA mediator 失衡 | 81 对 targeted quantitative LC-MS/MS、标准品 EIC/谱图 | qRT-PCR、TCGA/大队列、62 人 scRNA、8 例空间转录组 | 从 non-targeted 发现转向更大 targeted 队列，并把代谢物量与酶/细胞来源/空间区域一一对齐 |
| Acidic pH–SAT1–acetylspermidine, 2023 | 非靶向/定量代谢组筛酸性 pH 产物 | N1-acetylspermidine 定量与 SAT1 操控 | SAT1 knockdown/overexpression、免疫分析、血管生成、体内肿瘤 | 相关轴要升级为机制，至少需要候选酶扰动、命名代谢物定量和表型 rescue/反证 |
| ATF6–LCFA–microbiota, Nat Metab 2025 | 人 CRC 配对组织与 ATF6 小鼠/类器官脂肪酸发现；公开 MTBLS7387 | C22:4/C22:5/C22:6 同法标准品；来源表含 251 完整处理配对 | 类器官 D3-FA elongation、FASN 抑制、无菌/FMT、BONCAT/细菌生长/H2S | 人体丰度、来源示踪、宿主扰动和微生物功能分别闭环；不能用人体静态丰度替代其中任何一层 |
| Extrinsic lipids in CRC, Gut 2026 | 152 例发现队列定量总脂肪酸，28 例独立验证 | GC-MS/SIM 定量；组织学选区保证肿瘤细胞比例 | Apc 小鼠口服稳定同位素示踪；无菌/SPF；CD36 与 CPT1A 抑制；5 例患者类器官 | 独立人群 + in vivo 来源通量 + 2D/3D 功能是脂质机制标杆；同时提示长链 PUFA 程序不天然等于某一 CRC 亚型特异 |
| Carnitine/acetylcarnitine–CPT1A, Oncogene 2026 | 1,257 人发现/验证代谢组；400 例独立验证 | carnitine、acetylcarnitine、β-hydroxybutyrate 面板；并非 C20:4-LCAC 身份验证 | AOM/DSS 高脂饮食模型、代谢物处理、CPT1A silencing、FXR agonist/β-hydroxybutyrate、炎症与屏障表型 | carnitine 类积累既可能伴随输入/FAO 增强，也可能来自利用障碍；静态 acylcarnitine 不能单向解释通量 |
| 13C-SpaceM, Nat Metab 2024 | MALDI-AIF、显微图像和同位素分布建模连接到单细胞 | METASPACE 5% FDR、自然丰度校正、bulk LC-MS 对照 | U-13C glucose、ACLY 两条 shRNA、近单细胞空间脂肪酸合成/摄取 | bulk 平均可完全掩盖同一扰动下的代谢亚群；转录酶表达不能替代实际底物来源 |
| Spatial isotope deep tracing, Nat Commun 2025 | LC-MS/MS 与 AFADESI-MSI 的 MSITracer | 3种 LC 条件、人工剔除假阳性、连续组织切片 LC-MS/MS 对照 | U-13C glucose/glutamine、多个器官和时间点、动力学拟合 | 原生丰度只反映反应净结果；通路活性需要 isotopologue fraction 与时间维度 |
| Human glioma in-vivo tracing, Nature 2025 | 肿瘤/皮层代谢组和同位素网络 | 病理医生定量肿瘤含量，空间导航取材 | 8位患者术中 U-13C glucose 持续输注，平行小鼠模型，追踪 UDP-glucose、TCA 和 purine isotopologues | 人体代谢机制的强证据可以来自临床 tracing，但必须把血浆稳态、取材区域和组织组成一并控制 |

## 三、我们当前已经达到什么

### 已经具备

1. 60 例、30 对组织的四 panel 原始 LC–MS；primary Rmu-vs-RN 配对设计明确。
2. 缺 pooled QC 的事实已公开，使用患者内配对、三套归一化和 targeted EIC 重定量做替代稳健性。
3. 八候选统一证据账本；8/8 有峰界内 MS2，共 252 张。
4. feature 1717 有原始诊断碎片、跨色谱同样本相关和配对差值相关。
5. feature 3222 有 carnitine class motif；修改鸟苷有核糖丢失和跨加合物一致。
6. GSE236696 上皮敏感性、TCGA paired tumor-normal 和 MTBLS8090 阴性外部代谢组已经分层报告。
7. DreaMS/P2b/BioAware 的算法结论与生物学结论分开，网络证据不直接当身份标签。
8. GSE236696 的患者配对多胺/酸性/趋化轴、三种上皮门控和表达匹配随机集已完成；GSE236697 单病例空间定位也已完成并保留不共定位的阴性结果。

### 尚未达到

1. 没有一个核心候选达到 authentic-standard RT + MS2 + spike-in 的 MSI Level 1。
2. 没有独立带 Rmu/CMS/MSI 标签的原始组织代谢组复现。
3. 没有 pooled QC/blank，无法恢复经典 QC-CV 和漂移模型。
4. 没有源研究准确恶性上皮/CNV 标签；当前 GSE 上皮门是 marker-based sensitivity analysis。
5. 没有 isotope tracing、基因/药物扰动、rescue 或类器官；所以不能声称通量、酶活或治疗靶点。
6. 八候选统计来自候选选择后的 discovery panel；不能宣称全 feature-space FDR 确认。
7. 单病例空间转录组不能提供人群复现；其肿瘤与癌旁测序深度差异很大，禁止把 spot 当生物学重复或用 spot p 值制造显著性。

## 四、现实约束下的最强可发表闭环

在不能获得新组织、类器官和示踪条件时，目标应是**高水平算法应用 + 严格临床发现**，而不是伪装成完整湿实验机制论文。

### 计算端必须完成

1. 冻结 DreaMS/P2b/BioAware 模型和候选协议；在真实 MTBLS13729 谱图上报告注释覆盖、冲突、置信度和新增候选。
2. 用 raw mzML 对所有论文候选做峰界和 MS2 consensus，不再依赖通用桥接表。
3. 做 ion-family 去冗余和同位素/加合物/源内碎片图，避免重复发现。
4. 做全 feature-space 统计或明确两阶段 selective inference；八候选单独作为 discovery-priority panel。
5. 把 negative controls（3180、外部不复现轴、decoy network）写进主文，而不是只展示成功案例。

### 最小实验端（若只能购买少量标准）

1. 第一优先 N1,N8-diacetylspermidine：最可能把 feature 1717 直接升级为命名发现。
2. 第二优先 C20:4/C18:1/C18:0/C16:0 acylcarnitine 组合：验证是否为类群重塑。
3. 第三优先 m7G、m2G/Gm、m2²G：解析位置异构体。
4. 每个标准至少做同法 RT、多个 collision energy 的 MS2 和样本 spike-in；能有同位素内标则加半定量。

## 五、论文主叙事的正确强度

当前可成立的主叙事是：

> A DreaMS-enabled, evidence-calibrated reanalysis of a paired colorectal tissue metabolomics cohort recovered reproducible ion families that were missed by the original m/z–RT-only annotation table. The strongest candidates define modified-guanosine, long-chain acylcarnitine and acetylated-polyamine abundance programs in the Rmu discovery subgroup. Raw DDA fragmentation, cross-adduct/cross-chromatography concordance and external transcriptomic context support these programs, while standards and independent subtype-resolved cohorts remain necessary for exact identities and mucinous specificity.

不能成立的叙事是：

> We proved mucinous-specific purine/FAO/polyamine flux reprogramming and identified its causal enzyme.

## 六、主要来源

- MetDNA: https://www.nature.com/articles/s41467-019-09550-x
- NetID: https://pmc.ncbi.nlm.nih.gov/articles/PMC8733904/
- MetDNA3: https://www.nature.com/articles/s41467-025-63536-6
- DeepMet: https://www.nature.com/articles/s41586-025-09969-x
- Isotope-based metabolite identification: https://pmc.ncbi.nlm.nih.gov/articles/PMC12027066/
- CRC AHCY mechanism study: https://pmc.ncbi.nlm.nih.gov/articles/PMC10447251/
- Annotation/quantification/reporting guide: https://www.nature.com/articles/s41592-021-01197-1
- X13CMS and flux boundary: https://pmc.ncbi.nlm.nih.gov/articles/PMC7323898/

## 七、三个候选轴的逐项对标与缺口

| 本地轴 | 最接近的机制范式 | 标杆论文实际做了什么 | 我们已经有 | 当前缺口 |
|---|---|---|---|---|
| 修饰鸟苷/嘌呤 | METTL1–tRNA m7G–translation/tsRNA | 蛋白组或RNA修饰筛选；TRAC-seq/MeRIP；METTL1 gain/loss；催化失活突变；翻译组；CCND3或tsRNA表型 rescue | 离子家族、原始MS2、患者配对、独立Level-1反证、purine共变、外部蛋白/RNA背景 | 标准品、RNA修饰定量、writer/turnover扰动、来源示踪 |
| 乙酰化多胺 | acidic pH–SAT1–N1-acetylspermidine–neutrophil | non-targeted筛选后定量；SAT1 knockdown/overexpression；细胞内外代谢物；免疫细胞分析；中性粒细胞抗体反证；体内肿瘤 | 1717强MS2、跨色谱、配对丰度、酸性/趋化外部背景及空间阴性边界 | 精确异构体标准、SAT1蛋白/活性、分泌证据、髓系因果 |
| 长链acylcarnitine | FAO entry-versus-utilization | acylcarnitine类别定量之外，还测CPT1/CACT/CPT2/β氧化、OCR、同位素palmitate和药理/遗传扰动 | carnitine类MS2、原作者广泛carnitine program、外部FAO蛋白/RNA轴偏低 | 标准品与类别定量、同位素内标、OCR/示踪、关键酶扰动 |

由此得到的直接裁决是：三个轴都可形成高质量**机制假说**，但目前只有“算法发现—峰级谱学—患者配对丰度—外部背景”四层；没有一个轴达到“结构终证—来源示踪—酶扰动—表型/rescue”的完整机制门。

## 八、2024–2026 新标杆进一步给出的硬门

1. Gut 2024/2025 CRC lipid study 先在 40 对组织做 untargeted lipidomics，再在 81 对组织做 targeted quantitative LC-MS/MS，并用标准 EIC、qRT-PCR、大队列、scRNA 和空间转录组把代谢物、酶和细胞来源对齐。启示：非靶向发现不能直接承担最终定量和机制结论。
2. Nature Metabolism 的 SORD–糖驱动 CRC 转移研究把 global metabolomics 与 `13C6-fructose/13C6-glucose` tracing、SORD 机制和转移表型结合。启示：静态糖/有机酸丰度必须通过示踪和扰动才能成为代谢通量机制。
3. `13C-SpaceM` 和 spatial isotope deep tracing 显示 bulk abundance 会隐藏单细胞/区域异质性。启示：即使 TCGA/GSE 路径方向一致，也不能把 bulk 或 pseudobulk 分数等同于代谢流。
4. 2026 Spatially-guided MEtabolomics 将 DESI-MSI、H&E、bulk LC/GC-MS 和 RNA-seq 联合，并指出大量 bulk differential metabolites 可能来自非肿瘤区域。启示：MTBLS13729 的组织组成仍是 acylcarnitine/polyamine 解释中的现实混杂。
5. 2025 human glioma tracing 在 8 位患者手术期间给予 `[U-13C]glucose`，同时获得影像导航的 cortex、enhancing 和 non-enhancing tumour，并以病理量化 tumour content。启示：即便患者数少，直接同位素输入、动脉稳态和空间取材也能回答静态 abundance 永远回答不了的 nutrient fate。
6. 2025 spatial isotope deep tracing 明确指出 native metabolite abundance 是多条生成和消耗反应的净结果，不能等同 pathway activity；其 LC-MS/MS、AFADESI-MSI、U-13C glucose/glutamine 和时间动力学组合是我们使用 `flux` 一词前必须达到的层级。

据此把文章证据等级冻结为三层：

| 层级 | 要求 | 当前项目位置 |
|---|---|---|
| Descriptive discovery | 患者设计、FDR、效应量、EIC/MS2、去冗余 | 已完成，候选身份强度不一 |
| Mechanism-supporting | 独立队列、细胞/空间来源、多组学正反证 | 已达到一部分；亚型代谢组外部复现缺失 |
| Causal metabolism | 标准终证、tracing/flux、扰动、phenotype/rescue、体内 | 未达到 |

因此当前工作可以写“mechanistic context/hypothesis”，不能写“causal metabolic mechanism”。

### 公共数据计算闭环的现实上限

如果完全不能新增标准、tracing 或扰动，计算工作仍可达到一篇严谨的 **mechanism-supporting clinical discovery / algorithm application**，但不能靠继续堆公开转录组跨过因果门。高质量的公共数据闭环应至少同时具备：

1. 一个冻结 primary abundance endpoint（本项目为 paired Rmu-vs-RN），并把 subtype interaction 分开；
2. 原始 EIC、峰界 MS2、离子家族折叠和反例候选；
3. 原论文逐条增量审计，区分 source recovery、source-table-absent family 和真正的新结构；
4. 至少一个独立人体 abundance/targeted 数据集作路径背景，以及一个细胞/空间数据集作组成或定位敏感性；
5. 正反证并列：外部方向冲突时收敛到 context-dependent，而非挑选同方向队列；
6. 明确的标准品与机制实验优先级，使论文给出可执行假说而不是不可证伪故事。

继续增加 RNA-seq cohort 的边际价值已经明显低于标准终证。对 MTBLS13729，下一次真正跨等级的跃迁依次是：`m7G/m2G/Gm/m2²G` 与 `N1,N8-diacetylspermidine` 同法 RT+MS2、长链 acylcarnitine 标准组合、独立 subtype-resolved tissue metabolomics；只有再加入 isotope tracing/扰动/rescue 才能称代谢机制。

## 九、MTBLS7387 的正式复算如何改变我们的脂质主线

我们已经从 ATF6 论文 Fig. 3 的公开来源表重算人体配对脂肪酸数据，而不是只引用作者结论：

1. 来源矩阵为 `502` 行，即 `251` 对完整肿瘤–癌旁样本，共 `186` 个分子特征；
2. 按论文的配对 t 检验和全特征 BH-FDR，`56` 个特征达到 FDR<0.05，其中 `38` 个升高、`18` 个降低；`52/56` 同时通过 Wilcoxon FDR<0.05；
3. C20–C24 共 `59` 个特征，其中 `17` 个达到 FDR<0.05，`14` 个升高、`3` 个降低，说明长链重塑存在但不是单向全升高；
4. 游离 arachidonic acid 仅 `+0.0865 log2`、FDR `0.243`，并不显著；两个 hydroxy-C20:4 峰分别为 `+0.683`（FDR `9.58e-4`）和 `+0.414`（FDR `0.0418`）；
5. 同一分子式/链长的不同 RT 峰可以反向：hydroxy-C20:2 的 RT 5.88 峰 `+1.136`，RT 6.30 峰 `-1.469`。因此“C20:4/C20:2 类别升高”不能替代异构体级鉴定；
6. early CRC 的 99 对与 late CRC 的 152 对呈不同广度：前者有 5 个显著 C20–C24 特征，后者有 15 个；年龄分层是异质性描述，不是新的机制标签。

样本数字也已分层审计：论文文字报告 `259` 人；MetaboLights 样本表含 `258` 个 tumour 和 `258` 个 adjacent 条目；Fig. 3 处理矩阵只有 `251` 个完整配对。公开文件能确认净少 `7` 个处理配对，但不能确认排除原因；另有一处疑似 `315Tu2/315u` 标识拼写差异。正文必须使用 `251-pair processed analysis cohort`，并在可重复性补充材料中保留 259/258/251 三层，而不是选一个数字覆盖其余数字。

这项外部复算只支持 **long-chain-fatty-acid remodeling context**。它不能证明 MTBLS13729 feature 3222 就是 C20:4 acylcarnitine，不能证明 Rmu 特异、ATF6 激活、FAO 通量或酶活变化。投稿图见 `data/mtbls13729/mtbls7387_lcfa_context_figure_v1/mtbls13729_mtbls7387_lcfa_context.png`。

新增主要来源：

- ATF6–LCFA–microbiota: https://pmc.ncbi.nlm.nih.gov/articles/PMC12460170/
- MTBLS7387: https://www.ebi.ac.uk/metabolights/editor/MTBLS7387/descriptors
- Official analysis code: https://github.com/adamsorbie/Coleman_Sorbie_et_al_2025
- Extrinsic lipids in CRC, Gut 2026: https://pubmed.ncbi.nlm.nih.gov/41856524/
- Carnitine/acetylcarnitine–CPT1A, Oncogene 2026: https://www.nature.com/articles/s41388-026-03835-4
- 13C-SpaceM: https://www.nature.com/articles/s42255-024-01118-4
- Spatial isotope deep tracing: https://www.nature.com/articles/s41467-025-63243-2
- Human glioma in-vivo U-13C glucose tracing: https://www.nature.com/articles/s41586-025-09460-7

## 十、Neu5Ac 主轴对标：从游离代谢物到黏蛋白糖型还缺哪一层

候选级亚型审计后，feature 703 Neu5Ac 是本项目唯一通过候选面板 subtype-sensitivity 门的充分覆盖节点。但顶级 glycomics/机制论文说明，游离 Neu5Ac pool 只能定位到“可用唾液酸资源”，不能直接说明糖链载体、连接方式或细胞来源。

| 对标工作 | 实际测量层 | 它能回答什么 | 它不能替我们回答什么 |
|---|---|---|---|
| CRC N-glycan MALDI-MSI, Front Pharmacol 2024 | 病理标注区域内约100种 N-glycan 的空间相对丰度；mucinous/non-mucinous 分层 | 黏液型与非黏液型 CRC 具有可分的 N-glycome，空间与组织组成重要 | 公开补充表只有53行组成目录，无患者级强度；不能复算 Neu5Ac pool 或 linkage-specific sialylation |
| CRC O-glycomics, Theranostics 2022 | 微切割癌上皮/正常黏膜；PGC-LC-ESI-MS/MS；部分合成糖链标准 | CRC 中 core-2、α2-3 sialylation 与 sialyl-Lewis 型糖链富集，且与正常 core-3/α2-6 结构不同 | 研究不是黏液型亚型专门复现，也不测游离 Neu5Ac |
| MUC2 glycosite spatial MS, Nat Commun 2026 | StcE on-slide mucinase、MALDI-MSI 定位、LC-MS 解析完整 MUC2 glycopeptide | 三例黏液癌内不同 MUC2 glycoform 占据不同区域，直接证明 bulk pool 无法指定 glycan destination | 只有三例且无非黏液型对照，不能提供亚型人群统计或 feature 703 身份复现 |
| CRC AHCY, Nat Metab 2023 | untargeted/MSI发现 + 人/鼠背景 + 13C tracing + shRNA/药理 + organoid/in-vivo phenotype | 把静态代谢物变化升级为酶节点、流向和功能因果 | 我们当前没有对应的 GNE/NANS/SLC35A1 或 sialyltransferase perturbation/rescue |

由此冻结 Neu5Ac 轴的四级升级路径：

1. **当前已达：** paired Rmu abundance、source-Level-1 identity bridge、raw-MS2/跨面板恢复、Rmu-vs-Rtu interaction、TCGA composition sensitivity；
2. **最小结构闭环：** 同法 Neu5Ac 标准、RT/MS2、样本 spike-in，最好加同位素内标半定量；
3. **生物学去向闭环：** linkage-aware O/N-glycomics 或 MUC2 glycopeptide readout，区分 free pool、CMP-Neu5Ac donor 与 α2-3/α2-6 terminal incorporation；
4. **机制闭环：** GNE/NANS/SLC35A1 或候选 sialyltransferase 扰动，加 Neu5Ac/CMP-Neu5Ac/glycan readout 与表型 rescue；若声称通量，再加同位素来源与时间维度。

在完全无新湿实验的条件下，论文可达到第1级并用独立 glycomics 作结构背景，最高措辞为 `mucinous-relative selective Neu5Ac/mucin-glycan remodeling`。不能写 `enhanced sialylation flux`、`specific MUC2 glycoform accumulation` 或 `GNE-driven causal program`。

新增来源：

- Mucinous/non-mucinous CRC N-glycomic MALDI-MSI: https://pmc.ncbi.nlm.nih.gov/articles/PMC10808565/
- CRC core-2 O-glycomics: https://pmc.ncbi.nlm.nih.gov/articles/PMC9254241/
- MUC2 glycosite mapping and spatial MS: https://www.nature.com/articles/s41467-026-72853-3

## 十二、2026-08-31 raw-UMI组成审计后的最终对标与停机规则

最新原始患者级单细胞结果解决了一个计算条件下仍可解决、且审稿人必问的问题：转录信号究竟
来自cell composition还是within-epithelium state。既往黏液型CRC单细胞工作用3例mucinous与
4例classical样本观察到MUC2/FCGBP/REG4/SPINK4等goblet-like特征；另一项6对黏液癌
scRNA+spatial研究进一步用空间定位、共培养和免疫荧光连接细胞互作。它们共同说明，仅报告bulk
或全上皮差异不足以区分谱系比例和细胞状态。

本项目现在达到的计算闭环是：

1. 官方raw UMI、患者pseudobulk，不把cell当重复；
2. 作者cluster label构成的固定goblet-lineage fraction；
3. unadjusted、composition-adjusted与right/MMR敏感性并列；
4. 结果显示MUC2/SPDEF/NXPE1主要随组成衰减，而AGR2/SLC35A1保留正区间；
5. host NEU1/NEU3、蛋白模块、full-space FDR和独立代谢物复制的负结果同时保留。

这已经达到高质量**mechanism-supporting clinical discovery**对cell-source/组成混杂的要求，
但仍没有达到机制论文的biochemical source与flux门。近年强机制工作使用类器官、稳定同位素
tracing、遗传/药理扰动和rescue来区分pool size与reaction activity；动态单细胞代谢组和空间
同位素方法也明确指出高丰度不等于高通量。因此本项目的新增raw-UMI证据只能写成
`epithelial secretory-folding and Golgi donor-transport capacity`，不能写成SLC35A1 transport
flux、Neu5Ac synthesis flux或goblet-mediated causality。

### 最高性价比停机规则

- **停止**继续堆新的bulk RNA队列、network enrichment或相似的相关性图；它们不能跨过当前硬门。
- **无新湿实验时**，直接冻结Package A，完成算法增量、患者内代谢物、raw-UMI组成调整、外部
  糖组结构和全部负证据的投稿图表。
- **若只能增加一种实验组合**，优先当前LC方法中的Neu5Ac authentic standard + spike-in +
  同位素内标，并在同一样本增加linkage-aware O-glycan/MUC2 readout；这同时补身份终证和
  destination两个最大缺口。
- **若目标升级为因果机制**，必须另行获得可干预模型并加入isotope tracing、SLC35A1/GNE/NANS
  或具体glycosyltransferase扰动、目标代谢物/糖链readout和rescue。公共数据再分析不能替代这些门。

新增来源：

- Mucinous-vs-classical CRC single-cell profiling: https://pmc.ncbi.nlm.nih.gov/articles/PMC9870908/
- Six-patient mucinous CRC single-cell/spatial study: https://pmc.ncbi.nlm.nih.gov/articles/PMC11111627/
- EGFR/KRAS CRC organoid isotope tracing: https://pmc.ncbi.nlm.nih.gov/articles/PMC12162862/
- Dynamic single-cell metabolomics and isotope tracing: https://www.nature.com/articles/s41467-025-59878-w

## 十一、2026-08-31 更新：Neu5Ac 主轴收敛为 donor–carrier–core–linkage 解耦

为避免把 feature 703 的游离 Neu5Ac 升高直接等同于表面糖链高唾液酸化，我们按 2022 CRC
O-glycomics 提出的生物合成层次，预先冻结 donor、carrier、core 和 linkage 分支，并在 TCGA
COAD/READ 的 42 例黏液型与 329 例常规型原发肿瘤中进行临床协变量、组织组成和 MSI 敏感性
审计。结果不是一条统一上升的 sialylation pathway，而是显著解耦：

| 冻结分支 | clinical+lineage beta | BH q | 加 MSI beta | BH q | 裁决 |
|---|---:|---:|---:|---:|---|
| Neu5Ac donor supply/transport | +0.480 | 3.30e-8 | +0.448 | 5.36e-7 | 稳健相对富集 |
| secretory mucin carrier | +0.922 | 5.34e-11 | +0.859 | 4.79e-10 | 稳健相对富集 |
| mucosal core-3/Sda lineage | +0.879 | 1.76e-8 | +0.853 | 1.32e-7 | 肿瘤间相对保留 |
| core-2/sLeX transcript composite | -0.009 | 0.915 | +0.011 | 0.895 | 不支持整体转录升高 |
| alpha2-3 O-glycan sialylation | -0.439 | 0.0093 | -0.434 | 0.0141 | 相对降低 |
| ST6GAL1 | -0.742 | 8.50e-5 | -0.719 | 2.23e-4 | alpha2-6 N-glycan route 相对降低 |
| ST6GALNAC1 | +0.788 | 7.99e-8 | +0.761 | 5.36e-7 | 截短型 O-glycan route 相对升高 |
| GCNT3 | +0.417 | 0.0096 | +0.462 | 0.00217 | 分支酶相对升高 |

这组结果必须与外部患者结构数据联合解释。两个 MUC 病例在肿瘤间比较中具有最高的 core-3，
但相对各自正常组织 core-3 仍分别下降约 `34%`；与此同时 core-2、sLeX/A 与 core-2+alpha2-3
配对升高，alpha2-6 大幅下降。因此“黏液型中 core-3 较高”表示**黏膜分泌谱系的相对保留**，
并不表示肿瘤转化时 core-3 增加。最符合全部数据的模型是 hybrid mucin glycome：黏液型肿瘤
同时保留较强的 MUC2/core-3/Sda 黏膜谱系，并获得 core-2/sLeX 等肿瘤相关糖抗原；游离
Neu5Ac 资源池、供体转运、黏蛋白载体、糖链核心和末端连接方式不能互相替代。

近年对标进一步界定了新颖性和缺口：2026 年 988 例 CRC 的 computational sialylome 只在
转录层关联 mucinous histology、MSI 和 BRAF，且其 TCGA 部分与本项目重叠；2025 年
ST6GAL1–PD-L1 工作通过 knockdown/overexpression、lectin、IP、蛋白稳定性与小鼠治疗建立了
具体 alpha2-6 N-glycan 因果路线，但本项目的 ST6GAL1 和外部 MUC alpha2-6 方向均为负；
2026 年 MUC2 空间糖肽研究则证明同一黏液癌内不同 MUC2 glycoform 占据不同空间区域。故本项目
的可发表增量不是“首次发现 CRC sialylation”，而是利用 raw-MS2/配对定量恢复入口，并用相互
独立的代谢物、转录分支和患者糖组结构揭示 donor–carrier–core–linkage 解耦。

新增来源：

- Pan-cohort computational CRC sialylome, Biology 2026: https://pmc.ncbi.nlm.nih.gov/articles/PMC13162982/
- ST6GAL1–PD-L1 CRC mechanism, Advanced Science 2025: https://pmc.ncbi.nlm.nih.gov/articles/PMC12622430/

## 十二、跨论文机制硬门：不是“多组学越多越好”，而是每个因果问题必须有对应测量

为防止继续用 RNA、网络或相关性替代代谢机制，我们把近年代表性工作按其真正跨过的证据门重新
编码。符号 `✓` 表示该论文直接测量该层；`—` 表示该层不是研究目标，不能从其他列借结论。

| 工作 | 人体/临床发现 | 结构或定量终证 | 来源/通量 | 节点扰动 | 表型/体内 | 空间/细胞来源 | 对本项目的直接启示 |
|---|---|---|---|---|---|---|---|
| CRC AHCY, Nat Metab 2023 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | bulk untargeted 只负责发现入口；机制由 tracing、AHCY 抑制/shRNA、类器官和体内肿瘤共同建立 |
| SORD–sugary drink–CRC metastasis, Nat Metab 2025 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 转录高表达不足；需要底物去向、两个 KO clone、迁移/侵袭及多种转移模型 |
| CMS3–CPS1, 2024 | ✓ | targeted | `15N-NH4 -> UMP` | CPS1 inhibitor | organoid/cell vulnerability | IHC/亚型 | “亚型代谢”需要 subtype-resolved tracing 和选择性脆弱性，不能只靠 TCGA score |
| Diet-enhanced polyamine depletion, Nature 2025 | ✓ | targeted metabolite panel | 多前体 in-vivo isotope tracing | diet/drug combination | 小鼠肿瘤/治疗 | — | 即便起点是 n=10 人体 metabolomics，也用体内示踪判定 proline/arginine/ornithine 来源 |
| Human liver global flux, Nat Metab 2024 | 人体组织 | 69 个标准/RT/MS2 + isotope pattern | 多底物 13C + uptake/release + MFA | — | 组织功能 | intact slice | flux 需要 isotopologues、交换速率、模型拟合和假设诊断，不能由 abundance 命名 |
| Patient–PDX metabolic fidelity, Nat Metab 2025 | 配对患者–PDX | targeted/annotated panel | 患者与 PDX 平行 U-13C glucose | — | 连续传代 | 病理分区 | 模型系统会改变代谢；患者相关性与模型机制必须先做 fidelity audit |
| CRC O-glycomics, Theranostics 2022 | 配对患者组织 | PGC-LC-MS/MS + 部分合成标准 | — | — | — | LCM 上皮 | glycan core/linkage 是直接结构 readout，transferase RNA 不能替代 |
| MUC2 spatial glycopeptide, Nat Commun 2026 | 3块MUC/2位患者 | glycopeptide LC-MS | — | — | — | StcE + MALDI-MSI | 肿瘤MUC2以低唾液酸化/非唾液酸化糖肽为主且O-acetylated Neu5Ac低于健康结肠；bulk free Neu5Ac不能指定carrier destination |

这些工作给出一个对本项目不可再压缩的逻辑顺序：

1. **身份门回答“它是什么”；** 同法标准、RT、MS2、spike-in 或结构正交证据；
2. **丰度门回答“它是否改变”；** 锁定分母、配对统计、FDR、效应量、检测覆盖和 QC；
3. **来源门回答“它从哪里来”；** isotopologue fraction、前体输入和必要的时间维度；
4. **去向门回答“它去了哪里”；** 对 Neu5Ac 必须直接测 CMP-Neu5Ac、具体 linkage、carrier 或
   degradation product；
5. **节点门回答“谁控制它”；** 基因与药理至少一种，最好正反向和催化失活对照；
6. **功能门回答“它是否改变表型”；** organoid/cell/animal，并有 rescue 或正交反证；
7. **人群门回答“是否可泛化”；** 独立 subtype-resolved cohort，不能用重叠 TCGA 或同队列另一
   LC panel 冒充独立验证。

对 MTBLS13729 的裁决因此非常具体：当前已跨过丰度门，在身份门具有强同队列 Level-1 bridge，
并用外部结构层与 TCGA 分支建立了可检验的去向假说；但没有跨过同法标准、同一样本 glycan
destination、独立丰度复制、来源/通量、节点扰动和功能 rescue。继续增加公共 RNA cohort 不能
替代这些门，边际价值已经低于一次同法标准和 linkage-aware O-glycan 测量。

新增来源：

- SORD–sugar–CRC metastasis: https://www.nature.com/articles/s42255-025-01368-w
- CMS3–CPS1 metabolic vulnerability: https://pmc.ncbi.nlm.nih.gov/articles/PMC11696116/
- Diet-enhanced polyamine depletion: https://www.nature.com/articles/s41586-025-09564-0
- Global human liver 13C flux: https://www.nature.com/articles/s42255-024-01119-3
- Patient–PDX metabolic fidelity: https://www.nature.com/articles/s42255-025-01338-2
