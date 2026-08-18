# 代谢重编程高置信注释 —— 对标分析与课题定位

> 状态：方向已定案（2026-08-17 用户确认）；本文档 = 对标文章盘点 + 遗留问题 + 我们的解法 + 癌症代谢重编程短期评估。
> 纪律：每步过严格因果门；不承诺"必然提高"；命令由用户自己在 conda 里跑（CPU-only）。
> 关联：[[dreams-weighted-rule-noise-plan]]（微调方向仍在，本文档是"交付/包装"侧的并列线）。

## 0. 用户定案（2026-08-17）

1. **锚定应用**：代谢重编程状态分析 → 代谢通路剖析 → 顺带发现新代谢物。
2. **数据库**：全都要看（须高质量）；**HMDB + GNPS 定位为"生化代谢网络评估模块"**。
3. **置信度路线**：**诱饵 FDR + 校准后验**（确认，弃 idpp / SIRIUS 集成为主路线）。
4. **优先级**：倾向短期内可出**决定性成果**的方向；先评估**癌症代谢重编程**。

## 1. 置信度路线（确认）

排序分用现有 `DreaMS_similarity`，叠加：

1. **诱饵库**：用已有 4.8M 负边（不同分子对）+ passatutto 式碎片树诱饵；
2. **FDR/q-value**：target-decoy 竞争（passatutto / mokapot 路线）；
3. **后验校准**：Platt / isotonic（`scikit-learn`），在留出标注集上校准成 P(correct | score)；
4. **Schymanski 分级**：2a（谱库命中≥阈值）/ 2b（诊断证据 = 3,486 规则 + 分子式）/ 3 / 4 / 5；
5. **类级先验（可选）**：CANOPUS 式类预测，做通路/生化合理性先验。

库清单：`matchms`（余弦）、`mokapot`（Percolator 式重打分）、`passatutto`（代谢组诱饵）、
`idpp`（仅作 reference-free 概率对照）、`scikit-learn`（校准）、`RDKit`；外部对照 `SIRIUS/CSI:FingerID`。

## 2. 对标文章（新 + 遗留问题 + 我们的解法）

### A. 注释 / 生成工具（工具层对标）

| 对标 | 遗留问题 | 我们怎么解 |
|---|---|---|
| **MetGenX**（Nat Commun 2026，structure-informed 去新生成注释，对标 CFM-ID/SIRIUS/MSNovelist） | de novo 生成精度受限；database-free 生成难；生成结构无置信度校准 | DreaMS 检索 + 3,486 规则作正交证据，约束 + 校验生成结构，并给 FDR 后验 |
| **MS-Net**（Anal Chem 2026，多相似度网络注释，SIRIUS-CSI + MSNovelist） | 依赖多工具堆叠；注释率仍低 | 单一自监督 encoder 出检索分，替代"多工具拼凑" |
| **LM 引导代谢物发现**（Nature 2025） | 候选多但优先级与验证是瓶颈 | 置信度给候选排序 + 后验，解决"该先验证谁" |
| **MSNovelist**（Nat Methods 2022；2025 综述指出其对库外全新结构失效） | 对库外全新结构失效 | 嵌入检索 + 规则证据兜底库外场景 |

### B. 代谢重编程 / 通路剖析（生物学层对标）

| 对标 | 遗留问题 | 我们怎么解 |
|---|---|---|
| **¹³C-SpaceM**（Nat Metab 2024，空间单细胞同位素示踪脂肪酸从头合成） | 只读靶向脂肪酸，需 ¹³C 示踪，覆盖不了全代谢组 | 无示踪的 reference-free 全代谢组重编程剖析 |
| **Digital twins / ¹³C-scMFA**（Cell Metab 2025，脑肿瘤通量估计） | 通量要 ¹³C + scRNA 联合；非靶特征注释仍是瓶颈 | 把"非靶注释"这环补成高置信，作其下游输入 |
| **CRC 位置/组织特异代谢重编程**（MTBLS13729，2025 空间 + 非靶） | 注释率低，多数差异特征无身份 | 同类队列上把注释率/置信度做上去 |

### C. 通路 / 网络 / 问题定义（框架层对标）

| 对标 | 遗留问题 | 我们怎么解 |
|---|---|---|
| **mummichog** 通路富集（2013；2025 对比显示低注释 → 富集 FP/FN、方法间不一致） | 低注释导致 FP/FN | 高置信注释作输入 + **HMDB/GNPS 生化代谢网络评估模块** |
| **CCMN 化学分类驱动网络分析**（大连化物所 2024，用未鉴定特征、优于 mummichog/MetaboAnalyst） | 靠化学分类，缺结构级置信 | 化学规则(3,486) + 后验 = 分类 + 结构双置信 |
| **注释率仅 10–15%**（SinS 2025 / Warwick Dunn；JACS Au 2025 暗物质视角） | 85–90% 是"暗物质" | DreaMS 本身从无标注谱学——这是它的主场 |
| **癌症代谢组学诊断综述**（Trends Anal Chem 2025） | 仅 9% 研究做到量化、Level 1 罕见 | 诱饵 FDR + 校准后验 → 可量化的 Level 1/2 置信 |
| **idpp 识别概率**（Metz 2025）+ 测量精度扩展（Chang 2025） | reference-free 概率，但需建 RT/CCS 性质库 | 诱饵 FDR + Platt 后验走等价路线，无需性质库 |
| **passatutto**（Scheubert 2017，诱饵 FDR） | 无跨数据集通用 FDR 阈值 | 数据集自适应诱饵 FDR |

## 3. 引用清单（下载状态标注）

已下载到 `docs/papers/`：

| # | 文献 | 期刊/年 | 下载 |
|---|---|---|---|
| 1 | MetGenX — Structure-informed deep generation enables de novo metabolite annotation | Nat Commun 2026, 10.1038/s41467-026-72149-6 | ✅ metgenx_2026 |
| 2 | Scheubert et al. — Significance estimation ... spectral matching (passatutto) | Nat Commun 2017, 10.1038/s41467-017-01318-5 | ✅ passatutto_2017 |
| 3 | Buglakova et al. — Spatial single-cell isotope tracing (¹³C-SpaceM) | Nat Metab 2024, PMC11422168 | ✅ spacem_2024 |
| 4 | Hoffmann et al. — High-confidence structural annotation absent from libraries (CSI:FingerID/SIRIUS) | Nat Biotechnol 2022, PMC8926923 | ✅ hoffmann_2021 |
| 5 | Li et al. — mummichog | PLoS Comput Biol 2013, 10.1371/journal.pcbi.1003123 | ✅ mummichog_2013 |

需机构权限（付费墙，已记 DOI）：

| # | 文献 | 期刊/年 | DOI |
|---|---|---|---|
| 6 | MS-Net — Multi-Similarity-Based Network Annotation | Anal Chem 2026 | 10.1021/acs.analchem.6c01026 |
| 7 | El Abiead et al. — Language model-guided anticipation/discovery of mammalian metabolites | Nature 2025 | 10.1038/s41586-025-09969-x |
| 8 | Stravs et al. — MSNovelist | Nat Methods 2022 | 10.1038/s41592-022-01486-3 |
| 9 | Meghdadi et al. — Digital twins for in vivo metabolic flux (brain cancer) | Cell Metab 2025 | (Cell Metab 2025) |
| 10 | Metz et al. — Introducing 'identification probability' (idpp) | Anal Chem 2025 | 10.1021/acs.analchem.4c04060 |
| 11 | Chang et al. — Measurement precision & identification probability | Anal Chem 2025 | 10.1021/acs.analchem.5c01067 |
| 12 | Dührkop et al. — SIRIUS 4 | Nat Methods 2019 | 10.1038/s41592-019-0344-8 |
| 13 | Dührkop et al. — CANOPUS | Nat Biotechnol 2021 | 10.1038/s41587-020-0740-8 |
| 14 | Schymanski et al. — Communicating confidence (5 级) | ES&T 2014 | 10.1021/es5002105 |
| 15 | Cao et al. — Unintentional fragments & the dark metabolome | JACS Au 2025 | 10.1021/jacsau.5c01063 |
| 16 | Functional annotation — outstanding challenges (综述) | Anal Bioanal Chem 2025 | (review) |
| 17 | Lights and shadows of MS metabolomics in cancer diagnosis | Trends Anal Chem 2025 | S0165993625003978 |
| 18 | Fondrie & Noble — mokapot (软件) | J Proteome Res 2021 | 10.1021/acs.jproteome.1c00410 |

## 4. 课题定位（一句话）

> 在一个 **癌症疾病 vs 对照** 的代谢重编程队列上，用 DreaMS 自监督嵌入 + **诱饵 FDR + 校准后验**
> 做 **reference-free 高置信注释** → **通路剖析**（HMDB + GNPS 作生化代谢网络评估模块）→
> 在**"暗物质"里发现新代谢物**；对标 MetGenX / MS-Net / SIRIUS + mummichog，证明**注释覆盖率与置信度双双提升**。

## 5. 癌症代谢重编程专题 —— 短期决定性成果评估

### 5.1 为什么先做癌症

1. **文献最厚、对标最多**：¹³C-SpaceM（2024）、digital twins（2025）、CRC/GBM/前列腺队列全是近两年，可直接对标；
2. **公开数据最多**：MassIVE/GNPS、MetaboLights 有大量"肿瘤 vs 癌旁 / 处理 vs 对照"队列；
3. **代谢重编程是癌症 hallmark**，机制故事清晰（Warburg、TCA、谷氨酰胺、一碳代谢、脂质从头合成）；
4. **"决定性成果"最好定义**：差异代谢物能直接落到已知重编程通路，与同位素示踪结论交叉验证。

### 5.2 "决定性成果"的短期可量化定义（门槛）

在一个公开癌症队列上，达成以下任一条即为"决定性"：

1. **注释覆盖率**：把差异特征的高置信注释率从 10–15% 基线显著抬升（记录绝对数字，不预设"必然提高"）；
2. **置信度**：给每个注释配 FDR/q-value + Schymanski 等级，做到 Level 2 可量化；
3. **通路**：差异代谢物显著富集到癌症重编程通路（TCA / 谷氨酰胺 / 一碳 / 脂肪酸），与 ¹³C-SpaceM / digital twins 结论交叉一致；
4. **新代谢物**：在暗物质里报出 N 个候选新代谢物，用 3,486 规则 + 后验置信度支持。

### 5.3 可行性 / 风险

| 模块 | 现状 | 风险 |
|---|---|---|
| DreaMS 嵌入 + 检索 | 已预训练 + Atlas 已集成 GeMS-C1/MassIVE | 低 |
| 置信度（诱饵 FDR + 后验） | 需新建，路线清晰、库现成（mokapot/passatutto/sklearn） | 中 |
| 队列挑选 | 需在 MassIVE/GNPS 挑"疾病 vs 对照 + 足够 MS2"的公开癌症数据 | 中 |
| 注释率"决定性提升" | 需过严格因果门验证，不预设结论 | 中 |

### 5.4 分步里程碑（每步过因果门）

- **M1 数据门**：挑 2–3 个候选癌症队列，核 MS2 数量 / 对照设计 / 可复现性，选定一个；
- **M2 检索基线**：DreaMS 嵌入 + 检索，记覆盖率/命中率基线（对照 10–15% dark matter）；
- **M3 置信度模块**：诱饵库 + target-decoy FDR + Platt 后验 + Schymanski 分级，在留出集验证 FDR 校准；
- **M4 通路剖析**：差异代谢物 + 富集（HMDB/GNPS 网络评估模块），交叉验证重编程通路；
- **M5 新代谢物 + 对标**：暗物质候选 + 规则/后验支持，对标 MetGenX/SIRIUS/mummichog；
- **M6 工具证明 + 包装**：注释覆盖/置信度 vs 基线，封装 Gradio/CLI。

## 5.5 真实对标数字（已从 PDF 抽取，2026-08-17）

| 方法 | 关键数字 | 我们的机会点 |
|---|---|---|
| **MetGenX**（Nat Commun 2026） | DB 受限 top-1 **55.9%**（NIST 1388）/ **68.5%**（真实样本 1681）；**DB-free top-1 仅 21.7%**（top-3 35.0%）；用 **Word2Vec**（512 维）做谱嵌入 + 模板检索 + LightGBM 重排；仅找到 2 个新代谢物 | DreaMS 是 Transformer（1024 维），检索表征远强于 Word2Vec；DB-free 21.7% 是最大软肋，正是我们"检索 + 规则 + FDR"能打的地方 |
| **COSMIC/Hoffmann**（Nat Biotechnol 2022） | 置信度 = **KDE P 值 + 强制方向 SVM**（非诱饵）；315 个 HMDB 外结构、1,715 高置信、12 胆汁酸(9 确认) | 我们用 **target-decoy FDR**（passatutto 路线）替代 KDE，可在无大库打分分布时更稳 |
| **passatutto**（Nat Commun 2017） | empirical Bayes + target-decoy；**无跨数据集通用阈值**；+139% 注释（区间 -92% 到 +5705%） | 数据集自适应诱饵 FDR = 我们确认的路线 |
| **¹³C-SpaceM**（Nat Metab 2024） | 空间单细胞同位素示踪，但**只读靶向脂肪酸** | 无示踪 reference-free 全代谢组重编程 |
| **mummichog**（PLoS Comput Biol 2013） | m/z→通路，低注释致 FP/FN | 高置信注释作通路输入 |

## 5.6 候选队列（GNPS 数据门，133 个代谢组学+癌症中筛出）

| MassIVE ID | 内容 | 规模 | 类型 |
|---|---|---|---|
| MSV000100574 / MSV000100560 | **胰腺癌**代谢组+脂质组 | 1550 / 1008 文件，Q Exactive | 人类肿瘤队列 |
| MSV000085877 / 085884 / 085889 | **结直肠癌**性别差异（MTBLS1130/1129/1124） | 各 ~550 文件，Xevo G2-S | 人类 CRC |
| MSV000093325 | 早期**乳腺癌**代谢检测 | 667 文件，Q-Exactive | 人类 |
| MSV000092468 | **结直肠癌**代谢分型 | 1230 文件，Q Exactive | 小鼠 CRC |
| MSV000091475 | 肿瘤 mtDNA 突变驱动有氧糖酵解 | 2147 文件 | 小鼠肿瘤机制 |
| MSV000087155 | **180 种癌细胞系**代谢组 | 2797 文件，6550 QTOF | 泛癌细胞系（无肿瘤组织对照） |
| MSV000089514 | 胰腺癌细胞 ¹³C₆-葡萄糖示踪 | 3218 文件 | 细胞系示踪 |

> 注：需逐一下载确认 MS2 密度 / 肿瘤-vs-对照设计 / 是否含谱库匹配所需元数据，再做 M1 最终选定。

### 5.6.1 M1 核验快照（GNPS catalog 抽取，2026-08-17）

| MassIVE ID | 物种 | 仪器 | 文件数 | 体量 | **MS2 谱数(spectra)** | 设计 |
|---|---|---|---|---|---|---|
| MSV000100574 | 人 | Q Exactive | 1550 | 182 GB | **2,074,840** | 胰腺癌 vs 对照（raw LC-MS/MS，已见 .mzML） |
| MSV000100560 | 人 | Q Exactive | 1008 | 165 GB | **2,068,008** | 同上（配对数据集） |
| MSV000085877 | 人 | Xevo G2-S QTof | 560 | 41 GB | 0（未计数） | 结直肠癌性别差异 MTBLS1130 |
| MSV000085884 | 人 | Xevo G2-S QTof | 558 | 44 GB | 0 | 同上 MTBLS1129 |
| MSV000085889 | 人 | Xevo G2-S QTof | 550 | 42 GB | 0 | 同上 MTBLS1124 |
| MSV000092468 | 鼠+人 | Q Exactive+TSQ Altis+Xevo G2 等 | 1230 | 690 GB | 0 | CRC 代谢分型（GEMM+人，含靶向 TSQ） |

**关键读数：**
- **胰腺癌**（100574/100560）：`spectra` 字段 **207 万张 MS2**，是三者中 MS2 密度碾压级最高；且 result 页确认有 **.mzML**（非纯 .raw）。`complete:false` 表示仍在补传，但 mzML 已可及。→ **MS2 密度最优 + 肿瘤-vs-对照设计**。
- **结直肠癌性别差异**（085877/884/889）：已发表（MetaboLights MTBLS1130/1129/1124，Yuping Cai），人 CRC 组织 + 性别亚型，设计干净；但 `spectra=0`（GNPS 未计数），MS2 密度需自行下载转换核验。→ **设计最干净 / 已发表**。
- **结直肠癌 GEMM**（092468）：机制最强（adenosylhomocysteinase 靶点、甲硫氨酸循环），但 鼠+人+多仪器+含靶向(TSQ Altis 三级杆无 MS2)，管线最杂。→ **机制故事最强但最重**。

**M1 结论**：首选 **胰腺癌 100574** 做"高 MS2 密度 + 肿瘤-vs-对照"的工具证明；并行以 **CRC 性别差异** 作"已发表队列"的可对标复现线。GEMM 留作 M4 机制拓展。

### 5.6.2 M1 关键反转：CRC 队列是 MS1-only，DreaMS 用不了（2026-08-17）

**直接核验**：下载 MTBLS1130 真实 mzML（`menLCCstage1_26.mzML`，79.9MB），grep `ms level` 标签：

- **982 张谱中 979 张 = ms level 1（MS1），0 张 MS2。**

**根因**：MTBLS1130/1129/1124 是经典**非靶 MS1 profiling** 工作流——采谱协议只说 "MS data acquisition"（无 DDA/MSE/碎裂），鉴定靠 MetDNA + 靶向 MS/MS 匹配 METLIN/HMDB，但那批靶向 MS2 谱**未存入**这些 profiling mzML。GNPS catalog 里 `spectra=0` 与此一致。

**含义（硬约束）**：DreaMS 是 **MS2 碎片谱编码器**，输入必须是二级谱。因此：

| 候选 | MS2 证据 | DreaMS 可用性 |
|---|---|---|
| **胰腺癌 100574/100560** | "raw LC-MS/MS"（DDA）+ GNPS `spectra=2,074,840 / 2,068,008`（MS2 谱计数） | ✅ 可用 |
| **CRC MTBLS1130/1129/1124** | mzML 实测 **MS1-only**，MS2 谱未沉积 | ❌ 不可用（除非另找其靶向 MS2 数据） |
| CRC GEMM 092468 | 多仪器含靶向 TSQ（三级杆无 MS2）；待核 | ⚠️ 待核 |

**修正后的 M1 选定**：**胰腺癌 100574** 为唯一 DreaMS-ready 候选（MS2 密度碾压级 + mzML 可及 + 肿瘤-vs-对照）。CRC 队列虽设计/发表最优，但 MS1-only 是硬阻断，**不作为 DreaMS 注释主线**（其 MAF 注释表可作通路层面的旁证对照）。

**剩余待办（进入 M2 前的最后一块）**：胰腺癌 100574 的**样本级分组**（肿瘤 N vs 对照 N、哪些 `PF_n.mzML` 是肿瘤/对照）——MassIVE FTP 被防火墙挡、文件清单走 JS，需在 M2 下载数据时一并读取其 metadata 文件核验。

## 6. 待确认 + 下一步

1. **队列方向**：先看哪类癌症队列 —— 结直肠（数据最丰富、位置特异重编程是现成对标）/
   胶质瘤（13C-SpaceM、digital twins 直接对标） / 前列腺 / 泛癌？或先各挑候选并列给我选。
2. **M1 是否现在就启动**：去 MassIVE/GNPS 挖候选队列清单（只读、不训练）。

---
*本文件由 Claude 整理，2026-08-17；引用 PDF 已下载至 `docs/papers/`。*
