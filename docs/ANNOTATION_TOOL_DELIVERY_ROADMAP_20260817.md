# 面向非靶向代谢组学的高精度 LC-MS/MS 谱图解析工具 —— 交付路线与四条先验增强方向的严格评估

> 状态：2026-08-17 定稿；四方向文献已联网核实回填（共 60 篇，见文末汇总）。
> 立场：本文件**只做严格批判与路线建议**，不承诺"必然提高"；文献全部来自联网核实（不靠记忆拼凑）。
> 读者：项目方交付前，供课题组内部对齐口径用。

---

## 0. 交付目标与当前瓶颈

**交付目标**：给项目方一个面向非靶向代谢组学的、高精度的 LC-MS/MS 谱图解析（注释）工具，且能落到真实生物学任务上（疾病代谢变化 / 药物引起的代谢重编程等），有可核查的精度与注释率。

**我们已有的家底**（都已在仓库内落盘）：

| 资产 | 位置 | 成熟度 |
|---|---|---|
| 3,486 条化学碎裂规则（335 核心 + 3,151 MassBank） | `data_library/`、`dreams/models/chem_aware/` | 高（但语义/来源/适用条件未结构化） |
| MCES 结构距离标签与分层任务（T0–T3） | `tasks/README.md` | 高 |
| DreaMS 错误图谱 + 峰级因果定位（FP/FN） | `data/validation/dreams_structure_residual_atlas*` | 最高（有干预证据） |
| 反事实峰微调 + 消融 | `data/e1/counterfactual_formal`、`docs/COUNTERFACTUAL_PEAK_FINETUNE_STAGE_REPORT.md` | 中高（困难样本 +1pp，未过全量门） |
| 隐式化学因子 / 双重映射（embedding→概念→峰） | `docs/DOUBLE_MAPPING_STATUS_20260816.md` | 中低 |
| DreaMS 复现与分析笔记 | `dreams_analysis/` | 高 |

**瓶颈（诚实陈述）**：纯化学先验（规则重叠、规则注入、反事实峰）的增量已经见顶——规则面板增益跨 0，反事实微调困难样本约 +1pp 且全量未验证。要继续提高注释率，需要**正交于"碎裂化学"的新先验**，或**换范式（检索→生成）**。这正是下面四方向的动机。

**本文件的一句话立场**：四条方向都值得做，但它们在"能不能提高精度、提高多少、风险多大、多久能交付"上差异极大。方向四（打包交付）是无论如何都要做的；方向二（RT）最便宜、最稳；方向三（生成）最有亮点但风险最高；方向一（代谢网络）增量真实但必须避开"谱图相似→结构相似"的循环。

---

## 1. 方向一：代谢网络 + 生物代谢物先验

### 1.1 我们已有 vs 缺什么

- **已有**：化学碎裂规则（分子内"怎么碎"）、MCES 结构距离（分子间"差多少"）、Murcko 骨架。
- **缺**：分子间"怎么互相转化"的**反应可达性**先验——即"这个候选是不是能由样本里共同检出的其它代谢物经已知酶促反应生成"。

关键区分：代谢网络先验有**两种来源**，效果完全不同——

1. **谱图相似网络**（GNPS cosine networking）：基于谱图相似度连边。→ 会**复现我们错误图谱里的 FP**（共享碎片→误聚），本质是循环。
2. **反应/质量差网络**（KEGG/MetaCyc 反应边，biotransformation mass difference）：基于已知酶促转化的质量差连边。→ 这才是正交于谱图的新信息，但需要网络覆盖与样本内的共同检出。

### 1.2 严格评估

- **增量来源**：主要是**注释率**（把已注释代谢物沿反应边传播给未注释邻居）与**置信度**（生化合理性重排），**不是谱图匹配精度本身**。期望"代谢网络提升 Top-1 谱图精度"要非常克制。
- **与现有工作的衔接点（检索后强化）**：我们已有的 3,486 条规则库里，"中性丢失"天然就是 mass difference 的一种弱形式；把规则库从"碎裂质量差"扩到"生物转化质量差"是**同一张数据结构的自然扩展**。更关键的是——**网络方法（MetDNA/BAM/NetID）所假设的"反应对结构相似"信号，我们手里已经有等价物：MCES**。所以代谢网络先验对我们不是"从零引入"，而是"把已有 MCES + 规则库挂到反应边上再传播"。这与用户"扩充规则库"的直觉一致，但扩充方向应是**有生化来源的反应差**，不是更多 MassBank 经验峰。
- **风险**：
  1. 覆盖偏差：KEGG/MetaCyc/Reactome 最丰富的是初级/模式生物代谢，植物/次生代谢物/微生物天然产物覆盖不足——而我们很多库（natural_products、plant、enveda）恰恰是这些。
  2. 循环论证：若网络由谱图相似度构建（GNPS cosine），等于把我们错误图谱里的 FP（共享碎片误聚）硬编码进先验——**任何谱图边必须被我们的因果 FP/FN 图谱或 MCES 结构一致性门控**。
  3. **mass-difference 边混淆**：很多生物转化是等重的（羟基化 vs 某些位移、脱羧 vs 氨化），且加合物/碎片/同位素差会伪装成反应（NetID 专门区分这些）——naive delta-mass 库 FP 很高，边置信必须用精确质量+RT+分子式+MCES 联合校准。
  4. **种子依赖 + 误差传播**：MetDNA/BAM/NetID 都依赖正确 anchor 种子，一个错种子递归放大错误 → 置信必须随跳数衰减并独立证据门控。
  5. 假设强度：默认"样本内代谢物遵循已知生化"，对药物诱导/外源物场景不成立。
- **判定**：**P1 优先**（低成本高注释率回报），但先做"反应质量差先验"的轻量正交特征，不上完整知识图谱。**验收指标是注释率+置信度，不是 Top-1 谱图精度。**

### 1.3 文献（15 篇，联网核实）

1. **Sharing and community curation of MS data with GNPS Molecular Networking** — Wang et al. *Nature Biotechnology*, 2016. https://www.nature.com/articles/nbt.3597 | GNPS 开放平台：谱库匹配 + 谱图相似分子网络。| 谱图相似"社交网络"与 GNPS/MassIVE 数据生态的来源（注意是 Nat Biotechnol 非 Nat Methods）。
2. **Reproducible molecular networking using GNPS** — Aron et al. *Nature Protocols*, 2020. https://www.nature.com/articles/s41596-020-0317-5 | GNPS-MassIVE 可复现分子网络构建协议 + 可重分析"活数据集"。| 网络构建与公共数据来源，可挖网络先验并用于基准。
3. **Feature-based molecular networking (FBMN)** — Nothias et al. *Nature Methods*, 2020. https://www.nature.com/articles/s41592-020-0933-6 | 把分子网络锚定到色谱 feature 而非原始谱，支持异构体/定量/离子迁移。| 与我们"峰级 + LC-MS/MS 原生"的 atlas 天然对接。
4. **SIRIUS 4** — Dührkop et al. *Nature Methods*, 2019. https://www.nature.com/articles/s41592-019-0344-8 | 同位素模式 + 碎裂树 + CSI:FingerID，大规模 rank-1 正确结构 >70%。| "纯化学"基线，必须超过或互补；其分子式/指纹输出是生物先验层的输入。
5. **CANOPUS** — Dührkop et al. *Nature Biotechnology*, 2021. https://www.nature.com/articles/s41587-020-0740-8 | 从 MS2 预测 2,497 个化合物类（ClassyFire/NPClassifier），无参考谱也能预测。| 无库生物/类别先验，可与我们的隐式因子发现融合来约束候选空间。
6. **NetID** — Chen et al. *Nature Methods*, 2021. https://www.nature.com/articles/s41592-021-01303-3 | 全局网络优化，用 mass/RT/MS2 赋峰并以质量差连边（加合物/碎片/同位素/生物转化），可注释无 MS2 峰。| 最接近"生物质量差先验"的现有范本；其"加合物/碎片 vs 生物转化"边类型区分正是我们错误图谱诊断的问题。
7. **MetDNA** — Shen et al. *Nature Communications*, 2019. https://www.nature.com/articles/s41467-019-09550-x | 从种子沿 KEGG 反应网络递归传播，每实验累计注释 ~2,000 代谢物。| 反应网络传播（非谱图相似）大幅提注释率的铁证——正是我们的覆盖瓶颈。
8. **MetaCyc 2019 update** — Caspi et al. *Nucleic Acids Research*, 2020. https://doi.org/10.1093/nar/gkz862 | 2,749 条证据通路、6 万+ 文献。| 高质量通用反应质量差/通路来源，构建 biotransformation 边库。
9. **KEGG for taxonomy-based pathway/genome analysis** — Kanehisa et al. *Nucleic Acids Research*, 2023. https://doi.org/10.1093/nar/gkac963 | KEGG pathway/MODULE/NETWORK 反应网络。| 标准代谢知识图谱（Reactome 对人有类比作用）。
10. **Spec2Vec** — Huber et al. *PLOS Computational Biology*, 2021. https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008724 | word2vec 式谱 embedding 打结构相似度，比 cosine 更可扩展。| 与 DreaMS 直接可比的 embedding 相似度；验证"谱 embedding 恢复结构"——网络边应建在其上。
11. **MS2DeepScore** — Huber et al. *Journal of Cheminformatics*, 2021. https://jcheminf.biomedcentral.com/articles/10.1186/s13321-021-00558-4 | Siamese 网络从 MS2 预测结构 Tanimoto，MC-dropout 不确定性。| 其"从谱预测结构相似度"≈我们的 MCES/DreaMS 目标；不确定性校准是置信门控传播的范本。
12. **MolNetEnhancer** — Ernst et al. *Metabolites*, 2019. https://www.mdpi.com/2218-1989/9/7/144 | 分子网络 + MS2LDA 子结构 + in silico 注释 + ClassyFire 分类融合。| "谱 + 子结构 + 类别"多先验融合模板，我们的规则+MCES+错误图谱是它的泛化。
13. **BAM: biotransformation rules + global molecular networking** — Martin, Bittremieux, Hassoun. *Analytical Chemistry*, 2025. https://pubs.acs.org/doi/10.1021/acs.analchem.4c01565 | 生物转化规则作用于高相似 anchor 谱，给未知邻居排序（数亿谱上 24.2% 正确）。| 2025 最新证据：生物转化质量差 + 全球网络直接提升我们这类问题的覆盖。
14. **3D-MPEA** — Fan, Jia. *Analytical Chemistry*, 2024. https://pubs.acs.org/doi/10.1021/acs.analchem.4c00256 | 图注意力 + LSTM 于多层分子网络（同位素间隔 + MS2 相似）传播注释。| 2024"GNN-on-network"混合范本，可用我们的 DreaMS embedding 替换其图特征。
15. **ESP: ensemble spectral prediction for metabolite annotation** — Li et al. *Bioinformatics*, 2024. https://doi.org/10.1093/bioinformatics/btae490 | MLP(ECFP) + GNN 谱预测 + rank learning，平均候选名次提升 23.7%/37.2%。| 图嵌入 + rank learning 融合提升排序，与我们的 MCES 标签 + DreaMS encoder 兼容。

### 1.4 建议

1. 建立 `biotransformation_mass_difference` 库（KEGG RClass / MetaCyc 反应式 → 前体−产物质量差），与现有规则库共用 `match_vectors` 结构。
2. 只做"共检出 + 反应可达"的候选重排/传播，输出"反应证据"作为模块 2 的一类新证据，不做进主干损失。
3. 在植物/天然产物库上先测覆盖，覆盖不足则此方向降级。

---

## 2. 方向二：加入 RT（保留时间）先验

### 2.1 我们已有 vs 缺什么

- **已有**：DreaMS 预训练**已用 retention order 目标**（模型已接触 RT 顺序信号）；部分库（lipid 等）带 RT 元数据。
- **缺**：把**绝对/预测 RT** 作为显式正交先验，用于候选过滤与重排。

### 2.2 严格评估

- **增量来源**：RT 是**最强正交先验之一**，尤其对**同分异构体**（同 m/z 不同结构 → RT 不同）。这对我们错误图谱里最难的"同分子式近结构候选"正是对症。
- **与现有工作的衔接点**：DreaMS 的 retention order 目标是"两张谱谁先出"，我们可把它升级为"RT 作为额外 token/元特征"，但更稳妥是**后验重排**（不改主干）。
- **风险**：
  1. 跨实验室/色谱柱/梯度迁移困难，RT 难复现 → 需要预测 + 校准标准。
  2. RT 库规模远小于谱库（SMRT ~80k、RepoRT ~143k，比 MS/MS 库低几个数量级），且 SOTA 预测 MAE ~25–35 s 相对峰宽偏大，只能过滤、不能单独确认身份。
  3. 需要 query 自带 RT（很多公开谱库没有），且 RT 必须是**可校准/已知系统**上的测量，否则先验不可用甚至有害 → RT 头必须**按"RT 是否可用 + 系统是否匹配"门控**。
- **关键设计杠杆（检索确认）**：**rank-based（保留顺序）比 absolute RT 更抗跨实验室迁移**，且 DreaMS 预训练本就用 retention-order 目标——加"顺序/rank 先验"与现有监督对齐、风险最低；absolute RT 过滤则需 method-aware 模型或校准标准（PredRet/IRTS）才可靠。
- **判定**：**P0 优先**（最便宜、最稳、直接对症同分异构体）。但只做"后验重排/过滤"特征，不进主干；先确认我们数据里可用 RT 的占比。

### 2.3 文献（11 篇，联网核实）

1. **Retip: Retention Time Prediction for Compound Annotation in Untargeted Metabolomics** — Bonini, Kind, Tsugawa, Barupal, Fiehn. *Analytical Chemistry*, 2020. https://pubs.acs.org/doi/10.1021/acs.analchem.9b05765 | 五类 ML 回归器（RF/BRNN/XGB/LightGBM/Keras）预测 RT，MAE ~0.57–0.78 min，在 MS-FINDER 异构体搜索中砍掉 68% 候选。| "预测 RT→过滤候选"的可嵌入模板，正是我们要叠加的用法。
2. **SMRT (METLIN small molecule dataset for ML-based RT prediction)** — Domingo-Almenara et al. *Nature Communications*, 2019. https://www.nature.com/articles/s41467-019-13680-7 | ~80,038 条 RPLC 实测 RT + 结构/描述符，DL 回归器 top-3 命中 ~70%。| 事实上的公开 RT 训练集，量化了纯 RT 先验能加多少排序增益。
3. **PredRet: Prediction of Retention Time by Direct Mapping between Chromatographic Systems** — Stanstrup, Neumann, Vrhovšek. *Analytical Chemistry*, 2015. https://pubs.acs.org/doi/10.1021/acs.analchem.5b02287 | 跨系统 RT 直接映射（GAM），系统间中位误差 0.01–0.28 min。| 跨实验室/色谱柱 RT 迁移的基础机制——不用每实验室重训。
4. **DeepRT** — Ma, Ren, Yang, Ren, Yang, Liu. *Analytical Chemistry*, 2018. https://pubs.acs.org/doi/10.1021/acs.analchem.8b02386 | 胶囊网络做肽 RT（R²≈0.99），带 DeepRT+ 迁移扩展。| 证明紧凑的 RT 头能跨 LC 模式泛化、小样本可微调。
5. **GNN-RT: Prediction of LC Retention Time with GNNs** — Yang, Ji, Lu, Zhang. *Analytical Chemistry*, 2021. https://pubs.acs.org/doi/10.1021/acs.analchem.0c04071 | 分子图 GNN（不用手工描述符）在 SMRT 上相对误差 ~4.9%/3.2%。| 结构-only RT 预测已足够强，与 DreaMS 式分子编码器同源。
6. **RT-Transformer** — Xue, Wang, Ji, Li. *Bioinformatics*, 2024. https://academic.oup.com/bioinformatics/article/40/3/btae084/7613958 | ResGAT + 1D Transformer，SMRT 预训练 + 41 外部集迁移（MAE ~27–33 s），过滤 ~52–60% 候选。| 当前 SOTA，明确把 RT 当"MS/MS 之外的候选过滤"——正是我们要的用法。
7. **Graphormer-RT: method-independent RT via graph transformers** — Stienstra, Nazdrajić, Hopkins. *Analytical Chemistry*, 2025. https://pubs.acs.org/doi/10.1021/acs.analchem.4c05859 | 方法编码器（global graph node）单模型跨 191 RP + 49 HILIC 方法（MAE ~29–42 s），定位为 RT"基础模型"。| 对跨实验室问题的最强现答：编码"方法"而非固定系统——启发我们的 method-conditioned RT 头。
8. **RepoRT: a comprehensive repository for small molecule retention times** — Kretschmer, Harrieder, Hoffmann, Böcker, Witting. *Nature Methods*, 2024. https://www.nature.com/articles/s41592-023-02143-z | 精心整理 ~143k RP + ~4k HILIC RT，含柱/梯度/流动相/pH 元数据。| 最大的方法标注 RT 语料，其元数据粒度是诚实跨实验室评估的前提。
9. **SIRIUS 4** — Dührkop et al. *Nature Methods*, 2019. https://www.nature.com/articles/s41592-019-0344-8 | 集成注释引擎（CSI:FingerID）挑战集 >70% 识别，RT 作为正交轴重排/解异构、不参与核心打分。| "MS/MS 为主、RT 正交过滤"的典范架构——加 RT 时应照抄。
10. **ROASMI: repurposing retention data via retention order** — Sun et al. *Journal of Cheminformatics*, 2025. https://link.springer.com/article/10.1186/s13321-025-00968-8 | D-MPNN + RankNet 预测 pH 依赖的保留**顺序**（非绝对 RT），跨 71 RPLC 集，改善 SIRIUS/MS-DIAL 注释。| 与我们最对齐：DreaMS 已学 retention order，rank-based 更抗跨实验室迁移。
11. **A standardized nontargeted metabolomics method for cross-lab food profiling** — Odenkirk et al. *Food Chemistry*, 2025. https://www.sciencedirect.com/science/article/abs/pii/S0308814625031851 | 内标 RT 标准品（IRTS）实现跨实验室/仪器色谱对齐。| 让绝对 RT 跨实验室可比的校准标准路线——若要用 query RT，这是前提。

### 2.4 建议

1. 统计 `data/` 各库带 RT 的比例（决定此方向是"主力"还是"锦上添花"）。
2. 用 Retip/SMRT 类方法训一个冻结的 RT 预测器，作为候选重排分数，与 DreaMS 相似度加权融合。
3. 只在与 DreaMS 相同的锁定测试集上评估 RT 带来的 Top-1/注释率增量，避免过拟合。

---

## 3. 方向三：DeepMet / 谱图→SMILES 生成（更高层 NLP）+ DreaMS encoder→SMILES decoder

### 3.1 我们已有 vs 缺什么

- **已有**：DreaMS 编码器（1024 维，冻结即可用）、annotated01 ~7.6 万 IK14 带 SMILES（生成训练的种子数据）。
- **缺**：一个把谱图 embedding 解码成 SMILES 的**生成头**；以及"MS 谱作为语言"的更高层表征。

这是**范式转换**：从"检索库内结构"变成"生成库外结构（de novo）"。用户明确指出这部分重要——我同意，它是四个方向里**交付亮点最大**的，也是**风险最高**的。

### 3.2 严格评估

- **先澄清"DeepMet"（检索确认，重要）**：DeepMet 是**真实的具体方法**，但它**不是谱图→SMILES 解码器**，而是一个**生成式化学语言模型**——把 SMILES 当语言、自回归训练在已知哺乳动物代谢物上，学习"潜在生物合成逻辑"，从而**预判**未见过的类代谢物结构并按采样频率排序。论文：Qiang, Wang et al., *Nature* **651**:211–220 (2026), "Language model-guided anticipation and discovery of mammalian metabolites"。它的质谱连接是**下游**的（先精确质量筛，再 CFM-ID 匹配 + 元学习，~70% 注释）。所以 DeepMet 本质是**更强的化学先验，不是谱解码器**。若你记得的是"把谱当语言做注释的检索方法"，那更可能是 **DeepMASS/DeepMASS2**（检索式，非生成）——两者必须分清。
- **方向三其实是两件互补的事**：① DeepMet 式**生成先验**（预判类代谢物结构）；② **谱条件 SMILES 解码器**（把 DreaMS embedding 接到预训练 SMILES decoder 上，Spec2Mol/MetGenX 路线）。你说的"DreaMS encoder 做 SMILES 生成"对应②，"更高层 NLP"对应①。
- **增量来源**：检索路线永远受限于"库里有没这个结构"；生成路线能注释库外分子（dark chemical space），直接拉高注释率上限。
- **与现有工作的衔接点**：我们已证明 DreaMS embedding 可解码化学概念（macro-AUPRC 0.659）与局部结构环境——这为"从 embedding 解码 SMILES"提供了可读性基础；但"可解码概念"≠"能生成正确分子"。
- **最关键的设计结论（检索确认）**：纯自回归 free decoding 的精确命中很低（Mass2SMILES 平均 Tanimoto ~0.40、MS2Mol 近匹配 21%）；**2026 最强（MetGenX 55.9% top-1）走的是"检索模板条件生成"，不是自由解码**。所以我们的正确路线是 **DreaMS-embedding → 检索近邻结构 → 用其指纹/分子式条件 decoder 生成 SMILES**，而不是纯 decoder 自由生成。
- **风险（必须写清）**：
  1. **SMILES 有效性与幻觉**：自由解码会输出"语法对但结构错"或化学上不可行的 SMILES（比"我不知道"更危险）；须 RDKit 有效性过滤 + 校准置信分（学 MS2Mol 的 confidence scorer）。
  2. **评估难**：生成结构对错需真结构算 Tanimoto/MCES，而 de novo 场景恰恰没有真结构——评估要用 exact-match/Hits@k/MRR + 有效性率 + 独立注释判官，不能只看 Tanimoto（0.4–0.7 往往不是正确结构）。
  3. **数据量**：7.6 万配对做对齐够、做生成预训练不够 → 正确姿势是"SMILES decoder 在百万级**未配对**结构（PubChem/COCONUT/HMDB）上预训练，再在数万谱-结构对上对齐"（MetGenX 路线）。
  4. **检索与生成的优化目标分叉**：2026 有理论证明优化指纹预测损失可能**恶化**分子检索——生成头与检索头不能混用一个目标。
- **判定**：**P2 优先**（亮点大、风险高），先做"冻结 DreaMS encoder + 预训练 SMILES decoder 的检索条件生成" smoke，用有效性+化学合理性+MCES 做第一阶段门，不承诺"生成即正确"。

### 3.3 文献（17 篇，联网核实）

1. **DeepMet: Language model-guided anticipation and discovery of mammalian metabolites** — Qiang, Wang, Lu, Xing et al. (Rabinowitz, Skinnider, Wishart). *Nature* **651**:211–220, 2026. https://doi.org/10.1038/s41586-025-09969-x | SMILES-as-language 生成模型预判类代谢物结构并按采样频率排序（AUC 0.98 分离留出代谢物）。| 生成式化学先验的最强范本；单靠先验 top-1 仅 ~29%（质量筛）——正好论证"先验不够，需要谱条件化"。
2. **MassGenie** — Shrivastava, Swainston, Samanta, Roberts, Wright Muelas, Kell. *Biomolecules* **11**(12):1793, 2021. https://doi.org/10.3390/biom11121793 | ~400M Transformer 把 MS/MS→SMILES 当机器翻译，~6M 增强 SMILES + in-silico 碎片训练，CASMI 正模式 53% exact。| 最早"谱→SMILES 翻译"证明，通用 encoder-decoder 可生成有效未见结构。
3. **MSNovelist** — Stravs, Dührkop, Böcker, Zamboni. *Nature Methods* **19**(7):865–870, 2022. https://doi.org/10.1038/s41592-022-01486-3 | SIRIUS/CSI:FingerID 指纹预测 + LSTM SMILES 生成，GNPS 25% / CASMI2016 26% 首名。| 两段式（谱→指纹→SMILES）设计基准，若选分段 decoder 应对照。
4. **Spec2Mol** — Litsa, Chenthamarakshan, Das, Kavraki. *Communications Chemistry* **6**:132, 2023. https://doi.org/10.1038/s42004-023-00932-3 | Speech2Text 式 encoder-decoder，SMILES decoder 预训练后对齐谱 encoder，与碎裂树法相当（含未见结构）。| "给谱 encoder 接 SMILES decoder"最贴切的架构蓝图，可直接迁移到 DreaMS embedding。
5. **MS2Mol** — Butler, Frandsen, Lightheart et al. (Enveda/UCSD). *ChemRxiv*, 2023. https://chemrxiv.org/doi/full/10.26434/chemrxiv-2023-vsmpx-v3 | seq2seq Transformer de novo 生成，EnvedaDark 226 未知天然产物 21% 近匹配，带置信打分器。| 同时展示 dark chemical space 前景与纯自回归解码天花板（~21%）。
6. **MIST** — Goldman, Wohlwend, Stražar, Haroush, Xavier, Coley. *Nature Machine Intelligence* **5**(9):965–979, 2023. https://doi.org/10.1038/s42256-023-00708-3 | 把化学式峰 + 隐式中性丢失特征注入 Transformer + 多任务子结构/指纹预测，>70% 标准物上超过指纹检索。| 最强证据：把化学知识注入谱 encoder（而非只检索）能赢——DreaMS 条件化的设计线索。
7. **Mass2SMILES** — Elser, Huber, Gaquerel. *bioRxiv*, 2023. https://www.biorxiv.org/content/10.1101/2023.07.06.547963v1 | Transformer-encoder + temporal-CNN 从高分辨 MS/MS 预测 SMILES + 60 官能团 + 化学式，CASMI2022 平均 Tanimoto ~0.40。| 诚实量化了 SMILES 有效性与低 exact-match 问题——DreaMS decoder 会继承。
8. **MSBERT** — Zhang, Yang, Xie, Wang, Zhang, Lu. *Analytical Chemistry* **96**(42):16599–16608, 2024. https://doi.org/10.1021/acs.analchem.4c02426 | BERT 掩码峰 + 对比预训练，recall@10 0.908，谱相似检索上超过 Spec2Vec 和 DreaMS。| DreaMS 的直接对照/替代 embedding，其 latent 可喂共享 SMILES decoder。
9. **MetGenX** — Wang, Zhang et al. (Zheng-Jiang 组). *Nature Communications* **17**, 2026. https://doi.org/10.1038/s41467-026-72149-6 | "structure→structure"：检索谱相似模板，编码其指纹+化学式，6 层 transformer 解码 SMILES；NIST 55.9% top-1 / 76.1% top-3，跨电离模式。| 2026 SOTA：**模板/检索条件生成**（非自由解码）才是提准确率的关键——应嫁接到 DreaMS embedding 的策略。
10. **DeepMASS / DeepMASS2** — Ji, Du et al. *bioRxiv*, 2024 (GUI/webserver 2026). https://www.biorxiv.org/content/10.1101/2024.05.30.596727 | 把谱当语言、学谱语义相似 + 化学空间定位来排序库候选，称超过 SIRIUS/CFM-ID/MetFrag/MS-Finder。| 用户可能与之混淆的**检索式**方法；引用以澄清边界并界定"非生成注释"能到什么程度。
11. **MoleculeSTM** — Liu, Nie, Wang, Lu, Qiao, Liu, Tang, Xiao, Anandkumar. *Nature Machine Intelligence*, 2023. https://arxiv.org/abs/2212.10789 | SMILES/图 ↔ 文本对比模型（281K PubChemSTM），零样本结构-文本检索与编辑。| 提供谱↔结构↔文本对齐的 SMILES/文本编码解码机制。
12. **MolT5** — Edwards, Lai, Ros, Honke, Cho, Ji. *EMNLP*, 2022. https://arxiv.org/abs/2204.11817 | T5 在 100M ZINC SMILES + C4 文本预训练，ChEBI-20 微调做分子 caption / text→SMILES。| 现成的高有效性自回归 SMILES decoder（77M–800M），可初始化 DreaMS 条件头。
13. **MolXPT** — Liu, Zhang, Xia, Wu, Xie, Qin, Zhang, Liu. *ACL*, 2023. https://aclanthology.org/2023.acl-short.138/ | 350M GPT 交织 PubMed 文本 + 关联 SMILES，44% 参数达到接近 MolT5 的翻译 + 零样本生成。| 单一 SMILES-文本生成模型可当 decoder，无需额外分子-语言脚手架。
14. **Text2Mol** — Edwards, Zhai, Ji. *EMNLP*, 2021. https://aclanthology.org/2021.emnlp-main.47/ | 跨模态分子-文本检索（Hits@1, MRR）+ ChEBI-20 基准。| 确立 Hits@k/MRR 的检索评估框架，可复用于 DreaMS-embedding ↔ SMILES 候选排序。
15. **Chemformer** — Irwin, Dimitriadis, He, Bjerrum. *Machine Learning: Science and Technology* **3**:015022, 2022. https://doi.org/10.1088/2632-2153/ac3ffb | BART 式 encoder-decoder（span-masking + SMILES 增强）在 ~100M ZINC SMILES 预训练。| 预训练 SMILES seq2seq 的 decoder 是谱条件 SMILES 输出的好起点。
16. **MolFormer** — Ross, Belgodere, Chenthamarakshan, Padhi, Mroueh, Das. *Nature Machine Intelligence* **4**:1256–1264, 2022. https://arxiv.org/abs/2106.09553 | 线性注意力 Transformer，1.1B PubChem/ZINC SMILES MLM，强 MoleculeNet 特征。| 高容量 SMILES encoder，可与 DreaMS 配对做谱↔结构对比对齐。
17. **Small molecule retrieval from tandem MS: what are we optimizing for?** — De Waele, Wydmuch et al. *arXiv* 2602.16507, 2026. https://arxiv.org/abs/2602.16507 | 证明优化指纹预测损失可能**恶化**分子检索，给出后悔界。| 直接指导 DreaMS→SMILES 头的训练/评估：检索名次与结构相似度是不同目标，不能混。

### 3.4 建议

1. 先确认"DeepMet"具体指什么（是方法还是思路），避免张冠李戴。
2. 第一版：冻结 DreaMS → 线性/浅层 decoder → SMILES，输出 `(SMILES, 有效性, 与检索候选的 MCES)`，与检索结果并列。
3. 验收门：SMILES 有效性 ≥ X%、生成与检索 Top-1 的 MCES 中位数 ≤ Y、不产生比检索更差的排序。

---

## 4. 方向四：真实生物学应用 + 基准 + 交付包装（大杂烩多模态）

### 4.1 我们已有 vs 缺什么

- **已有**：全套谱图解析与化学证据能力。
- **缺**：① 严格基准（精度/FDR 的公开可核查数字）；② 1–2 个端到端的真实生物学案例；③ 一个能对外讲清楚"我集成了什么、各模块消融、边界在哪"的交付叙事。

### 4.2 严格评估（这部分最需要"批评"）

用户原话是"拼凑已有方法（DreaMS、DeepMet、CANOPUS、MS2DeepScore、SIRIUS…）做一个大杂烩，就说我们多模态、全面、性能好"。**这里必须踩刹车，并且检索给出了四条必须遵守的诚实性约束**：

- **"就说性能好"不可取**：没有基准数字支撑的"性能好"是过度承诺，会直接反噬交付信誉。正确说法是"在基准 X 上分级精度=…、覆盖率=…、FDR（decoy 校正）=…，各模块消融如下"。
- **分级报告，绝不折叠成一个数字（检索确认）**：CASMI 2022 的四个等级（adduct → 化学式 → 化合物类 → 2D 结构）必须分开报。一个"class 层 99%、2D 结构层 2%"的工具**不能**被描述为"99% 准确"。
- **覆盖率 + FDR 才是头条，不是 Top-1（检索确认）**：5% feature 覆盖率下的高精度对非靶向生物学几乎无用。必须报 ① 注释了 % 的检出 feature、② 用 decoy/盲标准估的 FDR、③ Top-1/Top-k 只是三级指标。**没有 decoy/null 模型就不能断言 FDR**。
- **decoy 评估，否则可能"假赢"（检索确认，最重）**：Gupta 2026 证明"忽略谱图的结构-only 基线"能靠"参考库 vs PubChem 化学空间偏差"部分"解决"MassSpecGym。我们必须在**与参考库同化学空间生成的 decoy（Spectraverse 式）**上评估，而不是朴素 PubChem 候选，否则任何准确率都不可解释。
- **"大杂烩"融合有陷阱（检索确认）**：SIRIUS 的化学式、CANOPUS 的类别、MS2DeepScore 的相似分、MetFrag 的候选名次、生成式 SMILES——语义不同，**不能平均**。需要**校准的 rank-fusion 层 + 每输出置信 + 明确优先级**；而且"多组件一致投票"**不是独立证据**（它们共享训练数据/库偏差，投票会放大同一系统性错误）。
- **逐模块消融 + 单一留出集是硬要求**：每个组件单独报 Top-1/Top-k/覆盖率/FDR，再报集成的边际增益，全在同一批谱、无泄漏；否则"集成打败所有组件"无效。
- **生物学案例的边界**：工具能证明的是"在该公开数据集上注释了 N 个结构、其中 M 个过了置信门槛、通路富集到哪些生物学过程"；**不能**直接宣称"发现了某疾病的代谢机制"——那是下游结论，需要生物学验证。

### 4.3 文献（17 篇，联网核实）

1. **DreaMS (Self-supervised learning of molecular representations from millions of tandem mass spectra)** — Bushuiev, Bushuiev, Samusevich, Brungs, Sivic, Pluskal. *Nature Biotechnology*, 2025. https://doi.org/10.1038/s41587-025-02663-3 | 数千万无标注 GNPS/MassIVE 谱自监督预训练，微调后碎裂/元素任务 SOTA。| 我们的基座模型，也是"大杂烩"所有声称必须对照的基准点。
2. **MassSpecGym: A benchmark for the discovery and identification of molecules** — Bushuiev et al. (Pluskal & Sivic 组). *NeurIPS 2024 (Spotlight)*. https://proceedings.neurips.cc/paper_files/paper/2024/hash/c6c31413d5c53b7d1c343c1498734b0f-Abstract.html | 231,104 谱 / ~29k 分子，de novo 生成 + 检索 + 谱模拟三任务，泛化挑战划分。| 必须报 Top-1/Top-k/有效性的基准；诚实 SOTA：de novo Top-1 ~2–3%、检索 hit@1 ~10–15%。
3. **CASMI 2022 (Critical Assessment of Small Molecule Identification)** — Fiehn Lab. 2022. https://www.fiehnlab.ucdavis.edu/casmi/ | 500 LC-MS/MS 盲测（±ESI、不在公共库），四级打分（adduct/式/类/2D 结构）。| 分级报告标准——交付工具必须按级分别报准确率。
4. **SIRIUS 4** — Dührkop et al. *Nature Methods*, 2019. https://doi.org/10.1038/s41592-019-0344-8 | CSI:FingerID + 碎裂树，挑战集 >70% 结构识别。| 集成的"化学式 + in-silico 碎裂 + 库搜索"腿，提供 de novo 相邻结构候选。
5. **CANOPUS** — Dührkop et al. *Nature Biotechnology*, 2021. https://www.nature.com/articles/s41587-020-0740-8 | 从谱预测 2,497 化合物类（~99.7% CV），无需参考谱。| 集成的"类"输出层——对无结构匹配的化合物，诚实报 class 层能拉高有效覆盖。
6. **MS2DeepScore** — Huber, van der Burg, van der Hooft, Ridder. *Journal of Cheminformatics*, 2021. https://doi.org/10.1186/s13321-021-00558-4 | Siamese 从谱对预测结构 Tanimoto，MC-dropout 不确定性，RMSE ~0.15（滤后 0.10）。| 谱相似/类似物腿；其不确定性是 FDR 校准现成的逐对置信信号。
7. **MS-DIAL** — Tsugawa, Cajka, Kind, Ma, Higgins, Ikeda, Kanazawa, VanderGheynst, Fiehn, Arita. *Nature Methods*, 2015. https://doi.org/10.1038/nmeth.3393 | 开源 DIA 去卷积 + 鉴定/定量流水线。| 把原始 LC-MS/MS 变成 feature 表 + MS2 谱的前端，注释器的输入来源。
8. **MZmine 3** — Schmid, Heuckeroth, Korf et al., Pluskal. *Nature Biotechnology*, 2023. https://doi.org/10.1038/s41587-023-01690-2 | 重写 MZmine 3，数千样本/小时，集成 LC-IMS-MS / 成像。| 处理前端备选；Pluskal 组血统直连我们已在用的 DreaMS/MassSpecGym 生态。
9. **GNPS Molecular Networking** — Wang et al. *Nature Biotechnology*, 2016. https://doi.org/10.1038/nbt.3597 | GNPS 分子网络、社区谱库、"活数据"重分析。| 社区数据/验证基座（注释沉积与复核处），也是 MS2DeepScore 训练数据来源。
10. **MetFrag relaunched** — Ruttkies, Schymanski, Wolf, Hollender, Neumann. *Journal of Cheminformatics*, 2016. https://doi.org/10.1186/s13321-016-0115-9 | in-silico 碎裂 + 库搜索 + RT/参考打分；Top-1 从 ~6–9%（纯谱）升到 71–89%（+RT+参考）。| 说明 RT/正交元数据的超值——集成应"加权证据"而非信任单一分数。
11. **FBMN (Feature-based molecular networking)** — Nothias et al. *Nature Methods*, 2020. https://doi.org/10.1038/s41592-020-0933-6 | 用 LC 解析 feature（同位素/加合物/RT）替代峰级网络，支持定量 + SIRIUS/CANOPUS in-silico 注释。| 把定量 feature + 类注释 + 结构注释合成一条管线的蓝图——我们"大杂烩"必须复现的集成。
12. **MSHub (Auto-deconvolution and molecular networking of GC-MS data)** — Aksenov et al. *Nature Biotechnology*, 2021. https://doi.org/10.1038/s41587-020-0700-3 | NMF 自动去卷积 + GC-MS 分子网络。| GC-MS 臂（方法简报，非综述）；注释最佳实践综述见 #14/#15。
13. **Software Tools and Approaches for Compound Identification of LC-MS/MS Data in Metabolomics** — Blaženović, Kind, Ji, Fiehn. *Metabolites*, 2018. https://doi.org/10.3390/metabo8020031 | 综述谱库、in-silico 碎裂、RT/CCS 建模与 CASMI 结果。| 前-ML 工具全景图，集成必须自我定位。
14. **From Samples to Insights into Metabolism (LC-HRMS)** — Ivanisevic, Want. *Metabolites*, 2019. https://doi.org/10.3390/metabo9120308 | 强调注释是瓶颈；走完整非靶向管线，强调置信/鲁棒 > 覆盖。| 交付工具必须继承的"置信语言（鉴定等级）"——避免过度承诺。
15. **Application of LLMs/Transformer-Based Models for Metabolite Annotation** — Liu, Zhang, Ge, Liu, He, Shen. *Health and Metabolism*, 2025. https://www.sciltp.com/journals/hm/articles/2504000541 | 综述 LLM/transformer 用于 RT 预测、MS2 生成、脂质预测、de novo 结构注释。| 为集成的"生成式"臂提供依据，界定 2025/2026 ML 代谢组学能守住的声称。
16. **Reverse metabolomics for the discovery of chemical structures from humans** — Gentry et al., Dorrestein. *Nature*, 2024. https://www.nature.com/articles/s41586-023-06906-8 | "先合成后检索"：新结合胆汁酸匹配公共数据集，四个 IBD 队列中克罗恩病升高。| 现成的端到端生物学案例（疾病代谢变化）+ 公共数据，可直接复现来演示注释器。
17. **Confronting spurious evaluations of computational methods in small molecule MS** — Gupta, Xu, Herbst, Wang, Wishart, Skinnider. *bioRxiv*, 2026. https://www.biorxiv.org/content/10.64898/2026.05.03.722532v2.full | 结构-only 模型靠参考库 vs PubChem 化学空间偏差匹配/超过多个 MassSpecGym 方法；提出生成式 decoy（Spectraverse）。| **对诚实策略最重要的一篇**——定义"不过度承诺"的含义与我们必须在其中评估的 decoy 空间。

### 4.4 建议

1. **基准先行**：在 MassSpecGym / CASMI 类公开基准上，跑出 DreaMS（我们）+ 融合各模块的消融表，得到可引用的精度/注释率/FDR 数字。
2. **两个生物学案例**：选 1–2 个公开、可复现、有原始数据的疾病/药物代谢数据集，端到端跑通，输出"注释率 + 通路证据"，结论严格限于工具表现。
3. **诚实包装**：对外表述用"集成 DreaMS/规则/RT/网络/生成 的多模态注释管线，基准 Top-1=…、注释率=…、FDR=…"，不用"全面、性能好"这类无数字的形容词。

---

## 5. 与现有闭环的集成位置

不推翻原架构，四条方向作为**正交证据源**并入：

```
模块1 表征学习（DreaMS，检索主分数）
        │
        ├─ 检索后重排：RT 先验（方向二）+ 代谢反应先验（方向一）
        ├─ 并列生成：DreaMS→SMILES decoder（方向三），输出库外候选
        │
模块2 化学证据解释（规则 + 峰级因果 + 隐式因子）
        │
        └─ 新增证据类型：反应可达证据、RT 符合度证据、生成置信度
            ↓
       统一基准 + 生物学案例（方向四，交付层）
```

**纪律不变**：规则/网络/RT/生成都只做**特征、重排、证据**，**不直接定义正负样本或 embedding 距离**；标签仍由 InChIKey + 严格 10 ppm 定义。最终测试集只读一次。

---

## 6. 分阶段执行优先级

| 阶段 | 内容 | 验收 | 优先级 |
|---|---|---|---|
| S0 | 基准地基：MassSpecGym/CASMI 跑出 DreaMS 基线数字，**分级报告 + decoy FDR** | 有可引用的分级精度/覆盖率/FDR（decoy 校正） | P0 |
| S1 | RT 先验（统计可用 RT → 预测器 → 重排融合） | Top-1/注释率增量 + 不降检索 | P0 |
| S2 | 代谢反应质量差先验（轻量库 + 共检出传播） | 注释率增量 + 覆盖审计 | P1 |
| S3 | DeepMet 确认 + DreaMS→SMILES 生成 smoke | SMILES 有效性 + 化学合理门 | P2 |
| S4 | 生物学案例 ×2（IBD 反代谢组学 / 药物代谢重编程）+ 多模态集成交付 | 端到端注释率 + 校准 rank-fusion + 逐模块消融表 | P0（必须） |

**一句话结论**：先做 S0/S1/S4 把"能交付、可核查"的闭环合上；S2/S3 是增量亮点，风险与回报成正比，需分门把关，绝不混进"没有数字的性能好"。

---

## 附：文献汇总（四方向共 60 篇，已联网核实）

| 方向 | 文献数 | 状态 |
|---|---|---|
| 1 代谢网络/生物先验 | 15 | 已核实 |
| 2 RT 先验 | 11 | 已核实 |
| 3 DeepMet/SMILES 生成 | 17 | 已核实 |
| 4 生物学应用/基准/包装 | 17 | 已核实 |
