# 从“反向检索”到真正的新研究对象：反向代谢组学、RDD、DeepMet 与 DreaMS Atlas 的范式审计

**日期：2026-09-12**  
**状态：深度文献调研结论；用于方向决策，不是实施合同**  
**替代关系：本文替代已撤回的 `REVERSE_METABOLOMICS_BEYOND_SEARCH_INNOVATION_LANDSCAPE_20260912.md`。**

## 0. 结论先行

上一版判断错在两点：第一，把“把标准谱作为查询”误当成尚待我们发明的反转；第二，在没有完成文献占位审计前，就把项目包装为“化学探针场”。这两点都不成立。

师兄提示的真正价值不是矩阵转置，而是**改变研究的基本单位**：

- 传统非靶向分析以“样本里这个峰叫什么”为基本问题；
- 反向代谢组学以“这个预先提出的化学结构或化学类，在什么生物系统中出现并与什么表型相关”为基本问题；
- Reference Data-Driven（RDD）分析进一步把“纯标准物”换成“带来源元数据的复杂参考样本”，因此即使结构未知的谱，也可作为食物、微生物或环境来源的证据；
- DreaMS Atlas 则把单次检索换成整个公共谱空间中的图结构，但它本身仍主要是一个化学邻域资源，不是疾病机制推断系统。

因此，**“用 DreaMS 把标准谱反向搜索样本”不构成创新**。Reverse Metabolomics、StructureMASST、microbeMASST、RDD 和 food-readout 工作已经覆盖了标准/结构/参考样本到公共生物语境的大部分直接路径。

本轮检索后，最值得继续验证、且尚未被这些工作等价覆盖的研究任务，不是另一个搜索器，而是：

> **学习一个经过技术可检测性校准的“化学—生物语境联合模型”：既能从谱图或结构预测其可能出现的组织、来源和疾病语境，也能从一个生物语境反向提出最值得寻找的已知及未知化学邻域；所有结论都在项目/实验室留出和队列内定量层面验证。**

这是一个**待证空白**，还不是“我们已经首创”的结论。它必须同时通过本文第 7 节的排除实验，才能成为项目主方向。

## 1. DeepMet 到底是什么，和这件事是什么关系

### 1.1 模型本体

DeepMet 不是 DeepMind 的项目，也不是 Transformer。它是一个三层 LSTM 化学语言模型：先在 ChEMBL 药物样结构上预训练，再用 2,046 个已在人类组织或体液中检测/定量的代谢物 SMILES 微调，用自回归采样频率学习“代谢物样化学空间”。论文报告它能够生成后来加入 HMDB 5.0 的多数代谢物，并把高采样频率作为代谢物先验。[DeepMet, Nature 2026](https://www.nature.com/articles/s41586-025-09969-x)

### 1.2 它做了三件不同的下游事

1. **结构先行的定向发现。**从模型高优先结构中购买/合成化合物，获得参考谱，再在同平台尿液或血液中用 RT 和 MS/MS 寻找；该阶段确认了 17 个预测物，但部分经文献复核属于数据库漏收的已知物。
2. **未知谱的常规结构注释。**从实验未知峰出发，用 DeepMet 先验、CFM-ID 预测谱、同位素、RT 等证据排序候选；随机森林融合将一组 held-out 评估的准确率提高到约 70%。这仍是“未知谱→结构”。
3. **参考谱先行的公共样本搜索。**论文还把新获得的参考 MS/MS 搜索了来自 MetaboLights 和 Metabolomics Workbench 的 35,460 个人体组织和细胞样本。这一应用和反向代谢组学方向相邻。

所以准确结论是：

> DeepMet 的**模型创新**是代谢物样结构的生成先验；它的论文**包含**结构/标准先行的定向发现和公共样本搜索，但 DeepMet 不是一个反向代谢组学搜索算法，更不是 DreaMS Atlas 式的谱图基础模型。

这一区分很重要。否则会把 DeepMet 的结构先验、标准验证和仓库搜索三个不同贡献混成一个不可复现的“大模型故事”。

## 2. 为什么“反过来”有意义，而不是数学上的同一件事

谱图相似度通常是对称的，但科研问题不是。

| 维度 | 传统前向注释 | 反向代谢组学 |
|---|---|---|
| 起点 | 队列中显著或未知的 feature | 预先定义的结构、标准谱或化学家族 |
| 候选分母 | 当前样本中的所有 feature | 预注册化学空间中的所有候选结构/谱 |
| 推断单位 | 单个 feature 的身份 | 单个化学假设在样本、组织、物种和表型中的分布 |
| 多重检验 | 常在看到差异峰后再注释 | 可在看队列结果前固定化学类和假设 |
| 实验顺序 | 先发现表型差异，再买标准 | 先获得标准/候选谱，再寻找生物学所在 |
| 输出 | “这个峰可能是谁” | “这个化学假设在哪里、和什么表型相关、下一步应在哪验证” |

[Nature 2024 的 Reverse Metabolomics](https://www.nature.com/articles/s41586-023-06906-8) 并不是简单交换 query 和 library。其完整贡献包含组合合成、2,430 张候选参考谱、公共仓库检索、元数据归纳、IBD 独立队列定量、同实验 RT/MS/MS、免疫和 PXR 功能实验，以及细菌产物验证。真正改变的是**先提出化学空间，再寻找生物学问题**。

但反向搜索也没有取消身份边界。原论文明确指出：MS/MS 检出依赖丰度、电离和 DDA 选择；公共库不覆盖所有分子；没有分离、NMR 或 X-ray 时不能排除某些异构体。因此，“仓库命中”是发现证据，不自动等于 Level 1 身份、真实未检出或机制。

## 3. 文献占位地图：哪些方向已经有人做了

### 3.1 逐范式比较

| 范式/代表工作 | 查询或参考对象 | 输出对象 | 是否利用未知谱 | 已经占据的空间 |
|---|---|---|---|---|
| Reverse Metabolomics, Nature 2024 | 合成物/标准的 MS/MS | 出现样本、组织、表型及后续验证 | 主要围绕预定化学类 | 标准/候选先行的公共仓库发现闭环 |
| StructureMASST, Nat Biotechnol 2026 | 名称、SMILES、SMARTS、亚结构、质量修饰 | 4,990 个数据集中的谱匹配和元数据分布 | 支持类似物和盲修饰 | “结构/亚结构/类似物→公共生物语境”已被直接实现 |
| RDD, Nat Biotechnol 2022 | 食物、微生物等复杂参考样本的整个 MS/MS 集 | 研究样本的来源/暴露读数 | **是，且是核心价值** | “参考来源代谢组→样本来源解释”已建立 |
| Open-source RDD, JASMS 2026 | GNPS molecular network + 分层参考集 | PCA、热图、Sankey、来源统计 | 是 | RDD 已有可用开源平台，不只是概念 |
| Food molecular fingerprints, 2025 预印本 | 500 余种食物中的 6,128 个谱学离子标志 | 饮食类别和相对摄入评分 | 是，仅约 6% 有结构 | “参考样本谱集合→临床样本表型/摄入预测”已有多场景验证数据，但尚未完成期刊同行评议 |
| microbeMASST, Nat Microbiol 2024 | 一张已知或未知 MS/MS | >60,000 微生物单培养中的生产者分类 | 是 | “谱→微生物来源”已实现 |
| MetSummarizer, 2021 | 样本对固定谱特征的 presence/absence | 表型分类、混合物成分分解 | 可用未命名谱特征 | “谱特征面板→样本分类”不是空白 |
| METASPACE-ML, Nat Commun 2024 | IMS 离子证据 + 样本上下文 | context-specific annotation + FDR | 间接 | “利用组织/物种上下文改善注释”已有成熟邻近工作 |
| TidyMass2, Nat Commun 2026 | 已注释或未注释 LC-MS feature | 七类来源与 feature-based 功能模块 | 是 | “不完整身份下的来源/功能解释”已有系统实现 |
| KGMN/MetDNA3 | 已知锚点、反应网、谱图网、相关网 | 从 known 向 unknown 传播结构 | 是 | “代谢网络+谱图邻域+样本共变”已被充分占据 |
| mummichog/PAIRUP-MS | m/z、差异统计、跨队列信号 | 无需完整身份的通路/跨队列功能 | 是 | “跳过身份直接讲通路”已有长期方法学 |
| IsoNet, Nat Commun 2025 | 稳定同位素时间/同位素体分布 | 未知反应和动态代谢路径 | 是 | 真正的未知反应/通量主张已有同位素证据标准 |
| DreaMS Atlas | 约 2.01 亿公共 MS/MS 的表示与聚类 | 全局化学邻域图和元数据邻域 | 是 | “把所有谱做成可查询 Atlas”已经是 DreaMS 自己的贡献 |
| DeepMet | 已知人类代谢物结构 | 未发现结构先验、候选和参考谱 | 结构生成本身不依赖未知谱 | “从已知代谢物预判新结构”已占据 |

关键资料包括：[RDD 原论文](https://www.nist.gov/publications/enhancing-untargeted-metabolomics-using-metadata-based-source-annotation)、[GNPS RDD 定义](https://ccms-ucsd.github.io/GNPSDocumentation/tutorials/rdd/)、[2026 开源 RDD 平台](https://pubmed.ncbi.nlm.nih.gov/41701920/)、[food readout](https://pmc.ncbi.nlm.nih.gov/articles/PMC12633204/)、[microbeMASST](https://pubmed.ncbi.nlm.nih.gov/38316926/)、[METASPACE-ML](https://www.nature.com/articles/s41467-024-52213-9)、[TidyMass2](https://doi.org/10.1038/s41467-026-68464-7)、[KGMN](https://www.nature.com/articles/s41467-022-34537-6) 和 [PAIRUP-MS](https://pmc.ncbi.nlm.nih.gov/articles/PMC6347288/)。

### 3.2 这张地图否定了什么

以下都不能再作为核心创新：

1. 用 DreaMS 代替 cosine/MASST 做标准谱搜索；
2. 用标准谱在疾病队列中统计“有没有”；
3. 把一组标准或参考谱做成疾病分类特征；
4. 对标准谱进行开放修饰/类似物搜索；
5. 把标准命中沿 Rhea/KEGG 或分子网络传播；
6. 用复杂来源样本作为伪谱库解释研究样本；
7. 把未知 feature 放进通路或来源模块；
8. 做另一个更大、更漂亮的谱图 Atlas；
9. 把上述模块串联后用新名词包装。

## 4. DreaMS Atlas 真正应该教我们的东西

DreaMS Atlas 的启发不是“我们也做一个 Atlas”。它的关键动作是把基础模型能力转化成一种此前难以直接操作的**研究对象**：公共仓库的全局谱图邻域。Atlas 中每个节点代表一组 DreaMS/LSH 聚类的谱，用户可以查询局部邻域、连接数据集元数据并找到带结构注释的邻居。[DreaMS Atlas 文档](https://dreams-docs.readthedocs.io/en/latest/tutorials/atlas.html)

同理，我们的新生物学应用只有在创造了新的推断对象时才成立。例如：

- 不是“某标准命中了多少张谱”，而是“一个化学邻域对不同生物语境的校准条件分布”；
- 不是“病例组是否更多”，而是“在控制项目、实验室、仪器和可检测性后，该化学邻域在何种语境中仍然富集”；
- 不是“给每个未知谱命名”，而是“从某个组织/疾病语境反向提出仍未命名但可重复出现的化学邻域”。

这才是“跳出注释流程”，而不是在注释流程末端加一个图或分类器。

## 5. 当前最强的新任务假设：学习化学—语境联合分布

### 5.1 任务定义

以 DreaMS 或后续更鲁棒的谱表示作为化学视图，以 Pan-ReDU 等项目级元数据作为语境视图，学习：

\[
P(C\mid S, A, D),\qquad P(S\mid C, A, D)
\]

其中：

- \(S\)：标准谱、实验谱或谱邻域；
- \(C\)：组织、物种、来源、疾病、暴露等生物语境；
- \(A\)：仪器、离子模式、碰撞能、采集方式等分析条件；
- \(D\)：该文件/项目对该类分子的可检测性与 MS/MS 选择概率。

两种方向对应两个以前分开的任务：

1. **chemical-to-context：**输入标准谱/结构，输出经过可检测性校准的语境分布；
2. **context-to-chemical：**输入组织/疾病/来源，输出最值得验证的已知与未知谱邻域。

Reverse Metabolomics 和 StructureMASST 主要做第 1 项的检索/计数；DeepMet 主要学习未条件化的代谢物样结构分布；RDD 和 food readout 用特定参考来源解释样本；METASPACE-ML 用已知 context 提高 IMS 注释。**本轮范围检索没有发现一个在 LC-MS/MS 公共仓库上，把这两种方向、显式技术可检测性和项目外泛化统一起来的成熟模型。**这只是范围检索后的候选空白，不等于已完成全球论文/专利查新。

### 5.2 为什么这可能是一个真正不同的生物学工具

它的输出不再是一个候选名字，而是三个可检验对象：

1. **语境特异性：**某谱邻域在何种组织/疾病/来源中超出其技术可检测性预期；
2. **未知化学家族：**即使没有结构名，也能发现稳定地与某一语境绑定的谱邻域；
3. **待验证化学假设：**对一个目标语境，提出最值得合成/购买标准的候选，而不是从几万个数据库候选中盲选。

这和 DeepMet 可以形成真正的正交关系：DeepMet 给出“像代谢物的结构”先验；语境模型给出“像这个组织/疾病会出现的结构或谱邻域”先验。数学上是从近似 \(P(\text{structure})\) 走向 \(P(\text{structure or spectrum}\mid\text{context})\)，不是把 DeepMet 接在 DreaMS 后面。

### 5.3 为什么不能现在就说它一定成立

公共仓库几乎没有可靠的真阴性。一个分子没有 MS/MS 命中，可能是：

- 样本中确实不存在；
- 浓度低或电离差；
- 色谱/离子模式不适合；
- DDA 没有选中它；
- 碰撞能和仪器导致谱不匹配；
- 元数据缺失或标签错误。

Reverse Metabolomics 本身明确承认丰度、电离和 MS/MS 采集的限制；Pan-ReDU 的存在说明跨仓库元数据必须先统一；METASPACE-ML 甚至为不同 context 单独抽样并强制项目/实验室多样性。因此，若不显式建模 \(D\)，所谓 \(P(C\mid S)\) 很容易只学到“哪个实验室常测什么”。MetSummarizer 已展示过内标和批次信息可以伪装成很强的表型分类信号。

## 6. 两个次级前沿，以及它们为何不能单独立项

### 6.1 检出率校准的队列级反向荟萃分析

工作定义：输入一组预注册标准/结构，在多个公开病例—对照队列中重新提取 MS1、确认 MS2，按研究内部效应量而不是仓库命中次数做随机效应荟萃分析。

价值：把“文件命中多”升级为队列内统计，部分解决平台和上传偏差。

不足：Reverse Metabolomics 已经对选中胆汁酸回到独立 IBD 队列做定量；因此单做“多队列重分析”更像严谨化和规模化，不足以成为新范式。它适合作为联合语境模型的验证层。

### 6.2 锚点化学邻域的转化与来源迁移图

工作定义：从标准或可靠锚点出发，检索其类似物/质量修饰，并比较这些邻域在食物、微生物、宿主组织和疾病中的分布，提出食物→微生物→宿主的候选转化链。

价值：从“这个分子在哪里”走向“这个化学家族如何跨生态位变化”。

不足：StructureMASST 已支持指定和盲修饰、同文件共现；BAM/KGMN 已使用生物转化规则和分子网络；food readout 已讨论宿主/微生物修饰；IsoNet 证明真正的反应发现需要动态同位素证据。因此，该方向若没有同位素、培养或至少独立来源数据，只能叫“转化假设图”，不能宣称反应或通量。

## 7. 决定这个方向是否值得做的七个硬门

只有全部通过，才应进入大规模实现。

### 门 1：任务不可被普通搜索等价实现

在相同谱库与数据上，与 cosine、DreaMS/FASST、StructureMASST 的文件级命中和简单元数据计数比较。若新方法只是提高若干搜索分数，不能立项。

### 门 2：项目、实验室和仪器真正外推

训练/验证必须以**项目或实验室为组**留出，而不是随机切谱。还要报告 unseen instrument、unseen study、unseen chemical identity/scaffold。若随机切分好而 project-held-out 崩溃，说明学到的是仓库批次。

### 门 3：可检测性模型胜过裸命中计数

用已知标准谱构造正例；把“未命中”视为 unlabeled 而非负例。至少采用 positive–unlabeled 或显式 observation model，并与匹配仪器/模式/质量范围的背景比较。若校准后不优于简单命中率，联合模型没有必要。

### 门 4：双向检索都产生增量价值

- chemical-to-context：已知分子的组织/来源标签必须在 identity-held-out 下恢复；
- context-to-chemical：目标 context 的 held-out 谱邻域必须被前排召回；
- 不能只报告一个容易方向。

### 门 5：未知谱不是被已知谱数量支配

在去除所有已知库匹配后，未知谱邻域仍应在独立项目中重复，并产生可验证的 context enrichment。否则只是更复杂的标准谱库搜索。

### 门 6：技术变量不能预测出同样结果

做 metadata-only、instrument-only、lab-only、precursor-mass-only 和 shuffled-context 对照；同时做 degree-preserving/label-permuted decoy。真实模型必须在 project-cluster bootstrap 下超过这些对照。

### 门 7：生物学验证回到原始队列

仓库级结果只能生成候选。至少一个发现必须回到原始 raw data：MS1 峰形与同位素、MS2 质量、队列内定量、协变量校正、独立队列复现。若主张精确结构，仍需同平台标准 RT+MS/MS；若没有标准，只能报告谱邻域、化学类或连接性级证据。

## 8. 与我们现有算法资产的真实关系

### 8.1 DreaMS embedding

它可能解决跨碰撞能/仪器的谱匹配，使同一化学实体在不同项目间更容易对齐。这是联合语境模型的必要底座之一。但官方 DreaMS 和我们的噪声微调都必须在 project-held-out 条件下证明增量，不能把内部检索提升直接折算成生物语境提升。

### 8.2 P2b

P2b 是候选组内的谱学重排器。它可以在 exact-identity 或 near-isomer 层提供第二证据，但不能定义 context 标签，也不能把它的候选任务增益当作联合语境模型增益。

### 8.3 峰级双重映射与 ChemAware

它们真正可能增加的不是“又一个规则分数”，而是：当模型把某谱邻域与一个生物语境关联时，定位哪些峰/中性丢失支撑关联，并用删峰或反事实峰干预验证忠实性。若做不到峰级定位和目标干预，化学解释仍是相关性装饰。

### 8.4 BioAware

BioAware 不应再以 Rhea 一跳传播来改写身份。若联合语境模型建立，BioAware 可以在后续把未知谱邻域组织成候选来源/反应假设，并明确支持、冲突和弃权。它不是第一阶段教师，也不是这项新任务的创新替身。

## 9. 最小、严谨、能快速否定的预实验

这不是立刻开发完整平台，而是用最小成本判断新任务是否真的存在。

### 9.1 数据对象

选择 Pan-ReDU 中元数据最完整、样本量足够的 3–5 个 context，而不是先选疾病故事。优先选择具有较明确生物边界和公开参考数据的 context，例如：

- 植物食物 vs 动物食物；
- 微生物单培养 vs 人体样本；
- 粪便 vs 血浆 vs 尿液；
- 药物相关 vs 非药物相关样本。

这些不是最终论文问题，而是系统正控。

### 9.2 三个模型，不得越级

1. **B0：**标准 MASST/FASST 或 DreaMS 最近邻 + 命中计数；
2. **B1：**只用技术元数据的可检测性/批次模型；
3. **M1：**化学谱表示 + 技术可检测性 + context 的联合模型。

只在 M1 同时超过 B0、B1，并在 project/lab-held-out 下稳定后，才进入 context-to-chemical 和未知谱发现。

### 9.3 预注册主指标

- chemical-to-context macro-AUPRC（类别不平衡时优先于 AUROC）；
- context-to-chemical Recall@K；
- project-cluster bootstrap 置信区间；
- 已知结构、未知谱、seen/unseen scaffold 分层；
- 仪器/实验室泄漏差值；
- 假阳性成本和 abstention 覆盖率。

### 9.4 立即停止条件

- 项目留出后 M1 不优于技术元数据模型；
- 去掉 precursor m/z 后性能坍塌，说明只是质量捷径；
- 未知谱子集无跨项目重复性；
- DreaMS 改进相对 cosine/FASST 无增量；
- 任何结果必须依赖已被查看的疾病标签调阈值。

## 10. 对师兄思路的当前最佳解释

师兄不是在建议把 MTBLS13729 的候选谱和标准库交换位置；也不只是建议“做 reverse metabolomics”。更可能的深层启发是：

> 不再让每个疾病队列决定我们研究哪几个显著峰，而是先建立一个可复用的化学参考坐标或化学假设空间，然后把大量生物样本投影到该空间中；再从化学空间与生物语境的联合分布中发现问题。

Nature 2024 完成了这一思想的第一版：预先合成化学类，再寻找疾病所在。RDD 完成了另一版：预先建立来源参考代谢组，再读取食物或环境来源。DreaMS Atlas 完成了谱空间版：先建立全局化学世界，再查询邻域。

我们若要再前进一步，不能重复这三者，而应把**离散命中/人工查看**升级为**可校准、双向、项目外泛化的化学—语境联合推断**。这才是当前文献地图下值得验证的“第五种反转”：

- 不只是查询方向反转；
- 而是从“注释一个对象”转向“学习化学世界与生物世界之间的条件分布”。

## 11. 当前决策

1. **撤销**“化学探针场”作为既定创新名称与结论；
2. **明确否定**“DreaMS + 反向检索”“BioAware + reverse metabolomics”“另建一个 Atlas”作为核心创新；
3. **保留**师兄的范式启发，但把它转化为可证伪的联合分布任务；
4. **先做第 9 节的小型方法存在性实验**，不立刻绑定 MTBLS13729；
5. 只有通过七门后，再选择疾病或生物系统做论文级应用；
6. 若失败，就诚实回退到 Reverse Metabolomics/RDD 的应用型工具改进，而不再用新名词包装。

## 12. 主要来源

1. Gentry EC et al. [Reverse metabolomics for the discovery of chemical structures from humans](https://www.nature.com/articles/s41586-023-06906-8). *Nature* 626, 419–426 (2024).
2. Gentry EC et al. [Reverse metabolomics protocol](https://www.nature.com/articles/s41596-024-01136-2). *Nature Protocols* (2025).
3. Qiang G et al. [Language model-guided anticipation and discovery of mammalian metabolites](https://www.nature.com/articles/s41586-025-09969-x). *Nature* 651, 211–220 (2026).
4. El Abiead Y et al. [Structure-centric searching enables global mapping of the public metabolome](https://www.nature.com/articles/s41587-026-03082-8). *Nature Biotechnology* (2026).
5. Jackson S et al. [Enhancing untargeted metabolomics using metadata-based source annotation](https://www.nist.gov/publications/enhancing-untargeted-metabolomics-using-metadata-based-source-annotation). *Nature Biotechnology* (2022).
6. GNPS. [Reference Data-Driven Analysis documentation](https://ccms-ucsd.github.io/GNPSDocumentation/tutorials/rdd/).
7. Mendoza Cantu A et al. [An Open-Source Platform for Reference Data-Driven Analysis of Untargeted Metabolomics](https://pubmed.ncbi.nlm.nih.gov/41701920/). *JASMS* 37, 803–806 (2026).
8. Gauglitz JM et al. [Learning molecular fingerprints of foods to decode dietary intake](https://pmc.ncbi.nlm.nih.gov/articles/PMC12633204/) (2025 preprint; not yet journal peer reviewed).
9. Zuffa S et al. [microbeMASST](https://pubmed.ncbi.nlm.nih.gov/38316926/). *Nature Microbiology* 9, 336–345 (2024).
10. Wadie B et al. [METASPACE-ML](https://www.nature.com/articles/s41467-024-52213-9). *Nature Communications* (2024).
11. Wang X et al. [TidyMass2](https://doi.org/10.1038/s41467-026-68464-7). *Nature Communications* 17, 1755 (2026).
12. Shen X et al. [KGMN: metabolite annotation from knowns to unknowns](https://www.nature.com/articles/s41467-022-34537-6). *Nature Communications* (2022).
13. Huang S et al. [PAIRUP-MS](https://pmc.ncbi.nlm.nih.gov/articles/PMC6347288/). *PLoS Computational Biology* (2019).
14. Li S et al. [Mummichog](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003123). *PLoS Computational Biology* (2013).
15. DreaMS. [DreaMS Atlas documentation](https://dreams-docs.readthedocs.io/en/latest/tutorials/atlas.html).
16. Pan-ReDU. [Enabling pan-repository reanalysis for big data science of public metabolomics data](https://pmc.ncbi.nlm.nih.gov/articles/PMC12103507/) (2025).
17. Goldman S et al. [MetSummarizer](https://pmc.ncbi.nlm.nih.gov/articles/PMC8050247/) (2021).
18. Gao Y et al. [Charting unknown metabolic reactions by stable-isotope tracing metabolomics](https://www.nature.com/articles/s41467-025-60258-7). *Nature Communications* (2025).
