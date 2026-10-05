# 【已撤回，不得作为路线依据】从“反向检索”到“化学探针场”：反向代谢组学之后还能真正反转什么

> **撤回说明（2026-09-12）：**本文把尚未理解清楚的师兄思路过早收敛为作者自拟的“化学探针场/生化程序断层扫描”，并且把 DeepMet 的模型、结构发现、标准验证和仓库搜索粗糙地混成一个概念。该概念不是由现有证据推出，不能作为创新结论、实施合同或项目决策依据。事实纠错见 `REVERSE_METABOLOMICS_FACTUAL_RESET_20260912.md`；当前方向判断只以 `REVERSE_METABOLOMICS_PARADIGM_AND_NOVELTY_AUDIT_20260912.md` 为准。

**日期：2026-09-12**  
**性质：深度调研与研究命题报告；不是实施合同，不包含尚未验证的性能主张**

## 执行摘要

师兄提出“把标准品谱图作为输入，反过来检查不同生物样本中是否存在以及有多少”时，关键并不在于把余弦相似度的两个参数交换位置。谱图相似度本身近似对称，单纯交换 query 和 library 在数学上没有新意。真正的反转发生在**研究对象、统计单位和最终输出**：

- 传统非靶向代谢组学以样本为起点，先发现差异 feature，再追问它是什么；
- 反向代谢组学以一个预先提出的化学结构或标准谱为起点，追问它在哪里、与什么表型相关；
- 更进一步的新范式不应停在“哪里出现了这个分子”，而应把标准及其可解释的化学邻域当成一组**生物系统探针**，追问一个样本激活了哪些生化转化程序、产生了哪些未知邻域成员，以及下一份最值得测量或合成的标准是什么。

调研显示，第一层反转已经高度成熟：MASST/FASST 能以谱图搜索公共仓库，StructureMASST 已能直接输入结构、亚结构或质量修饰并映射至样本元数据；DeepMet 已经从已知哺乳动物代谢物生成候选结构、购置或合成标准并回搜人类样本；DreaMS Atlas 已经把 2.01 亿张公共 MS/MS 组织成表示空间。因此，**“DreaMS + reverse metabolomics 搜索”只能算更换相似度引擎，不足以成为核心创新。**

本报告建议的唯一主攻命题是：

> **标准锚定的生化程序断层扫描（暂名）**：一个标准谱不是最终答案，而是化学坐标原点。系统从该原点沿着可解释的结构/反应方向，在每个样本中同时测量精确成员、未知类似物、转化边、共同变化与峰级证据，输出样本特异的“化学探针场”和“生化程序活性”，并以不确定性决定下一份最有信息量的标准，而不是逐个给未知峰强行命名。

这一命题与现有工作最实质的差异是：**从 molecule occurrence 转向 transformation-program measurement，从单次检索转向主动实验设计，从身份结论转向可校准的多层证据。**它能真正使用我们已有的 DreaMS 表示、噪声鲁棒性、P2b 谱学证据、ChemAware 规则、双重映射和 BioAware 上下文，而不把这些模块简单串成一个更复杂的候选排序器。

但必须守住边界：这仍是待验证的研究假设，尚不能称为首创。完整的论文与专利检索、跨队列原型、诱饵探针和标准品回收实验通过后，才有资格确定名称和创新主张。

---

## 1. 师兄的想法究竟“反转”了什么

### 1.1 不是矩阵转置，而是科学问题转置

设未知实验谱为 $x$，标准谱为 $s$，相似度为 $K(x,s)$。若 $K$ 是对称函数，把 $K(x,s)$ 写成 $K(s,x)$ 不会产生新算法。变化发生在外层问题：

| 范式 | 起点 | 被搜索空间 | 统计单位 | 输出 |
|---|---|---|---|---|
| 常规注释 | 样本中的未知 feature | 标准谱库/结构库 | feature | 最可能身份 |
| 经典反向代谢组学 | 合成物/标准物 | 公共样本仓库 | 化合物 | 出现在哪些样本/表型 |
| 本报告建议的进一步反转 | 标准锚定的化学假设或转化程序 | 多队列的谱图、丰度和上下文 | 探针 × 转化 × 样本 | 生化程序活性、未知邻域成员、下一实验 |

所以，师兄真正提供的是一种研究设计：过去是“数据先来，再努力解释”；现在是“化学假设先来，用所有样本回答它”。要产生新范式，还需再推进一步：从“一个分子是否出现”变成“一个化学程序如何在生物系统中展开”。

### 1.2 反向代谢组学为什么有生物学意义

它至少改变了四件事：

1. **把选择偏倚显式化。**常规流程只追逐显著差异 feature，稀有但机制关键的分子可能从未进入候选；化学假设先行可主动检查它。
2. **把未命名物保留下来。**标准的近邻、修饰物和共现物即使不能给出唯一结构，也可形成可复核的化学家族证据。
3. **把公共数据变成实验对象。**同一化学假设可跨组织、物种、疾病和实验条件检验，不再局限于单一队列。
4. **把标准品成本用于最有价值的问题。**若采用主动选择，标准品不是漫无目的补库，而是用于最大幅度降低一个生物学假设的不确定性。

---

## 2. 文献谱系：哪些方向已经有人做完了

### 2.1 谱图或结构反向搜索已经成熟

- [MASST](https://www.nature.com/articles/s41587-019-0375-9.pdf) 将单张 MS/MS 查询扩展到公共仓库，核心问题是“这张谱在什么数据中出现”。
- [FASST/MASST+](https://www.nature.com/articles/s41587-023-01985-4) 将仓库级搜索加速至可扩展规模。
- [Reverse metabolomics](https://www.nature.com/articles/s41586-023-06906-8) 合成 N-酰基酰胺、脂肪酸酯和胆汁酸结合物，获得标准谱后搜索公共数据并关联表型；其价值来自“先提出化学空间，再寻找人类证据”，不是相似度方向本身。
- [Nature Protocols 的反向代谢组学指南](https://www.nature.com/articles/s41596-024-01136-2) 已将这一过程标准化为 MS/MS-first 的大数据发现策略，并明确要求回到原始研究做 feature extraction 和统计分析。
- [StructureMASST](https://www.nature.com/articles/s41587-026-03082-8) 已允许以化学名称、SMILES、SMARTS、亚结构和质量修饰搜索跨仓库数据，并把结果映射到物种、组织、疾病和技术元数据。它还支持要求母体与类似物在同一文件或数据集中共现。

**裁决：**“用我们的 embedding 代替 cosine，然后标准谱反查队列”是有用工程，但不是足够的新问题。

### 2.2 从标准或候选化学空间预见未知代谢物也已有强工作

- [DeepMet](https://www.nature.com/articles/s41586-025-09969-x) 从 2,046 个哺乳动物代谢物学习可生成化学空间，产生超过 50 万个候选结构；其高优先候选中大量不在 HMDB，并通过购置/合成标准在人类体液中确认一部分预测。
- [AllCCS](https://www.nature.com/articles/s41467-020-18171-8) 用反应扩展结构库并结合预测 CCS、MS/MS 和实验数据注释未知物。
- [Propagated nearest-neighbour suspect library](https://www.nature.com/articles/s41467-023-44035-y) 在仓库级分子网络中把已知结构邻域传播到未知谱。

**裁决：**“生成潜在代谢物—预测谱图—回搜数据”也不能单独作为我们的新意。

### 2.3 从已知物向未知物传播、从未知 feature 推通路，也已高度拥挤

- [MetDNA](https://www.nature.com/articles/s41467-019-09550-x) 使用代谢反应网络从可靠种子递归注释未知物。
- [KGMN](https://www.nature.com/articles/s41467-022-34537-6) 联合反应网络、MS/MS 相似网络和全局峰相关网络。
- [NetID](https://www.nature.com/articles/s41592-021-01303-3) 以全局网络优化整合质量、RT、MS/MS、加合物、碎裂和生化转化。
- [PIUMet](https://www.nature.com/articles/nmeth.3940) 和 [TidyMass2](https://www.nature.com/articles/s41467-026-68464-7) 说明即使大量 feature 未被唯一注释，也可通过网络模块得到疾病相关生物学信号。
- [稳定同位素追踪未知反应](https://www.nature.com/articles/s41467-025-60258-7) 能把质量差、原子差和反应类别连接起来；这类工作拥有静态队列不具备的通量方向证据。

**裁决：**简单把 Rhea/KEGG/HMDB 图放在 DreaMS 后面传播候选，既不新，也容易把错误种子放大。我们此前 BioAware 的失败与这一文献边界一致。

### 2.4 亚结构、碎裂 motif 和活性导向也已有先例

- [CANOPUS](https://www.nature.com/articles/s41587-020-0740-8) 在无法得到唯一结构时预测化学类别。
- [MS2LDA 2.0](https://www.nature.com/articles/s41467-026-75038-0) 从重复碎片和中性丢失中学习 Mass2Motif，用亚结构层次组织未知谱。
- [StAR-MS](https://www.nature.com/articles/s41467-026-73030-2_reference.pdf) 以已知活性相关亚结构和诊断离子筛选肠道微生物来源分子，并继续做结构与活性验证。
- [Cell 的胆汁酸修饰研究](https://pure.mpg.de/rest/items/item_3590391_3/component/file_3709862/content) 使用 MassQL/诊断碎片在超大谱图空间发现胆汁酸修饰多样性。

**裁决：**“用若干诊断峰搜索某个化学类”也不能独立成为创新；我们的机会是用可学习、可干预验证的峰—概念映射替代完全手写且单骨架的规则。

### 2.5 DreaMS Atlas 真正值得学习的地方

[DreaMS 论文](https://www.nature.com/articles/s41587-025-02663-3.pdf) 的 Atlas 不是“又做了一次注释”。它把 2.01 亿张 MS/MS 的 embedding 近邻关系预计算成一个可复用的全局化学空间，使后续研究者不必从自己的小数据集重新构图。[官方 Atlas 文档](https://dreams-docs.readthedocs.io/en/latest/tutorials/atlas.html) 说明每个谱节点同时带 embedding、峰、前体、RT 和 MassIVE 元数据。

它的启示是：**算法最有价值的下游应用，往往不是把一个旧指标提高一点，而是把过去无法查询的对象变成新的公共科学基础设施。**我们应学习这种“创造新可查询对象”的思路，而不是复制 Atlas 的图形形式。

---

## 3. 现有路线尚未解决的共同空白

### 3.1 occurrence 不等于 biochemical program

StructureMASST 可以回答某结构或类似物在哪些样本中出现，但不能直接回答：

- 一个化学骨架在该样本中主要经历哪类修饰；
- 多个结构近邻是否构成同一个协同变化的生化程序；
- 观测到的是底物、产物、同源类似物、原位碎片还是共洗脱污染；
- 哪一类反应方向在病例中系统性增强；
- 下一份标准测什么，最能区分这些解释。

### 3.2 单分子身份是过窄、也常不可验证的终点

在没有同平台标准 RT 的情况下，MS/MS 高相似最多通常支持 Level 2 候选；异构体、共隔离和 in-source fragment 仍可能造成错误。若整篇生物学故事依赖一个候选名字，审稿人完全可以要求标准品。相反，若主要结论是“一个标准锚定、跨多个未知成员复现、由多条独立谱学证据支持的转化程序在疾病中增强”，单个未知物的身份就不再是唯一承重墙。

### 3.3 仓库热图不是生物学结论

公共数据有物种、组织、疾病、仪器、实验室和上传习惯的混杂。[Pan-ReDU](https://www.nature.com/articles/s41467-025-60067-y) 之所以重要，就是因为跨仓库元数据必须统一；反向代谢组学指南也明确要求返回原始队列做 feature-level 定量和统计。任何“某谱在病例文件更多”都只能生成假设，不能替代队列内对照。

### 3.4 现有工作很少把“下一标准”作为模型输出

Reverse metabolomics 和 DeepMet 仍然需要研究者预先决定合成/购买哪些化合物。真实标准谱库的建设成本和跨平台采集负担早已被 [Curatr](https://academic.oup.com/bioinformatics/article/34/8/1436/4741357) 等工作明确指出；[WEIZMASS](https://www.nature.com/articles/ncomms12423) 也说明高置信身份依赖同条件下的标准谱和正交性质。若一个化学类有数千个可能成员，枚举标准非常昂贵。真正面向实验效率的系统应计算：哪个候选标准能够最大幅度消除当前关于结构邻域、反应方向或表型关联的不确定性。

这里必须谨慎：主动学习和最优实验设计并不是新概念，也已用于代谢网络优化、代谢位点测量等相邻任务。潜在新意只能来自一个更窄的组合命题——**以恢复样本特异化学邻域和转化程序为目标，主动决定下一份应获取的标准谱**。本报告尚未证明该组合无人做过，必须继续做正式查重。

### 3.5 搜索置信度和生物学置信度没有被统一

谱图匹配 FDR、存在性、定量差异、网络模块和机制解释常被分开报告。[代谢组谱库匹配显著性研究](https://pmc.ncbi.nlm.nih.gov/articles/PMC5684233/) 证明不同项目需要不同阈值，固定 cosine cutoff 并不可靠。一个新平台必须区分：

1. 谱图是否像；
2. 该分子/家族是否真的存在；
3. 是否在特定表型中改变；
4. 是否支持某类转化；
5. 是否支持通量或因果机制。

前四层可以逐步计算，静态数据不能自动越到第五层。

---

## 4. 推荐的新科学对象：标准锚定的“化学探针场”

### 4.1 工作定义

给定标准或高可信参考谱 (p_j)，不只返回每个样本中最相似的谱，而是在样本 (s) 中构造：

\[
\mathcal{F}_{j,s}=\{E^{exact}_{j,s}, E^{analog}_{j,s}, E^{transform}_{j,r,s}, A_{j,s}, C_{j,s}, U_{j,s}\}.
\]

- (E^{exact})：校准后的精确成员谱学证据；
- (E^{analog})：标准周围未知化学邻域的证据；
- (E^{transform}_{r})：某类转化 (r) 的结构、质量差、峰级和共现一致性；
- (A)：MS1/EIC 层的相对丰度或检出率；
- (C)：与同一化学程序内其他成员的条件内共变；
- (U)：由诱饵、仪器条件、异构体竞争和缺失传播得到的不确定性。

样本不再只得到一张 feature 表，而得到一组“探针场读数”。病例—对照、时间、组织或物种比较的对象是这些探针场及其转化轴，而非仅仅若干被强行命名的峰。

### 4.2 为什么叫“断层扫描”而非搜索

单个标准像一个投影角度，只能照亮化学空间的一部分。多个结构相关标准从不同位置照射同一未知邻域，可逐步恢复：

- 哪些成员共享核心骨架；
- 哪些峰对应公共核心，哪些峰区分修饰；
- 哪个质量差/反应方向在样本中出现；
- 哪一组未知物随表型共同变化；
- 哪个标准最值得补测以解除歧义。

因此，标准不是“把未知物认出来”的一次性钥匙，而是对未知代谢空间进行多角度测量的探针。

### 4.3 最关键的输出不是名字，而是三张图

1. **Probe × Sample 图：**某标准及其化学邻域在样本中的出现、丰度和置信度；
2. **Probe × Transformation 图：**从标准出发哪些修饰/反应方向被观测并在何种表型增强；
3. **Transformation × Peak 图：**每个转化判断由哪些碎片、中性丢失和峰 token 支持，删去这些峰是否削弱判断。

三张图共同形成“化学假设—生物上下文—谱学证据”的闭环。

---

## 5. 它与现有方法究竟有什么不同

| 工作 | 输入 | 主要输出 | 尚缺少的环节 | 本命题的增量 |
|---|---|---|---|---|
| MASST/FASST | 单张谱 | 仓库匹配 | 化学程序、主动实验 | 从匹配升级为标准邻域的程序读数 |
| Reverse metabolomics | 合成标准集合 | 出现和表型关联 | 学习型邻域、转化算子、下一标准 | 标准不只检索自身，而测量未知邻域与程序 |
| StructureMASST | 结构/亚结构/修饰 | 全局分布 | 队列内定量模型、峰级因果、主动设计 | 把结果组织成可检验的样本特异程序 |
| DreaMS Atlas | 任意谱及其全局近邻 | 全局化学网络 | 标准锚定生物程序 | 将 Atlas 作为底层化学坐标，不复制 Atlas |
| MetDNA/KGMN/NetID | feature + 网络 | 身份传播/全局注释 | 标准探针场、主动标准、跨仓库统计 | 网络只约束转化假设，不覆盖谱学身份 |
| MS2LDA/CANOPUS | 未知谱 | motif/类别 | 标准锚定、样本程序和定量 | motif 成为探针场的可解释局部证据 |
| StAR-MS | 活性相关诊断亚结构 | 活性候选物 | 通用表示、跨骨架、自动不确定性 | 用双重映射学习谱峰—概念，不限单一手写骨架 |
| TidyMass2/PIUMet | 差异 feature | 功能模块 | 标准谱锚定和精确谱学证据 | 模块由可追溯的化学探针场构成 |

这一比较说明，真正可能构成方法学创新的不是任一单独组件，而是以下四项同时成立：

1. 标准锚定但允许未知邻域成员参与；
2. 估计的是样本特异转化程序，不是只做 identity ranking；
3. 峰级证据、诱饵和队列定量共同给出分层置信度；
4. 系统用信息增益提出下一份标准/实验。

少掉第 2 项会退化为 StructureMASST；少掉第 3 项会退化为漂亮但不可靠的网络图；少掉第 4 项仍是一次性分析工具，而不是新的发现闭环。

---

## 6. 我们已有算法资产在新范式中的正确位置

### 6.1 噪声微调后的共享 embedding

它的任务不是“证明某疾病机制”，而是让同一探针跨仪器、碰撞能和缺峰条件仍有稳定邻域，同时避免被少数共享高强度峰错误吸引。它决定探针场是否可跨数据集迁移。当前共享权重的有效提升应按项目冻结台账单独报告，不能与下游专家数字相加。

### 6.2 P2b 谱学专家

P2b 可作为精确/类似物证据的一条独立通道，尤其用于常规候选排序。但其 near-core 安全边界必须保留；它不能定义转化程序，也不能被包装成新 embedding。

### 6.3 ChemAware 规则与双重映射

这是本新范式最可能形成独特壁垒的模块。作用不是用规则投票替代模型，而是回答：

- 探针与未知邻域成员共享什么碎裂概念；
- 哪些峰支持某个修饰方向；
- 哪些峰只是仪器/碰撞条件特异；
- 定向删峰是否削弱该关系，而匹配随机删峰不产生同等效应。

也就是说，双重映射将“embedding 邻近”变成可证伪的化学关系。

### 6.4 BioAware 上下文

BioAware 不再用 Rhea 一跳分数直接改写身份。它只负责把多个探针场读数在样本内组织成反应/离子家族/共变模块，并标记支持、冲突和证据不足。它回答“这些化学事件是否作为一个程序共同出现”，不回答“这个谱一定叫什么”。

### 6.5 真实 MS1 定量和生物学分析

MS/MS 负责探针场的结构证据，MS1/EIC 负责队列内存在和相对丰度。必须使用患者配对、批次/QC、缺失、离子家族去冗余和多重校正。静态丰度只能支持“程序相关的丰度重塑”，不能支持通量。

---

## 7. 六个候选创新方向的淘汰矩阵

| 候选方向 | 新颖性 | 与现有资产契合 | 无湿实验可行性 | 被已有方法覆盖风险 | 裁决 |
|---|---:|---:|---:|---:|---|
| DreaMS 版 MASST | 低 | 高 | 高 | 极高 | 仅工程基线 |
| 标准谱反查 MTBLS13729 | 低 | 中 | 高 | 极高 | 不作为论文主线 |
| 标准锚定未知 analogome atlas | 中 | 高 | 高 | 中高 | 可作子模块 |
| 标准锚定生化程序断层扫描 | 高 | 极高 | 中高 | 中 | **主攻** |
| 主动选择下一标准/合成物 | 高 | 高 | 中 | 低中 | 第二阶段核心增量 |
| 直接用全谱预测疾病标签 | 中低 | 中 | 高 | 高 | 不推荐，解释性和跨队列弱 |

### 为什么不优先做“直接疾病分类”

[DeepMSProfiler](https://www.nature.com/articles/s41467-024-51433-3) 和 [MassCube](https://www.nature.com/articles/s41467-025-60640-5) 已经能从原始 LC–MS 构建表型分类与贡献热图。该任务容易取得漂亮 AUC，却可能只学习批次、仪器和队列差异，也无法解决标准品可信度问题。它不是师兄想要的那种化学—生物学新对象。

---

## 8. 最小但决定性的验证：先证明新“对象”存在

现在不应立刻大规模建平台。先做一个不依赖表型调参、可以快速证伪的原型。

### 8.1 数据选择

选择一个同时具有以下条件的公开项目，而不是把 MTBLS13729 硬塞进去：

- 至少 100–200 个生物样本，病例/对照或明确时间/处理设计；
- MS1 + MS2 原始数据完整，有 pooled QC/blank，元数据可解析；
- 至少一个已有标准品验证的化学家族或反应轴，可作阳性对照；
- 仍保留大量未注释但高质量 MS/MS，可检验未知邻域；
- 最好有独立队列或跨中心数据。

MTBLS13729 可作为后续 CRC 应用和配对组织验证，但其无 pooled QC、Rmu 仅 10 对、标准真值少，不适合承担首个方法学成败裁决。

### 8.2 阳性和阴性对照

每个标准探针同时建立：

- 真实标准谱；
- 前体质量匹配、峰数/强度匹配的 decoy 谱；
- 化学结构相近但生物来源不合理的 entrapment 探针；
- 打乱反应类型或端点的 transformation decoy；
- 标签置换的 phenotype null；
- 同一化学家族内留一标准，检验邻域恢复能力。

### 8.3 第一阶段只回答四个问题

1. 标准探针能否在跨条件数据中可靠回收已知成员？
2. 探针邻域能否找回被留出的同家族/转化成员，并显著超过质量差、cosine 和随机网络？
3. 一个预先冻结的转化程序分数能否在独立队列复现，而不是只在单 feature 上显著？
4. 双重映射定位的峰是否对该程序关系具有目标删峰特异性？

四问中任一关键门失败，就不能进入主动标准选择或生物学新机制包装。

### 8.4 第二阶段才做主动标准选择

将候选标准 (c) 的优先级定义为预期信息增益，而不是模型置信度最高：

\[
\text{priority}(c)=\mathbb{E}[H(\Theta\mid D)-H(\Theta\mid D,y_c)]-\lambda\,\text{cost}(c),
\]

其中 $\Theta$ 是关于化学邻域/转化程序的后验，$y_c$ 是测得该标准后的新证据。实验上用 retrospective simulation 验证：隐藏已有标准，再比较主动策略、随机选择、按丰度选择和按模型置信度选择，用多少个标准能恢复相同的正确邻域与生物学结论。

这一步无需立即购买标准，就能在已有带标准真值的数据上验证“减少标准品数量”的方法学价值。

---

## 9. 论文级生物学故事应如何形成

一篇强工作不应是“我们多注释了若干代谢物”，而应形成以下链条：

1. **化学假设：**某类已知信号分子/宿主—微生物共代谢骨架可能存在尚未描绘的转化空间；
2. **探针场：**由少量真实标准及可解释 embedding 方向发现一组精确成员和未知邻域成员；
3. **程序统计：**整组转化轴而非单个 feature 在发现队列改变，并在独立队列复现；
4. **来源证据：**组织、物种、微生物培养、饮食或干预元数据支持来源，而不是仅依赖 Rhea 邻接；
5. **谱学闭环：**峰级解释、质量差、共洗脱/离子家族排除和诱饵 FDR 共同成立；
6. **验证效率：**主动选择表明少量标准即可最大化对关键分支的确认；
7. **结论边界：**有标准的成员可提高身份等级；无标准成员报告为 family/transformation-resolved unknown，不冒充 Level 1；静态队列不声称通量。

这种故事的生物学主语是“转化程序”或“化学家族生态”，不是某个脆弱的候选名称。因此，它既能抵抗“你凭什么说这个代谢物是真的”这一质疑，也没有回避标准品：标准被重新用于锚定和最大信息增益，而不是被假装不再需要。

---

## 10. 当前最重要的反思与结论

### 10.1 我们此前为什么仍未理解师兄

我们一直把问题压回熟悉的模块边界：DreaMS 做相似度、P2b 重排、BioAware 加网络、MTBLS13729 做应用。这样得到的是一条更长的注释 pipeline，却没有改变论文的科学对象。师兄的启发恰恰是：**不要再以“每张未知谱最终叫什么”组织整个项目。**

### 10.2 真正值得学习 DreaMS Atlas 的不是“做 Atlas”

DreaMS Atlas 的创新在于让 2.01 亿张谱构成一个可直接查询的化学世界。我们的对应目标不应是复制另一个大图，而是创造一个此前不可查询的对象：

> **任何一个标准或化学假设，在所有样本中张成的、带转化方向、丰度、上下文、峰证据和不确定性的生物探针场。**

### 10.3 唯一建议

先把“标准锚定生化程序断层扫描”作为研究命题，不急于给平台起最终名字，也不急于把 MTBLS13729 填进去。下一步应是：

1. 完成更正式的论文/专利查重，重点检索 standard-anchored chemical neighborhood、active standard selection、transformation-program inference；
2. 选择一个有标准真值和独立队列的公开化学家族，做 retrospective 留标准验证；
3. 先用冻结 DreaMS、P2b、规则和双重映射构造严格基线，证明“探针场/程序分数”优于单分子搜索和普通网络传播；
4. 通过后才决定是否训练新的 probe-field embedding 或 BioAware operator；
5. 最后再把通过的方法迁移到 CRC/MTBLS13729，寻找新的生物学转化程序。

这条路线既不是照搬 reverse metabolomics，也不是把我们已有成果重新包装。它把已有模块变成测量一个新科学对象所必需的仪器，并把最昂贵的标准品验证转化为可优化的主动实验步骤。

---

## 主要参考文献

1. Bushuiev et al. [Self-supervised learning of molecular representations from millions of tandem mass spectra using DreaMS](https://www.nature.com/articles/s41587-025-02663-3.pdf). Nature Biotechnology, 2025.
2. Wang et al. [Mass spectrometry search tool for identification of small molecules from mass spectrometry data](https://www.nature.com/articles/s41587-019-0375-9.pdf). Nature Biotechnology, 2020.
3. Bittremieux et al. [Fast analog searching and spectral networking](https://www.nature.com/articles/s41587-023-01985-4). Nature Biotechnology, 2024.
4. Gentry et al. [Reverse metabolomics for the discovery of chemical structures from humans](https://www.nature.com/articles/s41586-023-06906-8). Nature, 2024.
5. Charron-Lamoureux et al. [A guide to reverse metabolomics—a framework for big data discovery strategy](https://www.nature.com/articles/s41596-024-01136-2). Nature Protocols, 2025.
6. Wang et al. [Structure-centric searching enables global mapping of the public metabolome](https://www.nature.com/articles/s41587-026-03082-8). Nature Biotechnology, 2026.
7. [Language model-guided anticipation and discovery of mammalian metabolites](https://www.nature.com/articles/s41586-025-09969-x). Nature, 2026.
8. Shen et al. [Metabolic reaction network-based recursive metabolite annotation for untargeted metabolomics](https://www.nature.com/articles/s41467-019-09550-x). Nature Communications, 2019.
9. Zhou et al. [Metabolite annotation from knowns to unknowns through knowledge-guided multi-layer metabolic networking](https://www.nature.com/articles/s41467-022-34537-6). Nature Communications, 2022.
10. Wang et al. [Metabolite discovery through global annotation of untargeted metabolomics data](https://www.nature.com/articles/s41592-021-01303-3). Nature Methods, 2022.
11. Pirhaji et al. [Revealing disease-associated pathways by network integration of untargeted metabolomics](https://www.nature.com/articles/nmeth.3940). Nature Methods, 2016.
12. Wang et al. [TidyMass2](https://www.nature.com/articles/s41467-026-68464-7). Nature Communications, 2026.
13. Dührkop et al. [Systematic classification of unknown metabolites using high-resolution fragmentation mass spectra](https://www.nature.com/articles/s41587-020-0740-8). Nature Biotechnology, 2021.
14. van der Hooft et al. [MS2LDA 2.0](https://www.nature.com/articles/s41467-026-75038-0). Nature Communications, 2026.
15. [Substructure-activity relationship mass spectrometry for activity-guided metabolite discovery](https://www.nature.com/articles/s41467-026-73030-2_reference.pdf). Nature Communications, 2026.
16. Gauglitz et al. [Enhancing untargeted metabolomics using metadata-based source annotation](https://www.nature.com/articles/s41587-022-01368-1). Nature Biotechnology, 2022.
17. Pan-ReDU consortium. [Pan-ReDU](https://www.nature.com/articles/s41467-025-60067-y). Nature Communications, 2025.
18. Scheubert et al. [Significance estimation for large scale metabolomics annotations by spectral matching](https://pmc.ncbi.nlm.nih.gov/articles/PMC5684233/). Nature Communications, 2017.
19. Nash et al. [DecoID improves identification rates in metabolomics through database-assisted MS/MS deconvolution](https://www.nature.com/articles/s41592-021-01195-3). Nature Methods, 2021.
20. Alseekh et al. [Mass spectrometry-based metabolomics: a guide for annotation, quantification and best reporting practices](https://www.nature.com/articles/s41592-021-01197-1). Nature Methods, 2021.
21. Kirwan et al. [Curatr: a web application for creating, curating and sharing a mass spectral library](https://academic.oup.com/bioinformatics/article/34/8/1436/4741357). Bioinformatics, 2018.
22. Shahaf et al. [The WEIZMASS spectral library for high-confidence metabolite identification](https://www.nature.com/articles/ncomms12423). Nature Communications, 2016.
