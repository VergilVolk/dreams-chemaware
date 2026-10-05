# 反向代谢组学方向的彻底重置：从“反向检索”到“标准品高效的主动化学验证”

> **2026-09-12 裁决：本文提出的“主动挑选标准品/最小实验消歧”仅保留为未来验证工具，不再承担方法学主创新。它未跳出候选检索与身份确证范式。新的 atlas 方向见 `REFERENCE_ANCHORED_PEAK_OPERATOR_ATLAS_20260912.md`。**

**日期：2026-09-12**  
**状态：方法定义与查重后的研究合同；不包含新模型结果，不授权继续 A0–A2 联合语境模型**

## 0. 结论先行

此前的错误不是实现不够复杂，而是科学问题没有先定义清楚：

1. 把参考谱作为 query 搜索生物样本，只是检索方向和研究设计的改变；相似度本身通常仍是对称的。
2. 这种“参考结构/标准谱 → 公共样本分布”已被 reverse metabolomics、MASST、StructureMASST、microbeMASST、RDD 和多种伪靶向工作覆盖，不能单独作为新算法。
3. A0–A2 把公共数据库的 sample type / project observation 当作预测目标，实际上主要学习可检测性、实验室、仪器和数据库覆盖度；它没有在真实队列中建立标准谱 × 样本 × 定量证据，也没有解决身份验证成本。
4. 仅仅提出“主动买标准品”也不够新。SAMBA 已经用疾病条件下的代谢网络模拟来排序值得测量/购买的代谢物；主动学习和信息增益本身在化学实验设计中也早已存在。

因此当前唯一值得继续查证的方法命题是：

> **给定真实队列中已经观测到、与表型相关、但被异构体和采集条件混淆的一组谱学候选，如何选择最少的一批同平台标准品或定向再采集实验，使最多的身份歧义、化学转化分支和生物学结论获得可审计的消歧；每次实验结果再更新候选图和下一轮实验。**

这不是“另一个搜索器”，而是一个**闭环实验决策问题**。目前只能称为待证研究空白，不能宣称首创。

## 1. 师兄思路的准确含义

师兄说“把标准品谱图输入，再去看不同生物样本里有没有、多少”，其真正启发有两层。

第一层是 chemistry-first：先定义要追踪的化学假设，再观察它在样本和表型中的分布，而不是先挑显著 feature、再为它找名字。Nature 的 reverse metabolomics 正是先组合合成并实测一批潜在代谢物，再搜索公共数据、定位表型关联，最后做队列复现和功能验证。[Nature 2024](https://www.nature.com/articles/s41586-023-06906-8)

第二层是实验闭环：标准品不只提供一个库谱分数，而是改变证据等级。只有同平台的 RT、MS/MS、必要时加标共洗脱等正交证据，才能把一部分候选从 putative annotation 推进到更高置信度。社区最佳实践也明确区分了参考库匹配与同方法 authentic standard 验证。[Nature Methods 2021](https://www.nature.com/articles/s41592-021-01197-1)

所以它不是简单的矩阵转置。真正的非对称性来自：

- 化学假设在看表型之前被冻结；
- 标准品实验会产生新的、不可由旧样本分数等价替代的证据；
- 研究目标从“给每个峰起名字”变成“用有限实验最大化可确认的化学—生物学知识”。

## 2. 现有工作已经做到哪里

### 2.1 已有，不能再冒充创新

| 已有能力 | 代表工作 | 对本项目的约束 |
|---|---|---|
| 单个标准/参考谱搜索公共数据 | [MASST](https://www.nature.com/articles/s41587-019-0375-9) | DreaMS 替换 cosine 只能算检索组件改进 |
| 结构、子结构、类似物、修饰容忍的全公共库搜索 | [StructureMASST](https://www.nature.com/articles/s41587-026-03082-8) | “输入结构后看在哪里出现”已经成熟 |
| 合成潜在代谢物后反查人类数据和表型 | [Reverse metabolomics](https://www.nature.com/articles/s41586-023-06906-8) | chemistry-first 本身已被证明，但其成功依赖合成、队列和功能实验 |
| 从复杂参考样本推断来源 | [RDD](https://www.nature.com/articles/s41587-022-01368-1) | 来源/语境传播不是我们的独占创新 |
| 反应网络递归注释 | [MetDNA](https://www.nature.com/articles/s41467-019-09550-x) | Rhea/KEGG 一跳传播不是新算法 |
| 由疾病代谢模型排序值得测量的代谢物 | [SAMBA](https://pmc.ncbi.nlm.nih.gov/articles/PMC10914266/) | “选择买哪些标准品”若只依赖疾病/网络预测，已有直接近邻 |
| 主动学习减少化学标注工作量 | [JCIM 2024](https://pubmed.ncbi.nlm.nih.gov/38170877/) | uncertainty / information gain 是通用工具，不是创新本身 |
| 大规模谱空间图 | [DreaMS Atlas](https://www.nature.com/articles/s41587-025-02663-3) | 再画一个 atlas 没有价值；必须让图支持此前不能做的实验决策 |

### 2.2 Nature reverse metabolomics 真正强在哪里

该论文的贡献不是“把 query 和 library 对调”。它同时完成了：

- 以组合合成扩展可检验的化学假设空间；
- 实测约 2,430 个候选化合物的 MS/MS；
- 在约 12 亿张公共谱中搜索；
- 用 ReDU 元数据定位物种、组织和疾病语境；
- 在多个 IBD 队列复现；
- 对关键胆汁酸共轭物做同实验 RT/MS/MS、定量、免疫/PXR 和微生物产生验证。

任何只完成“参考谱 → 样本命中”的项目，都只复现了这条证据链的一小段。

## 3. 对仓库现有工作的裁决

### 3.1 A0–A2 chemical-context joint model：撤回主线资格

代码审计确认：

- A0 从 StructureMASST/FASSTrecords 的预计算公共命中出发；
- A1 按 project 和 identity 做切分；
- A2 用 technical baseline、ClassyFire superclass 和 Morgan 邻居去预测 project-level sample-type observation；
- 它没有把真实标准/参考谱逐一送入 MTBLS13729 的 119 个原始样本；
- 它没有建立可靠的 MS1 定量；
- 它没有产生新的标准品验证证据。

即使 project holdout 分数变好，也很难区分结构—生物语境信号与离子化、仪器、项目组成和公共库采样偏差。该任务与 MetSummarizer、RDD、样本来源分类已有工作高度邻近。A0–A2 只保留为**检测偏差负对照**，不得继续扩成 DreaMS/BioAware 主模型。

### 3.2 MTBLS13729 reverse-probe 原型：保留为工程基线，不是论文创新

该原型真实完成了 reference spectrum × real sample × evidence 的一部分，并发现公共参考谱到冻结 MS1 feature 的桥接可工作；但它仍有硬边界：

- 参考谱不是本实验平台的 authentic standards；
- raw MS/MS 对同质量 feature 的选择能力弱；
- MTBLS13729 缺 pooled QC，且 Rmu 只有 10 对；
- MS2 的未检出不能当作样本中不存在；
- 它不足以证明新的精确身份或通量。

因此它适合作为“普通反向检索/伪靶向”基线和 CRC 后续应用，不承担新方法的成败。

### 3.3 当前模块的真实资格

| 模块 | 当前真实状态 | 在新命题中的合法角色 |
|---|---|---|
| 官方 DreaMS | 可用的跨条件谱表示基线 | 构建候选相似邻域 |
| 新噪声微调 embedding | 尚未证明全面优于官方 DreaMS | 通过独立门后才能替换基线 |
| P2b | 特定协议有小幅盲测增益，但 near-core 退化 | 可作为谱学证据通道/对照，不能冒充新 embedding |
| ChemAware | 化学特异性审计未通过 | 不能作为身份真值；未来只在特异性通过后参与峰证据 |
| 双重映射 | 尚未完成系统映射和忠实性门 | 未来用于解释“哪条分支被哪个峰消歧” |
| BioAware | 固定一跳和多轮直接覆盖均未建立可靠增量 | 只可用于实验优先级先验，不得覆盖谱学身份 |

## 4. 真正可能成立的新方法：主动化学消歧闭环

### 4.1 研究对象

基本单位不是一张 query 谱，也不是一个代谢物名称，而是一个**队列中的歧义组**：

\[
G_i = \{\text{同一局部质量/RT区域内仍可能成立的候选身份及异构体}\}.
\]

每个组同时有：

- 多样本 MS1 强度和检出；
- 多条件 MS/MS；
- 官方/新 DreaMS 相似度；
- raw peak、neutral-loss、adduct/isotope/co-elution 证据；
- 候选结构与可采购标准列表；
- 表型关联，但表型只决定研究价值，不作为身份真值。

### 4.2 可选择的动作

动作不只等于“买一个标准品”：

- 在同平台注入一个 authentic standard；
- 标准品加标到代表性 biological matrix；
- 为一个前体追加指定 CE 的 PRM/MS2；
- 对异构体增加正交分离或 ion mobility；
- 在确有条件时购买同位素内标做定量。

每个动作都有成本，并会排除或支持歧义组中的一部分候选。

### 4.3 优化目标

不是选择当前置信度最高的标准，而是选择**最能改变结论**的实验：

\[
a^*=\arg\max_a \frac{
\mathbb E[\Delta H_{identity}(a)]
+\lambda_1\mathbb E[\Delta H_{program}(a)]
+\lambda_2 V_{biology}(a)
-\lambda_3 R_{false\ claim}(a)
}{cost(a)}.
\]

其中：

- \(\Delta H_{identity}\)：动作后候选身份不确定性减少多少；
- \(\Delta H_{program}\)：一个化学转化家族/程序的不确定性减少多少；
- \(V_{biology}\)：该歧义组是否跨样本稳定、与表型相关、能否在独立队列验证；
- \(R_{false\ claim}\)：异构体、共洗脱、DDA 缺失和网络先验造成的错误声明风险；
- \(cost(a)\)：购买、合成、仪器时间和方法开发成本。

标准实验完成后，候选图和不确定性必须更新，再选下一动作。一次性排序不构成闭环。

### 4.4 与现有工作的实质区别

- 相对 MASST / StructureMASST：它们回答“在哪里命中”；本方法回答“下一次做哪个实验才能最大限度消除真实队列中的身份歧义”。
- 相对 reverse metabolomics：原作大规模合成一个预定义家族；本方法力图在预算受限时主动选择最有信息量的少数验证动作。
- 相对 SAMBA：SAMBA根据疾病代谢模型预测值得测量的代谢物；本方法根据**已经观测到的谱学歧义图和预期消歧结果**选择实验。
- 相对 DreaMS Atlas：Atlas 是化学谱空间地图；本方法是地图上的实验决策和证据更新。
- 相对 BioAware：生化网络只能调整研究优先级，不能把候选变成已确认身份。

这一区别必须通过实验对照证明，不能靠命名成立。

## 5. 最小决定性验证：先证明问题，再训练模型

### 5.1 数据要求

首个方法学数据集不能用 MTBLS13729 独自承担。应选择同时具有：

- 至少数百个真实样本或多个公开项目；
- 原始 MS1 + MS2、可解析元数据、最好有 QC/blank；
- 一批同平台 authentic standards 或可被隐藏的 Level 1 真值；
- 足够多同式/近异构候选；
- 最好有独立队列或跨仪器复现。

Reverse-metabolomics 原作的合成标准数据、DeepMet 的标准验证集、公开标准混合物/样本对可以作为候选；必须先核查许可、原始数据和真值粒度。

### 5.2 retrospective masked-standard benchmark

将已有同平台标准逐步隐藏，模拟“尚未购买”：

1. 冻结其余公共/本地谱库、队列候选图和所有阈值；
2. 每轮由策略选择一个要揭示的标准或实验；
3. 揭示该标准的 RT/MS/MS 后更新候选图；
4. 记录在相同标准预算下能正确消歧多少身份、多少近异构错误和多少表型相关歧义组。

必须比较：

- random；
- highest abundance；
- largest phenotype effect；
- highest current confidence；
- chemical diversity only；
- SAMBA/biological-prior only（适用时）；
- uncertainty only；
- 本方法的 expected ambiguity reduction per cost。

横轴必须是累计标准/实验成本；纵轴必须至少包含：

- 严格身份恢复数；
- near-isomer 正确消歧数；
- corrected / introduced；
- 能被确认的独立表型关联数；
- 校准误差和 abstention；
- 新分子式、新 scaffold 和跨仪器分层。

### 5.3 必须防止的泄漏

- 被隐藏标准的谱、RT、同实验复现谱不得进入选择器；
- 同一 IK14、分子式和 scaffold 必须分层报告，必要时整组留出；
- 不能把“未被 DDA 触发”当真阴性；
- phenotype 只进入研究价值项，不能进入身份标签；
- 公共数据库出现频率、Rhea degree、库中谱数必须作为单独基线；
- 每个动作的后验更新规则需预注册，不能根据答案事后改变。

### 5.4 第一阶段通过门

只有同时满足以下条件才进入算法训练与 CRC 应用：

1. 在至少一个完整 held-out standard 集上，相同成本下严格消歧数显著高于所有简单基线；
2. formula/scaffold-cluster bootstrap CI 下界大于零；
3. near-isomer 不退化，且 introduced 受控；
4. 选择优势不能被 abundance、library degree、commercial availability 单独解释；
5. 在第二项目或第二仪器上方向复现；
6. 无标准的成员仍只报告 family/transformation-resolved hypothesis，不冒充 Level 1。

## 6. 现阶段 DreaMS 项目的正确接入方式

### 6.1 第一版不等待所有模块成功

第一版用：官方 DreaMS + raw spectral evidence + MS1/RT/adduct/isotope 证据，建立可审计候选图和主动选择基线。这样可以先检验“主动消歧”是否真比随机/显著性/置信度排序更省标准。

### 6.2 新 embedding 的真正价值

噪声微调若成功，不是为了在论文图中多一个分数，而是应使跨仪器/跨 CE 的同一化学实体形成更稳定候选图，同时减少共享高峰导致的异构体混淆。它必须在 masked-standard benchmark 中表现为：

- 更准确的未验证歧义后验；
- 更高的标准选择效率；
- 更少的 introduced；
- 更好的跨平台迁移。

若只提升一个旧检索面板而不提高这些指标，就不是该生物学方法的核心增益。

### 6.3 双重映射与 ChemAware 的真正价值

二者只有在能回答“这个标准实验为什么排除了候选 B、支持候选 A”时才进入：

- embedding 方向可解码到具体碎片/中性丢失；
- 证据能定位到具体峰；
- 目标删峰/保峰能改变该候选分支，而匹配随机操作不能；
- 置信度可校准并跨结构留出复现。

未通过这些门时，规则只可用于生成待验证假设，不能影响身份标签。

### 6.4 BioAware 的真正价值

BioAware 可以提高某个歧义组的生物学研究价值或建议哪条反应分支值得验证；它不得直接修改谱学身份概率。这样既使用生物先验，也避免把数据库流行度当成化学证据。

## 7. 生物学论文如何闭环

方法论文不能以“多注释几个名字”收尾。理想链条是：

1. 在高质量发现队列中找到一个表型相关、但含多个候选身份的化学转化家族；
2. 主动策略用少量标准优先消歧最关键分支；
3. 已测标准以同平台 RT/MS/MS/加标证据提高身份等级；
4. 未测成员只报告为由诊断峰支持的 family/transformation-resolved unknowns；
5. 整个化学程序在独立队列复现；
6. 公开数据/RDD/microbeMASST 仅提供组织、饮食或微生物来源支持；
7. 静态丰度只声称 abundance remodeling，不声称通量；没有功能实验时不写酶活性因果。

MTBLS13729 可在方法通过后作为 CRC 应用。当前最强冻结锚仍是 feature 3222 的 C20:4-acylcarnitine-like 信号，但它只能作为 class-level/Level 2a-supported 入口，不能独自承担新机制。

## 8. 立即执行顺序

1. **停止** reverse-context A0–A2 和任何疾病/样本类型分类扩容。
2. **不再提交**当前 reverse-probe 的新模型；仅保存为普通反向检索基线。
3. 建立候选公开数据表，逐项核验标准真值、原始样本、QC、近异构体和独立队列是否齐备。
4. 先做纯回顾性的 masked-standard learning-curve；不训练大模型，不接 BioAware。
5. 若主动选择相对所有简单基线没有显著优势，立即停止该方法命题，而不是继续加模块。
6. 若通过，再把新 embedding、峰双重映射、BioAware 按独立消融逐个加入。
7. 最后迁移到 CRC/MTBLS13729，形成少量标准最大化生物学结论的应用。

## 9. 当前最诚实的一句话

我们现在还没有一个已经成立的“极牛反向代谢组学算法”。我们有一个经查重后仍可能值得验证的、比“反向搜库”更深的问题：**在真实队列的谱学歧义图上，主动选择最少的验证实验，最大化可被严格确认的化学—生物学知识。** 下一步是用隐藏标准的回顾性实验把它证明或判死，而不是再先写一个模型。
