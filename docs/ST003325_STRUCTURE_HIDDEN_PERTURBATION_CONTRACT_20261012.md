# ST003325 结构隐藏扰动复现合同

日期：2026-10-12  
状态：在读取扰动效应结果前冻结；数据集为 development/consumed，不是独立外测

## 1. 科学问题

本实验不建设 Atlas，也不以公共数据库命中作为成功条件。它只问一个更基础、可证伪的问题：

> 不使用结构名称、谱库身份或通路名单时，独立 STS 编辑 clone 中的 LC-MS 响应能否跨三种编辑机制复现；冻结的 MS/MS 化学关系能否在完整 MS1 响应之上改善这种复现？

ST003325 只承担事件级遗传扰动开发正控。它不能证明跨研究泛化，也不能证明公共数据层有效。

## 2. 冻结分母与设计边界

- 数据：Metabolomics Workbench ST003325 / PR001844；`development/consumed`；`sealed=false`。
- 两个具有 pooled-QC DDA 的面板：AN005444 HILIC negative、AN005445 RP positive。
- 每面板 72 个研究进样：CE、KO、PTC、WT 各 3 个 clone；每 clone 6 个 plate。
- plate 1--3 全部为 Basal，plate 4--6 全部为 Basal+DHEAS。条件与 plate 编号完全绑定。
- 因此不得把每面板 72 次进样当 72 个独立生物重复，也不得声称已从一般 plate-set 效应中无条件识别纯 DHEA-S 因果效应。
- 表型盲 QC 后有 3,080 个 measurement-qualified features；其中 804 个 feature--scan links 在 3 个 pooled-QC DDA 文件中至少复现 2 次。804 个链接只对应 760 个唯一 MS/MS evidence units；任何谱学方法按 760 个单元计票，不能把 44 个重复映射加倍。

## 3. 冻结响应对象

对每个 clone `c` 和 feature `i`：

主分析先在每个面板内对 QC 合格 features 的 `log1p` intensity 做逐进样 median centering（完全不读取 GROUP），再计算：

`D_ci = median(y_ci, plates 4--6) - median(y_ci, plates 1--3)`。

未做逐进样 centering 的 `log1p` 结果作为固定敏感性分析；二者方向不一致时主结论失败。

这称为 `design-labelled plate-set response`。在本数据中它与 DHEA-S 标签相同，但由于 plate 绑定，正式文字不得省略该限制。

所有预处理、缺失处理和 feature universe 在使用编辑标签前固定。不得按已知 steroid、STS pathway、library hit 或单 feature p 值预筛选。

## 4. 结构隐藏的跨编辑机制任务

外层依次留出 CE、KO、PTC 中一个完整 edit class。对每个外层 fold，再依次留出一个 WT clone：

1. 训练集只含另外两类编辑的 6 个 clone 和剩余 2 个 WT clone；
2. 用训练 clone 的 `D` 学习带方向的 response signature；
3. 测试集为留出 edit class 的 3 个 clone和留出的 1 个 WT clone；
4. 主原子比较是每个留出 edit clone 的 signature score 是否高于留出 WT，合计 27 个 mutant--WT 比较；
5. 同时报告三个 edit-class cluster 的结果，不能把 27 个比较当 27 个完全独立重复。

方向、feature 权重和任何降维只可由训练 fold 决定。立即运行的 MS1 基线不选择 top-K：训练 signature 是“另外两种编辑机制的 clone response 中位数”减去“两个训练 WT clone response 中位数”，随后做 L2 归一化；测试 clone 的 score 为其相对训练 WT centroid 的 response 与该 signature 的 cosine。两个面板的 combined score 是面板 cosine 的等权平均，避免按 feature 数量让 RP positive 自动占更高权重。不得查看留出 edit class 后调 K、阈值或谱学平滑强度。

## 5. 冻结比较世界

按以下顺序比较；只有前三个世界可立即运行，后续世界等待算法 coder 提供冻结谱学输出：

1. `MS1-all`：全部 3,080 个 measurement-qualified features；这是必须击败的结构非依赖强基线。
2. `MS1-MS2-observed`：804 个 replicated-MS2-linked features，但不使用碎裂关系；控制 DDA 选择与 feature 子集效应。
3. `raw-cosine / m/z-RT / shuffled-MS2`：控制普通谱图相似、质量邻近与任意图平滑。
4. `WSE`、official DreaMS、Noise V1：同一 760-unit query graph、同一 fold、同一响应估计。
5. 冻结 P2b：只作为一个局部谱学证据族，不与其原始通道重复计票；near-core 允许弃权。
6. ChemAware：Stage-1、Phase-A、V2 保持资产身份；仅按各自资格作开发消融，不默认生产增益。
7. `public-reference evidence`：可选增量世界。只有排除 ST003325/PR001844 及 raw hashes、并超过研究内谱学和 matched-shuffle 时才有资格保留；失败不否定主方法。

谱学方法的主增量为其相对 `max(MS1-all, MS1-MS2-observed, raw-cosine, shuffled-MS2, WSE)` 的差值，而不是相对随机的差值。

## 6. 终点与通过门

必须同时报告：

- 27 个 mutant--WT 原子比较的正确数与准确率，机会水平 50%；
- 每个留出 edit class 的正确数，及三类最差值；
- signature score margin，而不只报告正负；
- clone/edit-class cluster bootstrap 或完整小样本置换区间；
- feature/spectral-unit coverage、弃权、重复谱学单元和不同面板的分解；
- label permutation 与保持 plate、panel、强度、缺失和 MS2 获取倾向的 matched null。

进入谱学增量阶段的最低门：`MS1-all` 在 CE、KO、PTC 三个留出类均高于 50% 且总体 permutation p < 0.05。否则 ST003325 不能充当事件级正控，应以负结果封存。

谱学层通过门：相对所有适用强基线的 27-comparison 准确率增量严格为正，edit-class cluster 区间下界大于 0，三个留出 edit class 同号，且错误/弃权没有恶化。由于只有三个 edit-class clusters，区间分辨率有限；即使通过也只允许称 development evidence。

## 7. 禁止主张

本实验不能单独支持：DHEA-S 的无混杂因果效应、精确分子身份、STS 通量或酶活、跨研究迁移、疾病机制、公共 Atlas 增量或通用方法已经成立。其唯一晋级作用，是证明一个结构隐藏的事件级响应对象在独立编辑机制之间确实存在，并为冻结算法提供真实生物学压力测试。

