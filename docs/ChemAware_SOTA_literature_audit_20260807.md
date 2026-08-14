# ChemAware 架构的最新文献、创新性与 SOTA 可行性审查

日期：2026-08-07

## 1. 审查结论

当前项目值得继续，但不能把“化学规则库 + DreaMS 微调”本身作为主要创新，也不能在实验前承诺通用分子检索 SOTA。

最可辩护的科学假设是：

> 机制性裂解规则可以作为谱峰与候选分子子结构之间的中间锚点。它们在不破坏 DreaMS 原有表示的条件下，改善质量接近候选中的局部结构排序，并产生能够通过删峰和随机化检验的化学解释。

对应的论文目标应为：MassSpecGym v1.5 上具有竞争力的候选检索；困难异构体或规则冲突子集上获得稳定增益；峰级解释在干预试验中显著优于随机解释；在独立数据来源上保持性能和校准。只有在相同 split、相同候选池和无泄漏条件下显著超过最强修正基线，才宣称检索 SOTA。

## 2. 最新方法对创新边界的约束

| 已有方向 | 代表工作 | 已经覆盖的内容 | 对本项目的约束 |
|---|---|---|---|
| 质谱基础模型 | DreaMS | 大规模自监督预训练、1024 维谱图表示、峰 token、任务微调 | 继续使用官方 embedding checkpoint 合理；重新训练基础模型没有必要 |
| 全局谱图—分子对齐 | JESTR、MSAlign、SpecBridge、CMSSP | 对比学习、冻结基础模型、轻量投影、候选排序 | 普通 triplet、InfoNCE 或冻结 DreaMS 接分子编码器不构成核心创新 |
| 局部峰—结构对齐 | FLARE | 弱监督峰—原子双向细粒度对齐和可视化 | 仅加入局部 cross-attention 或展示热图不足以形成新意 |
| 领域知识与亚结构预测 | MIST、SIRIUS/CSI:FingerID、MIST-CF | 中性丢失、分子式、指纹/亚结构概率和候选排序 | 单独的规则解码头、指纹头或“化学可解释概率”已有充分先例 |
| 碎裂主题和子结构模式 | MS2LDA / MS2LDA 2.0 | 碎片与中性丢失共现模式、Mass2Motif、自动注释辅助 | 规则库或碎裂 motif 的建立具有工程价值，不能单独支撑算法创新 |
| 谱图对解释 | TransExION | 查询谱和参考谱的峰/中性丢失对齐以及 attention 可视化 | attention 可视化只能作为对照，不能直接称为忠实解释 |
| 生成与正向模拟 | GLACIER、MADGEN、MS-GPT、GLMR | 分子到谱图模拟、生成式检索或 de novo 结构生成 | 当前谱图 embedding 项目不宜竞争完整 de novo 或正向模拟 SOTA |

## 3. 可以保留和需要重写的创新点

### 可以作为核心创新

1. **规则锚定的三方局部对齐**：把可观测峰或中性丢失、化学规则、候选分子子结构放入同一局部对齐目标，而非只从全局 embedding 解码规则。
2. **支持—冲突—不可观测三状态标签**：缺少碎片峰不等同于结构不存在；加合物、碰撞能和仪器条件不足时，应标记为不可观测。
3. **规则冲突驱动的困难负样本**：规则相似但 MCES 远、或结构相近但规则冲突的样本用于 hard mining，直接针对 Task 0 暴露的粒度失配。
4. **因果忠实解释**：关键峰删除、随机等量峰删除、规则和参数随机化共同验证解释是否真正影响候选排序。
5. **经过审计的多轴评估**：同时报告候选检索、MCES 局部几何、规则 AUPRC、解释干预效应、校准和跨来源稳健性。

### 不能作为主要创新

- 建立 335 条机制规则或扩展 MassBank 经验规则；
- 对 DreaMS 使用 triplet loss、InfoNCE、MSE 或 preservation loss；
- 使用 MCES 作为距离或排序监督；
- 从 embedding 预测规则、官能团或指纹；
- 使用 attention 热图解释模型；
- 简单融合 DreaMS、规则引擎、MS2DeepScore 或 TransExION 的分数。

## 4. SOTA 可行性判断

### 通用候选检索 SOTA

现阶段不应承诺。FLARE 已报告细粒度对齐的强结果，MSAlign 和 SpecBridge 已展示冻结基础模型的高性价比对齐，GLACIER 等正向模型也形成了强检索路线。更重要的是，2026 年 MassSpecGym 审计指出早期论文中存在预训练重叠、候选捷径和指标实现差异；历史最高数字需要在 v1.5 下重新验证。

### 困难异构体与解释 SOTA

具备可能性。现有工作通常把解释呈现为 attention 或局部相似度，可验证的机制规则锚点和系统性干预评价仍有空间。该目标需要建立同一协议下的 FLARE/TransExION 解释基线，并证明规则锚点提高局部排序或解释忠实性，而非只提高规则重构准确率。

### 最现实的论文定位

“竞争性检索 + 机制规则锚定 + 因果忠实解释 + 防泄漏评估”的 Pareto 前沿。检索没有全面第一仍可形成完整论文；若同协议 Rank@1/MCES@1 显著超过最强修正基线，再追加检索 SOTA 声明。

## 5. 可行工程架构

### 模块一：ChemAware embedding

- 谱图侧：官方 DreaMS embedding checkpoint，输出 `H_peak` 和 `z_spec`。
- 分子侧：冻结 ChemBERTa 或轻量 GNN，输出 `H_atom` 和 `z_mol`。
- 参数更新：第一阶段只训练投影层和 adapter；通过局部对齐门控后，再解冻 DreaMS 顶部 1–2 层。
- 学习目标：全局候选对齐、峰—原子局部对齐、峰/质量差—规则—子结构锚定、MCES 局部排序、原始表示保持。
- 输出：`z_chem`、谱图—候选分数和局部对齐矩阵。

### 模块二：峰级化学解释

- 提取峰、质量差、规则和候选子结构之间的证据边；
- 对每条候选规则输出支持、冲突或不可观测；
- 通过关键峰删除、随机峰删除和随机化审查计算 comprehensiveness、sufficiency 与分数下降；
- 输出支持峰、规则来源、冲突项和解释可信度。

## 6. 数据与评估要求

1. 主评估采用 MassSpecGym v1.5；质量候选轨和分子式候选轨分开报告。
2. 按 IK14 连接层匹配；统一 canonical SMILES、候选池、ECFP4 和 MCES 参数。
3. 发布 DreaMS、分子编码器和所有辅助模型的预训练重叠清单。
4. 必做空谱、零强度、谱峰乱序、候选乱序和 spectrum-blind 控制。
5. 报告 Rank@1/5/20、MCES@1、困难子集指标、rule macro-AUPRC、删峰效应、校准和 paired bootstrap 置信区间。
6. 使用 Spectraverse 或来源留出作为独立外测；NIST20 仅在许可和 DreaMS 原协议可严格复刻时使用。

## 7. 六道实验门控

- **G0 评估器可信**：v1.5 split、IK14、候选池和控制试验全部通过。
- **G1 复现强基线**：冻结 DreaMS + 冻结分子编码器 + 轻量投影达到合理水平。
- **G2 局部对齐有效**：peak—atom 分支在困难检索上产生显著增益。
- **G3 规则锚点有效**：去掉规则锚点后 MCES@1 或近异构体性能显著下降。
- **G4 解释忠实**：关键峰删除效应显著高于随机峰，且随机化审查通过。
- **G5 跨来源成立**：独立来源性能和校准不发生不可接受的崩溃。

停止条件包括：强基线无法复现；规则增益只由规则频率或分子类别捷径解释；谱图置乱后性能不下降；外部来源性能崩溃。

## 8. 主要参考文献

- DreaMS: https://www.nature.com/articles/s41587-025-02663-3
- MassSpecGym: https://proceedings.neurips.cc/paper_files/paper/2024/hash/c6c31413d5c53b7d1c343c1498734b0f-Abstract-Datasets_and_Benchmarks_Track.html
- MassSpecGym in the Wild / v1.5 audit: https://openreview.net/pdf?id=I1PUDXXYot
- FLARE: https://pmc.ncbi.nlm.nih.gov/articles/PMC12873900/
- JESTR: https://pmc.ncbi.nlm.nih.gov/articles/PMC11601792/
- MSAlign: https://arxiv.org/abs/2605.19752
- SpecBridge: https://arxiv.org/abs/2601.17204
- MIST: https://www.nature.com/articles/s42256-023-00708-3
- SIRIUS methods: https://v6.docs.sirius-ms.io/methods-background/
- TransExION: https://pmc.ncbi.nlm.nih.gov/articles/PMC11134763/
- MS2LDA 2.0: https://pubmed.ncbi.nlm.nih.gov/42409848/
- Attention is not Explanation: https://aclanthology.org/N19-1357/
- Sanity Checks for Saliency Maps: https://papers.neurips.cc/paper_files/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html
- Faithfulness evaluation framework: https://proceedings.mlr.press/v162/dasgupta22a.html
