# BioAware 负离子网络专家：协议修复、开发结果与确认边界（2026-09-01）

> **已被 v2 化学完整性报告取代。** 本文使用的 v1 MoNA 候选库把所有负离子记录都视为 `[M-H]-`；后续审计发现 36,663 条记录中有 8,551 条不满足结构理论 `[M-H]-` 质量或结构—InChIKey 一致性。本文保留为开发历史，不得再引用其中 595-query 性能作为正式结果。当前有效结果见 `docs/BIOAWARE_NEGATIVE_NETWORK_EXPERT_V2_CHEMICAL_INTEGRITY_20260901.md`。

## 1. 当前结论

本轮得到了一条**值得进入独立外部确认**的方法学结果，但尚不能声称 SOTA：

> 在 8 个 MetDNA3 外部负离子单元、595 个可评价 `[M-H]-` 查询上，冻结官方 DreaMS 的 Recall@1 为 72.44%。候选特异网络专家在 identity-purged LOSO 中提高 3.36 pp（21 修正、1 新增，formula-cluster 95% CI 1.36–5.59 pp）；在更严格的 leave-biological-source-out 且同时净化测试身份和分子式后提高 3.03 pp（24 修正、6 新增，CI 0.77–5.67 pp）。候选内联合网络特征置换的平均变化为 −0.04 pp，真实结果的经验单侧 p=0.0099。

这是**已打开外部队列上的开发与压力测试证据**，不是新盲测。必须在一个从未用于开发、具有原始 MS1/MS2、Level-1 真值和 `[M-H]-` 候选覆盖的队列上冻结验证，才能形成确认性或 SOTA 声明。

## 2. 首先修复的协议错误

早期“16 panel 外部评价”实际只评价了 8 个正离子 panel。原因是原 MassSpecGym 参考 HDF5 仅支持 `[M+H]+`、`[M+Na]+`，负离子查询被静默排除。修复后：

- 正离子旧协议只有 8 个实际 panel；
- 新建独立的 MONA negative 参考库与官方 DreaMS 缓存复核；
- 仅把 `[M-H]-` 纳入当前正式范围，因为 MONA negative MGF 没有逐谱 adduct 元数据；
- 重新编码 15 个确定性抽样谱，缓存与当前冻结官方 checkpoint 的 cosine 最低 0.999999821，证明负离子 embedding 缓存一致；
- 任何跨任务 baseline 数值不得混用，所有比较必须使用同一 query/candidate/tie 协议。

协议完整性报告：`data/validation/bioaware_external_protocol_integrity_20260901/report.json`。

## 3. 官方 DreaMS 负离子基线

严格候选协议：外部 Level-1 `[M-H]-` 查询；MONA negative 中 10 ppm 候选；每个 IK14 取最大 cosine；至少 2 个唯一 IK14；并列按失败处理。

| 指标 | 数值 |
|---|---:|
| 外部 `[M-H]-` Level-1 谱 | 1,646 |
| 真值存在于 MONA negative | 1,075 |
| 最终可评价查询 | 595 |
| 唯一真值身份 | 177 |
| 唯一真值分子式 | 149 |
| 候选对 | 2,229 |
| 官方 DreaMS Recall@1 | 0.72437 |
| 官方错误 | 164 |

八个单元全部有覆盖，单元基线 Recall@1 约 0.67–0.88。基线报告：`data/validation/bioaware_metdna3_external_negative_dreams_v1/report.json`。

## 4. 方法是什么、与 MetDNA/KGMN 有何区别

本方法仍然是 embedding 后的候选专家，不改变 DreaMS embedding。它不是简单沿网络传播后加固定分数，而是构造并学习候选特异证据：

1. **DreaMS 候选组**：官方 cosine 提供冻结候选集合和低置信度边界；
2. **已知反应拓扑**：候选是否是质量候选、是否连接到身份隔离的高置信种子、最短深度、种子支持数和节点度；
3. **原始 MS2 边验证**：step-0 已知边的完整性与 bottleneck，及 step-1 预测边的增量证据；
4. **组内 pairwise 学习**：训练候选相对次序，而不是把路径存在性直接当身份；
5. **安全门控**：仅在 DreaMS Top1–Top2 gap ≤0.05、网络提案概率 ≥0.75、且提案具有原始 step-0 MS2 边验证时改排；否则回退 DreaMS；
6. **可追溯输出**：每个查询保留 DreaMS Top1、网络 Top1、最终 Top1、干预概率、边验证状态和回退原因。

方法学创新应表述为：

> 用身份隔离的样本内反应上下文生成候选特异拓扑证据，再用原始 MS2 边验证和风险门控学习何时覆盖 DreaMS 的困难候选；网络证据不足、冲突或 DreaMS 已高置信时严格回退。

不能把它写成“首次利用代谢网络注释”，也不能说已经完成生物上下文 embedding 微调。

## 5. 消融结果：提升到底来自哪里

所有消融使用相同的 identity-purged 八单元 LOSO、相同候选图、相同阈值。

| 模型 | ΔRecall@1 | 修正/新增 | 风险净收益 `C-2I` | 解释 |
|---|---:|---:|---:|---|
| 仅 DreaMS spectral score 学习 | 0.00 pp | 0/0 | 0 | 单调校准不能改变候选顺序 |
| 仅 mass-membership，无边门控 | +1.85 pp | 12/1 | 10 | 反应库收录本身是强先验 |
| 已知拓扑，去掉 mass-membership | +1.85 pp | 15/4 | 7 | 并非只有数据库收录效应 |
| 仅原始 step-0 边 | +1.18 pp | 14/7 | 0 | 有信号，但单独使用风险过高 |
| 光谱 + 原始边 | +0.17 pp | 1/0 | 1 | 原始边不能单独修复 DreaMS |
| 光谱 + 已知拓扑 | +2.02 pp | 12/0 | 12 | 稳健但保守 |
| 完整光谱+网络 | +2.02 pp | 12/0 | 12 | DreaMS 权重压制部分网络纠错 |
| **网络-only + 原始边门控** | **+3.36 pp** | **21/1** | **19** | 当前开发集最佳配方 |

关键结论：增益不是普通的 DreaMS cosine 校准；网络-only 显著优于完整融合，说明在 DreaMS 已出错的低置信候选组中，继续把 spectral score 强行放入提案模型会降低纠错能力。DreaMS 仍负责候选生成、低置信门和最终回退安全，而非进入网络提案分数。

消融报告：`data/validation/bioaware_metdna3_external_negative_loso_ablation_v2/report.json`。

## 6. 泛化与反事实审计

### 6.1 身份和分子式净化

- 所有测试真值 IK14 均从训练中删除：+3.36 pp，21/1；
- 测试真值分子式也从训练中删除：仍为 +3.36 pp，22/2；
- 四折 leave-biological-source-out（BV2cell、脑、肝、血浆整体留出）且身份净化：+2.86 pp，22/5；
- leave-source-out 再加分子式净化：+3.03 pp，24/6，四个来源折均为正。

来源净化报告：`data/validation/bioaware_metdna3_external_negative_source_loso_v1/report.json`。

### 6.2 候选特异网络置换

在每个 query 内联合置换完整网络特征块，保留候选数、特征边际分布和块内协方差，但破坏候选—网络对应：

| 指标 | 真实网络 | 100 次置换均值 | 置换 95% 上界 | 经验 p |
|---|---:|---:|---:|---:|
| ΔRecall@1 | +3.03 pp | −0.04 pp | +0.34 pp | 0.0099 |
| `C-2I` | 12 | −1.31 | 1.05 | 0.0099 |
| 修正数 | 24 | 0.87 | 3 | 0.0099 |

这排除了“任何同分布网络数字都能产生提升”的解释。置换报告：`data/validation/bioaware_metdna3_external_negative_candidate_permutation_v1/report.json`。

### 6.3 尚未解决的 component-purge

删除测试真值所在整个 step-0 反应连通分量后，结果为 1 修正/1 新增、净 0。但每折训练只剩约 69–104 个查询和 25–39 个身份，训练集严重坍缩。因此该结果既不能证明邻域泄漏，也不能证明跨反应分量泛化；需要更大的独立训练集或多个相互独立的反应网络分量后再回答。

## 7. 当前冻结工件和推理接口

- 冻结工件：`data/validation/bioaware_metdna3_negative_network_expert_v1/artifact.json`
- 工件 SHA256：`e05890f570d2db7b7de32a189cf21bb78c42c67e29c9b7e0b46c53ce3b8c1e1c`
- 推理模块：`annotation/bioaware_negative_expert.py`
- 推理安全策略：拒绝 truth、phenotype、case/control 等列；并列回退；高置信 DreaMS 回退；缺原始边验证回退；输出明确 abstention reason。
- 相关测试：15 个新协议/模型测试全部通过，另有 4 个部署接口安全测试通过。

冻结模型用全部已打开的 595 查询拟合，只能用于**下一个新外部队列**。它在训练集上的重代入结果仅是工程检查，不是性能结果。

## 8. 与 MTBLS13729 生物学应用的关系

当前 MTBLS13729 BioAware v1 的 21-query 一步 Rhea 试验为 0 修正/1 新增，已明确失败。不能把本轮 MetDNA3 negative 开发结果倒灌成 MTBLS13729 已提升的结论。

新专家若用于 MTBLS13729，必须满足：

1. 对 `[M-H]-` MS2 查询形成与冻结协议一致的候选组；
2. 使用同一样本/同 panel 的高置信种子构造身份隔离网络特征；
3. 原始 MS2 可验证 step-0 反应边；
4. 不使用 Rmu/RN/Rtu、差异丰度、q-value 或疾病标签选择候选；
5. 先做注释准确性验证，再做丰度和生物学故事；
6. 不把静态丰度差异写成通量或酶活改变。

## 9. 下一步固定决策

1. **不再调当前阈值或特征。** 当前配方和工件已冻结；
2. **寻找新确认队列。** 最低要求：原始 MS1/MS2、Level-1 身份、`[M-H]-`、可形成不少于 100 个多候选查询；
3. **NetID mouse-liver-negative 不能直接作确认。** 本地有约 1,230 张 MS2，但人工标准库只有少量代谢物；把 NetID 自己传播出的 8,191 个标签当真值会自证循环；
4. **确认端点预注册。** Primary：Recall@1；Secondary：MRR、修正/新增、`C-2I`、formula-cluster CI；必须报告覆盖率和弃权率；
5. **确认通过后再做上下文 embedding。** 反应邻居是不同分子，不能无条件拉近。可学习的上下文 adapter 必须以候选组内排序为目标、带 clean-embedding preservation，并在无网络证据时严格退化为原 DreaMS；它与当前后验专家是下一阶段，不是当前已完成结果。

## 10. 可发表性边界

当前证据足以支撑“方法候选和开发结果”，还不够支撑“全面超越 DreaMS”或“SOTA”：

- 正离子外部结果尚未稳定改善；
- 负离子结果只覆盖 `[M-H]-`；
- 8 个 MetDNA3 单元已参与模型发现；
- component-purged 泛化尚未证明；
- 尚无新独立外部队列。

只有冻结模型在新队列上显著优于同协议官方 DreaMS，并且对强基线（例如传统谱图相似度、MetDNA/KGMN 风格传播、mass-membership prior）均保持增量，才可讨论 SOTA。
