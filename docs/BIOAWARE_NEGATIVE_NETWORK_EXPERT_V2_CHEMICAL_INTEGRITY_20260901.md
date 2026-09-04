# BioAware 负离子网络专家 v2：化学完整性修复与严格开发证据（2026-09-01）

## 1. 当前最可靠结论

BioAware v2 是冻结官方 DreaMS embedding 之后的候选排序专家，不改变 embedding。它在**已经打开的四个生物来源**上采用 leave-biological-source-out，同时从训练中删除测试真值身份和分子式：

| 模型 | 查询 | DreaMS R@1 | 最终 R@1 | 增益 | 修正/新增 | formula-cluster 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| `full_bioaware`（主模型） | 548 | 0.70255 | 0.73723 | **+3.47 pp** | **19/0** | **+1.11 至 +6.29 pp** |
| `full_no_edge_gate`（高召回消融） | 548 | 0.70255 | 0.74270 | +4.01 pp | 23/1 | +1.51 至 +6.94 pp |

主模型四个来源的增益均为正，并且没有新增 Top-1 错误。候选内联合打乱全部网络特征、保持 DreaMS 分数和特征块协方差不变后，主模型 null 平均增益仅 +0.044 pp，95% 上界 +0.365 pp；观察值 +3.47 pp 的经验单侧 `p=0.0099`。

这支持“候选特异网络证据具有增量价值”，但仍是**打开队列上的开发与压力测试**，不能称为独立外部验证或 SOTA。

## 2. 为什么必须推翻 v1 数字

v1 把 MoNA negative 的所有记录都按 `[M-H]-` 使用。结构—质量审计显示：

- 总记录 36,663；
- SMILES 可解析 36,608；
- SMILES 计算 IK14 与记录 InChIKey 一致 36,519；
- 同时满足结构一致和理论 `[M-H]-` 质量 10 ppm 的只有 **28,112（76.68%）**；
- **8,551 条被拒绝**，包括源内脱水离子、其它负离子形式、质量元数据错配和少量结构标识错误。

一个具体污染例是：核糖结构的理论 `[M-H]-` 约 149.045，而部分 MoNA 记录前体为 131.034；旧协议会把它作为质量 131 的 `[M-H]-` 候选，与戊二酸竞争。这不是模型应学习的化学规律。

因此，595-query v1 结果只保留为历史，当前正式开发基准必须使用：

`data/validation/mona_negative_library_chemical_integrity_v1/approved_m_h_library_rows.npy`

审计报告：`data/validation/mona_negative_library_chemical_integrity_v1/report.json`。

## 3. 干净的官方 DreaMS 基准

固定协议：外部 Level-1 `[M-H]-` 查询；经结构验证的 MoNA `[M-H]-` 参考；feature m/z 10 ppm；每个 IK14 取最大官方 DreaMS cosine；至少两个唯一候选；并列算错。

| 指标 | 数值 |
|---|---:|
| 可评价查询 | 548 |
| 真值身份 | 164 |
| 真值分子式 | 136 |
| 候选分子 | 465 |
| 候选对 | 2,003 |
| 官方 DreaMS Recall@1 | 0.70255 |
| 官方错误 | 163 |

基准报告：`data/validation/bioaware_metdna3_external_negative_dreams_v2_chemically_filtered/report.json`。

该数值只能与同一 548-query 候选协议比较，不能与任何其它 DreaMS 基线拼接。

## 4. 方法定义

主模型的候选特征包括：

1. 冻结官方 DreaMS cosine；
2. 候选质量是否被反应网络覆盖；
3. 身份隔离的种子到候选的路径覆盖、逆深度、种子支持和节点度；
4. 原始 MS2 step-0 边完整性和 bottleneck；
5. step-1 预测边相对 step-0 的增量。

训练是查询组内 pairwise logistic ranking。部署只在以下条件同时满足时改排：

- DreaMS Top1–Top2 gap `<=0.05`；
- 网络提案相对当前 DreaMS Top1 的概率 `>=0.75`；
- 提案具有非零原始 step-0 完整边和 bottleneck；
- DreaMS、网络提案均无并列。

否则严格回退到 DreaMS。推理不使用真值、疾病标签或 P2b。

## 5. 消融：3–4 pp 来自哪里

在相同 548-query、identity-purged 8-unit LOSO 下：

| 配方 | ΔR@1 | 修正/新增 | 解释 |
|---|---:|---:|---|
| spectral-only | 0.00 pp | 0/0 | 普通 DreaMS 单调再校准不能改序 |
| mass-membership only | +2.19 pp | 13/1 | 网络收录先验本身有信号 |
| known topology | +2.74 pp | 18/3 | 路径拓扑增加纠错，也有风险 |
| raw step-0 only | +2.01 pp | 16/5 | 原始边单独使用不安全 |
| network-only | +3.47 pp | 22/3 | 网络块是主要增量来源 |
| spectral + known topology | +2.92 pp | 16/0 | 稳健融合 |
| `full_bioaware` | **+3.10 pp** | **17/0** | 8-unit 保守模型 |
| `full_no_edge_gate` | +4.01 pp | 24/2 | 更高召回、引入错误 |

在更严格的整生物来源与分子式净化下，`full_bioaware` 为 +3.47 pp、19/0；因此主模型不是仅靠同来源另一色谱模式复现。

消融报告：`data/validation/bioaware_metdna3_external_negative_loso_ablation_v3_chemically_filtered/report.json`。

## 6. 新增错误与安全门的作用

高召回配方唯一新增错误的提案具有强拓扑覆盖，但 step-0 bottleneck 为 0；它仅靠 step-1 极小增量进入。主模型的 raw-step0 硬门拦住该提案，代价是少修正 4 个错误，收益是 19/0 的高精度干预。

这说明安全门不是装饰：它把“网络可达”与“已有原始 MS2 支持”分开。不能用单个错误反向设置更细阈值；当前固定硬门来自预注册规则，尚需新队列验证。

机制审计：

- `data/validation/bioaware_metdna3_negative_transition_mechanisms_v2b_full_bioaware/`
- `data/validation/bioaware_metdna3_negative_transition_mechanisms_v2_full_no_edge_gate/`

## 7. 冻结工件

主部署工件：

`data/validation/bioaware_metdna3_negative_network_expert_v2_chemically_filtered/artifact.json`

- SHA256：`a04f9a7d02f726702f1c03ec4bac2e9ac2e471a3f722422165555774e7944c74`；
- 配方：`full_bioaware`；
- 普通数组 scaler/coef，不依赖 sklearn pickle；
- 化学完整性报告、source-LOSO 和候选置换报告哈希均写入工件；
- 推理实现：`annotation/bioaware_negative_expert.py`；
- 全数据重拟合输出 20/0 仅是工件执行检查，不是性能证据。

## 8. 创新性和论文边界

不能声称“首次使用代谢网络”。MetDNA/KGMN 已使用种子传播、反应网络、谱图网络和峰相关网络。本工作的可辩护增量是：

> 在冻结 DreaMS 候选空间中，以身份隔离的样本内反应路径构造候选特异证据，再以原始 MS2 边验证和显式回退门学习是否覆盖低置信 DreaMS Top1；同时对参考库离子形式做结构—质量化学完整性约束。

当前还不能声称：

- SOTA 或独立外部泛化；
- 正离子模式改善；
- 改变或改善 DreaMS embedding；
- 生物网络证明代谢通量、酶活或疾病机制。

## 9. 下一步唯一高价值门

冻结 v2 后，不再在这四个来源上调特征、C、gap 或概率阈值。下一步必须寻找或构建一个从未打开的、带 Level-1 真值和原始样本级 MS1/MS2 的 `[M-H]-` 队列，原样加载工件一次性验证：

1. 同协议官方 DreaMS 基线；
2. 冻结 BioAware v2；
3. corrected/introduced、formula-cluster CI、候选置换/度保持 decoy；
4. 数据库覆盖、候选数、来源与仪器分层。

只有该门通过，才能讨论确认性性能或与 SOTA 的正式比较。
