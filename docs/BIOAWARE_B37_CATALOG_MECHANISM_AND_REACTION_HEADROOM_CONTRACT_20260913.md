# BioAware B37：目录增益机制与反应证据剩余空间审计

## 目的

B36 已经证明候选级 BioAware 体系在六个已开放开发域上可以提高 Top-1，但其增益主要来自候选的目录属性，而不是已经建立的反应关系。B37 不训练共享 embedding，也不追求更高点估计；它回答两个进入下一阶段前不可回避的问题：

1. B36 的约 +5.6 个百分点究竟来自 Rhea 网络节点属性，还是来自谱库/数据库可观测性？
2. 在目录校准器已经纠错之后，当前反应证据是否还含有至少 3 个百分点的可辨识剩余空间？

## 不可混淆的三类证据

- 谱图证据：`spectral_score`，来自 DreaMS 当前候选相似度。
- 网络节点/目录证据：`network_member`、`known_log_degree`、质量窗口中的已知候选覆盖率、参考谱数量。这些是候选和数据库的属性，不是反应边证据。
- 反应关系证据：已知路径比例、路径深度、种子支持、一步/两步边完整度与瓶颈等。这些才对应“同一样本的反应网络能否消解候选”。

## 固定协议

- 沿用 B36 的 860-query、六域、候选图和阴性模式任务。
- 外层整域留出；训练时删除外层 truth identity 和 formula。
- 内层留域预测选择干预门；外层结果不参与训练或门控选择。
- 每个消融臂使用相同 HGB 容量、相同随机种子调度、相同门控网格。
- `catalog_full` 必须与正式 B36 的 `spectral_plus_catalog` 在 860 个 query 上逐条一致；不是只对 aggregate 数字。

## 目录机制拆解

固定报告以下臂：谱图单独、四个目录特征单独、拓扑二元组、可观测性二元组、全部目录特征，以及四个 leave-one-feature-out 臂。任何低维臂接近或超过 full catalog，说明此前的“BioAware”增益可以由更窄的目录校准机制解释。

## 反应证据可辨识性

每个 query、每个反应特征报告：

- 是否非零、是否在候选组内变化；
- truth 是否严格超过所有错误候选；
- 错误候选是否严格超过 truth；
- 对 DreaMS baseline error 和 catalog residual error 的可恢复数量；
- 对 baseline-correct 和 catalog-correct query 的潜在伤害数量；
- 与四个目录特征的 query-centered Spearman 相关性。

“任一反应特征可恢复”使用答案选择特征，只是上限，不是可部署模型结果。若 catalog residual oracle headroom 小于 3pp，当前图和当前标量聚合不足以支撑反应上下文 embedding；若大于等于 3pp 且证据在多数 query 内有变化，才进入原始路径/候选上下文模型，而不是把目录属性硬蒸馏进干净谱图 encoder。

进入原始路径诊断不等于进入共享 embedding。后者还必须满足：每个来源均有至少 3pp 剩余上限、每个来源多数 query 的反应证据在候选组内可辨识，并且 B36 的反应特异性门已经通过。任一项失败，`pass_to_shared_embedding=false`。

## 为什么现有提升没有进入共享 embedding

目录属性依赖候选、数据库和检索环境，同一张原始谱图在不同参考库中会得到不同目录证据；共享 clean-spectrum encoder 在只看到谱图时无法忠实恢复这些变量。强行注入会把数据库熟悉度当成分子固有谱学属性。真正的 BioAware 表示学习需要在推理时提供样本与候选上下文，或者先证明反应证据对应稳定、谱图可见且跨库可迁移的方向。

## 声明边界

B37 是已开放开发域的机制审计。它既不是独立盲测，也不是 SOTA、代谢机制、共享 embedding 改进或生物因果证据。

## 2026-09-13 本地完整复放结果（正式服务器运行前）

本地对 860 个 query 完成全部 12 个消融臂，并与 B36 的 catalog 臂逐 query 复放：所有候选、干预、Top-1 转换均一致，最大概率误差为 `5.55e-17`。

| 证据臂 | Recall@1 增量 | corrected / introduced | risk net, lambda=2 |
|---|---:|---:|---:|
| network member | +4.77pp | 42 / 1 | 40 |
| degree | +3.95pp | 38 / 4 | 30 |
| topology: member + degree | **+5.93pp** | **55 / 4** | **47** |
| observability: mass coverage + reference density | +0.47pp | 4 / 0 | 4 |
| full catalog | +5.58pp | 50 / 2 | 46 |
| full catalog 去掉 mass coverage | **+6.16pp** | **55 / 2** | **51** |

因此当前增益的主因是“候选属于代谢知识图以及其节点连接度”，不是参考谱数量或质量窗口覆盖率，也尚不是特定反应边。`topology` 与 `full catalog` 的 paired formula-cluster 差异跨 0，说明二者性能相当；不能因为 topology 点估计略高就宣称它显著胜出。

目录动作后剩余 244 个错误。“事后从十个反应特征中任选最有利者”的 oracle 可覆盖其中 61 个，即 7.09pp 总体上限，但该上限不是动作：所有单特征无门控策略均有大量新增错误，例如 `edge1_bottleneck_mean` 为 51 修正 / 80 新增，`known_log_seed_support_mean` 为 48 / 82。上限也不跨来源均匀：BV2cell 与 KGMN 只分别覆盖 1 和 2 个 residual error，而 Mouse brain、Mouse liver、NIST plasma、ST001154 分别覆盖 12、19、10、17 个。

这组结果只允许进入“保留完整路径身份、方向、种子和反应类型的表示诊断”，不允许直接微调共享 clean-spectrum encoder。当前 `pass_to_shared_embedding=false`。

## 正式服务器确认（job 2336398）

正式 10,000-bootstrap 运行完成，依赖、单元检查、逐 query B36 replay 和独立结构验证全部通过；正式点估计与本地完整复放完全一致。结果目录为 `data/validation/bioaware_b37_catalog_mechanism_2336398`。

对 topology 臂的 55 个修正和 4 个新增进一步做候选属性对账：

- 46/55 个修正（83.6%）属于 `truth network_member=1`、DreaMS baseline 错候选 `network_member=0`；
- 其余 9/55 个修正在 truth 与错候选均为网络成员时发生，才需要 degree 提供额外区分；
- 4 个新增错误中，3 个是网络内更高 degree 的错误候选压过正确候选，另 1 个是网络成员错误候选压过网络外 truth；
- 修正样本中 truth 相对 baseline 的 `known_log_degree` 平均差为 +1.4205；新增样本中最终错误候选相对 truth 的平均 degree 优势为 +1.9682。

因此当前约 5.93pp topology 增益的第一机制是代谢网络 membership 先验；节点中心性只是次级补充，并同时构成主要伤害源。它不是反应传播，也不能证明高 degree 候选在生物学上更正确。
