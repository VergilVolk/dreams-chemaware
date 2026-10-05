# BioAware B44：冻结目录专家的独立 MassBank 外部验证结果

日期：2026-09-13  
状态：一次性外部验证已完成；`pass_external_confirmation=false`；B44 自此视为已消耗测试集，不得在其上重新选特征、门或超参数。

## 1. 工程与协议有效性

- 便携谱图包校验通过：6,909 张必要谱图，来源 MassBank 哈希固定；
- 官方 DreaMS checkpoint：`8928f908606c0bd652c5a4107d3c35102f660622958c225a1f625abe4b1ba245`；
- 纯 NumPy HGB 工件：120 棵树、1,560 节点；导出前在 104,670 个输入上相对原 sklearn 模型的最大概率误差为 `1.11e-16`，分类不一致为 0；
- 893 个 query / 395 个分子式；query 身份、候选身份和 truth formula 均与 B42 开发宇宙隔离；
- 排序和并列、候选分子 max-reference cosine、冻结门、5 次目录特征置换负对照均按预注册协议执行。

因此，本结果没有证据指向便携化、pickle 替换、checkpoint 或行映射错误。它是算法外部迁移失败，而不是已知工程失败。

## 2. 主要结果

| 面板 | n | DreaMS R@1 | BioAware R@1 | Delta | corrected / introduced | formula-cluster 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| 负离子主面板 | 345 | 0.7507 | 0.7449 | -0.58 pp | 4 / 6 | [-2.33, +1.21] pp |
| 正离子次级面板 | 548 | 0.8431 | 0.8376 | -0.55 pp | 4 / 7 | [-1.50, +0.37] pp |
| 全体 | 893 | 0.8074 | 0.8018 | -0.56 pp | 8 / 13 | [-1.45, +0.34] pp |

全体 MRR 下降 0.289 pp，macro query AUC 下降 0.352 pp。负离子主面板 MRR 和 AUC 同样为负。所有正式门均失败。

门只干预 28/893 个 query（3.14%），其中 8 个受益、13 个受损、7 个 Top-1 状态不变。负离子仅干预 13 个 query，其中 4 个受益、6 个受损。当前门没有形成可接受的风险收益比。

## 3. 特异性对照

负离子真实目录特征的 R@1 Delta 为 -0.58 pp。五次 query 内目录特征置换对照的 Delta 分别为 0、0、0、-0.87、+0.29 pp。真实特征减平均置换对照为 -0.46 pp，formula-cluster 95% CI 为 [-2.32, +1.34] pp。

因此，B44 不支持“正确的 KEGG/Rhea 候选归属或目录度数提供了可迁移的增量排序证据”。

## 4. 对 B42/B43 的修正解释

B42 的约 +6 pp 和冻结工件在 B42 OOF 上的 +6.98 pp 没有迁移到 formula/identity 隔离的 MassBank。外部结果与开发结果相差约 7.5 pp。最合理的当前解释是：

1. B42 的主要信号依赖开发来源中的目录覆盖结构和候选生成分布；
2. B43 已显示较大收益来自“truth 被目录收录而错误候选未收录”的成员不对称；
3. B44 改变数据库、分子式和候选身份后，这种不对称不再稳定；
4. B43 中 mapped-competition 的开发信号也未证明能跨到 B44；
5. 固定的绝对目录成员/度数特征和统一 gate 对数据库覆盖偏移缺乏校准。

不得继续称 BioAware-Catalogue v1 为跨数据库稳定提升、SOTA、reaction propagation、sample-context 或 shared-embedding 改进。

## 5. 没有被本结果否定的方向

B44 是孤立谱库的候选静态目录先验测试。它没有同一样本代谢物种子、峰相关、样本共现、真实反应路径或组织/疾病上下文。因此它只否定当前 `candidate-static catalogue topology` 工件的外部部署，不否定真正的样本上下文 BioAware。

真正的 BioAware 下一代必须使用 phenotype-blind 的同一样本高置信种子和候选到种子的反应路径，并证明真实网络优于 degree-preserving、seed-permuted 和 edge-permuted 对照；否则仍只是数据库覆盖先验。

## 6. 后续纪律

1. 冻结并归档 B44；不得在 B44 上调门或选新特征。
2. 对 B44 `per_query.csv.gz` 和 `candidate_scores.csv.gz` 只做解释性失败审计：目录覆盖对称性、degree shift、置信校准、候选数、margin、corrected/introduced 路径。
3. 如果继续静态目录专家，开发时删除绝对 membership 捷径，只在候选目录覆盖模式相同的组内研究相对 topology；必须另锁 B45 外部集。
4. BioAware 主线转回样本上下文：高置信种子集合、一步/有向反应超边、峰相关与可学习 abstention；先证明动作相对匹配 null 有净增益，再讨论共享 embedding。

## 7. 正式产物

- 服务器结果：`data/validation/bioaware_b44_massbank_once_2336677/`
- 便携包：`data/validation/bioaware_b44_portable_bundle_20260913_v3/`
- 运行日志：`bioaware_b44_portable_2336677.out`

