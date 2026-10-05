# BioAware B0-M0 最终裁决（2026-09-05）

## 结论

B0-M0 v4 的闭环置换实现通过了数学不变量，但没有通过预注册的覆盖与平衡联合门，
因此不得进入正式 embedding signal test，也不得据此训练 relation adapter。

这不是“Rhea 没有信号”或“BioAware 无效”的证据。它说明的是：在当前 DreaMS
训练谱库与当前 Rhea 映射的交集中，无法同时构造规模充分且化学混杂受控的
single-spectrum reaction-neighbour 对照设计。

## v4 证明已经修复的部分

- 所有设计均满足 closed target-slot multiset；
- target log-degree SMD=0；
- target log-spectra SMD 约为 4e-16；
- 所有分类协变量 mismatch=0；
- P3 identity overlap=0；
- 未读取 embedding，未拟合模型，未使用 phenotype 或 P2b。

因此，v3 的独立 dummy column 导致的目标边际漂移已经消除。

## 无可用 operating point

| Tanimoto caliper | groups | source identities | source formulas | max SMD | 结论 |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0.04 | 304 | 169 | 148 | 0.033 | 平衡好，覆盖不足 |
| 0.06 | 399 | 217 | 193 | 0.058 | 平衡好，覆盖不足 |
| 0.08 | 459 | 241 | 215 | 0.089 | 平衡合格，组数与 identity 不足 |
| 0.10 | 546 | 273 | 246 | 0.136 | 组数合格，但 identity 与平衡失败 |

预注册门为 groups>=500、source identities>=300、source formulas>=200、所有连续
SMD<=0.10。没有任何冻结设计同时满足这些条件。不得事后把 identity 门降到 241、
把 SMD 门放宽到 0.14，或只展示 0.08/0.10 中更好看的部分。

## provenance 纠正

v4 的 control target 不是来自全部 9,610 个观测 identity，而是来自 1,782 个
eligible observed Rhea-edge target slots。旧报告中的
`controls_from_full_observed_universe=true` 是错误描述，代码已改为明确报告：

`control_target_pool = eligible observed Rhea-edge target-slot multiset`

该更正不改变 v4 数值或失败裁决。

## BioAware 路线决策

1. 冻结 B0-M0 v4 为负面的可辨识性预检验，不再对当前数据继续调 matching 阈值。
2. 不开展正式的 single-spectrum reaction-neighbour adapter 训练。
3. 保留已经成立的 BioAware v2 样本上下文候选专家结果；它解决的是样本内候选消歧，
   与 single-spectrum 通用 embedding 是不同科学问题。
4. 若未来获得更大的 Rhea-映射谱库或独立样本级 MS1/MS2 队列，重新预注册新的
   B0 设计；不得复用已经看过结果的 v4 门作为盲验证。
5. 现有 459 个平衡组可用于机制探索和功效估计，但必须明确标为 exploratory，
   不能作为共享 embedding 改善或 SOTA 的正式证据。

