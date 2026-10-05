# BioAware B39-M0/M1 本地执行结果（2026-09-13）

## 结论

B39 已完成从候选级路径比例到原子反应事件的工程重构，并严格复现旧 B3/B9 计算。当前结果建立了内部四来源固定动作扫描的输入资格，但尚未建立反应特异增益，也尚未改变 embedding。

## B39-M0：原子事件账本

- 1,738 个 query，6,695 个 query-candidate，38,999 个 candidate-context；
- 15,943 条 `query-candidate-seed-reaction-direction` 原子事件；
- 来源分层保留：合成 rotation（S）15,350 条、sample-local LOSO（L）396 条、hidden-standard（H）197 条；
- 方向分层：supported 870、supported-bidirectional 515、conflicted 853、unknown 13,705；
- 当前候选组中，每个 `seed-reaction` 事件只支持一个候选，但该结果对 KEGG pair-key 不能被解释为独立反应特异性；
- 11,166 条 Rhea 事件可恢复非 currency 参与物，只有 2,414 条在可见上下文中完整，完整率 21.62%；
- ST manifest 的 162/162 个 evaluation identity 原本均在对应样本 seed 集中，现有 ST 只属于 leave-one-seed-out 反事实恢复，不是 prospective unknown annotation；
- 现有六来源中 prospective unknown 来源数为 0。

## B39-M1：内部逐边实验层证据

- 8,701 条内部 candidate-seed 边；
- 6,815 条有共丰度匹配非邻居对照；
- 454 条有谱学匹配非邻居对照；
- 454 条同时具有谱学与共丰度对照；
- 映射回 15,350 条内部原子反应记录后，681 条事件同时具有两层证据；同一 candidate-seed pair 的多个 reaction ID 仍被视为依赖记录，不能重复计数。

最关键的工程验证：

- 逐边谱学结果重新聚合后，精确复现 B9 的 37,688 个 candidate-context，最大绝对误差 `1.11e-16`；
- 逐边共丰度结果重新聚合后，精确复现 B3 的 37,688 个 candidate-context，最大绝对误差 `1.11e-16`。

第一次本地实现曾错误地从所有 seed 而不是同一 rotation 的 seed 中选择谱学匹配对照，导致 B9 32,645 个数值不一致。历史复现门阻止该错误进入动作评估；修正后才通过正式 M1。

## 科学解释

1. B37 的 `+5.93 pp` 仍是 catalog membership/degree 先验，不能改称反应传播。
2. B38 的失败主要否定候选级路径比例，不等于否定原子反应事件。
3. B39 证明可在不读取排序结局的前提下，把方向、seed、reaction、候选、hyperedge 完整度和匹配对照重新对齐。
4. 当前真正瓶颈不是候选特异性，而是实验层覆盖：严格双证据只覆盖 681 条内部事件，外部 ST/KGMN 尚未逐边重建。
5. 因为只有内部 S 层具逐边双证据，下一步只能做内部固定动作发现；在 L/H 和新的 prospective 来源通过前，不能声称部署增益或跨来源反应特异性。

## 下一步准入顺序

1. 冻结 12-cell 动作定义与统一 DreaMS 风险层；
2. 只在内部 S 层做模型无关固定动作扫描，并报告相对 B37 topology 的增量；
3. 对 ST 重建真实 sample-local candidate-seed 谱学/共丰度边，对 KGMN 重建 hidden-standard 谱学边；缺失层不得填零；
4. 真实事件必须同时优于 matched non-neighbour、within-context seed permutation 与结构匹配 rewire；
5. 只有固定动作通过后才训练 context-conditioned candidate embedding；
6. 只有谱图可恢复分量跨 seed/context 保持方向后，才允许尝试 shared clean-spectrum encoder。

## 不得越界的表述

当前可以声称的是“原子事件与逐边匹配证据的可复现基础设施已建立”。不能声称 BioAware 已带来新的 reaction-specific Recall@1 增益、prospective 生物样本增益、shared embedding 改进或 SOTA。
