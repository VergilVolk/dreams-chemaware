# BioAware B47-U0：参考谱多重性与离子分支审计契约

## 目的

B47 当前对每个候选 IK14 的所有参考谱取最大 DreaMS cosine。候选参考谱数量不等，且未知 query adduct 下 `[M+H]+` 与 `[M+Na]+` 候选分支共同竞争。U0 在不读取注释真值、疾病表型或任何 BioAware 输出的前提下，量化这两种 nuisance 对候选顺序的影响。

U0 不是性能实验，不选择模型，也不声称 mean、expected-max 或任一 adduct 分支优于现行 max。它只冻结 U1 必须比较的评分协议。

## 预注册统计量

1. 候选参考谱数的完整分布，以及候选内 `max(score)-mean(score)`。
2. 固定同一个 query–candidate，随机无放回暴露 `k={1,2,4,8}` 张参考谱时最大分数的精确期望。它由次序统计量解析计算，不用 Monte Carlo，也不使用结果标签；主曲线限定在同一批 `n≥8` 候选上，避免不同 k 的样本构成变化冒充剂量效应。
3. `max` 与“均匀抽一张参考谱的期望值（mean）”之间 Top-1 身份、Top-1 adduct 和 margin 的变化。
4. 仅按参考谱数量选候选的 spectrum-blind shortcut 与 max winner 的重合率。
5. 同时存在 `[M+H]+` 和 `[M+Na]+` 候选分支的 query 比例、分支间最高分差和 pooled winner 的 adduct 构成。
6. 上述敏感性在 ST001122 与 ST003356 两个来源中分别报告，避免总体结果由单一来源驱动。

## 不可越界的解释

- `max` 与 `mean` 排名翻转只证明聚合器敏感，不证明哪一个更准确。
- 候选内 max lift 随参考谱数增长是极值机会的直接量化，但不能替代带真值的检索比较。
- 两个 adduct 分支竞争说明 neutral-mass/ion-form 假设尚未建模；不能据此假定任一分支为真。
- U0 工程门通过后进入 U1。U1 才能在冻结 split、相同候选集和相同 adverse-tie 规则下比较真实 Recall@k、MRR、macro query AUC 与校准。

## 禁止输入

- sealed annotation truth 或任何 truth/label/correct 列；
- phenotype、case/control、疾病分组；
- P2b、B30–B46 或其他算法输出；
- 根据 U0 结果删除 query、candidate 或 adduct 分支。
