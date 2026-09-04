# Noise L0 结果与 L1 clean-input action learnability 协议

日期：2026-09-03  
状态：L0 已完成；L1 在任何结果产生前冻结

## 1. L0 的决定性结论

L0 在 E4-A clean-duplicate continuation 的同一共享 query/reference encoder 下，
对 36,934 条 R0 动作逐条重放 target 与两条 frozen matched-random control，并在完整
候选图上计算严格 rank 和 margin。

- 20,997 条 positive、12,178 条 neutral、3,759 条 harmful；
- 564 个 action-row correction、229 个 action-row introduction；这些是嵌套动作行，
  不得相加冒充 unique-query 模型收益；
- 历史与 mature paired advantage 的 Pearson 相关为 0.9491；双方均非中性时方向
  一致率为 99.44%；
- candidate-gradient 的 target-specific advantage 从 step 3 的 0.01336 单调增加到
  step 6 的 0.02083，四个 formula-cluster CI 下界均大于零；
- role-confounder step 5 为 17 corrected / 0 introduced，但覆盖较低。

所以旧动作并未普遍失效。现阶段瓶颈是强 margin 信号能否仅由 clean input 跨 formula
识别，以及能否通过 no-op 避免约 10.2% 的 harmful action。L0 不是训练结果。

## 2. L1 唯一问题

在完全不更新 DreaMS 权重的条件下，检验 clean spectrum 是否能够预测每条固定
family/step action 的 full-list target-minus-random advantage，并在 held formula 上选出
净纠错、低新增错误的动作。

## 3. 允许和禁止的输入

允许：

1. 未修改 clean spectrum 的 m/z、intensity、稀疏度、entropy；
2. precursor、adduct、collision energy、instrument 等实际存在的采集字段；
3. clean spectrum 的 label-free contextual peak tokens；
4. clean spectrum 的 frozen mature embedding；
5. 预先固定的 action family、attenuation 和 step。

禁止：identity、formula、正确候选、错误候选、任何 candidate score、baseline rank/margin、
target path、control path、action outcome、P2b、P3。Identity/formula 只可用于权重、拆分
和结果审计。

## 4. 固定模型与对照

- 五折 formula-disjoint OOF；每个 formula 在且仅在一个 held fold；
- 每个训练 formula 总权重相同；
- 主模型为固定参数的 HistGradientBoosting：分别预测连续 paired advantage、positive
  概率与 harmful 概率；
- action-family-only 对照使用训练侧 formula-equal cell 均值；
- permutation 对照在训练侧每个 family/step 内置换 outcome，held outcome 不动；
- 不搜索模型深度、学习率或阈值。

## 5. 预注册 no-op 策略

主策略只接受同时满足：`P(positive)>=0.70`、`P(harmful)<=0.10`、预测 gain `>=0.01`
的动作。每个 query 最多选择预测 gain 最大的一条，其余为 no-op。另完整报告 moderate
和 strict 灵敏度，但不得代替主策略裁决。

## 6. L2 放行门

主策略必须同时满足：

1. positive AUPRC 高于 family-only 和 permutation；
2. 至少 500 queries、300 identities、150 formulas；
3. 全 no-op population 的 paired-advantage formula CI 下界大于零；
4. Top-1 delta formula CI 下界大于零；
5. corrected > introduced，且 corrected - 2*introduced > 0；
6. near delta 不为负；
7. Top-1 相对 family-only 和 permutation 的 paired formula CI 下界均大于零；
8. 五个外层 fold 均 formula-disjoint。

只有全部通过，才授权一次小规模 paired targeted-vs-matched-random shared-encoder L2。
L1 本身不产生新 embedding，也不构成模型性能提升。
