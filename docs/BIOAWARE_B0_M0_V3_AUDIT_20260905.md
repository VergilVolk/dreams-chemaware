# BioAware B0-M0 v3 审计与 v4 修正合同

## v3 的裁决

v3 不是反应信号检验，而是只使用协变量的匹配设计预检验。它没有读取
embedding、没有拟合模型、没有使用 phenotype 或 P2b。因此 v3 的失败不能解释为
“Rhea 没有生物学信号”或“BioAware 无效”。

v3 达到了覆盖门：814 个匹配组、372 个 source identity、326 个 source formula，
且 P3 identity overlap 为 0。但平衡门失败：Tanimoto SMD=0.130、target log-degree
SMD=0.451、target log-spectra SMD=0.136。

更重要的是，v3 的实现没有兑现“目标边际按构造保持”的合同。它使用每行独立的
dummy column；当 968 个 oriented contrasts 被丢弃时，被丢弃行对应的 target slot
仍可能被其他行使用。因此不完整匹配时，保留的真实 target multiset 和 control
target multiset 不相同。degree/spectra 的大 SMD 是这一实现缺陷的直接表现。

结论：v3 不允许进入 embedding signal test。

## v4 的唯一修正

v4 使用方阵 Hungarian assignment：

- 每个 oriented reaction contrast 同时是一行和一个 target slot；
- 合法非对角分配表示该行使用另一个 Rhea target 作为 unrecorded non-neighbour；
- 对角分配表示该行 unmatched，并同时占回自己的 target slot；
- 所有保留行只能形成闭合置换环。

因此即使只匹配部分反应边，保留的真实 target slot multiset 与 control target slot
multiset 仍精确相同。`target_log_degree` 与 `target_log_spectra` 的 SMD 必须在
浮点容差内为 0，否则脚本立即失败。

v4 还冻结四个 Tanimoto caliper（0.04/0.06/0.08/0.10）。四个设计全部报告；
它们只使用结构与观测协变量，不读取 embedding 或任何下游结果。在满足全部覆盖与
平衡门的设计中选择样本量最大的一个；若没有设计通过，则只保存最佳平衡诊断，
`pass_to_signal_test=false`，不得放宽阈值或训练 B1。

## v4 通过门

- groups >= 500；
- source identities >= 300；
- source formulas >= 200；
- 所有连续协变量 SMD <= 0.10；
- 所有分类协变量组内 mismatch 为 0；
- closed target marginal 必须精确保留；
- P3 identity overlap 必须为 0。

只有 `pass_to_signal_test=true` 才允许读取冻结 DreaMS embedding，开展下一步无模型
配对信号检验。通过仍不等于候选排序增益、共享 embedding 改善或 SOTA。

