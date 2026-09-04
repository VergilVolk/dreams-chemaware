# Noise-v3 A4-B 正例证据恢复与两层专家策略：预注册协议

**日期：** 2026-08-25  
**状态：** B0 诊断脚本已完成；结果未知；禁止在看到正式结果后调整主剂量或进入门

## 1. 已终止的动作策略

以下策略在等覆盖率审计中产生显著负净收益，不再作为独立动作策略：

- gradient-only；
- shared-only + gradient；
- 对所有查询执行固定比例随机遮峰；
- 整组条件特异峰删除；
- 不含 no-op 和 harm 约束的全覆盖动作策略。

它们只能作为负对照或候选生成器，不能进入正式微调数据。

## 2. 为什么先扩展动作空间

A4 非线性教师能预测动作收益和伤害，但相对 confounder-only 的风险净收益差尚未显著；它在历史策略之外只新增修正 19 个。继续在同一删峰空间调网络结构会放大开发集选择偏差。

因此先检验第二种大梯度来源：真实同身份、跨条件参考谱所提供的正例证据。

## 3. B0 固定实验

对每个 P3-disjoint A4 查询：

1. 使用同 IK14、同加合物的其他真实谱图构成归一化正例原型；
2. 将官方查询 embedding 以固定剂量向原型移动；
3. 在原 strict-10 ppm、同加合物候选组中重新排名；
4. 对照为同一候选组中错误身份的真实谱图原型；
5. 不增加合成峰，不选择每个查询的最优剂量，不更新 DreaMS 权重。

剂量固定为 0.10、0.25、0.50；主剂量固定为 0.25，另外两个剂量只报告剂量趋势。

## 4. B0 进入门

主剂量 0.25 必须同时满足：

1. 在 S1c/S2/S3A/A4 之外新增修正至少 80 个；
2. 风险净收益 `corrected - 2*introduced` 的 formula-cluster bootstrap CI 下界大于 0；
3. baseline-error 中 target 相对错误身份原型对照的 Top-1 差值，formula-cluster CI 下界大于 0；
4. introduced 不超过 corrected 的一半；
5. 官方 embedding 与候选图的 baseline 分数逐对复现，任何不一致均 fail-closed。

未通过时，不通过扩大剂量、best-of-three 选择或降低 CI 标准补救。

## 5. B0 的科学边界

真实身份原型只用于构造训练目标。它在部署时不可获得，因此 B0 结果是几何可训练性证据，不是检索算法，也不是性能提升。

若 B0 通过，后续学生模型只接收原始查询谱图；原型方向 stop-gradient，作为训练期教师。

## 6. 两层专家 v2

### 专家一：安全动作专家

- 主体为 confounder-only；
- 优先 25%/50%/75% 软衰减；
- 低峰数和低部署置信度查询默认 no-op；
- 目标是最低 introduced，而不是最大覆盖率。

### 专家二：扩展/恢复专家

- 若 B0 通过，加入真实跨条件正例教师方向；
- shared/unmatched 峰动作必须同时通过 benefit 与 harm 预测；
- 使用多种子不确定度下界，而不是只使用均值效用；
- 负责安全专家未覆盖的 positive-deficit 和边界错误。

### 门控

只允许部署可得特征：候选Top1–Top2分差、候选熵、候选数、谱峰数、动作角色、剂量、峰强度/mz、规则证据和可用采集元数据。真实正例margin、错误家族、DreaMS是否正确等标签字段禁止输入。

## 7. v2 必须增加的模型对照

1. confounder-only 固定策略；
2. 原 A4 单一 MLP 教师；
3. 两层专家但不含规则特征；
4. 两层专家含规则特征；
5. 两层专家去掉不确定度门；
6. no-op。

规则只有在“同结构含规则模型”相对“不含规则模型”的配对 formula-cluster CI 下界大于 0 时，才允许进入微调策略。

## 8. v2 的显著性要求

两层专家必须同时满足：

- 相对 confounder-only 的风险净收益配对 formula-cluster CI 下界大于 0；
- 相对单一 MLP 教师的风险净收益不下降；
- 新增修正至少 80；
- positive-deficit 与 near 分层分别报告，不得由 easy 层抵消；
- introduced 的峰角色、剂量、峰数、置信度和结构层完整输出；
- 模型、阈值、剂量和特征在进入封存测试前全部冻结。

只有满足这些条件，才进入候选条件化 token adapter 微调。

## 9. 当前产物

- `tasks/diagnose_noise_v3_a4b_positive_evidence.py`
- `tasks/validate_noise_v3_a4b_positive_evidence.py`
- `tasks/test_noise_v3_a4b_positive_evidence.py`
- `tasks/run_noise_v3_a4b_positive_evidence.sbatch`
