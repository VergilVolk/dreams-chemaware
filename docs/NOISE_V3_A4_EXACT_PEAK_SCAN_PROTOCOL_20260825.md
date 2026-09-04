# Noise-v3 A4：全错误空间逐峰精确干预协议

## 科学问题

S3A 已证明候选条件化梯度存在净信号，但合并 outcome-oracle 仅覆盖 799 个错误，仍不足以支撑 +4 pp。A4 检验：一阶梯度和粗峰角色是否遗漏了具有大真实效应的峰级动作。

## 固定数据范围

- 训练侧严格 10 ppm 完整候选图：23,876 查询；P3 identity overlap 必须为 0。
- 处理全部 1,805 个官方 DreaMS Top-1 错误，不允许按错误类型预筛。
- 每个错误匹配 3 个官方正确安全对照；匹配顺序为 `formula+near → formula → near → global`，距离使用 baseline margin、候选数与峰数；控制查询最大复用 3 次。
- 控制匹配层级、距离和复用次数全部落盘，不能静默降级。

## 正交动作矩阵

- 峰：每条查询所有真实 fragment token；precursor/padding 永久排除。
- identity-only 峰也扫描，但只作为负向机制对照，`policy_eligible=false`。
- 剂量：0.25、0.50、0.75、1.00；1.00 表示删除，其余表示降权后重新最大强度归一化。
- 每个变体重新运行官方 DreaMS，并对完整候选图计算严格 rank：`1 + #negative(score >= positive)`。
- 明确保存在并列时导致 rank 失败的 adversarial negative，禁止使用普通 argmax 掩盖并列负例。

## 记录字段

每个峰动作记录：

- query / token / m/z / intensity；
- 候选条件化峰角色；
- 输入梯度、预测一阶增益与梯度排名；
- 四档剂量下的 positive score、hardest-negative score、margin、rank、MRR；
- 最强错误候选 molecule index 与 spectrum row；
- error family、positive-deficit/negative-excess 和规则证据仅作描述性协变量。

## 决策门

1. A4 必须在 S1c/S2/S3A 之外新增至少 157 个可恢复错误；
2. 合并 action oracle 至少达到 956，进入策略训练的稳妥门为 1,000；
3. 同时报告安全对照中可被动作破坏的查询，不能只报纠正；
4. 报告 gradient Top-1/3/6/12/25/50 对 exact oracle 的覆盖率，决定梯度预筛是否允许继续使用；
5. 若动作空间仍不过门，先构建 positive-deficit 的真实跨条件正例分支，不允许用更复杂策略模型掩盖 headroom 不足。

## 明确边界

- A4 是训练数据上的动作空间实验，不是微调结果。
- per-query best action 使用了结果，只是 oracle。
- 化学规则不能定义动作标签；规则贡献必须在后续 formula-group OOF 策略中通过有/无规则配对消融证明。
- A4 不向谱图添加来自其他分子的峰。

