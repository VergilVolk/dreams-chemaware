# E4-A 三臂共享 embedding 因果归因最终结果

日期：2026-09-01  
训练作业：2328357（array 0–2）  
汇总：`noise_final_e4a_causal_attribution_complete`  
裁决：Attribution gate 未通过；停止扩大或调参当前 E4-A 定向噪声训练配方。

## 1. 协议有效

所有工程与科学不变量均通过：三臂完整、正式、同一初始化、同一非动作配置、同一输入 provenance、同一四轮 action/safety batch schedule、同一 held queries 和 initialization。matched controls 在训练前冻结且不使用 outcome；完整候选图用于 clean retrieval；P2b 禁止；P3 未消费。

因此本次失败不能归因于 baseline、候选图、sampler、seed、训练预算或实现漂移。

## 2. 单臂相对 official

| 训练臂 | Recall@1 增量 | corrected / introduced | risk net | near 增量 | MRR 增量 |
|---|---:|---:|---:|---:|---:|
| clean duplicate | +0.5065 pp | 37 / 7 | 23 | +0.5539 pp | +0.3560 pp |
| matched random | +0.5065 pp | 37 / 7 | 23 | +0.5539 pp | +0.3560 pp |
| targeted | +0.5403 pp | 38 / 6 | 26 | +0.5816 pp | +0.3735 pp |

单臂相对 official 的正向 CI 只证明三种 continuation 训练均改善该 development fold，不能证明定向峰语义的独立贡献。

## 3. 预注册主比较

### Targeted − matched random

- Recall@1：+0.03377 pp；paired corrected / introduced = 2 / 0；
- formula-cluster Top-1 95% CI：0.0000 至 +0.09275 pp，严格下界不大于 0；
- MRR：+0.01745 pp，95% CI −0.00224 至 +0.04776 pp；
- near Recall@1：+0.02769 pp；
- mean full-candidate margin：+0.0000401，95% CI −0.0001548 至 +0.0002345；
- Top-1 molecule 变化：3；其中 wrong-to-different-wrong：1。

### Targeted − clean duplicate

- Recall@1：+0.03377 pp；paired corrected / introduced = 3 / 1；risk net = 1；
- formula-cluster Top-1 95% CI：−0.02798 至 +0.11324 pp；
- MRR 和 full-candidate margin 的 CI 均跨 0；
- Top-1 molecule 变化：7；wrong-to-different-wrong：3。

### Matched random − clean duplicate

- Recall@1：0；paired corrected / introduced = 1 / 1；
- Top-1、MRR 和 full-candidate margin 的 CI 均跨 0。

## 4. 根因定位

按净正确点估计，targeted 相对 official 的 32 个净纠正中，clean continuation 已经贡献 30 个；定向动作只增加 2 个，即点估计约 6.25%，且统计下界为 0。不能再把 +0.5403 pp 称为“定向噪声带来的提升”。

训练动力学同时表明，targeted action view 并非无效：第四轮平均 action margin 从 clean-view 0.30062 增至 augmented-view 0.31290；matched-random 从 0.30197 降至 0.29793。但 held-action clean-query 子集三臂完全相同，均为 30 corrected / 2 introduced。由此可将失败精确定位为：

1. 模型能编码被修改后的谱图；
2. 当前加性 action-rank/consistency 损失没有把 targeted−random 的方向特异性写入 clean query；
3. 三臂共同的 clean/listwise continuation 支配了最终权重更新；
4. candidate-gradient/confounder 使用的候选特权信息没有形成跨 formula、clean-input 可识别的机制；
5. 当前结果不支持通过增加 epoch、action exposure、学习率或解冻层数来补救。

## 5. 后续唯一授权

不进入 multifold、P3、LR/layer/epoch 网格，也不训练 action selector。下一步仅授权一个无大规模训练的梯度归因审计：在同一 query/action/control microbatch 上分别计算 clean、targeted、matched-random 梯度，报告 `targeted−random` 差分梯度相对共同 clean 梯度的范数、余弦、formula 一致性、clipping 损失以及对 full-list margin 的一阶预测。

- 若差分梯度近零或跨 formula 方向不一致：当前成熟 action 只能保留为反事实解释/headroom，不再进入 clean encoder；
- 若差分梯度稳定但被共同 loss 或 clipping 淹没：才允许实现显式 paired counterfactual advantage loss，并先做一个小型 paired pilot；
- 任何后续结果仍以 clean full-list targeted−matched-random 为主终点。
