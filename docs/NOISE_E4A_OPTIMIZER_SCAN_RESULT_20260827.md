# E4-A 直接噪声微调：优化器扫描结果

日期：2026-08-27  
任务边界：只改变共享 DreaMS embedding；不使用 P2b、教师或下游重排器。

## 固定条件

- 同一 formula fold 0、seed 20260828；
- `curriculum + action_scope=all + views=4`；
- 解冻最后 1/7 Transformer block 与官方 projection head；
- candidate-gradient 50% step 3-6 + role-confounder 100% step 1-5；
- held formula 仅用于最终开发评价，不用于训练或 checkpoint 选择。

## 结果

| 配置 | Recall@1 增益 | near 增益 | corrected / introduced | risk net | preservation | 裁决 |
|---|---:|---:|---:|---:|---:|---|
| low LR: 5e-7 / 2.5e-6, clip 1 | +0.287 pp | +0.332 pp | 20 / 3 | 14 | 0.99807 | 欠拟合 |
| high LR: 2e-6 / 1e-5, clip 1 | **+0.540 pp** | +0.582 pp | 38 / 6 | **26** | 0.99524 | **安全最优** |
| base LR, clip 2 | +0.456 pp | +0.498 pp | 34 / 7 | 20 | 0.99640 | 不优于 high LR |
| base LR, clip 4 | +0.507 pp | +0.582 pp | 37 / 7 | 23 | 0.99578 | 次优 |
| base LR, 6 epochs | +0.507 pp | +0.554 pp | 36 / 6 | 24 | 0.99602 | 收益趋于饱和 |
| high LR, clip 2 | **+0.591 pp** | **+0.665 pp** | 42 / 7 | 28 | 0.99453 | 保持门失败 |

所有六组的 formula-cluster Recall@1 CI 下界均高于零。`high LR + clip 2` 虽然点估计最高，
但 embedding 平均保持度跌破预注册的 0.995 门，因此不进入多 fold。安全选择为
`backbone LR=2e-6, head LR=1e-5, clip=1, epochs=4`。

## 梯度诊断

- 原配置 clip=1 时约 90% 以上 step 触发裁剪，平均实际缩放约 0.35-0.46；
- 放宽到 clip=2 后裁剪率仍约 61%-75%；
- clip=4 后裁剪率下降到约 33%-51%；
- 提高学习率或放宽裁剪均能增加纠正，但后者更快损伤全局保持。

因此此前确有优化受限，但它只解释约 0.1-0.15 pp 的额外收益，不能解释与历史 3.85 pp
动作 oracle 的差距。固定 N-arm 已接近其可转移上限；继续靠更强更新主要交换新增错误和空间漂移。

## 下一步

1. 对安全最优配置完成 5 formula folds x 3 seeds；
2. 若方向稳定，将真实同身份跨条件正例作为 P-arm，与当前 N-arm 在同一共享编码器内联合训练；
3. P-arm 不使用 prototype teacher，不添加外部分子峰，只使用真实同身份、同加合物、跨仪器或
   碰撞能差异的谱图对及困难异构体负例；
4. P3 在开发结束前保持封存。

## 声明边界

本结果是 formula-fold 开发结果，证明直接噪声微调能够显著改变并改善共享 embedding；尚未构成
全外部测试结论。历史 3.85 pp 仍是逐查询事后选择动作/no-op 的 oracle headroom。
