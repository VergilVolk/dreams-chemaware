# Noise final dynamic-direct Phase A：四臂共享 embedding 结果

日期：2026-09-05  
状态：正式开发 fold 结果；停止进入第二 seed  
原始作业：`2331284`；精确续跑作业：`2331352`

## 1. 本次实验回答的问题

本实验检验的不是冻结 encoder 上的动作空间容量，而是：在成熟 E4-A shared encoder 上，以同一初始化、同一训练 query、同一候选集合、同一 epoch schedule、同一总 action dose、同一优化器和同一 seed，动态的 N+P 定向动作分配是否能比以下三个严格对照产生更好的 **clean-spectrum shared embedding**：

1. `clean_continuation`：相同成员与训练预算，但 action view 为 clean；
2. `matched_random`：相同权重，payload 换为来源匹配的方向对照；
3. `static_target`：使用真实 target payload，但 query 内动作权重不使用动态预测分配；
4. `dynamic_np`：使用真实 target payload和 formula-OOF 动态 N+P 权重。

四个臂均直接更新同一个 query/reference encoder；推理只输入 clean spectrum。P2b 未使用，P3 未消费。

## 2. 数据与协议

- Outer formula fold：0（开发 fold）
- Held queries：5,923
- Near queries：3,611
- 初始化：成熟 E4-A fold-0 shared encoder
- 初始化 Recall@1：0.9387134898
- Official DreaMS Recall@1：0.9329731555
- 训练：4 epochs；最后一个 Transformer block + projection head
- 学习率：backbone `2e-6`，head `1e-5`
- Action curriculum：0.35 / 0.45 / 0.55 / 0.55
- 训练前在当前成熟 E4 geometry 中完整回放 304,209 条 N/P actions，并重新执行 formula-crossfit 和共同 schedule
- 所有臂使用相同 schedule membership/order；每 query 的 static 与 dynamic 总 action dose 完全匹配，并保留显式 no-op mass

## 3. Clean held retrieval 结果

| Arm | Recall@1 | 相对 official | 相对成熟 E4 初始化 | corrected / introduced（相对初始化） | risk net, lambda=2 | Near delta | MRR delta | Formula-cluster 95% CI（相对初始化） |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clean continuation | 0.9398953 | +0.6922 pp | +0.1182 pp | 11 / 4 | +3 | +0.1939 pp | +0.0893 pp | [-0.0257, +0.2893] pp |
| matched random | 0.9398953 | +0.6922 pp | +0.1182 pp | 14 / 7 | 0 | +0.1662 pp | +0.0822 pp | [-0.0414, +0.2988] pp |
| static target | 0.9398953 | +0.6922 pp | +0.1182 pp | 12 / 5 | +2 | +0.1939 pp | +0.0921 pp | [-0.0223, +0.2805] pp |
| dynamic N+P | 0.9397265 | +0.6753 pp | +0.1013 pp | 11 / 5 | +1 | +0.1662 pp | +0.0808 pp | [-0.0384, +0.2682] pp |

相对 official 的正值包含成熟 E4 初始化已有的 +0.5740 pp，不能归因于本轮 dynamic N+P。相对初始化的四臂增量均很小，且 formula-cluster CI 均跨零。

## 4. 关键配对因果比较

| Comparison | Delta Recall@1 | corrected / introduced | risk net, lambda=2 | Near delta | Formula-cluster 95% CI | Holm-adjusted p |
|---|---:|---:|---:|---:|---:|---:|
| dynamic N+P vs matched random | -0.0169 pp | 2 / 3 | -4 | 0 | [-0.0945, +0.0598] pp | 1.0 |
| dynamic N+P vs static target | -0.0169 pp | 0 / 1 | -2 | -0.0277 pp | [-0.0485, 0] pp | 1.0 |
| dynamic N+P vs clean continuation | -0.0169 pp | 1 / 2 | -3 | -0.0277 pp | [-0.0607, 0] pp | 未作为 multiplicity 主检验 |

预注册的两个核心方向门均失败：dynamic 没有优于 matched-random，也没有优于 static-target。`pass_to_second_seed=false`。

## 5. 严格裁决

1. **本轮确实产生了新的 shared-encoder checkpoints，但没有证明定向 dynamic N+P 带来独立 embedding 增益。**
2. 四臂共同出现的约 +0.10 至 +0.12 pp 相对初始化增量主要属于 shared continuation/训练预算效应；不能记在 N/P 定向动作名下。
3. `dynamic_np` 相对三个对照均少净 1 个 held query，因此当前动态分配实现不得推广到第二 seed、其余 folds 或 P3。
4. 当前新的数值最优 Recall@1 是 clean、matched-random、static-target 三臂并列；在 MRR、near 和新增错误综合考虑下，`clean_continuation` 是最稳妥的本轮 checkpoint，但其相对初始化增益也未达到 formula-cluster 显著性。
5. 成熟 E4 初始化仍是当前噪声路线的正式基准。不得用本轮 dynamic checkpoint 替代它，也不得把 `+0.6753 pp vs official` 写成动态噪声贡献或 SOTA 证据。
6. 该结果否定的是这一个冻结的 dynamic weighting + direct-training kernel 在 fold 0 上的增量效果，不是否定既有 N/P 动作在冻结几何中的动作容量，也不授权回到随机噪声、P2b 或无边界超参扫描。

## 6. 工程与统计完整性

- 正式 schedule、四臂和 summary 均完成；独立验证器通过。
- 共同 schedule 的全部 gates 通过，30 个 cells 均保留。
- Dynamic preservation mean：0.9964052，超过 0.995 门。
- Dynamic 各 epoch clip fraction：0.4923、0.4691、0.4565、0.4368；未触发预注册的系统性 clipping 门，但高裁剪率应作为后续失败归因的候选因素，不能单独据此宣称根因。
- 失败发生在科学增量门，而不是数据缺失、序列化、GPU、checkpoint replay 或结果验证。

## 7. 冻结制品与来源

服务器运行根目录：

`data/validation/g8r_noise_final_dynamic_direct_e4_phase_a_final/fold_0_run_2331284`

主要制品：

- `schedule/report.json`
- `schedule/epoch_schedule.csv.gz`
- `arms/{clean_continuation,matched_random,static_target,dynamic_np}/report.json`
- `arms/{clean_continuation,matched_random,static_target,dynamic_np}/held_per_query.csv.gz`
- `arms/{clean_continuation,matched_random,static_target,dynamic_np}/final_shared_encoder.pt`
- `summary/report.json`

本地完整日志：`noise_dd_resume_2331284_2331352.out`  
本地日志 SHA256：`5dd5f60fed6caf8519e8665bb8fd7fb3c9cd4765a67766fb4821c43017d0af3e`

四臂 report SHA256：

- clean continuation：`b8fbbb30c569592da204e469d6e6fc64f010abc084d3541aeaaf47ba79cab23e`
- matched random：`9931d322f257a695156cbd88192cc4cf6eda6bdf21743a73fbfb2d53a4d414c1`
- static target：`0cce06b3398c450467205c439614f713961b863c7c2d9a965382d988f9ea2ba6`
- dynamic N+P：`5bae67bbed25675f56ea6c642dc6b36de6f954cffe8cdc359ff2196b45010e44`

## 8. 后续边界

下一步只能针对已经观察到的配对失败做归因：为什么 current-geometry crossfit 可预测 action value，却没有在同剂量、同 schedule 下转化成 clean embedding 增益。任何新训练必须先解释 dynamic 与 static/matched-random 几乎完全重合的原因，并保留本次四臂作为不可回写的因果基线。

