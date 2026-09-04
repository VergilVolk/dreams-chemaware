# Noise E4-B2：系统梯度处理筛选与独立确认协议

日期：2026-09-02  
状态：预注册；零权重更新；E4-B1 后续筛选/消融

## 1. 因果起点

E4-B1 显示，`candidate_gradient a=0.50 step=6` 在 official-error 样本中同时具有正的 target-vs-matched-random margin 和跨 formula 梯度共识，但 paired-advantage 梯度与 same-query clean-ranking 梯度未可靠正向对齐。其余成熟 cell 也不能同时通过三门。因此下一问题不是继续扫学习率，而是系统检验：哪一种动作 cell、梯度冲突处理和参数层范围，能在保留动作下降能力的同时与 clean embedding 目标相容。

## 2. 面板隔离

- 仅使用 outer formula fold 0 的训练侧；P3、P2b 禁止使用。
- 排除 E4-B1 面板出现过的全部 formula。
- formula 由固定哈希全局分配至 `screen` 或 `confirm`，两者在所有 cell 间完全不重叠。
- 每个 cell 内仅保留同时具有 official-error 与 official-correct query 的 formula；两个状态使用完全相同的 formula 集。
- 每个 cell、每个 split 目标32个配对 formula；数据不足时允许确定性缩减，但每边不得少于24个。
- query 选择只使用 cell、formula、baseline state 和固定哈希；不得使用 target/random action outcome。

## 3. 固定消融矩阵

九个成熟动作 cell 全部保留：

- `candidate_gradient a=0.50`：step 3–6；
- `role_confounder a=1.00`：step 1–5。

参数层范围：

- `joint`：projection head + 最后一个 transformer block；
- `head`：仅 projection head；
- `backbone`：仅最后一个 transformer block。

梯度处理：

- `raw`：原 paired-advantage 梯度；
- `pcgrad0`：只删除其相对于 clean 梯度的负投影；
- `anchor_0.10`、`anchor_0.25`：加入范数匹配的 clean anchor；
- `pcgrad_anchor_0.05`、`pcgrad_anchor_0.10`：先删负投影，再加入小剂量 clean anchor。

共18个 `method × scope` 配置。该矩阵固定报告，不得按结果删除失败配置。

## 4. 防止伪通过的终点

对每个 `split × cell × baseline_state × configuration`，以 formula 为单位报告：

1. target-minus-selected-random forward margin；
2. 处理后梯度与 clean 梯度 cosine；
3. 处理后梯度的 leave-one-formula-out consensus；
4. 动作下降保留率 `dot(g_action,g_safe)/||g_action||^2`；
5. 处理后梯度与原动作梯度 cosine；
6. 处理后/原动作梯度范数比。

clean alignment 可被 anchor 人为提高，因此不能单独放行。动作下降保留率的下界必须至少0.5，防止配置退化为 clean duplicate。

## 5. Screen 与 confirm

Screen 只使用 `screen` formula。预筛门要求：

- error forward-margin 95% CI lower > 0；
- error consensus、error clean alignment 和 paired-correct clean alignment 的均值均 > 0；
- error action-descent retention 均值 >= 0.5。

通过者按五个终点的固定 rank-sum 排序；每个 cell 最多保留一个配置，全局最多三项。若无人通过，不读取 confirm 结果作补救选择。

Confirm 只检验 screen 锁定项，使用 `alpha=0.05/(候选数×5)` 的单侧 bootstrap 下界，并同时要求：

- error forward margin > 0；
- error gradient consensus > 0；
- error clean alignment > 0；
- error action-descent retention >= 0.5；
- paired-correct clean alignment > 0。

## 6. 工程与结论边界

- `model.eval()`；使用 `torch.autograd.grad`；optimizer steps 固定为0。
- 只载入 E4-A clean-duplicate continuation checkpoint。
- 每个 formula 的 target 与两条 matched-random control 使用相同候选参考。
- 输出临时目录完整写入后原子发布。
- 单卡 sbatch，明确 `--gpus=1`，不额外申请超过分区默认值的内存。
- E4-B2 不产生新 embedding。即使 confirm 通过，也只授权编写一个小规模梯度手术 overfit 协议，不直接授权大训练、P3或多折。

