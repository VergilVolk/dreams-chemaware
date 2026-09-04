# Noise E4-B0：目标动作梯度归因协议

日期：2026-09-01  
状态：预注册；只做梯度审计，不更新权重

## 1. 已冻结的因果起点

E4-A 三臂实验只有一个工程差异：同一 query 的 action view 分别为 clean duplicate、两条冻结匹配随机路径中由哈希预先选定的一条、或 R0 定向路径。三臂共享初始化、训练 query、候选参考、采样键、优化器和训练日程。

结果中，targeted 相对 matched-random 仅多净改正 2 个 query（+0.0338 pp），formula bootstrap 下界为 0；因此不能继续扩大训练或调参。与此同时，targeted action view 的训练 margin 确实上升，说明下一问题不是“动作有没有作用”，而是“动作产生的梯度为什么没有稳定转移到 clean embedding”。

## 2. E4-B0 唯一问题

在不执行任何 optimizer step 的条件下，分别于：

1. 官方 DreaMS 初始化；
2. E4-A clean-duplicate continuation checkpoint；

测量同一 query、同一正负参考、同一定向动作和两条冻结匹配随机动作产生的梯度。区分四个对象：

- 同 query 的 clean-rank、margin-floor 与 official-preservation 公共梯度；
- E4-A 原目标 action branch 梯度；
- 原目标 branch 减冻结随机 branch 的差分梯度；
- 显式 paired advantage：要求 target margin 至少高于预先选定随机 control 0.01。

这四者不得再合并成一个含义含混的“教师梯度”。

## 3. 固定样本设计

- 只使用 outer formula fold 0 的训练侧；held fold、P3、P2b 均不进入。
- R0 action manifest 不含 outcome 字段；selection 不读取 corrected、introduced、target rank 或 target margin。
- 32 个 formula cluster，每个 formula 固定 4 个不同 query：2 个 candidate-gradient、2 个 role-confounder，共 128 actions。
- 通过跨 formula 的确定性轮换覆盖完整 9 个成熟 curriculum cells：candidate-gradient step 3–6 与 role-confounder step 1–5。
- 每个 query 固定 4 个 positive spectra、8 个 negative molecules；target 与两条 matched controls 使用完全相同的候选参考。
- 每个 formula 单独形成一个梯度 microbatch；统计单位是 formula，不把 128 条 action 当作独立重复。

## 4. 必报量

对两个 checkpoint 分别报告：

- target minus selected-random margin 及 target minus two-control mean margin；
- 原 target branch、target-minus-random branch、paired-advantage branch 的梯度范数；
- 各分支相对于当前 target total gradient 的范数比例；
- 各分支与同 query 公共 clean gradient 的 cosine，包括 head 与 backbone 分层；
- 在 clip=1.0 下当前总梯度的裁剪比例；
- paired-advantage gradient 与其余 31 个 formula gradient 总和的 leave-one-formula-out cosine；
- 上述 formula-level 均值的 5,000 次 bootstrap 95% CI。

## 5. 唯一放行门

只看 clean-continuation checkpoint，并要求同时成立：

1. target-minus-selected-random margin 的 formula CI 下界严格大于 0；
2. paired-advantage gradient 的跨 formula consensus CI 下界严格大于 0；
3. paired-advantage gradient 与公共 clean gradient alignment CI 下界严格大于 0。

三门同时通过，才允许一个小规模 paired-counterfactual advantage pilot。任何一门失败都停止训练，并把失败定位为：动作优势已消失、跨 formula 梯度互相抵消，或动作梯度与 clean-space 目标冲突。

## 6. 工程边界

- 模型保持 `eval()`；只对与 E4-A 相同的 head + 最后一层 backbone 求导。
- 使用 `torch.autograd.grad`，不创建 optimizer、不调用 `backward()`、optimizer steps 固定为 0。
- 梯度逐支路计算后立即释放；不同时保留四套大梯度。
- clean checkpoint SHA 必须与冻结 E4-A 配对汇总中的 clean-duplicate checkpoint 完全一致。
- 输出采用临时目录写入，三个制品齐全后原子发布；失败运行不会占据正式输出目录。
- 正式作业必须申请一张 GPU；登录节点只允许提交，不运行模型。

## 7. 结论边界

E4-B0 只能确定梯度可用性、跨 formula 一致性与 clean-space 相容性。它不产生新 embedding、不构成检索提升，也不授权在门失败后继续搜索学习率或扩大训练。
