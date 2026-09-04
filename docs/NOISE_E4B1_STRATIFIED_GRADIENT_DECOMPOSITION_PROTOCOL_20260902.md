# Noise E4-B1：分层动作梯度分解协议

日期：2026-09-02  
状态：预注册；零权重更新；E4-B0 后续机制定位

## 冻结起点

E4-B0 已经排除“动作信号消失”和“梯度过小”：target 相对 frozen matched-random 的 margin 优势在官方初始化和 clean-continuation checkpoint 均约为 +0.0103。失败发生在 pooled gradient：跨 formula 共识不为正，且 paired-advantage 与 clean 排序梯度不稳定对齐；现有 target-minus-random 分支则与 clean 梯度显著反向。

E4-B1 不训练模型。它只回答 pooled failure 究竟由哪个动作 cell、基线状态和模型层产生。

## 固定面板

- 只审计 E4-A clean-duplicate continuation checkpoint；官方初始化的总体结论已经由 E4-B0 冻结。
- outer formula fold 0 的训练侧；P3 与 P2b 禁止使用。
- 九个成熟 cell：candidate-gradient 0.50 的 step 3–6，role-confounder 1.00 的 step 1–5。
- 每个 cell 分成 `official_error` 与 `official_correct` 两个预注册状态。
- 每个 cell×state 确定性选择 32 个 formula，每个 formula 只取一个 query；共 18×32=576 actions。
- 选择只使用 cell、formula、query 与官方基线状态，不使用 target/random action outcome。
- 每个 target 与两条 frozen matched-random control 共享完全相同的正负候选参考。

## 分层轴

主检验单位为 `selector × step × baseline_state`，共18组。附加描述轴为：

- selector × baseline state；
- selector × score-error family；
- selector × near/non-near × baseline state；
- head 与最后一个 backbone block。

错误机制来自冻结 real-error signature，不在本阶段重新拟合阈值。

## 主检验

对每个主组报告：

1. target-minus-selected-random margin；
2. paired-advantage gradient 与 same-query clean gradient alignment；
3. paired-advantage gradient 的 leave-one-formula-out consensus；
4. head/backbone 分层 alignment 与 consensus；
5. 现有 target-minus-random branch 的 clean alignment；
6. 梯度范数比与 clip=1.0 的缩放。

每组的统计单位是 formula。除普通95% formula bootstrap CI外，候选资格还使用对18组×3个主终点的单侧 Bonferroni 下界：`alpha=0.05/(18×3)`。不得根据结果删除 cell 或更改分层。

## 候选资格与停止规则

只有 `official_error` 组同时满足以下条件，才进入后续小型条件化 overfit 设计池：

- target-random margin 的 multiplicity-adjusted lower bound > 0；
- gradient consensus 的 multiplicity-adjusted lower bound > 0；
- clean alignment 的 multiplicity-adjusted lower bound > 0。

E4-B1 本身不授权训练。如果无组通过，则停止把现有九格动作作为统一 shared-encoder 教师；优先扩展谱图可观测的条件特征或重新设计动作，而不是调学习率、增加 epoch、扩大解冻层数或提高 loss 权重。

## 工程边界

- `model.eval()`，仅求导 head + 最后一个 transformer block；
- `torch.autograd.grad`，无 optimizer、无 `backward()`、无权重更新；
- 每个 action 的大梯度在统计后立即释放；只在 CPU 保存18个主组的梯度和；
- 输出临时目录完整写入后原子发布；
- 单卡 sbatch，使用分区默认单卡主存，不额外申请超过38GB。

## 结论边界

该结果只能定位可学习子空间和冲突来源，不是新 embedding、检索提升或 P3 结果。
