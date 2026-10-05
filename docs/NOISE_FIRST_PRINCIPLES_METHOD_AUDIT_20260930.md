# Noise shared-encoder：近期方法的第一性原理审计

日期：2026-09-30  
范围：Noise native Stage-1、Stage-2 residual、relation T1/T3 V1/V2/V3、未提交的 V4 草案，以及 P2b/RRF 对 encoder 路线的启示。  
目标：解释为何现有路线反复出现 margin 上升但 Recall@1 只小幅上升或下降，并判定什么才可能继续逼近 `+5 pp`。

## 1. 先定义真正要优化的量

部署时没有 action view、teacher 或 reranker。共享 encoder `f` 对 query 谱和候选 reference 谱编码，候选分子得分是其多张 reference 谱中的最大余弦：

```text
score(q, molecule m) = max_(s in references(m)) cosine(f(q), f(s))
prediction(q)        = argmax_m score(q, m)
```

因此 Recall@1 只由真分子和当前最高 rival 在最大值边界附近的次序决定。平均 positive-negative margin 不是充分统计量：大量容易关系继续被推远可以显著提高 mean margin，却不能修正 rank-1 尾部；同时共享 reference 的移动可能让其他 query 翻转。

以 corrected held fold-0 的 `18,333` queries 计：

- official Recall@1 约 `93.187%`；
- 当前最高 encoder 点估计 V1 seed-3407 为 `94.2399%`，错误约 `1,056`；
- 若目标是相对 official `+5 pp`，需达到约 `98.187%`，即从当前再净修约 `724` 个 query，消除约 `69%` 的剩余错误；
- 现有任一次 continuation 的净修正只有 `69–102` 个量级，距离目标不是一个学习率或一次额外 epoch。

## 2. 监督信息到底来自哪里

Noise 目前只有三类可能的信息：

1. **动作增强的身份不变性**：把带噪/删峰/混淆后的 action view 与同分子实测谱拉近。这是真正新增的信息，也是 Stage-1 targeted 对 matched control 明显获胜的来源。
2. **实测同身份多谱关系**：同一分子的不同实测谱提供 hard positive，能拓宽分子内谱图变化范围。
3. **真实候选竞争关系**：当前 encoder 下 molecule-max 得分最高的错误分子，是与部署 Recall@1 最直接一致的 hard negative。

固定旧 negative、均匀采样大量容易 clean relation、或把旧 action 数量简单扩大，都不会创造第四类信息。它们只能改变已有梯度的剂量。

## 3. 各方法逐项裁决

| 方法 | 实际监督与目标 | 已知结果 | 第一性原理裁决 |
|---|---|---|---|
| Stage-1 native hard-positive | targeted action anchor、同身份 measured positive、错误候选 measured negative；原生 cosine triplet hinge | targeted vs official `+0.49637 pp`；targeted vs matched control `+1.26548 pp`，CI `[+0.87586,+1.65425]` | **成立且应保留。** 它证明 action content 有因果信息；但绝对净增仅 91 query，`risk_net_lambda2=-127`，说明局部纠正伴随大量新错。 |
| Stage-2 residual | 在 Stage-1 当前几何中筛 active 且 targeted 优于 control/clean 的单动作 residual；小步 continuation | 对 Stage-1 仅约 `+0.1036 pp`，约 19 个净 query | **说明旧 action 的可收割残差很快饱和。** 继续在同一旧边界中挖更细，不可能自然放大到 5 pp。 |
| T1/T3 V1 | 48,544 clean queries；2,730 action queries、19,894 actions；multi-hinge + truncated molecule-max listwise | 对 Stage-1 主种子 `+0.55637 pp`，复证 `+0.37637 pp`；当前最高 `94.2399%` | **有真实几何增益，但 action 归因不成立。** action 名义全局剂量仅 `2.81%`，exact row bridge 只完整保留 `47.15%`；增益主要证明 relation curriculum 可用，不证明是完整 Noise action 注入。 |
| T1/T3 V2 | clean stream 加每个 exact action 独立一步；19,894 events 来自 2,730 queries | fold-1 clean-only `-0.830 pp`；action rotation `-1.487 pp` | **剂量单位错误。** 平均 7.29 action/query，动作多的 query 被重复更新；optimizer step 也未与 clean control 匹配。恢复 exact provenance 不等于恢复正确学习信号。 |
| T1/T3 V3 | 每 query 一次；action query 内 `0.5 clean T1/T3 + 0.5 exact action hinge`；matched target/control | targeted `-0.7705 pp`、control `-0.9855 pp`；targeted-control `+0.2150 pp`；两臂 margin 均约 `+0.070` | **query dose 修好了，但全局信号仍错。** 只有约 2,730/48,544 query 含 action，因此 action 名义全局剂量仍约 `2.81%`；约 `97.19%` 是两臂共同 clean continuation。共同 margin 大涨、排名下降，直接证明目标与部署尾部错位。V3 还错误地从 Stage-1 而不是当前 V1 冠军开始。 |
| V4 草案（未提交、已撤销） | 只训练 action queries；`0.5 clean exact hinge + 0.5 action exact hinge` | 未运行 | **仍是错误范式。** “50% action”只是名义 loss 系数，不是梯度份额；clean hinge 不是 preservation；单一旧 p/n 与 molecule-max 全候选部署错位；rotation-0 只用约 13.7% exact events；未在 V1 几何重挖。不得提交。 |
| P2b | DreaMS/entropy/neutral-loss 的固定局部候选融合，不更新 encoder | formula-isolated OOF `+3.91 pp`；sealed P3-main `+1.07 pp`；near-core 有害 | **证明原始谱证据含 encoder 未收录的信息，但不是共享 encoder 增益，且不可无条件部署。** |
| strict RRF | DreaMS、cosine、top-k overlap 的固定 rank fusion | official geometry 探索 `+1.77 pp`；叠 Stage-1/GNPS 确认失败 | **仅是机制探针。** 固定朴素排名信号分布绑定，不能作为 encoder 或稳定 reranker 结果。 |

## 4. 为什么反复出现 margin 上升、排名下降

这不是偶然，也不是单一代码 bug，而是五个结构问题叠加：

1. **均值目标与尾部指标错位**：hinge/listwise 在大量容易 query 上继续增加 margin；Recall@1 只看边界附近的少数翻转。
2. **固定 negative 很快过时**：encoder 更新后 molecule-max 的 argmax rival 会切换。改善旧 negative 只会暴露下一个 rival。
3. **共享 reference 造成 see-saw**：同一 reference 谱被许多 query 复用。为 query A 移动它，可能让 query B 从正确变错误；现有 query balancing 没有归一 reference/molecule multiplicity。
4. **全角色对称更新缺少稳定约束**：clean/action/positive/negative 全部 live 能传递信号，但也让候选库整体漂移。局部 clean hinge active 时继续移动所有角色，inactive 时又完全无法保护旧排序。
5. **旧几何监督失效**：Stage-1 action 的 exact negative 和动作活性是在旧 encoder 下确定的。V1 已显著改变几何；不重新编码与重挖，就不知道某个动作是否仍 active、是否仍优于 control、是否仍对应当前 top rival。

V3 的数字是最直接的实证：targeted 与 control 的 mean margin 分别增加 `+0.070779/+0.070080`，但 Recall@1 分别下降 `0.7705/0.9855 pp`。动作内容方向仍略正（targeted-control `+0.215 pp`），然而共同漂移远大于动作收益。

## 5. 对“全部动作”和“大剂量”的纠正

“把所有动作都用到”不是充分条件。若多个动作来自同一 query，它们高度相关；按动作数给 optimizer dose 会把动作 multiplicity 错当独立样本量。相反，每 query 随机只取一个动作也不够，因为可能抽到 V1 当前几何中已经 inactive 或过时的动作。

正确原则是：

- query 是统计剂量单位；
- action 是 query 内候选增强，不是独立样本；
- 必须在**当前冠军几何**中评价全部动作；
- 每 query 只选择当前仍提供真实 residual 信息的动作，或在 query 内按当前有效性归一；
- 报告实际 active fraction、loss 份额、参数梯度范数与 targeted/control 梯度差异，不能把 loss 系数 `0.5` 写成“50% 信号进入 optimizer”。

## 6. 当前 5 pp 瓶颈的真实位置

现有方法已经证明 action 能提供信息，但没有解决两个决定净增的量：

```text
净修正 = corrected - introduced
```

Stage-1 为 `309-218=91`；T1/T3 主种子相对 Stage-1 为 `288-186=102`。两次都能纠正约四分之一当前错误，但同时引入大量新错。要从当前 `94.24%` 到相对 official `+5 pp`，需要约 `+724` 净 query；这要求同时显著扩大当前 residual error 覆盖，并把 corrected/introduced 交换率提高数倍。

因此瓶颈不是“动作数量不足”或“梯度有没有到 encoder”，而是：

> 如何只移动当前错误/近边界关系，同时防止共享候选和已正确 query 被连带翻转。

## 7. 唯一仍有第一性原理依据的下一代方向

下一步不应是 V4 的局部 triplet 放大，而应是 **V1-current dynamic residual ranking with stability**：

1. 从当前最高 V1 seed-3407 encoder 开始，在训练公式上重新编码完整 query/candidate 图。
2. 按部署公式计算每个 query 的完整 molecule-max 排名，区分当前错误、低 margin 正确、高 margin 正确三类。
3. 对每个 residual query，在 V1 当前几何中重新评价全部 registered actions；动作必须仍 hinge-active，并在完整候选边界和相同 exact relation 上均优于 matched control。**不得要求 action margin 优于 clean**：hard-positive action 本来就应比 clean query 更难；与 clean 的差值只作难度诊断，不能作为淘汰门。
4. positive 使用当前最难但身份可靠的多张同分子实测谱；negative 至少同时包含当前 V1 molecule-max top rival、原 exact negative 和真实 near/subset rival。合成 negative 暂不使用。
5. correction loss 只作用于当前错误/近边界 query，并直接围绕当前 molecule-max rival；每轮更新后重新挖一次，避免 fixed-negative 失效。
6. 对已正确 query 和高复用 reference 加入冻结 V1 排名/embedding 的 trust-region preservation，专门压低 introduced。它不是把 teacher 当新答案，而是限制 continuation 不得破坏当前冠军已经正确的关系。
7. query 与 reference/molecule 两层都做 multiplicity 归一；matched control 共享 query、关系、batch、seed、optimizer state 和 step，唯一差异仍是 targeted/control action content。
8. 预注入必须在真实选中事件上直接比较 backbone/head 参数梯度，而不是只检查 action 数组不同或叶子 embedding 有梯度。

这条路线同时使用 Stage-1 已证实的 action 新信息、T1/T3 的 molecule-level 部署对齐，并正面处理 introduced/see-saw。它仍不能保证 5 pp；在重建训练折 residual atlas 并估计可修错误覆盖前，任何“保证 5 pp”的承诺都没有科学依据。

## 8. 当前冻结裁决

- 保留：Stage-1 action-specific 因果结论；V1 seed-3407 当前最高 encoder 点估计；P2b 的限定面板证据。
- 作为机制证据保留：Stage-2 小 residual、RRF 信息互补探针。
- 停止：V2 one-action-one-step；V3 全量 clean continuation；未提交 V4 局部 hinge 放大。
- 在完成 V1-current full-candidate residual atlas、真实 action 活性与梯度审计前，不再提交新的 GPU 微调作业。
