# E6：把已验证改错峰动作直接蒸馏进共享 DreaMS embedding

## 目的

E5C 证明固定的 `candidate_gradient + role_confounder` 直接噪声训练可以改变模型权重，
但在 held formula fold 上只有约 `+0.54 pp`。后加的固定 intensity 引导没有显著增量。
这不能否定噪声微调；它说明把每个样本不同的有效动作压成一个全局固定动作，会丢失此前
S3A/A4 矩阵中最有价值的异质性。

E6 不使用 P2b、RAW/token 重排器或推理期候选专家。它复用 R1 已冻结的 882 个
训练图官方错误：每个错误对应一个在 S3A/A4 中真实把该 query 从错误变为正确的原始谱图
峰操作。该操作只用于非 held formula 的训练样本选择。

## 训练语义

对每个训练样本，同时将以下谱图送入同一个可训练 DreaMS：

1. 原始 clean query；
2. 对原始峰执行 R1 `target_path + attenuation` 后的 augmented query；
3. 同分子真实 positive spectra；
4. 困难错误候选及其他 hard negatives。

损失由 clean groupwise ranking、augmented groupwise ranking、clean/augmented consistency、
官方 margin floor 和 clean embedding preservation 组成。只解冻最后一个 Transformer block
与官方 projection head；训练时 backbone 保持 `eval()`，因此没有 dropout 混杂。

推理时只输入普通 clean spectrum，query 和 reference 共享同一套新权重，输出新的 embedding。

## 泄漏边界

- 公式五折在动作选择前冻结；held formula 的动作和结果完全不进入训练。
- R1 的 `teacher_rank/teacher_margin` 只决定训练集里采用哪个 raw-spectrum augmentation；
  随后从训练表删除，不进入 loss 或 sample weight。
- held 评价覆盖该 fold 的全部 clean queries，不只覆盖“教师可救”的 query。
- P2b、候选重排分数和 P3 均禁止使用。
- `3.694 pp`（882/23876）是训练图动作覆盖上界，不是模型性能承诺。

## 首轮严格配对设计

六个 arm 构成完整对照：

| 动作选择 | 每身份视图 | safety 权重 |
|---|---:|---:|
| fixed S3A curriculum | 2 | 2 |
| fixed S3A curriculum | 4 | 2 |
| fixed S3A curriculum | 4 | 4 |
| outcome-mined S3A+A4 | 2 | 2 |
| outcome-mined S3A+A4 | 4 | 2 |
| outcome-mined S3A+A4 | 4 | 4 |

这样每一个 outcome-mined arm 都有同视图数、同 safety 权重的 fixed 对照，避免把采样量或
安全正则差异误判为动作选择收益。

## 进入多折的硬门

主比较为 `outcome-mined, views=4, safety=2` 对
`fixed, views=4, safety=2`。只有同时满足以下条件才进入五折多 seed：

1. held clean Recall@1 相对官方 DreaMS 为正；
2. formula-cluster bootstrap CI 下界大于 0；
3. corrected 大于 introduced，且风险净收益为正；
4. near Recall@1 和 MRR 不下降；
5. clean embedding preservation 不低于 0.995；
6. 相对 matched fixed control 的 paired formula CI 下界大于 0，且净修正为正。

若失败，结论只限定为“当前共享编码器容量/损失无法吸收训练折的 outcome-mined 动作”，
不能反推 S3A/A4 的峰动作无效。下一步应先检查训练折 action transfer、梯度夹角与容量，
不能再次转向下游重排器来替代 embedding 微调。

## E6 实测裁决（2026-08-28）

六个 arm 均完成，无运行错误。结果否定了 `outcome_mined` 作为共享 embedding 训练分布，
但确认了 fixed replicated policy 的直接噪声微调信号。

| arm | held Recall@1 delta | corrected / introduced | preservation | formula CI |
|---|---:|---:|---:|---|
| fixed, views=2, safety=2 | +0.4052 pp | 30 / 6 | 0.99517 | [0.102, 0.812] pp |
| fixed, views=4, safety=2 | +0.5065 pp | 39 / 9 | 0.99163 | [0.146, 0.990] pp |
| fixed, views=4, safety=4 | +0.4727 pp | 36 / 8 | 0.99425 | [0.137, 0.924] pp |
| mined, views=2, safety=2 | 0.0000 pp | 8 / 8 | 0.99769 | CI crosses zero |
| mined, views=4, safety=2 | +0.0507 pp | 14 / 11 | 0.99539 | CI crosses zero |
| mined, views=4, safety=4 | +0.0169 pp | 10 / 9 | 0.99694 | CI crosses zero |

相同 views/safety 下，mined 相对 fixed 分别为 `-0.4052/-0.4559/-0.4559 pp`，
三个 formula-cluster CI 均完全低于 0。因此差异不能归因于学习率、安全权重或训练步数。

### 根因

`outcome_mined` 的 711 个训练动作由以下来源组成：

- A4 个例 exact-single-peak：434（61.0%）；
- S3A `role_shared`：192（27.0%）；
- S3A `candidate_gradient`：46（6.5%）；
- S3A `role_unmatched`：25（3.5%）；
- S3A `role_confounder`：14（2.0%）。

这与 S3A 全局矩阵的结论方向相反：`candidate_gradient` 和 `role_confounder` 才具有跨样本净正效应，
而 `role_shared` 全局显著增加错误。逐 query oracle 从一个全局有害家族中挑出的少数成功个例，
不构成可迁移规律。这是选择偏差，不是“大梯度”。

训练日志进一步确认：outcome-mined action 在第 1 epoch 已有约 96% 的 augmented margin 为正，
到第 4 epoch 约 98%，所以 augmented ranking 本身几乎是容易样本；clean/action cosine gap 却约 0.159，
是 fixed action 的约 8 倍。模型可以在训练 query 上把 clean margin 从负值推到正值，但该方向高度异质，
在 held formula 上只产生 14/11 的不稳定翻转。相反，fixed action 第 1 epoch augmented pass 仅约
16--20%，提供的是跨 identity 一致的困难梯度，并在 held fold 得到显著净修正。

### 决策

1. 永久停止“只要某 query 被某动作改正，就把该动作并入共享训练”的 oracle union。
2. `role_shared` 不得进入直接微调主线；A4 单峰动作必须先通过家族/化学层面的跨 identity 复制门。
3. 保留 `candidate_gradient + role_confounder` fixed curriculum，当前它是真实、显著的共享 embedding
   改进基线；E5C 的 `+0.5403 pp, 38/6, preservation=0.99524` 仍是当前单折最优安全配置。
4. 下一扩展只能加入全局复制为净正、方向对照显著、且能增加 identity/formula 覆盖的动作家族。
   首选 `recurrent_union_mix`（固定动作审计 294/27），但必须进行低权重、严格 paired 的增量试验；
   `consensus_projection` 因 456/187 风险较高且 E5C intensity 增量失败，降级。
5. 所有后续结果必须同时报告相对官方 DreaMS 和相对 fixed-noise 基线的 paired formula CI。

## E7 recurrent-union 增量裁决（2026-08-28）

E7 在 `errors-only + views=2 + safety-weight=2` 的 fixed curriculum 上加入
`recurrent_union_mix, dose=0.50`：

- fixed control：Recall@1 `+0.4052 pp`，29 corrected / 5 introduced；
- weight 0.025/0.05/0.10：均只比对照多修正1条、0新增；
- weight 0.20：比对照多修正3条、0新增，但 preservation=0.99412；
- 所有配对增量 CI 下界均为0，没有权重通过 multifold 门。

因此 recurrent transfer 方向低风险但增量太小，应停止权重扫描。还必须注意：E7 为了
错误聚焦把成熟的 `all + views=4` 改成了更弱的 `errors-only + views=2`，所以不能用
E7 的约0.4 pp否定已经通过15次 formula-held-out 运行、三seed均值 `+0.635 pp`
的 direct shared-embedding 基线。

## E8：动作方向迁移与参考漂移的因果消融

代码审计确认，历史 `direct_action_loss` 仍有两个未单独裁决的机制：

1. clean/action consistency 对两端同时反传，可能把 beneficial action 方向一并拉回；
2. 正负 reference 也接收 rank 梯度，优化器可能移动少数 reference 来降低损失，
   而不是让 clean query 学到动作方向。

E8 固定成熟学习率、完整非held分区、`action_scope=all`、views=4、同一seed/fold，
预注册以下因果分解：historical symmetric/shared；student-action stop-gradient；
frozen official raw-action target；official reference anchors；两项修正联合；以及联合修正下
candidate terminal 与 candidate+confounder terminal 的策略分解。

所有 arm 仍保存一个共享 DreaMS encoder，P2b 禁止，P3不使用，推理只输入 clean spectrum。
只有联合修正相对历史对照的 formula-cluster CI 下界大于0、near不降、corrected大于
introduced且 preservation>=0.995，才允许进入多折。否则必须接受约0.6 pp是当前
固定动作的可转移水平，不得把3.85 pp事后动作oracle写成权重性能。

### E8 实测裁决

成熟对照 `symmetric/shared + curriculum` 在 held fold 0 得到 `+0.5740 pp`，
38 corrected / 4 introduced，near `+0.6092 pp`，preservation=0.99527。因果消融结果为：

- `student_action_stopgrad/shared` 与对照 Top-1 完全相同（1 corrected / 1 introduced 的互换）；
- `official_action/shared` 与对照逐 query Top-1 完全相同；
- 将 reference 冻结为 official embedding 后，相对成熟对照下降 `-0.2195 pp`，
  formula-cluster CI 全负；
- candidate terminal 与 combined terminal 相对 curriculum 分别下降 `-0.3714/-0.3208 pp`，
  formula-cluster CI 全负。

因此 E8 排除两项旧嫌疑：对称 consistency 没有抹掉可用方向，共享 reference 更新也不是
虚假的捷径。相反，共享 query/reference 几何与多步 curriculum 是当前净收益的必要组成。
不得继续扫描 stop-gradient、official-action target、official-reference anchor 或 terminal-only。

### E9：动作陈旧性审计

代码对账发现另一个更上游的错位：原始 S3A 在每个删峰步骤都用当前查询状态重新计算
最难负候选、峰角色和输入强度梯度；E4-A/E8 则把 official DreaMS 上一次性挖出的
`target_path` 固定重放四个 epoch。学生 embedding 与困难负例发生变化后，训练动作从未更新。

E9 固定成熟 E8 权重，在已消费的 held formula fold 上忠实重跑 S3A 选择器，并逐 action 比较：

1. frozen path 与 current-student path 的 exact/first-token/Jaccard；
2. 当前最难负谱是否漂移；
3. 两条路径在同一 student、同一完整候选图上的 margin、corrected/introduced；
4. online-minus-frozen 的 formula-cluster Top-1 与 margin CI。

只有路径/负例确有漂移，且 online 相对 frozen 的 Top-1 不降、修正不少于新增、Top-1 或 margin
的 formula CI 下界大于0，才允许 E10 做 epoch-wise online re-mining。该方案仍是噪声微调：
候选只在训练期构造峰扰动，推理时仍由一个共享 DreaMS encoder 将 clean spectrum 映射为 embedding；
P2b 与 P3 均禁止。

### E9 实测裁决

E9 严格复现成熟 E8 held ranks（0 mismatches）。旧路径与当前学生路径只有44.5%完全相同，
但首峰一致率87.96%、平均Jaccard=0.833、最难负谱一致率87.10%，说明变化主要发生在后续低优先级峰。
在线路径相对冻结路径仅10 corrected / 9 introduced，Top-1 `+0.0119 pp`；formula CI
`[-0.1655,+0.2019] pp`。margin增量同样跨0。candidate-gradient为10/8且不显著；
role-confounder为0/1。动作陈旧性假设因此未通过，不启动epoch-wise online re-mining。

下一责任问题必须按query而不是action-row回答：成熟E8之后，允许no-op并从全部成熟步骤中为每个query
选择最佳动作时，究竟还剩多少独立错误可恢复。E9-B先计算这一严格oracle上限；若它不足以补足
五点目标，继续训练selector或扫描学习率在数学上都不可能达到目标，必须先扩展可复制动作家族。
