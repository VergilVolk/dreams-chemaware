# Noise 微调的真实提升瓶颈与下一轮决策（2026-08-30）

## 结论先行

当前瓶颈不是“没有有效峰动作”，也不是“学习率还不够大”。已经证实存在两层能力：

1. **共享 embedding 的可重复增益**：E4-A 在 5 formula folds × 3 seeds 上平均
   `+0.635 pp Recall@1 / +0.522 pp near`，15/15 个 held-fold CI 下界为正；
2. **动作空间的 outcome-aware 上限**：互补的 N/P/no-op 动作在已消费开发折上可达到
   约 `3.35–4.93 pp` 的逐查询上限。

两者之间约四个百分点的缺口是 **action-to-weight transfer gap**：逐查询、候选感知、
看过动作结果后选择的局部纠错，尚未被压缩成一个只读取 clean spectrum、同时编码 query
和参考谱的共享 DreaMS 函数。

## 一、证据层必须严格分开

| 层级 | 最强已确认结果 | 它证明什么 | 它不证明什么 |
|---|---:|---|---|
| 固定共享 encoder | E4-A 平均 `+0.635 pp` overall，`+0.522 pp` near | 固定 candidate-gradient + role-confounder 噪声能改变并改善共享 embedding | 不等于动作 oracle 的 3–5 pp |
| 单个固定 P 动作 | recurrent union 等在冻结/成熟几何上约 `+0.8–1.1 pp` | 真实重复谱可提供正证据补全方向 | 施加动作后的即时结果不等于 clean 推理权重增益 |
| no-op-aware 动作并集 | 约 `3.35–4.93 pp` | 若已知每条 query 哪个动作有效，监督空间足够大 | 不是可部署教师、不是模型、不是可直接蒸馏的统一函数 |
| P2b | 独立候选专家 | embedding 后仍可增加谱学判别 | 与本文件的 embedding 微调严格正交 |

因此，不能再用“动作 oracle 有 4.93 pp，所以学生理应得到 4.93 pp”推导训练失败；
也不能因为学生目前只有 0.6 pp，就否定动作空间。真正要测的是转移率。

## 二、提升瓶颈的因果排序

### 1. 最大瓶颈：逐查询 oracle 不是一个可学习的统一噪声分布

历史 3–5 pp 上限允许每条 query 在多个动作、剂量、参考策略与 no-op 之间读取结果后选择。
这相当于函数：

`action = f(clean spectrum, true positive, wrong candidates, observed outcomes)`。

部署时共享 encoder 只有 clean spectrum。它必须学习：

`embedding = g(clean spectrum)`。

前者包含真候选、错候选和事后结果，后者没有。若把前者所有成功个例直接堆入训练，
同类 clean 输入会收到异质甚至互相抵消的目标。E6 已经直接证明这种选择偏差：
outcome-mined 动作在增强视图上几乎都容易排对，却不能在 held formula clean query 上迁移；
相反，全局重复为净正的 fixed candidate-gradient/role-confounder 可以稳定迁移。

### 2. P-arm 的信息缺口比 N-arm 更难

1,805 个官方错误中，1,439 个包含 positive deficit。N-arm 可以从当前谱图与错候选中识别
“过强的混淆证据”并压低它；P-arm 的 recurrent/consensus 动作则使用同身份真实参考谱补回
clean spectrum 中没有的峰或强度。学生若只看到 clean spectrum，部分目标在信息论上不可辨认。

这解释了：

- candidate-gradient/role-confounder 固定策略能转移；
- global positive-guided dose 能产生动作优势，但安全正则一加回，增量接近零；
- 更大 guided weight 主要扩大 embedding 漂移和 near 异构体新增错误。

P-arm 要进入权重，动作选择必须首先在 formula 隔离下表现为 **跨身份可重复的规律**，
而不能只表现为该 query 的事后成功。

### 3. 当前联合训练把稀疏 P 梯度稀释在成熟 N/safety 梯度中

E5 日志已经显示 guided weight=1 时几乎每步触发全局梯度裁剪，只保留约 9–19% 原始范数；
固定 N-only 的保留比例明显更高。现有训练器分别 backward N、P、safety 后统一 clip，
只报告合并后的梯度范数，无法知道：

- P 梯度是否被 N/safety 支配；
- 哪些 query 的 P 与 safety 方向冲突；
- 在同一动作家族内，corrective 与 harmful control 是否相互抵消。

因此下一轮不能靠继续扫 guided weight、LR 或 safety weight解决。必须先做分支梯度审计，
再采用分支归一化或交替更新；否则参数变化只反映最大损失项。

### 4. 现有 E14 的 teacher target 过于激进且缺少动作特异的反例

现有 selected teacher 只保留 `clean wrong -> action rank 1`，并把 action 的绝对 margin 当 clean
目标。它没有同时提供：

- 同一个 action 在 mature-correct query 上造成 introduced error 的反例；
- official wrong 已被 mature E4-A 修正后必须保护的 hard-protected 集；
- action margin 相对 clean margin 的置信度/复制性权重。

更合理目标是保守的 **margin delta transfer**，而不是复制绝对 action margin：

`m_target = m_clean + rho * clip(m_action - m_clean, 0, cap)`，其中 `rho < 1`。

动作特异 harmful/protected controls 必须与 corrective examples 同批出现；否则共享 encoder
只学“看到该模式就移动”，无法学到“哪些相似谱不能移动”。

### 5. 共享几何和 preservation 构成真实但次级的容量约束

共享 encoder 会同时移动 query、positive 和 negative。冻结参考时看似很大的 query margin
改善，重编码参考后可能抵消。另一方面 preservation>=0.995 是必要安全线：历史新增错误
几乎全部集中在同分子式 near 异构体，过大位移会先破坏本来正确的局部边界。

提高 LR、解冻更多层或放松 preservation 可以增加 corrections，但现有证据显示 introduced
与尾部漂移同步增加。容量升级只能在动作标签可复制、分支梯度可辨认之后进行。

## 三、明确排除的错误解释

1. **不是单纯 LR 太低**：2e-6/1e-5 已是 E4-A 的稳定最优区间；更强剂量主要增加漂移。
2. **不是动作不存在**：固定矩阵和方向对照已证明多个动作有正特异性。
3. **不是 head-only 表达力问题的翻版**：E4-A 已实际更新最后一层 Transformer 与 head 并获得稳定权重增益。
4. **不是 P2b 能替代的问题**：P2b 是下游专家，不能证明 embedding 变好。
5. **不是把全部七层解冻就能解决**：标签异质和信息缺口未解时，增加容量只会更快记忆训练动作。

## 四、E14 审计发现与修正

### 已发现的实现错误

原 teacher 把每条 selected query 的 `formula_fold` 错写为被排除的 `outer_fold`。训练器会重算
真实 fold 并报错，所以旧版 E14 无法按合同执行。现已改为 materialize query 自己的真实 fold，
并增加回归测试。

### 已发现的策略错误

旧版代码声称使用“60 个经过筛选的动作”，实际把 E10B/E11/E12B 的全部 recipe 都交给
逐查询 oracle；这会再次从全局有害动作中挑少数成功个例。现已改成两级准入：

1. 只加载三份正式报告的 `passing_fixed_cells`；
2. 在当前 outer-train 几何的四个 formula 子折中，要求动作至少修正 10 条、总
   `corrected > 2*introduced`、每折 risk-net 不为负且至少两折严格为正。

只有通过两级门的动作才能用于逐 query teacher 选择。验证器还会检查 selected action 全部来自
replicated-safe 集合。

### 名称边界修正

当前 teacher checkpoint 与动作查询均排除了 student outer fold，所以 outer held evaluation 是干净的；
但 teacher 在 outer-train 内读取动作结果，因此不是“每条训练查询的 OOF 预测”。正式名称改为
**outer-fold-isolated privileged teacher**。

## 五、下一轮不是大扫参，而是三段式转移实验

### E14-M0：安全动作教师重建

先运行五个 outer folds，必须输出：

- prior-safe / current-geometry replicated-safe 动作数；
- 每动作 corrected、introduced、risk-net 和四个 inner formula fold 的 risk-net；
- selected query/identity/formula 数；
- selected action family 分布与 margin-delta 分布；
- outer held formula overlap=0、P3未使用、P2b禁止、动作 replay 精确。

任何 fold 没有足够的 replicated-safe actions 或 selected identity/formula 覆盖，停止，不训练。

### E14-M1：一折、严格配对的转移诊断

必须保持同一个成熟 E4-A 初始化、同一个 N/safety stream、同一个随机顺序，比较：

1. warm continuation control；
2. replicated-safe selected actions，仅 clean/action rank；
3. 加保守 margin-delta transfer；
4. 第3项 + action-specific harmful/protected controls。

在训练前先报告各分支独立梯度 norm、cosine、clip contribution。若 P 梯度在归一化前比 N/safety
小一个数量级以上，采用交替 P step 与 N/safety maintenance step，不再先求和后统一裁剪。

### E14-M2：五折多 seed

只有第4臂相对 warm control 同时满足以下条件才允许扩大：

1. formula-cluster CI 下界 > 0；
2. incremental corrected > introduced，且 corrected−2×introduced >0；
3. near、positive-deficit、cross-condition 均不退化；
4. preservation vs mature mean >=0.995，并报告 p01；
5. harmful/protected action controls 的 margin 不下降；
6. 相同效果在至少两个 seed 方向一致。

完成五折多 seed 后，才讨论是否进入封存评估。3–5 pp 仍是研究目标，不是下一轮的通过阈值；
下一轮首要目标是证明 P-arm 对成熟 E4-A 有显著、可重复的 **增量转移**。

## 六、当前可执行裁决

1. 保留 E4-A 作为 noise embedding 的正式基线和初始化；
2. 停止固定 P 剂量、guided-weight、safety-weight 与 LR 大扫；
3. 停止使用未经过 fixed-cell 和跨 formula 复制门的 outcome-selected actions；
4. 先重建修正后的 E14-M0；
5. M0 通过后，补齐 conservative delta target、action-specific risk controls 与分支梯度审计，
   再启动 M1 共享 encoder 微调。

这条路线仍然只输出新的 DreaMS embedding 权重；P2b、候选后处理和 P3 均不进入教师、特征或损失。
