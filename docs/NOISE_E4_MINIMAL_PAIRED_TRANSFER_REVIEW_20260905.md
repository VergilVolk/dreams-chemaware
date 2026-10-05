# Noise 微调七项问题与 E4 最小改动方案

日期：2026-09-05  
状态：设计审查；尚未授权实现或大训练

## 一、不可遗忘的七项问题

1. **监督目标错配**：现有 dynamic-direct 只对 target action 再做一次 identity listwise loss，没有显式优化 target 相对 control 的优势，也没有要求 clean embedding 继承该优势。
2. **M2 预测目标错配**：M2 预测 action 在冻结 geometry 中是否有效，不是该 action 的训练梯度能否改善独立 clean retrieval。Action effectiveness 不能替代 training influence。
3. **有害动作未硬清零且标签冲突**：`zero_weight_fraction=0`；全部动作获得正权重。81 条 action 同时被标为 positive 和 harmful，正向监督集合不闭合。
4. **强制平均 action dose 稀释 no-op**：0.35/0.45/0.55/0.55 的平均剂量会把低但非零 utility 一并放大；低置信 query 不能真正退出增强。
5. **P 数量压过成熟 N**：正式 schedule 中 N 约占 17.7%，P 约占 82.3%；六个 P families 与两个 N families 直接混合，使成熟 E4 的 N curriculum 被高覆盖、高风险 P 动作稀释。
6. **全量 replay 不等于全量训练曝光**：304,209 条 action 全部被评估，但仅 49,444 条进入四轮 schedule，约 16.3%。所有动作必须被审判，但不得把 harmful/uncertain 全部当正样本训练。
7. **Control 语义被合并**：N control 是同身份 matched-random robustness view；P control 是朝 wrong reference 方向的反事实错误 view。两者不能共享同一种“正确身份 augmentation”语义。

附加但不单列为第八项：当前 conditional policy 在四轮内冻结，且 dynamic-direct 相对 E4 同时改变了 full-list loss、batch query 数、每轮 optimizer steps、action family 组成和 exposure。它不是对 E4 的单变量修补，因而失败后无法精确定位某一个改动。

## 二、为什么不采用全新 meta-learning 架构作为第一步

全量虚拟更新或元影响学习能够更直接估计 transfer，但它会同时改变训练算法、计算图、内外层拆分、梯度估计和采样单位。当前首要任务是判断成熟 E4 的 N actions 能否在修复目标表达后产生独立增量，不应先引入新的大型架构。Meta-influence 只保留为后备诊断，不作为第一轮训练。

## 三、选定方案：E4-Paired Margin Transfer（E4-PMT）

E4-PMT 直接复制成熟 E4 的网络与训练主干，只允许以下最小改动。

### 3.1 完全保持不变

- 初始化：fold-aligned mature E4 checkpoint；
- 一个 shared query/reference encoder；clean spectrum inference；
- 解冻 final Transformer block + official projection head；
- backbone LR `2e-6`，head LR `1e-5`；
- 4 epochs、batch-actions=4、views-per-identity=4；
- 每 query 4 positive spectra、8 negative molecules；
- formula holdout、identity equal weighting；
- E4 的 clean rank、margin floor、preservation、安全 query 和 gradient clipping；
- 第一轮只使用成熟 N curriculum：candidate-gradient attenuation 0.50 step 3-6；role-confounder attenuation 1.00 step 1-5。

### 3.2 只修三个训练缺口

1. **成对执行**：每个 N target 与其冻结 matched-random control 在同一 query、同一正负候选、同一 batch 中执行。
2. **硬路由**：依据 outer-train 当前 E4 replay，把动作互斥分成 corrective、robustness-only、harmful、uncertain。只有 corrective 的 target 能产生纠正梯度；harmful/uncertain 的 target 正权重严格为零。No-op 不再由平均剂量倒推。
3. **Action-conditioned clean margin floor**：把 E4 原本的 clean margin floor 从“不得低于初始化”改为“对严格 corrective action，clean margin 至少继承冻结 target-control advantage 的一个预注册比例”。形式为：

   `floor = relu(m_initial + alpha * clip(adv_target_control, 0, cap) - m_clean)`

   其中 teacher advantage 只决定 clean margin 目标，不把 action embedding 当作推理模块；最终仍直接更新 shared encoder。

### 3.3 N control 的处理

- Target 与 matched-random control 均保留正确身份 loss，避免通过故意破坏 random control 制造表面 preference；
- 额外 preference 只要求 target margin 不低于 control margin；
- 真正写入 clean embedding 的信号来自 action-conditioned clean margin floor，而不是对 control 做梯度上升。

### 3.4 全覆盖但不把所有动作当正例

- 28,509 条成熟 N actions 全部完成一次互斥路由；
- corrective actions 在重复前全部至少训练一次；
- robustness-only actions 进入 control/鲁棒性分支；
- harmful 和 uncertain actions只进入安全统计或 no-op，不进入 corrective loss；
- schedule 按 formula -> identity -> N family -> cell 循环，禁止 raw row count 决定曝光。

## 四、最小扫描实验

不扫描学习率、层数、epoch 或新动作。仅扫描 action-conditioned margin 的转移比例：

- `alpha=0`：严格 E4 continuation 对照；
- `alpha=0.25`：保守继承；
- `alpha=0.50`：中等继承；
- matched-random paired control：同一严格 corrective membership，但 target payload 换为 matched random。

四臂共享初始化、query order、候选采样、训练步数和随机种子。第一轮仅 fold 0 / one seed。主比较为 `alpha arm - matched-random` 和 `alpha arm - alpha=0`；formula-cluster CI 下界必须严格大于零，corrected > introduced，near/MRR 不下降，才能授权第二 seed。

`alpha=0.25/0.50` 不是性能超参网格，而是验证动作优势是否能以小剂量写入 clean margin 的必要剂量检查。两者均失败时停止该 N transfer 表达，不扩大训练。

## 五、P 的进入顺序

P 不进入第一轮。N-PMT 通过后：

1. 先增加 P-transfer；
2. 对 P wrong-direction control 使用反事实负方向语义；
3. P-transfer 必须相对已通过 N checkpoint 有独立正增量；
4. P-intensity 最后单独审查，只有 harmful 硬门和 paired clean 增量同时通过才允许加入。

因此不是放弃 P，而是阻止 P 在尚未证明 transfer 时再次淹没 N。

## 六、相对 DreaMS 的可证实范围

- 已证实成熟 E4 fold-0：相对 matching official DreaMS `+0.5740 pp`；38 corrected / 4 introduced。
- Dynamic-direct 最终绝对值：`+0.6753 pp`，但其相对 E4 的 +0.1013 pp 不显著且不优于对照，不能归因给 N/P dynamic。
- 成熟 N action 在 E8 geometry 的 outcome-aware residual 上界约可再提供 `+0.6753 pp`，故 **仅 N 路线的同几何上界约为 +1.249 pp vs official**；这是上界，不是预测。
- E4-PMT 第一轮合理目标是证明独立增量，而不是承诺 5 pp。若能稳定转移 N residual 的 20%-50%，总提升约为 `+0.71` 至 `+0.91 pp vs official`；完全转移才接近 `+1.25 pp`。
- 更高的 3-5 pp 需要后续 P-transfer 等新覆盖被安全转移。现有 4.93 pp 是 outcome-aware action/no-op 容量，不能直接写成模型可达性能。

## 七、冻结裁决

下一步不得重写 E4 主干，不做全量 meta-learning，不混入 P，不扫 LR/layers/epochs。先用 E4-PMT 在成熟 E4 checkpoint 上完成一个 N-only、target/control 成对、harmful exact-zero、全 qualified-action coverage 的小型四臂因果实验。只有它证明相对 clean 和 matched-random 的严格正增量，才继续扩大。

## 九、实现冻结（2026-09-05）

唯一正式入口为 `sbatch tasks/run_noise_final_e4_pmt_phase_a.sbatch`。它复用 fold-aligned 成熟 E4 checkpoint 和 run 2331284 已完成的 current-geometry full-candidate replay，不重新训练 M2，也不重新发现动作。先将全部成熟 N action 划分为 corrective、robustness-only、harmful、uncertain；harmful 优先，只有 strict-positive corrective 进入 target loss，其余 corrective weight 精确为零。四臂共享同一 corrective membership 与 coverage-first schedule，分别为 clean duplicate、matched random、PMT alpha 0.25、PMT alpha 0.50。

PMT treatment 在同一 batch 编码 clean、target、matched control 与相同正负候选；control 保留 identity loss 和 initialization margin floor，因此不得通过破坏 control 制造 target-control 优势。新增 clean floor 只继承 outer-train current-geometry 的 clipped positive advantage。所有合格 action 必须在任何 recycling 之前至少曝光一次。

## 十、七项问题的真实处置边界

本轮不得宣称一次解决七项：

1. 目标函数错位：本轮直接修复，加入 target-control preference 与 clean margin inheritance。
2. M2 预测的不是训练影响：本轮不使用 M2，绕开而非解决“训练影响预测”问题；是否可转移由四臂 held-formula 因果比较回答。
3. harmful 仍获正权重：本轮在 N 分支直接修复，harmful 优先且 target corrective weight 精确为零。
4. 强制平均 action dose：本轮直接移除，不再给每个 query 强制分配 action mass。
5. P 压过 N：本轮通过 N-only 隔离混杂，尚未解决未来 N/P 联合配比。
6. 304,209 actions 未完整曝光：本轮对全部 28,509 mature N actions 完成路由，并对全部 strict corrective N actions coverage-first；尚未解决 P 与全部 304,209 action 的训练曝光。
7. N/P control 语义不一致：本轮只保留 N matched-random，从实验中隔离该问题；尚未建立可统一 N/P 的损失语义。

因此 E4-PMT 是对 1、3、4 的实质修复，对 2、6 的受控局部处理，对 5、7 的隔离实验，不是七项问题的最终总解。只有本轮证明 N 的独立 clean-embedding 转移后，才设计 P-transfer 的单独配对合同。
