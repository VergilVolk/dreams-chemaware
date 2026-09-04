# Noise E14 实现失败审计与 E15 强制工程合同（2026-08-31）

## 结论

E14 的一折结果不能用于裁决噪声微调路线。它只裁决了一个被错误压缩和错误监督的实现：

- 已验证的大动作空间被压成每个 query 一个动作；
- 条件有效动作先经过全局安全过滤，异质性被提前删除；
- harmful/protected risk controls 与 corrective actions 共用同一个正向增强损失；
- 644 条 P-arm 动作在一个 epoch 内被循环抽取多轮；
- 全局梯度比例只由每个分支最前面的 4 个样本决定；
- selected 路径没有执行命令行声明的 `positive_deficit_errors` 范围。

因此，E14 的 `pass_to_multifold=false` 只意味着当前实现不能扩大，不意味着动作空间、噪声微调或共享
embedding 路线失败。

## 一、必须分开的证据层

| 层级 | 已确认结果 | 正确含义 |
|---|---:|---|
| 共享 encoder 权重 | E4-A 五折三 seed 平均约 `+0.635 pp` overall、`+0.522 pp` near | 直接噪声微调能够稳定改善共享 embedding |
| A4/历史动作并集 | 约 `+3.853 pp` headroom | 动作监督空间足够大，不是已训练权重 |
| C1 support-disjoint teacher | `80,250` examples，教师空间约 `+2.475 pp` | 跨重复谱正证据存在，不是部署模型 |
| consensus projection | 固定动作约 `+1.127 pp` | 正例强度/共识方向有效 |
| recurrent union mix | 固定动作约 `+1.118 pp`，`294/27` corrected/introduced | 真实重复峰转移具有高安全性 |
| no-op-aware unions | 约 `3.4–4.9 pp` | outcome-aware 上界，不是共享权重性能 |

任何报告不得把 headroom、动作即时效果和共享 encoder 权重增益混为同一个指标。

## 二、E14 已确认的实现错误

### 1. 单动作坍缩

`build_noise_final_e14_crossfit_p_teacher.py` 对每个 query 只保留 margin 最大的一个 correcting action，
并强制 query 不重复。多动作一致性、剂量稳定性、动作不确定度和互补峰路径全部丢失。

### 2. 错误空间覆盖不足

E14 fold 4 最终只有 `644 queries / 389 identities / 183 formulas`。它没有统一纳入：

- A4 exact Top-25/Top-50 安全动作；
- C1 的 support-disjoint 正例教师实例；
- 完整 N-arm 与 P-arm 的互补动作账本；
- no-op 与 harmful action 的对称记录；
- ChemAware 规则作为条件路由特征。

### 3. 风险对照监督语义错误

`supervision_kind` 只被用于日志统计。risk controls 与 corrective examples 最终进入同一个
`guided_noise_loss`，并同时优化 action rank、clean/action consistency 和 self-transfer。这不是负对照，
而是把已知有害动作重新当成正向增强。E14 中 risk arm 相对 delta 的 `6 corrected / 11 introduced`
与该错误完全一致。

### 4. P-arm 循环过采样

约 `389 identities x 2 views = 778` 个 P 样本，需要服务约 `1,589 x 4 = 6,356` 次 epoch 内抽取。
游标耗尽后重新归零并洗牌，导致相同小样本每轮反复出现，破坏了动作曝光、N/P 比例和 formula
有效样本量。

### 5. 梯度标定不具代表性

N、safety、P-corrective、P-risk 每个分支只取列表最前面的 4 个样本估计梯度范数，再用一个全局
比例控制整次训练。该估计没有随机化、formula 分层、identity 分层或动作家族平衡。

### 6. 配置与实际样本范围不一致

命令行和输出 tag 声称 `positive_deficit_errors`，但 selected 路径绕过了该过滤。结果文件虽记录
该合同为 false，实验名称仍会造成误读。

## 三、E15 不可绕过的工程合同

### A. 多动作账本

每个 query 必须同时保留：

1. `corrective_actions`：最多 K 个、跨内折或跨重复证据支持的修正动作；
2. `harmful_actions`：造成 introduced error 或显著 margin 下降的动作；
3. `no_op`：永远允许；
4. action family、dose、step、reference policy、error family、near/mid、仪器条件和 ChemAware
   证据字段。

禁止用 `argmax(action_margin)` 把整个 query 压成一个动作标签。

### B. 风险分支与修正分支分离

`corrective_loss` 和 `risk_loss` 必须是两个不同函数、两次独立 backward 记录：

- corrective branch 可以训练 clean/action ranking、保守 margin delta 和一致性；
- risk branch 不得使用 corrective self-transfer，不得把 harmful action 当成正教师；
- risk branch 只定义 clean protection、trust region 和对 corrective gradient 的 veto/projection；
- 必须报告 corrective-risk gradient cosine、投影比例和被拒绝更新数。

### C. 有限曝光采样

- 每条 corrective/risk action 每 epoch 的曝光次数必须显式上限化；
- 禁止游标耗尽后在同一 epoch 归零循环；
- identity、formula、error family、action family 四层均衡；
- 输出实际 unique actions、draws、maximum exposure、p50/p90 exposure。

### D. 稳健梯度标定

- 不得使用固定的 `[:4]`；
- 至少 32 个分层微批次或覆盖至少 128 个 identity-action observations；
- 使用中位数/截尾均值与 epoch EMA；
- 每个动作家族单独报告 norm 与 safety cosine；
- 分支独立裁剪或 PCGrad 后才能合并。

### E. 三道训练门

1. **32-query overfit gate**：证明损失、方向和参数更新能实现教师修正；
2. **256-query identity-held gate**：证明不是记忆 query/action；
3. **one-formula-fold gate**：相对完全匹配的 warm continuation control，formula-cluster CI 下界
   大于 0，`corrected > introduced`，`corrected - 2*introduced > 0`，near 不退化，preservation
   mean >= 0.995。

任一门失败都不允许多折、多 seed 或 P3。

## 四、提交纪律

每个训练 sbatch 必须按顺序执行：

1. 静态 AST/合同测试；
2. 输入、哈希、formula 隔离和 P3 overlap 预检；
3. sampler exposure dry-run；
4. loss branch unit test；
5. gradient calibration smoke；
6. 只有上述全部通过才启动训练；
7. 训练后立即运行输出验证器；
8. 输出目录 fail-closed，禁止覆盖。

预检与训练必须位于同一 sbatch 中，`set -euo pipefail` 保证任何一步失败都会阻止后续训练。

## 五、当前裁决

- E4-A 保留为正式 noise-embedding 基线；
- E14 保留为负面工程诊断，不扩大；
- 下一阶段先完成 E15 合同预检和最小可学习性门，不直接跑大规模训练；
- P2b、候选后处理和 P3 不进入噪声微调教师、特征或损失。

## 六、E15-M0 已实现的审计工件

E15-M0 只构建训练账本，不训练模型。它合并四个既有证据源：

- R0/S3A 的固定 N-arm 动态峰轨迹；
- A4 的 Top-50 精确峰干预；
- C1 的 support-disjoint 正例教师；
- E14 在成熟 cross-fit 几何上已经计算的完整 P-action outcomes。

输出严格分为：

- `all_action_ledger.csv.gz`：所有来源与条件支持统计；
- `corrective_actions.csv.gz`：每 query 最多4个、优先来源/家族多样性的修正动作；
- `harmful_actions.csv.gz`：每 query 最多4个有害动作；
- `no_op.csv.gz`：每个训练 query 恰好一个 no-op；
- `conditional_support.csv.gz`：动作机制在 identity、formula 和 formula-fold 层的支持量；
- `report.json`：来源完整性、曝光 dry-run、隔离与 provenance。

该阶段禁止启动训练。只有完整账本、独立风险损失、有限曝光 sampler 和32×4分层梯度标定的
行为测试全部通过，才允许设计32-query overfit gate。

## 七、E15-M1 与 M2 的落实状态

E15-M1 已在 fold 4 完整通过：7,943 个被选动作全部与不可变来源逐行复现，rank 与监督语义
零错配；梯度预备面板固定为 4 sources × 2 branches × 16 actions = 128 个不重复动作。M1 的
source-equal 权重只作为标定候选，不能直接当训练权重，因为稀疏来源会获得过大的单动作权重。

E15-M2 的代码合同如下：

1. 从当前成熟的 fold-4 共享 encoder 初始化，所有 margin floor 与 preservation target 在该初始化
   上重新计算，禁止把 official/E14 旧几何的绝对 margin 或 raw margin delta 冒充当前尺度；动作
   强弱只使用 M1 的 source-local percentile，并映射为当前几何中的固定 0.01–0.05 目标增量；
2. R0 从峰路径重建，A4 从精确 token 重建，C1 从 support-disjoint 同分子 prototype 重建；
3. E14 只有与 `selected_actions.csv.gz` 或 introduced `risk_controls.csv.gz` 精确 join、拥有真实
   reference rows 的动作才可执行；其余行必须计入 non-executable，不得替换；
4. corrective 与 risk 使用不同函数。harmful payload 在 risk loss 中完全不可访问，只作为选择
   clean protection query 的路由标签；同一 source/query 的多个 harmful 标签只生成一次保护样本；
5. 32-query 容量门中，每个 source/query 每 epoch 恰好产生一个 optimizer step，每来源固定8个
   不重复真实错误。多动作优先但不强加到每个来源：E14 exact 侧表本身为534 actions/534 queries，
   因此是结构性单动作来源；R0/A4/C1 承担多动作步内聚合验证，全局至少保留6个真实多动作 query。
   每个 query 最多16个家族多样动作并在步内平均，动作数绝不增加优化步数；
6. 风险分支保护当前 query 候选图中的全部负分子，不得只保护 Top-4 后静默放过其余候选；
7. 128-action 面板不仅 action 唯一，而且每个 source/branch 的16项必须来自16个不同 query 与
   identity；它执行32个互不回卷的微批次，实测各 source/branch 梯度 norm 与 cosine；正式
   source 权重由实测 corrective norm 归一化并限制在 16 倍总动态范围内；
8. corrective/risk 梯度分开求取；发生冲突时仅投影 corrective 的风险冲突分量，再与 risk 梯度
   合并。每个 epoch 的 action 暴露和 query step 都必须逐项对账；
9. M2 仅是训练面板容量门。通过后才允许 identity holdout；它本身不是泛化结果，也不能与 P3
   或其他候选协议的官方 DreaMS 数字混合。

对应入口为 `tasks/run_noise_final_e15_m2_overfit.sbatch`。该作业固定单卡，不申请额外内存，
输出使用 `${SLURM_JOB_ID}` 隔离，依次执行单元测试、可执行面板构建、共享 encoder 训练和
checkpoint/逐 query 验证。

## 八、E15-M2 实测结果与 M3 放行边界

E15-M2 的不可变容量运行 `fold_4_run_2327814` 已通过全部门：32 个来源平衡的真实错误 query、
60 个 corrective actions 中修正 27 个且新增 0 个，corrective margin 平均增加约 0.241；全局
初始化保持度约 0.99275。它证明多动作账本、分离损失和共享 encoder 可以在受控面板上实现
大幅教师转移，但这是训练面板容量证明，不是身份外泛化，也不是 Recall@1 的 pp 结果。

运行后审计同时固定三项修正：

1. C1 没有显式 action view，原日志的普通均值会被 NaN 污染；全局和分来源日志现在使用有限值均值，
   NaN 只允许留在确实无该字段的局部分支；
2. risk 梯度范数只有约 `1e-7` 时，数值噪声曾触发极大 PCGrad 比例；现在风险范数低于 `1e-6`
   时禁止投影，并显式报告两分支范数与投影是否激活；
3. 梯度 cosine 改为 float64 计算并裁剪到 `[-1, 1]`，避免数值显示超过物理范围。

M3 不得从 M2 overfit 权重初始化。它必须从成熟 E4-A fold-4 clean-only 权重重新开始，并先冻结
256 个 identity-unique held queries（128 errors，四来源各32；另含128个 pooled correct controls）和256个完全未训练的
sentinel identities。held/sentinel identity 必须从训练 query、动作教师 reference 以及所有可训练
candidate references 中删除。只有 split 校验、参考谱 IK14 逐行校验、训练容量和配套 E4-A
权重来源全部通过，才允许启动 identity-held 训练。权重来源以通过的不可变 M2 报告中实际加载的
E4-A、official 和 architecture 三个 SHA256 为准；不得依赖历史 `decision.json` 中随脚本版本变化的
非本质字段。

这里的“四来源各32”只适用于 held errors，因为它们用于测量四种 corrective 机制的外推。held
correct controls 允许同一 query 在多个来源账本中出现，故只要求128个 identity-unique controls 且
报告其来源分布，禁止把来源标签强行做成互斥的32/32/32/32。harmful controls 是 pooled clean-risk
保护样本，也只要求身份隔离且总体非空，不要求按 corrective source 人为分层。训练容量逐来源报告 action、query、
identity 和 held-removal retention；阻塞条件是某一来源被删除为空，而不是未经数据推导的
`identity>=500` 或 `harmful query>=500`。

## 九、E15-M3 split 实测与配对训练合同

M3 split 已正式通过：256 个 held queries/identities（128 errors + 128 correct）、158 formulas，
其中201个为 near；错误侧四来源严格各32。另冻结256个 sentinel identities。训练侧保留1,930个
corrective actions、1,292个 corrective queries、485 identities，以及941个 pooled harmful actions、
467个 risk queries。12,700条动作教师 references 和2,871条账本 query rows 均通过 HDF5 IK14
逐行核验；held/train/sentinel identity overlap 均为0。

R0 在身份隔离后仍保留75个 corrective actions/53 queries/50 identities，但没有独立 harmful actions。
因此 M3 风险保护固定为 pooled clean-risk，不再错误假设每个 corrective source 都有自己的 harmful
分支。R0 corrective 仍保留并参与多动作训练。

M3 训练采用两个完全相同的成熟 E4-A 起点：

- warm control：相同 clean queries、候选 references、optimizer steps 和 pooled risk protection，永不读取
  action payload 或 teacher target；
- noise arm：在同一训练日程上增加 multi-action corrective loss，跨来源动作按 query 聚合，每个 query
  每 epoch 最多一次 optimizer step，每条动作不回卷；
- 两个预注册学习率配置只用训练内部 identity-disjoint dev 与 label-free sentinel preservation 选配置和
  epoch；256-query held 明细直到选择完成后才读取；
- held/sentinel/internal-dev identities 从所有可训练候选 references 删除，内部 dev 排名也过滤 held/
  sentinel references；最终 held 才在原始冻结候选图上一次性评价；
- M3 主比较为 noise versus paired warm control；noise versus mature initialization 作为补充。任何
  headroom、动作即时效果或 P2b 结果均不得混入共享 embedding 增益。

## 十、E15-M3 结果：实现失败，不裁决噪声路线

M3 最终选择 conservative epoch 1，但内部 identity-dev 的 noise-control 为 `0 corrected / 0
introduced`，held 为 `0 corrected / 1 introduced`，即 `-0.391 pp`；near 为 `-0.498 pp`。保持度
约0.99998，说明权重几乎没有形成可改变排序的更新。该结果不进入 formula-fold。

这次运行同时暴露四项必须修复的实现错误：

1. split 中有1,930 corrective actions/1,292 queries/485 identities，训练最终只剩281 actions/198
   queries/133 identities。除了身份隔离造成79个无合法负候选 query，主要压缩来自用成熟 full-graph
   rank 再次过滤旧账本，且没有先在成熟几何逐动作回放其真实方向；
2. `initial_margin` 在完整候选图上计算，loss references 随后排除了 held/sentinel/internal-dev
   identities。目标 floor 与实际训练候选不是同一个候选协议，违反 baseline matching；
3. control 使用初始化处近乎为零的 risk floor，平均 loss 仅约 `5.6e-7`，control 与 initialization
   排名完全相同。它不是有效的 warm continuation 对照，无法分离“正常 clean ranking 训练”与
   “动作监督增量”；
4. `selected_internal_epoch_eligible=false` 后程序仍继续读取并评价 held。正确的 fail-closed
   顺序应当是：内部开发门未通过就立即退出，held 文件及其标签都不得读取。此次256-query held
   因而已被消耗，只能用于解释本次失败，不能继续用于调参、选 epoch 或正式复测。

因此，M3 只否定当前实现。该256-query held panel 从此标记 consumed，不允许继续用于模型选择或
正式复测。下一阶段必须：在成熟 E4-A 几何上对所有训练动作重新执行并记录 action rank/margin；
在排除列表生效后重新计算完全匹配的 clean baseline margin；保留 safe margin-strengthening actions，
而不是只保留 full-graph 当前错误；control 使用与 noise 相同的 clean ranking/preservation loss，唯一
差异只能是 action/teacher 项；最后冻结新的 identity-held panel再评价。
