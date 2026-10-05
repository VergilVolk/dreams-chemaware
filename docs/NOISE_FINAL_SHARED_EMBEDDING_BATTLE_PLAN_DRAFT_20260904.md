# 噪声微调最终大决战：共享 embedding 训练决策草案

日期：2026-09-04  
状态：共同决策草案；关键选项冻结前不得提交正式大训练  
唯一目标：利用已经验证的 N/P 定向峰动作，训练一个推理时只接收 clean spectrum、query/reference 完全共享的 DreaMS encoder，使其 clean embedding 检索性能超过成熟 E4 起点。

## 1. 已经成立、不得再重做的事实

1. 正式图包含 23,876 个 query、1,805 个 official DreaMS Top-1 errors；P3 不进入开发。
2. 成熟 E4-A shared encoder 是当前最强、fold 对齐的可继续训练起点。fold 0 起点为 `seed_20260830/fold_0`，在对应 held fold 相对 official 为 `+0.574 pp`、38 corrected/4 introduced。
3. 当前几何已经完整回放 304,209 条 actions：N 28,509、P-intensity 215,436、P-transfer 60,264。
4. M2 已经证明三个来源均存在跨 formula 的 clean-visible 条件动作信号；P-transfer 最强，N 次之，P-intensity 的 cell prior 很强但 query-specific 增量较小。
5. M2 没有训练模型，不能作为 embedding 提升结论。
6. E14、E15、L2、CPG 等历史失败已经证明：单动作坍缩、硬阈值清空 family、把 risk 当 corrective、巨型 paired-gradient、结果后选和蒸馏替代直接训练均不得再次出现。
7. P2b、ChemAware、下游候选专家均不进入本轮。它们不能充当噪声微调教师或特征。

## 2. 本轮允许使用的数据

### 2.1 训练监督

- outer formula fold 0 之外的 17,953 个 query；M2 报告覆盖 1,993 identities、857 formulas。
- 同身份真实复现谱仅用于构建 P 动作；不得把 held formula outcome 或 P3 outcome 回填训练。
- N 使用已冻结的峰衰减路径；P 使用已冻结的强度/共识与真实复现峰转移 recipe。
- 每条动作必须有 target payload 和独立 control payload，并在成熟 E4 当前几何下重新计算完整候选 molecule margin。

### 2.2 候选边界

- 每个 query 使用冻结 official candidate graph 的完整候选 molecule 集合。
- 同一 molecule 的多张参考谱按冻结协议取最大相似度。
- held formula 可以作为已知谱库中的候选负分子出现，但不得作为训练 query、正监督或权重标签；因此该结果是固定谱库条件下的检索学习，不夸大为完全 inductive 新库泛化。

### 2.3 最终评价

- 开发评价：outer fold 0 的 5,923 个 held queries，约 225 formula clusters；该 fold 已被历史开发使用，不能称独立确认。
- 主要比较同时相对 mature E4 initialization、clean continuation、真正 matched-random control 和 static-target control。
- recipe、阈值、epoch、模型选择全部冻结后，才增加第二 seed、其余 formula folds，最后一次性使用 P3。

## 3. 必须全部保留的成熟动作空间

### 3.1 N：9 cells

- `candidate_gradient`：attenuation 0.50，step 3、4、5、6。
- `role_confounder`：attenuation 1.00，step 1、2、3、4、5。
- `role_shared` 永久禁用。

### 3.2 P-intensity：12 cells

- `matched_intensity_transport`
- `prevalence_attenuation`
- `consensus_projection`
- 每个 family 固定 dose 0.25、0.50、0.75、1.00。

### 3.3 P-transfer：9 cells

- `recurrent_peak_graft`
- `balanced_peak_exchange`
- `recurrent_union_mix`
- 每个 family 固定 dose 0.10、0.25、0.50。

“全部利用”定义为：30 个 cells 在训练 schedule 中均有真实、可审计的曝光，且任何来源都不能被硬阈值清零；它不等于把 304,209 条高度相关 actions 在每个 epoch 全部重复训练，也不等于对高风险 P-intensity 无条件等权。

## 4. 对当前大训练实现的阻断性审计

现有 `run_noise_final_dynamic_direct_e4_phase_a_big.sbatch` 不能在不讨论以下问题的情况下直接视为最终版。

### 4.1 `dynamic` 的定义没有兑现

合同原定义是 epoch 0 使用成熟 E4 replay，后续 epoch 使用上一轮 current-geometry evidence 更新权重。当前实现只在训练前运行一次 M2，此后四个 epoch 重复同一 schedule 和同一权重。

这有两种合法设计，但必须二选一并如实命名：

- **冻结 conditional policy**：四个 epoch 固定 M2 权重。因果对照最干净、成本最低，回答“初始化几何学到的条件策略能否转移”。
- **epoch-adaptive policy**：每轮重算 outer-train target/control advantage，下一轮更新权重。更接近原合同，但若四臂各自更新，会破坏共同 exposure；若由 NP arm 更新后复制给对照，又会造成 arm 间耦合。

建议：首个四臂因果核采用冻结 conditional policy，并把名字从 `dynamic` 明确为 `conditional_frozen`；只有它显著优于对照后，才单独做 epoch-adaptive 增量试验。该建议尚未冻结。

### 4.2 static-target 对照当前没有匹配总 action dose

当前 dynamic 每个 query 的 action mass 可变，平均约 0.392；static 固定为 0.500。它们不仅动作分配不同，总噪声剂量也不同。因此 `dynamic - static` 不能纯粹归因于条件加权。

必须修改为：static-target 对每个 query 复制 conditional arm 的**相同总 action mass/no-op mass**，只在该 query 的 families/cells/actions 内均匀分配。这样两臂唯一差异才是条件 utility，而不是噪声总量。

### 4.3 P 的 control 不是统一意义上的 matched random

N control 是冻结的匹配随机峰路径；P 当前 control 由 hardest-wrong-reference 构造，是 wrong-direction counterfactual。后者可能主动损害模型，不能命名为中性的 matched-random augmentation；用它作为唯一对照可能放大 target-control 差异。

需要共同选择：

- 保留 wrong-direction 作为机制 negative control，同时为 P 新增真正 count/intensity/mz/dose-matched random control，后者作为主要训练因果对照；或
- 接受 wrong-direction 仅回答“正确方向是否优于错误方向”，但不声称优于普通随机噪声。

建议第一种；这不是扩展动作搜索，而是修正对照语义。该建议尚未冻结。

### 4.4 当前 schedule 没有最大化 30-cell 覆盖

当前 schedule 在训练前按 dynamic utility 为每个 `identity × family` 只抽一个 action，之后四个 epoch 重复它。虽然全局 gate 可看到 30 cells，但同一 identity 的替代 dose/cell 大量未曝光；这会再次浪费动作矩阵。

建议使用 deterministic epoch-cycling：

- 每个 epoch 每个 `identity × family` 最多一个 action；
- 同一 epoch 不重复；四个 epoch 跨 cell/dose 轮换；
- 30 cells 每轮均有最低覆盖；
- 四个因果 arms 使用完全相同的 epoch-specific membership/order；
- membership 由 identity/formula/family/cell 的哈希轮换决定，不读取 outcome；utility 只决定 loss weight。

### 4.5 训练样本权重与合同不一致

当前 trainer 使用 `1 / formula_query_count`，实现 formula 等权，但没有在 formula 内进一步对 identity 和 query 等权。多 query identity 仍可重复施加梯度。

建议采用三级权重：formula 总权重相等；formula 内 identities 等权；identity 内 queries 等权。这样同时尊重 formula-disjoint 目标和 identity 去重复。该建议尚未冻结。

### 4.6 M2 clean feature 仍混合了两个 geometry

M2 的 global clean embedding 来自成熟 E4 当前几何，但 contextual peak tokens 仍来自 official fine-tuned DreaMS token cache。M2 的显著性成立，但不能严格称所有 clean features 都来自当前 mature E4 geometry。

需要共同选择：

- 直接冻结现有 M2，避免增加一次模型选择；或
- 用 mature E4 重新编码 25,275 张 reachable spectra 的 contextual peak tokens，仅在 outer-train 重做同一 crossfit，并以预先冻结的 Brier 对照判断是否替换。

建议重新编码，因为最后一个 Transformer block 已变化，而 P-transfer 的可见性主要依赖峰级上下文；但必须作为一次预注册替换，不能看 held retrieval 结果后选择。该建议尚未冻结。

### 4.7 风险模型还不足以声称解决新增错误

harmful AUPRC 的绝对值仍低：N 0.130、P-intensity 0.134、P-transfer 0.074。当前取 `max(full, cell-only)` 是保守措施，但 M2 没有报告：

- harmful Brier、ECE/calibration slope；
- dynamic weight top decile 的 harmful enrichment；
- 各 source/family 的 weighted positive mass 与 weighted harmful mass；
- baseline-correct low-margin queries 的 action exposure；
- P-intensity 在加权后是否真正压低历史 introduced 风险。

这些必须在构建 schedule 后、加载 117M 训练模型前计算。风险诊断不用于事后删 cell，只用于验证权重确实完成了预注册的降险；失败时重新校准 risk/no-op，不允许靠 held outcome 选阈值。

### 4.8 现有统计门仍需补齐

当前汇总以两个 Holm-adjusted one-sided sign-flip p 值作为主门，但合同还要求 clustered CI 下界严格大于零。最终汇总必须同时要求：

- conditional target vs true matched-random：Holm-adjusted p < 0.05 且 formula CI low > 0；
- conditional target vs exposure-matched static target：Holm-adjusted p < 0.05 且 formula CI low > 0；
- conditional target vs clean continuation：点估计 > 0，另报告 CI；
- corrected > introduced、corrected - 2×introduced > 0；
- near Recall@1、MRR、preservation 不下降；
- clip fraction 不系统饱和；
- 按 source/family/exposure decile 报告 introduced 与 candidate switch。

## 5. 建议的直接共享-encoder训练核

### 5.1 模型与初始化

- 初始化：fold-matched mature E4-A high-LR checkpoint，不回退 official DreaMS。
- 一个 shared encoder 同时编码 clean query、action view 和 candidate references。
- 推理时只有 clean spectrum；没有 selector、P2b、residual head 或蒸馏模块。
- 第一轮保持成熟 E4 的可训练范围：projection head + 7 层 Transformer 中最后一层（layer 6）。两层或全模型解冻只能在第一轮证明语义转移后作为独立 capacity test。

### 5.2 建议保留的简单直接损失

`L = L_clean_full_list + L_action_mixture_full_list + 2 L_margin_floor + 5 L_preserve`

- `L_clean_full_list`：原始 clean query 在完整候选 molecule 集合上的 listwise cross-entropy。
- `L_action_mixture_full_list`：no-op 与全部选中 target actions 的归一化混合；总质量严格为 1。
- `L_margin_floor`：保护 clean query 相对成熟 E4 初始化的完整候选 margin，不允许因噪声训练系统恶化。
- `L_preserve`：同时保护 query 与 candidate-reference geometry。

首轮不加入 paired residual distillation、不加入 target-minus-control 巨型梯度项、不改变网络架构。control payload 只属于独立对照 arm，不在 target arm 中反向“推开”。

### 5.3 当前有效动作剂量必须共同决定

当前 conditional 权重的平均 no-op 为 0.608，即平均 action mass 约 0.392。由于总 loss 还额外包含一份 clean loss，平均有效系数近似为 clean `1.608`、action `0.392`；action/clean 约为 0.24。这很安全，但也可能继续造成动作信息转移不足。

可选方案：

1. **保守原剂量**：保持当前 utility/no-op，`lambda_action=1`。归因最连续，可能低估动作容量。
2. **中等 curriculum**：保持每个 query 内相对 utility，不改变 cell 选择；仅把总 action mass 按 epoch 预注册为约 0.35、0.45、0.55、0.55，并保证 no-op ≥0.35。所有 target/control/static arms 匹配同一总 dose。可能提高转移，但增加一个新的训练假设。
3. **增大 action loss 系数**：固定权重但提高 `lambda_action`。这最容易重演 L2 clipping，不建议作为首选。

建议方案 2，但必须由共同决策冻结；训练前需用 32 个分层 microbatches 验证 action/clean 梯度范数、cosine 和 clip fraction。

### 5.4 优化参数

- 现有候选：4 epochs、backbone LR `2e-6`、head LR `1e-5`、weight decay `1e-4`、temperature 0.10、grad clip 1.0、no AMP、batch query 1。
- 这些参数来自成熟 E4 high-LR 配方，第一因果核不与动作/权重同时做 LR sweep。
- 必须在正式训练前报告 32 个分层 microbatches 的未裁剪总梯度、各 loss branch 范数与 cosine；若系统 clipping，先按共同决定的规则降低 action dose，而不是事后挑学习率。

## 6. 因果训练臂与每一臂回答的问题

建议第一轮仍是一个 sbatch 串行四臂，但修正对照语义和 dose：

1. `C0 clean_continuation`：成熟 E4 继续训练本身能涨多少。
2. `C1 true_matched_random`：相同 query、相同总 dose、相同 cell/count/intensity/mz 与 schedule，但随机峰 payload；回答方向语义是否超过普通增强。
3. `C2 exposure_matched_static_target`：相同 target payload 与每 query 总 action/no-op mass，但在 cells/actions 内静态均分；回答 clean-conditional weighting 是否有增量。
4. `NP conditional_target`：完整 N + P-intensity + P-transfer 条件直接微调。

P wrong-direction payload 单独保留为训练前/训练后机制诊断，不作为 C1 的唯一替代。

若第一轮通过，第二阶段才做：

- N-only conditional；
- P-intensity-only conditional；
- P-transfer-only conditional；
- 必要时 N + P-transfer（检验高收益低风险组合）。

这些消融用于归因和去除负贡献，不得在第一轮结果出来前凭直觉删掉任何成熟策略。

## 7. 训练前必须生成的审计页

正式模型加载前输出并人工确认：

1. 30 cells 的 action/query/identity/formula 数量与四 epoch 覆盖率；
2. 每个 arm、epoch、query 的 action mass + no-op = 1；
3. 四臂 epoch-specific membership/order 完全一致；
4. true matched-random 的 peak count、intensity、m/z、dose 分布与 target 匹配；
5. static 与 conditional 每 query 总 dose 完全一致；
6. formula→identity→query 三级权重和；
7. 各 family ESS、最大单 action/query/identity exposure；
8. weighted positive/harmful mass 与 top exposure decile 风险；
9. mature E4 checkpoint、current embeddings/tokens、graph、ledger 的 SHA；
10. outer-held formulas、P3、P2b 均未进入训练标签；
11. 32 个分层 microbatch 梯度审计；
12. smoke run 后 clean/control/target 三种 payload 均可执行且无 OOM、NaN 或全程 clipping。

## 8. 最终共同决策清单

在写新的唯一 sbatch 前，必须共同冻结以下五项：

1. 首轮采用冻结 conditional policy，还是实现 epoch-adaptive weights？
2. P 的主要对照是否增加真正 matched-random，同时保留 wrong-direction 只作机制诊断？
3. 是否用 mature E4 重编码 contextual peak tokens 后重做一次 M2？
4. action dose 采用当前保守剂量，还是四 epoch 中等 curriculum？
5. 训练聚合是否采用 formula→identity→query 三级等权？

本草案的建议组合是：**成熟 E4 tokens + 冻结 conditional policy + true matched-random 主对照 + exposure-matched static + deterministic cell cycling + 三级等权 + 中等 action-dose curriculum + 最后一层与 head 直接训练。**

这套组合的目的不是承诺未经观察的 +5 pp，而是让四臂第一次同时满足：全部成熟策略真实曝光、条件加权可归因、随机增强可归因、训练单位正确、风险可量化、推理只用 clean embedding。用户确认五项选择后，才据此生成版本化的新 trainer、独立测试和唯一 GPU sbatch；旧 sbatch 不覆盖，保留为冻结历史实现。

## 9. 用户授权后的精简实施冻结（2026-09-04）

用户要求删除过多步骤、在合理范围内尽可能提升性能。最终实施只保留直接影响 shared embedding 转移的修正：

1. 保留现有 M2；本轮不重新编码 mature-E4 contextual peak tokens。
2. 首轮使用冻结的 clean-conditional policy；不实现 epoch-adaptive replay，避免四臂 exposure 分叉。
3. 保留现有来源适配 control：N 为 matched-random peak path，P 为 wrong-reference direction control；结果中明确区分语义，不把 P control 宣称为普通随机噪声。本轮不再新建一套大型 P-random 矩阵。
4. 30 cells 使用 outcome-independent deterministic epoch cycling；每个 epoch 每个 identity-family 最多一个 action，跨 epoch 优先遍历不同 action。
5. conditional 与 static 每个 query 的总 action/no-op dose 完全相同；static 只取消 query 内条件分配。
6. 四轮平均 action dose 冻结为 0.35、0.45、0.55、0.55，最大单 query action dose 0.65，no-op 至少 0.35。
7. 训练权重使用 formula→identity→query 三级等权。
8. 损失、学习率、解冻范围保持简单：full-list clean/action mixture、margin floor、preservation；head + 最后一层；backbone/head LR 为 2e-6/1e-5。
9. 不加入蒸馏、CPG、在线 selector、新 action family、额外层或学习率扫描。

正式入口冻结为：

```bash
sbatch tasks/run_noise_final_dynamic_direct_e4_phase_a_final.sbatch
```

该作业申请一张 GPU、不显式申请内存，使用 `${SLURM_JOB_ID}` 唯一目录；先运行全部轻量单测和静态实现审计，再依次执行 mature-E4 preflight、304,209-action current-geometry replay、M2、四 epoch 共同 schedule、四臂直接 shared-encoder 训练和 paired formula-cluster 汇总。P control 的结论边界维持为“正确方向相对错误方向”，真正的模型性能仍以 NP 相对 mature E4、clean continuation 和 official DreaMS 为准。
