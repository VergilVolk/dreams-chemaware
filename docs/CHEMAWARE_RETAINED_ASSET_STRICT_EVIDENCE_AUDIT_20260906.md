# ChemAware 保留成果严格证据审计

**冻结日期**：2026-09-06  
**审计对象**：截至冻结日期被描述为“保留成果”“核心资产”“通过门槛”或可用于 shared embedding 微调的 ChemAware 结果，以及被借作依据的 Noise/BioAware/P2b 结果。  
**审计结论**：**ChemAware 已经形成多层可举证成果：结构条件教师与动作的候选判别能力；两条 formula-disjoint 经验化学关联；clean-visible mass/rule shared-kernel 的开发集增益；冻结候选侧谱学门的一次性内部 test 净纠错；以及 ICEBERG direct 训练后 clean shared embedding 的首次正向 proof-of-effect。当前尚未完成的是把化学特异增量在完整图、匹配对照和多折条件下稳定复现。已有成果不能被这一层尚未完成而抹零。**

**A2 后续更新**：24-action、零 optimizer step 梯度筛选已完成。三个非 naive
路由的 correct influence 绝对值均严格为正，但 correct−candidate-swapped 与
correct−peak-permuted 的 formula-cluster 区间均未严格为正；`naive_action_minus_clean`
严格有害。因此当前 action-view 注入路由未通过 E2 特异性门，且不得用
correct−control 残差作为训练梯度。详见
`CHEMAWARE_A2_GRADIENT_TRANSFER_SCREEN_RESULT_20260906.md`。

本文替代 `CHEMAWARE_ACTION_ASSET_AND_TRANSFER_AUDIT_20260906.md` 中所有关于证据等级和“核心候选资产”的判断。旧文仅保留为探索过程记录。

## 1. 本次审计回答的不是“做了什么”，而是“能主张什么”

每个结论必须明确区分以下五层：

| 层级 | 可回答的问题 | 最低证据要求 |
|---|---|---|
| E0：定义有效 | 规则或动作是否明确、可执行、不会偷看真值 | 完整输入条件、动作方向、适用域、失败条件和负对照 |
| E1：冻结动作有效 | 不更新模型时，动作能否在未参与选择的数据上净纠错 | formula/identity 隔离；预先冻结动作；corrected、introduced、风险效用及聚类区间 |
| E2：动作特异梯度有效 | 相比等预算对照，动作梯度是否改善未修改 clean query 的真实边界 | target-control 梯度增量、单步 clean influence、冲突率、覆盖率及区间 |
| E3：共享 embedding 有效 | 部署时只给 clean spectrum，模型是否优于匹配基线和等预算对照 | 同图、同初始化、同训练量；多折多 seed；全指标；formula-cluster CI；安全性 |
| E4：外部泛化有效 | 在完全独立且未参与开发的数据上是否复现 | sealed external dataset、冻结协议、实验室/仪器分层及预设统计分析 |

“教师很强”“动作后排序改善”“梯度非零”“训练可运行”均不能替代 E3。报告标准采用透明、完整、准确地说明数据、模型、评价和不确定性的原则；TRIPOD+AI 也特别要求避免混淆 validation 与 evaluation，并报告开放科学信息和已知未知项。它是报告框架，不被当作本项目的偏倚评分工具。参见 [TRIPOD+AI statement, BMJ 2024](https://www.bmj.com/content/385/bmj-2023-078378)。

## 2. 总裁决

| 被审计对象 | 最强可成立事实 | 原先容易形成的过强说法 | 严格裁决 |
|---|---|---|---|
| ICEBERG teacher | 在一个刻意富集 official errors 的 MassSpecGym 派生 panel 上，候选结构条件谱预测可区分候选 | “结构教师已经独立验证” | 仅为同域机制/headroom；不是独立泛化 |
| ICEBERG top-3 conflict attenuation | confirmation 为 3 corrected / 2 introduced、净纠错 +1、Recall@1 `+0.7519 pp`，margin 对两个置乱对照的聚类区间为正 | “已经证明 shared embedding 会提升” | **保留为积极的 E1 候选动作结果**；2:1 风险偏好下效用为 −1，说明安全性结论对风险权重敏感，不能直接外推 E3 |
| 两条 C2H6N 经验规则 | formula-disjoint confirmation 中结构条件与峰出现有统计关联 | “两条化学规则已经会纠错” | 仅为内部关联假设；实际训练 query 覆盖门失败 |
| 五视图谱学共识门 | 一次性内部 test 为 9/0、+0.254 pp，聚类区间为正 | “稳定化学规则或 embedding 成果” | 可保留为稀疏的候选侧谱学纠错信号；不是化学、不是 embedding、不是外部泛化 |
| error-conditioned spectrum-only adapter | corrected full graph 的 1,929-query inner 上 `+0.3629 pp`、9/2，公式簇区间为正 | “这是化学增益” | **保留为已通过的 shared-embedding 基线与训练骨架成果**；它是所有化学注入必须超过的非化学对照 |
| mass / rule-mass shared kernel | 同一 held inner 上 mass `+0.8813 pp`；rule-mass `+1.2442 pp`，32/8，且比 mass 与 shifted-rule 的配对区间均为正 | “规则机理与外部泛化已证明” | **保留为 clean-visible shared-embedding development 成果**；后续独立 576-query observation screen 不稳定，故尚不能升级为确认性/外部结果 |
| ICEBERG direct shared v2 | clean shared embedding 在 180-query inner-all 上 `+0.5556 pp`、1/0；selected 49-query 上 `+2.0408 pp` | “化学动作已稳定注入” | **保留为首次 learned shared-embedding proof-of-effect**；单臂、小样本、CI 下界为 0，且 preservation/clip 门失败，必须做匹配四臂复现 |
| direct ICEBERG shared trainer | 已产生上述 proof-of-effect，并可计算 clean/action/safety 三项 loss | “当前 loss 已解决稳定迁移” | 迁移并非零，但尚未证明 action-specific superiority；需要 correct、swapped、permuted、clean-duplicate 同预算比较 |
| rule-teacher gradient audit | correct rule 梯度范数 2.656、为 base 的 19.9%，与 base cosine 0.325；correct−shifted differential 在 11/11 个可训练张量非零 | “只要梯度非零就会提升” | **保留为注入瓶颈定位证据**：化学方向存在但弱、且与通用目标仅部分对齐；不是检索性能结果 |
| `direct_projected_guarded` | 已实现 clean/safety 主梯度与 action 辅助梯度分离；采用单侧 PCGrad-style 冲突投影，action 范数上限为主梯度 25%；合同测试通过 | “新方法已经提升性能” | **仅保留为历史安全工程实现**；PCGrad/cap 不创造 correct-control 特异性，Stage-1 与 A2 结果均不支持继续把它作为当前下一训练方案 |
| Noise E4-A `+0.6362 pp` | 在旧 23,876-query cohort 上曾有多折多 seed 正结果 | “当前可用的共享 embedding 证据底座” | 旧图语义已撤销；只能作为历史工程起点，不能支持当前 corrected-fullgraph 性能结论 |
| BioAware/P2b | 候选上下文在各自任务中存在纠错信息 | “ChemAware embedding 可据此提高 3–5 pp” | 非 ChemAware shared embedding 结果；只提供对照设计经验 |

因此，当前可发表式结论不是“没有成果”，而是：

> 我们已经获得可复核的结构条件候选判别、经验化学关联、clean-visible shared kernel、稀疏候选纠错及 learned clean-embedding proof-of-effect；其中 ICEBERG top-3 动作在 formula-disjoint confirmation 上实现净纠错 +1、Recall@1 +0.7519 pp，并在 margin 指标上显著优于匹配置乱对照。尚待解决的是安全阈值的风险权重敏感性，以及 chemical treatment 相对等预算 continuation 和 pseudo-action controls 的稳定、可归因 shared-embedding 增量。

## 3. 逐项证据链与偏倚审计

### 3.1 ICEBERG teacher：教师能力存在，但独立性不成立

本地报告 `data/validation/chemaware_full_manifest_iceberg_teacher_700_v1/report.json` 明确记录：

- 700 个 query 中有 619 个 official errors、81 个 official-correct；这不是自然分布样本，而是错误富集的 teacher-development cohort。
- official DreaMS Recall@1 为 0.1157，ICEBERG 为 0.5886，candidate-swapped 为 0.1329，peak-permuted 为 0.2986。
- 选择规则优先取 official errors，再取 near-boundary correct；报告自己声明 `diagnostic_not_formal_generalization=true`。
- ICEBERG 权重训练于 MassSpecGym，本地图也来自 MassSpecGym；报告自己给出 checkpoint-overlap warning。
- ICEBERG canonicalization 去除了立体化学，因此结果只涉及 connectivity-level 区分，不能支持立体异构体主张。

这些结果足以支持“候选结构条件的预测谱含有判别信息”，不支持“独立结构教师已经验证”，更不支持任何 shared embedding 增益。

### 3.2 ICEBERG top-3 动作：正向结果成立，但安全裁决依赖风险权重

现有输出记录的冻结 setting 为 `conflict_attenuate / strength=0.75 / top_k=3`：

- discovery：276 queries，8 corrected / 2 introduced，Recall@1 `+2.1739 pp`；
- confirmation：133 queries，3 corrected / 2 introduced，Recall@1 `+0.7519 pp`；
- confirmation 的 absolute margin、相对 candidate-swapped margin、相对 peak-permuted margin 的 formula-cluster CI 下界均为正；
- 21 个 setting 在 discovery 上筛选，fold 2 用作 confirmation；fold 3 尚未做 embedding evaluation，fold 4 保留。

原脚本 `tasks/audit_chemaware_direct_action_bank.py` 的 hit1 门为：

```text
hit1_delta >= 0
corrected >= introduced
```

按原门槛，confirmation 有 `corrected-introduced=+1`，所以脚本 PASS 有明确的数据依据。若采用后来更保守的 2:1 引错惩罚，则：

```text
U = corrected - 2 * introduced
U_confirmation = 3 - 2 * 2 = -1
```

因此，正确结论不是“该结果失败”，而是：**动作的净纠错和 margin 改善是正成果；是否达到部署级安全门，取决于预先声明的风险权重。** 原实验没有预先把 2:1 定义为唯一主要终点，审计不能事后用它抹除原有正结果；但后续实验必须同时报告 1:1 净纠错和 2:1 保守风险效用，并在实验前冻结主要裁决规则。

此外还有四个限制：

1. teacher panel 刻意富集 official errors，不能估计全局收益或风险；
2. 700-query graph 与 ICEBERG 训练域同源，不能算外部确认；
3. 输出明确写有 `formal_training_authorized=false` 和 `global_inner_outer_evaluated=false`；
4. 本地只有粘贴输出；未找到与哈希 `ae462bb1edcfe23147f6858c713121504e3f46073b91848afd491f6d14101241` 对应的 action-bank `.npz`，也未找到输出所引用的 `iceberg_predictions_f16.npy`，无法在本机逐 query 重放 117 条最终训练动作。现存 graph 与 teacher report 的 SHA-256 与输出记录一致，但这不足以替代缺失的预测张量和 action bank。

**裁决**：保留为当前最重要的 ChemAware E1 正成果和 shared-embedding 微调候选。它证明了结构条件动作可改善候选边界，但由于样本小、2:1 风险效用为负、同域 teacher 及本地工件不完整，尚不能称作部署安全或 E3 成果。

### 3.3 两条经验规则：统计关联成立，化学机制与纠错能力均未成立

`data/validation/chemaware_empirical_structure_rules_v1/report.json` 从 9,854 个有效 SMILES 中，经 discovery 筛出 21 个候选，并在 formula-disjoint confirmation 中保留两条：

| 父结构条件 | 观察事件 | confirmation effect | formula-cluster 95% CI | BH q | confirmation 支持 |
|---|---|---:|---:|---:|---:|
| `fr_NH0` | C2H6N, 44.0495 Da | 0.07257 | [0.02794, 0.12262] | 0.02520 | 107 molecules / 70 formulas |
| `fr_piperzine` | C2H6N, 44.0495 Da | 0.15880 | [0.06036, 0.25965] | 0.03150 | 25 molecules / 24 formulas |

正面证据是：formula-disjoint、within-formula structure negative controls、21 个 confirmation hypotheses 统一做 BH 校正，且没有查看 embedding evaluation folds 或 outer fold。

但不能越过以下边界：

- 报告明确限定为 `association_not_mechanistic_claim=true`；它测量的是峰出现率差，不是碎裂机理的直接证据。
- 两条规则共享同一个 C2H6N observation channel，且都是含氮结构谓词；不能算两个独立化学机制。
- 知识库中附带的文献来自 observation record，并没有直接验证本次新挖掘的 parent-predicate association。
- 映射到实际 action graph 后，discovery 只有 10 queries / 9 formulas，confirmation 只有 6 queries / 5 formulas；`CHEMAWARE_RULE_ACTION_COVERAGE_FAIL`，四个覆盖门全部失败，`formal_training_authorized=false`。
- 没有 corrected/introduced、没有 matched action controls、没有 clean embedding 结果。

**裁决**：两条记录只保留为待外部复核的知识库种子，不得称作“有效纠错规则”或“微调监督资产”。

### 3.4 五视图谱学共识：最强的内部稀疏纠错信号，但不属于化学规则或 embedding

`data/validation/chemaware_spectral_consensus_applicability_v4_frozen/report.json` 显示：

- discovery formula-grouped OOF：34 次触发，30/4，风险效用 22，`+0.2356 pp`；
- formula-disjoint confirmation：4 次触发，4/0，`+0.1133 pp`，公式簇区间约 `[+0.0269,+0.2313] pp`；
- 100 次标签置乱的经验 `p=0.0198`，分辨率粗且置乱中有一次效用达到 6；
- gate 使用 DreaMS margin、五种参考谱相似度 margin、投票和候选数，部署时必须读取整个 candidate reference set。

随后冻结全部参数并一次性消费内部 test：3,539 个可评价 query 中触发 9 次，9/0，Recall@1 `+0.254 pp`，formula-cluster 区间 `[+0.086,+0.448] pp`。这是本轮审计中最强的冻结内部证据。

但它仍有清晰上限：

1. v1-v4 开发过程重复查看同一个 confirmation，v4 的 4/0 不能视为全新独立确认；真正更强的是随后一次性 test 的 9/0。
2. 触发率仅 `9/3539=0.254%`，证明高精度小区域，不证明广覆盖。
3. 五个 view 高度相关；历史审计给出的错误 query 有效秩约 1.56–1.60，不能称作五种独立证据。
4. 它是候选侧谱学 gate，不使用分子结构规则，也不是只输入 clean spectrum 的 embedding。
5. test 仍来自本项目内部数据，不是跨实验室、跨数据集外部验证。

**裁决**：保留为“内部冻结、稀疏、高精度的候选侧谱学纠错信号”；它可作为后续方法必须达到的 safety/oracle benchmark，但不能归入 ChemAware 共享 embedding 成果。

### 3.5 corrected full graph 上的 shared-embedding 基线：训练骨架本身已经证明有效

`data/validation/chemaware_full_align_full_errorcurr500_v1/report.json` 是不能遗漏的正结果。它在 corrected 83,619-query graph 上训练 spectrum-only shared residual adapter，并在 formula-disjoint inner 的 1,929 queries / 1,210 formula clusters 上得到：

- Recall@1：0.904614 → 0.908243，即 `+0.3629 pp`；
- MRR：`+0.1970 pp`；
- 9 corrected / 2 introduced；
- formula-cluster 95% CI：`[+0.0964,+0.7851] pp`；
- preservation mean：0.998208；所有四个预设 gate 通过。

同一协议下 uniform continuation 为 `+0.2592 pp`、7/2，但区间跨零；error-conditioned formula-equal arm 同为 `+0.2592 pp`，区间为正。主结果说明“完整候选边界 + error-conditioned identity-equal continuation”可以稳定改善 official shared embedding。它不是化学归因，却是 ChemAware 注入必须超过的强基线和已经验证的训练骨架。

Phase A 的八臂正式 Slurm 结果进一步复现了共同 continuation 增益：`none`、mass 和 rule-mass 等多臂均达到 `+0.4666 pp`、12/3，公式簇区间为正。正确归因是“共享训练骨架有效、原 KL 化学传递没有提供额外增量”，而不是“ChemAware 一无所获”。

### 3.6 mass / rule-mass clean-visible shared kernel：强开发成果与后续稳定性问题必须同时保留

`data/validation/chemaware_rule_mass_pairfirst_full_inner_v1/report.json` 在同一 1,929-query held inner、pair-first molecule aggregation 下给出：

- mass shared kernel：`+0.8813 pp`、20/3；
- rule-response：`+1.2442 pp`、33/9；
- rule-mass：`+1.2442 pp`、32/8，formula-cluster 95% CI `[+0.6491,+1.7847] pp`；
- rule-mass 相对 mass：`+0.3629 pp`，配对区间 `[+0.0551,+0.7637] pp`；
- rule-mass 相对 mass-shifted rule control：`+1.0368 pp`，配对区间 `[+0.3173,+1.1428] pp`。

这是真实的 one-spectrum-to-one-embedding、query/reference 同函数、candidate-structure-free、pair-fusion-before-molecule-max 的 **clean-visible shared-embedding development 成果**。它没有更新 DreaMS 权重，输出是拼接 kernel embedding；但“没有微调”不等于“不是 embedding 成果”。

后续 `data/validation/chemaware_psd_observation_locked_confirmation_v1/report.json` 在新的 576-query / 484-formula panel 上显示 mass 仅 `+0.3472 pp`、3/1、区间跨零，learned 120-channel observation embedding 为 `-0.5208 pp`、1/4。该 panel 使用的是重构后的 120 observation channels，并非原 rule-mass 的 214 neutral-loss + 102 fragment response 的原样复现，因此它不能事后抹除 `+1.2442 pp`，但确实证明该收益对规则定义、panel 与表示方式敏感，尚缺独立复现。

**裁决**：mass `+0.8813 pp` 与 rule-mass `+1.2442 pp` 必须保留为强开发结果；当前问题是跨规则版本和独立 panel 的稳定性，而不是“结果不存在”。

### 3.7 direct ICEBERG shared embedding：已经出现正向 proof-of-effect，但尚缺化学归因复现

`tasks/train_chemaware_iceberg_direct_shared.py` 的 `direct_dual_listwise` 实际优化：

```text
L = lambda_clean * CE(clean list)
  + lambda_action * CE(action list)
  + safety_weight * CE(protected clean list)
```

direct 模式将 consistency、margin floor 和 preserve 置零；现有 pilot 还关闭 peak contrast。该 loss 可以让 action view 本身更容易按 identity 排序，却没有直接约束“由 action 产生的参数更新必须改善未修改 clean query”。数学上，若 `g_a` 是 action 梯度、`g_c,val` 是 clean validation 梯度，小步更新对 clean loss 的一阶影响约为：

```text
Delta L_clean,val ≈ -eta <g_c,val, g_a>
```

而属于化学动作的特异增量应比较：

```text
Delta_specific ≈ -eta <g_c,val, g_target - g_matched_control>
```

该式只定义 target 与 matched control 的**配对归因统计量**。不得把
`g_target - g_matched_control` 直接作为 optimizer 梯度，因为这会最大化仍然具有
真实 identity 标签的 control loss；matched controls 必须 audit-only，
harmful/uncertain 的 corrective weight 必须为零而不是负数。

现有 `data/validation/chemaware_iceberg_direct_shared_correct_cpu_v2/report.json` 已经给出实际 clean shared-embedding 变化，而不是只有代码：

- inner-selected 49 queries：Recall@1 `+2.0408 pp`、MRR `+1.0204 pp`、1/0；
- inner-all 180 queries：Recall@1 `+0.5556 pp`、MRR `+0.2778 pp`、near Recall@1 `+0.7937 pp`、1/0；
- 部署输入仍是一张 clean spectrum，query/reference 使用同一个 encoder。

这证明 direct action training **曾经把参数更新传回 clean embedding 并改变真实候选排名**。但它仍是 development-only 单臂：formula bootstrap 下界为 0，outer 未评估，preservation 与 clipping gate 未通过，也没有同批次 clean-duplicate、candidate-swapped 和 peak-permuted 结果。因此它是 learned shared-embedding proof-of-effect，不是稳定化学特异增益。

该 proof-of-effect 使用的旧版 `direct_dual_listwise` 将 consistency、margin floor 和 preserve 置零；action CE 也没有直接约束相对 matched control 的 clean transfer。当前代码已新增 `direct_guarded_listwise`：仍不读取教师分数、教师 embedding 或一致性蒸馏，但恢复 official-embedding preservation 与 clean margin floor，并把 pilot 的梯度裁剪上限从 1 调至 5。此修复解决安全锚和严重 clipping 问题；数学上，属于化学动作的特异增量仍必须比较：

```text
Delta_specific ≈ -eta <g_clean,val, g_target - g_matched_control>
```

**裁决**：保留 `+0.5556 pp` 全体与 `+2.0408 pp` selected 的 proof-of-effect；新的 A2 筛选已经证明当前三种 action-view 路由没有形成 matched-control 特异性，因此不再直接进入 corrected full graph 四臂训练。下一步先解析已有逐 action 产物并验证 clean-spectrum observability。

### 3.8 梯度瓶颈与注入修复：方向存在，旧 loss 让它被淹没

`data/validation/chemaware_rule_teacher_gradient_audit_v3/report.json` 在相同 official 初始化、相同候选 batch 和相同 13,638,656 个可训练参数下给出：

- base retrieval 梯度范数 `13.3783`；
- normalized rule-KL 梯度范数 `2.6564`，仅为 base 的 `19.86%`，与 base cosine `0.3254`；
- rank-equivalent KL 为 base 的 `23.99%`，cosine `0.4067`；
- positive margin transfer 为 base 的 `16.56%`，cosine `0.5713`；
- correct 与 shifted-rule 梯度范数分别为 `2.6564` 与 `1.2305`，差分范数 `2.1770`，11/11 个可训练张量均有非零差分。

这排除了“规则根本没有梯度”的解释。旧 Phase A/B 的科学瓶颈是：化学支路比 common continuation 弱约 4–6 倍，只部分同向，随后又遭遇高比例全局裁剪；因此训练结果主要复刻 continuation。它曾支持先把 auxiliary 梯度与 primary retrieval 梯度拆开并限制尺度，但只能解决优化冲突，不能证明或创造化学特异性。

作为历史修复，仓库已实现 `direct_projected_guarded`。它不是声称逐字复现经典 PCGrad，而是以 clean retrieval 为受保护主任务、只投影 action 辅助梯度的单侧 PCGrad-style 版本：

1. primary 梯度只来自 clean full-candidate ranking、正确样本 safety、official margin floor 与 official-embedding preservation；
2. auxiliary 梯度只来自正确/对照 action view 的 full-candidate listwise loss，不读取教师分数、教师 embedding 或候选侧 reranker 分数；
3. action 梯度与 primary 点积非正时移除冲突分量；存活 action 梯度范数强制不超过 primary 的 `0.25`；
4. 四个 arm 使用完全相同的采样、步数、学习率、安全池和梯度规则；只有动作身份不同；
5. 全局 grad clip 提高到 `5`，并逐 epoch 记录 raw cosine、冲突投影率、范数封顶率和实际 action/primary 比例；
6. prefix cache 只覆盖训练与 inner 评价真实访问行，并显式闭包 action/safety 的 query 与全部 candidate reference rows；若闭包遗漏会在建模前 fail-fast。Graphormer bias 从逐谱 `O(L²)` 物化改为 exact per-token `O(L)` 缓存，避免“Model ready 后长时间无输出”的假卡死；缓存每 30 秒输出进度与 ETA。2026-09-06 首版收缩曾漏入 safety query rows，触发 `KeyError: 144813`；补齐后又暴露旧 `paired_evaluation` 会在子集评价前映射整图 rows，触发 `KeyError: 2623`；随后空 near 分层的 `delta_near_recall1=None` 又在 gate 中被错误地与零比较。现已修复训练缓存闭包、subset-only rank/margin 评价和 optional-stratum gate：空分层明确记录为 not applicable，不再伪装为通过或失败；三类路径均有回归合同。

方法选择也曾按算力性价比做过收缩。经典 [PCGrad](https://papers.neurips.cc/paper_files/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html) 直接投影冲突任务梯度；[GradNorm](https://proceedings.mlr.press/v80/chen18a.html) 解决任务损失尺度平衡，但不保证 action 不伤害 clean retrieval，也不解决化学身份特异性；[CAGrad](https://proceedings.neurips.cc/paper_files/paper/2021/hash/9d27fdf2477ffbff837d73ef7ae23db9-Abstract.html) 还要解多目标折衷；基于 clean validation 的 [meta-reweighting](https://proceedings.mlr.press/v80/ren18a.html) 需要额外无偏 meta split 和双层优化，当前小 action bank 下更容易造成验证泄漏和高方差。新的 A2 结果进一步表明：此时继续更换梯度冲突算法属于解决错误问题，因为公共 identity gradient 本身已经是正向的，真正缺失的是 correct 相对 controls 的可识别差异。

历史 Stage-1 按顺序停止只运行了 `clean_duplicate` 与 `correct_synthetic`；两臂 Recall@1 增量相同且都退化，因此没有继续运行两个 pseudo-action controls。随后零更新 A2 同时检查 correct、candidate-swapped 与 peak-permuted，确认当前 action CE、straight-through 和 pair routing 都没有 correct-specific influence。继续跑四臂训练已被停止，而不是等待补跑。

**裁决**：`direct_projected_guarded` 的静态编译、CPU 合同和安全工程实现仍是可复核事实，但它已不再是活跃训练候选；A2 的 absolute-positive/control-nonspecific 结果说明下一步应审计 boundary switch 与 clean observability，而不是再调 PCGrad、cap 或 optimizer。

### 3.9 Noise E4-A：历史正结果因数据语义更正而不能前移为当前证据

旧报告在 5 folds × 3 seeds 上记录 E4-A 平均 Recall@1 `+0.6362 pp`、MRR `+0.4246 pp`、near Recall@1 `+0.5300 pp`，累计 559 corrected / 104 introduced。单看旧 protocol，这是多折多 seed 的正向 shared-embedding 结果。

但 `docs/NOISE_CORRECTED_FULLGRAPH_METHOD_RESET_20260906.md` 已明确撤销旧正式图：旧 23,876-query cohort 源于对 `SIMULATION_CHALLENGE=False` 的错误语义解释，不能承载新结论。新 corrected development graph 为 83,619 queries；旧 N actions 仅覆盖 1,451/5,957 official errors（24.36%），即便零风险全纠正，query-specific 全图上限也只有 `+1.735 pp`。

**裁决**：`+0.6362 pp` 只能写成“旧 cohort 上的历史结果”，不能再称当前模型证据或 ChemAware 可用底座。checkpoint 和训练代码可作为工程初始化参考，但必须在 corrected graph、匹配 official baseline 下重新评价后才恢复证据资格。

### 3.10 BioAware 与 P2b：只能借鉴实验逻辑，不能借用性能数字

它们支持的共同经验是：候选上下文可能含有 DreaMS 缺失的判别信息；直接后处理增益不自动迁移到 shared encoder；必须设置 matched-capacity、identity-permuted、clean-duplicate 和等预算 continuation 对照。

它们不是 ChemAware 的训练结果，不能用于宣称 ChemAware 达到或接近 3–5 pp，也不能把候选排序增益相加到 embedding 指标上。

## 4. 风险偏倚矩阵

| 资产 | 选择独立性 | 确认独立性 | 多重选择控制 | 风险敏感评价 | 可复现工件 | 部署输入匹配 | 结论 |
|---|---|---|---|---|---|---|---|
| ICEBERG teacher | 差：error-enriched | 差：训练域同源 | 不适用/机制审计 | 有 corrected/preserved，但非自然率 | graph/report 哈希核对通过；预测张量本地缺失 | 否：候选结构 | headroom only |
| ICEBERG top-3 action | 21 settings 用 discovery 选择 | formula-disjoint fold 2 | 仅分割，无全局风险校正 | 1:1 净效用 +1；2:1 保守效用 −1 | action bank 本地缺失 | 动作仅训练时可用；已有小图 transfer 信号 | positive E1, safety-sensitive |
| 两条经验规则 | discovery/confirmation 分离 | formula-disjoint | 21 项 BH | 无纠错评价 | 核心输入哈希在；脚本哈希未写入报告 | 理论上训练时可用 | association seeds |
| 谱学共识 gate | formula OOF | 4/0 confirmation 被多版本重复查看；9/0 test 一次性 | 100 permutations，分辨率有限 | 9/0 为正 | frozen gate/ledger/hash 在 | 否：需要候选参考谱 | internal candidate signal |
| spectrum-only error-conditioned adapter | corrected full graph train/inner formula 隔离 | outer 未读 | 三个路由变体 | 9/2，CI 正 | report/preflight/hash 在 | 是 | passed embedding baseline |
| mass/rule kernel | 训练公式选 beta，1,929 held inner | 后续 576 panel 不是同一规则表示 | 开发 grid 已记录 | rule-mass 32/8，绝对及配对 CI 正；新 representation 不稳定 | report/hash 在 | 是：clean-only shared kernel | strong development result |
| direct ICEBERG trainer | 单臂开发 | 180-query inner | 无 matched arm | 1/0，但 CI 下界 0、preservation/clip 失败 | report/preflight 在 | 是：clean-only | learned proof-of-effect |
| Noise E4-A | 旧 protocol 多折 | 旧语义图 | 旧报告有 CI | 旧结果为正 | 历史工件在 | clean-only | current claim revoked |

## 5. 当前应当保留并向前推进的成果

### 5.1 已经取得的正成果

1. **结构条件教师有明确候选判别能力**：ICEBERG correct teacher 在 700-query error-enriched panel 上 Recall@1 为 0.5886，明显高于 candidate-swapped 0.1329 和 peak-permuted 0.2986；这是强 mechanism/headroom 证据。
2. **ICEBERG top-3 conflict attenuation 已产生 formula-disjoint 正向动作结果**：confirmation 为 3/2、净纠错 +1、Recall@1 `+0.7519 pp`，三个 margin 检验的 formula-cluster CI 下界均为正。这是当前最值得前移的 ChemAware 动作成果。
3. **发现两条经多重校正的结构条件—峰出现关联**：`fr_NH0` 与 `fr_piperzine` 对 C2H6N 峰的 confirmation effect 和 BH q 均通过预设统计门。这是化学规则库重构的实证起点。
4. **存在一次性冻结验证的高精度候选纠错区域**：谱学共识 gate 在内部 test 上 9/0、`+0.254 pp`、公式簇区间为正；这是当前最强的候选侧安全信号。
5. **clean-visible shared kernel 已取得显著开发增益**：mass `+0.8813 pp`；rule-mass `+1.2442 pp`、32/8，并显著超过 mass 与 shifted-rule 对照。
6. **shared-embedding 微调骨架已通过较大 inner gate**：spectrum-only error-conditioned adapter 在 1,929-query inner 上 `+0.3629 pp`、9/2、公式簇区间为正；Phase A continuation 又得到 `+0.4666 pp`、12/3。
7. **ICEBERG direct 已产生首次 learned clean-embedding 正向变化**：inner-all `+0.5556 pp`、1/0，selected `+2.0408 pp`；尚缺 matched controls 和稳定性，而不是没有发生迁移。
8. **已经定位 shared embedding 的具体证据断点**：现有瓶颈是让 chemical treatment 稳定超过强 continuation control，而不是“ChemAware 没有信息”或单纯“代码没跑起来”。
9. **已经完成但不再晋级的安全工程修复**：`direct_projected_guarded` 分离 primary/action 梯度、单侧投影冲突、将 action 范数封顶为 primary 的 25%，并恢复 clean preservation 与 margin floor；缓存范围和表示也已压缩。它是可运行的历史实现，但新的 A2 结果说明“减少冲突”并不等于“获得 correct-action specificity”，故当前不授权继续训练。

### 5.2 尚未完成、但不能据此把已有成果归零的层级

- chemical treatment 相对同预算 clean/matched-control 的稳定训练后增量尚未得到结果；
- action-specific clean-gradient transfer 已完成 24-action 首轮直接测量并失败：绝对 influence 为正，但 correct−两个 matched controls 的区间未严格为正；
- 外部 sealed validation 尚未完成；
- 因此现在不能把任何一个数字称作“已确认的化学特异 fine-tuning 增量”，但可以且必须完整报告 rule-mass `+1.2442 pp`、spectrum-only fine-tuning `+0.3629 pp`、ICEBERG direct proof-of-effect `+0.5556 pp`、ICEBERG E1 `+0.7519 pp` 和候选 gate `+0.254 pp`，并分别注明证据层级。

### 5.3 不再保留为正成果

- learned 120-channel rule embedding；
- 把 ICEBERG action-bank 的脚本 PASS 解释成“部署安全”或“shared embedding 已提升”；
- 把 direct trainer 的可运行性本身当成果；实际 `+0.5556 pp` proof-of-effect 另行保留；
- 旧 Noise E4-A 的 `+0.6362 pp` 作为当前 corrected-graph 结果；
- BioAware/P2b 的数值作为 ChemAware 或 embedding 结果。

## 6. 每项资产要“前移”必须补齐什么证据

### 6.1 ICEBERG conflict attenuation

必须依次完成，不能跳级：

1. **服务器已完成**：action bank、逐 query action、teacher prediction 和 split ledger 已哈希锁定，A2 重放误差小于 `3.58e-7`；本机仍缺 action bank 与逐 action CSV，不能在本地逐行复算。
2. **已完成但风险敏感**：唯一 setting 已冻结并通过 E1 margin 对照；同时必须保留 confirmation 的 3/2、1:1 净效用 +1 与 2:1 风险效用 −1。
3. **未完成**：仍需在未参与 ICEBERG 训练和动作选择的独立数据上验证 teacher 与动作，并分开报告自然分布和 error-enriched panel。
4. **A2 首轮已完成并失败**：target 的绝对 influence 为正，但未胜过 candidate-swapped 与 peak-permuted；matched-random/clean-duplicate 不再通过追加 GPU 审计来挽救当前路由。先对已有 CSV 做 reference-switch 与 raw/normalized 分解。
5. **当前禁止进入**：只有新的、预先冻结且满足 control 不反传、非负 correct signal 和 clean-observability 的 E2 路由通过后，才允许最小 pilot；不得把对照残差当训练梯度。
6. **当前禁止进入**：只有最小 pilot 先通过，才可用 corrected full graph 做 formula-disjoint 多折多 seed，并报告 Recall@1/5/10/20/50、MRR、micro/macro AUC、corrected/introduced、near-core、分层覆盖及 multiplicity-corrected CI。

### 6.2 两条经验规则

1. 外部谱库复现 parent predicate 与 C2H6N 峰的关联，并按 adduct、collision energy、instrument 分层。
2. 补足直接文献或碎裂路径证据；在此之前统一称 empirical association，不称 mechanism。
3. 扩充为多个独立 predicate/observation family，避免由同一 C2H6N channel 垄断证据。
4. 达到预设 query/formula 覆盖门后，才评价 support-boost 的 corrected/introduced；必须与 predicate-swapped、matched observed peak 和 clean duplicate 对照。
5. E1 和 E2 均通过后，才允许进入 shared embedding pilot。

### 6.3 谱学共识 gate

1. 保持 gate 完全冻结，在新的外部 dataset 一次性验证；不得再调 threshold。
2. 报告触发率、校准、不同仪器/碰撞能/adduct 的分层风险和所有 retrieval 指标。
3. 若用作训练路由器，必须证明 routed training 相比同数量 random/error-only sampling 有 clean embedding 增量；否则只保留为候选 reranking/safety oracle。

### 6.4 mass/rule kernel

保留原 pair-first rule-mass 定义、beta 与哈希，在新的 sealed panel 上做**原样复现**；不要拿改变为 120-channel learned observation 后的失败冒充原方法复现。必须同时报告 official、mass、rule-response、rule-mass、shifted-rule，并检验 rule-mass 相对 mass 和 shifted-rule 的配对增量。若原样复现成功，它可以直接作为 3,072-dimensional shared embedding 使用，也可作为后续 1,024-dimensional student 的逐候选残差教师。

### 6.5 Noise E4 checkpoint

先在 corrected 83,619-query graph 上对原 checkpoint 做只读完整评价，确认初始化、candidate graph 和 official baseline 完全匹配；只有结果重新成立，才允许把它作为 ChemAware pilot 的初始化。之后仍须设置 equal-budget clean continuation 和 matched-random，不能把历史数字当先验胜利。

## 7. 后续任何结果的最低报告模板

每个新实验必须同时给出：

1. **冻结主张**：主要终点、最小有意义效应、方向、风险权重和停止规则；
2. **数据流**：开发、选择、确认、test、external 的 formula/identity 重叠矩阵；
3. **完整分母**：总 query、可评价 query、动作覆盖、触发、corrected、introduced；
4. **匹配基线**：同一个 checkpoint、同一个 candidate graph、同一个 tie policy；
5. **因果对照**：clean duplicate、matched random、chemical identity permutation、geometric perturbation；
6. **全指标**：Recall@1/5/10/20/50、MRR、micro/macro AUC，以及预设 safety metrics；
7. **不确定性**：以 formula 为聚类单位的 CI；多 hypothesis/arm/metric 做预设 multiplicity correction；
8. **逐案例账本**：每个纠错和引错的 query、formula、候选变化、动作和分数；
9. **复现信息**：代码 commit、脚本哈希、输入/输出哈希、环境、命令、随机种子和失败运行；
10. **主张边界**：明确它是 teacher、frozen action、reranker、shared embedding 还是 external generalization。

只报告最好 arm、只报告 Recall@1、只报平均值不报引错、从开发集挑结果后称 confirmation、或把候选侧后处理写成 embedding 提升，均视为证据不合格。

## 8. 最终结论与推进条件

严格审计后，最重要的修正有三条：

1. **ChemAware 已有 shared-embedding 成果**：rule-mass clean-visible kernel 为 `+1.2442 pp` 的开发结果；error-conditioned learned adapter 为 `+0.3629 pp` 的通过结果；ICEBERG direct 为 `+0.5556 pp` 的 learned proof-of-effect。三者性质不同，必须分别陈述，但绝不能概括为零。
2. **ChemAware 已有化学/结构条件成果**：ICEBERG top-3 E1 为 `+0.7519 pp`、3/2；两条经验规则通过 formula-disjoint 与 BH 门；ICEBERG teacher 对匹配置乱有大幅候选判别优势。
3. **ChemAware 已有候选侧安全成果**：谱学共识一次性内部 test 为 9/0、`+0.254 pp`。它与 P2b 的广覆盖但 near-core 不安全形成互补：前者可作高精度 gate，后者只能作受限候选特征。
4. **尚未完成的核心归因问题已经进一步收缩**：在进入训练后性能比较前，必须先确认 correct action 是否实际改变 reference/candidate boundary，以及这种差异能否由未修改 clean spectrum 在 formula/identity 隔离条件下学到。当前三种 action-view 路由没有通过这一特异性门。

当前不再预设 shared embedding 微调为主线，也不提交 `direct_projected_guarded`。先用既有 `per_action.csv` 做零 GPU 的 reference-pair switch、raw/normalized influence、action role/fold 分层及 E1--E2 相关性审计，再做 clean-spectrum observability gate。若 correct-specific boundary event 可由 clean spectrum 跨 formula 学到，才设计只让 correct 非负信号反传、controls audit-only 的最小 embedding pilot；若不可观测，则把 ICEBERG E1 保留为 candidate-conditioned 证据，进入 reranker/hybrid 的综合研判。同时保留 **rule-mass shared kernel + 冻结谱学 gate** 作为无需等待微调的候选侧/扩展 embedding 基准。

## 9. 核心证据定位

- ICEBERG teacher：`data/validation/chemaware_full_manifest_iceberg_teacher_700_v1/report.json`
- ICEBERG action-bank 审计脚本：`tasks/audit_chemaware_direct_action_bank.py`
- ICEBERG direct trainer：`tasks/train_chemaware_iceberg_direct_shared.py`
- direct gradient core：`tasks/chemaware_direct_training_core.py`
- direct four-arm Slurm：`tasks/run_chemaware_direct_action_views_pilot.sbatch`
- direct causal summarizer：`tasks/summarize_chemaware_direct_action_views.py`
- rule-teacher gradient audit：`data/validation/chemaware_rule_teacher_gradient_audit_v3/report.json`
- 经验规则：`data/validation/chemaware_empirical_structure_rules_v1/report.json`
- 经验规则 action coverage：`data/validation/chemaware_empirical_rule_action_coverage_v1/report.json`
- 当前知识库：`dreams/models/chem_aware/chem_action_knowledge_v3.json`
- 谱学共识 frozen gate：`data/validation/chemaware_spectral_consensus_applicability_v4_frozen/report.json`
- 谱学共识一次性 test 记录：`docs/CHEMAWARE_CHEMICAL_LIBRARY_REASSESSMENT_20260902.md`
- locked PSD/rule confirmation：`data/validation/chemaware_psd_observation_locked_confirmation_v1/report.json`
- mass 开发记录：`data/validation/chemaware_mass_kernel_embedding_full_inner_v1/report.json`、`data/validation/chemaware_mass_pair_pairfirst_full_inner_v2/report.json`
- Noise corrected-graph 撤销记录：`docs/NOISE_CORRECTED_FULLGRAPH_METHOD_RESET_20260906.md`

以上路径是证据源，不意味着其中每份旧报告的 `PASS` 或文字结论自动获得本次严格审计认可；本文件的裁决以原始计数、split、对照和适用边界为准。
