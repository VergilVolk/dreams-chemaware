# ChemAware 动作资产与迁移证据总账（决策前审计）

> **证据状态更正（2026-09-06）**：本文是第一轮盘点稿。后续严格审计保留了 ICEBERG top-3 的 E1 正向动作结果、两条经验化学关联和冻结候选侧纠错结果，同时把它们与尚未完成的 shared-embedding 验证分开。24-action A2 筛选现已完成：三条非 naive 路由均有绝对正的 clean influence，但均未显著超过 matched controls；详见 [A2 梯度筛选结果](./CHEMAWARE_A2_GRADIENT_TRANSFER_SCREEN_RESULT_20260906.md)。关于结果强度、复现边界和下一阶段条件，以该结果与 [ChemAware 保留成果严格证据审计](./CHEMAWARE_RETAINED_ASSET_STRICT_EVIDENCE_AUDIT_20260906.md) 为准；不得把“A2 当前路由失败”误写成“ChemAware 成果为零”。

**日期**：2026-09-06  
**状态**：动作盘点完成；A2 首轮机制筛选失败；路线决策尚未作出；禁止把本文件解释为已选择 embedding 微调或候选重排  
**范围**：只讨论 ChemAware 及与其直接相关的 Noise/BioAware 动作、教师、训练注入经验

## 1. 当前唯一 priority

当前任务不是立即实现一个新排序器，也不是立即继续提交共享 embedding 微调。当前唯一优先级是：

1. 把仓库中已经尝试过、曾出现正信号或具有明确科学含义的动作全部整理出来；
2. 把“冻结动作有效”“教师可区分”“梯度可到达”“共享 embedding 真正提升”四种证据拆开；
3. 解释每个动作为什么有效或为什么没有迁移，提炼共同经验；
4. 在证据总账完成后，再决定下一阶段是 embedding、候选重排，还是二者并行。

在第 4 步之前，不把任何一种实现形式预设为答案。

## 2. 统一证据等级

| 等级 | 问题 | 合格证据 | 不能冒充什么 |
|---|---|---|---|
| A0：化学/谱学定义 | 动作是否有明确、可执行的对象与方向？ | 父结构条件、谱峰事件、加合物/离子模式、允许的强度变换均明确 | 不能说明它能纠错 |
| A1：冻结动作效应 | 在不更新模型时，动作能否修正当前候选边界？ | held-formula、corrected/introduced、matched control、CI | 不能说明 embedding 会学会该能力 |
| A2：梯度可迁移性 | 动作产生的更新是否会改善干净输入？ | target-control 增量梯度、clean validation influence、分支范数与夹角 | 不能只用“梯度非零”代替 |
| A3：共享 embedding 增益 | 部署时只输入干净谱图，是否仍优于等预算控制？ | 同图、同初始化、同训练量下超过 clean/matched control，公式聚类 CI 为正 | 不能用动作 headroom 或后处理增益代替 |
| A4：外部泛化 | 是否能在未参与选择的独立数据复现？ | sealed external set、多折多 seed、完整安全指标 | 不能用反复查看的内部折代替 |

过去最主要的概念错误，是把 A1 或“非零训练梯度”直接记成 A3。

## 3. ChemAware 自身动作资产

### 3.1 旧观察字典与质量规则

| 资产 | 真实含义 | 已有证据 | 当前裁决 |
|---|---|---|---|
| 335 条旧记录 | 文献/软件包中记录的碎片质量与中性丢失现象 | 316 条可读；没有可执行的父结构前提 | 只能作为候选观察，不是机制规则 |
| 120 个正离子 observation channels | 把旧记录去重后形成的 fragment/NL 质量通道 | locked 576-query confirmation 上 Recall@1 比 official 低 0.5208 pp | A0 不完整，A1 失败；禁止整体广播进训练 |
| mass CountSketch | 仅使用谱峰 m/z 与强度构造的部署可见核 | inner 1,929 identities 上曾有 +0.881 pp；locked 576-query 确认仅 +0.3472 pp，CI [-0.4132,+1.0331] pp | 是谱学表示资产，不是化学机制；证据不稳定 |
| rule-response / rule-mass | 观察通道响应与 mass kernel 拼接 | 原始 pair-first inner 上 rule-mass +1.244 pp，较 mass-only +0.363 pp；后续 Phase A 的 KL/PMT 注入未超过 no-teacher | clean-visible shared kernel 有显著开发信号；已失败的是当时的 KL/PMT 注入，不能据此抹除 kernel 结果 |
| shifted-rule / row-permuted control | 规则质量或行关系置换 | shifted PMT 反而优于正确规则 PMT | 证明旧规则优势主要不是正确化学身份信息 |

关键经验：质量通道可以提供额外的谱学分辨率，但“某个质量差出现过”不等于“当前母体可通过该机制产生此峰”。只有质量值、没有父结构条件和实验域条件的条目，不应得到化学监督权重。

### 3.2 新重构的结构条件经验规则

现有 v3 知识库只正式保留了两条 formula-disjoint confirmation 通过的经验关联：

| 父结构条件 | 观察峰 | confirmation 证据 | 覆盖 | 当前裁决 |
|---|---|---:|---:|---|
| `fr_NH0` | `C2H6N`, 44.0495 Da | effect 0.07257；95% CI [0.02794,0.12262]；BH q=0.0252 | 107 molecules / 70 formulas | A0/A1 规则入库通过 |
| `fr_piperzine` | `C2H6N`, 44.0495 Da | effect 0.15880；95% CI [0.06036,0.25965]；BH q=0.0315 | 25 molecules / 24 formulas | A0/A1 规则入库通过 |

但是把规则映射到实际训练 query 后，discovery 仅 10 queries / 9 formulas，confirmation 仅 6 queries / 5 formulas。覆盖门失败，因此：

- 两条规则是目前最干净的“化学规则种子”；
- 它们可用于验证规则执行器、控制设计和局部案例解释；
- 它们不足以单独承担全局 3–5 pp 的目标；
- 不能因为统计入库通过就启动 117M 参数模型的正式训练。

### 3.3 结构教师与局部碎片动作

| 教师/动作 | 设计 | 最好已有信号 | 主要否证 | 当前等级 |
|---|---|---|---|---|
| MolFormer 全分子几何 | 用 SMILES embedding 定义候选结构关系 | correct 与伪教师均能形成训练信号 | correct 未稳定超过 identity-permuted、random-marginal、same-formula mismatch；最终 embedding 几乎相同 | A2/A3 失败 |
| Morgan connectivity teacher | 同分子式候选内的 ECFP 几何 | 冻结线性 probe 在 922 queries 上比三种置乱平均高 6.69 pp | `corrected-2*introduced` 为负；真实 PEFT 无稳定 rank 变化；存在指纹碰撞 | 诊断资产，不是可部署或已迁移资产 |
| ChemBERTa residual | 跨模态结构残差 | 训练与选择流程可运行 | locked 576-query confirmation：0 corrected / 0 introduced，CI 跨零 | A1/A3 失败 |
| 单键切割碎片 | 候选结构单键切割后的可解释质量集合 | inner 全体 truth > hard-negative 为 62.07% | 21 个 official errors 中仅 4 个方向正确 | 全局相关但不对准错误边界 |
| MAGMa 多步碎片 | 多解候选碎片与 observed peak 匹配 | formula-held confirmation 中 differential 在 7 个 raw errors 上救回 2 个 | raw recurrence 本身更强；mask-rotated/structure-permuted 控制同样能救回错误；未过训练门 |
| MAGMa peak-token probe | 从最终 peak token 解码碎片 Morgan bits | correct AUPRC 0.42976 | 仅比 peak-permuted 高 0.00871，比 same-formula structure-permuted 高 0.00106 | 最终 token 几乎没有身份特异碎片可读出性 |
| motif-event interaction | 候选特异局部 Morgan 环境 × fragment/NL event | 4,916-query 版本可产生大量正的冻结策略修正 | correct 与 matched-capacity、结构置乱、峰置乱的配对差异 CI 均跨零；小版甚至显著弱于 base | A1 化学特异性失败 |

这些结果共同说明：全分子结构相似、局部碎片“可生成”、以及峰级结构标签，都没有自然地对准 DreaMS 当前真正犯错的候选边界。覆盖率高不等于错误纠正率高，标签能被读取也不等于其梯度会改善 clean retrieval。

### 3.4 ICEBERG 结构条件动作

ICEBERG 是目前 ChemAware 中最值得保留、同时最容易被夸大的动作来源。

#### 教师本身的候选判别能力

在 50-query action-rich inner panel 上：

- official DreaMS Recall@1 = 0.50；
- correct ICEBERG = 0.80；
- candidate-swapped = 0.04；
- peak-permuted = 0.42；
- official 的 25 个错误中，ICEBERG 救回 19 个。

这证明 ICEBERG 包含候选结构条件的碎裂判别信息；但 panel 很小且富集困难样本，只属于 A1 教师 headroom。

#### 已尝试的三类使用方式

| 方式 | 结果 | 裁决 |
|---|---|---|
| 候选分布蒸馏 | correct 与 spectrum-only、candidate-swapped 同为 90.617%；peak-permuted 更高 0.104 pp | 化学分布没有迁移 |
| 训练样本路由 | correct router 比 random router 低 0.156 pp | 教师适合选候选，不代表适合选训练样本 |
| observed-peak differential action | 用 true structure 与 hardest same-formula negative 的预测强度差，衰减更支持错误候选的 observed peaks | 当前唯一通过 matched action control 的 ICEBERG 动作族 |

#### 当前最强 ICEBERG 动作

冻结全局动作选择为：`conflict_attenuate`, strength `0.75`, top-k `3`。它只改变三个已观测峰的强度，不新增 m/z。

- discovery：276 queries，8 corrected / 2 introduced，Recall@1 +2.1739 pp；
- confirmation：133 queries，3 corrected / 2 introduced，Recall@1 +0.7519 pp；
- confirmation mean margin +0.01311；对 candidate-swapped +0.01242；对 peak-permuted +0.01064；三者 formula-cluster CI 下界均为正；
- 5,929 个已审计 correct actions 中，`corrective_rank=102`，`corrective_margin=1,075`，但按当前严格条件最终只有 117 个训练动作；
- 该 700-query teacher panel 明确偏向 official errors。后续 direct shared-embedding 单臂已在 180-query inner 上产生 +0.5556 pp 的 clean-embedding proof-of-effect，但未配齐同批 matched controls，不能据此完成化学归因。

因此它的准确裁决是：

> ICEBERG top-3 conflict attenuation 已通过 A1 冻结动作特异性门；direct training 也已出现 clean shared-embedding 正向 proof-of-effect。新的 24-action A2 审计证明 action 梯度能形成绝对正的 clean-margin influence，但 action-query、straight-through 和 action-routed pair 三条路由均未显著超过 candidate-swapped 与 peak-permuted，因此当前注入方式没有通过 action-specific 梯度门。A3 中 correct action 相对 clean-duplicate 和 pseudo-action controls 的严格增量仍未确认。

旧版 `direct_dual_listwise` 直接把 clean CE、action-view CE 和 safety CE 相加，只能保证模型学习 action view 的身份；它没有数学上保证 action 的改进会迁移回未修改的 clean spectrum，并把 `corrective_margin` 与真实 `corrective_rank` 混为同权监督。`direct_projected_guarded` 曾恢复 official-embedding preservation、clean margin floor，并对 action 辅助梯度执行单侧冲突投影与 25% 范数上限；这些是有效的安全工程修复，但 PCGrad/cap 只能处理冲突和尺度，不能创造 correct-control 特异性。结合 Stage-1 同预算结果和新的 A2 FAIL，它不再是当前被授权的下一训练方案，只保留为历史实现资产。

### 3.5 候选局部谱学动作（不是严格化学规则）

| 资产 | 冻结结果 | 价值 | 限制 |
|---|---|---|---|
| 五视图谱学共识门 | formula-disjoint confirmation 4 corrected / 0 introduced；历史独立 internal test 9 / 0 | 当前最稳定的高精度错误发现器与安全路由信号 | 读取候选参考谱；五个视图高度相关；不能称结构化学机制 |
| P2b neutral-loss fusion | development +3.91 pp；sealed P3 +1.07 pp | 证明局部 neutral-loss 几何能补 DreaMS | near-core -4.23 pp；是候选时后处理，不是 shared embedding |
| error-conditioned spectrum-only continuation | inner +0.363 pp，9 / 2，CI 为正 | 说明“找对错误并集中训练”本身有价值 | 没有新增化学信息，是所有 ChemAware 训练必须超过的控制 |

这组资产很重要，因为它们定义了强控制：任何结构化学动作如果只达到相似结果，就不能声称增益来自化学知识。

## 4. Noise/BioAware 能借鉴什么，不能照搬什么

### 4.1 Noise

Noise 中值得保留的动作资产：

- `candidate_gradient`：attenuation 0.50，成熟步数 3–6；
- `role_confounder`：attenuation 1.00，成熟步数 1–5；
- P-transfer 与 P-intensity：用于扩覆盖，但风险异质，不能合并成一个正监督池；
- clean/action 双视图、candidate boundary、preservation、margin floor、identity/formula-equal sampling、完整分支梯度审计。

但必须分清结果：

- E4-A 是当前最稳定的 shared-embedding 开发结果：5 folds × 3 seeds，平均 Recall@1 +0.6362 pp；
- 3.346–4.93 pp 是 action-rich cohort 的条件容量，不是模型提升；
- dynamic N+P 相对 matched-random 为 -0.0169 pp；
- candidate-boundary 相对 clean duplicate 为 -0.0675 pp，100% clipping；
- corrected full graph 上旧 N actions 仅覆盖 24.36% errors，即使全部零风险修正，全局 query-specific 上限也只有 +1.735 pp。

可迁移的不是某个已经成功的动作损失，而是三条方法经验：

1. action headroom 必须与 shared transfer 分账；
2. action、clean、safety 分支必须先做梯度范数/夹角和 clipping 审计；
3. 任何 treatment 都必须超过同预算 clean duplicate 与 matched random。

### 4.2 BioAware

BioAware candidate expert 在外部开发集上曾有 +3.47 至 +4.01 pp，并有 19/0 或 23/1 的冻结候选纠错；这证明候选上下文有价值，不证明 shared embedding 已获得该能力。其 B6 matched arms 在现有日志中没有产生 Top-1 改变。

BioAware 最值得借鉴的是流程：

- 先验证 candidate universe 完全一致；
- 测量专家增量梯度与 generic gradient 的范数和 cosine；
- 要求单步更新后真实 truth-vs-baseline-wrong clean margin 上升；
- 使用 direct-onehot 与 soft-teacher 成对臂，而不是只跑 treatment。

不能照搬的结论是“既然后处理能提高 4 pp，微调也应提高 4 pp”。B6 已说明这一步迁移并不自动成立。

## 5. 为什么现有动作没有自然变成 embedding 提升

设干净谱为 `x`，候选条件动作后的谱为 `T_a(x)`，共享编码器参数为 `theta`。现有直接训练近似优化：

```text
L = CE(theta; x, y) + lambda_a CE(theta; T_a(x), y) + lambda_s L_safety
```

动作冻结排序提高，只说明：

```text
margin(T_a(x)) > margin(x)
```

它不推出一次 action-view 更新会改善干净谱：

```text
margin_clean(theta - eta * grad L_action) > margin_clean(theta)
```

对小步更新，clean validation loss 的一阶变化由梯度内积控制：

```text
Delta L_clean ~= -eta * <g_clean_validation, g_action>
```

真正属于动作的增量还必须扣除等形控制：

```text
transfer_specificity(a)
  = - <g_clean_validation, g_target_action - g_matched_control>
```

这里的 target-control 差只能作为配对归因统计量，不能把
`g_target_action - g_matched_control` 当作 optimizer 梯度；后者会主动最大化仍然
保留真实 identity 标签的 control loss，违反 control 不反传和 harmful/uncertain
零 corrective weight 的既有合同。

因此：

- “动作后当前模型排得更好”与“用动作训练后干净谱排得更好”是两个不同问题；
- shared parameters 只保证梯度作用于同一参数，不保证作用方向正确；
- 如果 action view 已经变成容易样本，其 CE 梯度可能反而更小，最终仍由 clean continuation 主导；
- 117 个动作面对 117M 参数模型，若没有跨 query 的共同梯度方向，增加 epoch 只会放大过拟合或被 preservation 抵消；
- hard negative 若由冻结模型一次选定，模型更新后动作语义会陈旧；但在线刷新也不能替代化学特异性对照。

24-action A2 筛选现已补到总体层，但逐 action 的 raw gain、gradient norm、
reference-pair switch 和 E1--E2 相关性尚未汇总。完成这些 CPU 诊断与
clean-spectrum observability gate 之前，不指定下一种 loss。

## 6. 当前资产分级

### 保留为核心候选资产

1. **ICEBERG top-3 conflict attenuation**：目前最强的结构条件、matched-control 通过的冻结动作；停在 A1。
2. **两条结构条件 C2H6N 经验规则**：目前最严格的可执行化学规则种子；覆盖不足。
3. **高精度谱学共识门**：最稳定的纠错/安全路由信号；不能冒称化学规则。
4. **mass CountSketch**：部署可见的谱学补充表示；严格确认仍不确定。
5. **Noise E4-A checkpoint/trainer contract**：当前最可靠的 shared-embedding 工程与对照底座；不是 ChemAware 因果结果。

### 保留为诊断或未来候选排序资产

1. ICEBERG 完整候选分布；
2. Morgan frozen probe；
3. P2b neutral-loss fusion；
4. BioAware candidate expert；
5. MAGMa 多解碎片与单键切割集合。

这些资产都证明候选条件信息存在，但尚未证明适合压缩进单谱 embedding。

### 当前不应继续扩算力的方向

1. 120 个 observation channels 的整体广播；
2. 旧 335/3,151 条记录直接当机制规则；
3. MolFormer/ChemBERTa whole-molecule residual；
4. scalar PMT、rank-equivalent KL 和旧 rule-mass teacher；
5. motif-event hashed interaction；
6. ICEBERG candidate distribution 蒸馏或 sampling router；
7. 没有 matched action/clean control 的单臂 direct-action 训练。

## 7. 决策前必须补齐的矩阵

下一步不是先写最终 trainer，而是对保留的动作建立同一张 evidence matrix：

| 必测项 | target action | matched chemical control | matched geometric control | clean duplicate |
|---|---:|---:|---:|---:|
| 冻结 clean/action margin | 必须 | 必须 | 必须 | 必须 |
| corrected / introduced / candidate switch | 必须 | 必须 | 必须 | 必须 |
| action branch gradient norm | 必须 | 必须 | 必须 | 必须 |
| 与 clean-error gradient 的 cosine | 必须 | 必须 | 必须 | 必须 |
| 与 protected-correct gradient 的冲突率 | 必须 | 必须 | 必须 | 必须 |
| 单步更新后的 clean truth-vs-hardest-negative margin | 必须 | 必须 | 必须 | 必须 |
| formula-crossfit influence CI | 必须 | 必须 | 必须 | 必须 |
| 可覆盖 official errors 与理论净上限 | 必须 | 必须 | 必须 | 必须 |

优先对象应是：ICEBERG top-3 conflict attenuation、两条结构条件规则的 support boost、高精度谱学共识动作，以及 Noise 的 candidate-gradient/role-confounder 作为外部动作基线。

## 8. 完成矩阵后如何决策

### 若选择 embedding 微调

必须观察到：target-control 的 clean influence 为正、公式外推稳定、覆盖足够，并在同预算小 pilot 中超过 clean duplicate 和 matched random。否则不能因为动作 A1 很强就继续扩大微调。

### 若选择候选重排

适用条件是：候选结构/候选参考谱信号在 A1 很强，但无法从单张 clean spectrum 产生稳定的 A2/A3 迁移。此时重排不是“退而求其次”，而是尊重信息可观测边界。

### 若二者兼做

只能按信息类型分工：

- embedding 学习广覆盖、部署时从 clean spectrum 可见的稳定谱学不变量；
- reranker 使用稀疏、候选依赖、结构条件的信息；
- 两者必须分别报告独立增量，不能把串行总收益全部归给 ChemAware embedding。

## 9. 当前结论

目前还没有证据支持直接宣布“下一步应只做 embedding”或“下一步应只做 reranker”。已有证据支持的更精确结论是：

1. ChemAware 已经找到候选结构条件信息，但其有效性主要停留在冻结候选动作层；
2. 当前 shared-embedding 失败的核心不是模型容量不足：三条非 naive 路由都能改善各自 clean-input 候选边界，但这种方向没有显著超过 matched pseudo-actions，缺的是化学特异性而不是梯度可达性；
3. 真正的化学规则库仍过于稀疏，两条合格规则不能承担全局目标；
4. 谱学共识、P2b、BioAware、ICEBERG 均说明候选侧有增量，但也共同提示信息可能天然依赖候选上下文；
5. 下一项最高优先级工作是用既有 `per_action.csv` 完成 A2 的 reference-switch、raw-vs-normalized influence、action-role/fold 分层和 E1--E2 相关性，并增加 clean-spectrum observability gate；随后再决定 embedding、reranker 或 hybrid。

在该矩阵完成之前，不提交新的全局训练，不把动作 headroom 写成模型性能，也不以工程可运行代替科学路线成立。
