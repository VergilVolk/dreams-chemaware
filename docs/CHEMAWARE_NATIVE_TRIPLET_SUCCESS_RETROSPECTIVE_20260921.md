# ChemAware 原生 Triplet 微调成功复盘与历史路线统一裁决

日期：2026-09-21

状态：`ROLE3_POSITIVE / STRONGEST_CHEMAWARE_LEARNED_SHARED_FINETUNE_TO_DATE / CHEMICAL_ATTRIBUTION_PENDING`

## 0. 结论先行

这次结果在一个严格限定下确实是空前的：它是迄今 ChemAware 路线中，第一次在完整的 1,929-query role-3 图上，通过直接更新 DreaMS 权重，获得严格正公式簇区间的、幅度超过 1 pp 的共享 clean-spectrum embedding 增益。

正式结果为：

- official Recall@1：`0.9046137895`
- ChemAware native-triplet best Recall@1：`0.9227579057`
- Recall@1 增量：`+1.8144116122 pp`
- formula-cluster bootstrap 95% CI：`[+0.8154763981,+2.8703289304] pp`
- corrected / introduced：`57 / 22`
- 净纠正：`35`
- 风险效用 `corrected - 2*introduced`：`13`
- MRR：`+1.0789269051 pp`
- micro-AUC：`+0.8079210824 pp`
- macro-AUC：`+0.9342078491 pp`
- Recall@3：`+0.5184033178 pp`
- Recall@5：`+0.0518403318 pp`
- Recall@10：`+0.0518403318 pp`
- Recall@20/50：维持 `1.0`

但“空前”不能被扩大成以下错误表述：

1. 这不是 ChemAware 所有方法中的最大候选排序增益；候选重排器在同一 role-3 面板上曾达到 `+3.1104 pp` 和 `+3.9399 pp`。
2. 这不是新的 outer 或外部数据结果；formula role 3 已被历史开发反复使用。
3. 这不是已经完成化学因果归因的结果；当前尚未运行与 correct triplet 完全匹配的 official-hard-only 和 content-permuted triplet 训练对照。
4. 这不是“训练越久越好”。最佳模型在 global step 3,000，后期模型已明显退化。

因此最准确的总裁决是：

> ChemAware 首次把成熟候选侧化学证据，转换成大规模、身份标签正确、与 DreaMS 原生 cosine 检索目标同构的 hard-negative triplet 课程，并在极短的原生对比微调后获得当前最强的 learned shared-embedding 开发结果。结果证明“这套课程能够产生更好的共享 embedding”；其中有多少增量必须归因于正确化学语义，仍需同预算 null-triplet 对照确认。

## 1. 为什么它是 ChemAware 微调路线的实质突破

### 1.1 与历史 learned shared-embedding 结果的量级比较

| 历史资产 | 查询规模 | Recall@1 增量 | 严格含义 |
|---|---:|---:|---|
| error-conditioned spectrum-only adapter | 1,929 | `+0.3629 pp` | 训练骨架有效，但不是化学归因 |
| rule-kernel Phase A common continuation | 1,929 | `+0.4666 pp` | none、mass、rule-mass 基本相同，增益来自共同 continuation |
| 旧 ICEBERG direct shared proof | 180 | `+0.5556 pp` | 1/0，小样本，CI 下界为0，安全门失败 |
| orthogonal event adapter smoke | held fold | `+0.7258 pp` | 比 clean-only 少 `0.0518 pp`，动作增量未成立 |
| rule-mass explicit shared kernel | 1,929 | `+1.2442 pp` | 干净谱图共享核；没有更新 DreaMS 权重 |
| **本次 native-triplet best** | **1,929** | **`+1.8144 pp`** | **直接更新 DreaMS，共享 embedding，CI 严格为正** |

本次点估计比旧 `+0.4666 pp` common continuation 高 `1.3478 pp`，比旧 `+0.5556 pp` direct proof 高 `1.2588 pp`，比未更新权重的 rule-mass shared kernel 高 `0.5702 pp`。这些差值是历史定位，不是跨实验配对因果估计。

### 1.2 最佳模型实际是一个短程、低剂量更新

从 PyTorch Lightning checkpoint 元数据直接读取：

| Checkpoint | Epoch | Global step | Recall@1 增量 | 与 official embedding 平均 cosine |
|---|---:|---:|---:|---:|
| `best.ckpt` | 1 | 3,000 | `+1.8144 pp` | `0.7212` |
| `last.ckpt` | 214 | 412,000 | `+0.4147 pp`，CI跨0 | `0.3057` |

epoch 从0计数，因此 best 大约产生于第二个 epoch。每个 epoch 为1,922个 batch；step 3,000 相当于约1.56个 epoch。训练集有7,688个 triplet event，batch size 为4，因此这一最佳点恰好属于“一到两遍高质量课程”的量级。

后期 `last.ckpt` 的 Recall@3、Recall@5 分别下降 `0.1555 pp` 和 `0.2074 pp`，并产生71个纠正与63个新错误。它证明过度训练会持续扭曲原始 DreaMS 全局几何；15小时长跑的大部分计算没有带来收益。

## 2. 这次方法到底做对了什么

### 2.1 化学不再充当软标签，而只负责挑选“该学哪个错误边界”

对 query 谱图 `q`：

- `p` 是同一真实分子身份的其它参考谱；
- `n` 是不同分子身份的候选参考谱；
- ChemAware 只决定哪些错误候选分子进入 hard-negative 集合。

训练仍使用 DreaMS 原生目标：

```text
L(q,p,n) = max(0, margin - cos(f(q),f(p)) + cos(f(q),f(n))).
```

这里所有监督标签都是真实身份关系。化学规则不会直接规定 embedding 数值、不会把规则分数当身份概率，也不会要求学生复制 candidate-conditioned teacher 的绝对分值。

这是与历史路线最根本的区别：

```text
历史：化学教师分数/动作位移 -> 强迫 shared embedding 回归该数值
本次：化学证据 -> 选择有信息的错误候选 -> 原生身份 triplet 学习
```

即使某部分化学教师强度不可由单张谱恢复，它也不会作为不可实现的连续 target 被硬塞进1,024维向量；只有当被选中的负例在谱图中确实存在可学习差异时，身份 triplet 才形成一致梯度。

### 2.2 训练目标和正式检索指标第一次真正同构

正式检索使用 query/reference embedding cosine，并在候选分子内部对参考谱取最大值。当前训练也直接优化 query、同身份参考谱和错误候选参考谱之间的 cosine margin。

这解决了旧方法的三重错位：

1. 不再拟合与最终名次不等价的 teacher KL、Huber residual 或动作幅度；
2. 不再先把化学动作压成一个丢失候选身份的标量；
3. 不再把 candidate-conditioned 非对称分数强行解释为一个共享对称 Gram 几何。

因为 query、positive、negative 都经过同一个 encoder，所得相似度天然属于同一个共享、对称、可部署空间。

### 2.3 Triplet 质量和覆盖第一次达到足以微调116M模型的规模

本次训练池不是几十个动作，而是：

- 7,688个全局去重的 query-negative 分子对；
- 4,032个独立 anchor queries；
- 2,518个公式；
- 49,980条 positive reference edges；
- 55,663条 negative reference edges；
- 6,658个 action-hard events；
- 2,295个 correct-vs-multinull specific-hard events；
- 所有 event 都有真实同身份 positive 和真实异身份 negative；
- 没有通过复制 triplet 虚增数量。

每个 query 的候选来源包括：

1. official DreaMS 当前最难错误候选；
2. 最多2个由正确规则中心排序出的 action-hard 错误候选；
3. 最多1个正确规则中心相对三个 content-permuted 中心具有优势的 specific-hard 错误候选。

这形成了“全覆盖稳定底座 + 化学困难样本加密”，而不是只训练少数 oracle-rescued errors。

### 2.4 多参考谱随机采样提高了训练密度

每个 triplet event 保存同身份分子的全部可用 positive references，以及错误候选分子的全部 negative references。DreaMS 原生 `ContrastiveSpectraDataset` 每次访问动态抽取1个 positive 和1个 negative。

因此同一个 query-negative 分子事件在不同 epoch 可以看到不同参考谱组合，7,688个事件对应的实际谱图配对远多于7,688。这比历史固定单 pair 路由更不容易把某一次 reference 选择偶然性当成化学规律。

### 2.5 直接继承成熟 DreaMS 训练栈，去掉了自造注入器

本次只自定义 triplet 构造，以下全部复用原生实现：

- `ContrastiveSpectraDataset`
- `ContrastiveHead`
- 官方 spectrum preprocessor
- 官方 projection head
- cosine triplet-margin loss
- Adam optimizer路径
- 同一个116M backbone

初始化来自官方 `official_embedding_slim.pt`，学习率为 `5e-6`，margin 为 `0.1`，batch size 为4，全 backbone 从epoch 0可训练。

过去大量时间消耗在自造 adapter、PEFT parametrization、candidate residual、frozen-prefix cache、特殊梯度路由和多重损失权重上；即使工程正确，它们也改变了已被 DreaMS 验证过的优化几何。本次把创新严格限制在“训练样本选择”，从而保留成熟训练动力学。

### 2.6 角色隔离和身份审计阻止了最危险的伪提升

- formula roles 0/1：triplet优化数据；
- formula role 2：构造审计与配方选择；
- formula role 3：模型开发评价；
- 本次运行没有读取 formula role 4。

训练和验证池的全部 positive/negative identity edges 均通过审计；official role-3 rank replay mismatch 为0。虽然 role 3 已是历史反复使用的开发面板，不再具备独立 confirmation 资格，但当前数值至少不是候选图漂移、baseline错配或身份边错误造成的。

## 3. 历史 ChemAware 为什么长期表现差

### 3.1 科学对象错了：把 candidate-conditioned 能力当作 spectrum-only 能力

ICEBERG、规则矩阵和候选 residual 能看到候选结构、候选参考谱或候选专属化学响应；部署时单张 clean spectrum encoder 看不到这些特权变量。历史方案却经常直接要求 shared embedding 复制候选教师输出。

这不是简单的模型容量不足，而是可观测性缺口：教师知道的信息不一定是学生输入的函数。

### 3.2 目标几何错了：非对称候选残差未必能装进共享点积

一个共享 embedding 必须产生对称、全局一致的相似度矩阵。逐 query、逐 candidate 的残差可能互相冲突，也可能不是任何共享 PSD Gram 矩阵的变化。旧方法常在没有验证这一点时直接蒸馏。

本次 triplet 不预先指定完整目标矩阵，只施加身份正确的局部次序约束；共享 encoder 自己寻找可实现的几何。

### 3.3 动作经过 pair selection 后丢失了化学区别

A2审计发现：

- correct 与 candidate-swapped 在16/24动作上选择同一 reference pair；
- correct 与 peak-permuted 在19/24动作上选择同一 reference pair。

一旦 pair 相同，单位范数梯度方向也相同，所谓“正确化学动作”只剩下标量强度差。继续调学习率、epoch或gradient cap无法恢复已经在表示步骤中丢失的信息。

本次不再压缩成一个单 pair；它保留错误候选分子的多参考谱集合，并通过数千个不同 query-negative 边界训练。

### 3.4 数据太少、过度富集错误、覆盖不足

历史关键动作经常只有24、49、133、180或272个 query，并从 official errors 中筛出，因此适合证明 headroom，不适合稳定微调116M参数模型。direct-prior全图方案又只有258个active events，覆盖仍不足。

本次覆盖4,032个 query 和2,518个公式，且包含每个 query 的 official-hard 边界，不再仅依赖极少数事后可纠正动作。

### 3.5 损失下降不等于候选名次改善

历史 rule-selected boundary adapter 的训练 pair loss 曾下降约63%，但 fold-3 Recall@1 为0变化；crossview event loss也曾从0.12909降到0.01649，而检索只提高0.0829 pp、区间跨0。

这些失败证明模型可以拟合动作损失，却不一定修正正式 top-1 边界。本次损失本身就是 positive-vs-negative cosine ranking constraint，因此目标更接近最终名次。

### 3.6 化学增量长期被普通 continuation 混淆

rule-kernel Phase A 中：

- none：`+0.4666 pp`
- mass：`+0.4666 pp`
- rule-mass：`+0.4666 pp`
- rule-response：`+0.4147 pp`

当所有臂几乎相同，结果只能归因于共同训练过程，不能归因于化学教师。旧项目多次把“比official高”误当作“化学有效”。

当前 `+1.8144 pp` 显著大于这些历史普通 continuation 点估计，但由于没有在本次相同池、相同3,000 steps下训练 official-only/null-triplet controls，仍不能做严格化学归因。

### 3.7 优化器和长训练曾扭曲化学剂量

旧 residual 四臂所有 optimizer step 都发生gradient clipping；global gradient ratio、clipping和Adam自适应共同改变了原始化学dose。不同teacher强度不再对应不同参数位移。

本次不再额外混合chemical loss，因此没有“化学梯度权重”可被clip重新解释。但当前 last checkpoint 仍显示普通 triplet 训练本身也会过拟合；关键安全机制应是短程训练与冻结 checkpoint，而不是继续增加epoch。

### 3.8 工程复杂性掩盖了科学错位

历史缺模块、prefix cache漏行、设备不一致、CLI漂移、role matrix、恢复状态机等问题消耗了大量资源。更严重的是，工程终于跑通后容易把“PASS contracts”误认为“科学目标正确”。

本次成功并不是因为写了更复杂的注入器，而是因为删掉了大多数特殊机制，只保留经审计的triplet构造与官方训练栈。

## 4. 以前的重排器到底提高了多少

“以前的重排器”至少对应三套不同资产，必须分开报告。

### 4.1 ChemAware canonical truth-blind candidate policy

在同一1,929-query role-3开发面板上：

- Recall@1：`0.9046137895 -> 0.9357179886`
- 增量：`+3.1104199067 pp`
- corrected / introduced：`64 / 4`
- formula-cluster 95% CI：`[+2.2199,+4.0683] pp`
- 风险效用：`64 - 2*4 = 56`

这是candidate-conditioned开发重排结果，不是shared embedding，也没有完成合法的新outer评估。

### 4.2 ChemAware multi-null symmetric residual V2

这是后续更强、控制更完整的ChemAware候选重排器。在同一role-3面板上：

- Recall@1：`0.9046137895 -> 0.9440124417`
- 增量：`+3.9398652151 pp`
- corrected / introduced：`93 / 17`
- formula-cluster 95% CI：`[+2.8191,+5.1921] pp`
- 风险效用：`93 - 2*17 = 59`
- 相对candidate-rotated truth-blind control：`+3.3178 pp`
- paired formula-cluster 95% CI：`[+2.3060,+4.3776] pp`

这说明正确多空化学语义在候选侧确有很强的边界识别能力。但role 3是已使用开发面板，旧role 4也已被其它实验消费；该结果不能冒充新的outer确认。

### 4.3 P2b固定rank fusion：不是ChemAware V2

P2b是独立的固定权重局部rank fusion：

- nested formula-OOF开发，5,037 queries：`+3.91 pp`，280/83，CI `[+2.91,+4.87] pp`
- sealed P3-main，3,000 queries：`+1.07 pp`，89/57，CI `[+0.24,+1.89] pp`
- P3 near-core，496 queries：`-4.23 pp`，20/41，区间全负

因此“P2b提高约4 pp”只适用于开发面板；它在sealed主面板上是`+1.07 pp`，并在near-core明确有害。

## 5. 重排器与本次embedding之间的真正关系

同一1,929-query面板上：

| 方法 | 净纠正 | Recall@1增量 |
|---|---:|---:|
| ChemAware canonical reranker | 60 | `+3.1104 pp` |
| ChemAware multi-null V2 reranker | 76 | `+3.9399 pp` |
| 本次 native-triplet embedding | 35 | `+1.8144 pp` |

本次embedding的点增益相当于V2候选重排点增益的46.1%，相当于canonical reranker的58.3%。这只能作为同面板的描述性“headroom保留率”，不能称为严格蒸馏效率，因为本次没有回归V2最终策略输出，而是从同源多空证据中挖掘hard negatives。

真正的重要变化是：

```text
候选重排器告诉我们“哪些错误候选值得处理”
              ↓
triplet构造把它变成“同身份positive vs 该错误negative”
              ↓
DreaMS原生loss只学习可由谱图支持的那部分区分
              ↓
部署时不再需要候选结构、规则分数或重排器
```

因此这不是传统的teacher-logit蒸馏，而是privileged hard-negative curriculum transfer。

## 6. 当前结果仍暴露出的三个问题

### 6.1 化学特异归因尚未完成

本次训练池同时包含每个query的official-hard negative与ChemAware action-hard/specific-hard negatives。没有相同数量、相同难度、相同参考谱复用和相同3,000-step预算的null训练臂，因此不能把全部`+1.8144 pp`归因于正确化学规则。

### 6.2 全局margin并未全面改善

mean positive margin从`0.33270`下降到`0.32684`，但Recall@1、MRR和AUC上升。这说明模型修正了一批关键top-1边界，同时压缩了许多已正确query的富余margin。它不是全局无代价改善；后续必须监测校准、跨库检索和安全子群。

### 6.3 checkpoint选择仍有随机性

当前ModelCheckpoint每1,000个train steps监控`Train loss`，而DreaMS模块记录的是batch-level loss。best恰好落在step 3,000，可能同时包含真实早停最优与batch噪声。结果本身已经冻结有效，但下一次正式实验必须预注册固定step快照或平滑train loss，不能让随机单batch决定唯一release checkpoint。

## 7. 最低成本的下一项决定性实验

不要再跑301 epochs。固定官方初始化、学习率`5e-6`、batch size 4、margin 0.1，并只保存step 1,000/2,000/3,000/4,000。构造以下同预算训练臂：

1. `official_hard_only`：只使用official当前最难错误候选；
2. `correct_multinull_hard`：当前成功池；
3. `content_permuted_A/B/C`：分别使用三个匹配null中心构造的hard negatives；
4. 可选`difficulty_matched_random`：匹配official rank、候选数、参考谱数和公式分布。

必须固定：

- 完全相同的query/formula覆盖；
- 完全相同的triplet event数；
- 完全相同的positive/negative reference multiplicity分布；
- 同一初始化、step数、batch order、seed和评估协议；
- correct相对每个control的paired formula-cluster CI；
- Recall@1/3/5/10、MRR、micro/macro AUC、margin与embedding preservation。

因为当前最佳点只有3,000 steps，这一归因实验的单臂成本应远低于本次15小时长跑。它回答的不是“还能不能再高一点”，而是最重要的科学问题：

> `+1.8144 pp`中有多少来自成熟DreaMS hard-negative continuation，有多少来自正确ChemAware语义选择的独立增量？

role 3只能作为已使用开发面板做机制复算；论文级泛化结论必须另建未使用的新面板或外部来源，不能继续把role 3称作独立确认，也不能重新利用已消费的旧role 4调参。

## 8. 允许与禁止的表述

允许：

> 在已使用的formula-role-3开发图上，ChemAware多空证据选择的身份监督hard-negative triplet，经DreaMS原生短程微调后，使共享clean-spectrum embedding的Recall@1提高1.8144个百分点，公式簇bootstrap区间严格高于零；这是目前ChemAware learned shared-embedding微调路线的最强结果。

禁止：

- “化学规则已经独立贡献+1.8144 pp”；
- “outer提高1.8144 pp”；
- “达到3–5 pp shared embedding提升”；
- “训练213个epoch带来提升”；
- “超过重排器”；
- “已经外部泛化”。

## 9. 冻结证据与来源

- 当前模型评估：`data/validation/chemaware_high_coverage_native/run_2340721_resume_2340524/role3_evaluation_after_stop.json`
- 当前triplet审计：`data/validation/chemaware_high_coverage_native/run_2340524/triplets/report.json`
- 当前多空证据：`data/validation/chemaware_high_coverage_native/run_2340524/evidence/report.json`
- 当前最优模型：`data/validation/chemaware_high_coverage_native/run_2340721_resume_2340524/training/best.ckpt`
- 当前结果封存：`docs/CHEMAWARE_HIGH_COVERAGE_NATIVE_ROLE3_SEAL_20260921.md`
- canonical candidate policy：`data/validation/chemaware_truthblind_candidate_policy/run_2338337/policy/report.json`
- multi-null V2：`data/validation/chemaware_multinull_deployment_safe_full_20260919/report.json`
- P2b正式记录：`docs/P2B_RANK_FUSION_FORMAL_RECORD_20260823.md`
- 历史系统批判：`docs/CHEMAWARE_SYSTEMATIC_CRITIQUE_AND_RESEARCH_RESET_20260912.md`
- 历史严格证据审计：`docs/CHEMAWARE_RETAINED_ASSET_STRICT_EVIDENCE_AUDIT_20260906.md`

关键冻结指纹：

```text
current best.ckpt:
09c419dcabf2ff424fe38bb33eaefcd25ccab0b6d125daf03b7959c25fff5838

current role3 evaluation:
01eb4069ad3c7dc36f49e24c0e7e1e7dac20722f228ac2b5eea1e9c01ca6813d

canonical candidate-policy report:
64a997a4fc05f442c04e2746a5ac6e1022272bb21417b237cde279ea66aa47bf

multi-null V2 report:
b437687031b6aef43f2328df10d6b2dcc69c6213b1ce7562c16bc42149830e6f
```
