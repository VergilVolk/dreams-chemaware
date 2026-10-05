# Noise N+P+A4 正式结果前复盘与下一步分支（2026-09-06）

## 1. 复盘边界

当前双 GPU fold-0 作业已经提交，本文不改变该作业的训练代码、数据、初始化或统计门槛，也不提前把运行中的配置判为成功或失败。唯一目标仍是：保留成熟 E8 shared encoder，利用既有 N、P、A4 动作直接改善 clean-spectrum shared embedding，而不是另训 teacher 或推翻 noise 科学逻辑。

需要严格区分三件事：

1. 动作在冻结几何中有改错能力；
2. 动作产生的梯度能够无机械截断地进入优化器；
3. 参数更新能够跨 identity/formula 泛化到未消费 clean query。

当前已经较好解决第 1、2 项的接线问题，但第 3 项尚无正证据。4--5 pp 仍是正式 promotion gate，不是已经得到的结果。

## 2. 当前版本真正保留和修复了什么

- 使用 corrected MassSpecGym candidate graph：83,619 queries、6,220 formulas、6,220,661 spectrum edges；fold 0 有 18,333 held queries。
- 初始化固定为 mature E8 shared encoder，不回到 official，不改变一层解冻、LR、四 epoch 的成熟训练主干。
- N 保留 candidate-gradient 与 role-confounder 的完整多步路径；P 保留 E10B/E11/E12B 全 60 recipe/reference cells；A4 从 outcome-free exact scan 重放 Top-50 proposal × 4 doses。
- 所有动作都在同一 E8、同一 corrected molecule-max candidate geometry 中重评分；旧 graph outcome 和 held formula 不进入训练。
- 同一 query 的多 action 不再拆散或用 action 数偷乘 optimizer dose；ledger 最多保留 16 corrective 与 8 harmful actions，并在 source/family 内均衡。
- corrective 只把 action 相对 clean 与 matched control 的保守 margin 增量写回 clean molecule boundary；action/control 均 detached，不使用 teacher embedding regression。
- harmful action 不作正增强，而进入独立 clean risk/protection 分支；corrective 与 risk 使用 PCGrad veto。
- 旧的 global clipping 断档已经被共同预缩放替代；本地 routed run 的 PCGrad+clip retention p10 为 0.999927，clip-event fraction 为 0。
- 正式 evaluator 同时计算 Recall@1/2/3/5/10/20、MRR、mean/median rank、macro-query AUROC/AUPRC、micro-candidate AUROC/AUPRC、margin、Top1--Top2 gap、corrected/introduced/risk-net、near subset、formula-cluster CI 与 MassSpecGym `[M+H]+` 10-ppm pooled pairwise AUROC/AUPRC。

这些修复证明“配置打开但训练没有吃到动作”的多处旧故障已被排除；它们不证明 held embedding 已经提升。

## 3. 当前证据能说到哪里

### 3.1 形式上的 4 pp 难度

fold 0 有 18,333 queries，official embedding 有 1,225 个 Top-1 errors。绝对 +4.0 pp 等于至少约 734 个净 Top-1 翻转，即使以较弱的 official 为分母，也要消除约 59.9% 的错误；E8 已优于 official，所以相对 E8 所需消除的剩余错误比例只会更高。该目标数学上可能，但绝不是一次普通 continuation 可以自然得到的幅度。

### 3.2 本地联合 ledger 的结构

当前 92-query 开发 ledger 有 848 corrective rows / 69 corrective queries / 56 formulas：

| source | corrective rows | 占 corrective rows |
|---|---:|---:|
| E10B | 475 | 56.0% |
| E11 | 249 | 29.4% |
| E12B | 88 | 10.4% |
| N mature | 28 | 3.3% |
| A4 exact | 8 | 0.9% |

69 个 corrective queries 中，55 个同时有两个及以上来源，但 P 仍提供 95.8% 的 corrective rows。query 内 family equalization 防止 recipe 数直接变成剂量，却不能自动保证 N/A4 在 query 分布或参数梯度上有独立贡献。

### 3.3 本地 encoder 结果

32-query routed run：

- train-corrective：5 corrected / 0 introduced，Recall@1 `+15.625 pp`；
- risk panel：1 corrected / 0 introduced，Recall@1 `+3.125 pp`；
- 10-query inner-held corrective：0 corrected / 0 introduced；
- 32-query outer-held sample：0 corrected / 0 introduced，平均 margin `-0.00115`。

因此目前只证实“动作边界能够写进训练 query”，没有证实“能传到未见 identity/formula”。本地 integrated clean-control 因赶正式服务器进度被停止，当前 N+P+A4 还没有本地 action-vs-control 因果差值；正式作业中的 paired control 将补上这一证据。

## 4. 尚未根本解决的局限，按优先级排序

### P0-A：所谓 99.99% retention 不是端到端更新保留率

当前 retention 只计算 PCGrad projection 与最终 clip 的比例，不包含冻结 calibration scale。开发 run 中：

- corrective scale = `10.4427`；
- common global scale = `0.005483`；
- 相对 raw corrective gradient 的显式乘数约为 `0.0573`；
- risk gradient 的显式乘数约为 `0.00548`。

AdamW 对共同标量近似具有尺度不变性，因此不能把 `0.0573` 直接解释成只剩 5.7% 参数更新；同样，也不能用 99.99% 宣称端到端损失已经完全消失。正式结果后必须补报每个分支的 optimizer-update norm、与未缩放更新的 cosine、Adam epsilon/weight-decay 占比，才能回答真正的信号保留。

### P0-B：corrective 更新在 epoch 前段集中，后面存在长 clean-only 尾巴

训练循环把 shuffled corrective batches 与 protect batches 从 index 0 开始配对；corrective batches 用完后，其余 steps 全是 protect/clean continuation。开发 run 32 corrective vs 64 protect queries，动作只活跃于 50% steps。正式 fold 0 有 65,286 outer-train protect queries，而 corrective query 只能来自少数当前 E8 errors；动作活跃比例预计显著低于 50%，且全部集中在 epoch 前段。

这不会违反“一 query 一次”的剂量合同，却可能产生新的时间断档：刚写入的 corrective boundary 在同一 epoch 后续大量 continuation steps 中被部分覆盖。下一版本应保持总步数、总剂量和每 query 一次不变，只把 corrective steps均匀铺到完整 epoch，并记录 action update 到 epoch 末的存活率。

### P0-C：clean-control 不是同幅值的 action-specific 对照

`clean_control` 与 routed arm 共享初始化、protect stream、step 数和评估，但它把 corrective gradient 直接置零；它没有用 matched-random/shuffled action delta 替换真实 action delta。因此 routed-vs-control 的差值同时包含“增加了一份 corrective gradient”和“这份梯度确实来自正确 action payload”两层作用。

正式 paired CI 可以证明该额外 direct-transfer 分支是否有增量，却不能单独证明 source/recipe/action specificity。若正式结果为正，仍需一个同 query、同 norm、同 schedule 的 source-matched shuffled-delta control；若结果为零，则该对照也是定位 payload 是否被抹平的最短路径。

### P0-D：总 corrective gradient 可能经 candidate reference 而非 query 主体满足

margin-transfer 的 action/control 已 detached，但 live clean margin 同时由 clean query、positive references 和 negative references构成；当前 corrective loss 对这些 clean spectra 均可反传。代码计算了 corrective batch 的 reference preservation 值，却没有在 paired-margin-only corrective loss 中直接加入该 preservation 项。全量 protect stream 提供总体锚定，但不能证明每次 action update 主要落到 query 表征，而不是通过移动少数 reference 来满足 margin。

正式结果后必须做一次不改数据的梯度归因：分别 detach positive/negative references、detach query，比较三部分参数梯度 norm、cosine 和单步 margin effect。只有 reference 分支明显主导时，才运行 query-only corrective 消融；不能先验地再次冻结整个 shared reference encoder。

### P0-E：全量 P/A4 router 是非流式物化，工程失败不能误判为科学失败

P router 对 65,286 outer-train queries 构造 60 target + 60 direction-control views，并在编码前把全部 spectrum tensors、metadata 和 1024-D encoded array保留在内存中。理论规模约为 7.83 million encoded views，仅 float32 encoded matrix 就约 32 GB，尚未包含 tensor/list/DataFrame、candidate contexts 与输出 action bank。A4 虽只覆盖 frozen scan 映射到的约 5,000 queries，也同样先全量物化。

因此当前作业若在 P/A4 route 阶段 OOM、超时或磁盘失败，只能裁决实现不是 action 方法。下一工程版本必须按 query chunks 流式 encode/score/write，最后原子合并 hash；不能减少 recipe、query 或 action 来假装解决运行问题。

### P1-F：rich action payload 仍被压缩为有限 candidate-margin 增量

当前方法优于逐峰 scalar gate，也没有压成单一 1024-D teacher direction；但每个 action 对训练的最终信息仍是对固定 8 个 hard-negative molecules 的 margin delta。动作造成的其他 embedding geometry变化不会进入 corrective target。若正式结果出现“训练 query 改错、held margin 不动”，应比较 margin-transfer 与 action-induced parameter-tangent/edge-gradient transfer，而不是回到 teacher embedding regression。

### P1-G：hard negatives 固定于 E8 初始化

每个 query 只使用初始化时的 top-8 negative molecules、每个 molecule 最多两张 references。这样避免旧的动态 reference switch 断档，但当模型移动时，新 hardest competitor 可能进入 Top-1 而不受 corrective/risk loss约束。若正式结果表现为 margin 上升但 introduced 增多，下一步应使用 `initial top-k ∪ epoch-boundary current top-k` 的冻结并集，禁止逐 step 抖动。

### P1-H：全量动作存在来源覆盖不等于来源贡献已被识别

当前 per-query source/family equalization 解决 recipe multiplicity，不解决不同来源覆盖同一批容易训练错误、或 P 在几乎所有错误 query 上存在 corrective action导致 N/A4 没有独立统计支撑。正式报告必须增加 source leave-one-out 的冻结梯度 replay/影响估计；在证据出现前不能声称 N、P、A4 都对 held 提升作出了贡献。

### P1-I：promotion gate 尚未把“全部指标提升”机械化完整

evaluator 已计算完整面板，但当前最终 gate 只对 Recall@2--20、MRR、macro/micro AUROC/AUPRC 和 `[M+H]+` pairwise AUROC 做 non-regression。它尚未逐项机械检查 mean/median rank、margin、Top1--Top2 gap、near 全面板、pairwise AUPRC以及全 adduct pairwise面板。正式结果可以完整读取这些值，但在 promotion 前必须补齐 gate；不得用一个 `broad_metrics_nonregressive=true` 掩盖未检查字段。

### P1-J：three seeds 仍只有一个 formula fold

三 seed 只能排除随机初始化/采样幸运，不能证明跨 outer folds 泛化。fold 0 三 seed 全过才授权其余 folds；单 fold 即使超过 4 pp，也只能叫 fold-0 result。

## 5. 正式结果出来后的判读，不再反复横跳

| 观察 | 首要解释 | 下一步，保持 E8/N+P+A4 主线 |
|---|---|---|
| P/A4 route OOM、超时或输出过大 | 非流式工程瓶颈 | chunked router；相同全部 actions、相同 hash 合并，不改变科学样本 |
| signal gate 失败 | calibration/PCGrad/clip 仍破坏分支 | source/formula-stratified calibration；报告 optimizer update，而非扫 LR |
| train-corrective 强，outer-held 近零 | 写入成功、共享泛化失败 | 先均匀 interleave corrective steps；再做 query/reference 梯度归因 |
| held margin 上升但 Recall@1 不动 | hard-negative coverage/阈值翻转不足 | initial∪epoch hard-negative union，保持 epoch 级冻结 |
| routed 与 clean-control 几乎相同 | action-specific payload 未产生独立贡献 | 同 norm source-matched shuffled-delta control；不回到 teacher |
| routed 优于 clean-control但 introduced 高 | reference/risk 保护不够或 competitor switch | query-only corrective 消融 + expanded risk negatives |
| Recall@1 达标但其他指标退化 | 局部翻转破坏全局 geometry | 不 promotion；按退化指标定位 near/reference/pairwise 分支 |
| 三 seed 全面达标 | fold-0 真实正结果 | 冻结 checkpoint/hash，扩 outer folds；最后才做独立 P3 下游验证 |

## 6. 最小的下一轮开发顺序

无论本次结果方向如何，先从现有输出做只读诊断，不重训 teacher、不换架构：

1. 读取每 seed 的 route counts、source/formula/identity coverage、calibration、每 step PCGrad/clip、完整 held per-query 表；
2. 增加 optimizer-update norm 与 corrective step position 的离线/单步 replay；
3. 计算 query vs positive-reference vs negative-reference 的梯度归因；
4. 只有发现明确主瓶颈后，做一个保持全部动作和训练预算的最小因果修改：优先 `front-loaded vs uniformly-interleaved`；
5. 若现有 routed arm 为正，再补同 norm shuffled-payload control以确认 action specificity；
6. 若 margin 提升但 Top-1 不翻转，再引入 epoch-level hard-negative union；
7. 不扫 LR、层数、teacher 架构或动作阈值来替代上述归因。

## 7. 当前最诚实的裁决

这次版本是迄今第一条把 corrected graph、mature E8、N/P/A4 多动作、clean-boundary transfer、risk veto 和完整 held evaluator同时接通的正式路线，值得让服务器跑完。它解决了旧 90% clipping 和 action-loss 语义错位，但尚未根本证明训练信号能跨 formula 泛化；本地 held=0 是必须正视的预警，而不是推翻动作科学逻辑的理由。

真正最可能的剩余瓶颈依次是：corrective 更新在 epoch 中前置后被 continuation 稀释、对照没有同幅值 action specificity、corrective gradient 经 reference 绕行、固定 hard-negative覆盖不足，以及 full router 的非流式工程规模。正式结果的价值就在于按第 5 节把这些可能性一次分开，而不是再根据一个总 Recall@1 拍脑袋换路线。

## 8. 第二轮代码追踪：动作库完整，不等于参数更新完整

进一步沿 `routing_ledger -> training_actions -> batch loss -> parameter gradient` 逐行追踪后，需要修正第一轮复盘的一处口径：当前方法不是把每类动作都“直接微调进 encoder”，而是只让 corrective action 的有限 candidate-margin 信息决定 clean-boundary gradient；其余动作主要停留在路由或 query membership 层。

### 8.1 `paired_margin_only` 中 action tensor 不产生参数梯度

正式入口固定 `--corrective-objective-mode paired_margin_only`。在该模式中：

- `lambda_action_rank=0`；
- `lambda_counterfactual=0`；
- `lambda_action_safety=0`；
- action/control margins 在构造 target 前全部 `detach`；
- 唯一 corrective loss 是 clean margin 到 action-conditioned scalar margin target 的 Smooth-L1。

因此 action spectrum 虽然经过 encoder forward，却没有 action-view parameter gradient。其峰级干预模式经过如下信息漏斗：

`101-token action -> 1024-D action embedding -> 8 fixed molecule margins -> min(action-clean, action-control) -> ReLU -> cap -> family/query normalization -> clean-boundary gradient`

这仍比单一 query gate 更丰富，也确实不是 teacher distillation；但动作改变了哪个峰、在 1024-D 空间形成什么方向，最终只通过最多 8 个 margin 标量影响更新。严格说，它是 action-conditioned clean hard-negative fine-tuning，而不是 raw action view 本身参与的完整直接微调。

### 8.2 当前 safety 路径没有消费 harmful action 的具体内容

ledger 会选择最多 8 条 harmful actions/query，并把它们的 action/control tensors写入 `action_spectra.npz`。trainer 随后只从 harmful rows 提取 `risk_queries`，`ProtectExample` 只包含 clean query、positive references与negative references；harmful action tensor、source、family、peak和dose均不进入 `protective_batch_loss`。

正式配置中 `maximum_clean_queries=0`，所以 `clean_queries=all 65,286 outer-train queries`，而 `protect_queries = clean_queries union risk_queries`。由于 risk queries 本来就是 outer-train 的子集，harmful routing 在正式作业中甚至不会增加一个 protect query。结果是：harmful动作只影响ledger统计，不改变正式risk batch内容或权重。

这不代表应该把 harmful action 当正增强。正确修复方向是让 harmful action 形成 action-specific veto/constraint，例如用其梯度或边界退化方向投影 corrective update，而不是最小化 harmful action 的 identity loss。

### 8.3 robustness-only actions 完全不进入 training actions

当前 ledger 只选择 `selected_corrective | selected_risk`，robustness-only 与 uncertain 留在 audit ledger，不进入训练 NPZ。开发路由中共有 871 条 robustness-only actions，覆盖 24 queries/23 formulas；在16个 official-correct P queries上，960条动作中574条为 robustness-only。这类动作不能提供纠错标签，但可能是跨条件不变性和动作机制泛化的重要安全样本。

下一版若需要扩大 action 机制覆盖，应为 robustness-only 建立独立、低剂量、同 identity 的 invariance branch；它不能获得 corrective margin reward，也不能与 harmful 混合。

### 8.4 当前真正的“约 90%断档”更可能是 optimizer duty cycle

gradient calibration只把一个 paired corrective step 与一个 protect step 的梯度 norm配平，没有校正整个 epoch 中两类 optimizer updates 的次数。训练循环中每个 corrective query每 epoch一次、每个 outer-train protect query也每epoch一次；corrective batches耗尽后继续执行大量risk-only optimizer steps。

fold-0 outer-train共有65,286 queries，official有4,732个errors；mature E8预计错误更少但正式值待输出。开发P路由在60个错误中有58个存在corrective action。若正式corrective query数量约为4--5千，则每epoch约只有6%--8%的optimizer steps含action transfer，后续92%--94%为protect-only updates。当前 `action_retention_p10` 只在含action的steps上统计，因此即使显示99.99%，也完全看不到这一epoch级duty-cycle稀释。

必须新增三个量：

1. `action_active_optimizer_step_fraction`；
2. 全epoch累计corrective/risk update norm比例；
3. 最后一个corrective step之后，action-induced margin improvement到epoch末的survival ratio。

### 8.5 margin cap进一步压平了强动作差异

开发selected corrective actions的保守gain中位数为0.0566，32.1%的actions达到或超过0.10 cap。乘以0.5 transfer fraction后，至少这32.1%的动作都被压成相同的0.05目标增量。按source看，中位gain约为：N `0.0267`、A4 `0.0253`、E10B `0.0552`、E11 `0.0792`、E12B `0.0573`。

cap是必要的安全保护，不能因强动作多就直接放大；但当前报告没有记录per-edge非零率、cap命中率或归一化后的有效action数，所以无法判断“848条corrective actions”中真正提供不同训练信息的有多少。

## 9. 不回到蒸馏的后继直接微调结构

若当前正式结果未达到门槛，优先后继不是新teacher，而是把现有动作语义完整送入同一个shared encoder。

### 9.1 Query-complete、epoch-balanced scheduler

- 所有corrective queries和全部65,286 protect queries仍每epoch曝光一次；
- 不循环或复制corrective action；
- 把多个protect microbatches做gradient accumulation/平均，配到均匀分布于全epoch的corrective optimizer steps；
- optimizer step数不再由protect query数单独决定，消除长risk-only尾巴；
- calibration从per-step norm改为per-epoch integrated update budget。

这项修改不改变动作、模型、LR、层数、总query覆盖或总数据剂量，只修复时序和累计优化剂量。

### 9.2 三类动作各用其正确语义

1. `corrective`：保留当前clean-boundary margin-transfer；再加入极低剂量、source-normalized的真实action-view identity gradient，使峰级action payload能够产生参数方向。成功action loss较小时按冻结branch gradient norm校准，不能再由“动作容易”自动归零。
2. `robustness-only`：只提供有界invariance/identity-consistency，不获得corrective target。
3. `harmful`：只形成action-specific gradient veto、clean/reference floor或反事实约束，绝不作为positive imitation。

三类分支必须分别记录raw norm、source/formula composition、PCGrad cosine、clip与optimizer update；不能再次先相加再只报总梯度。

### 9.3 最小因果对照

在相同E8、ledger、queries、references、epochs和optimizer budget下比较：

- current scalar-margin transfer；
- current transfer但uniform interleave；
- uniform interleave + same-norm真实action-view gradient；
- source-matched shuffled action/control gradient。

第3臂只有严格优于第2和第4臂，才能说明新增收益来自真实峰action payload，而不是普通额外排序剂量。该设计仍是直接shared-encoder fine-tuning，不生成teacher embedding target。

## 10. 结果出来后新增的可证伪签名

| 假说 | 必须看到的签名 | 若看不到 |
|---|---|---|
| epoch尾部覆盖action更新 | action block后margin改善、epoch末明显回落 | 不优先改scheduler |
| action payload被scalar化丢失 | real与shuffled scalar臂接近，真实action-gradient臂胜出 | 不引入action-view branch |
| reference绕行 | reference-side gradient/update显著大于query-side，train强held弱 | 保持shared reference，不做query-only冻结 |
| fixed negatives过时 | introduced queries的最终top competitor多数不在initial top-8 | 不做negative refresh |
| P覆盖掩盖N/A4 | source leave-one-out显示P-only近似全部效果 | 不用配额强行抬高N/A4 |
| cap压平有用强度 | cap-hit actions在uncapped离线单步中有稳定额外clean influence | 保持0.10 cap，不扫剂量 |

第二轮结论因此比“梯度已经保留99.99%”更严格：当前版本解决了action-step内部的裁剪，却仍可能在action-to-gradient信息压缩、harmful/robustness未消费和epoch累计剂量三个层面损失绝大多数可迁移信息。这些问题可以在不推翻E8、不丢弃任何动作、不引入teacher的条件下逐项修复。
