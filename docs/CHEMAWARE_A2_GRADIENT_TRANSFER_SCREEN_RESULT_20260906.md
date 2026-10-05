# ChemAware A2 动作迁移梯度筛选结果与特异性重审

**记录日期**：2026-09-06  
**实验性质**：24-action、formula-stratified、零 optimizer step 的一阶机制筛选  
**正式状态**：`CHEMAWARE_ACTION_TRANSFER_GRADIENT_FAIL`  
**性能主张**：本实验不评估训练后检索性能，`performance_pp_claim=null`；这不是
“0 pp”，也不撤销此前已经分级保留的 E1、rule-mass 或 direct proof-of-effect。

## 1. 完整运行事实

- 8 个 action-transfer 核心合同和 4 个审计合同全部通过；preflight 通过。
- 冻结动作设置严格重放为 `conflict_attenuate / strength=0.75 / top_k=3`，
  动作种子为 `20260935`。
- 117 个 eligible 动作中按 formula 分层抽取 24 个动作，覆盖 24 个 formula；
  本轮没有审计全部 117 个动作。
- official checkpoint SHA-256 为
  `8928f908606c0bd652c5a4107d3c35102f660622958c225a1f625abe4b1ba245`；
  action-bank SHA-256 为
  `ae462bb1edcfe23147f6858c713121504e3f46073b91848afd491f6d14101241`。
- clean embedding 最小重放 cosine 为 `0.9999998808`；动作 margin 最大绝对重放
  误差为 `3.5763e-7`，远低于 `5e-4` 容差。因此当前 FAIL 不是 checkpoint、
  action seed、候选图或 action 重构漂移造成的。
- 模型共 117,101,029 个参数；最后一个 transformer block 与 projection head 共
  13,638,656 个参数被置为可求梯度。`optimizer_steps=0`、`weights_updated=false`，
  没有生成或修改微调 checkpoint。

## 2. 冻结结果

主指标是等学习率缩放更新范数下，对未修改 clean query 的
`positive - hardest-negative` margin 一阶增益：

```text
u_route = layerwise_LR * g_route
I = - <grad_theta m_clean, u_route> / ||u_route||
```

| 梯度路由 | correct | correct − candidate-swapped | correct − peak-permuted | 裁决 |
|---|---:|---:|---:|---|
| action query only | 1.3944 `[1.1957, 1.5745]` | +0.0843 `[-0.0558, 0.2194]` | −0.0683 `[-0.1731, 0.0002]` | FAIL |
| action forward / clean backward | 1.4415 `[1.2492, 1.6172]` | +0.0453 `[-0.0765, 0.1591]` | −0.0503 `[-0.1467, 0.0111]` | FAIL |
| action-routed clean pair | 1.4501 `[1.2362, 1.6400]` | +0.0250 `[-0.1330, 0.1512]` | −0.0678 `[-0.2045, 0.0213]` | FAIL |
| naive action − clean | −0.6443 `[-0.7938, -0.5052]` | −0.5867 `[-0.8999, -0.3101]` | −0.5472 `[-0.7904, -0.3421]` | 有害反例 |

三个非 naive 路由的 correct 绝对区间均严格为正，且 24/24 动作的一阶方向均
有利于自身 clean margin。这证明动作视图产生的梯度可以到达可部署 shared
encoder，并能形成普通的同身份排序更新；它不是“完全没有迁移”。但是：

- correct 相对 candidate-swapped 的优势只有 correct 绝对量的约 1.7%--6.0%，
  且区间均跨 0；
- peak-permuted 的均值反而比 correct 高约 3.5%--4.9%；
- 三条不同路由得到几乎相同的结果。

因此本轮只否决 **E2：当前三种注入路由具有 correct-action-specific clean
gradient**。它不否决 **E1：冻结正确动作相对匹配动作对照能改善候选边界**，
也不把已有 shared-embedding 开发结果改写成零。

### 2.1 已完成的逐 action reference-pair 分解

后续 Slurm 作业已在同一冻结 `run_2331965/screen24` 上完成 CPU 后处理，状态为
`CHEMAWARE_ACTION_TRANSFER_SPECIFICITY_SUMMARY_COMPLETE`。24 个动作的 pair route
选择重合如下：

| 对照 | 正参考相同 | 负参考相同 | 正负 pair 完全相同 | 至少一端改变 |
|---|---:|---:|---:|---:|
| candidate-swapped | 18/24（75.0%） | 20/24（83.3%） | 16/24（66.7%） | 8/24（33.3%） |
| peak-permuted | 22/24（91.7%） | 20/24（83.3%） | 19/24（79.2%） | 5/24（20.8%） |

更细分地，candidate-swapped 中只换正参考、只换负参考、两端都换分别为
4/24、2/24、2/24；peak-permuted 中分别为 1/24、3/24、1/24。

这直接证实了 pair route 的主要不可识别机制：对 candidate-swapped 的 16/24 和
对 peak-permuted 的 19/24 比较，correct/control 传给 clean loss 的 `(c+, c-)`
完全相同。此时两臂梯度只差一个正标量，单位更新范数指标会把该标量消掉，因而
不可能携带 correct-action specificity。剩余真正发生 reference switch 的样本仅
8 个和 5 个；在看到其条件 influence 前，既不能声称这些 switch 有益，也不能用
全体均值把它们否定。

这项结果把失败定位到“动作信息在 reference selection 时大面积坍缩”，而不是
模型没有梯度、动作 E1 无效或训练轮数不足。扩大 epoch、学习率、PCGrad/cap 或
直接审计全部 117 个梯度都不会修复该结构性丢失。

## 3. 为什么当前梯度不特异

### 3.1 listwise identity CE 的公共分量压倒动作差异

正例位于候选分子列表首位。固定每个分子的 max-reference 后，listwise CE 对
query embedding 的梯度为：

```text
dL/dq = (sum_j p_j c_j* - c_true*) / T
```

correct、candidate-swapped 和 peak-permuted 三臂共享同一个 identity 标签、同一
候选列表、同一算子、同一剂量，只改变三个 observed peak 的 intensity。只要
动作没有改变 active positive reference、hard negative 或大幅改变 softmax 权重，
三臂的梯度就由同一个 `c_true*` 与同一批负候选主导。模型学到的是“把该 query
推向真实身份”，而不是“识别这三个 peak 为什么具有正确结构证据”。

### 3.2 straight-through 只换了反向路径，没有增加化学监督量

`action_forward_clean_backward` 使用 action embedding 计算分数，但把梯度送入
clean-query Jacobian。其参数梯度近似为：

```text
g = J_clean^T (sum_j p_j(action) c_j* - c_true*) / T
```

它能保证更新到 clean encoder，却不能凭空产生 correct 与 pseudo-action 的差别。
当三臂的候选概率与 active reference 相近时，straight-through 只能搬运公共的
identity gradient。这解释了它与 `action_query_only` 的结果几乎相同。

### 3.3 pair 路由只有“换边”时才携带动作信息

`action_routed_clean_pair` 由动作视图选择正参考 `c+` 与负参考 `c-`，再在 clean
query 上优化：

```text
L_pair = softplus((tau - q_clean dot (c+ - c-)) / T)
```

如果 correct 与 control 选择同一 `(c+, c-)`，归一化后的方向就是同一个
`J_clean^T(c- - c+)`；动作只改变一个会被单位范数归一化消掉的标量。此时该路由
在数学上不可能表现出 action specificity。现有汇总没有报告三臂 reference-pair
重合率，必须先从已有 `per_action.csv` 计算，不能再烧 GPU 猜测。

### 3.4 correct action 变成容易样本，梯度反而可能更小

E1 的作用正是让动作后的正确候选更容易胜出。对交叉熵而言，越容易的样本梯度
越小。因此“冻结动作更有效”与“动作 CE 提供更强的微调梯度”并不单调对应。
这也解释了为什么 `g_action - g_clean` 在 24/24 动作上方向有害：它主要从一个
更弱的 easy-example 梯度中减去更强的 clean error 梯度。

### 3.5 当前动作是候选条件信息，shared embedding 存在可观测性门

该 ICEBERG 动作由 true structure 与 hardest same-formula negative 的预测差异
定义。它天然回答“相对于这个具体错误候选，哪些 observed peaks 更冲突”，而
单张 clean spectrum 的部署 embedding 不接收候选身份。训练可以使用这种特权
信息，但只有当不同 formula/identity 上存在可由 clean spectrum 学到的稳定规律
时，它才能沉淀成 candidate-independent embedding。117 个稀疏动作对 13.6M
可训练参数只提供了局部边界事件；当前结果尚未证明这些事件共享可学习的
clean-visible 规律。

## 4. 撤回“对照中心化残差梯度”方案

不得使用：

```text
g_chem = g_correct - mean(g_candidate-swapped, g_peak-permuted)
```

原因不是实现困难，而是目标本身不合格：

1. 梯度下降使用该量，等价于最小化 correct loss 的同时最大化对照 loss；两个
   对照仍是同一真实 identity 的 label-preserving 谱图，主动把它们训练坏是
   反学习。
2. 对照的职责是反证与归因，不是负监督。已有 Noise 合同要求 matched control
   不反传，harmful/uncertain 的 corrective weight 精确为零，而不是取负权重。
3. 三个高维梯度十分接近时，相减会放大很小的估计噪声和 control-generator
   artifact；117 个动作不足以稳定估计 13.6M 维的自由残差方向。
4. 用同一批对照既构造更新又裁决“correct 是否胜过对照”会形成循环定义；
   即使数值通过，也不能作为独立特异性证据。

因此，对照差只能用于 **paired audit statistic**，不能直接送入 optimizer。

## 5. 特异性必须分成三个门

| 层级 | 要回答的问题 | 当前状态 |
|---|---|---|
| E1 source specificity | 正确结构动作在冻结模型上是否优于同剂量伪动作 | 已通过；confirmation `+0.7519 pp`、3 corrected / 2 introduced，安全性仍敏感 |
| E2 transfer specificity | 只使用正确动作产生的非负训练信号，是否比等预算对照更改善 clean boundary | 当前三条路由未通过 |
| E3 deployment observability | 动作所携带的差异能否由推理时单张 clean spectrum 稳定恢复 | 未验证；这是 shared embedding 与 candidate-conditioned 方法的分流门 |

E1 通过不能替代 E2；E2 的绝对正 influence 也不能替代 correct-control 增量；
E2 若通过但 E3 不通过，正确形态应是 candidate-conditioned scoring/hybrid，而不应
继续把不可观测信息硬塞进 spectrum-only embedding。

## 6. 下一步只分析现有产物，不训练

从服务器现有 `screen24/per_action.csv` 完成以下六项后，才允许设计新 loss：

1. 同时比较 `predicted_clean_margin_gain`、单位范数 gain、raw gradient norm、
   LR-scaled update norm；检查“correct 更容易所以梯度更小”是否主导结果。
2. 计算 pair route 中 correct/control 的 `(positive_reference,
   negative_reference)` 完全重合率、只换正参考率、只换负参考率；若几乎不换边，
   直接淘汰 pair route。
3. 计算 E1 forward specificity
   `m_correct - m_control` 与 E2 influence specificity 的逐 action、formula-macro
   Spearman 关系；若无正关系，E1 动作效果没有转成训练影响。
4. 分开报告 `corrective_rank` 与 `corrective_margin`。后者只改善连续 margin，
   不应与真正纠错动作同权解释。
5. 分 discovery folds 0--1 与 action-family confirmation fold 2 报告；任何新的
   operator/gate 只能在 discovery 内确定，再冻结到未用于该选择的 formula fold。
6. 检查 correct、swapped、permuted 的 active molecule/reference 是否相同，以及
   influence 差异是否仅由梯度范数而非方向产生。

这些都是 CPU 汇总，不需要重新加载 117M 模型。当前缺少本机
`screen24/per_action.csv`，所以不能诚实地把上述逐 action 机制写成已经验证。
仓库已加入只读汇总器及唯一 Slurm 入口；作业会自动解析最新的完整
`run_*/screen24`，校验 action bank 哈希，并把结果原子写入同目录的
`specificity_report.json`：

```bash
sbatch tasks/run_chemaware_action_transfer_specificity_summary.sbatch
```

这是申请恰好 1 张 GPU 的 Slurm 作业；不手动指定任何内存参数，不加载模型、
不求新梯度、不更新权重，也不需要手工填写 job id。只有目录中同时存在 `report.json`、
`per_action.csv` 和 `COMPLETE.json` 才会被解析；已有
`specificity_report.json` 时 fail-closed，不覆盖既有结果。

## 7. 符合现有教训的后续路线边界

在上述 CPU 审计完成前不指定新 winner。可保留的设计原则只有：

- optimizer 只接收 correct branch 的非负信号；harmful、uncertain 和未支持动作
  的 corrective weight 必须精确为零；matched controls 只用于选择与最终反证，
  不反传、不给负权重；
- action 必须改变一个真实训练事件，例如经过 cross-fit 冻结的 query budget、
  clean candidate boundary 或低维 peak/token target；仅把 action view 当第二个
  同身份正样本已经被本轮结果否决；
- 不再做参数空间的自由投影、残差或 coordinate mask。若 reference-switch 与
  influence 关系成立，可检验 `boundary-budget`：在 discovery formula 内由冻结
  correct-control advantage 得到非负 `w_i`，只优化 correct action 指定的 clean
  boundary；control 只生成审计/匹配臂，不反传。该方法还必须超过同 query 数量的
  error-only 与 weight-permuted 选择对照，否则只能解释成普通 hard-example mining；
- 若要让动作语义真正进入 encoder，可检验 `peak-action observability`：从全部已
  审计 action（不仅是 117 个正动作）构造 observed-peak 的
  `attenuate / keep / unobservable` 三态标签，用 clean peak tokens 预测；只以 correct
  action 标签训练低容量 auxiliary head，controls 只做 formula-held 反证。只有该
  head 在 identity/formula 隔离 OOF 中显著胜过 peak-permuted、candidate-swapped 和
  label-permuted null，才允许把它与 clean retrieval 联训，推理时丢弃 auxiliary
  head；
- 必须增加 clean-spectrum observability gate：用未修改谱图在 formula/identity
  隔离 OOF 中预测 action 是否有益、动作峰集合或动作边界类别，并胜过标签置乱、
  matched-peak 与 candidate-swapped null。通过后才有理由继续 shared embedding；
- 既有 MAGMa/MIST 类 peak-token probe 相对置乱对照只出现约 0.008 的 AUPRC 增量，
  因而不能预设 `peak-action observability` 会通过；它是信息可辨识性门，不是下一
  个待烧算力的模型承诺；
- 若 correct action 很少改变 reference/candidate boundary，或 clean-spectrum
  observability 失败，就停止强迫 candidate-independent 注入，保留 E1 作为候选
  条件证据，进入 reranker/hybrid 的综合研判。该分流不是现在预先转向。

## 8. 当前裁决

1. 本轮运行完整、可信、成本有边界；没有运行时失败，也没有更新权重。
2. 三个正向路由证明普通 action-to-clean 梯度可达，但没有证明化学特异梯度。
3. `naive_action_minus_clean` 已被严格否决；对照中心化残差同样不得进入训练。
4. 现在最缺的不是另一种 optimizer 技巧，而是确认动作是否改变训练边界，以及
   该差异是否可由 clean spectrum 学到。
5. 在逐 action CPU 证据补齐前，禁止提交 117-action full confirmation、bounded
   embedding pilot 或任何全局微调。
