# Noise 共享编码器、P2b 与近期重排器正向成果总账

**冻结日期**：2026-10-01  
**范围**：只记录以下三条线：

1. 最近的 Noise 微调共享 DreaMS encoder；
2. P2b 局部候选重排器；
3. 最近的三通道 RRF 光谱重排器。

本文不纳入 BioAware、ChemAware、生物学应用，也不把早期 E4/E8 工程试验、失败课程或未运行方案包装成正向成果。所有“提升”必须同时注明数据集、基线、是否改变 encoder、置信区间和当前资格。

---

## 1. 一页结论

| 方法 | 是否更新共享 encoder | 评价协议 | 最重要的正向结果 | 当前资格 |
|---|---:|---|---|---|
| Noise native Stage-1 hard-positive | 是 | corrected MassSpecGym held fold-0，18,333 queries | 对 official Recall@1 `+0.49637 pp`；targeted 对 matched control `+1.26548 pp`，公式簇 CI `[+0.87586,+1.65425] pp` | 已成立的 Noise 动作因果信号；当前最干净的 action-specific encoder 证据 |
| Noise native Stage-2 residual | 是 | 同一 held 图 | 对 official 约 `+0.6000 pp`，比 Stage-1 约 `+0.1036 pp` | 正向点估计，但增量未达到升格标准；不覆盖 Stage-1 |
| **Noise relation-complete T1/T3，seed 3407（V1）** | **是** | MassSpecGym held + GNPS Gold/Silver identity/formula-disjoint | MassSpecGym 对 official `+1.0527 pp`；GNPS identity `+1.20055 pp`、formula `+1.25451 pp`，两者校正 CI 均严格为正 | **当前最高且已外部泛化确认的 Noise encoder**；仍不是 5 pp 成果 |
| Noise relation-complete T1/T3，seed 3408 | 是 | 同一协议复证 | 对 Stage-1 Recall@1 `+0.37637 pp`；MRR、AUPRC、margin 方向为正 | 方向复现；Recall@1 CI 下界 `-0.02868 pp`，未独立显著 |
| Noise relation-complete T1/T3 v2 exact-action rotation | 是 | train-side fold-1 diagnostic，16,742 queries | exact row-level bridge 全部恢复，但三臂 Recall@1 均下降 | **NO-GO；held 未消费，不构成性能结果** |
| P2b frozen local rank fusion | 否 | formula-isolated nested OOF | Recall@1 `+3.91 pp`；near Recall@1 `+5.83 pp` | 开发证据强，但不是封存外测 |
| P2b frozen local rank fusion | 否 | sealed P3 main，3,000 queries | Recall@1 `+1.07 pp`，CI `[+0.24,+1.89] pp` | 封存主面板正向重排结果；near-core 有害，不能通用部署 |
| RRF(dreams, cosine, top-k overlap) | 否 | official geometry held 探索 | Recall@1 `94.954%`，corrected/introduced `425/101`，risk-net `+223` | 信息存在性与互补性探针 |
| 同一 RRF 叠加 Stage-1 并做 GNPS 独立确认 | 否 | frozen confirmation run 2347032 | 无可保留增益 | **确认失败，永久关闭；不得作为部署或性能成果** |

最严格的结论是：

- 当前最好的共享 encoder 是 T1/T3 主种子 V1：MassSpecGym `94.2399%`，相对 official `93.1875%` 累计约 `+1.0527 pp`；在两个独立 GNPS 面板上又分别得到 `+1.20055 pp` 和 `+1.25451 pp`，说明约 1--1.3 pp 的提升能够外部迁移，但仍不是 `+5 pp`。
- 最可靠的 Noise action-content 因果证据仍是 Stage-1 targeted 相对 matched control 的 `+1.26548 pp`。
- P2b 的可引用封存增益是 P3-main 的 `+1.07 pp`，不是开发集的 `+3.91 pp`，更不是 encoder 增益。
- RRF 的 `94.954%` 是 official 几何上的探索性重排结果；Stage-1/GNPS 确认失败后，不能进入最终系统。

---

## 2. 共同评价协议与数字口径

### 2.1 corrected MassSpecGym held graph

最近 encoder 结果使用同一个冻结候选图：

- held queries：`18,333`；
- near subset：`6,781` queries；
- 评价单位：query 对候选分子；
- 同一候选分子的多张参考谱按 molecule-max 聚合；
- 主指标：Recall@1/2/3/5/10/20、MRR、mean/median rank、macro-query AUROC/AUPRC、micro-candidate AUROC/AUPRC、正例对最强负例 margin、Top1–Top2 gap、corrected/introduced/risk-net、near-subset 和 formula-cluster paired CI；
- official DreaMS Recall@1：约 `93.1875%`，即 `1,249/18,333` 个 Top-1 错误；
- Stage-1 Recall@1：约 `93.6835%`，即 `1,158/18,333` 个 Top-1 错误。

`risk_net_lambda2 = corrected - 2 * introduced`。它比净修正更严格，但不是 Recall@1 本身；应与 Recall、CI 和 margin 一起报告，不能单独改判。

### 2.2 AUC 的正确名称

本文的 pooled AUC 是：

> MassSpecGym strict-10-ppm pooled pairwise AUROC/AUPRC

它不是 DreaMS 论文 NIST20 图 4b 的精确复现。即使数值接近 `0.85`，数据集、pair ledger 和去重协议不同，不能称为“论文 0.85 复现”。

### 2.3 GNPS Gold/Silver 独立外部确认（run 2347472）

该评估使用冻结的 GNPS Gold/Silver strict-10-ppm benchmark；GNPS 谱从未进入 Noise 训练或 checkpoint 选择。两个面板分别为：

- identity-disjoint：`10,995` queries，`87,518` candidate molecules，`175,171` directed spectrum pairs；
- formula-disjoint：`5,261` queries，`24,401` candidate molecules，`47,724` directed spectrum pairs。

V1 相对 official DreaMS 的完整关键结果：

| 指标 | identity-disjoint | formula-disjoint |
|---|---:|---:|
| official Recall@1 | `85.35698%` | `86.80859%` |
| V1 Recall@1 | **`86.55753%`** | **`88.06311%`** |
| Recall@1 增量 | **`+1.20055 pp`** | **`+1.25451 pp`** |
| 24-family 校正公式簇 CI | **`[+0.46048,+1.89026] pp`** | **`[+0.22931,+2.29860] pp`** |
| V1 MRR | `0.912741` | `0.928331` |
| V1 macro-query AUROC | `0.941290` | `0.938052` |
| official pooled strict-10-ppm AUROC | `0.928697` | `0.903277` |
| V1 pooled strict-10-ppm AUROC | **`0.941279`** | **`0.929902`** |
| pooled AUROC 增量 | **`+1.25819 pp`** | **`+2.66248 pp`** |
| corrected / introduced | `328 / 196` | `156 / 90` |
| risk-net lambda=2 | `-64` | `-24` |
| near Recall@1 增量 | **`+1.50158 pp`** | **`+1.97011 pp`** |

分阶段的 GNPS Recall@1 贡献为：

- official -> Stage-1：identity `+0.70941 pp`，formula `+0.98841 pp`；
- Stage-1 -> V1：identity `+0.49113 pp`，formula `+0.26611 pp`；
- official -> V1：identity `+1.20055 pp`，formula `+1.25451 pp`。

因此可以严格写成：V1 是当前 Noise 线第一个在 MassSpecGym 内部 held graph 和两个 GNPS 外部 disjoint 面板上均获得正向 Recall@1 的共享 encoder。不能写成 NIST20 `0.85` 的精确复现，也不能把负的风险网隐藏起来。

---

## 3. 共享 encoder 路线

### 3.1 Stage-1：原生 DreaMS hard-positive Noise 微调

#### 3.1.1 算法

Stage-1 不再自创 teacher、gate、蒸馏或残差注入器，而是保留 DreaMS 原生训练内核，只改变 triplet 的内容。

动作来自冻结的七类有效 Noise 来源：`A4_exact`、`E10B`、`E11`、`E12B`、`N_mature`、`P_guided_original` 和 `V4_gradient_path`。不同来源保留各自 action payload；matched control 只替换动作内容，不改变 query、边界关系、训练剂量或优化条件。

对每个可由原生输入格式表示的有效 Noise 动作：

```text
anchor   = 同一 query 的 registered targeted action view
positive = 同一真实分子的独立实测谱
negative = 该动作指定的 exact false candidate 实测谱
```

对无法原生表示的动作，使用实测 fallback：

```text
anchor   = clean measured query
positive = exact measured true spectrum
negative = exact false candidate measured spectrum
```

关键科学修正是：动作谱只作为 hard positive anchor；已经被实验证明有害的 `clean query -> action view -> false candidate` bridge 不进入训练。

#### 3.1.2 原生 DreaMS 内核

- `ContrastiveSpectraDataset` 的 triplet 语义；
- 原生 `ContrastiveHead` 与官方 1024 维线性 head；
- `SpectrumPreprocessor`：100 peaks，precursor intensity `1.1`；
- cosine triplet-margin loss，margin `0.1`；
- Adam，learning rate `5e-6`，weight decay `0`；
- batch size `4`，FP32；
- backbone 从 epoch 0 解冻；
- 从官方 fine-tuned embedding 初始化；
- 一次完整 pass；同一 query 不在同一个 optimizer batch 重复；
- targeted 与 registered same-query control 使用相同关系、顺序、步数、seed 和优化器配置。

#### 3.1.3 正向结果

在 corrected held fold-0：

| 对比 | Recall@1 变化 | corrected / introduced | 净修正 | 解释 |
|---|---:|---:|---:|---|
| targeted vs official | **`+0.49637 pp`** | `309 / 218` | `+91` | 绝对共享 encoder 提升 |
| targeted vs mature E8 | **`+0.22910 pp`** | — | — | 相对此前成熟 Noise encoder 的增量 |
| targeted vs matched control | **`+1.26548 pp`** | `455 / 223` | `+232` | action content 的 matched causal contrast |

targeted 对 matched control 的公式簇 paired CI：

```text
[+0.87586, +1.65425] pp
```

这是本项目目前最强、最清楚的 Noise-specific 因果证据：在训练关系、剂量和优化条件匹配时，targeted action content 比 same-query control 更有利。

#### 3.1.4 必须同时报告的限制

- targeted 对 official 的 `risk_net_lambda2 = 309 - 2*218 = -127`；
- `+1.26548 pp` 是 targeted-minus-control，不是 targeted-minus-official；
- 对 official 的绝对提升是 `+0.49637 pp`；
- Stage-1 当时没有保存可用于完整 paired 比较的 pooled-AUC 数值账本，只保留了部分方向门；因此本文不虚构 Stage-1 AUROC 增量；
- 这证明动作内容有效，但没有达到 `+2 pp` 或 `+5 pp` 的绝对 encoder 目标。

### 3.2 Stage-2 residual：只续训仍然有效的动作边界

Stage-2 从 Stage-1 targeted checkpoint 继续：

- 重新编码 outer-train candidate graph；
- 只选当前几何中仍然 hinge-active、targeted margin 优于 matched control、且优于 clean 同边界的动作；
- 每个 query 最多一个 residual action；
- 与 clean preservation 关系一起训练；
- late continuation learning rate `1e-6`。

正向点估计：

- 对 official Recall@1 约 `+0.6000 pp`；
- 对 mature E8 约 `+0.3327 pp`；
- 对 Stage-1 仅约 `+0.1036 pp`，约 19 个净 query。

Stage-2 的 Stage-1 增量没有形成足够强的 paired-CI/风险证据，因此不替换 Stage-1。它的价值是证明 residual continuation 还能收获很小的正向余量，而不是建立新的冠军。

### 3.3 最新结果：relation-complete T1/T3 shared-encoder continuation

#### 3.3.1 为什么改为 T1/T3

Stage-1 的单 triplet 每个 query 只暴露一个局部边界。T1/T3 的目标是让训练关系更接近部署时的 molecule-level competition：

- **T1**：一个 query 内，所有被选中的正参考谱与所有被选中的负参考谱形成 multi-relation hinge；
- **T3**：先对每个候选分子的多张谱做 exact molecule-max，再在完整候选分子集合上做 listwise softmax；
- 同一 query 内先平均，再对 query 平均，避免参考谱多的分子或动作多的 query 自动获得更大剂量。

对单个 anchor `a`、正谱集合 `P`、负分子集合 `N_m`：

```text
T1 = mean_m mean_(p in P, n in N_m) relu(0.1 + sim(a,n) - sim(a,p))
T3 = -log softmax_m(max_(s in molecule m) sim(a,s) / 0.07)[true molecule]
L  = 0.5 * T1 + 0.5 * T3
```

所有 clean/action anchors 和所有 measured references 都由同一个共享 encoder 实时编码并接收梯度。

#### 3.3.2 训练语料

outer held fold-0 不参与训练。训练语料统计：

| 项目 | 数量 |
|---|---:|
| train queries | `48,544` |
| 含 same-formula rival 的 queries | `37,948`（`78.1724%`） |
| complete-candidate queries | `37,731`（`77.7254%`） |
| attached targeted actions | `19,894` |
| 有 action 的 queries | `2,730` |
| candidate molecules | `186,425` |
| measured reference rows | `516,753` |
| positive references，中位数 | `4` |
| negative molecules，中位数 | `2` |

每个 query 最多 4 个正参考、4 个负分子，每个负分子最多 3 张参考谱；额外加入最多 2 个 boundary molecules。所有 19,894 个可表示 action 都进入语料，没有 query 触发 40-action cap。

#### 3.3.3 训练内核

- warm start：Stage-1 targeted champion；
- 恢复 Stage-1 原生 Adam moments；
- 完整 DreaMS `ContrastiveHead` 和 backbone 全部解冻；
- Adam `5e-6`，weight decay `0`；
- FP32，100 peaks，precursor intensity `1.1`；
- 每个 eligible query 恰好一次；最多 4 queries/step、64 unique spectra/step；
- 有 action 的 query：clean anchor 权重 `0.5`，其全部 actions 共同平分 `0.5`；无 action 时 clean 权重 `1.0`；
- 一个 epoch；
- 推理时只有共享 encoder，不需要 reranker、teacher、候选特征或 action view。

#### 3.3.4 主种子 3407：完整 held 指标

绝对指标：

| 指标 | T1/T3 seed 3407 |
|---|---:|
| Recall@1 | **`0.942398953`** |
| Recall@2 / @3 / @5 | `0.983581520 / 0.993127148 / 0.997981781` |
| Recall@10 / @20 | `0.999781814 / 1.000000000` |
| MRR | `0.967624980` |
| mean / median rank | `1.088474 / 1` |
| macro-query AUROC | `0.977094977` |
| macro-query AUPRC | `0.967624980` |
| positive-vs-best-negative margin | `0.434560850` |
| signed Top1–Top2 gap | `0.436262577` |
| micro-candidate AUROC / AUPRC | `0.969442965 / 0.906253315` |

相对 Stage-1：

| 指标 | 变化 | 24-hypothesis formula-cluster CI |
|---|---:|---:|
| Recall@1 | **`+0.55637 pp`** | **`[+0.10521,+1.05232] pp`** |
| MRR | `+0.31711 pp` | `[+0.08951,+0.59155] pp` |
| macro-query AUROC | `+0.25471 pp` | `[-0.00281,+0.53902] pp` |
| macro-query AUPRC | `+0.31711 pp` | `[+0.07457,+0.58280] pp` |
| positive-vs-best-negative margin | **`+0.066262`** | `[+0.055570,+0.078115]` |
| signed Top1–Top2 gap | **`+0.066443`** | `[+0.055975,+0.078414]` |

Top-1 transition ledger：

```text
corrected  = 288
introduced = 186
net        = +102
risk_net_lambda2 = -84
```

因此，主种子是严格正向的 Recall@1、MRR、AUPRC 和 margin 结果；macro AUROC 点估计为正，但校正 CI 下界略低于 0，不能称显著。

对 official 的累计 Recall@1：

```text
0.49637 pp + 0.55637 pp ≈ +1.05274 pp
```

取原账本四舍五入口径可写成约 `+1.053 pp`。

#### 3.3.5 主种子 near-subset

| 指标 | 绝对值 | 相对 Stage-1 |
|---|---:|---:|
| Recall@1 | `0.893231087` | `+0.58988 pp`，CI `[-0.28601,+1.64059] pp` |
| MRR | `0.939592205` | `+0.33745 pp`，CI 跨 0 |
| macro AUROC | `0.961702980` | `+0.31319 pp`，CI 跨 0 |
| margin | `0.280061351` | **`+0.042338`，CI `[+0.027375,+0.059664]`** |
| signed Top1–Top2 gap | `0.283662500` | **`+0.042763`，CI `[+0.028461,+0.059690]`** |

near transitions：`150 corrected / 110 introduced`，`risk_net_lambda2=-70`。near 的几何 margin 明确改善，但 near Recall@1 仍未显著。

#### 3.3.6 复证种子 3408

绝对指标：

| 指标 | T1/T3 seed 3408 |
|---|---:|
| Recall@1 | `0.940598920` |
| Recall@2 / @3 / @5 | `0.984999727 / 0.993999891 / 0.998309060` |
| MRR | `0.967087409` |
| macro-query AUROC / AUPRC | `0.976745984 / 0.967087409` |
| micro-candidate AUROC / AUPRC | `0.971325685 / 0.908730640` |

相对 Stage-1：

| 指标 | 变化 | 24-hypothesis formula-cluster CI |
|---|---:|---:|
| Recall@1 | `+0.37637 pp` | `[-0.02868,+0.78093] pp` |
| MRR | `+0.26336 pp` | `[+0.04063,+0.50229] pp` |
| macro-query AUROC | `+0.21982 pp` | `[-0.00743,+0.45974] pp` |
| macro-query AUPRC | `+0.26336 pp` | `[+0.05217,+0.50330] pp` |
| margin | **`+0.070541`** | `[+0.060367,+0.083162]` |
| signed Top1–Top2 gap | **`+0.070687`** | `[+0.060616,+0.082641]` |

transitions：`265 corrected / 196 introduced`，净 `+69`，`risk_net_lambda2=-127`。

复证种子确认了方向和几何 margin，但 Recall@1 的 multiplicity-corrected CI 差约 `0.029 pp` 才严格高于 0。因此不能写成“双种子 Recall@1 均显著”；正确表述是“双种子点估计均为正，主种子显著，复证种子 MRR/AUPRC/margin 显著而 Recall@1 未过校正下界”。

#### 3.3.7 pooled AUC

这些是绝对值；当前结果没有给出与 Stage-1 的 pooled-AUC paired delta，因此只能报告绝对性能：

| 种子 | MassSpecGym 10-ppm AUROC / AUPRC | `[M+H]+` 10-ppm AUROC / AUPRC |
|---|---:|---:|
| 3407 | `0.849613 / 0.868891` | `0.857623 / 0.873918` |
| 3408 | `0.857806 / 0.875473` | `0.866696 / 0.881161` |

这两个种子的差异也说明：不能只凭一个 pooled AUC 点估计选择 checkpoint，更不能把 `0.8576/0.8667` 称为 NIST20 `0.85` 复现。

#### 3.3.8 当前资格与不能掩盖的问题

T1/T3 是当前最高 absolute shared-encoder 点估计，但有五个重要边界：

1. 两个种子 Recall@1 都为正，但只有主种子校正 CI 严格为正；
2. 两个种子的 `risk_net_lambda2` 都为负，说明 introduced 仍然偏多；
3. 训练中 clean query 占全部 48,544 query，而 action 只覆盖 2,730 query；按每个 action-query 的 action 总权重 0.5 计算，全局 action-attributable anchor dose 约为 `2,730*0.5/48,544 = 2.81%`；
4. 每个 action-query 的所有 action 在同一 query objective 中平分 0.5，因此平均约 7.29 个 action/query 时，单动作剂量很小；
5. T1/T3 corpus 使用动作谱作为 anchor，但没有逐动作保留 Stage-1 的 exact positive/negative 边界；其正负池按 query 的 measured relation 重新构造。因此它强力证明 relation-complete T1/T3 continuation 有效，却不能单独证明 `+0.556 pp` 主要由 Noise action content 造成。

此外，训练产物保存了 slim backbone/head，但没有保存更新后的 Adam state；因此可用于推理与比较，不能无损继续恢复这一轮之后的优化器动力学。该 sbatch 也没有完成 GNPS Gold/Silver 外部面板确认。

**最终裁决**：

- `seed 3407` 是当前最高 MassSpecGym held Recall@1 的 shared encoder research checkpoint；
- Stage-1 仍是 action-specific causal attribution 最干净的 checkpoint；
- T1/T3 的合理公开主张是“在 Stage-1 上进一步获得约 `+0.56 pp` 的主种子提升，并在复证种子上保持正向点估计和显著 margin/MRR 改善”；
- 不得写成“已稳定提升 5 pp”“双种子 Recall@1 均显著”或“动作贡献已被完全隔离证明”。

### 3.4 T1/T3 v2：恢复 exact bridge 后的最新 NO-GO

#### 3.4.1 v2 真正修复了什么

run `noise_relation_t1_t3_v2_run_2347230` 针对 v1 的两个真实缺口进行了修复：

1. clean T1/T3 stream 不再混入 action anchors；
2. 每个 Stage-1 action 恢复其原始的 exact hard-positive bridge：`action spectrum -> exact positive row -> exact negative row`，并独立占一个 action event。

语料审计确认：

| 项目 | 数值 |
|---|---:|
| 全部 action events | `25,687` |
| train-fold action events | `19,894` |
| fold-1 diagnostic events | `5,793` |
| outer-fold events | `0` |
| bad-registry / ragged / unrepresentable rejects | `0 / 0 / 0` |

v1 generic measured pool 对 exact row-level bridge 的保留情况：

| exact relation | v1 pool 中存在的比例 |
|---|---:|
| exact positive row | `63.9841%` |
| exact negative row | `65.2659%` |
| positive 与 negative exact rows 同时存在 | `47.1549%` |

因此，v1 确实没有逐动作完整保留约 `52.8451%` 的 **exact row-level boundary**。但这不等于 v1 丢掉了 52.8% 的全部化学监督：即使 exact row 缺席，同分子正谱和同一负分子的其他实测谱仍可能保留 identity/molecule-level 关系。正确措辞应限定在 row-level exact bridge。

#### 3.4.2 fold-1 三臂结果

fold-1 有 `16,742` queries。它位于 Stage-1 outer-train 分布中，仅用于选择，不是 outer held 性能主张。

共同 warm start：

```text
Recall@1 = 0.958607096
MRR      = 0.977140468
mean margin = 0.403598982
```

| arm | Recall@1 after | ΔRecall@1 | ΔMRR | Δmean margin |
|---|---:|---:|---:|---:|
| clean-only | `0.950304623` | **`-0.83025 pp`** | `-0.54622 pp` | `+0.069027` |
| v1 averaged checkpoint | `0.949826783` | **`-0.87803 pp`** | `-0.56408 pp` | `+0.061440` |
| exact action-rotation | `0.943734321` | **`-1.48728 pp`** | `-0.93469 pp` | `+0.040722` |

预注册选择规则要求最佳 arm 的 fold-1 Recall@1 delta 严格大于 0。三臂全部为负，因此：

```text
selected_arm = null
proceed_to_held = false
```

外层 held fold 没有被编码或评价，GNPS 也没有启动。该结果不能改变 v1 的 held 正向结果，也不能称为新的外层性能下降；它只判定 v2 当前训练协议不值得消费 held。

#### 3.4.3 最重要的科学诊断

这次不是单一问题，而是三个相互独立的信号：

1. **所有 arm 都出现 margin 上升但 Recall/MRR 下降。** 这直接证明 mean margin 不是排名质量的充分统计量。T1/T3 把大量已经容易的关系推得更远，同时边界尾部发生更多 rank flip；以后不能再用 margin 增长替代 Recall、MRR 和 transition ledger。
2. **clean-only 与 v1 averaged 在 fold-1 几乎相同。** averaged 只比 clean-only 再低约 `0.0478 pp`，说明 v1 中被平均到 2.81% 全局剂量的 action 并不是 fold-1 下降的主要来源；共同的 clean relation continuation/分布遗忘已经贡献大部分下降。
3. **action-rotation 比 clean-only 额外低约 `0.6570 pp`，但这个差不能被当成 action 因果效应。** v2 让每个 action event 独占一个 optimizer step，而 clean step 最多容纳 4 个 query；19,894 个 action events 来自 2,730 个 action queries，即平均 `7.29` 个 events/query。同一 action-query 因动作多而被重复更新，优化步数也没有与 clean-only 做 matched-step control。这违反了“以 query 为剂量单位、同一 query 不因动作多而被反复轰炸”的科学合同。

还有一个选择协议限制：Stage-1 warm start 已经使用 outer-train 数据，而 v2 continuation 只训练 folds 2/3/4，再用 fold-1 检查遗忘。因此 fold-1 对继续训练存在结构性的 catastrophic-forgetting 压力；它适合做安全停止门，却不能单独回答某个方法在真正未见分布上的泛化价值。

#### 3.4.4 v2 的正确结论

- exact action bridge 的 provenance 修复成功；
- action-rotation 的剂量设计不合格；
- 当前 v2 三臂均不应进入 held；
- v1 seed 3407 的 `94.2399%` 仍是最高已评价 shared-encoder 点估计；
- 不能说“exact action 无效”，只能说“one-action-one-step、按动作数累积剂量的 v2 训练协议失败”。

如果以后重做，只允许保留 v2 的 exact bridge table，不允许复用其动作剂量：每个 query 每轮至多抽取一个 action，动作跨轮 rotation；总 optimizer step 与 matched clean control 完全相同，并在训练前比较 targeted/control 的原始 triplet 梯度，而不是靠训练后的内部统计自证。

### 3.5 query-balanced exact-bridge V3：fold-1 NO-GO

V3 不修改 v2 历史文件，而是复用 v2 已验证的 exact action-event ledger，并重新定义训练剂量：

```text
无 action 的 query:
    L_query = clean relation-complete T1/T3

有 action 的 query:
    L_query = 0.5 * clean relation-complete T1/T3
            + 0.5 * native exact cosine-triplet hinge
```

关键合同：

- 每个 clean query 在一个 pass 中恰好出现一次；
- 每个 action-query 只选择一个 exact event，动作多不会增加 optimizer steps；
- query 内动作按确定性 coverage-first rotation 选择，下一 rotation 在该 query 内换下一个动作；
- targeted/control 共享 query、exact positive、exact negative、batch、seed、Adam state 和 optimizer step count；
- 两臂唯一差异是 `targeted_action_spectra` 对 `control_action_spectra`；
- clean、action、positive、negative 全部进入同一个 live shared encoder；
- 使用 Stage-1 原生 Adam moments、`5e-6`、FP32、100 peaks、margin `0.1`；
- 第一作业只做 fold-1 targeted/control screen，不接触 outer held 或 GNPS；
- NO-GO 时不把 slim/Adam 大文件写回共享盘，避免无意义 quota 消耗。

本地检查：Python 编译通过，V3 core tests `7/7` 通过，SBATCH 合同测试通过，`bash -n` 通过。测试直接覆盖四类 triplet roles 的梯度、targeted/control 注入前梯度差异、query 剂量、coverage-first rotation、exact-bridge batch packing 和 fold-1 选择逻辑。

服务器 run `noise_relation_t1_t3_v3_run_2347335` 已完成 fold-1 screen。共同 warm start 为 Stage-1 targeted checkpoint：Recall@1 `0.958607096`、MRR `0.977140468`、mean margin `0.403598982`。结果如下：

| arm | Recall@1 after | ΔRecall@1 | ΔMRR | Δmean margin |
|---|---:|---:|---:|---:|
| targeted | `0.950901923` | **`-0.77052 pp`** | `-0.49543 pp` | `+0.070779` |
| matched control | `0.948751643` | **`-0.98555 pp`** | `-0.63436 pp` | `+0.070080` |

targeted 相对 matched control 为 Recall@1 `+0.21503 pp`、MRR `+0.13893 pp`，但 mean-margin 差仅 `+0.000699`。因此 exact targeted action 的方向仍优于 same-query control，但两臂共同的 clean relation-complete T1/T3 continuation 主导了更新，并使 fold-1 排名退化。V3 没有进入 outer held 或 GNPS，不能产生新的 held 性能主张，也不能覆盖 run 2347055。

V3 的根本错误不是 query balancing 没执行，而是优化目标的全局剂量仍然错误：约 `2,730 / 48,544` 个训练 query 带 action，且 action 在这些 query 内权重仅为 `0.5`，故 action-attributable 全局剂量约为 `2.81%`；其余约 `97.19%` 是两臂完全相同的 clean T1/T3 更新。共同 margin 上升约 `0.070`、两臂 margin 差不足 `0.001`，与这一诊断一致。此外，V3 从 Stage-1 checkpoint 而不是当前最高的 V1 seed-3407 checkpoint 开始，未遵守“从最好已有 encoder 继续”的继承要求。

后续不得复用 V3 的全量 clean T1/T3 pass。唯一仍值得保留的是：V2 的 exact action-event ledger、按 query 计量、同 query matched control，以及 full shared encoder 的四角色梯度。若继续，必须从 run 2347055 的 V1 seed-3407 checkpoint 开始，只在 action-query 上使用原生 exact cosine triplet，并用同一 exact positive/negative 边界上的 clean anchor 做成对 preservation；不得再让非 action query 的共同 relation loss 淹没动作差异。

入口：

```bash
sbatch tasks/run_noise_relation_t1_t3_v3_2gpu.sbatch
```

---

## 4. P2b：冻结局部候选证据融合重排器

### 4.1 算法定位

P2b 不更新 encoder。它在 strict-10-ppm、同加合物候选分子集合中，用原始峰和中性丢失证据重排 DreaMS 候选。

冻结打分：

```text
score = 0.10 * DreaMS_similarity
      + 0.00 * sqrt_cosine
      + 0.10 * entropy_similarity
      + 0.80 * neutral_loss_sqrt_cosine
```

- normalization：`absolute`；
- minimum support：`1`；
- minimum advantage：`0.0`；
- 排除 query 自身谱；
- 同一候选分子的多张谱取 molecule-max；
- 模型选择按分子式隔离。

主信号是 neutral-loss sqrt cosine；DreaMS 与 entropy 提供较小修正。它是固定权重 rank fusion，不是 learned reranker，也不是 embedding 微调。

### 4.2 formula-isolated nested OOF 开发结果

开发集：`5,037` queries、`2,522` identities、`1,082` formulas、`2,094` near queries。

| 指标 | official DreaMS | P2b | 变化 |
|---|---:|---:|---:|
| Recall@1 | `0.8606` | `0.8997` | **`+3.91 pp`** |
| MRR | `0.9161` | `0.9411` | **`+2.50 pp`** |
| near Recall@1 | `0.7612` | `0.8195` | **`+5.83 pp`** |
| corrected / introduced | — | `280 / 83` | 净 `+197` |

Recall@1 formula-cluster bootstrap 95% CI：`[+2.91,+4.87] pp`。这是分子式隔离开发证据，不是封存外测结果。

### 4.3 sealed P3 main

P3-main-real-pristine：`3,000` queries。

| 指标 | official DreaMS | P2b | 变化 |
|---|---:|---:|---:|
| Recall@1 | `0.8793` | `0.8900` | **`+1.07 pp`** |
| MRR | `0.9304` | `0.9361` | **`+0.57 pp`** |
| macro-query AUC | `0.9209` | `0.9265` | `+0.56 pp` |
| corrected / introduced | — | `89 / 57` | 净 `+32` |

- Recall@1 formula-cluster CI：`[+0.24,+1.89] pp`；
- McNemar exact `p=0.0101`；
- MRR formula-cluster CI 为正；
- macro-query AUC 点估计为正，但 CI 跨 0；
- 净修正 `32/362 = 8.8%` 的 official Top-1 错误。

这是 P2b 最稳妥、可引用的封存正向成果。

### 4.4 near-core 硬边界

near-core：`496` queries。

| 指标 | official DreaMS | P2b | 变化 |
|---|---:|---:|---:|
| Recall@1 | `0.4879` | `0.4456` | **`-4.23 pp`** |
| MRR | `0.6990` | `0.6704` | `-2.86 pp` |
| corrected / introduced | — | `20 / 41` | 净 `-21` |

因此 P2b 不能直接用于 MCES 0–2 极近异构体候选集合。开发 OOF 的 near `+5.83 pp` 与 sealed near-core 的 `-4.23 pp` 必须同时出现，不能只摘录前者。

### 4.5 消融中仍有价值的正向发现

- neutral-loss-only 在开发 OOF 已贡献约 `+3.91 pp`；
- 完整 P2b 在开发 OOF 与 NL-only Top-1 几乎相同；
- 但在 sealed P3-main，完整 P2b 相对 NL-only 仍提高 Recall@1 `+0.97 pp`，CI 为正；
- 因而可解释为“中性丢失提供主增益，DreaMS/entropy 的小权重在封存主面板提供泛化修正”。

### 4.6 当前资格

P2b 是有正式封存正向证据的局部重排模块，但不是通用重排器：

- 可引用：P3-main Recall@1 `+1.07 pp`；
- 不可引用为 encoder gain；
- 不可声称解决近异构体；
- 在没有新的标签盲安全门与新封存面板前，不能对 near-core 无条件启用。

---

## 5. 近期三通道 RRF 重排器

### 5.1 算法

对每个候选分子分别做 molecule-max，再融合三条候选排序：

1. DreaMS embedding cosine；
2. greedy sqrt-intensity cosine；
3. top-20 fragment-presence overlap。

冻结 RRF：

```text
RRF(candidate) = sum_channel 1 / (60 + rank_channel(candidate))
```

它零参数、零训练、标签盲、只使用 rank，不使用不同量纲的 raw score。严格平局规则要求真分子严格高于所有对手；平局不赠送给真分子。

### 5.2 official geometry 上的探索性正向结果

| 方法 | strict Recall@1 | ties | corrected | introduced | net | risk-net λ2 | C:I |
|---|---:|---:|---:|---:|---:|---:|---:|
| DreaMS solo | `93.318%` | `42` | `24` | `0` | `+24` | `+24` | — |
| cosine-max | `92.904%` | `102` | `412` | `448` | `-36` | `-484` | `0.92` |
| top-k overlap max | `81.667%` | `2,417` | `281` | `462` | `-181` | `-643` | `0.61` |
| RRF(d,c) | `91.960%` | `602` | `145` | `66` | `+79` | `+13` | `2.20` |
| RRF(d,t) | `92.653%` | `698` | `233` | `65` | `+168` | `+103` | `3.58` |
| **RRF(d,c,t)** | **`94.954%`** | `13` | **`425`** | **`101`** | **`+324`** | **`+223`** | **`4.21`** |

三方 RRF 在这一 official-geometry 探针上相对 re-encoded official ledger 净提升约 `324/18,333 = 1.767 pp`。

这里的 `DreaMS solo=93.318%` 是该探针自己的 strict-tie/re-encode 账本，不是 canonical official `93.1875%` 的替代值。原报告记录了 re-encoded proxy mismatch；因此 RRF 的 `425/101` transition ledger 和 `94.954%` 只在这套探针几何内解释，不能用 `94.954-93.1875` 再造一个跨账本增益。

分层：

- near-subset Recall@1：`88.28% -> 90.95%`，`212 corrected / 47 introduced`；
- same-formula non-near：Recall@1 `95.38%`，`170/41`；
- unclassified：Recall@1 `98.94%`，`43/13`。

这证明原始谱图中存在与 DreaMS 排名互补的强度无关碎片存在性信息；单独使用 top-k overlap 很差，但其排序投票与 DreaMS/cosine 组合时可显著提高交换比。

### 5.3 为什么该正向结果不能进入最终系统

该结果随后按冻结协议做了唯一确认：

- DreaMS channel 替换为 Stage-1 champion；
- 固定 `RRF(d,c,t,k=60)`；
- 同时报 corrected MassSpecGym 与 GNPS Gold/Silver identity/formula-disjoint 面板；
- run：`noise_stage1_rrf_confirmation_run_2347032`。

确认结果：

- internal Recall@1 delta 非正；
- formula-cluster CI 下界非正；
- risk-net λ2 非正；
- GNPS identity-disjoint 与 formula-disjoint 的大部分 retrieval rank、MRR、macro AUC/AUPRC 指标退化；
- 只有 micro-candidate 与 pooled pairwise AUC/AP 没有退化；
- `reranker_promotion_authorized=false`；
- `embedding_gain_claim_authorized=false`。

重叠分解解释了原因：official 几何上 425 个 corrections 中，`166` 个已被 Stage-1 修正，只有 `259` 个落在 Stage-1 的 1,158 个残余错误上，即 `22.3%` 的 residual headroom；固定 naive signals 与 Stage-1 的可修正错误高度重叠，而不是可稳定叠加。

### 5.4 当前资格

RRF 应保留为一条重要的正向机制发现和失败边界：

- 正向事实：在 official geometry 的 held 探针上，三通道 rank consensus 达到 `94.954%`、risk-net `+223`；
- 机制事实：原始碎片存在性排序含有 DreaMS 未完全编码的信息；
- 最终事实：叠加 Stage-1 并做 GNPS 独立确认失败，因此它不是当前部署组件、不是新的性能增益、也不是 encoder 提升。

不能只报 `94.954%` 而隐去 run 2347032；这会把探索性发现误写成已确认算法成果。

---

## 6. 三条线之间不能混算的数字

| 数字 | 真正含义 | 不能写成 |
|---|---|---|
| Stage-1 `+1.26548 pp` | targeted action content 相对 matched control 的因果差 | 对 official 的绝对提升 |
| Stage-1 `+0.49637 pp` | shared encoder 相对 official 的绝对 Recall@1 提升 | action causal contrast |
| T1/T3 `+0.55637 pp` | seed 3407 相对 Stage-1 的增量 | 双种子稳定的 0.56 pp |
| T1/T3 cumulative `~+1.053 pp` | seed 3407 相对 official 的累计 encoder 提升 | `+1.265 + 0.556` 的简单相加 |
| P2b `+3.91 pp` | formula-isolated nested-OOF 开发结果 | sealed 或 external 增益 |
| P2b `+1.07 pp` | sealed P3-main reranker 增益 | encoder 增益或 near-core 增益 |
| RRF `94.954%` | official geometry 上的探索性 fixed-rank-fusion结果 | Stage-1 上确认成功的最终系统 |
| pooled AUROC `0.85–0.87` | MassSpecGym 10-ppm pairwise 指标 | NIST20 论文 0.85 精确复现 |

不同方法、基线和测试集的 pp 不允许直接相加。尤其不能把 T1/T3 encoder、P2b 和失败确认的 RRF 叠成一个虚构的“总提升”。

---

## 7. 当前可对外使用的严谨表述

### 7.1 shared encoder

> 使用原生 DreaMS 对比学习内核构造 action hard-positive triplets 后，Noise Stage-1 在 18,333-query corrected held graph 上相对 official DreaMS 提高 Recall@1 0.50 个百分点；targeted action content 相对 matched same-query control 的提升为 1.27 个百分点，公式簇配对区间严格为正。进一步使用 relation-complete multi-hinge 与 molecule-max listwise 目标，从 Stage-1 继续训练，在预注册主种子上再提高 0.56 个百分点，并在复证种子上得到 0.38 个百分点的同向点估计。当前最高 absolute Recall@1 为 94.24%，相对 official 累计约提高 1.05 个百分点。

必须紧接一句：

> T1/T3 的风险网仍为负，复证种子的 Recall@1 校正区间轻微跨零，因此这是有真实几何改善的中等幅度结果，而不是已经稳定达到 5 个百分点。

### 7.2 P2b

> 固定权重的 DreaMS/entropy/neutral-loss 局部候选融合在 formula-isolated nested OOF 中提高 Recall@1 3.91 个百分点，并在封存 P3-main 面板上提高 1.07 个百分点；其增益主要来自中性丢失证据。该方法在极近异构体 near-core 上有害，不能无条件部署。

### 7.3 recent RRF

> 三通道 RRF 在 official geometry 的探索性完整候选重排中达到 94.954% Recall@1 和正 risk-net，证明原始谱图的碎片存在性排名包含与 DreaMS 互补的信息；但固定配方叠加 Stage-1 并在 GNPS 面板确认时失败，因此该结果仅作为机制证据，不作为部署或性能主张。

---

## 8. 当前资产与复现入口

### 8.1 Stage-1 / Stage-2

- `tasks/build_noise_dreams_native_triplets.py`
- `tasks/train_noise_dreams_native.py`
- `tasks/train_noise_dreams_native_residual_stage2.py`
- `tasks/run_noise_dreams_hard_positive_native_2gpu.sbatch`
- `docs/NOISE_DREAMS_NATIVE_HARD_POSITIVE_BRIDGE_20260925.md`
- `docs/NOISE_DREAMS_NATIVE_RESIDUAL_STAGE2_20260926.md`
- `docs/NOISE_5PP_AND_AUC_CORRECTED_LEDGER_20260927.md`
- Stage-1 result root：`data/validation/noise_dreams_hard_positive_native_fold_0_run_2344820/`

### 8.2 T1/T3

- `tasks/build_noise_relation_t1_t3_corpus.py`
- `tasks/noise_relation_t1_t3_core.py`
- `tasks/train_noise_relation_t1_t3.py`
- `tasks/run_noise_relation_t1_t3_2gpu.sbatch`
- `tasks/test_noise_relation_t1_t3_core.py`
- result root：`data/validation/noise_relation_t1_t3_run_2347055/`
- primary checkpoint：该 run 的 seed-3407 `final_slim.pt`
- replicate checkpoint：该 run 的 seed-3408 `final_slim.pt`

v2 现已有 train-side NO-GO：`data/validation/noise_relation_t1_t3_v2_run_2347230/`。它没有 outer held 结果，不得覆盖 run 2347055；其可复用资产仅限 exact action-event ledger 与最终 Adam-state 保存实现。

V3 修复实现：

- `tasks/noise_relation_t1_t3_v3_core.py`
- `tasks/train_noise_relation_t1_t3_v3.py`
- `tasks/summarize_noise_relation_t1_t3_v3_fold1.py`
- `tasks/run_noise_relation_t1_t3_v3_2gpu.sbatch`
- `tasks/test_noise_relation_t1_t3_v3.py`
- `tasks/test_noise_relation_t1_t3_v3_sbatch.py`

### 8.3 P2b

- `tasks/build_g8r_p2_listwise_cache.py`
- `tasks/train_g8r_p2b_rank_fusion.py`
- `tasks/g8r_p2_rank_fusion_core.py`
- `tasks/eval_g8r_p2b_on_sealed_p3.py`
- `data/validation/g8r_p2b_rank_fusion.json`
- `data/validation/g8r_p2b_locked_ablation.json`
- `data/validation/g8r_p2b_p3_final.json`
- `docs/P2B_RANK_FUSION_FORMAL_RECORD_20260823.md`

### 8.4 RRF

- `tasks/GLM_probe_full_rerank_rrf.py`
- `data/validation/GLM_full_rerank_rrf_probe_20260929_v2_strict/report.json`
- `data/validation/GLM_sealed_result_full_rerank_rrf_20260929.md`
- confirmation run：`data/validation/noise_stage1_rrf_confirmation_run_2347032/`

---

## 9. 最终项目裁决

1. **当前 shared-encoder 最高点**：T1/T3 seed 3407，Recall@1 `94.2399%`，相对 official 约 `+1.053 pp`。
2. **当前最强 action-specific 因果证据**：Stage-1 targeted vs matched control `+1.26548 pp`，公式簇 CI 严格为正。
3. **当前最强已封存 reranker 正向证据**：P2b P3-main `+1.07 pp`；near-core 禁用边界必须保留。
4. **近期 RRF 的最终状态**：探索性 official-geometry 结果很强，但 Stage-1/GNPS 确认失败，永久不升格。
5. **最新 T1/T3 v2 状态**：exact bridge ledger 修复成功，但 one-action-one-step 剂量过冲；fold-1 三臂全部下降，held 未消费，v1 结果不变。
6. **尚未实现的目标**：没有任何一条已确认的 shared-encoder 路线达到 `+4–5 pp`；把不同基线、不同数据集、encoder 与 reranker 的数字相加不能弥补这一事实。

这份总账以后应作为 Noise encoder 与 reranker 数字的唯一解释入口；任何新结果都必须在对应小节追加数据集、基线、完整指标、paired CI、corrected/introduced/risk-net 和外部面板状态后，才能改变上述裁决。
