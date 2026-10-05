# Noise E4-DEB Phase A：直接候选边界训练与覆盖审计结果

日期：2026-09-05  
状态：fold 0 / seed 20260905 已完成；原配方停止，不进入多折

## 科学问题

在同一个成熟 E4 shared clean-spectrum encoder、同一训练日程和同一 N 动作集合下，显式保留逐候选 `target-control` 边界的直接损失，能否优于 clean continuation 与 matched-random augmentation？

## 冻结制品

- 原始训练：`data/validation/g8r_noise_final_e4_deb/fold_0_run_2331540`
- 正式续评估：`summary_resume_2331557/report.json`
- 覆盖审计：`coverage_audit_2331558/report.json`
- 三个 checkpoint SHA256：
  - clean duplicate：`894be99f63d475006bccf4d4c321cb0631c3e6e79e64484fd1113ef699faa720`
  - matched random：`c9a1cefa4a2547e49cab25c21a2bd8f73cdc6006d08e9fcfcc9e5369fa6bd2cc`
  - candidate boundary：`e625e2f087b8dd311c74d5e3ce49a9cdcb3384a48f8d3bfb2095101f11fbbeb0`

## 结果

同一 held formula fold 共 5,923 queries。相对匹配的 official DreaMS：

| arm | Recall@1 | delta | corrected / introduced | MRR | macro query AUC |
|---|---:|---:|---:|---:|---:|
| clean duplicate | 0.940571 | +0.7598 pp | 58 / 13 | 0.965801 | 0.970455 |
| matched random | 0.940402 | +0.7429 pp | 58 / 14 | 0.965759 | 0.970362 |
| candidate boundary | 0.939895 | +0.6922 pp | 56 / 15 | 0.965393 | 0.969724 |

Candidate boundary 相对 clean duplicate：

- Recall@1 `-0.0675 pp`；7 corrected / 11 introduced；risk-net(lambda=2) = -15。
- MRR `-0.0408 pp`；near Recall@1 `-0.0831 pp`。
- mean positive-vs-best-negative margin `-0.007410`，formula-cluster 95% CI `[-0.011359, -0.004199]`，严格为负。
- macro query AUC `-0.0730 pp`，但 micro candidate AUC `+0.0316 pp`、micro candidate AUPRC `+0.0830 pp`。

Candidate boundary 相对 matched random：

- Recall@1 `-0.0507 pp`；7 corrected / 10 introduced；risk-net(lambda=2) = -13。
- MRR `-0.0366 pp`；near Recall@1 `-0.0554 pp`。
- mean positive-vs-best-negative margin `-0.006895`，formula-cluster 95% CI `[-0.010704, -0.003728]`，严格为负。

所有 paired promotion gates 均失败，`pass_to_multifold=false`。这不是“小样本下尚未显著的正趋势”：最关键的 hardest-boundary margin 已显著朝错误方向移动。

## 覆盖分解

Outer-train ledger 含 304,209 actions、17,953 queries、857 formulas；其中 1,097 queries 是 action-covered current-geometry errors。当前 E4-DEB 只训练 N 臂 strict-positive error actions：1,311 actions、382 queries、123 formulas，即只覆盖可见错误的 34.82%。

| source | action-covered errors | strict-positive error queries | corrected error queries | introduced queries |
|---|---:|---:|---:|---:|
| N | 561 | 382 | 140 | 61 |
| P-intensity | 1,097 | 837 | 450 | 408 |
| P-transfer | 656 | 462 | 238 | 76 |

N/P action-oracle union 可纠正 621/1,097 errors（56.61%）。P 在 N 之外新增 481 个独立错误；N 与全部 P 的纠错交集为 107，N-only 为 33。上述均为 outer-train 动作覆盖/结果诊断，不能写成 held 或部署增益。

## 根因裁决

1. **训练覆盖过窄。** N-only errors scope 排除了 65.18% action-covered errors，并遗漏 P 独有的 481 个可纠正错误。
2. **路由标签精度不足。** N 的 382 个 strict-positive error queries 中只有 140 个被动作真正纠正；`target-control advantage > 0.01` 不等价于跨越 Top-1 边界。
3. **损失优化了错误层级。** micro candidate AUC/AUPRC 略升而 hardest-negative margin、Top-1、MRR 和 macro AUC下降，说明大量较容易候选边压过了真正决定检索结果的临界负候选。
4. **梯度标定参照失效。** 初始化时 action gradient norm median 约 25.23，而 preservation/safety gradient median 约 `4.29e-6`。请求的缩放约 `1.70e-7`，实现却保留 effective scale = 1，随后每个 epoch 100% clipping。用近零 preservation gradient 标定 action branch 在数学上不稳定，也没有实现预期平衡。
5. **P 风险异质。** P-transfer 的 corrected/introduced query 比明显优于 P-intensity；不能把 275,700 个 P actions 作为一个等价正监督池，也不能因扩大覆盖而恢复旧 dynamic N+P 混合。

## 停止与下一步边界

- 停止原样扩大 E4-DEB、学习率扫描和多折训练。
- 保留三个 checkpoint 和全部 per-query ledger 作为负结果与配对对照。
- 下一轮仍从同一成熟 shared encoder 出发，直接微调 clean embedding；不引入下游专家或 P2b。
- 首先修复梯度标定：以非零 clean-ranking boundary gradient 为参照，按 formula/source 分支归一化；不得再以近零 preservation gradient 决定 action scale。
- 监督分层：actual top1 correction 为高权重，strict-positive but not corrected 为低权重 partial-margin，harmful 为零 corrective weight并进入独立安全约束。
- 候选损失优先作用于当前 top wrong molecule 与被动作改变的临界边；其余候选只承担 preservation，不再让大量容易边主导更新。
- 扩展顺序为 N → N+P-transfer → 通过独立风险门后的高精度 P-intensity；每一步均配 clean 与 matched control，并保持总 query/action exposure 匹配。

## DreaMS 原文 0.85 与本任务评估边界（更正）

原文约 0.85 是 NIST20 `[M+H]+` 谱图对上的 pooled ROC-AUC，不是 Top-1。
官方 notebook 以同 IK14 为正例、前体质量在 10 ppm 内但 IK14 不同为负例，并将
NIST20 中与 MoNA contrastive 微调集重叠的 IK14 排除；论文描述最终约 750,000 个
pair。公开 notebook 的最终图读取的是一个 100 万 pair 的中间预测表，过滤后为
750,535 行。

此前列出的两个文件名来自官方源码，但用途不同：

- `nist20_clean_spec_entropy_[M+H]+_retrieval.pkl` 是 notebook 构造的 NIST20 谱表；
- `nist20_clean_spec_entropy_[M+H]+_50k_pairs_retrieval.pkl` 只在训练期
  `SpecRetrievalValidation` callback 中被读取。公开仓库没有该 50k pair 表的生成
  代码，也没有证据表明它就是论文 Fig. 4b 的最终约 750k pair 表。

因此，不能要求用户下载这两个文件，也不能用缺少它们阻塞本项目模型晋级。E4-DEB
的正式裁决只使用匹配的冻结 MassSpecGym query/candidate graph。后续统一报告该任务的
Recall@1/2/3/5/10/20、MRR、macro/micro AUROC/AUPRC、margin 与风险覆盖；另在冻结的
MassSpecGym 10-ppm spectrum-pair ledger 上报告 pooled ROC-AUC，作为与原文相同数学
定义的域内指标，但明确不称为 NIST20 精确复现。只有拿到并验证原论文 pair ledger
及 MoNA-disjoint 清单后，NIST20 才可作为外部补充结果。
