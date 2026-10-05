# Noise corrected-fullgraph 方法重置（2026-09-06）

状态：方法发展中；禁止提交正式训练；禁止声称已获得 4–5 pp encoder 提升。

## 1. 旧正式图被撤销

旧 `g8r_error_atlas_listwise_cache.npz` 的 23,876-query cohort 来自对
`SIMULATION_CHALLENGE=False` 的错误语义解释。它只能保留为历史受限 cohort，不能承载新的正式结果。
因此 `run_noise_final_direct_boundary_v2_phase_a.sbatch` 已 fail-closed；旧 graph、旧
held ledger、旧 action route label 都不得移植为纠正图结论。

新的 train-side development graph 为：

- 83,619 queries；9,854 identities；6,220 formulas；
- 392,229 candidate molecules；6,220,661 query-candidate spectrum edges；
- 87,848 unique reachable spectra；
- same-adduct、query-centred strict 10 ppm、positive molecule first、spectrum-max molecule score；
- P3-disjoint；P3 outcome 未消费。

## 2. official 全指标基线

完整结果：`data/validation/noise_corrected_official_full_metrics_v1_20260906/report.json`。

主要 retrieval：

| 指标 | official |
|---|---:|
| Recall@1 | 0.928760 |
| Recall@2 / 3 / 5 / 10 / 20 | 0.978438 / 0.990851 / 0.997417 / 0.999725 / 1.000000 |
| MRR | 0.959622 |
| mean / median rank | 1.114328 / 1.0 |
| macro-query AUROC / AUPRC | 0.968687 / 0.959622 |
| micro-candidate AUROC / AUPRC | 0.956287 / 0.864155 |
| positive-vs-best-negative margin | 0.340170 |
| Top1–Top2 gap | 0.351168 |

near subset 有 28,188 queries、4,036 errors，Recall@1=0.856819；全图共有 5,957 errors。

同一冻结 graph 的 spectrum edges 还给出：

- 全 adduct 6,220,661 directed pairs：MassSpecGym 10-ppm pooled pairwise AUROC=0.847501；
- `[M+H]+` 5,931,207 directed pairs：AUROC=0.855089，AUPRC=0.843585；
- 该 0.855 与论文约 0.85 数值接近，但不是 NIST20/MoNA-disjoint ledger，必须标记
  `exact_nist20_paper_replication=false`。

字面“所有指标 +5 pp”在数学上不可能：Recall@2 以上、macro AUROC、micro AUROC 的剩余
headroom 分别小于或等于 5 pp。正式可实现门槛应为 Recall@1 与 near Recall@1 绝对
+4–5 pp；未饱和 AUROC/AUPRC 显著正增益；饱和 Recall@k 报剩余错误消除比例且不得退化；
rank/margin/gap 按自身单位和方向报告。

## 3. 旧 action 覆盖并不是自然上限

旧成熟 N actions 映射到纠正图后只覆盖：

- 35,028 actions；9,516 queries；
- 1,451/5,957 official errors（24.36%）；
- 即使覆盖错误全部修正且零引入，query-specific correction 的全图上限也只有 +1.735 pp。

因此旧 S3A `+3.854 pp` 不能外推为全图希望：它评估的是 action-rich cohort，不是全部
83,619 queries。共享泛化仍可能存在，但必须在完整 held formula/query 图上证明。

## 4. 两个完整 held-fold proxy 的反证

fold 0 有 18,333 held queries / 1,249 held formulas。训练使用其余 65,286 queries，
评估使用该 fold 全部 queries 和其全部 candidate edges。

| shared map | Recall@1 delta | corrected / introduced | pooled pairwise AUROC delta | 裁决 |
|---|---:|---:|---:|---|
| bounded diagonal, 2 epochs | +0.0055 pp | 15 / 14 | -0.0109 pp | 否定 |
| bounded low-rank-16, 2 epochs | +0.1091 pp | 53 / 33 | -0.0362 pp | 容量增加仍远离目标 |

这否定了“只把旧 action presence 当权重，再训练一个全局 shared map”的方案。旧 action
payload 被丢弃时，增加映射容量不能恢复 4–5 pp。

## 5. 发现并修正 matched-control 注入断档

原 S3A 的 target path 与 matched control 用于因果审计；此前实现却把“两个 strict matched
control path 完整”错误提升为直接训练 target action 的资格门。

在固定随机 16-query 的真实 official encoder smoke 上：

- candidate-gradient 合法 target prefixes：step 3/4/5/6 分别覆盖 13/13/12/11 queries；
- role-confounder 合法 target prefixes：step 1/2/3/4/5 分别覆盖 5/4/3/3/1 queries；
- target-first 实际发布 65 actions / 13 queries；
- control-gated 实际只发布 22 actions / 7 queries；
- 丢弃 43/65=66.15% 合法 target actions，并把 role-confounder 整支清零。

修正后的 `build_noise_corrected_full_action_bank.py`：

- 在当前 encoder/candidate geometry 每一步重新选择 target、更新 candidate context；
- outer-held formula 不发布；
- 所有合法 target prefix 发布；matched controls 仅作为可用时的诊断，不再是训练资格；
- 不计算 target/control rank、corrected、introduced 或 teacher margin；
- 完整运行预计产生数十万级 actions，但正式数量必须由全量结果给出，不能从 16-query
  smoke 外推为事实。

## 6. target-only 直接注入合同

`direct_target_boundary_objective` 不做蒸馏，也不依赖 matched control：

1. clean query 与 raw target action 都直接对同一 live molecule-max identity boundary 优化；
2. action-eligible query 获得一次固定 clean-boundary 额外剂量；
3. 同一 query 内所有 action loss 先平均，1 条复制成 10 条不会改变 loss 或 clean gradient；
4. action safety 只阻止 target view 相对 clean boundary 退化；
5. teacher embedding、teacher margin、teacher residual 均不存在。

数值测试已证明 action multiplicity 从 1 到 10 时 loss 与 clean gradient 不变。

真实 117M encoder 的 13-query / 65-action CPU smoke 进一步证明：

- clean-boundary、action-rank、action-safety、action→clean 五个分支均进入真实计算图；
- 初始 action→clean 梯度中位数 0.070，只是 clean-boundary 2.168 的约 3.2%，不能直接接受；
- 未校准版本 4/4 optimizer steps 被 global clip；
- 仅做总梯度 p90 缩放后，loss scale=0.1033，clip fraction 从 100% 降为 0%，但分支比例不变；
- 当前实现进一步把 action→clean 冻结校准为 clean-boundary 梯度中位数的 25%，倍率上限 8，
  再由各分支最大范数三角上界确定一次性 global scale。倍率和 scale 都在 optimizer 前冻结，
  不读取 action outcome 或 held 指标。

## 7. 进入服务器前尚缺的门槛

1. 在更大的固定随机 query panel 上确认 target-first eligibility、error coverage 与 formula 覆盖；
2. 把 target-only objective 接入一个剥离旧 R0/old-held-ledger 依赖的最小 corrected trainer；
3. 在更大 action panel 上复验 25% transfer 梯度比例、clip fraction <10% 和 zero-change gate；
4. 冻结全指标 evaluator 与 promotion gate；
5. 之后才允许在服务器进行只读资产核验并生成 `#SBATCH --gpus=1` 的唯一原子输出作业。

服务器训练的正式主终点仍必须是未消费 formula panel、多 seed、query-weighted Recall@1
绝对提升至少 4 pp、multiplicity-corrected formula-cluster CI 严格为正，并同时报告本文件
第 2 节全部指标。开发 proxy、action eligibility 和 0.855 pairwise baseline 都不是 encoder
提升结果。
