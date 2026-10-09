# 全模块统一算法 v1：证据资格驱动的联合模型

日期：2026-10-08  
状态：2026-10-09 保真与泄漏修订后，核心模型、输入装配器、冻结器与一次性外测入口已建立；尚无合格的真实统一模型胜利结果，禁止开启 Enveda。MassSpecGym 与 GNPS Gold/Silver 均为 `development/consumed`。

> 本修订优先于较早的“全部模块 active/严格正权”设计。ChemAware Stage-1、Phase-A、V2 是三个独立资产；RRF 保持 `retired_audit_only`；逐候选 applicability 必须显式；权重允许精确为零；三个 raw P2b 通道与 frozen P2b composite 互斥。外测规则见 `UNIFIED_ALGORITHM_DATA_LEAKAGE_LEDGER_20261009.md`。

## 1. 唯一决策对象

对查询谱图 `q` 的同一候选集合 `C_q`，只有 registry 中资格合格、且在候选 `c` 上 applicable 的模块产生证据。当前默认生产集合为：

- official DreaMS（冻结对照）；
- Noise V1（条件稳健共享表示）；
- ChemAware Stage-1 encoder（独立确认的化学表示）；
- weighted spectral entropy（强经典全谱证据）；
- frozen P2b-on-Noise composite（P2b 家族唯一 enabled 代表）。

Phase-A、MSG、ChemAware V2 仍是 provisional/development-only；RRF 已退役；BioAware B47 尚无合格事件级外测。它们保留在总账，但不会因“全模块”名称自动进入生产 bundle。

原始分数先在每个 query 的候选集合内变成 rank-logit evidence，避免直接混合量纲。查询级可靠性只使用 label-free 摘要：可用率、top gap、离散度、模块赢家一致性。真值、正例 margin、corrected/introduced 或 panel 性能均不得成为输入。

设标准化证据为 `e_m(q,c)`，可靠性网络输出可精确为零的非负精度 `p_m(q)>=0`：

```text
w_m(q,c) = availability_m(q,c) * p_m(q) /
           sum_j availability_j(q,c) * p_j(q)

S(q,c) = sum_m w_m(q,c) * scale_m * e_m(q,c)
         - lambda * Var_w[scale_m * e_m(q,c)]
```

整个专家或单个候选可以弃权。缺失/域外证据以 `availability=0` 表示“没有观察”，不是反对候选。训练目标是 query-listwise likelihood；anti-collapse 默认关闭，避免为了保留模块数量而强迫有害专家参与。

## 2. 资产资格

| 资产 | 当前角色 |
|---|---|
| Noise V1 | enabled，共享 encoder 主资产 |
| ChemAware Stage-1 | enabled，必须保持独立 checkpoint 身份 |
| ChemAware Phase-A | provisional，开发比较，不进入最终 bundle |
| WSE | enabled，强基线与独立证据 |
| frozen P2b-on-Noise | enabled；不得与 raw P2b 三通道共现 |
| raw P2b 三通道 | audit-only derivative；只能用于预注册替代实验 |
| Noise RRF 1.7 | retired audit-only，不得进入生产 |
| Noise MSG | provisional；晋级前不得进入最终模型 |
| ChemAware V2 | development-only、候选稀疏；独立确认前不得进入最终模型 |
| BioAware B47 | conditional-unqualified；无真实事件上下文时 unavailable |

历史负结果决定证据资格。失败资产可以保留为审计和负对照，但不能污染生产分数。每个 family 必须做 leave-one-family-out 消融，并在同一候选轴上证明净价值才可晋级。

## 3. 已实现文件

- `tasks/grand_unified_evidence_core.py`：候选级 applicability、rank-logit、可零权融合、listwise loss、严格排名；
- `tasks/build_grand_unified_evidence_bundle.py`：GNPS development bundle 与 query/candidate ID 精确对齐；
- `tasks/grand_unified_components_v2.json`：证据资格 registry；
- `tasks/train_grand_unified_evidence_model.py`：公式分组开发训练/验证与模型冻结；拒绝 final test bundle；
- `tasks/seal_grand_unified_external_bundle.py`：把 Enveda component scores、冻结 panel、模型、registry 和 checksums 绑定；
- `tasks/evaluate_frozen_grand_unified_external.py`：独立的一次性外测与不可自动重试 opening ledger；
- `tasks/test_grand_unified_evidence_core.py`：候选缺失语义、精确零权、ID 对齐、资产资格和外测边界合同测试；
- `tasks/run_grand_unified_all_modules.sbatch`：只训练 development bundle 的 GPU 作业。

## 4. 开发运行

装配器只生成 GNPS development bundle，写入 `evaluation_role=development`、`truth_status=consumed`、`allow_final_claim=0`。额外数组必须带完全一致的 `query_ids` 和 `candidate_ids`；未提供模块保持 missing/unavailable，不能伪造成零分。

```bash
python tasks/build_grand_unified_evidence_bundle.py \
  --panel identity_disjoint \
  --panel-npz data/validation/GLM_gnps_identity_panel_reconstruction/panel_identity_disjoint.npz \
  --base-scores <frozen_method_scores.npz> \
  --extra chemaware_stage1_encoder=<aligned.npz>::scores_identity_disjoint \
  --extra p2b_noise_v1_frozen=<aligned.npz>::scores_identity_disjoint \
  --out <train_bundle.npz>

sbatch --export=ALL,TRAIN_BUNDLE=<train_bundle.npz> \
  tasks/run_grand_unified_all_modules.sbatch
```

开发训练结束只生成 `model.pt` 与 freeze `report.json`，并明确记录 `external_test_opened=false`。训练器的 `--test-bundle` 参数只用于 fail-closed 报错，不执行评测。

## 5. Enveda 一次性开启

Enveda 必须先由修订后的 score-blind/fail-closed 链重建；修订前启动的作业 `2354608` 不能自动满足新合同。GNPS 上唯一模型冻结后，才允许生成 component scores、由 sealer 绑定所有哈希，并运行独立 evaluator。Evaluator 在加载标签前以 exclusive-create 写 opening ledger；无论成功还是异常，账本都保留且禁止自动重试。主比较基线必须在命令中预先声明，不能在 Enveda 上按最高 MRR 事后选择。

## 6. 当前真正缺口

缺口不是继续把所有历史模块塞进模型，而是：

1. 在同一 GNPS candidate graph 上补齐五个 enabled 资产的可信分数与 SHA-256；
2. 完成双面板 leave-one-family-out、最强单方法比较、corrected/introduced、near 风险与 formula-cluster paired CI；
3. 冻结唯一 seed、超参数、模块集合、abstention 与主比较基线；
4. 用新 fail-closed 链重建 Enveda，并且只打开一次。

ChemAware V2、MSG 与 B47 只有各自通过独立资格门后才能更新 registry；RRF 不再晋级。若冻结统一模型在 Enveda 为负，结论就是外推失败，不能回到同一外测集调参。
