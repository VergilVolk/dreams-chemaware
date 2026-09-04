# DreaMS 噪声微调路线：成果与可视化总账（2026-08-29）

## 1. 必须保留的成果

1. **真实共享 embedding 已得到稳定提升。** E4-A 在 5 个 formula fold × 3 个 seed 的开发图 OOF 中，Recall@1 平均提高 **+0.635 pp**，near 提高 **+0.522 pp**；15/15 个 fold 的公式簇置信区间下界均大于 0。该结果来自同一个共享 query/reference encoder 的新权重，不是 P2b、reranker 或后处理头。
2. **峰级动作空间存在更大的可迁移潜力。** S3A、A4、E10–E13 等冻结动作/教师审计证明，candidate-gradient、positive-guided consensus/transfer 与 no-op-aware 组合能覆盖远高于当前权重增益的错误空间。历史常用数字约为 **3.35–3.85 pp**，后续扩展动作空间的 held-development 上限接近 **4.93 pp**。
3. **上限与模型结果严格分开。** 3.35–4.93 pp 是逐 query 选择动作或 outcome-aware oracle/headroom，不是已训练权重；目前能够写成“共享 embedding 实际提升”的最强稳健数字仍是 **+0.635 pp overall / +0.522 pp near**。
4. **动作安全结构已经被重复识别。** `candidate_gradient` 是稳定正向主轴；`role_confounder` 是较小但低风险的补充；`role_shared` 尤其大剂量会产生灾难性新增错误；`role_unmatched` 仅有小幅、有限收益。
5. **E14 的目标不是寻找另一个后处理器。** 它冻结 60 个经过筛选的动作，使用外层 formula 隔离的特权教师，把更高比例的动作空间能力迁移进同一个共享 encoder；P2b 仍被禁止进入教师、特征和损失。这里的教师 checkpoint 与动作记录都排除学生 held fold，但动作是在 outer-train 内按结果挖掘的，因此不得误称为“每条训练查询的 OOF 预测”。

## 2. 核心可视化索引

### A. 动作矩阵与连续干预

- `data/validation/g8r_noise_v3_s3a_extended_matrix/s3a_action_matrix.png`
  - 每种动作、剂量和连续步数对应的 corrected、introduced、net correction、新错误覆盖。
  - 关键证据：candidate-gradient 的净修正随步骤增长；role-shared 大剂量净损失快速扩大。
- `data/validation/g8r_noise_v3_s3a_extended_matrix/s3a_complete_case_trajectories.png`
  - 在相同 six-step-complete query 上比较净修正轨迹及每步边际效应。
- `data/validation/g8r_noise_v3_s3a_extended_matrix/s3a_transition_destinations.png`
  - 错误修正和新增错误最终落入 identity / near / mid / far 的结构层级。

### B. 统一证据总账与风险边界

- `data/validation/g8r_noise_final_e0_unified_matrix_local_audit/e0_risk_weighted_action_matrix.png`
  - 使用 `corrected - 2 × introduced` 统一比较动作安全性。
- `data/validation/g8r_noise_final_e0_unified_matrix_local_audit/e0_correction_risk_tradeoff.png`
  - corrected–introduced 散点；明确提示动作 headroom 不是模型增益。
- `data/validation/g8r_noise_final_e0_unified_matrix_local_audit/e0_a4_dose_response.png`
  - 峰衰减剂量越大，可恢复错误增加，但 collateral risk 同时更快增加，且梯度与精确动作的一致性下降。

### C. 经验噪声分布与采集条件

- `data/validation/g8r_noise_final_e1_empirical_smoke_v4/e1_peak_prevalence_landscape.png`
  - 同一 identity 内峰出现率 × 共识强度的二维分布，是经验缺失噪声的取样基础。
- `data/validation/g8r_noise_final_e1_empirical_smoke_v4/e1_empirical_jitter_by_condition.png`
  - 不同仪器/CE关系下的强度抖动与质量抖动经验分位数。
- `data/validation/g8r_noise_final_e1_empirical_smoke_v4/e1_pairwise_dropout_reliability.png`
  - 可用于剂量估计的可靠 replicate pairs，以及不同采集条件下的缺峰比例和安全上限。

### D. 化学规则与噪声的接口试验

- `data/validation/rule_noise_pilot/rule_noise_pilot.png`
  - 3,486 条规则在强度比例峰遮蔽下的保留率，以及 true-vs-wrong 规则 Jaccard margin。
  - 这是 ChemAware 与噪声路线的接口证据，不是 ChemAware embedding 已成功的证明。

### E. 错误空间与人工案例（相邻证据）

- `data/validation/e0_failure_audit/failure_audit.png`
- `data/validation/e0_failure_audit/manual_cases/case_01_spectra.png` 至 `case_30_spectra.png`
- `data/validation/e0_failure_audit/manual_cases/case_01_structures.png` 至 `case_30_structures.png`
- `data/validation/dreams_structure_residual_atlas_large_v2/mechanism_summary.png`
- `data/validation/dreams_structure_residual_atlas_large_v2/paired_common_structure_residual_atlas.png`
- `data/validation/dreams_structure_residual_atlas_large_v2/structure_residual_atlas.png`

这些图用于解释噪声动作为什么可能有效、哪些结构区容易引入新错误；它们不单独构成微调性能结果。

## 3. 当前结论图景

| 证据层 | 已确认内容 | 不允许的夸大 |
|---|---|---|
| 冻结动作矩阵 | 某些峰动作可显著改变正确/错误候选的相对 margin；candidate-gradient 最稳定 | 不能称为新 embedding 或部署性能 |
| 经验噪声 E1 | 噪声剂量可从真实重复谱和采集条件估计 | 不能据此断言训练会改善检索 |
| 共享 encoder E4-A | 真实权重 OOF 平均 +0.635 pp overall、+0.522 pp near | 尚未等于 3.85–4.93 pp 动作上限，也未完成全新外部盲测 |
| E14 | 试图提高动作到共享权重的转移率 | 运行前没有性能承诺；必须按 formula-isolated cross-fit 结果裁决 |

## 4. E14 固定提交顺序

1. `sbatch tasks/run_noise_final_e14_teacher_build.sbatch`
2. 教师构建通过且 60 个动作完整复现后：`sbatch tasks/run_noise_final_e14_selected_transfer_pilot.sbatch`
3. 五折/多 seed 完成后：`sbatch tasks/run_noise_final_e14_selected_transfer_summary.sbatch`

任何阶段如果动作 replay、outer-formula 隔离、P3 overlap、clean preservation 或 corrected/introduced 对账失败，后续阶段必须停止。
