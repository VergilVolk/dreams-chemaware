# DreaMS 噪声路线可视化画廊（2026-08-29）

> 阅读边界：S3A/A4/E0 图描述冻结动作或教师空间；只有共享 encoder 的多折多 seed 结果才是模型权重性能。

## 1. S3A 预注册错误机制动作矩阵

![S3A action matrix](D:/DreaMS/data/validation/g8r_noise_v3_s3a_extended_matrix/s3a_action_matrix.png)

## 2. 相同 query 上的六步动作轨迹

![S3A complete trajectories](D:/DreaMS/data/validation/g8r_noise_v3_s3a_extended_matrix/s3a_complete_case_trajectories.png)

## 3. 修正与新增错误的结构去向

![S3A transition destinations](D:/DreaMS/data/validation/g8r_noise_v3_s3a_extended_matrix/s3a_transition_destinations.png)

## 4. 风险加权动作矩阵

![E0 risk-weighted matrix](D:/DreaMS/data/validation/g8r_noise_final_e0_unified_matrix_local_audit/e0_risk_weighted_action_matrix.png)

## 5. 修正—新增错误权衡

![E0 correction risk tradeoff](D:/DreaMS/data/validation/g8r_noise_final_e0_unified_matrix_local_audit/e0_correction_risk_tradeoff.png)

## 6. A4 剂量—收益—风险曲线

![E0 A4 dose response](D:/DreaMS/data/validation/g8r_noise_final_e0_unified_matrix_local_audit/e0_a4_dose_response.png)

## 7. 真实重复谱中的峰出现率景观

![E1 peak prevalence](D:/DreaMS/data/validation/g8r_noise_final_e1_empirical_smoke_v4/e1_peak_prevalence_landscape.png)

## 8. 采集条件下的强度和质量抖动

![E1 empirical jitter](D:/DreaMS/data/validation/g8r_noise_final_e1_empirical_smoke_v4/e1_empirical_jitter_by_condition.png)

## 9. 缺峰估计的可靠性与安全上限

![E1 pairwise dropout reliability](D:/DreaMS/data/validation/g8r_noise_final_e1_empirical_smoke_v4/e1_pairwise_dropout_reliability.png)

## 10. 化学规则证据在峰遮蔽下的行为

![Rule noise pilot](D:/DreaMS/data/validation/rule_noise_pilot/rule_noise_pilot.png)

## 11. DreaMS Top-1 错误空间

![Failure audit](D:/DreaMS/data/validation/e0_failure_audit/failure_audit.png)

## 12. 结构残差的机制总览

![Residual mechanism summary](D:/DreaMS/data/validation/dreams_structure_residual_atlas_large_v2/mechanism_summary.png)

## 13. 结构—embedding 残差图谱

![Paired residual atlas](D:/DreaMS/data/validation/dreams_structure_residual_atlas_large_v2/paired_common_structure_residual_atlas.png)

![Full residual atlas](D:/DreaMS/data/validation/dreams_structure_residual_atlas_large_v2/structure_residual_atlas.png)

## 14. 30 组人工病例

人工谱图与结构病例位于：

- `D:/DreaMS/data/validation/e0_failure_audit/manual_cases/case_01_spectra.png` 至 `case_30_spectra.png`
- `D:/DreaMS/data/validation/e0_failure_audit/manual_cases/case_01_structures.png` 至 `case_30_structures.png`

## 15. 当前读图结论

- 最稳健正动作：`candidate_gradient`。
- 可作为低风险补充：`role_confounder`。
- 明确淘汰区：大剂量 `role_shared`；它修正少、引入多，并且新增错误主要落入近邻/中等相似错误候选。
- 单纯加大剂量不是答案：可修错误数增加的同时，风险增长更快，动作精确性下降。
- 经验噪声必须按真实 replicate、仪器和 CE 条件校准；不能再使用统一随机删峰代替真实采集变化。
- 当前真实共享 embedding 成果为 +0.635 pp overall / +0.522 pp near；动作空间 3.35–4.93 pp 是下一阶段的可迁移上限，不是现成模型性能。

