# 噪声微调 P/N/S 共享编码器首轮协议（2026-08-27）

## 目标与边界

本阶段只训练一个共享 DreaMS 编码器。推理时输入原始干净谱图，输出新的 embedding；query 与 reference 使用同一套权重。

- P 流：真实、同分子、同加合物、跨采集条件的正谱对。
- N 流：已经在 5-fold × 3-seed 中稳定验证的 `curriculum/all/views4` 定向峰噪声策略。
- S 流：官方 embedding 保持与官方正确排序 margin floor。
- 禁止：P2b、下游重排分数、特权教师、结果标签选动作、P3 查询身份。

## P 流样本定义

从冻结 strict-10ppm 候选图的正分子候选谱中枚举。必须同时满足：

1. query 与 positive 为不同谱图行；
2. 两者均为 real/train；
3. IK14 相同且 adduct 相同；
4. 仪器不同，或碰撞能均可用且差值至少 10；
5. 身份在 P3 允许训练清单中；
6. 不写入 corrected/introduced/teacher/P2b 等结果字段。

公式在任何训练之前用固定 seed 分为 5 折，训练折与评价折按 formula 完全隔离。

## 固定的 N 流基线

- policy=`curriculum`
- action_scope=`all`
- views_per_identity=4
- unfreeze=最后 1 个 Transformer block + 官方 projection head
- backbone LR=`2e-6`; head LR=`1e-5`
- epochs=4; grad clip=1.0; dropout off
- 正样本候选 4；负分子 8

该配方此前 15 次 held-formula 运行全部为正，3-seed 聚合约为 overall +0.635 pp、near +0.522 pp。P/N 扫描不得改变上述参数。

## 首轮唯一变量

P 流权重固定扫描 `{0.125, 0.25, 0.5, 1.0}`。P 流独立随机数流，不能改变 N/S 的采样序列，因此与固定 N-only 结果构成配对比较。

## 通过门

候选配置必须同时满足：

1. 原 E4-A 全部门通过；
2. overall Recall@1 增益不低于完全相同 seed/fold 的固定 N-only；
3. risk-net=`corrected - 2*introduced` 不低于 N-only；
4. held cross-condition pair cosine 提高；
5. held cross-condition query Recall@1 不下降。

符合者按 risk-net、overall 增益、cross-condition cosine 增益依次选择。首轮只是单 held-formula 开发折；选中配方必须再做 5 folds × 3 seeds，之后才能进入封存测试。

## 产物

- `tasks/build_noise_final_pn_positive_manifest.py`
- `tasks/train_noise_final_e4a_direct_augmentation.py`（P/N/S 扩展，N-only 默认行为保持）
- `tasks/test_noise_final_pn_shared_encoder.py`
- `tasks/run_noise_final_pn_positive_manifest.sbatch`
- `tasks/run_noise_final_pn_weight_scan.sbatch`
- `tasks/summarize_noise_final_pn_weight_scan.py`
- `tasks/run_noise_final_pn_weight_scan_summary.sbatch`

本地已通过 Python 编译、静态契约检查、非递增 HDF5 索引回归测试，以及 P-arm loss 的有限梯度单元测试。正式数据仅在服务器，故正式 cardinality 与性能必须由服务器作业给出。

## 五个百分点目标的新增硬门

23,876 个查询上提高 5 个百分点需要至少净修正 1,194 个查询；官方错误总数为 1,805。历史 3.853 pp 是逐查询观察动作结果后的 no-op oracle，不是权重成绩；P2b 的 3.91 pp 是下游排序成绩，也不是共享 embedding。

因此新增 `audit_noise_final_pn_fivepoint_headroom.py`：对 N-arm 训练期动作教师可恢复错误与 C1 支持谱互斥 P-arm 可恢复错误取独立 query 并集，同时逐 formula fold 检查是否覆盖 5 pp。该审计仍是 outcome-aware 上限，只能作为进入高容量训练的必要条件。

若通过，容量扫描只改变三个预注册维度：

1. hardest negative 从官方 Top-8 扩为候选图中至多 20 个负分子；
2. baseline-error identity 每轮额外采样 0/2/4 个 N-arm 视图；
3. 解冻最后 1/2/3 个 Transformer blocks，并随深度降低 backbone 学习率。

任何配置仍必须超过完全相同 P 权重的一层 comparator，且 risk-net 不降低、introduced 不扩大、P/N/S 原始门全部通过。若五点 headroom 不通过，容量扫描 fail-closed，不允许用更大学习率掩盖监督空间不足。
