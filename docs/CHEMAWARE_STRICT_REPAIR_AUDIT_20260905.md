# ChemAware 严格修复审计（2026-09-05）

## 结论

截至本审计，**没有证据证明现有 ChemAware 化学先验提高了官方 DreaMS 共享 embedding 的检索性能**。当前可归因的化学增益为 **0**，因此不得启动新的全局 GPU 微调，也不得将任何单臂 checkpoint 标记为 PASS 或 release eligible。

此前出现的 **+0.4666 个百分点**来自所有 Phase-A 训练臂共同具有的普通继续训练：官方基线 90.4614%，八个训练臂均为 90.9279%。因为化学臂、无教师臂和对照臂完全相同，这个数不能归因于化学规则。

## 新增的锁定证据

### 严格观察通道产品空间

- 锁定集合：576 个 identity-distinct query、484 个分子式；此前两次屏幕看过的 780 个 fold-2 分子式全部排除。
- 官方 DreaMS：94.2708%。
- mass：94.6181%，点估计 +0.3472 个百分点，但只有 3 个分子式改善、1 个变差；分子式聚类 bootstrap 95% CI 为 **[-0.4132, +1.0331] 个百分点**，不能确认提升。
- 120 通道严格人工观察字典（均匀或学习权重）：93.7500%，相对官方 **-0.5208 个百分点**。
- 判定：`CHEMAWARE_PSD_OBSERVATION_FAIL`，不允许训练。

证据目录：`data/validation/chemaware_psd_observation_locked_confirmation_v1/`。

### ChemBERTa 跨模态结构残差

- 结构教师：12,453 个分子的 ChemBERTa 768 维向量；仅作训练标签。
- 拟合：fold 0 的 2,565 个身份、1,418 个分子式。
- 选择：fold 1 的 1,992 个身份、1,244 个分子式。
- 锁定确认：复用上述完全相同的 576 个 query。
- 官方 DreaMS：94.2708%。
- 真实结构教师：94.2708%，1 个修正、1 个引入，净增益 0。
- 身份乱序结构教师：94.2708%；未训练随机映射：94.2708%。
- 真实教师相对官方的分子式聚类 95% CI：**[-0.6198, +0.3099] 个百分点**。
- 判定：`CHEMAWARE_CROSSMODAL_STRUCTURE_FAIL`，`pass_to_gpu_pilot=false`。

证据目录：`data/validation/chemaware_crossmodal_structure_embedding_v3/`。

## 已修复的八类问题

1. **单臂结果冒充因果 PASS**：单臂只能是 `ARM_COMPLETE` 或 `ARM_SAFETY_FAIL`；checkpoint 永远保持 `causal_chemistry_pass=false`、`release_eligible=false`。
2. **没有干净输入可观测性门**：新增 formula-OOF 审计；当前真实结果事件不足、CI 下界为 0，`pass_to_gpu_training=false`。
3. **数据语义不清**：独立重建 137,830 条允许行，与目标集合完全一致；明确 `SIMULATION_CHALLENGE` 只作成员审计，不作谱图来源或筛选条件。
4. **候选监督被压成单个 margin**：改为每个 query-reference 残差的 molecule-balanced Huber 监督，并增加 shifted、row-permuted 等容量对照；旧 KL/PMT 只允许历史复现。
5. **规则库概念错误**：335 条旧记录中，当前 kernel 可读 316 条，但可执行的母体结构前提规则为 0；重构后只有 120 个正离子观察通道，215 条隔离，且明确不是机理规则库。
6. **注入架构未经低成本证伪就微调**：新增 PSD 观察空间和 ChemBERTa 跨模态产品空间审计；两条路线均已在锁定查询上判负并阻断 GPU。
7. **算力没有硬预算和提交门**：新 pilot 上限为 5 臂 × 1 GPU × 2 小时，数组并发 1，总上限 10 GPU·小时；训练身份最多 512、epochs 最多 2。授权有效期 24 小时，启动时重新验证全部证据哈希。授权检查同时存在于提交脚本、作业入口和 Python 训练程序内部；手工 `srun/python` 也不能绕过。
8. **开发结果与 outer 释放混用**：新增不可变 release freeze 与一次性 outer seal。只有多臂因果开发 PASS 才能冻结；outer 失败后同一候选不得重试。

## GPU 入口状态

- 唯一保留的训练入口：`tasks/run_chemaware_candidate_residual_arm.sbatch`。
- 唯一允许的提交方式：`tasks/submit_chemaware_candidate_residual_pilot.sh`，但当前由于可观测性和规则准入均未 PASS，实际会在 `sbatch` 前被阻断。
- 11 个旧训练入口已改为历史隔离入口：不申请 GPU、不展开数组、1 分钟内以退出码 64 终止。
- 纯汇总任务已取消 GPU 请求。
- 静态审计结果：`CHEMAWARE_GPU_ENTRYPOINTS_SAFE`；证据位于 `data/validation/chemaware_gpu_entrypoint_audit_v1/report.json`。

## 下一条科学路线

现有证据已经否决“按精确质量匹配的人工中性丢失字典”和“通用 ChemBERTa 均值向量的线性残差”。若继续，新的化学监督必须改变信息质量，而不是再调损失权重：优先考虑经过结构真实性验证的峰级子式/碎片子结构标签，或专门按结构相似性训练的分子教师；先通过 clean-spectrum formula-disjoint 解码门和匹配乱序对照，之后才允许 10 GPU·小时以内的小试。

在新标签通过这些 CPU 门之前，正确动作是 **不提交任何 ChemAware 训练作业**。
