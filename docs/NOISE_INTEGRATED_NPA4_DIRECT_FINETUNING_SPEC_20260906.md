# Noise N+P+A4 统一直接微调冻结规格（2026-09-06）

## 1. 唯一目标与当前状态

本规格只服务于 noise 路线：把成熟噪声动作揭示的真实同分子候选边界直接写入 clean-input DreaMS shared encoder。它不训练下游专家，不引入 bio-aware 或 chem-aware 目标，也不回归 teacher embedding、teacher margin 或 teacher residual。

最终目标是在同一冻结 MassSpecGym query/candidate graph 上，相对精确 E8 初始化获得严格、可重复的 4--5 pp 全面提升。该数值是 promotion gate，不是预先保证；只有 outer-formula-held、clean-input、paired formula-cluster 结果满足本规格第 10 节，才允许声称实现。

截至本规格冻结时：

- N、P、A4 三类动作已经在同一 E8 几何中接通；
- 本地开发账本包含 5,804 条路由动作、1,272 条入训动作、92 个查询；
- 其中 848 条纠错动作和 424 条风险动作，纠错/风险各覆盖 69 个查询；
- 32-query N+P+A4 直接微调臂正在运行；
- 4--5 pp 的正式 encoder 结果尚未产生，不得提前写成已完成。

## 2. 冻结数据与初始化

### 2.1 Candidate graph

固定目录：

`data/validation/noise_corrected_candidate_graph_v1_20260906`

固定统计：

- 83,619 queries；
- 9,854 IK14 identities；
- 6,220 formulas；
- 392,229 candidate molecules；
- 6,220,661 candidate edges；
- 87,848 spectra；
- fold 0 outer held：18,333 queries；
- fold 0 outer train：65,286 queries。

任何训练或比较必须使用该 graph 的相同 query、candidate molecule aggregation、positive 定义和 exclusion。禁止跨任务拼接 baseline。

### 2.2 初始化

唯一初始化为成熟 E8 shared encoder：

`data/validation/g8r_noise_final_e8_direct_transfer/curriculum_all_views4_blocks1_blr_2e-06_hlr_1e-05_e8_baseline_symmetric_shared/seed_20260830/fold_0/final_shared_encoder.pt`

SHA256：

`8047b3f58c6808c86b320ac94b9e610610384040fa3a03e8d70550eb438a24af`

Official checkpoint 只作为第二基线：

`data/e1/official_embedding_slim.pt`

SHA256：

`8928f908606c0bd652c5a4107d3c35102f660622958c225a1f625abe4b1ba245`

主比较必须是 candidate vs exact initial E8；official vs candidate 是补充比较。

## 3. 动作资产：三类必须同时进入

### 3.1 成熟 N 多步动作

必须保留全部 9 个成熟 N cells，以及每个 query 的完整多步路径：

- `candidate_gradient`：最多 6 步，发布第 3--6 步；
- `role_confounder`：最多 5 步，发布第 1--5 步。

动作在当前 E8 几何中逐步重放，每一步重新计算 candidate context。不得把多步路径压成一个 action，不得因 matched control 缺失而删除 target action；缺失控制时使用 detached clean fallback 并明确记录。

### 3.2 成熟 P 全 registry

必须使用全部 60 个冻结 recipe/reference cells：

- E10B：19 cells；
- E11：16 cells；
- E12B：25 cells。

包含的成熟 family 至少为：

- consensus projection；
- recurrent union mix；
- matched intensity transport；
- recurrent peak graft；
- balanced peak exchange；
- transport then union；
- consensus then union。

reference policy 必须保留 top3、farthest3、maxmin6、condition6、maxmin12 的差异，不得先平均成单一方向。

### 3.3 A4 全提案多剂量动作

唯一允许读取的 A4 来源为：

`data/validation/g8r_noise_v3_a4_exact_peak_scan/exact_peak_scan.h5`

只读取 outcome-free proposal metadata：真实 token、candidate role、gradient rank、policy eligibility 和冻结剂量。不得读取任何 `result_*` outcome，不得使用后来 one-action teacher 的赢家选择。

每个 outer-train query 重放 Top-50 proposal，剂量固定为 0.25、0.50、0.75、1.00，并建立匹配峰控制。正式路由必须使用 `--max-queries 0 --maximum-gradient-rank 50 --formal`。

## 4. 当前几何中的动作路由

每条 N/P/A4 动作必须在同一个 E8 checkpoint、同一个 corrected graph 和同一个 molecule-max candidate boundary 中重新评分。历史 outcome 不能直接作为当前训练标签。

冻结路由语义：

- `corrective`：clean 为错误，action 相对 clean 和 matched direction control 都达到预注册优势；
- `harmful`：动作造成明确 margin 损害，只能进入风险保护分支；
- `robustness_only`：clean 正确且动作未造成超出 slack 的破坏，只用于审计/保护；
- `uncertain`：纠错权重严格为零。

默认阈值：

- paired advantage：0.01；
- harm margin：0.01；
- robustness slack：0.005。

不得降低阈值来扩大动作数，不得把 harmful/uncertain 当成正增强。

## 5. 多动作合并与剂量

统一账本由 `tasks/build_noise_corrected_routed_action_ledger.py` 生成。

冻结约束：

- 每 query 最多 16 条 corrective actions；
- 每 query 最多 8 条 harmful/risk actions；
- corrective 选择按 source/family round-robin，保留 N/P/A4 与 family 多样性；
- 未入训动作仍保存在完整 routing ledger 中；
- optimizer 的基本单位是完整 query action-set；
- 同一 query 的全部动作在同一步、同一候选语义中使用；
- action multiplicity 不增加 query 总训练剂量；
- 每个 epoch 每个 query 和每条选中 action 恰好曝光一次，不 recycle、不拆散。

## 6. 直接写入 clean encoder 的目标

主训练入口：

`tasks/train_noise_corrected_routed_direct.py`

默认主模式固定为：

`--corrective-objective-mode paired_margin_only`

对每条 corrective action，先在 detached E8/current action view、detached matched control view 与 clean view上计算同一 molecule-level positive-vs-hard-negative margin。可转移量为保守交集：

`min(relu(action_margin - clean_margin), relu(action_margin - control_margin))`

该量只决定 clean query 应获得的边界增量；梯度只落到 clean-input shared encoder。action 和 control 不作为 embedding 回归目标，不反传 teacher target，因此本方法不是 teacher embedding distillation。

harmful action 不进入 positive imitation。它只标记需要 clean/reference preservation 和 risk-floor 的查询。

## 7. 已修复的历史实现故障

以下修复是本规格的一部分，禁止回退：

1. 旧 E4 的好 action loss 更小、梯度压力反而低，不能把 action loss 简单替换 clean loss；现改为 action-to-clean molecule-margin transfer。
2. 旧流程 89%--93% step 发生 global clipping；现对分支先独立校准，再施加共同 pre-clip scale，默认目标总 norm 为 0.05。
3. corrective 与 risk 梯度使用 PCGrad 风险 veto；必须记录投影和裁剪后的 action signal retention。
4. corrective scale 允许小于 1，也允许在 raw transfer 梯度很小时有限放大，禁止旧 `max(1, requested_scale)`。
5. 旧 E14/E15 把每 query 多动作压成一个路径；现 query-complete、多来源、最多 16 条。
6. 旧 A4 `query_row` 为空导致 pandas groupby 静默丢弃；现以 query row、IK14、formula 三重一致映射，零映射即失败。
7. A4 全路由器的谱张量为 101 tokens 而角色向量曾错误使用 100；现按真实 tensor 长度构造并检查 token 边界。
8. N 路由器曾把 action IDs 保存为 object array；现强制 Unicode string array，所有训练加载保持 `allow_pickle=False`。
9. 同一 query 的 action 曾使用不同 hard-negative tensor 语义；现固定 query-level molecule references，语义漂移即失败。
10. clean duplicate 曾冒充 action control；现 clean-control 是同 provenance、同 query、同 schedule、同 calibration、同保护预算的独立训练臂。
11. 风险预算曾随机抽到大量 E8 原本错误的 query；有限预算现在优先 clean-rank=1 查询，以直接约束 introduced errors。

## 8. 信号传输硬门槛

训练必须输出以下量：

- raw corrective gradient norm；
- raw risk gradient norm；
- margin-transfer gradient norm；
- calibration scale；
- PCGrad projection retention；
- global clipping retention；
- final action signal retention。

硬门槛：

- action signal retention p10 >= 0.50；
- clip-event fraction <= 0.10；
- `legacy_90pct_loss_reproduced = false`；
- 任何 corrective/margin-transfer gradient 为零或非有限值立即 fail closed；
- 正式作业至少 32 个 calibration batches、128 个 query observations。

本地 4-query 接线测试已达到约 99.99% action retention、0% clip events；这只证明 90% 信号损耗的机械故障已修，不等价于 held 性能已提升。

## 9. 数据隔离与训练对照

- outer split 单位为 formula；fold 0 action routing 和训练不得消费 held formula；
- 可选 inner holdout 单位为 IK14 identity；本地开发固定 `inner-holdout-fold 0`；
- route outcome 只能来自 outer-train；
- 所有最终评估只输入 clean spectrum；
- P2b 禁止，P3 不消费；
- clean-control 与 routed-direct 使用相同 E8 初始化、训练 queries、risk queries、clean queries、epoch、batch、reference、learning rate、calibration seed 和评估 queries；唯一差异是是否施加 routed action-to-clean transfer。

## 10. 完整评估与 promotion gate

正式评估必须同时报告：

- Recall@1/2/3/5/10/20；
- MRR、mean rank、median rank；
- macro-query AUROC/AUPRC；
- micro-candidate AUROC/AUPRC；
- positive-vs-best-negative margin；
- Top1--Top2 gap；
- corrected、introduced、risk-net；
- near subset 的同组指标；
- formula-cluster paired confidence interval；
- MassSpecGym `[M+H]+`、10-ppm、IK14 pooled pairwise AUROC。

最后一项必须命名为 `MassSpecGym 10-ppm pooled pairwise AUROC`，不得称为 NIST20 论文 0.85 的精确复现。

候选 checkpoint 只有同时满足以下条件才可 promotion：

1. 相对 exact initial E8 的 outer-held Recall@1 提升至少 4.0 pp；
2. 相对等预算 clean-control 的 paired formula-cluster CI 下界严格大于 0；
3. relative official baseline 的增量单独报告，不与 E8 增量混算；
4. near-subset CI 严格为正；
5. MRR、Recall@2/3/5/10/20、macro/micro AUROC/AUPRC、margin、Top1--Top2 gap和 pairwise AUROC 不允许出现预注册的实质退化；
6. introduced 不得抵消 corrected，`risk_net_lambda2 = corrected - 2 * introduced` 必须为正；
7. 至少三个 seed 都满足主门槛，才进入 multifold；不得用单个幸运 seed 宣称 4--5 pp。

## 11. 本地开发运行冻结配置

当前动作臂：

- 32 corrective queries；
- 32 risk queries，优先 E8 clean-rank=1；
- 64 clean/protect queries；
- 5-fold inner identity split，fold 0 holdout；
- 2 epochs；
- 4 queries/batch；
- 8 calibration batches；
- one unfrozen encoder block；
- head LR `1e-5`，backbone LR `2e-6`；
- preclip target `0.05`；
- seed `20260906`。

本地结果只用于证明三源联合训练、信号传输、对照和 held-identity 行为，不用于宣称正式 4--5 pp。

## 12. 正式服务器执行边界

正式服务器作业必须：

- 使用两张 GPU：`#SBATCH --gpus=2`；GPU0 顺序完成 N 构建与 N 路由，GPU1 同时顺序完成 P60 与 A4 路由；统一 ledger 后，每个 seed 的 routed-direct 与等预算 clean-control 各占一张卡并行训练；
- 使用唯一 job-ID 根目录和原子 staging/rename；
- 先在 E8 几何中正式重放 full N、full 60-cell P、full A4 Top-50 x 4 doses；
- `--formal` 且所有 query limit 为 0；
- 生成 formal-authorized unified ledger 后才启动训练；
- 每个 seed 独立输出，禁止覆盖；
- routed-direct 与 clean-control 成对运行；
- 先完成 fold 0 三 seed，达到第 10 节才进入其余 folds；
- 任何 provenance hash、outer-fold、tensor ID、candidate reference、signal gate 或 metric schema 不一致立即停止。

旧入口 `tasks/run_noise_final_direct_boundary_v2_phase_a.sbatch` 不得提交；`tasks/run_noise_corrected_e4_np_fold0.sbatch` 是 fail-closed 遗留入口，也不得提交。正式提交必须使用依据本规格新生成并通过 preflight 的双卡入口。

## 13. 不允许再次发生的路线偏移

- 不把 noise 直接微调改写成 teacher embedding 蒸馏；
- 不因为一个窄 gate/teacher 失败就否定 N/P/A4 动作；
- 不扔掉成熟 E8 初始化；
- 不把多动作压成逐峰标量或单一 1024 维方向；
- 不用历史 outcome 选择 A4 赢家；
- 不用 pooled AUROC 替代 retrieval 指标，也不把不同任务的 baseline 混在一起；
- 不在本地小样本尚未胜过 clean-control 时声称已经获得 4--5 pp；
- 不为追求动作数量降低阈值、泄漏 held formulas 或消费 P3。
