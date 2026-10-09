# ChemAware 统一整合资产保真审计（2026-10-09）

> **修订状态（2026-10-09，晚于下文初始审计）：** 初始审计发现的资产混名、RRF 违规激活、query-only availability、候选轴未校验、强制正权与指标缺失均已在 registry v3、bundle、core 和 trainer 中修复；11 个合同测试通过。下文第 3 节的原始实现描述保留为审计证据，不能当作当前代码状态。当前仍未解决的整合 blocker 是 P2b 衍生通道与已冻结 composite 的重复计票；本次已将 raw P2b channels 改为 `audit_only_derivative`，冻结 composite 为唯一 enabled 代表。真实统一模型结果仍不存在。

## 裁决

当前 `grand_unified` 仍是尚未产生真实结果的联合模型原型，不能称为已完成的多模块算法整合。但初始版本的 ChemAware 资产身份、适用域、候选轴对齐和评测协议缺陷已经修复；当前不应提交正式训练的原因是仍未完成 P2b 家族去重后的真实数据 bundle、逐模块消融与冻结准入。

这不是 ChemAware 结果失效，而是整合层没有尊重已有证据边界。

## 必须保留的 ChemAware 资产身份

| 资产 | 已有结果 | 正确资格 | 统一模型中的正确角色 |
|---|---:|---|---|
| Stage-1 native-triplet encoder | role-3 1,929 queries，Recall@1 `+1.8144 pp`，57/22，formula-cluster CI `[+0.8155,+2.8703] pp` | 已封存的同源确认结果；GNPS 外推仅约 `+0.30/+0.21 pp` | 独立 encoder 专家，不得与 Phase-A 合并 |
| Phase-A max-boundary encoder | role-2 1,975 queries，Recall@1 `+2.1266 pp`，54/12，CI `[+1.2761,+3.0303] pp` | 强开发结果；未完成同等级 role-3 独立确认；GNPS 外推约 `+0.23/+0.40 pp` | Stage-1 的备选/消融专家，不是默认“最佳已确认” encoder |
| V2 multi-null symmetric residual reranker | role-3 1,929 queries，Recall@1 `+3.9399 pp`，93/17，CI `[+2.8191,+5.1921] pp`；相对 truth-blind rotated null `+3.3178 pp` | 最强候选侧开发信号；outer 已消耗且没有新独立确认 | 稀疏、候选条件化的化学残差专家；必须带逐候选适用域，不能当成全局 dense score |

## 初始实现对不起这些资产的具体位置（已修复，保留作审计记录）

1. `grand_unified_components_v2.json` 只有一个含糊的 `chemaware_encoder`，Stage-1 与 Phase-A 被合并，无法做资产级消融、选择或资格声明。
2. `run_GLM_chemaware_gnps_scores.sbatch` 默认只生成 Phase-A 分数，却命名为通用 `chemaware_encoder`；已封存的 Stage-1 没有作为独立输入进入当前统一模型。
3. V2 重排器被登记为普通 `active` 模块，但当前 bundle 的 availability 只有 query×module；它不能表达“本 query 中仅部分候选具有合格化学证据”。零分与无证据会被混淆。
4. bundle 对额外模块只检查数组长度，不校验 query ID、candidate ID、候选顺序和哈希。候选轴即使错位也可能静默通过。
5. 旧 registry 明确将 `noise_rrf_1_7` 标为 `retired_audit_only`（Stage-1/GNPS confirmation failed）；新 registry 无新证据却重新标为 `active`。
6. 联合模型强制所有可用模块严格正权重，并用 anti-collapse loss 阻止权重归零。这会强迫已证伪、域外或局部有害专家污染最终排序；“每个模块都参与”不是科学约束。
7. 当前训练器只报告 Recall@1/5/10、MRR 和 near Recall@1；没有官方基线差值、corrected/introduced、formula-cluster CI、AUROC、模块消融、多 seed 或安全子集分析。
8. 当前只有六个小型合成合同测试；没有真实候选轴对齐测试、稀疏候选 availability 测试、已退休模块隔离测试、ChemAware 三资产身份测试或外部评测回放测试。
9. 仓库中没有 `grand_unified` 的真实运行产物。因此它目前是设计提案，不是已经优于任何单模块的算法结果。

## 数据集边界

- MassSpecGym/MassBank 同源角色面板：可做资产复现、课程构造和机制消融，不能再次包装为全新外部确认。
- GNPS Gold/Silver：已经被反复查看和用于模型选择，只能作为消耗过的开发/压力测试面板。它已经证明 ChemAware encoder 的 `+1.8/+2.13 pp` 不会原幅度外推。
- Enveda-180：可以成为新的终局外部面板，但必须先完成 score-blind 去重、候选构造、身份/结构隔离、跨条件设计和 manifest 哈希冻结；模型清单与权重冻结后只能揭盲一次。
- BioAware 样本上下文不能在标准谱库中伪造；没有真实事件输入时必须 unavailable，而不是补零后进入融合。

## 当前剩余修复顺序

1. P2b raw channels 与 `p2b_noise_v1_frozen` 永不在同一 joint model 共现；raw channels 只可在单独、预注册的 composite replacement 实验中使用。
2. 生成并审计真实 GNPS candidate-keyed bundle，显式给出每个模块的可用率、来源 SHA-256 与缺失语义。
3. 在 GNPS development/consumed 上完成多 seed、leave-one-family-out 消融和最强单方法对照；ChemAware V2 仍只可作为逐候选 residual。
4. 报告 Recall@1/3/5/10/20/50、MRR、macro/micro/pooled AUROC、corrected/introduced/risk-net、formula-cluster CI、identity/formula/near 子集和 open-set FDR-coverage。
5. 统一系统必须分别对照 official、WSE、Noise V1、P2b、Stage-1 ChemAware、Phase-A ChemAware、V2 reranker 与简单预注册融合；若不能胜最强单模块，不得进入 Enveda。
6. 完成冻结账本后，才允许一次性提交 Enveda 外部评测。

## 对现阶段的准确表述

ChemAware 已经证明两件不同的事：化学选择的困难三元组可以改善共享谱图 embedding；候选结构已知时，化学残差重排具有更大的开发集增益。尚未证明的是：这些增益能在新分布上保持原幅度，或把所有模块放入一个强制共用的神经融合器就会进一步提升。当前任务应是保真整合与独立验证，不是把历史资产重新混成一个无法解释的总分。
