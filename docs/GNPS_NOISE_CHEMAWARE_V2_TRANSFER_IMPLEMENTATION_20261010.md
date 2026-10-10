# Noise V1 × ChemAware V2 冻结动作迁移实现（2026-10-10）

## 1. 目的与资格

本实现只回答一个已经足够具体的问题：历史冻结的 ChemAware V2 候选动作，能否在不改变
V2 决策几何、不重新训练和不选择阈值的条件下，为冻结 Noise V1 候选分数提供净增益。

评测数据固定为已经被多轮开发消费的 GNPS Gold/Silver 10-ppm 双面板：

| 面板 | Queries | Candidate molecules | Directed spectrum pairs | 资格 | 泄漏状态 |
|---|---:|---:|---:|---|---|
| identity-disjoint | 10,995 | 87,518 | 175,171 | development / consumed | 已用于开发，不是外部确认 |
| formula-disjoint | 5,261 | 24,401 | 47,724 | development / consumed | 已用于开发，不是外部确认 |

本作业不读取任何其他开发数据集，也不访问 Enveda-180。Enveda-180 继续保留给全部模型、
策略和报告规则冻结后的单次最终外测。

## 2. V2 的准确算法语义

V2 不是候选分子结构模型，也不是 dense candidate score。它在历史 official DreaMS 几何中，
结合 query/reference 谱图的 fragment-mass 与 neutral-loss rule kernel，输出：

1. abstain；或
2. 一个需要提升的候选槽位。

历史 V2 参数保持不变：

- `contrast_representation = symmetric_summary`；
- 三个 content-permuted null；
- `dose = 2.0`；
- `threshold = 0.3565609484787261`；
- selection 结果 78 corrected / 17 introduced；
- held-inner 结果 93 corrected / 17 introduced。

轻量仓库只保留了 V2 结果账本，没有保留 nuisance/residual estimator 的序列化参数；账本本身
不能对新 GNPS query 推理。因此作业优先复用已有的 `truthblind_policy.joblib`；若它缺失，则在
同一作业内按历史固定算法恢复一次序列化 bundle，并在打开 GNPS 标签之前冻结。恢复合同校验
三套 null、symmetric summary、deployment-safe selection、固定 dose、1,978/1,929 query 分母、
全部机制门及 corrected > 2×introduced。旧 sklearn/runtime 下得到的具体 threshold 和纠错计数
不作为跨环境字节级条件；这些实际值必须随运行报告完整保存，不能冒充旧点估计。

## 3. Noise 迁移规则

V2 selector 永远使用 official DreaMS embedding 和历史 rule-kernel 特征。Noise V1 只作为
deployment score base，不进入 V2 特征计算：

1. V2 abstain：保留该 query 的全部 Noise pair scores，逐值不变；
2. V2 选择的候选已经是 Noise Top-1：同样保留全部 Noise pair scores，逐值不变；
3. V2 选择的候选不是 Noise Top-1：只将该候选中最高的 reference-pair score 提升到略高于
   当前 Noise 最大值，使该候选成为 Top-1；
4. 不修改其他候选顺序，不拟合融合权重。

这个定义隔离了两个问题：V2 仍按经过开发的输入分布作决策；实验只检验该冻结动作能否跨越
更强的 Noise 基线。直接用 Noise embedding 替换 V2 的 official embedding 会改变历史阈值与
utility 的含义，因此被明确禁止。

## 4. 输出与评测

主 pair-score cache 为：

`noise_v1_chemaware_v2_cache`

每个面板另存一个 truth-blind action ledger：

- `query_index`；
- `abstained`；
- `selected_candidate`；
- `official_top_candidate`；
- `deployment_top_candidate`；
- `output_top_candidate`；
- `changed_deployment_top1`。

汇总同时报告 selected、abstained、真正改变 Noise Top-1、以及原本已经与 Noise Top-1 一致的
query 数量。标签只由下游统一 evaluator 打开；selector 与 action ledger 不读取 identity、formula
或 correctness。

候选系统分别与 official DreaMS、Noise V1、WSE 和 P2b-on-Noise 比较。统一 evaluator 输出：

- Recall@1/2/3/5/10/20、MRR、mean/median rank；
- macro-query AUROC/AUPRC；
- micro-candidate AUROC/AUPRC；
- 10-ppm pooled pairwise AUROC/AUPRC；
- near-structure 全套检索指标；
- margin 与 signed Top1-Top2 gap；
- corrected、introduced、risk-net；
- 10,000 次 formula-cluster paired bootstrap CI。

## 5. 运行入口

```bash
sbatch tasks/run_gnps_noise_chemaware_v2_transfer.sbatch
```

如冻结策略位于其他目录，只允许显式指向已有 bundle：

```bash
CHEMAWARE_V2_POLICY_DIR=/absolute/path/to/frozen_v2 \
  sbatch tasks/run_gnps_noise_chemaware_v2_transfer.sbatch
```

输出目录：

`data/validation/gnps_noise_chemaware_v2_transfer_run_<jobid>`

最终汇总为 `final_report.json`。运行前合同测试入口：

```bash
python -u tasks/test_chemaware_v2_noise_transfer.py
```

## 6. 预先固定的解释边界

- GNPS 结果无论多好都只能作为 development / consumed 证据。
- 历史 V2 的 `+3.0839/+3.9399 pp` 不能与 Noise 的增益相加，也不能作为本轮预期结果。
- 本轮不搜索阈值、dose、null 数量、融合权重或候选子集。
- 若 Noise+V2 不能在两个面板上同时形成相对 Noise 的正向、风险可接受的增量，则停止这条
  迁移路线；不得复用同一 GNPS 结果继续调参。
- 只有在统一系统于 GNPS 开发域完成冻结后，才允许一次性进入 Enveda-180。

## 7. 实现文件

- `tasks/export_chemaware_v2_grand_fusion_scores.py`：支持把冻结 official-geometry 动作迁移到任意
  对齐的 GNPS pair-score cache，并输出完整动作账本。
- `tasks/run_gnps_noise_chemaware_v2_transfer.sbatch`：GNPS-only 单次运行入口。
- `tasks/test_chemaware_v2_noise_transfer.py`：动作改变、一致和 abstain 三类合同测试。
