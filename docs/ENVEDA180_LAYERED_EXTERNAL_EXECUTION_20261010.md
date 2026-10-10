# Enveda-180 分层算法外测执行状态（2026-10-10）

## 当前裁决

作业 `2354608` 只保留为构建审计。它没有载入后来登记为 required 的 MSnLib 排除源，且当时的合同错误地把仓库中不存在的 MassBank 全量 MGF 设为硬依赖，因此不能用于最终一次性模型外测。

正式执行仍有三个不可倒置的科学阶段，但前两个工程阶段已合并为一次 Slurm 提交：

1. fail-closed、无模型的 v2 面板重建；
2. 冻结算法的 score-blind 分数组件生成；
3. 独立的一次性 opening evaluator。

同一个作业先完成第一阶段，成功后才进入第二阶段；不自动执行第三阶段，因此不会因工程错误提前消耗 Enveda 外测资格。

## 冻结算法

`tasks/enveda180_layered_freeze_contract_v1.json` 在任何 Enveda 模型分数产生前固定：

- Noise V1：共享谱图坐标；
- weighted spectral entropy：主 Top-1 输出；
- frozen P2b-on-Noise：局部竞争与候选列表输出；
- ChemAware V2：只作化学解释证据，不自动改判；
- BioAware：谱库无真实样本事件，状态为 unavailable。

官方 DreaMS 与 Noise checkpoint 路径和 SHA-256 已固定。GNPS 5% FDR margin 阈值也预先写入合同，Enveda 不重新校准。

## 单次提交：重建面板并生成盲分数

```bash
git pull
sbatch tasks/run_prepare_gnps_enveda_benchmarks.sbatch
```

新目录固定为：

- `data/validation/enveda180_scoreblind_manifest_v2_20261010`
- `data/validation/enveda180_cross_condition_benchmark_v2_20261010`

六个仓库实际存在的已消费来源全部 required，任一缺失即失败。MassBank 已消费面板通过完整的 `ik14/formula` 元数据排除；仓库不存在的全量 MassBank MGF 不再作为虚假硬依赖，其谱图哈希未覆盖被明确保留为残余边界。输出目录已存在时拒绝覆盖。新 panel 额外保存 `query_adduct`，使正负离子评价不再被统一伪写为 `[M+H]+`。

面板构建成功且 report 显示全部 required exclusion source 为 loaded 后，同一作业自动进入 score-blind 阶段：

- 用现有官方 DreaMS 与冻结 Noise V1 checkpoint 各编码一次；
- 原样复用 WSE 和 frozen P2b 实现；
- 构建阶段只读取 query/candidate 指针，不读取 molecule label、正确身份或评价指标；
- 不运行 evaluator，不生成 Recall/AUC，不创建 opening ledger；
- 输出 `method_scores_unopened.npz` 和 `scoreblind_report.json`。

只有 score-blind 作业完成、文件回传并验证后，才编写和提交一次性 opening 命令。打开后的任何结果都不得用于改模型、换主方法、调 P2b、重校阈值或复活 ChemAware hard promotion。
