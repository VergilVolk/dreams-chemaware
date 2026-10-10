# 逐实体统一证据工作流实现（2026-10-10）

## 结论

仓库现在已有一个不训练融合头、不把不同证据强行相加的统一工作流。它的统一对象是实验实体及其候选证据记录，而不是由所有模块共同生成的单一 Top-1 分数。

实现入口：

- `tasks/build_unified_entity_evidence_ledger.py`
- `tasks/test_unified_entity_evidence_ledger.py`
- `tasks/run_unified_entity_evidence_ledger.sbatch`

GNPS Gold/Silver 在本项目中已被反复用于开发，运行结果必须标为 `development_consumed`。该作业不读取 Enveda-180。

## 模块职责

| 层 | 当前资产 | 账本中的作用 | 明确不做的事 |
|---|---|---|---|
| 谱图坐标 | Noise V1 | 为每个 query 保存稳定坐标键、候选分数和排名 | 不因另一模块出现而丢弃 Noise 表示 |
| 独立检索 | weighted spectral entropy | 当前 GNPS Top-1/open-set 最强基线，作为主候选检索列 | 不伪装成学习模型，也不与 Noise 分数相加 |
| 局部竞争 | P2b frozen on Noise V1 | 保存冻结 composite 的候选分数与排名 | 不重复加入三个 P2b 原始通道 |
| 碎裂化学 | ChemAware V2 action ledger | 保存 query 适用性、所选候选及相对 Noise 的 corrected/introduced | 不再执行 hard promotion，不把缺失写成零分 |
| 样本事件 | BioAware | 只接收质量门通过的真实 candidate-specific event | GNPS 无样本事件时整层为 unavailable |

## 两张核心表

每个面板产生两张 gzip CSV：

1. `candidate_evidence_<panel>.csv.gz`：一行对应一个 query-candidate molecule。只保存部署可见的 Noise/WSE/P2b 原始分数与排名、ChemAware applicability/selection 和 BioAware event applicability，不含真值身份或正确性。
2. `entity_evidence_<panel>.csv.gz`：一行对应一个实验实体。保存固定实体 ID、Noise 坐标键、三个谱学模块各自的 Top-1、ChemAware 候选、BioAware 可用性、结构声明状态、输出层级和不能升级结构身份的原因。
3. `evaluation_truth_<panel>.csv.gz`：单独保存已消费开发面板的 query identity/formula、near 标记和 candidate truth。它只用于评测，不能回流到前两张部署账本。

评价真值只进入隔离的 `evaluation_truth` 文件和 `report.json` 开发指标，不参与实体输出层级。没有 authentic-standard/正交确认时，输出为 `ranked_structure_hypothesis`，不会因为三个谱学模块一致就自动升级为可信结构。

## 同分母消融

`report.json` 对 Noise、WSE 和冻结 P2b 在完全相同的 query/candidate 轴上分别报告 R@1/2/3/5/10/20、MRR、mean/median rank。这里的“消融”是独立模块能力表，不是重新训练一个融合器再删除输入。

ChemAware 单独报告：

- applicable queries 与覆盖率；
- 改变 Noise Top-1 的次数；
- corrected、introduced、`corrected - 2*introduced`；
- 部署语义固定为 `evidence_only_no_score_override`。

BioAware 单独报告通过质量门的事件覆盖率。没有事件文件或没有合格事件时，状态为 unavailable，并写出原因。

## known-only 边界

可选 entity manifest 用 `reference_status` 明确 known/unknown。若同时存在两类实体，脚本只物化分区，并要求独立生物学 endpoint 和留出条件；它不会从候选排名自行制造生物学增量。

GNPS 谱库面板全部是已知 library queries，因此其结果必须写为：

`not_evaluable_all_entities_known`

这不是失败，而是阻止把封闭谱库检索错误包装成 unknown-entity biological gain。真正的 known-only 对 known+unknown 增量必须在跨平台或独立扰动数据上运行，并以实体找回和表型复现为终点。

## 可选真实样本接口

`--entity-manifest` 接收以 `panel,query_index` 为键的 CSV，可携带 `entity_id`、`qc_pass`、`reference_status`、`orthogonal_structure_status`、precursor/RT、blank/QC、adduct 和 ion family。

`--bioaware-events` 接收以 `panel,query_index,candidate_id` 为键的事件 CSV。只有 `event_quality_pass=true` 的事件进入账本；事件 ID、类型、最高分及数量均保留。静态目录成员关系不能通过该接口冒充事件。

## 服务器运行

```bash
sbatch tasks/run_unified_entity_evidence_ledger.sbatch
```

默认复用作业 `2358497` 生成的 ChemAware action ledger。若以后有真实实体清单和事件输入：

```bash
ENTITY_MANIFEST=/path/entities.csv.gz \
BIOAWARE_EVENTS=/path/events.csv.gz \
sbatch tasks/run_unified_entity_evidence_ledger.sbatch
```

该作业只装配现有输出并运行同分母统计，不重新编码谱图、不重新训练参数、不扫描阈值，也不访问 Enveda。
