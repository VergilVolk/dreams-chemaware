# Contrast Atlas 模型无关逐 pair 评分交接合同

## 目的

Atlas 生物端需要比较 WSE、Official、Noise、Full23 等方法在**同一个冻结候选宇宙**上的实际决策。固定候选宇宙，不等于固定 WSE 已经选出的 reciprocal 边。因此，本合同把算法交接点前移到完整 directed-pair 轴：各方法必须先独立评分完整 block，再形成自己的 block 决策和 reciprocal 边。

本合同不训练或改变任何算法，也不把模型排序分数解释成身份概率。它只验证交接数据是否完整、truth-blind、fail-closed 且可追溯。

## 冻结轴与分母

合同绑定：

- `data/validation/contrast_atlas_model_independent_identity_axis_v5_20261027/report.json`
- `directed_complete_candidate_pairs.tsv.gz`：291,330 条 directed pair；
- `query_target_source_blocks.tsv.gz`：768,054 个 query×target-source block；
- 其中 139,129 个非空 block 承载全部 291,330 条 pair；
- 其余 628,925 个 block 是没有物理化学候选的结构性弃权。

“完整 block 评分”指：交付全部 291,330 条 pair，并对全部 139,129 个非空 block 给出方法自己的决策。不得只交 Top-k、WSE 互惠边或算法认为容易的 block。零候选 block 仍保留在冻结轴分母中，但不要求算法伪造评分。

## 必交三张表

### `method_pair_scores.tsv.gz`

必须与冻结 pair 轴逐行严格对齐，保留：

- `directed_pair_id`、`query_block_id`、两个轴内 index；
- 原始 `raw_score`；
- `score_available`；
- 缺失时的 `unavailable_reason`。

`raw_score` 是方法实际用于排序的原始输出，禁止为了适配 Atlas 无条件截断到 `[0,1]`。它可以天然落在 `[0,1]`，但合同要求 `raw_score_clipped=false` 和 `raw_score_transformation=none`。排序分数不是校准概率，manifest 必须声明 `probability_interpretation=false`。

如最终方法分数缺失，该行的 `raw_score` 必须为空，不能用 WSE、零分或其他专家回填。内部模型若有预先冻结并明确属于该模型定义的 fallback，应由算法端先形成最终方法分数；生物端不会另行补分。

### `method_block_decisions.tsv.gz`

必须覆盖全部 139,129 个非空 block。决策只能来自该方法在完整 block 上的分数：

- 任一候选缺少最终分数：`abstain_incomplete_block_evidence`；
- 最优分数并列：`abstain_tied_top1`；
- 唯一 Top-1 但未达到方法自身冻结准入规则：`abstain_method_rule`；
- 方法回答：`selected_unique_top1`。

这允许不同模型有不同的合法回答与弃权，但不允许某模型沿用 WSE 的候选选择或 WSE 的 Top-1。

### `reciprocal_edges.tsv.gz`

reciprocal 边只在两个方向都由该方法选为唯一 Top-1 时成立。每条边必须同时保留：

- `left_to_right_directed_pair_id` 与原始方向分数；
- `right_to_left_directed_pair_id` 与原始方向分数。

不得只保存平均值、最小值或单方向分数后丢失方向信息。不同实体的化学关系不属于此身份轴合同，应走独立的 relation-track 合同。

## Truth-blind 边界

评分、block 决策和 reciprocal 选择阶段不得读取：

- response beta；
- response SE；
- phenotype；
- registered relation；
- identity truth。

这些字段也不得出现在三张交接表中。响应信息只能在方法自己的身份边完全冻结并通过 validator 以后，由 Atlas 下游读取。

## Manifest 最低要求

包内 `report.json` 使用 `contrast_atlas_method_pair_score_package_v1`，登记：

- 方法 ID、版本和工件 SHA-256；
- raw score 语义与方向；
- 未截断、非概率解释；
- truth-blinding；
- 缺失 fail-closed 且未用 WSE/外部方法补分；
- 模型专属完整 block 决策；
- 三张输出表的 SHA-256；
- 冻结轴三个 SHA-256。

精确机器合同见：

- `tasks/contrast_atlas_method_pair_score_contract_v1.json`
- `tasks/validate_contrast_atlas_method_pair_scores.py`

运行示例：

```bash
python tasks/validate_contrast_atlas_method_pair_scores.py \
  --package-dir data/validation/<method_complete_axis_package>
```

validator 的 `PASS` 仅表示 schema、完整性、隔离和决策一致性合格；它不证明身份准确、unknown 增量、研究级响应增量或生物学有效性。

## 旧接口的用途边界

`run_contrast_atlas_open_local_edge_enrichment.py::apply_algorithm_edge_scores` 会在 WSE 已经选出的固定 reciprocal 边上替换权重，并将输入截断到 `[0,1]`。该接口保留用于 **fixed-WSE-edge sensitivity analysis**：回答“在同一组 WSE 边上换权重会怎样”。

它不满足本合同，不能用于声称 Official、Noise 或 Full23 在完整候选轴上形成了自己的身份对应，也不能用于模型之间的正式 Atlas 应用增量比较。本轮不修改该旧逻辑，以免把历史敏感性结果与新的完整轴比较混为一谈。
