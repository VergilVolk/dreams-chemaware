# Contrast Atlas 真实负 shard 入库资格重放

日期：2026-10-13  
状态：正式的 `development/consumed` 负对照重放；不是新阳性结果

## 裁决

三个已有真实 shard 均在其原本失败的科学层级被统一入库控制器阻断：

| shard | 数据集与独立分母 | 失败层 | 冻结结果 | Atlas 裁决 |
|---|---|---|---|---|
| ChemProp cross-design v4 | MSV000094899 vs MSV000096724/MSV000093571；12 drugs、36 queries | strict entity mapping 到 biological response | mass 7/36；cosine 0.70 10/36，增量 CI 跨 0；WSE 7/36 | `FAIL_ENTITY_INCREMENT` |
| ST003325 | PR001844；12 clones、27 held-edit comparisons | within-study response object | 10/27，median margin -0.03970，clone permutation p=0.7126 | `FAIL_RESPONSE_OBJECT` |
| LCNEC | 263 discovery entities | strict entity coverage / public evidence | 6/263 有候选，coverage 2.28%；public certificate 0/263 assigned | `ABSTAIN_INSUFFICIENT_COMPARABLE_EVIDENCE` |

`shards_replayed=3`，`shards_admitted=0`，`all_historical_stops_preserved=true`。

这不是“方法失败了三个数据集”的合并结论。三个 shard 分别回答不同问题：ST003325 的 response object 本身未成立；ChemProp 的跨设计实体对应没有带来可靠响应恢复增量；LCNEC 的跨平台可比较覆盖不足。将三者压成一个平均准确率会破坏科学含义。

## 实现与工件

- 执行器：`tasks/audit_contrast_atlas_shard_admission.py`
- 结果：`data/validation/contrast_atlas_negative_shard_admission_20261013/report.json`
- 简表：`data/validation/contrast_atlas_negative_shard_admission_20261013/admission_ledger.csv`

执行器核查原始报告状态、数据角色、独立分母、关键计数、配对 CI、表型盲状态、显式 dustbin 及原停止门，并记录五个输入文件的 SHA-256。任何数值或角色漂移均直接报错。

## 下一门

必须先有一个真实、结构隐藏、基于独立生物学单位且通过预注册门的 within-study response object，才允许进入 shared-pattern 或 study-association 检索。目前唯一候选是 MSV000096214/TT7 raw truth-blind 重建；其作者处理表结果只属于 `development/consumed` 正控，不能代替 raw 结果。

截至本文件生成时，TT7 positive consensus 已完成；negative polarity 正在 gap filling，尚未生成 `negative/report.json`，因此没有提前打开 genotype effect stage。

