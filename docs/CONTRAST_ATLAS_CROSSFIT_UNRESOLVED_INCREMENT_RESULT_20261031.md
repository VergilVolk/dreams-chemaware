# Contrast Atlas：已知实体交叉留出下的 unresolved 响应增量（2026-10-31）

## 结论

本轮从现有 **15 个研究来源、51 个研究级 contrasts、429 个唯一 full InChIKey** 的输入池出发，完成系统级交叉留出实验。通过预设支持门后，正式终点覆盖 **8 个研究、41 个 study×contrast 单元、59 个 held-out full InChIKey 和 17 个 source dyads**。429 个输入 InChIKey 按固定 SHA-256 规则拆成互不重叠的两半：一半形成 train-known 响应分数，另一半只作为 held-out-known 经验真值；随后交换两半重复一次。

在当前 WSE unresolved reciprocal edges 上，加入 unresolved 响应没有通过内部增量门：

| 终点 | 结果 |
|---|---:|
| 输入池来源 / contrasts / full InChIKeys | 15 / 51 / 429 |
| 终点 eligible studies / study×contrast units / held-out keys | 8 / 41 / 59 |
| feature-side fold-swapped rows（读取 truth availability 前） | 943 |
| 合格 base contrast-pair rows | 401 |
| fold-swap 后 rows | 802 |
| 有多响应轴的研究 | 8 |
| source dyads | 17 |
| target-contrast 两两比较 | 3,072 |
| train-known micro / dyad-macro concordance | 0.5241 / 0.5003 |
| unresolved-only micro / dyad-macro concordance | 0.4678 / 0.5390 |
| 固定 known+unresolved micro / dyad-macro concordance | 0.4977 / 0.5051 |
| 固定组合相对 train-known 的 dyad-macro 增量 | **+0.00476** |
| dyad bootstrap 95% interval | **[-0.03208, +0.04427]** |
| dyad exact sign-flip 单侧 p | **0.40970** |
| leave-one-study mean delta 范围 | **[-0.01872, +0.02235]** |

允许的结论是：

> 在 development/consumed 的 15-source、51-contrast 响应集合中，使用当前 WSE unresolved reciprocal edges，未解析实体响应没有在实体不相交的已知响应交叉留出中，为 train-known 提供稳健的 contrast-specific 排序增量。

这否定的是当前 WSE endpoint 的 unresolved 增量，不是否定 Contrast Atlas 的总科学问题，也不是 Full23 或最终统一算法的结果。

## 实验合同

1. 从 common-policy library-supported 记录中，每个 `study × full InChIKey` 只保留一个不读取响应的代表离子。
2. 固定拆分：`SHA256(split_salt + "|" + full_InChIKey)` 的前 64 bit 对 2 取模。
3. 输入池的 fold 0 / fold 1 分别包含 218 / 211 个唯一 full InChIKey；正式终点中为 27 / 32，两半无实体重叠。
4. 两次 fold swap 分别用一半形成 train-known，另一半形成 held-out-known truth。
5. 只保留跨 provisional parent project、每个 known fold 至少 3 个共享 InChIKey、至少 3 条 unresolved reciprocal edges，且两端研究各至少有两个 contrast 的记录。
6. 在每个 `truth fold × source pair` 内，对 train-known、held-out truth 和 unresolved 分别去掉 query-axis 与 candidate-axis 主效应：

   `interaction = score - query-axis mean - candidate-axis mean + source-pair mean`

7. 终点只在固定 query contrast 和固定 target study 内比较 target contrasts，因此来源身份或候选数量本身不能直接决定排序。
8. train-known 与 unresolved 的双中心化及 RMS 归一化先在不读取 held-out support/value 的 943-row feature-side universe 上冻结；随后才按 held-out support 限定评价行。两者等权相加，没有拟合 alpha，也不读取 held-out response value 调权。

## 为什么它检验 unresolved 的条件增量

- **train-known**：第一半 exact-InChIKey 实体，构成可用已知响应基线。
- **held-out-known truth**：另一半 exact-InChIKey 实体，只参与评价。
- **unresolved-only**：两端均为 common-policy unresolved 的 reciprocal WSE edges，与两套 known 实体键无交集。

因此，本实验问的是：在已有 train-known 响应之外，加入 unresolved 是否能更准确恢复一组从未参与 train-known 计算的已知响应排序，而不是 unresolved 是否仅因边更多而产生更多显著调用。

实体不相交并不等于外部独立。三套响应仍来自相同研究样本和实验标签，source dyads 也共享研究；bootstrap、sign-flip 和留一分析都只是开发期敏感性证据。

## 结果解释

unresolved-only 的 dyad-macro concordance 为 0.5390，但 micro concordance 为 0.4678。这种方向分裂说明结果受 dyad 大小和研究组成影响，不能把较高的 macro 点估计单独写成增量。

最重要的配对比较是固定组合减 train-known：平均只增加 0.00476，区间跨零，17 个 dyad 的全部 `2^17=131,072` 种符号配置给出 exact sign-flip p=0.40970；删除任一研究或任一 dyad 后，平均 delta 都可能转负。三个预设通过条件均未满足：

1. dyad-bootstrap interval 下界没有大于 0；
2. sign-flip 单侧 p 没有小于等于 0.05；
3. leave-one-study 后增量并非始终为正。

train-known 本身的 dyad-macro concordance 也只有 0.5003，说明跨研究 held-out exact-InChIKey 响应不是天然稳定的机制真值。它是合格的经验响应锚点，但不能被表述成普遍生物学真值。

## 与统一算法的直接接口

本轮冻结的是研究级评价终点，而不是 WSE 的边。Official、Noise、Full23 或最终统一算法必须：

1. 在 294,236 条完整、模型无关候选轴上独立评分；
2. 各自形成 directional 和 reciprocal decisions；
3. 在相同 known split、response operator、eligibility 和分母上重算本终点；
4. 比较其固定 known+unresolved 组合相对 train-known 的 dyad delta。

禁止只给 WSE 已选边替换一列分数；那只能评价 WSE 固定拓扑上的重加权，不能证明算法增量。

## 证据边界

- 全部研究、contrasts 和结果均为 `development/consumed`，不是封存外部验证。
- GNPS full-InChIKey 库匹配不是 MSI level-1 身份。
- common identity policy 与代表离子选择使用 WSE，可能偏向 WSE。
- unresolved 可能包括已知分子的其他加合物、源内碎片或同一化学家族成员，不等于新分子。
- dyads 共享研究；dyad bootstrap 和 sign-flip 不是独立研究置信区间。
- 当前结果没有证明 unknown 信息增量、Full23 增量、生物学发现或机制。

## 可复现产物

- 实现：`tasks/run_contrast_atlas_crossfit_unresolved_increment.py`
- 测试：`tests/test_contrast_atlas_crossfit_unresolved_increment.py`
- 正式输出：`data/validation/contrast_atlas_crossfit_unresolved_increment_v10_20261031/`

运行命令：

```powershell
python -u tasks/run_contrast_atlas_crossfit_unresolved_increment.py `
  --collection data/validation/contrast_atlas_open_response_collection_v10_frozenbase_msv000080918_v7_20261031 `
  --library-evidence data/validation/contrast_atlas_global_biological_adjudication_v1_20261030/entity_library_evidence.tsv.gz `
  --unresolved-score-ledger data/validation/contrast_atlas_known_anchor_unresolved_test_v2_20261031/known_anchor_unresolved_scores.tsv.gz `
  --output data/validation/contrast_atlas_crossfit_unresolved_increment_v10_20261031 `
  --split-salt "contrast-atlas-known-crossfit-v1|20261031" `
  --min-known-per-fold 3 --min-unresolved-edges 3 `
  --bootstrap 20000 --sign-flips 99999 --seed 20261031
```

五项针对性测试通过；同参数重复运行的输出校验和完全一致。
