# BioAware v2-0：实验特征图与两层证据对照

## 本轮目的

本轮不再把“一跳 Rhea 邻居”直接当成改写谱图排序的理由。先建立与表型完全隔离的实验数据层，再检验生化网络证据是否同时得到队列内部实验关系支持。

## 冻结边界

1. 特征图不读取候选结构、真值、肿瘤/癌旁、组织学、差异丰度或排序结果。
2. 所有 MS2 谱先按 10 ppm、20 s 映射到 MS1 feature。
3. 同一样本内的重复 MS2 先聚合，再对样本等权聚合，避免 DDA 重复采集次数支配 feature embedding。
4. feature 图边由官方 DreaMS cosine top-k 冻结；共检出和丰度 Spearman 只作独立实验支持诊断。
5. 测试仍是原 v1 的 21 个 query，pseudo-truth 和候选集完全不变。
6. seed 不再被“query 必须具有至少两个候选”这一测试条件截断；所有达到原冻结阈值且落在 Rhea 的 Level 2a-supported feature 均可作为证据 seed。
7. 两层证据路径必须是：query feature → 实验图中的 seed feature → Rhea reaction → candidate。
8. leave-query-out 与 leave-truth-identity-out 保留；同一 seed 化合物的多个 feature 不重复累积证据。

## 固定四对照

- DreaMS：原始谱图分数。
- archived-v1：原 21 seed 的一跳 Rhea，必须精确复现 20/21 降到 19/21、0 修正/1 新增。
- expanded-Rhea-only：扩大 seed 后仍只使用 Rhea，判断 seed 覆盖变化本身的影响。
- two-layer：扩大 seed，但只有实验图相邻 seed 的 Rhea 路径可贡献分数。

## 当前本地验证

- 语法检查、单元测试、shell 静态检查通过。
- 真实 pos_rp 2 万条 MS2 烟雾测试：映射 5,212 条 MS2，形成 2,236 个 feature 节点、8,187 条无重复自环边。
- feature embedding 单位范数、边端点、节点唯一性验证通过。
- archived-v1 精确复现 20/21 → 19/21，0 修正/1 新增。
- 仅 pos_rp 的不完整图不能为 neg_rp 测试 query 提供证据，two-layer 正确 abstain，且报告被标记为 non-formal。

## 正式运行与裁决

正式任务先构建 neg_rp 与 pos_rp 全量图，再运行四对照。只有同时满足以下条件才允许进入 BioAware v2-1：

1. expanded seed 数量严格大于 archived-v1；
2. two-layer 的 corrected > introduced；
3. two-layer Recall@1 不低于冻结 DreaMS；
4. two-layer 至少实际干预一个 query；
5. 所有输出继续明确标为 Level 2a spectral pseudo-truth 安全性/机制审计，不能写成标准品确认准确率。

若 two-layer 仅做到零新增但零修正，则说明保守门有效、信息量不足；下一步扩大外部可验证 benchmark，而不是在这 21 个已经 95.2% 饱和的 query 上调权重。

## 2026-08-28 正式结果

- 全量实验图通过：neg_rp 965 个节点、13,593 条边；pos_rp 5,291 个节点、78,468 条边。
- seed 从 21 行扩展为 75 行、55 个化合物。
- archived-v1 精确复现：20/21 → 19/21，0 修正、1 新增。
- expanded-Rhea-only：1 修正、1 新增，Recall@1 净变化为 0。
- two-layer：只有 2/21 query 获得网络证据、5 条路径；0 修正、0 新增、0 干预。
- 正式门未通过。结论是安全性提高但证据覆盖不足，不是注释准确率提升。

expanded-Rhea-only 唯一合理修正为肌苷支持次黄嘌呤；唯一新增仍为多种氨基酸/犬尿氨酸经转氨反应共同支持 2-氨基丁酸，从而覆盖 GABA。后者缺少这些反应共同需要的 α-酮丁酸，说明一跳网络把“不完整反应物侧”误当成了完整反应证据。

## v2-1 候选机制（仅 post-hoc）

新增 `audit_mtbls13729_bioaware_v2_hyperedge.py`，要求 seed 所在反应侧的其余非通用参与物也必须存在于 leave-query/truth-out 高置信 seed 中。

在已看过的 21 例上，该规则删除 183 条不完整路径、保留 119 条路径，并得到 1 修正、0 新增、21/21。由于规则由本批错误启发，该 +4.76 pp 只能作为机制拟合，不得报告为正式性能。该规则还会删除 MTBLS1905 中原先 evaluation-only 的 guanine→guanosine 单例修正，说明它是保守安全门而不是已经证明的普适最优规则。下一步必须在未看过的新 cohort/benchmark 上冻结验证，并加强自动 seed 覆盖。

## 文件

- `tasks/build_mtbls13729_bioaware_v2_feature_graph.py`
- `tasks/evaluate_bioaware_v2_two_layer.py`
- `tasks/validate_mtbls13729_bioaware_v2_feature_graph.py`
- `tasks/validate_mtbls13729_bioaware_v2_two_layer.py`
- `tasks/test_mtbls13729_bioaware_v2_feature_graph.py`
- `tasks/run_mtbls13729_bioaware_v2_feature_graph.sbatch`
