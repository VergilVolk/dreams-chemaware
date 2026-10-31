# Contrast Atlas：known-anchor / unresolved 实体不相交检验

日期：2026-10-31
证据角色：`development / consumed`
正式结果：`FAIL_CONTRAST_SPECIFIC_INTERACTION_GATE`

## 结论先行

这次不是再挑一个小型疾病案例，而是在现有公共响应集合上进行系统检验：用一组具有完整 InChIKey 谱库支持的实体定义经验响应锚点，再询问**完全不同的一组 unresolved MS/MS 实体**能否恢复同一批研究之间的有向响应关系。

结果表明：

- unresolved WSE 响应与 known anchor 存在粗粒度描述性相关；
- 只去掉来源对均值或单侧 contrast 主效应后，相关仍为正；
- **同时去掉查询 contrast 和候选 contrast 主效应后，相关降为 `rho=0.0598`，leave-one-study-out 范围为 `[-0.0318, 0.1857]`，轴标签置换 `p=0.146`；**
- 开放检索的 Top-1/3/5 分别为 `8/24/41`（202 个方向），相对于均匀随机的 dyad-resampling 差值区间全部跨零；
- 因此，当前 WSE unresolved 路线尚未证明 contrast-specific 的 Atlas 检索能力，不能作为生物学发现结果。

这不是“Atlas 不值得做”。它给出了一个此前缺失的大规模、实体不相交开发终点：新的统一算法必须改善**双中心化后的特异关联和开放 Top-1**，而不能只提高粗相关或连边数量。

## 科学问题与数据隔离

### Known anchor

从 GNPS library-supported 记录中，每个 `study × full InChIKey` 只保留一个响应盲选择的观测离子：先按 WSE、再按 matched peaks、最后按确定性 entity ID 排序。响应方向和疾病/药物名称不参与实体选择。

- 734 个 `study × InChIKey` 选择；
- 429 个唯一 full InChIKey；
- 覆盖 15 个研究来源；
- 每个研究内 contrast 使用 `beta / sqrt(beta^2 + SE^2)` 表示带不确定性的有向响应；
- 两研究之间以共享 full InChIKey 上的 signed cosine 形成经验 known anchor。

谱库支持的 InChIKey 不是标准品和保留时间确认，因此不是 MSI level 1 身份；known anchor 也不是预先成立的生物学机制真值。

### Unresolved test

测试侧只使用两端均为 unresolved 的 reciprocal WSE 边：

- 32,870 条 reciprocal unresolved 边；
- known 与 unresolved 的 `(study_id, entity_id)` 及 raw feature 交集均为 **0**；
- unresolved 配对与目标选择不读取谱库身份、condition 语义名称或登记关系；
- 但响应分数必然使用由实验标签估计的 `beta/SE`，所以它不是与样本独立的外部 truth。

按当前 provisional parent-project map 排除同母项目后，获得 615 个连续评分 contrast pair、50 个研究来源对。615 行中的 495 行（80.5%）、101 个稳定 anchor 中的 71 个（70.3%）涉及 ChemProp 三个来源，因此不能把该集合称为广泛独立的公共研究验证。实体不相交也不保证 compound/family 不相交：unresolved 仍可能是已知分析物的其他加合物、碎片或同一化学家族成员。

## 连续响应结果

| 分析层 | 行数 | Spearman rho | leave-one-study-out rho 范围 | 轴标签置换 p |
|---|---:|---:|---:|---:|
| 原始响应关系 | 615 | 0.2240 | 0.1698 至 0.3167 | 0.001 |
| 去来源对均值 | 610 | 0.2038 | 0.1412 至 0.3064 | 0.001 |
| 去候选侧主效应 | 508 | 0.1694 | 0.0829 至 0.2886 | 0.004 |
| 去查询侧主效应 | 580 | 0.1676 | 0.0804 至 0.2863 | 0.002 |
| **双中心化：同时去两侧主效应** | **478** | **0.0598** | **-0.0318 至 0.1857** | **0.146** |

置换不是 cell-wise 随机打乱。它在每个完整的研究对响应矩阵内整体置换左、右 contrast 标签，从而保留行列依赖结构。该 p 值仍只是“研究对内 contrast 轴可交换”假设下的开发 null，不是普适生物学 p 值。

source-pair dyad bootstrap 也只能作为描述性敏感性分析，因为不同 dyad 会共享同一个 study。双中心化 dyad 区间为 `[-0.1040, 0.2534]`，同样不支持关系特异增量。

## 稳定锚点与开放检索

稳定 anchor 要求：

- 至少 5 个共享 exact InChIKey；
- `|known signed cosine| >= 0.5`；
- leave-one-InChIKey-out 后方向始终不变；
- unresolved 端至少 3 条 reciprocal 边。

获得 101 个稳定 anchor，其中 72 个 known/unresolved 方向一致（71.29%）。但这只是符号一致率，不是开放检索。

开放检索把 101 个 anchor 双向展开为 202 个查询；它们只来自 32 个无序 source-pair dyad，并非 202 个独立重复。其中 142/202 个方向涉及三个 ChemProp accession；这证明覆盖集中，但不能单凭该比例断言信号由 ChemProp 驱动。候选包括所有不同 provisional parent project 的 contrast。修复了一个会静默删除“只有 unresolved 证据、没有 known anchor”的竞争候选的 v1 实现错误后，最终结果为：

| 终点 | 观察数 | 均匀随机期望 | 观察减期望的 source-pair dyad 区间 |
|---|---:|---:|---:|
| Top-1 | 8 | 5.06 | -1.75 至 8.10 |
| Top-3 | 24 | 15.19 | -3.61 至 23.01 |
| Top-5 | 41 | 25.31 | -0.82 至 34.85 |

最差并列名次的中位数为 12。三个区间都跨零；Top-1 尤其没有形成稳健增量。

## 科学裁决

### 本轮允许陈述

> 在 development/consumed 的 15 来源响应集合中，实体不相交的 unresolved WSE 响应与 library-supported exact-InChIKey 经验锚点存在粗粒度描述性相关；该相关在同时移除查询和候选 contrast 主效应后明显减弱且未通过条件置换，开放 Top-1 也未超过 dyad-resampling 下的随机参考。因此当前 WSE endpoint 尚不具备合格的 contrast-specific Atlas 检索能力。

### 本轮不允许陈述

- unresolved 已提供独立生物学增量；
- 当前结果证明跨研究泛化；
- 101 个稳定 anchor 是 101 条正确生物学关系；
- Top-3/Top-5 已显著超过随机；
- WSE 已经实现 Contrast Atlas；
- Full23 或统一算法已有应用增量；
- 任一高排名跨疾病/跨干预关系是机制发现。

例如当前少量 Top-1 包含肥胖—植物细菌处理、皮肤 AD—植物处理、药物—镧响应等跨域组合。它们说明经验 known-response coherence 不能自动等同于生物学正确性，而不是值得直接写成机制的发现。

## 对统一算法的直接用途

本检验从现在起作为**冻结的开发终点**，而不是论文成功结果。WSE、Official、Noise 和最终统一算法应在相同研究、相同 response operator 和相同 unresolved 候选轴上比较；每个模型可形成自己的合法 reciprocal decision，不能固定 WSE 最终选边。

晋级至少需要同时满足：

1. 双中心化 known-anchor / unresolved 相关为正，leave-one-study-out 最低值大于 0；
2. 轴标签条件置换通过预设门；
3. 开放 Top-1 相对随机的跨来源敏感性为正，而不是只改善 Top-5；
4. 收益不由 ChemProp 来源独占；
5. known-only、known+unresolved、机会匹配 known 和 shuffled unresolved-MS2 分开比较；
6. 任何结构依赖模块只在具有合法候选结构时启用，不能把身份排序有效性外推到跨质量化学关系。

即使统一算法通过本门，也只证明开发期的 unresolved-response specificity；仍需未参与开发的独立生物学控制才能形成文章主张。

## 复现

正式输出：

- `data/validation/contrast_atlas_known_anchor_unresolved_test_v2_20261031/report.json`
- `data/validation/contrast_atlas_known_anchor_unresolved_test_v2_20261031/known_anchor_unresolved_scores.tsv.gz`
- `data/validation/contrast_atlas_known_anchor_unresolved_test_v2_20261031/stable_known_anchors.tsv`
- `data/validation/contrast_atlas_known_anchor_unresolved_test_v2_20261031/entity_disjoint_retrieval.tsv`
- `data/validation/contrast_atlas_known_anchor_unresolved_test_v2_20261031/checksums.sha256`

运行命令：

```powershell
python -u tasks/run_contrast_atlas_known_anchor_unresolved_test.py `
  --collection data/validation/contrast_atlas_open_response_collection_v10_frozenbase_msv000080918_v7_20261031 `
  --library-evidence data/validation/contrast_atlas_global_biological_adjudication_v1_20261030/entity_library_evidence.tsv.gz `
  --edges data/validation/contrast_atlas_complete_identity_wse_alignment_v5_msv000080918_20261031/source_pair_frozen_edges.tsv.gz `
  --parent-projects tasks/contrast_atlas_parent_projects_v2_msv000080918.tsv `
  --output data/validation/contrast_atlas_known_anchor_unresolved_test_v2_20261031 `
  --min-known 5 --min-unresolved 3 --anchor-abs 0.5 `
  --bootstrap 2000 --seed 20261031
```

实现 SHA256：`ff82d903265266fbdd99448dbd96729cb1faff5bac88047fd60b3fe6c76c7142`。结果目录另含逐文件 SHA256。五项最小回归测试通过。
