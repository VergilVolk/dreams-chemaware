# Contrast Atlas：三项规范化母药重叠的跨项目生物学控制结果

日期：2026-10-31

结果状态：`DRUG_CONTROL_PANEL_COMPLETE`

选择状态：`post_hoc_independent_contextual_adjudication_not_a_sealed_benchmark`

## 结论先行

这次得到的是一个真实、可复算的 **Atlas 生物学控制面板**，不是统一算法已经成功的证据。

在三套现有资产中：

- 2019 年 MTBLS896 单菌株筛选含 271 种药物、76 个菌株和 4 个无菌 pH 对照；
- 2026 年 MTBLS6166 预印本的 Zenodo 处理表含 272 个非内标药物标签、60 个人源及 29 个动物来源群落；
- 已消耗的 ChemProp WSE 登记表含 12 种精确药物关系；
- 按预先写明的盐型归一化规则，三者的完整母药交集只有 **omeprazole、sertraline、telmisartan 三种药物**。

因此，这三种药物不是从更多合格交集中挑出的“漂亮案例”，而是当前资产的**完整规范化母药三方交集**。例如 `SERTRALINE_HYDROCHLORIDE` 被显式归一化到 `SERTRALINE`；原始标签保留在逐行结果中。该交集仍然很小，而且是在看过 ChemProp 开发结果后进行的背景核实，不能充当封存外测。

最重要的生物学结论是：**同一化学实体没有一个脱离实验背景的全局响应真值。** 三种药物分别形成稳健阳性、技术条件依赖阳性和群落依赖响应三类控制。Contrast Atlas 必须保存实验背景、对照定义和异质性，而不能把“同一分子”直接压缩成一个统一方向标签。

## 数据角色和计算边界

### MTBLS896：单菌株与 pH 控制

2019 年 Nature 研究测量 76 个肠道菌株对 271 种药物的作用。本文使用官方 Figshare 处理结果 `TableS3_drug_fold_changes.csv`，对每种药物和测试单元复算：

```text
positive = consumed_pct > TableS3 adaptive threshold AND pFDR <= 0.05
```

该研究是 ChemProp 之外的独立项目，但不是为本项目预留的前瞻外测。原论文及数据入口：

- https://pmc.ncbi.nlm.nih.gov/articles/PMC6597290/
- https://doi.org/10.6084/m9.figshare.8119058

### MTBLS6166 / Zenodo 13943903：群落响应

2026 年预印本测量 60 个人源和 29 个动物来源肠道群落。这里使用 Zenodo 发布的 `activity_summary_table.csv` 中 `Type_metab` 作者判定；论文方法说明该判定会将群落时间趋势与相应无菌对照比较。我们**没有**从原始九时间点、四重复数据重新估计 beta、SE 或 FDR。

该项目与 MTBLS896 是不同实验项目，但来自同一研究谱系，不能写成跨实验室独立复现；论文目前仍是未同行评议预印本。论文明确将完整原始代谢组和元数据登记在 MTBLS6166，而不是 MSV000096589。来源：

- https://pmc.ncbi.nlm.nih.gov/articles/PMC12871589/
- https://zenodo.org/records/13943903

### ChemProp WSE：已消耗开发结果

WSE 排名来自 MSV000094899 与 MSV000093571 的同母项目跨设计检索，且两者均已用于开发。本结果只用于说明三类生物学控制在当前 Atlas 检索中的表现，不构成外部算法验证。

## 完整结果

WSE 排名按 `all / known / unresolved` 报告 `similar` 响应面名次。登记表中三种药物均为 `expected_signed_direction=unspecified`、`signed_truth_claimed=False`；因此这里的 `similar` 面是描述性检索面，不是预先成立的正确方向。

| 药物 | MTBLS896 菌株阳性 | pH 对照阳性 | MTBLS6166/Zenodo 群落阳性 | 已精确映射人源群落阳性 | WSE 094899→093571 | WSE 093571→094899 | 生物学角色 |
|---|---:|---:|---:|---:|---:|---:|---|
| Omeprazole | 75/76 | 3/4 | 69/88 | 58/58 | 4 / 16 / 4 | 3 / 17 / 2 | 无菌对照校正后的群落阳性，但存在明显 pH/自发降解风险 |
| Telmisartan | 12/76 | 0/4 | 83/89 | 56/59 | 5 / 1 / 6 | 6 / 1 / 6 | 跨项目稳健生物学正控；known-only 已足够 |
| Sertraline | 0/76 | 0/4 | 49/87 | 34/57 | 5 / 4 / 5 | 5 / 8 / 5 | 单菌株阴性、复杂群落阳性，属于背景依赖对照 |

群落标识表还存在一个不得静默修复的问题：activity 表含 `WLS-250`，identifier 表则含 `WLS-240`。三种药物中 `WLS-250` 均有可评价记录；本分析将其标为未映射样本，不把它推断为人源，也不将其改名为 `WLS-240`。

## 逐药物科学裁决

### Telmisartan：合格的生物学阳性控制，不是 unknown 增量

Telmisartan 在 MTBLS896 中有 12/76 个菌株达到表中阳性规则，4 个 pH 对照均为阴性；后续群落表中 83/89 个可评价群落为作者阳性。两项实验共同支持真实生物学活性。

然而，在 ChemProp WSE 检索中，known-only 的 `similar` 面在两个方向均为第 1 名；unresolved 则均为第 6 名。因此 telmisartan 证明“这项药物活性具有外部生物背景”，但**不证明未解析信号提供新增信息，也不验证整个非靶向响应向量应为同向**。

### Omeprazole：最值得继续检验的 unresolved 候选，但只能作条件性陈述

Omeprazole 在 75/76 个菌株中达到表中阳性规则，但 3/4 个 pH 对照也达到同一规则，说明单凭母药下降无法区分微生物作用与非生物降解。后续群落项目的 `Type_metab` 已按作者方法与无菌对照比较，69/88 个可评价群落为阳性；已精确映射的人源记录为 58/58 阳性。

在已消耗 WSE 检索中，unresolved 相对 known-only 的描述性 `similar` 面排名变化为：

- MSV000094899 → MSV000093571：16 → 4，改善 12 位；
- MSV000093571 → MSV000094899：17 → 2，改善 15 位。

这使 omeprazole 成为一个有外部生物背景支持的 unresolved 候选。但它仍然不是 unknown 增量的证明：`known` 与 `unresolved` 分别基于独立的 WSE Top-1 图，互惠边分别为 36 与 2,126；排名变化没有做等机会校准。药物是在查看开发结果后核实，使用的是 WSE 而不是 Full23，自研算法也没有参与本项结果。这里的 `unresolved` 仅指没有严格谱库支持，不等于新分子。

### Sertraline：不能当简单阴性，应作为响应背景异质性控制

Sertraline 在 MTBLS896 的 76 个单菌株中无一达到表中阳性规则，4 个 pH 对照也均为阴性；但在后续群落表中，49/87 个可评价群落被作者判为阳性。

这不是一个应被“救回”的矛盾标签，而是 Atlas 必须表达的信息：单菌株筛选与完整群落的测量对象、覆盖和实验条件不同，响应不可无条件运输。群落互作或未覆盖菌株是可能解释，但本结果本身没有区分这些机制。它不能被简化为全局阳性或全局阴性。当前 WSE 中 unresolved 相对 known-only 的 `similar` 面一个方向退后 1 位、另一个方向改善 3 位，也没有稳定 unknown 增量。

## 对 Atlas 方法的直接含义

这项结果支持的不是“再建一张药物数据库”，而是一个明确的响应知识单元：

```text
化学实体
+ 实验来源和生物系统
+ 干预/对比定义
+ 技术与无菌对照
+ 有向效应和不确定性
+ 身份或化学关系证据等级
```

同一实体在不同背景下的记录可以一致，也可以异质；Atlas 应返回这些背景化记录及证据，而不是先将其合并成一个全局符号。只有同一实体身份成立时才能做实体级累积；化学类似物可以支持模式关系，但不能冒充同一分子。

本面板还把两个必须分开的命题固定下来：

1. **药物活性在什么实验背景下成立？** Telmisartan 提供正控，omeprazole 提供技术条件控制，sertraline 提供 assay/context 不可无条件运输的控制。
2. **unresolved 或自研算法是否增加正确答案？** 本轮尚未证明。必须在相同完整候选集合、相同响应算子和明确机会控制下比较 WSE、Official、Noise 和 Full23；各模型必须独立选边，不能继承 WSE 最终边。

## 允许与不允许的论文陈述

允许：

> 在当前三方完整规范化母药重叠中，不同项目的处理结果显示药物响应具有稳健、技术条件依赖和 assay/context 不可无条件运输三类模式。已消耗的 WSE 检索中，omeprazole 的 unresolved `similar` 面在两个方向均比 known-only 排名更高，而 telmisartan 的 known-only `similar` 面已为第 1；登记关系没有预设有符号真值，这些结果只建立背景化响应控制，尚未证明 unknown 或统一算法增量。

不允许：

- 三种药物构成大规模独立外测；
- MTBLS896 和 MTBLS6166 是跨实验室独立复现；
- `Type_metab` 是我们从原始数据重算的效应或 FDR；
- omeprazole 的母药下降全部由微生物导致；
- sertraline 是全局阴性；
- unresolved 排名改善已经证明 unknown 每单位证据更有信息；
- `unresolved` 等于从未见过的新化学分子；
- 外部母药下降验证了 ChemProp 整体非靶向响应向量的 `similar` 面；
- Full23、Noise、ChemAware 或 BioAware 已在此面板产生应用增量；
- 本轮结果构成机制发现。

## 下一项决定性工作

三药面板应保留为**生物学校准与错误解释控制**，而不是文章的唯一主实验。当前最直接的算法—应用闭环仍是：在完整 Atlas 候选块上生成 WSE、Official、Noise 和 Full23 的独立逐候选分数，然后检验这些模型是否改变三项控制及更大响应集合中的实体对应和研究级回答。

Full23 目前缺少整个 55,127-entity Atlas 的七通道证据，现有少量旧边重加权不能替代完整计算。得到完整打分之前，本报告不会用“等待算法”为由虚构增量，也不会把 WSE 结果改名为统一算法结果。

## 复现

```powershell
python tasks/build_contrast_atlas_independent_drug_control_panel.py `
  --mtbls896-fold-changes data/reference/contrast_atlas_drug_controls_20261031/mtbls896_processed/TableS3_drug_fold_changes.csv `
  --community-activity data/reference/contrast_atlas_drug_controls_20261031/activity_summary_table.csv `
  --community-identifiers data/reference/contrast_atlas_drug_controls_20261031/community_identifiers.csv `
  --wse-registered-ranks data/validation/contrast_atlas_complete_identity_wse_open_retrieval_v5_msv000080918_20261031/registered_relation_ranks.tsv `
  --output-dir data/validation/contrast_atlas_independent_drug_control_panel_v1_20261031

python -m pytest -q tests/test_contrast_atlas_independent_drug_control_panel.py
```

正式输出：

- `data/validation/contrast_atlas_independent_drug_control_panel_v1_20261031/report.json`
- `data/validation/contrast_atlas_independent_drug_control_panel_v1_20261031/drug_control_summary.tsv`
- `data/validation/contrast_atlas_independent_drug_control_panel_v1_20261031/mtbls896_drug_strain_calls.tsv`
- `data/validation/contrast_atlas_independent_drug_control_panel_v1_20261031/mtbls6166_drug_community_calls.tsv`
- `data/validation/contrast_atlas_independent_drug_control_panel_v1_20261031/wse_consumed_exact_drug_ranks.tsv`
- `data/validation/contrast_atlas_independent_drug_control_panel_v1_20261031/checksums.sha256`

所有分母、数据角色、输入 SHA256、实现 SHA256 和声明边界均写入 `report.json`。
