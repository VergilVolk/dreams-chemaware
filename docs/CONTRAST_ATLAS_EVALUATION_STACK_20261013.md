# Metabolomic Contrast Atlas 冻结评测栈

日期：2026-10-13  
状态：开发期方法合同；尚无跨来源正向结果，尚无独立确认  
中心对象：`study-level contrast signature`，不是单谱图、结构注释、网页或疾病专属模型

## 1. 今天纠正的两个统计错误

### 1.1 证据世界不得改变候选题目

`all`、`known_only`、`measurement_matched_known` 和
`unresolved_shuffled_ms2` 必须面对完全相同的 query-target candidate
universe。某个世界缺少已知实体或合格模式时，只能对该题弃权，不能删除 query、target
或缩小分母。否则所谓 unknown increment 会混入候选难度差异。

实现现在将候选宇宙与实际证据表分离；外部四世界只调用核心的 `all` 身份视图。核心内部的
`all/known_only/unresolved_only` 仅为身份组成诊断，不再与四个配对评测世界混名。

### 1.2 共享样本中的 feature effect 不得假设独立

多个 feature 的 `beta` 和 `SE` 来自同一批生物样本，通常相关。原型曾用

`sqrt(sum((w_i SE_i)^2))/sum(w_i)`

聚合 pattern SE，这隐含零协方差，可能产生虚假精度。当前默认改为只凭边际 SE 能支持的
完全正相关上界：

`SE_pattern = sum(|w_i| SE_i)/sum(w_i)`。

以后若某研究提供按相同生物学单位同步生成的 bootstrap draws，可用经验协方差替换该保守
上界；不同 feature 各自独立 bootstrap 不合格。

## 2. 不可交换的阶段顺序

1. **研究内效应冻结**：从原始样本和设计估计 entity/independent-ion-family 的
   `beta, SE, detectability, QC`；不同研究的原始强度不合并。
2. **表型盲化学证据冻结**：构建 strict entity links 与 calibrated pattern loadings；
   禁止读取 beta、疾病/处理标签、p 值和验证标签。
3. **四个配对证据世界冻结**：全量、已知结构、测量质量匹配的已知结构、未知实体 MS2
   分层置乱。测量匹配与置乱只允许使用预先声明的技术协变量。
4. **固定候选宇宙检索**：同一 query registry、同一 target graph、同一 source filter；
   缺证据返回 abstention。
5. **内层化学对齐 null**：在 target 内置换 pattern ID，保留 effect/SE 分布与覆盖；
   得到方向性经验 p 值和 query-world 内 BH q 值。该 null 不读取生物学相关性真值。
6. **冻结输出和阈值**：保存输入 SHA、随机种子、覆盖阈值、候选分母和 calibration 状态。
7. **单独开封 truth ledger**：每个候选必须有显式 truth；未列出不自动当负例。truth 必须
   登记独立复现/设计性反转或救援/正交功能复现/预设 decoy 等依据，并证明在 Atlas 评分前
   冻结且 selection blind。
8. **选择性 risk-coverage**：同时报告 coverage、错误回答率、match R@1/MRR 与 no-match
   accuracy；开发集选 operating point 后不得在 sealed 集重选。
9. **外层匹配 truth null**：在 query 与预声明技术 strata 内置换 relevance labels，检验
   检索是否超出技术匹配机会水平。
10. **未知信息增量裁决**：按预声明 biological cluster bootstrap；`all` 在每一 draw 中与
    当次最强控制比较。总体验证、每个 source holdout 和每个 instrument holdout 的增量 CI
    下界都必须严格大于零，同时错误率增量 CI 上界不得大于零。

任何阶段失败都只降低对应主张；不得放宽相似度、减少 no-match 或事后改 truth 挽救。

## 3. 四个评测世界回答不同反事实

| 世界 | 保留内容 | 回答的问题 |
|---|---|---|
| `all` | 合格 known + unresolved | 完整方法能做到什么 |
| `known_only` | 只有结构已知或谱库支持实体 | 未知信息之外的常规可解释部分有多强 |
| `measurement_matched_known` | 在强度、检出率、QC CV、缺失、MS2 峰数等技术协变量上与 unresolved 匹配的 known | 增量是否只是未知信号更多/更易测 |
| `unresolved_shuffled_ms2` | 保留全部研究内 effects，但在预声明技术 strata 内错配 unresolved 的 MS2 pattern blocks | 增量是否来自真实化学对应，而非未知实体数量或效应分布 |

最后一个世界只消融 unresolved 的 MS2-pattern 对应；strict identity 与其他证据若要主张独立
价值，必须另设家族消融，不能把所有模块合成一个总分。

## 4. 当前实现

- 输入与泄漏合同：`tasks/contrast_atlas_schema_v1.json`
- 核心多层检索：`tasks/contrast_atlas_core.py`
- 四世界构建：`tasks/build_contrast_atlas_control_worlds.py`
- 固定 candidate universe 检索：`tasks/run_contrast_atlas_control_retrieval.py`
- 完整 pretruth 前缀：`tasks/run_contrast_atlas_pretruth_stack.py`
- 内层 pattern null：`tasks/calibrate_contrast_atlas_pattern_null.py`
- truth 开封：`tasks/build_contrast_atlas_query_outcomes.py`
- risk-coverage：`tasks/evaluate_contrast_atlas_selective_curve.py`
- 外层 matched truth null：`tasks/calibrate_contrast_atlas_outer_truth_null.py`
- paired unknown increment：`tasks/adjudicate_contrast_atlas_unknown_increment.py`

pretruth 执行器故意不接受 truth 文件，并在报告中写入 `truth_opened=false`。这不是工程拆分，
而是防止标签渗入化学证据与阈值开发的实验隔离。

## 5. 当前实证状态

### 5.1 已完成但只能称开发证据

- 三个历史真实 shard admission replay：3/3 保持停止，0/3 合格进入 Atlas。
  - ChemProp：12 drugs、36 queries；旧同图前结果为 mass 7/36、cosine 10/36、WSE 7/36，
    药物 cluster CI 不支持可靠增量。
  - ST003325：12 clones、27 comparisons，10/27；median margin -0.03970，permutation
    p=0.7126。
  - LCNEC：263 entities，6/263 具备候选资格，public certificate 0/263。
- 合成合同覆盖 core、控制世界、固定候选分母、pretruth 栈、内外层 null、truth 开封、
  selective curve 与 paired increment。合成测试只证明实现合同，不证明生物学有效。

### 5.2 TT7 / MSV000096214

当前为 `development/consumed`。正离子已有 4,562 个 consensus groups、52 个样本/文件；负离子
对齐仍在本地运行。只有负离子完成并由 table audit 给出 `pass_to_effect_stage=true`，随后 effect
report 给出 `pass_to_posthoc_t4=true`，才允许转换为 Contrast Atlas record。TT7 是强遗传正控
shard，不是总方法，也不能单独证明跨来源泛化。

### 5.3 本地第二来源文件审计

| 资产 | 本地可观测文件 | 当前可用角色 | 限制 |
|---|---:|---|---|
| MTBLS13729 | 135 files，3,958,268,726 bytes；63 mzML、72 HDF5 | 不同 source family 的 disease-side `development/consumed` shard；可构建 no-match 与研究内 contrast | 已深度使用；本地并非四面板 240 mzML 的完整副本；不能当 sealed confirmation |
| ST001154 pilot | 14 files，636,762,101 bytes；8 mzML | 资格/接口 pilot | 文件是预选择子集，不能代表无偏研究级 contrast |
| LCNEC preflight | 8 files，635,803,990 bytes；2 ZIP | consumed disease query/压力测试候选 | 当前未形成统一 contrast record；不得从个别 dark hits 推成 Atlas 正结果 |

进一步的面板级审计已证明 **0 个 MTBLS13729 面板获准转换**：neg-RP 虽有完整 30 对登记
设计，但本地只有 20/60 mzML、19/60 HDF5 且无统一 MS1-MS2 link；pos-RP 登记 59 样本、
29 个完整配对，本地只有 13/59 mzML、16/59 HDF5。详见
`CONTRAST_ATLAS_MTBLS13729_READINESS_RESULT_20261013.md`。因此目前仍没有合格的第二来源
contrast record。若未来恢复完整资产，而 TT7 与 MTBLS13729 又没有预先登记的生物学关联，
正确 truth 仍是 no-match/decoy，而不是事后寻找共同通路。

## 6. 下一科学动作

1. 等待 TT7 raw 链自然结束，严格按两个门转换记录；失败则保留负结果。
2. MTBLS13729 本地资格审计已 FAIL；只允许恢复缺失 raw/可复现全量矩阵后按原门重审，禁止
   用当前 63 个跨面板残缺 mzML 或预选 EIC 表拼接。
3. 并行审计另一个不同 source family；只有完整研究级定量宇宙与 MS1-MS2 linkage 同时闭合，
   才能用同一 schema 构建 no-match/query 压力测试。
4. 另行寻找具备相同扰动或预设反转/救援的独立来源正配对。没有这种 truth，Atlas 只能证明
   可运行和可弃权，不能证明能检索生物学关系。
5. 真实数据必须比较四世界，并报告所有分母、data role、consumption 与 leakage 状态。

当前最重要的未解决科学问题不是数据库规模，而是：在严格固定候选宇宙、控制共享样本相关性、
预注册 truth 且包含 no-match 的条件下，unresolved MS/MS 是否在独立来源留出中带来正的净
检索增量。这个问题尚未被回答。
