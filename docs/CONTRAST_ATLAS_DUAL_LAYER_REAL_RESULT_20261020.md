# Metabolomic Contrast Atlas：七来源双层真实结果

日期：2026-10-20  
证据角色：development / consumed  
结论状态：中心科学门未通过；不得宣称 unknown 增量或外部验证

## 1. 不再改变的科学目标

Metabolomic Contrast Atlas 的核心对象是研究级、有方向、有幅度和不确定性的生物学响应。系统应允许一项新的非靶向 LC-MS/MS 研究在公共研究集合中检索相似、相反或无可比证据的响应，并明确区分：

1. **身份层**：跨研究是否观测到同一化学实体，以及该实体的响应是否复现。只有这一层允许把对齐后的效应作为同一实体证据累积。
2. **化学关系层**：即使不是同一实体，局部化学相关实体是否共同支持可验证的响应模式关联。该层不能合并为同一分子的 beta/SE，但不是普通邻居查询，也不是可有可无的低等级附属物。

两层共同服务开放式研究级响应检索。任何一个局部匹配错误只能修改具体算子，不能再次重定义总课题。

## 2. 固定数据边界

现有集合包含：

- 7 个独立来源家族；
- 18 个研究级对比；
- 446 个生物学单元；
- 54,142 张代表 MS/MS；
- 24,185 个 response-eligible feature entities；
- 66,616 条响应记录，其中 1,602 条 library-supported，65,014 条 unresolved；
- 每个查询有 14–17 个 source-disjoint 候选；
- 6 条预登记的有向设计相关关系；
- **0 条预登记 signed-direction truth**。

`unresolved` 仅指冻结分区下无谱库支持，不代表独特的新分子。未登记的跨研究配对均为未裁决对象，不能作为阴性。

## 3. 身份层真实结果

严格 planar identity 基线在全部 264 个固定候选对中仅 56 个可评分，208 个不可评分；其中 130 个没有任何共享严格身份，78 个共享身份不足 3 个。

在 6 条预登记设计关系中：

- 可回答：2/6；
- 弃权：4/6，且均为零共享严格身份；
- signed-similar Top-1：1/6；
- unsigned Top-1：2/6。

可回答的两条均来自 pediatric VEO-IBD 与 adult IBD200 的双向关系，共享 12 个严格身份。HFD 的四条双向/性别关系全部因零共享严格身份不可回答。

另一个五来源已知身份正例校准集上，同前体 10 ppm + official DreaMS 图得到：

- all：91/95 answered 正确，95/124 truth groups 被回答；
- known：93/94 answered 正确，94/124 truth groups 被回答；
- unresolved：没有可校准的已知身份回答。

这是已知身份正例子集上的开发期校准，不是开放集 FDR。`all` 没有超过 `known`，因此 unknown identity increment 未建立。当前图还是同前体质量窗口下的谱学对应，不是完整的中性质量—加合物实体解析。

## 4. 化学关系层真实结果

七来源透明 fragment-measure baseline 在 `all`、`known`、`unresolved` 三个世界分别保留相同的 264 个固定候选对，所有候选均有分数。

6 条登记关系的 Top-1 计数如下：

| Evidence world | Signed similar | Signed opposite | Unsigned response | Composition |
|---|---:|---:|---:|---:|
| all | 0/6 | 0/6 | 0/6 | 0/6 |
| known | 0/6 | 1/6 | 1/6 | 2/6 |
| unresolved | 0/6 | 0/6 | 0/6 | 0/6 |

`known` 中的 Top-1 包括：

- VEO-IBD → adult IBD200：unsigned-response Top-1；
- VEO-IBD ↔ adult IBD200：composition 双向 Top-1；
- female HFD → mouse HFD：signed-opposite Top-1。

这些只能称为“登记设计相关关系的描述性排名”。因为预登记 signed direction 为 `unspecified`，不能把 signed-similar 或 signed-opposite 的结果判为正确机制，也不能把 composition Top-1 当作响应检索成功。

更关键的是，`all` 和 `unresolved` 没有让任何登记关系达到任一指标的 Top-1。因此当前关系算子没有显示 unresolved MS/MS 相对 known-only 的新增正确知识。

## 5. 中心裁决

当前七来源资产是真实的公共响应底座，但还没有证明 Metabolomic Contrast Atlas 的中心命题：

- 已裁决的正确开放式生物学检索：0；
- 已裁决的错误开放式生物学检索：0；
- 原因：没有登记 signed-direction truth，开放 Top-1 均不能自动判真或判错；
- unknown-information increment：**NOT DEMONSTRATED**；
- 外部确认：无，所有来源均为 development / consumed。

这不是把方法降格为身份匹配器的理由。它说明当前严格身份基线覆盖不足，而当前 fragment relation baseline 也没有捕获登记关系。下一步必须改进和验证两个具体算子，同时保持研究级响应检索这一总目标不变。

## 6. 下一步唯一高性价比科学动作

不扩到 18 个来源，不再增加通用协议，也不围绕单个疾病重写方法。直接在现有 7 来源上完成一个可裁决的双层实验：

1. 为已有 HFD/IBD 关系补充**事先定义且有独立依据的可检验响应命题**，明确哪些只要求设计相关、哪些预期同向、哪些方向不可预设。
2. 身份层只改善中性质量/加合物兼容和 no-match 风险；不得用同向响应反推身份真值。
3. 关系层开发真正的局部化学响应匹配：分离化学组成、响应符号/幅度和谱学关系，并用 response reassignment 检查它是否只识别来源组成。
4. 在相同候选全集上给出 correct / incorrect / abstain；只有 `all` 新增的正确回答超过 `known` 且错误风险不升高，才算 unknown 增量。

在这一步通过以前，不宣称已经形成新的生物学发现方法；但也不再把任何单一身份实验、疾病实验或关系算子改写成整个课题。

## 7. 可复算产物

- 脚本：`tasks/summarize_contrast_atlas_dual_layer_real_results.py`
- 报告：`data/validation/contrast_atlas_dual_layer_real_result_v1_20261020/report.json`
- 身份层登记关系：`registered_identity_layer_results.tsv`
- 关系层登记关系：`registered_relation_layer_results.tsv`
- 开放 Top-1（含并列）：`open_relation_top_hits.tsv`

