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

### 4.1 组成中心化的响应核：真实开发结果

为直接分离“研究的化学组成相似”和“MS/MS 碎片证据随有向响应共同变化”，新增了一个真值盲算子。每个 study × contrast × evidence world 内先计算 `beta/sqrt(beta^2+SE^2)`，再减去该研究内均值，将中心化响应权重作用于单位谱图，最后对所得 signed fragment vector 做 L2 标准化并跨研究比较。登记关系仍在完整候选表写出之后打开。

相对未中心化 fragment-measure baseline，IBD 双向关系的 signed-similar 排名发生如下变化：

| Evidence world | VEO-IBD → adult IBD200 | adult IBD200 → VEO-IBD |
|---|---:|---:|
| all，旧基线 | 5/17 | 4/17 |
| all，centered kernel | 3/17 | 1/17 |
| unresolved，旧基线 | 5/17 | 7/17 |
| unresolved，centered kernel | 3/17 | 3/17 |
| known，centered kernel | 1/17 | 1/17 |

相应的 composition 排名为：`all` 13/17 与 8/17，`unresolved` 17/17 与 8/17，`known` 双向 1/17。由此可见，centered kernel 的 `all/unresolved` 排名改善不是简单复制 composition 排名；但 `known` 世界仍无法区分响应核增量和强烈的来源组成相似性。

采用固定种子 `20261020` 做 31 次 study × contrast × identity-stratum 内响应权重重排。对 direction-agnostic alignment-strength 排名，真实 IBD 关系取得同样好或更好 null 排名的比例为：

- VEO-IBD → adult IBD200：all 0.387、known 0.065、unresolved 0.194；
- adult IBD200 → VEO-IBD：all 0.129、known 0.097、unresolved 0.161。

这些是非校准 sensitivity，不是随机化 p 值。它说明 known IBD 信号最强，unresolved 有提示性但尚不充分；HFD 关系没有稳定改善。centered kernel 因而是值得保留的关系层候选算子，但尚未通过 unknown-information gate。

### 4.2 IBD200 板次混杂否证

成人 IBD200 元数据进一步显示，19 个健康对照全部位于 Plate3，而 158 个 CD/UC 病例分布于 Plate1、Plate2、Plate3 和未标注板次。原始 `adult_all_ibd_vs_healthy` 响应因此与板次完美混杂，不能作为纯疾病效应。

在唯一同时包含病例和健康对照的 Plate3 内重算 36 个 CD/UC 与 19 个 HC：

- 固定 QC+MS/MS 候选仍为 746 个；
- 662 个实体获得有限且正的 SE，其中 library-supported 135、unresolved 527；
- 84 个 Plate3 内效应不可估计的实体明确弃权；
- 原始与 Plate3 beta 的 Pearson 相关为 0.859；
- 不确定性收缩响应权重相关为 0.782；
- beta 符号一致率为 81.72%。

使用 Plate3 响应重跑 centered kernel 后：

| Direction / world | 原 alignment-strength rank | Plate3 rank |
|---|---:|---:|
| pediatric → adult, all | 5 | 5 |
| pediatric → adult, known | 1 | 6 |
| pediatric → adult, unresolved | 6 | 5 |
| adult → pediatric, all | 1 | 7 |
| adult → pediatric, known | 1 | 1 |
| adult → pediatric, unresolved | 3 | 5 |

最醒目的成人→儿童 `all` Top-1 消失，儿童→成人 `known` 也从 1 降到 6。Plate3 的 unresolved signed-similar 排名仍为 3/4，但 alignment-strength 仅为 5/5，且 31 次分层响应重排中同样好或更好的比例为 0.194/0.290。

因此，原 IBD Top-1 不能继续作为关系层成功证据。更严格的结论是：IBD 跨研究响应存在提示性的正向谱学关系，但当前排名对成人效应估计和板次混杂敏感，尚未形成稳健、可裁决的 biological-response retrieval。

### 4.3 来源内响应轴特异性：HFD 正控与 obesity 反证

全局检索排名仍可能由来源差异驱动。现有集合提供了一个不新增数据即可进行的更严格正控：mouse HFD 查询在 `MSV000088023` 同一目标来源内面对四个对比，其中两个是 female/male HFD-vs-normal，两个是 female-vs-male sex contrasts。目标来源、谱图集合和 composition score 完全相同，只有响应权重不同。

使用 centered-kernel alignment strength：

- `all`：male HFD rank 1、female HFD rank 2、normal-sex rank 3、HFD-sex rank 4；
- `unresolved`：同样精确恢复两个 HFD 对比为 Top-2；
- 两个 HFD 与最强 sex decoy 的分离边际分别为 0.00827（all）和 0.00718（unresolved）；
- 31 次分层响应重排中，得到同样大或更大边际的次数均为 3/31；
- `known` 因目标对比仅有 5 个 known entities、有效响应支持低于冻结门槛而弃权。

这是第一个真正排除“只识别目标来源”的响应轴正控结果。它证明 unresolved-only 谱学响应在这个 exact-intervention 场景中能够回答 known-only 无法回答的问题，但**不能**证明 measurement-matched unknown accuracy increment：known-only 没有形成可比较的有效回答，31 次 feature-level reassignment 也只是 sensitivity，不是校准随机化检验。

随后对唯一另一个双来源、多对比条件族 obesity 做 post-hoc 泛化检查。四个 obesity 查询分别在另一来源的四个候选中检索两个 obesity 对比：

| Evidence world | exact obesity Top-2 / 4 queries | 重排 query-replicates 中 exact Top-2 比例 |
|---|---:|---:|
| all | 0/4 | 0.185 |
| known | 0/4 | 0.105 |
| unresolved | 1/4 | 0.161 |

因此 HFD 能力没有广泛泛化到 diabetes/depression 分层下的 obesity 响应。当前最准确的科学边界是：centered kernel 在预登记 exact-intervention HFD 正控中显示来源内响应轴特异性，但对异质条件的通用检索尚未成立。

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
- 组成中心化响应核：`tasks/run_contrast_atlas_centered_response_kernel.py`
- 中心化结果：`data/validation/contrast_atlas_centered_response_kernel_v2_20261020/report.json`
- IBD200 板次审计：`tasks/audit_ibd200_plate_confounding_response.py`
- Plate3 重算集合：`data/validation/contrast_atlas_open_response_collection_ibd200_plate3_v2_20261020/report.json`
- Plate3 centered-kernel 结果：`data/validation/contrast_atlas_centered_response_kernel_ibd200_plate3_v2_20261020/report.json`
- HFD 来源内正控：`data/validation/contrast_atlas_hfd_source_conditioned_axis_v3_20261022/report.json`
- Obesity 来源内泛化反证：`data/validation/contrast_atlas_obesity_source_conditioned_axis_v1_20261022/report.json`
