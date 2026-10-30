# Metabolomic Contrast Atlas：PRISM 严格候选的身份与离子家族裁决

日期：2026-10-30

数据角色：**development / consumed**。本结果不是封存外测，也不是新的独立项目复制。

## 结论先行

成人 IBD200、儿童 VEO-IBD 与 ST001000 两个参与者独立队列形成的严格筛选结果，不能写成“13 个代谢物在多队列中复现”。正确的证据层级是：

| 层级 | 数量 | 科学含义 |
|---|---:|---|
| 三个队列拟合均通过各自 BH 过滤的检验 | 17 | CD/UC、模式和重复观测均未独立 |
| ST001000 feature-mode 观测 | 15 | 同一个源实体对可跨模式重复 |
| 成人—儿童源实体对假说 | 13 | 尚未完成 ST001000 样本链接 MS/MS 身份确认 |
| 表型盲离子家族折叠后的候选分析物家族 | **9–13** | 9 是按同模式共洗脱和预设正离子质量差折叠后的工作分母；13 是不作跨实体对折叠的上限 |
| 当前策略允许的同一 observed-ion 合并 | **2/13** | 两端均达到共同 Gold/Silver 谱库政策且全 InChIKey 相同；仍不是标准品确认 |
| MSI level-1 或标准品确认身份 | **0/13** | 当前 ST001000 结果没有样本链接 MS/MS |

因此，本轮真正得到的是一组**表型盲优先化的身份/离子家族假说**，不是 13 个已确认分子，更不是 13 项独立生物学发现。

## 一、这一步在整个 Atlas 中的位置

Atlas 的大规模主体仍是 15 个来源、51 个研究级对比和完整开放候选池。本轮没有把课题缩回一个 IBD 小案例；它是在全局检索之后回答一个决定性问题：

> 当一个研究级 IBD 联系在公共响应检索中获得支持时，支撑该联系的化学边中有多少真的有资格被视为同一实体，又有多少只是共质量、相似谱或同一离子家族？

这一资格检查是 Atlas 从“研究级检索分数”进入可解释生物学证据的必要步骤。它不能替代 51 个对比的全局结果，也不能单独证明 unknown 或自研算法增量。

## 二、完全表型盲的化学证据

### 2.1 精确输入谱图

13 对源实体的谱图按实体数字 ID 精确定位到 MGF 中的 `SCANS`：

- 成人：`data/reference/contrast_atlas_ibd_cross_study_20261014/adult_ibd200_consensus.mgf`，SHA256 `6aeba8b4c1e9cc94e127985cc18bde9a4d1df8c99030d470d4cec888bc5465e6`；
- 儿童：`data/reference/contrast_atlas_ibd_cross_study_20261014/pediatric_veoibd_consensus.mgf`，SHA256 `c628ce08fc8215942469ec8fb193c6b3c189484032f83cc9ba9b190281c64fd1`。

成人原始共识谱与修正 PEPMASS 的应用谱在这 13 个 scan 上峰列表一致。冻结跨研究分数是 canonical WSE：同前体 10 ppm、0.02 Da 峰容差、互惠 Top-1。结果表另外给出从精确原始峰重算的 sqrt-cosine；它只是表型盲诊断，不能冒充冻结模型结果。

### 2.2 共同身份政策

当前共同身份政策使用 GNPS Gold/Silver positive `[M+H]+` 候选：前体 10 ppm、WSE `>=0.70`、匹配峰 `>=6`。只有两端均过政策且全 InChIKey 相同，才允许在当前 development 身份轨中条件性合并为同一 observed-ion 假说。

`library_supported` 仍不是标准品确认。库中的名称不能自动传播给 unresolved 端，也不能把共质量异构体或不同加合物写成同一个已验证分子。

## 三、13 个源实体对逐项裁决

| 成人→儿童实体 | 前体 ppm | 跨谱 WSE | 原始峰 sqrt-cosine / 匹配峰 | 共同库证据 | 裁决 |
|---|---:|---:|---:|---|---|
| `1816→40840`，m/z 283.263 | 0.177 | 0.729 | 0.678 / 23 | 两端过政策、全键相同 | 条件允许同一 observed-ion 合并；结构名不确认 |
| `928→13473`，m/z 305.186 | 0.885 | 0.783 | 0.711 / 27 | 两端低分候选的全键和平面键均冲突 | **禁止合并** |
| `1431→33374`，m/z 371.258 | 0.054 | 0.849 | 0.815 / 44 | 同一候选但两端均低于政策 | 禁止合并 |
| `1463→35038`，m/z 375.289 | 0.346 | 0.915 | 0.764 / 78 | 两端过政策、全键相同 | 条件允许同一 observed-ion 合并；结构名不确认 |
| `1432→33375`，m/z 389.269 | 0.308 | 0.878 | 0.863 / 86 | 同一候选但两端 WSE 均低于 0.70 | 禁止合并 |
| `1970→40190`，m/z 411.362 | 0.851 | 0.259 | 0.272 / 10 | 仅儿童端有低于政策的候选 | 禁止合并 |
| `1849→41300`，m/z 427.357 | 0.070 | 0.770 | 0.747 / 9 | 同一候选但两端均低于政策 | 禁止合并 |
| `1041→16870`，m/z 595.348 | 0.672 | 0.522 | 0.537 / 29 | 当前严格库中双方均无 10-ppm 候选 | 禁止合并；只保留历史 stercobilin-like 假说 |
| `1304→29583`，m/z 597.364 | 2.042 | 0.561 | 0.845 / 13 | 双方均无 10-ppm 候选 | 禁止合并 |
| `1284→28832`，m/z 617.330 | 0.340 | 0.751 | 0.779 / 10 | 名义候选全键相同，但两端 WSE=0、0 匹配峰 | 禁止合并 |
| `1251→27774`，m/z 817.582 | 1.297 | 0.441 | 0.570 / 61 | 仅儿童端达到政策 | 禁止合并 |
| `1261→27793`，m/z 834.609 | 1.378 | 0.353 | 0.366 / 9 | 仅儿童端达到政策 | 禁止合并 |
| `1256→27770`，m/z 839.564 | 1.537 | 0.578 | 0.700 / 2 | 双方均无候选 | 禁止合并 |

汇总：

- `2/13`：当前冻结策略下条件性 observed-ion 合并；
- `2/13`：只有单侧命名锚点；
- `4/13`：双方指向同一但低于政策的候选；
- `1/13`：两端低分候选冲突；
- `1/13`：只有单侧低分候选；
- `3/13`：没有合格的 10-ppm 库候选；
- `0/13`：标准品或 MSI level-1 身份。

高 WSE 不是身份许可。例如 m/z 389.269 的跨研究 WSE 为 0.878、原始峰 sqrt-cosine 为 0.863，但两端库证据仍未过冻结政策；它只能作为高优先级谱学假说。

## 四、13 对实际上折叠为 9 个候选分析物家族

离子家族折叠完全不读取疾病效应，只使用同一 acquisition mode、RT 差 `<=0.03 min`，以及预先列出的正离子质量差（H2O、NH4-H、Na-H、K-H；质量差残差 `<=0.01 Da`）。

### 4.1 三个明确的多离子家族

| 家族 | ST001000 正离子观测 | 化学依据 | 合并的源实体对 |
|---|---|---|---|
| F1，neutral mass 约 594.342 | C8 `595.3494@3.21`、`617.3314@3.21`；另有 HILIC `617.3279` | Δ=21.982000 Da，与 Na-H 理论值差 0.058 mDa | `1041↔16870`、`1284↔28832` |
| F2，neutral mass 约 816.574 | C8 `817.5808`、`834.6087`、`839.5631`，均 RT 4.02 | 分别兼容 `[M+H]+/[M+NH4]+/[M+Na]+`；中性质量最大差约 1.35 mDa | `1251↔27774`、`1261↔27793`、`1256↔27770` |
| F3，neutral mass 约 388.261 | C8 `371.2578@3.74`、`389.2685@3.73` | Δ=18.010700 Da，与 H2O 理论值差 0.135 mDa；兼容 `[M+H-H2O]+/[M+H]+` | `1431↔33374`、`1432↔33375` |

另外，m/z 305.185 和 617.33 各自在 HILIC/C8 中重复出现，但已经属于同一源实体对，不增加源假说数。

411.3606@1.69 与 427.3556@1.71 共洗脱且相差约 15.995 Da，可作为氧化关系候选；它不是常见正离子加合物，因此没有被合并成同一个分子家族。595.3494 与 597.3618 来自不同 LC 模式，也没有被按约 2.012 Da 差强行合并。

只在 13 个优先源实体对内部折叠时，9 个工作家族为：

- unresolved：6；
- known：2；
- mixed known/unresolved：1。

这里的“9”仍是分析物家族假说，不是 9 个已鉴定代谢物。合理报告范围是 **9–13**。当同一规则扩展到全部 101 个可测试实体对时，一条严格 known 边还会与一条未通过筛选的 unresolved 水损失边归入同一 mixed 家族，因此完整分母中的 9 个优先家族组成变为 unresolved 6、known 1、mixed 2。两种数字回答的分母不同，不能混写。

## 五、历史注释为什么不能替代当前严格证据

历史 GNPS `MQScore` 与当前冻结 WSE 不是同一分数，也不使用同一证据合同：

- m/z 595.348 两端都曾匹配同一 Bronze Stercobilin 记录，MQScore 0.755/0.917；Bronze 被当前 Gold/Silver 合同排除，因此只能写 `stercobilin-like`。
- 817/834/839 家族在历史结果中包含 cholic-acid、二聚体和 in-silico 胆汁酸候选；这些标签支持“值得做胆汁酸家族核验”，不支持三个独立胆汁酸身份。
- m/z 389.269 的成人历史 Gold 候选全 InChIKey 与当前共同库候选冲突；儿童端是从谱传播出的无键 in-silico 候选。
- 即使是当前两条策略允许合并的源实体对，历史标签也与当前共同库标签不一致或证据不足。因此只能合并 observed-ion 响应假说，不能传播结构名称。

原 ST001000 论文已经报告了 IBD 中的胆汁酸和 tetrapyrrole 变化。这构成生物学背景，但不能反过来证明本轮某一 m/z 的身份；否则会形成按预期机制选择结构的循环论证。

## 六、对 unknown 增量的正式裁决

原 101 个可测试源实体对中，严格三拟合通过率为：

- unresolved：`9/82`；
- known：`2/12`；
- mixed：`2/7`。

unresolved 对 known 的单侧 Fisher 检验为 OR `0.616`、`p=0.855`，没有 unknown 富集证据。

离子家族折叠后，13 个阳性源实体对变成 9 个家族。为避免只折叠阳性分子，本轮已经把同一规则应用到全部 101 个可测试源实体对：

| 分母 | unresolved | known | mixed | unresolved vs known 单侧 Fisher |
|---|---:|---:|---:|---:|
| 源实体对 | 9/82 | 2/12 | 2/7 | OR 0.616，`p=0.855` |
| 表型盲家族 | **6/72** | **1/10** | **2/8** | OR 0.818，`p=0.798` |

完整折叠得到 90 个候选分析物家族、13 条候选离子关系边和 9 个多实体对家族；最大家族含 3 个源实体对。家族定义没有读取响应，且 known/unresolved 使用同一规则。

这仍是冗余敏感性分析，不是 family-level replicated-findings FDR：上游 q 值在 feature×diagnosis 层计算，一个家族只要有任一成员通过就记为优先家族。尽管如此，结论没有因统一折叠而改变。

因此当前允许的结论仍是：

> 未定结构实体参与了 IBD 响应假说的优先化，但本面板没有证明 unresolved 相对 known 的条件增量。

## 七、这对 Atlas 科学主线意味着什么

这不是 Atlas 的失败，也不是一项新疾病机制。它完成了大规模检索之后最缺的一层：把“研究间响应联系”拆成可合并同一实体、不可合并的谱学假说以及离子家族冗余。

当前三条可保留的结果是：

1. 成人 IBD、儿童 VEO-IBD 与 ST001000 两个参与者独立队列之间存在一组方向一致的质量匹配响应候选；
2. 这些候选高度冗余，13 个源实体对最多对应约 9 个分析物家族；
3. 只有 2 个源实体对达到当前共同谱库政策，且 0 个达到标准品确认；大多数 unresolved 候选必须保持弃权。

最有价值的后续验证对象不是任意挑 13 个峰，而是三个多离子家族：595/617、817/834/839、371/389。它们分别具备明确加合物/水损失关系、跨队列方向信息和足够强的源谱证据，最适合开展样本链接 MS/MS、保留时间和标准品/胆汁酸专库核验。

在获得这些证据前，论文只能把本结果写成**跨队列响应候选优先化与身份风险控制**。不得写成“13 个代谢物复现”“新胆汁酸机制”或“unknown 已证明优于 known”。

## 八、可复算产物

脚本：

- `tasks/adjudicate_contrast_atlas_prism_strict13_identity.py`

结果目录：

- `data/validation/contrast_atlas_prism_strict13_identity_adjudication_v1_20261030/strict13_identity_evidence.tsv`
- `data/validation/contrast_atlas_prism_strict13_identity_adjudication_v1_20261030/strict13_candidate_ion_relation_edges.tsv`
- `data/validation/contrast_atlas_prism_strict13_identity_adjudication_v1_20261030/strict13_candidate_analytical_families.tsv`
- `data/validation/contrast_atlas_prism_strict13_identity_adjudication_v1_20261030/strict13_candidate_analytical_family_summary.tsv`
- `data/validation/contrast_atlas_prism_strict13_identity_adjudication_v1_20261030/strict13_unique_feature_observations.tsv`
- `data/validation/contrast_atlas_prism_strict13_identity_adjudication_v1_20261030/report.json`
- `data/validation/contrast_atlas_prism_strict13_identity_adjudication_v1_20261030/checksums.sha256`

完整 101 对的统一家族分母敏感性：

- `tasks/analyze_contrast_atlas_prism_family_level_increment.py`
- `data/validation/contrast_atlas_prism_family_level_increment_v1_20261030/retained_pair_family_assignments.tsv`
- `data/validation/contrast_atlas_prism_family_level_increment_v1_20261030/retained_family_summary.tsv`
- `data/validation/contrast_atlas_prism_family_level_increment_v1_20261030/retained_candidate_ion_relation_edges.tsv`
- `data/validation/contrast_atlas_prism_family_level_increment_v1_20261030/report.json`
- `data/validation/contrast_atlas_prism_family_level_increment_v1_20261030/checksums.sha256`

脚本通过 `py_compile`；全部输出 SHA256 已复核一致。
