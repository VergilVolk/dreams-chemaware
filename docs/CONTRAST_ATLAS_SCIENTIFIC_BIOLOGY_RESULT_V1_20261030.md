# Metabolomic Contrast Atlas：科学结果与主张边界 V1

日期：2026-10-30

数据角色：**全部 development / consumed；没有封存外部生物学验证**

研究对象：51 个研究级对比、同前体 10 ppm 身份假说轨；不同质量化学关系轨不纳入本结果

## 一、结论先行

当前 Atlas 已经得到一项可复现的**方法学正控**，但尚未得到新的疾病机制：

> 以小鼠高脂饮食尿液响应为查询，在 50 个其他研究级对比中，人类无 T1D 背景下的肥胖尿液响应在 all 和 unresolved 世界均为 similar-face 第 1；known-only 世界不能形成校准答案。

这证明未定结构的 MS/MS 实体能够携带研究级响应检索信息。它不能被写成新发现，原因是原始 2026 年论文已经联合使用 `MSV000097426` 和 `MSV000084112`，并直接报告了同一跨物种肥胖联系及 m/z 396.1684 信号。当前结果的资格是：**在更大的开放候选池中，结构未知信息找回了一个已发表联系的 consumed positive control**。

目前没有证据支持以下更强主张：

- 50 个新化学实体已得到双端验证；
- 40% 的新增实体是葡糖醛酸苷；
- 三个 N-acyl amino acid / metanephrine / N-acyl taurine 通路已被鉴定；
- Full23 或其他自研统一算法已改善 Atlas；
- Atlas 已产生一项独立的新疾病机制。

## 二、先纠正会改变科学结论的错误

### 2.1 相似与相反响应不能机械二选一

旧汇总在 query-level maxT p 值并列时继续选择一个 `response_face`，导致大量查询被机械标成 opposite。相似和相反两面的统计量必须独立保留；旧的“45/49 opposite”及其衍生故事失效。

### 2.2 `MSV000086360` 不具备当前显著性推断资格

Atlas 登记表把根、茎、叶组织切片文件当成 96 个独立生物单位。原论文说明成年植株被切成大量组织段，并明确提示只有极少的植株级重复，不能把这些切片当作独立植株。该来源恰好是旧全局结果的强 hub，因此所有以它为目标或查询的 beta、SE、置换显著性和跨项目联系均撤回，直到获得 plant-ID crosswalk 并按植株聚类。

原论文：<https://pmc.ncbi.nlm.nih.gov/articles/PMC9246715/>

### 2.3 HFD—肥胖的“50 条双端诊断边”不是双端 FDR 结果

主分支后续解释使用 2026-10-27 旧 WSE 图的 771 条边，并用两端 `|beta / SE| >= 1.96` 定义 50 条“diagnostic”边。该阈值是未校正正态近似，不是两端 BH FDR。

在 2026-10-29 完整候选轴与冻结总账中：

| 项目 | 结果 |
|---|---:|
| all-world 互惠身份假说 | 596 |
| similar / opposite 非零贡献边 | 319 / 277 |
| 两端未经多重校正 `|z| >= 1.96` | 37 / 16 |
| 两端 `q <= 0.05` | **1 / 1** |

因此，研究级响应富集可以成立，但不能把 37 或旧图的 50 条边写成 37/50 个已验证实体。

### 2.4 精确质量枚举不等于结构或通路鉴定

三组约 `+2.016 Da` 的前体对只能说明候选质量差接近 `H2`。在两项研究各自内部，这三组谱图的 weighted spectral entropy similarity 仅为：

| 质量对 | 小鼠内部 WSE | 人类内部 WSE |
|---|---:|---:|
| 198.112 / 200.128 | 0.395 | 0.386 |
| 314.179 / 316.194 | 0.579 | 0.604 |
| 336.217 / 338.232 | 0.395 | 0.625 |

这些数值不足以确认同一化学家族。现有分子式枚举还预设 `[M+H]+`、有限 CHNOS 元素范围和价态规则；即使某个分子式在该受限搜索内唯一，也不能区分结构异构体，更不能据此宣布 N-acyl amino acid、metanephrine 或 N-acyl taurine 通路。三条通路只能保留为待验证假说。

## 三、完整候选轴上的严格结果

### 3.1 小鼠 HFD 尿液查询人类肥胖尿液

查询：`MSV000097426::mouse_hfd_vs_nfd`，6 HFD vs 6 NFD 小鼠。

候选：`MSV000084112::obesity_without_diabetes`，无 T1D 人群内 obese vs normal weight，12 vs 12；模型同时调整 sex、age 及 obesity-by-diabetes 设计项。

候选全集：50 个其他研究级对比。

| 化学世界 | 互惠边 | similar z | similar rank | query maxT p | opposite z | opposite rank |
|---|---:|---:|---:|---:|---:|---:|
| all | 596 | **4.388** | **1/50** | **1/257 = 0.003891** | -1.718 | 42/50 |
| known-only | 17 | 0.785 | 7/50 | 0.996109 | -1.981 | 45/50 |
| unresolved-only | 575 | **4.398** | **1/50** | **1/257 = 0.003891** | -1.250 | 36/50 |

其他证据：

- similar 总强度 109.149，置换均值 86.992、SD 4.999；
- strongest edge 仅占 similar 总强度 0.825%，effective edge count 202.36，说明不是单峰劫持；
- unresolved–unresolved 贡献约占 similar 强度 95.1%；
- 但只有 1 条 similar 边达到双端 `q <= 0.05`，研究级结果主要来自许多中弱证据的累积。

**科学裁决**：这是目前最清楚的 unresolved 条件增量，但仍是 WSE 开发结果，不是 Full23 增量，也不是封存外部确认。

### 3.2 已发表 m/z 396.1684 锚点被正确找回

完整轴包含：

- 人：`MSV000084112::A009834`，m/z 396.168307；
- 鼠：`MSV000097426::A003244`，m/z 396.168715；
- 质量差 1.03 ppm，跨研究 WSE 0.9518；
- 两端 MS/MS 均含 m/z 220.1366，前体中性丢失分别为 176.0317 和 176.0321 Da；两端还包含约 m/z 113.023 和 85.028 的葡糖醛酸相关碎片。

这为该边的 glucuronide 身份提供直接谱学支持。不过在人类端，本地 obesity effect 为 `beta=+0.832, q=0.545`；小鼠端为 `beta=+1.532, q=2.48e-5`。所以该边方向一致，却不是本地双端 FDR 边；其贡献约为 similar 总强度的 0.71%。

原论文已经报告同一 m/z 396.1684:220.1366 葡糖醛酸苷—苷元对，以及该葡糖醛酸苷在人类肥胖尿液中升高：<https://www.nature.com/articles/s41467-026-73398-1>

因此它是可靠的**已发表特异锚点**，不是 Atlas 新发现。

### 3.3 “广泛 glucuronidome”目前只得到有限直接支持

对 319 条 similar 边逐端回查代表 MS/MS，并以 176.0321 ± 0.02 Da 中性丢失作为保守直接证据：

| 子集 | 边数 | 任一端有 NL176 | 两端均有 NL176 |
|---|---:|---:|---:|
| 全部 similar 边 | 319 | 32 | **8** |
| 两端未校正 `|z| >= 1.96` | 37 | 2 | **0** |
| 两端 `q <= 0.05` | 1 | 0 | **0** |
| 前体 m/z >330 | 113 | 28 | **8** |
| 前体 m/z <=330 | 206 | 4 | **0** |

这是保守筛查，不能排除没有明显 NL176 的葡糖醛酸苷；但它足以证明“m/z >330 即葡糖醛酸苷”“20/50 为葡糖醛酸苷”没有证据。当前只有 8 条 similar 边在两端都出现直接 NL176，其中 m/z 396.1684 是最清楚的一条。

## 四、第二个可信响应家族：IBD

### 4.1 成人 IBD200 ↔ 儿童 VEO-IBD

`IBD200_GROUP::adult_all_ibd_vs_healthy` 查询 `MSV000097610::pediatric_veoibd_vs_healthy`：

| 世界 | 互惠边 | similar z | rank | query maxT p |
|---|---:|---:|---:|---:|
| all | 232 | **3.379** | 2/50 | **0.007782** |
| known-only | 17 | 2.163 | 1/50 | **0.011673** |
| unresolved-only | 206 | 2.395 | 4/50 | 0.085603 |

同物种、同粪便基质的 IBD 响应得到支持，但 known-only 已经能够找回，因此这条联系不能证明 unknown 独立增量。成人 CD/UC、疾病位置、儿童 VEO-IBD 和 5-ASA 等治疗暴露仍构成异质性；只能称疾病家族响应复现，不是共同机制确认。

### 4.2 `MSV000096564` ↔ VEO-IBD 是优先假说，不是独立确认

`baseline_ibd_vs_control` 查询 VEO-IBD 时，all `z=3.236, p=0.038911`，known-only `p=1`，unresolved-only `z=4.064, p=0.003891`。这是很强的 unresolved 候选。

但官方临床试验 `NCT05973500` 是单组、开放标签、42 天地中海饮食 Crohn 研究，不招募健康志愿者；当前本地 11 Crohn + 11 control 的 control 来源、招募和批次 crosswalk 未被公开材料解释。补齐 provenance 前，这一联系只能作为开发假说，不能承担第三个独立 IBD 队列确认。

<https://clinicaltrials.gov/study/NCT05973500>

## 五、必须保留的负结果与弃权

- 小鼠 HFD 尿液 ↔ 小鼠 HFD 血浆：female 和 male 两个对比的 similar/opposite query-level maxT 均未通过。相同干预不保证跨 biofluid 同向，不能为了“找回 HFD”调阈值。
- 成人 IBD200 ↔ `MSV000096564`：两面均无信号。
- industrialization feces ↔ VEO-IBD：两面均无信号。
- obesity saliva ↔ obesity urine 的强 opposite 汇总缺少双端显著实体且生物液、T1D、抑郁分层均不同，不进入生物学故事。
- COVID、skin AD、lanthanide 和多数独有设计当前没有同条件独立来源；它们应成为真实 no-match / abstention 任务，而不是被不相关候选强迫产生答案。
- 旧 different-mass relation 轨由 fragment-cosine mutual top-1 构成，出现大量跨质量且无化学资格的边；当前不得用于科学解释。

## 六、当前 Atlas 真正做到了什么

### 已做到

1. 在 51 个已消耗研究级对比的开放候选池中，使用未定结构 MS/MS 响应找回了原论文已经建立的小鼠 HFD—人类肥胖尿液联系。
2. 该答案在 unresolved-only 世界为 rank 1 且通过 query-level maxT；known-only 不通过，形成了明确的 unknown 条件增量正控。
3. 同一分析正确显示 HFD 跨 biofluid 不复现、成人 IBD 队列并非彼此都相似，说明 Atlas 不应只按疾病名匹配。
4. 每条研究级联系可以追到具体实体、谱图、效应方向和不确定性，并能发现 WSE 身份误连和来源设计错误。

### 尚未做到

1. 没有封存独立数据确认一个新增研究级答案。
2. 没有证明 Full23、Noise 或 ChemAware 相对 WSE 增加正确的生物学答案。
3. 没有得到可用于 effect pooling 的大批已确认同一实体；绝大多数仍是身份假说。
4. 没有合格的不同质量化学关系轨。
5. 没有确认三条新代谢通路或新疾病机制。

## 七、论文中允许的核心表述

允许：

> 在一个包含 51 个已消耗研究级对比的开发 Atlas 中，未定结构的 MS/MS 实体使小鼠高脂饮食尿液响应在 50 个候选中找回原论文已报道的人类肥胖尿液响应；known-only 分析未形成校准答案。该结果证明结构未知信息可以增加研究级响应检索能力，但仍需独立数据与统一算法比较。

禁止：

- “Atlas 发现了 HFD—人类肥胖的新跨物种机制”；
- “50 条新实体得到验证”；
- “三个代谢通路已确认”；
- “40% 是葡糖醛酸苷”；
- “Full23 已改善生物学检索”；
- “51 个对比产生了 49 个正确答案”。

## 八、数据与证据定位

- 完整候选得分：`data/validation/contrast_atlas_complete_identity_wse_open_retrieval_v3_20261029/open_local_edge_candidate_scores.tsv.gz`
- 实体贡献总账：`data/validation/contrast_atlas_connection_evidence_ledger_release_v1_20261029/connection_edge_contributions.tsv.gz`
- 响应对象：`data/validation/contrast_atlas_open_response_collection_v9_frozenbase_chemprop_20261023/response_entities.tsv.gz`
- 小鼠代表谱：`data/validation/msv000097426_hfd_ms2_anchored_20261014_v1/representative_spectra.mgf`
- 人类代表谱：`data/validation/msv000084112_matched_ms2_anchored_20261014_v1/representative_spectra.mgf`

本文件是科学结果裁决，不改变冻结数据与算法输出。主分支后续的 HFD 化学解释必须以本边界为准。
