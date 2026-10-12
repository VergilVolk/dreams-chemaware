# ChemProp 跨设计 M1b 结果与实体门裁决

日期：2026-10-11  
证据角色：development / consumed  
泄漏状态：feature 对应不读取药物标签或丰度响应；药物标签仅用于对应冻结后的检索评价  
最终裁决：**同图 v4 实体层不通过增量门；relation transport 不启动；ChemProp 路线封存**

## 1. 本实验回答的问题

在没有天然 row ID 对应时，能否仅凭跨表可见的 m/z、RT 和 MS/MS，将
MSV000094899 的 12-drug T0/T2 响应迁移到 MSV000096724（共享 3 drugs）和
MSV000093571（共享 9 drugs），并比最简单的 10-ppm 质量对应更准确地找回同一药物？

三批数据来自同一 ChemProp 项目和同类 Q Exactive HF 平台。本结果是跨设计开发实验，
不是跨实验室、跨仪器或独立生物学外测。

## 2. 数据、分母和实现

- query 原始非空 MS/MS：8,055 features；冻结 QC 主分析：3,538 features；
- 生物学检索：12 drugs x 3 replicate queries = 36 queries；其中 9 个 query 为 3-way 检索，27 个为 9-way 检索；
- 响应量：`(Bac T2 - noBac T2) - (Bac T0 - noBac T0)`，不做 vehicle subtraction；
- candidate generation：precursor m/z ±10 ppm；
- raw cosine：sqrt-intensity、0.02 Da；WSE：`ms_entropy 1.5.3`、0.02 Da；
- 所有 feature 对应在药物标签和丰度响应不可见时建立；
- 对应冻结后，使用匹配 features 的稳健缩放响应向量作同药物检索；
- 主比较为相对 mass-nearest 的按 12 drugs 聚类 bootstrap，分母始终为 36 queries。

工件：

- `data/validation/chemprop_cross_design_overlap_audit_v5_20261011/report.json`
- `data/validation/chemprop_cross_design_overlap_audit_v5_20261011/candidate_edges_10ppm.csv.gz`
- `data/validation/chemprop_cross_design_response_v4_20261011/report.json`
- `data/validation/chemprop_cross_design_response_v4_20261011/retrieval_results.csv`
- `data/validation/chemprop_cross_design_response_v4_20261011/feature_pair_manifests.csv.gz`

v1/v2 先在全部 nonempty-MS/MS reference 中选谱学 Top-1、再过滤 endpoint QC；v3 又让 mass 基线保留了
QC 合格但无非空 MS/MS 的 reference。三者候选图均不完全一致，作为失效历史保留，不得引用为正式比较。
v4 固定所有方法共同使用 `10 ppm ∩ endpoint QC ∩ nonempty MS/MS` reference graph。

## 3. 对应层结果：已知物准确率提高，但不能外推到未知物

| reference design | 8,055 中有 10-ppm candidate | 3,538 QC 中有 candidate | cosine >= 0.70（QC） | 已知物子集 cosine Top-1 | 已知物子集 WSE Top-1 |
|---|---:|---:|---:|---:|---:|
| 23-drug | 3,188/8,055 (39.58%) | 1,750/3,538 (49.46%) | 773/3,538 (21.85%) | 89/117 (76.07%) | 95/117 (81.20%) |
| 45-drug | 3,783/8,055 (46.96%) | 2,063/3,538 (58.31%) | 1,345/3,538 (38.02%) | 121/164 (73.78%) | 131/164 (79.88%) |

WSE 在不完整的 library-supported 子集上比 cosine 分别多答对 6 和 10 个 query。
这些分母偏向已知、可注释化合物，只能说明跨设计谱学对应可行，不能估计未知实体身份准确率。
10 ppm、cosine threshold 和 mutual Top-1 都只是规则通过，不是真值。

## 4. 生物学主终点

分层随机期望为 `(9×1/3 + 27×1/9)/36 = 16.67%`。正式同图结果如下：

| truth-blind feature mapping | pair features 23 / 45 | R@1（36） | MRR（36） | R@1 相对 mass | R@1 95% CI | MRR 相对 mass | MRR 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|
| mass nearest 10 ppm | 1,520 / 1,074 | 7/36 (19.44%) | 0.4249 | reference | — | reference | — |
| mass + RT <= 0.5 min | 1,235 / 904 | 5/36 (13.89%) | 0.3829 | -5.56 pp | [-19.44,+5.56] pp | -0.0421 | [-0.1155,+0.0258] |
| cosine Top-1, no threshold | 1,520 / 1,074 | 7/36 (19.44%) | 0.4106 | 0.00 pp | [-8.33,+8.33] pp | -0.0144 | [-0.0528,+0.0257] |
| cosine >= 0.70 | 689 / 589 | 10/36 (27.78%) | 0.4588 | +8.33 pp | [-2.78,+19.44] pp | +0.0339 | [-0.0267,+0.0953] |
| cosine >= 0.80 | 475 / 450 | 7/36 (19.44%) | 0.4334 | 0.00 pp | [-11.11,+11.11] pp | +0.0085 | [-0.0514,+0.0650] |
| mutual cosine >= 0.70 | 606 / 466 | 8/36 (22.22%) | 0.4240 | +2.78 pp | [-8.33,+13.89] pp | -0.0009 | [-0.0726,+0.0780] |
| WSE Top-1 | 1,520 / 1,074 | 7/36 (19.44%) | 0.4230 | 0.00 pp | [-8.33,+8.33] pp | -0.0020 | [-0.0437,+0.0384] |

没有任何 MS/MS 方法达到预注册门：相对 mass-nearest 的 R@1 或 MRR 药物簇 CI 下界均未严格大于零。
cosine 0.70 有最好的点估计，但这是合同中预列的多个方法之一，不能因点估计最好而升级为成功。
WSE 的已知物对应准确率更高，却没有转化为生物学响应检索增量，这是本轮最重要的机制性负结果。

## 5. 预设 high / low 分层

| method | high signal：27 queries R@1 / MRR | low signal：9 queries R@1 / MRR |
|---|---:|---:|
| mass nearest | 25.93% / 0.4900 | 0.00% / 0.2299 |
| cosine >= 0.70 | 33.33% / 0.5184 | 11.11% / 0.2799 |
| WSE Top-1 | 25.93% / 0.4841 | 0.00% / 0.2396 |

low-signal 的 9-way R@1 对 cosine 0.70 为 11.11%，等于机会水平；WSE 为 0%。
所以当前信号更接近强药物转化线索，而不是已证明可迁移的、分散的内源代谢响应。

## 6. 负对照与限制

每个 method x batch 都执行 200 次独立 feature-alignment permutation 和 200 次 reference-drug-label permutation。
正式主比较的 paired CI 均跨零。23-drug 只有 3 个候选和 3 个药物，
随机区间很宽，不能单独支撑结论。

本实验没有接入但不再为该场景追加开发的项目资产：

1. official DreaMS 与 Noise V1 的同图 checkpoint embedding；本地缺少两个冻结 checkpoint，不能伪造替代；
2. 冻结 P2b；它需要 DreaMS 通道，且原合同是候选参考谱重排，须在同图 embedding 完成后原样计算，不能把 WSE 或 cosine 冒充其 DreaMS 通道；
3. ChemAware 局部证据的合格消融；
4. ChemProp/ChemProp2 官方 edge/decoy-FDR 输出的任务对齐比较；
5. 跨设计的母药及母药网络成分排除、no-match 错误率和未知身份真值。

## 7. 科学裁决与下一步

**实体门失败。** 当前可以保留的结论是：跨设计 MS/MS 能建立一批可信候选对应，WSE 在已知物子集上优于 raw cosine；
但截至 WSE，这种改善没有产生可重复的药物响应恢复增量。不得启动 relation transport，不得把共享碎片关系称为反应或机制，
也不得宣称通用 Atlas 能力成立。

本轮至此封存。若算法 coder 日后已经生成完全冻结、无需为本场景训练或调参的 official/Noise/P2b 分数，
可以一次性加入同一 edge 表作补充消融；它们不是继续占用生物学方法研发时间的前置条件。
生物学主线转向无标签也能成立的公共谱学证据空间，并在非药物真实研究中检验公共参考信息的条件增量。

这项结果不否定总的 structure-agnostic 方法目标；它否定的是“较好的谱学实体对应会自动产生生物学迁移增量”这一捷径。
