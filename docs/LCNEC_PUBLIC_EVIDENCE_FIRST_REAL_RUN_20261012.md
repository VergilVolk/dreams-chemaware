# LCNEC 公共证据层首次真实运行：单源弃权与双族就绪清单

日期：2026-10-12
状态：正式 `development/consumed` 中间结果；预注册合同 `docs/LCNEC_PUBLIC_EVIDENCE_SPACE_M0_CONTRACT_20261011.md` 约束下产出

## 一句话

在 263 个表型盲 LCNEC QC ion-family 查询谱上，首个真实公共证据 edge table（GNPS 负库单源、双证据族、冻结阈值）已建成并首次运行证书引擎：**0/263 赋值（单源全弃权，符合设计），40/263 已达双证据族门、只差第二独立来源**。

## 数据、分母与角色

- 查询：263 条 HSST3n QC-qualified ion family 代表谱（负模式，表型盲选择，见 M0 合同）。
- 参考：`data/reference/unified_v3/unified_neg.mgf`，29,564 条 GNPS 来源负模式谱，29,008 条通过最少峰数/前体验证进入门控。
- 门控：前体 5 ppm；证据族：pinned `weighted_entropy_similarity`（tolerance 0.02 Da）+ MS2DeepScore 2.x 公共双极性模型（embedding API，余弦）。
- 阈值（冻结，非 LCNEC 校准 shard = GNPS identity-disjoint 对抗面板）：**跨分子式错误对 q99.9 分位** → WSE ≥ 0.645、MS2DeepScore ≥ 0.915。同分子式异构体在任何分数下不可分（残余错误 ≥98% 为同分子式），异构体消歧按设计交给多来源证书一致性。
- 角色：全部 `development/consumed`；证书输出明确不 claim 分子身份/反应/功能/生物关联。

## 冻结数字链

| 层级 | 数量 | 占 263 |
|---|---|---|
| 存在 5 ppm 质量兼容公共候选的 query | 147 | 55.9% |
| ≥1 证据族过冻结地板的 query | 64 | 24.3% |
| **≥2 证据族同时过地板的 query（双族就绪）** | **40** | **15.2%** |
| 获得证书的 query | 0 | 0% |

- 门控后 pair 总数 480；WSE ≥0.10 留 362，MS2DS ≥0.50 留 397；edge 表（WSE ≥0.30 或 MS2DS ≥0.70）563 行。
- 双族就绪单元 63 个（40 query），certificate_score 中位 0.557（IQR 0.423–0.698），其 `public_sources` 全部 =1——**唯一阻塞是第二独立来源**。
- 仪器 provenance：源 MGF 未记录（`not_recorded`），证书引擎按单一 bucket 处理。

## 判读

1. 单源弃权门在真实数据上工作正常：0 赋值不是失败，是引擎的设计行为。
2. 公共负库对 LCNEC 暗空间的**质量覆盖天花板**已被量化：即便第二来源就绪，双族门上限也只有 40/263（15.2%）；55.9% 的 query 连质量兼容候选都没有。M0-A 的 `median certified coverage < 0.20` 停止线在此数据上大概率触发——这是提前可见的诚实边界。
3. 解锁清单（服务器侧，本地不可达）：(a) MoNA-neg MGF（`data/models/mona_neg_full.mgf`，141 MB）作为第二独立来源补打分；(b) 85×263 EIC 矩阵（集群 run_2352292 的完整 `all_qc_family_eic_matrix.npz`）。二者齐备前不产出任何 M0 生物学数字。

## 机器可读锚点

- edge 表：`data/validation/lcnec_public_edges_gnpsneg_v1/`（edges.csv、query_universe.csv、report.json，sha256 在证书 report 内）
- 冻结阈值：`data/validation/lcnec_public_evidence_thresholds_v1/`（thresholds.json 全量 + thresholds_flat.json 引擎格式）
- 证书运行：`data/validation/lcnec_public_certificates_gnpsneg_v1/`（certificates.csv、unit_evidence.csv.gz、report.json 含全部输入 sha256）
