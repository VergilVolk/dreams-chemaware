# GLM：Noise/P2b 文章线 GNPS 统一基准预注册协议

日期：2026-10-03
性质：**预注册**。在 `run_noise_gnps_article_benchmark_1gpu.sbatch` 出结果之前冻结方法矩阵、指标、端点与声明规则。此后任何改动必须以修订条目追加，不得回改。

---

## 1. 冻结资产（全部已封存，禁止再动）

| 资产 | 内容 | 凭证 |
|---|---|---|
| GNPS Gold/Silver 10ppm benchmark v1 | 329,607 谱 MGF + manifest；identity-disjoint 10,995 queries/175,171 对；formula-disjoint 5,261/47,724；两面板并用 52,871 谱；[M+H]+ | `checksums.sha256` + `gnps_pair_score_cache.benchmark_fingerprint` 逐字节校验 |
| official / Noise V1 embeddings | run 2347472 的 `official_embeddings.npz` / `v1_embeddings.npz` | SHA256 链 |
| ms_entropy 实现 | pinned 官方 wheel 1.5.2（cp311），作业内 node-local 安装 | 文件名即版本 |
| 评估协议 | molecule-max 聚合、strict ties 计负、公式簇配对 bootstrap（10,000 次，族 24 假设） | `evaluate_gnps_gold_silver_10ppm_embeddings.py` 现行冻结版 |

**已知复测一致性**：V1 vs official 在本图上已有测量（run 2347472：identity +1.2005pp CI [+0.460,+1.890]；formula +1.2545pp CI [+0.229,+2.299]）。本基准用同一图同一协议重推，**复测值若与已测值偏离超过舍入差即为实现事故**，须先查错再谈结果。

## 2. 方法矩阵与资格分层

| 层 | 方法 | 训练语料 | GNPS 重叠 | 榜单资格 |
|---|---|---|---|---|
| S1 经典谱学（无训练） | greedy cosine；modified cosine；weighted spectral entropy（pinned ms_entropy 1.5.2） | 无 | 无 | 严格 OOD 榜 |
| S1b P2b 单通道（诊断用） | sqrt_cosine；unweighted entropy（P2b 冻结特征）；neutral-loss sqrt cosine | 无 | 无 | 严格 OOD 榜（标注"单通道诊断"） |
| S2 DreaMS 家族 embedding | official DreaMS（微调用公开语料，非本项目训练）；Noise V1（本项目，MassSpecGym 训练） | 见各自语料 | 无身份/公式重叠（benchmark 构建期排除） | 严格 OOD 榜 |
| S3 冻结候选重排器 | P2b(official)；P2b(V1)——固定权重 0.10/0.00/0.10/0.80、absolute 归一化（DreaMS 通道 (x+1)/2）、support≥1、advantage≥0 | P2b 权重来自 MassSpecGym 公式隔离 OOF | 无 | **单独信息层**，绝不与 S1/S2 混排 |
| S4 公开神经基线（未入本期） | Spec2Vec；MS2DeepScore 2.0 | 公开权重均用过 GNPS 语料 | **有** | 只进"现成实用榜"，不进严格 OOD 结论；需先落权重到 `third_party/` 并经 `extend_gnps_article_score_bundle.py` + `gnps_pair_score_cache` 契约接入 |
| 排除 | BioAware 全线 | — | — | GNPS 无样本上下文变量，硬塞即伪 |

## 3. 指标（每方法 × 每面板，全部预指定，不做挑选）

- 检索：Recall@1/2/3/5/10/20、MRR、mean/median rank、near 子集 Recall@k、正例对最优负例 margin、top1-top2 gap（原始+带符号）。
- 判别：macro-query AUROC/AUPRC、micro-candidate AUROC/AUPRC、**GNPS 10ppm pooled pairwise AUROC/AUPRC**（DreaMS 论文同款数学，命名必须带 "GNPS"，禁止与论文 NIST20 0.85 直接比大小）。
- 安全：corrected / introduced / C:I 比值 / risk-net(λ=2) / 严格平局数。
- 配对：对 official_dreams 的公式簇配对 CI（recall@1、MRR、macro AUROC/AUPRC、margin、signed gap）。
- 曲线：逐点 ROC 与 PR 曲线表（`curves_*.csv.gz`）供论文作图。

## 4. 预指定端点与判定规则

- **E1（一致性闸）**：复测 V1 vs official 的 recall@1 delta 落在已测 CI 内。不过 → 实现事故，全表作废查错。
- **E2（主声明候选 1）**：Noise V1 > official_dreams 于两面板 recall@1（点估计为正且 CI 下界>0）。成立 → "本项目 encoder 在模型盲外部基准上显著优于官方 DreaMS"。
- **E3（主声明候选 2）**：P2b(V1) vs Noise V1 于两面板 recall@1。若两面板均正 → 重排增益跨数据集；若仅一面正或近核有害复现 → 声明降级为"数据集特定"，如实报告。
- **E4（定位声明）**：DreaMS 家族最优 vs S1 经典最优（两面板 recall@1 与 pooled AUROC）。此为"学习表示是否优于经典相似度"的领域定位句。
- **排序规则**：headline 排序按 identity-disjoint recall@1；分层内排序；S3 永远单独成层展示。

## 5. 声明语言红线

1. GNPS 数值不得表述为 NIST20 复现；0.94 对 0.85 的跨数据集比较禁止出现。
2. S4 未落榜前，禁止写"优于全部近年方法"；只可写"优于经典谱学基线与 DreaMS 家族内部基线"。
3. 任何方法不得因 GNPS 表现回改实现或超参（无选择自由度：权重冻结、容差 0.02Da 固定、top-100 固定）。
4. near 子集、formula-disjoint 面板的表现必须与主面板同表呈现，不得只报有利面板。

## 6. 停止条件

- E1 失败 → 停，查实现。
- 作业内四类合同测试任一失败 → 停。
- 经典分数出现非有限值/NaN（缓存层已 fail-closed）→ 停。
- 结果落盘后本协议不得修改；修订仅以"修订 N"追加并注明理由。

## 7. 产出物清单（论文侧）

1. 六联主图（a 双面板 Recall@1 条形；b Δvs official 森林+CI；c recall@k；d pooled ROC；e PR；f corrected/introduced 散点+λ=2 线）——`plot_noise_gnps_article_benchmark.py`（schema 对齐 `evaluate_noise_gnps_article_benchmark.py` 输出）。
2. formula 面板曲线补充图。
3. 方法×指标汇总表 CSV（含层、语料、重叠风险列）。
4. `evaluation/report.json` 与全部 `paired_*.csv.gz`、`curves_*.csv.gz`、`queries_*.csv.gz` 原始留存。
