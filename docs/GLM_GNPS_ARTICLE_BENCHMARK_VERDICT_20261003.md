# GLM：GNPS 文章基准第一轮判决书（run 2349091）

日期：2026-10-03
对应预注册：`GLM_GNPS_ARTICLE_BENCHMARK_PREREGISTRATION_20261003.md`（结果落盘前冻结，未回改）
运行：`data/validation/noise_gnps_article_benchmark_run_2349091/`（10 方法 × 2 面板 + 全部曲线/查询表 + 已渲染图）

---

## 1. 端点裁决

| 端点 | 判定 | 证据 |
|---|---|---|
| **E1 一致性闸** | **通过** | V1 vs official 复测 identity +1.2005pp（与 run 2347472 点估计完全一致，CI [+0.50,+1.91] vs [+0.46,+1.89]，差异仅 bootstrap 种子） |
| **E2 主声明** | **成立，可声明** | Noise V1 > official：identity +1.20pp CI [+0.50,+1.91]；formula +1.25pp CI [+0.25,+2.26]，双面板下界>0 |
| **E3 P2b 跨数据集** | **方向成立** | P2b(V1)−V1 点估计 +0.57/+0.19pp；P2b(official)−official +1.62pp CI [+0.82,+2.44] / +1.22pp CI [+0.04,+2.33]（显著）。P2b(V1)−V1 的正式 CI 待 `paired_query_table_ci.py` 补算 |
| **E4 学习 vs 经典** | **劈裂判决** | Recall@1：经典最优（WSE）≥ 学习最优（点估计，显著性待补）；pooled pairwise AUROC/AUPRC：Noise V1 双面板第一 |

## 2. 双榜单（按预注册分层，headline 按 identity recall@1）

**Recall@1**：

| 排名 | 方法 | identity | formula |
|---|---|---:|---:|
| 1 | weighted_spectral_entropy（经典，零训练） | **87.37%** | **88.27%** |
| 2 | p2b_noise_v1_frozen（重排层） | 87.13% | 88.25% |
| 3 | p2b_sqrt_cosine（经典单通道） | 87.04% | 88.16% |
| 4 | p2b_official_frozen（重排层） | 86.98% | 88.03% |
| 6 | noise_v1（本项目） | 86.56% | 88.06% |
| 8 | official_dreams | 85.36% | 86.81% |
| 9-10 | cosine / modified cosine | 85.04/85.03% | 87.17/87.15% |

**Pooled pairwise AUROC / AUPRC**（DreaMS 论文口径的同构数学）：

| 方法 | identity AUROC | identity AUPRC | formula AUROC | formula AUPRC |
|---|---:|---:|---:|---:|
| **noise_v1** | **0.9413** | **0.7337** | **0.9299** | **0.7504** |
| weighted_spectral_entropy | 0.9402 | 0.7292 | 0.9265 | 0.7474 |
| p2b_noise_v1_frozen | 0.9416 | 0.7247 | 0.9248 | 0.7370 |
| official_dreams | 0.9287 | 0.6519 | 0.9033 | 0.6484 |

## 3. 可声明 / 不可声明

**可声明**：
1. 本项目 encoder 微调在模型盲外部基准上显著优于官方 DreaMS（E2，双面板 CI 下界>0）；且把 pooled pairwise AUROC 从 0.9287/0.9033 提到 0.9413/0.9299。
2. 冻结 P2b 重排对外部数据同样有效（两个基底、双面板方向全正；official 基底显著）。
3. **指标劈裂现象**：Top-1 检索与 pairwise 判别在本基准上排名不一致——pairwise 判别最强的表示（V1）不是 Top-1 最强的分数（WSE）。这与谱熵文献的立场一致，值得单独成段。

**不可声明**：
1. "学习表示在 Top-1 检索上优于经典相似度"——本基准不支持，且点估计方向相反（WSE 87.37% 全场第一）。WSE vs V1 / vs P2b(V1) 的差距（0.2~0.8pp）**尚无配对 CI，显著性未定**。
2. "优于全部近年方法"——S4（Spec2Vec/MS2DeepScore 2.0）未落榜。
3. GNPS AUROC 与 DreaMS 论文 NIST20 0.85 的任何直接比较（预注册红线）。

## 4. 次级事实

- 纯 cosine 在 Top-1 上与 official 打平（−0.32pp n.s. / +0.36pp n.s.）但 pooled AUPRC 显著更高（0.703 vs 0.652）——official embedding 的 pair 级判别在外部数据上偏弱，被本项目微调修复。
- WSE 的 corrected/introduced = 470/249（比值 1.89，全场最优交换比）；全部方法 risk_net(λ=2) 仍为负。
- near 子集：WSE 0.8251/0.8010 > V1 0.8144/0.7996 > official 0.7994/0.7799——经典方法在最难段也不输。
- modified_cosine ≈ cosine（前体位移在 10ppm 同加合物图上几乎无用武之地）。

## 5. 待办（按优先级）

1. **补两个正式检验**（`tasks/paired_query_table_ci.py`，秒级 CPU，读已存表）：
   - WSE vs noise_v1（E4 显著性裁决）
   - p2b_noise_v1_frozen vs noise_v1（E3 显著性裁决）
2. S4 公开神经基线（需先落权重到 `third_party/`，经 pair-score cache 契约接入，单列"现成实用榜"）。
3. 图已渲染（`figures/`）；按真实 10 方法数据微调一次标签/图例排布后即为论文版。

## 6. 溯源

run 2349091；bundle `method_scores.npz`（+metadata）、`evaluation/report.json`、`figures/{main_benchmark_figure,formula_panel_curves}.pdf/.png`、`figures/summary_table.csv`、逐方法 `queries_*.csv.gz` 与逐点 `curves_*.csv.gz` 全部留存于 RUN_ROOT。
