# 集成学习互补性：撤回与真值盲重审（2026-10-07）

**状态：S1 margin-selection 结果正式撤回（RETRACTED）。** 本文记录两次独立的真值泄露如何制造了"S1 = oracle (+4.93/+5.44pp)"与"跨面板 router +2.93/+4.03pp"两个假象，给出修正后的完整诚实结果，并说明什么结论幸存、什么结论替代它们。

## 1. 撤回一：S1 信号是真值泄露（答案键泄露）

- 原始 S1 定义：对每个 query，取 14 个方法中 `positive_vs_best_negative_margin` 最大的方法，采纳其排名。
- **泄露机制**：`positive_vs_best_negative_margin` 由**真阳性候选**的分数计算（rank≤2 时恒等于带符号 top1−top2 gap，rank≥3 时为 s(阳性)−s(top1)）。任何把阳性排第一的方法天然获得大正 margin，跨方法 argmax 因此几乎必然选中"已答对的方法"——这不是算法，是把答案键当特征。
- **证据**：S1 恰好等于 oracle-any（92.30/93.71%，9 方法）；其 847/331 个错误全部落在所有方法共同失败的 query 上（泄露信号无法提升不可恢复集）。
- **修正后的真值盲同构策略**（选 top1_top2_gap 最大者的排名，纯部署可计算）：identity **+0.25pp**（87.61 vs WSE 87.37）、formula **+0.65pp**（88.92 vs 88.27），配对 bootstrap CI 均跨零。
- 涉及文件：`tasks/GLM_mine_complementarity.py`（S1/S6 使用泄露信号）、`tasks/GLM_validate_s1_margin_selection.py`、`deliverables/GLM_gnps_article_ladder/run_local/s1_validation.json`、`complementarity_mining.json`（S1/S6 行）。提交 `e06720f`、`5246550` 的主张作废。

## 2. 撤回二：跨面板 router 是面板重叠记忆

- router v2（HGB + 一致性特征）报告 identity +2.93pp / formula +4.03pp（CI 全正）。
- **泄露机制**：identity-disjoint 与 formula-disjoint 面板共享 **5,237/5,261** 个 formula 面板 query 谱（query_row 与 query_ik14 双重叠）。"在一方面板训练、另一方面板测试"实际把测试 query 原样喂给了训练过程，树模型直接记忆。
- **证据**：v3 泄露清除后（训练 query 的谱与结构对测试集均未见过），增益坍缩：
  - Split A（formula 全部 5,261 训练 → identity 未见过 5,715 测试）：HGB **+0.18pp** [−0.28, +0.63]
  - Split B（identity 未见过 5,715 训练 → formula 全部 5,261 测试）：HGB **+0.46pp** [−0.02, +0.93]
  - Split C（identity 内结构对半分割）：HGB **+0.29pp** [−0.13, +0.71]
  - 学习型加权融合（pair 级 logistic，仅 Split B 方向 CI 为正）：**+0.59pp** [+0.10, +1.12]；A/C 方向不显著。
- 涉及文件：`tasks/GLM_router_v2.py`（结果作废，保留为反面教材）、`router_v2.json`。

## 3. 修正后的完整结果（全部真值盲 / 泄露清除）

基准：GNPS Gold/Silver 10ppm，identity-disjoint（10,995 q）与 formula-disjoint（5,261 q），14 方法冻结打分束（`bundle/method_scores.npz`），逐方法 R@1 经双源验证与冻结评估器逐 query 表一致。

### 3.1 无监督（无标签、部署可计算）

| 策略 | identity Δ vs WSE (pp, CI95) | formula Δ vs WSE (pp, CI95) |
|---|---|---|
| U1 选 raw gap 最大方法 | +0.04 [−0.32, +0.40] | +0.25 [−0.27, +0.78] |
| U2 选 batch-z gap 最大方法 | −1.81 [−2.27, −1.36] | −1.65 [−2.34, −1.01] |
| U3 选 batch-百分位 gap 最大方法 | −0.81 [−1.22, −0.40] | −0.66 [−1.25, −0.08] |
| U4 query 内 z 分数融合 | −0.07 [−0.36, +0.24] | +0.14 [−0.32, +0.57] |
| U5 Borda 融合 | −0.13 [−0.42, +0.16] | +0.08 [−0.38, +0.53] |
| U6 RRF 融合 | −0.02 [−0.32, +0.27] | +0.17 [−0.29, +0.63] |

**没有一个无监督策略显著超过最佳单方法（WSE）。**

### 3.2 监督（训练面板用标签，测试 query 未见过）

见 §2 v3 三分割表。最强为 learned fusion +0.59pp（单方向 CI 为正）；router ≤ +0.46pp（CI 跨零）。

### 3.3 参照（不可部署）

- 最佳单方法 WSE：87.37 / 88.27。
- oracle-any（14 方法）：93.18 / 94.47（**+5.81 / +6.20pp 真实但仅为上界**）。

## 4. 机制：为什么 oracle 空间捕获不了

1. **方法内校准极好**：top1−top2 gap 对"本方法是否答对"的 AUC = 0.86–0.93（14 方法 × 2 面板）；最高置信五分位准确率 99.3–100%。
2. **方法间错误高度相关**：逐对正确性 φ 均值 = 0.763 / 0.751。方法们一起失败。
3. **赢家中奖者不自知**：WSE 错误但 oracle 可恢复的 639/326 个 query 里，能答对的方法自身 gap 百分位均值仅 **0.115 / 0.107**，99.6/99.5% 的赢家处于自身置信分布 60 分位以下。**互补性恰好埋藏在所有方法都最不自信的区域，任何基于置信度的路由在结构上都无法开采。**
4. 我们的方法在可恢复集里贡献最多独特命中（identity 639 次中 noise_v1 胜出 287 次，14 方法第一；ms2deepscore 253、official_dreams 249、spec2vec 230 次之）——但同样无法被置信度路由利用。

## 5. 幸存的部署价值：弃权门控（risk–coverage）

| 覆盖率 | WSE acc (id / form) | noise_v1 acc (id / form) |
|---|---|---|
| 50% | 99.60 / 99.73 | 99.42 / 99.54 |
| 60% | 99.41 / 99.52 | 99.03 / 99.37 |
| 80% | 96.46 / 96.74 | 95.59 / 96.34 |
| 100% | 87.37 / 88.27 | 86.56 / 88.06 |

真值盲 gap 可靠地划分"可自动信赖"与"需要正交证据"两个世界：60% 覆盖带精度 ≈99.5%，而弃权带（低置信）准确率≈抛硬币。**置信度的正确用途是门控（何时信库检索、何时转 MSn/化学专家/人工），不是路由（信哪个方法）。** 这与本项目 identifiability-gated 范式（CLAIM_IDENTIFIABILITY_GUIDED_UNTARGETED_METABOLOMICS_20261005）直接互锁：低置信带正是该范式的服务对象。

## 6. 对外主张边界（论文语言）

- **可以**：(1) 14 方法模型盲 GNPS 梯队本身；(2) oracle 互补性 +5.81/+6.20pp 的存在性与结构刻画（赢家不自知、φ≈0.76）；(3) 无监督与泄露清除后的监督选择/融合全部 ≤ +0.6pp 的**否定性结果**（对"多方法集成自动涨点"预期的可引用反例）；(4) gap 校准 AUC 0.86–0.93 与 risk–coverage 门控的部署读数；(5) noise_v1 是可恢复集最大单一贡献者 + pooled AUROC 榜首（0.9413）。
- **不可以**：任何"集成/路由/选择达到 92–94%"或"捕获 oracle 全部空间"的表述；任何引用 `s1_validation.json`、`complementarity_mining.json` 中 S1/S6 行、`router_v2.json` 的数字。
- 两次泄露本身作为方法论警示写入论文附录：**oracle 味置信信号**（由真阳性分数导出的任何量）与**面板重叠记忆**（分割必须按 query 谱+结构双去重）是集成评估的两个系统性陷阱。

## 7. 工件清单

| 文件 | 内容 | 状态 |
|---|---|---|
| `tasks/GLM_truthblind_fusion_analysis.py` | U1–U6 + 泄露清除 router，双源验证 | 当前有效 |
| `tasks/GLM_router_v3.py` | 三分割泄露清除协议 | 当前有效 |
| `tasks/GLM_gap_calibration.py` | gap→正确性校准 | 当前有效 |
| `tasks/GLM_risk_coverage.py` | risk–coverage + 失败带结构 | 当前有效 |
| `tasks/GLM_zenodo_resume.py` | 通用续传下载器 | 工具 |
| `deliverables/.../truthblind_ensemble.json` | §3.1 表 | 当前有效 |
| `deliverables/.../router_v3.json` | §2/§3.2 表 | 当前有效 |
| `deliverables/.../gap_calibration.json`、`risk_coverage.json` | §4/§5 | 当前有效 |
| `deliverables/.../s1_validation.json`、`complementarity_mining.json` | 旧 S1 结果 | **RETRACTED**（保留作审计痕迹） |
| `tasks/GLM_mine_complementarity.py`、`GLM_validate_s1_margin_selection.py`、`GLM_router_v2.py` | 泄露实现 | 归档为反面教材 |

> 一句话总结：**集成路由的 5pp 蛋糕在地图上（oracle），但钥匙不在置信度里——赢家不知道自己赢；置信度真正可靠的用途是把 60% 的 query 圈进 99.5% 精度自动带，把剩下的硬币抛给正交证据范式。**
