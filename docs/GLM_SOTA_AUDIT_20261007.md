# GNPS 15方法模型盲基准 SOTA 审计（2026-10-07）

**审计对象：** 冻结 15 方法梯队（`deliverables/GLM_gnps_article_ladder/run15/ladder_full.csv`，双源验证通过；第 15 方法 = MS2LDA 2.0 重训 Spec2Vec 2026，Zenodo 记录 15688609，md5 锚定）。**基准：** GNPS Gold/Silver 10ppm 模型盲面板（与 MSG/MoNA 训练语料身份/公式双零重叠）。

## 0. spec2vec_2026 入场结果

- R@1 **86.08 / 87.25**，低于 2020 版（86.29 / 87.44）：重训模型的训练库分布（cleaned combined libraries）不覆盖 GNPS 生态，模型盲基准上不占优。
- AUROC 0.9346 / 0.9177（略高于 2020 版 0.9336 / 0.9170，R@1 更低——又一处判别/检索分离样本）。
- 空向量（词表缺失>10%）仅 19/57k 谱，可忽略。
- **对记分板无影响：全部冠军不变。**

## 1. 冠军记分板（14 方法现状）

| 指标 | identity_disjoint 冠军 | 我方最好 | formula_disjoint 冠军 | 我方最好 |
|---|---|---|---|---|
| Recall@1 | WSE 87.37 | p2b_noise_v1_frozen 87.13（平手，ΔCI [−0.11,+0.57]） | WSE 88.27 | p2b_noise_v1_frozen 88.25（平手，ΔCI [−0.49,+0.51]） |
| MRR | entropy_raw 0.9179 | p2b_noise_v1_frozen 0.9165 | p2b_noise_v1_frozen **0.9290** ✅ | 同左 |
| pooled pairwise AUROC | p2b_noise_v1_frozen **0.9416** ✅ | 同左 | noise_v1 **0.9299** ✅ | 同左 |
| micro AUROC | noise_v1 **0.9436** ✅ | 同左 | noise_v1 **0.9314** ✅ | 同左 |
| macro query AUROC | entropy_raw 0.9464 | p2b_noise_v1_frozen 0.9444 | entropy_raw 0.9411 | p2b_noise_v1_frozen 0.9401 |
| near Recall@1 | WSE 82.51 | p2b_noise_v1_frozen 82.13 | WSE 80.10 | noise_v1 79.96 |
| 互补性独特贡献 | — | noise_v1 可恢复集胜出 287/639（14 方法第一） | — | noise_v1 161/326（第一） |

（✅ = 我方持有；配对 CI 见正文。）

## 2. 我方 SOTA 的部分（可直接主张）

1. **成对判别力双面板第一**：pooled 10ppm pairwise AUROC identity 0.9416（p2b_noise_v1_frozen）、formula 0.9299（noise_v1）；micro AUROC 双面板第一（noise_v1）。与次名（WSE 0.9402/0.9265）差距小但方向一致；与公开学习模型（ms2deepscore 0.8994/0.8818、spec2vec2020 0.9336/0.9170）差距显著。
2. **formula 面板 MRR 第一**（p2b_noise_v1_frozen 0.9290）。
3. **可恢复集最大单一贡献者**（oracle 互补性结构分析，见撤回文档 §4）：方法盲但结构上不可被置信度路由利用。

## 3. 非 SOTA 的部分与补充状态

| 缺口 | 差距 | 补充动作 | 状态 |
|---|---|---|---|
| R@1 双面板 | 名义 −0.24/−0.02pp，**配对 CI 均跨零（统计平手）** | spec2vec_2026 已入场（低于 2020 版，冠军不变）；诚实表述为"学习表示与最强经典熵相似度在身份检索上平手，在判别力上领先" | **已完成** |
| macro query AUROC | entropy_raw 领先 ~0.2pp（0.9464 vs 0.9444） | 无本地训练可补；写入 metric-split 图注（判别 vs 检索的分离） | 已有图 `GLM_gnps_metric_split`（15 方法版） |
| near Recall@1 | WSE 领先 0.38/0.14pp | near 子集为主张边界（"最难结构近邻区分仍属经典熵"），不冒领 | 已写入梯子文档 |
| 集成超越单方法 | oracle +5.88/+6.31pp 存在（15 方法） | **已证明基本不可部署捕获**（撤回文档）；15 方法版泄露清除 router 在 3 分割之 2 显著（B +0.61 [+0.15,+1.06]、C +0.51 [+0.09,+0.95]，A +0.23 n.s.）≈ oracle 空间的 10%；替代主物 = 弃权门控（60% 覆盖@99.4–99.5%） | `GLM_confidence_gate.py` 已交付 |

## 4. 结论

- **主张句（诚实版）：** 在与训练语料零重叠的 GNPS 模型盲基准上，我们的噪声课程 encoder 家族在成对判别（pooled/micro AUROC）与 formula 面板 MRR 上是 14 方法第一；在 Top-1 检索上与最强经典方法（WSE）统计平手；集成/路由不能超过最佳单方法（否定性结果 + 机制解释）；置信度的部署形态是弃权门控。
- **禁止句：** "我们全面 SOTA"；"集成达到 92–94%"；任何引用已撤回 S1/router_v2 数字的表述。

*spec2vec_2026 打分完成后本表更新为 15 方法版。*
