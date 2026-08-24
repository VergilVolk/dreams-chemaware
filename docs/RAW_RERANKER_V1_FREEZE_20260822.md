# RAW Reranker v1 冻结记录（2026-08-22）

**性质**：候选组内 pairwise ranking reranker，**不动 DreaMS 权重、不生成新 DreaMS checkpoint**。DreaMS 负责 ±10ppm 候选召回，RAW 峰级特征重排器负责重打分。

---

## 一、冻结配置（唯一正式版本）

| 项 | 值 |
|---|---|
| 模型 | 逻辑回归（LogisticRegression, fit_intercept=False） |
| 特征（12 维） | `dreams_similarity` + 11 个 RAW 特征：sqrt_cosine, linear_cosine, entropy_similarity, intensity_coverage_min, intensity_coverage_mean, matched_peak_fraction_min, top10_match_fraction, neutral_loss_sqrt_cosine, neutral_loss_coverage_min, neutral_loss_coverage_mean, peak_count_ratio |
| 训练目标 | within-query 差分 `x(q,c⁺) − x(q,c⁻)`，正例赢过 Top-5 困难负例（hard_k=5） |
| 正则 C | 0.01（formula-group OOF 选定） |
| 峰匹配容差 | 0.02 Da |
| 候选图协议 | strict-10ppm、同 adduct、逐 IK14 去重（取最高分） |
| gate 阈值 | 0.24098341166973114（DreaMS Top1−Top2 分差，OOF 选定） |
| gate require_disagreement | false |
| 训练/评估数据 | g8r_train（训练+OOF）/ g8r_val（开发） |

---

## 二、开发集（g8r_val）正式结果

- 整体 Recall@1：0.8081 → 0.8516（**+4.35pp**，净 +27）。
- formula-cluster bootstrap CI：[0.011, 0.069]；McNemar p=0.0009 → 显著。
- near（MCES 0–2）层：corrected 20 / introduced 7 / **净 +13**，CI 下界 +0.004、McNemar p=0.019 → 显著（唯一显著层）。
- 其余层（3–5 / 6–10 / >10）净 +4 / +4 / +6，均不显著。
- unseen-formula 层：+4.33pp，与 seen 基本一致。

**口径对账**：ungated 全量重排 = 45 corrected / 18 introduced；gated 最终模型 = 44 / 17。差异为 gate-off 反事实的 1 corrected + 1 introduced（净 0）。**正式转换表以 gated 44/17 为准**；差异 query ID 见 `data/validation/g8r_raw_reranker_reconcile.json`。

---

## 三、结论边界（严谨表述）

> RAW 重排器显著改善自然检索中的 near 结构错误（净 +13），但修正率跨层均匀（33%），属"通用检索改善覆盖 near"，**不是 near 特异突破**；其能力尚未覆盖 hard-tail near 异构体（预挖 near 面板净 +3 不显著）。

- 可以说：得到一个**有望优于官方 DreaMS 的两阶段检索系统**（开发集显著）。
- 不能说：微调出了更好的 DreaMS 权重；DreaMS 权重与 embedding space 均未变。
- 最终泛化：等待 Test-A / Test-B 一次性盲测。

---

## 四、冻结后纪律

- 冻结后禁止根据 Test-A/B 再调整任何参数、阈值或特征。
- Test-A/B 只用于当前 RAW-v1 的一次性验证；下一代（峰 token + 候选组 listwise + 峰级 RAW 证据）另锁 Test-C，不复用 Test-A/B 作全新盲测。
- 噪声/反事实遮蔽后续仅作增强项，不充当主架构。

## 五、代码版本哈希

（提交前在服务器记录以下文件 SHA256：`tasks/train_g8r_raw_reranker.py`、`tasks/audit_g8r_raw_reranker_44_17.py`、`tasks/audit_g8r_raw_reranker_bootstrap.py`、`tasks/audit_g8r_raw_reranker_reconcile.py`，以及 `data/validation/g8r_raw_reranker_cache{,_val}.npz`。）
