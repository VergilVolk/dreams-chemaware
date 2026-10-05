# BioAware B47-U1 谱学一元校准节点（2026-09-21）

## 结论先行

U0 证明 B47 的候选分数对参考谱数量敏感，但它不等于“参考谱越多越坏”。在
`83,619-query corrected development graph` 上，直接把 molecule-max 换成参考谱均值会让
Recall@1 下降约 8.29 pp；直接惩罚参考谱数量同样有害。因此 U1 只允许小剂量、公式隔离、
风险受控的重复谱共识校准。

## U1：小型冻结配方族

- 输入：official DreaMS query-reference cosine；不使用 Rhea、表型、P2b 或 B47 真值。
- 候选集：与 corrected graph 完全相同；same-adduct、strict 10 ppm。
- 33 个预注册配方：`max`，以及 max 向 mean / top-2 mean 的小剂量收缩，配合极小的
  log-reference-count 项。
- 选择：5-fold formula OOF；ties 对正例不利；要求 Recall@1、MRR、near 和公式等权效应
  同向，且 `corrected > 2 * introduced`。

OOF 结果：

- Recall@1：0.928760 → 0.929442，+0.0682 pp；
- formula-cluster 95% CI：[+0.0249, +0.1135] pp；
- near Recall@1：+0.1348 pp；
- corrected / introduced：131 / 74；
- λ=2 risk net：-17；
- fold 0 轻微反向。

因此 U1 的“允许身份切换”分支没有通过安全门，不能带回 B47。

## U1b：顺序泄漏纠错与严格安全 veto

U1b v1 曾报告 45 / 0 和 +0.0538 pp，但该结果无科学资格。corrected graph 的监督评价合同
规定 positive molecule 固定为每个候选组的第一项；v1 在 baseline 并列时调用 `argmax`，也会
固定选择第一项。于是所谓“保持原稳定 Top-1 后消歧”实际借用了正例在数组中的位置。这是候选
顺序泄漏，不是谱学证据。

U1b v2 已修为：只有原始 max Top-1 本来唯一、校准 Top-1 也唯一且两者身份相同时，才允许采用
校准分数；baseline 并列一律拒绝消歧。复算结果：

- corrected / introduced：0 / 0；
- Recall@1：+0.0000 pp；
- near Recall@1：+0.0000 pp；
- formula-cluster 95% CI：[0, 0]；
- 五个 outer folds 的 Recall@1 均为 0；
- Top-1 identity switches：0。

这不是失败的安全策略，而是一个数学边界：若不允许改变唯一 Top-1，又不允许借候选顺序解除
并列，就不可能修正任何 Top-1 错误。U1b 因此冻结为 `STOPPED_INVALIDATED_V1 / V2_NOOP`。

## 为什么 U1/U1b 不能直接解锁 B47

U1 允许身份切换但风险不合格；U1b 禁止身份切换后成为 Top-1 no-op。二者都不能作为新的 B47
seed scorer。不得降低 B47 的 `score >= 0.80`、`margin >= 0.05` 或 identity 门来迁就结果。

## 下一步唯一问题

先运行 U1c 真值盲分母审计，定位 `<200 identities` 首次发生在哪一层：

1. unique Top-1 身份多样性；
2. score/margin 绝对门；
3. 跨样本 feature consensus；
4. Rhea coverage、currency/hub safety；
5. sample-candidate collapse。

若在 score/margin 或 consensus 前已有 >=200 个安全 Rhea 身份，才值得构建更强的原始峰
谱学 unary（RAW/entropy/neutral-loss）。若在网络覆盖或身份集中层已低于 200，继续调谱学
分数不能解决问题，应停止 U1 并重设 event benchmark/seed source。

## 主张边界

U1 是 corrected development graph 的内部谱学开发结果；U1b v1 是已撤销的顺序泄漏结果，
U1b v2 是结构安全但 Top-1 no-op 的审计。它们都不是 B47 accuracy、外部转移、反应网络增益、
SOTA 或 shared-embedding improvement。
