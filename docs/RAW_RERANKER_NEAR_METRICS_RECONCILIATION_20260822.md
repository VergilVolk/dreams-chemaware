# RAW Reranker 的两种 near 指标：定义、关系与结论边界（2026-08-22）

**目的**：终结此前"预挖 near 硬负例面板"与"真实检索 near Top-1 错误"两个口径之间的混乱，明确各自能/不能得出的结论。

---

## 一、两个指标的定义（截然不同的候选集）

### 指标 A：near hardest-negative 准确率（预挖 near 硬负例面板）

- **候选集**：`g8r_locked/val.json` 的 `neg` 字段——**故意挖掘的 446 个质量近邻困难负例**（独立 HDF5 行，非 anchor），按 MCES 分级 near（0–2）/ mid（3–5）。
- **定义**：对每个有 ≥1 个 near 负例的 anchor，`s(pos) > max 过该 anchor 的 near 负例` 的占比。
- **结果**（统一口径后）：near 净 +3（corrected 18 / introduced 15 / 186 anchor），formula-cluster bootstrap CI 跨 0，**不显著**。
- **性质**：这是"**最难中的最难**"诊断——预挖负例是 DreaMS 最易混淆的近异构体，是压力测试，不是自然检索分布。

### 指标 B：near Top-1 检索错误的修正（anchor-vs-anchor）

- **候选集**：`g8r_locked/val.json` 的 **2000 个 anchor 自身**，按 strict-10ppm 同 adduct 构成候选池（逐分子去重、取最高分）。
- **定义**：对每个 query，Top-1 候选是否等于 query 自身 IK14；"near 错误"= 错误 Top-1 候选与 query 的 MCES ∈ 0–2。
- **结果**（44/17 分解）：near（MCES 0–2）baseline 错误 60 → 修正 20（33.3%）→ 引入 7 → **净 +13**。
- **性质**：这是**自然严格 10ppm 检索**里的 near 错误，是部署场景的真实任务。

---

## 二、为什么两个指标给出不同结果（净 +3 vs 净 +13）

不是矛盾，而是**候选难度不同**：

| | 指标 A（预挖） | 指标 B（anchor-vs-anchor） |
|---|---|---|
| near 负例来源 | 故意挖的最难异构体 | 自然候选池里的 near 分子 |
| near 负例难度 | 极高（DreaMS 最易错） | 混合（易+难） |
| RAW reranker 净收益 | +3（不显著） | +13（33% 修正率） |

**结论**：RAW reranker 对**自然检索中的 near 错误**有效（33%、净+13），但对**故意挖掘的最难 near 负例**仍吃力（净+3、不显著）。二者必须分开报告，不能互相替代。

---

## 三、各自的结论边界（严谨版）

1. **指标 A 不显著** → 只能说"RAW reranker 对最难 near 硬负例的局部排序改善证据不足"，**不能**说"RAW reranker 对 near 无效"。
2. **指标 B 净+13、33% 修正率** → RAW reranker **确实修正了真实检索中的 near Top-1 错误**（且修正率在各 MCES 层均匀 33%/33%/31%，说明是"通用检索改善"，不是"near 特异突破"）。
3. **两者合起来** → RAW reranker 的 +4.35pp 总体增益里，near（0–2）贡献了净 +13，是各层最大；但这是"通用改善恰好覆盖了 near 错误"，不是"专门解决了 near 异构体"。
4. 以上均为 **g8r_val 开发集**结果；最终泛化结论仍需 Test-A/Test-B 一次性验证。

---

## 四、对后续方向的含义

- RAW reranker 作为"总体严格 10ppm 检索改善器"已有多层证据（整体 +4.35pp CI>0、unseen-formula 保持、near 净+13、gate 有效）。
- 但"最难 near 异构体分离"（指标 A）仍未显著解决；若这是课题核心目标，后续应进入审核员说的"峰 token + 候选组内 listwise 排序"，而非继续在 RAW 特征上增量。
- 反事实峰遮蔽（第三类噪声）仍需等待"共享主峰是 near 误判因果来源"的确认，暂不启动。
