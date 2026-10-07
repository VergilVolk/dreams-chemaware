# LCNEC 暗特征三级主张架构 v2（门控范式落地，2026-10-07）

**性质：** 任务二应用方法论的首个落地数字。GNPS 模型盲基准校准的置信度门控（cosine 分子级 top1−top2 gap）迁移到 LCNEC 优先暗特征反向搜索，产出逐模块主张表。**工作流交付，非性能主张。**

**v2 修正（审计发现）：** v1 的 `top1_name` 联接自 pilot 时代的 `lcnec_dark_neighbor_identities.csv`——该表早于 unsorted-mz cosine 修复（分子与分数均已过期，name 列实为 USI）。v2 直接从 GNPS manifest 联接正确分子（SMILES/USI/仪器），并新增卤素合理性旗标。**旧 identities 表不得用于任何决策。**

## 1. 输入与校准链

- 暗特征：30 个 LCNEC priority dark 模块；分子级（ik14 折叠）重扫（`GLM_lcnec_dark_molecule_rescan.py`），0.7% 类似物容差门，52,871 张 GNPS gold/silver 库谱。
- 门控阈值（GNPS cosine_greedy，保守取 identity 面板）：@99% 精度 gap ≥ **0.584**；@99.5% gap ≥ **0.741**。

## 2. 三级结果（30 模块；三轴 = 谱学门控 × 合理性旗标 × 质量一致性）

**质量轴（本轮审计新增）：dark 前体 vs 候选 [M+H]+ 的偏差全部 >10 ppm（−23.8 至 −6963 ppm）——30 个模块没有任何同质量候选，全部 top 命中都是 0.7% 类似物容差门扫进来的类似物匹配。**

| 级 | 定义 | 数量 | 模块（m/z → top 命中） | Δppm | 卤素 |
|---|---|---|---|---|---|
| **A 门控通过** | cos ≥0.85 且 gap ≥0.584 | **1** | 221.985 → C8H7FN6O（cos 0.949，gap 0.826） | **−4883** ⚠类似物 | ⚠ F |
| **B1 高分争议** | cos ≥0.85，gap 不过 | 4 | 273.081（0.981/gap 0.012） | −52 | ⚠ F₂ |
| | | | 342.088（0.967/0.205） | −6232 | — |
| | | | 207.015（0.934/0.011） | +4353 | — |
| | | | 313.120（0.924/0.006） | −24 | ⚠ F₃ |
| **B2 次命中** | 0.70–0.85 | 3 | 190.072 / 251.079 / 244.023 | −5488 / −4142 / −450 | — |
| **C 暗** | <0.70 | 22 | 无谱学主张 | — | 12/29 top 命中含卤 |

## 3. 读数（最终版）

1. **零分子级命中**：质量轴加入后，30 个暗特征没有一个具备同质量候选——谱学 gap 门控通过的 221.985 也是 −1.09 Da 的类似物（谱极相似但质量不符）。**任何 top-hit 工作流都会在这里产生 8 个假"高置信鉴定"；三轴架构把它们全部拦在正确层级。**
2. 候选化学空间本身可疑：top 命中多为 GNPS 合成库（组合化学板）含氟/含卤结构——对内源组织代谢物而言是谱巧合类似物，B1/B2 不构成购买工单，只构成"药物暴露/污染排查 + MSn"的排查线索。
3. 221.985 的正确处置：**污染/用药暴露排查优先于任何标准品采购**（QC/blank 序列相关性、进样序位置）。
4. 22 个暗特征 = 库覆盖缺失，无候选主张。

## 3.5 法证层（EIC矩阵对照，冻结资产）

对 A/B1/B2 全部 8 个模块用冻结 EIC 矩阵（85 进样 = 68 study + 9 QC + 2 blank + 6 dilution）做样本类型分解 + 表型效应联查（`GLM_lcnec_dark_forensics.py` → `lcnec_dark_forensics.json`）：

- **8/8：blank 0/2 检出（零污染）、QC 9/9 稳健、study 68/68 全检出**；
- **8/8 带强表型效应**：|log2FC| = 1.06–3.12，q ≤ 4.6×10⁻¹¹（其中 342.088 达 +2.92/q=1.4e-19、190.072 达 +3.12/q=2.1e-17）；
- 处置更新：Tier-A 221.985 的污染假说被 blank 屏除——它是真实、重现、与表型强关联的成分；含氟类似物邻居使**药物暴露假说**成为首选检验（用药记录联查 → MSn），其余 B1/B2 走 MSn/标准品路线；
- 组合结论：**这 8 个是"库覆盖缺失但信号真实"的暗物质核心**——三级架构把它们与假鉴定区分开，法证层证明它们值得正交证据投入。

## 4. 边界（随表携带）

- 阈值校准域（库-库谱）与迁移域（真实样本 DDA）存在域移位；@99% 保证近似成立。
- 质量轴判据：|Δ| ≤10 ppm 视为同质量候选；>10 ppm 一律类似物（本表全部如此）。
- Tier A 为 Level 2 文献匹配主张（且本表 Tier A 实为类似物线索，非分子鉴定）；Level 1 需标准品。卤素旗标为启发式，不替代用药史与 QC 排查。
- n=30，无统计主张。
- 摘要句（可进稿件）：*Transferring benchmark-calibrated confidence gates to reverse-search annotation of 30 LCNEC dark modules yielded no molecule-level identification: every top hit is an analog-mass match (23-7000 ppm off the candidate [M+H]+), mostly fluorinated synthetic-library structures; one module passed the spectral gap gate and is flagged exogenous-until-proven. The three-axis architecture (spectral gate, chemical plausibility, mass consistency) is what prevents these from being reported as identifications.*

## 5. 工件

| 文件 | 内容 |
|---|---|
| `tasks/GLM_lcnec_dark_molecule_rescan.py` + `lcnec_dark_molecule_rescan.{json,csv}` | 分子级重扫（gap 精确） |
| `tasks/GLM_lcnec_dark_three_tier.py`（v2） + `lcnec_dark_three_tier.{json,csv}` | 双轴三级主张表（SMILES/USI/卤素旗标） |
| `tasks/GLM_lcnec_dark_forensics.py` + `lcnec_dark_forensics.json` | 法证层：blank/QC/study 分解 + 表型效应 + 处置 |
| `deliverables/GLM_gnps_article_ladder/run15/gate_cosine_{identity,formula}.json` | 阈值校准来源 |
