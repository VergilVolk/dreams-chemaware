# LCNEC 暗特征三级主张架构 v2（门控范式落地，2026-10-07）

**性质：** 任务二应用方法论的首个落地数字。GNPS 模型盲基准校准的置信度门控（cosine 分子级 top1−top2 gap）迁移到 LCNEC 优先暗特征反向搜索，产出逐模块主张表。**工作流交付，非性能主张。**

**v2 修正（审计发现）：** v1 的 `top1_name` 联接自 pilot 时代的 `lcnec_dark_neighbor_identities.csv`——该表早于 unsorted-mz cosine 修复（分子与分数均已过期，name 列实为 USI）。v2 直接从 GNPS manifest 联接正确分子（SMILES/USI/仪器），并新增卤素合理性旗标。**旧 identities 表不得用于任何决策。**

## 1. 输入与校准链

- 暗特征：30 个 LCNEC priority dark 模块；分子级（ik14 折叠）重扫（`GLM_lcnec_dark_molecule_rescan.py`），0.7% 类似物容差门，52,871 张 GNPS gold/silver 库谱。
- 门控阈值（GNPS cosine_greedy，保守取 identity 面板）：@99% 精度 gap ≥ **0.584**；@99.5% gap ≥ **0.741**。

## 2. 三级结果（30 模块；双轴 = 谱学门控 × 合理性旗标）

| 级 | 定义 | 数量 | 模块（m/z → top 命中） | 卤素旗标 |
|---|---|---|---|---|
| **A 门控通过** | cos ≥0.85 且 gap ≥0.584 | **1** | 221.985 → C8H7FN6O（cos 0.949，gap 0.826，entropy 0.421） | ⚠ **含 F（外源待证）** |
| **B1 高分争议** | cos ≥0.85，gap 不过 | 4 | 273.081 → C14H10F2N4（0.981/gap **0.012**）⚠F₂；342.088 → C20H29N3O2（0.967/0.205）；207.015 → C12H15NO2（0.934/0.011）；313.120 → C14H15F3N4O（0.924/0.006）⚠F₃ | 3/4 含氟 |
| **B2 次命中** | 0.70–0.85 | 3 | 190.072（C8H18N2OS）、251.079（C13H17NO4）、244.023（C15H17NO2） | 无 |
| **C 暗** | <0.70 | 22 | 无谱学主张；12/29 个 top 命中含卤素（含 F₆、BrCl 等）→ 库覆盖缺失信号，不报告为候选 | — |

## 3. 读数

1. **唯一过门控的命中带合理性旗标**：221.985 的谱学证据充分（gap 0.826），但 C8H7FN6O 含氟——组织代谢组学中含氟结构默认外源（LCNEC 患者化疗/环境暴露可能）。**门控通过 ≠ 可报告；主张需要双轴（谱学×合理性）都过。** 该模块的正确下一步：查用药记录/进样序列污染排查/标准品。
2. **最高分仍是最争议**：273.081 cosine 0.981、gap 0.012——传统 top-hit 工作流的"自信错误"，门控拦截送升级；且其候选 F₂，同样待合理性审查。
3. 22 个暗特征的低分"top 命中"多为含卤药物样化学式（F₆、BrCl）——反向搜索在此分数段没有信息量，暗特征的正确表述是"库覆盖缺失"。
4. B1/B2 共 7 个模块是正交证据工单（标准品清单含 SMILES/USI/仪器，`lcnec_dark_three_tier.csv`）。

## 4. 边界（随表携带）

- 阈值校准域（库-库谱）与迁移域（真实样本 DDA）存在域移位；@99% 保证近似成立。
- 类似物容差门使竞争分子含类似物，gap 语义 = 证据受争议。
- Tier A 为 Level 2 文献匹配主张；Level 1 需标准品。卤素旗标为启发式（F/Cl/Br/I→外源待证），不替代用药史与 QC 排查。
- n=30，无统计主张。
- 摘要句（可进稿件）：*Confidence thresholds calibrated on a model-blind benchmark were transferred to dark-feature reverse search: of 30 modules, one passed the spectral gate (and carries a fluorine plausibility flag), four high-scoring modules were blocked by contested molecule-level gaps, and 22 remained dark.*

## 5. 工件

| 文件 | 内容 |
|---|---|
| `tasks/GLM_lcnec_dark_molecule_rescan.py` + `lcnec_dark_molecule_rescan.{json,csv}` | 分子级重扫（gap 精确） |
| `tasks/GLM_lcnec_dark_three_tier.py`（v2） + `lcnec_dark_three_tier.{json,csv}` | 双轴三级主张表（SMILES/USI/卤素旗标） |
| `deliverables/GLM_gnps_article_ladder/run15/gate_cosine_{identity,formula}.json` | 阈值校准来源 |
