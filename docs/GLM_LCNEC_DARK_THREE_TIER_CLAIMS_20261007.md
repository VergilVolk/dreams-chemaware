# LCNEC 暗特征三级主张架构（门控范式落地，2026-10-07）

**性质：** 任务二应用方法论的首个落地数字。把 GNPS 模型盲基准上校准的置信度门控（cosine 分子级 top1−top2 gap）迁移到 LCNEC 优先暗特征模块的反向搜索，产出可审计的逐模块主张表。**这是工作流交付，不是性能主张。**

## 1. 输入与校准链

- 暗特征：30 个 LCNEC priority dark 模块（`priority_dark_modules.mgf`）。
- 反向搜索：分子级（ik14 折叠）重扫 v2（`GLM_lcnec_dark_molecule_rescan.py`，修正 v1 只存 top-3 谱邻居、无法计算分子级 gap 的问题），0.7% 类似物容差前体门，52,871 张 GNPS gold/silver 库谱。
- 门控阈值（GNPS 模型盲基准 cosine_greedy 校准，保守取 identity 面板值）：
  - @99% 精度：gap ≥ **0.584**（覆盖 52.8%）
  - @99.5% 精度：gap ≥ **0.741**（覆盖 43.8%）

## 2. 三级分类结果（30 模块）

| 级 | 定义 | 数量 | 模块 |
|---|---|---|---|
| **A 门控通过** | cosine ≥0.85 且 gap ≥0.584 | **1** | m/z 221.985 → C8H7FN6O（cos 0.949，gap 0.826，entropy 0.421） |
| **B1 高分争议** | cosine ≥0.85 但 gap 不过 | **4** | 273.081（cos 0.981，gap **0.012**）、342.088（0.967/0.205）、207.015（0.934/0.011）、313.120（0.924/0.006） |
| **B2 次命中类似物** | 0.70–0.85 | 3 | 190.072、251.079、244.023 |
| **C 暗特征** | <0.70 | 22 | 无谱学主张（top 命中多为含卤/药物样化学式，属库覆盖缺失而非候选） |

## 3. 读数（不越界）

1. **30 个暗特征里只有 1 个能过门控**——暗特征空间正是门控范式的升级带，与 GNPS 上"60% 自动带之外 ≈ 抛硬币"的定量结构一致。
2. **最高分恰是最争议**：m/z 273.081 cosine 0.981 但分子级 gap 仅 0.012——基准上验证的"分数高 ≠ 可靠、gap 才是决策量"在真实数据上复现。该模块若按传统 top-hit 报告会是"自信的错误"。
3. B1 四个模块是**正交证据的明确工单**：标准品购买（`lcnec_dark_neighbor_identities.csv` 已列 USI/质量/仪器）或 MSn。
4. C 级的"top 命中"（cosine 0.02–0.38 的含卤/大分子化学式）不应报告为候选——这是库覆盖缺失的信号。

## 4. 边界（必须随表携带）

- 阈值在校准域（库-库谱、10ppm 检索）与迁移域（真实样本 DDA、0.7% 类似物门）之间存在域移位；Tier A 的 99% 精度保证只近似成立，**Tier A 仍是 Level 2 文献匹配主张，Level 1 需标准品**。
- 类似物容差门使"竞争分子"含类似物，gap 语义是"证据受争议"而非严格的同分异构体区分。
- n=30，无统计主张。
- 摘要句（可直接进稿件方法段）：*Confidence thresholds calibrated on the model-blind GNPS benchmark were transferred to dark-feature reverse search; 1/30 modules passed the gate, 4 high-scoring modules were blocked by contested molecule-level gaps and routed to orthogonal evidence, and 22 remained dark.*

## 5. 工件

| 文件 | 内容 |
|---|---|
| `tasks/GLM_lcnec_dark_molecule_rescan.py` + `data/validation/GLM_track2_census/lcnec_dark_molecule_rescan.{json,csv}` | 分子级重扫（gap 精确） |
| `tasks/GLM_lcnec_dark_three_tier.py` + `lcnec_dark_three_tier.{json,csv}` | 三级主张表 |
| `deliverables/GLM_gnps_article_ladder/run15/gate_cosine_{identity,formula}.json` | 迁移阈值的校准来源 |
