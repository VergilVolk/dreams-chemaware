# Stage-2 冻结合同：全局化学重塑的跨平台复现、暗实体增量和转换边一致性

**日期：** 2026-10-07
**前置：** Stage-1 PASS（`LCNEC_GLOBAL_REMODELING_STAGE1_PASS`）
**性质：** 冻结实验设计；Stage-1 结果出来前不得修改本合同的任何阈值或规则。

## 1. 科学问题

Stage-1 通过三分解（家族整体 / 家族内 / 孤立残差）的正交性、非平凡性和
k-稳定性后，Stage-2 回答三个否决实验：

| # | 问题 | 通过条件 | 失败后果 |
|---|---|---|---|
| 2A | 家族内部替换是否跨平台复现？ | 在 LIPn 独立平台上，同家族不同实体的效应差异方向一致率 ≥ 60% | 降为单平台描述性 |
| 2B | 暗实体是否给 known-only 增加独立信号？ | 加入暗实体后，家族分解的 sign-flip p 值改善 ≥ 一个数量级 | 降为已知代谢物谱学扩展 |
| 2C | 转换边（PMD + 碎片支持）的患者方向一致性是否超过 PMD 置换零？ | BH-FDR ≤ 0.05 的转换边数 ≥ 5，且置换零中位数 p ≥ 0.20 | 移除转换层，仅保留家族+孤立 |

## 2. Stage-2A：LIPn 跨平台复现

### 2.1 数据

- LIPn 平台的 mzML 包：`MTB22_P073_LIPn_mzML_public.zip`（336.9 MB，Zenodo 19005638）
- LIPn overview：`04_MTB22_P073_LIPn_mzML_overview_v1.txt`
- 同一 34 对患者（TU/NG），但独立平台/色谱条件

### 2.2 流程（表型盲冻结）

1. 在 LIPn 平台上重建同一 263 家族的 EIC 定量（与 Stage-1 完全相同的
   `quantify_lcnec_hsst3n_all_qc_eic.py`，仅换 zip + overview 输入）。
2. 使用 Stage-1 冻结的 DreaMS 嵌入和 mutual 3-NN 图（**不重建**）。
3. 计算 LIPn 患者效应矩阵 d^{LIPn}_{p,i}。
4. 对每个 Stage-1 家族组件（component），计算 LIPn 平台的家族内方差。
5. **复现判定**：Stage-1 家族内效应差的方向与 LIPn 家族内效应差的方向
   一致的实体对比例 ≥ 60%（对非 singleton 组件中的全部实体对）。

### 2.3 门

- `cross_platform_consistency_fraction >= 0.60`
- 置换检验 p ≤ 0.05（随机重排 LIPn 实体-家族映射后的分布作零）

## 3. Stage-2B：暗实体增量

### 3.1 定义

- **已知代谢物**：在任何公共谱库（GNPS/MoNA/MassBank）中 Top-1 余弦 ≥ 0.80
  的实体（从 Stage-1 的嵌入 + 本地 GNPS 库确定）。
- **暗实体**：其余实体（Top-1 余弦 < 0.80 或无候选）。

### 3.2 实验

在**同一冻结图和同一患者效应矩阵**上：
- 仅使用已知代谢物列做三分解 → 报告 family/within/isolated 分数 + sign-flip p
- 使用全部 263 列做三分解（= Stage-1 结果）→ 同上
- **增量** = 全部 - 已知物-only 的 (a) non-trivial coverage, (b) sign-flip p 值改善

### 3.3 门

- 暗实体至少贡献 30% 的非 singleton 覆盖（`dark_coverage_fraction >= 0.30`）
- 加入暗实体后 family 或 within 的 sign-flip p 值至少改善 10×（一个数量级）

## 4. Stage-2C：转换边一致性

### 4.1 数据

使用 Stage-1 输出的 `primary_decomposition.npz`（含 family_id、mz、效果矩阵）
和 `tasks/GLM_transformation_edges.py` 模块。

### 4.2 流程

1. 从 `primary_decomposition.npz` 提取 263 家元的精确质量（如需中性质量，
   由 precursor_mz 减质子换算）。
2. 以 `edge_candidates()` 构建全部候选转换边（ppm ≤ 20 Da）。
3. 以 `edge_differentials()` 计算 g_{p,e}。
4. 以 `patient_consistency()` 计算每条边的方向一致性 sign-test p。
5. 以 `transformation_null()`（PMD 置换）估计零分布。
6. BH 校正后报告 FDR ≤ 0.05 的边。

### 4.3 门

- `n_edges_with_fdr_05 >= 5`
- `null_median_p >= 0.20`（置换零的中位 p 值不能太低，否则说明边构造有偏）

## 5. 产出

- `lcnec_global_remodeling_stage2/run_<JOBID>/`
  - `lipn_consistency.json`（2A）
  - `dark_increment.json`（2B）
  - `transformation_edges.csv` + `edge_null.json`（2C）
  - `report.json`（终态：PASS / STOP / PARTIAL）

## 6. 三个实验的裁决

| 2A | 2B | 2C | 终态 |
|---|---|---|---|
| ✓ | ✓ | ✓ | **PASS**：进入 Stage-3（前瞻干预设计） |
| ✓ | ✓ | ✗ | **PARTIAL**：家族+暗增量成立，转换层降为描述性 |
| ✗ | ✓ | — | **PARTIAL**：仅暗增量成立，跨平台降为单平台 |
| ✓ | ✗ | — | **STOP**：降为已知代谢物谱学扩展，不进入 Stage-3 |
| ✗ | ✗ | — | **STOP**：整线关闭 |

## 7. 已冻结的参数

| 参数 | 值 | 依据 |
|---|---|---|
| ppm_tolerance | 20 | 覆盖 orbitrap 精度 + 同位素漂移 |
| cross_platform_consistency_threshold | 0.60 | 34 对患者中 ≥ 20 对同向 |
| dark_coverage_threshold | 0.30 | 暗实体至少占非 singleton 的 1/3 |
| sign_flip_improvement | 10× | 一个数量级 |
| n_edges_fdr_05_min | 5 | 至少 5 条独立转换边 |
| null_median_p_min | 0.20 | 零分布中位不能太低 |
| BH FDR level | 0.05 | 标准水平 |
| graph | Stage-1 冻结的 mutual 3-NN | 不重建 |
