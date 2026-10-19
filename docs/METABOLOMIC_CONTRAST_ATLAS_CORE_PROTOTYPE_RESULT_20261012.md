# Metabolomic Contrast Atlas 最小核心原型结果

日期：2026-10-12  
数据角色：`synthetic_contract_smoke`  
正式性：否；仅验证软件语义，不是生物学或方法性能结果

## 结果

机器合同、执行器和单元测试已经建立：

- `tasks/contrast_atlas_schema_v1.json`
- `tasks/contrast_atlas_core.py`
- `tasks/test_contrast_atlas_core.py`
- `tasks/fixtures/contrast_atlas_smoke/`

`python -m pytest -q tasks/test_contrast_atlas_core.py`：`3 passed`。

端到端输出为 `data/validation/contrast_atlas_core_smoke_20261012_v2/`。输入包含 13 个 contrast records、13 个 pattern loadings、9 个 strict links；query 为合成 study `Q/q`，3 个目标均来自不同 source family。

在 `all` 世界：

| 目标 | strict entities | strict score | common patterns | pattern score | 输出 |
|---|---:|---:|---:|---:|---|
| S | 4 | 0.9993 | 2 | 1.0000 | similar response |
| O | 4 | -0.9993 | 2 | -1.0000 | opposite response |
| L | 1 | 1.0000 | 1 | 1.0000 | abstain: insufficient comparable evidence |

目标 L 即使单个证据完全同向也必须弃权，证明覆盖门优先于高相似度。所有比较同时要求 query 和 target 两侧 coverage 过门。

## 已通过的失败保护

1. evidence table 出现 `beta/effect/phenotype/outcome` 等列立即拒绝；
2. strict entity link 不是 contrast pair 内一对一立即拒绝；
3. link 端点不存在或未通过 QC 立即拒绝；
4. 同一 independent unit 被多个合格实体重复使用立即拒绝；
5. 默认拒绝同一 source family 作为跨来源结果；
6. known-only、unresolved-only 和 all 分开计算，不允许以总 feature 数替代独立证据数。

## 尚未通过、不得宣称

- 没有真实研究结果；
- 没有 empirical null、FDR、cluster bootstrap 或 source-held-out calibration；
- 没有 measurement-matched known control；
- 没有证明未知信息增量；
- 没有证明 WSE、DreaMS、Noise、P2b 或 ChemAware 改善生物学检索；
- 没有独立 Atlas 外测。

下一步必须先把 ChemProp、ST003325 和 LCNEC 的既有负结果转换成统一 schema 并重放；只有统一核心保持原 FAIL/abstain，才进入 MSV000096214 consumed 正控和第二独立来源。

