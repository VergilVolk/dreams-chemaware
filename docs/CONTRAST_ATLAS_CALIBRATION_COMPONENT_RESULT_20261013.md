# Contrast Atlas 化学对齐经验 null 组件

日期：2026-10-13  
状态：开发组件；只完成合成语义验证

## 方法

对已经通过 shared-pattern coverage 门的 query--target contrast pair，固定：

- query 与 target 各自 pattern 数量；
- 每项研究的 pattern effect 与 SE 边际分布；
- 双方共同可观测 pattern 集合；
- query 的 effect--pattern 对应。

只置换 target 中 `(effect, SE)` 与 `pattern_id` 的对应，重算 uncertainty-weighted signed cosine。分别计算 similar-tail、opposite-tail 和 two-sided empirical p，并在同一 query/world 的所有目标间做 Benjamini--Hochberg 校正。

该 null 回答：观察到的高分是否依赖具体化学模式上的方向对齐，而不是任意重排效应都能得到。它不控制 source、instrument、ion mode、feature count 或 biological-label confounding，因此只能作为内层校准；正式 Atlas 仍需外层 matched-source null 与 held-out source family。

## 实现与验证

- 实现：`tasks/calibrate_contrast_atlas_pattern_null.py`
- 测试：`tasks/test_calibrate_contrast_atlas_pattern_null.py`
- 固定种子：`20261013`
- 默认 draws：999
- 默认 minimum common patterns：8
- 默认 FDR：0.10

定向合成测试中，10 个共同模式的完全同向和完全反向响应分别得到对应方向 empirical p <= 0.01。BH 校正与证据不足弃权测试通过。

旧两模式 smoke 的重放结果位于 `data/validation/contrast_atlas_pattern_null_smoke_20261013/`：9 个 world--target rows 中 `0` 个达到校准资格；两个模式即使 descriptive cosine 为 `+1/-1`，仍返回 `abstain_insufficient_patterns_for_null`。这说明校准层没有把极小分母的完美相关升级为发现。

当前 Contrast Atlas 相关测试总计 `6 passed`。没有真实数据 calibrated association，因此不得报告方法准确率。

