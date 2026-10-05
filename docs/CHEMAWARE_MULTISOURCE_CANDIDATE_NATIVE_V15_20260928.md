# ChemAware 多源候选化学证据原生 Triplet 路线 V15

日期：2026-09-28

状态：`IMPLEMENTED / LOCAL SOURCE-GEOMETRY AUDIT PASS / NO NEW WEIGHT UPDATE`

## 核心裁决

SIRIUS 不是路线本身，只是一个可替换的候选证据源。路线本身是：

> 用独立于 DreaMS 相似度的候选结构条件化化学证据，找出受保护 Phase-A
> embedding 当前仍未分开的真实候选边界，再把这些边界转换成 DreaMS 原生
> identity triplet；不蒸馏教师分数，不改训练器和损失。

可接入的来源包括 SIRIUS Tree/CSI、ICEBERG/GLACIER/FIORA/FraGNNet 的
structure-to-spectrum 分数，以及未来通过独立控制门的碎片子图证据。来源分数
不可直接相加或平均。

## 为什么这次不是重做旧实验

旧 ICEBERG candidate-residual 训练失败，证明连续教师残差不能稳定传入共享
embedding；它没有否定 ICEBERG 的候选排序信息。现有冻结 ICEBERG 审计在
700 个偏难训练 query 上，对 619 个 official errors 救回 348 个，并显著超过
candidate-swapped 与 peak-permuted controls。V15 只使用这个来源选择 `(q,p,n)`，
进入梯度的监督仍是正确 identity positive 与错误 identity negative。

## 多源裁决

对当前 Phase-A active candidate pair `(t,c)`，来源 `s` 的支持条件为：

1. 来源在自己的适用域内可比较 `t` 与 `c`；
2. `score_s(t) > score_s(c)`；
3. `t` 是该来源适用候选集合中的严格第一；
4. 若该来源有 matched controls，则正确优势严格大于每个 control 优势；
5. 任一其他可比较来源若给出 `score_s(c) >= score_s(t)`，该 pair 整体丢弃。

不同来源只通过支持集合和冲突集合组合，不做数值融合。保留下来的边界按支持
来源数和 Phase-A 当前难度排序，每个 query 最多追加两个 singleton native
triplet。完整 Phase-A pool 必须是新训练池的逐字节数组前缀。

## 本地零更新结果

现有 ICEBERG 冻结结果已映射为 truth-blind common ledger：

- 700 queries；
- 3,788 candidate rows；
- 不导出 label；
- 每行同时保留 correct、candidate-swapped、peak-permuted 三个冻结分数；
- graph SHA256：`55b67c6684bf6750cc7d722df5029daac7b5e43856d01b922ec46e1141bfa948`；
- teacher SHA256：`e38d53c86c4d9a8be564c8ac7d09f4806972a9adf31fee53c8b5fcba960da0fb`。

使用 official geometry engineering stand-in 做构造审计时：

- 完整保留 5,956 个 Phase-A events；
- 新增 419 个不重复 source-proven triplets；
- 覆盖 297 queries、262 formula clusters；
- 364 个 error-boundary events，55 个 active-margin events；
- 271 / 621 个当前错误 winner 有直接 source proof；
- 条件 current-winner headroom 为 43.6393%；
- identity contract、exact-prefix、singleton、no-fusion gates 全部通过。

这些数字是 source/geometry headroom，不是 embedding 性能提升。正式数量必须由
受保护 `+2.1266 pp` Phase-A checkpoint 重新编码后得到。

## 唯一训练入口

```bash
sbatch tasks/run_chemaware_multisource_native.sbatch
```

该任务严格申请一个 GPU、不手动指定内存；直接加载 Phase-A checkpoint；先执行
最低覆盖门，再训练 250/500/750/1000 四个 checkpoint；role 2 只相对 Phase-A
进行 paired selection，且公式簇 bootstrap CI 下界必须为正。没有合格增量时保留
Phase-A 并停止，不打开 role 3。

如已有完成的 SIRIUS score directory，可选：

```bash
SIRIUS_SCORE_DIR=data/validation/chemaware_sirius_source/run_x/scores \
  sbatch tasks/run_chemaware_multisource_native.sbatch
```

SIRIUS 不存在时，ICEBERG 单源路线仍可独立运行；未来 GLACIER 等来源只需输出同一
ledger schema，无需修改 DreaMS 训练代码。
