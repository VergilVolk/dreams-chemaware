# Contrast Atlas 同轴算法交付边界（2026-10-29）

## 结论

当前完整身份候选轴上，**只有 WSE 已完成逐对、逐块、全分母交付**。Official DreaMS、Noise V1、B 和 Full23 在该轴上的合格直接交付均为 **0 / 291,330 对**。旧 embedding 缓存只能说明可复用计算量，不能作为同轴算法结果。

本文件固定算法端与生物端的交接合同；不改变候选轴，不读取响应真值，也不把不同质量的化学关系送入身份排序器。

## 冻结身份轴

| 对象 | 分母 | SHA-256 |
|---|---:|---|
| observed-ion registry | 54,861 study-entities | `81c6251a29d5abc5c1ac1135ca57f6c749a8a0c4ee7c6985fbea01ae8e484b52` |
| directed complete candidate pairs | 291,330 | `741f3a6a4735d100121c16ec095d59eda872fe0cdde8c38cd25fe903c22994e2` |
| query-by-target-source blocks | 768,054 total；139,129 nonempty | `8108c15bee5884f9ae662d22efb09e432740ba176ba194c2c142610ed4b16ffd` |

轴文件位于 `data/validation/contrast_atlas_model_independent_identity_axis_v5_20261027/`。它是 15 个 source 上由同前体 10-ppm 及显式离子冲突排除规则产生的**身份假说轴**，不是分子身份真值。全部 291,330 对和每个非空候选块必须原样保留。

## 当前可用交付

| 方法 | 当前轴合格逐对分数 | 状态 |
|---|---:|---|
| WSE | 291,330 / 291,330 | 已发布；全分母、truth-blind |
| Official DreaMS | 0 / 291,330 | 未交付 |
| Noise V1 | 0 / 291,330 | 未交付 |
| B | 0 / 291,330 | 未交付 |
| Full23 v1 | 0 / 291,330 | 未交付 |

WSE 发布包位于 `data/validation/contrast_atlas_complete_identity_wse_release_v1_20261029/`。此前 fixed-edge、mutual-top1 或疾病定向输出均不是这条完整轴上的可替代交付。

## 旧缓存的潜在覆盖

下表是只读资产盘点，表示“若精确映射到冻结 registry 后，可少算多少编码”，不表示已完成同轴评分。

| 旧资产 | 可映射实体或候选对 | 相对完整轴 | 完整非空块 | 解释边界 |
|---|---:|---:|---:|---|
| Official DreaMS embedding cache | 40,182 / 54,861 entities；195,908 / 291,330 pairs | 67.25% pairs | 88,252 / 139,129（63.43%） | 其余端点仍须编码；不能只评价完整缓存子集 |
| Noise V1 embedding cache | 16,329 / 54,861 entities；34,378 / 291,330 pairs | 11.80% pairs | 未形成合格全轴块交付 | 缓存集中于三个 ChemProp source，不能代表全 Atlas |
| ChemAware Stage-1 cache | 94,596 / 291,330 pairs | 32.47% pairs | 未形成合格全轴块交付 | 可作为后续通道缓存，不是 Full23 输出 |
| B / Full23 bundle | 无完整当前轴 evidence bundle | 0% direct delivery | 0 | 必须由相同完整块重新生成全部所需通道 |

## 全量重编码的理论上限与 fail-closed 分母

15 份当前 MGF 均在本地。以 encoder 至少 3 个合格峰的输入条件计算：

| 项目 | 可评分上限 | 必须 fail closed |
|---|---:|---:|
| entities | 54,318 / 54,861 | 543 |
| directed candidate pairs | 284,354 / 291,330（97.61%） | 6,976 |
| nonempty candidate blocks | 134,533 / 139,129（96.70%） | 4,596 |

这是输入可编码性的理论上限，不是模型实际覆盖。6,976 对与 4,596 个块必须以明确的 `missing_required_evidence` 或等价状态留在固定分母中；不得删除、换候选或用 WSE/其他模型分数填补。

## 最短合格适配器合同

1. **输入不变。** 读取上述三个冻结文件并核对 SHA；不得按任何模型阈值、Top-k、mutual-nearest、响应方向或登记关系增删候选。
2. **完整块推理。** 对每个 query-by-target-source block 一次性提供其全部候选。Full23 的 query-local 特征不得在 singleton 化的边表上重算。
3. **冻结模型。** 逐项登记方法版本、checkpoint SHA、输入清洗合同及每个 source 的 embedding/evidence cache SHA；Stage-1、Phase-A、Noise 与其他通道不得混名。
4. **精确对齐。** 输出必须与 291,330 行候选轴一一对应，优先保持原始行序，并同时带稳定 pair key、block index、有限分数和 availability/reason 字段。
5. **缺失不改分母。** 零候选、单候选、多候选、端点不可编码和缺少必要通道均保留；不可评分块明确弃权，不能静默丢弃或回填成其他方法的判决。
6. **truth-blind。** 编码、评分、校准和候选准入均不得读取 beta、SE、表型、登记关系或身份真值。生物端只在冻结分数交付后原样运行相同响应检索。
7. **最小交付物。** 每个方法交付逐对分数、逐块决定、availability mask、运行报告、输入/输出校验和及端到端命令；随后才能与 WSE 在相同 291,330 对及 139,129 个非空块上比较。

最短实施路径是：先写一个 current-axis adapter，复用可精确映射的旧缓存，同时对缺失的 Official 和 Noise 端点进行全量重编码；再生成 B/Full23 所需的其余证据通道与完整候选块推理。不能用部分 cache 的交集替代这条路径。

## 当前不得宣称

- 不得宣称 Official、Noise V1、B 或 Full23 已在该 Atlas 完整轴上优于 WSE；它们尚未交付同轴结果。
- 不得把旧 fixed-edge、mutual-top1、ChemProp-only 或疾病定向分数称为 291,330 对同轴评测。
- 不得把理论可编码上限或缓存可覆盖比例写成实际模型覆盖、性能或增益。
- 不得由本轴报告身份准确率、FDR 或 no-match 特异性；当前没有覆盖全轴的独立身份真值。
- 不得把身份候选分数外推为不同质量化学关系的有效分数，也不得合并不同实体的 beta/SE。
- 不得宣称 unresolved-information increment、Full23 biological increment 或 BioAware increment；这些均需在相同固定分母上另行检验。
- 所有现有 Atlas 数据均为 `development/consumed`，不得称为封存外部生物验证。

因此，当前生物端可继续报告 WSE 的系统级开发结果，但任何“自研算法使 Atlas 更好”的结论，都必须等待上述同轴交付完成。
