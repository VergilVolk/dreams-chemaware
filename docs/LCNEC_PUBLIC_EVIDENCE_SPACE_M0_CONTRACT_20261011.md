# LCNEC 公共谱学证据空间 M0 冻结合同

日期：2026-10-11  
状态：**development/consumed 预注册；尚未运行真实公共层；不得作为独立生物学验证**

## 1. 要回答的问题

在完全不使用 TU/NG 表型构建公共单元、证书或样本表示的条件下，独立公共
MS/MS 是否能把 LCNEC 的 263 个表型盲 QC-qualified ion families 变成具有
可审计覆盖率和弃权的样本级化学组成；这种组成是否比 MS1、研究内 MS/MS
和伪公共层更好地保留个体信息，并且没有主要编码批次、注射顺序或测量深度？

M0 不检验分子身份、代谢通路、反应、肿瘤机制或临床效用。

## 2. 冻结研究对象与数据地位

- 数据集：LCNEC HSST3n；**development/consumed**，曾参与既往方法探索；
- 目标宇宙：仅由 9 pooled QC、2 blank 和 6 dilution injections 决定的
  263/1,138 families；表型标签未参与选择；
- 技术资格门：5 ppm、15 s、至少 6 个 QC、blank/QC <= 0.20、dilution
  Spearman rho >= 0.70；
- MS/MS：263/263 条代表 QC 谱已存在；只能作为观测谱学证据，不能当身份；
- 样本：68 study injections，34 个患者各 TU/NG 一对；另有 17 个技术注射；
- 丰度：预期为 85 x 263 的 EIC matrix。当前本地只有 32,752-byte
  `.filepart`，因此真实 M0 运行状态是 `INPUT_NOT_EVALUABLE`；不得从旧报告
  反推或重建矩阵数值。

公共参考层必须排除本 LCNEC study、其镜像、重处理副本及相同原始文件哈希。
公共来源必须以真实 study/source 为单位，不能把同一文件的多个谱图计为独立来源。

## 3. 冻结表示与四个世界

所有世界使用同一 263-family 行列、同一 abundance/reliability 变换和同一样本。

1. **MS1-only**：`log1p(EIC) * reliability`，不读取公共 MS/MS；
2. **study-internal spectral**：仅使用 263 条研究内代表谱的冻结谱学关系，
   无公共来源；
3. **real-public**：独立公共来源形成的多来源、多证据族 unit certificate；
4. **matched-shuffle public**：保持每个 query 的 m/z bin、峰数、仪器、来源数、
   certificate score 及逐 query 覆盖/弃权状态，在冻结技术分层内置换
   `public_unit_id`；以重复置换的完整 null 分布裁决，不挑选单次置换。

真实公共层只能在 WSE 与算法组交付的合格冻结表示中选择证据族；不得把
official DreaMS、Noise、P2b、ChemAware 及其派生组合重复计为相互独立证据。
公共证书至少要求两个合格证据族、两个独立 source、source dominance <= 0.80；
并列或无资格候选进入 `__NO_MATCH__`。所有阈值必须来自非 LCNEC 的 consumed
calibration shard，并在读取 TU/NG 前冻结。

## 4. 标签打开顺序与终点

### M0-A：无表型构建及技术否证

构建证书和样本 unit occupancy 时禁止读取 `GROUP_CODE`、TU/NG、疾病、结局或
结构真值。首先报告：

- 263 queries 中获得证书的数量及比例；
- 每个样本按 `log1p(abundance) * reliability` 加权的 certified coverage、
  dustbin fraction 和公共 unit 数；
- coverage 与 TIC、总检测 family 数、MS2 获取深度、注射顺序和 batch 的关联；
- 公共 unit 的 source dominance、仪器覆盖、冲突和 leave-one-source-out 稳定性。

若 median certified coverage < 0.20，或任何单一公共 source 支配主结论，M0
停止；低覆盖样本的已认证部分不得归一化后冒充完整化学组成。

### M0-B：隐藏标签的个体保持

只有 M0-A 冻结后，使用患者 ID 作为**仅评价**标签：对 68 个 study samples
做 leave-one-sample-out mate retrieval，候选为其余 67 个样本，唯一正确项为
同患者另一组织，均匀随机 R@1 为 `1/67`。报告 R@1、MRR、34-patient cluster
bootstrap CI，并与四世界同图比较。

同时预测 batch、注射顺序分位、TIC、检测 family 数和缺失率；这些是负终点。
若 real-public 对患者 mate retrieval 不超过前三世界最强值，而主要改善技术
终点预测，则失败。

### M0-C：最后打开 TU/NG

只有 M0-A/B 的方法和超参数均冻结后才读取 TU/NG：

- 在患者内计算 unit-level paired effect；
- 使用患者级 sign flip，不把 263 families 或多个 units 当独立样本；
- 比较 real-public 与 MS1-only、study-internal、matched-shuffle；
- 报告 effect、34-patient CI、multiplicity、coverage 与 leave-one-source-out；
- 若存在预先冻结的 library-supported ledger，才附加 known-only、unresolved-only、
  known+unresolved；不得用本队列显著性定义 unknown 子集。

M0-C 只生成 LCNEC 内部表型关联。没有独立队列或前瞻扰动时，不得写成机制。

## 5. 晋级与停止

只有同时满足以下条件，才能进入第二个来源：

1. real-public 在 patient-cluster CI 下超过前三个世界的最强基线；
2. 增量在 leave-one-source-out 下方向稳定；
3. 增量不能由 coverage、TIC、batch、注射顺序或检测深度解释；
4. unresolved 加入已知信息后有条件增量，而非只增加节点数量；
5. matched-shuffle 不能复现主结果。

任一条件失败，本路线降级为带 provenance 的公共谱学查询/覆盖工具，不称
通用生物学发现方法。LCNEC 阳性也只允许选择一个此前未参与规则开发的完整
非药物来源做一次冻结验证；LIPn 与 HSST3n 为同一患者、不同平台，不是独立
生物学队列，且既有 5 ppm 头部审计仅覆盖 6/263（2.28%），不能承担全局验证。

## 6. 已实现和当前唯一硬缺口

- `tasks/build_public_spectral_evidence_certificates.py`：多来源、多证据族、可弃权证书；
- `tasks/project_public_evidence_certificates_to_samples.py`：证书到样本 unit occupancy，
  明确保留 dustbin 和 coverage；
- `tasks/shuffle_public_evidence_certificates.py`：保持 coverage/confidence 的匹配伪公共层；
- 合同测试已覆盖合格 assignment、单来源弃权、泄漏来源拒绝、完整 query dustbin、
  样本投影、outcome-column 拒绝和 matched-null 不变量。

真实执行仍缺：完整 85 x 263 EIC matrix，以及排除了 LCNEC 派生数据、包含
冻结 WSE/合格算法分数和真实 source/instrument provenance 的公共候选 edge table。
在二者齐备前不输出任何 LCNEC M0 生物学数字。
