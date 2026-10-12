# LCNEC P0：跨平台谱学实体效应迁移合同

日期：2026-10-10  
状态：实现完成；LIPn 表型盲实体宇宙就绪；HSST3n→LIPn 全局候选覆盖门失败，路线降级为窄技术验证  
替代范围：本文取代旧 `GLM_GLOBAL_REMODELING_STAGE2_CONTRACT_20261007.md` 的 Stage-2A/2B 主终点。旧 Stage-2C 仅保留为描述性探索。

## 1. 要回答的问题

在不使用 LIPn 肿瘤/癌旁标签选择匹配的前提下，HSST3n 的一个 QC 合格 ion family 能否：

1. 在 LIPn 中以受控的 link-FDR 找到同一实验实体；
2. 在 LIPn 中复现预先规定方向且超过最小效应的配对肿瘤效应；
3. 使 `unresolved_entity` 在完整多重检验宇宙中贡献 known-only 之外的复现实例。

同一 34 对患者、不同平台只属于技术复现。无论结果多强，均不得称独立生物学队列。

## 2. 冻结顺序

读取 LIPn 表型前必须完成并哈希：

- HSST3n 与 LIPn QC 实体 manifest；
- ion family、前体质量、离子形式、RT 辅助映射和谱图证据；
- 带 dustbin/no-match 的候选链接、`match_p` 和 `match_weight`；
- 同时声明 `phenotype_blind=true` 和 `validation_effects_unseen=true` 的 `link_freeze.json`。

链接表禁止出现 beta、effect、p-value、phenotype、disease 或 group label。分析器会拒绝此类输入并核对冻结哈希。

## 3. 身份状态

只允许四类：

- `confirmed_known`；
- `library_supported`；
- `unresolved_entity`；
- `artifact_or_redundant_ion`。

旧版以 Top-1 cosine 0.80 区分 known/dark 的规则废止。检索分数不能定义真实身份状态。

## 4. 联合检验

对发现实体 i 与验证候选 j：

- `p_match` 检验二者不是同一实体；
- `p_direction` 检验验证效应未沿发现方向变化；
- `p_min_effect` 检验验证效应未超过预注册最小幅度。

候选对使用三者最大值。若 i 有多个验证候选，以表型盲冻结权重做 weighted Bonferroni 并集；发现端和验证端再以最大 p 值构成两研究 partial conjunction。默认在完整发现实体宇宙上用 BY 校正。

`selected_joint_candidate_for_audit` 只说明哪个候选产生了校正后的最小联合 p 值，不得重新解释为表型盲 Top-1 链接。

## 5. 输出和主终点

实现：`tasks/analyze_entity_effect_transport.py`  
合同测试：`tasks/test_entity_effect_transport.py`

输出：

- `pair_replication_audit.csv`：逐候选匹配、方向、最低效应和加权联合证据；
- `entity_replication_results.csv`：逐发现实体的完整宇宙 q 值及 known-only q 值；
- `report.json`：分母、身份分区、泄漏声明、文件哈希和结论边界。

P0 主终点：

- 固定 link-FDR 下可进入效应检验的实体覆盖；
- 完整宇宙 BY q≤0.05 的复现实体数；
- 其中 `unresolved_entity` 的复现实例数；
- 反向、低于最小效应、歧义和 no-match 数量。

unknown 增量不能仅由“多出一个显著实体”宣布。正式增量还需质量、强度、缺失率和 MS/MS 信息量匹配的 known 对照，以及独立患者或 held-out 功能终点。

## 6. 当前执行状态

本地 Stage-1 主矩阵仅有 `.filepart`，LIPn 定量矩阵也不存在，因此不得产生真实 LCNEC 数字。LIPn 原始 zip 已完整存在（336,864,469 bytes；SHA-256 `6badd39a687d2b646f57c21a90115ef3f2fe6c58b439726c579bca15d97acfd1`），overview 也存在；缺口是跨平台实体提取、表型盲链接及效应矩阵，而不是原始包本身。当前只允许：

2026-10-10 已完成 LIPn 85/85 文件 acquisition audit：68 个 study、9 个 pooled QC、2 个 blank、6 个 dilution 均有 DDA；pooled QC 共 24,362 条 MS2，每文件唯一前体中位数 1,859。`pass_to_feature_headroom=true` 仅授权下一步表型盲实体构建，不是跨平台复现结果。

随后仅用 17 个技术对照注射完成 LIPn QC headroom，未读取 phenotype label：46,048 条 MS2 event 形成 1,586 个前体-RT family，其中 382 个同时满足 QC 重现性、blank 洁净度与 dilution 响应。该 382 是冻结代表谱图的候选实体宇宙；仍不能在缺少校准 `match_p` 和患者级效应时进入联合复现检验。

代表谱图冻结完成后，5 ppm 的表型盲候选空间仅有 6 个 pair，覆盖 HSST3n 的 6/263（2.28%）；其中 3/6 的未校准 sqrt-intensity cosine≥0.70，并存在两个发现 family 指向同一验证 family。`pass_to_global_transport=false`。该结果冻结为 headroom 阴性：禁止通过放宽 ppm、事后加入质量变换或读取效应来扩大候选集。LCNEC 后续至多校准并验证这 6 对，不能承担广覆盖 unknown-entity 方法的主证据。

1. 运行合成合同测试；
2. 在服务器核对完整 Stage-1 `primary_decomposition.npz`；
3. 不沿用 HSST3n 绝对 RT 直接提取 LIPn：先在 LIPn 独立形成 ion family 和 MS/MS 实体；
4. 表型盲构建 LIPn 实体和链接冻结包；
5. 冻结最小配对 log2FC 后一次运行本分析器。

任一必需文件缺失即停止。不得沿用旧 sbatch 的“LIPn 缺失则跳过 2A 后继续总体裁决”。

## 7. 结论边界

通过 P0 只支持：未定结构实验实体能够在同一患者的独立平台上完成受控技术效应迁移。

它不支持：

- 独立患者复现；
- 精确结构；
- 生物来源；
- 代谢反应、通量或酶活；
- 因果功能。

