# MTBLS13729 八候选统一证据账本（2026-08-30）

## 结论

原始 positive-RP mzML 重审计后，冻结八候选**全部具有峰界内 DDA MS2**，合计 `252` 张谱。此前“通用桥接表没有 accepted link”等同于“没有 MS2”的判断已被推翻。现在的主要瓶颈不是采集覆盖，而是：标准品终证、位置/同分异构体解析、候选选择后的统计边界，以及独立 Rmu 队列复现。

最重要的两项是：

1. **feature 1717**：当前最强的新候选轴。Rmu 9/9 上升，原始 MS2 中 `100.0759` 为 73/73 基峰；同源 HILIC HMDB0041947 候选在跨样本和配对差值中高度一致。可写为 `N1,N8-diacetylspermidine-like`，不可去掉 `-like`。
2. **feature 3222**：当前最强的类别锚点。59 张固定窗 DDA MS2、25 个样本通过强 carnitine motif，支持长链酰基肉碱类别；不能确认 C20:4 双键位置和立体结构。

## 八候选逐项账本

| feature | 当前最稳妥身份 | Rmu 配对效应 | 峰界内 MS2 | 正交/诊断证据 | 论文角色 | 禁止越界 |
|---:|---|---:|---:|---|---|---|
| 1597 | methylguanosine isomer family `[M+H]+` | +3.72 log2, 9/9 | 42/30 samples | 30/42 核糖丢失；与 7489 跨加合物一致 | 修饰鸟苷家族成员 | 不指定 m7G/m2G/Gm |
| 1717 | N1,N8-diacetylspermidine-like / acetylated-polyamine family | +3.01 log2, 9/9 | 73/45 | 100.0759 为 73/73 基峰；HILIC 跨样本 rho 0.860、配对差值 rho 0.719、85 项中 rank 1 | 最强新候选轴 | 无标准品不得写确切名称或 MSI Level 2/1 |
| 3019 | dimethylguanosine isomer family `[M+H]+` | +2.40 log2, 9/9 | 32/32 | 32/32 核糖丢失；与 8481 跨加合物一致 | 修饰鸟苷家族锚点 | 不区分 1,7-/N2,N2- 等位置异构体 |
| 3180 | unknown chlorinated/exogenous-like feature | +1.57 log2, 9 pairs | 24/24 | 谱图复现但缺乏合理内源身份 | 生物学合理性阴性对照 | 不进入内源代谢机制故事 |
| 3222 | long-chain acylcarnitine; C20:4-acylcarnitine-like | +1.78 log2, 8/10 | 30/30；固定窗 59/42 | 25 样本 strong carnitine motif；85.0281/60.0808 稳定 | FAO 利用受限/酰基肉碱锚点 | 不确认双键位置、立体结构、通量 |
| 4966 | C7H9N5O nitrogenous heterocycle / purine-like isomer family | +2.44 log2, 10/10 | 23/23 | 110.0347、153.0404、137.0817、135.0298 复现 | 嘌呤样/含氮杂环轴 | 不命名 preQ1；同式异构体未排除 |
| 7489 | methylguanosine isomer family `[M+Na]+` | +3.67 log2, 8/10 | 3/3 | 3/3 核糖丢失；与 1597 质量差一致 | 支撑性加合物 | 谱数少；不可作为独立结构终证 |
| 16425 | unknown reproducible lipid-like feature | +1.73 log2, 7/8 | 25/25 | 多个 25/25 复现碎片，但正离子证据不诊断 LPE | 探索性脂质峰 | 不能写 LPE 亚类或脂肪链 |

## 统计边界

- 八候选的 Rmu 配对效应是**候选筛选后的发现证据**，不是整个 feature space 的全局 FDR 结论。
- `n=8–10` 对适合报告效应、方向一致性、精确符号检验和留一患者稳定性，不适合把亚型机制写成确认级结论。
- primary endpoint 是 Rmu 肿瘤对癌旁；Rmu-vs-Rtu interaction 必须单独报告。当前不能把任何候选写成已证实 `mucinous-specific`。
- 静态丰度、转录和蛋白只能形成机制假设；没有同位素示踪/扰动时，不得声称通量或酶活改变。

## 下一步优先级

1. **N1,N8-diacetylspermidine 标准品**：当前信息增益最高，最可能把 1717 从强候选升级为命名代谢物。
2. **C20:4/C18/C16 acylcarnitine 标准组合**：确认类别、RT 和链级排序，检验是否是单峰还是类群重塑。
3. **m7G、m2G/Gm、m2²G 标准组合**：解决修饰鸟苷位置异构体，而不是只重复验证核糖丢失。
4. **16425 和 4966 反证优先**：先证明旧 LPE/preQ1 具体名称不成立，再决定是否值得购买标准品。
5. **独立组织队列**：只有具有配对正常和 Rmu/MSI/CMS 信息的原始数据，才能复核亚型异质性。

## 可复核产物

- `tasks/build_mtbls13729_candidate_evidence_ledger.py`
- `data/mtbls13729/candidate_evidence_ledger_v1/candidate_evidence_ledger.csv`
- `data/mtbls13729/candidate_evidence_ledger_v1/report.json`
- `data/mtbls13729/frozen_candidate_ms2_coverage_v1/`
- `data/mtbls13729/frozen_candidate_ms2_consensus_v1/`
- `data/mtbls13729/polyamine_crosschrom_audit_v1/`
- `data/mtbls13729/modified_guanosine_ms2_audit_v1/`
- `data/mtbls13729/c20_4_anchor_ms2_audit_v1/`

## 2026-08-30 扩展总账更新（优先于“八候选”口径）

全 feature-space 重提取、扩展 DDA 共识和原论文 source table 对账后，论文候选从 8 个探索峰扩展为 15 个证据分层节点。新增的高价值结果包括：

- source-table 同模式 RPLC 精确重映射：myristoylcarnitine（0.28 ppm，RT −2.86 s）、N1-acetylspermine（2.06 ppm，−1.46 s）、methylthioadenosine（−1.24 ppm，−2.50 s）、isoleucine（1.58 ppm，+6.64 s）和 phenylalanine（1.58 ppm，−3.19 s）；
- 跨面板同样本正交复核：carnitine、hypoxanthine 和 tryptophan 均通过，N1,N8-diacetylspermidine-like 继续保持最强跨色谱证据；
- taurine 虽有强 DreaMS 共识，但跨面板组织内与配对差值相关失败，正式降级；
- feature 722 的弱 DreaMS synephrine 投票被原论文 Level-1 phenylalanine 的前体/RT 证据推翻，这是“谱图模型候选必须受标准/色谱证据约束”的实例；
- 四个后验定义模块在 9–10 个 Rmu 配对中全部同向且留一代谢物方向稳定：乙酰化多胺–MTA、嘌呤/修饰核苷、长链酰基肉碱和大中性氨基酸。

扩展后的机器可读总账：`data/mtbls13729/integrated_biology_ledger_v1/integrated_candidate_ledger.csv`。完整裁决见 `docs/MTBLS13729_INTEGRATED_BIOLOGY_RESULT_20260830.md`。
