# Validation Candidate Board（2026-09-19）

## 1. 用途与边界

本表只整理仓库中已经冻结、可追溯的真实样本候选，用于选择最少的同平台标准品实验。它不是新的身份判定，也不以表型、通路合理性或 BioAware 网络支持替代谱学身份。

共同规则：

- 身份裁决时不得读取疾病标签、患者效应、通路结果或任何隐藏真值；这些信息只用于身份裁决完成后的研究价值排序。
- “标准品可用”必须区分：仓库已有公共标准谱、可采购线索、组内实物库存、以及是否仍有可加标的原样/混合样。当前仓库没有证明下列 authentic standards 已在组内，也没有证明所有队列仍有剩余提取物。
- library spectrum、精确质量、DreaMS、离子家族、跨色谱相关和网络 context 均不能单独产生 Level 1 身份。
- 同平台升级至少需要 authentic standard 的 RT 与 MS/MS；存在位置异构体或共洗脱风险时还需竞争标准并跑、样本 spike-in/co-injection 和碰撞能系列。
- 任何失败都保留原有有界结论，不为维持故事更换阈值、删除竞争候选或用生物学合理性覆盖化学冲突。

## 2. 候选总表

| 对象 | 冻结原始证据与样本 | A/B 竞争问题 | 标准品/湿资源状态 | 当前允许身份层级 | 核心资源缺口 |
|---|---|---|---|---|---|
| MTBLS13729 f1597 | positive-RP，`m/z 298.114285`，RT `217.00 s`；42 张 peak-resolved MS2/30 样本，30/42 支持核糖丢失；与 f7489 `[M+Na]+` 一致 | m7G vs m2G；少数 `m/z 152.0558` 还要求保留 Gm/共洗脱/CE 依赖解释。公开 m7G/m2G 最佳 sqrt-cosine 仅差 `0.00450` | 仓库已有 MassBank authentic-standard 谱；没有 m7G、m2G、Gm 实物在手或剩余提取物库存证明 | methylguanosine positional-isomer family，MSI level 3-like | 三个竞争标准、同法 RT、完整镜像谱、multi-CE、样本 spike-in、双加合物核对 |
| MTBLS13729 f3019 | positive-RP，`m/z 312.130679`，RT `248.88 s`；32 张 peak-resolved MS2/32 样本，32/32 支持核糖丢失；与 f8481 `[M+Na]+` 一致 | m2²G 与 m2,7G/1,7G 等二甲基鸟苷位置异构体；`312→180` 只支持相容性，不排除位置异构体 | 仓库有公共方法/谱学背景；没有竞争标准实物或剩余提取物库存证明 | dimethylguanosine positional-isomer family，MSI level 3-like | m2²G 及可得竞争异构体、同法 RT/MS2、multi-CE、spike-in、双加合物核对 |
| MTBLS13729 f703 | positive-RP，`m/z 310.113362`，RT `49.27 s`；33 张 peak-resolved MS2；与 source Level-1 negative-HILIC Neu5Ac 在59个共同样本相关 `0.959`，source rank 1 | 冻结账本没有一个合格的命名 runner-up；真正问题是 positive-RP 峰是否与 Neu5Ac standard 同法共洗脱，还是同质量/共洗脱/降解峰 | source 面板已有 Level-1 Neu5Ac 身份，当前 positive-RP 未做同法 authentic-standard injection；组内 Neu5Ac 实物和样本余量未记录 | same-cohort orthogonal source recovery；不得把 positive-RP feature 写成新的同法 Level 1 | Neu5Ac standard、同梯度 RT+同 CE MS2、pooled/sample spike-in、峰肩/降解检查；最好有 isotope internal standard |
| MTBLS13729 f1717 | positive-RP，`m/z 230.185931`，RT `89.57 s`；73 张 peak-resolved MS2/45 样本，`100.0759` 为 73/73 基峰；跨色谱相关 `rho=0.860` | N1,N8-diacetylspermidine-like vs 其他 acetylated-polyamine/位置异构体；仓库没有冻结一个已被充分谱学表征的唯一 B 候选 | 未完成兼容 experimental library match、同法 standard 或 spike-in；标准实物和样本余量未记录 | acetylated-polyamine ion family / N1,N8-diacetylspermidine-like；低于 MSI Level 2 | N1,N8-diacetylspermidine standard及可得位置异构体、同法 RT、multi-CE MS2、spike-in |
| MTBLS13729 f3222 | positive-RP，`m/z 448.339463`，RT `631.07 s`；30 张 peak-resolved MS2/30 样本，固定窗59张；25样本有 strong carnitine motif | C20:4-acylcarnitine-like vs 其他链/双键位置/立体异构体及较弱 ion-form 假设；类别碎片不能定位双键 | 仓库建议 C20:4/C18/C16 标准组合，但没有标准在手、同位素内标或样本余量证明；独立 MTBLS8090 未复现泛CRC升高 | long-chain acylcarnitine class / C20:4-acylcarnitine-like，MSI level 3-like | 同法 acylcarnitine 标准组合、multi-CE、RT/MS2、spike-in；精确定量需 isotope internal standard；生物推广需独立Rmu队列 |
| LCNEC quinolinate | HSST3n，`m/z 166.014725`，RT `54.79 s`；5个匹配碎片，coverage `0.846`，sqrt-cosine `0.896`；34对中28对同向 | quinolinate vs same-formula top-five rival；冻结直接竞争例为 3-nitrobenzoate，谱学领先仅约 `0.061` | 34对 raw arm 与QC工件在仓库；未记录 quinolinic acid standard 已在手或剩余LCNEC提取物可加标 | MSI Level-2 compound hypothesis；不是 exact/Level-1 identity | authentic quinolinic acid；必要时3-nitrobenzoate竞争标准；同HSST3n RT/MS2、co-injection；剩余提取物/pooled matrix确认 |
| LCNEC ascorbate | HSST3n，`m/z 175.024979`，RT `39.74 s`；15个匹配碎片，coverage `0.973`，sqrt-cosine `0.975`；34对中32对同向 | ascorbate vs frozen runner-up D-glucuronolactone；谱学领先约 `0.305`，但仍缺同法RT | 34对 raw arm 与QC工件在仓库；未记录 ascorbic acid standard 已在手、氧化控制或剩余提取物可加标 | MSI Level-2 compound hypothesis；不是 exact/Level-1 identity | fresh ascorbate standard、D-glucuronolactone对照、氧化/稳定性控制、同HSST3n RT/MS2及co-injection |

## 3. 逐对象验证合同

### 3.1 f1597/f3019：modified-guanosine 竞争标准包

**目的。** 解决位置异构体，而不是再次证明核糖丢失或重复计算丰度。

**步骤。**

1. 先核实 m7G、m2G、Gm、m2²G 以及可得二甲基鸟苷竞争异构体的实物库存、纯度、盐型和采购周期；没有实物不得标记 `READY`。
2. 冻结当前 positive-RP 梯度、离子模式、RT窗和现有样本谱；身份判定人员不读取Rmu/Rtu标签。
3. 标准单独进样并做碰撞能系列；m7G 与 m2G 必须同批并跑，不能只运行预期阳性标准。
4. 在同一 pooled/sample matrix 做加标或co-injection，检查共洗脱、完整镜像谱和峰肩；同时核对 `[M+H]+/[M+Na]+` 是否共洗脱。
5. Level 1 升级门：前体误差不超过 `5 ppm`、同法共洗脱、主诊断碎片及至少两个次级离子的相对关系一致，并在碰撞能系列下稳定；所有实际可得竞争标准必须被排除。

**停止门。** 两个竞争标准仍共洗脱且完整谱不可分、样本出现多峰/混合谱、或加标产生独立峰肩时，停止唯一命名，保留相应 methyl-/dimethylguanosine isomer family。不得用转录、蛋白或通路方向裁决异构体。

### 3.2 f703：Neu5Ac 同法升级

**目的。** 将 positive-RP feature 的跨面板强桥接变成同方法化学验证；不是重新发现 Neu5Ac。

**步骤。**

1. 核实普通 Neu5Ac authentic standard、可加标的 pooled/sample extract 与稳定同位素内标是否实际存在。
2. 按原 positive-RP 方法运行 standard、样本和样本+standard；锁定 RT 后再比较 MS/MS，不得按结果移动峰界。
3. 检查共洗脱、主/次碎片、峰肩和可能的降解/共洗脱峰；定量升级时再引入 isotope internal standard。
4. positive-RP Level 1 只在同法 RT、MS/MS 和 spike-in 均支持时成立；negative-HILIC source Level 1 不能自动转移成另一色谱模式的 Level 1。

**停止门。** standard 与 feature 不共洗脱、加标出现独立峰肩、MS/MS存在实质冲突或样本余量不足以完成加标时，维持 `same-cohort orthogonal source recovery`。不得将 free Neu5Ac 丰度直接解释为糖链linkage、来源、通量或净表面唾液酸化。

### 3.3 f1717：acetylated-polyamine family

**步骤。** 先确认 N1,N8-diacetylspermidine 及竞争位置异构体可得性；同positive-RP方法运行multi-CE standard、样本和spike-in，比较RT、完整谱及 `100.0759/114.0916/72.0445` 等离子关系。

**停止门。** 若没有竞争标准、标准不共洗脱、或位置异构体不可分，继续报告 acetylated-polyamine family / `-like`，不得写 N1,N8 确证、SAT1致因或MSI Level 2/1。

### 3.4 f3222：long-chain acylcarnitine class

**步骤。** 先核实C20:4与代表性C16/C18标准组合；同法运行RT、multi-CE MS/MS和spike-in，检查carnitine motif、链级顺序及是否存在多个色谱峰。若进行定量，另用稳定同位素内标。

**停止门。** 单一标准只能支持类别/链级而不能排除双键位置和立体异构体；独立Rmu队列未复现时，停止亚型/普适CRC机制扩展。任何结果均不得由静态丰度推出FAO通量、CPT1A活性或酶活。

### 3.5 LCNEC quinolinate

**步骤。** 核实 quinolinic acid 与3-nitrobenzoate标准、原HSST3n方法可重现性、pooled matrix/剩余提取物；在盲于患者效应的条件下完成两标准并跑、RT、multi-CE MS/MS和co-injection。身份裁决完成后才关联QPRT/NAD背景。

**停止门。** 两标准不能在本方法下区分、quinolinate不共洗脱、或样本峰呈混合谱时，维持Level-2 hypothesis。蛋白QPRT下降不能确认代谢物身份，也不能证明利用瓶颈或NAD通量。

### 3.6 LCNEC ascorbate

**步骤。** 核实 fresh ascorbic acid 与D-glucuronolactone标准；预先固定样品处理、避光/低温/抗氧化和进样时间规则；同HSST3n方法比较RT、multi-CE MS/MS及co-injection，并记录氧化产物。

**停止门。** 氧化/降解使standard不稳定、竞争物不可区分、或样本峰不共洗脱时维持Level-2 hypothesis。不得宣称抗氧化依赖、治疗作用、统一PPP激活或代谢通量。

## 4. 首批实验对象（最多三个）

### 1. f703 Neu5Ac

最接近低成本闭环：已有33张positive-RP MS2、source Level-1锚、59样本跨面板相关和明确的同法standard+spike-in门。它能最快补上MTBLS13729主生物轴最薄弱的“本方法身份”一环。其价值是身份升级，不是新算法首次发现。

### 2. f1597/f3019 modified-guanosine 竞争标准包

这是候选板中最真实的近异构体裁决问题，能检验“算法排序是否经得住竞争标准”，而不是只验证一个预期阳性名称。把两个feature作为一个实验包，是因为它们共享鸟苷骨架、同一色谱方法、加合物核对和multi-CE采集；但每个feature必须独立给出身份结果。若资源有限，最低先做m7G+m2G并跑和m2²G；Gm及其他二甲基异构体不能被文字性排除。

### 3. LCNEC quinolinate（条件启动）

它是LCNEC中最具机制区分度的化学确认目标，而且存在明确same-formula竞争物与较小谱学margin；验证成功会比重复一个已经很强的库谱匹配更有信息量。只有在确认quinolinic acid/3-nitrobenzoate标准、原HSST3n方法和可加标matrix后才转为`READY`。否则不自动用生物学故事替代，而由资源审计重新排序。

f1717虽是强候选，但当前连兼容experimental library match与标准实物都未确认；f3222需要更大的标准组合和独立Rmu复现才能升级主要主张；ascorbate效应最大且谱学分离较强，但相对quinolinate的机制消歧信息较低。因此三者暂列第二批，不代表否定其现有有界结果。

## 5. 状态与来源

当前所有对象均为 `RESOURCE_AUDIT_REQUIRED`；没有任何湿实验因本表而自动获得授权或被标记为已购标准。

主要冻结来源：

- `data/mtbls13729/candidate_evidence_ledger_v1/candidate_evidence_ledger.csv`
- `data/mtbls13729/integrated_biology_ledger_v3/integrated_candidate_ledger_v3.csv`
- `docs/MTBLS13729_BIOLOGY_CLOSURE_AND_MINIMAL_VALIDATION_20260830.md`
- `docs/MTBLS13729_ACETYLATED_POLYAMINE_MECHANISM_AUDIT_20260830.md`
- `docs/MTBLS13729_ACYLCARNITINE_BIOLOGY_DECISION_20260820.md`
- `data/validation/lcnec_hsst3n_identity_claim_defense_v1/report.json`
- `data/validation/lcnec_hsst3n_manuscript_evidence_package_v2/report.json`

