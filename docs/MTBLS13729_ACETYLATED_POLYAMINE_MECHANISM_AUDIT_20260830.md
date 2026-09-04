# MTBLS13729 乙酰化多胺机制链复核（2026-08-30）

## 1. 裁决摘要

`feature 1717` 是当前信息密度最高的新候选之一，但仍只能写为
`N1,N8-diacetylspermidine-like / acetylated-polyamine family`。它满足：

- Rmu 配对组织中 9/9 上升，平均约 `+3.01 log2`；
- 峰界内 73 张原始 DDA MS2、覆盖 45 个样本，`m/z 100.0759` 为 73/73 基峰；
- 同源 HILIC HMDB0041947 候选在 59 个样本中与 RP feature 1717 的 Spearman
  `rho=0.860`，患者配对差值 `rho=0.719`，在 85 个 HILIC 注释中排名第 1；
- 原研究正文没有讨论 spermidine、polyamine 或 acetylated polyamine，因此“Rmu
  乙酰化多胺候选轴”是重分析增量，不是复述原论文结论。

但是标准品同法 RT、标准品 MS2 和 spike-in 共洗脱尚未完成；HILIC 来源表也没有
碎片可靠性字段。因此它不是 MSI Level 1，也不能去掉 `-like`。

## 2. 六患者配对单细胞证据

对 GSE236696 的 6 对黏液型肿瘤/癌旁单细胞原始矩阵重新做患者级 pseudobulk。
统计单位始终是患者，不以细胞数扩充样本量。

在冻结的 broad epithelial gate 中：

| 转录程序 | 肿瘤-癌旁均值 | 同向患者 | 精确符号翻转 p |
|---|---:|---:|---:|
| 多胺乙酰化/分解（SAT1/PAOX/SMOX） | +0.690 | 6/6 | 0.03125 |
| 多胺合成 | +0.504 | 6/6 | 0.03125 |
| 酸性/乳酸应答 | +1.208 | 6/6 | 0.03125 |
| 中性粒细胞募集趋化程序 | +2.378 | 6/6 | 0.03125 |
| 嘌呤合成/回收 | +0.371 | 6/6 | 0.03125 |

患者间“多胺乙酰化/分解变化”与“酸性应答变化”Spearman `rho=0.886`，6! 精确置换
`p=0.0333`。它们与趋化变化不相关（`rho=-0.143`, `p=0.803`），所以不能把三个轴
写成已经闭合的连续因果链。

## 3. 上皮门控与随机基因集反审计

为防止 broad gate 的优先分配规则制造信号，冻结并重跑三种门控：

1. `broad_frozen`：此前冻结的 PTPRC 阴性、至少两个上皮 marker 的优先门；
2. `competitive`：上皮分数必须超过所有其他谱系至少 0.15；
3. `canonical_strict`：至少三个上皮 marker、至少两个 canonical anchor，且相对其他
   谱系至少领先 0.05。

严格门在部分样本只保留 12–13 个细胞，因此只能做敏感性分析，不能替代主门。

| 程序 | broad | competitive | canonical strict | 表达匹配随机集结论 |
|---|---:|---:|---:|---|
| 多胺乙酰化/分解 | +0.690, 6/6 | +1.174, 5/6 | +1.173, 5/6 | competitive/strict 幅度经验 p≈0.02；broad p≈0.06 |
| 多胺合成 | +0.504, 6/6 | +0.371, 4/6 | +0.509, 4/6 | 只在 broad 幅度上优于匹配空集；跨门不稳 |
| 酸性/乳酸应答 | +1.208, 6/6 | +1.031, 6/6 | +1.165, 6/6 | 三门均明显优于匹配空集 |
| 趋化程序 | +2.378, 6/6 | +2.296, 6/6 | +2.123, 5/6 | 三门幅度均优于匹配空集 |
| 嘌呤 | +0.371, 6/6 | +0.556, 5/6 | +0.562, 5/6 | 本次统一均值口径为边缘信号；旧 10 基因中位数口径仍较强 |

所以最稳转录背景是酸性应答和趋化；多胺乙酰化/分解有跨门方向与幅度证据，但
样本只有 6 人且严格门细胞数低，应写成支持性假设而非已确认机制。多胺合成不进入
核心机制表述。

## 4. 空间转录组复核及否定性结果

GSE236697 为源研究另一个单病例肿瘤/癌旁 Visium 数据。重分析获得肿瘤 3,481 个
QC spot、癌旁 1,725 个 spot。该样本的肿瘤测序深度远低于癌旁，因此 tumour-normal
差异只作描述，不做 spot 级显著性：

- 肿瘤中酸性/乳酸应答和趋化程序描述性升高；
- 长链 FAO 程序描述性降低；
- 多胺乙酰化/分解的 raw mean score 较高，但 rank score 差异极小；
- 控制 library size 和 detected genes 后，趋化程序与源论文单核/巨噬细胞 compartment
  marker 的 partial Spearman 为 `0.273`；
- 多胺乙酰化/分解与酸性应答、趋化程序的深度校正后相关分别只有 `0.010` 和 `0.040`。

因此空间数据支持“酸性/趋化生态存在”和“趋化偏向髓系区域”的描述，却没有支持
SAT1/乙酰化多胺与这些区域共定位。它是限制机制链强度的有效阴性结果。

## 5. 与外部机制论文的关系

既往研究已经证明在实验模型中：酸性 pH 可通过 SAT1 诱导 N1-acetylspermidine，
SAT1 抑制会降低中性粒细胞募集、血管生成和肿瘤生长。这个结果为我们选择
SAT1/酸性/趋化轴提供先验，但不能迁移为本队列的因果结论。我们本地候选更接近
N1,N8-diacetylspermidine-like，也不是该论文被标准品确认的 N1-acetylspermidine。

高水平 CRC 代谢机制论文通常还包含 targeted quantitative LC-MS/MS、标准品、空间
定位、候选酶扰动、同位素示踪和 rescue。我们目前没有这些湿实验条件，所以论文定位
必须是“DreaMS 驱动的严格重分析与机制假设优选”，不是治疗靶点机制终证。

## 6. 当前可写与不可写

可以写：

> DreaMS-enabled reanalysis prioritized a recurrent acetylated-polyamine ion family that was strongly
> elevated in the Rmu discovery subgroup. Patient-paired mucinous scRNA-seq independently supported
> epithelial polyamine-acetylation and acidic-response programs, whereas single-patient spatial data did
> not establish their colocalization. The result therefore motivates, but does not prove, an acidic
> acetylated-polyamine microenvironment in mucinous colorectal cancer.

不可写：

- feature 1717 已被确认为 N1,N8-diacetylspermidine；
- SAT1 导致了 feature 1717 升高；
- 乙酰化多胺造成了中性粒细胞募集；
- 已证明 Rmu 多胺通量重编程。

## 7. 信息增益最高的下一步

1. N1,N8-diacetylspermidine 标准品：同法 RT、多个碰撞能 MS2、样本 spike-in；
2. 同时测 N1-/N8-acetylspermidine，排除 230→100 碎片的结构/位置非特异性；
3. 若只能做一次追加实验，优先标准品身份终证，而不是继续扩展转录相关性；
4. 若完全不能做湿实验，将该轴放在算法应用论文的“high-confidence hypothesis”层，
   同时把空间不共定位和 MTBLS8090 未直接复现写入主文限制。

## 8. 工件

- `data/mtbls13729/candidate_evidence_ledger_v1/`
- `data/mtbls13729/polyamine_crosschrom_audit_v1/`
- `data/external/GSE236696/polyamine_mechanism_v1/`
- `data/external/GSE236696/polyamine_gate_specificity_v1/`
- `data/external/GSE236697/spatial_metabolic_axes_v1/`
- `tasks/analyze_gse236696_polyamine_mechanism.py`
- `tasks/audit_gse236696_polyamine_gate_specificity.py`
- `tasks/analyze_gse236697_spatial_metabolic_axes.py`

