# 黏液型 CRC 风险结构与代谢轴：GSE281917/281918 和 TCGA 反证式审计（2026-08-30）

## 目的

检验 MTBLS13729 中的修饰鸟苷/嘌呤、乙酰化多胺和长链酰基肉碱丰度程序，是否与独立黏液型 CRC 队列内部的风险表达结构一致。该分析只提供转录背景，不把 RNA 轴当作代谢物、通量或酶活真值。

## 数据与不可绕过的设计边界

- GSE281917：140 例 mucinous CRC，GPL24676/NovaSeq 6000；
- GSE281918：119 例 non-mucinous CRC，GPL16791/HiSeq 2500；
- 作者补充方法概述 CPM、分位数归一化、log2 和中位数中心化，但 GEO 明确显示 histology 与 platform 完全重合；
- 因此 MuC–NMuC 跨系列效应不可识别。ComBat 或在模型中同时加入 platform 都不能从完全共线设计中恢复 histology 效应；
- 主分析限定在 GSE281917 内部：冻结四个代谢轴，计算论文冻结的 23 基因 MuC23 风险分数；
- TCGA COAD/READ 的 42 例黏液型原发肿瘤用于定向复算，但 TCGA 已用于相关轴分析，所以不是全新盲测。

## GSE281917 主结果

四轴与 MuC23 风险分数不存在直接共享基因。对分期、年龄、性别做残差秩校正后：

| 轴 | partial-rank rho | 95% bootstrap CI | BH q | 留一病例范围 |
|---|---:|---:|---:|---:|
| modified-nucleoside processing | -0.310 | [-0.469, -0.142] | 4.46e-4 | -0.334 至 -0.296 |
| purine synthesis/salvage | -0.498 | [-0.633, -0.345] | 2.37e-9 | -0.527 至 -0.486 |
| carnitine/long-chain FAO | -0.012 | [-0.192, 0.168] | 0.892 | -0.030 至 0.011 |
| polyamine acetylation/catabolism | +0.262 | [0.089, 0.416] | 0.00267 | 0.245 至 0.285 |

线性尺度 HC3 模型没有任何轴通过四轴多重校正，说明关系主要是秩单调而不是稳定线性效应；不能把它写成独立预后系数。四轴均未显示独立分期关联。

## Bulk 组织组成敏感性

使用此前在 GSE236696 分析前冻结的 broad-lineage marker sets，对 epithelial、myeloid、B/plasma、T/NK、endothelial 和 fibroblast 分数做额外残差化。MuC23 与 fibroblast（rho=0.738）和 endothelial（rho=0.640）高度相关，说明风险签名本身带强基质成分。

| 轴 | 临床校正 rho | 临床+六类组成 rho | 95% CI | BH q | 裁决 |
|---|---:|---:|---:|---:|---|
| modified-nucleoside processing | -0.310 | -0.050 | [-0.229, 0.124] | 0.574 | 主要由组成解释 |
| purine synthesis/salvage | -0.498 | -0.254 | [-0.464, -0.086] | 0.0139 | GSE 内仍保留 |
| carnitine/long-chain FAO | -0.012 | -0.132 | [-0.338, 0.035] | 0.268 | 阴性 |
| polyamine acetylation/catabolism | +0.262 | +0.105 | [-0.079, 0.271] | 0.310 | 主要由组成解释 |

嘌呤轴删除任一基因后仍保持负向，rho 为 -0.319 至 -0.185，最弱 p=0.0343。单基因层面 ADK、PNP、ADA、APRT 为负，GDA 为正；因此这是可重复的多基因轴，但不是每个节点同向。

## TCGA 定向复算

在 42 例 TCGA 黏液型 CRC 中，用同一 MuC23 系数、同一代谢轴和同一 broad-lineage marker sets：

| 轴 | 临床校正 rho | 95% CI | 临床+组成 rho | 95% CI | 结论 |
|---|---:|---:|---:|---:|---|
| purine synthesis/salvage（预定 primary） | -0.508 | [-0.745, -0.198] | -0.144 | [-0.511, 0.251] | bulk 风险方向复现；组成独立性未复现 |
| modified-nucleoside processing | -0.575 | [-0.779, -0.252] | -0.185 | [-0.541, 0.233] | 同上，次级 |
| carnitine/long-chain FAO | -0.149 | [-0.506, 0.235] | +0.122 | [-0.312, 0.528] | 阴性 |
| polyamine acetylation/catabolism | +0.320 | [-0.027, 0.602] | +0.422 | [0.024, 0.685] | 次级正向；三项次级 BH q=0.054，需独立复现 |

TCGA 因样本只有 42 例且组成协变量较多，组成校正 CI 较宽。结果不能证明 GSE 信号是假，也不能声称组成独立复现；正确结论是：MuC23 高风险与“低嘌呤/修饰核苷处理、较高多胺乙酰化”bulk 表达状态跨队列方向一致，但其细胞来源和独立预后性未确定。

## 与 MTBLS13729 代谢物结果的关系

MTBLS13729 的 Rmu 发现亚组中修饰鸟苷样离子显著升高，而高 MuC23 风险状态中 purine synthesis/salvage 转录轴更低。两者不是直接矛盾，也不能自动拼成机制。至少存在三种竞争解释：

1. modified-nucleoside turnover/clearance bottleneck：降解/回收不平衡导致组织池积累；
2. 组织组成：基质、免疫和上皮比例同时影响 bulk RNA 与代谢物；
3. 不同队列/亚群上下文：Rmu-vs-RN 是 10 对配对丰度发现，MuC23 是跨患者风险表达分层。

没有 isotope tracing、标准品定量和细胞来源实验，不能选择其中任何一个作为既定机制。

## 可进入论文的最高强度表述

> In two mucinous CRC transcriptomic cohorts, the MuC23 risk program was accompanied by lower bulk purine/modified-nucleoside processing scores and a directionally higher polyamine-acetylation score. The purine association survived broad lineage adjustment in GSE281917 but not in the smaller TCGA mucinous subset, indicating a reproducible risk-associated bulk state without establishing cell-autonomous metabolic reprogramming.

禁止写：

- MuC23 风险由嘌呤通量下降驱动；
- 修饰鸟苷积累由 METTL1/回收障碍导致；
- SAT1 驱动黏液型复发；
- 这些 RNA 结果验证了 feature 1597/3019/1717/3222 的精确身份。

## 可复核工件

- `tasks/analyze_gse281917_mucinous_metabolic_axes.py`
- `data/external/GSE281917/mucinous_metabolic_axes_v1/report.json`
- `tasks/audit_gse281917_metabolic_axis_composition.py`
- `data/external/GSE281917/mucinous_axis_composition_audit_v1/report.json`
- `tasks/audit_tcga_mucinous_risk_axis_replication.py`
- `data/external/TCGA_COADREAD_Xena_20260830/mucinous_risk_axis_replication_v1/report.json`

