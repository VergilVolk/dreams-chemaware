# MTBLS13729 外部 CRC sialyltransferase-score 与黏液型关联审计（2026-08-31）

## 核心结果

2026 年 *Biology* 论文整合 TCGA、Sidra-LUMC 和 CPTAC-2 共988例 CRC 转录组。其公开补充
Table S3 中，980例有 histology 信息：

| Histology | Sialyl-Low | Sialyl-High | Total | High fraction |
|---|---:|---:|---:|---:|
| Mucinous adenocarcinoma | 69 | 85 | 154 | 55.2% |
| Non-mucinous adenocarcinoma | 588 | 238 | 826 | 28.8% |

由原始2×2计数重算，mucinous进入Sialyl-High组的odds ratio为`3.043`，95% CI
`[2.142, 4.325]`，Fisher exact p=`6.58e-10`；来源表报告χ²=`40.884`、FDR q `<0.001`。

## 这个 score 到底测了什么

补充 Table S1 显示，`Sialylome Activity score`只包含20个sialyltransferase genes：

- ST3GAL1–6；
- ST6GAL1–2；
- ST6GALNAC1–6；
- ST8SIA1–6。

因此它是**唾液酸转移酶转录复合分数**，不是：

- free Neu5Ac；
- CMP-Neu5Ac或donor pool；
- 实测O/N-glycan结构；
- glycan flux；
- 酶活性。

## 对本地主轴的价值

该结果独立于 MTBLS13729，支持“黏液型 CRC 中唾液酸相关糖基化程序值得优先研究”。它与本地
`hybrid mucin glycome`兼容：一个总sialyltransferase composite可以整体偏高，同时ST6GAL1/
alpha2-6分支下降、ST6GALNAC1等其他分支上升。

它不能代替本地分支级分析，因为把20个方向可能不同的基因合成一个分数，会隐藏
donor–carrier–core–linkage decoupling。它也不能复现feature703的患者内丰度。

## 独立性边界

- 相对 MTBLS13729：独立；
- 相对本地TCGA COAD/READ转录分析：不完全独立；
- 原因：外部论文三队列中包含TCGA；Sidra-LUMC和CPTAC-2是新增队列，但公开表没有按队列分别
  报告mucinous关联，无法计算真正的leave-TCGA-out效应。

因此论文中只能称 `external pooled transcriptomic context with partial TCGA overlap`，不能称
三次独立复制，也不能称独立代谢物验证。

## 可用于论文的表述

> In an external integrated transcriptomic analysis of 980 histology-annotated CRC cases,
> mucinous tumours were enriched in a high sialyltransferase-expression group (55.2% versus
> 28.8%; reconstructed OR 3.04, 95% CI 2.14–4.33). Because the analysis partially overlaps
> TCGA and does not measure Neu5Ac or glycan structures, it was retained as contextual support
> rather than independent metabolite replication.

## 工件

- 审计脚本：`tasks/audit_external_crc_sialylome_mucinous.py`
- 冻结补充与报告：`data/external/CRC_sialylome_mucinous_Biology2026_20260831/`
- 来源 DOI：`10.3390/biology15090705`

