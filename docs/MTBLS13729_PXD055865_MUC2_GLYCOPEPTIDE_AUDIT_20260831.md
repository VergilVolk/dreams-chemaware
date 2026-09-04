# PXD055865 MUC2 糖肽外部证据审计（2026-08-31）

## 一、审计目的与裁决

本审计检验公开数据 PXD055865 是否能够补足 MTBLS13729 主线中缺失的
“游离 Neu5Ac 最终进入什么糖链和载体”这一层。

结论是 **PASS_WITH_LIMITATION**：该数据提供了人工复核的 MUC2 糖肽、
O-acetyl-Neu5Ac/O-acetyl-GalNAc 候选以及空间异质性证据，能够作为
carrier/destination 层的外部结构背景；但它不测游离 Neu5Ac 丰度，且只有两位
独立黏液型结直肠癌患者，不能称为 MTBLS13729 游离 Neu5Ac 的独立复制。

## 二、真实研究设计

- 论文：Lowery 等，*Glycosite mapping and in situ mass spectrometry imaging of
  MUC2 glycopeptides via on-slide mucinase digestion*，Nature Communications，2026。
- 数据集：ProteomeXchange/PRIDE PXD055865。
- 肿瘤标本：Colon1a、Colon1b、Colon2 共三块黏液型结直肠癌组织。
- 独立患者数：**2**。Colon1a 与 Colon1b 来自同一位患者，不能作为两次独立重复。
- 对照：1 份独立 healthy colon。
- 本地使用文件：Supplementary Data 2（人工复核糖肽列表）和 source spectra workbook。

因此，任何“n=3 patients”“三例独立复制”或正式 tumour-versus-normal 群体统计均不成立。

## 三、可复核数据结果

分析脚本：`tasks/audit_external_pxd055865_muc2_glycoforms.py`。

| 标本 | 患者 | 去重 MUC2 糖肽 | 含 Neu5Ac | O-acetyl-Neu5Ac | O-acetyl-GalNAc |
|---|---|---:|---:|---:|---:|
| Colon1a | Patient1 | 439 | 4 | 3 | 1 |
| Colon1b | Patient1 | 390 | 5 | 5 | 4 |
| Colon2 | Patient2 | 451 | 1 | 0 | 0 |
| HealthyColon | HealthyDonor | 21 | 1 | 1 | 0 |

去重单位为 `Sequence + Protein Name`，折叠了重复电荷态和 Colon2 的两个肿瘤区域。
source-spectrum workbook 含两张专门支持 di-O-acetyl-Neu5Ac 的 HCD/ETD 图表，
并含 O-acetyl-GalNAc 的 source sheet。

上述数量只说明“在不同深度的发现实验中鉴定到了什么”，**不是无偏丰度**。肿瘤标本数量、
切区和鉴定深度均远高于 healthy colon，不能用 661 vs 21 等鉴定数比较生物丰度。

## 四、论文作者层面的可用结论

原论文报告：

1. 黏液型 CRC 中 MUC2 glycoform 具有明显空间分层和肿瘤内异质性；
2. healthy colon 中 extended glycans 上可见高丰度 di-/tri-O-acetylated Neu5Ac；
3. Colon1a/1b/2 中 mono-/di-O-acetylated Neu5Ac 整体较低；
4. 肿瘤区域出现 putative O-acetylated GalNAc 证据；
5. 作者同时明确警告：同批样本仍需要深入 glycomics；sialic acid 可能发生 in-source
   fragmentation；putative O-acetylated GalNAc 仍需排除技术伪影。

本地补充表审计用于确认这些证据确实存在于公开结构化数据中，而不把鉴定数重新包装成丰度。

## 五、与 MTBLS13729 的整合

MTBLS13729 中已经观察到：

- Rmu 10/10 患者游离 Neu5Ac 增加；
- 游离 Neu5Ac 增幅显著超过 CMP-Neu5Ac 和 UDP-GlcNAc；
- 两个表型盲 mono-O-acetyl-Neu5Ac-like 精确质量峰不随游离 Neu5Ac 增加；
- TCGA donor/carrier/core/linkage 分支不支持简单“全局高唾液酸化”。

PXD055865 增加的不是“又一例 Neu5Ac 上升”，而是 carrier-resolved 边界：

> 游离 Neu5Ac pool 的扩张可以与 MUC2 上 O-acetylated sialic-acid destination 的降低、
> 低唾液酸化糖肽占优和替代性 O-acetyl-GalNAc 候选同时出现。

这与 **donor–carrier–core–linkage decoupling** 方向相容，也进一步否定
“free-pool 上升必然等于所有黏蛋白末端统一高唾液酸化”的捷径。

## 六、允许和禁止的论文表述

允许：

- “An independent carrier-resolved mucinous CRC dataset provides structural context consistent
  with destination-level remodeling of MUC2 glycans.”
- “The external data are directionally compatible with, but do not replicate, free-pool/carrier
  decoupling.”
- “MUC2 O-glycoforms show strong inter- and intra-tumour heterogeneity.”

禁止：

- “PXD055865 independently replicated free Neu5Ac elevation.”
- “Three independent mucinous patients confirmed the mechanism.”
- “Tumour O-acetyl-Neu5Ac abundance was estimated from glycopeptide identification counts.”
- “O-acetyl-GalNAc is definitively established.”
- “Free Neu5Ac was proven to flow into or away from MUC2.”

## 七、仍缺的决定性验证

1. MTBLS13729 同法 Neu5Ac authentic standard：RT、MS2、spike-in；
2. 同一样本或替代组织的 linkage-aware O-glycan / MUC2 glycopeptide readout；
3. 4-/7-/8-/9-O-acetyl-Neu5Ac 的标准或 ion-mobility/CCS 区分；
4. 若要声称通量或因果，必须增加同位素示踪、节点扰动和 rescue。

## 八、工件与来源

- 审计结果：`data/external/PXD055865_2026_MUC2/audit_v1/report.json`
- 标本汇总：`data/external/PXD055865_2026_MUC2/audit_v1/specimen_summary.csv`
- 去重存在矩阵：`data/external/PXD055865_2026_MUC2/audit_v1/muc2_glycopeptide_presence.csv`
- 论文：https://www.nature.com/articles/s41467-026-72853-3
- 数据：https://proteomecentral.proteomexchange.org/cgi/GetDataset?ID=PXD055865

