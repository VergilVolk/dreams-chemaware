# MTBLS13729 外部 CRC N-glycomics 补充表审计（2026-08-30）

## 目的

判断 2024 年 CRC MALDI-MSI/N-glycomics 研究的公开补充材料能否作为 feature 703 Neu5Ac 的独立患者级复制。

## 输入

- 论文：*N-glycan signatures and spatial glycomic heterogeneity in colorectal cancer*（PMC10808565）；
- Europe PMC supplementary ZIP：`data/external/CRC_Nglycomics_PMC10808565_20260830/supplementaryFiles.zip`；
- 有效工作簿：`data/external/CRC_Nglycomics_PMC10808565_20260830/supplementary/Table1.XLSX`；
- 根目录下原名为 `Table1.XLSX` 的 20 KB 文件实际为 Europe PMC HTML 落地页，已删除，避免以后被误读为工作簿。

## 工作簿结构与可复算性

有效工作簿只有一个 sheet、一个 `A1:I53` 表格。它是研究队列中检出的 N-glycan 质量与组成目录，包含：

- 101 条 m/z–composition 记录；
- 19 条名称中含 `NeuAc` 的组成，其中 `NeuAc1` 14 条、`NeuAc2` 5 条；
- group 字段分布为：2-branch 32、3-branch 10、4-branch 28、Bisect 13、HYBRID 4、MAN 7、PAUCI 7。

该表没有患者 ID、组织学亚型、tumour/normal 标签或任何强度列。因此无法从公开补充表重算：

1. mucinous vs non-mucinous 的患者级 N-glycan 差异；
2. free Neu5Ac pool 与具体 sialylated N-glycan 的患者内相关；
3. feature 703 的丰度效应或 Rmu-vs-Rtu interaction；
4. 任何 glycan linkage、细胞来源或代谢通量。

## 对当前论文的正确作用

这份补充表确认该外部研究确实覆盖多种含 NeuAc 的 N-glycan composition，但只能作为“CRC N-glycome 中存在多种 sialylated destination”的结构背景。它不能作为 feature 703 的独立丰度复制，也不能证明本地 free Neu5Ac 被用于某一种 glycan linkage。

因此在主文中的最高允许表述为：

> Independent CRC glycomics provides structural context for multiple sialylated glycan destinations, while the public supplement lacks patient-level intensities required to replicate the mucinous-relative free-Neu5Ac signal.

## 决策

- 保留该研究作为 linkage-aware 验证必要性的依据；
- 不把19条 NeuAc-containing compositions计为19次独立验证；
- 独立复制硬门仍为未通过；下一步仍需患者级强度矩阵或新的靶向/糖组验证。
