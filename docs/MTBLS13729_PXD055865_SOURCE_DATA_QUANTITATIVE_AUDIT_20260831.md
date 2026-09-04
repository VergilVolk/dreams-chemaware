# PXD055865 补充文件与 Source Data 定量边界审计（2026-08-31）

## 结论先行

PXD055865 的公开补充材料比早期只审计 Supplementary Data 2/3 时更有价值，但仍不能被包装成独立患者队列复现。

1. Nature Communications 页面实际提供 8 个补充文件，现已全部下载并逐类盘点；
2. MOESM8 解压后只有 258 张 MALDI PNG，覆盖主图 1/2 与 8 个补充图目录，没有 CSV、XLSX、TXT 或其他数值矩阵；
3. MOESM1 的补充图 20、22–25 在图内给出了 Neu5Ac、AcNeu5Ac、Ac2Neu5Ac、Ac3Neu5Ac 的 `normalized level`，可以做描述性比值核对；
4. 健康结肠的 AcNeu5Ac/Neu5Ac、Ac2Neu5Ac/Neu5Ac、Ac3Neu5Ac/Neu5Ac 图示比值分别为 `11.91/44.33/182.17`；4 个肿瘤区域的前两项范围仅为 `0.0094–1.0000` 和 `0.0042–2.2798`；
5. 但肿瘤来自 2 位独立患者（Colon1a 与 Colon1b 同一患者），健康结肠只有 1 位供者，`normalized level` 是图示强度标尺，不是校准浓度或统一积分面积，因此不能做群体统计，也不能称独立丰度复现。

最合理的作用是加强 **pool–carrier–modification decoupling**：MTBLS13729 的游离 Neu5Ac 池增加，并不要求黏蛋白载体上的 O-acetyl-sialic-acid destination 同步增加；PXD055865 反而显示健康结肠 O-acetyl 指纹很强，而两个肿瘤患者之间高度异质。

## 完整补充材料盘点

已冻结文件：

- MOESM1：47 页 Supplementary Figures/Tables/Text；
- MOESM2：1 页 reporting summary；
- MOESM3：蛋白/肽层 Supplementary Data；
- MOESM4：糖肽鉴定表；
- MOESM5–6：编辑/同行评议相关 PDF；
- MOESM7：图 4 与补图 26/27/30/31 的谱峰源数据；
- MOESM8：97 MB Source Data ZIP。

MOESM8 中 258 张 PNG 的目录分布为：Figure 1（24）、Figure 2（15）、Supplementary Figure 7（9）、8（47）、11（10）、12（31）、13（12）、14（60）、15（17）、32（33）。没有 Supplementary Figures 20–25 的数值源表。

因此，MOESM8 解决的是图像来源可追溯问题，不解决数值重算、跨样本归一化或不确定度估计。

## 补充图可读出的定量层

以下数字均为 MOESM1 图内 `NL` 标签的人工转录，并由渲染后的原页复核：

| 区域 | Neu5Ac | AcNeu5Ac | Ac2Neu5Ac | Ac1/Neu5Ac | Ac2/Neu5Ac |
|---|---:|---:|---:|---:|---:|
| Colon1a tumour（P1） | 1.42e6 | 1.42e6 | 4.05e5 | 1.0000 | 0.2852 |
| Colon1b tumour（P1） | 5.79e5 | 5.79e5 | 1.32e6 | 1.0000 | 2.2798 |
| Colon2 tumour 1（P2） | 3.38e6 | 7.14e4 | 1.80e4 | 0.0211 | 0.0053 |
| Colon2 tumour 2（P2） | 2.58e6 | 2.43e4 | 1.08e4 | 0.0094 | 0.0042 |
| Healthy colon（1 donor） | 1.57e5 | 1.87e6 | 6.96e6 | 11.9108 | 44.3312 |

健康结肠还有 Ac3Neu5Ac `2.86e7`，对应 Ac3Neu5Ac/Neu5Ac `182.17`。

肿瘤邻近组织的图示值同样已结构化保存，但不把 tumour/adjacent 的 `NL` 比值写成 fold change，因为图没有证明这些 display levels 经过可跨运行比较的统一归一化。

## 新的生物学解释

这批外部证据不支持“mucinous CRC 全局高唾液酸化”或“游离 Neu5Ac 升高必然流向 MUC2 O-acetyl-sialylation”。相反，它支持一个更精确的分层模型：

1. **free pool**：MTBLS13729 中 Rmu 的游离 Neu5Ac 同患者升高；
2. **activated donor**：CMP-Neu5Ac 未同步显著上升；
3. **carrier**：PXD055865 确认 MUC2 是可定位载体，但肿瘤糖肽以 Tn/T 和低比例 sialylated glycoforms 为主；
4. **modification destination**：健康结肠存在很强的多 O-acetyl Neu5Ac 指纹，肿瘤患者之间的 O-acetyl/Neu5Ac 比例高度异质；
5. **linkage/core**：独立 O-glycomics 与 TCGA 支持 core/linkage 重塑，而不是一个统一“更多 sialic acid”轴。

因此，论文创新点应继续冻结为 **donor–carrier–core–linkage decoupling**，而不是首次发现 Neu5Ac 或建立通量因果。

## 仍可继续做的公开数据复算

PRIDE API 显示，构成补充图 20–25 的 Thermo RAW 文件确实公开，关键文件包括：

- Colon1a R1/R2 `StcEx2`；
- Colon1b R1/R2 `StcEx2`；
- Colon2 Tumour1/Tumour2/Adjacent1/Adjacent2；
- HealthyColon StcE。

总量约 3–4 GB。下一步可用 ThermoRawFileParser 提取同一组 fingerprint-ion XIC，并冻结统一 ppm、RT、峰面积和运行内归一化协议。这能把“图示 NL”升级为可复核的描述性积分，但仍只有 2 位肿瘤患者和 1 位健康供者，不能解决独立人群统计功效。

## 可复核工件

- 审计脚本：`tasks/audit_external_pxd055865_source_data.py`
- 结构化报告：`data/external/PXD055865_2026_MUC2/source_data_audit_v1/report.json`
- 258 张图像清单：`source_image_inventory.csv`
- 图内 NL 转录：`sialic_fingerprint_normalized_levels.csv`
- O-acetyl/Neu5Ac 比值：`sialic_fingerprint_ratios.csv`

## 绝对禁止的表述

- “PXD055865 独立复制了 Rmu 游离 Neu5Ac 丰度增加”；
- “健康与肿瘤差异经过统计显著性检验”；
- “normalized level 等于校准浓度或统一积分峰面积”；
- “Colon1a 与 Colon1b 是两位患者”；
- “已确定 4/7/8/9-O-acetyl-Neu5Ac 位置异构体”。

