# MTBLS13729 外部 CRC O-糖组学黏液型结构审计

## 审计对象与口径

- 外部研究：12 对原发 CRC/匹配正常结肠黏膜的 PGC-LC-MS/MS O-glycomics；另有 6 例转移灶。
- 原始补充表：`data/external/CRC_Oglycomics_PMC9254241_20260830/supplementary_tables.xlsx`。
- 文件 SHA256：`a0937f8f8b74365d47da58d569731fd53606e68ce767a01790fb119c544560d3`。
- 亚型定义严格以 Supplementary Table S2 为准：T2、T3 为 MUC；T1、T4、T6–T12 为 AC；T5 为神经内分泌癌并排除于 AC/MUC 描述性比较。
- 只报告两个 MUC 病例相对 9 个 AC 原发肿瘤的数值位置，以及 T2-C2、T3-C3 的配对方向；`n=2` 不做亚型显著性检验。

## 必须披露的元数据冲突

Supplementary Table S8 将 T6 的 `Type_2` 错写为 MUC，但 Table S2 和 Table S7 均将 T6 标为 AC；正文又写成“8 AC + 1 NEC + 2 MUC”，合计只有 11 例，而 Table S2 实际列出 9 AC + 1 NEC + 2 MUC 共 12 例。本审计使用病例级信息最完整的 Table S2，并保留该冲突作为外部数据质量限制。

## 结构特征结果

| O-糖链结构特征 | T2 | T3 | 9个AC中位数（范围） | MUC在11个AC/MUC肿瘤中的位置 | T2-C2 / T3-C3 |
|---|---:|---:|---:|---|---:|
| core 2 | 61.41 | 44.22 | 26.41（15.25–40.89） | 第1 / 第2高 | +46.46 / +40.13 |
| sialyl Lewis X/A | 7.64 | 20.05 | 0（0–5.06） | 第2 / 第1高 | +7.64 / +4.49 |
| core 2 + sialyl Lewis X/A | 4.30 | 8.07 | 0（0–5.06） | 第3 / 第1高 | +4.30 / +8.07 |
| α2-6 sialylation | 23.24 | 15.00 | 49.70（36.00–62.58） | 倒数第2 / 倒数第1 | −50.08 / −70.43 |
| core 2 + α2-3 sialylation | 33.51 | 50.01 | 40.04（9.21–59.07） | 第7 / 第2高 | +27.73 / +44.29 |
| 总 α2-3 sialylation | 46.51 | 89.05 | 103.89（56.53–117.36） | 第11 / 第8高 | +3.45 / +26.18 |
| core 3 | 6.71 | 8.14 | 0.35（0–1.31） | 第2 / 第1高 | −34.01 / −34.64 |

关键结果不是笼统的“全部唾液酸化增加”，而是：

1. 两个外部 MUC 病例的 core-2 和 sialyl-Lewis X/A 均位于原发腺癌/黏液癌队列最上端；
2. α2-6 唾液酸化在两个 MUC 病例中位于最下端，且相对各自正常组织大幅下降；
3. 总 α2-3 唾液酸化并不高于 AC，但两例相对自身正常组织均上升；core-2 上的 α2-3 唾液酸化也均配对上升；
4. MUC 肿瘤仍保留高于 AC 的 core-3 相关结构，但相对各自正常黏膜总体 core-3 大幅下降，说明这是“肿瘤间相对保留”而非“从正常到肿瘤绝对增加”。

## 七个高特异性 TACA 的描述性结果

- `H2N2F1S1f`：T2=10.99、T3=15.49，分别为 11 个 AC/MUC 原发肿瘤第2、第1高；9个AC中位数为0。
- `H2N2S1d`：T2=3.36、T3=6.48，分别第2、第1高；9个AC中位数为1.39。
- 其余五个 TACA 中，MUC 也多高于 AC 中位数，但并非均处于队列顶端，因此不能写成统一的 MUC 特异 TACA 程序。

## 对 MTBLS13729 主轴的意义

该外部队列不能复现 free Neu5Ac 的定量丰度，也不能证明 feature 703 的身份或因果来源；但它提供了独立样本、独立组织处理和独立 PGC-LC-MS/MS 技术的结构层正交支持。与 MTBLS13729 的 free Neu5Ac 上升、Neu5Ac 模块亚型交互和 TCGA 黏蛋白/唾液酸程序合并后，最符合数据的模型是：

> 黏液型 CRC 存在选择性的唾液酸-黏蛋白糖链重塑，表现为 core-2/sLeX/A 肿瘤相关糖链扩张、α2-6 架构丢失，以及部分 core-3 糖链在 MUC 中相对 AC 保留；这不是全局 hypersialylation，也不等同于已证明的 Neu5Ac flux 增加。

## 结论边界

- **可以写**：independent structural support；mucinous-relative selective sialyl/mucin-glycan remodeling；orthogonal convergence。
- **不能写**：独立 free-Neu5Ac 丰度复制；linkage 已由 MTBLS13729 直接测得；糖基转移酶因果；代谢通量；MUC 人群级确认。
- 两例 MUC 的结果应在正文作为外部结构验证/一致性分析，在摘要中使用 `supported by an independent glycomics cohort`，不能使用 `replicated` 或 `validated in an independent cohort`。
