# MTBLS13729 Neu5Ac–黏蛋白混合糖组审计（2026-08-31）

## 一、核心裁决

当前三层证据不支持“黏液型结直肠癌发生全局 hypersialylation”，也不支持“正常 core-3
程序被简单替换为癌相关 core-2 程序”。更符合数据、且可被后续实验证伪的模型是：

> **Rmu 中游离/可提取 Neu5Ac pool 增大；黏液型肿瘤相对常规型保留更强的分泌型黏蛋白与
> core-3/Sda 黏膜谱系程序，同时获得癌相关 core-2/sialyl-Lewis X/A 糖链；不同
> α2-6、α2-3 连接和载体蛋白并不同步，形成 hybrid mucin glycome。**

这个模型明确区分三种比较：

1. `Rmu tumour - matched normal`：回答同患者 pool-size 改变；
2. `mucinous tumour - conventional tumour`：回答组织学亚型相对保留/富集；
3. 外部 `MUC tumour - matched normal` 与 MUC 在全部肿瘤中的排序：同时回答绝对肿瘤改变与
   肿瘤间相对位置。

三种比较不得混成同一个“升高/降低”结论。

## 二、本地代谢物入口：free Neu5Ac

- MTBLS13729 positive-RP feature 703 通过 source Level-1 Neu5Ac 的同队列正交身份桥；
- 锁定 detection-masked targeted-EIC 在 10 对 Rmu/RN 中 `10/10` 为正，平均
  `+1.935 log2`（约 `3.82-fold`）；
- 跨面板 `log2(EIC+1)` 口径同为 `10/10`，平均 `+1.975 log2`；
- discovery peak-picker matrix 因 P24 缺失为 `9/9`、`+1.881 log2`，只作缺失敏感性；
- Neu5Ac 是五个冻结模块中唯一在 raw 与 PQN 两种归一化下通过 subtype-sensitivity BH 门的
  模块（Rmu-vs-Rtu 差值约 `+2.14–2.21 log2`）；
- 它没有通过全部 13,155 个 MS1 target 的 full-space FDR10。因此它是预定义候选面板中的
  亚型锚，不是全空间确认 biomarker。

## 三、TCGA 结构锚定的糖链分支审计

### 3.1 协议

- 42 个黏液型、329 个常规型 TCGA COAD/READ primary tumours；
- HC3 OLS 调整 age、side、stage、sex；
- 组成敏感性模型再加入六类 broad-lineage expression proxies；每个 outcome 从 proxy 中删除
  与自身重叠的基因，避免机械过调；
- MSI 完整病例模型另行加入 MSI；`n=364`；
- 分支基因来自独立 CRC PGC-LC-MS/MS O-glycomics 的 biosynthetic interpretation，结果产生前
  固定；所有 composite 和单基因共同做 BH 校正；
- 工件：`data/external/TCGA_COADREAD_Xena_20260830/glycan_branching_v2/`；脚本：
  `tasks/audit_tcga_mucinous_glycan_branching_v1.py`。

### 3.2 结果

| 分支/节点 | lineage-adjusted beta | BH q | +MSI beta | +MSI BH q | 裁决 |
|---|---:|---:|---:|---:|---|
| Neu5Ac donor supply/transport | +0.480 | 3.30e-8 | +0.448 | 5.36e-7 | 稳健相对富集 |
| secretory mucin program | +0.922 | 5.34e-11 | +0.859 | 4.79e-10 | 最强亚型程序 |
| normal-mucosal core-3/Sda | +0.879 | 1.76e-8 | +0.853 | 1.32e-7 | 黏液谱系相对保留 |
| core-2/sLeX biosynthesis | -0.009 | 0.915 | +0.011 | 0.895 | 转录 composite 不支持整体升高 |
| α2-3 O-glycan sialylation | -0.439 | 0.0093 | -0.434 | 0.0141 | 相对降低，和笼统 sialyl-high 冲突 |
| ST6GAL1 | -0.742 | 8.50e-5 | -0.719 | 2.23e-4 | α2-6 N-glycan route 相对降低 |
| ST6GALNAC1 | +0.788 | 7.99e-8 | +0.761 | 5.36e-7 | 特定 O-GalNAc/sialyl-Tn route 上升 |
| ST6GALNAC3 | -0.127 | 0.419 | -0.136 | 0.424 | 无确认差异 |
| GCNT3 | +0.417 | 0.0096 | +0.462 | 0.00217 | core-3/4 branching context 上升 |

关键单基因中，`GNE`、`NANS`、`SLC35A1`、`MUC2`、`SPDEF`、`B3GNT6`、`B4GALNT2`
均为显著正向；`ST3GAL2`、`ST6GAL1` 和 `B4GALT3` 为显著负向。core-2 composite 的近零
并非“没有生物学”，而是其组成基因方向相反，不能用一个平均分代表糖链产物。

### 3.3 该结果为什么不是外部 O-glycomics 的反证

外部两个 MUC 肿瘤的 core-3 丰度在 11 个 AC/MUC 肿瘤中排第 2/第 1，远高于 9 个 AC 的
中位数 `0.35`；但相对各自正常黏膜仍下降 `-34.01/-34.64`。因此：

- “MUC 相对 AC 保留 core-3”与 TCGA 的 `B3GNT6/B4GALNT2/GCNT3` 相对富集一致；
- “癌变相对正常丢失 core-3”与外部配对糖组一致；
- 两个命题并不矛盾，矛盾来自把 tumour-normal 与 subtype contrast 混为一谈。

## 四、独立 O-glycomics 给出的结构层

两个外部 MUC 病例同时表现为：

- core-2 为全部 AC/MUC 肿瘤第 1/第 2；配对变化 `+46.46/+40.13`；
- sialyl-Lewis X/A 为第 2/第 1；配对变化 `+7.64/+4.49`；
- α2-6 总特征为倒数第 2/第 1；配对变化 `-50.08/-70.43`；
- core-3 为第 2/第 1，但配对变化 `-34.01/-34.64`；
- core-2 上 α2-3 结构配对增加 `+27.73/+44.29`，但 total α2-3 并不高于 AC。

这说明结构终产物同时包含“癌相关 TACA 获得”和“黏膜谱系残余保留”。由于外部 MUC
只有 `n=2`，它是 independent structural support，不是人群级 subtype replication。

## 五、相对最新文献的真正增量

### 5.1 2026 transcriptomic CRC sialylome

一项 988-case TCGA/Sidra-LUMC/CPTAC-2 的 ssGSEA 研究报告高 sialylation signature 与黏液型、
MSI、BRAF、晚期和 immune-excluded transcriptional state 相关。该研究没有直接测 glycan
structure、linkage 或 free Neu5Ac，并且其 TCGA 数据与本项目的 TCGA context 部分重叠。

本项目不能把该论文当“独立 glycomics 复现”；真正增量是把笼统 `sialyl-high` 拆成互相
分离的 donor、mucin、core、linkage 和 carrier hypotheses，并以直接 O-glycomics 证明这些
层并不同步。

### 5.2 2025 ST6GAL1–PD-L1 CRC mechanism

该机制研究通过 ST6GAL1 knockdown、SNA lectin pulldown、PD-L1 immunoprecipitation、蛋白
稳定性/泛素化、细胞功能和小鼠治疗实验，证明特定 α2-6 N-glycan route 可具有免疫功能。
本项目的 mucinous-relative `ST6GAL1` 转录为负，外部 MUC 总 α2-6 特征也最低，因此不能把
free Neu5Ac 或 secretory-mucin enrichment 直接映射成 ST6GAL1–PD-L1 机制。

### 5.3 2026 MUC2 spatial glycopeptide study

StcE on-slide mucinase + MALDI-MSI + LC-MS 在来自 2 位患者的 3 块黏液型 CRC 中显示 MUC2
glycoforms 可在同一肿瘤内形成不同空间区域。更关键的是，肿瘤 MUC2 O-glycopeptide 整体以
低唾液酸化/非唾液酸化结构为主，mono- 和 di-O-acetylated Neu5Ac-containing glycans 低于健康
结肠。该结果与本地 free Neu5Ac pool 增加并不矛盾，而是直接支持本项目的关键边界：bulk free
Neu5Ac 和 bulk transferase RNA 均不能指定糖链载体、连接或空间去向。数据规模和缺乏非黏液型
人群对照使其不能承担 subtype 复现。

## 六、最小可发表结论与决定性验证

### 当前可以写

> Paired tissue metabolomics, lineage-adjusted tumour transcriptomics and an independent structural
> O-glycomics cohort converge on a hybrid mucin glycome in mucinous CRC: an expanded free Neu5Ac
> pool and secretory-mucin/donor programme coexist with relative retention of mucosal core-3 features
> and acquisition of core-2/sialyl-Lewis tumour-associated glycans. The discordance across donor,
> transferase and structural layers argues against global hypersialylation.

### 投稿前最有价值的最小验证

1. **身份/定量门：** 同法 Neu5Ac standard RT+MS2、sample spike-in，最好加 isotope internal
   standard；这确认本地入口，但不回答糖链去向。
2. **结构门：** 在同一批样本或可得替代组织上测 linkage-aware O-glycomics；最低限度同时量化
   core-2/sLeX、core-3/Sda、sialyl-Tn、α2-3 和 α2-6。
3. **载体/空间门：** MUC2 glycopeptide 或 StcE/LC-MS readout；若无法实现，可用 SNA/MAL-II
   加 MUC2/epithelial marker 的组织学共定位作较低等级替代。
4. **来源门：** 配对测 free Neu5Ac、ManNAc、CMP-Neu5Ac；只有加入 isotope incorporation
   才能声称 donor flux。

## 七、禁止表述

- `global hypersialylation`；
- `ST6GAL1/PD-L1 pathway is activated in Rmu`；
- `free Neu5Ac proves increased glycan incorporation`；
- `core-3 is increased during tumour transformation`；
- `independent free-Neu5Ac replication`；
- `glycosyltransferase causality`、`flux reprogramming` 或 `therapeutic target established`。

## 八、可复核工件

- `data/mtbls13729/neu5ac_glycan_publication_figure_v1/`
- `data/mtbls13729/neu5ac_glycan_publication_figure_v2_final/`
- `data/external/CRC_Oglycomics_PMC9254241_20260830/mucinous_structural_audit_values.csv`
- `data/external/TCGA_COADREAD_Xena_20260830/glycan_branching_v2/`
- `tasks/audit_tcga_mucinous_glycan_branching_v1.py`
- `docs/MTBLS13729_EXTERNAL_OGLYCOMICS_MUCINOUS_AUDIT_20260831.md`
