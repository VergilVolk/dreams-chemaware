# MTBLS13729 生物学应用：GSE236696 肿瘤上皮方向性复核

## 结论摘要

MTBLS13729 的本地代谢组结果现已获得一条患者配对、细胞分辨率的方向性证据链。经 2026-08-30 逆向审计后，最稳的生物学表述是：

1. **嘌呤合成/回收是最稳的上调转录方向。** 在 6 对肿瘤/癌旁上皮 pseudobulk 中，固定嘌呤轴 6/6 上升，平均肿瘤−正常差值 `+0.634`，两侧精确 sign-flip `p=0.03125`。该方向跨三种轴计分和留一基因保持稳定；全基因表达匹配的 20,000 个随机轴中，效应幅度经验 `p=0.0113`。但与 FAO 作为两个共同主终点做两侧 Holm 校正后为 `p=0.0625`，应称为强方向性证据，而非已通过的独立显著复现。
2. **长链 FAO 下降具有最强的基因集特异性，但患者一致性强度依赖计分定义。** 固定肉碱穿梭/长链 FAO 轴在原计分下 6/6 下降，平均差值 `-0.567`，两侧精确 `p=0.03125`；替代的逐基因配对计分为 5/6 下降，精确 `p=0.0625–0.09375`。全基因表达匹配随机轴的效应幅度经验 `p=0.00020`、6/6 一致性经验 `p=0.00510`；共同主终点 Holm 校正仍为 `0.0625`。
3. **修饰核苷加工轴降级为探索性支持。** 原计分为 5/6 上升、平均 `+0.261`、精确 `p=0.125`，且留一基因和替代计分明显不稳定；不能用它验证具体修饰鸟苷身份。

这些结果支持本地“嘌呤周转增强”和“长链酰基肉碱积累/FAO 利用受限”作为跨组学机制假设，但不能把方向一致写成黏液型特异性。TCGA COADREAD 的 32 对肿瘤/正常样本显示嘌呤轴普遍升高、FAO 轴普遍下降；42 例黏液型与 329 例常规型肿瘤在 364 例高覆盖 MSI 敏感性模型中，FAO 无差异（beta `+0.010`, p=`0.911`），嘌呤在黏液型中反而较低（beta `-0.232`, p=`0.0091`；五轴主模型 BH q=`0.070`）。因此当前更准确的解释是：两轴属于一般 CRC 肿瘤程序，本地 Rmu 队列可能具有更强静态丰度表现，但黏液型交互尚未证实。

## 数据与统计设计

- 数据集：GEO `GSE236696`，6 名黏液型结直肠腺癌患者的肿瘤和匹配癌旁单细胞转录组，共 12 个样本。
- 原始数据：36/36 个 GEO 原始文件均按远端公布字节数完成核对。
- 质量控制：沿用源研究阈值，每细胞检测基因数 `>200`、线粒体 UMI 比例 `<25%`。
- 统计单位：患者；细胞只用于构造每位患者、每个谱系的 pseudobulk，绝不把细胞数当作生物学重复。
- 主分析谱系门：至少两个上皮 marker 且 `PTPRC=0`，允许无法可靠归类的细胞弃权；12/12 样本均保留至少 50 个上皮细胞。
- 敏感性分析：至少三个上皮 marker，最低 30 个细胞/样本。
- 代谢轴：在查看 GSE236696 结果前固定；不依据本队列结果增删基因，也不构造事后最优总分。

## 主结果

| 固定轴 | 上皮平均配对差值 | 同向患者 | 精确 sign-flip p | 患者 bootstrap 95% CI | 裁决 |
|---|---:|---:|---:|---:|---|
| 修饰核苷加工 | +0.261 | 5/6 上升 | 0.1250 | [+0.038, +0.523] | 方向支持 |
| 嘌呤合成/回收 | +0.634 | 6/6 上升 | 0.03125 | [+0.351, +0.978] | 强方向支持；两主轴 Holm=0.0625 |
| 甲硫氨酸/SAH 循环 | +0.351 | 5/6 上升 | 0.0625 | [+0.156, +0.507] | 次级支持 |
| 长链肉碱穿梭/FAO | -0.567 | 6/6 下降 | 0.03125 | [-0.946, -0.242] | 方向支持；替代计分 5/6；两主轴 Holm=0.0625 |
| 鞘脂代谢 | +0.069 | 2/6 上升 | 0.9688 | [-0.137, +0.400] | 不支持统一方向 |

由于只有 6 对患者，bootstrap 区间用于描述效应稳定性，不能替代小样本精确检验。两条主轴的最小可达两侧精确 p 值即为 `2/2^6=0.03125`；两个共同主终点的两侧 Holm 校正均为 `0.0625`。若按既有代谢组方向做内部单侧检验，两轴 Holm 均为 `0.03125`，但该方向并未在公开时间戳中预注册，因此只能作为敏感性分析。

## 基因级驱动

轴分数并非由单个高表达基因支配：

- 嘌呤合成/回收：`ADA +1.184`（6/6）、`HPRT1 +0.877`（6/6）、`IMPDH1 +1.181`（5/6）、`GMPS +0.504`（5/6）、`APRT +0.403`（5/6）。
- 修饰核苷加工：`WDR4 +1.014`（5/6）、`METTL1 +0.760`（5/6）、`TRMT5 +0.758`（6/6）、`TRMT10C +0.527`（6/6）、`THUMPD3 +0.480`（6/6）。
- 长链 FAO：`ACADS -0.834`（6/6）、`SLC25A20 -0.847`（5/6）、`ETFDH -0.710`（5/6）、`CPT1A -0.529`（5/6）。

因此可以写“多个互补节点共同支持轴方向”，不能写“整条通路所有基因均一致变化”。

## 敏感性与反例

将上皮门收紧至至少三个 marker 后，三条关键轴仍保持方向：修饰核苷 `+0.351`、嘌呤 `+0.549`、FAO `-0.408`；但由于有效细胞更少，精确检验不再全部达到 `p<0.05`。这支持方向稳健，却不应包装成第二次独立统计复现。

细胞数平衡审计进一步将每位患者的肿瘤/正常上皮细胞下采样至相同数量，并进行 2,000 次重复：嘌呤平均效应的 95% 下采样区间为 `[+0.491,+0.774]`，FAO 为 `[-0.702,-0.417]`。即使所有 12 个样本统一只取 91 个细胞，区间仍分别为 `[+0.352,+0.864]` 和 `[-0.790,-0.221]`。因此两条平均方向不是细胞数不平衡造成；修饰核苷在统一 91 细胞时只有 34% 重采样达到至少 5/6 同向，再次支持其降级。

鞘脂轴在上皮中不一致；它在 T/NK compartment 中呈 6/6 上升，提示免疫组成或免疫状态相关性，但当前不纳入主故事。嘌呤轴在部分免疫谱系也可升高，因此“上皮存在该程序”成立，“该程序仅限肿瘤上皮”不成立。

## 与本地代谢组和独立蛋白组的闭环

| 证据层 | 修饰鸟苷/嘌呤轴 | 长链酰基肉碱/FAO 轴 |
|---|---|---|
| MTBLS13729 配对代谢组 | 修饰鸟苷样模块 10/10 上升，平均约 `+2.95 log2`；存在 purine-like 共变特征 | C20:4 arachidonoylcarnitine-like feature 3222 上升，并有长链酰基肉碱类别积累 |
| 独立黏液型 CRC 池化组织蛋白组 | 嘌呤固定面板 LMC/NC 与 RMC/NC 中位 `+0.42/+0.32` | FAO 固定面板 LMC/NC 与 RMC/NC 中位 `-0.29/-0.43`；配对部位比较 12/12 同向下降 |
| GSE236696 患者配对上皮 pseudobulk | 嘌呤轴 6/6 上升；修饰核苷轴 5/6 上升 | FAO 轴 6/6 下降 |

三层证据使用不同测量技术、不同队列和不同统计单位，因此方向一致具有价值；但也正因为测量对象不同，不能把这些数值合并成一个跨组学 p 值或“机制总分”。池化蛋白组不能提供患者级复现，当前上皮门也不是源研究 Seurat cluster 或恶性 CNV 标注的逐细胞复现，而是基于公开 marker 的保守组成敏感性分析。

## 论文中可以与不可以写的内容

可以写：

- paired metabolomics identified a modified-guanosine/purine-turnover axis and a long-chain-acylcarnitine accumulation axis;
- pooled mucinous CRC proteomics and patient-paired epithelial pseudobulk provide directionally concordant support for enhanced purine metabolism and constrained long-chain FAO;
- the epithelial result weakens, but does not eliminate, the explanation that bulk-tissue effects arise solely from cell-composition shifts.

不可以写：

- feature 1597 已经被确认是 m7G，或 feature 3019 已经被确认是 m2²G；
- feature 3222 已经通过标准品确认为 arachidonoylcarnitine；
- FAO flux、CPT1A 活性或某个甲基转移酶的因果作用已经被证明；
- 6 对单细胞队列构成大样本外部验证。
- 两条轴已被证明为黏液型 CRC 特异；TCGA 目前支持的是一般 CRC 肿瘤−正常程序，而非黏液型富集。

## 下一步最小闭环

1. 标准品优先级：`m7G` 与 `m2²G`；如预算允许，再加入 `m2G`、`Gm`。同法 RT 与完整 MS2 是位置异构体升级的必要条件。
2. 酰基肉碱优先级：C20:4、C16:0、C18:0、C18:1；首先确认 feature 3222，再验证类别级积累是否保持。
3. 在现有数据内完成患者级三轴联合图，但保持三轴并列，不依据结果构造复合分数。
4. BioAware 只负责离子家族折叠、证据槽、网络一致性与冲突弃权；网络不能替代结构标准品。

## 可复核产物

- `data/external/GSE236696/paired_axis_screen_v1/summary.json`
- `data/external/GSE236696/paired_axis_by_lineage_v3/summary.json`
- `data/external/GSE236696/paired_axis_by_lineage_v3/lineage_gene_paired_results.csv`
- `data/external/GSE236696/paired_axis_by_lineage_v3/lineage_axis_deltas.png`
- `data/external/GSE236696/paired_axis_by_lineage_markers3_min30_v1/summary.json`
- `data/external/GSE236696/epithelial_axis_adversarial_audit_v1/summary.json`
- `data/external/GSE236696/epithelial_axis_genomewide_matched_null_v1/summary.json`
- `data/external/GSE236696/epithelial_axis_cell_balance_v1/summary.json`
- `data/external/TCGA_COADREAD_Xena_20260830/mucinous_axis_analysis_v4/summary.json`
- `data/external/TCGA_COADREAD_Xena_20260830/mucinous_axis_analysis_v4/adjusted_axis_effects.png`
- `data/external/mucinous_crc_proteomics_2021/axis_reanalysis_v1/summary.json`
- `data/external/mucinous_crc_proteomics_2021/axis_reanalysis_v1/target_protein_heatmap.png`

## 数据来源

- GEO GSE236696: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE236696
- Source study: https://pmc.ncbi.nlm.nih.gov/articles/PMC11111627/
- Independent mucinous CRC tissue proteomics: https://mdpi-res.com/d_attachment/curroncol/curroncol-28-00305/article_deploy/curroncol-28-00305.pdf
