# MTBLS13729 生物学部分逆向审计与缺口清单（2026-08-30）

## 一、审计后的总裁决

此前故事方向有价值，但部分措辞超过了数据能支持的强度。审计后应按三层证据表述：

1. **稳固层：一般 CRC 肿瘤程序。** TCGA COADREAD 32 对肿瘤/正常中，嘌呤轴平均升高 `+0.779 z`，29/32 同向，BH `q=3.34e-8`；长链 FAO 轴平均下降 `-1.669 z`，32/32 同向，BH `q=1.16e-9`。这与 MTBLS13729 的嘌呤样共变和长链酰基肉碱积累方向相容。
2. **中等层：黏液型患者上皮中的方向一致。** GSE236696 六对上皮 pseudobulk 中嘌呤 6/6 上升、FAO 原计分 6/6 下降；原始两侧 p 均为 `0.03125`，但两个共同主终点 Holm 后均为 `0.0625`。嘌呤稳健性强于 FAO。
3. **探索层：修饰核苷加工程序与黏液型特异性。** 修饰核苷轴只有 5/6、p=0.125，且对留一基因与计分方法敏感；TCGA 的黏液型对常规型比较也不支持嘌呤/FAO 的黏液型特异富集。

这意味着项目不是“没有生物学结果”，而是主结论应从“黏液型特异代谢机制已验证”收敛为：**算法辅助发现了两个可复核的静态代谢轴，它们与一般 CRC 的嘌呤增强和 FAO 受限程序一致；Rmu 中的幅度和具体离子家族值得进一步验证，但亚型特异性尚未成立。**

## 二、逐项纠偏

| 原表述或隐含结论 | 审计发现 | 修正状态 |
|---|---|---|
| GSE236696 两条轴“独立显著复现” | 两侧原始 p=0.03125；两共同主终点 Holm=0.0625 | 改为强方向性复核 |
| FAO 6/6 是完全稳健结果 | 原计分 6/6；替代逐基因计分 5/6，p=0.0625–0.09375 | 保留平均方向，降低一致性强度 |
| 修饰核苷转录轴支持具体候选 | 留一 RNMT/TRMT10C/TGS1/DCP2 后仅 3/6，留一 THUMPD3 后 2/6 | 降级为探索性，不用于身份确认 |
| 上皮门复现了源论文的恶性上皮结果 | 当前是公开 marker 构造的 broad epithelial gate，不是源 Seurat cluster/CNV 标注 | 仅称组成敏感性分析 |
| 两轴可能为黏液型特异 | TCGA 黏液型 vs 常规型未支持；嘌呤甚至方向较低且 5 轴 q=0.070 | 删除“已证实特异” |
| 跨组学方向一致可证明通量/酶活 | 所有现有数据仍是静态丰度、转录或池化蛋白 | 明确禁止通量和酶因果措辞 |

## 三、现有结果的最强部分

### 3.1 嘌呤轴

- GSE236696：原计分和两个替代计分均为 6/6 上升；均值范围约 `+0.371` 至 `+0.634`。
- 留一基因后均值始终为正；去掉 HPRT1 时同向性从 6/6 降为 5/6，说明 HPRT1 重要但不是唯一驱动。
- 在 21,400 个候选背景基因中按平均表达、检出率和样本间标准差逐基因匹配，并抽取 20,000 个同规模随机轴后，均值方向经验 `p=0.0113`；说明该轴的效应强度超出普通可观测基因集背景。6/6 同向本身的匹配背景经验 `p=0.112`，因此证据主要来自效应幅度和跨计分稳定性，而不能只强调 6/6。
- TCGA 配对肿瘤/正常中 29/32 上升，构成一般 CRC 背景的强外部支撑。

### 3.2 长链 FAO/酰基肉碱轴

- GSE236696 原计分均值 `-0.567`，6/6；替代计分均值仍为负，但为 5/6。
- 全基因表达匹配的 20,000 个随机轴中，均值方向经验 `p=0.00020`，6/6 同向经验 `p=0.00510`；这是当前 GSE 轴中最强的基因集特异性结果。
- TCGA 配对肿瘤/正常 32/32 下降，是当前最稳的外部转录背景。
- MTBLS13729 feature 3222 与类别级长链酰基肉碱积累提供静态代谢物层锚点；但这不能区分输入增加、氧化受限、线粒体数量变化或细胞组成。

### 3.3 多胺乙酰化候选轴：丰度强，原始 MS2 和跨色谱证据已补齐，标准品终证仍缺

- feature 1717 在 59 个样本中检出率为 `94.9%`，Rmu 有值的 9/9 对全部升高；四种归一化的平均效应为 `+2.859` 至 `+3.009 log2FC`，精确 sign-flip p 均为 `0.00390625`，所有留一患者均值仍大于 `+2.38 log2FC`。因此“存在一个强而独立的 m/z 230.185931 多胺样丰度轴”有扎实局部证据。
- 通用 MS1–MS2 桥接产物虽然给出 `0` 条 accepted link，但原始 mzML 峰界重审计找到 `73` 张 DDA MS2、覆盖 `45` 个样本；`m/z 100.0759` 在 `73/73` 中为基峰，和既往 CRC 中 N1,N8-diacetylspermidine 的 `230.2 -> 100.0` MRM 转换一致。旧桥接表的“0”是工程漏接，不是没有实验 MS2。
- 同源 independent HILIC 峰 HMDB0041947/N1,N8-diacetylspermidine 与 RP feature 1717 在 `59` 个样本中 Spearman `rho=0.860`，组织标签内残差 `rho=0.756`，`29` 对肿瘤-癌旁差值 `rho=0.719`；在 `85` 个可评估 HILIC 注释中相关性排名第 `1`。这构成强正交证据，但 HILIC MAF 没有碎片/可靠性分数，且 HILIC m/z 自身偏离理论值超过 10 ppm。
- 独立 MTBLS8090 的 35 对 CRC 中，N1-acetylspermidine、N1-acetylspermine 和 spermidine 均未在三项 BH 校正后显著。因此外部数据没有复现“泛 CRC 多胺乙酰化升高”，但因缺少 N1,N8-diacetylspermidine 本身和 Rmu 标签，也不能否定本地候选。
- 正确论文位置从“质量匹配探索轴”升级为“最强新候选代谢轴”，写作必须使用 `N1,N8-diacetylspermidine-like` 或 `candidate`。只有同法标准品 RT、标准品 MS2 与加标共洗脱通过后，才可去掉限定词并升级为命名代谢物。

## 四、仍缺乏且必须进一步挖掘的内容

### P0：不增加新数据即可完成

1. **用源研究精确细胞注释复核。** 当前本地只用 marker gate。应优先取得作者 Seurat 对象或 cluster-to-cell barcode 表；若不可得，再做基于拷贝数异常的恶性上皮识别和源论文 14 个 cluster 的标签转移。
2. **全表达基因匹配零分布已完成。** 33,660 个符号中 21,400 个基因满足背景资格；按平均表达、检出率和样本间标准差逐靶基因匹配。嘌呤和 FAO 通过效应幅度特异性，修饰核苷未通过。下一层如需正式竞争性通路检验，应再用 CAMERA/rotation 处理基因间相关性。
3. **改用计数模型复核 pseudobulk。** 现有差值轴适合方向审计，但正式论文应增加患者配对的 edgeR/DESeq2 pseudobulk，报告每个轴的 CAMERA/fgsea 或受限基因集检验；n=6 时仍以效应和方向为主。
4. **拆分细胞组成与细胞内改变。** 对每患者报告上皮亚群比例、轴内表达和总组织贡献三部分，避免把 gate 后残余亚群构成误称为细胞内代谢重编程。
5. **TCGA 高覆盖 MSI 敏感性已补完。** 改用标准化 `CDE_ID_3226963` 并以旧字段补缺后，371 例中 364 例 MSI 完整。MSI 调整后，FAO 的黏液型差异仍接近零（beta `+0.010`, p=`0.911`）；嘌呤在黏液型中反而较低（beta `-0.232`, p=`0.0091`），进一步否定“黏液型特异嘌呤增强”。但 BRAF/KRAS 实测结果只有几十例，不能稳定纳入模型；CMS 和可靠连续纯度字段也不在当前矩阵中。
6. **细胞数平衡敏感性已补完。** 患者内等细胞数与全样本统一 91 细胞的 2,000 次重采样中，嘌呤和 FAO 的平均效应区间始终不跨零；修饰核苷的患者一致性明显下降。下一步仍需源 cluster/CNV 标注处理门内亚群组成。

### P1：决定结构与生物学故事能否升级

1. **feature 3222 标准品。** C20:4 酰基肉碱同法 RT、完整 MS2 和加标共洗脱；未完成前保持 arachidonoylcarnitine-like。
2. **修饰鸟苷标准品。** m7G 与 m2²G 为最低组合；m2G、Gm 为扩展组合。必须解决位置异构体而非只复现核糖丢失。
3. **feature 1717 标准品。** N1,N8-diacetylspermidine 的优先级不低于 feature 3222：当前已有 73 张峰界内 MS2、稳定的 100.0759 产物离子和跨色谱定量一致，差的正是能终结位置异构体争议的标准品 RT/MS2/加标共洗脱。
4. **独立原始组织代谢组复核。** 优先寻找含配对肿瘤/癌旁、可获得 mzML/mzXML、能分辨黏液型或 MSI/CMS 的数据；只用作者差异表不足以确认同一离子。
5. **公开空间数据桥接。** GSE236696 源研究含空间证据，但当前尚未把轴定位到黏液池、侵袭前沿或特定上皮亚群。空间转录只能定位表达程序，不能替代空间代谢物测量。

### P2：有资源再做

1. 标准品确认后的靶向 EIC 在独立队列复现。
2. 公开同位素示踪或扰动数据中验证嘌呤合成/回收和 FAO 的方向；只能作为外部机制证据，不能冒充本队列通量。
3. 将 BioAware 网络作为候选证据与冲突检测模块，不用网络邻近性代替身份或表型标签。

## 五、最短论文路径

1. 主轴保留“嘌呤/核苷周转”和“长链酰基肉碱/FAO 利用受限”，修饰核苷具体酶轴降为探索性。
2. 主生物学结果以 MTBLS13729 的配对静态丰度为发现，以 GSE236696、池化蛋白组和 TCGA 为不同强度的方向性支撑，不合并 p 值。
3. 将“mucinous-specific”改为“observed in the Rmu subgroup and concordant with broader CRC programs”；只有交互检验或亚型间直接比较通过，才恢复“specific”。
4. 若只能做一个额外实验，优先 feature 3222 标准品；若只能做一个额外计算，优先源细胞注释/CNV 恶性上皮复核。

## 六、外部文献与数据的正反证分流

1. **MTBLS8090 是必须保留的阴性外部代谢组。** 35 对 CRC 肿瘤/癌旁中，24 个不看表型预冻结的 C≥12 酰基肉碱类中位效应均值为 `-0.196 log2`，Wilcoxon `p=0.273`、符号置换 `p=0.429`。它不复现泛 CRC 的 LCAC 积累，但没有 Rmu 标签，既不能验证也不能否定 Rmu 特定现象。
2. **2026 空间代谢组提示阶段/空间分离。** 23 份组织的 AFADESI-MSI 研究报告原发 CRC 中嘌呤、氨基酸和胆碱代谢升高，而酰基肉碱更突出于肝转移并被解释为不完全 FAO/线粒体功能异常。它支持“嘌呤与酰基肉碱不是所有阶段都同步”的框架，但不验证 feature 3222 或黏液型特异性。
3. **一般 CRC 嘌呤证据较充足。** 近期靶向核苷代谢和跨肿瘤嘌呤研究支持 CRC 中嘌呤合成/回收需求增加；这些工作提高通路合理性，但不能确认本地修饰鸟苷位置异构体。
4. **FAO 机制具有情境依赖性。** CRC 的 p53-过氧化物酶体 FAO-嘌呤研究提供可实验验证的通路连接；另一方面，耐受、酸性或转移环境也可能提高脂肪酸利用。因此静态 FAO 转录下降与酰基肉碱积累只能形成“利用受限”假设，不能推广为所有 CRC 的 FAO 通量下降。
5. **CMS 是优先于继续强调 mucinous 的分层框架。** CMS3 具有代谢特征，CMS1 与 MSI/免疫相关，CMS4 受基质成分影响。下一外部队列若有 CMS 标签，应先检验两轴在 CMS/MSI 中的异质性，再决定是否恢复组织学亚型叙事。
6. **多胺文献只提供先验，不提供本地身份。** 既往研究在 CRC 尿液或组织中报告乙酰化多胺升高，但标本类型、具体分子和分析方法均不同。MTBLS8090 的三项可用相关多胺也未显著复现，故文献合理性不能替代 feature 1717 的实验 MS2 与标准品。

主要外部入口：

- MTBLS8090: https://www.omicsdi.org/dataset/metabolights_dataset/MTBLS8090
- GSE236696: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE236696
- 2026 CRC liver-metastasis spatial metabolomics: https://pmc.ncbi.nlm.nih.gov/articles/PMC13208894/
- CRC consensus molecular subtypes: https://pmc.ncbi.nlm.nih.gov/articles/PMC4636487/
- UCSC Xena public TCGA matrices: https://xena.ucsc.edu/download-data/

## 七、可复核产物

- `data/external/GSE236696/epithelial_axis_adversarial_audit_v1/summary.json`
- `data/external/GSE236696/epithelial_axis_genomewide_matched_null_v1/summary.json`
- `data/external/GSE236696/epithelial_axis_cell_balance_v1/summary.json`
- `data/external/TCGA_COADREAD_Xena_20260830/mucinous_axis_analysis_v4/summary.json`
- `data/external/TCGA_COADREAD_Xena_20260830/mucinous_axis_analysis_v4/axis_results.csv`
- `data/external/TCGA_COADREAD_Xena_20260830/mucinous_axis_analysis_v4/paired_tumor_normal_axis_results.csv`
- `data/external/TCGA_COADREAD_Xena_20260830/mucinous_axis_analysis_v4/adjusted_axis_effects.png`
- `data/external/MTBLS8090/lcac_replication/lcac_replication_report.json`
- `data/mtbls13729/polyamine_axis_adversarial_audit_v1/summary.json`
- `data/mtbls13729/polyamine_axis_adversarial_audit_v1/local_normalization_audit.csv`
- `data/mtbls13729/polyamine_axis_adversarial_audit_v1/mtbls8090_polyamine_context.csv`

## 八、声明边界

本审计不改变原始代谢组统计结果，只修正跨组学验证的强度和可写结论。静态组织代谢组、转录组和蛋白组均不能独立建立通量、酶活性或因果方向。
