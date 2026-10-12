# MSV000096214 结构隐藏 MS1 遗传反事实结果

日期：2026-10-12  
状态：正式 development/consumed 正控结果；不是封存外测，不是 MS/MS 增量结果

## 裁决

在作者公开、已处理的 Table S1 中，结构注释完全隔离后，tt7 的“独立等位复现—替代突变体特异性—tt7×tt4 上位塌缩”全部通过预注册门。这证明该遗传设计有足够信号，值得进入 raw、表型盲 feature reconstruction 与谱学增量阶段。

它不证明 DreaMS、Noise、P2b、ChemAware 或未知 MS/MS 已产生生物学增量，也不证明端到端 unbiased discovery。Table S1 是作者用 MZmine 处理并按 apigenin internal standard 与 seed weight 归一化的 feature universe；论文结果公开，且 genotype 与文件编号整块绑定。

## 数据、分母和泄漏状态

- 数据集：MassIVE `MSV000096214`；`development/consumed`；`sealed=false`。
- 49 个实际沉积生物样本、10 genotype；WT、3 条 tt7、tt4、tt3、fls1、3 条对应 tt7×tt4。
- 920 features：ESI+ 457、ESI− 463；49×920 主矩阵无非数值或缺失单元。
- annotation、metabolite family、annotation level 和作者 molecular-network cluster 写入单独 `SEALED_annotations.csv.gz`，本次分析未读取。
- 精确 target-set 推断单位是 6 个 single-mutant genotype 中选择 3 个 target 的全部 `C(6,3)=20` 种分配；最小可达 p 值为 0.05。

## 冻结主结果

主预处理：`log1p` 后在每个 sample、每个 polarity 内做中位中心化；正负离子等权。每次完整留出一条 tt7，只用另两条 tt7 与 WT 构造全部 feature、无 top-K 的方向一致 signature。

- 真实 tt7 target set 的 12 个 pairwise 比较（3 held alleles × WT/tt3/tt4/fls1）最差 AUROC：`1.000`。
- 20 种 genotype-cluster 分配中真实 tt7 集合唯一排名第 1；精确右尾 `p=1/20=0.05`。
- 三条 held allele 相对 WT 的 AUROC/AUPRC 均为 `1.000/1.000`。
- 9 个 held-tt7 vs alternative-mutant mean margin 全部为正；最小为 `+0.4655`。
- 三条 single-tt7 minus paired-double score margin：`+0.5084`、`+0.4783`、`+0.4661`；3/3 为正。
- 三条 double mutant 均比 paired single tt7 更接近 tt4 score center。
- raw-log1p sensitivity 的所有门与主分析相同。

## 表型盲稳健性

预设五个世界全部得到相同 gate pattern，且真实 target set 的最差 AUROC 都为 1.000、精确 p 都为 0.05：

| 世界 | features | 结果 |
|---|---:|---|
| all | 920 | 全部门通过 |
| pooled-QC CV ≤20% | 812 | 全部门通过 |
| pooled-QC CV ≤30% | 867 | 全部门通过 |
| ESI+ only | 457 | 全部门通过 |
| ESI− only | 463 | 全部门通过 |

3 个 pooled-QC 的 feature CV 中位数为 9.28%，90% 分位为 21.52%，95% 分位为 32.57%。这些敏感性说明结果不是由单一极性或低重复性 features 单独驱动。

## 必须保留的反证边界

1. 作者论文已经报告 tt7 的强代谢表型，因此这是 consumed positive control，不是新生物学发现。
2. Table S1 是否包含作者选择过的“specialised metabolites”而不是完整 raw feature universe，不能由补充表本身排除。
3. tube/file ordinal 与 genotype 整块绑定；等位复现和双突变塌缩削弱单调漂移解释，但不能消除所有 block/batch confounding。
4. 只有 20 个 genotype-cluster 分配；`p=0.05` 是分辨率极限，不得写成高精度显著性。
5. 当前未使用 MS/MS 峰、embedding、谱学邻域、公共 reference 或结构 annotation；不能据此声称算法带来生物学价值。

## 后续决策

主线转向仓库已冻结的 raw mzXML 重建合同 `GLM_MSV000096214_STRUCTURE_HIDDEN_CONTRACT_20261012.md`。该路线只用 pooled-QC 定标、在开 genotype 前检测/对齐/补空，并在效应阶段使用 genotype-cluster 穷举而非重复样本伪推断。

尝试把作者 feature 直接以 10 ppm/6 s 连到 pooled-QC MS2 时，首个 negative QC 的 11,105 个 MS2 得到 0 个主链接；诊断显示若干最近候选存在约 14--16 ppm 的系统质量偏移。因此该直接链接按冻结规则停止，不事后放宽窗口救阳性。raw reconstruction 将从同一原始数据内部建立 MS1 feature 与 MS2 precursor 对应，避免依赖作者表与原始隔离质量的未校准偏移。

机器可读结果：

- `data/validation/msv000096214_structure_hidden_table_20261012/report.json`
- `data/validation/msv000096214_structure_hidden_ms1_baseline_20261012/report.json`
- `data/validation/msv000096214_structure_hidden_ms1_robustness_20261012/report.json`
