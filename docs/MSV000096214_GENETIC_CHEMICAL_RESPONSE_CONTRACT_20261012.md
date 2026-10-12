# MSV000096214 遗传反事实化学响应合同

日期：2026-10-12  
状态：主科学问题在打开补充表 abundance/annotation values 前冻结；development/consumed。2026-10-12 仅查看表结构后、首次效应计算前加入第 8.1 节的可识别性修正与精确计分式。

## 1. 方法学问题

本实验不以 Atlas、数据库注释或大规模公共邻域为前提。核心问题是：

> 在结构身份隐藏时，MS/MS 定义的化学关系能否把遗传学的“等位复现—路径特异性—双突变上位/救援”转化为可迁移、可证伪的未知化学响应？

这比“哪些代谢物显著”更严格。一个合格的结构未知化学单元必须同时满足：独立 tt7 等位/突变线方向复现、相对 tt3/tt4/fls1 的特异性，以及在 tt7×tt4 双突变中按上游阻断预测减弱。结构注释只在这些结果冻结后解盲。

## 2. 数据和分母

- 数据集：MassIVE `MSV000096214`；Arabidopsis thaliana seeds；Bruker impact II；positive/negative LC-MS/MS。
- 角色：`development/consumed`；论文和数据已公开，不是外部封存。
- 登记 50 个生物样本、10 个 genotype，每 genotype 原计划 5 个重复。
- 实际沉积 49 个完整正负离子样本；`fls1-3` sample 33 两极性均缺失，因此 fls1 为 4 个重复，其余每 genotype 5 个。
- genotype：WT、3 条独立 tt7（115 EMS、tt7-8、tt7-9）、tt4、tt3、fls1、3 条对应 tt7×tt4。
- 研究样本 98 个 mzXML 全部含 MS2，共 1,095,667 MS2；6 个 pooled-QC 文件另有 66,958 MS2；2 个 blank。
- 补充 Table S1 报告 920 个 feature，但在本合同冻结前不得读取其 abundance 和 annotation values。

## 3. 标签盲数据构建

1. 只从补充表读取 feature ID、ion mode、m/z、RT 与 49 个实际沉积生物样本的 abundance；annotation、metabolic family、annotation level、作者 cluster 和显著性列隔离保存，不进入开发。
2. QC、blank、缺失、重复性、ion-family 去冗余和 feature--MS2 linkage 不读取 genotype。
3. 谱学证据优先由 pooled-QC 构建。样本 DDA 只有在报告 genotype-specific acquisition propensity 且不以 genotype 选择代表谱时才能补充。
4. 同一 peak-list hash 只形成一个 spectral unit；多个 abundance features 共享一张谱时不得重复计票。
5. 所有算法通道使用同一个 feature universe、同一 49-sample graph 和同一外层 split。

## 4. 三个冻结真值任务

### G1：独立 tt7 等位留出

依次完整留出 `tt7_115`、`tt7-8`、`tt7-9`。训练只使用另外两条 tt7 与 WT 学习 response；测试时区分留出 tt7 与 WT、tt3、tt4、fls1。主终点为 allele-held-out macro AUROC、AUPRC、margin 和 genotype-balanced accuracy。三条留出线必须同号，不能由某一条线单独驱动。

### G2：路径特异性

将 G1 训练 signature 用于 tt3、tt4、fls1。合格 response 不能仅区分“任何 flavonoid mutant vs WT”；tt7 score 必须高于三个 alternative single mutants。按 genotype 聚类报告错误，不按 49 个样本制造伪重复。

### G3：遗传上位/救援

对三条 lineage 分别比较单 tt7 与对应 `tt7×tt4`。预注册方向是双突变的 tt7-like score 下降，并更接近 tt4/WT reference。主分母为 3 条 lineage，不是 30 个进样。若只有 published annotation 或已知 kaempferol features 显示救援，而结构隐藏的整体/家族响应不显示，则 structure-agnostic 方法失败。

G1 是进入谱学比较的最低正控；G2 和 G3 是功能特异性门。任何一个失败，均不得把可视化网络或分类准确率包装成新方法。

## 5. 比较世界

按同一 split 比较：

1. `MS1-all`：完整 QC 合格 abundance response；
2. `MS1-MS2-observed`：只限制到有 MS2 的 features，不使用碎裂关系；
3. m/z/RT、raw cosine、FBMN/作者网络（如可重建）、MEMO/MS2LDA 的适用版本；
4. WSE、official DreaMS、Noise V1；
5. 冻结 P2b，且 composite 与原始通道不能重复计票；
6. ChemAware 各 checkpoint 仅按资格开发消融；
7. degree、polarity、mass、peak count 与 acquisition propensity 匹配的 shuffled-spectral null。

主增量为完整方法减去以上适用世界的最大值。模块必须可归零；失败的 RRF 和 BioAware B44--B46 不进入。

## 6. 真正的新科学对象

潜在贡献不是新的 kNN 图，而是 `genetically adjudicated spectral response unit`：一个谱学单元/家族只有在训练等位中复现、留出等位中预测成功、alternative mutants 中保持特异、并在上游双突变中按方向消退，才获得有限的功能关系证书。

该证书只支持“与该遗传反事实响应一致的化学单元”，不支持精确结构、直接酶底物、反应方向或通量。公共参考可在事后提供检索线索，但不是证书成立条件。

## 7. annotation 解盲与 unknown 增量

G1--G3 和所有谱学权重冻结后，才打开 Table S1 的 annotation level：Level 1--3 为 library/class-supported，Level 4 为 unresolved。固定比较：known-only、measurement-matched known、unresolved-only、known+unresolved、unresolved shuffled-MS2。

主 unknown 估计是 `known+unresolved - max(known-only, matched-known, unresolved shuffled-MS2)` 在 G1--G3 联合终点上的增量。若 unknown-only 有信息但加入 known 后无增量，只能说未知信号携带信息，不能说改善发现。

## 8. 通过门和外推边界

- G1/G2 联合门：三条 allele-held-out 对 WT 的 AUROC 均 >0.5；并对 6 个 single-mutant genotype 中任取 3 个作为伪 target 的全部 `C(6,3)=20` 种分配做穷举。主统计量为每个 held target 相对 WT 和 3 个 complement genotype 的 12 个 genotype-balanced pairwise AUROC 的最小值；真实 tt7 集合必须是 20 种分配中唯一最优，即精确 `p=1/20=0.05`。由于分辨率限制，不写成 `p<0.05`；
- G2：三种 alternative single mutants 的 tt7 false-positive 均不恶化；
- G3：三条 lineage 的 single-to-double margin 全部按预注册方向；
- 谱学增量：相对全部强基线同号，allele/lineage cluster CI 下界严格大于 0，且 coverage/abstention 不恶化；
- unknown 增量：matched-known 和 shuffled-MS2 后仍为正。

这里只有 3 条 tt7 allele 与 3 条 double-mutant lineage，统计分辨率有限。即使全过，也只是同一研究、同一物种、同一仪器的 development evidence。通用主张需要另一条生物合成轴或另一实验室的独立扰动研究，且规则完全冻结。

## 8.1 表结构后、效应计算前的精确实现冻结

本节加入时已经确认 Table S1 为 920 features、49 个实际生物样本、正负离子各约一半；程序曾在格式调试时意外打印前两行 abundance/annotation，但没有计算 genotype effect、挑 feature、看分组均值或据此选择规则。以下实现由实验设计与统计可识别性决定，不使用这两行的内容。

1. 主预处理为 `log1p(abundance)`，随后在每个 sample、每个 ion mode 内减去 feature 中位数；raw `log1p` 为固定 sensitivity。
2. 对每个 held target，使用另两条 target genotype 与 WT。逐 feature 计算两个 `median(target)-median(WT)`；仅保留方向一致者，权重为共同方向乘两者绝对效应的较小值。无 top-K、p-value 筛选、annotation 或作者 network。
3. 正负离子分别计算样本相对 WT 中位向量与 signature 的 cosine，再对两极性等权平均。给 WT 自身计分时 leave-one-WT-sample-out 重建中心与 signature，避免把该样本同时作为中心和测试点。
4. G1 报告三条 held allele 相对 WT 的 AUROC/AUPRC、均值 margin；G2 报告每条 held allele 相对 tt3、tt4、fls1 的 pairwise AUROC 与最差 specificity margin。联合主统计量和 20 种穷举检验按第 8 节执行。
5. G3 对每条真实 lineage 使用其 held-allele signature，报告 `mean(single tt7 score)-mean(corresponding tt7x tt4 score)`；还要求 double mutant 到 tt4 score 中心的绝对距离小于 single mutant。主分母严格为 3 lineage。3/3 同向仅是遗传复现门，不声称独立 `p<0.05`。
6. 所有 sample-level bootstrap 只作为条件于冻结 signature 的描述性区间；不得替代 genotype/lineage 分母。首次运行只裁决 MS1 正控是否值得进入谱学增量实验，不证明未知 MS/MS 已有价值。

