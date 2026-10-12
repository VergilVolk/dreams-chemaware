# MSV000096214 遗传扰动 LC-MS/MS readiness 结果

日期：2026-10-12  
状态：正式数据/设计就绪结果；未打开 abundance、annotation 或论文结果表

## 裁决

`MSV000096214` 通过进入 truth-blind feature-table 审计的全部门，是当前比 ST003325 更合格的事件级开发数据。其优势来自遗传反事实设计，而不是公共数据规模或 Atlas 覆盖。

## 数据角色和分母

- 数据集：MassIVE `MSV000096214`；`development/consumed`；`sealed=false`。
- 仓库：215 files，34,975,433,409 bytes；其中 106 peak mzXML，34,949,658,872 bytes。
- 登记 50 个生物样本、10 genotype、正负离子各一次；实际沉积 49 个完整 paired-polarity 样本。
- 唯一缺失：fls1-3 的 sample 33，positive 与 negative 均未沉积。因此 fls1 每极性 n=4，其余 genotype 每极性 n=5。
- 技术控制：6 pooled-QC files、2 blanks。
- 研究样本：98/98 文件含 MS2；1,332,483 total scans，其中 1,095,667 MS2。
- pooled-QC：66,958 MS2。

## 设计资产

数据包含 WT；三条独立 tt7 线（EMS、tt7-8、tt7-9）；tt4、tt3、fls1 三个路径位置对照；以及三条与 tt7 lineage 对应的 tt7×tt4 双突变。这个设计允许同时测试等位复现、路径特异性和遗传上位/救援，不需要先知道 feature 的结构。

## 泄漏状态

本次只读取了 MassIVE inventory、FAIR sample metadata、params.xml 和每个文件的 `.scans` 层级清单。未读取任何 feature abundance、差异分析、annotation、molecular network 或作者 cluster。因此 readiness 数字无 outcome leakage；但数据与论文公开，后续所有使用均登记为 development/consumed。

## 限制

通过 readiness 不等于方法有效。论文已报告强 tt7 代谢表型，所以普通 MS1 分类很可能容易；真正门槛是：在完整留出一条 tt7 allele 时仍能预测，并利用双突变反事实排除“任何 mutant 都相似”，最后证明 unresolved 谱学信息对 matched-known 有条件增量。

机器可读报告：`data/validation/msv000096214_perturbation_readiness_20261012_v2/report.json`。

