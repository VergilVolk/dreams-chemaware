# ChemProp 跨设计 M1b 冻结合同

日期：2026-10-11  
状态：在读取跨设计响应检索结果前冻结；development/consumed，不是跨实验室外测

## 科学问题

在没有共享 feature ID、且不使用待预测药物响应建立谱学对应时，MS/MS 证据能否相对 MS1/m/z–RT 基线提高同一项目内 endpoint screen 到 time-series 的生物学响应恢复？

这只是通用 structure-agnostic 方法的第一个有监督能力测试，不定义总方法。

## 数据与分母

- query：MSV000094899，12-drug、T0–T8；本轮只取 T0/T2，3 biological replicates/drug；
- endpoint batch A：MSV000096724，23-drug screen；共享 3 drugs，2 biological replicates；
- endpoint batch B：MSV000093571，45-drug screen；共享 9 drugs，3 bacterial replicates + 1 abiotic control；
- 三者属于同一研究项目、同类 Q Exactive HF 平台，不得称为跨实验室、跨仪器或独立外测；
- query 原始 MS2 feature 分母 8,055；主生物学分析固定使用上一轮表型盲 QC 的 3,538 features；两者的 overlap 分母分别报告；
- endpoint features 独立执行 blank >= 3x 过滤并要求非空 MS/MS。

## 预先固定的药物分层

按论文选择规则，在本轮结果打开前固定：

- high ChemProp1 signal（至少一个 score >= 1 edge，9 drugs）：Cilnidipine、Clomifen、Erythromycin、Ketoconazole、Lansoprazole、Loratadine、Metronidazole、Omeprazole、Sertraline；
- low-signal challenge（3 drugs）：Montelukast、Simvastatin、Telmisartan。

必须同时报告 pooled、high 和 low；不得依据本轮结果重新分组。

## 响应终点

两类设计都使用：

`(Bac T2 - noBac T2) - (Bac T0 - noBac T0)`。

不用 T3–T8，不用 drug label 建立 feature 对应，不使用响应共同优化对应。time-series 每个 replicate 是一个 query；endpoint replicates 只在研究内平均。总计 36 queries：23-drug batch 为 3 drugs x 3 queries、3-way；45-drug batch 为 9 drugs x 3 queries、9-way。不同 batch 的候选数不同，必须分层报告，pooled 仅作汇总。

## 冻结对应方法

1. precursor mass 最近邻，10 ppm；
2. precursor mass 10 ppm + RT 绝对差 <= 0.5 min；0.5 min 是 5-min gradient 的固定 10%，不由本轮结果选择；
3. 10 ppm 候选内 sqrt-intensity raw MS/MS cosine Top-1；
4. 方法 3 加 cosine >= 0.70 或 >= 0.80 abstention；
5. 方法 3 的真正双向 mutual Top-1 + cosine >= 0.70。

所有方法必须在同一 query QC universe 报告 feature coverage。10 ppm、cosine 或 mutual 只表示候选/规则通过，不等于正确实体。

## 主要终点与负对照

- 生物学主终点：同药物跨设计检索 R@1、MRR、rank、margin；
- 对应层辅助终点：可用 pair 数、coverage，以及不完整 MQScore >= 0.80 planar InChIKey 子集上的 Top-1；
- 负对照：reference drug label permutation；matched pairs 的 reference feature independent permutation；
- 主比较：每个谱学方法减 precursor-mass 最近邻，按 12 drugs 聚类 bootstrap 95% CI；
- MS/MS 增量通过门：R@1 或 MRR 的 paired drug-cluster CI 下界严格 > 0，且 high/low 同方向、alignment permutation 明显较差；
- 若谱学方法不超过 mass/RT 基线，则记录负结果，不放宽阈值、不加入 relation transport 挽救。

## 尚未纳入首轮脚本、但正式结论前必须补齐

- WSE、official DreaMS、Noise V1、冻结 P2b：必须复用相同候选图、QC universe 和终点；
- ChemProp/ChemProp2 原始 edge/decoy-FDR 输出：作为原任务竞争者；
- 母药、药物相关库节点与母药所在网络成分排除；
- known/unknown 分层和真实 no-match 错误率；unknown 没有身份真值时不得借用 known 子集准确率。

Relation transport 只有在严格实体层完成后才能作为独立增量假设测试，且永不升级为身份、反应方向或机制真值。
