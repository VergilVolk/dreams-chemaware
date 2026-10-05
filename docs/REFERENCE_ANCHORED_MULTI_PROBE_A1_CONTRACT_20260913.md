# 多标准锚定化学坐标 A1：低成本证伪合同

## 目的

A1 不建设样本 Atlas，也不训练 DreaMS。它只回答一个前置问题：一个标准身份相对于许多第三方标准的谱学响应排列，是否包含超出直接两谱相似度的结构邻域信息。

## 固定数据与成本

- 仅使用已冻结的 153 张公共参考谱、73 个 panel-specific 身份和已抽取的小型 MGF；
- 不读取 119 个样本的原始 mzML，不编码大模型，不使用 GPU 计算；
- Slurm 只申请集群要求的 1 张 GPU，限时 20 分钟，算法本身运行在 CPU；
- 不使用 phenotype、MS1 丰度、P2b、BioAware 或任何身份纠正结果。

## 新对象

对 query 身份 q 和候选身份 c，分别计算它们对其余标准身份集合 A 的六通道响应：fragment sqrt-cosine、entropy、top-10 峰、强度覆盖、匹配峰比例和 neutral-loss cosine。比较 q 与 c 时，q 和 c 都从 landmark 集合 A 中移除。

`multi_anchor_profile(q,c)` 是六条第三方标准响应排序的一致性，不能退化为 q 与 c 的直接谱图匹配。为排除“只是学到前体质量相近”，强基线固定为：

`direct_mass = 0.5 * direct_multichannel + 0.5 * min(m1,m2)/max(m1,m2)`。

主方法固定为：

`multi_anchor_augmented = (direct_multichannel + precursor_mass_similarity + multi_anchor_profile) / 3`。

权重不根据结果搜索。

## 隐藏真值

RDKit Morgan 指纹仅在所有谱学分数构建完成后用于评估：结构不得进入 multi-anchor 或 direct score。每个 query 的输出是其化学邻域恢复质量，而不是精确身份命中。

主要指标为 NDCG@5；另报选中邻居 Tanimoto、top-3 structural hit、候选内 Spearman 和 structural regret。

## 进入下一阶段的门

只有同时满足以下条件，才允许进入 A1b 官方 DreaMS 多锚点确认：

1. multi-anchor augmented 相对固定 direct+mass 强基线的 IK14-cluster bootstrap NDCG@5 95% CI 下界严格大于 0；
2. 正负离子 panel 的 NDCG@5 均不下降；
3. 至少 50 个身份，且每个 panel 至少 20 个身份；
4. 查询与候选身份从其 pair-specific landmark 集合中排除。

若失败，结论是当前 73 身份面板未支持“多标准坐标”增量；停止样本级扩建，而不是调权重追分。即使通过，也必须先用官方 DreaMS 响应矩阵复核；A1 本身不直接授权样本级算子建设。

## 声称边界

A1 不是注释率、未知结构解析、MSI Level 1、峰位点解释或生物学结果。即使通过，也只证明第三方标准响应具有关系型结构 headroom；下一阶段仍须显式构建保留、平移、丢失和新生峰，并做 study/instrument-held-out 验证。
