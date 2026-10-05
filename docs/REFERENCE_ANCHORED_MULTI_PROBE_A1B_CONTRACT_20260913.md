# A1b 官方 DreaMS 多标准关系坐标确认合同

## 唯一问题

在官方 DreaMS embedding 中，一个标准身份相对其余第三方标准的响应排序，能否在直接 query-candidate cosine 之外，提高结构近邻恢复？

## 冻结输入

- 153 张公共参考谱；
- neg_rp 35 个身份、pos_rp 38 个身份；
- 69 个跨 panel 去重 IK14；
- 官方 `official_embedding_slim.pt` 与既有 DreaMS 架构 checkpoint。

不读取生物样本、不读取表型、不使用 P2b、不使用 E6，不把结构标签输入打分函数。

## 冻结算法

1. 同一身份多张参考谱对另一个身份的 direct score 取最大 official cosine。
2. 对每个 query-candidate 对，从 landmark 集排除 query 与 candidate。
3. 用两者对第三方 landmarks 的 official-cosine 排名一致度定义 `official_profile`。
4. 主候选固定为 `0.5 × official_direct + 0.5 × official_profile`。
5. 不扫描权重、不按结果选 panel、不删除困难身份。

## 评价

结构 Morgan 指纹只在所有谱学分数冻结后打开。每个 query 评价 selected structural similarity、Top-3、NDCG@5 与候选内 Spearman。重复跨 panel IK14 在 bootstrap 中作为同一 cluster。

## 前进门

主候选相对 official direct 的 NDCG@5 IK14-cluster bootstrap 95% CI 下界必须大于零，且 neg_rp、pos_rp 各自均不下降。mass-control 结果单独报告，但不改变主门。

通过只允许进入一个样本级峰算子原型；不构成注释、MSI 身份、Atlas 或生物学发现。
