# 多标准锚定化学坐标 A1：结果、边界与下一步

## 冻结结果

A1 使用 153 张公共参考谱、73 个 panel-specific 身份；跨离子模式去重后是 69 个独立 IK14。比较 query 与 candidate 时，两者均从第三方 landmark 标准集合中排除。结构 Morgan 指纹只用于最终评价，不参与谱学打分。

预注册主比较通过：`multi_anchor_augmented` 相对 `direct_mass_fusion_50_50` 的 NDCG@5 增量为 `+0.03769`，IK14-cluster bootstrap 95% CI `[+0.01953,+0.05804]`；候选内结构 Spearman 增量为 `+0.02844`，CI `[+0.01512,+0.04254]`。正、负离子模式的 NDCG 方向均为正。

## 必须正视的基线问题

`direct_mass_fusion_50_50` 不是这批数据上的最强直接基线：

- neg_rp NDCG@5：direct-multichannel `0.70433`，direct+mass `0.67652`；
- pos_rp NDCG@5：direct-multichannel `0.65379`，direct+mass `0.63774`。

因此原始门虽然按合同通过，却不能单独证明多锚点超过最强直接谱学读出。新增只读审计用冻结逐身份分数比较固定的 `profile_fusion_50_50` 与 `direct_multichannel`，不允许重新拟合、搜索权重或删除样本。A1 是开发屏，A1b 是尚未观察的官方 embedding 确认，因此允许把 A1 中机制一致、表现最好的固定融合配方预注册到 A1b；A1b 内不再选权重。

另一个重要事实是：profile 单独使用并不稳定；可靠正信号来自固定的“直接谱学证据 + 第三方标准响应坐标”融合。这是 A1b 的固定设计依据。

## 当前能够支持的结论

第三方标准响应 profile 含有值得独立复核的关系型结构信号。当前结果不是精确身份恢复、样本注释、代谢物发现或生物学重编程证据。

## 成本受控的 A1b

1. 先运行冻结强基线审计；失败即停止，不编码模型。
2. 若通过，仅编码现有 153 张参考谱的官方 DreaMS embedding。
3. query/candidate 从 landmark 集严格排除。
4. 主比较固定为 `official direct cosine` 对 `0.5 × official direct + 0.5 × official relational profile`；不扫描融合权重。
5. precursor mass 仅作为控制报告，不作为制造增益的主基线。
6. 主比较的 IK14-cluster NDCG CI 下界大于零，且两个 panel 都不下降，才允许进入样本级 retained/shifted/lost/gained 峰算子。

即使 A1b 通过，也只能称为“官方 DreaMS 空间中的多标准关系坐标 headroom 成立”，不能称为新 Atlas 或新代谢物发现。
