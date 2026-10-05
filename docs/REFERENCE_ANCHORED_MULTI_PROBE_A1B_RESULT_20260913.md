# A1b 官方 DreaMS 多标准关系坐标：冻结结果与停止裁决

## 结果

A1b 完整通过了执行与来源校验：153 张参考谱由官方 `official_embedding_slim.pt` 编码，checkpoint SHA256 为 `8928f908606c0bd652c5a4107d3c35102f660622958c225a1f625abe4b1ba245`；共 73 个 panel-specific 身份、69 个独立 IK14。query 和 candidate 均从第三方 landmark 集排除，没有使用结构、表型、样本丰度或 P2b 参与打分。

预注册主比较失败：

- `official_profile_fusion_50_50` 相对 `official_direct` 的 NDCG@5：`-0.00590`，IK14-cluster bootstrap 95% CI `[-0.01941,+0.00822]`；
- neg_rp 面板 NDCG@5：`-0.01268`；
- pos_rp 面板 NDCG@5：`-0.00354`；
- 两个 panel 都下降，因此 panel 非劣门失败。

profile 单独使用显著有害：

- NDCG@5：`-0.07144`，95% CI `[-0.10426,-0.03985]`；
- selected structural similarity：`-0.09719`，95% CI `[-0.14941,-0.05020]`；
- Top-3 structural hit：`-0.12319`，95% CI `[-0.23188,-0.01449]`。

质量控制也不能挽救该方法：`official_multi_anchor_augmented` 相对 `official_mass_50_50` 的 NDCG@5 为 `-0.00845`，95% CI `[-0.02563,+0.00867]`。

唯一正向读数是固定融合的 candidate Spearman `+0.01892`，95% CI `[+0.00126,+0.03591]`。它与 NDCG@5、selected similarity 和 panel 方向不一致，只能解释为候选列表整体或尾部排序的小信号，不能事后替换预注册主终点，也不能支持样本级应用。

## 与 A1 的对账

A1 的原始多通道固定融合相对 `direct_multichannel` 的 NDCG@5 为 `+0.02962`，95% CI `[+0.01317,+0.04767]`；A1b 的官方 embedding 固定融合则为 `-0.00590`。因此不能把 A1 的约 `+2.96pp` 说成 DreaMS embedding 的提升。

目前最合理、但仍属于待验证的机制解释是：A1 的互补性来自显式 fragment、neutral-loss、coverage 等多通道响应；官方 DreaMS 单一 cosine 把这些通道压缩后，其 landmark 排名缺少足够的局部化学分辨率。A1b 只否定“用官方 cosine 响应直接构造多标准关系坐标”，不证明所有多标准测量范式都不可能。

## 裁决

`pass_to_sample_level_peak_operator=false`。按照 A1b 预注册合同：

1. 不进入 retained/shifted/lost/gained 样本级峰算子；
2. 不扫描融合权重；
3. 不按 panel、化学类别或次要指标事后挑选成功子集；
4. 不将 A1 的原始多通道结果包装为 DreaMS embedding 改进、Atlas 或生物学发现。

如果后续继续这一思想，只允许先做一次不训练、低成本的失败归因审计：逐 query 对账 raw-multichannel 与 official-cosine 的 landmark 排名、Top-5 邻居换位和谱图重复数影响。该审计只能决定是否值得重新设计“显式多通道标准响应场”，不能绕过本次停止门。

## 当前科学结论

第三方标准响应的关系信息在显式多通道谱学空间中存在初步结构邻域信号，但该信号没有迁移到官方 DreaMS 单一 cosine 空间。现有证据不支持构建基于官方 DreaMS profile 的样本级化学地图。
