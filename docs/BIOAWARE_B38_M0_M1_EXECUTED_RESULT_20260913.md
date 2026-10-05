# BioAware B38-M0/M1 执行结果与机制裁决（2026-09-13）

## 结论先行

B37 的 `network_member + known_log_degree` 目录拓扑先验在相同的六域开发协议上
仍能精确复现 `+5.9302 pp`，但显式一步反应路径并未在其上产生增量。加入真实直接
路径后总增益降至 `+5.0000 pp`，相对 topology 为 `-0.9302 pp`；加入更丰富的事件
汇总后相对 topology 为 `-1.9767 pp`。因此当前路径动作不得进入候选上下文模型或
共享 embedding 微调。

## M0：显式事件账本

- 固定负离子评价图：860 queries、3,314 candidate rows。
- 完整嵌套训练图：1,738 queries、6,695 candidates、38,999
  query-candidate-seed contexts。
- 全训练图显式事件：15,943；负离子评价图显式事件：8,382。
- 506/860 queries 至少存在一条直接路径；494/860 在至少一个 seed context 中存在
  候选间路径差异。
- 内部设计对每个 held-out identity 有七个 rotation；实现保留全部七个，禁止任取其一。
- 主路径可识别性门通过。
- 联合数据层门未通过：路径候选的谱学配对控制覆盖 11.4%，共丰度控制覆盖 85.4%；
  外部 ST/KGMN 不具备旧 B3/B9 控制缓存。缺失被显式报告，没有按零证据填充。

## M1：容量匹配的路径增量检验

环境固定为 scikit-learn 1.7.2，并逐 query 精确复现 B37 topology：

| 方法 | 相对 DreaMS Recall@1 | corrected / introduced |
|---|---:|---:|
| catalog topology | +5.9302 pp | 55 / 4 |
| topology + explicit direct-path fraction | +5.0000 pp | 46 / 3 |
| topology + enriched explicit-event summary | +3.9535 pp | 41 / 7 |

真实直接路径相对 topology：

- Recall@1：`-0.9302 pp`；
- left-better / right-better：3 / 11；
- identity-cluster 95% CI：`[-2.164, +0.110] pp`；
- formula-cluster 95% CI：`[-2.176, +0.118] pp`；
- ST001154：`-6.6667 pp`；Mouse liver：`+1.1364 pp`；其余四域为 0；
- 相对 20 个 degree-preserving rewires 的经验上尾 p：`0.9048`；
- enriched event summary 相对 topology：`-1.9767 pp`，identity/formula CI 均全负。

## 可以和不可以声称什么

可以声称：

1. B37 的稳定开发增益主要是代谢网络目录成员与度先验；
2. 当前粗粒度直接路径特征没有提供 topology 之外的反应特异增量；
3. 20 个度保持重连证明，单纯增加一个图支持特征产生的变化不足以归因于真实反应边；
4. ST001154 暴露了路径先验的跨域负迁移风险。

不可以声称：

1. +5.93 pp 是反应传播或生化机制增益；
2. 当前结果改变了 DreaMS embedding；
3. BioAware 已达到 SOTA 或完成独立外部确认；
4. “更多路径特征”可以修复问题——富集汇总反而显著更差。

## 下一步冻结决策

`pass_to_b38_m2_path_set_model = false`。不得直接训练原计划的路径集合模型，也不得把
这批路径蒸馏进 clean-spectrum encoder。

下一次高性价比检验必须重新定义可验证动作，而不是继续加总路径数：只保留方向可确认的
Rhea 事件，并要求真实边同时得到 matched-non-neighbour 谱学或共丰度支持；对每个事件
保留 seed、reaction、direction 与证据来源。只有该严格事件子集在 topology 之上取得
正的跨域、cluster-CI 和 degree-rewire 增量，才允许训练上下文候选 embedding。目录
membership/degree 继续作为独立工程基线，不得注入共享 clean-spectrum embedding。

## 本地可复核工件

- `data/validation/bioaware_b38_m0_localcheck_20260913_v4/report.json`
- `data/validation/bioaware_b38_m1_localcheck_v2_20260913/report.json`
- M0/M1 的 event、candidate、transition、selection 与 provenance 哈希均在各 report 中。

