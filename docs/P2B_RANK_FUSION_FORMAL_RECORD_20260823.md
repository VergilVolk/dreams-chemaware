# P2b 局部候选排序器：正式结果、实现与使用边界

**冻结日期**：2026-08-23  
**项目角色**：最终算法的强制保底模块；不等同于 DreaMS 权重微调，也不替代 ChemAware 主线。

## 1. 科学定位

P2b 解决的是一个明确而局部的问题：在前体质量误差不超过 10 ppm、加合物一致的候选分子集合中，DreaMS 全局表征给出候选后，利用原始碎片峰和中性丢失证据重新排列候选。

它保留 DreaMS 对谱图的全局表征能力，并补充 DreaMS 单一余弦相似度未充分利用的局部谱峰关系。它不是新的 DreaMS checkpoint，也不是“噪声微调成功”的证据。论文中应称为 **frozen local rank fusion / 局部候选证据融合排序器**。

## 2. 输入、候选协议与输出

### 输入

- 查询 MS/MS 谱图；
- strict-10 ppm、同加合物候选库；
- 官方 DreaMS embedding 与候选相似度；
- 查询谱图与候选谱图的原始峰匹配和中性丢失匹配统计。

### 候选聚合

- 排除查询谱图本身；
- 同一候选分子的多张谱图取该评分的最大值；
- 评价单位为候选分子身份（IK14），不是谱图行；
- 数据划分和模型选择在分子式层面隔离。

### 输出

- 候选分子的局部融合分数；
- 更新后的候选顺序；
- DreaMS、碎片峰相似度、熵相似度和中性丢失相似度，供模块二解释和冲突审查。

## 3. 冻结实现

冻结配置：

```text
score = 0.10 * DreaMS_similarity
      + 0.00 * sqrt_cosine
      + 0.10 * entropy_similarity
      + 0.80 * neutral_loss_sqrt_cosine
```

- normalization：`absolute`
- minimum support：`1`
- minimum advantage：`0.0`
- 参与候选排序的四个预注册特征：
  1. `dreams_similarity`
  2. `sqrt_cosine`
  3. `entropy_similarity`
  4. `neutral_loss_sqrt_cosine`

虽然 sqrt cosine 的冻结权重为 0，它仍属于预注册搜索空间和消融对象。当前冻结模型中真正起主要作用的是中性丢失相似度；DreaMS 和 entropy 提供较小修正。

## 4. 开发阶段证据：公式隔离嵌套 OOF

评价集包含 5,037 个查询、2,522 个分子身份、1,082 个分子式和 2,094 个 near 查询。所有权重选择都在五折、分子式隔离的嵌套 OOF 中完成；封存 P3 未参与训练和选择。

| 指标 | 官方 DreaMS | P2b | 变化 |
|---|---:|---:|---:|
| Recall@1 | 0.8606 | 0.8997 | **+3.91 pp** |
| MRR | 0.9161 | 0.9411 | **+2.50 pp** |
| near Recall@1 | 0.7612 | 0.8195 | **+5.83 pp** |
| 修正 / 新增错误 | — | 280 / 83 | 净修正 197 |

Recall@1 的分子式聚类 bootstrap 95% CI 为 **[+2.91, +4.87] pp**，五个 outer fold 的总体和 near 方向均非负。

这组结果证明：局部谱峰和中性丢失证据在开发协议下含有可复现的候选排序信息。它不能被表述为外部盲测提升约 4 个百分点。

## 5. 封存 P3 证据

### P3-main-real-pristine（主结论面板，n=3,000）

| 指标 | 官方 DreaMS | P2b | 变化 |
|---|---:|---:|---:|
| Recall@1 | 0.8793 | 0.8900 | **+1.07 pp** |
| MRR | 0.9304 | 0.9361 | **+0.57 pp** |
| macro query AUC | 0.9209 | 0.9265 | +0.56 pp |
| 修正 / 新增错误 | — | 89 / 57 | 净修正 32 |

- Recall@1 分子式聚类 bootstrap 95% CI：**[+0.24, +1.89] pp**；
- McNemar exact p = **0.0101**；
- MRR 的分子式聚类 CI 为正；
- AUC 点估计为正，但 CI 跨 0，不能称显著。

因此，可公开声称：**冻结 P2b 在封存主面板上显著提高了 Recall@1，并减少了约 8.8% 的 DreaMS Top-1 错误（净修正 32 / 原错误 362）**。

### near-core（压力测试，n=496）

| 指标 | 官方 DreaMS | P2b | 变化 |
|---|---:|---:|---:|
| Recall@1 | 0.4879 | 0.4456 | **-4.23 pp** |
| MRR | 0.6990 | 0.6704 | -2.86 pp |
| 修正 / 新增错误 | — | 20 / 41 | 净新增 21 |

Recall@1 的分子式聚类 CI 完全为负。因此，**冻结 P2b 不能直接用于 MCES 0–2 的极近异构体候选集合**。这是当前算法的硬边界。

## 6. 消融与机制判断

- neutral-loss cosine 单独使用时，开发 OOF Recall@1 提升约 **+3.91 pp**；
- 完整 P2b 开发 OOF 提升约 **+3.91 pp**，与 NL-only 的 Top-1 净差为 0；
- 在 P3 主面板，完整 P2b 相对 NL-only 仍提高 Recall@1 **+0.97 pp**，CI 为正；
- 说明中性丢失是主要信号，DreaMS/entropy 的增量在开发 OOF 很小，但在封存主面板上提供了可检测的泛化收益。

因此，论文不应写成“复杂多特征模型取得提升”，而应写成：**DreaMS 全局表征与中性丢失主导的局部证据具有互补性；少量熵相似度修正提升了封存主面板上的稳健性。**

## 7. near 安全防线

封存结果之后的候选歧义审计表明：若候选集合包含 MCES 0–2 的 near 候选，则回退到 DreaMS，可在 near-core 上完全阻止 P2b 的新增错误；但该规则使用了结构标签，只能作为 consumed-P3 的机制诊断，不能作为当前无标签部署模型或新的封存测试结论。

后续可部署安全门必须仅依赖推理时可得信息，例如：

- DreaMS 与 P2b 排名是否冲突；
- Top-1/Top-2 分差；
- 谱峰和中性丢失证据的支持/冲突；
- 模块二给出的解释置信度。

任何新门控都必须重新锁定开发集和测试集，不能在已消费 P3 上选择阈值。

## 8. 最终系统中的固定位置

```text
实验谱图
  -> ChemAware/DreaMS 表征与候选生成
  -> P2b 局部候选证据融合排序（强制保留）
  -> 化学证据解释与置信度
  -> 低置信/冲突样本回传峰级微调
```

P2b 的三项长期作用：

1. 提供已经通过封存主面板验证的性能保底；
2. 输出 DreaMS 与局部化学证据冲突的真实错误样本，作为后续噪声微调样本来源；
3. 为模块二提供可追溯的碎片峰和中性丢失证据。

## 9. 可复现工件

### 核心代码

- `tasks/build_g8r_p2_listwise_cache.py`
- `tasks/audit_g8r_p2_cache_headroom.py`
- `tasks/train_g8r_p2b_rank_fusion.py`
- `tasks/g8r_p2_rank_fusion_core.py`
- `tasks/audit_g8r_p2b_ablation.py`
- `tasks/eval_g8r_p2b_on_sealed_p3.py`
- `tasks/audit_g8r_p3_candidate_ambiguity_router.py`

### 冻结结果

- `data/validation/g8r_p2b_rank_fusion.json`
- `data/validation/g8r_p2b_locked_ablation.json`
- `data/validation/g8r_p2b_p3_final.json`
- `data/validation/g8r_p3_candidate_ambiguity_audit.json`

### 关键哈希

- P2 cache SHA256：`b103b2574593a629a63a3b24d6d801260271a1f856c6fb712d3f574ef7245731`
- P2 cache audit SHA256：`b562c7e5748ef6c04c593d0463969cc61b1c3de76e592f984dc36a65fe7987ae`
- selection report SHA256：`af669c313403f5320d33cc4b26cc61e853e5a89b6a5a4837eb72bf8204bbc4e1`
- frozen artifact SHA256：`3ed4af01661556be198058db4dacdfb4f7e1893928586231cc02476ff28046fc`
- P3 lock summary SHA256：`97c8e336e9a39caa01a75d537dfa85e0c93e8dcccf09903fb92a534b2251f7cb`
- HDF5 SHA256：`ccda2c4114d9b21413977df03376ca0fc097956a7fa304b861a3154a2b81e64f`
- reference library SHA256：`cad6ee05aee1ccfb8d4923b2405795a5538cb1876247b0d098db033343361385`

## 10. 论文表述边界

### 可以写

- “在分子式隔离的嵌套 OOF 开发评价中，P2b 将 Recall@1 提高 3.91 个百分点。”
- “在封存的 3,000-query 主面板上，冻结 P2b 将 Recall@1 提高 1.07 个百分点，分子式聚类 bootstrap CI 为正，McNemar p=0.010。”
- “P2b 的增益主要来自中性丢失证据，并在主面板上表现出与 DreaMS 的互补性。”

### 不可以写

- “DreaMS 权重被提高了 4 个百分点。”
- “P2b 已解决近异构体检索。”
- “P2b 在所有任务上达到 SOTA。”
- 把 consumed-P3 的结构标签回退规则写成正式部署性能。

## 11. 当前裁决

P2b 已经达到“必须保留的性能与证据模块”标准，但没有达到“独立主创新”或“近异构体解决方案”标准。后续优化不得覆盖该冻结工件；新策略以增量模块形式开发，只有在新的封存测试上同时满足主面板非劣、near 面板改善和新增错误受控，才能替换 P2b。
