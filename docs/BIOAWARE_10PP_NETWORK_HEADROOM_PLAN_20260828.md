# BioAware 网络专家：创新边界、+10pp 头寸与下一阶段方案（2026-08-28）

## 1. 结论先行

当前 BioAware **有方法学创新潜力，但尚未达到 SOTA，也不是 MetDNA3 的完整复现**。

已经复现或借鉴的部分：

- MetDNA/MetDNA3 的代谢反应网络递归传播；
- MetDNA3 的 MS1 预映射、feature-edge raw-MS2 约束思想；
- MetDNA3 的 SMN 对照阈值（Morgan/Dice ≥ 0.4）与 HILIC RT 相对误差 30%；
- NetID 类方法的“局部谱学证据不足时，需要全局一致性约束”思想。

属于本项目自己的组合与约束：

- 将官方 DreaMS embedding 检索分数作为候选 unary evidence；
- 将 reaction path、observed MS1 node、raw-MS2 edge 分开保存，不把网络可达性直接冒充身份标签；
- query truth identity 从每个 seed fold 中剔除，7 次 held-out rotation 多数投票；
- 唯一候选、严格优势、冲突弃权和 corrected/introduced 非对称风险门；
- 未来将 DreaMS unary 与样本矩阵、reaction hyperedge、ion-form group 放入同一个候选分配问题，而不是简单做网络扩散。

因此，真正可形成创新的方法不是“又一个一跳网络加分器”，而是：

> **DreaMS-guided, evidence-factorized, risk-controlled global metabolite assignment**：把谱图 embedding、反应网络、真实 feature-edge MS2、RT、离子形式与跨样本强度关系作为可审计的独立因子，在全局候选分配中联合求解，并允许弃权。

## 2. 当前开发结果

固定任务为 117 个 HILIC query，官方 DreaMS Recall@1 = 95/117 = 81.20%，共 22 个错误。提高 10 个百分点至少需要净修正 12 个 query。

| 证据臂 | 结果 | 判定 |
|---|---:|---|
| known MRN + observed MS1 + full raw-MS2 path | 3 corrected / 0 introduced，+2.56pp | 正向但仅 2 个身份，CI 跨 0 |
| SMN（Dice ≥ 0.4） | 4 corrected / 8 introduced，−3.42pp | 网络过密，保留为负对照 |
| identity-isolated RT（30% 窗口） | 0 corrected / 3 introduced，−2.56pp | RT 模型 R²=0.644，但不能单独做 override |
| reported + predicted eMRN raw-path | 5 corrected / 6 introduced，−0.85pp | 比 reported-only 多 1 个独立修正，仍未过门 |

当前四臂的实际修正并集为 7 个错误；把 SMN/RT 的 truth-headroom 也按 oracle 计入，乐观并集仍只有 11 个，低于 +10pp 所需的 12 个。**因此继续扫网络权重在数学上也无法达到目标。**

## 3. 为什么不能称 SOTA

1. 目前最好的 +2.56pp 是 consumed development ablation，公式 bootstrap CI 下界为 0；
2. 只有两个独立修正身份，不能证明跨化学空间泛化；
3. RP 仍封存，尚无一次性外部验证；
4. MetDNA3 报告的是网络注释覆盖与 Top-N correct rate，我们当前报告的是 strict candidate Recall@1，任务定义不可直接横比；
5. 当前版本缺少 MetDNA3 完整的 RT descriptor table、全局峰相关/离子形式层和最终 redundancy removal；
6. 未与 NetID、MetDNA3、SIRIUS/CSI:FingerID 等在同一个锁定协议上比较。

## 4. 为什么不能承诺 +10pp

“必须 +10pp”可以作为工程目标，不能作为预先保证。当前数据已证明：

- SMN 增大覆盖也同步增大错误；
- RT 对总体保留时间有预测力，但对局部异构体候选不够特异；
- predicted reaction edges 增加路径，但 raw-MS2 bottleneck 仍会支持错误候选；
- 所以缺的不是更多边，而是**候选特异、样本级的全局约束**。

只有在新证据层对当前未解决错误新增至少 3 个独立 headroom，且正确 query 风险小于一半修正头寸时，+10pp 才重新成为数学可达目标。

## 5. 下一阶段优先级

### G1：样本矩阵与 ion-form 全局图（最高优先级）

在 MTBLS13729 的 60 样本 MS1 feature matrix 上构建：

- 同位素、加合物、中性丢失、in-source fragment 的同 RT feature group；
- 跨样本 abundance correlation，配对病人设计仅用于下游生物学，不进入身份标签；
- 每个 feature 的候选集合与 DreaMS unary score；
- reaction/structure/raw-MS2 edge 只作为 pairwise factor；
- 货币代谢物、高度数 hub、冲突路径显式降权或弃权。

先做 outcome-blind graph calibration；无标准答案的 MTBLS13729 不用于选择超参数。性能选择必须转到有 Level-1 标准或可锁定真值的外部样本矩阵。

### G2：全局候选分配，而非局部加权

定义每个 feature 的候选变量，目标函数由以下项组成：

- unary：DreaMS、质量误差、RT；
- pairwise：reaction probability、raw-MS2 edge、ion-form/coelution、sample correlation；
- constraints：同一 feature 只能选一个候选；同一离子组的 neutral identity 必须一致；高冲突时 abstain；
- safety：corrected − 2×introduced、公式隔离 OOF、degree-preserving decoy、随机网络对照。

优先实现可解释的 ILP/因子图基线，再决定是否用 GNN。这样可以明确每一条证据究竟修正或引入了什么错误。

### G3：候选特异碎裂模型

对全局图仍无证据的错误，增加候选结构→碎片/中性丢失的结构特异 likelihood。化学规则库只作为碎片证据，不作为身份标签。该层必须在 formula/scaffold 隔离下验证，避免记忆常见分子式。

### G4：冻结与外部裁决

开发门：

1. 新层独立增加 ≥3 个 error query；
2. corrected > introduced，corrected − 2×introduced > 0；
3. formula-cluster CI 下界 >0；
4. 每个公式 fold 非负；
5. 随机网络与 degree-preserving decoy 不得复现；
6. 联合 optimistic headroom ≥14（比 +10pp 所需 12 留出风险余量）。

全部通过后才冻结工件并开启 RP。外部结果未出前，禁止使用“SOTA”表述。

## 6. 主要可复现产物

- `data/validation/bioaware_metdna3_candidate_edge_decision_v1/`
- `data/validation/bioaware_metdna3_smn_headroom_v1/`
- `data/validation/bioaware_metdna3_rt_headroom_v1/`
- `data/validation/bioaware_metdna3_candidate_edge_ms2_step1_v1/`
- `data/validation/bioaware_metdna3_predicted_edge_increment_v1/`
- `data/validation/bioaware_10pp_headroom_v1/`

方法依据：[MetDNA3](https://www.nature.com/articles/s41467-025-63536-6)、[MetDNA2](https://www.nature.com/articles/s41467-022-34537-6)、[NetID](https://doi.org/10.1038/s41592-021-01303-3)、[MrnAnnoAlgo3 官方源码](https://github.com/ZhuMetLab/MrnAnnoAlgo3)。

## 7. 2026-08-29：G1/G2 实施更新

G1 已实现为严格区分 discovery pilot 与 formal EIC 的全 MS1 峰网络。非正式本地压力测试覆盖 neg-RP 3,798 个峰和 pos-RP 13,155 个峰，分别得到 129 和 1,479 条丰度相关支持边；正式运行必须先完成统一 targeted-EIC 重定量，并锁定每个样本的提取参数。

G2 新增两类候选特异证据，均不使用 P2b：

- 公式隔离、身份 held-out 的 DreaMS embedding→Morgan 指纹解码器，在原先 11 个完全未解决错误中新增 2 个独立头寸（3-Pyridol、PC(16:0/20:4)）；
- 335 条主规则库形成的候选参考谱规则似然，在固定三种读法中覆盖 1-Methyladenine、7-Methylguanine、3-Pyridol 和 SDMA。

两类新证据对未解决错误的并集为 5；与旧模块实际并集 8 相加，当前 consumed-development 的实际可触达头寸为 13，首次超过 +10pp 所需的 12。但是解码器/规则单独使用都会引入大量错误，所以只能说明“数学可达”，不能称为模型提升。下一阶段必须用全局候选分配、证据一致性和弃权，把这些头寸转成 corrected − 2×introduced > 0 的安全增益。

## 8. G3 融合实现审计（2026-08-29）

G3-v1 在 117 个查询、88 个分子式的 nested formula-OOF 中完全弃权（0 corrected / 0 introduced）。诊断显示这不是 headroom 消失，而是量纲错误：未经校准的 decoder、rule、reaction、SMN 和 RT 原始值直接叠加到 DreaMS cosine，L2 后各残差权重仅约 0.01–0.06，五个外层折中的四个没有改变任何 Top-1。该版本作为安全负结果归档，不进入 RP。

G3-v2 的预注册修正是候选组内证据归一化与机制级共识：每一独立证据族只保留一个 0–1 相对票，多个 rule 特征先在族内合并，避免规则数量造成重复计票；公式隔离 OOF 只学习非负族权重；只有至少两个独立证据族共同支持且内层 OOF 风险净收益为正时才允许覆盖 DreaMS，否则弃权。该修正针对已定位的量纲问题，不改变 P2b 禁用、表型禁用和 RP 封存约束。

实测 G3-v2 为 2 corrected / 0 introduced，Recall@1 由 0.8120 提升至 0.8291（+1.71pp），但 formula-cluster bootstrap 95% CI 为 [0, 0.0446]，下界未超过零。因此它是当前最安全的 BioAware 开发正信号，不是显著提升，也不能打开 RP 或宣称 SOTA。

G3-v3 进一步测试了同分子式其他查询的 leave-one-query-out 软共识；不施加“同分子式同身份”或候选唯一性硬约束。实测 0 corrected / 0 introduced，说明当前 22 个多查询分子式组不足以稳定学习可迁移的 peer residual。该分支归档。下一步不再在 117-query 已消耗开发集上扫权重，而应扩大候选条件化碎裂监督：利用候选结构与参考谱学习“谱图是否支持这个具体候选”，再以严格结构/分子式隔离评估。当前 13 个联合 headroom 仍只是 oracle 上界。
