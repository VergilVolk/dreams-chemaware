# BioAware v1 实现、外部试点与阶段裁决

日期：2026-08-27

## 一、不可混淆的模块边界

BioAware v1 是 **冻结 embedding 之后的生化网络证据专家**。它不训练 DreaMS，不改变 Noise/ChemAware embedding，也不替代 P2b。推理输入是候选级谱学得分与表型盲高置信种子，输出是候选网络支持、反应路径、冲突/弃权状态和保守重排结果。

疾病标签、差异倍数、p 值、通路富集结果不允许进入身份评分。注释冻结后才能进行 Rmu/Rtu/RN 生物学比较，防止用疾病假设选择身份、再用该身份证明疾病假设。

## 二、已完成的工件

1. `annotation/bioaware.py`
   - 显式 `compound -> reaction -> compound` 一跳反应超图；
   - 默认无方向传播，避免把 Rhea 书写方向误当生理通量方向；
   - 货币代谢物和高度数种子过滤；
   - 反应规模与节点度数降权；
   - leave-query-out 与 leave-truth-identity-out；
   - 谱学高置信冲突时保持 DreaMS 并报警；
   - 无网络证据时严格回退 DreaMS；
   - 每次干预保留 seed、reaction 和 contribution 路径。
2. `tasks/build_bioaware_rhea_cache.py` 与 `tasks/validate_bioaware_rhea_cache.py`
   - 从 Rhea 官方 TSV 建立版本化结构反应缓存；
   - 结构键使用 full InChIKey/IK14，不用名称字符串连接；
   - 保存输入、脚本和输出 SHA256。
3. `tasks/evaluate_bioaware_expert.py`
   - 唯一 Top-1，平分计错；
   - corrected/introduced、McNemar、formula/query cluster bootstrap；
   - 度数与反应侧规模保持的网络 rewiring 诱饵；
   - 正式模式至少 10 个网络诱饵，少于 10 必须显式标记 development；
   - 种子输入、阈值、度数、货币节点和最终合格数审计。
4. 数据接入
   - `tasks/prepare_mtbls1905_bioaware_benchmark.py`：真实已知目标外部基准；
   - `tasks/prepare_mtbls13729_bioaware_pilot.py`：表型盲 Level 2a 谱学参考试点；
   - 网络外候选全部保留并获得零网络支持，严禁通过删除候选虚增性能。
5. 集群入口
   - `tasks/run_bioaware_rhea_cache.sbatch`；
   - `bioaware_rhea_cache_20260827.zip` 与 `tasks/run_bioaware_rhea_offline_install.sbatch`：计算节点无外网时使用；
   - `tasks/run_bioaware_mtbls13729.sbatch`。

## 三、Rhea 缓存实测

- 反应：17,656；
- participant rows：78,843；
- 唯一 IK14：10,152；
- 因通配/非法结构跳过 participant：9,675；
- 无效反应：566；
- participants SHA256：`ab8ecb5515c35c042d055bf0cac7035b9f7c81771e3214cec4959d6f7001b556`；
- reactions SHA256：`93aefc03df8791e51b8954920ea16b9805a08ad16679d703542277f506ed567c`。

缓存通过两侧反应、结构键、重复行、度数复现、方向契约和哈希验证。

## 四、MTBLS1905 真实外部已知目标试点

### 4.1 输入与基线复现

- 查询谱：36；
- 已知目标：18；
- 候选行：358；
- 官方 DreaMS Top-1：0.750，精确复现已有外部报告；
- 真值 Rhea 覆盖：100%；
- 全候选 Rhea 覆盖：15.6%。

候选覆盖低意味着网络只能作为稀疏附加证据，不能成为主评分器；网络外候选仍参加完整排名。

### 4.2 评价型 leave-target-out 上限诊断

将其余已发表目标作为评价型种子，并对每个 query 删除其真值身份的全部种子：

- DreaMS：0.750；
- BioAware：0.778；
- corrected / introduced：1 / 0；
- 干预率：2.78%；
- formula-cluster bootstrap 95% CI：0 至 0.0968；
- 10 个度数保持网络诱饵均为 0 增益；
- 总门：FAIL，因为效应只有一个 query，CI 下界为 0，McNemar 不显著。

这只能说明“真实网络可能提供一个正确修正案例”，不能称为性能提升，更不能称为部署结果。

### 4.3 可部署自动种子

最初只从 36 个目标 query 取种子且生成阈值 0.70、评价阈值 0.80，导致 8 条输入全部被过滤。该输入契约不一致已经修复。

修复后从全部 8,600 条表型盲 QC MS/MS 建种子，冻结条件为 strict 10 ppm、DreaMS Top-1 不低于 0.80、Top1-Top2 margin 不低于 0.05、结构进入 Rhea：

- 合格种子只有 1 条、1 个化合物；
- 可用 query-candidate 网络路径为 0；
- DreaMS/BioAware 均为 0.750；
- 无修正、无新增。

因此当前可部署失败的首要原因是 **高置信独立种子饥饿**，不是已经证明反应网络无用，也不是通过放宽阈值就可以补救。

## 五、阶段裁决

### 已成立

1. 完整、可复现、可回退、可解释的 BioAware v1 工程链已经建立；
2. Rhea 真实网络与度数保持伪网络可以严格对照；
3. MTBLS1905 官方 DreaMS 基线被精确复现；
4. 网络不会覆盖强谱学证据，首轮未引入错误；
5. 一个 leave-target-out query 被真实网络修正，而伪网络未复现。

### 尚未成立

1. BioAware 显著提升外部注释准确率；
2. BioAware 提升新增注释数量；
3. BioAware 改善 MTBLS13729 的长链酰基肉碱注释；
4. 网络证据可以蒸馏进 contextual adapter；
5. 任一静态队列结果可证明通量或酶活性改变。

所以 BioAware v1 当前状态是 **系统完成、有效性门未过**。不得写成算法成功，更不得把 +1/36 包装成显著结果。

## 六、下一步最小但关键的增量

当前不应增加图深度或直接上 GNN。首先扩大独立、表型盲、可校准种子：

1. 用 cohort 内 MS1 同位素/加合物/离子家族把同一结构的多条证据合并，但不能把同一谱重复计为独立证据；
2. 用冻结的 DreaMS/P2b/峰级证据生成 Level 2a 种子，要求跨谱一致、候选 margin 与校准后错误率共同过门；
3. 在 MTBLS13729 中优先限定肉碱穿梭与长链脂酰基反应的一跳子图，降低货币节点和无关路径；
4. 每次隐藏一个种子身份，比较真实网络、度数保持 rewiring、质量/分子式匹配伪反应；
5. 只有 corrected > introduced、公式聚类 CI 下界大于 0、伪网络不能复现，才冻结 BioAware 评分；
6. 注释冻结后再检验 Rmu/Rtu/RN 丰度与反应邻接的一致性。

如果 MTBLS13729 仍因种子不足而无有效路径，BioAware 保留为路径解释器，不进入身份重排；下一步应补谱库/标准品证据，而不是把网络权重调大。

## 七、创新性边界

“在代谢网络上扩散”本身不是创新，MetDNA、NetID、KGMN/MetDNA2 和 MetDNA3 已覆盖这一大方向。本项目只有在以下联合证据成立后才具有明确方法学增量：

1. DreaMS/ChemAware 的谱学似然；
2. 峰级双重映射给出的具体碎裂证据；
3. typed reaction hypergraph 的正交生化证据；
4. 冲突感知、可弃权的证据门；
5. seed deletion、degree-preserving rewiring 和路径级解释共同证明网络贡献不是数据库流行度伪影。

当前已实现第 3、4、5 项的工程骨架；第 1、2 项的正式融合以及跨队列有效性仍待验证。
