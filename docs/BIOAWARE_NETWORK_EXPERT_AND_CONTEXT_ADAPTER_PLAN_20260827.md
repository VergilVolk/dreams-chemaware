# BioAware 生化网络专家与可选上下文 embedding 方案

日期：2026-08-27

## 1. 决策摘要

生化代谢网络应成为平台的第四类证据，但必须分成两个互不混淆的产物：

1. **近期主线：BioAware 后验专家。** 在一个真实 LC-MS 队列内，将冻结的谱图证据、离子家族、共检出/丰度相关和经审计的生化反应连接成候选图，进行带弃权的候选重排和置信度更新。它不改变 DreaMS embedding。
2. **后期研究线：可选的 context adapter。** 只有 BioAware 后验专家在跨队列、pathway-holdout 和网络诱饵实验中稳定有效后，才把其上下文后验蒸馏进一个可开关的小型 adapter。它输出 contextual embedding，但不取代通用 ChemAware embedding。

不采用“HMDB 邻居一律拉近”的全局微调。代谢反应相邻并不等于结构或 MS2 相似；辅因子、高度节点和通路流行度会把无关谱图聚成一团，并压制罕见或未知代谢物。

## 2. 它与现有三条算法线的关系

```text
原始谱图
  -> Noise/ChemAware shared encoder -> z_chem（通用、样本无关）
  -> P2b 谱学候选专家             -> S_spec（候选级、样本无关）
  -> 双重映射                     -> 化学概念与具体峰证据
  -> BioAware 图专家               -> S_bio（队列/样本上下文相关）
  -> 校准、冲突检测、弃权           -> 最终注释与证据路径

可选研究分支：z_context = normalize(z_chem + A(graph_context))
```

- Noise 微调解决采集变化和错误峰依赖，改变共享 embedding。
- ChemAware 微调用化学概念辅助监督改变共享 embedding。
- P2b 只使用谱学候选证据，是 embedding 后专家。
- BioAware 只引入生物反应和队列共现上下文，是另一个正交专家。
- 表型、疾病组别和差异丰度方向不得进入候选身份评分；只能在注释冻结后用于生物学检验。

## 3. 为什么不是只用 HMDB

HMDB 适合做代谢物词表、结构标识、组织/体液存在证据和文献链接，但不应成为唯一反应图。正式图谱建议组合：

- **HMDB**：代谢物身份、同义词、biospecimen 元数据；
- **Rhea**：经文献人工整理、质量/电荷平衡、带方向和化学计量的生化反应；
- **Reactome**：人类反应、酶、亚细胞区室和通路层级；
- **KEGG/MetaCyc**：补充反应和跨物种覆盖；
- **GNPS/MassBank/DreaMS**：仅作为谱图证据，不能重复充当独立生物网络证据。

所有数据库统一到结构主键：优先 `full InChIKey + charge state`，候选检索可使用 IK14，但网络建图不得仅用名称字符串。每条跨库映射保留来源、版本和歧义状态。

## 4. 图不是普通邻接矩阵，而是有类型的反应超图

### 4.1 节点

1. `feature`：队列中的 MS1 feature；
2. `spectrum`：与 feature 链接的 MS2；
3. `candidate`：候选化学结构；
4. `reaction`：显式反应事件；
5. `enzyme/gene`：催化或运输实体；
6. `pathway`：只用于解释和后验汇总，不直接作为身份标签。

### 4.2 边

- 观测边：feature–spectrum、同位素、加合物、源内碎片、共洗脱、跨样本丰度相关；
- 化学边：候选结构对应、分子式一致、峰级碎片/中性丢失支持；
- 生化边：substrate→reaction→product、enzyme→reaction、reaction→pathway；
- 每条边都有 `source/version/direction/stoichiometry/compartment/confidence`。

必须保留显式 reaction 节点。把多底物反应压成任意代谢物两两相连，会让 ATP、H2O、NAD(P)H、CoA 等辅因子成为超级枢纽，制造大量伪传播。

### 4.3 边过滤

- 辅因子/货币代谢物不作为普通传播种子；
- 反应边要求候选质量差、分子式变化和反应式相容；
- 同一 LC-MS 队列内至少有另一项独立观测支持；
- 跳数衰减，第一版最多一跳；
- 方向只在数据库、组织区室和证据允许时使用，不能从静态丰度推断通量方向。

## 5. BioAware v1：带弃权的后验候选专家

### 5.1 输入

对 query feature `q` 和候选结构 `c`：

- `S_spec(q,c)`：冻结 ChemAware/DreaMS/P2b 的校准谱学得分；
- `S_peak(q,c)`：双重映射中的诊断峰、碎片和中性丢失证据；
- `S_ion(q,c)`：同位素、加合物、离子家族与 RT 证据；
- `S_rxn(q,c)`：候选和高置信种子之间的反应可达性；
- `S_cohort(q,c)`：不使用表型标签的共检出、共洗脱和跨样本相关；
- `U_*`：各证据的不确定度、覆盖和冲突标记。

### 5.2 传播

种子向量 `y` 只能来自标准品或预先冻结的高置信谱学后验。第一版可用带类型权重的 personalized PageRank：

`f = (1-alpha) y + alpha P_rel f`

其中 `P_rel` 不是单一邻接矩阵，而是对反应、离子家族和队列共现边分别归一化后加权；每种关系权重只在训练队列的 OOF 中估计。

### 5.3 最终得分与弃权

`S_final = S_spec + g(q,c) * Delta_bio`

`g` 是部署时可得的风险门：谱学 margin、网络种子覆盖、路径长度、节点度数、关系类型一致性和冲突强度。输出必须区分：

1. 谱学强 + 网络一致；
2. 谱学强 + 网络冲突（保持谱学结果并报警）；
3. 谱学弱 + 网络提供附加支持（可重排但降置信）；
4. 仅网络支持（只列为假设，不升级至 MSI Level 2）；
5. 证据不足（弃权）。

## 6. 可选 BioAware context adapter

### 6.1 为什么它不是当前第一步

网络后验依赖组织、物种、样本类型和队列覆盖。直接写入一个通用 spectrum embedding，会把常见组织、常见通路和数据库热门代谢物的偏见固化在模型中。基础 encoder 还必须服务纯谱库检索和未知代谢物，因此要保留 context-free 版本。

### 6.2 正确的 adapter 形式

候选图先产生上下文向量 `h_ctx`，再用零初始化残差：

`z_ctx = normalize(z_chem + lambda(q) * A(z_chem, h_ctx))`

- `A` 是小型 relation-aware adapter；
- `lambda(q)` 由上下文可靠度决定，缺少网络证据时严格为 0；
- 训练时冻结或低学习率保护 `z_chem`；
- 推理可同时输出 `z_chem` 和 `z_ctx`，不能只保留后者。

### 6.3 生化辅助损失

不使用“反应邻居直接拉近”的无类型对比损失，改用：

- 反应类型/方向预测：`L_relation = CE(r | z_i, z_j)`；
- 质量和结构匹配的反应边排序：真反应对高于 degree/formula-matched decoy；
- 后验蒸馏：仅蒸馏在 OOF 中被 BioAware 专家稳定纠正、且独立谱学证据不冲突的样本；
- preservation：无上下文、未知通路和高置信原始正确样本保持原 embedding。

这使模型学习“二者是什么关系”，而不是错误地学习“二者必须相似”。

## 7. 与化学规则库和双重映射的真正连接

化学规则库不再只是 3,486 个数字命中，而是连接两类变化：

1. **谱内变化**：碎片、中性丢失、同位素模式；
2. **分子间变化**：反应前后分子式/结构变化、酶和反应方向。

对一个被网络支持的候选，解释报告必须给出双路径：

```text
谱图路径：query -> 诊断峰/中性丢失 -> 化学概念 -> candidate
生化路径：高置信 seed -> reaction -> enzyme/pathway -> candidate
```

如果两条路径冲突，不能平均后给出高置信结论，必须保留冲突状态。这是相对于单纯 diffusion 的核心方法学价值。

## 8. 关键创新点与已有方法的差异

MetDNA 已做反应网络递归传播；NetID 已做质量、RT、MS2、离子和生化转换的全局优化；KGMN/MetDNA2、MetDNA3 已做知识层和数据层联合。因此“我们也做图传播”不能构成创新。

本项目可形成的方法学增量是：

1. DreaMS/ChemAware 提供跨仪器的谱图似然；
2. 峰级双重映射给出可干预的碎裂证据；
3. typed reaction hypergraph 提供正交生化上下文；
4. evidence-gated propagation 防止网络覆盖谱学强证据；
5. counterfactual conflict test：删除一个种子、打乱同度节点或反转反应方向后，结论应按预期衰减；
6. annotation 置信、网络路径和生物学结论分层，不循环证明。

## 9. 预注册评估与停止门

### 9.1 数据划分

- 身份、分子式、Murcko scaffold 三层报告；
- 训练/开发/测试按 cohort 分离，不能只在同一谱库拆谱；
- pathway-holdout：整条通路从种子和调参中拿掉；
- leave-one-seed-out：隐藏真种子后恢复；
- 人体队列训练后至少在一个独立队列复核。

### 9.2 对照

- 谱学基线；
- 相同覆盖率的随机候选重排；
- degree-matched 节点诱饵；
- 保持节点度数的网络 rewiring；
- 反应方向打乱；
- 去掉 cohort 共现；
- 去掉生化网络，仅保留离子家族。

### 9.3 主要指标

- Top-1、MRR、候选校准误差；
- 固定错误发现率或固定 abstention 下的注释覆盖；
- corrected / introduced 和风险加权净收益；
- unseen-formula、unseen-scaffold、near-isomer 分层；
- 网络冲突检出率；
- 注释冻结后才做通路/机制恢复。

### 9.4 Go/No-Go

BioAware v1 只有同时满足以下条件才进入产品管线：

1. 至少两个 cohort 的 OOF Top-1 或固定 FDR 覆盖净提高；
2. corrected > introduced，且风险加权净收益公式聚类 CI 下界大于 0；
3. pathway-holdout 不为负；
4. degree-matched/rewired decoy 不复现收益；
5. 网络冲突样本能够弃权，而不是被强制翻转；
6. network-only 注释不被错误升级。

任何一条失败，网络只保留为解释/探索模块，不蒸馏进 embedding。

## 10. 最小闭环实验：优先 MTBLS13729 长链酰基肉碱

这是当前性价比最高的生物学试点，因为已有表型盲类别证据和 MS2 支持，不需要从零找故事。

### B0：知识图谱最小化

- 只收录肉碱穿梭、长链脂酰-CoA、长链酰基肉碱和 beta-氧化的一跳反应；
- 用 Rhea/Reactome 反应式和方向，HMDB 做名称/结构映射；
- 显式记录 CPT1/CPT2/SLC25A20/ACADVL/HADHA-B，但静态丰度不推断通量或具体酶活。

### B1：表型盲种子与候选

- 种子仅来自标准品/高置信谱库或稳定类别 MS2；
- 不使用 Rmu/Rtu/RN 标签、差异倍数或 p 值；
- 将现有 20 个长链酰基肉碱特征和 31 个有 MS2 的优先候选映射到图中。

### B2：留一恢复与诱饵

- 每次隐藏一个已知候选，检验邻居能否把它从同质量候选中恢复；
- 对同度节点 rewiring、质量差匹配的伪反应和方向打乱重复；
- 比较 frozen spectral、network-only、posterior fusion。

### B3：冻结后生物学检验

- 锁定注释与置信度后，再比较 Rmu-RN、Rtu-RN 和交互；
- 报告具体分子种、反应对、候选酶和稳态丰度方向；
- 若缺标准品，结论保持为类别/候选机制，不写 flux reprogramming。

## 11. 第二个试点：MTBLS1905 生物意义错误面板

核苷/嘌呤、氨基酸、含硫、泛酸和色氨酸错误面板用于外部案例，不用于训练或调参。它回答：当谱学 embedding 在有生物意义的近邻上犯错时，BioAware 是否在不看疾病标签的条件下提供正确的反应路径，并在谱学证据冲突时选择弃权。

## 12. 执行顺序

1. 构建版本化的 `compound_id_map` 和 Rhea/Reactome 反应超图；
2. 给现有注释管线增加 feature/candidate/reaction 三层图导出，不改任何模型；
3. 在 MTBLS13729 做 leave-one-seed-out + 网络诱饵；
4. 若通过，做 frozen posterior fusion 和跨队列验证；
5. 接入双重映射，输出谱峰路径和生化路径；
6. 只有上述步骤通过，才训练可选 context adapter；
7. Noise/ChemAware shared embedding 的训练继续独立推进，BioAware 不替代它。

## 13. 文献锚点

- MetDNA：Nature Communications 2019，https://www.nature.com/articles/s41467-019-09550-x
- NetID：Nature Methods 2021，https://www.nature.com/articles/s41592-021-01303-3
- KGMN / MetDNA2：Nature Communications 2022，https://www.nature.com/articles/s41467-022-34537-6
- MetDNA3：Nature Communications 2025，https://www.nature.com/articles/s41467-025-63536-6
- Rhea reaction curation/direction：https://www.rhea-db.org/help/reaction-curation
- Reactome data model/download：https://reactome.org/documentation/data-model/ ，https://reactome.org/download-data/

## 最终裁决

生物网络不是 Noise 微调的替代品，也不应成为把代谢物强行聚类的新借口。它的最佳位置是：先作为可审计、可弃权、样本上下文相关的候选专家，解决“谱图证据不足但队列中存在独立生化证据”的问题；通过严格外部验证后，再蒸馏为可选 context adapter。这样既能提高非靶向代谢组学注释和机制连贯性，又不会污染面向未知化学空间的通用 embedding。
