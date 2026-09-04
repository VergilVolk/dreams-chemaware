# BioAware 双轨算法创新方案（2026-08-30）

## 1. 不可混淆的定义

### Track A：BioAware-Context 候选重排器

输入包括 query 谱图、候选身份、同一样本的观测 feature/种子和反应网络。输出是候选上下文化分数。该模块依赖候选集合与样本上下文，因此是 **learned contextual reranker**，不是通用谱图 embedding。

### Track B：BioChem-Embed 共享谱图表示

训练阶段使用真实错误、采集变化、typed biochemical relation 和峰级反事实作为监督；推理阶段只输入一张原始谱图，由一个共享编码器输出新 embedding。该模块才属于 DreaMS embedding 微调。

P2b、Track A 和 Track B 不得互相替代性能主张。最终系统可按 `BioChem-Embed -> P2b -> BioAware-Context` 串联，但每层必须先独立通过锁定评价。

---

## 2. Track A：把 BioAware 重排器做得更精准

## 2.1 当前错误模型

BioAware v1 失败并非只因权重不合适，而是候选证据定义错误：

1. 一条一跳路径可同时支持多个同式候选；
2. 枢纽/货币代谢物制造大量非特异路径；
3. reported、predicted 和 SMN 网络证据嵌套，不能重复投票；
4. 图外、无种子、缺失 feature 和明确冲突被混成零；
5. 网络支持没有要求反应方向、必要参与物和实验 transformation 同时成立；
6. query 内 min-max 放大微小噪声；
7. 固定门控不能预测何时会引入新错误。

## 2.2 新分数必须是受约束残差

保持官方谱学分数为锚：

\[
S(q,c)=S_0(q,c)+g(q,c,O)\,\Delta_{ctx}(q,c,O)
\]

- `S0`：冻结并校准的 DreaMS/新 embedding 谱学 unary；
- `Delta_ctx`：候选特异的上下文 log-likelihood residual；
- `g`：风险门控，只决定是否允许 residual 生效；
- 上下文不可覆盖谱学证据强、网络证据不完整的候选。

不得直接学习一个任意总分，否则模型会重新学习 DreaMS 并在小样本上过拟合。

## 2.3 候选特异上下文因子

每个候选必须显式区分以下量：

1. `available`：候选和种子是否可映射到网络；
2. `seed_reliability`：种子的谱学/RT/标准品等级；
3. `reaction_type`：氧化、还原、甲基化、酰化等 typed relation；
4. `direction`：候选是底物还是产物；
5. `hyperedge_completeness`：必要参与物中有多少被真实 feature 支持；
6. `feature_edge`：m/z transformation、raw-MS2 transformation、共洗脱/峰形；
7. `hub_penalty`：种子和反应的度、currency 状态；
8. `conflict`：网络支持与谱学、RT、离子家族的显式冲突；
9. `missingness`：图外、未观测和真实反证分别编码。

Rhea 一跳不再是最终分数，而是生成 typed candidate--seed hyperedge 的来源。

## 2.4 模型结构

首版采用可审计的 hierarchical residual，不直接上无约束 GNN：

1. 对每条 seed--reaction--candidate path 编码；
2. path 内以反应完整性和 seed reliability 做乘性门控；
3. 同一 evidence family 内用保守聚合，禁止 `max`；
4. network、spectral transformation、physicochemical 三个依赖组分别校准；
5. 候选组内用 listwise softmax 排序；
6. 加入 `unknown/abstain` 候选和 selective-risk head。

只有该结构在大规模开发集上通过后，才比较 Set Transformer/GNN；复杂模型必须相对同输入的线性/树模型证明增量。

## 2.5 损失与安全约束

\[
L_A=L_{listwise}+\lambda_I L_{introduced}+\lambda_P L_{preserve}+\lambda_C L_{calibration}
\]

- `L_listwise`：组内真候选排序；
- `L_introduced`：官方正确而上下文改错的样本加大代价；
- `L_preserve`：强 DreaMS margin 时 residual 接近零；
- `L_calibration`：输出可解释为风险/概率；
- 选择指标为固定 FDR 下新增正确注释数，不以开发 Recall@1 单指标选模。

## 2.6 必要负对照

- degree-preserving Rhea rewiring；
- 同质量差但非反应边；
- 相同 seed 数/反应度的随机候选；
- 打乱样本归属但保留候选结构；
- 去掉 raw-MS2 transformation；
- 去掉方向/完整参与物；
- DreaMS-only 风险门控。

真实网络必须显著优于所有匹配对照，否则只是网络密度或候选频率效应。

---

## 3. Track B：真正改变共享 embedding

## 3.1 核心原则

反应相邻代谢物是不同分子，不能无条件拉近。训练目标应同时满足：

- 同一身份跨条件不变；
- 同式近异构体可分；
- 生化关系可由类型化算子读取；
- 关系方向能映射到具体峰/中性丢失；
- 干净谱图检索不退化。

## 3.2 共享编码器

\[
z=E_\theta(x), \qquad z\in S^{d-1}
\]

query 与 reference 必须使用同一 `E_theta`，全部参考 embedding 在每次正式 checkpoint 下重新编码。推理时不读取 Rhea、候选、样本或表型。

## 3.3 Typed reaction operator

对具有真实实验谱图的反应对 `(a,b,r)`：

\[
\hat z_b=T_r(z_a),\qquad
L_{rel}=-\log\frac{\exp(\cos(\hat z_b,z_b)/\tau)}
{\sum_{j\in C_r}\exp(\cos(\hat z_b,z_j)/\tau)}
\]

`C_r` 必须包含：

- 同分子式/近结构错误候选；
- 相同质量差但非 Rhea 关系的 decoy；
- 度和化学类别匹配的随机网络 decoy。

关系算子学习“如何变化”，而不是把所有反应邻居变成同一簇。

## 3.4 Identity / near-isomer 双约束

\[
L_{id}=1-\cos(E_\theta(x_i),E_\theta(x_j))
\]

其中 `i,j` 为同身份跨采集条件谱图。

\[
L_{rank}=\log\{1+\exp[(m+s_n-s_p)/\tau]\}
\]

其中 `n` 优先来自已确认 DreaMS 错误图的 near-isomer。正例收紧和负例分离必须在同一 anchor 下评价。

## 3.5 峰级反事实约束

仅使用通过匹配随机对照的动作，不复用未经特异性验证的整组删峰：

\[
L_{cf}=\max(0,\,\delta-\Delta_{target}+\Delta_{matched-random})
\]

目标动作来自：

- 采集条件下经验缺失的峰；
- 支持错误候选的共享主峰；
- typed reaction 预期的碎片/中性丢失；
- ChemAware 规则对应且具有独立随机对照的峰。

峰动作只作为训练监督，不能把真值候选或 P2b 分数输入推理编码器。

## 3.6 总损失

\[
L_B=\lambda_{id}L_{id}+\lambda_{rank}L_{rank}+\lambda_{rel}L_{rel}
+\lambda_{cf}L_{cf}+\lambda_{off}L_{official-preserve}
\]

训练顺序：

1. 冻结 backbone，只检验 relation head 是否存在可读信号；
2. 最后一层 adapter/LoRA，所有 backbone dropout 关闭并保持 eval；
3. 若共享 clean embedding 的外层 OOF 增益成立，再有限解冻最后 Transformer block；
4. 不允许从教师/oracle 提升直接推断学生一定能获得同等 pp。

## 3.7 Track B 通过门

- formula、scaffold、dataset/instrument 三种隔离均报告；
- overall、near-isomer、cross-condition 均不退化；
- corrected > introduced 且 `corrected - 2*introduced > 0`；
- 多 seed cluster-bootstrap CI 下界大于零；
- official embedding preservation 达标；
- 相同容量、无 `L_rel`/无 `L_cf` 的 adapter 不得复现全部增益；
- sealed external Level-1 retrieval 和 MTBLS 注释率在固定 FDR 下改善。

---

## 4. 两条线如何协同而不泄漏

1. Track B 先输出候选无关的新 embedding；
2. P2b 独立读取 embedding 和谱学峰特征；
3. Track A 最后读取样本内上下文，只做低风险 residual；
4. Track A 的网络分数和 P2b 绝不作为 Track B 推理输入；
5. Track A 可用于发现 Track B 的失败亚型，但不能充当同一测试集上的 teacher；
6. 每层分别报告独立增量及组合增量。

---

## 5. 最近三步

### 2026-08-30 本地证据表示复核

已将候选上下文扩展为来源侧完整度、目标侧共产品完整度、完整 hyperedge、候选竞争数、计量元数据和方向可用性，并通过合成歧义路径单元测试。开发数据重放显示：

- MTBLS13729：50 个候选中 22 个有任意一跳证据，只有 8 个有来源侧完整路径；候选级完整 hyperedge 数量更少；
- MTBLS1905 自动种子：36 个 query 上没有可用 Rhea 路径；published-seed headroom 仅用于覆盖诊断；
- 当前离线 Rhea 缓存的 `direction_semantics` 全部为 `canonical_lr_not_physiological`，因此左右侧只能作为数据库规范方向，不能声称生理反应方向；正式 typed-direction 实验必须补充有方向证据的 Rhea/Reactome/酶学映射；
- 当前已暴露开发集上的 `1 corrected / 0 introduced` 只是假设生成结果，样本极小且使用 published seeds 时不是部署性能。

这说明 A0 必须先解决可用路径覆盖和方向来源，不能直接用当前一跳分数训练复杂 adapter。

### A0：重排证据可辨识性审计

在现有候选证据账本上，不改 Top-1，先检验 candidate-specific typed hyperedge 对“DreaMS 错、候选可被网络纠正”的 AUPRC，并与 degree/mass-difference matched decoy 比较。若真实网络不优于 decoy，Track A 不训练。

### B0：typed relation 冻结表示探针

只使用双方均有真实谱图的 Rhea reaction pairs；按公式/骨架隔离，用 `T_r` 与 relation-matched decoys 验证官方 embedding 是否包含反应类型可读信号。该阶段不声称检索提升。

### B1：最小共享 adapter

只有 B0 过门后，用 `L_id + L_rank + L_rel + L_cf + L_official-preserve` 训练共享 adapter；先做内层 OOF 和多 seed，不触碰封存测试。

这三步分别回答“网络能否安全重排”“生化关系是否可读”“共享 embedding 能否真正学到”，不得合并成一次大训练。
