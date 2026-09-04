# BioAware × MetDNA3 集成与生物上下文表示学习契约

**冻结日期：2026-08-30**  
**当前目标：** 不再从零发明代谢网络重排器。优先严格复现 MetDNA3；若完整官方资产不可得，则立即以公开且完整的 KGMN/MetDNA2 作为成熟基线。在不改作者候选生成、传播和去冗余逻辑的前提下，仅研究 DreaMS 能否改善实验 feature 网络中的 MS2 边判定；只有通过外部与 decoy 评价的网络 margin，才允许进入共享 DreaMS embedding 微调。

## 0. 复现可用性硬门（2026-08-30 补充）

本地源码审计发现，公开 `MrnAnnoAlgo3` 仓库包含 MRN3 核心传播函数，但 `MRN3main.R` 正式运行所需的 `obj_mrn*.rda`、`info_mrn*.rda`、`md_mrn*.rda` 并未随仓库提供；README 也明确把完整功能和示例数据放在需注册的 MetDNA3 网站。因此：

- 仅有 GitHub 核心代码时，**不得声称完整复现 MetDNA3**；
- 若提供作者完整工作目录（至少含 `table_ms2_edges.rda` 与 `ms2_data.rda`），可走精确 MetDNA3 桥接；
- 若没有这些资产，正式基线切换为公共数据与代码齐全的 KGMN/MetDNA2；MetDNA3 只保留为算法设计参照，不作为已复现对照；
- 该判断由 `tasks/audit_metabolic_network_framework_reproducibility.py` 自动输出，禁止人工越过。

这不是降低目标，而是把“最新但缺资产的论文”与“真正能运行、能比较、能改进的算法”分开。论文中可以把 MetDNA3 作为最新方法学参照，但实证基线必须可复现。

## 1. 为什么选择 MetDNA3 作为主框架

代表方法解决的是不同层次的问题：

| 方法 | 已有成熟能力 | 对应本项目瓶颈 | 本项目处理 |
|---|---|---|---|
| MetDNA / MetDNA3 | 从可靠种子出发，在知识层反应网络和实验 feature 层之间递归传播；MetDNA3加入预测 MRN、MS1 预映射、raw-MS2 feature-edge 约束和逐步去冗余 | BioAware v1 的一跳覆盖低、没有成熟递归传播、网络扩张后错误传播 | **主复现框架**；作者传播和去冗余保持不变 |
| KGMN / MetDNA2 | 联合反应网络、谱图相似网络和峰相关网络 | BioAware 的证据彼此割裂 | 作为 MetDNA3/KGMN 消融思想，不再另造一个多层图算法 |
| NetID | 同时建模同位素、加合物、碎片和生化转化边，进行全局一致性分配并使用 target-decoy/FDR | 离子形式混淆、同一 feature 多候选、局部贪心冲突 | 后续作为 MS1/ion-family 全局一致性对照；不是首个结构级 near-isomer 排序器 |
| JESTR / FLARE 类候选表示方法 | 在候选组中建模分子结构或峰—结构局部对应 | DreaMS 固定 cosine 难分同式近异构体 | 只借鉴“候选组内 ranking”和“局部证据”训练形式；不冒充代谢网络方法 |

选择 MetDNA3 的原因不是它最新，而是它已经覆盖当前最明显的三处缺失：数据—知识双层映射、递归传播和冗余消除。重写这些部分既慢又难以证明创新。

## 2. 当前瓶颈与成熟算法的对应措施

### 2.1 种子稀少和一跳覆盖低

- **MetDNA3 的措施：** MS1 预映射后逐轮传播，每轮新注释可成为下一轮种子。
- **我们不能做的事：** 为提高覆盖无约束地扩 Rhea/预测边。
- **集成原则：** 保留作者传播深度、候选生成和置信度逻辑；BioAware 只提供 DreaMS 种子谱学分数、路径解释和冲突/弃权信息。

### 2.2 密图、共同底物和预测反应造成错误传播

- **MetDNA3 的措施：** 知识边必须能映射到实验 feature 边，且 feature-edge 的 MS2 相似度达到阈值；最终执行逐步去冗余。
- **仍存在的缺口：** 作者的固定 modified-dot-product 对采集条件漂移和共享主峰敏感；论文也承认扩网会降低特异性。
- **唯一首轮创新：** 用同一共享 DreaMS encoder 为作者已经生成的同一批 `table_ms2_edges` 重新打分，并与作者分数做四个冻结对照：
  1. 作者 modified-dot-product；
  2. 官方 DreaMS；
  3. 通过外部评价的 noise-tuned DreaMS；
  4. 作者分数与 DreaMS 的保守交集。

### 2.3 网络证据不是候选特异证据

- **NetID 的措施：** 不把单条生化边直接当身份；联合非生化离子关系、质量、RT、MS2 和全局约束。
- **我们的最小改进：** 首轮不再训练通用 BioAware 融合器。先检验 DreaMS feature-edge 是否提高真反应边相对 degree-preserving / formula-matched decoy 边的分离，以及是否降低错误传播。
- **通过条件：** 在预注册阈值下，传播后的正确 Top-1/Top-3 或固定 FDR 覆盖增加，同时 introduced 不增加；不能只报告 edge AUC。

### 2.4 证据相关、校准差和风险不可控

- **NetID/MetDNA3 的启示：** 使用 target-decoy/FDR、标准品和去冗余，而不是把相关证据当多张独立票。
- **BioAware 的保留价值：** 只负责证据路径、冲突、缺失和回退；不再用 query 内 min-max、多票计数或零权重证据开门。
- **冻结主指标：** coverage-risk、固定 1%/5%/10% FDR 注释数、corrected/introduced、formula/scaffold/study 分层。

## 3. 不修改作者算法的工程接口

固定作者源码：

- 仓库：`third_party/MrnAnnoAlgo3`
- commit：`978ae62b33bde75a066032953ed912a716274288`
- 作者固定 `ms2_cutoff = 0.5`
- 作者会从 `02_result_MRN_annotation/table_ms2_network.rda` 读取 `from, to, ms2_score`，随后执行原始过滤、data-to-knowledge 回映射、递归注释和去冗余。

作者仓库声明为 CC BY-NC-ND 4.0。为避免把改写作者源码误作我们的实现，本项目不修改或重新分发其 R 源码；桥接仅通过作者公开的输入/中间文件接口完成，论文中明确标注原算法与新增模块的归属。

桥接流程：

1. 从作者工作目录导出原始 `table_ms2_edges.rda` 和 `ms2_data.rda`；
2. 将作者 feature 名和谱图无损导出为 edge CSV + MGF；
3. 用一个冻结的共享 DreaMS checkpoint 对所有 feature 谱图编码；
4. 仅对作者原始 edge 集合计算 DreaMS cosine；
5. 在开发数据上拟合并冻结把 cosine 转为 `[0,1]` edge reliability 的校准器；
6. 严格核对 `from/to` 后写回新的 `table_ms2_network.rda`；
7. 在新的、独立工作目录运行未修改的 `MrnAnnoAlgo3` 后半段。

禁止：改变作者 edge 候选集合、用真值逐 edge 选最高分数、在外部测试集调阈值、把 `(cos+1)/2` 当校准概率、覆盖原作者工作目录。

## 4. 第一阶段必须冻结的四个比较臂

| 臂 | edge score | 用途 |
|---|---|---|
| A | 作者 modified-dot-product | 完整复现基线 |
| B | 官方 DreaMS 校准分数 | 检验预训练表示是否改善 feature-edge |
| C | noise-tuned DreaMS 校准分数 | 检验更好 shared embedding 是否迁移到网络算法 |
| D | `min(calibrated author, calibrated DreaMS)` 或双阈值交集 | 针对密图/假边的唯一风险控制创新 |

四臂共享：同一 feature、同一候选 edge、同一种子、同一 MRN、同一 RT/CCS 设置、同一传播和去冗余代码。只有 edge reliability 不同。

## 5. 评价顺序与停止规则

### M0：作者复现

- 作者输出数量、种子数、传播轮数、Top-N 和 FDR 与论文/作者示例合理一致；
- 若不能复现，停止所有“改进”比较。

### M1：edge 级别机制验证

- identity/study 隔离；
- 真反应 feature-edge 对比 degree-preserving、m/z/RT/degree 匹配 decoy；
- 报告 AUPRC、校准、固定 FDR recall，不以 edge AUC 单独过门。

### M2：传播结果

- Top-1、Top-3、Top-10；
- target-decoy FDR 下的注释覆盖；
- corrected/introduced；
- 深度、节点度数、seen/unseen reaction、formula/scaffold/study 分层；
- 至少一个外部 study 保持完全未暴露。

### M3：最小创新裁决

只有 D 臂相对 A 臂在外部数据上提高固定 FDR 覆盖或准确率、且不增加 introduced，才能称为“DreaMS-constrained MetDNA3”改进。否则只保留复现和负结果。

## 6. 网络先验如何指导 embedding 微调

不能把反应相邻的不同代谢物直接拉近。网络只允许提供**训练关系的可靠度**：

1. 在训练 study 上交叉拟合 MetDNA3；
2. 仅保留真实网络显著优于 degree-preserving decoy、且跨折方向一致的候选 margin；
3. 构造 `(query spectrum, same-identity positive reference, hard wrong candidate)`；
4. 网络只决定该 ranking triple 是否可信及权重，不定义 identity positive；
5. 用一个共享 query/reference DreaMS encoder 优化组内 ranking，同时加入官方 embedding preservation；
6. 推理时只输入单张干净谱图，不需要代谢网络、候选或表型。

训练目标：

`L = w_net * softplus((m + s_neg - s_pos) / tau) + lambda_preserve * (1 - cos(z, z_official))`

其中 `w_net` 来自交叉拟合且通过 decoy 的网络置信，不来自测试真值。必须同时比较：随机网络权重、仅身份对比学习、MetDNA3 网络教师、DreaMS-constrained MetDNA3 教师。

## 7. 当前能说和不能说的结论

现在可以说：我们选择成熟 MetDNA3 作为主网络算法，并找到了不修改其传播逻辑的精确 DreaMS edge 接口。

现在不能说：BioAware 已经复现 MetDNA3、已经超过 MetDNA3、DreaMS edge 一定更好、网络先验已经改善 embedding、或已经达到 SOTA。

## 8. 下一批可执行产物

- `tasks/export_metdna3_official_bridge.R`：导出作者 edge 与 feature 谱图；
- `tasks/encode_metdna3_feature_dreams_embeddings.py`：官方/实验 shared checkpoint 编码；
- `tasks/build_metdna3_dreams_ms2_network.py`：严格按作者 edge 集合打分；
- `tasks/inject_metdna3_ms2_network.R`：验证并生成作者可读取的 RDA；
- `tasks/run_metdna3_dreams_edge_bridge.sbatch`：服务器执行入口；
- `tests/test_metdna3_dreams_network_bridge.py`：边集合、顺序、重复、范围和无真值依赖测试。

## 9. 代表文献与代码

- MetDNA3, Nature Communications 2025: https://www.nature.com/articles/s41467-025-63536-6
- MrnAnnoAlgo3: https://github.com/ZhuMetLab/MrnAnnoAlgo3
- KGMN / MetDNA2, Nature Communications 2022: https://www.nature.com/articles/s41467-022-34537-6
- NetID, Nature Methods 2021: https://www.nature.com/articles/s41592-021-01303-3
- NetID code: https://github.com/LiChenPU/NetID
- JESTR: https://pmc.ncbi.nlm.nih.gov/articles/PMC12233093/
- FLARE: https://pmc.ncbi.nlm.nih.gov/articles/PMC12873900/
