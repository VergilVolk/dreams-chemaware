# BioAware 动作发现重置与决战合同（2026-09-06）

## 0. 结论先行

BioAware 目前不是“没有信号”，而是存在两个已经在打开开发数据上达到目标量级、但尚未完成独立确认的候选动作：

| 动作 | 固定协议 | Recall@1 增益 | corrected / introduced | 独立 corrected identity / formula | 当前资格 |
|---|---|---:|---:|---:|---|
| 同分子式不确定性动作 | 482 query；8-unit LOSO；truth identity + formula purge | **+5.81 pp** | 39 / 11 | 19 / 18 | 开发候选教师 |
| 四来源高召回网络动作 | 548 query；leave-source-out；truth identity + formula purge | **+4.01 pp** | 23 / 1 | 12 / 11 | 开发候选教师 |
| 四来源安全网络动作 | 同上，要求 raw step-0 edge | **+3.47 pp** | 19 / 0 | 8 / 7 | 高精度开发基线 |

它们都通过了候选网络特征联合置换反证，但都没有通过独立外部确认。冻结外部数据上当前最好结果仍接近零，核心失败表现是动作不触发，而不是大量触发后方向相反。

因此 BioAware 的正确顺序固定为：

1. 先证明一个可部署动作在真实候选组内稳定净改正 3--5 pp；
2. 再扩大该动作的独立 identity、formula、source 和外部触发覆盖；
3. 冻结动作后优先实现候选组内重排器；
4. 只有当网络动作包含可由单张谱图恢复的信息时，才允许尝试蒸馏到共享 spectrum-only embedding；否则使用显式 context-conditioned 表示，不能把信息论上不可实现的目标伪装成优化失败。

当前不得声称 BioAware 已经达到 SOTA，也不得声称已经改进共享 embedding。

## 1. 这次审计修正了什么

统一证据账本位于：

- `tasks/audit_bioaware_action_evidence_ledger.py`
- `data/validation/bioaware_action_evidence_ledger_v2_20260906/report.json`
- `data/validation/bioaware_action_evidence_ledger_v2_20260906/query_action_ledger.csv.gz`
- `data/validation/bioaware_action_evidence_ledger_v2_20260906/strict_changed_actions.csv.gz`
- `data/validation/bioaware_action_evidence_ledger_v2_20260906/same_formula_changed_actions.csv.gz`
- `data/validation/bioaware_action_evidence_ledger_v2_20260906/baseline_error_headroom.csv.gz`

账本不拟合模型、不调阈值、不使用 P2b、不使用 phenotype，也不改变 embedding。它重新验证每条 query 的 truth、baseline、proposal、final、corrected、introduced，并同时报告 query 数和去重后的 identity/formula/pair 数。

最重要的纠偏是：query-level 的 39/11 或 23/1 不能当成 39 或 23 个独立生物化学机制。多个组织/色谱条件会重复同一 truth--wrong pair。论文和决策必须以 query 与 identity/formula 两个层级同时报告。

## 2. 已经做过什么，以及它真正告诉了我们什么

### 2.1 一跳 Rhea 固定加分

MTBLS13729 的早期一跳 Rhea pilot 只有 21 个可评估 query，DreaMS 为 20/21，BioAware 改为 19/21：0 corrected、1 introduced。该结果否定的是“固定的一跳邻接分数可直接覆盖 DreaMS”，不是反应网络本身。

教训：

- 网络相邻不是 identity 标签；
- 高度节点、currency metabolites 和大反应会产生伪支持；
- 必须以候选为单位建证据，并允许缺失、冲突和弃权；
- 21 个 query 无法支撑模型选择。

### 2.2 feature graph 与局部安全门

随后加入真实 feature、样本内 seed、离子家族和局部路径检查，少量 query 出现安全修正，但覆盖率过低。它证明真实样本上下文能作为补充证据，却没有建立跨库性能。

教训：安全门能抑制 introduced，但如果门依赖某个库特有的 raw-edge 覆盖，迁移时会完全失活。

### 2.3 MetDNA3 对齐的候选网络动作

当前 548-query 负离子任务使用 11 个候选特征：DreaMS 分数、mass membership、最短路径可用性/深度/seed 支持/节点 degree、step-0 和 step-1 raw-MS2 路径完整度与瓶颈、预测边增量。它在严格 source + identity + formula purge 下取得 +3.47 至 +4.01 pp。

这证明：

- candidate-specific 网络分配中存在真实增量；
- 该增量不是简单的“只在 DreaMS 低置信时乱换”，因为联合候选置换的 p95 远低于观察增益；
- raw step-0 gate 提高精度，但牺牲覆盖；
- full-no-edge 是更高召回动作，safe 是高精度动作，二者应作为两个不同教师保留。

但它也暴露出严重混杂：安全动作的 19 个修正中，proposal 的网络 degree 全部高于 baseline。当前动作可能同时使用了真实反应支持和数据库收录/节点流行度先验，必须用 degree/path-matched null 拆开。

### 2.4 同分子式不确定性动作

同分子式任务把候选限制在化学完整、精确 `[M-H]-`、与查询分子式一致的候选组，再以 DreaMS margin <= 0.05 作为唯一 intervention gate，得到 +5.81 pp、39/11。

这是目前最强的开发动作，但并不自动优于四来源安全动作：

- 优点：覆盖更宽，修正 19 个独立 identity；
- 缺点：introduced 增至 9 个独立 identity，干预精度约 0.51；
- 它需要部署时具有可信分子式约束；
- 它是在已打开的数据上经诊断形成的动作，不能作为外部确认。

两个候选教师的 corrected-query Jaccard 仅约 0.41--0.51，说明它们既有重叠也有互补；不能简单平均，也不能用当前 truth 事后挑选。下一版需要显式学习“安全网络动作”和“同分子式高召回动作”的适用条件。

### 2.5 外部冻结测试

当前冻结外部结果：

- ST001154 HILIC：161 query，0 intervention，0/0；
- ST001154 extension：150 query，3 interventions，0/1；
- KGMN 200STD hidden seed：162 rotations、55 unique query，0 intervention，0/0；
- 多 panel 汇总也只有约零增益，置信区间跨零。

这说明最大的现实瓶颈是 action coverage/calibration transport。不能把“零干预”解释成“网络方向错误”，也不能把开发 +3--6 pp 写成跨数据库稳定增益。

### 2.6 B0 单谱反应邻居因果设计

B0-M0 尝试在谱库交集中构造严格的 reaction-neighbour vs matched non-neighbour 单谱对照。最终因 coverage 与 balance 无法同时满足而停止。这是可识别性失败，不是网络信号的负实验。

### 2.7 B4/B5/B6 共享 embedding 尝试

现有共享 encoder 实验未建立 Recall@1 增益。这里既有动作覆盖不足，也有工程/目标错配：

- 不同阶段对 teacher logit、query z-score、reference aggregation 的定义不一致；
- 两候选时 query z-score 几乎把概率固定成同一对数几率，压掉强弱信息；
- 只保留 teacher-corrected query，丢掉 harmful/no-op 决策边界；
- hard truth 与 soft teacher 混为一个损失；
- max-over-reference 只给 argmax reference 梯度；
- 训练与评估的 reference multiplicity 不同；
- 每步 clipping、只解冻末块且没有 branchwise 梯度审计，使“没有移动”无法区分是梯度未注入还是目标不可实现。

因此 B4--B6 的零增益不能被包装成 BioAware 科学假设失败，但同样不能继续作为共享 embedding 的正证据。

## 3. 与代表方法的差距：不是缺一个更大的神经网络

| 方法 | 它实际解决的问题 | 我们已实现 | 我们仍缺失 |
|---|---|---|---|
| MetDNA / MetDNA3 | seed 到候选的递归传播；知识网络与 data layer 交互；预测反应扩边 | identity-isolated seed、已知/预测路径、局部 raw-MS2 边 | 可靠路径多重性、反应类型与方向、交叉拟合递归传播、校准 FDR |
| KGMN | 反应网络 + knowledge-guided MS2 + global peak correlation；全局删除冲突和离子冗余 | 局部候选路径和 raw-MS2 bottleneck | 样本级 ion family、RT/coelution/correlation、全局冲突优化 |
| NetID | mass/RT/MS2 与生化、加合物、碎片、同位素边的全局网络优化 | 部分生化质量差与 MS2 | abiotic vs biochemical edge typing、全局 assignment、无 MS2 feature 联动 |
| MS-Net | 谱图、结构、taxonomy 的多相似度网络与迭代传播 | 谱图 + 反应网络候选证据 | 结构/来源证据的校准、迭代传播防确认偏差、candidate rescue 的独立外测 |

主要文献与代码：

- MetDNA3: https://www.nature.com/articles/s41467-025-63536-6
- KGMN/MetDNA2: https://www.nature.com/articles/s41467-022-34537-6
- NetID: https://www.nature.com/articles/s41592-021-01303-3
- NetID code: https://github.com/LiChenPU/NetID
- MetDNA3 code: https://github.com/ZhuMetLab/MrnAnnoAlgo3
- MS-Net: https://pubmed.ncbi.nlm.nih.gov/42216864/

本项目不能把“Rhea 一跳 + DreaMS 分数”包装成新方法。真正可能形成方法创新的是：在严格候选组内，将反应路径可靠性、谱图变换一致性和样本 feature 图作为互相独立的因子，并用可审计的风险门控/全局冲突约束决定是否改写 DreaMS。

## 4. 当前动作为什么没有迁移

### 4.1 独立样本量小于表面 query 数

相同代谢物和同一错误对跨组织/LC 条件重复，query-level bootstrap 会高估机制复现。必须同时按 formula、identity、truth--wrong pair 和 source 聚类。

### 4.2 特征聚合过早丢失信息

raw edge 文件已经含有：

- `identity_paths`
- `complete_ms2_paths`
- `node_combinations_evaluated`
- `path_truncated`
- `best_bottleneck`
- `median_bottleneck`

现有 ranker 最终只保留“是否至少一条完整路径”和“平均 best bottleneck”。多路径是否重复支持、路径是否只靠单一边、best 与 median 是否冲突、搜索是否截断，全被压掉。这会把一个偶然高分路径和多条稳定路径视作同一种证据。

### 4.3 网络 degree 与真实生化证据混在一起

高 degree 节点更容易获得路径、seed 与 raw edge，现有模型对 degree 系数持续为正。候选置换只证明“网络特征与候选的对应关系有用”，不能证明增益独立于 catalog popularity。必须增加 degree/path-availability-matched null。

### 4.4 只做 candidate-local，未做 feature-global

KGMN/NetID 的强项不是多加一列候选分数，而是利用同一样本所有峰之间的 RT、共洗脱、强度相关、同位素/加合物/源内碎裂关系消解冲突。我们当前动作没有使用这一层，所以对外部数据的网络覆盖一旦下降，便没有第二条证据维持干预。

### 4.5 共享 spectrum-only embedding 存在信息边界

如果同一张谱图在不同样本上下文中应产生不同候选排序，那么确定性的 `E(spectrum)` 不可能完整编码该样本特异信息。可行选择只有两种：

1. 网络动作与谱图内部可见模式稳定相关：可蒸馏到共享 embedding；
2. 动作依赖样本 feature graph：必须使用显式 context-conditioned query/candidate embedding 或后验 ranker。

在动作一致性尚未量化前直接训练共享 encoder，会把“不可由输入恢复的信息”当监督，结果必然接近零或互相抵消。

## 5. 下一轮动作空间：固定、分层、可反证

### A. 两条现有动作必须保留

- `A_safe`：要求 raw step-0 edge；目标是 introduced 接近零；
- `A_formula`：同分子式 + DreaMS 低 margin；目标是扩大 corrected 覆盖。

二者独立报告，不先平均成一个 teacher。

### B. 立即补齐的高性价比动作

1. `A_path_reliability`：完整路径数、identity path 数、completion ratio、median bottleneck、best--median spread、跨 fold 支持率；
2. `A_degree_normalized`：固定 degree 或 path availability 后比较候选，使用 seed-per-degree、complete-path-per-degree、edge support efficiency；
3. `A_reaction_fragment_concordance`：反应精确质量差/元素变化与 query--seed 的 fragment/neutral-loss 差异是否一致；
4. `A_ion_family_veto`：同样本 RT/coelution/correlation、同位素、adduct、in-source fragment 关系，只作 veto 后再评估是否可作 proposer；
5. `A_recursive_crossfit`：最多两轮；每轮 seed 必须由另一 fold 产生；每次传播要求两个独立证据通道；
6. `A_global_conflict`：同一 feature 多候选、同一候选多 feature、离子家族冲突的全局 assignment。

### C. 每条动作的强制反证

- spectral-only；
- within-query joint candidate permutation；
- degree/path-availability-matched network null；
- seed-label shuffle；
- edge direction/reaction-class scramble；
- sample co-occurrence shuffle；
- same-dose/no-op control。

任何动作若只击败 random candidate permutation、却不能击败 degree/path matched null，不得解释成生化反应证据。

## 6. 固定评价与验收门

### 6.1 开发任务

- 真实 query 与真实候选组；
- official DreaMS baseline 与动作使用同一候选聚合、并列规则和 reference multiplicity；
- outer split 至少隔离 source + truth identity + formula；若样本量允许再报告 scaffold/reaction-class holdout；
- 所有阈值仅由 outer-train 内层 OOF 选择；outer-test 不得参与动作选择；
- 每个 query 记录 baseline、proposal、final、corrected、introduced、证据路径和弃权原因。

### 6.2 动作进入冻结外测的最低门

- Recall@1 净增益至少 3 pp；
- formula-cluster CI 下界 > 0；
- corrected > 2 x introduced；
- 至少 20 个 corrected identity、15 个 corrected formula；
- 每个来源方向非负；
- near/isomer 子集不退化；
- candidate permutation 与 degree/path-matched null 均通过；
- 预注册 coverage--precision 曲线，不在测试集调门。

### 6.3 外部确认门

- 冻结工件，零 refit；
- 外部动作必须有非零且足够的 identity-level activation；
- Recall@1、MRR 和风险加权净收益均不退化；
- formula/identity-cluster CI 下界 > 0；
- 至少一个完全独立数据集达到 >=3 pp，另一个方向一致，方可讨论通用性能；
- 与 MetDNA3/KGMN/NetID 必须在同一任务和同一候选图上比较，不能横向拼接论文数字。

3--5 pp 是动作晋级门，不是可以预先保证的结果。任何脚本不得通过反复看 test 后调阈值制造该数字。

## 7. 从动作到模型的唯一允许顺序

### Phase 1：动作账本与作用域（已完成）

统一两条强动作、消歧其协议、去重独立证据、列出外部失败和 retrospective headroom。

### Phase 2：Action Atlas v3（当前）

在所有 baseline error 与 matched-correct control 上构造完整候选因子：路径可靠性、degree-normalized 网络支持、反应--碎片一致性、ion-family/RT/coelution、availability/conflict/unknown。先做低容量、候选组内 OOF，不上大模型。

### Phase 3：冻结 ranker

若动作通过开发门，冻结 action recipe、feature schema、scaler/model、阈值和 provenance；先做独立外测。若只能在 sample context 下工作，就保留为 BioAware ranker，这是正确的产品形态，不是失败。

### Phase 4：共享 embedding 可蒸馏性门

先计算同一 identity 在不同 context 中的动作方向一致性，并训练 spectrum-only probe 预测 action sign。只有 identity/formula 隔离 OOF 显著高于 matched null，才允许蒸馏到 `E(spectrum)`。

### Phase 5：embedding 或 context adapter

- 可蒸馏：同一 shared encoder，候选组内 listwise margin，corrective/harmful/no-op 全量监督，损失与评估 reference aggregation 同构，逐分支记录梯度范数与余弦；
- 不可蒸馏：使用 `E(spectrum, sample_graph)` 或 candidate-context adapter；推理时显式输入上下文；
- 两者均不得把 candidate identity、truth、phenotype 或 P2b 分数作为 embedding 监督捷径。

## 8. 当前最优决策

短期最高性价比不是再次训练 13.6M/116M 参数模型，而是：

1. 保留 `A_safe` 和 `A_formula` 两个已有 3--6 pp 开发动作；
2. 立即补 path reliability 与 degree-matched specificity；
3. 将候选局部动作接入样本级 ion-family/global-conflict 层，扩大外部 activation；
4. 达到独立 identity 覆盖门后冻结一个低容量 reranker；
5. 外部通过后，再用可蒸馏性门决定 shared embedding 或 context-conditioned embedding。

这条路线既不放弃 BioAware embedding，也不让一个尚未被证明可由单谱恢复的 context 信号继续浪费 GPU。

## 9. Action Atlas v3 实测裁决（2026-09-06）

本节是在上述计划冻结后得到的实测结果。它改变了我们对旧 BioAware 增益来源的解释，因此优先级高于前文对旧动作的暂定表述。

### 9.1 旧 +5.81 pp 已精确复现，但不是反应特异证据

在相同 482-query、145-identity、117-formula 的已打开开发协议上，旧 `current_v4` 被逐 query 精确复现：

| 固定动作 | Recall@1 增益 | corrected / introduced | corrected identity | formula-cluster 95% CI |
|---|---:|---:|---:|---:|
| spectral only | 0.00 pp | 0 / 0 | 0 | [0, 0] |
| spectral + network degree | **+6.02 pp** | 38 / 9 | 20 | [+2.80, +9.77] pp |
| 旧 current v4 全特征 | **+5.81 pp** | 39 / 11 | 19 | [+2.11, +10.00] pp |
| current v4 去 degree | +3.94 pp | 25 / 6 | 13 | 正向但更窄 |
| path reliability 去 degree | +2.70 pp | 24 / 11 | 13 | 未达到动作晋级门 |
| current + 全部 path reliability | +5.60 pp | 41 / 14 | 22 | 未优于 degree control |

最重要的比较不是“全模型是否高于 DreaMS”，而是“精确反应路径是否在 degree 之上提供额外信息”。答案目前是否定的：

- `current_v4 - degree_control = -0.21 pp`，paired formula CI 跨 0；
- 补回完整路径数、completion ratio、median/best bottleneck、path truncation 等字段后，仍未超过 degree control；
- path-reliability-only 明显低于 degree control。

### 9.2 degree-fixed、path-availability-conditioned 反证失败

我们固定 spectral score 与候选节点的精确 degree，并在 `unit × degree quintile × known-path/edge-availability` 层内联合置换其余网络字段。94.45% 的候选行被实际移动。

- 旧动作 Recall@1 增益：+5.81 pp；置换零分布均值：+5.85 pp；经验 `p=0.624`；
- 旧动作 `corrected - 2×introduced = 17`；置换零分布均值：16.86；经验 `p=0.574`。

因此，现有 precise-path/bottleneck 字段未显示超出 degree 与粗路径可用性的候选特异价值。旧 5.81 pp 必须降级解释为：

> 一个在已打开同分子式压力集上有效的 catalog/network-coverage prior，而不是已验证的生化反应路径动作。

这不等于 degree 没有工程价值；它意味着 degree 不可作为 BioAware 生物动作、不可作为 shared-embedding 教师，也不可据此宣称反应网络机制成立。

### 9.3 当前真正缺失的动作

下一动作必须以 degree control 为基准，而不是只以 DreaMS 为基准。合格动作要证明：

1. 在匹配 degree、路径数量、完整路径数量、谱图组合机会、极性与分离模式后，真实反应边的谱学支持强于非反应边；
2. 反应精确质量/元素变化与 query--seed 的 fragment/neutral-loss 变化一致；
3. 或同一样本内候选--种子的丰度共变在 degree-matched、seed-shuffled null 之外成立；
4. 在 source + identity + formula 隔离 OOF 中，相对 degree-only 至少增加 3 pp，且 corrected identity 足够、introduced 受控；
5. 若相对 degree-only 未通过，则只能保留 degree prior 作为工程对照，不得进入 embedding。

### 9.4 下一步固定顺序

1. 构造 opportunity-calibrated reaction evidence：用完整路径数、组合数、degree、谱图数校准 bottleneck 的偶然最大值偏差；
2. 若仍无独立增益，停止在旧 bottleneck 上继续调模型，转向 reaction-transform × fragment/neutral-loss concordance；
3. 再评估 `reaction probability × shrinkage abundance correlation × seed confidence / hub penalty`；
4. 只有某条动作相对 degree-only 达到预注册门，才冻结低容量候选组内 ranker；
5. 再通过 spectrum-only 可预测性检验决定 shared embedding 或 context-conditioned embedding。

对应工件：

- `tasks/develop_bioaware_action_atlas_v3.py`
- `data/validation/bioaware_action_atlas_v3_20260906/report.json`
- `data/validation/bioaware_action_evidence_ledger_v2_20260906/report.json`

### 9.5 opportunity-calibration 草稿的预运行否决

第一版 opportunity-calibration 草稿在正式产出前被终止，不能作为结果。预运行审计发现：

- 只按 `source × LC` unit 留出，会让同一 biological source 的另一 LC 模式留在训练集；
- 以其他真实路径作为 kNN 对照，不等同于 degree/path-matched 非反应路径；
- train residual 未做 nested cross-fit，且 fallback 会重新接纳相关候选；
- comparator 未包含所有被用于 calibration 的 opportunity 主效应，残差可能只是把遗漏的主效应重新带回模型；
- 145 个 identity / 117 个 formula 对“相对 full-opportunity baseline 再增加 3 pp”明显功效不足。

因此该草稿未完成、未保存结果、不得提交服务器。下一步先做全资产 support/power audit；正式设计必须同时满足：以 biological source 为外层、nested identity/formula cross-fit、零 fallback、真实非反应 matched controls、full-opportunity comparator，以及 identity-equal 增量统计。
