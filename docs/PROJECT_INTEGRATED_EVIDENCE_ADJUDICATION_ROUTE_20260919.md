# DreaMS 多尺度证据裁决：整体研究路线与论文边界

日期：2026-09-19  
状态：项目总路线主记录；后续局部路线文档不得与本记录冲突  
目标：把既有 Error Atlas、Noise、局部谱学、ChemAware、BioAware 与真实样本验证组织成一条可证伪、可投稿的研究路线，而不是把独立协议的百分点相加。

## 0. 对专家建议的裁决

| 专家建议 | 裁决 | 原因与修正 |
|---|---|---|
| 用一个统一问题组织碎片资产 | 接受 | 统一的是“证据何时有资格改变排序”，不是假定所有模块组成已验证系统。 |
| 区分 spectrum-only 与 sample-aware | 接受 | 两种模式有不同可见信息和公平基线；Mode B 必须在相同强 unary 上证明增量。 |
| 建立 Bridge Panel | 条件接受 | 目前没有合格 panel；只能先写资格与冻结规则，不能预占正结果。 |
| 把所有旧结果排成七图递进链 | 拒绝 | 多数结果的数据、候选、指标和 split 不同；只有协议可连接的结果允许直接比较。 |
| 把旧 Failure Atlas 当正式起点 | 修正 | 旧 23,876-query cohort 已撤销正式资格；主图必须重建于 83,619-query corrected graph。 |
| 把 Peak-token 正增益作为已知事实 | 拒绝 | `+0.842/+1.001 pp` 无 canonical 来源，现存结果不支持该结论。 |
| 把 Noise E4 视为成熟底座并继续扩展 | 修正 | 旧 E4-A 属历史 cohort；只允许 repaired V3 的一次冻结裁决，且当前尚未运行。 |
| ChemAware 立即扩到 evidence topology | 推迟 | 先做 one-time outer；topology 只在 outer 后以零学习信息量审计启动，不能救 outer。 |
| BioAware 继续开发静态图先验 | 拒绝 | B44--B46 已形成外部反转/null/有害证据；B47 必须转向 prospective exact sample events。 |
| reverse metabolomics 作为验证加速器 | 条件接受 | 只用于检索、资源选择与外部分布支持；不得成为身份真值或表型引导打分。 |
| 现在就承诺完整 multi-scale 大文章 | 拒绝 | 稿件形态由 ChemAware outer、B47 独立增量和真实验证三道硬结果决定。 |

## 1. 总命题

本项目研究的是：

> 在预训练 MS/MS 全局表示之后，如何识别仍未被全局表示解释的证据，并仅在这些证据对候选判别提供可复现增量时改变排序。

内部简称：**多尺度证据裁决（multi-scale evidence adjudication）**。

该命题包含三个事实层级，不预设三者都产生正结果：

1. **可观测性**：局部峰、质量关系、候选条件信息或样本事件中是否存在增量信号；
2. **可迁移性**：该信号能否进入 candidate-independent 的共享谱图表示；
3. **可部署性**：在推理时不读取真值的条件下，能否安全地改善候选排序，并在外部数据或标准品上成立。

“教师动作有效”“开发集重排有效”和“共享 embedding 已提升”是三个不同主张，必须分别评价。

## 2. 两种运行模式

### Mode A：Spectrum-only annotation

运行时可见：query MS/MS、候选参考谱、前体/离子信息及预先冻结的候选条件特征。

可包含：

- 官方或经正式验证的共享 DreaMS encoder；
- 局部峰与 peak-token 证据；
- 质量差、显式规则及 ChemAware 候选条件策略；
- P2b 等强重排器作为比较对象。

该模式回答：没有样本上下文时，哪些信息在全局 embedding 之外仍能安全改善候选判别？

### Mode B：Sample-aware annotation

运行时额外可见：真实 LC-MS 实验中的 sample-feature 事件、离子家族、真值盲种子、精确反应事件和预注册的样本证据。

该模式回答：在强 spectrum-only unary 之上，样本特异上下文是否仍有独立增量？

Mode B 不等于把 catalogue membership、网络度数或历史错误频率加入 Mode A。B44--B46 已经证明，这些静态先验可能内部有效而外部反转。

## 3. 当前证据裁决

### 3.1 Error Atlas：保留为论文起点

专家稿引用的 `23,876 queries / 1,805 errors / 13,784 near queries / 1,676,787 pairs` 属于旧 cohort。该 cohort 因 `SIMULATION_CHALLENGE=False` 的语义误读已失去正式主结果资格，只能保留为历史资产或补充材料。当前 formal failure atlas 必须在 corrected graph 上重建；现有规模为 `83,619 queries / 5,957 errors / 28,188 near queries / 4,036 near errors / 6,220,661 edges`。

因此，旧图中的 positive deficit、ambiguity、pair-overlap 等只能称为待复核表型，不能直接称作因果机制。重建时每个数字必须绑定 manifest、候选协议、DreaMS checkpoint、split、tie policy 与哈希；未经统一 provenance 的数字不得进入定量主图。

允许主张：剩余错误呈现多种可重复的分数和峰级表型。  
暂不允许主张：这些表型已经分别构成被严格因果确认的错误机制。

### 3.2 Noise：表示修正与迁移瓶颈

保留的事实：

- 峰级/动作级 corrective headroom 存在；
- 历史 E4-A `+0.635 pp` 只在旧 cohort 上成立；其 targeted 相对 matched-random 仅 `+0.03377 pp` 且区间触零，不能作为当前 shared encoder 已改善的证据；
- 多个后续 targeted transfer 版本未稳定胜过 matched random/shuffled；
- 2026-09-16 V3 修复了 exact-injection 实现，但修复本身不是 held 性能结果。

当前仅允许一次冻结 V3 裁决。它已具备服务器单 GPU 入口，但修复后的 exact-injection V3 尚未运行，且本地缺少两个服务器侧冻结 ledger，状态是 `READY_ON_CLUSTER_ONLY / NOT_YET_RUN_AFTER_REPAIR`，不是“已失败”也不是“已成熟”。若 targeted 未同时胜过 official、initial E8 与 matched shuffled，停止正向 embedding 扩展；保留“candidate-level corrective signal 难以压缩进共享表示”的负边界，但必须报告效应量、置信区间和对照，不能只用训练失败支撑信息瓶颈。

### 3.3 Local spectral / Peak-token / A1：待统一 provenance 的资产线

专家稿中的 Peak-token `+0.842 / +1.001 pp` 找不到 canonical 冻结来源，现有报告反而显示：旧 pair cache 的最佳 single token overall 为 `-0.00419 pp`、near 为 `+0.2467 pp`；确认面板的 token-only 为 `-1.5585 pp`，raw signal 为 `+0.255 pp` 且区间跨零。除非找回唯一来源与协议，这两个正增益数字从路线中删除。

RAW-v1 是历史候选后处理资产：开发集 `+4.35 pp`，两个冻结测试分别 `+0.45 pp` 与 `+0.73 pp`，区间均跨零。A1 则是 153 spectra、73 panel IDs、69 independent IK14 上的关系坐标小面板，explicit multichannel 相对 direct multichannel 的 NDCG@5 为 `+2.962 pp`，IK14 区间为 `+1.317` 至 `+4.767 pp`；它不是 Recall@1、身份准确率或 embedding 提升。A1b 的 official-cosine fusion 为 `-0.590 pp`，区间跨零，两面板均下降，已触发停止门。A1 与 A1b 可作为“显式局部关系可见，但未迁移到一个 cosine”的直接机制对照，不能外推为“global embedding 缺乏这些信息”。

Peak-token、RAW multichannel、A1/A1b 可以共同支持“局部/显式关系是否在 global cosine 之外有增量”的研究问题，但它们不是同一分母，不构成递进消融。

近期任务是建立统一资产表：数据、split、候选、基线、指标、corrected/introduced、是否外层、是否已消费。只有协议可连接的结果进入同一主图；其余进入补充材料或作为假设生成证据。

### 3.4 ChemAware：当前最接近可部署的候选条件方法

当前 canonical truth-blind policy 已完成无真值 replay，并在 held-inner 显示较强绝对增益；但相对 nuisance-only 和 same-feature-direct 的机制增量尚未在内层显著成立，outer fold 4 仍未开启。

canonical outer role 4 包含 16,198 个未触碰 query，当前确无 outer lock 或结果目录。科学合同已具备一次性 outer 条件，但开封前恢复审计发现：seal 后没有持久状态机，普通异常会删除 staging 中的冻结预测而保留不可重入的 seal；若 truth 已读取但 report 未发布，则可能形成不可恢复的 outer 消费事故。因此当前状态是 `OUTER_READY_SCIENTIFICALLY / BLOCKED_BY_POST_SEAL_RECOVERY`，不是可立即提交。必须先在合成数据上实现并故障注入验证 prediction receipt、truth-open receipt、持久 staging、same-attempt 受限恢复与幂等只读 validator，再重新做 GO/NO-GO。修复不得改变 policy、baseline 或科学门。通过恢复审计后，outer 必须同时裁决：

- 总 Recall@1 增益及 formula-cluster CI；
- corrected 与 introduced；
- 相对 nuisance-only 的化学增量；
- 相对 same-feature-direct 的 residualization 增量；
- 各 Recall@k 不退化。

在 outer 之前不启动新的模型矩阵、GNN 或 evidence-topology 训练。关系组织只作为 outer 之后的条件性 P1：先做零学习信息量/空模型审计，证明确有增量后才允许 tiny edge-aware head。

### 3.5 BioAware：从静态先验转向真实样本事件

B30/B37 的内部正结果保留为“强但不可移植的 metabolic catalogue/risk prior”这一发现，不得称为 sample-specific biology。B44 的外部反转、B45 的 coverage-neutral null 和 B46 的 dual-context 负结果永久保留，不得用于继续调门。

B47 当前本地只确认到候选图：51,976 queries、10,578 global identities、2,078,709 query-reference rows，其中 ST001122 为 33,829 queries、ST003356 为 18,147 queries；服务器日志曾报告 51,976 query / 82,302 reference embeddings，但 embedding、seed 与 U0 尚未同步到本地，因此仍不能把日志当成完整可核验工件，更没有揭盲性能。B1 已于 2026-09-21 实现 fail-closed artifact registry 与 U0 单一入口；必须在服务器完成哈希闭环后才进入 B2。近期顺序固定为：

1. 对齐服务器与本地产物及哈希，不覆盖已有冻结结果；
2. U0：reference multiplicity、adduct mixing、candidate exposure；**已于2026-09-21完成**，max对expected-single造成22.24% both-unique Top-1翻转，且两个来源复现；
3. 强 spectrum-only unary；
4. ion-family / neutral-entity 层；
5. truth-blind seeds 与 exact event ledger；
6. matched degree/coverage/seed/rewired/sample-permutation nulls；
7. 一次性外部门。

B1 provenance已完成，但原始max-seed未达到200个独立identity，且U0显示严重reference-count敏感性；因此不得降低seed门或直接进入B2。只有先在非B47标签开发数据上冻结coverage/adduct-aware强unary、再真值盲重建seed，并且 exact sample event 在该强 unary、ion-family 与全部结构化空模型之上通过预注册门，才允许声称 sample-aware 增量。

### 3.6 P2b：冻结强基线，不再扩线

P2b 是固定权重 local rank fusion，不应称为一般 learned reranker。冻结证据为：开发 5,037 queries 上 `+3.91 pp`；sealed P3-main 3,000 queries 上 `+1.07 pp` 且区间严格为正；near-core 496 queries 上 `-4.23 pp`。它保留为强基线、near-core failure boundary 和真实应用中的第三意见，正负结果必须同表呈现。除非统一外层协议显示其仍有不可替代增量，否则不再继续研发。

### 3.7 Biology 与 reverse：验证加速器，不是身份真值来源

MTBLS13729、LCNEC 和组内任务只从已有候选池选择可验证对象，不再扩展新疾病故事。优先级按实验可执行性而非生物叙事评分：

- 原始 MS/MS 与样本是否存在；
- A/B 竞争结构是否真实；
- 标准品是否已有或可采购；
- RT、MS/MS、必要时 co-injection 是否可完成；
- 身份确认后是否产生清晰的生物学增量。

当前优先核验顺序是：MTBLS13729 的 f1597/f3019 近异构体问题、f703 Neu5Ac 成熟验证链，以及在资源真实可得时的 LCNEC quinolinate/ascorbate。f1717 只能报告 acetylated-polyamine family，缺标准时禁止指定 N1/N8；f3222 只能报告 C20:4 acylcarnitine-like class，丰度不能推出通量、CPT1A 或酶活。LCNEC 当前是独立的 Level-2 生物学资产，不是“新算法发现”。

Reverse metabolomics 只用于 standards/resources-first 检索和外部分布支持。表型不得进入身份打分；公共命中不得替代同平台标准确认。若 masked benchmark 不能胜过简单检索基线，它只保留为物流与标准采购工具，不升级为算法模块。

## 4. Bridge Panel 的资格

Bridge Panel 是未来连接两种运行模式的冻结交集，不是当前已存在结果。进入条件：

- 同一 query 同时具备 MS/MS、冻结候选、可靠独立身份、sample-feature event 和样本上下文；
- Mode A 与 Mode B 使用同一候选、tie policy 和 spectrum-only unary；
- panel 在任何 context 阈值或模型选择之前冻结；
- truth identity 不进入 seed/context 构造；
- 足够的 source、formula、错误和 reaction-reachable 错误支持聚类推断。

在满足以上条件前，Fig.6 只能是设计图或移入展望，不能预留一个必然为正的结果位。

## 5. 论文结构：固定主问题，条件性结果图

建议主文先按五张必备图组织，另外两张为条件图：

1. **Fig.1（必备）**：冻结 failure atlas 与错误异质性；
2. **Fig.2（必备）**：分面式证据地图，展示 global embedding 外证据的可观测性与迁移边界；除 A1↔A1b 外，不使用共享效应量坐标对不同协议结果排序；
3. **Fig.3（必备）**：ChemAware spectrum-only truth-blind adjudication 与 outer；
4. **Fig.4（必备）**：Noise targeted transfer 的正式裁决，允许为严格负结果；
5. **Fig.5（必备）**：BioAware shortcut、外部反转与 B47 事件化修复路线；
6. **Fig.6（条件）**：通过资格审计后的 Bridge Panel；
7. **Fig.7（条件）**：至少一个由独立身份验证支持的真实应用闭环。

若 Fig.6/7 未及时满足，不能靠开发集或候选名称填充。此时应缩窄为方法与边界论文，生物学文章独立推进。

## 6. 投稿裁决

本项目不预先承诺期刊。投稿形态由三个硬结果决定：

- ChemAware outer 是否通过全部注册门；
- B47 是否建立 sample-aware 独立增量；
- 是否完成至少一个标准/独立身份支持的真实应用闭环。

若只有第一项成立：方法论文。  
若第一、三项成立：方法加应用论文。  
若第二、三项也成立：再考虑完整 multi-scale system 叙事。

## 7. 统一纪律

1. 每个正文数字必须进入 claim registry，并绑定文件哈希与证据等级。
2. `complete/pass` 必须注明是工程合同、开发科学门、外层科学门还是外部确认。
3. 不同 query/candidate/tie/split 的百分点不得相加或画成递进 waterfall。
4. 负结果保留，但只能否定它实际检验的命题。
5. 新探索只有在既有证据给出明确可证伪问题时才允许启动。
6. 不覆盖冻结目录；修订使用新版本路径并保留旧报告。
7. 任何真实样本身份主张必须单列证据层级与禁止措辞。
