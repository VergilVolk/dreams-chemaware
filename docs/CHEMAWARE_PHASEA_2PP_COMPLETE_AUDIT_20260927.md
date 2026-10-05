# ChemAware Phase-A 2 pp 完整审计与下一代直接微调路线

日期：2026-09-27

状态：`AUDIT_COMPLETE / NO_NEW_HELDOUT_OPENED / NO_WEIGHT_UPDATE`

## 0. 审计结论

当前最准确的结论不是“triplet 还不够多”，而是：

1. `+2.1266 pp` Phase-A 是目前**通过既定安全标准的最强 role-2 shared-embedding 开发结果**；它从 official DreaMS 独立初始化，不是从 `+1.8144 pp` 模型继续训练。
2. 单看 Recall@1 点估计，pair-expanded step 2000 曾达到 `+2.2785 pp`，比 Phase-A 多3个正确 query；但它牺牲6个 Recall@3 query、产生更大几何漂移，且相对前一基线的增量区间跨零、风险效用为负，因此不是合格替代品。
3. Phase-A 相比 stage-1 的主要优势是**少制造错误**，而不是纠正更多 official errors：两者都纠正54个，Phase-A把 introduced 从24降到12。
4. 后续方法失败的共同原因不是训练不够，也不是 triplet 数量不够，而是监督单位错位、pairwise loss 与 candidate-list retrieval 错位、最硬负例的高方差、event-level 不等权，以及 warm-start 后重置 Adam 造成的优化轨迹断裂。
5. 下一条最强、最稳妥路线应是：**一次连续的 official-init DreaMS 原生训练；在不重置 Adam 的前提下在线重挖当前 candidate winner；按 query 组织 semi-hard、化学特异 hard 和 safety 边界；先保持原生 triplet loss，再用受控 canary 比较 query-listwise multi-negative loss。**

目前不应宣称已经达到5 pp。要在1,975-query role-2面板达到 `+5 pp`，需要99个净纠正；Phase-A只有42个，仍需57个净纠正。若不新增 introduced error，相当于还要纠正151个剩余 official errors 中的37.75%。

## 1. 证据等级

| 结果 | 面板 | Recall@1变化 | 证据含义 |
|---|---:|---:|---|
| historical native best | role 3, 1,929 queries | `+1.8144 pp`，CI `[+0.8155,+2.8703]` | 已确认的 shared embedding 开发结果；不是新outer |
| Phase-A step 2000 | role 2, 1,975 queries | `+2.1266 pp`，CI `[+1.2761,+3.0303]` | 当前最强合格 role-2 开发结果；未独立 role-3 确认 |
| pair-expanded step 2000 | role 2, 1,975 queries | `+2.2785 pp` vs official | 最高未入选 R@1 点；安全门失败，不能替代 Phase-A |
| cross-view consensus teacher | held confirmation, 2,413 queries | candidate-side `+1.9478 pp` | 证明跨谱化学证据有候选裁决能力；不是 embedding 结果 |
| V7/V8 true-support continuation | role 2 | 均低于 Phase-A | 合格负结果；说明“更多当前硬边界”不自动转成更好 embedding |
| V9 exact boundary | construction only | 23 queries / 58 events | 覆盖停止；没有训练，不是性能负结果 |
| V10 cross-view direct | local smoke only | 4/20 errors proven | 工程通过；全量覆盖和性能仍未知 |

## 2. Phase-A 到底做了什么

Phase-A 从 official checkpoint 出发，严格复用 DreaMS 的：

- `ContrastiveSpectraDataset`；
- `ContrastiveHead`；
- 共享 backbone 和 projection head；
- cosine triplet hinge；
- Adam；
- `lr=5e-6`、margin `0.1`、batch size `4`；
- 端到端更新全部 backbone 参数。

唯一改变的是训练 triplet 分布。训练池共5,956个事件：

| 角色 | 事件数 | 占比 |
|---|---:|---:|
| official-correct query 的 max-boundary safety | 3,605 | 60.53% |
| official-error 的 official hardest boundary | 777 | 13.05% |
| official-error 的 ChemAware boundary | 550 | 9.23% |
| 未修改的 DreaMS 10-ppm replay | 1,024 | 17.19% |

训练覆盖4,032个 query、2,518个公式、427个 official training errors。每个 focused event 都是单一 `(q,p*,n*)`，其中 `p*` 和 `n*` 是 official embedding 下定义候选最大分数的精确 reference。

原生损失是：

`L_tri(q,p,n) = [0.1 + cos(z_q,z_n) - cos(z_q,z_p)]_+`。

部署时 molecule candidate 的分数是：

`S(q,c) = max_(r in R(c)) cos(z_q,z_r)`。

Phase-A 的关键贡献，是第一次让 focused triplet 中的 `p*`、`n*` 与这个 molecule-max 决策边界一致，同时保留大规模 safety 和官方 replay。

## 3. 为什么 Phase-A 能成功

### 3.1 监督只规定身份顺序，不蒸馏候选教师数值

ChemAware 负责选择错误候选；真正进入梯度的标签仍然是“同一 InChI connectivity positive / 不同 identity negative”。这避免把 candidate-conditioned、非对称、可能不可由单谱观测的教师分数硬塞进共享对称 embedding。

### 3.2 它优化了真正决定 top-1 的 reference pair

旧池把同一 candidate 的全部 references 存入集合，再随机抽一个 negative。Phase-A 固定当前 candidate maximum，消除了最直接的 reference sampling mismatch。

### 3.3 它的比例恰好同时提供纠错和保护

错误边界占22.28%，safety加replay占77.72%。这不是普遍最优常数，但在当前数据和2,000步下形成了有效平衡。错误 query 平均有3.108个事件，正确 query通常只有一个 guard，所以困难 query 得到加密，同时绝大多数优化暴露仍用于稳定全局几何。

### 3.4 2,000步是暴露预算最优，不是收敛

5,956个事件、batch size 4，每个 epoch 正好1,489步。2,000步只相当于：

- 全部事件完整看过1次；
- 第二轮只看2,044个事件，即34.32%的随机子集。

这解释了轨迹的非单调性：step 2,500 的 micro-AUC 更高，但 Recall@1显著更低。继续降低 triplet loss 并不等于继续改善 top-1。

### 3.5 它主要解决了“不要破坏”，而不是扩大纠错覆盖

在 role 2：

| 模型 | corrected vs official | introduced vs official | 净纠正 |
|---|---:|---:|---:|
| stage 1 | 54 | 24 | 30 |
| Phase-A | 54 | 12 | 42 |

所以 Phase-A 的优势来自12个更少的 introduced errors。这个事实也解释了为什么后来“多加纠错 triplet”的方案很容易失败：它们增加 gross corrections 的同时又重新打开了错误引入通道。

## 4. 必须纠正的“最佳”表述

Pair-expanded step 2000 的 role-2 Recall@1 是0.918987，对 official 为 `+2.2785 pp`，高于 Phase-A 的0.917468和 `+2.1266 pp`。但：

- pair-expanded corrected/introduced 为70/25，Phase-A为54/12；
- pair-expanded Recall@3 为0.988354，Phase-A为0.991392，少6个命中；
- mean positive margin 为0.31026，Phase-A为0.37009；
- cosine-to-official 为0.67983，Phase-A为0.79529；
- 相对 stage-1 的 corrected/introduced 为38/23，`corrected-2*introduced=-8`；
- formula-cluster 增量区间跨零。

因此：

> Phase-A 是当前最强的合格平衡解；pair-expanded 是更高但不稳定、未通过安全门的单指标点。

这一区分不能再混淆。

## 5. 后续扩容为什么没有超过 Phase-A

### 5.1 Pair-expanded：有效梯度多了，但独立决策信息没有同比增加

它把训练池扩到9,957个事件，step 2000 纠正更多错误，但 introduced 同时翻倍以上，并损害 Recall@3。说明它增加了能产生梯度的 spectrum pairs，却没有同等增加互不冲突的 candidate decisions。

文献也明确指出：最硬负例并非总是最好。单位球高维 embedding 中，过近负例的梯度方向容易被噪声主导；随机负例又往往太容易，因此需要控制难度而不是追求 hardest-only。[Sampling Matters in Deep Embedding Learning](https://openaccess.thecvf.com/content_ICCV_2017/papers/Wu_Sampling_Matters_in_ICCV_2017_paper.pdf)

### 5.2 Specific replay：化学比例提高了，但覆盖仍集中在极少数 query

最佳 step 500 相对 stage-1 只有 `+0.3038 pp`，19/13，未过门。化学事件更“纯”并没有解决独立 query coverage，也没有解决 pairwise/listwise 错位。

### 5.3 Multi-condition：扩展条件覆盖，却重新训练出一个更不稳定几何

它从 official 重新训练，扩到6,602 anchors、11,280 events。step 3,000 仍比 Phase-A少1个净 query；相对 Phase-A 是27 corrected / 28 introduced。说明“多条件”本身有信息，但现有 event distribution 无法同时重建 Phase-A 的保护性。

### 5.4 V6 residual：当前错误是对的对象，但训练轨迹断裂

V6 从 Phase-A 权重冷启动新 Adam。step 100 已低于 Phase-A 5个净 query；所有后续点都没有恢复。它同时重放大量已经完成使命的 Phase-A events，fresh Adam 会再次推动这些边界，并与 residual correction 冲突。

### 5.5 V7/V8 true support：化学证据被 identity broadcast 稀释

V8 的4,425个 correction events 中：

- 只有1,565个与化学证据来自同一个 query；
- 2,860个是 identity broadcast；
- 只有803个同时保持 source query 和 contrasted candidate。

它最好的 step 500 相对 Phase-A仍为10 corrected / 11 introduced。数量的64.63%来自监督单位迁移，正是“数量增加但信噪比下降”。

### 5.6 V9：完全对齐后，证据只剩23个独立 query

V9 删除 identity broadcast 后，151个当前 errors 中只有23个 query、23个公式具有 same-query、same-current-false proof，共58个事件。因此它正确地停在训练前。这里暴露的是化学证据覆盖瓶颈，而不是代码问题。

### 5.7 V10：跨谱证据是合理方向，但仍不能单独解决目标错位

cross-view consensus teacher 在2,413-query held confirmation 上候选侧提高 `+1.9478 pp`，而 reversed control 为负，证明跨谱化学证据有真实方向性。但本地 V10 smoke 的20个当前 errors 只保留4个严格证明。它可以提供高精度 chemical negative，不能独自承担全局训练课程。

## 6. 当前区分能力不足的根本原因

### 6.1 Pairwise训练目标与candidate-list top-1不完全同构

当前每步只比较一个 positive 和一个 negative；正式决策却是：

`S(q,c+) > max_(c != c+) S(q,c)`。

一个 pair 被推开，不代表真正的当前 winner 被推开，也不代表第二、第三候选不会接替成为新 winner。Smooth-AP、SoftSort和contextual metric learning的共同启示，是 retrieval 应尽量在候选集合层面建模，而不是把所有信息压成互相独立的 triplet。[Smooth-AP](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123540647.pdf)、[SoftSort](https://proceedings.mlr.press/v119/prillo20a.html)、[Contextual Similarity Optimization](https://proceedings.mlr.press/v202/liao23b.html)

### 6.2 Reference maximum 会随参数变化，Phase-A pair 是静态的

Phase-A 的 `p*`、`n*` 在 official geometry 下正确；训练后 `argmax_r` 可能切换。静态 pair 继续满足 margin 后就不再提供纠错梯度，或者开始优化已经不决定名次的 reference。

### 6.3 原生 triplet 对正负相似度施加等强度梯度

源码中 loss 对 `s_p` 和 `s_n` 是对称的 `m-s_p+s_n`。但真实边界可能是“positive已经足够高，只需降低一个 false candidate”，或者相反。Circle Loss 与 Multi-Similarity 的核心正是根据每个 pair 的优化状态分配不同权重。[Circle Loss](https://openaccess.thecvf.com/content_CVPR_2020/html/Sun_Circle_Loss_A_Unified_Perspective_of_Pair_Similarity_Optimization_CVPR_2020_paper.html)、[Multi-Similarity Loss](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Multi-Similarity_Loss_With_General_Pair_Weighting_for_Deep_Metric_Learning_CVPR_2019_paper.pdf)

### 6.4 Event均匀不等于query均匀

错误 query 平均3.108个 focused events，正确 query通常1个。扩容方法又使少数 identity/query 拥有数十个相关 pairs。uniform DataLoader 实际按“事件数”赋权，导致证据丰富的 query 支配梯度。

### 6.5 化学教师看到的变量多于部署encoder

候选结构、其它同身份谱、候选特异 action 都是训练期 privileged information。它们可以很好地选择 negative，但不能保证单张 query spectrum 足以恢复完整裁决。DreaMS 原论文也指出其未微调 embedding 对近质量小结构差异敏感性不足，并通过同身份/近质量异身份 triplet 改善；这支持 hard-negative curriculum，但不证明任意候选规则都能编码到单向量。[DreaMS](https://www.nature.com/articles/s41587-025-02663-3)

### 6.6 所有 post-Phase-A continuation 都存在 fresh-Adam confound

受保护 checkpoint 是 weights-only。加载后重新创建 Adam，动量和二阶矩全部归零。因此“从Phase-A继续”并不是原训练轨迹的连续课程。V6/V7/V8 在50–100步即退化与此一致，但尚无保持 optimizer state 的配对实验，所以这仍是待验证机制，不能单独归因。

### 6.7 评估指标本身接近饱和

role 2 official 的 miss 数：R@1=205、R@3=31、R@5=12、R@10=4、R@20=0、R@50=0。Phase-A 后为163、17、8、4、0、0。因此 R@20/R@50 已无提升空间，R@10最多只能提高4/1975=`0.2025 pp`。要求所有 Recall@k 都提高3–5 pp在这个面板上数学上不可能；应该报告误差减少比例和保持不退化。

## 7. 文献审计后的方法裁决

### 7.1 应保留

- DreaMS 原生 shared encoder、cosine similarity、端到端更新；
- 真实 identity positive/negative；
- molecule-max reference alignment；
- safety boundaries 和官方 replay；
- formula-disjoint construction/evaluation；
- chemical rule 只负责 negative selection 或可信度，不作为连续教师 target。

### 7.2 应立即改变

- 从固定 official hard negative 改为当前 checkpoint 的 online winner；
- 从 hardest-only 改为“winner + semi-hard band + chemical-specific”组合；
- 从 event-equal 改为 query-packet equal；
- 从 weights-only 冷启动 residual 改为单次训练内 remine，保留 Adam state；
- 从一个 pair 的损失统计改为每个 query 的 candidate-set 统计；
- checkpoint按冻结 retrieval ledger选，不按 train loss选。

### 7.3 不应直接照搬

- XBM/超大负例队列：它能扩大 hard-negative搜索，但本项目的问题不是“找不到负例”，而是 hardest/false/conflicting negatives过多；若用，只能存 candidate-valid、formula-compatible references。[Cross-Batch Memory](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Cross-Batch_Memory_for_Embedding_Learning_CVPR_2020_paper.pdf)
- PCGrad/CAGrad：可诊断 correction/safety/replay 梯度冲突，但随机梯度操纵并不自动保证收敛；先测梯度 cosine，再做小型消融，不能直接成为主线。[PCGrad](https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html)、[CAGrad](https://proceedings.neurips.cc/paper/2021/hash/9d27fdf2477ffbff837d73ef7ae23db9-Abstract.html)
- SAM：会增加优化开销，而且近期工作显示其可能压低 feature rank；对已有共享 embedding 未必合适，不是当前最高性价比。
- 继续扩大 identity-broadcast triplet：V8 已经给出负证据。

## 8. 下一代主方法：连续在线候选集课程

这是当前优先级最高、与已有证据最一致的方法。

### 8.1 单次优化轨迹

从 official DreaMS 启动一次 Trainer 和一次 Adam：

1. step 0–1,500：精确 Phase-A curriculum；
2. step 1,500：对 roles 0/1 全部训练 query 重新编码并重挖；
3. step 1,500–2,500：每250步刷新 candidate set，但不重置模型或 Adam；
4. 固定保存1,500/1,750/2,000/2,250/2,500；
5. role 2只做一次预注册选择；未通过不打开 role 3。

### 8.2 每个 query 的训练 packet

对 query `q`，用当前模型构造：

- `p*`：当前 true candidate 的 max reference；
- `n_win`：当前最高分 false candidate 的 max reference；
- `n_semi`：位于 margin band 内、但不是极端最近的一个 semi-hard negative；
- `n_chem`：若存在，加入 same-query/candidate-aligned、correct-vs-null specificity 为正的 ChemAware negative；
- `g_safe`：若 query 当前正确，保留最近边界 guard。

每个 query 先对 packet 内 loss 求均值，再对 query 求均值。化学只决定 `n_chem` 是否进入，不广播到其它 query，不改变 identity label。

### 8.3 第一阶段仍保持原生 triplet loss

`L_q = mean_(n in packet(q)) [0.1 + s(q,n) - s(q,p*)]_+`。

这只改变采样组织和 remine 时机，不改变 DreaMS encoder、cosine 或 identity supervision。它是风险最低的主实验。

### 8.4 第二阶段受控比较 candidate-listwise loss

若原生 packet 版本仍停在约2 pp，则在相同 candidate packets 上比较：

`S_tau(q,c) = tau_r log sum_(r in R(c)) exp(cos(z_q,z_r)/tau_r)`

`L_rank(q) = log(1 + sum_(c != c+) exp((S_tau(q,c)-S_tau(q,c+)+m)/tau_c))`。

它有四个优势：

1. `S_tau` 是 molecule-max 的平滑近似；
2. 一个 query 同时看到多个 negatives；
3. 每个 query 贡献一次，避免 event multiplicity偏置；
4. 目标直接压低所有可能接替 winner 的候选。

SupCon 证明多正多负的 batch contrastive 通常优于单 triplet/max-margin；Smooth-AP和contextual metric learning支持直接对 retrieval ranking建模。[Supervised Contrastive Learning](https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html)

### 8.5 保护机制

先使用现有 replay 与 safety packet。若梯度审计仍显示系统性冲突，再加入一个小的 L2-SP：

`L_total = L_rank + lambda ||theta-theta_official||^2`。

L2-SP 是对起始权重的直接正则，不是教师蒸馏；文献建议它作为小数据 fine-tuning 的保留基线。[Explicit Inductive Bias for Transfer Learning](https://proceedings.mlr.press/v80/li18a.html)

## 9. 最小决定性实验矩阵

只允许四个臂，共享同一 official 初始化、seed、candidate graph、训练 query 和总 optimizer steps：

| Arm | 目的 |
|---|---|
| A | 精确 Phase-A复现，作为同步基线 |
| B | 连续 Adam + online remine + query packets + 原生 triplet |
| C | B，但 `n_chem` 用 matched null candidate替换 |
| D | B 的 candidate packets + query-listwise loss |

选择顺序：

1. B必须先安全超过A；
2. B必须以 formula-cluster paired CI 超过C，才能归因化学；
3. D只有在 Recall@1、MRR、Recall@3、micro/macro AUC 全部不劣于B时才升级；
4. role 2选择后只打开一次 role 3；
5. 新 external/generalization panel 不得参与方法选择。

必须记录：

- corrected / introduced / corrected-2*introduced；
- current winner coverage；
-每个query的active negative数；
- correction、safety、replay三类梯度 cosine；
- embedding cosine-to-official；
- candidate winner switch rate；
- identity-equal与query-micro两套指标；
- formula-cluster bootstrap CI。

## 10. 更大评测集

当前1,975-query role-2面板适合 checkpoint selection，但不足以证明跨条件稳健性。仓库已有 full-role 评估入口，可以在冻结模型后报告全部 role-2/3 query，同时以 identity-equal 指标为主，避免多谱 identity 支配 micro metric。

真正外部泛化应使用新的、未参与 triplet 构造的数据源。MassSpecGym提供 molecule retrieval、生成和谱图模拟标准任务及结构泛化 split；其2026年审计又提示必须锁定 corrected split、checkpoint和metric实现，避免泄漏、shortcut和metric divergence。[MassSpecGym 2024](https://papers.neurips.cc/paper_files/paper/2024/file/c6c31413d5c53b7d1c343c1498734b0f-Paper-Datasets_and_Benchmarks_Track.pdf)、[MassSpecGym in the Wild 2026](https://arxiv.org/abs/2606.19624)

外部评测只能在方法冻结后进行，不能拿来继续调参。

## 11. 最终科研判断

Phase-A 的成功不是因为“化学 triplet越多越好”，而是因为它把有限化学信息放在正确的检索边界上，同时用大量 guard/replay抑制破坏。它的瓶颈也非常清楚：只优化静态单pair，无法持续追踪 candidate winner；而后续扩容把“更多梯度”误当成“更多独立纠错信息”。

要从2 pp向5 pp推进，核心不是再挖几千个 triplet，而是让每次更新都满足三件事：

1. 当前确实决定候选名次；
2. 在同一 query 的候选集合内有可验证的化学方向；
3. 不与正确 query 的保护梯度冲突。

当前最有胜算的实现是“连续Adam轨迹 + online candidate winner remine + query-balanced multi-negative packets”；candidate-listwise loss是其后的决定性算法升级，而不是从头再造 adapter 或做教师蒸馏。

## 12. 可复算资产

- Phase-A冻结账本：`docs/CHEMAWARE_PHASEA_2PP_RUN_2343962_FROZEN_LEDGER.json`
- Phase-A池：`data/validation/chemaware_phasea_pool_rebuild_local_20260927/train_pool.npz`
- 本次审计程序：`tasks/audit_chemaware_phasea_discrimination.py`
- 本次审计输出：`data/validation/chemaware_phasea_discrimination_audit_20260927/report.json`
- Phase-A构造器：`tasks/build_chemaware_max_boundary_native_triplets.py`
- DreaMS原生训练入口：`tasks/train_chemaware_dreams_native.py`
- V8根因审计：`docs/CHEMAWARE_EXACT_BOUNDARY_RESIDUAL_V9_20260927.md`
- V10直接跨谱triplet：`docs/CHEMAWARE_CROSSVIEW_BOUNDARY_DIRECT_V10_20260927.md`
