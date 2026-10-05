# 中期报告逐句仓库核对与逻辑纠偏

日期：2026-09-30  
核对对象：用户提供的《基于自监督学习的非靶向代谢组学 LC–MS/MS 代谢物注释方法研究》中期报告文本  
裁决原则：以仓库中已冻结协议、结果账本、负结果、角色隔离和 claim limit 为准；不以叙事流畅性替代证据等级。

## 一、总裁决

这份报告的主要问题不是“多数数字造假”，而是**把不同数据角色、不同算法形态、不同时间阶段和不同证据等级的真数字，拼接成了一个已经闭环的串行算法**。与此同时，它又漏掉了 P2b 等真正有封存正证据的后融合方法，使全文同时出现“错误合并”和“关键遗漏”。这会造成比单个数字抄错更严重的科学误解。

报告暗示的主线是：

> 自监督模型进一步训练 → 局部化学重排 → 样本生化关系重排 → 完整注释流程。

仓库实际形成的是五个相互关联、但尚不能串成同一个累计性能管线的层次：

1. **协议与候选图底座**：corrected development graph、统一候选聚合、公式/身份隔离、聚类置信区间和停止门。
2. **共享 encoder 路线**：Noise 与 ChemAware native training；推理时只用谱图，但训练监督并非纯自监督。
3. **候选条件重排路线**：ChemAware residual/rule reranker、P2b、BioAware B30；它们读取候选侧或分支侧信息，不能与共享 embedding 混称一个模型。
4. **样本局部事件路线**：BioAware B47 U3；目前证明的是候选特异事件产额超过结构化空模型，不是注释准确率提高。
5. **真实队列应用路线**：MTBLS13729 与 LCNEC；已形成证据校准的候选和生物学假说，但精确新代谢物声明仍为 0，关键身份仍待同法标准品。

因此，报告应从“已形成三段式注释算法”改写为：

> 项目已建立统一、可审计的候选检索与评价底座，并在共享编码器、候选条件重排、样本局部事件证据和真实队列应用四条互补路线取得不同成熟度的结果；这些结果尚未组成经过同一独立外部面板验证的端到端累计性能系统。

## 二、最根本的九处逻辑错位

### 1. “自监督”被扩大成了整个后续方法的训练性质

DreaMS backbone 的来源是自监督预训练，但报告中的主要正结果使用了已知分子身份、正确参考谱和错误候选：

- Noise Stage-1 是 identity-supervised native triplet；
- Noise T1/T3 是 multi-relation hinge 加 molecule-level listwise softmax；
- ChemAware native triplet 由化学证据选择 hard negative，但损失仍是身份监督 triplet；
- ChemAware residual reranker 和 BioAware B30 是候选重排器。

所以，课题题目可以保留“基于自监督学习”，正文必须写成“**以自监督预训练的 DreaMS 为基础，进行身份监督的对比微调和候选条件重排**”。否则会把预训练来源与下游监督方式混为一谈。

### 2. 把“共享 embedding”和“候选重排器”当成同一种算法

共享 encoder 在推理时只需要谱图并改变全部 query/reference 的 embedding；候选重排器则在冻结或既定谱学分数之上读取候选侧证据并改变候选顺序。两者的输入、部署方式、可迁移性和基线均不同。报告把 Noise 94.24%、ChemAware 94.40% 和 BioAware 78.30%并列成一个流程，读者会自然误以为三者可顺次叠加；仓库没有这样的累计实验。

### 3. 把开发面板上的最优点写成了已完成方法

ChemAware 94.40% 是 1,929-query role-3 上的 post-outer 开发重排结果；Noise 94.24% 是 corrected held fold-0 上预注册主种子的最高共享 encoder 点估计；BioAware 78.30% 是 1,631-query 的打开多来源开发协议。三者均不是“同一独立外部基准上的三个模块”。GNPS Gold/Silver 已封存，但仓库总览仍明确记载统一外部数值尚未产出。

### 4. 把不同阶段的对照差拼成同一条因果解释

Noise 的 `+1.26548 pp` 是 Stage-1 targeted 相对 matched control 的 action-content 因果差；最高 94.24%来自随后 T1/T3 continuation。T1/T3 相对 Stage-1 主种子再增 `+0.55637 pp`，但该阶段没有保留逐动作 exact boundary，仓库明确禁止把这 `+0.556 pp` 单独归因于 Noise action content。报告却用前一阶段的 1.27 pp 对照来解释后一阶段的 1.05 pp 累计结果。

### 5. 把新 MassBank 规则库与旧 ChemAware 94.40%结果建立了不存在的血缘关系

16,272 条规则是 2026.03 MassBank 分层来源的独立证据语料；94.40%则来自更早的多零对称残差 V2 重排器。二者分别成立，但后者不是“加入这 16,272 条规则后”的结果。正确写法必须拆成两个结果，不能前后相接制造因果暗示。

### 6. 把“事件产额”提升成“生物信息能够改善注释”的中间结论

B47 U3 的 4,996 是 candidate-specific queries，2,239 是 potential ranking-change opportunities。U3 没有打开 annotation truth，没有 corrected/introduced，也没有 accuracy；且有效 seed identity、候选 identity/formula 集中度等门失败，`pass_to_frozen_event_ranking_evaluation=false`。因此它证明的是“在当前冻结构造中，真实样本局部精确反应事件多于两类结构化 null”，不能写成已经证明 BioAware 能提高外部注释准确率。

### 7. 把 corrected development graph 称为一个整体“训练集”

83,619 queries 是 corrected development graph 总分母，内部存在 outer held fold、公式角色和开发/确认/outer 使用历史。把全部 83,619 都称为“训练查询”会掩盖角色隔离。建议统一称“corrected candidate development graph”，具体训练语料另报 48,544 queries 或相应 role 池。

### 8. 把真实队列工作压缩为“候选生成”，反而漏掉项目最扎实的应用贡献

MTBLS13729 已不只是对齐与候选生成，而是形成 phenotype-blind 注释、定量、外部糖组与组织学证据校准的 hybrid mucin glycome remodeling 假说；LCNEC 已完成 34 对患者、QC 漏斗、跨平台背景和四个 Level-2/connectivity 候选。报告既不能把这些候选写成确认身份，也不应把现有应用闭环降格成“正在开始候选生成”。

### 9. 报告漏掉 P2b 和近期 RRF，导致“重排器”谱系残缺

P2b 是项目当前证据等级最高的算法性后融合结果之一：在预冻结 P3-main 3,000-query 面板上，Recall@1 从 87.93%提高到 89.00%，`+1.07 pp`，89 corrected/57 introduced，formula-cluster CI `[+0.24,+1.89] pp`，McNemar `p=0.0101`。它不更新 encoder，而以固定权重融合 DreaMS、entropy 和 neutral-loss similarity。报告完全漏掉 P2b，却用 ChemAware role-3 开发重排器代表全部“局部化学信息”，使证据等级倒挂：E2-dev 的大点估计被突出，E3 sealed 的较小但更硬结果被删除。

近期三通道 RRF 也应记录，但资格必须写完整。它在 official-geometry held 探针上得到 `94.954%`、425 corrected/101 introduced、risk-net `+223`，相对该探针的 re-encoded official ledger 净 `+324/18,333 = +1.767 pp`。这个结果证明 raw cosine 与 top-k fragment-presence ranking 含有 DreaMS 未完全编码的互补信息；但固定 RRF 叠加 Stage-1 并在 GNPS 进行唯一确认时失败，`reranker_promotion_authorized=false`。因此它是**正向机制发现 + 确认失败边界**，不是可部署的新一代重排器性能。

这两项恰好说明为什么全文必须把两条轴分开：

| 轴 | 改变什么 | 推理输入 | 代表方法 | 正确性能口径 |
|---|---|---|---|---|
| W：shared embedding | 更新同一个 DreaMS encoder，使 query/reference 全部重新编码 | 谱图 | Noise Stage-1/T1/T3；ChemAware native/Phase-A | 相对 official 或前一共享 checkpoint 的 held/role 增益 |
| X：post-embedding reranking | 保持 embedding 不变，在每个 query 的候选集合上融合局部证据 | 谱图分数、峰/中性丢失、候选侧证据或排名 | P2b；ChemAware V2；RRF；BioAware B30 | 同一冻结候选集合内的 corrected/introduced、risk-net 和 paired CI |

W 与 X 可以在未来形成部署组合，但仓库当前没有一个已通过独立确认的统一 `W+X` 累计系统。不能把 W 的 `+1.05 pp`、P2b 的 `+1.07 pp`、V2 的 `+3.94 pp` 或 RRF 探针的 `+1.767 pp`相加。

#### 9.1 原报告遗漏或弱化的正向结果总表

以下结果应按资格呈现，而不是只挑最大百分数：

| 路线 | 正向结果 | 资格与正确位置 |
|---|---|---|
| Noise Stage-1 shared encoder | 对 official `+0.49637 pp`；targeted 对 matched control `+1.26548 pp`，CI 严格为正 | W / action-content 因果信号与绝对 encoder 增益分开报告 |
| Noise T1/T3 shared encoder | 主种子对 Stage-1 `+0.55637 pp`；absolute 94.2399%，累计对 official 约 `+1.0527 pp` | W / 当前最高 absolute Noise encoder 点估计；复证种子同向但 CI 略跨 0 |
| ChemAware native shared encoder | role-3 `+1.8144 pp`，57/22，CI `[+0.8155,+2.8703]` | W / 已确认开发证据；outer 与化学独立归因未闭环 |
| ChemAware Phase-A shared encoder | role-2 `+2.1266 pp`，54/12，CI `[+1.2761,+3.0303]` | W / protected development；尚未通过 role-3/outer |
| P2b fixed fusion | sealed P3-main `+1.07 pp`，89/57，CI `[+0.24,+1.89]` | X / E3；当前最硬的封存重排正结果，同时 near-core `-4.23 pp` |
| ChemAware V2 residual reranker | role-3 dev `+3.9399 pp`，93/17；candidate-rotated contrast `+3.3178 pp`且 CI 正 | X / E2-dev；无新独立确认，不是 embedding |
| ChemAware direct-first + backoff | role-3 dev `+3.7325 pp`，78/6 | X / post-outer development；应作为 V2 的稳健回退谱系报告 |
| ChemAware rule-mass / mass probes | rule-mass `+1.2442 pp`，32/8，CI 正；mass `+0.881 pp`，CI 正 | X / E1 机制探针；支持“化学先验作为核权重”的方向，不是主性能 |
| RAW-v1 bounded residual | dev `+4.35 pp`；P1 选择性 `+0.0355`且 CI 正 | X / 历史开发资产；Test-A/B 增益小且 CI 跨 0，不得升格 |
| 三通道 RRF | official-geometry 探针净 `+1.767 pp`，94.954%，425/101 | X / 互补信息机制证据；Stage-1/GNPS 确认失败，永久不升格 |
| BioAware B30/B35 reranker | 1,631-query overall `+2.88 pp`，50/3；合格负离子去重子集 `+6.24 pp` | X / 打开多来源开发证据；不得称独立外部 |
| BioAware B47 U3 event layer | 4,996 candidate-specific queries；2,239 opportunities；null lift `+22.69%` | 事件层 / 真值盲机会证据；集中度门失败，无 accuracy |
| 真实应用 | MTBLS13729 hybrid mucin glycome；LCNEC 四个 Level-2/connectivity 候选与跨组学背景 | 应用层 / 可验证假说与优先级，不是确认新代谢物 |

这张表比“Noise—ChemAware—BioAware 三段式”更忠实，因为同一个命名空间内部也可能同时包含 W 层和 X 层。例如 ChemAware native/Phase-A 是 embedding，ChemAware V2/direct-first 是 reranking；不能因它们都叫 ChemAware 就写成同一种算法。

## 三、按原报告行号逐句裁决

下表只审查具有项目事实含义的句子；封面、学校模板和经费空表不作科学裁决。

| 原行 | 原意 | 裁决 | 仓库核对与必须修改 |
|---:|---|---|---|
| 10 | 课题为“基于自监督学习……” | **可保留题目，正文需限定** | DreaMS 是自监督预训练底座；当前主要增量算法含身份监督 triplet/listwise 和候选重排，不能统称为自监督训练。 |
| 36 | 非靶向代谢组学注释率低、MS/MS 结构注释困难 | **保留，但避免无来源的“90%以上”** | 作为背景判断成立；若正式提交，“90%以上”需给文献与口径，不能由仓库实验替代外部引用。 |
| 37① | 项目以自监督 MS/MS 模型为核心，综合谱图、结构和样本代谢信息 | **方向正确，架构表达错误** | 应写成四条互补路线，而非一个已串联模型。共享 encoder、候选重排和样本事件层的输入与证据等级不同。 |
| 37② | 多条件真实谱和扰动谱继续训练模型 | **部分正确** | Noise Stage-1/T1/T3 成立；但下游训练是 identity-supervised。ChemAware 原生路线则由化学证据选 hard negative，不应并入“实验条件扰动”一句。 |
| 37③ | 用碎片、中性丢失等化学证据重排候选 | **可保留但需分算法** | 94.40%属于 multi-null residual candidate reranker；16,272 条 MassBank 新规则是另一条尚待完整性能闭环的证据来源。 |
| 37④ | 用样本内代谢物与候选的已知生化关系提供上下文证据 | **可作为 B47 方法定义** | 必须明确当前只得到 event-yield 结果，尚未获准打开排序真值。 |
| 37⑤ | 93.19%→94.24% | **数字成立，解释需重写** | official 约 93.1875%，T1/T3 主种子 94.2399%，累计约 +1.0527 pp；这不是一个单阶段结果。 |
| 37⑥ | 化学信息使 Top-1 错误相对下降约 41% | **数字成立但证据级别被隐藏** | 对应 V2 role-3：90.4614%→94.4012%，93 corrected/17 introduced，错误率相对下降约 41%；它是 post-outer 开发重排结果，不是独立外部确认，也不是共享 embedding。 |
| 37⑦ | 样本代谢信息使 75.41%→78.30% | **历史开发结果成立** | B35/B30 的 1,631-query 打开开发协议成立；不是 B47 外部结果，也不能与 94.24/94.40 串行累计。 |
| 37⑧ | 已形成从自监督学习到真实样本辅助注释的完整流程 | **删除或根改** | 尚无端到端累计实验、统一独立外部基准或标准品终证。改为“建立了统一评价底座和多条互补算法路线”。 |
| 37⑨ | 摘要未报告 P2b | **关键正结果遗漏** | 应至少补入：固定权重后融合在 sealed P3-main 上 `+1.07 pp`，同时 near-core `-4.23 pp`；这是目前最硬的封存重排证据。 |
| 65 | 单一全局谱图相似度不足以解决结构注释 | **保留** | 这是项目合理的问题定义；可加上候选集、参考谱多重性和近结构异构体是具体困难。 |
| 66① | 先通过真实谱和扰动谱进一步训练共享模型 | **保留但注明监督性质与两阶段** | Stage-1 action-specific triplet 与后续 T1/T3 relation-complete continuation 不应压成一个算法动作。 |
| 66② | 再用局部化学证据重排 | **需拆分** | 一条是推理时 candidate reranker；另一条是训练时用化学证据选择 hard negative、推理仍为 spectrum-only encoder。 |
| 66③ | 最后用真实样本代谢关系进一步增强 | **不得写成已串联的“最后一步”** | B47 是单独、真值盲的事件机会层；当前没有被接入并通过排序性能评价。 |
| 67① | 83,619 查询、9,854 身份、6,220 公式、392,229 候选、6,220,661 边 | **数字正确** | “83,619 个训练查询”改为“83,619-query corrected development graph”。 |
| 67② | 78.06%有同公式干扰、33.71%有 MCES-near 干扰 | **正确** | 另一个 75.08%是“负候选边中与真值同公式”的比例，不要与 78.06%混用。 |
| 67③ | 按公式隔离、分子聚合、多指标评价 | **原则正确** | 需增加公式/身份/source cluster CI、并列算失败、开发/确认/outer 角色不可互换。 |
| 68 | 已形成一个综合三类信息的计算方法 | **过度完成化，改写** | 已形成的是研究体系和可审计组件，不是已验证的单一端到端方法。 |
| 70 | 将真实条件变化谱、明确分子对应和同公式难负例用于训练 | **基本正确** | “自监督模型训练”改成“自监督预训练模型上的身份监督微调”。 |
| 71① | 同时比较多张正参考和多个负候选，完整候选集优化 | **对应 T1/T3，成立** | 这是 T1/T3 continuation，不是 Stage-1 native triplet。 |
| 71② | 48,544 queries、186,425 candidates、516,753 references、37,948 same-formula | **正确** | 还应写 complete-candidate queries 为 37,731；该语料来自 outer-train，fold-0 18,333 未训练。 |
| 72① | 18,333 上 93.19%→94.24%，+1.05 pp | **正确但需给阶段链** | official→Stage-1 为 +0.49637 pp；Stage-1→T1/T3 主种子 +0.55637 pp；累计约 +1.0527 pp。 |
| 72② | matched perturbation 对照低 1.27 pp，因此 1.05 pp 来自方向性谱图变化 | **因果拼接错误** | `+1.26548 pp`只证明 Stage-1 targeted action content 优于 matched control；T1/T3 的增量没有完成 action-content 归因，不能据此解释全部 94.24%。 |
| 73 | 过度增加 hard negative 会损伤正确候选和原几何 | **保留** | 与 Stage-3/5、T1/T3 v2 和多个 ChemAware 后继路线的停止结果一致；应给出这是负结果约束，不是泛泛经验。 |
| 76 | 化学证据需候选条件化，不能全局统一加分 | **保留，是核心认识** | 这是 ChemAware 从规则直加分转向 candidate-specific residual/course selection 的关键范式变化。 |
| 77① | 已整合多来源 fragment/neutral-loss 证据并做候选替换等控制 | **部分正确** | 不同实验的控制不能汇成一个未说明版本的“已整合方法”。需逐一标算法版本和面板。 |
| 77② | MassBank 16,272 条候选规则，在公式隔离分析中有稳定区分信号 | **正确** | 这是独立 evidence-corpus 资格结果，不等于后续模型性能。 |
| 78① | 90.46%→94.40%，错误率相对下降约 41%，93/17 | **数字正确** | 明确是 ChemAware multi-null symmetric residual V2、role-3 1,929-query、post-outer development result。 |
| 78② | candidate replacement control 证明候选化学信息起实质作用 | **方向可保留，措辞收紧** | V2 对 candidate-rotated 对照仍 +3.3178 pp 且 CI 正，说明候选条件信号不是纯谱学分数平移；但不能把它归因给新 MassBank 16,272 规则。 |
| 79 | 规则可检测不等于能区分当前竞争候选 | **保留** | 这是仓库中最重要、最可发表的方法学认识之一。 |
| 79后 | 未呈现 P2b 与 RRF 谱系 | **新增独立小节** | P2b 是 E3 sealed 后融合；RRF 是 official-geometry 的 `+1.767 pp`探索性互补性探针，但 Stage-1/GNPS 确认失败。二者都不是 embedding。 |
| 82 | 1,631-query 上 75.41%→78.30%，50/3 | **正确但必须标“打开开发”** | 六来源均非负，但不是独立外部性能；BioAware 后续 B44 外部反转、B45 null、B46 harmful 的负结果不能在中期报告中消失。 |
| 83 | 数据库频率、度数、覆盖可造成假生物学证据 | **正确且必须强化** | U0 的 reference multiplicity、B44–B46 的外部失败是转向 B47 的原因，不应只写成一般风险。 |
| 84 | 仅保留同样本高置信种子、精确反应约束并做结构化 null | **正确描述 B47 U3** | 还需写 truth/phenotype 关闭、query 不得自作 seed、currency/hub 控制和 reaction signature 冻结。 |
| 85 | 51,976 queries、216,793 rows、4,996 candidate-specific；rewired max 91、seed permutation max 4,072、lift 22.69% | **数字正确** | 本地缺服务器 report.json，仓库当前证据来自 2026-09-30 完成账本；提交前应同步并哈希封存服务器工件。 |
| 86① | 真实样本中存在不能由随机网络或覆盖解释的候选特异关系 | **表述过强** | 只能说“超过已注册的 degree-rewired 与 seed-context permutation null”；集中度门失败，不能宣称已排除全部覆盖/枢纽混杂。 |
| 86② | 尚未打开真值，不能解释为 accuracy 提升 | **正确，必须保留并前置** | 再补：2,239 是机会数，不是修正数；`pass_to_frozen_event_ranking_evaluation=false`。 |
| 88 | MTBLS13729/LCNEC 仅处于对齐、候选生成和优先级筛选 | **过时且低估** | 两队列已有成体系应用闭环；应具体报告已完成的证据漏斗、生物学主张和负边界。 |
| 89 | 未有同平台标准品，不把候选写成确定结构或新发现 | **正确** | 继续保留；尤其 MTBLS13729 的位置异构体与 LCNEC 四候选仍为 Level-2/connectivity。 |
| 90 | 有样本信息才启用上下文重排 | **原则正确，现实状态需改** | 当前 B47 尚未获准进入排序评价，不能写成已经部署的条件分支。应写成未来部署合同。 |
| 92–93 | 固定当前方法，在独立数据验证泛化 | **方向正确，但“固定方法”过早** | 需分别冻结 Noise encoder、ChemAware encoder/reranker 和 BioAware event layer；不能冻结一个虚构的总模型。GNPS/MoNA、同法标准品和 role-safe 面板各回答不同问题。 |
| 94 | 扩大 seed/reaction/candidate 覆盖，并比较真实与 null 对排序的贡献 | **正确但需先处理失败门** | 先解决 U3 身份/公式集中度和服务器工件闭环；未过门不得打开真值或启动 embedding。 |
| 95 | 标准品、RT 和 MS/MS 终证关键候选 | **正确** | 应具体列优先级：Neu5Ac；修饰鸟苷竞争标准；乙酰化多胺异构体；酰基肉碱组合；LCNEC quinolinate/竞争身份。 |
| 101 | 完成主要算法，形成完整流程 | **删除“完成”与“完整流程”** | 应写“完成统一协议与若干算法节点，端到端独立验证尚未完成”。 |
| 102 | 候选图和统一评价体系已建成 | **基本正确** | 把“训练查询”改为 development graph，并写明 official baseline、角色隔离和重复消费限制。 |
| 103 | Noise 93.19%→94.24%，matched control 证明增益来自方向性变化而非训练量 | **前半正确，后半过度归因** | 必须拆成 Stage-1 因果结果和 T1/T3 绝对最高点两个条目。 |
| 104 | ChemAware 90.46%→94.40%、41%、93/17，candidate control 证明化学信息有效 | **数字可报，归属需限定** | 写明 residual reranker、role-3 post-outer development、非共享 embedding、非 MassBank-16,272 直接结果。 |
| 105① | BioAware 75.41%→78.30% | **可报历史开发结果** | 同时必须报告 B44–B46 外部负结果，说明为何旧静态目录/拓扑路线已关闭。 |
| 105② | 4,996 外部事件为直接验证样本信息贡献打下基础 | **可保留为基础设施结果** | 必须同句报告 2,239 opportunities、集中度门失败、truth unopened、无 accuracy。 |
| 106 | 真实队列正进行候选生成，关键候选待标准品 | **边界正确、进度过时** | 改为“已完成证据校准应用与优先级排序；精确身份仍待同法标准品”。 |
| 101–106整体 | “主要成果”只写 Noise/ChemAware/BioAware 三块 | **成果分类本身错误** | 应按 W 共享权重、X 后融合重排、B47 事件层和真实应用分栏；P2b sealed、RRF 机制探针、ChemAware native/Phase-A 都应各归其位。 |

## 四、必须补回、原报告却基本隐去的负结果

中期报告模板明确要求如实报告失败。以下负结果不是“旁枝”，而是当前方法设计的因果依据：

1. **BioAware B44–B46**：外部迁移 `-0.56 pp`、coverage-neutral topology `+0.2326 pp`且 CI 跨 0、dual-context `-0.7714 pp`。它们关闭了静态目录 membership、degree/topology、path/diffusion 和 aggregate context ledger 的继续调参路线。
2. **ChemAware 独立性边界**：V2 role-3 点增益虽大，但规范策略 outer 未过预注册 `+3 pp`门；共享 encoder 的化学独立增量尚待 listwise 两臂实验，不能用 curriculum 正结果自动证明“化学内容是原因”。
3. **Noise 归因边界**：Stage-1 action-content 因果信号成立；T1/T3 continuation 的新增 `+0.556 pp`不能单独归因给动作内容。T1/T3 v2 exact-action rotation 为 NO-GO，且暴露 query 剂量与 optimizer step 不匹配。
4. **BioAware B47 排序门失败**：真实事件产额超过 null，但身份/公式集中度不足，未获准揭盲排序。
5. **应用终证缺口**：MTBLS13729 full exact-FDR10 为 0；LCNEC 精确新代谢物声明为 0；二者均缺关键同法标准品。

这些结果应作为“研究如何从错误假设收敛到当前可辩护方法”的主线，而不是在正结果之后用一句“仍有局限”带过。

## 五、建议采用的正确论文/中期报告主线

### 5.1 一句话主张

> 本项目围绕可信的非靶向 LC–MS/MS 代谢物注释，建立了严格角色隔离的候选检索与评价体系；在此基础上，分别验证了实验变化感知的共享编码器训练、候选条件化化学重排和样本局部精确反应事件构造，并在两个真实队列中形成证据校准的候选与可验证生物学假说。现阶段最强证据是开发/确认面板上的约 1–4 pp 增益及真值盲事件富集，端到端独立外部性能和 Level-1 身份仍待完成。

### 5.2 结果应按“问题—算法—证据—边界”组织，而非按三个模块顺次叠加

| 结果节点 | 应报告的核心数字 | 可以声称 | 不可以声称 |
|---|---|---|---|
| 统一候选图 | 83,619 queries；official R@1 92.876%；5,957 errors | 建立了统一、困难且可审计的开发分母 | 这是外部测试集；83,619 都参与训练 |
| Noise Stage-1 | official +0.49637 pp；targeted-control +1.26548 pp，CI 正 | action content 有因果优势；共享 encoder 有小幅绝对提升 | 当前 checkpoint 比 official +1.265 pp |
| Noise T1/T3 | 94.2399%；对 Stage-1 +0.55637 pp；累计约 +1.0527 pp | 当前最高共享 encoder 点估计 | 全部增益由方向性 Noise action 导致 |
| ChemAware Phase-A encoder | role-2 +2.1266 pp，54/12，CI `[+1.2761,+3.0303]` | 当前更高的 protected development shared-weight 点 | 已通过 role-3/outer；化学独立归因已完成 |
| ChemAware native | role-3 +1.8144 pp，57/22，CI 正 | 化学筛选课程可改善共享 encoder | 化学内容的独立增量已闭环；已外部确认 |
| P2b frozen fusion | P3-main +1.07 pp，89/57，CI `[+0.24,+1.89]` | 当前最稳妥的 sealed reranker 正结果 | encoder gain；解决 near-core；与 W 层直接相加 |
| ChemAware V2 reranker | role-3 +3.9399 pp，93/17；candidate-rotated contrast +3.3178 pp | 候选条件重排有强开发信号 | 是 16,272 MassBank 规则训练结果；是外部 SOTA |
| 三通道 RRF | 探针 94.954%；425/101；相对探针 ledger +1.767 pp | raw fragment ranks 与 DreaMS 有互补信息 | 已确认或可部署的新 reranker；相对 canonical official 跨账本作差 |
| BioAware B30/B35 | 1,631-query +2.88 pp，50/3 | 打开多来源开发协议上的低风险增益 | 独立外部泛化；B47 的结果 |
| BioAware B47 U3 | 4,996 candidate-specific；2,239 opportunities；null lift +22.69% | 真值盲精确事件结构超过已注册 null | accuracy 提升；2,239 corrections；已获准揭盲 |
| MTBLS13729/LCNEC | hybrid mucin glycome；四个 LCNEC Level-2/connectivity candidates | 可审计候选、负边界和实验优先级 | 确认新代谢物；机制/通量已证实 |

## 六、可直接替换原摘要的版本

> 非靶向代谢组学的核心瓶颈，不仅是谱图相似度不足，还包括近质量/同分子式候选竞争、参考谱多重性、候选侧化学证据的条件依赖，以及真实样本上下文中的覆盖与枢纽混杂。本项目以自监督预训练的 DreaMS 谱图编码器为基础，建立了包含 83,619 个查询、392,229 个候选分子和 6,220,661 条 query–reference 关系的 corrected development graph，并通过公式/身份隔离、分子级参考谱聚合和聚类置信区间统一评价协议。在共享编码器路线中，Noise action-specific native triplet 相对官方 DreaMS 提高 Top-1 0.50 个百分点，targeted action 相对同查询等剂量 matched control 提高 1.27 个百分点；进一步采用 multi-relation hinge 与 molecule-level listwise 目标后，主种子 Top-1 达到 94.24%，相对官方累计提高约 1.05 个百分点，但后续增量尚不能全部归因于 action content。ChemAware 化学 hard-negative 原生微调在 role-3 提高 1.81 个百分点。与共享编码器分开的后融合路线中，固定 P2b 在 sealed P3-main 上提高 1.07 个百分点，但在 near-core 上降低 4.23 个百分点；多零对称残差重排器在 role-3 开发面板上由 90.46%提高至 94.40%，但尚无新独立确认。三通道 RRF 在 official-geometry 探针中显示约 1.77 个百分点的互补性信号，但在 Stage-1/GNPS 确认中失败，仅保留为机制证据。BioAware 在 1,631-query 打开开发协议中由 75.41%提高至 78.30%，其外部静态目录/拓扑路线随后因迁移失败而关闭；新 B47 路线在两个外部研究的真值盲分析中获得 4,996 个候选特异查询和 2,239 个潜在排序机会，事件产额超过结构化空模型，但集中度门未过，尚未打开注释真值或计算准确率。项目已在 MTBLS13729 和 LCNEC 队列形成证据校准的候选与生物学假说；关键结构仍待同法标准品、保留时间和完整 MS/MS 终证。综上，本阶段完成的是统一评价体系和多条互补算法路线，而非已经外部验证的单一端到端注释流程。

## 七、提交前不可妥协的修订清单

1. 全文删除或改写“已形成完整流程”“主要算法开发完成”“三层依次提高注释”等句子。
2. 所有百分数后补上：算法版本、数据面板、角色、基线和是否独立外部。
3. Noise 结果拆成 Stage-1 因果结果与 T1/T3 绝对结果，禁止用 `+1.26548 pp`解释全部 94.24%。
4. ChemAware 16,272 条规则与 V2 94.40%分开成两个独立段落。
5. BioAware B30/B35 与 B47 U3 分开；在 4,996 后同句写“无 accuracy、集中度门失败、未获准揭盲”。
6. 补入 B44–B46、Noise v2、ChemAware outer 门和应用标准品缺口等关键负结果。
7. 把 83,619 从“训练查询”改为“corrected development graph queries”。
8. 把真实队列一节更新为实际完成的 MTBLS13729/LCNEC 证据链，同时保留 Level-2/未终证边界。
9. 在提交前把 B47 U3 服务器 `report.json`、完成日志和哈希同步到本地；否则 9月30日完成账本只能作为服务器侧证据记录，不能替代可复算工件。
10. 图示不要画成三个模块串联；应画成“统一协议底座上四条互补路线”，并用实线/虚线区分已验证性能、事件机会和待验证接口。
11. 新增“shared embedding 与 post-embedding reranking”总表：P2b、V2、RRF 必须归 X 层；Noise T1/T3、ChemAware native/Phase-A 必须归 W 层。
12. 报告 P2b 时同时给 `+1.07 pp` sealed main 与 `-4.23 pp` near-core；报告 RRF 时同时给 `+1.767 pp`探针与 GNPS 确认失败，不能只摘正数。

## 八、主要仓库依据

- `docs/NOISE_ENCODER_P2B_AND_RERANKER_POSITIVE_RESULTS_20260930.md`：Noise Stage-1、T1/T3、语料、因果边界和后续 NO-GO。
- `docs/GLM_POSITIVE_RESULTS_AND_ALGORITHM_DOSSIER_20260930.md`：W/X/A 角色划分、P2b E3 sealed 资格和全部正结果卡片。
- `docs/NOISE_5PP_AND_AUC_CORRECTED_LEDGER_20260927.md`：targeted-control 与 targeted-official 的口径分离。
- `docs/GLM_CHEMAWARE_ALGORITHM_POSITIVE_RESULTS_LEDGER_20260930.md`：ChemAware reranker、native encoder、MassBank 规则语料、outer 与化学归因边界。
- `docs/BIOAWARE_ALGORITHM_POSITIVE_RESULTS_COMPLETE_LEDGER_20260930.md`：B30/B35、B47 U3 事件结果、集中度失败门和 claim limit。
- `docs/BIOAWARE_B47_U3_EVENT_YIELD_PROTOCOL_AMENDMENT_20260921.md`：U3 truth-blind event-yield 合同及禁止把机会数写成 accuracy 的边界。
- `docs/PROJECT_FULL_EVIDENCE_INVENTORY_20260925.md`：统一证据等级、负结果、MTBLS13729 与 LCNEC 应用边界。
- `docs/GLM_PROJECT_OVERVIEW_20260930.md`：最新总体定位、候选图、GNPS 未产出状态和应用主张。
- `docs/SYSTEMATIC_REPOSITORY_REVIEW_20260922.md`：历史纠错、评估集重复消费、归因缺口与服务器工件边界。
