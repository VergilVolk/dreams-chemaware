# ChemAware 化学库与既有探索重新审计

**日期**：2026-09-02  
**状态**：停止新增训练；先修正数据合同和科学假设  
**范围**：只讨论 ChemAware，不涉及 BioAware、代谢网络或具体生物学应用

## 1. 结论

此前把四种性质不同的资源都口头称为“化学库”，导致了监督语义混乱：

1. MassSpecGym 的已标注谱集合提供同一化学身份在不同实验条件下的重复谱，也提供严格质量窗口内的异分子候选；
2. `unified_v2` 是因字段误读而漏谱的历史代表谱库，不能再作为当前完整检索库；
3. 335 条核心质量模体与 3,151 条 MassBank 记录派生项是观测质量模体，不是分子身份标签；
4. 候选结构和单键切割片段只能构成需要校准的弱教师，不能直接定义谱图距离。

因此，当前不允许继续通过增加 loss、增加 PEFT 层数或延长训练来推进 ChemAware。必须先把检索库、训练重复谱库、经验模体库和结构弱教师拆成独立工件，并分别声明可以和不可以提供的监督。

## 2. 已核验的数据事实

### 2.1 MassSpecGym HDF5：当前真正的数据母库

本地 `MassSpecGym_MurckoHist_split.hdf5` 共 231,104 条谱，其中：

- `SIMULATION_CHALLENGE=True` 为 119,029 条，`False` 为 112,075 条。该字段表示是否进入 MassSpecGym 的 spectrum-simulation benchmark 子集，不是实验谱与合成谱的来源标签，不能据此筛谱；
- 全部谱覆盖 28,929 个 IK14、34,259 个“IK14 × 加合物”组；
- 27,998 个组至少有两张谱，共覆盖 224,843 条谱；
- 8,327 个重复组含多个仪器类型；
- 16,421 个重复组至少有两条可观测碰撞能，16,004 个组的可观测碰撞能确实不同；
- 训练折中有 129,261 条谱同时具有 strict-10ppm、同加合物的同身份参考谱和异身份候选；验证折中有 23,902 条。

上述 strict-10ppm 数量尚未剔除 P3 身份，也未排除完全相同的峰表，只是本地上界，不是正式可训练数量。HDF5 还没有可审计的原始来源库字段，因此只能主张跨仪器、跨碰撞能重复，不能在未补充来源台账前主张跨来源重复。正式服务器工件仍必须应用 P3 allow-list，并重新报告 query、identity、formula 和 scaffold 隔离后的数量。

### 2.2 `unified_v2`：因字段误读而不完整的历史代表谱库

`unified_v2` 最终包含 294,575 条谱、207,787 个完整 InChIKey、182,158 个 IK14。旧构建器把 `SIMULATION_CHALLENGE=True` 错读为“模拟谱”，在任何结构/前体校验前丢弃了 119,029 条合法的已标注谱；剩余记录中有 111,456 条 MassSpecGym 谱和 1,340,636 条 GNPS 谱通过过滤。去重后来源分布为：

- GNPS：278,866 条；
- MassSpecGym：15,709 条。

当前去重键为“完整 InChIKey × 极性 × 推断加合物”，每个键最多保留一谱；实际 294,575 个键全部只有一谱。代表谱按库质量等级和峰数选择。旧版写出的 MGF 没有保留 `INSTRUMENT_TYPE`、`FOLD`、`SIMULATION_CHALLENGE` 或 `COLLISION_ENERGY`。

一键一谱的设计适合压缩检索 gallery，但不适合训练跨仪器、跨碰撞能的身份不变性；字段误读则使 `unified_v2` 连“完整代表谱库”也不能成立。它只能作为错误历史和回归对照。

修正后的 `unified_v3` 已完成全量构建和独立迁移审计：231,104 条 MassSpecGym 输入中仅 663 条未通过前体—结构校验，230,441 条进入代表谱竞争；最终库含 297,899 条代表谱、210,813 个完整 InChIKey、184,379 个 IK14。相对 v2 净增 3,324 个代表键、3,026 个完整 InChIKey 和 2,221 个 IK14，没有删除任何旧代表键；另有 338 个共同键从 GNPS 代表谱改由 MassSpecGym 代表谱占据。最终 19,371 条 MassSpecGym 代表谱中，`True` 4,115 条、`False` 15,256 条；全部保留 fold、碰撞能字段和 benchmark 标记，16,957 条保留非空仪器类型。来源库 provenance 仍未补齐。

这个净增幅远小于 119,029 条被误筛记录，原因是代表库会按“完整 InChIKey × 极性 × 加合物”去重，且与 GNPS 竞争。不能据此认为其余记录无价值：它们的主要价值在非去重重复谱训练库，而不在一键一谱 gallery。

### 2.3 当前 922-query 图：人为富集的机制诊断，不是全局训练或 benchmark

当前本地发现图有 922 个 query、457 个 query identity、460 个候选 identity 和 927 条可达谱。它来自验证折中预先筛选的质量合格、跨条件正对和 mass-dense 单元：

- 原始验证折 45,185 条谱；
- 质量过滤后 25,534 条；
- 2,735 个跨条件单元；
- mass-dense 后 928 个单元；
- 再做分子身份和质量邻接连通分量隔离的 discovery/confirmation 划分。

该图的每个 query 在直接候选组中都只有一张正参考谱。457 个 query identity 中 400 个可在 `unified_v2` 找到，但 `unified_v2` 的一键一谱合同仍不能提供多正样本。因此缺少多正样本不是“公共库没有覆盖”，而是检索库去重和当前图构建共同造成的。图目录名中的 `cached_real_diagnostic` 是历史命名，不构成谱图来源证据；图内 775 个 `SIMULATION_CHALLENGE=True` query 也不得被解释为模拟谱。

### 2.4 质量模体库：不能作为身份或机制标签

核心 JSON 含 335 项：214 个中性丢失、102 个碎片质量、8 个同位素模式、9 个氢重排及少量其他项。

MassBank 派生 JSON 含 3,151 项：

- 3,072 个 `CF`；
- 79 个标成 `NL` 的 `mass_diff`；
- 每项 support 都为 1；
- 每项都被标记为 `enabled_by_default=true` 和 `recommended_action=keep`。

实现审核确认：79 个 `mass_diff` 来源于 `abs(precursor_mz - exact_mass)`，语义是前体/加合物偏移，不是碎片中性丢失；3,072 个 `CF` 来自每条记录按 m/z 排序后的最低三峰，解析时没有保留峰强度。因此它们是记录级观测，不是经过跨分子支持的碎裂规则，更不是候选结构的片段真值。

G0 后续已把 MassBank `mass_diff` 单列为 `precursor_exact_mass_offset`，这是必要的语义修复；但整个 3,151 项工件仍只允许用于覆盖、冲突、质量控制和不确定性分析。

### 2.5 “679.8 万谱”是未完成 provenance 的历史总量，不是当前可引用数据合同

`tasks/README.md` 写有“6,797,516 谱图、271,594 个独特 InChIKey（GNPS + MassBank + MoNA + MSnLib）”，但当前代码与数据目录中未找到与该数字一一对应的构建脚本、逐来源计数、去重规则、MS2/MSn 层级说明或冻结 manifest。这个数字只能视为历史汇总口径，不能在对外汇报中作为已经审计的训练集规模。

它也不与 `unified_v3` 的 297,899 张谱矛盾：前者意图描述保留重复条件的全谱集合，后者明确是按“完整 InChIKey × 极性 × 加合物”压缩的一键一代表谱检索库。二者的计数单位、去重规则和用途不同，不能直接比较。只有当 679.8 万谱的来源、谱级别、身份解析、重复谱与损坏记录均形成可复算台账后，才允许恢复这一数字。

更重要的是，历史总库声称包含 MSnLib，而当前已把 MSnLib 的合格后期采集谱保留为外部压力测试候选。因此未来若重建非去重训练库，必须显式排除这部分 MSnLib 谱及其身份；同一资源不能同时充当训练数据和独立评价数据。

### 2.6 MSnLib：可形成严格隔离的外部压力测试，但不是绝对来源独立

本地 18 个 `*_ms2.json` 文件共 1.51 GB，含 570,948 个非空行；其中 570,912 行可解析为导出 MS2/pseudo-MS2 库谱，36 行存在 JSON 断裂或控制字符错误并被显式剔除。这里“一行”是一张导出库谱，可能由多个原始扫描合并，不等于一个 raw scan。可解析部分覆盖 47,659 个完整 InChIKey、46,908 个 IK14 和 21,107 个分子式；522,837 条记录的原始采集日期晚于 2022-11-30。

按 IK14 比较，14,501 个 MSnLib 身份与 MassSpecGym 重叠，3,227 个与本地 MoNA 重叠。进一步要求：采集晚于 2022 年 11 月、完整结构可解析、常用加合物、至少 5 个峰、chimeric QC 通过、身份与 MassSpecGym/MoNA 均不重叠、规范化峰哈希也不与两库重叠，得到 270,202 条合格谱和 41,915 个“身份 × 加合物”参考组。只有 554 个组还能找到不同原始文件中的第二张非重复谱作为 query；strict-10ppm 内至少存在一个错误身份候选后，最终剩 406 个可评价 query、379 个 IK14、338 个分子式、3,338 条候选边。候选数中位数为 6，295 个 query 含同分子式错误候选。

这 406 个 query 已在不读取任何模型分数的前提下按连通块拆分：discovery 162、confirmation 122、final test 122。主面板进一步要求 query 与真参考使用相同 merge type，并仅保留 `[M+H]+` / `[M-H]-`，三拆分为 113/86/86；final test 主面板中 66 个 query 含同分子式干扰物。拆分后 query 身份、分子式、非空 Murcko 骨架、Morgan-Tanimoto ≥ 0.8 的 query 对和候选参考谱均无跨拆分重叠；最终测试输入不含结构和真值字段，测试 key 单独保存，尚未读取模型分数。

该面板最准确的名称是“未来预注册的外部压力测试”。它的时间、身份、谱哈希和化学分组隔离是可审计的，但 DreaMS 论文曾用同一 MassIVE accession `MSV000094528` 做过含氟性质评估，因此不能宣传为从未被 DreaMS 研究触及的完全新来源。若要作最强外部泛化主张，仍应再增加一个不同 accession、不同实验室的封存面板。

## 3. 既有探索真正证明了什么

### 3.1 P2b 证明的是局部谱学互补性，不是“化学规则有效”

P2b 使用查询谱和候选实验谱之间的 DreaMS cosine、峰匹配、熵相似度和中性丢失相似度。冻结配置中 80% 权重来自中性丢失 sqrt cosine。

开发阶段 5,037-query、分子式隔离嵌套 OOF：

- Recall@1：86.06% → 89.97%，+3.91 pp；
- near Recall@1：76.12% → 81.95%，+5.83 pp。

但冻结 P3 的完整门未通过：

- 主面板 3,000 query：87.93% → 89.00%，+1.07 pp，89 修正 / 57 新错；
- near-core 496 query：48.79% → 44.56%，−4.23 pp，20 修正 / 41 新错；
- nearmid 661 query：54.46% → 51.44%，−3.03 pp。

因此，P2b 的局部中性丢失证据对主面板存在可重复互补性，但对极近、同分子式异构体不是可靠身份依据。它不能作为全局教师无条件蒸馏，也不能把开发 OOF 的 +3.91 pp 当作共享嵌入的预计增益。

### 3.2 小图 PEFT 只证明容量和安全性，尚未证明化学归因

当前共享 encoder 的最好本地结果在单个外折上为 2 corrected / 0 introduced，即 +1.01 pp。增加 preservation、训练轮次或第二个 Transformer block 只改善连续 margin，没有增加离散纠错数。

Morgan、MolFormer、候选 hardness、峰规则注意力和单键切割片段均未在严格对照下给出超过 clean listwise continuation 的稳定增量。结构单键切割在整体候选上有方向，但在 21 个官方错误中只有 4 个方向正确；峰到子结构 probe 的 AUPRC 只比峰置乱高 0.0079。

这些结果共同说明：当前瓶颈不是 PEFT 参数太少，而是监督没有对准官方 DreaMS 的残余错误。

### 3.3 传统谱学多数不能替代 DreaMS，但存在稀疏的高置信残差信号

在既有 formula-isolated observability cohort 上，对五个原始谱学视图（entropy、sqrt cosine、linear cosine、top-10 peak match、intensity coverage）做不看身份标签的 3/5 候选多数投票。直接用多数替代 DreaMS 在 discovery 上净降 1.69 pp，在 confirmation 上净降 1.39 pp；confirmation 为 116 corrected / 165 introduced。因此“多个传统谱相似度一致”本身仍不是安全教师。

随后只使用可观察的排序几何——DreaMS top-2 分差、各视图 top-2 分差、投票一致度和候选规模——在 discovery 公式组内进行五折 cross-fit，并按 `corrected - 2 × introduced` 一次性冻结高精度门，阈值为 0.75。该门在 formula-disjoint confirmation 只触发 4 次，结果为 4 corrected / 0 introduced。冻结 scaler、系数、截距、阈值和输入哈希后，才首次编码并一次性消费此前未读的 test split；3,539 个可评价 query 中触发 9 次，9 corrected / 0 introduced，Recall@1 从 84.459% 到 84.713%，即 +0.254 pp，公式簇 bootstrap 的差值区间为 +0.086 至 +0.448 pp。

这个结果是方向性证据，不是共享 embedding 结果：门控在推理时读取 query 与整个候选参考谱集合的五种相似度，且触发率极低。100 个 discovery 标签置乱控制中有 1 个在 confirmation 获得比观察门更高的风险净值，经验 `p=0.0198`；因此不能把 4/0 或 9/0解释为已经找到普适化学规律。九个 test 触发案例均排除了 cohort 哈希定义下的完全重复峰表；其中 5 个有已知跨仪器正参考、2 个有明确不同碰撞能正参考，但仍不足以证明跨实验室泛化或结构因果性。

进一步的依赖性审计撤回了“五视图相互验证”的说法。约 77% 的可评价查询中五种评分选择完全相同的候选，约 99% 最多出现两个不同赢家；在 DreaMS 错误查询上，五种 truth-vs-wrong 优势的相关矩阵有效秩仅为 1.56–1.60。冻结门触发的 discovery/confirmation/test 34/4/9 个案例里，五种评分全部同票。它们实际是同一峰匹配关系的若干高度相关汇总，而不是五种独立化学证据。

另有一个 provenance 限制：历史 cohort 报告记载“按稳定哈希 60/20/20”，当前同名生成脚本已改成分层贪心分配，二者不一致。实际冻结的 discovery/confirmation/test 三份 manifest 已逐行核验：并集恰好覆盖 18,810 条 selected spectra，任意两份的 row、identity 和 formula 交集均为 0，报告计数与实际一致。因此上述结果可以从冻结 manifest 与 pair table 复现，但不能声称当前 builder 能字节重建历史 cohort。

## 4. 重新定义 ChemAware 科学问题

官方 DreaMS 已经用“同一分子的不同谱为正、0.05 Da 内不同分子为负”的 triplet 做过端到端微调。因此，仅在 MassSpecGym 上重复同身份正对和近质量负对，不是新的 ChemAware 科学问题，只能作为域内 continuation 对照。

以 `ChemAware` 为项目代号，当前真正可检验、且区别于官方 DreaMS 的第一阶段问题应为：

> 在 query/reference 共用同一个、推理时只接收 MS/MS 谱和前体信息的 DreaMS encoder 时，能否把“传统峰匹配评分能够识别、但 DreaMS 全局 embedding 丢失的稀疏候选判别信息”压缩进共享单谱 embedding；并在分子身份、分子式和骨架隔离的评价中，相对官方 DreaMS 及同采样、同容量、同错误聚焦策略的 clean listwise continuation 获得稳定的候选检索增益，同时不损害近异构体和原有高置信正确查询？

这里第一阶段真正待证的是“极稀疏、高置信的候选判别峰匹配证据是否提供独立增量，并且能否在不读取候选结构或候选谱的部署阶段被共享单谱向量表示”。重复谱只用于估计实验条件不变性，近质量候选只用于定义检索边界；二者本身都不是 ChemAware 新意。所谓“化学感知”也不来自把经验质量列表当成身份标签。由于当前教师不含结构、反应或子结构真值，即便第一阶段成功，也应表述为谱图证据保持；只有第二阶段的结构—碎片监督在 matched spectral control 之上仍有增量，才能表述为化学感知。

修正后的 metadata-only 全量候选清单现已建立：83,619 个 query、9,854 个身份、6,220 个分子式、392,229 个候选分子节点和 6,220,661 条 query–reference 谱边。308,610 条负候选分子边中 75.08% 与真分子同分子式；78.06% 的 query 至少含一个同分子式干扰物，33.71% 至少含一个已有 MCES-near 标注的干扰物。每个身份的 query 数从 1 到 507，说明后续必须 identity-equal 训练/评价并用 formula-cluster bootstrap，不能按 83,619 张谱直接计算一个受高重复身份支配的总体数字。

## 5. 四类资源的严格合同

### A. 代表谱检索库

用途：部署时候选检索。允许每个身份/加合物保留少量高质量代表谱。必须保留完整来源和质量字段，但不承担多正样本训练。

### B. 重复谱训练库

用途：学习同一 IK14、同加合物跨仪器/跨碰撞能的不变性；只有存在可信来源字段时才允许声称跨来源。必须保留所有谱行和实验元数据；按 identity 等权，不能让多谱身份支配 loss；训练和评价按 identity、formula、scaffold 隔离。

### C. 经验质量模体库

用途：冲突、QC、覆盖和解释。不得给 candidate identity 加分，不得作为 fragment-structure truth，不得再称 3,151 个记录项为 3,151 条机制规则。

### D. 候选结构弱教师

用途：训练期的候选特异审计或低权重辅助。必须有 correct/permuted/matched-marginal/same-formula-mismatch 对照，并且在相同 observable mask 和相同 query ledger 上比较。不能在推理时引入候选输入后仍称共享谱图 embedding。

## 6. 已完成预检与下一步入口

已完成的预检给出三条约束：

1. `SIMULATION_CHALLENGE` 不得作为来源筛选；修正后的 P3-disjoint 训练并集为 19,403 个身份、137,830 条谱，和 P3 身份交集为 0；
2. P2b 的 exact neutral-loss 分数可被 PSD Gaussian set kernel 高精度近似，说明“共享向量可表示性”不是主要瓶颈；
3. 但 P2b 本身弱于 DreaMS，且已在 P3 near-core 明确退化，因此它不是安全的全局教师。
4. P3-disjoint identity-continuation 控制池已分别冻结：`[M+H]+` 为 78,039 anchors / 9,779 identities / 2,938,393 正边 / 2,992,814 负边；`[M+Na]+` 为 5,580 anchors / 600 identities / 143,583 正边 / 145,871 负边。两者都只是 DreaMS 目标的域内控制。
5. 原始谱学多数全局失败；冻结高精度门在独立 confirmation 和一次性 test 上分别为 4/0 与 9/0，但只支持“存在稀疏候选判别信号”，不支持部署或共享 embedding 增益主张。

再次训练前只允许完成以下数据与假设工件：

1. `unified_v3` 已生成并审计；后续只需补充可验证的来源库 provenance，不能把 benchmark 标记代作来源；
2. 已完成非去重控制池；下一步须以它重建不依赖 `SIMULATION_CHALLENGE` 的完整 P3-disjoint candidate graph，并重新生成所有依赖图的缓存与哈希；
3. 下一实验只比较两个严格 matched 的训练臂：相同稀疏度、相同 DreaMS 错误聚焦但不看原始谱学一致性的 clean control，与额外使用冻结候选判别信号的 ChemAware 臂。已消费的 internal test 不得再参与选模、改阈值或机制选择。

只有修正 candidate graph、matched control 和新的未消费外部评价面板同时冻结后，才允许做上述两臂配对。普通同身份 triplet、经验质量模体、P2b 全局蒸馏和 Morgan whole-molecule 教师均不作为 ChemAware 实验臂。

## 7. 当前裁决

- `unified_v2` 仅保留为字段误读下的历史回归工件，不再作为当前代表谱库；
- `unified_v3` 已纳入全部 MassSpecGym 标记类别并保留现有条件元数据，当前只承担代表谱检索；
- 新建独立、非去重、元数据完整的重复谱训练清单；
- MassBank 记录派生项退出身份训练主线；
- P2b 保留为局部谱学证据和可嵌入性研究对象，但因 near-core 失败，不得全局蒸馏；
- 所有新增训练继续暂停，直到上述数据和可嵌入性合同通过。

## 8. 可复现工件

- 审计脚本：`tasks/audit_chemaware_chemical_library_contract.py`
- 当前化学库合同审计：`data/validation/chemaware_chemical_library_contract_audit_v3/report.json`
- v2→v3 迁移审计：`data/validation/chemaware_reference_library_migration_v2_to_v3/report.json`
- 统一库构建：`tasks/build_reference_library.py`
- 修正后的 P3 allow-list：`data/validation/g8r_p3_allow_recovered_corrected_v3_20260902/report.json`
- 训练谱系审计：`data/validation/chemaware_training_lineage_audit_v3/report.json`
- P2b 共享核审计：`data/validation/chemaware_neutral_loss_shared_kernel_v1/report.json`
- 真值盲谱学多数审计：`data/validation/chemaware_label_free_spectral_consensus_v1/report.json`
- 冻结门及 confirmation：`data/validation/chemaware_spectral_consensus_applicability_v4_frozen/report.json`
- 五种谱学评分依赖性审计：`data/validation/chemaware_raw_view_dependence_20260902/report.json`
- 修正候选清单：`data/validation/chemaware_corrected_candidate_manifest_v1/manifest.json`
- 候选规模与困难度审计：`data/validation/chemaware_corrected_candidate_manifest_v1/audit.json`
- 冻结门 feature ablation：`data/validation/chemaware_gate_feature_ablation_20260903/report.json`
- observability 标签/谱哈希冲突敏感性：`data/validation/chemaware_observability_label_conflicts_v2_20260903/report.json`
- 一次性 test 结果：`data/validation/chemaware_frozen_spectral_gate_test_20260902/report.json`
- test 触发案例审计：`data/validation/chemaware_frozen_gate_trigger_case_audit_20260902/report.json`
- observability cohort 工件谱系：`data/validation/chemaware_observability_cohort_provenance_audit_20260902/report.json`
- P2b 特征：`tasks/audit_e0_observability_residual.py`
- P2b 冻结结果：`data/validation/g8r_p2b_p3_final.json`
- 当前本地图：`data/validation/chemaware_shared_v2_cached_real_diagnostic/graph.npz`
