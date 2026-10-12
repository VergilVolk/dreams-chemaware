# 非靶向代谢组学生物学应用边界重置：从未知实体复现到功能三角验证

> **2026-10-11 二次纠偏：** open-set 实体效应迁移与 response transport 都只是通用方法中的分析算子，不再有“唯一主线”。总地基是无需标签的可观测化学表示；必须同时支持无标签样本、有设计的表型研究和单谱查询。ChemProp2 M1 实际是 natural feature ID 下的 MS1 response baseline，并未使用谱学表示。总纲见 `UNBIASED_STRUCTURE_AGNOSTIC_DISCOVERY_CORE_20261011.md`。

日期：2026-10-10  
状态：方法学主线建议；尚未形成阳性结果  
适用范围：统一算法完成谱学证据装配之后的生物学应用。本文不改变 GNPS/Enveda 的算法评测边界，也不把提案写成成果。

## 0. 结论

本项目不应再把“多注释几个分子”“把未知峰画成网络”“用公共仓库找上下文”或“用质量差推断转换”当成拓宽非靶向代谢组学边界。这些方向已有成熟先例，而且本仓库的静态 BioAware、化学转换和后验融合实验已经说明：更多关系不自动产生更可靠的生物学。

当前最有机会形成真实方法增量的主线是：

> **把未定结构的 LC–MS/MS 信号注册为可弃权、可跨条件追踪的实验实体；把实体对应的不确定性显式传播到跨队列效应复现；只对已复现且仍有生物学价值的未知实体，使用独立扰动、时间/剂量和稳定同位素来源证据完成一次前瞻功能闭环。**

可简称为“谱学实体效应迁移与功能三角验证”，但在通过隐藏真值和真实数据门之前不建议创造新缩写。

这条路线有两个彼此独立的贡献层：

1. **统计方法层：**在不知道精确结构时，控制“同一实体且效应复现”的错误率，而不是先硬匹配一个峰再做差异分析；
2. **生物学层：**让一个临床发现的未知实体进入可控模型，证明它具有可重复的扰动响应和可解释的前体掺入，并由前瞻干预使该实体和表型按预注册方向改变。

第一层失败，整条路线停止。第一层通过但第二层未完成，只能称跨研究实体复现方法，不能称机制或功能代谢组学。

## 1. 仓库证据的重新裁决

### 1.1 已经可用的地基

- `build_unified_entity_evidence_ledger.py` 已把 Noise、WSE、冻结 P2b、ChemAware V2 和真实 BioAware event 分栏保存；这解决的是证据记录和真值隔离，不是生物学推断。
- LCNEC Stage-1d 在 **34 对患者、263 个 QC 合格实体**上通过患者 bootstrap：family/within/isolated 分量分别为 0.3222/0.5082/0.1696。这支持“全局效应可被分解”，但仍是单队列、单主平台证据。
- LCNEC 有 pooled QC、blank、稀释和配对设计，适合做 ion family 和同患者跨平台技术复现。
- B47 的 ST001122/ST003356 原始 MS/MS 已在本地，能继续检验 candidate-specific event；它没有自动获得疾病复现或因果含义。
- MTBLS13729 有真实临床生物学轴，可作为“丰度、来源、通量、因果不能混写”的压力测试，但其样本量、QC/blank 和全 feature 发现能力不足以裁决主方法。

### 1.2 仍未成立的前提

- 尚未证明未知实体在来源、仪器和色谱留出下具有可用的 link-FDR–coverage。
- LCNEC 已完成表型盲技术 headroom：HSST3n 263/263、LIPn 382/382 个 QC 合格实体均有代表 MS/MS，但 5 ppm 仅形成 6 个候选对，覆盖发现端 6/263（2.28%）；因此全局运输覆盖门已失败，尚无效应迁移结果。
- 尚未证明 known+unknown 在独立生物学终点上超过 matched known-only。
- 尚无一个“疾病未知实体—受控扰动—稳定同位素—前瞻表型”的闭环。
- B44–B46 的静态目录、拓扑和聚合上下文是负结果；B47 在真实事件门通过前不能进入主方法。

### 1.3 必须修正的旧合同

旧 Stage-2B 用“谱库 Top-1 cosine ≥ 0.80”定义 known。这个定义会把检索器性能和实体身份混在一起：同一个真实已知物可能因谱图质量被归入 dark，一个未知物也可能因错误高分被归入 known。新主分析必须按冻结证据状态定义：

- `confirmed_known`：同方法 authentic standard 或等价正交确认；
- `library_supported`：谱库匹配但未达确认；
- `unresolved_entity`：无可接受库身份或明确 no-match；
- `artifact_or_redundant_ion`：污染、同位素、加合物、二聚体、源内碎片或错误 MS1–MS2 链接。

known-only 主对照使用前两类中预先规定的集合；unknown 增量只允许来自 `unresolved_entity`。不得再用模型得分阈值定义生物学“暗物质”。

## 2. 文献前沿：哪些主张已经被占据

| 已有方向 | 已完成的关键能力 | 本项目不能冒领的说法 | 尚可争取的交界 |
|---|---|---|---|
| PAIRUP-MS（2019） | 匹配跨数据集未知信号，并使联合关联发现约增加 6 倍 | “未知峰可跨数据集进入关联分析” | 加入 MS/MS open-set 对应，并控制 match-and-replicate 联合错误率 |
| GromovMatcher（2024） | 以强度相关几何和最优传输自动对齐研究，优于 metabCombiner/M2S | “第一次用全局结构匹配跨研究 feature” | 谱图证据、dustbin/no-match、效应复现而非仅对齐准确率 |
| Reverse metabolomics（2024） | 合成谱图→公共数据上下文→四个 IBD 队列→培养与功能实验 | “第一次从谱图搜索疾病上下文” | 从临床未知实体正向进入可控功能实验，而非 chemistry-first 搜索 |
| 药理代谢图谱（2025） | 1,520 个药物、2,269 个 putative metabolites、>340 万依赖及 MoA 验证 | “第一次用代谢响应指纹推断药物机制” | 未知实体能否在来源/仪器留出时提供独立 MoA 增量 |
| IsoNet（2025） | 全局示踪和同位素体网络发现未知反应，并用合成/体外反应确认 | “第一次用同位素发现未知反应” | 将临床复现的同一未知谱学实体接入示踪和干预闭环 |
| MID-based identification（2025） | 20 个独立 13C tracer；用标记参考材料识别大量 unknown peaks | “第一次用 MID 补充 MS/MS 身份” | 把 MID 当独立功能/来源坐标，并与跨研究实体错误率共同评价 |
| TidyMass2（2026） | 来源数据库整合和未知 feature 功能模块分析 | “第一次让未注释 feature 可解释” | 不靠数据库来源标签，而靠实体复现、扰动和示踪支持来源/功能 |
| MS2LDA 2.0（2026） | 大规模 Mass2Motif 与自动注释引导 | “第一次发现未知碎裂程序/亚结构” | motif 只能作化学解释，不承担实体同一性或机制真值 |
| StructureMASST（2026） | 结构/亚结构级公共谱图和元数据搜索 | “第一次做全球谱图上下文映射” | 公共上下文仅作外部描述，不作为主因果证据 |
| mammalian dark metabolome preprint（2026） | 26 个 tracer、18 个小鼠组织，为数千未知峰赋予前体来源并发现新代谢物家族 | “第一次给未知物建立生物合成坐标” | 临床发现的实体能否被可靠迁移到该类功能坐标并预测可干预表型 |

因此，创新不能落在任一单组件。可辩护的新颖性只能是：**跨研究谱学实体不确定性、效应复现统计和前瞻功能实验之间的闭环。**

## 3. 主方法：不确定性感知的实体效应迁移

### 3.1 分析对象

每个研究先在不读取表型的条件下冻结四层对象：

1. `ion_feature`：单峰及 MS1/MS2 观测；
2. `ion_family`：去除同位素、加合物、源内碎片和共洗脱冗余后的分析物单元；
3. `cross_study_entity`：跨研究可能代表同一分析物的概率对应集合；
4. `structure_hypothesis`：候选集合和 no-match，永不反向决定前三层真值。

当前统一证据账本可作为第 1–2 层输入，但还需增加样本级丰度、MS/MS 获取倾向、批次、平台和冻结哈希。

### 3.2 Open-set 对应，不强制 Top-1

对发现实体 i 和验证实体 j，估计表型盲对应概率：

\[
q_{ij}=P(M_{ij}=1\mid E_{mass},E_{ion-family},E_{MS/MS},E_{condition}).
\]

要求：

- 显式 dustbin/no-match；
- 默认一对一，只有预定义的未分离异构体或平台拆分才允许多对一；
- 不把共享数据库 Top-1 候选作为主对应证据；
- 分数在来源、仪器和分子式隔离训练折上校准；
- 以同质量、同离子形式、相似峰数/谱质的不同分子作 hard decoy；
- 报告固定 1%/5% link-FDR 下覆盖、Brier/ECE 和来源/分子式簇 CI，而非只报 Top-1。

Noise、WSE、冻结 P2b 是独立输入或独立基线；ChemAware V2 只作候选化学解释；BioAware 只有真实 event 才可进入次级分析。不得把模块票数当对应概率。

### 3.3 把对应不确定性传播到效应复现

发现队列为实体 i 给出预注册效应 `beta_D(i)`；验证队列为每个候选 j 给出 `beta_V(j)`。匹配候选和权重必须在读取验证表型前冻结。

主结论是合取命题：

1. i 和 j 为同一实体；
2. 验证效应与发现效应同向；
3. 验证效应超过预注册的最小相关效应，而不只是样本量驱动的微小 p 值。

继续保留 URSEA 的 intersection–union 结构，以匹配 p 值、方向性关联 p 值及最小效应检验中最大者作为 pair-level 证据，再对一对多候选做读取表型前冻结的加权并集校正。全局使用 selective replicability/r-value 或保守依赖校准，不在入选子集上直接 BH。

必须新增三项旧版缺失的输出：

- 标准化效应及跨平台 prediction interval；
- 发现—验证效应异质性和反向效应数；
- 对 `q_ij` 多重抽样后的结论敏感性。

这样避免“谱图可能匹配 + 验证端 p<0.05”被误写为稳定生物学效应。

### 3.4 known+unknown 的增量终点

不是比较“用了多少 feature”，而是在完全相同的数据、预处理和终点上比较：

1. `confirmed/library-supported known only`；
2. 数量、MS1 强度、缺失率、QC CV、MS/MS 信息量匹配的 known 对照；
3. known + unresolved entities；
4. known + m/z/RT-only unresolved entities；
5. known + 打乱的谱学对应；
6. known + 完整谱学实体迁移。

主终点为同一 replicability FDR 下新增的真实复现实体数；功能数据中的主终点为药物/靶点/来源/仪器留出后的 MoA 或节点检索增量。只有第 6 项相对所有对照的聚类 CI 下界严格大于 0，才支持“未知实体有独立生物学增量”。

## 4. 决定性扩展：功能三角验证

跨队列复现仍然只是关联。真正拓宽边界需要对一个已经复现的未知实体建立三个相互独立的坐标：

### 4.1 疾病坐标

- 独立患者或独立生物样本中的同一实体；
- 方向和最低效应通过 replicability gate；
- 不依赖精确结构成立。

### 4.2 干预坐标

- 两个机制不同、但指向同一生物学轴的正交干预；
- 时间和剂量序列；
- 同时记录细胞死亡、增殖、总离子量和批次，排除通用应激；
- 至少一个预注册 held-out 干预用于评价，不能用同一批干预既选实体又宣称预测成功。

### 4.3 生物合成坐标

- 至少一个能区分竞争来源假说的稳定同位素前体；
- 报告 isotopologue/MID、自然丰度校正和时间依赖；
- 单次掺入只能支持“前体贡献/来源”，不能自动声称净通量或酶活性；
- 若要写通量，必须有动力学和可识别的 flux model。

### 4.4 前瞻闭环

预注册：某干预应使同一未知实体向疾病反方向移动，并改善一个独立表型 readout；必要时加入 rescue。成功标准不是“网络更漂亮”，而是：

> 同一谱学实体被独立确认迁移成功；其时间/剂量响应和 tracer 来源符合预注册竞争假说；前瞻干预同时改变该实体和功能表型；替代解释被负对照排除。

精确结构可在这之后通过标准、合成、NMR 或正交分离升级，但不再是关联统计成立的前置条件。反过来，如果功能实验必须先知道精确结构才能执行，就应诚实承认该实例仍属于传统结构确认路线。

## 5. 三层执行计划

### P0：现在即可做的证伪，不增加湿实验

1. **LCNEC Stage-2A/B 已完成 P0 headroom 重写与首次证伪。**身份改用冻结状态，分析器实现 open-set link-and-replicate；但真实候选覆盖仅 2.28%，故路线降级为最多 6 对的窄技术验证，不再承担广覆盖主结论。
2. **构建隐藏身份对应基准。**使用跨仪器真实重复谱和 hard decoy，比较 m/z、m/z+RT、cosine、WSE、official DreaMS、Noise、Noise+P2b、PAIRUP-MS、metabCombiner 与 GromovMatcher。
3. **回顾性 evidence masking。**在 reverse metabolomics 和 IsoNet 中依次隐藏结构、来源、示踪或功能真值，测试方法是否会在证据不足时弃权，并在揭盲后恢复正确低分辨率结论。
4. **修正实体账本接口。**增加 `identity_status`、`sample_abundance_pointer`、`acquisition_propensity`、`cross_study_link_set`、`effect_estimate` 和 `effect_claim_level`，但仍不把表型写回谱学分数。

### P1：公共功能数据的开发门

- **IsoNet（MassIVE MSV000097764）**：检验谱学实体链接能否保留独立的 isotopologue/reaction truth；此数据用于开发，不是本项目的新生物学发现。
- **ChemProp2（MSV000096724、MSV000093571、MSV000094899）**：检验时间方向和微生物药物转换响应；按 drug、source 和 batch 留出，并与时间序列/FBMN 基线比较。
- **药理代谢图谱（S-BSST1724）**：只有确认可取得原始或兼容的 MS/MS 和样本级设计后才进入；当前论文确认了大规模代谢响应，但网页证据不足以确认其适合跨研究 MS/MS 实体迁移。
- **2026 mammalian dark metabolome preprint**：可作前体来源隐藏真值压力测试，但因仍是预印本且体系为小鼠多组织，不作为临床外部确认。

P1 的通过条件：在来源/仪器/干预留出下，unknown entity 相对 matched known-only 对 held-out 功能标签有严格正增量；否则扰动路线降级为工程扩展。

### P2：一个前瞻生物学闭环

只选一个在 P0/P1 后仍满足以下条件的实体或小模块：

- 在疾病队列和独立样本中复现；
- 不是 blank、源内碎片或冗余离子；
- 在可用细胞、类器官、微生物或组织模型中可检测；
- 至少有两个可区分的来源/机制假说；
- 有现实可行的正交干预、tracer 和表型 readout。

优先级不是最大 fold-change，而是“能否以最少实验区分竞争假说并形成闭环”。若 LCNEC 无独立患者或模型中检测不到同一实体，应换数据，不得用同患者多平台代替独立生物学。

## 6. 各现有路线的新位置

| 路线 | 新位置 | 不再允许的角色 |
|---|---|---|
| 逐实体统一证据账本 | 主方法的数据结构地基 | 自称完成生物学整合 |
| URSEA v0.1 | 统计核心，升级效应大小和功能门 | 单独作为完整应用方法 |
| 可辨识结论/六轴证据 | 治理与实验选择层 | 仅凭框架图声称创新 |
| 谱图实体扰动响应 | P1/P2 高风险扩展 | Gate 1 前建立大规模“功能身份” |
| 全局化学重塑 | 描述性表型压缩和候选生成 | 把图平滑/内部替换写成机制 |
| PMD/转换边 | 仅在同位素或时间方向支持后作为次级证据 | 静态 abundance 差推断反应或通量 |
| BioAware | 真正 event-level context 的可选证据 | 静态目录/拓扑复活 |
| MTBLS13729 | 结论边界与反例 | 独立方法裁决或 flux 证明 |
| LCNEC | 6 对候选的窄技术校准/反例；证明停止门有效 | 广覆盖实体运输或独立患者复制 |

## 7. 失败门和允许主张

| 最远通过层级 | 允许主张 | 禁止主张 |
|---|---|---|
| 只通过隐藏身份对应 | 可跨条件校准谱学实体对应 | 未知实体复现疾病生物学 |
| 再通过同患者跨平台 | 技术效应迁移 | 独立队列复现 |
| 再通过独立患者/样本 | 未定结构实体具有可复现关联 | 来源、通量或因果机制 |
| 再通过公共扰动留出增量 | 未知实体含独立功能响应信息 | 单一实体的疾病因果作用 |
| 完成前瞻干预+tracer+表型 | 对该实体支持限定条件下的功能机制 | 泛化到所有未知代谢物 |

立即停止或降级的条件：

- 可接受 link-FDR 下覆盖不足，或增益只存在于随机谱划分；
- unknown 增量消失于 QC/强度/缺失/MS2 信息量匹配后；
- 匹配使用了验证表型、功能标签或共享候选身份形成循环；
- 同患者跨平台阳性但独立生物学为零；
- tracer 只证明掺入却被写成通量，或静态 PMD 被写成反应；
- 结论依赖一个未经审计的源内碎片、污染物或错误 MS1–MS2 链接；
- 前瞻干预只改变实体而不改变独立表型，或只改变表型而实体未按预测移动。

## 8. 论文结构建议

统一论文不应是“算法冠军 + 两个故事”的拼接，而应形成一条因果清晰的价值链：

1. **算法部分：**在冻结候选图上证明谱学证据如何改善排序与 open-set 风险；
2. **实体部分：**证明改进能转化为固定 link-FDR 下更多可靠跨条件实体，而不是只提高小候选池 R@1；
3. **生物学部分：**证明这些实体带来 matched known-only 之外的可复现效应或 held-out 功能增量；
4. **决定性案例：**一个未知实体完成独立复现、扰动、tracer 和前瞻表型闭环；
5. **负结果：**保留静态 BioAware、无增量 transformation 和无法复现实体，说明停止门真正工作。

最强的一句话主张不是“无需注释即可解释代谢组”，而是：

> We test whether an unresolved LC–MS/MS entity can be carried, with calibrated correspondence uncertainty, from clinical association to independent replication and a prospective perturbation–isotope functional experiment, without treating exact structural annotation as a prerequisite or a substitute for mechanism.

## 9. 主要文献

1. Yu et al. PAIRUP-MS. PLoS Computational Biology, 2019. https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006734
2. Breeur et al. GromovMatcher. eLife, 2024. https://elifesciences.org/articles/91597
3. Gentry et al. Reverse metabolomics. Nature, 2024. https://www.nature.com/articles/s41586-023-06906-8
4. Schuhknecht et al. Human metabolic map of pharmacological perturbations. Nature Biotechnology, 2025. https://www.nature.com/articles/s41587-024-02524-5
5. Wang et al. IsoNet. Nature Communications, 2025. https://www.nature.com/articles/s41467-025-60258-7
6. Isotope tracing-based metabolite identification. 2025. https://pmc.ncbi.nlm.nih.gov/articles/PMC12027066/
7. A Functional Metabolomics Framework to Track Microbiome Drug Metabolism. 2026. https://pmc.ncbi.nlm.nih.gov/articles/PMC12889426/
8. Wang et al. TidyMass2. Nature Communications, 2026. https://www.nature.com/articles/s41467-026-68464-7
9. Torres Ortega et al. MS2LDA 2.0. Nature Communications, 2026. https://www.nature.com/articles/s41467-026-75038-0
10. StructureMASST. Nature Biotechnology, 2026. https://www.nature.com/articles/s41587-026-03082-8
11. Mapping the mammalian dark metabolome by in vivo isotope tracing. bioRxiv preprint, 2026. https://pmc.ncbi.nlm.nih.gov/articles/PMC13060188/

