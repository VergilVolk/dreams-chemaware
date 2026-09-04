# MTBLS13729 生物学论文蓝图与完成门（2026-08-30）

## 一、论文定位

### 推荐定位

**算法驱动的非靶向代谢组学再分析 + 严格证据校准的临床发现论文。**

不是完整的代谢因果机制论文。当前没有新组织、标准品终证、同位素示踪、基因/药理扰动或类器官表型，不能把静态丰度包装成通量和治疗靶点。

### 一句话贡献

> A phenotype-blind DreaMS-enabled raw-MS2 and ion-family reanalysis recovered modified-guanosine and acetylated-polyamine signals missed or biologically unresolved by the original m/z–RT annotation table, defining a context-dependent Rmu metabolic state and refining the interpretation of the author-reported carnitine program.

**当前优先版本：**

> Evidence-calibrated reanalysis of paired colorectal tissues recovered six parallel abundance programs and three source-anchored metabolites in an orthogonal LC-MS panel. Proline/P5C–matrix remodeling aligned with a broad CRC program, whereas elevated Neu5Ac was embedded in a mucinous-relative but internally heterogeneous mucin-glycan context; neither abundance program establishes flux or causality.

## 二、主线结构

### 主轴：修饰鸟苷/嘌呤周转

- 1597：methylguanosine isomer-family `[M+H]+`；
- 7489：同家族 `[M+Na]+`，只作支撑；
- 3019：dimethylguanosine isomer-family `[M+H]+`；
- 8481：同家族 `[M+Na]+`，只作支撑；
- 4966：独立 C7H9N5O purine-like companion feature。

主轴成立的理由：原始峰界内 MS2、核糖丢失、跨加合物一致、Rmu 患者内一致、与独立嘌呤样 feature 共变、独立外部数据提示背景依赖。

### 平行轴 1：乙酰化多胺

- 1717：acetylated-polyamine / N1,N8-diacetylspermidine-like；
- 73/73 MS2 中 m/z 100.0759 为 base peak；
- 跨 HILIC/RP 样本相关和配对差值相关；
- 文献把 SAT1–acetylspermidine 与酸性微环境、髓系/中性粒细胞联系起来，但本研究不声称该因果链。

### 平行轴 2：长链酰基肉碱

- 3222：long-chain/C20:4-acylcarnitine-like；
- 原作者已发现广泛 carnitine program；
- 本研究的新增是更细的长链锚点及 carnitine-shuttle imbalance hypothesis；
- 外部蛋白组和上皮 RNA 轴向下支持“利用瓶颈”分支，但 2026 年 carnitine/acetylcarnitine–CPT1A 工作支持“输入/利用增加”分支；两者共同说明静态丰度不足以判定通量。

### 为什么不强行合并成一条因果链

1717 和 3222 与修饰鸟苷模块的患者内相关不显著。当前数据支持三个并行丰度程序，而非“修饰鸟苷→多胺→FAO”的单链。把三个轴合成一个事后总分会过拟合 n=10，应禁止。

## 三、预定主要终点和次级终点

### Primary biology endpoint

Rmu 肿瘤与匹配 RN 中，离子家族折叠后的 modified-guanosine module 的患者内 log2FC：

- 10/10 正向；
- mean 约 `+2.95 log2`；
- exact sign-flip `p=0.001953`；
- 三套 phenotype-blind PQN 方向保持。

该终点仍是候选选择后的发现级结果，不宣称全 feature-space FDR。

### Secondary endpoints

1. feature 1717 Rmu-vs-RN 配对变化与跨色谱一致性；
2. feature 3222 Rmu-vs-RN 配对变化和 carnitine 类 MS2；
3. feature 4966 与 modified-guanosine module 的患者内相关；
4. 外部 pooled mucinous proteomics 与 GSE236696 上皮 pseudobulk 的轴方向；
5. 独立 Level-1 CRC 组织数据中的背景依赖/反向结果。

### Exploratory only

- Rmu-vs-Rtu interaction；
- 临床 MMR/BRAF 小样本分层；
- 单病例空间转录组；
- feature 16425；
- 所有具体酶归因。

## 四、主图设计

### Figure 1：算法到真实候选

1. 240 mzML / 4 panels / paired cohort 设计；
2. DreaMS/P2b 候选生成与 phenotype-blind evidence ledger；
3. 原作者 345 条注释与新候选对账；
4. 新增/已知/降级三类候选数量。

### Figure 2：修饰鸟苷离子家族

1. 1597/7489 与 3019/8481 的加合物关系；
2. representative MS2 和 132.042 Da 核糖丢失；
3. 10 对 Rmu 患者内变化；
4. 三种归一化稳健性；
5. 4966 的患者内相关。

### Figure 3：外部反证式验证

1. OEP00006137 Level-1 四唯一峰按 MSI/MSS 分层；
2. 原始 mzXML 重提取与作者矩阵相关；
3. 外部方向与 MTBLS13729 不同，明确“context-dependent rather than universal”；
4. SAH 是并行环境，不与模块患者级耦合。

### Figure 4：平行代谢程序与机制边界

使用 `integrated_biology_evidence.png`：

- DreaMS 五个生物学候选效应；
- 原作者已知 carnitine program；
- pooled mucinous proteomics 三轴；
- 六对 epithelial pseudobulk 三轴；
- 标题明确“parallel abundance programs, not a causal chain”。

Figure 4b 使用 `mtbls13729_mtbls7387_lcfa_context.png` 加入独立配对人体脂质背景：

- MTBLS7387 的 251 对处理矩阵中，C20–C24 有 17 个 FDR 显著特征，14 升、3 降；
- 游离 arachidonic acid 不显著，两个 hydroxy-C20:4 峰显著升高；
- feature 3222 必须与外部 free/hydroxy fatty acids 分栏，图注明确“pathway context, not identity replication”；
- ATF6 与 2026 Gut 论文放在 evidence ladder 的 causal benchmark 层，不能把它们的 tracing/干预结论迁移到 MTBLS13729。

### Figure 5（有标准品后）

1. 标准品/样本/样本+spike-in EIC 共洗脱；
2. 多碰撞能 MS2 镜像；
3. 位置异构体分离；
4. targeted relative quantification。

若无标准品，不应伪造 Figure 5；应把文章定位停在 discovery/application。

## 五、补充表

1. 所有候选 peak-resolved MS2 谱图与样本覆盖；
2. 8 候选投稿准备度矩阵；
3. 原作者 345 条注释逐项 overlap；
4. ion-family/adduct/isotope/source-fragment 去冗余；
5. 每种归一化的患者级 paired effects；
6. negative controls 和 failed validation；
7. 外部队列原始下载、损坏文件和 RT 窗敏感性；
8. BioAware/P2b 只作为算法应用，不作为身份真值。

## 六、最低验证闭环

### 若只能买少量标准

P0：

1. N1,N8-diacetylspermidine；
2. N2,N2-dimethylguanosine；
3. m7G、m2G、Gm 至少两至三个能覆盖 1597 的位置异构体。

P1：

4. C20:4 acylcarnitine；
5. C16:0/C18:0/C18:1 acylcarnitine 组成类群面板。

每个标准至少执行：同一色谱方法 RT、同碰撞能 MS2、样本 spike-in 共洗脱。只有谱库 cosine 而无 RT 不足以升级到 MSI Level 1。

### 若完全不能做湿实验

文章只能主打：

- evidence-calibrated reanalysis；
- raw-MS2 ion-family discovery；
- independent raw-data re-extraction；
- external multi-omics context；
- negative/null result transparency。

此时应避免“mechanism”放在标题核心，可用 “metabolic programs” 或 “mechanistic hypotheses”。

## 七、与高水平机制论文的差距

| 机制论文硬门 | 当前状态 | 结论 |
|---|---|---|
| 标准品 RT+MS2 | 未完成 | 精确身份不成立 |
| 同位素 tracing | 未完成 | 来源/通量不成立 |
| 基因或药理扰动 | 未完成 | 酶因果不成立 |
| rescue | 未完成 | 功能因果不成立 |
| 独立 subtype-resolved metabolomics | 未找到 | Rmu specificity 未确认 |
| 外部多组学 | 已完成，但异质 | 仅作方向性背景 |
| 原始 MS2/EIC | 已完成 | 发现级证据强 |
| 负结果/decoy | 已保留 | 提高可信度 |

## 八、可投稿结论的最大强度

### 当前可写

> DreaMS-enabled reanalysis recovered reproducible modified-guanosine and acetylated-polyamine ion families in paired mucinous colorectal tissues. The modified-guanosine module was uniformly elevated in the Rmu discovery subgroup and was supported by raw fragmentation, cross-adduct consistency and purine-axis covariation. Long-chain acylcarnitine accumulation, interpreted alongside discordant external FAO and carnitine–CPT1A evidence, motivated competing carnitine-shuttle imbalance hypotheses but did not establish flux direction.

### 当前不可写

- “我们证明了黏液型 CRC 的 METTL1 驱动修饰鸟苷通量”；
- “SAT1 通过 N1,N8-diacetylspermidine 招募中性粒细胞”；
- “FAO 被抑制/激活”；
- “发现了确定的新代谢物”；
- “达到 MSI Level 1”；
- “三个轴组成一条因果链”。

## 九、文章现实定位

### 无标准品时

目标是强算法应用/代谢组学方法与资源型文章，而不是顶级机制论文。卖点是：同一公开队列中，深度谱学模型和证据校准如何找回传统注释表遗漏的信息，并用外部原始数据与反证限制结论。

### 有 3–5 个标准品时

可以显著升级为“算法发现 + 结构确认 + 临床配对发现”，具备更强的一区潜力。仍不能打包票，期刊层级取决于标准结果、独立复现和算法主文是否同时有锁定测试增益。

## 十、完成清单

- [x] 原论文补充表全量获取与审计；
- [x] 8 候选原始 MS2 证据账本；
- [x] 修饰鸟苷跨加合物与患者级模块；
- [x] 1717 跨色谱一致性；
- [x] 3222 carnitine 类 MS2；
- [x] 独立 Level-1 组织代谢组与原始 mzXML 重提取；
- [x] pooled mucinous proteomics 和 paired epithelial RNA 三轴；
- [x] 单病例空间结果和阴性边界；
- [x] 原论文增量矩阵；
- [x] 论文级综合证据图；
- [x] MTBLS7387 251 对人体脂肪酸来源矩阵复算、样本流失审计与长链脂质背景图；
- [x] ATF6 2025 与 Gut 2026 脂质机制研究的完整证据链对标；
- [ ] 标准品；
- [ ] 独立 Rmu/CMS 组织代谢组；
- [ ] 示踪/扰动/表型机制实验。

## 十一、2026-08-30 跨队列机制综合更新

新增 `mechanism_evidence_matrix_v1` 后，当前主线必须从“黏液型特异机制”改为：**Rmu 发现亚组中的高幅度三轴代谢池表型，嵌在广义 CRC 核苷/嘌呤、多胺和 FAO 程序中。**

- 19 条证据来自 6 个数据源，其中 14 条独立于 MTBLS13729 发现队列；
- ST001087 对 dimethylguanosine/polyamine 家族给出正方向，但 OEP00006137 的 Level-1 dimethylguanosine 为负方向；
- TCGA paired tumour-normal 强支持一般 CRC 的 modified-nucleoside/purine 上升和 FAO 下降；mucinous-conventional 调整模型不支持 subtype-specific enhancement；
- GSE236696 和 pooled mucinous proteomics 提供机制背景，但不能替代外部代谢物复现、通量或扰动。

因此正文中的最高强度表述应使用 `Rmu-associated abundance programs`、`context-dependent remodeling` 和 `FAO-utilization bottleneck hypothesis`，避免 `mucinous-specific flux`、`METTL1/SAT1-driven` 或 `causal mechanism`。

完整矩阵与结论见 `docs/MTBLS13729_CROSSCOHORT_MECHANISM_SYNTHESIS_20260830.md`。

## 十二、结构证据新增裁决

feature 1597 与 MassBank 7-methylguanosine/N2-methylguanosine authentic 谱的最佳 sqrt-cosine 为 `0.6712/0.6667`，差值只有 `0.00450`。这支持 methylguanosine family compatibility，却也是位置异构体不可分的直接证据。与此同时，本地五个主候选的峰界内 MS2 全部记录为同一 `30 eV`，所以跨样本复现成立、跨碰撞能稳定性尚未测试。主文结构证据图必须同时展示正证据和身份天花板，不能把“有原始 MS2”写成“已经完成标准品确认”。

结构证据主图：`data/mtbls13729/structural_evidence_figure_v1/structural_evidence.png`。

## 十三、外部长链脂质背景的最新裁决

MTBLS7387 给出了真实、较大样本的人体配对复现，但只在通路/类别层面帮助 3222：

- 251 对完整处理样本、186 个脂肪酸特征；56 个全 panel 特征、17 个 C20–C24 特征达到配对 BH-FDR<0.05；
- C20–C24 中 14 升、3 降，说明是重塑而不是统一积累；
- free C20:4 不显著，两个 hydroxy-C20:4 峰显著升高，说明异构体/氧化状态必须保留；
- 论文所述 259、MetaboLights 258 对和处理矩阵 251 对是三个不同层级，已独立审计，排除原因在公开文件中未给出。

因此 feature 3222 的最高表述调整为：

> A recurrent long-chain acylcarnitine-like ion elevated in the Rmu discovery subgroup, embedded in an independently replicated CRC long-chain-lipid remodeling context.

不能调整为：

> An externally validated C20:4 acylcarnitine or proof of ATF6/FAO flux in mucinous CRC.

新工件：

- `data/external/mtbls7387_paired_lcfa_replication_v1/mtbls7387_paired_lcfa_replication.json`
- `data/external/mtbls7387_processed_pair_attrition_v1/report.json`
- `data/mtbls13729/mtbls7387_lcfa_context_figure_v1/mtbls13729_mtbls7387_lcfa_context.png`

## 十四、扩展原始数据复核后的主线修订（优先采用）

原“修饰鸟苷单主轴 + 两个平行轴”应升级为 **四个并行 abundance programs**：

1. 乙酰化多胺–MTA：平均模块 log2FC `+3.52`，10/10 同向；
2. 嘌呤/修饰核苷：`+2.36`，9/9 同向；
3. 长链酰基肉碱：`+1.66`，10/10 同向；
4. 大中性氨基酸：`+0.94`，10/10 同向。

四个模块均通过 leave-one-feature-out 方向稳定性，但仍是候选选择后的描述性分析。患者级模块相关只在 amino-acid–purine/nucleoside 上达到 rho `0.833`，六组比较 BH q `0.053`；acylcarnitine 与 purine/polyamine 近乎不相关。因此正文不得将四轴画成单一上游调控链。

新增 Figure 主图：`data/mtbls13729/convergent_biology_figure_v1/mtbls13729_convergent_biology.png`。该图同时显示模块效应、节点效应、互补证据层和患者级模块协调，标题已经写明 post-selection 与 flux/causality 边界。

完整身份纠错和机制对标见 `docs/MTBLS13729_INTEGRATED_BIOLOGY_RESULT_20260830.md`。
- `data/mtbls13729/mtbls7387_lcfa_context_figure_v1/mtbls13729_mtbls7387_lcfa_context.pdf`

## 十四、黏液型内部风险表达结构：新增但受组成限制的证据

GSE281917 的 140 例 MuC 中，MuC23 高风险与较低的 purine synthesis/salvage、modified-nucleoside processing 分数和较高的 polyamine acetylation/catabolism 分数相关。分期、年龄、性别校正后的秩相关通过多重校正，但风险分数与 fibroblast/endothelial marker scores 高度相关。

加入六类 broad-lineage marker 后，只有 GSE281917 的 purine 轴保留（rho=-0.254，95% CI [-0.464,-0.086]，q=0.0139）；在 42 例 TCGA 黏液型病例中，purine 轴只在临床校正模型中复现，组成校正后 CI 跨 0。由此只能写 **risk-associated bulk transcript state**，不能写 cell-autonomous purine mechanism 或独立预后因子。完整审计见 `docs/MTBLS13729_MUCINOUS_RISK_TRANSCRIPTOMIC_AUDIT_20260830.md`。

对应的主文/扩展图工件为 `data/mtbls13729/mucinous_risk_context_figure_v1/mtbls13729_mucinous_risk_context.png`：A 面板展示直接 Rmu–RN 代谢物丰度，B/C 面板展示 GSE281917 与 TCGA 的临床及 broad-lineage 校正风险关联。三类终点不得互相替代。

项目相对高水平机制论文的逐门审计见 `docs/MTBLS13729_MECHANISM_READINESS_SCORECARD_20260830.md`。该 scorecard 冻结了文章定位：当前是 algorithm-enabled, evidence-calibrated clinical discovery；没有标准终证、isotope tracing、perturbation 和 rescue 时，不得把标题或摘要写成 causal metabolic reprogramming。

## 十五、当前投稿版故事架构（覆盖前述早期版本）

### 15.1 文章不再以单一“修饰鸟苷机制”为唯一主轴

现有数据更适合一篇 **算法驱动、证据校准的临床发现论文**，按以下顺序叙述：

1. 原始 240 个 LC-MS 文件中，MS1 定量与峰界 MS2 被重新联通；
2. DreaMS/离子家族/谱学证据把原论文 345 条 m/z–RT 注释扩展成 18 个严格分层的生物学候选；
3. 三个 Level-1 节点在正交面板被重新找回：proline、glutamate、Neu5Ac；
4. 患者级结果形成六个平行 abundance programs，而非一个统一因果链；
5. 外部 bulk、单细胞、空间和蛋白组把“general CRC”与“mucinous-relative”两种效应拆开；
6. 失败候选、反向外部结果和无显著 matched-null 结果共同限定论文结论。

### 15.2 推荐主结果层级

**Result 1 — annotation recovery and evidence calibration**

- 原论文 345 条注释与 18 候选逐项对账；
- 9 source remap、3 orthogonal Level-1 recovery、5 source-table-absent family candidates、1 downgrade/control；
- 真实样本三路应用必须称 retained/changed/abstained，不得在无真值时称 corrected/introduced；
- P2b 在 pos-RP 注释 feature 数从 3,072 增到 3,243，但 tier-gained 47、tier-lost 370，说明“覆盖增加”不自动等于“质量提升”。

**Result 2 — paired abundance programs**

- acetylated polyamine–MTA、purine/modified nucleoside、long-chain acylcarnitine、expanded amino-acid/proline、Neu5Ac/mucin-glycan 等作为平行程序；
- 所有模块统计均标明 postselection/descriptive；
- interaction 与 Rmu-vs-RN primary endpoint 分栏。

**Result 3 — orthogonal identities and negative controls**

- 345/374/703 的 raw MS2、跨面板 sample/within-tissue/paired-delta correlation；
- 301/1695/725/458 的拒绝理由；
- source-linked 13 个候选的效应 Spearman 0.890，说明跨面板重定量与原论文方向一致，但不是独立复现。

**Result 4 — cross-cohort mechanistic context**

- general CRC proline/P5C–matrix；
- mucinous-relative, selective Neu5Ac/mucin-glycan program；
- modified-guanosine/polyamine/acylcarnitine 的正反外部证据；
- GSE matched-null、单病例空间和 pooled proteomics 的限制进入正文，不藏在补充材料。

五模块患者级协调也必须作为反证进入 Result 2：Neu5Ac 与 expanded amino-acid/purine 模块不相关；amino-acid–purine 的 rho 虽为 `0.833`，在扩展后的 10 组比较中 BH q=`0.0775`。任何“共同上游调控器”网络图都超出数据。

### 15.3 新主图顺序

1. **Figure 1**：研究设计、MS1–MS2 联通、候选证据阶梯与原论文增量四分类；
2. **Figure 2**：18-candidate ledger、六个 abundance programs、患者配对效应；
3. **Figure 3**：proline/glutamate/Neu5Ac 三个正交身份节点及两个失败身份对照；
4. **Figure 4**：`proline_sialic_crosscohort_summary`，显示 general CRC 与 mucinous-relative 分化；
5. **Figure 5**：修饰鸟苷、多胺和 acylcarnitine 的谱学家族与反证；
6. **Extended Data**：43-row mechanism evidence matrix、全部失败候选、归一化敏感性与外部异质性。

### 15.4 当前最短验证升级

如果只能增加一种湿实验，优先 **Neu5Ac 同法标准 + spike-in + linkage-aware glycan readout**，因为 proline/glutamate 已有强 source-Level-1 bridge，而 Neu5Ac 的生物学去向仍最不确定。若完全无湿实验，则投稿定位冻结为 `algorithm-enabled clinical discovery`，标题使用 `abundance programs`、`remodeling` 或 `mechanistic context`，不用 `mechanism` 或 `flux reprogramming` 作核心动词。

## 十六、竞争机制树：替代单一路径故事（2026-08-30 冻结）

当前五条 abundance axis 均存在至少两个能产生相同静态丰度方向的机制，因此论文不得用转录背景替代来源、通量或酶因果。14 个竞争机制已经固化到 `data/mtbls13729/competing_mechanism_trees_v1/`，完整解释见 `docs/MTBLS13729_COMPETING_MECHANISM_TREES_20260830.md`。

- 修饰鸟苷：writer deposition、RNA turnover、释放/组成三分；
- 乙酰化多胺：宿主 SAT1、相邻 positional isomer、微生物 biofilm 三分；
- 长链 acylcarnitine：fatty-acid entry、incomplete oxidation、substrate/composition 三分；
- Neu5Ac：precursor/donor pool、linkage-specific transfer、glycan degradation 三分；
- proline/glutamate：tumour-cell synthesis/redox 与 matrix/collagen source 二分。

因此主图中的每条路径箭头必须使用 `compatible with` 或虚线 hypothesis edge。只有标准品、来源/通量 readout 和节点扰动依次通过后，才能升级为实线机制边。

验证顺序也据此调整：最先购买 `m7G/m2G/Gm/m2²G`、`N1,N8-DiAcSpd/N1,N12-DiAcSpm/monoacetylspermidines` 和长链 acylcarnitine 组合标准。Neu5Ac 已有 source-Level-1 身份桥，其下一步重点是 free/donor/glycan 三层去向，而不是重复证明 nominal mass。

## 十七、亚型门后的最终主线重排（2026-08-30）

五模块与17个正相候选的 exact subtype-sensitivity 分析改变了主图优先级：

1. **主轴改为 Neu5Ac–mucin/glycan remodeling。** feature 703 是唯一跨归一化通过候选面板亚型 BH q<0.01 的充分覆盖节点；五模块层面的 q 为 `0.0016–0.0018`。
2. **多胺、酰基肉碱、氨基酸改为 general tumour parallel programs。** 它们可以支持 Rmu 内丰度背景和竞争机制，但不能写成 mucinous-specific。
3. **修饰鸟苷改为结构新颖性/低覆盖探索轴。** 1597 与3019仅各有2个严格有效 Rmu 患者，必须先解决异构体和检出覆盖，不能因效应大而升格。
4. **Figure 1–2 应首先展示终点分离。** 一张图同时给出 Rmu-RN 主效应与 Rmu-vs-Rtu interaction，并明确全空间 exact-FDR10 为0；随后才展示 feature 703 的 raw EIC/MS2、正交面板和外部 glycomic/transcriptomic context。
5. **最小实验优先级必须按目标分开。** 若强化生物学主轴，首选 Neu5Ac 同法 spike-in 加 linkage-aware O/N-glycan readout；若增加新化学实体，首选多胺与修饰鸟苷异构体标准。二者解决的是不同论文缺口，不能混为一个优先级。

独立 glycomics 的正确用法也已冻结：2024 CRC MALDI-MSI 显示 mucinous/non-mucinous N-glycome 可分；公开表中101条 composition 有19条含NeuAc，但没有患者级强度或组织学列，仍不足以复算患者级 Neu5Ac。2026 PXD055865 MUC2 空间糖肽研究证明黏液癌内 glycoform 异质性；三块肿瘤标本实际仅来自两位独立患者，Colon1a/1b属于同一患者，且鉴定数不能作为丰度。两者支持 linkage-aware 验证必要性，不构成 feature 703 的独立复制。PXD055865字段级审计见`docs/MTBLS13729_PXD055865_MUC2_GLYCOPEPTIDE_AUDIT_20260831.md`。

2022 CRC O-glycomics 现在可作为更直接但仍有限的外部结构层证据：两个 MUC 病例的 core-2 与 sLeX/A 均处于原发 AC/MUC 队列最高端，α2-6 处于最低端，且相对各自正常组织方向一致。主图或补图可放置 `free Neu5Ac abundance -> TCGA mucin/sialic program -> external core-2/sLeX/A structure` 三层证据链；图注必须标明外部 MUC `n=2`、Table S2 亚型裁决和“非 free-Neu5Ac 复制”。

2024 年 372 对患者匹配 CRC–正常黏膜的代谢生物地理研究新增一层独立患者组织背景。其官方
补充表将 Neu5Ac 定为标准品支持的 HILIC(-) Level 1，并显示正常黏膜 cecum-to-rectum 梯度
`+0.349`、p`<0.001`，肿瘤梯度衰减为 `+0.088`、p=`0.091`。这应进入 Result 4e 或 Extended
Data，证明解剖位置是 Neu5Ac 分析不可忽略的生物学维度。由于公开补充中没有 mucinous 字段，
它不能进入“独立黏液型复制”栏，也不解除 Rmu 全部右侧造成的 histology–location 混杂。

公开 Dash 图不能作为该队列的分析级患者数据：每种组织返回371个值、每亚部位固定53个、无
patient/pair ID，且直接回归不复现补充表。图中方向可作背景，正式效应与p值只引用补充材料。

## 十八、机制论文完成度硬门（2026-08-30）

机器可审计总账已升级为 `data/mtbls13729/mechanism_paper_completion_audit_v4_final/`。它在原16门基础上增加独立 Neu5Ac 空间背景门和公开数据复现边界门，共18个互不替代的证据门，防止把某一层成功外推为整篇机制闭环：

- 已通过：配对终点定义、Neu5Ac亚型发现、raw-data定量复核；
- 带限制通过：无 pooled QC/blank 条件下的配对归一化、同队列匹配背景、TCGA转录背景；
- 部分通过：精确MS2身份与空间/细胞来源；
- 明确未通过：完整13,155-target exact-FDR10、独立患者级代谢物复制、tracing/perturbation/rescue；
- 明确负结果：BioAware v1没有提高注释，患者级模块协调不支持单一上游链。
- 新增背景门：372 对独立 CRC 组织提供 Level-1 Neu5Ac 与疾病依赖空间梯度，但无黏液型亚组，
  因而只记为 `PASS_CONTEXT`，独立 Rmu abundance replication 仍为缺失。
- 新增复现边界门：公开 Dash 值的样本量与回归不复现补充表，只记为方向性背景，不进入患者级
  效应合并或配对分析。

因此“值得发表的现象”和“可以写成因果机制”必须分开。前者已经存在，即 `mucinous-relative hybrid mucin glycome`；后者尚未成立。任何投稿版本都必须把完成度总账中 `claim_forbidden` 字段当作摘要、主图和讨论的红线检查表。

## 十九、Hybrid mucin glycome 主图与投稿结构（2026-08-31）

新分支审计后，文章不再使用单线 `Neu5Ac up -> sialylation up`。Figure 4 的机制框必须并列三条
虚线证据流：

1. **free-pool/activated-donor layer：** free Neu5Ac 配对升高，但同患者 CMP-Neu5Ac 与
   UDP-GlcNAc 未同步；GNE/NANS/SLC35A1 supply/transport 仅作为黏液型相对转录背景；
2. **carrier/lineage layer：** MUC2/SPDEF/secretory-mucin 与 core-3/Sda 黏膜谱系在肿瘤间相对
   保留，但外部配对数据表明 core-3 在肿瘤转化中仍下降；
3. **structure/linkage layer：** 外部 MUC 显示 core-2/sLeX/A 获得与 alpha2-6 丢失，TCGA 的
   ST6GAL1 亦为负，排除简单 global hypersialylation 和 ST6GAL1–PD-L1 迁移。

三条线只允许在 `hybrid mucin glycome` 处虚线汇合，不允许画成 donor 直接驱动某个 linkage 的
实线因果箭头。正文新颖性写成 `donor–carrier–core–linkage decoupling`；不能写成首次发现
Neu5Ac，也不能写成首次发现 CRC 唾液酸化。

该结构已实现为主图 v2：`data/mtbls13729/neu5ac_glycan_publication_figure_v2_final/`。Panel A
为锁定 targeted-EIC，Panel B 为分支级 TCGA forest，Panel C 明示外部 MUC `n=2` 的配对结构，
Panel D 仅画虚线证据汇合并列出禁止外推。

投稿包分为三个等级：

| 等级 | 所需证据 | 可用标题/结论强度 |
|---|---|---|
| A：当前无新湿实验 | 锁定 Neu5Ac EIC、TCGA 分支、外部 n=2 O-glycomics、完整负证据与算法增量 | algorithm-enabled, evidence-calibrated clinical discovery |
| B：最小验证 | A + 同法 Neu5Ac standard/spike-in/内标 + linkage-aware O-glycan readout | validated selective mucin-glycan remodeling |
| C：因果机制 | B + 独立样本 authentic-standard CMP-Neu5Ac/ManNAc、MUC2 glycopeptide、isotope incorporation、节点扰动和 rescue | causal glycan-metabolic mechanism |

当前应优先完成 A 的可重复性和图表闭环，同时积极争取 B；在没有 C 的情况下，标题和摘要不得使用
`drives`、`flux reprogramming`、`enzyme activation` 或 `therapeutic target`。

同患者供体审计已经补上一个关键中间节点：10对Rmu中free Neu5Ac为10/10升高、均值
`+2.249 log2`，CMP-Neu5Ac和UDP-GlcNAc未同步；free-minus-donor/precursor的平均差为
`+1.693/+1.922 log2`，Holm-Wilcoxon p均`0.0273`。这把主图从“free pool + RNA donor”修正为
“free pool 与实测 activated donor 解耦 + RNA背景”，但CMP-Neu5Ac仍是Level 2，不能替代
Package B/C要求的标准品确认、独立组织和glycan destination。

来源分支也必须进入Figure 4的反证层：一般CRC中NEU1/NEU3升高，而mucinous相对conventional
在lineage/MSI校正后显著降低；因此不能把Rmu free pool画成NEU1/NEU3转录驱动。相反，
CMP activation/transport RNA轴相对升高而CMP-Neu5Ac pool不升，形成capacity–pool mismatch。
current-GDC已补齐NXPE1：mucinous相对conventional在临床+lineage模型中显著升高
（beta=`+0.621`、p=`0.000369`），但加入`MUC2/TFF3/SPDEF/FCGBP/AGR2`完整secretory-mucin程序后
效应消失（beta=`+0.064`、p=`0.734`；再加MSI为`-0.048`、p=`0.782`），leave-one-out和双marker
敏感性也支持这是分布式carrier state，而非单个标志或已证实独立驱动。一般CRC的50对current-GDC
tumour-normal中NXPE1为47/50降低；GSE236696六对黏液癌上皮pseudobulk也为6/6降低，但因低计数
和p=`0.0625`只作方向支持。因此Figure 4应把NXPE1画在`secretory carrier-linked capacity`层，
不是从free Neu5Ac直连出的实线酶反应；蛋白、活性、具体O-acetyl位置与glycan destination仍用
虚线问号。free Neu5Ac和CMP-Neu5Ac均有文献acceptor context，图中不得把本地free pool指定为
体内唯一或直接NXPE1底物。

原始negative-HILIC随后对`m/z 350.109269 [M-H]-`进行了表型盲全RT审计，冻结出4.29和
5.55分钟两个独立峰（分别50/60与54/60样本支持；47与56张RT分层MS2）。两个峰几乎都含
`m/z 87`，但Rmu完整配对BH q均为`0.930`，患者变化与free Neu5Ac亦不相关。Figure 4因此应把
`bulk mono-O-acetyl-Neu5Ac-like pool not increased`放在反证侧；Extended Data展示完整色谱、
患者热图和碎片频率。该结果不能定位4/7/8/9-O-acetyl异构体，也不能否定glycan-bound或
细胞型特异O-acetylation。完整审计见
`docs/MTBLS13729_OACETYL_NEU5AC_LIKE_AUDIT_20260831.md`。

### 19.1 算法—生物学归属边界

主文不得把 feature703 写成 E6/P2b 新发现。Neu5Ac 身份来自原论文 Level-1 节点的正交面板恢复；E6/P2b 的可报告增量分别是谱学证据稳定性和候选覆盖。当前本地缺少三路 feature-level 工件，因此只允许报告三路总体数字，候选级归属标记为待同步复核。对应冻结审计为 `docs/MTBLS13729_ALGORITHM_TO_BIOLOGY_INCREMENT_AUDIT_20260831.md`。

## 二十、独立患者raw-UMI与组成调整后的Figure冻结（2026-08-31）

主图已升级为`data/mtbls13729/neu5ac_evidence_convergence_figure_v3_final/`，v2结构图继续作为
糖链分支扩展图保留。v3四个panel固定为：

1. 同10位Rmu患者的free Neu5Ac、CMP-Neu5Ac和UDP-GlcNAc配对变化；
2. GSE178341中6例mucinous、53例conventional的raw-UMI患者级上皮系数，必须同时展示
   unadjusted与goblet-fraction-adjusted结果；
3. 15例mucinous、15例conventional独立蛋白组固定面板，所有区间跨零和BH阴性必须保留；
4. 非因果汇合模型，同时列出CMP pool不扩张、host NEU1/NEU3不支持、NXPE1不独立和蛋白模块
   未确认，以及独立代谢物复制/同法标准/同一样本糖组/tracing仍缺失。

上皮组成诊断的正确结论是：MUC2、SPDEF和NXPE1主要随goblet/secretory composition衰减，
AGR2和SLC35A1在组成调整后仍保留正区间。因此主文使用`selective epithelial secretory-folding
and Golgi donor-transport capacity`；不得使用`all goblet cells activate the pathway`或把调整模型
写成因果中介。完成度权威工件改为25门的
`data/mtbls13729/mechanism_paper_completion_audit_v10_final/`，前述v4/v8/v9仅作版本历史。
