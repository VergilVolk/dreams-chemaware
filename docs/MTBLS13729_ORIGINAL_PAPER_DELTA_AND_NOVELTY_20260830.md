# MTBLS13729 原论文对账、DreaMS 增量与新颖性边界（2026-08-30）

## 结论先行

本次重分析不是把原论文的 345 条注释重新命名。原论文已经建立了广泛的肉碱、嘌呤和脂质变化；DreaMS 重分析的真实增量分为三类：

1. **原表缺失的离子家族**：methylguanosine-like 与 dimethylguanosine-like 家族及其跨加合物证据；
2. **原名存在、但发现了不同色谱峰和不同生物学定位**：acetylated-polyamine / N1,N8-diacetylspermidine-like feature 1717；
3. **原通路已存在，但增加了更细的类别锚点和不同机制解释**：long-chain/C20:4-acylcarnitine-like feature 3222。

最强、最诚实的论文贡献不是“首次发现肉碱代谢或 N1,N8-diacetylspermidine”，而是：

> 以表型盲谱学、峰级原始 MS2、离子家族折叠和配对重定量，找回原作者 m/z–RT 注释表未解析的修饰鸟苷离子家族，并把乙酰化多胺和长链酰基肉碱信号放入可审计、证据分级的 Rmu 代谢状态中。

新增的 15 候选定量审计进一步把这条边界钉死：9 个为原身份重映射，1 个为原名存在但新增正交峰证据，4 个为原 identity table 未列出的算法候选家族，1 个因生物学不一致而降级。10 个 source-linked 节点的重提取效应与原表 Rmu 效应 Spearman `rho=0.830`，但来自同一队列，只能叫技术一致性。详细逐候选表见 `data/mtbls13729/biology_novelty_audit_v1/candidate_novelty_layers.csv`。

## 1. 原论文补充表已经包含什么

原论文 Table S4（`pr5c01260_si_005.xlsx`）包含：

| 项目 | 数量 |
|---|---:|
| UHPLC 注释 | 345 |
| Level 1 | 157 |
| Level 2 | 188 |
| Ltu vs Rtu nominal p<0.05 | 72 |
| Rtu vs Rmu nominal p<0.05 | 92 |
| Ltu vs normal | 133 |
| Rtu vs normal | 93 |
| Rmu vs normal | 93 |
| cancer vs normal | 226 |

原作者已覆盖：

- 24 条 carnitine 上下文；其中 11 条在 Rmu-vs-normal 表中；
- 13 条 purine/nucleoside 上下文；
- 6 条 polyamine 上下文；
- 15 条 LysoPE 上下文。

因此，凡是写成“本研究首次发现肉碱通路、嘌呤通路或多胺代谢异常”都不成立。

## 2. 八候选逐项对账

| feature | 当前可守身份 | 原作者是否有同名 | 严格 m/z+RT 匹配 | 真实增量 | 主文位置 |
|---:|---|---|---:|---|---|
| 1597 | methylguanosine isomer family `[M+H]+` | 否 | 0 | 原表未列的修饰鸟苷家族；42 张峰界内 MS2，30/42 见核糖丢失 | 主线 |
| 3019 | dimethylguanosine isomer family `[M+H]+` | 否 | 0 | 原表未列的二甲基鸟苷家族；32/32 见核糖丢失 | 主线 |
| 7489 | 1597 的 `[M+Na]+` 支撑 | 否 | 0 | 跨加合物证据，不是独立发现 | 补充 |
| 1717 | acetylated-polyamine / N1,N8-diacetylspermidine-like | 是 | 0 | 原作者 HILIC 同名，但本峰是独立 pos-RP 信号并在 Rmu 显著升高 | 主线的正交轴 |
| 3222 | long-chain acylcarnitine; C20:4-like | 否 | 0 | 原肉碱项目中新增长链锚点与肉碱穿梭失衡竞争假说 | 次级机制支撑 |
| 4966 | C7H9N5O purine-like isomer family | 否 | 0 | 原嘌呤背景中的新同式异构家族信号 | 次级伴随轴 |
| 3180 | unknown chlorinated/exogenous-like | 否 | 0 | 可复现但不可生物学解释 | 负向控制 |
| 16425 | unknown lipid-like；旧 LPE-like 未确认 | 否 | 0 | 结构不足 | 探索性补充 |

所有 8 个候选与原作者表的严格 m/z+RT 匹配数均为 0。这个结果只证明它们不是原表中同一色谱 feature，不能自动证明它们是新分子。

## 3. feature 1717：不能说“新发现这个名字”

原作者已经报告 `N1,N8-Diacetylspermidine`：

- HMDB0041947；
- HILIC；
- m/z 230.18311；
- RT 2.624 min；
- Level 2。

DreaMS feature 1717 为：

- positive-RP；
- m/z 230.185931；
- RT 1.493 min；
- 73 张峰界内 MS2 / 45 个样本；
- m/z 100.0759 在 73/73 谱图中均为 base peak；
- Rmu 9 对平均 `+3.009 log2`，9/9 正向，精确 sign-flip `p=0.003906`。

feature 1717 与同源 HILIC 注释在 59 个样本上的 Spearman `rho=0.860`，患者配对差值 `rho=0.719`，且在 85 个 HILIC 注释中相关排名第一。这使“同一乙酰化多胺家族”具有很强正交支持，但两个色谱 feature 并非严格同一峰。

允许写：

> A distinct positive-RP acetylated-polyamine/N1,N8-diacetylspermidine-like feature showed strong Rmu-associated accumulation and cross-chromatographic concordance.

禁止写：

> We newly discovered N1,N8-diacetylspermidine in this cohort.

## 4. feature 3222：不是肉碱通路新发现，而是解释修正

原论文已经把多种 carnitine 和 acylcarnitine 上升解释为 activated carnitine shuttle。原作者 Rmu-vs-normal 显著条目包括 dodecanoyl-, myristoyl-, octanoyl-, heptadecanoyl-, 9-hexadecenoyl-, linoleyl-, oleoyl-carnitine 等。

DreaMS feature 3222 的增量是：

- m/z 448.33946，C27H45NO4；
- 59 张前体/RT 匹配 MS2，30 张严格峰界内谱；
- carnitine 类诊断碎片在 25 个样本中强支持；
- Rmu 10 对平均 `+1.776 log2`，8/10 正向；
- 提供一个长链/C20:4-like 类别锚点。

独立 pooled mucinous CRC 蛋白组中，carnitine shuttle/long-chain FAO 固定面板整体偏低；GSE236696 的 6 对黏液型上皮 pseudobulk 中，该轴 0/6 上升。这与 acylcarnitine 堆积并列时，更适合提出“进入后利用受阻/线粒体处理瓶颈”假说，而不是简单等同于 FAO 增强。

但是，acylcarnitine 丰度同时受脂肪酸输入、CPT1、CACT/SLC25A20、CPT2、β-氧化、外排和组织细胞组成影响。静态丰度加转录/蛋白方向不能证明通量。因此只能写：

> The joint evidence is consistent with carnitine-shuttle imbalance, with increased entry, incomplete oxidation and impaired downstream utilization retained as competing hypotheses.

不能写：

> FAO flux is suppressed in Rmu tumors.

## 5. 修饰鸟苷是最强注释增量

原作者表有 guanosine、methylguanine 等嘌呤背景，但没有 methylguanosine/dimethylguanosine ion families。我们的证据包括：

- 1597/7489 和 3019/8481 两组跨加合物质量关系；
- 相同的 132.042 Da 核糖丢失；
- 1597 的 Rmu 模块 `+3.72 log2`，3019 `+2.40 log2`；
- 离子家族折叠后 10/10 Rmu 患者正向，平均约 `+2.95 log2`；
- 与独立 purine-like feature 4966 患者内相关 `rho=0.903`；
- 独立 Level-1 CRC 组织数据出现反方向/背景依赖结果，否定“泛 CRC 标志物”，提高了亚型/异构体特异假说的价值。

最合理的主命题是“Rmu 中修饰鸟苷/嘌呤周转轴”，而不是指定 METTL1 或某一个位置异构体。METTL1、tRNA m7G 和 tsRNA 文献只用于生成后续可检验机制，不用于提升当前身份或因果等级。

## 6. 当前论文能回答原论文什么未解决问题

| 原论文留下的问题 | 当前回答 | 证据等级 |
|---|---|---|
| m/z+RT 注释表是否漏掉可重复离子家族？ | 是；修饰鸟苷、乙酰化多胺和长链 acylcarnitine 候选有原始 MS2/离子家族证据 | 强发现级 |
| Rmu 的变化是否只是一个个孤立峰？ | 修饰鸟苷与独立 purine-like feature 构成患者内共变轴；多胺和 acylcarnitine 为平行轴 | 中等 |
| carnitine accumulation 是否等同 FAO 激活？ | 不能；外部蛋白/上皮 RNA 的 FAO 轴偏低支持利用受限分支，但 2026 carnitine/acetylcarnitine–CPT1A 研究支持输入/利用增强分支 | 竞争假说级 |
| 乙酰化多胺是否可能连接酸性微环境/免疫？ | 文献上可检验；本队列只有代谢物和有限表达/空间背景，尚无因果链 | 假说级 |
| 是否证明 Rmu 特异？ | 10 对发现集提示，独立 subtype-resolved 代谢组缺失 | 未确认 |
| 是否证明结构和通量？ | 否；需标准品与示踪/扰动 | 未完成 |

## 7. 投稿时的贡献层级

### 可以作为主结果

1. DreaMS-enabled raw-MS2 reanalysis recovered previously unlisted modified-guanosine ion families.
2. 离子家族折叠、峰界内 MS2、跨加合物和患者内配对共同支持 Rmu 修饰鸟苷轴。
3. feature 1717 的强峰级 MS2、跨色谱一致性和 Rmu 方向形成独立乙酰化多胺证据。
4. 原论文 carnitine 结论经外部多组学重新解释为“肉碱穿梭失衡”，输入增强、利用受限和不完全氧化均保留，而非单向激活或单向下降。

### 只能作为次级/讨论

1. mucinous specificity；
2. METTL1/WDR4 或 SAT1 因果；
3. neutrophil recruitment；
4. FAO flux；
5. 精确位置异构体。

## 8. 可复核工件

- 原论文补充表：`data/mtbls13729/source_paper_supplements/pr5c01260_si_005.xlsx`
- 对账脚本：`tasks/audit_mtbls13729_original_paper_delta.py`
- 对账结果：`data/mtbls13729/original_vs_dreams_biology_delta_v1/`
- 投稿准备度矩阵：`data/mtbls13729/manuscript_readiness_v1/candidate_manuscript_readiness.csv`
- 整合证据图：`data/mtbls13729/integrated_biology_evidence_v1/integrated_biology_evidence.png`

## 9. 主要文献

- 原 MTBLS13729 论文：https://pmc.ncbi.nlm.nih.gov/articles/PMC13459539/
- Acidic pH–SAT1–N1-acetylspermidine：https://pmc.ncbi.nlm.nih.gov/articles/PMC10563787/
- METTL1–tRNA m7G–CRC progression/metastasis：https://pmc.ncbi.nlm.nih.gov/articles/PMC12864287/
- tsRNA-GlyGCC–METTL1–5-FU resistance：https://pmc.ncbi.nlm.nih.gov/articles/PMC11330149/
- CRC AHCY mechanism benchmark：https://pmc.ncbi.nlm.nih.gov/articles/PMC10447251/
