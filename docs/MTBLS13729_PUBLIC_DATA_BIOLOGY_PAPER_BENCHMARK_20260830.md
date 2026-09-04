# MTBLS13729：高水平非靶向 LC–MS/MS 生物学论文对标与无新湿实验投稿边界

## 1. 先区分三种论文，不再混用门槛

### A. 算法方法论文

核心问题是“注释或检索是否更准、覆盖是否更高、错误是否可控”。必须具备：严格分子/骨架隔离的外部测试、与强基线比较、消融、校准/弃权、泄漏审计、运行成本，以及至少一个真实队列应用。真实应用用于证明效用，不替代盲测准确率。

### B. 公开数据驱动的生物学发现论文

核心问题是“在没有新湿实验时，能否从多个独立公开数据中得到同一预注册生物学命题”。必须具备：发现队列的原始数据重处理、独立队列的同分子或同轴复核、患者作为统计单位、负结果与异质性、身份分级、跨组学只作正交背景。可以形成 `mechanism-supporting discovery`，不能声称酶因果、通量或治疗靶点已经建立。

### C. 因果机制论文

核心问题是“哪个来源、哪个反应、哪个酶、哪个细胞和哪个表型”。典型闭环为：非靶向发现 → 标准品/靶向定量 → 独立队列 → 细胞/空间来源 → 同位素示踪 → 基因或药理扰动 → rescue → 动物/临床外推。没有 tracing、perturbation 和 rescue 时，不得把 B 类论文包装成 C 类。

## 2. 顶尖方法论文真正验证了什么

| 方法 | 关键设计 | 防止“覆盖高但不真实”的措施 | 对本项目的启示 |
|---|---|---|---|
| MetDNA / MetDNA3 | 反应网络邻居、数据驱动与知识驱动双层网络、递归传播 | seed/validation 拆分、RT 与 MS2 约束、自检评分、跨样本/平台验证 | BioAware 不能只靠一跳邻居投票；必须有数据层共洗脱/共变边、种子外验证和传播终止条件 |
| DeepMet + CFM-ID | 生成候选化学空间，再用预测 MS2 排序 | held-out metabolite、第二谱库复现、decoy 谱、synthetic standard；作者仍强调真实数据需正交确认 | 扩候选库必须同时增加 decoy 与校准；生成结构不能自动升级身份 |
| TidyMass2 | 将代谢物来源、MS1 候选和综合反应网络用于功能模块 | 多数据库网络、置信评分、冗余过滤、模块而非单峰解释 | 未注释峰可进入模块，但身份不确定性必须传播到模块结论，网络一致性不能当结构真值 |
| NetID/Binner | 同位素、加合物、碎片、共洗脱与生化质量差形成全局图 | process blank、RT/相关性、全局一致性、人工可视化复核 | 本项目应继续以中性分子/离子家族为统计单位，避免把同一代谢物多个离子重复当新发现 |

主要来源：

- MetDNA: https://www.nature.com/articles/s41467-019-09550-x
- MetDNA3: https://www.nature.com/articles/s41467-025-63536-6
- DeepMet: https://www.nature.com/articles/s41586-025-09969-x
- TidyMass2: https://www.nature.com/articles/s41467-026-68464-7
- NetID: https://pmc.ncbi.nlm.nih.gov/articles/PMC8733904/
- Binner: https://academic.oup.com/bioinformatics/article/36/6/1801/5603305

## 3. 顶尖机制论文真正多做了哪些层

| 标杆 | 非靶向发现 | 身份/定量 | 来源/空间 | 通量 | 扰动/rescue | 当前项目缺口 |
|---|---:|---:|---:|---:|---:|---|
| CRC AHCY, Nature Metabolism 2023 | 是 | targeted + MSI | GEMM/临床/组织 | `13C5` methionine | organoid、shRNA/药理、in vivo | 缺标准、tracer、AHCY/SAT1/METTL1/CPT 轴扰动 |
| Spatial isotope deep tracing, Nature Communications 2025 | 是 | 多 LC 与 MS2 | 全身 MSI | `U-13C` glucose/glutamine | 肿瘤模型 | 本项目静态丰度不能替代标记分数 |
| 单细胞动态代谢, Nature Communications 2025 | 是 | 单细胞代谢 | 肿瘤–巨噬细胞 | isotope tracing | 共培养/细胞互作 | 本项目单细胞 RNA 只定位转录背景，不测代谢物流 |

来源：

- AHCY: https://www.nature.com/articles/s42255-023-00857-0
- Spatial isotope tracing: https://www.nature.com/articles/s41467-025-63243-2
- Dynamic single-cell metabolomics: https://www.nature.com/articles/s41467-025-59878-w

## 4. MTBLS13729 当前已经达到的层级

| 层 | 状态 | 证据 |
|---|---|---|
| 原始队列重处理 | 已完成 | MS1 重定量、峰界 DDA、四面板对账 |
| source identity 对账 | 已完成 | 9 个原身份重映射；source-linked 效应 Spearman `rho=0.830` |
| 算法新增候选 | 已完成但仍为家族级 | palmitoylcarnitine、C20:4-acylcarnitine-like、methyl-/dimethylguanosine families |
| 离子家族去冗余 | 部分完成 | `[M+H]+/[M+Na]+`、核糖丢失、跨样本复现 |
| 独立外部背景 | 已完成 | OEP00006137、ST001087、GSE236696、TCGA、黏液型池化蛋白组、MTBLS7387 |
| 独立同分子同亚型复现 | 未完成 | 未找到第二个患者级黏液型组织代谢组 |
| 标准品位置异构体确认 | 未完成 | m7G/m2G/Gm/m2²G 与 feature 1717、C20:4 标准待验证 |
| 通量/酶因果/rescue | 未完成且现有数据不能补 | 无 tracer、无扰动、无 rescue |

## 5. 在“无新湿实验”约束下，最强可投稿形态

### 主命题

> Evidence-calibrated reanalysis of paired mucinous colorectal tissues recovers a context-dependent modified-guanosine/purine program together with parallel acetylated-polyamine and long-chain acylcarnitine abundance programs.

### 论文价值

1. **方法价值**：DreaMS/离子家族/峰界 MS2 提高未解析家族覆盖，证据校准同时展示一个真实纠错（phenylalanine）和一个真实降级（taurine）。
2. **生物学价值**：修饰鸟苷/嘌呤、多胺乙酰化、长链 acylcarnitine 三轴在本地 Rmu 中形成强配对表型，并由公开单细胞、蛋白组和外部代谢组提供方向性或反向异质性证据。
3. **概念价值**：修饰核苷不是泛 CRC 统一升高；acylcarnitine accumulation 也不能简化为 FAO 单向激活/抑制。论文主动保留 context/isomer/flux 的竞争解释。

### 不能承诺

- “发现了新的化学实体”；
- “证明黏液型特异性”；
- “证明 METTL1/SAT1/CPT1A 因果”；
- “证明 FAO、甲基化或多胺通量方向”；
- “达到 MSI Level 1”，除非同法标准共洗脱与完整 MS2 通过。

## 6. 还必须补齐的计算层闭环

1. 冻结 15 候选的新颖性表和失败候选表进入补充材料，避免选择性报告；
2. 用冻结 source identity 对 15 候选逐项评估 official DreaMS、E6 embedding 与 P2b：报告 correct/incorrect/abstain，不用 P2b-only 名称作真值；
3. 把 4 个 source-table-absent 家族在全部可达谱图中做离子家族、共洗脱、同位素和谱峰镜像图；
4. 以患者为单位完成模块效应、leave-one-feature-out、临床敏感性与所有负结果；
5. 将外部证据矩阵按 `same metabolite / same pathway / transcript context / contradictory` 四级分层；
6. 冻结一版主文结果、补充表和图，后续任何新候选只进入探索补充，不再改变主终点。

## 7. 决策

在没有新湿实验的现实条件下，本项目可以完成一篇扎实的“算法方法 + 公开原始数据驱动生物学应用”论文；不能负责任地承诺一篇已经建立酶—通量—表型因果链的纯机制论文。最有效的路线不是继续堆网络富集，而是把结构证据、患者配对丰度、独立背景、反证和算法增量全部冻结到同一证据矩阵中。
