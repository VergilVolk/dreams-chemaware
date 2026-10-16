# Metabolomic Contrast Atlas 主科学合同

日期：2026-10-12  
状态：恢复后的生物学应用总纲；取代把 Contrast Atlas 排除或把单个验证面板升级为总方法的定位  
论文位置：统一算法论文后半部的规模化应用与旗舰发现；不取代前部候选排序算法

## 1. 核心问题

> 能否把海量公共非靶向 LC-MS/MS 从“可搜索谱图的仓库”推进为“可检索生物学变化的知识空间”，使尚未获得精确结构身份的化学信号也能以效应方向、大小、不确定性和证据资格参与跨研究发现？

本方法的索引对象不是单张谱图、候选结构、kNN 连通分量或某个遗传实验，而是 **study-level contrast signature**：一项研究内部，在冻结 QC 与设计模型下估计的化学响应及其可观测边界。

`unbiased` 只表示进入发现不以结构名称、谱库命中或预选通路为条件；不否认提取、电离、色谱、DDA、样本选择和公共数据偏差。

## 2. 两层 Atlas

### 2.1 Spectral Knowledge Atlas

保存海量公共 MS/MS、来源、仪器条件、前体和离子信息、已知/未知状态、冻结谱学表示、碎裂模式与资格分层。它回答“有哪些可复核的谱学与化学证据”。首轮验证可复用现有公共索引，不要求重新下载两亿张谱图。

### 2.2 Biological Contrast Atlas

只收录具有可追溯 MS1 定量、真实样本设计、可核查 MS/MS 联结和合格 QC 的研究。每个 contrast 保存 study/source、数据角色、生物学独立分母、设计、entity 或 independent ion-family 的 `beta`、`SE/CI`、检测覆盖、blank/QC/缺失、MS2 获取倾向、身份状态、证据等级和禁止外推边界。

它回答“这些化学证据在什么条件下怎样变化”。不同实验室的原始峰强度不得直接合并；跨研究比较只发生在研究内效应冻结之后。

## 3. 两个对称入口

### A. 谱图/未知实体到生物学响应

输入未知 MS/MS、entity 或谱学模式，返回公共研究中的严格实体复现、共享化学响应、方向、效应区间、异质性和 no-match。谱学邻近本身不等于生物学功能。

### B. 生物学对比到公共响应

输入病例—对照、敲除、给药或其他 contrast，构建已知与未知共同参与的冻结响应签名；搜索其他研究中相似、相反或不可比较的响应。返回可比较覆盖和错误风险，不强迫每个研究都有匹配。

## 4. 核心算法：多层级、可弃权的化学响应检索

三个层级必须分开输出，禁止加权成一个无法解释的总分：

| 层级 | 允许回答 | 必须拒绝的情形 | 禁止升级 |
|---|---|---|---|
| 严格实体对应 | 两研究中足够支持同一分析实体的变化是否一致 | 身份概率低、离子形式/异构体不清、coverage不足 | 精确结构或通量 |
| 共享化学模式 | 实体集合是否呈现相似碎裂化学响应 | 常见碎片、单峰支撑、模式未校准、来源偏差 | 同一分子或反应 |
| 研究对比关联 | 疾病响应是否与某功能扰动显著相似或相反 | 可比较模式太少、设计混杂、效应不稳 | 因果机制 |

对研究 `d` 中 independent unit `i`，先估计 `beta_di, SE_di`。对表型盲、独立校准的化学证据 `k`，以 `w_ik` 形成研究级模式响应：

`C_dk = sum_i(w_ik * beta_di) / sum_i(w_ik)`。

实现必须传播 SE，折叠冗余离子，控制模式常见度、DDA获取倾向和相关性，并只在双方共同可观测区域比较。实体数量或离子数量不是独立生物学分母。

## 5. 双向、留出式验证

**功能扰动 -> 疾病：**在有真实标签的敲除/药物研究中建立冻结响应，隐藏身份或功能标签并检验能否恢复；通过后才用于疾病队列搜索。

**疾病 -> 功能扰动：**在疾病队列中冻结结构隐藏响应，在 Atlas 中提出功能扰动预测；随后只在未参与选择和建模的研究中验证。

两个方向都必须预测未用于建模的信息。失败后不得用静态反应数据库或事后文献邻接挽救。

## 6. 现有资产的正确位置

| 资产 | 在本方法中的角色 | 不是 |
|---|---|---|
| Noise / official DreaMS / WSE | 条件稳健谱学证据与强基线 | 单独的生物学结论 |
| P2b | 局部歧义下的冻结判别证据 | 与原始通道重复计票的专家 |
| ChemAware | 资格范围内的碎裂约束与开发消融 | 默认全局增益 |
| BioAware | 通过 B47 后的真实事件级条件证据 | 静态目录/拓扑先验 |
| entity/relation transport | 跨研究对应算子 | Atlas 总方法 |
| public certificate | 资格、来源和弃权层 | 科学发现本身 |
| ChemProp | 药物跨设计验证 shard | drug-response 总方法 |
| ST003325 | 冻结负 shard | 可反复调参的训练集 |
| MSV000096214/TT7 | 强遗传正控与谱学增量 shard | 总方法或旗舰疾病发现 |
| LCNEC/MTBLS13729 | 疾病端查询和旗舰候选 | 单独足以证明机制的案例 |

## 7. 最小可发表原型

先建小型、严格的 Contrast Atlas，不建网站。首轮只裁决三件事：

1. **结构隐藏响应恢复**：不使用结构名称，能否在 held-out 干预/疾病研究中恢复真实 contrast；
2. **跨研究检索**：完整留出研究或来源，能否检索到方向正确的独立响应，并在无对应时弃权；
3. **未知条件增量**：`known + unresolved` 是否严格超过 `known-only`、measurement-matched known 和 unresolved shuffled-MS2，且同一错误风险下成立。

必须比较 MS1-only、m/z/RT、raw cosine、WSE、official DreaMS、Noise、冻结 P2b、MEMO、PALS/MS2LDA 的适用版本，以及标签、谱图和对应置乱。不得只报告差异峰数、样本分类、UMAP 或网络相似度。

## 8. 通过门

- 至少两个独立 source family；同项目不同表不算独立来源；
- 谱学对应构建不得读取待预测效应；
- 报告 coverage、no-match、错误关联、相反方向和异质性；
- 未知增量在 source/instrument 留出中同号，cluster-level CI 下界严格大于零；
- 删除公共谱学层、unresolved 和各证据族的消融完整；
- 旗舰应用必须产生预注册、可失败且由独立扰动支持的预测；
- 任一层失败只降低该层主张，不通过扩大数据库或放宽相似度补救。

## 9. 数据边界

MassSpecGym、GNPS identity/formula panels、ChemProp、ST003325、MSV000096214 和现有 LCNEC/MTBLS13729 分析均为 `development/consumed`。Enveda-180 是候选排序算法的最终未开封外测，不自动成为 Contrast Atlas 生物学外测。Atlas 的独立确认必须另行登记，并在 schema、阈值、检索终点和停止门冻结后打开一次。

## 10. 近期执行顺序

1. 冻结机器可读 contrast record、strict-entity link、pattern loading 和 retrieval-result schema；
2. 用现有 consumed shards 做端到端 smoke 与负结果重放，不宣称外部验证；
3. 选择至少两个有 raw MS1/MS2、设计和 QC 的独立来源构建小 Atlas；
4. 完整研究留出，运行 strict entity / shared pattern / contrast association 三层检索；
5. 只有 unknown 条件增量与 source 留出通过后，扩大公共研究规模；
6. 在疾病端提出一个独立扰动可验证的旗舰预测。

当前机器合同与最小执行器分别为
`tasks/contrast_atlas_schema_v1.json` 和 `tasks/contrast_atlas_core.py`。它们只实现输入资格、三种证据世界、严格实体层、共享模式层、跨来源过滤和证据不足弃权；当前固定阈值是开发原型参数，不是生物学结论，也不得在打开独立确认集后调节。

方法竞争边界、双侧可观测覆盖、经验 null、主终点和首轮实证路线见
`METABOLOMIC_CONTRAST_ATLAS_METHOD_FRONTIER_20261012.md`。

## 11. 目标主张（当前尚未建立）

> We test whether public untargeted LC-MS/MS can be indexed by study-level chemical responses, rather than structure names alone, so that unresolved chemistry contributes to calibrated cross-study retrieval and falsifiable biological discovery.
