# DreaMS 项目全史阅读与综合裁决（2026-08-01 → 2026-10-29）

日期：2026-10-29
作者：GLM（生物应用线）
阅读方式：亲读全部 2026-10-08→10-29 核心结果文档（约 65 份逐字）+ 亲读早期奠基文档（second_report 590 行、ALGORITHM_INTEGRATION 权威状态页、BIOLOGY_FRONTIER_RESET、UNBIASED_CORE、10-11 科学决策、GLM_PROJECT_OVERVIEW_20260930 全 217 行）+ 全部 520 份 8 月以来文档经读史官逐字阅读并落盘结构化摘要（tasks/_history_review/，含批次与分块双套摘要，失败自动补读）+ 批次 2/3 摘要与 MASTER_INDEX 交叉核验。
本文冲突规则：与任何旧文档冲突时，以更晚的实现修复、冻结账本和机器工件为准。

---

## 时代一：规则注入与评价纠偏（08-01 → 08-22）

**主线**：把"化学碎裂规则注入 DreaMS"的原始设想，经过系统证伪收敛为"化学感知微调 + 独立解释模块"。

- 规则库两层建成：335 条核心可解释规则（NL 214 / CF 102 / ISO 8 / HR 9）+ 3,151 条 MassBank 经验模式；规则数据与执行代码分离，76,157 个 IK14 预计算缓存。
- annotated01 数据湖（3,263,214 谱 / 76,157 分子）审计后判定不可直接训练：23,713 个 IK14 混合极性、与 MassSpecGym 重叠 36.5%。
- **Task 0 否证实验（决定性）**：Rule Jaccard 与 MCES 的 Pearson 仅 −0.167/−0.194——规则不是全局结构距离尺。角色重分配：全局距离删除、可解码概念保留、困难样本发现器强化（1,925 个高 Jaccard/高 MCES 冲突对）、峰级解释证据保留。
- MIL 十折被简单线性聚合击败（0.224 vs 0.286）——"复杂聚合器创造解释"路线排除。
- **"0.85 事件"**：发现论文 0.85（NIST20 协议）、本地旧 0.8828（随机负样本）、严格 E0 0.6641（10-ppm 难候选池）不可横向比较——评价协议纪律由此建立。
- 反事实峰微调时代（P0 失败案例审计 → P1 规则噪声 pilot → 化学掩码 GPU 运行簿）：CPU head 审计发现量纲与实现问题，为后续"执行纠偏"文化奠基。
- 08-16/08-17 出现**平台化愿景**（ANNOTATION_TOOL_DELIVERY_ROADMAP 37.8KB、ANNOTATION_PLATFORM_ARCHITECTURE、PRODUCT_WHITEPAPER、0827 指导的"mzML→encoder→结构生成解码器"终极形态）——后被 PI 明确否定为"不是 Atlas 工程"的方向，但作为论文应用章节与下一代叙事保留。

**裁决**：规则不能硬塞进注意力（第一次汇报已证），也不能定义 embedding 距离（Task 0）。活下来的资产：两层规则库、annotated01 数据湖、MCES 分层评价、E0-E5 阶梯训练设计。

## 时代二：错误图谱与重排器纪律（08-22 → 08-30）

**主线**：从冻结盲测的失败学习，建立 P2b 与错误机制语言。

- **RAW-v1 盲测失败**：开发 +4.35pp → Test-A/B +0.45/+0.73pp（CI 跨 0）。教训：开发集增益必须经冻结盲测。
- **P2b 秩融合冻结**（0.1 DreaMS + 0.1 entropy + 0.8 neutral-loss）：封存 P3-main +1.07pp（CI [+0.24,+1.89]，McNemar p=0.0101）但 near-core −4.23pp 硬边界——最稳的系统级成果。
- 四层架构宪法（08-23 ChemAware 决策）：共享 embedding / P2b / 双重映射解释 / 生物网络后验。
- 噪声 v3 一天五轮（G1/S1a/S1b/S1c）：23,876-query 全图动作空间扫描。role-confounder 最干净、role-identity 负对照通过、shared/整组删除否决；oracle 上限 386→663→799→920（+3.853pp）但三进入门全 FAIL——**瓶颈诊断为 positive-deficit（910/1,029 未恢复错误）**。
- A4-B0 教师 +9.62pp、C1 支持互斥 +2.47pp 证明 P-arm 真实。
- 08-26 执行纠偏：D1b clean-only 被误当主结果被纠正；唯一主线锁定"原始谱图→同一编码器→新 embedding"。
- **E4-A 定向噪声微调成线**：+0.635pp overall / +0.522pp near，15/15 折 CI>0——第一个真实权重改善（与 3.85pp oracle 的差距命名为 action-to-weight transfer gap）。
- E5-E9 系统性证伪：全局固定引导稀释信号、outcome-mined oracle 是选择偏差（mined 比 fixed 差 0.4pp+）、冻结 reference 负贡献证明共享几何必要。
- BioAware 弧线：v1 工程全链建成但 MTBLS1905 1/0、MTBLS13729 20/21→19/21 净引入 1 错（GABA 被 Rhea 转氨路径以 0.002272 翻转）；MetDNA3 117-query 一跳 CI 跨 0、raw-MS2 在错误上显著反向；G3-v2 +1.71pp 被审计撤回（零权重计票+嵌套证据重复计数→0/0）→ 降级为基础设施，创新转向 BioChem-Embed。

## 时代三：MTBLS13729 疾病端全循环 + B 系列（08-29 → 09-21）

**主线**：疾病端完成"发现→扩张→反证→重排"完整循环；BioAware B20-B40 用外部面板逐个否决过度声明。

- 修饰鸟苷模块（10/10 同向 +2.953 log2，p=0.001953）→ 三轴→四模块→六轴扩张（proline/glutamate/Neu5Ac 正交找回）→ 外部 Level-1 反证（MSI −0.778）→ 亚型交互门裁决：**只有 Neu5Ac 通过**（+2.209/+2.142，q≈0.0017）→ 主轴重排为 Neu5Ac/mucin-glycan。
- Neu5Ac 深挖：donor 解耦（free +2.249 vs CMP +0.556，Holm p=0.0273）、O-acetyl 阴性（q=0.930）、NXPE1 被分泌程序吸收、GSE178341 raw-UMI 独立复制（AGR2/SLC35A1 q=0.0068 通过）→ 最终模型 = **hybrid mucin glycome + donor–carrier–core–linkage decoupling**，明确反对 global hypersialylation。
- 定位冻结为 "algorithm-enabled, evidence-calibrated clinical discovery"，25 门完成度总账明确未通过项（独立患者级复制、tracing/perturbation/rescue）。
- B 系列（B20-B47）：静态目录、拓扑、聚合上下文、双上下文全部负结果（B44 −0.56pp / B45 +0.23 n.s. / B46 −0.77pp）；U0 发现参考谱多重性大混杂（22.24% Top-1 被 max 聚合翻转）；U1c 分母瓶颈（Rhea-safe 127 < 200）；B47-U3-v3 原子事件协议冻结 READY_NOT_RUN。
- ChemAware 规则内核/R-CWT/V2 多零/Phase-A 时代：R0 切向投影 +4.0039pp 证明"几何不是瓶颈、可观测性是"；**原生三元组突破**（stage-1 role-3 +1.8144pp，CI [+0.82,+2.87]；Phase-A role-2 +2.1266pp，CI [+1.28,+3.03]）；V2 多零残差 +3.94pp（dev）但 GNPS 迁移 +0.47pp CI 跨零。

## 时代四：5pp 算术裁决与 GNPS 基准（09-21 → 10-08）

**主线**：+5–10pp 目标被算术杀死；诚实增量定位确立；GNPS 模型盲基准冻结。

- **GLM_PROJECT_OVERVIEW_20260930 的三层账与 5pp 算术裁决**：role-2 面板 +5pp=99 净纠正而最好净数 42；151 残余错误 58% 无任何化学方向；化学源穷尽（890<1,000 事件门）。可辩护定位 = +2~3pp 诚实增量 + 体系贡献，放弃 5–10pp 头条。
- GNPS Gold/Silver 10-ppm 基准冻结（模型盲）：2,091,446 → 329,607 过滤 − 31,971 MSG/MoNA IK14；identity-disjoint 10,995q/175,171 对、formula-disjoint 5,261q/47,724 对。
- 15 方法梯队：WSE R@1 87.37/88.27 为最强单法；official 85.36/86.81；Noise +1.20/+1.25 vs official；MS2DeepScore 79.12/—；**两次撤回**（S1 margin 答案键泄露、router v2 面板重叠泄露）；oracle 互补性 +5.88/+6.31 真实但 6 种 truth-blind 融合全部 CI 跨零；**幸存部署物 = 弃权门控**（WSE/Noise 60% 覆盖 99.0-99.5% 精度）。
- metric split 发现：pairwise AUROC ≠ Top-1 检索；margin 是有效弃权信号（AUC 0.848）而绝对分数不是（0.507）。
- 实体找回终点：5% FDR coverage WSE 67.11% > Noise 58.36% > official 55.36%。
- LCNEC：跨平台 5ppm 候选仅 6/263（2.28%）→ 全局运输 headroom 失败；P2 v2 因错误 entropy 实现作废。
- 0929 硬化十点合同 + V16 覆盖门停止（890<1,000 保护停止未训练）。

## 时代五：定位宣言与遗传反事实（10-08 → 10-14）

**主线**：从"算法冠军+数据集"转向"通用无偏结构无关发现范式"；TT7 成为第一个遗传正控。

- **生物学应用定位宣言（10-10/10-11 纠偏→10-12 冻结）**：五原则（无偏发现、结构无关推断、化学知情分析、可泛化方法、可验证生物学发现）、三平等入口（单谱查询/无标签样本/有设计研究）、八条封存路线。
- **Metabolomic Contrast Atlas 主合同恢复（10-12）**：两层结构（Spectral Knowledge Atlas + Biological Contrast Atlas）、双向查询、study-level contrast signature 为索引对象；竞争边界（iDMET/reverse metabolomics/DreaMS Atlas/MEMO/PALS/PAIRUP-MS 各占何位）；三层可弃权检索（严格实体/共享化学模式/研究对比关联）。
- 统一候选能量（WSE 锚定的连续凸残差融合）启动。
- **ST003325 正式负结果**：结构隐藏跨编辑机制 10/27（p=0.7126）——clone/plate 混杂同构教训。
- **MSV000096214/TT7**：作者表正控通过（真实 tt7 三元组在 20 种簇分配中唯一第一，12 pairwise AUROC 最差 1.00，3/3 上位塌缩）；**raw truth-blind 重建全链完成并通过**（正离子 4,562 特征 T=800 / 负离子 9,751 特征 T=400；T1 六检全过 p=0.0005 级、T2 精确 p=0.05、T3 全正；**1,988 个未解析 MS2 实体独立保留全部遗传拓扑**，包括 939 个 pooled-QC-only 敏感性全过）；T4 容量 FAIL 诚实记录。正确主张 = "unresolved-information sufficiency, not unknown-information increment"。
- HFD 跨研究首正控（10-14）：标准化有向 beta 14/15 Top-1（组成对照 0/15 全并列；置换 p=0.0002）——"研究效应可成为跨实验室可检索对象"的必要前提首次成立。
- MSV000088023 结构隐藏 HFD：207 个未解析特征 2/2 精确检索、动物级置换 p=0.0004、方向增量 CI 严格为正。
- 首个未知跨研究（097426→084112）：描述性 Top-1 正确但样本标签校准 p=0.30 FAIL；尿→唾液肥胖 WRONG_TARGET FAIL——"Top-1 不够，必须过样本标签零模型"的方法论教训。

## 时代六：Contrast Atlas 系统化与开放检索（10-14 → 10-29）

**主线**：从数据集配对走向多来源开放响应集合；两类全局算子被证伪；局部边证据成为唯一活口；统一算法线产出 Full23。

**系统化扩张**：Pan-ReDU 普查（939MB dump：629 任务→132 FBMN→95 有定量+MS2）→ 开放响应集合 v2→v10（10-22 达 12 来源 27 对比 751 生物单位 91,379 响应行；ChemProp 扩展后 15 来源 51 对比 238,412 行 54,861 实体）。
**严格准入文化**：MSV000089518 因生物单位门 FAIL_INDEPENDENT_BIOLOGICAL_UNIT_GATE 判退；IBD200 板次混杂（19 健康全在 Plate3）被 Plate3 重算否证；COVID 双源（085505+1866）资格审计 NO-GO（缺 subject crosswalk + 全 feature 矩阵）；运动/禁食 exact-axis 审计 NO-GO；肾脏 16 队列=13 独立项目族、6/6 干预无样本级随机化 → 归档。
**算子证伪链（最重要的负结果序列）**：
- 全局碎片组成基线：HFD 关系排名 12-14/17 → FAIL
- centered response kernel（组成中心化）：known 侧强但 unresolved 0/6 → 未过 unknown gate
- signed fragment/neutral-loss programs：表面 3/6 → **31/31 次响应重排零模型完全解释（FAIL_UNKNOWN_RESPONSE_SPECIFICITY）**
- 稀疏局部 Top-1 对齐：原始分看似改善 → 256 次响应谱重排校准后 0/5 → FAIL
- 协方差校正响应场（ridge）：HFD 正控把 sex contrast 排进 Top-2 → 冻结负结果
- ChemProp 实体级：12/12 同药至少一条 dual-FDR 边、9/12 最小 p；但 HFD/IBD 跨项目应力 0 过 → **信号不迁移**
**活口证据**：
- **来源内响应轴正控**：HFD 查询在同源四轴中精确恢复两个 HFD-vs-normal 为 Top-2（all+unresolved，跨 fragment/DreaMS 表示复现，31 次重排 3/31）——排除"只认来源"
- **酮食第二干预源准入**（MTBLS1066：20 个体 168 mzML，H_MCT−H_C 4v4 / BC_MCT−BC_C 3v4）
- **学习表示首次改变真实开放检索**（10-27 固定边实验）：official/Stage-1 重加权 41,082 条冻结边，exact-drug 12 单元配对中 10/1/1 胜 coverage-matched fragment（sign p=0.0059）；IBD 双向 Top-1（max-T p=0.0039）——"学习局部关系加权"成立但非 ChemAware 特有效应、非身份召回
- **cilnidipine = 唯一 unknown-required 候选**（10-29）：known 基线排 14/15，all/unresolved 排第一（p=0.0039）——但 fragment/DreaMS/Stage-1 也排第一 → 支持"未解析信号有响应信息"，不支持算法增量
- **m/z 595.348 IBD 实体解析**：两对特征（1041↔16870 / 1286↔28796）非别名——实体消歧范例
- MSV000095331 药物-微生物：修正 estimand 后 unresolved 完整世界 2/3 Top-1 但数量匹配后 0/3 → 未解析优势主要是覆盖效应
**算法线（10-19→10-27）**：统一 B（GNPS nested OOF 88.59/89.85 但 Enveda 反向迁移 −2.87pp）→ 23 特征 Additive（Enveda-to-GNPS +0.88/+1.26pp，跨源保持）→ **Full23 v1 冻结发布**（四面板全胜 official +24.49/+18.88/+2.89/+2.72pp、胜 WSE +4.56/+1.03/+0.88/+1.26pp，四公式簇 CI 下界全正；严格消融：V2 continuous-null 是最稳定独立增量族，自研包不能被 Official+WSE 学习组合替代 +2.76/+0.83/+0.81/+1.08pp）；SOTA 裁决分层：**非 SOTA**（开发集内部有效、跨域未确认、无未开封外测）；Full23 v1.1 multi-reference readout 为唯一授权的下一表示级实验。

---

## 全项目存活资产总账（截至 2026-10-29）

| 资产 | 数字 | 资格 |
|---|---|---|
| Full23 Additive v1 | Enveda id 77.59 / GNPS id 88.25 R@1；四面板 CI 下界全正 | consumed-development 冠军，可迁移 |
| 弃权门控范式 | WSE/Noise 60% 覆盖 99.0-99.5% 精度 | 部署级实证 |
| ChemAware stage-1 / Phase-A | role-3 +1.81 / role-2 +2.13（CI 严格正） | 已消费开发，无 GNPS 迁移 |
| P2b | 封存 P3-main +1.07（p=0.0101）| near-core −4.23 边界 |
| WSE | 15 方法天梯 R@1 冠军；实体找回 67.11%@5%FDR | 最强经典基线 |
| TT7 遗传正控 | raw 重建 T1-T3 全过；1,988 未解析实体保留拓扑 | consumed 正控 shard |
| HFD 跨研究正控 | 有向响应 14/15 vs 组成 0/15；p=0.0002 | consumed 首个跨实验室响应检索 |
| 来源内响应轴 | HFD Top-2 精确（跨两种表示） | consumed 正控 |
| IBD 实体链 | adult↔pediatric Top-1 max-T p=0.0039（学习表示） | consumed |
| MTBLS13729 Neu5Ac 主轴 | donor 解耦 Holm p=0.0273 + raw-UMI 复制 q=0.0068 | mechanism-supporting |
| 开放响应集合 | 15 源/51 对比/238,412 行/54,861 实体 | development 底座 |

## 已死路线总账（不再救）

全局碎片组成基线、centered kernel（作为 unknown gate）、signed fragment+loss programs、稀疏局部 Top-1、协方差校正响应场、ChemProp 跨项目迁移、肾脏 counter-response 专用估计量、运动/禁食 exact-axis、COVID 双源现资格、遗传反事实作为方法名、公共证据证书作为核心创新、kNN 连通分量作为化学家族、BioAware 静态目录/拓扑/聚合（B44-B46）、RRF、S1/router v2（泄露撤回）、RAW-v1、五种无监督 truth-blind 融合。

## 当前一句话现状

**方法骨架（研究级有向响应 + 双层身份/关系 + 可弃权检索 + 冻结谱学证据提供者）在 consumed 数据上已被反复证明可运行、可否证、可诚实弃权；两个已知正结果（HFD 家族检索、IBD 实体链）都还限定在同项目或已知物可答的范围；"未解析化学信息在独立来源留出下带来正确检索增量"这一中心命题仍是 0 个正例——唯一候选 cilnidipine 支持 unknown 有信息但不支持算法增量。**

## 决定性缺口与下一步（按 10-29 冻结程序）

1. **身份轨 4 世界 × 4 方法矩阵**（known-only / matched-known / shuffled-MS2 / all × WSE / official / Noise / Full23）在已冻结完整候选块上——下一个唯一授权的决定性计算
2. 关系轨单独评价（fragment/WSE/Official/Noise + 校准关系打分器；Full23 排除）
3. 只有"过最强对照 + 改变研究级答案 + 新独立证据确认"三条件齐备的案例才能成为旗舰生物学结果
4. 服务器侧欠账：MoNA-neg MGF + LCNEC EIC 矩阵（多来源证书与样本级投影前提）

## 阅读覆盖证明

- 亲读（本会话逐字）：10-08→10-29 约 65 份核心文档 + 早期奠基 8 份（含 590 行 second_report 全文、217 行 9-30 总览全文）
- 读史官逐字读毕并落盘：批次 2（52）、批次 3（52）、批次 4（51/52）、批次 7（48/52）+ 27 个分块（进行中，每块 17 文件逐文件追加摘要）
- 全部摘要文件：tasks/_history_review/*_digest.md（本文件与摘要互为证据）
