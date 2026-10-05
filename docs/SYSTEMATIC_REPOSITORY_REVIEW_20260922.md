# DreaMS 仓库系统性综合审查报告

日期：2026-09-22
性质：只读审查报告（未修改任何科研工件）
范围：docs/（约 250 篇）、tasks/（1700 个脚本，约 35.9 万行）、data/validation/（1071 个实验目录 / 5293 个文件）、deliverables/、.codex_* 等 GPT/Codex 工作产物、根目录日志与工件。
方法：核心索引文档（MASTER_SUMMARY、CLAIM_REGISTRY、AGENT_EXECUTION_REGISTRY、INTEGRATED_EVIDENCE_ADJUDICATION_ROUTE、VALIDATION_CANDIDATE_BOARD）全文精读 + 三路分块深审（早期阶段 / NOISE 主线 / CHEMAWARE+REVERSE）+ 直接抽查（data/validation 产出质量、codex 产物、tasks 结构）。

---

## 1. 项目全景：这个仓库实际是什么

一个以 DreaMS（预训练 MS/MS 谱图 embedding）为底座的科研项目，围绕一个总命题展开（见 `PROJECT_INTEGRATED_EVIDENCE_ADJUDICATION_ROUTE_20260919.md`）：

> 在预训练全局表示之后，识别仍未被全局表示解释的证据，并仅在这些证据对候选判别提供可复现增量时改变排序。

四条算法/应用主线 + 一个平台层：

| 主线 | 科学角色 | 当前最强真实结果 | 当前状态 |
|---|---|---|---|
| **Noise（噪声微调）** | 改共享权重 | 旧 cohort E4-A +0.635 pp（历史资格）；因果归因显示其中 ~94% 来自 clean continuation | corrected graph 上 V3 修复后未运行；已重置为 native triplet |
| **ChemAware（化学先验）** | 候选条件策略 / 课程 | native triplet role-3 图 +1.8144 pp（CI 严格为正），迄今最强 learned shared-embedding 结果 | 化学归因与 fold-4 outer 均未闭环 |
| **BioAware（生化网络）** | sample-aware 上下文 | 无外部正增益；B44 外部反转、B45 null、B46 负结果 | B47 truth-blind 分母已冻结，U1 STOPPED、U2 完成、U3 待做 |
| **生物学应用** | 真实发现 | MTBLS13729 修饰鸟苷模块 10/10 同向 +2.95 log2、C20:4 锚点（Level 2）；LCNEC quinolinate/ascorbate Level-2 | 验证候选板就绪，全部 RESOURCE_AUDIT_REQUIRED |
| **平台/解释层** | 基础设施 | 3,486 规则库、双重映射（macro-AUPRC 0.659/0.240）、FDR/Schymanski 注释平台、P2b 封存 +1.07 pp | P2b 冻结为强基线不再扩线 |

## 2. 时间线主干（四阶段）

1. **08-06–08-23 早期**：规则去标签化 → 因子挖掘全线阴性 → 反事实峰监督（28.1% vs 7.6% 因果证据成立）→ RAW-v1（+4.35 开发 → 盲测不显著）→ P2b（封存 P3-main +1.07 pp，near-core −4.23 pp）。
2. **08-24–09-06 重构**：NOISE S 系列动作矩阵 → A4/C1 教师头寸 → E4-A 多折 → 09-01 因果归因归零 → 09-06 corrected graph 重大重置（旧 23,876 cohort 因 `SIMULATION_CHALLENGE` 语义误读整体撤销，新图 83,619 queries / R@1 0.92876）。同期 ChemAware 经历 unified_v2 数据事故勘误（误弃 119,029 谱）与 Phase A 八臂同分判负。
3. **09-07–09-19 收敛与清算**：best-action V5–V11 与 Hybrid 注入全部失败（精确 0.25 注入下仍输 shuffled）→ 09-14 版本失败总账 → 09-16 注入器缺陷修复定性"实现错误而非科学否定"；ChemAware 09-12 范式批判（补上可表示性/可观测性环节）、outer preflight 判 NO-GO；BioAware B44–B46 外部反转定案，B47 转 truth-blind；09-19 建立 CLAIM_REGISTRY / AGENT_REGISTRY / 证据裁决总路线三件套。
4. **09-20–09-22 当前**：两条 native triplet 线（NOISE 合同冻结未跑；ChemAware role-3 已出 +1.8144 pp 正结果并已 seal 高覆盖工件），residual native triplet 官方缓存 20260922 已落盘（含完整 gates 审计）。

## 3. 各主线审查要点

### 3.1 早期阶段（深审报告要点）
- 强证据：错误集中于 MCES 0–2 局部混淆；共享主峰误聚/条件特异峰分离经干预验证；P2b 是期末唯一通过封存盲测的模块。
- 弱证据：所有因子/可解码方向（0/32 过全门）。
- 问题：残差图谱 v1 次日被 v2 推翻；top_overlap 大胜为 tie-breaking 假象；66% 训练 anchor 为模拟谱；G8R M1 复盘记录了 LLM 生成结论被人工驳斥的案例。
- 未闭环：Causal ChemMask GPU 正式训练、KPGT P1/P2、规则库结构化改造、P2b near-core 无标签安全门。

### 3.2 NOISE 主线（深审报告要点）
- 九阶段演进；约 8 条路线被明确否定；三次真正转折：瓶颈定性（oracle≠可学分布）、因果归因归零、图重置。
- 当前 native triplet 仅为冻结合同，docs 内无结果文档，主线"合同已冻结、裁决悬置"。
- 批判性发现：headroom（3.35–5.33 pp）与实际增益（+0.635 pp）长期脱节，+4 pp gate 接近数学不可达；图重置后旧数字仍被后续文档引作动机；"实现缺陷"解释有不可证伪倾向；工程事故（OOM/hash/JSON 类）占比过高；P-arm 信息论缺口自 08-30 提出后从未正面解决。

### 3.3 ChemAware + Reverse（深审报告要点）
- ChemAware 五阶段：重置 → 教师路线（Phase A 八臂同分，可归因化学增益=0）→ 证据分级 → 范式批判 → **native triplet 突破**（化学只选 hard-negative 课程 + DreaMS 原生 triplet loss，训练目标与部署检索首次同构；best 在 step 3,000 早停，last 退化）。
- Reverse metabolomics：同日三轮自我纠偏后定位于"技术可检测性校准的化学—生物语境联合模型"；A2 未超技术基线、A1b 在官方 embedding 空间确认失败停线，方向目前仅存负结果 + A0 酰基肉碱局部信号。
- 批判性发现：A1 基线偏倚；native triplet 缺 null-triplet 同预算对照（化学归因 PENDING）；role-3 面板被多轮开发重复消费，"开发持续进步"与"独立泛化"证据链从未合拢；outer seal 后无恢复状态机（最坏分支=truth 已消费而结果丢失）且未见修复完成记录。

### 3.4 BioAware + 生物学（经核心索引覆盖）
- B47 现状（AGENT_REGISTRY 权威）：B1 provenance 完成（18 工件 / 638 MB 哈希闭环，**但 artifact registry 目录本地缺失，仅服务器侧**）；U0 显示 max 聚合造成 22.24% Top-1 身份翻转（reference multiplicity 重大混杂）；U1 +0.0682 pp 但 131/74、risk-net=−17 → STOPPED_UNSAFE；U1b v1 的 45/0 系 argmax 并列泄漏已撤销；U1c 漏斗 4300→127，瓶颈在 Rhea 覆盖/度数而非谱学；U2 完成（Rhea-safe 127 身份 / 3,243 events）；U3 truth-blind 事件产量审计 READY。
- 生物学：MTBLS13729 主现象冻结为"mucinous-relative free-Neu5Ac pool expansion with pool-to-donor decoupling"；18-candidate ledger 已诚实拆分，无一是新标准确认；LCNEC 为独立 Level-2 资产；验证候选板首批 f703 Neu5Ac / f1597+f3019 鸟苷异构体 / LCNEC quinolinate，全部待资源审计。

### 3.5 tasks/ 与代码资产（直接盘点）
- tasks/ 1700 个 .py、约 35.9 万行（test 331 / validate 181 / build 177 / train 55 / run 25 / evaluate 35 / 其他 896）；tests/ 121 个；dreams/ 为上游 DreaMS 的 fork（algorithms/models/training/utils + api/cli）。
- "其他 896"未按命名约定分层，是最大的可检索性债务；脚本化科学（每个实验一个可重放脚本 + report.json）模式本身健全。
- 抽查 report.json 质量：结构化程度高（status/formal/data_contract/metrics/gates/contracts/claim_limit 字段齐全，fail-closed 布尔合同如 `B47_truth_opened:false`），U1 报告甚至自带 claim_limit 免责声明——可审计性属上乘。

### 3.6 data/validation 产出（直接抽查）
- 1071 个目录中 69 个完全空、5 个无 json/csv 产出（约 7% 空壳，多为中断 run）；其余产出结构统一。
- CLAIM_REGISTRY 抽查 4 条 artifact 路径：P2B-P3、peak_token_reranker、noise_corrected_official_full_metrics、B47-U1 均存在且内容匹配；**B47-B1-001 的 `bioaware_b47_b1_artifact_registry_20260921_v1/report.json` 本地不存在**（与 AGENT_REGISTRY 记录的"服务器侧工件未同步"一致，但 CLAIM_REGISTRY 未标注此缺口，属文档-工件不一致）。
- 最新 chemaware_residual_native_triplets_w030_official_cache_local_20260922 含三池 npz + 三重 content-permuted null + 完整 gates（outer_role_4_untouched=true 等），审计设计成熟。

### 3.7 GPT/Codex 工作（直接审查）
两条"GPT 的工作"脉络：
1. **科研过程本身**（docs/ 主体）：由 LLM agent 深度参与生成。质量极高的一面：预注册门、SHA256 冻结、E0–E4 证据分级、允许/禁止措辞清单、勘误不删旧文。风险一面：早期首轮结论存活率低（≥6 项"显著结果"一周内被证伪为工程假象）；存在 LLM 生成结论被人工驳斥的记录；in-flight 状态声明常无跟进闭环；命名替代链复杂。
2. **.codex_* 工程产物**：`.codex_tmp/report2*` 是一套完整的 PPT 生成流水线（源 44 页清华模板 → 布局 JSON → 38 页重建 PPT），带 deviation-log 自我审计（正文统一黑体 20pt、删除彩色卡片、CCS/RT 模块退出说明），最终产物在 deliverables/（`LC-MS代谢物注释课题组会汇报_20260906/0911_V7/V8.pptx`）。质量结论：流程严谨（模板审计→布局映射→渲染→montage 核对），但**全部停留在临时目录未归档**。`.codex_spreadsheet_audit` 仅剩 node_modules（空壳）；`.codex_tmp_a_prof` 为 DreaMS 论文 PDF/TXT（11.7 MB）；`.claude/` 仅 settings.local.json。另发现 27 个 `.chart-data-*` 临时目录与大量根目录散落日志/JSON，均属清理对象。

## 4. 跨主线系统性问题（按严重度）

1. **归因缺口是全局模式**：ChemAware native triplet（+1.81 pp，缺 null-triplet 对照）、Noise E4-A（+0.635 pp，其中 94% 来自 continuation）、U1（+0.068 pp，risk-net 为负）——三条线的"增量归因"均未闭合，而这是论文能否主张方法创新的命门。
2. **评价集重复消费 / 确认面板枯竭**：role-3 图、corrected development graph、P3 均被多轮开发消耗；fold-4 outer 是唯一未触碰面板且因恢复合同 NO-GO 无法开启。缺乏新确认面板使"开发持续进步"无法转化为"独立泛化"主张。
3. **文档-工件一致性有漏洞**：B47-B1 registry 本地缺失但 CLAIM_REGISTRY 未标注；大量 in-flight 状态无闭环。
4. **headroom-增益脱节与不可证伪的"实现缺陷"解释**：目标设定（+4 pp）与证据（0.6 pp）长期不符；失败后以工程事故解释续命，虽有 09-16 的真实修复佐证，但模式本身有风险。
5. **仓库卫生**：约 30 个临时目录（.codex_*、.chart-data-*、tmp/、scratch/）、根目录上百个 .out/.log 散落、69 个空 validation 目录、tasks/ 896 个脚本无命名分层。
6. **服务器-本地双轨**：多个正式工件（B47 embeddings/seeds、noise ledger）仅存服务器，本地哈希闭环不完整，违背仓库自身的 fail-closed 原则。

## 5. 当前最可信的资产清单（可进论文）

- P2b sealed P3-main +1.07 pp / near-core −4.23 pp（E3，正负同表）。
- ChemAware native triplet role-3 +1.8144 pp（E2 开发，归因待补）。
- corrected failure atlas 分母（83,619 queries / 5,957 errors，E2）。
- 峰级因果证据（定向删峰 28.1% vs 随机 7.6%，discovery/confirmation 隔离）。
- 化学双重映射（macro-AUPRC 0.659/0.240、175 桥、2 因子输入忠实性）。
- MTBLS13729 生物学包（表型盲冻结 + 配对定量 + 多归一化敏感性）。
- 全套负结果资产（E4-A 归因、B44–B46、A1b、V5–V11 失败总账）。

## 6. 建议优先级

1. **补 native triplet 的 null-triplet 同预算对照**（ChemAware 归因闭环）+ 恢复 outer seal 状态机后开 fold-4 —— 这是决定论文形态的两道硬门。
2. 同步 B47 服务器工件到本地并补全 CLAIM_REGISTRY 标注；完成 U3 事件产量审计。
3. 追溯或永久删除 `PT-UNV-001`（+0.842/+1.001 pp 无来源数字）。
4. 仓库清理：归档 .codex_tmp PPT 流水线与 deliverables 对应关系；删除 .chart-data-*、空 validation 目录；tasks/ 补命名分层索引。
5. 验证候选板（f703/f1597/f3019/quinolinate）进入实际资源审计，把生物学资产推向 Level 1。

---

*审查方法注记：本报告由核心索引文档精读、三路分块深审（早期/NOISE/CHEMAWARE+REVERSE 各自独立成文于审查过程）与直接抽查（data/validation、.codex_*、tasks 结构）汇总而成；BioAware/生物学部分以仓库自身权威索引（0919–0921）为准。所有审查为只读，未修改任何科研工件。*

---

## 附录（V2 增补）：初版综述遗漏的三条主轴

初版综述偏重算法裁决账本，遗漏了三条定义课题面貌的主轴，现补齐：

### A. 课题的对外科学问题不是"提升检索"，而是"开放世界结构反演 + 拒答"

`A教授交流_AI4SCIENCE代谢物注释PPT逐页讲稿_V3_20260902.md`（768 行终审版）给出全场唯一中心句：

> 给定一张未知小分子的 MS/MS 谱图及其采集条件，能否输出证据可追溯、置信度可校准的分层结构结论；当正确结构不在候选库中或证据不足时，能否可靠地拒答？

这一定位包含：一期任务卡（冻结输入/候选/输出/真值/拒答）、五类数据角色（无标注语料/参考谱库/候选普库/冻结基准/真实队列不可互相替代）、三层验收阶梯（封闭计算真值 → 盲标准品跨条件 → 真实队列送验命中率）、开放世界压力测试（移除真值后测高置信误接纳率）、以及与 A 教授方向的"规模化 Agent 并行提出/执行/反驳假设 + 统一裁判"合作接口。仓库内部那套 E0–E4 证据分级、claim registry、预注册门，正是这一对外验收协议的内化。算法线的一切 pp 数字都从属于这个更大的"可信注释"命题。

### B. 生物学应用是一个跨组学三角验证工程，不是单队列差异分析

`PROJECT_RESEARCH_MASTER_SUMMARY` 13.7–13.9 节与 MTBLS13729 系列显示的完整版图：本地 Rmu 队列表型盲冻结注释 + 患者内配对定量（修饰鸟苷模块 10/10 同向 +2.95 log2、随机模块对照 p=0.0007）只是第一层；其上叠加了独立蛋白组（OEP00006137 原始 180 个 mzXML 重提取、SAH +2.4 log2 复现）、单细胞（GSE236696 六对黏液型、嘌呤轴 6/6 上升 p=0.031/随机轴审计）、空间（GSE281697 多胺-酸性-趋化空间相关否定）、风险转录组（GSE281917/TCGA MuC23）、原论文 345 条注释的增量审计（八候选与原表 m/z+RT 匹配均为 0；1597/3019 修饰鸟苷家族是当前最强注释新增；1717/3222 均有原作者先例、只有类别级增量）。最终收敛为"修饰鸟苷/嘌呤共变主轴 + 乙酰化多胺平行轴 + 长链酰基肉碱平行轴"三轴并列、机制就绪度评分为 evidence-calibrated clinical discovery 层（明确未达 isotope-tracing 因果层）。

### C. 两条算法主线已在同一范式上会师（一手文档确认）

- ChemAware（`NATIVE_TRIPLET_SUCCESS_RETROSPECTIVE_20260921` 全文）：7,688 个 query-negative 分子对 / 4,032 anchor / 49,980 positive edges；化学只从三个来源挑选 hard negative（official 最难错误 + 规则中心 action-hard + 胜过三个 content-permuted null 的 specific-hard），训练完全复用 DreaMS 原生 ContrastiveSpectraDataset/hinge loss/Adam。best 在 step 3,000（约 1.56 epoch，与 official embedding 平均 cosine 0.72）；训到 412k 步退化到 CI 跨 0（cosine 跌至 0.31）——过度训练会持续扭曲 DreaMS 全局几何。
- NOISE（`NATIVE_TRIPLET_CONTRACT_20260920` 全文）：同一原生运行时，七源 32,114 行 strict Top-1 纠错动作经别名折叠后生成 clean-boundary + action-boundary 双 triplet，62,312 宽 clean 事件 + 动作事件 = 113,796 事件 / 28,449 步单 epoch（较照抄 301 epoch 减少 183 倍）；含对全零动作谱预处理器除零的逐 token 恢复审计。合同已冻结、双臂（targeted vs matched-shuffled）未跑。

两线共同的思想跃迁：**知识（化学规则/纠错动作）不再进入损失函数或注入谱图，只负责设计课程（选 hard negative）；训练目标与部署检索严格同构。** 这是对 0912 四层批判（判别力/分数修正/可表示性/可观测性）的正面回答。

---

## 附录（V3 增补，2026-09-29）：最新增量一手审查——BioAware U 系列代码 + 09-26~29 主线

### A. BioAware U 系列代码（一手通读）

**代码链**：`bioaware_b47_u3_core.py`（879 行，全文）+ `audit_bioaware_b47_u3_event_yield.py`（640 行，零假设接线与闸门段全文）+ `annotation/bioaware.py` 的 `degree_preserving_reaction_decoy`（422-469 行）+ U1/U1b/U1c/U2 清单。本地运行 `test_bioaware_b47_u3_event_yield.py`：**13 passed**。

设计裁决（代码层面确认）：
1. **真值盲由运行时强制**：audit 主流程逐表扫描 `truth/phenotype/disease/case_control` 禁用列名，命中即 `RuntimeError`；U3 分母必须逐级复现 U1c/U2 冻结数字，不复现即停。
2. **催化剂/载体剔除**：净化学计量变化为零的参与物不构成分子转化端点；currency 代谢物独立剔除。
3. **双重 10 ppm 身份契约**：query 侧与参考侧的中性质量各自独立对分子式过 10 ppm，然后才检验转换残差——"仅匹配质量差"被显式禁止（两个身份错误可能相消）。
4. **转换签名冻结（v3 核心修复）**：`degree_preserving_reaction_decoy` 只交换 `compound_id`、**不动 `formula` 列**——重连后每条反应边的期望非氢元素差签名保持原值，身份接线随机化而化学期望冻结。这正确实现了 U3 修正案"重连的边不得从重新分配的身份重定义期望转换"（v2 正是栽在这里并被作废退役）。
5. **目录记录 ≠ 独立观测**：同一 (query, seed) 的多条 Rhea 记录在 noisy-OR 聚合前塌缩；candidate specificity 按"该 seed 支持的全部竞争候选"计，与反应记录数解耦。
6. **三族零假设各取所需**：度保持重连（结构零假设）/ 同层置换完整 seed 载荷（样本上下文零假设，含 64 次重试找最优移位 + derangement 保证）/ 同 query 同 adduct 错误身份轮转（非方向性诊断——wrong-identity 数量大不作方向判定）。fail-fast：任一结构零假设 ≥ 真实值即写入 NO-GO 并停止消耗算力。
7. **算术 headroom 门**：51,976 查询上 +3 pp 需要 ≥ ceil(0.03×51,976)=1,560 个可改变决策，U3 必须先过此门才允许一次性真值评估。
8. **U4（自 pyc 反汇编）**：在打开 B47 真值前冻结"U3-v3 独占事件赢家"主动作——仅当事件赢家与唯一 DreaMS 赢家不同或解开平局时提升，其余候选序原样保留；浓度分层/多样性闸门不过则降级标注 prospective-mechanistic。

### B. 确认的问题清单（BioAware 最新代码与工件）

1. **U4 源码丢失**：`freeze/validate/test_bioaware_b47_u4_primary_action.py` 在全仓库仅存 `__pycache__/*.pyc`（2026-09-22 21:59 编译），`.py` 不存在、从未进入 git（u4 文件无任何提交记录）、无上传清单、无文档、无输出目录。U 系列最新一步的完整溯源链断裂。
2. **U2/U3/U4 结果不在本地**：`data/validation` 中 BioAware 输出止于 09-21 的 U1/U1b dev 目录；U2 文档给出 canonical 路径 `data/validation/bioaware_b47_u2_catalog_coverage_20260921_v1/report.json` 但该目录本地不存在（服务器侧）。U3-v3 是否已在服务器执行、结果如何，本地零记录。
3. **登记册滞后于实际推进**：AGENT_EXECUTION_REGISTRY（最后更新 09-22 14:44）记 `U3-V3 READY_NOT_RUN`，但同晚 21:59 已发生 U4 编译活动；U3 无结果文档、U4 无冻结记录——fail-safe 状态机出现断档。
4. **NOISE Stage2–6 输出同样仅存服务器**：本地 `noise_*` 目录最新停在 09-07；Stage-4 文档引用的 run 2344820/2345708 等正式结果均不可本地审计。
5. **主账本停更**：MASTER_SUMMARY 最后修改 09-01，整个 09-06~09-29 的 U 系列、NOISE Stage1–6、ChemAware V6–V15/PhaseA/MassBank、GNPS 基准均未入账。
6. **版本控制近乎缺失**：1143 个未跟踪文件、约 20 个已跟踪文件带未提交修改——近一个月的全部工作（含 U4 丢失源码）不在 git 内。
7. **已文档化的协议错误**（仓库自身账本，供统计）：U3-v1 未正确实现关键 null、v2 两项逻辑错误作废；Stage-4 v3/v4 构造错误 + run 2345708 判废；Stage-1/2/3 摘要曾只保留 AUC 布尔方向检查（09-27 修复为全数值 `auc_metric_comparisons`）。

### C. 09-26~29 主线一手补读后的叙事修正

1. **5 pp 与"1–2 pp"的语言学纠正**（`NOISE_5PP_AND_AUC_CORRECTED_LEDGER_20260927`）：Stage-1 targeted−control = **+1.26548 pp**（因果对比，CI [+0.876,+1.654]）；targeted−official = +0.49637 pp；Stage-2 绝对最优约 +0.60 pp；ChemAware +1.81441 pp 是独立实验。绝对规模 ~+0.5–0.6 pp，距 5 pp（≈917 个净决策）还差 ~4.40 pp。五种比较器禁止合并成一个 headline。
2. **ChemAware PhaseA = +2.1266 pp**（role-2 1,975 查询，CI [+1.276,+3.030]，自 official 独立初始化）：赢在 introduced 24→12 而非纠正更多；pair-expanded +2.2785 pp 因安全门被拒。PhaseA 审计给出后续扩容失败的五条根因（监督单位错位 / pairwise-listwise 错位 / 最硬负例高方差 / event≠query 剂量 / fresh-Adam 断裂）与四臂决定性矩阵——**C 臂即化学归因 matched-null 对照，本综述初版指出的全局归因缺口已进入正式实验设计**。
3. **GNPS Gold/Silver 10 ppm 基准**（v1，09-28）：2,091,446 谱 → 329,607 过滤谱，identity-disjoint 10,995 查询/175,171 对 + formula-disjoint 5,261 查询/47,724 对，排除 MassSpecGym/MoNA 身份，独立复核器逐 query 重验——这是"确认面板枯竭"问题的正面回应，且明确禁止称 NIST20 复现。
4. **MassBank 分层路线**（09-28）：MassBank 2026.03（MD5 封存）独立化学证据，16,272 条候选规则经双匹配容量对照 + 冻结折 3–4 一次性确认（八组 CI 下界全部严格为正）；动态参考谱转换解决"60 事件瓶颈"（1,658 事件/4,066 活跃 triplet）；12k 语料 = PhaseA 前缀 5,956 + 化学 1,658 + replay 4,386，待服务器单次训练 + role-2 门。
5. **V15 多源裁决协议**：来源只通过支持/冲突集合组合、禁止数值融合；ICEBERG 冻结审计（700 查询救回 348/619 official errors，胜过 swapped/permuted 对照）为第一来源；fragment_graph 新源在 dev100 v2 上 **specificity 门未过（UNQUALIFIED）**，诚实未授权挖掘。
6. **NOISE Stage4/5/6**：Stage-4 自我纠错三次后收敛为 query-calibrated residual v5（Adam 1e-6、每 query 一残差动作 + 一 clean 保持）；Stage-5 关闭为负结果（474 查询、未胜对照、AUROC −6.624 pp）；Stage-6（最新）把 Stage-1 关系扩展为完整噪声三元组（锚/正独立扰动、难负例需可混淆性提升 ≥0.01、全部视图经未修改 SpectrumPrepressor 精确往返），注册最低 20,000 新广域双难暴露，冠军仍是 Stage-1。
7. **BioAware 精神的一手表述**：不是"用生物学知识提升检索"，而是——在打开真值之前，把"同一样本内一个可信代谢物事件能否为某候选提供可区分于其全部竞争者的类型化生化转化证据"做成可审计的冻结构造；证据不足以改变排序时宁可不动作（U1b v2 的 0/0 是数学边界而非失败）；无效结果退役命名空间但保留错误记录。这是 0912 第四层（可观测性）批评在生物学侧的正面回答，与算法侧"知识只设计课程"完全同构。

### D. V3 结论

课题当前的统一战场是：**把"知识增量"从"更多梯度"重新定义为"更多独立的候选决策覆盖"**（PhaseA 审计 §6、V13 教训、U3 headroom 门、Stage-6 注册口径全部指向此处）。方法学风险最高的三件事依次是：U4 溯源链断裂（必须先找回源码或宣布作废重写）、服务器-only 结果的本地审计闭环、以及主账本/登记册与实际推进的脱节。
