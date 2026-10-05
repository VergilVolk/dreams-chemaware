# DreaMS 扩展项目全课题统筹总览（GLM 工作底稿）

**日期：** 2026-09-30
**作者：** GLM 会话审计（统筹任务）
**性质：** 工作底稿，非论文结果稿。所有数字以引用的冻结文档/机器工件为准；与旧文档冲突时，以更晚的专项纠错、冻结账本和机器工件为准。
**覆盖声明：** 本总览覆盖 2026-08-01 至 2026-09-30 的 docs/（368 篇，按线全读或经总账+注册表+定向首读覆盖）、tasks/（2,339 文件的类别纪律审计）、data/validation/（1,185 目录的注册表扫描）。9/25 项目总账（`PROJECT_FULL_EVIDENCE_INVENTORY_20260925.md`）之后的增量由本会话第一手补齐（union 预检、硬化合同、探索链、GLM V16）。

---

## 0. 总体裁决

项目不是一个"给 DreaMS 加化学规则"的单点尝试，而是围绕可信 LC–MS/MS 代谢物注释的完整研究体系：严格可复算的候选检索与评价协议、系统性错误图谱、两条共享 embedding 改善路径（微调 / 后融合）、一条收敛到真值盲原子事件的生物上下文路线（负结果链 + 待裁决协议）、真实非靶向数据注释平台、以及 MTBLS13729/LCNEC 应用闭环。

### 0.1 三层结果账（更新至 0930）

| 层级 | 当前最强结果 | 证据资格 | 必须同时说明的边界 |
|---|---|---:|---|
| 共享 embedding 微调（化学课程） | **ChemAware Phase-A：role-2 `+2.1266pp`**（0.89620→0.91747，54/12，CI `[+1.2761,+3.0303]`，run 2343962 step-2000） | E2 | role-3 非回归确认未消费；对已发布 stage-1 增量门未过（+0.6076pp，utility −13）；role-3 确认版另有 **stage-1 `+1.8144pp`**（run_2340721_resume_2340524 step-3000，57/22，CI `[+0.8155,+2.8703]`，role-3 开发已消费+验证加载器软泄漏注记） |
| 共享 embedding 微调（噪声课程） | **Noise Stage-1 绝对 `+0.49637pp` vs official**；因果对比 `+1.26548pp` vs 同 query 对照（CI `[+0.87586,+1.65425]`，18,333 held queries，run 2344820）；Stage-2 ~`+0.6000pp` 未晋级 | E2 | 5pp=约 917 个净决策，现约 110 个；~4.4pp 绝对缺口；oracle 上限 3.35–4.93pp 属动作空间容量而非部署性能 |
| embedding 后融合 | P2b 封存 P3-main `+1.07pp`（89/57，CI `[+0.24,+1.89]`）；V2 多零残差重排 role-3 dev `+3.9399pp`（93/17，CI `[+2.8191,+5.1921]`） | E3 / E2-dev | P2b near-core `−4.23pp`；V2 无新独立确认面板；RRF 融合 held 94.954% 但 run_2347032 GNPS 确认失败，**永久关闭** |
| 生物上下文 | **无可迁移正增益**（B44 `−0.56pp`、B45 `+0.23pp` n.s.、B46 `−0.77pp`；U1 `+0.0682pp` 风险不合格；U1b v2 结构性 no-op） | 冻结负结果链 | B47-U3-v3 原子事件协议 9/22 冻结，**READY_NOT_RUN** |
| 探索（禁止引用为正式） | fragment step-750 `+2.3797pp`、union step-1000 `92.0506%`（`+2.4304pp`） | 仅探索 | CI 跨 0；硬化合同前链路被取消资格；作为 V16 初始化与起点保留 |

### 0.2 5–10pp 目标的算术裁决（针对"ChemAware 微调出 5–10pp"的指令）

**role-2 面板（1,975 queries）上这个目标已被测量杀死：**
- +5pp = 99 个净纠正；Phase-A 现有 42 个净；剩余 151 个残余错误的 66% 需要被零引入地修复；
- 151 个错误的最终诊断分布（GLM 诊断 v2，2026-09-30）：`unanimous` 19（Phase-A 时代就该学到，部分已学——100 个事件碰撞）、`vetoed` 23（对称否决可回收，但只有 4 个成为胜者事件）、`contested` 18（争议臂，未训练）、`dominant_opposition_only` 32 + `no_dominant_signal` 56 = **88 个（58%）没有任何可用的化学方向**、`ambiguous_tie` 3；
- 全修 151 个错误的天花板 = `+8.25pp`；现实转化率（Phase-A 先例 42 训练面板胜者 → 42 个选择面板净纠正）下，任何硬化纪律内的化学线增量都在 **+0.3~1pp 量级**；
- 化学源已穷尽：fragment-graph + MassBank 合并 = 890 事件 < 1,000 覆盖门（GLM V16 测量）；静态规则 0 新事件；SIRIUS 面板已建未跑（V14）。

**+5pp 的真实来源只能是（按可信度排序）：** (1) 目标函数改造（2pp 天花板审计的 T1/T2 阶梯：多正例多负例 + 自适应配对加权）——已设计未运行；(2) 规模（Noise Stage-6 的 ≥20,000 双侧硬暴露，run_2346857 在跑）；(3) 组合性（Noise×ChemAware 堆叠从未测试——两线冠军都从 official 初始化）；(4) 结构生成器解码器（0827 平台路线，下一代项目）。

---

## 1. 冻结的评价与数据基础设施

### 1.1 骨干与官方基线

- 冻结 checkpoint：`official_embedding_slim.pt`，SHA256 `8928f908…`；架构包 `ssl_model_server.pt`（464 MB，**仅架构参数，权重永远来自 --official-checkpoint**——L215 合同）。
- corrected development graph：83,619 queries / 9,854 identities / 6,220 formulas / 6,220,661 边；official R@1 **0.928760**（5,957 错误）；near 28,188/4,036；macro AUROC 0.968687 / micro 0.956287 / pooled pairwise 0.847501。
- 封存 P3 面板：main 3,000（0.879333）/ isomer 1,989（0.794872）/ near-core 496（0.487903）/ near+mid 661 / exposed 851 / sim-to-real 609。
- 公式角色面板（GNPS 派生 manifest，7,939 units）：role 0/1 训练（4,032）、role-2 选择（1,975–1,978，official ≈ 0.89620）、role-3 确认（1,929）、role-4 outer（16,198，**从未打开**，post-seal recovery NO-GO）。
- **GNPS Gold/Silver 10ppm 基准**（9/28 封存，模型盲）：ALL_GNPS 2,091,446 → 329,607 过滤 − 31,971 MSG/MoNA IK14；identity-disjoint 10,995 queries/175,171 对；formula-disjoint 5,261/47,724；与 noise 线共享。official 在 GNPS 面板上的数值**尚未产出**（等待 Stage-6 / 化学线后续运行的统一 evaluator）。
- 评价语言：Recall@1/2/3/5/10/20、MRR、macro/micro AUROC/AUPRC、pooled 10ppm pairwise、margin、corrected/introduced、λ=2 risk-net、near 子集、公式簇配对 CI；AUC 为**共同主端点**（9/27 修正案）。DreaMS 论文 0.85 = NIST20 pairwise ROC-AUC，本项目对应值必须命名为 "MassSpecGym/GNPS 10ppm pooled pairwise AUROC"，禁止称"0.85 复现"。

### 1.2 数据资产

MassSpecGym HDF5 231,104 谱/28,929 IK14（train [M+H]+ 156,568；严格 10ppm 池 112,601 锚/5.1M 正/4.8M 负）；annotated01 3,263,214 谱/76,157 IK14（E1 禁用：36.5% MSG 泄漏）；GNPS ALL_GNPS.mgf 2,091,446；MoNA（neg 141MB+pos 304MB，DreaMS 论文切分 25,319/5,524+3,557/831）；统一库 265,011 pos + 29,564 neg（~207,787 IK14）；规则库 335 核心 + 3,151 MassBank（记录级观察，非规则）→ 分层扩展 8,132；处理队列 MS2 374,232 neg_rp + 419,676 pos_rp；GeMS-A10 预训练语料 23,517,534 谱。存储账（9/22）：validation .pt 39.85 GiB / .ckpt 6.94 GiB / .npy 20.09 GiB。

### 1.3 工程纪律

tasks/ 2,339 文件：audit 395 / test 402 / sbatch 416（全部 sbatch-only，登录节点禁跑程序，不显式指定 CPU/内存，`--gpus=N`）/ build 210 / train 62 / evaluate 40 / GLM 4。三层验证：合成 fixture 合同测试（导入生产核心）、fail-closed 溯源（嵌套 SHA256 树、篡改必败、冻结工件常量）、sbatch 文本本身被测试。负面结果保留证据账本、权重可删；封存一旦打开永不删除；claim registry（E0–E4，无正向 E4）本身不授权任何主张。

### 1.4 工件注册表（9/29 扫描）

1,185 目录：975（82.3%）含本地 marker（959 report.json）；complete 族 572 / fail 族 133（含 26 UNQUALIFIED）/ frozen-sealed 38 / smoke 46。活跃加速：8 月 354 → 9 月 831。注意：`GLM_full_rerank_rrf_probe_20260929_v2_strict` [COMPLETE] 与 `_LENIENT_TIE_GIFTED_DO_NOT_CITE` [DO_NOT_CITE 禁引] 在服务器存在——RRF 线本地任务文件已按 PI 指令删除，线已关闭。

---

## 2. Noise 线全史（0816–0930）

**"噪声"的最终语义：** 训练时对真实谱的峰干预（衰减/删除特定峰）或提供同身份跨条件真实参考谱，产生纠正视图，蒸馏进共享 encoder；推理只见干净谱。基底是错误图谱：23,876 queries / 1,805 official 错误 / 1,439 positive-deficit vs 385 negative-excess（旧 cohort；0906 修正图重置后旧数仅作动机引用）。

### 2.1 算法演化弧（选型与死因）

| 时代 | 选型 | 结局 |
|---|---|---|
| 0816–0821 掩码增广 → 合成 4 轴噪声 + InfoNCE 异构体负例 | G5–G7 均匀噪声微调 | **−3.1…−4.6pp AUC，红队杀死**；转向真实跨条件正例 |
| 0822 G8R 头部门（105 万参线性投影） | M1/M1b 双流 | 配对 margin ≈ −0.001、preservation 0.99465<0.995，**fail-closed**；衍生 RAW-v1 重排器线（见 §3） |
| 0824 v3 重置：候选条件化峰归因（梯度 a_k=I·dJ/dI vs 角色选择器） | S1a/S1b/S1c 单峰矩阵 → S2 序贯 → S3A 扩展矩阵 | role-confounder 干净（100% 单峰 99/18）；梯度 50% 广谱主力；oracle 663→799→920（+3.853pp 上限）；shared 角色与 100% 梯度删除净害；**4pp 需 956，差 36，全部门失败** |
| 0825 正证据臂 | A4-B0 原型教师（+159 超门）→ C1 支持排他教师 +2.47pp | 上限证据；**E4-A 直接微调成线**（见下） |
| 0826–0828 F/E/R 迁移时代 | E4-A 定向噪声微调（last-1/7 块+头，lr 2e-6/1e-5 clip1）+ P/N/S 联合 | **E4-A：+0.635pp/近集 +0.522pp，15/15 折 CI>0 —— 该线首个稳权重增益**（旧 cohort 资格）；E5–E9 引导/结果挖掘/在线重挖全部负（挖掘比固定差 −0.41…−0.46pp = 选择偏差实证）；oracle 3.35–4.93pp vs 实现 0.635pp = **~4pp 动作→权重迁移缺口**（命名瓶颈） |
| 0901–0919 混合注入器时代（E14→Hybrid） | E14 教师重建失败（单动作塌缩/风险当纠正损失/[:4] 校准）→ E15-M3 又败（held 0/1，−0.391pp，面板被消费）；E4-A 三臂归因门失败（targeted−matched-random +0.0338pp，CI 下界=0，STOPPED）；B0–B2 梯度手术无跨公式共识；L0/L1 可学习性通过（AUPRC 0.7065）但 30 单元动态条件加权 Phase A 被杀（−0.0169pp vs 全部对照，Holm p=1.0）；E4-PMT/E4-DEB 败（DEB margin CI 严格为负）；评审会撤回蒸馏路线并测定**迁移效率仅 ~13–17%**；0906 修正图重置（旧 23,876 图撤销）后 V4–V11 注入器时代：paired_margin_only 分离动作张量、one-best 丢弃 89% 严格行、AdamW 把动作信号衰减到 5–9%、V8 关闭参考梯度旁路（70–92% 能量在参考侧）、V10/V11 冻结 Injector V1（精确 0.25 优化器份额，工程金测通过）——但 held Top-1 仍不动：V7 +0.1309pp 且负于 shuffled；混合注入（job 2337358）vs E8 +0.0927pp、低于配对 shuffled −0.0273pp、risk-net −9 | **25+ 项失败模式入账本**（0914 版本失败账本），教训沉淀为 13 条现行规则；0915 状态=七源 32,114 动作库 + Injector V1 冻结 + E4 Faithful V1（字节精确复现）获批待跑 + Hybrid V2 修复已规格未提交；**无任何 ≥4pp 主张** |
| 0920–0922 原生三重奏重置 | 放弃一切注入器：noise 只改三重奏内容（anchor/positive/negative 角色），DreaMS 原生 loss/optimizer/preprocessor 不动；先 action-anchor 后 action-hard-negative（复用 ChemAware +1.8144pp 的训练运行时） | run_2344688 证伪 clean→action 桥（−0.0667）；0925 修复 = **run_2344820 Stage-1 冠军**（action 视图为硬正例 +0.1365，桥接禁入训练账本） |
| 0926–0928 Stage-2→5 残差阶梯 | S2 边际残差（每 query ≤1 动作，lr 1e-6）→ S3 多难度（10,146 事件/3,021 queries，lr 5e-6）→ S3 修复（校准关系选择）→ S4 宽正例（v3 2345708 被否：双倍旧剂量）→ S5 负残差 | S2 +0.6pp 未晋级；S3 **负**（vs Stage-1 −0.649pp；vs 对照 +0.2346 n.s.）——根因六条：剂量乘法、难度失衡、等压铰链、5e-6 晚期率、动作/干净分离、覆盖 4.63%；S5 关闭为仅负残差（474 queries、对照未胜、AUROC −6.624pp） |
| 0927–0930 方法论审计 + Stage-6 | 2pp 天花板审计（原生铰链五重低效：1正1负、等幅压力、谱级 vs molecule-max 端点错配、最难例噪声、视图质量）；T0–T3 因果阶梯设计（T1 多关系铰链 → T2 自适应配对加权 → T3 molecule-max 列表级，**未运行**）；Stage-6 双侧硬三重奏规模试验（≥20,000 新宽暴露/≥10,000 queries，靶=干净锚+噪声正+噪声负 vs 对照=干净锚，lr 5e-6，从 Stage-1 冠军精确续训） | **run_2346857 在跑（9/29 无标记）**——现役 5pp 赌注 |

### 2.2 现役数字账（5pp 修正账本 9/27）

| 证据 | ΔR@1 | 正确解释 |
|---|---:|---|
| Stage-1 targeted vs 注册配对对照 | **+1.26548pp** | 因果动作内容对比，18,333 held，455/223 |
| Stage-1 vs official | **+0.49637pp** | 绝对共享 encoder 增益 |
| Stage-2 vs official | ~+0.6000pp | 最强绝对点估计，未晋级 |
| 旧 E4-A（旧 cohort） | +0.6362pp | 机制证据，非修正图主张 |
| ChemAware | +1.8144pp | 另一线，禁止计入 Noise |

晋级合同（合取）：绝对 R@1 趋近/达到 +5pp vs official 与 mature-E8 双基线；R@1+MRR 公式簇 CI>0；全部 AUC/AUPRC 数值改进；margin/Top1−Top2 不回归；corrected−2×introduced>0（总集+near）；全部 Recall@k 方向达标；靶向胜同 query 同剂量对照。

---

## 3. 重排器与融合线（P2b / RAW-v1 / V2 多零 / RRF）

- **RAW-v1**（0822 冻结）：有界残差重排器，dev 0.8081→0.8516（+4.35pp）但 Test-A/B +0.45/+0.73pp CI 跨 0；P1 选择性 +0.0355 CI>0。6,612/10,000 训练锚是 SIMULATION 谱。
- **P2b 秩融合**（0823 冻结，E3）：`0.10·DreaMS+0.10·熵+0.80·NL-sqrt-cosine`；封存 P3-main **+1.07pp**（89/57，McNemar p=0.0101）；near-core **−4.23pp** 硬边界；7 项 SHA256 溯源。
- **V2 多零对称残差**（0919）：role-3 dev +3.9399pp（93/17）且胜全部 3 个内容置换零（+3.3178 CI>0）；V1 backoff +3.7325pp。**无新独立确认面板**；outer NO-GO。
- **RRF 全重排探针**（0929 服务器，已删本地任务）：held 94.954%；GNPS 确认失败（run_2347032 指标回退）；lenient-tie 变体 `DO_NOT_CITE`；**永久关闭**。
- 教训沉淀：候选侧证据（X 层）永远不能当共享权重（W 层）主张；near-core 是所有融合方法的共同墓地。

---

## 4. ChemAware 线全史（0807–0930）

**范式结晶（0920 后）：** 化学证据只做真值盲三重奏课程选择（query, true, false 关系准入），训练事件保持原生身份三重奏，DreaMS 原生 loss/optimizer 不动；化学分数永不进入损失或推理。

### 4.1 演化弧

| 时代 | 选型 | 结局 |
|---|---|---|
| 0807–0823 SOTA 审计 → P1 反事实微调 → P2b → 双映射解释 | 335 规则库降位为解释工具；MCES 局部排序 | "值得继续"；P2b 冻结（§3） |
| 0824–0902 数据合同时代 | 方向噪声 v1/v2（55,892 变体证伪）→ 大规模候选组重置 → 共享 embedding 重启（v2 符号峰残差适配器/rank-4 PEFT） | 适配器 +0.3629pp（pres 0.998）为局部正；ICEBERG 蒸馏失败；SIMULATION_CHALLENGE 勘误（119,029 谱语义错误，unified_v2 作废）；有效秩 1.56–1.60 撤回"五独立证据"；科学问题收窄合同（0902） |
| 0903–0908 教师时代 | ICEBERG 2.1 结构条件教师蒸馏 + 8 臂直接迁移 + A2 梯度筛选 | 8 臂全部=纯 continuation（+0.4666pp，化学增量=0）；A2 特异性失败（pair 塌缩 16/24）；**0905 GPU 全面封锁**（零可归因化学增益；≤10 GPU·h 预算制）；ICEBERG 残差机制审计关闭（10/272 vs clean 11/272） |
| 0912–0913 R-CWT 重置 | 四层天花板分解 H_chem→H_geom→H_obs→H_opt；R0 切向投影 +4.0039pp（85/3）证明**几何不是瓶颈，可观测性是主嫌**；跨视图 PEFT 5 臂合同 | 政策 v4 dev +2.6439pp、教师 role-2 +1.9478pp、oracle +4.7659pp —— GPU 结果最终 +0.0829pp CI 跨 0（事后关闭） |
| 0919–0921 重排器顶峰 + 原生突破 | V2 多零（§3）；**原生三重奏突破**：run_2340721_resume_2340524 step-3000 = role-3 **+1.8144pp**（7,688 事件/4,032 queries；best.ckpt 在 3,000 步，last.ckpt 41.2 万步退化至 +0.4147pp） | 封存 + 29 项 SHA256 账本 + 恢复救回（.filepart 3 ckpt，CRC 全验）；**软泄漏注记**：role-3 池曾作为 trainer 验证加载器 |
| 0922–0926 后继家族（全部关闭） | V2 残差课程 / 参考对齐 / 对扩展 / 特定重放 / 固定预算手术 / **max-boundary Phase-A** / 多条件 V5 | Phase-A（run 2343962 step-2000，5,956 事件=3,605 安全+777 错误+550 化学+1,024 重放，覆盖全部 427 official 错误 query）= **+2.1266pp 保护**；其余负：V5 差 1 净 query；V6 冷 Adam；V7/V8 低于 Phase-A（身份广播稀释 64.63%）；V9 覆盖停止（23/151 queries）；V11/V12 在线重挖家族关闭（自适应支持过拟合 151→73）；V13 受益边界替代关闭（4 ckpt 全 CI 跨 0） |
| 0927–0929 源多样化 + 硬化合同 | V14 SIRIUS（面板建好未跑）/ V15 多源（本地 stand-in +419 三重奏，无权重更新）/ MoNA 极性迁移协议（未提交）/ 分层规则 8,132（静态 0 新事件）/ MassBank 分层（1,658 化学事件，资格全 CI>0）/ fragment-graph 生产链 | 探索：seed run_2346306 → 续训 run_2346408 **step-750 = 92.00%（+2.3797pp，CI 跨 0）**；union 续训 step-1000 92.0506%；旧 union 链被取消资格（C 级准入、参考乘法、Adam 重置、仅种子确认）；**0929 硬化十点合同**（_singleton 事件、成对支配、≥1,000 事件/≥500 queries、精确缓存哈希、单原生 Adam 恢复+计数证明、role-3 非回归、固定步检查点） |
| 0929–0930 GLM V16（本会话） | 对称显著性否决 + 争议臂分流；可执行预注册（诊断门+账本哈希交叉核验） | 诊断 v2 定稿（151→19/23/18/32/56/3）；发射门 50→40（PI 修正，声明在案）；build **890 事件/623 queries/10 胜者**（对照生产 511/364）+ 争议臂 422 事件（预注册保留不训练）；**覆盖门 890<1,000 保护停止，未训练** |

### 4.2 冠军与资格

- **发布候选：stage-1** best.ckpt（step-3000，SHA 09c419dc…，tag chemaware-role3-positive-20260921）——role-3 +1.8144pp；边界：role-3 开发已消费、验证加载器软泄漏、化学归因实验（回溯 §7 零臂）从未执行。
- **开发冠军：Phase-A** run 2343962 step-2000——role-2 +2.1266pp；便携复现 SHA a8428329…（run_2345481/phasea_reproduction_gate），为 V13/V16 的初始化几何。
- **探索起点：fragment step-750**（+2.3797pp）——V16 的权重初始化（声明式混合设计：挖掘几何=frozen Phase-A 缓存，训练权重=step-750）。
- +5pp 需要 99 净纠正（role-2）/97（role-3）；最好净数 = 42。

### 4.3 已知矛盾（ChemAware 内部）

1. V13 step-1000 与 fragment step-750 数字逐位相同（92.00%=1817/1975）——面板量化巧合（1 query=0.0506pp）或报告回声；引用任一前必须核对 run manifest。
2. 双冠军基线不兼容（stage-1 vs Phase-A 选择宇宙不同）；Phase-A 对 stage-1 增量门未过却按绝对增益保护。
3. 最强信号（union +2.4304pp、V2 +3.94pp）都在被取消资格/无确认的方法学上。
4. role-4 状态：注册表称 BLOCKED（未开）与旧文档"已消费"表述冲突——以 0919 恢复审计为准：**从未打开**。

---

## 5. BioAware 线（负结果链 → B47-U3 待裁决）

**目标：** 用生物上下文（反应网络、样本特异事件）提高注释——静态目录/度数/传播全部证伪（B44 外部 −0.56pp、B45 拓扑 +0.23pp n.s.、B46 双上下文 −0.77pp）；收敛到"真值盲、样本特异、原子反应事件"严格路线。

**B47 尾声链（0921–0922，第一手）：**
- **B1/U0**：18 文件 638MB 溯源全过；参考谱多重性大混杂（中位 3/p90 24/max 287；max−期望单谱提升中位 0.0401；log 数×提升 Spearman 0.8297；**22.24% 双唯一 query 的 Top-1 身份被 max 聚合翻转**，双源复现 22.16%/22.39%；加合物分支竞争 23.43% query 暴露双假设）。
- **U1**：33 配方冻结 OOF +0.0682pp 但 131/74 风险不合格；**U1b v1 撤销**（argmax 候选顺序泄漏）、v2 = 结构安全 Top-1 no-op（0/0）——数学边界：不借候选顺序就不能修任何并列。
- **U1c**：分母瓶颈定位——51,976 事件/4,300 Top-1 身份 → 谱置信+共识 238 身份 → Rhea-safe **127**（首次 <200）；`RHEA_COVERAGE_OR_HUB_BOTTLENECK`。
- **U2**：目录覆盖——238 中 Rhea 129 / safe 127 / +严格 KEGG 141 / +EMRN 上限 149；89（37.4%）无目录边；`CATALOGUE_COVERAGE_UNDERPOWERED`；但事件级功效未测量（3,243 折叠种子事件）。
- **U3-v3**（0922 冻结，**READY_NOT_RUN**）：候选特异类型化事件层（query 特征–候选身份–种子特征–反应源/方向），加合感知中性质量重放、非氢元素 delta 签名冻结防重定义、跨候选特异性、三族结构零（度保持重连/样本间种子置换/query 内错身份置换）、fail-fast、+3pp 终门=1,560 可干预决策 headroom 门先行。

**Reverse/A 线：** A0–A2 上下文联合模型不胜技术基线（重置）；A1 多通道 NDCG@5 +2.962pp 但 A1b 官方空间 −0.590pp CI 跨 0，路线按预注册门停止；保留价值=标准品/碰撞能选择优化。

---

## 6. 应用生物学线（MTBLS13729 + LCNEC，总账 §23–25）

**MTBLS13729（黏蛋白糖组，30 患者/60 样本/4 面板）：** 四轴冻结——唾液酸/LacdiNAc/LacNAc 延伸/硫酸化同向（10/10，targeted EIC ~+1.935 log2；亚型交互 +2.209/+2.142，q≈0.0018/0.0016）；free Neu5Ac vs CMP-Neu5Ac/UDP-GlcNAc +1.693/+1.922（Holm p=0.0273）；外部 O-糖组两例 MUC 显示 core-2 与 sialyl-Lewis X/A 扩张、alpha2-6 下降（**不支持全局高唾液酸化**）；O-acetyl-Neu5Ac-like/NXPE1/MUC2 载体/单细胞审计多项未过正式门——**最终语言必须是 hybrid mucin glycome 重塑假说**。标准品优先级：f703 Neu5Ac > f1597/f3019 甲基鸟苷 > f1717 乙酰化多胺异构体 > f3222 酰基肉碱组合。

**LCNEC（34 对肿瘤/癌旁）：** 263 个 QC 资格家族漏斗（source 表 16.0% 重叠 → DreaMS 候选覆盖 60.1% → P2b top 一致 51.7% → 完整证据保留 25.1%）；可解析 19 个 source 正对照 17/19 一致（89.5%，诚实暴露两同分子式异构体错误）。四优先候选：ADP family +2.400 log2（33/34）、ADP-ribose +1.556（31/34，PARP1 +1.319/PARP2 +0.868 独立支持）、ascorbate +5.407（32/34，MSI Level-2）、quinolinate +2.047（28/34，QPRT −0.853/HAAO −1.103/IDO1 −1.063 q≤8.13e-10）。负边界：6 个患者级关系 0/6 过、16 项技术混杂 0/16 过；**精确新代谢物声明=0**。

---

## 7. 平台、Agent 与汇报资产

- **注释平台**（0817 架构账）：M0–M9 模块；消融 FP-proxy 0.764→0.000（+m/z）；注释率 ~5.9% 置信/94.1% 暗物质；校准 LOO 70.5%；decoy FDR 9 spec/s；Level 1/4 永不发射。**0827 指导的最终形态**：mzML→峰→encoder→结构生成器解码器+置信度——现平台是其中间形态，解码器路线未启动。
- **Agent 化科研组织**：claim registry（19 行）/ agent execution registry / validation board / integrated adjudication route；`PASS/complete` ≠ 科学主张通过。
- **PPT 流水线**：`.codex_tmp/report2*`（44 页审计→38 页重建）；deliverables 0906/0911 V7/V8；数字引用必须回总账校验。
- **A 教授交流**：V3 终稿（13 页）——开放世界拒答+弃权为中心问题；五数据角色；GeMS-A10 23,517,534 谱；禁用叙事清单在案。

---

## 8. 全项目矛盾与未闭合清单（跨线）

1. **旧 cohort（23,876）数字仍在后期文档中被引作动机**——0906 重置后它们只有历史资格（E4-A +0.635pp 等）。
2. 双冠军（stage-1/Phase-A）基线不兼容 + Phase-A 增量门未过（§4.3）。
3. role-4 outer：16,198 queries 从未打开；post-seal recovery 状态机未建——**任何 outer 主张都无资格**。
4. GNPS 面板 official 基线数值缺失（待 Stage-6/后续化学运行产出）。
5. Noise×ChemAware 组合性零实验；mature-E8 与 Phase-A/stage-1 从未同场对比。
6. 服务器-本地溯源缺口：Noise V3 账本、B47 registry、V16 输出仅服务器侧；本地只有报告/哈希。
7. 69 空目录 + 13 不可读目录（注册表口径 19 vs 总账 §28 的差异）应登记为中断/空壳。
8. Stage-1（noise）的 p01 保持 0.980–0.982（preservation>0.995 但 p01 弱）——未复核。
9. MassBank PI/NL 家族同账本（非独立确认）；"GLM" 前缀是命名空间非方法主张——论文禁止写 "GLM analysis"。
10. V16 发射门修正（50→40）为 PI 级事后修正，已声明；覆盖门（1000/500）未被修正且不应被修正至观测值。

---

## 9. 统筹建议（下一步三步）

**① 让 Stage-6 跑完并按晋级合同裁决（现役唯一 5pp 赌注）。** run_2346857 完成后：取 Stage-1/靶向/对照三臂全指标 + GNPS 双面板；无论过否，冻结工件上执行 2pp 天花板审计的零更新审计（关系/梯度对齐），然后 T1（多关系铰链）单变量推进——这是文献约束下唯一可信的 >2pp 路线，且其设计已预注册。

**② 化学线按测量关闭增量预期，转入"确权"模式。** 增量空间已被 V16 诊断测量（58% 错误无化学方向；890<1000 事件）。可选三支：(a) 接受 Phase-A/stage-1 为化学线最终结果，走 role-3 非回归确认 + GNPS 双面板 + MoNA 极性迁移（协议已建未跑）的**确权链**；(b) 争议臂 422 事件的预注册探索训练（唯一未训练化学质量）；(c) V14 SIRIUS 服务器运行（面板已建）作为"真正独立源扩展"的最后一次覆盖门尝试——失败即写"化学线覆盖天花板"结论文档。

**③ 论文定位：+2~3pp 的诚实增量 + 体系贡献，放弃 5–10pp 头条。** 可辩护故事：严格角色面板协议 + GNPS 模型盲基准 + 两条已确认共享权重增益（+1.81 role-3 / +2.13 role-2）+ 因果对照方法论（动作内容 vs continuation 分离）+ 负结果账本本身（错误图谱:58% 无化学方向 = 对全领域的可引用发现）+ LCNEC/MTBLS13729 应用闭环 + 平台。5–10pp 属于下一代（结构生成器解码器，0827 路线）立项叙述。

---

## 10. 来源与置信度

- 第一手：docs/ 现役 14 篇 noise（0916–0930）+ B47 尾声 6 篇 + 总账 718 行 + 0827 指导；本会话全部 chemaware 0926–0930 事件（union 预检、硬化合同、V16 设计/诊断/建造/覆盖停止）。
- subagent 全读审计：noise 8 月 35 篇、noise 9/1–15 43 篇、chemaware 65 篇、基础设施 49 篇 + 补遗、注册表 1,185 目录。
- 总账+注册表+定向首读覆盖：BioAware 60 篇（9/25 后零增量已验证）、应用生物学 68 篇（同前）。
- 机器清单：`data/validation/project_repository_artifact_manifest_20260925/`（docs/tasks/validation 逐文件 SHA256）。

**本总览的 GLM_ 前缀为命名空间约定；任何论文引用不得使用 "GLM analysis" 表述。**
