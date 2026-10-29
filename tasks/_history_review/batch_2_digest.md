# Batch 2 阅读摘要（2026-08-22 → 2026-08-30）

清单：`tasks/_history_review/batch_2_files.txt`（52 个文件，全部用 read 工具逐字完整阅读，长文件未跳读）。

---

## 逐文件摘要

1. `RAW_RERANKER_V1_FREEZE_20260822.md` | 2026-08-22 | 合同（冻结记录） | 冻结 RAW 候选组内重排器 v1（逻辑回归、12 维特征、C=0.01、gate 阈值 0.24098341166973114），不动 DreaMS 权重。开发集 g8r_val 上 Recall@1 0.8081→0.8516（+4.35pp），formula-cluster CI [0.011,0.069]，McNemar p=0.0009；near 层净 +13（p=0.019 唯一显著层）；正式转换表 gated 44/17。 | 开发集 PASS，但明确"非 near 特异突破"，最终裁决等 Test-A/B 盲测 | 把 g8r 系列工作从"改权重"切换为"冻结后重排"的第一份正式契约。

2. `STATUS_HANDOVER_20260822.md` | 2026-08-22 | 交接/计划 | 一句话状态：RAW-v1 已冻结，最终盲测脚本就绪待服务器提交。给出大规模三面板：有效 query 102,449（Test-A 35,662 / Test-B 32,096 / Test-C 34,691，各自 IK14 4,287/4,287/4,288），Test-C 留给下一代；候选库 166,842 谱（全 HDF5 231,104 减 g8r）。 | 仅计划（待盲测） | 把"过门标准"（CI 下界>0 且 MRR 不退）与五条纪律（不夸大、IK14 零重叠、配对检验、双口径分开、必须 sbatch）写死。

3. `RAW_RERANKER_V1_FINAL_VERDICT_20260822.md` | 2026-08-22 | 裁决（结果） | RAW-v1 冻结盲测：Test-A +0.45pp（CI [−0.0067,+0.0160]，p=0.444，59/50）、Test-B +0.73pp（CI [−0.0063,+0.0213]，p=0.312，77/64），效应从 +4.35pp 衰减至不足 1pp 且不显著；A/B 重叠 1,427 个 IK14。根因是"模型不能可靠判断哪些 query 值得推翻 DreaMS"（干预接近五五开）。 | FAIL（未显著超越官方 DreaMS）→ 永久归档，保留为 v2 峰级基线 | 本批次第一记警钟：开发集增益在冻结盲测下大缩水，催生 P1/P2/P3 与后续全部封存纪律。

4. `g8r_reranker_roadmap_20260822.md` | 2026-08-22（内含 08-23 复审） | 计划/审计 | RAW-v1 后路线：P1 风险受控残差重排、P2 候选组 listwise、P3 新锁定测试集。P1 开发结果 0.8435（29/7，gate 覆盖 8.06%）；四门控对照中 disagreement-only 与 learned gate 在 λ=2 效用下并列（15=15）；P3 锁集发现 HDF5 内模拟谱 119,029/231,104 与真实谱 112,075 混用，g8r 训练锚点 6,612/10,000 为模拟谱；P3 v3 定为真实谱 only，main=3,000、near-core≥450，本地容量审计 main 3,285 / near-core 497。 | P1 学习门控被判定不可宣称胜出；P3 协议反复重写后达成正式锁集设计 | 定义了整个批次赖以生存的评价基础设施（真实谱过滤、query/候选库分离、SHA256 fail-closed）。

5. `g8r_p2_listwise_design_20260823.md` | 2026-08-23 | 合同（设计） | P2 受保护 listwise：有界残差 s=s0+Δmax·tanh fθ、零初始化、分子 max 聚合、安全项 L_safe；兼容性复现门要求精确复现 0.8080645161 / 0.8516129032 / 44/17 / 0.4645161290。 | 仅计划（复现门任一不符即停） | 把"新实现必须先逐位复现旧数字"确立为工程纪律，为 P2b 铺路。

6. `P2B_RANK_FUSION_FORMAL_RECORD_20260823.md` | 2026-08-23 | 结果+合同（冻结） | P2b 局部候选排序器冻结：score=0.10·DreaMS+0.00·sqrt_cosine+0.10·entropy+0.80·NL。开发嵌套 OOF（5,037 query）Recall@1 0.8606→0.8997（+3.91pp，CI [+2.91,+4.87]pp），near +5.83pp，280/83；封存 P3-main（n=3,000）0.8793→0.8900（+1.07pp，CI [+0.24,+1.89]pp，McNemar p=0.0101，89/57，净修正相当于减少 8.8% Top-1 错误）；但 near-core（n=496）0.4879→0.4456（−4.23pp，20/41，CI 全负）。 | P3-main PASS / near-core FAIL（硬边界）→ 永久保留为强制保底模块 | 本批次最稳的系统级成果：证明局部峰/中性丢失证据与 DreaMS 互补，同时确立"near 必须回退"的安全边界。

7. `CHEMAWARE_NOISE_BIOPRIOR_DECISION_20260823.md` | 2026-08-23 | 决策（计划） | 四项架构决策：P2b 永久保留、ChemAware 峰级微调为主创新、双重映射为解释核心、生物网络先验后置为候选后验。盘点资产（概念 AUPRC 0.659 vs 基线 0.200；469 环境 0.240 vs 0.0447；175 桥梁）与旧噪声路线三宗罪（drop­out 混杂、随机遮峰与预训练重复、反事实样本 2,207 条口径陈旧）。 | 决策性文件；近期成败第一决定点定为"定向峰干预 vs 匹配随机删峰"因果门 | 全项目的架构宪法：排序器/微调/解释/生物学四层分工由此定形。

8. `G8R_REAL_ERROR_ATLAS_PROTOCOL_20260823.md` | 2026-08-23 | 协议（计划） | 真实错误图谱协议：四类转移（protected_correct/corrected/introduced/persistent_wrong）、双轴分解（positive deficit vs negative excess，参考分布按 near×候选数分层）、三层边（query/候选分子/谱图对）、335 规则仅作描述。 | 仅计划（代码完成待服务器） | 为后续全部 23,876-query 错误机制分析提供数据契约。

9. `CHEMAWARE_DIRECTIONAL_NOISE_V2_PROTOCOL_20260824.md` | 2026-08-24 | 协议+结果（M1 实测裁决） | 二代定向噪声：分子内共识峰（稳定核心≥60% 出现率 / 条件特异≤40% 且强度<0.20）。M1 实测：1,231 个完整配对查询中定向删除与匹配随机删除总体准确率均为 0.5589；552 个基线错误中定向修正 22 vs 随机期望 26.33，CI 跨 0。 | M1（整组硬删除）FAIL，pass=true 被推翻；转 M1-R 机制分臂 | 否定"整组删低频弱峰"，但把失败归因于算子/剂量混淆而非方向本身，开启 v3。

10. `CHEMAWARE_FINETUNE_RESET_20260824.md` | 2026-08-24 | 重置（计划） | 微调主线重置：诊断七大根因（数据饥饿 57,274→1,231、主动选低梯度峰、正例不够难、一次一个负例、目标错位、线性 head 容量错位、噪声角色错位）；新主线为大规模候选组内细粒度微调（S=αcos+βA+γA_NL）；G0 固化为全量审计合同（≥15,000 训练查询、≥2,500 IK14、335+3,151=3,486 维规则）。 | 仅计划（preflight 不过不训练） | 完成从"小样本因果配对"到"全量候选组监督"的战略转向。

11. `NOISE_FINETUNE_V3_RESET_20260824.md` | 2026-08-24 | 重置（计划） | 噪声微调 v3：候选条件化高梯度路线；重申旧候选条件化信号（删误候选独有峰 Δmargin +0.0494，CI [0.0330,0.0675]；删正确候选独有峰 −0.0558，CI [−0.0786,−0.0450]）未被 v2 推翻；定义峰四角色（identity-only/confounder-only/shared/unmatched）与梯度归因 a_k=I_k·dJ/dI_k。 | 仅计划（G1 通过才训练） | 把"梯度选峰"形式化为可审计动作空间，是 S 系列矩阵的理论起点。

12. `NOISE_V3_G1_RESULT_20260824.md` | 2026-08-24 | 结果 | G1 大规模干预审计（23,876 query、2,522 身份、26,990 峰选择、323,880 干预变体；官方前向复现余弦 mean=1.000000）。candidate-gradient 50%：baseline-wrong 修正 138（CI>0）但 baseline-correct 新增 113，净 +25；硬删除 288/560 净 −272；role-only（confounder）6,114 query：25%→100% 剂量净 +5/+17/+81，硬删除 near 75/15。 | 总 gate pass_to_training=false（否定全局施加 candidate-gradient），但 role-only 假设获支持，rescue+safety 双流结构成立 | 首次在完整候选组上量化"梯度能定位错误但伤及正确"的双刃性。

13. `NOISE_V3_S1A_RESULT_20260824.md` | 2026-08-24 | 结果 | S1a 单峰正交矩阵（3 选择器×4 剂量，192,296 单元）。candidate-gradient 净 +19/+25/−60/−272；role-confounder 净 +5/+17/+32/+81（100% 时 near 75/15）；role-identity 负对照净 −16→−460（50% target−random margin −0.00807，CI 全负）。 | role-confounder 为"当前最干净固定方向"，identity 方向负对照 PASS，random/整组删峰假设被裁决不成立 | 用负对照验证了实验方向敏感性，峰角色语义自此可信。

14. `NOISE_V3_S1B_HEADROOM_RESULT_20260824.md` | 2026-08-24 | 结果 | S1b 动作互补性与理论上限：1,805 官方错误（Recall@1 0.92440）中 386 个可被非对照动作修正，no-op-aware oracle +1.62pp；near 可修正 329/1,446（22.75%）；candidate-gradient 独有 282、confounder 独有 34。 | 策略设计门 PASS（>1pp）但高容量策略训练数据门 FAIL（正向 action 行 864<1,000，独立正 query 386）；+2pp/+4pp 数学不可达 | 首次给出动作空间的数学上限，迫使团队承认单峰枚举到头。

15. `NOISE_V3_S1C_RESULT_20260824.md` | 2026-08-24 | 结果 | S1c 扩展单峰矩阵（gradient 前 5 峰 + confounder 前 3 峰，394,752 行）：408 个独立可修正错误，oracle +1.709pp；较 S1b 只新增 22 个，正 action 行增至 2,176。 | FAIL（未达预注册 478 错误/+2pp 与 600-query 门槛）→ 转顺序多峰 S2；旧 headroom JSON 的 pass_to_policy_design=true 不作为 S1c 裁决 | 证明"继续加单峰 rank 只产生重复动作"，穷举路线终结。

16. `NOISE_FINETUNE_SYSTEMATIC_STRATEGY_20260824.md` | 2026-08-24 | 计划（总账） | 系统性补充实验方案：四字段策略分类（错误臂 P/N/B/S × 选择器 × 算子 × 目标）、无偏矩阵 S0-S3、历史路线证据总账（规则标签 AUC 0.88→0.59、G5/G6/G7 降 3.1-4.6pp 等）；附录预注册 S1a/S1c/S2。 | 计划+总账文件；核心增量定位为"候选条件化、因果对照、可选 no-op 的峰干预策略学习" | 本批次噪声线的百科全书式框架文档，后续 S/A/E/R 系列全部由此展开。

17. `NOISE_V3_S2_RESULT_20260824.md` | 2026-08-24 | 结果 | S2 动态序列矩阵（35,565 序列、88,596 配对单元、265,788 干预变体）：candidate-gradient 50% 动态 1/2/3 步净 +20/+54/+84；role-confounder 100% 三步 37/1 净 +36（最高精度）；gradient 100% 全部净负（−202/−169/−122）禁用。联合 no-op oracle 663 个错误 = +2.7768pp（S1c 正确口径 553 而非 408，汇总器前缀 bug 已修正）；near 可恢复 574/1,446（39.7%）。 | 允许进入策略学习的候选动作集收敛为 no-op + gradient50% 1-3 步 + confounder100% 1-3 步 | 确立两个可部署动作原语（广覆盖 gradient50%、高精度 confounder100%）。

18. `NOISE_V3_S3_COVERAGE_AND_POLICY_PLAN_20260824.md` | 2026-08-24 | 计划 | S3 错误覆盖扩展与策略聚合计划：样本账 663/1,805（36.7%）、仍有 1,142 错误（872 near）无动作可修；六大未覆盖机制（正例证据缺失、成组碎裂、非局部交互、候选关系、规则未入选择器、不可辨识谱）；S3A 冻结 8 动作×最多 6 步。 | 仅计划（S3A 已冻结待跑） | 把"动作空间天花板"问题显性化，指向 positive-deficit 必须单独处理。

19. `NOISE_V3_S3A_FORMAL_DECISION_20260824.md` | 2026-08-24 | 结果+裁决 | S3A 正交错误机制矩阵（48 单元、278,001 行、66,735 序列、834,003 变体、bootstrap 5,000 次）：candidate-gradient 50% 六步净 +98（+0.41pp 上限）；confounder step5 24/0 但仅覆盖 431 query、step6 转负（深度锁 5）；shared-role 全剂量净退化（step1 208/578）被否决；99.91% 新增错误流向同分子式、78.84% 流向 MCES 0-2。联合 outcome-oracle 799（+3.346pp），距 4pp 所需 956 尚缺 157。 | 允许进入动作空间扩展（A4），不允许直接训练最终策略 | 用 oracle 上限证明"先扩动作空间再学策略"的顺序正确性。

20. `NOISE_V3_A4_EXACT_PEAK_SCAN_PROTOCOL_20260825.md` | 2026-08-25 | 协议 | A4 全峰精确干预协议：全部 1,805 错误 + 每 3 个匹配正确对照（匹配率与复用落盘）；每峰 4 档剂量精确前向，并列时保存 adversarial negative。 | 仅计划（新增≥157、合并 oracle≥956 两道门） | 从"梯度近似"升级到"精确边际扫描"的合同。

21. `NOISE_V3_A4_FORMAL_RESULT_AND_A4B_DECISION_20260825.md` | 2026-08-25 | 结果+裁决 | A4 正式结果：25,275 谱、206,288 峰动作、825,152 变体、对照匹配边 5,415（83.7% 同分子式+相近难度）。25%→100% 可恢复错误 138/289/461/738，受损对照 183/375/652/1,078；梯度 Top-1 仅覆盖 411/776（Top-3 578、Top-25 703）；合并 oracle 920 = +3.853pp。 | 三个进入门全 FAIL（新增 121<157、920<956<1,000）→ 禁止拟合最终策略，转 A4-B 正例证据恢复；错误机制矩阵显示 1,029 未恢复错误中 910 含 positive-deficit | 本批次最重要诊断：剩余瓶颈不是"删哪个峰"而是"补回正例证据"。

22. `NOISE_V3_A4_NONLINEAR_ACTION_TEACHER_PROTOCOL_20260825.md | 2026-08-25 | 协议 | 非线性动作教师预注册：1,784 可干预错误+3,132 对照；多任务预测修正/伤害/margin；效用 U=P(correct)−2P(introduce)；严禁泄漏字段（scan_kind、baseline rank、错误家族等）。 | 仅计划（AUPRC≥2×阳性率、净收益 CI>0、introduced≤corrected/2、新增≥80 四门） | 防泄漏的输入黑名单成为后续所有策略模型的规范。

23. `NOISE_V3_A4_ACTION_TEACHER_FORMAL_RESULT_20260825.md` | 2026-08-25 | 结果 | 动作教师 OOF：修正任务 ROC-AUC 0.849/AUPRC 0.132（7.59×阳性率）、伤害 0.853/0.122（9.34×）、margin Spearman 0.540；40% 覆盖 182/39 净收益 104（CI [0.0115,0.0309]）。但相对 confounder-only 差 +14 不显著（CI [−0.0071,0.0122]），历史外新增仅 19<80。 | pass_to_counterfactual_training=false（不得绕过）；gradient-only/shared-only 显著负净收益被否定；确立"安全层 confounder + 扩展层教师"两层结构 | 证明动作结果可学习但旧动作空间独立覆盖不足，P-arm 成为唯一出路。

24. `NOISE_V3_A4B_POSITIVE_EVIDENCE_AND_TWO_EXPERT_PROTOCOL_20260825.md` | 2026-08-25 | 协议 | A4-B 正例证据恢复与两层专家预注册：B0 用同 IK14 同加合物真实谱原型以固定剂量 0.10/0.25/0.50 移动查询 embedding，错误身份原型为对照；列出已终止动作策略清单（gradient-only、shared-only、固定随机遮峰、整组条件峰删除、无 no-op 全覆盖）。 | 仅计划（B0 四门：新增≥80、CI>0、优于随机方向、introduced≤corrected/2） | 正式埋葬一批死路线并锁定 P-arm 实验设计。

25. `NOISE_V3_A4B_POSITIVE_EVIDENCE_RESULT_20260825.md` | 2026-08-25 | 结果 | B0 通过：4,998-query 面板上 baseline 0.6389→teacher 0.7351（+9.62pp），542/61，风险净收益 420，历史+A4 外新增 159（≥80 门过），公式簇 risk-net 0.0840 [0.0691,0.0997]，positive-deficit 修正 465、near 374。 | 四个预注册门全 PASS（但明确为训练期教师上限，非部署性能；alpha=0.50 的 1168/173 为探索性不得替换主剂量） | 证明 P-arm 方向真实存在，为 C1 与最终 P/N 联合训练提供依据。

26. `NOISE_V3_C1_CROSSFIT_TEACHER_RESULT_20260825.md` | 2026-08-25 | 结果 | C1 支持谱互斥扩展：23,876 查询全图上 80,250 个严格互斥交叉拟合样本（1,217 身份、627 分子式、49,667 near），baseline 0.5664→0.5912（+2.47pp），2,382/396，风险净 1,590，near +2.34pp，公式簇 risk-net 0.0198 [0.0168,0.0232]。 | 全部门 PASS，进入峰 token 学生阶段；B0 的 9.62pp 被诊断为直接正例对齐放大，C1 2.47pp 才是可信上限 | 确立 P-arm 的真实可迁移量级（约 2.5pp 而非 10pp）。

27. `NOISE_FINETUNE_EXECUTION_CORRECTION_20260826.md` | 2026-08-26 | 纠偏（审计） | 执行纠偏：D1b clean-only adapter 曾被误当噪声微调主结果（三 seed 净修正 +19/+19/+10，平均约 +0.067pp，第三 seed near 净 −6 未过一致性门）；明确三类数字（动作/教师上限、下游重排增益、真改编码器后的 clean 增益）只有第三类能回答噪声微调成败。P2b、C2-C、模块 2 全部暂时封存；F0-F6 执行顺序与七项强制核查（对称编码器、真正用噪声、逐 query 转换表等）。 | 纠偏文件；唯一主线锁定"原始谱图→同一编码器→新 embedding" | 本批次最关键的一次自我纠错，阻止预算继续流向下游重排器。

28. `NOISE_FINETUNE_FINAL_BATTLE_PLAN_20260825.md` | 2026-08-25（含 08-26 纠偏注记） | 计划（总账） | 最终决战计划：D/I/T/M 四级证据标记体系；完整结果总账（含早期 counterfactual head +2pp/100-query、G5/G6/G7 降 3.1-4.6pp、P0 dropout 诊断、C2-C 归档等）；D0-D6 实验顺序（clean adapter→P-arm→N-arm→P+N→概念→冻结评价）；D1 首折 epoch2 修正 10/引入 7 因 λ=2 效用为负正确回退 epoch 0，D1b 提高安全项权重至 8。 | 计划+实现状态；"当前唯一下一步"被 08-26 纠偏文件取代 | 噪声线的完整作战地图与证据分级语言，此后所有文档沿用 D/I/T/M 口径。

29. `0827指导.md` | 约 2026-08-27（文件未写明日期） | 计划（指导） | 课题最终收敛方案：做面向非靶向代谢组学的整合性 MS/MS 解析平台，解决"代谢暗物质注释率低"；功能为多格式输入统一转 mzML→谱图解析→DreaMS 式 encoder→1024 维 L2 归一化向量，并将 decoder 换成分子结构生成器+置信度评估。 | 仅计划（一页纸愿景） | 来自导师/上层的方向性输入，把项目从纯算法拉向"平台+落地"叙事。

30. `NOISE_FINETUNE_MATRIX_EXPANSION_AND_TRANSFER_PLAN_20260826.md` | 2026-08-26（含至 08-27 的多轮结果追加） | 计划+结果 | 矩阵扩展与迁移计划：三矩阵（动作效应/梯度兼容/clean-transfer 效率）；E1 经验噪声校准正式结果（3,412 身份-加合物组、402,458 共识峰簇、1,211,271 匹配峰对、104,952 可靠变异）；E2-M0 冻结 44 单元（corrective 28/robustness 12/negative 4）；E2-M1 18/28 corrective 过门；E2-M1b 收敛四机制家族 14 单元；E3 梯度兼容（acquisition 与 role-confounder 余弦仅 0.074、39.5% 逐 query 为负）；E4-M1 因 ~200 倍量纲压制作废；E4-M1b 联合训练安全回退 epoch 0；三单机制 pilot 全部未形成检索改进；E4-R 四项结构性错位诊断；R0 复现 36,934 动作记录；R2 两档学习率 held clean 分别 −0.084pp（1/6）与 −0.034pp（5/7），action-clean margin 差 0.079 未收缩。 | E4-M1/M1b、三 pilot、R2 均 FAIL（clean 迁移失败），但归因于实现结构而非噪声无效 | 本批次最密集的失败-诊断循环记录，最终指向 E4-A 直接噪声微调的正确形态。

31. `BIOAWARE_NETWORK_EXPERT_AND_CONTEXT_ADAPTER_PLAN_20260827.md` | 2026-08-27 | 计划 | BioAware 双产物决策：近期主线为后验网络专家（typed reaction 超图、带弃权 personalized PageRank、五态输出），context adapter 后置；数据源 HMDB+Rhea+Reactome+KEGG/MetaCyc；预注册 leave-one-seed-out、degree-matched decoy、pathway-holdout 等评估与 Go/No-Go 六门。 | 仅计划 | 把生物网络先验从"口号"变成有停止规则的工程线。

32. `NOISE_E4A_OPTIMIZER_SCAN_RESULT_20260827.md` | 2026-08-27 | 结果 | E4-A 直接噪声微调优化器扫描（fold 0、seed 20260828、解冻最后 1/7 block+head）：low LR +0.287pp；high LR（2e-6/1e-5, clip1）+0.540pp（38/6，risk net 26，preservation 0.99524）安全最优；high LR+clip2 +0.591pp 点估计最高但 preservation 0.99453 跌破 0.995 门 FAIL。clip=1 时 >90% step 触发裁剪（实际缩放 0.35-0.46）。 | 六组 CI 下界全>0；选定 2e-6/1e-5/clip1/4 epochs；优化受限只解释 0.1-0.15pp | 确定直接噪声微调的标准配方。

33. `NOISE_E4A_HIGHLR_MULTIFOLD_RESULT_20260827.md` | 2026-08-27 | 结果 | E4-A 多折确认：三 seed（23,876 query/fold）Recall@1 +0.611/+0.662/+0.632pp（均值 +0.635pp）、near +0.508/+0.537/+0.522pp（均值 +0.522pp）；每 seed 平均约 186 修正/35 引入，risk net 约 117；15/15 fold 公式簇 CI 下界>0，15/15 过安全门；最弱 fold overall 仍 +0.425pp；preservation 均>0.995 但 p01 约 0.980-0.982。 | 开发阶段 PASS：证明固定预注册峰噪声可稳定迁移进共享 DreaMS embedding | 本批次噪声线的里程碑成果——第一个真实的共享权重改善（但远低于 3.85pp oracle）。

34. `BIOAWARE_V1_IMPLEMENTATION_AND_PILOT_RESULT_20260827.md` | 2026-08-27 | 结果 | BioAware v1 工程完成（Rhea 缓存 17,656 反应、78,843 participant rows、10,152 唯一 IK14）；MTBLS1905 外部试点：36 query、18 已知目标，DreaMS 基线 0.750 精确复现；leave-target-out 0.750→0.778（1/0）但 CI [0,0.0968]、10 个度保持诱饵均 0 增益，总门 FAIL；可部署自动种子（8,600 QC MS2、阈值 0.80/margin 0.05）仅 1 条合格种子、0 条网络路径。 | 裁决：系统完成、有效性门未过；首要失败原因为高置信独立种子饥饿 | 诚实的外部负结果，防止把 1/36 包装成成功。

35. `NOISE_FINAL_PN_SHARED_ENCODER_PROTOCOL_20260827.md` | 2026-08-27 | 协议 | P/N/S 共享编码器首轮协议：P 流为真实同分子跨条件正谱对（仪器不同或 CE 差≥10）；N 流固定为已验证 curriculum/all/views4 配方（+0.635pp 基线）；首轮唯一变量为 P 权重 {0.125,0.25,0.5,1.0}；新增 5pp headroom 审计（5pp 需净修正 1,194，官方错误共 1,805）。 | 仅计划（通过门含 cross-condition cosine 提高且 Recall@1 不降） | P-arm 正式并入共享编码器训练的合同。

36. `NOISE_FINAL_POSITIVE_GUIDED_MATRIX_20260828.md` | 2026-08-28 | 协议 | 正例引导动作矩阵（英文）：P/N 冻结联合覆盖 922/1,805=3.86pp，剩余 883 错误中 795 为 positive-deficit；定义三个只用自身正分子真实谱的族（matched_intensity_transport / prevalence_attenuation / consensus_projection）×4 剂量，附最难错误分子方向对照。 | 仅计划（联合并集须新增≥272 才够 5pp 名义上限，目标 350 缓冲） | 把 P-arm 从"教师原型"改写成"可执行谱内动作"。

37. `BIOAWARE_MTBL13729_V1_FAILURE_AUDIT_20260828.md` | 2026-08-28 | 结果+审计（英文） | MTBLS13729 v1 失败审计：21 query（18 真值分子式簇）中 baseline 20/21=0.95238，BioAware 19/21=0.90476，0 修正/1 新增，干预率 1/21=4.76%，CI [−0.1667,0]，10 次度保持诱饵全 0。唯一新增错误为 feature neg_rp:1705：GABA（谱分 0.928094）被两条 Rhea 转氨路径（RHEA:66116/66124）支持的 2-aminobutanoate（0.902365+0.15×0.186671=0.930366，仅超 0.002272）覆盖；唯一 baseline 错误（hypoxanthine 被误排）无任何网络证据；13/21 query 无网络证据。 | 正式门 FAIL；判定为全局一跳 override 规则的负结果，非"生化上下文无用" | BioAware 转折点：数值脆弱的网络 override 被抓现行。

38. `BIOAWARE_LITERATURE_GROUNDED_REDESIGN_20260828.md` | 2026-08-28 | 重设计（计划，英文） | v1 作为排序算法退役，保留为审计负基线与解释源；v2 对齐 MetDNA3 双层设计+DeepMet 连续 metabolite-likeness 先验+NetID 全局一致性+TidyMass2 模块分离；列出 v1 八项缺失（无实验 feature 网络、无共检出/丰度一致、无 RT/同位素约束、无人/组织限制、无全局分配约束、强制 Top-1 等）；V2-0~V2-4 实验序列（含 GABA 歧义重放，预期输出 ambiguous 而非新身份）。 | 仅计划 | 用文献逐条对照解释失败原因，重建 BioAware 的科学地基。

39. `NOISE_FINAL_E5_GUIDED_SHARED_EMBEDDING_20260828.md` | 2026-08-28 | 结果（英文） | E5 系列：入场证据 P/N+正例引导+缺失峰迁移联合 1,257/23,876=5.26pp headroom（recurrent_union_mix 0.50 固定动作 294/27）。E5 pilot：guided weight 1.0 三臂 preservation 全部 0.978-0.979 FAIL（强度引导 +0.5909pp 最佳但保护崩）；E5-B 剂量扫描：0.10/0.25/0.50 单调恶化（0.9923/0.9884/0.9841），0.10 仍不过 0.995 门；发现 safety_ratio 语义错误（改覆盖数而非梯度），新增 safety_stream_weight；E5-C 最终：intensity 0.10/safety 4 过全门（+0.5065pp，36/6，0.996669）但不优于 N-only/2 对照（+0.5234pp）；0.10/2 仅差 +0.0507pp 且差 0.000339 未过保护门。 | 终止全局固定 intensity 剂量扫描；结论为"稀疏、逐 query 的信号被全局平均稀释"，下一步改 per-query 动作挖掘 | 证明固定全局引导动作不可行，同时留下 5.26pp 的监督空间上界。

40. `BIOAWARE_V2_0_FEATURE_GRAPH_EXECUTION_20260828.md` | 2026-08-28 | 结果 | v2-0 两层证据对照：全量实验图 neg_rp 965 节点/13,593 边、pos_rp 5,291 节点/78,468 边；seed 从 21 行扩至 75 行/55 化合物；archived-v1 精确复现 20/21→19/21（0/1）；expanded-Rhea-only 1 修正/1 新增净 0；two-layer 仅 2/21 query 获证据、0/0/0 干预。 | 正式门 FAIL（安全性升、覆盖不足）；post-hoc 超边规则（要求非通用共底物也在种子中）删 183 留 119 条路径得 1/0=21/21，但明示为机制拟合不得报正式性能 | 保守门有效但信息量不足，确认"一跳不够、两层才起步"。

41. `BIOAWARE_NETID_EXTERNAL_VALIDATION_CONTRACT_20260828.md` | 2026-08-28 | 合同（英文） | NetID 外部验证合同：原锁定 NetID v1.0（314 条人工审读），但可行性修正发现其为酵母阴性 MS1 峰/公式任务、公共 MSV000087434 的大规模 targeted-MS2 为肝数据，与 DreaMS 结构检索端点不匹配；结构排序基准改移 MetDNA3 Exploris。冻结依赖校正反应上下文分数（seed 侧非货币参与物分组、只计每组最强路径、低谱学 margin 才允许 override）。 | 合同（盲读规则：manual_curate.csv 只记哈希不读值）；通过门含 ≥200 可评 query、真实图胜过度保持诱饵 95 分位 | 展示了"在开盲前修正协议"的纪律，避免在错误端点上浪费封测。

42. `BIOAWARE_METDNA3_EXTERNAL_PROTOCOL_20260828.md` | 2026-08-28 | 协议（英文） | MetDNA3 Exploris 480（MSV000097913）为主外部基准：239 个 mzML（10,700,428,871 字节）、20 面板（5 样本型×HILIC/RPLC×正负离子）、1,652 个 Level-1 代谢物、30% 种子/70% held-out 十折。冻结划分：开发=NIST urine HILIC、内部验证=NIST urine RPLC、未触碰外部测试=16 个 BV2/鼠脑/鼠肝/NIST 血浆面板。 | 仅协议（claim gate：16 面板合并簇 CI>0、MRR 不降、修正>引入、≥12/16 面板非负、诱饵被击败） | 为 BioAware 找到任务对齐的真实外部考卷。

43. `NOISE_FINAL_E6_OUTCOME_MINED_DIRECT_20260828.md` | 2026-08-28 | 结果 | E6 outcome-mined 蒸馏（882 个可救错误）：mined 三臂 held 增益 0/+0.0507/+0.0169pp 全部 CI 跨 0；fixed 三臂 +0.4052/+0.5065/+0.4727pp 全部 CI>0；同配对下 mined 相对 fixed −0.4052/−0.4559/−0.4559pp（CI 全负）。根因：mined 动作构成 A4 单峰 434（61%）、role_shared 192（27%），而 shared 全局有害——逐 query oracle 从有害家族挑个例=选择偏差。E7：recurrent_union_mix 各权重仅多修 1-3 条、CI 下界全 0，停止扫描；E8 因果消融：冻结 reference −0.2195pp（CI 全负）、terminal-only −0.3714/−0.3208pp，证明共享 query/reference 几何+多步 curriculum 是必要组成；E9 动作陈旧性：在线路径与冻结路径 44.5% 相同、在线仅 10/9、CI [−0.1655,+0.2019]，假设 FAIL 不启动在线重挖。 | E6 永久否决 oracle-union 训练分布；保留 fixed curriculum（E5C +0.5403pp 38/6 仍为单折最优安全配置） | 一组漂亮的对照实验，把"选择偏差不能当训练信号"钉死。

44. `BIOAWARE_METDNA3_DEVELOPMENT_RESULT_20260828.md` | 2026-08-28 | 结果 | MetDNA3 HILIC 开发（117 query/91 IK14，DreaMS 95/117=81.20%，22 错误）：Rhea 一跳原始累加 4/7（−0.366pp）；dependency-corrected 4/2（+0.244pp，CI 跨 0）；OOF safe gate 4/7 失败。raw-MS2 数据层在错误实例上 truth>wrong 仅 13.6%，mean 差 −0.09697（CI [−0.1832,−0.0053]）为显著反向证据；错误分解：11/22 真值不在 Rhea、5/22 raw 支持错误候选；KEGG version1 补 3 个覆盖但 7 个成对实例 truth>wrong=0；eMRN step1-8 每层新增错误真值覆盖全为 0；MS1 预映射 step0 52.7%→step2 59.1% truth recall；全 MS1 特征图（104,988 技术重复对、39,077 稳定节点）两跳递归得 5 稳健 query/3 身份；F2-B 深度三开发消融 95/117→98/117（+2.56pp，3/0）但仅 2 身份、CI 含 0。 | 一跳/直接加分方案全部被否定；确立 F0（MS1 预映射）→F1（受控递归）→F2（raw-MS2 只验证 feature-edge）→F3（冻结门）路线 | BioAware 最有价值的负结果集：证明"网络更大≠注释更好"，缺的是观测预映射与候选特异性。

45. `MTBLS13729_FROZEN_P2B_APPLICATION_PROTOCOL_20260828.md` | 2026-08-28 | 协议 | MTBLS13729 三轨应用协议：官方 DreaMS / E6 实验性 embedding（单折 +0.405pp，30/6，CI [+0.102,+0.812]）/ 官方+冻结 P2b 三路；废弃旧 annotations_fdr.csv（先 Top-10 后 ppm 的错误协议）；MS1-MS2 桥去重后 neg_rp 374,232 张 MS2→24,846 证据/965 特征、pos_rp 419,676→86,646/5,291；三层验证（工程真值引用 P3、真实注释层 retained/changed 只述不裁、生物学层冻结后才读表型并检查 C20:4 锚点 m/z 448.339483/RT 630.436s）。 | 仅协议（E6+P2b 不列主结果，因 P2b 未对 E6 空间重校准） | 把算法成果安全地接入真实生物学队列的桥接合同。

46. `MTBLS13729_BIOLOGY_PIPELINE_STATUS_20260820.md` | 2026-08-20（含 08-29 冻结更新） | 结果/状态 | 生物学线总账：研究设计（Rmu-RN 主终点 n=10 对；neg_rp 30 对、pos_rp 29 对——P06-Ltu 官方文件 2MiB 零字节损坏）；数据硬门（无 QC/blank、四面板固定"肿瘤先癌旁后"的顺序趋势）；neg_rp 发现 3,786 目标、无 BH q<0.05、两发现峰之一 m/z 258.05 疑似批次伪影（Spearman rho=-0.972）；pos_rp 拆峰修正后 20 个独立长链酰基肉碱特征：Rmu-RN +1.35 log2（精确 p=0.0059）、交互 +1.16 log2（p=0.0308）、2,000 组随机面板背景 p≈0.0005；C20:4 锚点 feature 3101→3222 cosine 0.903；注释协议修复（先 ppm 建图再排序，注释率 13.43%→14.79%）；v2 谱库重建 265,011/29,564 谱（v1 因 GNPS PEPMASS 错误全作废）。08-29 更新：555 冻结候选中 8 个过名义门、6 个过 FDR 10%、3 个过 FDR 5%。 | 生物学结论封顶 Level 2"类别积累"，不写通量/酶；主假设=右半结肠黏液腺癌长链酰基肉碱稳态积累 | 本批次唯一"活到论文叙事"的生物学主线。

47. `BIOAWARE_10PP_NETWORK_HEADROOM_PLAN_20260828.md` | 2026-08-28（含 08-29 更新） | 计划+结果 | +10pp 头寸计划：117-query 任务 +10pp 需净修 12 个；四证据臂实测 known MRN+observed MS1+raw-MS2 3/0（+2.56pp，2 身份，CI 跨 0）、SMN 4/8（−3.42pp）、RT 0/3（−2.56pp，R²=0.644）、predicted eMRN 5/6（−0.85pp）；实际修正并集 7、乐观并集 11<12，扫权重数学不可达。08-29 更新：G1/G2 新增 Morgan decoder 头寸（11 未解决错误+2）与规则似然（+4），联合 headroom 13 首次过 12；G3-v1 未校准叠加全弃权（残差权重 0.01-0.06）；G3-v2 归一化+族共识 2/0=+1.71pp（0.8120→0.8291）但 CI [0,0.0446] 下界为 0；G3-v3 peer 共识 0/0 归档。 | +10pp 不可承诺；G3-v2 为"最安全开发正信号"非显著提升；RP 继续封存 | BioAware 数字的诚实账本——每个正信号都标注 CI 含 0。

48. `OFFICIAL_DREAMS_CHECKPOINT_BASELINE_LEDGER_20260829.md` | 2026-08-29 | 结果（基线总账） | 官方冻结权重 official_embedding_slim.pt（SHA256 8928f908606c0bd652c5a4107d3c35102f660622958c225a1f625abe4b1ba245）的跨任务基线：全量 gate 9,289 query R@1 0.930132；g8r locked 620/0.808065；P2 OOF 5,037/0.860631（near 2,094/0.761223）；全错误图谱 23,876/0.924401（near 子集 13,784/0.895096，1,805 错误中 1,446 near）；P3 六面板（main 3,000/0.879333、isomer 1,989/0.794872、near-core 496/0.487903、near+mid 661/0.544629、exposed 851/0.773208、sim-to-real 609/0.898194）；结构残差与跨条件分层（同仪器 0.769544 vs 跨仪器 0.689629；CE 差≤10 0.857555 vs >10 0.701272）；生物学面板 MTBLS13729 20/21、MTBLS1905 27/36、MetDNA3 95/117。 | 基线账本（明确"不同行不可横比"） | 让"官方 DreaMS 多强"第一次有了统一、可引用的口径。

49. `PPT_REPORT_SCRIPT_AFTER_SLIDE5_20260829.md` | 2026-08-29 | 报告讲稿（总结） | 组会汇报逐页脚本（第 6-27 页+附录）：评价语言统一（AUROC≠Recall@1）、难度阶梯（0.9244→0.4879）、三条路线分工、被否定的两条"合理"路线、错误机制分解（positive deficit 68.8%）、定向干预 vs 匹配随机（confirmation 28.1% vs 7.6%）、数据协议、四类成熟动作、动作矩阵（gradient +98 vs shared −874）、共享 encoder 配方、+0.635pp 多折结果、3.85pp→0.635pp 的四瓶颈分析（描述性迁移率约 16.5%）、P2b ±边界（+1.07/−4.23pp）、ChemAware 证据语言定位、双重映射三道门、BioAware 克制结论、生物学冻结流程、C20:4 2.34-3.39 倍升高、平台原型与分层系统总结。 | 汇报文档；严格区分"现在可以说/不能说" | 本批次结束时项目自我认知的快照与对外交付的口径。

50. `NOISE_ROUTE_VISUAL_GALLERY_20260829.md` | 2026-08-29 | 可视化画廊 | 14 张图索引（S3A 动作矩阵/轨迹/去向、E0 风险加权与剂量响应、E1 峰出现率与抖动、规则噪声、失败审计、结构残差图谱、30 组人工病例）。读图结论：最稳健正动作 candidate_gradient、低风险补充 role_confounder、大剂量 role_shared 淘汰（新增错误主要落入近邻候选）、经验噪声必须按真实 replicate 校准；当前真实权重成果 +0.635pp overall/+0.522pp near，动作空间 3.35-4.93pp 仅为上限。 | 图册（附阅读边界声明） | 为讲稿与论文准备的视觉资产清单，防止旧图被误读为模型性能。

51. `BIOAWARE_STRATEGY_CRITICAL_AUDIT_AND_REDESIGN_20260830.md` | 2026-08-30 | 审计/撤回 | 严格审计撤销 G3-v2 的 +1.71pp：工程漏洞一——gate 把权重为 0 的证据族计入支持（decoder 权重全折为 0 仍算票）；漏洞二——known/predicted/SMN 高度相关（known vs predicted 约 0.95、嵌套）却当独立票。严格重放后 1/0（+0.85pp），网络族合并后 0/0。另指出候选组内 min-max 放大极弱差异、family 内 max 是未校准 OR、缺失/图外/反证统一填 0、线性加和无条件依赖等六类问题；117-query 任务一个 query=0.85pp 规模不足；生物学侧全局峰图存在潜在循环性（union-family 用全样本相关边、传递闭包串联风险）。下一代方案：带 unknown 状态的概率因子图（三组较独立证据族+ILP 全局推断+固定 FDR 注释覆盖）。 | 撤回（+1.71pp 降为历史开发输出）；P0 修复 gate、RP/外部基准继续封存 | 本批次末尾最狠的一次自我打脸，保证论文不会建立在重复计票上。

52. `BIOAWARE_INNOVATION_REPOSITIONING_20260830.md` | 2026-08-30 | 重定位 | 创新性重判：种子传播（MetDNA 2019）、多层网络（KGMN 2022）、全局组合（NetID）、随机游走（ChemWalker 2023）、双层递归（MetDNA3 2025）均已覆盖；换 DreaMS/换 Rhea 只是组件替换。BioAware v1/v2 降级为内部基线与基础设施；唯一差异化主攻方向改为 typed biochemical counterfactual representation learning（BioChem-Embed）：身份不变性+近异构体判别+反应类型等变算子 T_r(z_A)→z_B+峰级忠实性四约束，推理仅输入 clean 谱图。 | 重定位文件（四阶段证据链 A-D 未完成前不得称"世界首创"）；立即停止扫网络权重/在 117-query 上追小数点 | 本批次的终点与下一批次的起点：生物网络从"算法主张"退为"监督关系来源"。

---

## 批次总评（2026-08-22 → 2026-08-30）

**这半个月的主线**是一条"从重排器到错误机制到共享 embedding"的收敛之路。开局（08-22）是 RAW reranker v1 的冻结与盲测：开发集 +4.35pp 的明星数字在 Test-A/B 上缩水到 +0.45/+0.73pp 且 CI 跨 0，团队立刻学会了"开发集结论必须经冻结盲测"这一课。随后 P2b 局部排序器（中性丢失主导的 0.1/0.1/0.8 融合）成为最稳的系统级成果：开发 OOF +3.91pp、封存 P3 主面板 +1.07pp（McNemar p=0.0101）显著，但 near-core −4.23pp 划出硬边界。08-23 的 ChemAware 决策文件把四层架构（共享 embedding / P2b / 双重映射 / 生物网络后验）定形，此后全部工作都在这个框架内推进。

**噪声微调是本批次最长的隧道**。08-24 一天之内完成了 v3 重置、G1、S1a、S1b、S1c 五轮迭代：23,876-query 全图上逐峰、逐剂量、逐序列地扫出动作空间——role-confounder 是最干净方向（100% 剂量净 +81）、role-identity 负对照验证方向有效、shared/整组删除被否决；S1b/S1c 的 no-op oracle 只有 +1.62~1.71pp，S2 动态序列升到 663 错误（+2.78pp），S3A 升到 799（+3.35pp），08-25 的 A4 全峰精确扫描（825,152 变体）把联合 oracle 推到 920（+3.853pp）但三个进入门全 FAIL，由此诊断出真正的瓶颈：1,439 个 positive-deficit 错误不是"删峰"能解决的。A4-B0（教师 +9.62pp）与 C1 支持互斥（+2.47pp）证明了 P-arm 方向真实。08-26 的执行纠偏文件是转折点：D1b clean-only 被误当噪声主结果、P2b/C2-C 被封存、唯一主线锁定"原始谱图→同一编码器→新 embedding"。经过 E0-E4/R0-R2 一系列失败（梯度量纲压制、clean→perturbed 方向反置、R2 迁移失败），08-27 的 E4-A 优化器扫描找到安全配方（backbone 2e-6/head 1e-5/clip 1/4 epochs），5-fold×3-seed 确认 **overall +0.635pp、near +0.522pp、15/15 fold CI>0**——第一个真实的共享权重改善。08-28 的 E5/E6/E7/E8/E9 则系统性地证明了：全局固定引导动作稀释信号、outcome-mined oracle 训练分布是选择偏差（mined 比 fixed 差 0.4pp+）、共享 reference 几何与多步 curriculum 是必要组成、动作陈旧性不构成瓶颈。

**BioAware 走了一条"工程成功、性能失败、定位降级"的完整弧线**。08-27 v1 建成全链路（Rhea 17,656 反应缓存、度保持诱饵、leave-out），但 MTBLS1905 只有 1/0 的不可显著修正且可部署种子饥饿（8,600 QC MS2 只出 1 条合格种子）；08-28 MTBLS13729 上 20/21→19/21 净引入 1 错（GABA 被 2 条转氨路径数值脆弱地翻转 0.002272）；文献对齐重设计后 v2-0 两层证据 0/0/0 保守弃权；MetDNA3 117-query 开发集上 Rhea 一跳 CI 跨 0、raw-MS2 在错误实例上是显著反向证据（−0.09697）、eMRN 扩网零新增错误覆盖；08-30 的严格审计撤销了 G3-v2 的 +1.71pp（零权重计票+嵌套证据重复计数，严格重放后 0/0），最终 BioAware 降级为基础设施，创新主张让位于 typed biochemical counterfactual representation learning。

**死掉的尝试**（有明确 FAIL/归档裁决）：RAW-v1 通用重排（盲测不显著，归档）；P1 学习门控（不优于 disagreement-only）；head-only 线性投影与"整组条件峰删除"（M1 0.5589=随机）；单峰 rank 枚举扩容（S1c +22 错误）；非线性动作教师直接进微调（新增 19<80）；E4-M1/M1b 量纲混合与 clean→perturbed 蒸馏（全回退 epoch 0）；E5 全局固定 intensity 引导（preservation 崩）；E6 outcome-mined 训练分布（CI 全负）；E9 在线重挖（CI 跨 0）；BioAware 一跳 override（0/1）、SMN（−3.42pp）、RT 单独覆盖（−2.56pp）、G3-v3 peer 共识（0/0）与 G3-v2 的 +1.71pp 主张（被撤回）。

**活到后面的资产**：P2b 冻结排序器（主面板 +1.07pp 显著、near 回退边界）；E4-A 定向噪声配方与 +0.635pp 共享 embedding（开发级、待外部总评）；23,876-query 错误图谱与 positive-deficit/negative-excess 机制分解；动作空间监督上界 5.26pp（P/N+正例引导并集 1,257 错误）；真实重复谱经验噪声校准（E1）；官方 checkpoint 基线总账（0.9244→0.4879 的难度阶梯）；MTBLS13729 生物学线（20 特征酰基肉碱类别 +1.35 log2、p=0.0059、随机背景 p≈0.0005，C20:4 锚点 2.34-3.39 倍，555 候选中 6 个过 FDR 10%）；MetDNA3 外部基准合同与 16 RP 面板封存；以及贯穿全部文档的方法学纪律——oracle/教师/权重三级数字不得互换、formula-cluster bootstrap+McNemar、IK14/分子式双重隔离、fail-closed 哈希封存。这半个月的净效果：项目从"堆叠看似合理的先验"进化为"错误机制驱动、证据分级严格、每一步可审计"的研究体系，并在结尾完成了一次方向重定位（BioChem-Embed）。
