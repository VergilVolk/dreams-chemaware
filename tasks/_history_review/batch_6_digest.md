# 批次 6 摘要（B20-B40 系列、ChemAware 规则内核、MSnLIB、化学先验、外测失败根因）

REFERENCE_ANCHORED_PEAK_OPERATOR_A0_RESULT_20260912.md | 2026-09-12 | 结果 | 标准品锚定峰级化学算子 A0 在困难多候选查询上获 +5.37 pp（0.5854→0.6392），加 MS1 后困难查询 +8.64 pp，但非通用方法。 | 全体 2,707 查询基线 Recall@1=0.7281，谱学算子 +1.03 pp；身份聚类 bootstrap 95% CI [0,0.0227] 下界未>0，正确 feature 入候选率仅 0.7610，57 校准身份仅 15 个多候选。 | 证明"酰基肉碱峰级证据有用"的局部正信号，要求下一阶段补 conserved-core peak set 与 top-k 候选。

BIOAWARE_2026_COMPETITIVE_VALUE_AND_NOVELTY_AUDIT_20260912.md | 2026-09-12 | 审计 | BioAware 应定位为"目录/网络先验风险受控 Top-1 路由器"而非反应网络推理模型；主要增益来自目录覆盖度而非反应路径。 | B35（259,176 谱/30,984 身份）Recall@1 +2.88 pp（50 修正/3 新错，p=5.52e-12），负离子域 +6.24 pp；SOTA 主张评分 0，算法创新性 4/10。 | 确立诚实命名边界与六步转硬创新路线（目录对照消融、外部盲测、属性化 sink 模型）。

BIOAWARE_B36_REACTION_SPECIFICITY_ABLATION_CONTRACT_20260913.md | 2026-09-13 | 合同（含正式结果） | 四臂消融判定反应上下文是否在目录先验之上提供独立增益。 | job 2336381：catalog-only +5.581 pp（50/2）最强，full +5.000 pp（47/4），reaction-only +1.395 pp（19/7）CI 跨零；full 相对 catalog-only −0.581 pp（p=0.332），错配对照 p=0.273，科学门失败。 | 裁决现有 5-6 pp 增益属谱学+目录覆盖专家，禁止进入共享 embedding 微调。

NOISE_BEST_ACTION_V7_CORRECTIVE_RESTORATION_SPEC_20260912.md | 2026-09-12 | 合同（已执行被拒） | V7 保留 E4/E8 直接共享编码器路线，仅修复 V6R 把辅助更新误算为纠正信号的科学缺陷，用 U_combined−U_noncorrective 反事实恢复纯纠正更新。 | 32,114 准入纠正行/3,482 训练查询，恢复目标 0.25 更新分数、0.90 保护保留；job 2336335 实测 routed 仅 +0.130911 pp 且低于 shuffled −0.060001 pp，被拒。 | 记录 V7 边界：信号诊断达标≠held 提升，单种子只能 pending_multiseed。

NOISE_BEST_ACTION_V8_QUERY_LOCAL_DIRECT_SPEC_20260913.md | 2026-09-13 | 计划（canary 合同） | V8 唯一语义变更是纠正残差内 detach 候选 reference，迫使梯度走 clean query 与 action view，检验梯度局部性。 | 依据 job 2336335：transfer 能量约 70.6% 在 reference 侧、payload 约 92.5% 在 reference 侧；四臂 canary（512 纠正查询/单 epoch），需角色/语义/局部性/优化器四门全过。 | 把 V7 根因（reference 旁路）转化为可独立检验的因果测试。

NOISE_BEST_ACTION_V7_RESULT_AND_ROOT_CAUSE_AUDIT_20260913.md | 2026-09-13 | 结果+审计 | V7 三臂完整结果：routed +0.130911 pp 仅比 clean 高 0.021819 pp、低于 shuffled 0.060001 pp，因果门失败。 | 67 修正/43 新错=净 24（需约 733 才达 +4pp）；根因一：约 92.5% payload 纠正能量走 reference 侧旁路；根因二：corrective 更新分数 p10=0.089913（合同 0.25），29% 步骤未达标；序列化崩溃为 numpy.bool_ JSON 编码。 | 确立"action-view 成功≠clean 编码器监督"的科学瓶颈论断，冻结 V7 为被拒诊断。
