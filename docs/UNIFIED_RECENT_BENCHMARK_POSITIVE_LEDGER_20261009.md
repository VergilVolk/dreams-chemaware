# 统一论文近期 benchmark 证据账本（2026-10-09，审计修订）

本账本整理可以继续使用的正向结果和决定性的零/负迁移结果，并同时记录数据集、分母、开发/封存资格和泄漏状态。它不是把不同任务上的百分点相加。MassSpecGym 与 GNPS Gold/Silver 均已被开发过程消费；Enveda-180 在本账本生成时仍未打开模型分数。

## 1. GNPS Gold/Silver 统一开发面板

数据集：identity-disjoint `10,995` queries、`87,518` candidate molecules、`175,171` directed spectrum pairs；formula-disjoint `5,261` queries、`24,401` candidate molecules、`47,724` directed spectrum pairs。角色：`development / consumed`。泄漏状态：模型选择已发生，绝不能再称终局外测；与 MSG/MoNA 的身份/公式排除不等于与所有 GNPS 生态训练语料隔离。

| 方法 | identity R@1 | formula R@1 | 可保留的正向结果 | 资格 |
|---|---:|---:|---|---|
| official DreaMS | 85.35698% | 86.80859% | 冻结共同基线 | development baseline |
| Noise V1 | 86.55753% | 88.06311% | 对 official `+1.20055/+1.25451 pp`；formula-cluster CI 均严格为正；micro AUROC 双面板第一，formula pooled AUROC `0.9299` 第一 | 强开发/迁移证据，非终局外测 |
| P2b frozen on Noise V1 | 87.13% | 88.25% | formula MRR `0.9290` 第一；identity pooled pairwise AUROC `0.9416` 第一；R@1 与 WSE 配对统计平手 | 冻结重排器开发结果 |
| weighted spectral entropy | 87.37% | 88.27% | R@1、near R@1 双面板名义第一 | 最强经典基线 |
| ChemAware Stage-1 encoder | 比 official `+0.3001 pp`，CI `[−0.2388,+0.8246]` | `+0.2091 pp`，CI `[−0.5357,+0.9454]` | 两个 R@1 CI 均跨 0，风险净收益为负；不能列为 GNPS 正迁移 | role-3 同源结果仍成立；GNPS 迁移判为零/未证实 |
| ChemAware Phase-A encoder | 比 official `+0.2274 pp`，CI `[−0.3109,+0.8037]` | `+0.3992 pp`，CI `[−0.3234,+1.1281]` | 两个 R@1 CI 均跨 0，风险净收益为负；不能列为 GNPS 正迁移 | role-2 强开发结果仍成立；缺 role-3 与 GNPS 确认 |

禁止引用：S1 oracle `+4.93/+5.44 pp`、router v2 `+2.93/+4.03 pp`。两者因真值/面板重叠泄漏正式撤回。RRF 1.7 的 GNPS 确认失败，保持 `retired_audit_only`。ChemAware 的同源大增益不得外推成 GNPS 迁移成功。Noise MSG 尚无可替代上述冻结结果的统一 GNPS 正结果。

## 2. MassSpecGym/ChemAware 同源角色面板

角色：开发、同源确认或历史封存面板，均已消费；不是最终独立外测。

| 资产 | 数据集与分母 | 正向结果 | 当前资格 |
|---|---|---|---|
| Noise V1 shared encoder | corrected MassSpecGym held，`18,333` queries | 对 official R@1 `+1.0527 pp`；随后在 GNPS 双面板迁移为 `+1.20055/+1.25451 pp` | 当前 Noise encoder 主资产 |
| ChemAware Stage-1 native-triplet encoder | role-3，`1,929` queries | R@1 `+1.8144 pp`，`57/22`，formula-cluster CI `[+0.8155,+2.8703] pp` | 已封存同源确认；统一模型默认 ChemAware encoder |
| ChemAware Phase-A max-boundary encoder | role-2，`1,975` queries | R@1 `+2.1266 pp`，`54/12`，CI `[+1.2761,+3.0303] pp` | 强开发结果；provisional 对照 |
| ChemAware V2 multi-null residual reranker | role-3，`1,929` queries | R@1 `+3.9399 pp`，`93/17`，CI `[+2.8191,+5.1921] pp`；相对 truth-blind rotated null `+3.3178 pp` | 最强化学候选残差信号；outer 已消费，必须逐候选适用，不可当 dense score |
| P2b frozen local rank fusion | sealed P3 main，`3,000` queries | R@1 `+1.07 pp`，CI `[+0.24,+1.89] pp` | 可引用封存重排结果；near-core 有害，非通用部署 |

## 3. 从小候选池走向实体找回

数据集：GNPS identity-disjoint 开发面板，`7,695` match queries + `3,300` no-match queries；阈值在混合 match/no-match population 上按 margin 校准。角色：`development / consumed`。这不是生物学结果。

在 5% FDR 下，coverage 的分母是 `7,695` 个 match queries：official DreaMS `55.36%`，Noise V1 `58.36%`，P2b frozen on Noise `65.65%`，WSE `67.11%`。因此现有最强实体找回基线仍是 WSE；Noise 的增量是真实但不足以宣称统一算法胜出。下一阶段必须在冻结模型后把同一 no-match/FDR 协议迁移到 Enveda，并进一步进入真实样本中的实体找回，而不是只报告封闭候选池 Top-1。

## 4. 当前准入裁决

统一模型只有同时满足以下条件才可打开 Enveda 模型分数：

1. GNPS 两个已消费开发面板上胜过预注册的最强单方法，而不只是 official DreaMS；
2. 报告 R@1/3/5/10/20/50、MRR、candidate macro/micro/pooled AUROC、corrected/introduced、风险净收益、formula-cluster CI、near 子集和实体找回 FDR-coverage；
3. ChemAware 三资产分开，V2 使用逐候选 applicability；RRF 不复活；BioAware 无真实事件上下文时完全 unavailable；
4. P2b 的三个原始通道与冻结 `p2b_noise_v1_frozen` composite 互斥；当前 only-enabled 代表是 composite，原始通道只能进入单独预注册的 replacement 实验。
5. 冻结模型、模块清单、阈值、候选轴和所有 SHA-256 后，只运行一次 Enveda；任何后续修改都不得回到同一 Enveda 面板重新调参。

## 5. 生物学应用边界

Enveda 是标准谱库外测，只能回答跨碰撞条件的候选排序与实体找回泛化。真正超出“注释—差异—买标准品”的应用仍需独立的事件级样本上下文、未知实体增量和前瞻扰动闭环。BioAware B44--B46 是静态目录/拓扑负证据；B47 在获得真实事件级 MS/MS 与共变输入并通过外部门以前，不进入统一模型性能主张。

## 6. 失效或尚未执行的近期结果

- `lcnec_p2v2/run_2354362` 的 23 个 medium/high 与“91% dark matter”由旧自制 entropy 实现产生；该实现对相同谱可返回 0，提交 `fc6b744`/`2453297` 已要求所有 P2 v2 输出重算。旧数字不得引用。
- P0-G 旧 no-match 实现没有移除阳性分子，旧 AUC 0.507 不再是有效方法学结论。修复后应以混合 match/no-match 的 FDR-coverage 报告为准。
- B47 U3-v3 只有 4,996 个 candidate-specific queries 和 2,239 个潜在干预机会；真值未打开，集中度与身份质量门失败，`pass_to_frozen_event_ranking_evaluation=false`。仓库中只有 2026-10-03 补救预注册，尚无补救结果。
- LCNEC global-remodeling Stage-1d 在 34 对患者、263 个冻结实体上通过进入 Stage-2 的门，但仓库中没有 Stage-2 结果。它不能替代跨平台复现、暗实体增量或转换边一致性。
