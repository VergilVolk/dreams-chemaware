# BioAware SOTA 执行契约：谱学专家、类型化上下文与共享 embedding

日期：2026-08-30

## 1. 当前真实状态

BioAware 尚不能称为 SOTA。已有的一步 Rhea 加分、117-query 小样本 listwise 与局部线性融合均不足以承担完整候选排序。经零净变化反应修复后，MTBLS1905 暴露开发集只保留 1 个有效纠正；117-query formula-OOF 的增益不显著。

当前最强的谱学基座仍是 P2b：开发 OOF 约 +3.91 pp，已打开 P3-main 约 +1.07 pp 且显著，但 P3-near-core 下降约 4.23 pp。因此最终系统必须对 near 候选 fail-safe，不能把 P2b 全局覆盖包装成 SOTA。

## 2. 与最新工作的区别和吸收

- MetDNA/MetDNA3：吸收数据层、知识层、递归传播、种子可靠度、冗余消解与 FDR；不声称发明网络传播。
- KGMN：吸收反应网络、谱图网络与峰相关网络的多层证据结构；不把相关网络当独立投票。
- NetID：吸收全局候选一致性、离子关系和 unknown 状态；后续采用可审计因子图/ILP，而非逐 query 加分。
- MS-Net 2026：吸收多相似度网络与 rank 2–50 rescue 的评价思想；谱学层先解决候选生成与多证据排序，BioAware 只提供有条件的残差。
- TidyMass2 2026：吸收跨数据库代谢网络、功能模块与显著性/随机网络比较；功能模块不能反向充当身份真值。
- JESTR 2025：吸收“候选集本身必须进入训练”的思想以及谱图—结构联合表示；不直接复制其实现。代码审计显示其公开候选正则分支把谱图 embedding 与候选分子 embedding 的平均余弦写入损失，而 `sim_l` 在该函数中没有实际参与计算，因此不能把论文点估计当作可直接移植的教师。我们的对应实现必须使用候选组内严格 listwise/ranking 目标、明确正例、near 负例和风险约束。

## 3. Track A：最终候选系统

### 3.1 谱学主专家

使用冻结 P2b 配方作为多谱学证据基座，并建立 structure-safe router：候选组存在 MCES 0–2 near 对时保持官方 DreaMS，否则允许 P2b。该规则依赖候选结构、但不使用真值；必须在新的外部面板冻结验证后才能成为正式系统。

### 3.2 BioAware 小残差

BioAware 不再承担全部排序，只输出类型化上下文 residual：

- 反应方向与冲突；
- 源侧和目标侧超边完整度；
- 独立种子/反应数；
- 候选特异性与竞争候选数；
- 节点度、反应大小、currency 过滤；
- 数据层 MS2、峰相关、RT、离子家族；
- 1→N / N→1 冗余与 unknown 状态。

只有经过 formula/scaffold/dataset 隔离、degree-preserving decoy、reaction-size decoy 和 target-decoy FDR 后，残差才可干预谱学 Top-1。

## 4. Track B：生物上下文共享 embedding

### 4.1 训练数据

新增正式 manifest 将真实训练谱分为：

1. 同分子跨仪器/碰撞能谱图：唯一检索正例；
2. MCES near 异构体：困难负例；
3. 有效 Rhea 跨身份反应对：辅助类型标签；
4. 分子式/质量匹配非反应对：校准对照。

任何 Rhea 左右侧非 currency `compound × stoichiometry` 签名相同的反应都在聚合前删除。反应邻居永远不是同分子正例。

本地非正式全量审计得到 13,430 个真实身份、2,582 个 near 对和 1,411 个有效反应对；正式服务器版本必须使用 P3 allow-list，预期收缩到 P3-disjoint 真实训练空间。

### 4.2 模型

采用零初始化、范数受限的 contextual peak adapter：查询谱和参考谱共享完全相同的 adapter；输入原始干净谱，输出新的归一化 embedding。关系分类头只在训练期存在，推理时丢弃。

损失：

\[
L=L_{same>near/control}+\lambda_{rel}L_{typed-relation}
+\lambda_{safe}L_{official-correct-margin}+\lambda_{pres}L_{preserve}.
\]

关系损失不直接规定反应邻居之间的余弦距离，而是在 `|z_a-z_b|` 与 `z_a⊙z_b` 上预测关系类型。其传入 adapter 的有效梯度被限制为主排序梯度的 25%，防止生物先验劫持谱学空间。

### 4.3 已完成的工程验证

- Rhea identity-noop 单元测试通过；
- manifest 全量本地非正式构建通过；
- 共享 adapter 端到端 CPU 训练烟雾测试通过；
- 零初始化保持官方 embedding；
- 关系头 warmup 只训练 head，不把随机关系梯度写入 adapter；
- 第二 epoch 确认关系梯度能到达 adapter；
- 梯度上限在烟雾测试中把关系有效权重自动缩至约 0.043，避免其压过主排序目标；
- 三个 sbatch 已拆分为 preflight、pilot、5-fold 和 aggregation，避免数组任务争抢 immutable 输出。

## 5. 不可跳过的运行顺序

1. `sbatch tasks/run_bioaware_embedding_preflight.sbatch`
2. 检查正式身份、near、reaction、control 与每折错误数；
3. `sbatch tasks/run_bioaware_embedding_pilot.sbatch`
4. pilot 必须同时满足排序梯度到达、关系梯度到达且受控、heldout/near 不退化、preservation ≥0.995；
5. pilot 通过后才运行 `sbatch tasks/run_bioaware_embedding_adapter.sbatch`；
6. 五折完成后运行 `sbatch tasks/run_bioaware_embedding_aggregate.sbatch`；
7. 只有 formula-cluster CI 下界 >0、MRR >0、near 不退化、corrected>introduced、五折 preservation 通过，才进入三种子与新外部冻结测试。

## 6. SOTA 声明边界

以下条件全部满足前禁止使用“SOTA”或“全面超过 DreaMS”：

- 与官方 DreaMS 使用完全相同 query/candidate/tie 协议；
- formula、scaffold、dataset/instrument 三种隔离均报告；
- 新外部数据未参与任何模型、阈值、路由或融合选择；
- 总体、near、跨仪器与生物相关类别均不退化；
- 报告 Recall@1、MRR、macro-AUC、coverage-risk、FDR、corrected/introduced；
- 至少三个种子方向一致；
- Track A 与 Track B 分别消融，并报告组合是否互补。

当前最合理的论文创新表述是：

> 在冻结多谱学候选专家之上构建可弃权的类型化代谢反应因子，并把反应类型作为受梯度约束的训练期辅助监督，学习候选无关、查询—参考共享的谱图 embedding；生物网络不直接替代谱学身份，而是在不破坏 near 异构体判别的条件下塑造可迁移表示。

## 7. 2026-08-30 晚间封存补充：外部转移与上下文表示分叉

### 7.1 MoNA 身份隔离转移面板

已从本地 MoNA negative-mode 库构建不与当前 MassSpecGym 开发 HDF5 共享 IK14 的固定面板：

- 1,301 个 query；
- 697 个 query 身份、722 个候选身份；
- 266 个分子式；
- 4,070 个候选分子、17,332 张候选谱；
- 同分子式、20 ppm precursor window、每分子 max-over-reference、并列对正例不利；
- 官方 DreaMS 本地基线 Recall@1 = 81.245%，MRR = 89.650%，共 244 个错误。

该面板可以证明“相对当前开发 HDF5 的身份隔离转移”，但不能证明 MoNA 从未进入 DreaMS 预训练。三种子最终模型必须预先冻结，外部面板只运行一次；禁止在 MoNA 上选种子、阈值或 adapter 强度。

### 7.2 两类 embedding 不能混叫

**B1：通用共享 embedding。** 生物关系只在训练期作为受限辅助监督。推理时 query/reference 都只输入干净谱图并共享一个编码器；输出仍是候选无关的通用 embedding。该分支用 MoNA 身份隔离面板验证。

**B2：样本上下文候选 embedding。** 推理时允许输入同一样本已观测高置信种子和类型化反应边，得到：

\[
z_c^{ctx}=\operatorname{norm}(z_c+\alpha_c\,\Delta_c).
\]

这是候选特异、样本特异的表示层，不应冒充通用 DreaMS embedding。无上下文、上下文冲突或门控为零时必须 bit-for-bit 回退到 `z_c`。反应邻居只作为 typed context message，永远不是 same-identity positive。

### 7.3 上下文证据张量与因子图当前结果

已用 identity-noop 过滤后的 Rhea 路径重建无泄漏张量，并把 truth/phenotype 完全拆出输入：

- MTBLS13729：204 条路径压缩为 130 个独立依赖边，去重 36.3%；
- MTBLS1905：38 条路径压缩为 30 个独立依赖边，去重 21.1%；
- complete path 按观测种子折叠；incomplete path 按 missing co-substrate signature 折叠；
- missing relation 明确为 unknown，不作为负证据；方向冲突单独编码。

固定、保守、显式 unknown 的 max-product 因子图在两个已消费开发集均为 0 修正/0 新增。这证明安全回退已实现，也说明固定规则传播没有可发布增益。下一步只允许训练“何时相信上下文”的门控 adapter，不再事后手调网络权重。

### 7.4 新增运行链

通用共享 embedding：

1. `sbatch tasks/run_bioaware_embedding_preflight.sbatch`
2. pilot 通过后运行五折；
3. 五折聚合过门后才进行三种子全开发重训；
4. `sbatch tasks/run_mona_identity_disjoint_transfer_preparation.sbatch`
5. 最终模型冻结后仅一次运行 `sbatch tasks/run_mona_bioaware_embedding_transfer.sbatch`。

样本上下文候选 embedding 的已消费机制回放：

1. `sbatch tasks/run_bioaware_context_development.sbatch`
2. 只回答上下文是否可学习及是否精确回退；
3. 无论结果多高，21/36 query 均不得作为外部 SOTA 证据；
4. 若公式隔离方向为正，再锁新的样本级外部数据集与 target-decoy/FDR 评估。
