# DreaMS 定向噪声微调复盘评委会与下一轮决策

日期：2026-09-05  
状态：正式复盘第二次修订；冻结历史证据，撤回候选残差蒸馏方案，尚未授权新的大规模训练

## 摘要

本项目已经证明两件不同的事实。第一，候选条件化的峰级动作具有真实且可重复的即时纠错能力；第二，直接用这些动作训练共享 query/reference encoder，可以获得稳定但较小的 clean retrieval 提升。二者之间仍缺少高效的信息传递机制。

当前最可靠的共享 noise-embedding 结果仍是 E4-A high-LR multifold：5 个 formula folds × 3 个 seeds 的 Recall@1 平均提升 `+0.6362 pp`，MRR 平均提升 `+0.4246 pp`，near Recall@1 平均提升 `+0.5300 pp`，15/15 次运行方向为正。最新 E4-PMT 的 fold-0 `clean_duplicate` 达到 Recall@1 `0.9404018`，相对匹配 official DreaMS 为 `+0.7429 pp`；但它仅是一折一 seed，尚不能替代 E4-A multifold 作为全局最强证据。

E4-PMT 同时给出否定性因果结果：显式 paired-margin treatment 虽使 held clean full-list margin 相对 matched-random 显著增加约 `0.00037–0.00038`，Top-1 却低 `0.0338 pp`，并新增 2 个错误。因此，“把 target-control 优势压成单个 hardest-negative margin floor”不是有效的传递算子。

下一轮撤回“缓存逐候选残差再让 clean student 拟合”的方案，因为它仍属于蒸馏，会再次引入教师压缩与传递损耗。最高性价比路线应回到 E4 已验证的直接共享编码器微调：成熟动作只负责构造真实峰级训练视图和定位具体困难候选；当前共享 encoder 同时编码 clean、target、正参考与当前困难负候选，并直接优化身份排序。matched-random 只作为独立等预算实验臂，不再与 target 一起接受对称正例损失。训练预算集中于错误和临界边界，harmful/uncertain target 不进入正向增强，其导致的候选切换转成 clean-query 安全边界。目标是减少“动作容量到 clean embedding”的传递损耗，同时严格控制新增错误。

## 1. 唯一科学问题

最终目标不是让被修改的 action spectrum 排名更好，也不是训练后处理器，而是：

> 训练一个推理时只接收原始 clean spectrum、query/reference 完全共享的 DreaMS encoder，使其在身份隔离和分子式隔离条件下，对完整候选分子图具有更好的排序几何。

因此，每个结果必须回答五个问题：

1. 输入是不是 clean spectrum；
2. query 与 reference 是否由同一个更新后的 encoder 编码；
3. 动作或教师是否使用了推理时不可见的正确身份、错误候选或 outcome；
4. 评价是否为 formula-held、完整候选 molecule retrieval；
5. 增益是相对 official、相对初始化，还是相对等预算因果对照。

任何缺少上述标签的 pp 数值不得与共享 embedding 性能相加。

## 2. 证据分级

| 等级 | 含义 | 可作何种声明 |
|---|---|---|
| A | 多折、多 seed、shared clean encoder、完整候选评价 | 可报告稳定模型权重增益 |
| B | 单开发折、shared clean encoder、严格配对对照 | 可报告因果方向，不可称最终性能 |
| C | 冻结 encoder 上 action/no-op oracle 或 teacher | 只能报告监督容量/headroom |
| D | adapter、reranker、P2b residual 或特权教师 | 不能称为新 embedding |
| X | schema、路径、split、baseline 或 gate 实现错误 | 科学结论作废，只保留事故证据 |

## 3. 全路线逐阶段账本

### 3.1 动作发现与早期策略

| 阶段/提交 | 做法与结果 | 严格结论 | 处置 |
|---|---|---|---|
| G5–G7 | 均匀随机删峰、加峰、m/z 抖动；旧协议下 0.8216/0.8354/0.8365，低于匹配 DreaMS 0.8676 | 无条件破坏不能提供纠错方向 | 永久停止作主策略；只保留 matched-random 对照 |
| v2 M0 | 从真实 identity-adduct 重复谱估计 conditional/shared/unique 峰 | 真实采集差异可校准剂量，不等于噪声标签 | 保留经验分布 |
| v2 causal/specificity | 55,892 变体；target 删除与匹配随机在错误和正确 query 上效应相近；specificity gates 全失败 | 整组 conditional deletion 没有错误特异性 | 永久停止整组删除 |
| v2 M1 | 1,231 complete-control queries；target 与随机 accuracy 均 0.5589；错误 query target 修正 22，随机期望 26.33 | target 不优于等剂量随机 | 不训练 |
| G1 | 建立 candidate-gradient 路径，但早期固定策略未达训练门 | 梯度可生成候选动作，不是最终训练标签 | 进入正交矩阵 |
| S1A | 单峰正交矩阵，首次同时控制峰角色、剂量与随机路径 | 定向选择优于只看删除剂量 | 保留矩阵 |
| S1B | no-op-aware headroom `+1.62 pp` | 动作空间有容量，非模型增益 | 仅作容量证据 |
| S1C | 394,752 query-action rows；union headroom `+1.709 pp` | 扩剂量增加覆盖也增加风险 | 保留低中剂量 |
| S2/redecision | 顺序动作与 candidate switch；通过 policy-design 容量门 | 多步路径优于孤立动作，但仍是 outcome-aware | 保留轨迹，不直接训练 selector |
| S3A | 48 cells、278,001 配对 action-step；union 799 errors，即 `+3.346 pp` | candidate-gradient 3–6 和 role-confounder 1–5 成为成熟 N 原语 | 冻结成熟 N；禁用 role_shared |
| A4 exact | 206,288 peak actions、825,152 variants；oracle 达 920 errors，即 `+3.853 pp`；75–100% 删除新增错误激增 | 峰级空间大，但高剂量 action 不适合训练 | 保留 action/harm ledger；禁用高剂量和 shared-only |

### 3.2 教师、峰 token 与错误的“高 pp”解释

| 阶段/提交 | 结果 | 没有证明什么 | 处置 |
|---|---|---|---|
| A4 nonlinear teacher | corrective ROC-AUC 0.849、harmful AUC 0.853；40% coverage 182/39 | 没有产生新 shared embedding；对简单策略无显著增量 | 保留风险标签，停止堆 selector |
| A4-B adapter | 线性约 `+0.760 pp`，非线性约 `+1.060 pp`；teacher rescue 542 中仅恢复 80，并新增 45 | 冻结 query adapter，不是统一 query/reference space | 归档，不作主模型 |
| C1 support-disjoint teacher | 80,250 examples；instance delta `+2.475 pp`；2382/396；near `+2.344 pp` | 教师使用正确身份支持，不能作为 deployable clean encoder | 仅保留真实重复谱和协议 |
| C2-A | token expert 相对 global control 的 CI 跨零 | contextual token 有信息，但简单方向头未证实增量 | token 作为未来输入资产 |
| C2-B/C2-C | 首次 86/0 来自实现错误；纠正后约 `-0.0209 pp`；P2b residual 为零或负 | 不能用 reranker 失败否定 noise | 永久移出 noise 主线 |
| D0/D1/D1b | D0 冻结正确合同；D1b 为 query-only clean adapter，14 seeds 平均约 `+0.067 pp` | 不是噪声训练，也不是共享 reference geometry | 只作 clean control |
| F1 v1–v4 | 外层从很小正值到 v4 full graph `-0.0844 pp` | 特权 P teacher 的 challenge 修正不能转成自然检索 | 停止 prototype/absolute P 蒸馏 |

### 3.3 E1–E4：从动作校准到共享 encoder

| 阶段/提交 | 结果 | 结论与教训 |
|---|---|---|
| E1 | 3,412 groups、2,588 identities、1.21M matched peak pairs | 是真实采集变化分布，不是性能 |
| E2 manifest/scan/sensitivity | 44 cells；28 corrective；18 初筛、14 sensitivity-pass | 冻结 encoder 上存在动作特异性；多重性校正必要 |
| E3 | 27,735 actions、1,104 identities、408 formulas；16 compatible/0 conflict | embedding tangent 相容不等于 117M 参数梯度或训练可实现性 |
| E4-M0 | 17,310 action targets | 绝对 action target 会压缩多动作、步数与候选结构 |
| E4-M1/M1b | shared/pooling adapter 未把动作容量转进 clean retrieval | clean loss 约大于 action loss两个数量级，pooling 不能重构峰间交互 |
| R0 | 忠实恢复 36,934 N action rows、matched controls 与 path | 必须永久保留的动作执行资产 |
| R1 | 882 recoverable queries，仅 462 identities、248 formulas | 动作容量高度集中 |
| R2 | action-view train/held 约 99%；clean transfer train 18.6%、held formula 1.17%；两个 LR 均为负 | 记住 action view 不等于 clean-input 泛化；停止 absolute embedding imitation |

### 3.4 E4-A：当前最可靠的共享 noise embedding

| 提交 | 结果 | 结论 |
|---|---|---|
| 2325996 三臂 curriculum | all/views4 `+0.44 pp`、errors/views4 `+0.44 pp`、errors/views8 `+0.49 pp` | 多步 curriculum 可直接改善 shared encoder |
| 2326069 optimizer scan | 最佳单折约 `+0.59 pp`；提高 LR/clip 只增加约 0.05 pp，并接近 preservation 边界 | LR 不是主瓶颈 |
| 2326084，5 folds × 3 seeds | 平均 Recall@1 `+0.6362 pp`；MRR `+0.4246 pp`；near `+0.5300 pp`；559/104 累计 corrected/introduced；15/15 CI lower bound >0 | 当前 A 级模型权重证据 |
| E4 single-family | candidate、confounder、acquisition 单独均未通过 multifold 门 | 单一 family 覆盖不足 |

E4-A 的局限同样必须保留：target action、clean ranking 和普通 continuation 同时变化，缺少等预算 clean/matched-random 对照；训练只解冻最后 1/7 Transformer block与 projection head；28,509 action rows 没有无放回全覆盖；训练负候选来自旧 geometry 截断。

### 3.5 P/N 扩展与 E5–E13

| 阶段/提交 | 结果 | 裁决 |
|---|---|---|
| P/N headroom | N recoverable 882；严格 P recoverable 173；union 922，`+3.8616 pp` | 容量仍非模型；正证据是主要缺口 |
| positive intensity matrix | consensus projection@0.75：`+1.1267 pp`，456/187 | 有效但全局风险过高 |
| positive transfer matrix | recurrent union mix@0.50：`+1.1183 pp`，294/27 | 更安全，但仅约 34.7% query 可执行 |
| E5/E5B/E5C | N-only `+0.5403 pp`；intensity 最好点估计 `+0.5740 pp`，但相对 N-only CI 跨零或 preservation 失败 | 停止固定 guided/safety weight 扫描 |
| E6 | fixed 约 `+0.4052 pp`；outcome-mined held 约 14/11 | 事后挑成功 action 引入选择偏倚；停用 |
| E7 | recurrent weight 0.025–0.1 相对 control 仅多 1 个；0.2 多 3 个但 preservation 失败 | 固定全局 P 权重无独立价值 |
| E8 | symmetric/shared `+0.5740 pp`；stopgrad 与 official-action target 无增量；official references 低 `0.2195 pp`；terminal-only 更差 | 保留 shared/current refs/curriculum；停用 stopgrad、official target/reference、terminal-only |
| E9 | online vs frozen 10/9，约 `+0.0119 pp`，CI 跨零 | action staleness 不是主瓶颈；停止在线重挖 action path |
| E9B | 成熟 N 在 E8 geometry 的额外 oracle `+0.6753 pp`，总 `+1.2494 pp` | 同 geometry 的 N 上界，不是预测 |
| E10 | fixed recurrent 相对 E8 `+0.8779 pp`，59/7；no-op union 总 `+3.3767 pp` | P 有即时效应，但需要条件路由 |
| E10B | 19 P cells 中 13 pass；union 总 `+3.799 pp` | 扩展仍有重复覆盖 |
| E11 | reference diversity 新增 39 unique；总约 `+4.457 pp`；maxmin6 风险恶化至 55/20 | 多参考并非单调更好 |
| E12A/B | relaxed recurrence fixed `+1.7896 pp`，137/31；union 只新增 28，约 `+0.4727 pp` | 固定效应增大，但独立覆盖小 |
| E13 | control `+0.5740 pp`；guided `+0.557–0.608 pp`，均未显著优于 control且 preservation <0.995 | 不再全局叠加 P |

### 3.6 E14/E15、梯度审计和 L 路线

| 阶段/提交 | 结果 | 处置 |
|---|---|---|
| E14 | selected-no-margin/delta 相对 continuation 约 `+0.2127 pp`，risk branch 反向 | 每 query 单动作、global filter、risk 混入 corrective、epoch 回收、4-sample calibration 等实现错误；trainer 作废 |
| E15 M0/M1 | 多动作与 replay 账本有价值 | status/schema/gate/overwrite 事故属于工程失败 |
| E15 M2 | 3,433 corrective、2,479 harmful；小 panel overfit 27/0 | 证明局部容量，不证明 holdout transfer |
| E15 M3 | 数据缩成 281 actions/198 queries/133 identities；held 0/1，`-0.391 pp` | split 删除候选、假 control、无效 risk、已失败仍读 held；trainer 作废 |
| E4-B0/B1/B2 | no-update 梯度分解与 surgery screen | 发现跨 formula 梯度共识不足；样本硬门、torch API 和实现事故使部分早期运行无效；仅保留最终审计数据 |
| L0 | 36,934 actions；candidate-gradient advantage 随 step 3–6 增长；role-confounder step5 17/0 | 动作在当前 geometry 中仍有效 |
| L1 | clean-token formula-OOF positive-action AUPRC 0.7065，高于 family-only 0.6098 和 permutation 0.6142 | clean input 有非零可识别性，但不足以证明训练 influence |
| L2 | role-confounder 被阈值清空；targeted vs random 4/3；每轮 clipping=1 | 静态阈值、family collapse 和损失实现失败，不能否定完整动作空间 |
| CPG0 | 计划保存逐候选 residual，但单测、OOM、门逻辑与超长重放失败 | 科学动机保留；现有制品不得冒充完成教师 |

### 3.7 最近两次严格因果实验

#### Dynamic-direct Phase A，jobs 2331284/2331352

四臂均从 mature E4 fold 0 初始化，训练 4 epochs：

| Arm | Recall@1 | vs official | vs mature E4 | 结论 |
|---|---:|---:|---:|---|
| clean continuation | 0.9398953 | +0.6922 pp | +0.1182 pp | continuation 效应 |
| matched random | 0.9398953 | +0.6922 pp | +0.1182 pp | 与 clean 并列 |
| static target | 0.9398953 | +0.6922 pp | +0.1182 pp | target 没有独立增量 |
| dynamic N+P | 0.9397265 | +0.6753 pp | +0.1013 pp | 低于三对照 `0.0169 pp` |

Dynamic vs matched-random formula CI 为 `[-0.0945,+0.0598] pp`，Holm p=1；`pass_to_second_seed=false`。该结果否定 frozen dynamic weighting + direct-training kernel，不否定 action capacity。

#### E4-PMT，job 2331467

E4-PMT 路由全部 28,509 mature N actions：15,520 corrective、9,034 robustness-only、2,891 harmful、1,064 uncertain；harmful/uncertain target corrective weight 精确为零。

| Arm | Recall@1 | vs official | vs mature E4 | 相对初始化 corrected/introduced |
|---|---:|---:|---:|---:|
| clean duplicate | **0.9404018** | **+0.7429 pp** | **+0.1688 pp** | 13/3 |
| matched random | 0.9402330 | +0.7260 pp | +0.1520 pp | 13/4 |
| PMT alpha=.25 | 0.9398953 | +0.6922 pp | +0.1182 pp | 12/5 |
| PMT alpha=.50 | 0.9398953 | +0.6922 pp | +0.1182 pp | 12/5 |

PMT 相对 matched-random 为 `-0.0338 pp`（0/2），相对 clean duplicate 为 `-0.0507 pp`（0/3）。但是 PMT 的 mean full-candidate margin 相对 matched-random 增加 `0.000366–0.000384`，formula CI 下界严格为正。故它学到了微弱连续位移，却没有学到有用的 Top-1 边界迁移。

## 4. 当前模型排名的严格表述

1. **最强多折证据**：E4-A high-LR multifold，平均 `+0.6362 pp Recall@1`。这是目前可以写进论文主结果表的开发 OOF shared-embedding 成绩。
2. **最强单折点估计**：E4-PMT 的 `clean_duplicate`，fold-0 Recall@1 `0.9404018`，相对 matching official `+0.7429 pp`。它尚需在未用于选择的 folds/seeds 复现。
3. **不能晋级的模型**：dynamic N+P 和两个 PMT treatment；它们相对各自因果对照没有正增量。
4. **不是模型的高数字**：`3.346–4.93 pp` 均为 outcome-aware action/no-op 容量，不能列入模型排行榜。

## 5. 为什么“把每种正收益方法依次训练一遍”不能直接相加

各臂相对 official 的增益包含同一个 mature E4 起点和大量相同 continuation 收益。它们不是正交增量：

- clean duplicate 与 matched random 只差 1 个净 query；
- dynamic Phase A 的 clean/random/static 三臂 Recall@1 完全相同；
- PMT 两个 alpha 完全相同，且低于 clean/random；
- E5/E7/E13 的 guided 增益基本被各自 no-guided control 解释；
- 同一 query 的动作会重叠修正，后一步还可能重新引入前一步错误。

因此，相加的合法条件只能是：从同一前置 checkpoint 出发，第二阶段相对等预算 continuation control 具有严格正的 formula-cluster paired CI，并报告 unique corrected、new introduced 与 candidate switch。未满足时串行训练只是增加优化步数，不能称为方法叠加。

## 6. 指标体系与 DreaMS AUC=0.85 的边界

### 6.1 下一轮必须同时报告

- DreaMS 原文同协议 NIST20 10-ppm spectrum-pair pooled AUROC；
- 同一 NIST20 pair ledger 上的 AUPRC、正负相似度分离及 bootstrap CI；
- Recall@1、@2、@3、@5、@10、@20；
- MRR；
- pooled/micro candidate AUROC 与 average precision；
- macro query AUROC：每个 query 内正分子对所有负分子，然后对 query 取宏平均；
- macro query average precision；
- median rank、mean rank、normalized rank percentile；
- 自动注释的 coverage at controlled error/FDR（1%、5%、10%）及其置信区间；
- Top-1 置信度的校准误差、Brier score 与 selective risk-coverage 曲线；
- positive-vs-best-negative margin 与 top-2 gap；
- near/MCES、positive-deficit、negative-excess、cross-condition 分层；
- corrected、introduced、risk net、wrong-to-different-wrong；
- formula 和 identity cluster bootstrap CI；
- 相对 official、相对初始化、相对 clean continuation、相对 matched-random 四套配对差异。

### 6.2 Top-10/Top-20 的实际信息量

现有 E8 fold-0 `held_per_query` 可直接重算：

| 指标 | official | E8 | delta |
|---|---:|---:|---:|
| Recall@1 | 0.932973 | 0.938713 | +0.5740 pp |
| Recall@5 | 0.997130 | 0.997299 | +0.0169 pp |
| Recall@10 | 0.999662 | 0.999662 | 0 |
| Recall@20 | 1.000000 | 1.000000 | 0 |

因此本候选图上的 Top-10/20 已接近或达到天花板，必须报告，但不能作为主要优化目标。对于“实验人员拿一个工具是否能获得可信注释”，最直接的两个终点是：单一答案场景的 Top-1 accuracy，以及允许模型拒答时在固定 1%/5% FDR 下可接受的注释覆盖率。MRR/Recall@5 描述人工复核短名单成本；pairwise AUROC 描述相似度判别，但不直接等于一个可接受注释的概率。

### 6.3 不能直接比较的两个 AUC

DreaMS 论文的 `AUC≈0.85` 是在 NIST20 上抽样约 750,000 个谱对，将同 2D InChIKey 谱对作为正类、10 ppm 前体质量邻近的不同分子作为负类，计算全局二分类 AUROC。我们的主任务是 MassSpecGym formula-held query 对完整候选分子的检索，现有 P3 报告的是 macro query AUC。两者的数据、采样单位、负例分布和聚合方式均不同。

所以当前不能说 clean duplicate 已超过论文的 0.85，也不能用本任务约 0.92 的 macro-query AUC去宣称超过。DreaMS 源码中的原始验证入口读取 `data/NIST20/nist20_clean_spec_entropy_[M+H]+_retrieval.pkl` 和 `data/NIST20/nist20_clean_spec_entropy_[M+H]+_50k_pairs_retrieval.pkl`；服务器是否具备合法授权文件必须先做只读确认，不能凭文件名推测内容。实际 pair 数、过滤条件、正负比例和文件 SHA 必须物化后再冻结。

从下一次候选 checkpoint 开始，必须冻结三个 evaluator：

1. 本任务 evaluator：同一 candidate graph 的 macro query AUC；
2. 论文复现 evaluator：原始 NIST20 10-ppm pair ledger 的 pooled/global AUROC，即论文约 0.85 所对应的指标；
3. 应用 evaluator：完整候选检索上的 Top-k/MRR 与固定 FDR 下的 coverage。

每个进入模型比较的 checkpoint 都必须与 official DreaMS 和该次训练初始化在三套 evaluator 上成对报告。缺少合法 NIST20 资产时，该 checkpoint 只能标记为“内部检索评估完成、论文 AUC 未完成”，不能进入最终模型排序。只有在对应 evaluator 上与同数据、同 pair ledger 的 official checkpoint 配对比较，才能说超过。

## 7. 最近实验给出的最重要机制教训

1. **普通 continuation 是真实强基线。** 任何新噪声臂不超过 clean duplicate，就没有证明动作语义贡献。
2. **动作有效不等于动作可传递。** action-view 可纠错、teacher AUC 高、oracle union 大，都不能替代 clean-input shared encoder 结果。
3. **标量 advantage 过度压缩。** PMT 提高平均 margin 却降低 Top-1，说明需要保留“哪个候选被推远”的向量结构。
4. **公共 rank loss 会淹没差分。** PMT 同时把 target/control 当正样本；显式 preference loss仅约 `1e-4`，而二者平均 advantage 已远高于 0.01 hinge，差分分支大多失活。
5. **no-op 是动作。** harmful、uncertain、不可识别动作必须零 corrective weight；不能用低但非零权重代替退出。
6. **P 与 N 语义不同。** N matched-random 是鲁棒性对照；P wrong-direction 是反事实错误方向。未经独立放行不得混损失。
7. **完整候选边界必须进入训练。** 只修正旧 hardest negative 会被新候选顶替。
8. **多动作不能在 sampler 中坍缩。** 必须保留 query-action-candidate 三层索引与无放回 exposure。
9. **新增错误必须是一级终点。** consensus projection 的 456/187 说明只看 corrected 会系统性选择危险策略。
10. **优化器排在监督之后。** LR、层数和 clip 扫描只带来约 0.05 pp 局部变化，无法修复目标错配。

## 8. 下一轮最高性价比方案：直接微调，不做蒸馏

名称：E4 Direct Error-Boundary Fine-tuning（E4-DEB）。它不是 teacher/student、logit distillation、embedding imitation 或 cached residual regression。成熟动作只做两件事：生成峰级训练视图；在训练折中指出应被重点学习的真实候选边界。模型始终使用一个当前共享 encoder 端到端编码 query 与 reference，并由真实 identity 标签直接优化。

### 8.1 为什么撤回上一版 CPD

上一版虽保留逐候选残差，但仍要求 clean student 拟合由 target/control 产生的外部数值目标，本质上没有消除“动作到学生”的蒸馏接口。它还没有回答动作视图本身能否通过共享参数、正确采样和强边界损失直接改善 clean retrieval。因此 CPD 不进入实现。

E4 已证明直接微调能够产生稳定增益；当前应修 E4 的信息利用率，而不是替换成新的教师架构。

### 8.2 动作如何进入训练

仅在 outer-train 内，根据完整当前候选图把成熟 N action 分为 corrective、robustness-only、harmful 与 uncertain。该划分是监督式困难样本挖掘，不产生要被学生拟合的 teacher score。

- corrective：target action 相对 matched-random 对具体错误候选产生可重复改善；其峰级 target view进入直接训练；
- robustness-only：不作为纠错动作，只能进入独立鲁棒性对照；
- harmful/uncertain：target view 的正向权重精确为零；
- harmful action 导致的“正确候选下降/新错误候选上升”被转换为原始 clean query 的安全 hard-negative pair，而不是把 harmful action 反号训练；
- no-op 始终允许；没有安全 corrective action 的 query 不强制接受噪声。

同一 query 的多个安全动作不压成一个分数。动作分别保留，但按 `query -> identity -> formula -> family -> cell` 均衡；在任一动作重复之前，所有合格动作至少曝光一次。

### 8.3 直接损失：动作负责造样本，身份标签负责训练

对 clean query `q`、target action view `q_t`、真实同身份参考集合 `P` 和当前候选分子 `C`，由同一 `E_theta` 计算 molecule-level 聚合分数。对 action 修复或暴露的每个错误候选 `c_j`，直接优化：

`L_clean_boundary(j) = softplus((gamma - [s_theta(q,P)-s_theta(q,c_j)]) / tau)`

`L_target_boundary(j) = softplus((gamma - [s_theta(q_t,P)-s_theta(q_t,c_j)]) / tau)`

两项都是真实身份监督；没有 `teacher_advantage`、没有 target embedding target、没有 clean residual target，也没有会在 0.01 处失活的 `target-control` hinge。target action 的作用是让共享参数在机制相关峰视图上学习同一个候选边界，clean 项则保证最终部署输入本身被直接优化。

必须同时保留：

1. 当前 Top-k 错误候选，而非仅旧 hardest negative；
2. action 前后的 candidate switch；
3. 少量全候选 listwise 项，防止只推远一个错误后由另一个候选顶替；
4. protected-correct 与历史 introduced query 的 clean margin floor；
5. query 与 reference 两端均求梯度，不冻结 reference geometry。

matched-random control 不与 target 在 treatment 内接受对称正例损失。它只形成一个独立训练臂：用完全相同的 query、候选边界、剂量、曝光和预算，把 target payload 换成预先冻结的 matched-random payload。这样 `targeted - matched_random` 才测量峰选择语义，而不会再次被公共 `0.5*(target_rank+control_rank)` 淹没。

### 8.4 训练预算必须集中到会改变决策的边界

此前 PMT 把大量算力花在已经正确且离边界较远的 query 上。E4-DEB 的训练单位改为 active candidate boundary：

- official/mature 错误 query；
- near-boundary 正确 query；
- 历史 introduced/candidate-switch query；
- 远离边界的正确 query 只作为低频全局 preservation sentinel。

每个 microbatch 按 active boundary 数量归一，而不是按 action row 数量平均。corrective、clean-safety 与 global-preservation 三支分别记录未加权梯度范数；正式权重由不少于 32 个 formula-stratified microbatches 的中位梯度范数校准，使 corrective 梯度在全局裁剪之前达到预注册的非微小占比。不得再由前 4 个样本估计，也不得通过无限提高 loss weight 把安全梯度压没。

全局 norm clipping 会等比例缩放已经合成的梯度，本身不是定向信号消失的首因；真正要控制的是裁剪前各分支的相对范数和方向。正式报告必须包含分支范数、余弦、裁剪比例和裁剪后有效步长。

### 8.5 最小而充分的开发实验

所有臂从同一个 fold-aligned `clean_duplicate` checkpoint 出发；第一轮只用成熟 N，避免 P 再次以数量压过 N：

1. `C0 clean_continuation`：等预算 clean 直接训练；
2. `C1 matched_random_direct`：同一 active boundary 和训练预算，action payload 换成 matched-random；
3. `Tcg candidate_gradient_direct`：只用 candidate-gradient 3–6；
4. `Trc role_confounder_direct`：只用 role-confounder 1–5；
5. `Tnp mature_N_combined_direct`：两种 N family-balanced 直接训练。

五臂不是超参数搜索，而是一次最小析因：确定两个成熟 N family 各自贡献以及组合是否相加。它们共享 seed、query order、每个 query 的正负候选、optimizer steps、可训练层、LR 与总 action exposure。不得用 raw action 行数让某一 family 获得更多梯度预算。

若 `Tnp` 严格超过 C0/C1，且至少一个单 family 臂解释该增量，则复制第二 seed 和其他 formula folds。之后才从通过的 N checkpoint 出发，对低风险 P-transfer 做 `N/no-P × P-transfer/matched-P-control` 的 2×2 实验；P-intensity 因 456/187 的高风险最后审查。

### 8.6 放行标准与传递效率

主门必须全部通过：

1. `Tnp-C1` 与 `Tnp-C0` 的 Recall@1 formula-cluster CI lower bound > 0；
2. MRR、macro query AUROC、micro candidate AUROC/AUPRC，以及冻结 MassSpecGym 10-ppm
   spectrum-pair pooled AUROC 对两个对照均不下降；NIST20 仅在原始 pair ledger
   可验证时作为外部补充，不是本任务晋级门；
3. corrected > introduced 且 `corrected - 2*introduced > 0`；
4. near Recall@1、Recall@5 与固定 FDR coverage 不下降；
5. preservation mean ≥0.995；
6. 每个新增错误均可追溯至 action、原候选、candidate switch、family/cell 与曝光次数；
7. treatment 与两个对照的训练步数、有效 batch、候选边界数和裁剪后累计更新范数匹配。

每次还要报告同一 held geometry 下的传递效率：

`transfer_efficiency = (Tnp net gain - C1 net gain) / held target-vs-matched-random safe-action headroom`

分子与分母必须来自同一 checkpoint、同一 held fold、同一候选图；历史 `3.85/4.93 pp` 不可直接作当前分母。按历史数字粗略比较，E4 只保留约 13%–17% 的动作容量，即缺口约 83%–87%，接近但不能精确写成 90%。E4-DEB 的直接目标就是提高这个比率，而不仅仅追求相对 official 的绝对正值。

### 8.7 成功与失败如何解释

- Tcg/Trc 均超过 C1：两种峰机制都可直接转入 clean embedding，可继续组合；
- 单 family 超过而组合不超过：不是动作弱，而是梯度/采样冲突，保留有效 family，停止盲目相加；
- Tnp 超过 C1 但不超过 C0：主要是普通 supervised continuation，不能归因给噪声；
- 所有 target 臂不超过 C1：现有动作只改善被改写的谱，不提供额外可泛化训练信息；此时才需要扩展动作机制，而不是调 LR；
- Recall@1 上升但同一冻结任务上的 MRR、macro/micro AUROC/AUPRC、10-ppm
  spectrum-pair AUROC 或 FDR coverage 下降：不能晋级为更好的通用 embedding。

## 9. 工程硬约束

任何正式作业之前，单个短 preflight 必须验证：输入路径与 SHA、schema、fold、rank replay、candidate completeness、target/control 同 membership、query-action multiplicity、无放回 exposure、三臂 batch/step 等价、all-true 与 expected-false contract、唯一运行目录和原子发布。

Slurm 作业必须包含 `#SBATCH --gpus=1`，不显式申请内存；登录节点只提交，不运行模型。失败运行不得占用正式完成目录。

### 9.1 已发生的工程事故账本

下列运行没有资格贡献科学结论，但每一类都必须转化为提交前测试。重复出现的相同错误合并为一类；这不是省略实验臂，而是避免把同一工程故障伪装成多个科学结果。

| 事故类别 | 已出现的具体表现 | 对结果的危害 | 永久修复要求 |
|---|---|---|---|
| 调度合同错误 | sbatch 未写 `--gpus=1`；N26 显式内存超限 | 作业未开始，浪费排队时间 | 静态解析 sbatch；必须一张 GPU、不得显式内存 |
| 路径与工作目录错误 | 从 `~` 提交相对路径失败；模块 import 失败；所需 held 文件不存在 | 运行前或运行中断 | 所有输入在 preflight 中解析为绝对路径并检查；不得临时猜路径 |
| 输出生命周期错误 | 完成/失败目录拒绝覆盖，重跑仍指向旧目录 | 修复代码无法执行 | `${SLURM_JOB_ID}` 唯一 staging；成功后原子发布；失败目录不占正式路径 |
| HDF5 索引错误 | 非递增 fancy index | 数据读取直接失败 | 排序读取后逆置恢复原顺序的数值单测 |
| schema/status 假设错误 | 缺列、状态名不符、`Namespace` 缺参数、NumPy `bool_` 无法 JSON 化 | 长作业在末端失败 | 对真实制品运行 schema fixture；JSON 递归转原生类型 |
| baseline/replay 误差 | baseline rank mismatch；2/882 replay mismatch；zero-init 改变 3 个 rank | treatment 与基线不可比较 | 同执行器 fresh-forward；显式浮点容差；mismatch 必须逐条物化 |
| 无效 gate | 所有布尔门均真却仍抛错；把经验样本阈值当成科学失败 | 错误阻断有效数据 | gate 逻辑做真/假双向单测；容量阈值与科学效应门分离 |
| 事后门槛与样本耗损 | 固定要求 24/16 formulas、每源 8 identities、每源 6 multi-action errors，但实际支持不足 | 自造小样本或迫使改门槛 | 先物化总体支持，再冻结可满足的分层设计；不在报错后追逐阈值 |
| 动作坍缩 | 多动作压成每 query 一个最优 action；全局 safe filter 删除条件动作 | 丢失高容量动作空间 | 保存完整多动作及 no-op；训练前报告每 query multiplicity |
| 损失语义混淆 | harmful/risk 与 corrective 共用正向损失；target/control 对称 rank 淹没差分 | 有害动作得到正梯度，定向信号被稀释 | corrective、risk、preservation、counterfactual delta 四类损失分离 |
| 暴露失真 | epoch 内循环回收少量 guided examples；30 万 ledger 动作并未完整训练暴露 | 少数样本被过采样，多数动作未被看到 | 无放回 sampler；报告每动作最大/中位曝光及零曝光比例 |
| 梯度校准失真 | 只取每支前 4 个样本估计梯度；未按 identity/formula 分层 | 权重由偶然样本决定 | 至少 32 个分层 microbatches、128 个 identity-action observations |
| split/candidate 破坏 | 为 identity holdout 删除参考后，query 失去所有负分子 | 评估任务被改变，样本大量丢失 | holdout 约束训练监督，不删除完整候选谱库；候选完整性门前置 |
| 资源与批处理错误 | retry 一次性编码导致 32 GB OOM | 长作业中断 | 自适应小 batch、逐块释放、OOM fixture；不以扩大 GPU 掩盖 |
| API/单测错误 | 使用不存在的 `torch.flatnonzero`；残差测试构造本身对称导致断言失败 | preflight 本应拦截却自身不可信 | 在服务器环境运行最小数值测试；fixture 必须先证明可区分 |
| 结果偷换 | action/oracle、adapter、reranker、P2b residual 被口头称为新 embedding | 夸大结论，错误指导下一轮 | 每份报告强制标注 A/B/C/D/X 证据等级及 inference contract |

这些事故说明：此前的低收益不全是路线失败，也不全是工程失败。已经有严格对照证明 PMT 算子本身无 Top-1 增量；也有多次工程失真使其他旧实验不能裁决路线。下一轮只能修复已被定位的候选级信息瓶颈，不能笼统宣称“修完工程就必然获得 3–5 pp”。

## 10. 最终裁决

当前不应把所有正值臂串行训练，也不应继续扫描 action dose、LR、蒸馏器或新 selector。应先以 `clean_duplicate` 为共同初始化，运行 E4-DEB 五臂直接微调：clean、matched-random、candidate-gradient、role-confounder、balanced combined N。它保留 E4 已有权重和端到端 shared encoder，只修复四个已定位的直接训练缺口：训练资源未集中到真实边界、harmful action 获得正监督、多动作曝光/家庭配比失真、当前 Top-k candidate switch 未进入损失。

论文表述只能是：E4-A 已获得稳定的 shared-embedding 提升；clean continuation 在一个 held formula fold 上进一步提高；成熟 N actions 的 paired scalar-margin transfer 未增加 Top-1，说明弱 hinge、对称 target/control ranking 与单标量继承无法保留动作信息。下一项实验改为 action-mined、identity-supervised 的直接共享编码器微调，不采用教师—学生蒸馏。任何更强声明必须等待 E4-DEB 多折结果、三套冻结 evaluator 和一次冻结 P3。

## 12. 2026-09-05 最终实现修订：逐候选直接边界训练

正式实现不再拟合 cached residual、embedding target 或 query-level scalar advantage。每个 query 保留完整的 `positive reference × negative molecule` 边矩阵；`target-control` 也保持同形矩阵，只用于逐边选择和连续加权。真实 molecular identity 始终是优化标签。

对每条当前困难候选边，执行三项直接更新：

1. clean 谱直接学习该真实候选边界；
2. target 峰动作谱直接学习同一身份边界；
3. 对逐候选 `target-control` 使用无死区 logistic preference；control 为 stop-gradient 方向对照，不能通过破坏 control 人为制造优势。

这不是蒸馏：不存在要被 clean encoder 拟合的教师 embedding、score 或 residual。逐边 action advantage 只决定把训练预算放在哪个真实身份边界上。非正 advantage 的边获得精确零 corrective 权重；所有 current Top-k negative molecules 每 epoch 用当前共享 encoder 刷新，动作造成的候选切换分子强制保留且每个负分子只出现一次。protected-clean 分支同样保护逐候选 margin，而不是只保护一个旧 hardest negative。

为降低一次开发作业成本，首轮由原计划五臂收敛为三臂：`clean_duplicate`、`matched_random`、`candidate_boundary`。treatment 同时完整保留 9 个成熟 N cells（candidate-gradient 3–6 与 role-confounder 1–5），所以本轮回答的是“新的逐候选传递算子能否超过两个等预算对照”，不再额外消耗两个单-family 训练臂。若通过，再用同一实现做 family attribution；若失败，不得以继续扫 LR 代替根因判断。

动作路由的 no-update 几何仍冻结在 mature E4；三臂模型则统一从 job 2331467 的 fold-0 最强 `clean_duplicate` checkpoint 初始化。这样既不重算和篡改动作证据，又继承当前最好权重。三臂曝光顺序哈希只包含 arm-invariant 的 query、identity、formula、cell 与 dose；峰路径和动态参考不进入等价性哈希。所有动作在任何重复前必须完成一次曝光。

训练前用相同的 32 个 formula strata 分别估计 corrective 与 safety 梯度范数；自动校准只允许保持或增强 corrective 分支，不能再次把动作信号缩小。正式结果必须报告裁剪前 head/backbone 范数、clip scale、有效 action-query 比例、逐 epoch candidate 刷新，以及完整候选 Recall@1/2/3/5/10/20、MRR、macro/micro AUROC/AUPRC、rank、near、corrected/introduced/risk-net。

NIST20 论文协议使用同一冻结 pair ledger 对 official 与候选 checkpoint 成对重建 pooled pairwise AUROC/AUPRC。若服务器缺少有授权的两个 NIST20 pickle，作业只生成 `unavailable` 制品，明确禁止拿内部 macro-query AUC 冒充论文约 0.85 的指标。

唯一正式作业为 `tasks/run_noise_final_e4_deb_phase_a.sbatch`；它申请一张 GPU、不显式申请内存、使用 job-id 唯一目录，并在加载 117M 模型前执行语法、静态合同和数值单测。首轮仍只能测量 N 动作约 3.35 pp 条件容量的传递效率；P 动作负责把总条件容量推近 4–5 pp，但在 N 算子没有显著超过两个对照前不得混入本轮，避免再次让高数量、高风险 P 掩盖 N 的因果贡献。
