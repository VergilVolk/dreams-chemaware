# 噪声微调路线：从 PPT 高潜力动作到共享 embedding 的完整根因审计

日期：2026-09-01  
版本：v3（逐阶段失败账本、瓶颈树与停止清单；取代同名文档 v1/v2 的不完整裁决）  
范围：只讨论直接改变共享 DreaMS embedding 的噪声微调；P2b、下游重排器和 ChemAware 不参与本裁决。

## 一、这次必须纠正的结论

上一版仍然把问题说浅了。它把“动作曝光不足、多动作/no-op 路由缺失、单向 transfer 不够强”列为最优先矛盾，但这些只是**在已经证明动作可被 clean encoder 学会之后**才应优化的工程项。现在最上游、尚未解决的是四个问题：

1. **因果归因没有闭合。** 已训练共享 embedding 的 `+0.635 pp` 是真实结果，但没有同训练步数、同 sampler、同候选集合的 clean-only 与 matched-random 学生对照；因此不能把全部增益严格归因于“定向峰级噪声”。
2. **动作可执行不等于 clean-input 可学习。** candidate-gradient、confounder、consensus projection 等动作使用了正确候选、错误候选或同身份参考谱这些训练期特权信息。冻结 encoder 上 action 能纠错，只证明反事实谱图有效，不证明原始 clean spectrum 足以让共享 encoder 跨 formula 学出这条规律。
3. **训练目标与最终检索图不一致。** 训练使用由官方 embedding 冻结选出的少量正负代表谱，最终评价却在完整候选分子/参考谱集合上重新排名；encoder 变化后，真正 hardest negative 会切换，而训练未必看见它。
4. **动作容量、动作转移和最终性能长期被混为一谈。** 3–5 pp 多为 outcome-aware action union/headroom；当前共享 encoder 的可靠收益仍是约 `+0.635 pp`。二者之间缺的不是一句“加大训练”，而是一条可检验的反事实监督到泛化表征的桥梁。

因此，路线没有被否定，但下一轮不能直接从“增加动作、提高曝光、加大学习率”开始。必须先回答：**定向动作是否比等剂量随机动作提供了可转移的额外梯度；哪些动作效果能跨 identity/formula 被 clean encoder 学会；如何在完整候选边界上表达这种梯度。**

## 二、已经被牢固验证、不能推倒重来的成果

### 2.1 真实错误图谱成立

23,876 个正式 query 中有 1,805 个官方 DreaMS Top-1 错误：

- positive-deficit only：1,242；
- positive-deficit + negative-excess：197；
- negative-excess only：188；
- comparative boundary：178。

至少 `1,439 / 1,805 = 79.72%` 的错误包含 positive deficit。另有 984 个错误命中共享主峰筛查、1,032 个命中 neutral-loss convergence、783 个可被至少一种传统谱学证据纠正。DreaMS 的核心缺陷是同分子式局部结构、近结构候选和峰级身份信息没有被充分组织进 embedding，而不是简单的随机噪声鲁棒性不足。

### 2.2 成熟 N-arm 动作真实有效

- `candidate_gradient a=0.50`：根据正确/错误候选的峰级 margin 方向逐步干预；第 3–6 步净收益累积，固定 step 6 为 140 corrected、42 introduced、net 98。
- `role_confounder a=1.00`：衰减匹配错误候选而不匹配正确候选的峰；安全性高，但覆盖窄，固定 step 5 为 24 corrected、0 introduced。
- `role_shared`：随着删除步数增加持续破坏身份信息，属于明确负结果，不应再回到无条件 shared-peak 删除。
- 与相同删峰数、相近强度和 m/z 的随机对照相比，定向干预的错误纠正率明显更高；这证明“删哪些峰”比“删多少峰”重要。

### 2.3 正例证据动作扩充了 N-arm 之外的空间

- consensus projection 最佳固定 cell：约 `+1.127 pp`，456 corrected、187 introduced；
- recurrent union mix 最佳固定 cell：约 `+1.118 pp`，294 corrected、27 introduced；
- 在官方 geometry 上，P/N outcome-aware union 的 recoverable error 数可超过 5 pp 所需数量。

这些是重要的动作空间证据，但仍然是冻结 encoder 上的反事实 action outcome 或 oracle union，不是共享 encoder 性能。

### 2.4 共享 embedding 的可靠现有成果

E4-A 解冻最后 1/7 Transformer block 与官方 projection head，5 formula folds × 3 seeds：

- Recall@1 平均 `+0.635 pp`；
- near Recall@1 平均 `+0.522 pp`；
- 每 seed 平均约 186 corrected、35 introduced；
- 15/15 fold-seed 方向为正。

这是目前噪声路线最牢固的“模型权重本身”成果。它不能被 action headroom 替代，也不能被后续失败实验抹掉。

## 三、为什么 PPT 中 3–5 pp 的潜力没有直接变成模型增益

### 3.1 首要断裂：反事实动作含有特权信息

`candidate_gradient` 的峰选择依赖正确候选和 hardest wrong candidate；`role_confounder` 依赖峰在正负参考谱中的角色；P-arm 的 projection/graft 依赖真实同身份参考谱。动作生成器因此知道“正确答案是谁、错误候选是谁、缺失的正例峰是什么”。

最终共享 encoder 推理时只收到原始谱图。它不可能逐 query 复现一个依赖候选答案的删峰路径，也不可能从完全缺失的峰中无条件恢复同身份参考谱的信息。正确目标不是让 encoder **模仿每个特权 action 的谱图或绝对 embedding**，而是把这些 action 当作反事实证据，提炼成跨分子可复用的峰上下文规律和候选 margin 约束。

这个区别解释了为什么 action oracle 很强、学生却弱：oracle 在回答“如果已知答案和竞争者，怎样改谱能过边界”；部署模型需要回答“只看任意新谱，怎样组织峰级表征，才能让正确参考普遍胜过相似错误参考”。两者不是同一个学习问题。

### 3.2 R2 已经定位：action view 学会了，clean transfer 没学会

R2 的 fixed-teacher transfer 不是“教师没效果”：

- teacher action accuracy 约 99.7–100%；
- student 对 action view 的 train/held accuracy 约 99.0%/98.8%；
- 但原始 clean query 对 teacher-corrected action 的转移，train 约 18.6%，held formula 仅约 1.17%；
- clean embedding 朝固定 action target 的平均 cosine 移动只有约 0.0022。

这说明模型有容量编码并复现被修改过的谱图，但没有把 action 中的候选特异信号抽象成 clean 谱图上的跨 formula 规则。增加 action 重复次数只能让 action view 记得更牢，不保证解决这个泛化断裂。

### 3.3 3.35–3.85 pp 是条件 oracle，不是固定策略净增益

固定 N-arm cell 的全任务净收益只有：

| 固定 cell | corrected | introduced | net | 全任务净增益 |
|---|---:|---:|---:|---:|
| candidate-gradient 50%，step 6 | 140 | 42 | 98 | `+0.410 pp` |
| role-confounder 100%，step 5 | 24 | 0 | 24 | `+0.101 pp` |

历史约 `+3.346 pp` 来自 S1c/S2/S3A 中，对每个 query 根据结果在多种 action/step/no-op 之间取最优的 union oracle。它证明动作库覆盖很多错误，却同时使用了结果来做路由。若没有可泛化的选择依据，不能把 union 直接写入单一共享 encoder。

### 3.4 动作容量会随 embedding geometry 变化

不能把官方 geometry 上 P/N 的 5 pp headroom，与成熟 E8 encoder 上的 residual headroom直接相加。模型更新后：

- hardest wrong candidate 会改变；
- 原本有效的峰路径可能陈旧；
- 新增错误形成新的 residual error space；
- 相同 action 的 margin 和安全性会变化。

E9 显示成熟 N action 的 frozen/online residual oracle相近（40 vs 38 corrected），说明 N-arm staleness 在该 fold 上不是首要矛盾；但 E10 在成熟 E8 geometry 上的 P/N/no-op total oracle 约 `+3.377 pp`，并没有自然保持官方 geometry 的 5 pp。所有容量结论必须在同一模型阶段、同一候选图上计算。

## 四、正式训练实现中真正存在的断裂

### 4.1 +0.635 pp 的定向噪声独立贡献尚未完成因果拆分

冻结 action scan 有 matched-random control，但 E4-A 训练只把 target action payload送入 `DirectExample`；匹配随机 action 没有进入成对学生训练。E4-A 同时优化 clean rank 与 action rank，因此普通监督式 listwise continuation 本身就可能产生一部分增益。

目前缺少严格的三臂学生实验：

1. clean-only duplicate-view control；
2. 等剂量、等曝光、等峰数的 matched-random noise；
3. target-directed noise。

三臂必须使用完全相同的 query、candidate batch、训练步数、学习率、解冻层和 seed，并做公式簇配对比较。没有这个实验，我们可以说“该训练方案得到 +0.635 pp”，但不能严谨地说“定向噪声本身贡献了全部 +0.635 pp”。

### 4.2 训练候选集合被官方 embedding 截断并冻结

训练通常取官方分数最高的少量正例参考谱，并为少量负分子取代表谱；最终评价却使用完整候选图。候选分子每 query 中位数为 3、p90 为 8、最大 21，单分子参考谱最多数百张。encoder 更新后，未被训练 batch 包含的负谱可能成为新的 hardest negative。

这会造成三类错误：

- 训练 margin 已经满足，但完整图 Top-1 仍错误；
- 修正原 hardest negative 后，第二/第三负候选接管 Top-1；
- 固定官方 hard negative 与当前学生 geometry 脱节，新增错误没有梯度来源。

所以“动作有效但最终净增益小”部分来自 listwise objective 与正式检索图不一致，而不只是 action loss 权重小。

### 4.3 动作曝光不足是真的，但不是第一优先级

R0 有 36,934 个 action row。fold 0 外折后有 28,509 row、7,766 eligible query；正式 sampler 每 identity 每轮只取 4 view，4 轮约 24,992 次 action presentation，且存在跨轮重复。它确实没有覆盖 PPT 中同一 query 的完整 candidate-gradient 3–6 步与 confounder 1–5 步。

但在 R2 已显示 clean transfer 几乎为零的前提下，先把曝光翻倍可能只会强化 action-view 拟合。正确顺序是先证明 target-minus-random 的 clean transfer存在，再解决完整动作覆盖。

### 4.4 outcome/no-op 路由缺失，但不能直接训练一个结果 oracle

E4-A 把 corrected、neutral、introduced action 以近似相同的 action rank/consistency方式使用，确实稀释了监督。E14 又曾把每 query 压成单一 action，导致条件动作空间丢失。

然而“把所有 outcome/no-op 标签直接喂给 selector”也不正确，因为 outcome 依赖正确身份和完整候选图，推理时不可得。合法做法是：

- outcome 只在 outer-train 中定义反事实监督；
- held identity/formula 完全不使用 action outcome；
- 学习可泛化的机制或 pairwise margin，而不是记忆 query→action ID；
- no-op 是安全基准，不是一个可由答案泄漏决定的部署标签。

### 4.5 原损失没有直接学习“定向动作优于随机动作”

E4-A 损失为：

`Lclean-rank + Laction-rank + 0.25 Lconsistency + 2 Lmargin-floor + 5 Lpreservation`

第 4 轮日志中 clean rank 与 action rank 约占记录 loss 的 91.5%，加权 consistency 约占 0.52%。更关键的是，没有以下差分量：

`[m(target_action)-m(clean)] - E[m(matched_random)-m(clean)]`

也就是说，冻结筛选阶段证明的“方向特异性”没有成为学生的核心优化对象；target action 只是一个增强样本。

### 4.6 symmetric/stop-gradient 不是现在的主瓶颈

上一版把双向 consistency 提得过高。E8 配对因子实验中，symmetric/shared、stop-gradient/shared 和 official-action/shared 的 Top-1 结果基本相同；改为固定官方 reference 反而显著变差。说明：

- 仅把 cosine consistency 改成 stop-gradient，不会解决 clean transfer；
- query 与 reference 必须由同一个共享 encoder共同移动；
- 关键应是候选边界与反事实 advantage，而不是绝对 action embedding 的单向回归。

### 4.7 risk branch 在关键试验中几乎没有形成有效反梯度

E15-M2 的小样本 overfit 能把 27 个错误纠正且不新增，证明网络容量和 payload 执行链可工作；但 harmful/risk 分支梯度范数约为 `10^-7` 量级，远弱于 corrective branch，并且分支冲突频繁。这意味着“已物化 harmful actions”不等于它们真的在保护 clean geometry。

风险约束应直接作用于：introduced query 的完整候选 margin、protected-correct 的 margin floor，以及 target-vs-harmful 的成对差分；不能只靠接近零的 preservation/routing 项。

### 4.8 M3 负结果是被筛空后的实现结果，不是路线裁决

E15-M3 最终只有 281 corrective actions、198 queries、133 identities，另有 79 个训练 query 因参考排除后失去合法负分子而被丢弃。held 上 0 corrected/1 introduced 只说明该训练池不足以支持 identity transfer；不能据此否定完整 E15 ledger 或噪声路线。

## 五、动作空间到底哪里仍未充分挖掘

### 5.1 N-arm 的问题不是动作名字少，而是机制抽象不足

candidate-gradient 目前主要是一条由特定正/负候选定义的贪心路径。未充分覆盖：

- 多 hard-negative 共同约束，而非只追逐当前第一错误候选；
- 多条近似最优路径与峰间协同；
- 删除一个峰后 hardest negative 切换的在线路径；
- neutral loss、相对 m/z、强度比例和 contextual peak token 层面的跨分子机制；
- 对 introduced error 形成的反方向机制。

继续枚举更多 step 只能加密同一条局部路径；真正需要扩的是“可跨 formula 复用的峰关系机制”。

### 5.2 role-confounder 是高精度安全动作，不是主覆盖动作

它严格依赖峰能区分正/负参考，覆盖天然有限。应保留为 safety-first 专家监督，而不能通过放宽规则强行承担 1,805 错误的大部分召回；否则会重新滑向 role-shared 的身份信息破坏。

### 5.3 P-arm 的主要难点是信息是否存在，而不是剂量还不够细

consensus projection 和 recurrent graft 在反事实谱上有效，但若 clean query 中完全不存在某峰，共享 encoder 无法凭空观察它。P-arm 必须分成：

1. **可预测强度不足**：峰存在但强度/上下文被错误编码，可通过 real-replicate强度与条件不变性学习；
2. **部分缺失但可由其余峰预测**：可做 masked/contextual peak prediction 辅助任务；
3. **不可由 clean 输入识别的真正缺失证据**：只能学习对缺失的鲁棒性，不能把 reference graft 当成 clean target 强制复现。

因此下一步应先做 clean-input predictability 分层，而不是把所有 positive-guided action 混进同一损失。

### 5.4 新增错误空间没有被作为独立学习对象闭环

每个获得净增益的动作都产生 introduced errors。过去多为计数、过滤或 safety weight，而没有系统构造：

- 原正确候选为何因动作失分；
- 新错误候选是旧 hardest negative、候选切换还是共享峰身份损失；
- 该错误能否由完整候选 margin 提前预测；
- 对应 harmful action 的反事实 advantage 是否显著；
- 同一机制在其他 formula 是否重复。

“corrected > introduced”不是充分安全机制。新增错误必须形成与 corrected 对称的机制图谱和非零反梯度。

## 六、重新排序后的决战优先级

### 优先级 0：冻结可比协议与实现等价性

先固定同一官方 baseline、完整候选图、formula folds、identity 排除和 fresh-forward reproduction；action payload、matched control、query/candidate rows 必须逐条 replay。任何训练臂只能改变预注册的一个因素。

### 优先级 1：做共享 encoder 的三臂因果归因实验

在同一 outer-train/held formula 上运行：

- A：clean-only duplicate views；
- B：matched-random noise；
- C：mature targeted noise。

三臂完全匹配样本、候选、步数、学习率、batch 和 seed。首要结果不是 C 相对 official，而是：

- `C - B` 的 formula-cluster CI；
- `C - A` 的 formula-cluster CI；
- corrected/introduced、near、完整候选 margin；
- action-view 与 clean-view 的转移率。

若 C 不显著优于 B，说明当前学生只获得普通增强/监督 continuation，必须停止扩大训练。若 C 优于 B 但 clean transfer低，进入优先级 2。

### 优先级 2：建立“动作可学习性/可识别性”门

按 action family、error family 和 evidence availability 分层，在 outer-train 学、identity/formula holdout 测：

- clean contextual peak tokens 能否预测 action advantage 的符号和大小；
- 峰角色能否跨分子泛化，而非记住 query/action ID；
- candidate-conditioned pairwise机制能否转化为共享相似度；
- P-arm 缺失证据属于可预测、仅鲁棒、还是不可识别。

只有 held formula 上 advantage sign、校准和净 margin 显著的 family 才进入共享 encoder。不可识别动作保留为解释性 headroom，不再被假装成可部署训练标签。

### 优先级 3：把训练目标对齐到完整候选检索

- 使用每 query 全部候选分子，或至少保证包含当前学生 geometry 下的动态 hardest negatives；
- 多正例参考谱以 molecule-level 聚合，避免单一参考偶然性；
- 定期刷新负候选，而不是永远使用官方 embedding 的 top negative；
- loss 直接优化完整 listwise/molecule-level margin；
- query/reference 始终共用同一 encoder。

### 优先级 4：学习反事实 advantage，而非模仿 action embedding

核心训练量应为：

`A_target = [m_theta(a_target)-m_theta(q)] - mean_r[m_theta(a_random_r)-m_theta(q)]`

并将其转成 clean query 的 candidate-margin 约束。target、matched-random、harmful 和 no-op 必须成组进入同一 microbatch。绝对 embedding consistency 只作弱正则；risk branch 对 introduced/protected-correct 的完整候选 margin 给出可测的非零梯度。

### 优先级 5：完成机制级动作空间，而不是继续单路径 step 扫描

- N-arm：multi-negative、beam/path diversity、峰对/峰组协同、neutral-loss 与 contextual-token 机制；
- P-arm：real replicate consistency、强度校准、masked peak context、可预测缺失与不可预测缺失分开；
- 每个新 family 都报告相对当前 geometry、相对既有 union 的 unique recoverable gain和 introduced mechanism。

### 优先级 6：再做 multi-action/no-op 曝光与采样

此时才对通过可学习性门的动作执行完整覆盖：

- 同 query 保留多个 corrective/harmful/control action；
- 不以 4 views/identity 丢弃动作；
- identity-equal 在 loss 聚合实现；
- action 不在 epoch 内无限循环；
- no-op 与 protected-correct 为独立安全支路。

### 优先级 7：最后调学习率、解冻层数和训练规模

89% clipping 和仅解冻最后 1 block 是真实限制，但历史 clip/LR 扫描只带来约 0.05 pp 并迅速破坏 preservation。只有前六项通过后，才比较 1–2 blocks、分层 LR、梯度投影和更长训练；否则只是更快拟合错误目标。

## 七、下一轮应当如何裁决，而不是再绕弯

下一阶段不直接叫“更大规模 E15”，而应分为四个不可跳过的门：

1. **Attribution gate**：targeted student 显著优于 matched-random 和 clean-only；
2. **Learnability gate**：action advantage 在 identity/formula holdout 可预测、可校准；
3. **Full-list transfer gate**：完整候选图上 clean query 的 margin/Top-1 改善，且相对配对 control 显著；
4. **Scale gate**：通过前三门后才扩大 action family、fold、seed、层数和训练量。

每一门都必须同时报告 corrected、introduced、candidate-switch、新增错误机制、near、MRR、完整候选 margin 与 formula-cluster CI。任何“action accuracy 高但 clean transfer 低”的结果只能判定为反事实可执行，不能再包装成 embedding 已学会。

## 八、最终判断

我们没有回到随机删峰，也不需要放弃噪声微调。PPT 已经解决了第一层问题：真实错误可被有方向的峰级干预纠正，而且峰的语义比随机剂量重要。真正长期绕弯的是把这个**候选条件的反事实能力**误当成了**共享 clean encoder 可直接吸收的监督**。

当前瓶颈的正确优先顺序是：

1. 定向噪声对共享 encoder 的独立因果贡献尚未拆清；
2. 特权 action 到 clean-input 跨 formula 表征的可学习性未建立；
3. 训练候选边界与最终完整检索图不一致；
4. 训练没有优化 target-minus-random 的反事实 advantage，risk 梯度也未真正生效；
5. 动作空间缺少机制级、多负候选、P-arm 可预测性与新增错误闭环；
6. 之后才是 multi-action 曝光、采样、学习率和解冻层数。

这不是降低目标，而是把“动作能改谱”与“模型能从原始谱学会”之间缺失的科学问题正面解决。只有这条桥搭好，官方 geometry 上的 3–5 pp action capacity 才有可能稳定转化为更强的共享 embedding，而不是继续在 oracle headroom 与 `+0.x pp` 学生结果之间往返。

## 九、整条噪声路线的逐阶段审计账本

本节不再按“成功/失败”二分。每个阶段分别回答四件事：当时真正验证了什么、没有验证什么、失败属于动作、监督、训练还是实现、今后如何处置。不同候选协议下的 Recall 不横向拼接；正式主任务统一以 23,876 query、1,805 个官方错误的冻结账本为准。

### 9.1 早期随机噪声 G5–G7：已证伪的起点，不得重启

- **做法**：均匀随机删峰、随机加峰、独立 m/z 抖动，以及不区分峰角色的随机遮挡。
- **结果**：在当时匹配的旧任务上，官方 DreaMS 为 0.8676，G5/G6/G7 仅为 0.8216/0.8354/0.8365。这个数值只用于同协议历史判案，不能与 23,876-query 主任务混用。
- **具体错因**：
  1. 把身份主峰和采集噪声峰等价处理；
  2. 删除比例取自人为均匀分布，而非真实重复谱的条件缺失分布；
  3. 随机添加的 m/z 缺乏真实碎裂来源；
  4. 独立 m/z 抖动破坏峰间共现和中性丢失关系；
  5. 训练只学到“对任意破坏保持不变”，没有获得纠正近结构错误所需的方向。
- **裁决**：永久停止作为主训练策略。仅保留“等剂量、等强度、等 m/z、等删峰数”的 matched-random 作为因果负对照。

### 9.2 初代错误图谱与整组定向删峰 v2：方向正确，干预单位错误

- **做法**：先把错误拆成 positive deficit 与 negative excess，再按 identity-adduct 低频/低强度条件峰整组删除；同时比较 shared/unique 峰。
- **结果**：第一轮 55,892 个变体表明，删 unique 峰会同时改变错误 query 和 protected-correct query；匹配对照后不存在稳定的错误特异 excess。M1 中 1,231 个完整配对 query，target 与随机对照 Top-1 都为 0.5589；baseline-wrong 中 target 修正 22 个，随机期望修正 26.33 个，identity/formula CI 均跨零。
- **具体错因**：
  1. 一次同时改变“选择哪些峰、删多少、如何删”三个变量，无法归因；
  2. 把一组 conditional peaks 整体硬删除，剂量过粗；
  3. 没有针对当前最危险的错误候选；
  4. 用 shared/unique 这种二元标签替代了峰对正负候选 margin 的真实贡献；
  5. 低频不等于噪声，跨碰撞能/仪器变化也不等于身份无关。
- **裁决**：永久停止整组 conditional deletion 和泛化的 shared/unique 训练标签。真实重复谱 prevalence 只保留为剂量校准、峰保护和采集关系建模证据。

### 9.3 Noise-v3 G1/S1A/S1B/S1C/S2/S3A：成熟动作空间的真正起点

- **可靠发现**：
  - `candidate_gradient a=0.50` 随 3–6 步累积，净修正从 76 增至 98；第 6 步为 140 corrected / 42 introduced，固定动作约 `+0.410 pp`。
  - `role_confounder a=1.00` 高精度、低覆盖；典型第 5 步 24 corrected / 0 introduced。
  - `role_shared` 全剂量有害；`a=1.00` 从第 1 步 208/578 恶化至第 6 步 151/1,025。
  - `role_unmatched` 多数单元净增很小，覆盖低。
  - S1c+S2+S3A 的 outcome-aware union 可覆盖 799 个错误，即 23,876 query 上 `3.346 pp` 的动作空间 headroom。
- **当时做对的地方**：第一次以 matched-random 控制证明“删哪些峰”而非“删了更多峰”决定性能；第一次同时画 corrected、introduced、net 和新增覆盖矩阵；第一次发现 candidate-gradient 的收益可小步积累、shared 删除的伤害也会积累。
- **仍未做够的地方**：
  1. candidate-gradient 只沿一个当前 hardest wrong candidate 的贪心路径前进，没有覆盖 Top-k 错误候选切换；
  2. confounder 只擅长降低负候选，不解决占 68.8% 的 positive-deficit-only；
  3. 每条 action 使用正确/错误候选身份，是训练期 privileged information；
  4. union 以答案选择动作，是容量上限，不是 deployable policy；
  5. 没有把“稳定净增”与“clean input 上可预测”分开筛选。
- **裁决**：保留 candidate-gradient 3–6 步与 role-confounder 1–5 步为成熟 N-arm 原语；永久禁用 role_shared；role_unmatched 从主线移出，只能在出现新机制证据时复查。不得再把 3.346 pp 说成已训练模型增益。

### 9.4 A4 exact peak scan：证明峰级空间很大，也证明高剂量会失控

- **规模**：1,805 errors + 3,193 matched-correct controls；206,288 个 peak actions，825,152 个变体。
- **剂量结果**：25% 为 138/183，50% 为 289/375，75% 为 461/652，100% 为 738/1,078。梯度 Top-1 捕获 411 个可纠错 action，Top-3 578，Top-25 703，Top-50 736。
- **容量结果**：与既有动作结合的 oracle 可修正 920/23,876，即 `3.853 pp`；仍低于当时 4 pp 的 956-query 门槛。剩余 1,029 个未恢复错误中约 910 个含 positive deficit。
- **具体错因**：
  1. “能把当前错误翻转”被错误等同于“适合训练”；
  2. 75–100% 的强干预通过破坏身份信息制造大量假纠正；
  3. gradient 只适合作为候选生成器，不足以作为最终 action；
  4. shared 峰和强峰删除是 introduced errors 的主要来源；
  5. 对低 margin、稀疏谱使用同剂量导致风险集中。
- **裁决**：保留 A4 action/outcome/harm 标签作为训练候选账本和安全证据；永久停止 75%/100% 全局删除、gradient-only 与 shared-only 策略。任何 action 进入训练必须同时通过 matched-random specificity、风险与 clean-input learnability。

### 9.5 A4 nonlinear teacher、A4-B 与 C1：教师很强，但多数知识不是 clean encoder 可直接获得

- **A4 teacher**：formula-OOF corrective ROC-AUC 0.849、harmful AUC 0.853；40% 覆盖下 182/39，risk net 104。但相对简单 confounder-only 没有显著优势，只新增 19 个既有矩阵外的修正，未达 80-query 新覆盖门槛。
- **A4-B frozen adapter**：线性约 `+0.760 pp`（100/62），非线性约 `+1.060 pp`（98/45），非线性未显著优于线性，且仍是冻结 query adapter，不是 backbone embedding。teacher-rescue 542 个中学生只恢复 80 个，约 14.8%，并新增 45 个错误。
- **C1 support-disjoint teacher**：80,250 examples、1,217 identities、627 formulas；teacher instance accuracy `+2.475 pp`，2382/396，near `+2.344 pp`。但它是 identity-label-supervised teacher，不是可部署检索结果。
- **具体错因**：
  1. teacher 看到了正确身份参考、错误候选或 action outcome，clean query 推理时看不到；
  2. 某些 positive graft 含原始 query 根本不存在的信息，确定性 encoder 不可能可靠“猜出”真正缺失峰；
  3. 学生被要求复制 action/prototype embedding，而非学习原始谱中可识别的机制；
  4. selector 主要在旧 action 空间内重新排序，没有创造足够新的独立覆盖；
  5. instance-level 大样本被误读为 query/formula-level 可迁移容量。
- **裁决**：保留 support-disjoint 协议和真实同身份重复谱关系；停止 clean→grafted/prototype 绝对方向模仿和在同一 action 集上继续堆非线性 selector。P-arm 必须改成真实 noisy-view→clean/identity consistency、contextual masked-peak prediction，且先证明目标可从 clean input 预测。

### 9.6 C2 peak-token 与 P2b residual：重要表征资产，但不是噪声微调成功

- **C2-A**：global control `+0.0461 pp`（161/124），token expert `+0.0835 pp`（181/114）；token 相对 global 的 formula CI 跨零，risk gate 失败。
- **C2-B/C2-C**：早期一次 86/0 的“最好 fusion”来自有误实现；纠正后约 `-0.0209 pp`（30/35），P2b residual 后续为零或负。
- **具体错因**：
  1. 用 token 特征做下游 query/candidate fusion，输出的不是新 embedding；
  2. 首次结果存在工程错误却一度被当成突破；
  3. 对 P2b 的边际残差失败被错误用于否定 noise route；
  4. 简单方向头没有显著超过全局控制。
- **裁决**：保留 contextual peak tokens 作为未来可学习峰机制的输入；永久从 noise 主线剔除 C2-B/C2-C/P2b residual。首次 86/0 结果不得引用。

### 9.7 D0/D1/D1b：有效的 clean control，被错误命名成噪声微调

- **结果**：D0 正确冻结 23,876-query、1,805-error、P2b-free 合同。D1/D1b 实际训练 clean-only query adapter；14 seeds 多为 0–0.2 pp，平均约 `+0.067 pp`，部分 near 下降。
- **具体错因**：没有 action/noise teacher；reference geometry 冻结；query-only adapter 不形成统一共享空间。
- **裁决**：仅作为 clean continuation/control 保存；不得称为噪声模型、不得代表 noise route 上限，也不作为新模型主线。

### 9.8 F1 v1–v4：P-arm 蒸馏失败的原因不是学习率，而是监督不可识别

- **结果**：v1 外层约 `+0.0338 pp`（11/9）；v2/v3 普通 Recall 很小且 identity-equal risk net 为负；v4 在训练 challenge 可修正 96 个，但外层 full graph `-0.0844 pp`（0/5），外层 challenge 16/14。
- **具体错因**：
  1. support-disjoint challenge 与自然完整检索图分布不同；
  2. 大量 baseline-correct 的轻微 teacher 改善稀释真正 error gradient；
  3. teacher direction 候选条件化，clean input 不可识别；
  4. pooling/query adapter 无法改变 Transformer 内峰间交互；
  5. clean→teacher/prototype 目标再次要求模型复制缺失证据。
- **裁决**：停止当前 P-arm margin/prototype distillation。保留真实重复谱，但把任务重写为 view-invariant identity geometry 与可预测峰语义，而不是模仿特权 teacher。

### 9.9 E1–E4-M1b：校准与动作发现有效，所谓“梯度兼容”不足以授权训练

- **E1**：3,412 identity-adduct groups、2,588 identities、1.21M matched peak pairs；是可靠采集变化分布，不是性能结果。
- **E2**：14 个 sensitivity-passing action cells；证明冻结 encoder 上的即时动作特异性。
- **E3**：27,735 actions、1,104 identities、408 formulas；计算的是 embedding tangent direction，不是 117M 参数空间真实梯度。“16 个 compatible、0 conflict”不能推出联合训练会成功。
- **E4-M0/M1/M1b 问题**：
  1. 把多步/多剂量 curriculum 压成单一小 target，丢失停步规则和收益幅度；
  2. clean rank loss 量级约为 action loss 的 200 倍，action 梯度被淹没；
  3. 平衡后 candidate/acquisition 分支仍可与 clean objective 反向；
  4. pooling residual 不能改变上下文峰 token 的交互；
  5. 训练方向是 clean→perturbed，而稳健表征更应检验真实 noisy/action view→clean identity geometry。
- **裁决**：保留 E1/E2；E3 只作描述。永久停止 E4-M0 绝对 target 压缩和 pooling residual imitation。

### 9.10 R0/R1/R2：动作可被记住，但无法跨 formula 迁移到 clean query

- **R0**：忠实恢复 36,934 action rows、1,991 identities、877 formulas，并保存 matched controls 和完整 path；这是必须保留的资产。
- **R1**：可纠错 query 882 个，但仅 462 identities、248 formulas，容量本身已高度集中。
- **R2**：学生 action-view 学习接近 99%，clean transfer 在训练约 18.6%，held formula 约 1.17%；高 LR held `-0.0844 pp`（1/6），低 LR `-0.0338 pp`（5/7）。
- **具体错因**：
  1. action-view 记忆被误认为 clean-view 泛化；
  2. 每 identity/action 压缩，动态路径信息损失；
  3. N、robustness、safety 与 unfreeze 同时变化，无法归因；
  4. moving-student target 漂移；
  5. 训练使用冻结少量正负代表谱，模型变化后真正 hardest negative 切换；
  6. clean input 不含候选条件，无法知道应向哪条 action path 移动。
- **裁决**：保留 R0；停止 absolute action embedding distillation 与 R2 组合 trainer。下一次必须以 matched-control 学生因子实验和完整 listwise margin 判断定向动作的独立增益。

### 9.11 E4-A direct shared encoder：当前最好真实 noise embedding，但因果归因仍不完整

- **做法**：共享 query/reference encoder，解冻最后 1 个 Transformer block + projection head；candidate-gradient 3–6 步、confounder 1–5 步；每 identity 4 views、4 epochs。
- **结果**：5 folds × 3 seeds，平均 Recall@1 `+0.635 pp`，near `+0.522 pp`；约 186 corrected / 35 introduced；15/15 fold-seed 为正。这是目前可引用的最好共享 noise embedding 结果。
- **仍然缺失的关键对照**：R0 已有 matched_control_paths，但 E4-A `DirectExample`/loss 未真正使用；没有与 targeted 完全同 sampler、同训练步数、同剂量的 matched-random student，也没有严格的 clean-only duplicate arm。因此 +0.635 pp 是可靠模型增益，但不能全部归因于定向噪声。
- **目标稀释**：clean rank + action rank 占主要损失，consistency 加权贡献约 0.5%；fold0 有 28,509 action rows、7,766 queries，但每 epoch 仅使用约 6,248 views，四轮约 24,992 次且有重复，未形成 action 层面的均衡覆盖。
- **候选错位**：训练 negative/reference 由官方 embedding 冻结截取，最终评估使用完整候选图。
- **裁决**：保留 E4-A checkpoint、结果与训练配置作为官方 noise baseline；下一步不是直接扩大它，而是先做 clean/random/targeted 三臂因果复现，并把训练 margin 对齐到完整候选边界。

### 9.12 优化器、层数与 clipping 扫描：不是当前主瓶颈

- **结果**：提高 LR/clip 在单 fold 仅约增加 0.05 pp，随后破坏 preservation；最佳附近约 89% 步骤发生 clipping。
- **裁决**：暂停一切大范围 LR、clip、epoch、解冻 1/2/3 blocks 扫描。只有目标、对照和 full-list transfer 通过后才做小型分层 LR 比较。继续扫描属于低性价比局部优化。

### 9.13 E5/E6/E7：positive/intensity/recurrent 具有 action 信号，但固定权重转移几乎没有增量

- **E5**：N-only `+0.5403 pp`（38/6）；intensity weight 0.10 为 `+0.5740 pp`（41/7）但 preservation 失败。权重/安全扫描相对 N-only 的 paired CI 均跨零。
- **E6**：固定策略约 `+0.4052 pp`（30/6）；outcome-mined families held 仅约 14/11。把 globally harmful 的 role_shared 中个别 oracle 成功动作选入训练，造成选择偏倚。
- **E7**：recurrent_union_mix 相对 fixed control，0.025/0.05/0.1 只多修正 1 个；0.2 多 3 个但 preservation 失败，CI 下界为零。
- **具体错因**：
  1. 全 query 固定 guided weight，不区分 clean-input 可判断的适用条件；
  2. positive action 的信息来源仍是 privileged same-identity reference；
  3. outcome mining 允许“任何能修正当前 query 的 action”，重新引入 oracle；
  4. additive loss 没有表达 targeted-vs-random advantage；
  5. 小增量被反复 weight scan，消耗算力却不扩大机制覆盖。
- **裁决**：停止 fixed intensity dose/safety、outcome-mined policy 和 recurrent weight 扫描。recurrent/top3 positive evidence 仅作为待验证机制保留，不再作为全局共享 loss 直接叠加。

#### P/N 五点容量与 positive-guided 两轮扩展：真正跨过的是 action union，不是模型权重

- **P/N frozen union**：N-arm 可恢复 882 个，P-arm 在正式 1,805 个错误中严格可恢复 173 个，重叠 133，union 为 922；主任务 headroom `3.8616 pp`，距离 5 pp 仍缺 272 个 unique errors。未覆盖 883 个错误中，positive-deficit-only 662、positive+negative 133，说明最主要缺口仍是正证据不足，而非继续削弱负候选。
- **positive-guided intensity matrix**：最佳 `consensus_projection@0.75` 固定 action 为 `+1.1267 pp`，456 corrected / 187 introduced，虽然方向特异性显著，但风险成本很高；它只为 frozen P/N union 新增 223 个 unique，扩到 1,145，仍差 49 个到五点。
- **positive peak transfer matrix**：最佳 `recurrent_union_mix@0.50` 固定 action 为 `+1.1183 pp`，294/27；再新增 112 个 unique，使 outcome-aware union 达到 1,257，表面上超过五点所需的 1,194。
- **为什么仍不能直接拿来训练**：
  1. 三个 union 都用结果决定每个 query 选哪条 action/no-op；
  2. positive-guided action 依赖真实同身份参考谱，推理时不可见；
  3. consensus projection 的 187 个 introduced 已经表明全局使用会严重伤害；
  4. recurrent transfer 的 27 个 introduced 虽低，但只有约 34.7% query 可执行，适用条件没有被 clean input 模型学会；
  5. 达到五点的是不同 action 的事后集合并集，不是一套共享权重。
- **裁决**：保留这两轮矩阵作为“positive evidence 确实是剩余主空间”的决定性证据；永久禁止把 1,257-query union 或五点容量写成模型成绩。下一步不是再扩 positive action，而是检验 recurrent/positive action 的 clean-input 可识别性及相对 matched-random 的学生增量。

### 9.14 E8/E9：共享编码与 curriculum 的必要性已证实，在线重挖没有价值

- **E8**：mature symmetric/shared 为 `+0.5740 pp`（38/4）；stopgrad 相对 symmetric 的 Top-1 增量严格为 0；official action target 为 0；fixed official references 相对方案 `-0.2195 pp`（1/14）；terminal-only 比 curriculum 差，combined 也未恢复。
- **E9**：online re-mining 与 frozen path 仅 10/9，约 `+0.0119 pp`，CI 跨零。
- **裁决**：
  - 保留 shared query/reference encoder 与多步 curriculum；
  - 永久停止 fixed official references、absolute official target、terminal-only 与 stopgrad 变体；
  - 停止当前 N action 的 epoch-wise online re-mining。注意：这不等于停止 full-list hard-negative refresh；前者重挖 action path，后者只让训练 margin 看见模型当前真正最难候选。

### 9.15 E9B/E10/E10B/E11/E12/E13：动作空间接近 5 pp，不等于学生接近 5 pp

- **E9B**：成熟 N action 在 held full task 的额外 oracle headroom `+0.675 pp`，总 headroom `+1.249 pp`。
- **E10**：recurrent_union_mix fixed 相对 mature E8 `+0.878 pp`（59/7）；no-op oracle 总 headroom `+3.377 pp`。
- **E10B**：19 个 P cells 中 13 个 fixed pass；union 总 headroom `+3.799 pp`。
- **E11**：reference diversity 再添 39 unique、union 增 `+0.658 pp`，总约 `+4.457 pp`；但 maxmin6 为 55/20，风险明显差于 top3 的 59/7。
- **E12B**：relaxed recurrence 最好 fixed 相对 E8 `+1.7896 pp`（137/31），但对 union 只新增 28 个、`+0.4727 pp`，仍未到 5 pp。
- **E13**：control `+0.5740 pp`；guided arms `+0.557–0.608 pp`、39–42/6，相对 control 无显著增量，且全部 preservation <0.995。
- **具体错因**：
  1. 不断放宽 recurrence/reference/dose 主要在重复覆盖同一批错误；
  2. oracle no-op union 使用结果选择动作，不是推理策略；
  3. reference diversity 增加了信息量，也增加错误分子证据污染；
  4. relaxed recurrence 提高 fixed action，却未给 shared encoder 带来可识别的条件；
  5. 把“五点容量门”误用成学生训练目标，诱发无止境 action scan。
- **裁决**：停止全局 reference-diversity、relaxed-recurrence 阈值和 dose 扩展。保留 action headroom 用于描述可干预空间，但不再以凑足 5 pp oracle union 作为下一次训练授权。

### 9.16 E14：明确的实现失败，不能被解释成路线失败

- **已确认实现错误**：
  1. 每 query 只保留一个“最佳” corrective action，破坏多动作 curriculum；
  2. global safe filter 删除本应按 query 条件成立的 action；
  3. risk controls 进入 corrective loss，造成 risk arm 6 corrected / 11 introduced；
  4. guided examples 在 epoch 内循环回收，部分样本过曝；
  5. gradient calibration 只看每分支前 4 个样本；
  6. selected scope 实际未应用 positive-deficit filter；
  7. 200-formula gate 失败于 183 后 post-hoc 改为 150，只能标记 amendment，不能当 prereg pass；
  8. 路径、缺文件、变量未初始化和输出完整性判断反复出错。
- **结果**：selected-no-margin/delta 相对 continuation 约 `+0.2127 pp`，但 risk branch 反而恶化；不足以判定 action transfer。
- **裁决**：永久停用 legacy E14 teacher/trainer；仅保留其 action artifacts 和失败证据。不得继续补丁式扩 fold。

### 9.17 E15 M0–M3：账本和小样本容量成立，正式 holdout 训练实现仍不合格

- **M0/M1 工程事故**：source status 过严且不匹配、validator 所有子 gate 为真却仍抛错、完成目录拒绝覆盖却没有自动新 run path、dtype/schema 不固定。这些是工程错误，不是科学失败。
- **M2**：可执行 panel 有 3,433 corrective / 2,479 harmful；32-query/60-action overfit 可达到 27 corrected / 0 introduced，证明局部模型容量存在。
- **M3 数据坍缩**：完整 ledger 到训练只剩 281 corrective actions、198 queries、133 identities；79 train queries 因“无合法负候选”被丢弃。所谓四源平衡和严格 identity exclusion 共同把有效监督砍掉大半。
- **M3 协议错误**：
  1. initialization margin 在 full graph 计算，实际训练 reference 却经过 identity exclusion 截断；
  2. paired warm control loss 约 `5.6e-7`、几乎不更新，不是真正等预算 continuation control；
  3. internal epoch 已判 ineligible，仍继续查看 held，导致 held panel 被消费；
  4. harmful/risk 分支虽分开记账，但其有效梯度近乎数值零；
  5. 过严的每 source quota 强行要求“平衡”，不符合各机制真实覆盖；
  6. 去除 held identity 的所有候选 reference 后，一些 query 失去负分子，暴露拆分单位和候选图不兼容。
- **结果**：held 0 corrected / 1 introduced，`-0.391 pp`；near `-0.498 pp`；preservation 0.99998，说明模型几乎没发生有用更新，而不是“稳定但没收益”。
- **裁决**：拒绝当前 E15-M3 trainer 和已消费 held panel；不得在其上调 LR、加 epoch 或直接 multifold。M0 ledger 可保留，但下一个 trainer 必须重写采样、候选、control 和 gate 流程。

## 十、瓶颈不是一个点，而是一条七层失真链

下面按因果顺序排序。上游未通过时，优化下游没有意义。

### 10.1 第一层：评价与归因失真

**表现**：把 action oracle、teacher instance、冻结 adapter、单 fold、共享 encoder、P2b residual 的 pp 混在一起；把 +3–5 pp headroom 当成学生应达到的已知答案。

**后果**：每次学生只有 +0.x pp 就被判断为“没有学到”，然后转向新的 teacher/action scan；但从未先测 targeted student 相对 matched-random student 的净优势。

**解决**：每个结果强制标注五个维度：`输入是否 clean`、`是否共享 encoder`、`是否 outcome-aware`、`是否 formula-held`、`候选图是否一致`。主因果终点必须是 paired `targeted - matched_random`，其次才是 `targeted - clean_only` 和 `targeted - official`。

### 10.2 第二层：动作有效性与输入可识别性被混淆

**表现**：candidate-gradient、confounder、positive graft 在被修改的谱上能翻转排名；学生 clean query 却不能跨 formula 重现。

**根因**：action 使用当前正确候选、错误候选、同身份参考或 outcome；这些信息推理时不存在。尤其对“真正缺失的峰”，原始谱没有足够统计信息时，任何 deterministic encoder 都无法知道应补哪一个结构证据。

**解决**：先做 learnability audit，而不是直接训练：只用 clean-query 可见的峰 token、precursor/adduct、谱稀疏度、局部中性丢失与采集元数据，预测每个 action 的 `target-minus-random margin gain`。按 formula cross-fit 报告校准、AUPRC 和可覆盖错误。不可预测的 action 只能作为 oracle 解释，不进入 clean encoder 监督。

### 10.3 第三层：两个主要动作没有覆盖完整错误机制

**candidate-gradient 的缺口**：

- 只沿单一 hardest negative 的局部一阶路径；候选切换后原路径失效；
- 偏向“压低某个负候选”，对 positive-deficit-only 的 1,242 个错误覆盖不足；
- 逐峰独立贡献忽视峰组合、峰对、中性丢失和结构同分异构体的非线性交互；
- 路径 step/dose 是 query-specific 的，却被全局 curriculum 平均化。

**role-confounder 的缺口**：

- 高精度但覆盖只有数百 query；
- 只能删除同时支持错误候选的混淆峰，无法恢复正确分子证据；
- 对近结构异构体常不存在单个“明显 confounder”。

**解决**：动作扩展不再按更多 dose，而按机制正交扩展：

1. Top-k candidate-switch-aware gradient：同时约束前 3–5 个负候选，动作后重新检查最难负候选；
2. positive-evidence reweighting：只重加权原谱已经存在、但 embedding 低估的峰，不凭空添加不可见峰；
3. peak-pair/neutral-loss interaction：以峰对和互补丢失为单位，而非单峰；
4. isomer-local contrast：在同分子式近结构候选内定义局部 listwise margin；
5. acquisition-calibrated invariance：真实重复谱中可预测变化做 noisy→clean/identity 学习；
6. explicit no-op/risk routing：适用性不确定时不干预。

每个新机制必须报告对冻结成熟 action union 的 **unique** 新覆盖与 introduced-error 来源；少于预设独立覆盖门槛就停止，不再靠 dose 扩展。

### 10.4 第四层：损失没有直接优化“定向比随机好”

**表现**：E4-A 的 consistency 贡献很小；E5/E7 guided weight 改变极少；E14/E15 risk branch 无效或反向。

**根因**：主要梯度仍来自普通 clean/action ranking；目标 action 与 matched-random 并未在同一 query、同一剂量下形成对比项。模型只需学会一般鲁棒性，不必使用动作语义。

**解决**：采用同 query 的配对反事实目标，显式比较 `margin(target_action)` 与 `margin(matched_random)`；同时保留 clean full-list ranking。risk branch 不是“把 harmful action 当 corrective 的反号样本”，而是保护 clean/no-op margin、惩罚 targeted action 造成的 candidate switch。每一分支先做梯度范数、余弦与数值非零审计。

### 10.5 第五层：训练候选边界与最终检索边界不一致

**表现**：训练修正了冻结错误候选，完整图上另一个分子成为 Top-1；M3 甚至因 reference exclusion 让 query 无负分子。

**根因**：正负 reference 由旧 official embedding 冻结截断；训练后 geometry 改变，candidate aggregation、多个参考谱和 hardest negative 都会变化。identity holdout 又错误地通过删 candidate references 实现，破坏真实任务。

**解决**：formula/identity split 只限制监督来源和 teacher reference，不能删除评价候选分子。训练 loss 必须使用完整候选分子聚合或动态 Top-k hard-negative cache；每 epoch 可刷新 hard negatives，但不重新 outcome-mine action。所有 clean/action/random 三臂共享完全相同候选边界。

### 10.6 第六层：多动作账本在 sampler 中再次坍缩

**表现**：E14 每 query 单 action；E15 从数千 action 缩到 281/198/133；低覆盖 source quota 反向支配样本选择；epoch 内循环回收导致过曝。

**解决**：采样单位分三层：identity → query → action family/step。每个 epoch 每 action 至多一次，不够的 source 不硬补齐；用 inverse-propensity 或 capped identity weights 防止高重复 identity 支配。必须输出每个 action、query、identity 的最大 exposure 以及 curriculum step 覆盖，不能只报总 batch 数。

### 10.7 第七层：优化器与层容量确实有限，但排在最后

**表现**：最后一 block+head、89% clipping、PPT 中峰交互目标难以通过 pooling/最后层小更新实现。

**解决**：只有前六层通过后，才比较最后 1 block、最后 2 blocks 和带 peak-token auxiliary objective 的版本；使用分层 LR、梯度累积和按分支校准后的 clipping。若 matched-random 因子实验都不能证明 targeted advantage，解冻更多层只会扩大错误监督。

## 十一、反复出现的工程错误及其永久修复规则

| 错误 | 历史表现 | 根因 | 永久修复 |
|---|---|---|---|
| HDF5 非递增索引 | `Indexing elements must be in increasing order` | 直接用未排序 query rows 取 HDF5 | 统一 `unique-sort-read-inverse` 读取函数并单测乱序、重复索引 |
| baseline rank mismatch | query 21286、R1 replay 2/882 等 | fresh forward、cache、tie/float 和候选聚合不一致 | 每个 executor 先输出 per-query rank hash；允许数值容差但不允许静默替换任务基线 |
| 状态/文件路径猜测 | A4 status 不匹配、fold4 文件缺失、`held` 未初始化 | 代码硬编码历史目录/schema | 输入 manifest 明确列文件、hash、schema version；启动前只读 preflight 检查所有路径 |
| 输出目录冲突 | completed result refusing overwrite | 固定目录兼作结果与重跑目录 | 每次自动生成唯一 run-id；完成目录永不覆盖；summary 只读聚合指定 run-id |
| validator 逻辑错误 | 所有 gates 为 true 仍抛异常 | 缺总 gate 或布尔聚合错误 | validator 单测同时覆盖 all-true、单项 false、缺字段；训练前本地 synthetic test 必须通过 |
| 任意阈值阻断 | 需要 8 identities、6 multi-action、200 formulas | 把期望样本量写成未核实硬门 | preflight 先报告真实可用数；阈值来自统计功效/协议，不能事后拍定 |
| action 被单选/过滤 | E14 单 action collapse、global safe filter | 数据结构在 builder/sampler 中丢多对多关系 | ledger 主键为 query-action；每阶段校验 multiplicity 分布与 source/action hash |
| 风险与纠正混损失 | E14 risk 6/11、E15 risk 近零 | 同一 loss 路由或数值尺度不匹配 | separate forward/loss/optimizer accounting；校准后报告非零梯度范数和分支余弦 |
| epoch 内过曝 | guided cursor 回绕 | batch 数大于 unique action 却重复抽样 | 无放回 epoch sampler；不足时减少该分支 batch，不循环复制 |
| 校准样本失真 | 每源前 4 个样本 | 非随机、非分层、样本太少 | 至少 32 个分层 microbatches、128 个 identity-action observations，固定随机种子 |
| split 破坏候选图 | training query 丢失所有 negative | 通过删 references 实现 identity holdout | holdout 只隔离监督/teacher identity；候选库保持完整，防止任务被改写 |
| 假 paired control | control loss 约 0、不更新 | control 不是等预算 continuation | control 与 treatment 共享 batch order、steps、optimizer、clean objectives，仅 action 替换不同 |
| 未通过 dev 仍看 held | M3 ineligible 后继续 held | gate 没有硬停 | dev gate false 立即退出且不加载 held 文件；已消费 held 永久标记，不重复调参 |

以后任何大规模提交前，必须自动验证：输入存在与 hash、schema、rank replay、action multiplicity、candidate completeness、三臂 batch equivalence、branch gradient、唯一输出目录和 dev hard-stop。审计不需要堆成大量重复程序，但这些合同必须由一个短、可复用 preflight 执行。

## 十二、明确的保留、暂存与停止清单

### 12.1 必须永久保留的资产

1. 23,876-query/1,805-error 冻结错误图谱、formula folds 和完整候选图；
2. S3A corrected/introduced/net 矩阵与 per-query action ledger；
3. candidate-gradient 3–6、role-confounder 1–5 的成熟路径；
4. A4 exact action/outcome/harm 数据；
5. E1 真实重复谱采集变化分布；
6. R0 的 36,934 action rows、matched controls 与 path payload；
7. E4-A 5×3 结果、checkpoint 和训练配置；
8. contextual peak tokens；
9. recurrent/top3 positive-reference 结果，作为机制证据而非现成训练监督；
10. 所有 introduced-error per-query 记录和 candidate-switch 类型。

### 12.2 立即永久停止的路线

1. 随机加峰、均匀随机删峰、独立 m/z jitter 作为主策略；
2. 整组 conditional/unique/shared 峰删除；
3. role_shared、gradient-only、75%/100% 全局删峰；
4. outcome-aware “任何能修正就进入训练”；
5. clean→action/prototype 的绝对 embedding 模仿；
6. fixed official reference、absolute official target、terminal-only、stopgrad 变体；
7. C2-B/C2-C/P2b residual 在 noise 路线中的任何使用；
8. D1/D1b query-only adapter 冒充 noise embedding；
9. current C1 margin/prototype distillation；
10. legacy E14 trainer 和当前 E15-M3 trainer；
11. 在未闭合归因前的大规模 LR/clip/layer/epoch 网格；
12. 为凑 5 pp oracle union 继续无边界扩 dose/reference/recurrence。

### 12.3 暂存但当前不值得继续投入的路线

1. role_unmatched；
2. A4 nonlinear selector；
3. global consensus projection/intensity transport；
4. global recurrent_union_mix weight scan；
5. reference diversity/maxmin replacement；
6. relaxed recurrence；
7. epoch-wise action online re-mining；
8. 只在 pooling/projection 上做峰 token 小头。

这些路线不是永久证明无效，而是当前 unique coverage、paired increment 或可学习性不足。只有新的机制证据能解除封存，不能因为“还有算力”重跑。

## 十三、下一轮唯一高优先级实验：先闭合因果，再扩大动作空间

### 13.1 实验 A：E4-A 的三臂严格复现

同一 initialization、formula-development fold、batch order、identity/query/action sampler、训练步数、optimizer、完整候选边界，运行：

1. **Clean-only continuation**：相同 clean ranking 预算；
2. **Matched-random noise**：与 target action 完全匹配删峰数、强度、m/z、role 和 curriculum step；
3. **Targeted mature noise**：candidate-gradient 3–6 + confounder 1–5。

唯一主终点是 full-list clean retrieval 上 `targeted - matched_random` 的 formula-cluster paired CI；其次是 `targeted - clean_only`。三臂均报告 corrected/introduced、near、MRR、candidate-switch 和 preservation。若 targeted 不显著优于 matched-random，立即停止扩大学生；这说明现有 +0.635 pp 主要是普通训练/鲁棒性而非动作语义。

### 13.2 实验 B：动作可学习性，而不是再训练一个 oracle selector

对 A 中每个 query/action 构造标签 `target action margin gain - matched random margin gain`。输入严格限制为 clean query 可见信息。formula-crossfit 检验：

- 是否能识别 positive、zero、harmful action；
- 校准后 high-confidence 子集是否同时有足够 identities/formulas；
- 是否对 mature union 提供 unique error coverage；
- 误选 action 为什么产生新错误。

若不可预测，不把该 action 用于 clean encoder；转为解释性/oracle 证据。若可预测，才进入 conditional sampler/no-op routing。

### 13.3 实验 C：完整候选 listwise counterfactual training

在 A/B 通过后，训练时保留完整候选分子及参考谱聚合；hard-negative cache 可按当前 student 刷新，但 action path 固定，防止重新 outcome mining。损失包含：

- clean full-list rank；
- target action full-list rank；
- target-vs-matched-random margin advantage；
- no-op/clean margin floor；
- 独立 harmful/risk protection；
- 真实同身份 view consistency。

这一步直接解决“修正旧 hardest negative、却被新候选顶替”的 candidate-switch 问题。

### 13.4 实验 D：P-arm 单独重建，禁止与 N-arm 过早混合

P-arm 只使用真实 same-identity replicate，区分三类：

1. 原谱已存在但被低估的峰：可直接 reweight；
2. 可由上下文/峰对预测的缺失证据：可做 masked contextual objective；
3. 原谱完全不可识别的缺失峰：只作 headroom/解释，不要求 clean encoder 幻觉补全。

P-arm 必须先单独通过 support-disjoint formula holdout，再与 N-arm 做 2×2 factorial；不能再次直接把 positive graft 固定权重叠到成熟 N 模型上。

## 十四、下一轮的硬停止规则

1. preflight 任一 rank/schema/action/candidate/control gate 失败：不提交训练；
2. action learnability 在 formula holdout 无正向 CI：该 action 不进入 clean encoder；
3. targeted student 不显著优于 matched-random：停止该 noise family；
4. introduced ≥ corrected 或 risk-net ≤ 0：停止该配置；
5. dev 未通过：不读取 held；
6. 单 development fold 未通过：不做 multifold、seed 或 P3；
7. 只改善 action-view、不改善 clean full-list：只能报告 action executability；
8. 任何新扫描若 unique error coverage 低于预注册门槛：停止，不以更多剂量续命；
9. P3 只在冻结模型、冻结协议和所有前置门通过后使用一次；
10. 任何 pp 都必须携带任务协议、输入类型、共享 encoder 状态、是否 outcome-aware 和 CI，不允许跨任务拼接。

## 十五、最终总裁决

此前并不是“噪声路线走错了”，而是同一条正确路线在七个接口上反复失真：有效 action 被当成可学习监督，特权信息被当成 clean 证据，oracle union 被当成模型目标，完整检索被缩成冻结少数候选，多动作被 sampler 压成单动作，risk/control 分支没有真实梯度，工程 gate 又多次改变数据和评价协议。

当前最重要的不是再找到一个看上去能修正几十个错误的峰操作，也不是再提高学习率，而是用 E4-A 已有成熟动作完成第一次真正的 **clean-only / matched-random / targeted student 因果闭环**。这一步通过，噪声微调才拥有可被科学归因的核心创新；这一步不通过，就应果断停止该 action family，而不是继续用 oracle headroom 掩盖 transfer 失败。随后只扩展 clean-input 可识别、full-list 可迁移、unique coverage 足够且 introduced 可解释的机制。这样才能把已经证实的峰级纠错潜力转化为稳定、更强、可用于后续生物学注释的共享 embedding。
