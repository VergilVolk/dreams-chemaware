# Noise L1 结果与 L2 配对反事实共享编码器合同

日期：2026-09-03  
状态：L1 已完成；L2 在任何训练结果产生前冻结

## 0. L2 首次预检失败与修正

首次 L2 在模型编码前触发 `pilot did not retain both mature strategies`。根因是先对全部
高置信 query 做无分层哈希截断，再事后检查 selector，低频 `role_confounder` 可能仅因
1,024-query pilot 截断而消失。这是子采样实现错误，不是动作或 L1 路线失败。

修正后先分别为 `candidate_gradient` 与 `role_confounder` 确定性保留 query 覆盖，再用
剩余名额做全局哈希填充；阈值、动作定义、损失和训练规模均不改变。若某 selector 在完整
outer-train L1 主门下确为零，程序必须在任何 GPU 编码前报告逐 selector 容量并停止，禁止
以放宽阈值伪造双策略覆盖。

## 0B. 第二次预检揭示的真实 family 路由结果

正式 L1 主阈值在 outer-train 实际选择了 3,391 个 `candidate_gradient` query，
`role_confounder` 为 0。第二次失败不是子采样问题，而是原 L2 合同错误地把“L0 中存在
有效动作”偷换成“L1 已证明该 family 可由 clean input 路由”。

修正后的 L2 只激活 `candidate_gradient`。`role_confounder` 仍保留为成熟反事实动作，
但在当前 clean-OOF 主门下路由为 no-op。不得降低冻结阈值强行纳入；只有未来独立的
clean-visible formula-OOF 路由结果通过同等统计门后，才能重新进入共享 encoder 训练。

## 1. 不可改变的路线裁决

噪声微调的两条成熟主策略仍为：

1. `candidate_gradient`，attenuation `0.50`，step `3–6`；
2. `role_confounder`，attenuation `1.00`，step `1–5`。

L0 已在成熟 clean-continuation geometry 上逐条重放 36,934 条 R0 action 及两条
frozen matched-random controls。历史与成熟 paired advantage 的 Pearson 为 0.9491，
非中性方向一致率为 99.44%；candidate-gradient step 3–6 的 formula-cluster advantage
下界均为正，role-confounder step 5 为 17 corrected / 0 introduced。因此，两条动作定义
没有失效，不允许以扩充 P-arm、重新扫描随机噪声或增加学习率替代其转移问题。

## 2. L1 证明了什么

L1 使用五折 formula-disjoint、identity-disjoint OOF，只读取 clean spectrum、成熟 clean
embedding、label-free contextual peak tokens、采集元数据及固定 action family/step。正确/错误
候选、candidate scores、baseline rank/margin、action path/outcome、identity/formula、P2b 和 P3
均不进入特征。

- positive AUPRC 0.7065，超过 family-only 0.6098 和 permutation 0.6142；
- 主 no-op policy 在 10,059 个 action-covered queries 中选择 4,606 个，覆盖 1,302
  identities / 688 formulas；
- 冻结动作结果为 72 corrected / 9 introduced，risk-net(lambda=2) 为 54，near delta
  为 +0.795 pp；
- 所有预注册 L2 放行门通过。

这只证明 clean-input action suitability 可跨 formula 学习；它不是新的 embedding，也不能把
子集 +0.626 pp 或 63 个净纠正冒充 23,876-query 全任务共享编码器增益。

## 3. 对历史 E4-A 的准确归因

历史 E4-A 的两个策略确实进入共享编码器训练，但只作为 additive augmented views：
`clean rank + action rank + consistency + margin floor + preservation`。目标中没有显式的
`targeted margin > matched-random margin`。严格三臂结果为 clean duplicate +0.5065 pp、
matched random +0.5065 pp、targeted +0.5403 pp；targeted 相对 matched random 仅 +0.0338
pp，formula CI 下界为 0。

因此主要错误按优先级为：

1. 训练目标没有约束定向动作优于等剂量随机动作；
2. fixed curriculum 没有根据 clean input 区分 positive / neutral / harmful action；
3. candidate-conditioned 特权动作没有通过合法的 clean-input 路由桥接到推理；
4. action sampling 未覆盖完整多动作结构，后续 E14/E15 又发生单动作坍缩和过度过滤；
5. 训练候选边界与最终 full-list 候选切换没有充分对齐；
6. harmful/no-op 风险分支没有形成独立且非零的保护梯度。

## 4. L2 唯一允许的实验

L2 是一个小规模、单 development formula fold、严格配对的共享 DreaMS encoder 试验：

- treatment：L1 OOF 高置信通过的 mature targeted actions；
- control：同一 query/action row 的预先冻结 matched-random control；
- 两臂共享初始化、训练 query、候选 references、batch order、optimizer steps、学习率、
  unfreeze 层、clean/listwise loss、no-op/risk protection 和 action exposure；
- 唯一差异是 targeted payload 与 matched-random payload；
- 主损失必须显式包含 paired counterfactual advantage，而不是只把 target 当普通增强；
- 推理和评价只输入 clean spectrum，query/reference 始终使用同一 encoder；
- 主终点是 full-list clean retrieval 的 `targeted - matched_random` formula-cluster paired CI。
- 同时报告相对 clean-continuation 初始化及官方 DreaMS 的结果用于定位总性能，但这两项不得替代
  `targeted - matched_random` 因果主终点，continuation 收益不得记入定向噪声净贡献。

L1 的 policy 汇总为方便审计而每 query 最多选择一条动作；L2 训练不得照搬为单动作账本。
所有满足 OOF 主门的 action rows 都应保留，同 query 可保留多个 family/step；采样按 query
聚合、按 clean-OOF 预测效用排序轮换、每 action 每 epoch 至多一次；每个 identity 每 epoch
最多贡献一个 query，并报告 action/query/identity 最大曝光。禁止把多动作无优先级平均灌入。

## 5. 分支与损失合同

每个训练单元必须成组包含 clean、targeted、其冻结 matched-random controls 和候选集合：

1. `L_clean_full_list`：clean query 的 molecule-level full-list ranking；
2. `L_action_rank`：action view 的候选 ranking；
3. `L_paired_advantage`：同 query 下 target margin 必须高于 matched-random margin；使用平滑
   paired hinge，并对 comparator stop-gradient，避免已经越过 margin 的样本令该分支整批
   静默为零，也避免通过主动破坏 matched-random comparator 虚构优势；
4. `L_no_op_floor`：未获高置信动作的 query 保持 clean margin；
5. `L_risk`：对 L0 harmful actions 独立保护 clean/full-list margin，不把 harmful payload
   当正教师；
6. 弱 consistency/preservation 仅作安全正则，不能主导动作语义。

训练前必须报告各分支梯度 norm、cosine、clip fraction；paired-advantage 或 risk 梯度为数值零
时禁止启动训练。

## 6. 训练数据和防泄漏合同

- L2 development fold 在启动前冻结；outer-held formula 的 action outcome 不进入训练；
- outer-train actions 必须使用不包含自身 formula 的 L1 OOF predictions；
- L1 动作表只覆盖有动作的 query，不能用其 formula 子集冒充完整 held fold；最终检索必须按
  D0 的冻结 `seed=20260825` 公式折规则覆盖该折所有 query；
- 不以 held identity 删除正式候选分子；拆分只隔离监督与 teacher source；
- matched controls 使用 L0 已冻结路径，不读取 target outcome 后选择；
- candidate hard-negative 可在共同初始化下构建，但 treatment/control 必须完全相同；
- no P2b、no P3、no ChemAware、no P-arm、no post-outcome action mining。

## 7. 放行与停止规则

只允许先运行一个小 pilot。进入多折前必须同时满足：

1. treatment 相对 matched-random 的 clean full-list Recall@1 formula CI 下界 > 0；
2. paired corrected > introduced 且 corrected - 2*introduced > 0；
3. near 不退化；
4. preservation mean >= 0.995；
5. targeted action-view 优势确实转移到 clean-query margin；
6. 每个动作、query、identity 曝光符合上限，无回卷、无单动作坍缩；
7. 至少两个训练 seed 方向一致后，才允许 multifold。

失败时先按“动作选择、paired loss、full-list candidate switch、risk gradient、采样曝光”归因，
禁止用 P-arm、更多 dose、更多 epoch、学习率扫描或解冻更多层掩盖失败。
