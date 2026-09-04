# 噪声微调当前上下文快照

> 2026-09-04 更新：本文早期的 F1-F5 执行顺序已被后续 E4-A causal、B0-B2、L0-L2 证据取代。当前唯一实施合同为 `docs/NOISE_FINAL_DYNAMIC_CONDITIONAL_DIRECT_FINETUNING_CONTRACT_20260904.md`。主线是动态条件加权的直接共享 embedding 微调；CPG 蒸馏降为通过直接训练门后的条件后备。不得再按下方历史顺序重启 F1 蒸馏或 E9 在线动作路径重挖。

**唯一目标：** 原始 MS/MS 谱图经过同一新编码器，输出改进后的统一 embedding；当前停止所有 P2b、RAW/token residual 和模块2工作。

## 已封存证据

- D0：23,876 queries、2,522 identities、1,082 formulas；官方错误1,805，near错误1,446；P-arm 80,250例，N-arm 4,916动作。
- C1 P-arm教师：支持谱与评价正谱逐行互斥；+2.47 pp，near +2.34 pp，2,382/396；这是教师上限，不是模型成绩。
- A4/N-arm：精确峰动作和动态矩阵已证明动作效应；candidate-gradient 50%与role-confounder是主要安全动作来源；shared-only和全局100%删除禁用。
- D1b：clean-only query adapter，不是真正噪声训练。三seed总体净修正+19/+19/+10，平均约+0.067 pp；第三seed near净减少6，未通过near一致性门。
- P2b/C2-C：embedding后模块，全部暂停，不得进入噪声教师、标签或损失。

## 不可再混淆

1. 动作/身份教师上限不等于学生模型性能。
2. 噪声微调必须让clean原始谱图输出新embedding。
3. 查询和谱库必须由同一编码器重编码；query-only校准不是完整统一空间。
4. 规则只能用于峰保护、动作风险和概念辅助，不能定义身份或距离。

## 当前执行顺序

1. F0：锁定对称编码、P3隔离、zero-init复现和逐query错误转移协议。
2. F1：单独把C1 P-arm教师蒸馏给clean谱图学生。
3. F2：依次加入confounder、candidate-gradient 50%和A4安全动作N-arm。
4. F3：组合已单独过门的P/N分支。
5. F4：仅在教师有效且adapter容量不足时解冻最后一个Transformer block。
6. F5：最后加入ChemAware规则峰保护与概念解码。

## F1硬门

- 模型输入：clean原始谱图；输出：候选无关的新embedding。
- positive-deficit、cross-condition教师迁移有效；overall不劣；near不显著下降。
- corrected>introduced，报告`corrected-2*introduced`。
- formula OOF、identity-equal、三seed；逐query保存旧/新正负分数与候选身份。
- P2b完全禁止；封存测试在模型和阈值冻结后只运行一次。

详细执行合同：`docs/NOISE_FINETUNE_EXECUTION_CORRECTION_20260826.md`。

## F1 v1 pilot 裁决（2026-08-26）

- 单 seed / fold0，最佳 epoch=3；完整自然候选图 outer Recall@1 仅 `+0.000338`（约 `+0.034 pp`），11 修正 / 9 新增，near 约 `+0.083 pp`，preservation=`0.99715`。
- 该结果仅证明共享 query/reference 峰级 adapter 可训练且未明显破坏原空间；量级远小于 C1 教师的 `+2.47 pp`，不构成有效性通过，禁止扩展为 5 folds x 3 seeds。
- 核心协议错位：C1 教师的增益定义在“隐藏其余同身份支持谱、只保留一个评价正谱”的 held-out-positive challenge；F1 v1 却只按完整自然图（基线约 94%）选择 epoch，困难正例信号被容易正例稀释。
- F1 v2 改为双协议：C1 challenge 负责选择 epoch；完整自然图仅作 preservation、overall、near 和风险净收益安全门。两者均使用同一个共享编码器输出，P2b 仍完全禁用。

## F1 v2/v3 机制裁决（2026-08-26）

- v2 通用 hard-rank：完整图和 challenge 的普通 Recall 略升，但 challenge 身份等权 `corrected-2*introduced` 始终为负，故最佳 epoch 正确回退到0。
- v3 候选 margin 蒸馏：69.4%训练候选对具有正教师 margin 增量，均值约0.031；损失正常下降，但 challenge 身份等权 Recall 反而下降约0.20–0.25 pp，仍回退到0。
- v2/v3共同问题：主监督覆盖三万多条“教师略微改善但基线已正确”的例子，真正“基线错误且教师纠正”的例子占比很小；主动移动已正确样本造成新增错误抵消修正。
- F1 v4 决策：主纠错梯度只来自 `baseline wrong -> teacher correct`；`baseline correct`只进入等量 safety stream，且优先纳入教师会破坏的样本。新增训练折 rescue 审计，用于区分优化失败与跨formula泛化失败。

## E8–E10 当前裁决（2026-08-28）

- E8 在 formula fold 0 上得到当前最强的真实共享 embedding：`curriculum + symmetric + shared` 相对官方 Recall@1 `+0.574 pp`、near `+0.609 pp`、MRR `+0.386 pp`，38 修正 / 4 新增，preservation `0.99527`。这是模型权重结果，但仍只是单个开发折。
- E8 因子实验已经否定三条错误归因：stop-gradient action target没有增益；冻结 official reference anchors 显著有害；terminal-only 显著有害。因此保留共享 query/reference 更新与多步 curriculum。
- E9 在同一个 E8 权重上在线重挖成熟 N-arm 动作，只得到 10 修正 / 9 新增，formula-cluster CI 跨0。动作路径存在变化，但 online 与 frozen 的最终排序近乎等价；动作陈旧不是主瓶颈。
- 原 E9-B 把 `40/2293=+1.744 pp` 的动作覆盖子集增益与完整 `5923-query` 的五点目标比较，分母不一致。正确的完整任务残余上限是 `40/5923=+0.675 pp`；连同 E8 已实现的34个净修正，现有 candidate-gradient / role-confounder N-arm 的总 oracle 上限约为 `74/5923=+1.249 pp`。
- 因此禁止继续扫描 N-arm 学习率、权重或在线重挖。下一门 E10 必须在成熟 E8 几何中重新测量 P-arm：真实同身份正谱引导的 `consensus_projection` 和 `recurrent_union_mix`，并以当前最难错误候选构造方向对照。
- E10 的训练准入是双门：至少一个固定 P-arm cell 同时满足正向 formula CI、`corrected > 2*introduced` 和方向特异性；P+N+no-op 的完整 5923-query oracle 总容量达到5 pp。任何一门失败，都不能把历史 frozen-action oracle 数字再次当成共享学生的训练承诺。

## E9-B v2 / E10 正式结果（2026-08-28）

- E9-B v2 已在完整5,923-query分母复现：成熟N-arm相对E8最多额外修正40个、`+0.675 pp`；E8+N oracle相对官方总计`+1.249 pp`，formula CI全正但远低于5 pp。因此继续扫描candidate-gradient/confounder权重没有容量依据。
- E10 的 `recurrent_union_mix@0.50` 在成熟E8几何上固定动作增益`+0.878 pp`，59修正/7新增，risk-net=45；formula CI `[+0.539,+1.306] pp`，相对wrong-direction control的formula CI也全正。这证明真实同身份重复谱中的稳定缺失峰具有强且特异的P-arm信号。
- E10 的七cell + N-arm + no-op oracle 相对官方达到`+3.377 pp`，formula CI `[+2.439,+4.545] pp`，但未达到5 pp。该结果仍是outcome-aware容量上限，不是新模型权重。
- E10尚未覆盖历史已通过的 `matched_intensity_transport`、`recurrent_peak_graft`、`balanced_peak_exchange`，也没有测量“强度校正后再补稳定缺失峰”的顺序组合，因此不能据E10宣布完整P-arm容量不足。
- E10-B固定扩展为19个positive cells及19个wrong-direction controls：完整复测E10七cell，补入上述三个历史动作，并预注册四个强度/缺峰顺序组合。只有扩展联合容量达到5 pp且存在固定安全cell，才进入条件噪声训练。

## E10-B 正式结果与 E11（2026-08-28）

- E10-B 的19个positive cells中13个通过固定安全与方向特异性门，证明P-arm不是少数剂量的偶然信号。最佳cell仍为`recurrent_union_mix@0.50`：相对成熟E8 `+0.878 pp`、59修正/7新增、risk-net=45。
- 新增成熟动作没有超过最佳单cell，但为联合oracle增加25个独特修正：E10-B相对成熟E8恢复191个，完整E8+P+N相对官方总容量`+3.799 pp`，formula CI `[+2.701,+5.080] pp`。
- 5 pp在5,923-query折上约需297个净修正；当前总计225个，仍缺72个。该缺口不能通过继续扫最佳cell剂量解决，因为新增动作的价值已表现为互补覆盖而非单cell峰值。
- E10/E10-B所有P-arm参考都取当前embedding最相似的3张同身份谱，可能系统性排除跨仪器/跨CE远端真实变体。E11只改变参考集合，不改变成熟动作：farthest-3、embedding max-min-6、采集条件分层6、max-min-12；每种参考策略固定搭配四个E10-B强动作及wrong-direction control。
- E11仍在完整5,923-query分母上复现E10-B联合，再报告新增独特修正和总容量。若参考多样性仍不能达到5 pp，则下一步必须对剩余oracle错误做新机制审计，而不是继续堆相似动作。

## E11正式结果与E12-A（2026-08-29）

- E11精确复现E10-B联合，参考多样性新增39个独特修正、无oracle损失，增量`+0.658 pp`，formula CI `[+0.379,+0.989] pp`。
- E8+E9/E10-B/E11完整联合相对官方达到`+4.457 pp`，formula CI `[+3.250,+5.862] pp`；点估计仍未达到5 pp。当前净恢复264个，5 pp约需297个，缺33个。
- 仅`maxmin6/12 + recurrent_union_mix@0.50`两个多样参考固定cell通过；最佳maxmin6为55修正/20新增、risk-net=15，安全性显著弱于top-3 recurrent的59/7、risk-net=45。因此参考多样性只能作为条件动作，禁止全局替换top-3。
- E11后仍有133个oracle错误。E12-A先做无新动作结果的可达性审计：排除正谱自匹配后测量真实同身份支持教师是否可纠正，并统计top-3/farthest/maxmin/condition参考在0.67、0.50、0.34支持率下可提供的稳定缺失峰数。
- E12-B准入要求：支持互斥正教师和50% recurrence eligibility均至少覆盖缺口33个。未过门则换残余机制，不盲目放宽阈值。

## E12-A正式结果与E12-B（2026-08-29）

- E11剩余133个oracle错误中111个为near；89个positive-deficit-only、18个both，合计107/133含positive-deficit。残余空间仍主要是正证据不足，不是纯negative-excess。
- 排除参考谱自身候选后，106个残余错误具有支持互斥正教师，82个可被其纠正，显著超过5 pp尚缺的33个。这证明残余监督容量真实存在，并非自匹配伪影。
- 0.67 recurrence仅84个残余错误有候选缺失峰；放宽到0.50后为127个，其中104个至少5峰、77个至少10峰。0.34相对0.50仅增加约1个，故排除0.34。
- E12-B固定为50%支持率；五种参考策略分别测试max=5/10与dose=0.25/0.50，并增加一个max=10、dose=0.50的support-weighted安全版本，共25个positive cells及25个wrong-direction controls。
- E12-B必须完整复现E11联合，所有固定cell在完整5,923-query上报告修正/新增、formula CI与方向特异性；只有固定安全cell和总容量5 pp同时通过，才进入条件共享编码器训练。
# 2026-09-04 dynamic-direct implementation checkpoint

- 主线仍是直接更新同一个 clean-spectrum query/reference DreaMS encoder；P2b、ChemAware、CPG residual 均不进入当前训练输入或损失。
- 已实现 model-free provenance preflight、动态权重核心、无重复分层 sampler、完整 30-cell N/P outer-train ledger 和独立验证器。
- 30 cells = N 的 9 个成熟 cells + P intensity/consensus 的 12 个 cells + P real-replicate peak-transfer 的 9 个 cells；禁止读取 `passing_cells`、`best_fixed_cell` 或 oracle union 删 cell。
- 纠正了一项提交前发现的关键 provenance 错误：L0/L1 的 action label 定义于 E4-A causal clean-duplicate fold-0 geometry，因此首轮不能用 high-LR multifold fold-1 checkpoint。首个开发试验冻结为 outer fold 0 / seed 20260828 / L0 指向的 exact clean-duplicate checkpoint。
- 权重实现也已纠正：不再把 identity/formula/family 逆频率直接乘进 action utility；那会覆盖动态信号。action utility 保持单调，等暴露由 sampler/层级聚合负责。
- 当前唯一获准作业为 `sbatch tasks/run_noise_final_dynamic_direct_preflight.sbatch`。该作业先运行 CPU 数值测试和静态实现审计，再验证所有 SHA/schema/fold，最后建立 outer-train 30-cell ledger；它不加载 DreaMS，不产生新的 embedding。
- 该作业通过后，下一步才是 GPU action/control replay + tiny overfit，再生成 Phase A 四臂正式训练。不得把 preflight/ledger PASS 写成性能提升。
