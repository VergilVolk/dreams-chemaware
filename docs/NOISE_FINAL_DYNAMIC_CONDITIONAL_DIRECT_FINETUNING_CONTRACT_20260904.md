# 噪声路线决战合同：动态条件加权的直接共享 embedding 微调

日期：2026-09-04  
状态：冻结实施指导；未经本合同规定的门，不得转向新架构或扩大训练  
范围：只改变统一 DreaMS query/reference embedding；P2b、候选重排器、ChemAware 和 P3 均不参与本阶段

## 0. 本合同取代什么

本合同不推翻既有动作矩阵、E4-A、E8、L0 或 L1。它取代以下已经失败或优先级不当的训练路线：

1. 把全部成熟动作当作同质 augmented views 的固定平均训练；
2. 把每个 query 压成一个 outcome-mined action；
3. 用一个全局 target-minus-random 梯度方向要求所有 formula 一致；
4. L2 中由巨大 paired-advantage loss 主导并造成全程 clipping 的实现；
5. 在 clean 可识别性尚未闭合前，枚举数十万逐候选 residual 的 CPG0 全量教师；
6. 重新扫描动作路径、剂量、学习率或重新建立已经完成的动作矩阵。

本阶段只有一个目标：在不依赖推理期候选专家的条件下，把已经验证的 N/P 峰级反事实信息更高比例地转化为 clean-spectrum 共享 encoder 的检索能力。

## 1. 不可改变的实证起点

### 1.1 已经成立

- 正式开发图为 23,876 queries、1,805 个 official DreaMS Top-1 errors；约 79.7% 的错误包含 positive deficit。
- E4-A 的真实共享 encoder 在 5 formula folds × 3 seeds 上平均 Recall@1 `+0.635 pp`、near `+0.522 pp`，15/15 方向为正。这是模型权重结果。
- 成熟 N 原语为 `candidate_gradient a=0.50 step 3-6` 与 `role_confounder a=1.00 step 1-5`。`role_shared` 永久禁用。
- P 动作中 consensus/intensity 与 real-replicate recurrent families 存在显著固定反事实效应；它们必须区分“原谱可见证据”和“真正缺失证据”。
- L0 在同一成熟 geometry 上保存 36,934 个 N actions；目标动作总计 564 corrected / 229 introduced，证明动作空间有正、负和异质效应。
- L1 仅使用 clean spectrum/contextual tokens，在 formula OOF 上预测正动作的 AUPRC 为 `0.7065`，高于 family-only `0.6098` 和 permuted `0.6142`。因此 clean 输入包含非零的动作可学习信号。

### 1.2 已经证伪

- E4-A 严格 causal three-arm 中，targeted 相对 matched-random 仅多净改正 2 个 query，即 `+0.0338 pp`，formula CI 下界为 0。旧固定训练不能把其大部分增益归因于动作语义。
- B2 的三个确认配置均未通过跨 formula 全局梯度共识。不同 formula 不应被强迫共享一个动作方向。
- L2 只激活 candidate-gradient、丢失 role-confounder，targeted 相对 matched-random 只有 4 corrected / 3 introduced，且每轮 clip fraction 为 1.0。它否定的是该静态阈值和损失实现，不是否定 clean-visible 条件选择。
- E9 的 online action-path re-mining 相对 frozen path 不显著；动作路径陈旧不是首要瓶颈，不再重挖路径。
- E6 的 instance outcome mining 主要选入 A4 单峰和全局有害的 role_shared，显著弱于 fixed curriculum。逐 query 看见结果后选动作不是可迁移的数据策略。

## 2. 对上一版提议的四项修正

### 2.1 不做逐样本二阶/全参数在线影响估计

117M 参数模型上逐 action 计算并保存全梯度既昂贵，也会重复 B0-B2。动态性采用两层低成本信号：

1. **clean-visible prior**：formula-crossfit 的 clean contextual feature 预测；
2. **current-geometry evidence**：在训练 batch 内无梯度记录 target 与冻结 matched-random controls 的完整候选 margin 差，作为下一 epoch 的滞后一阶校准量。

权重在一个 epoch 内冻结，禁止同一步用刚观察到的结果立即改自身权重。epoch 0 使用冻结 mature checkpoint 的 outer-train replay；随后只复用训练本来需要的 target forward，并为固定 controls 增加批量 no-grad forward，不重新编码 491,100 个全动作矩阵。不为每个 action 求 Hessian，不保存全参数梯度，不用 held fold 调权重。

### 2.2 P 臂不是末位附加项

残余错误以 positive deficit 为主，P 臂与 N 臂同级进入设计，但分成：

- `P-observed`：原始 query 已存在的峰，学习强度、峰间上下文和真实 replicate 一致性；
- `P-predictable-missing`：只有当 clean contextual tokens 在 formula-crossfit 中能预测其收益时，允许作为 masked/contextual augmentation；
- `P-unidentifiable-missing`：clean 输入无法识别的真正缺失峰，只保留为 headroom/解释证据，不监督 clean encoder 幻觉补峰。

### 2.3 动态更新的是训练权重，不是动作路径

N/P action payload、剂量和 matched controls 均从既有冻结 ledger 重放。训练中只更新软权重与当前完整候选分数；不重新搜索峰路径，不产生新的 outcome oracle。

### 2.4 CPG 蒸馏降为条件后备

直接微调是主线。只有在“某个 action family 的 clean-visible soft policy 显著、直接训练也显著，但 clean transfer 比 action capacity 低很多”时，才允许用 CPG residual 做压缩/增强。不得先运行 CPG0 全量枚举来代替直接训练。

## 3. 唯一允许的算法

### 3.1 数据与拆分

1. 外层拆分单位为 molecular formula；outer-held 只用于最后评价。
2. outer-train 内再固定 K 个 formula folds，用于产生 action soft-weight 的 crossfit 预测。
3. 任一 query 的 action outcome、identity 或 formula 不得进入预测自身权重的拟合折。
4. identity equalization 在 loss 聚合中实现；不能通过删除 candidate references 实现 holdout。
5. 完整候选谱库保持不变。训练监督可隔离，正式候选边界不可删减。

### 3.2 动作集合

#### N arm

完整保留 9 个成熟 cells：

- candidate-gradient：attenuation 0.50，step 3、4、5、6；
- role-confounder：attenuation 1.00，step 1、2、3、4、5。

同一 query 的多个 action 保留，不做 single-action collapse；no-op 永远存在。

#### P arm

P 侧必须区分“配方复用”和“结果行复用”：E10-B/E11/E12 的 held-fold outcome matrix 只证明过容量，不能作为 outer-train 的训练样本或权重标签。首版复用其冻结峰操作配方、真实同身份参考选择规则和 wrong-direction control 代码，在 outer-train 上重新物化 P actions，并在 inner formula folds 内产生 crossfit 标签；不读取 outer-held 的 `best_fixed_cell`、`passing_cells` 或 oracle union 来删 cell。每条 action 标注：

- clean peak 是否原本存在；
- evidence availability 类别；
- same-identity reference、采集关系、prevalence 与 payload；
- matched wrong-direction control；
- 是否属于真正不可识别的 missing evidence。

不可识别的 missing evidence 不进入 clean transfer loss。

### 3.3 Soft weight，不做硬阈值坍缩

对 action `a`、clean query `q`，定义：

- `p_clean(a|q)`：由不含该 formula 的训练折拟合的 clean-visible 正收益概率；
- `u_t(a,q)`：只在 outer-train 上计算的当前 encoder 在 epoch `t` 下，target action 相对两条 matched-random controls 的 molecule-level margin 增量；它是训练期校准量，不是推理期 selector；
- `r(a,q)`：只由 outer-train 的历史/当前 harmful、candidate switch 和 protected-correct margin 风险得到的惩罚。

最终权重为连续、有界、单调的函数：

`w_t(a,q) = cap[p_clean × sigmoid(u_t / tau_t) × (1-r)]`

其中 `tau_t` 只由上一 epoch 的 outer-train `u_t` 稳健尺度（预注册为 IQR）确定，并在当前 epoch 冻结。采用连续 sigmoid 而不是 `positive_part`：负但接近零的动作仍保留低曝光，避免训练初期某个成熟 family 因估计噪声被硬清零；明显有害动作则由 `r` 和 no-op 吸收。未在上一轮观察到的 action 使用同 family × clean-score decile × baseline-margin decile 的收缩估计，不能用 held outcome 回填。

合同要求：

- 不设置“只剩一个 family”的硬门；
- 每 query 权重和有上限，避免多动作 query 过曝；
- 动作权重只表达同一 family 内的条件效用；identity、formula 和 family 的等暴露由分层采样器实现，不把逆频率乘进动作效用；
- 报告 effective sample size、零权重比例、各 family 曝光和最大单 query 曝光；
- 任一 family 的有效样本量或总曝光低于预注册下限时，判为该 family 暂不可学习；不得用其他 family 冒名替代，也不得临时降低阈值。

首版允许使用固定的、预注册的简单单调函数；不得先训练复杂神经 selector。只有简单软权重通过后，才讨论更复杂 gating。

### 3.4 直接共享 encoder 损失

所有可训练分支作用于同一个 query/reference encoder。主损失为：

1. `L_clean_full_list`：clean query 对全部候选分子的 molecule-level listwise loss；
2. `L_target_full_list`：按 `w_t` 加权的 target action view 完整候选 listwise loss；
3. `L_real_view`：真实 same-identity replicate 的一致性/对比损失，只覆盖合法 P-observed 或可预测分支；
4. `L_safe_margin`：保护 baseline-correct、历史 introduced 和低 margin query，不允许其当前正负 margin 低于冻结安全基线；
5. `L_preserve`：对未被选中的 clean spectra 维持初始化 geometry。

matched-random controls 的角色是估计 `u_t` 和提供因果训练对照；不得通过反向传播故意把 random 分支变差。不存在 L2 式无界 `target-random` 巨大梯度项。

### 3.5 候选边界

- 每 query 使用 official frozen graph 中完整且完全一致的候选 molecule 集；这里的“完整”不是另造一个全局谱库。同一 molecule 的多张参考谱按冻结 official 协议聚合。
- hardest negative 由当前 full-list forward 的候选分数自然确定，而不是永久固定为 official DreaMS 的一个负谱。
- 这不是 E9 的 action-path re-mining：峰动作路径保持冻结，只让检索 loss 看见当前竞争边界。

### 3.6 优化范围

- 初始化使用与 outer fold 严格匹配、provenance 验证通过的成熟共享 checkpoint；不能用 fold 0 checkpoint 代替其他 fold。
- 首版保持 E4-A 已验证的 projection head + final Transformer block；不同时改变层数。
- 学习率、epoch、batch order 和 clean/action exposure 复用成熟 E4-A 配方；不先做 LR sweep。
- action 分支在至少 32 个分层 microbatches 上校准到不会淹没 clean branch；必须报告未裁剪梯度范数与 clip fraction。
- 若 clip fraction 接近 L2 的 1.0，训练立即失败，不得用最终指标掩盖。

## 4. 首轮最小因果实验，而非大网格

为同时隔离“定向 payload”“动态权重”和“N/P 互补性”，首轮分为两个顺序阶段，禁止一次启动大网格。

### 4.1 Phase A：四臂因果内核

只使用一个在运行前写入 use-history ledger 的开发 outer formula fold 和一个 seed，运行四个同预算 arms。该 fold 的结果属于模型选择，不是独立确认；历史上已用于 P 动作选择或 E14 事后修订的 fold 不得伪装成新 confirmatory fold：

实施冻结补记（2026-09-04）：首轮使用 `outer fold 0 / seed 20260828`，并且只允许以 L0 provenance 指向的 E4-A causal `clean_duplicate` checkpoint 初始化。原因是 L0/L1 的成熟动作优势就在该 geometry 上定义；其他 high-LR multifold checkpoint 在重新回放标定前不得混用。fold 0 已被历史开发使用，因此本轮明确属于开发性因果试验，不是独立确认。

1. `C0 clean-continuation`；
2. `C1 weight-matched random-control`：使用动态 NP 的逐 query 权重、剂量和曝光，但替换为冻结 matched-random payload；
3. `C2 static-target`：使用相同 N/P target payload、剂量和总曝光，但使用预注册的 family-balanced 静态权重；
4. `NP dynamic-direct`。

四臂共享初始化、query 顺序、candidate batches、optimizer steps、学习率、解冻层和随机种子。

Phase A 分别回答：

- `NP-C1`：定向 payload 是否超过等剂量随机增强；
- `NP-C2`：动态 soft weighting 是否超过相同 target actions 的静态平均；
- `NP-C0`：最终共享 embedding 是否超过普通 continuation。

`NP-C1` 是方向性主检验，`NP-C2` 是动态加权机制检验；两者在运行前固定，并以 formula cluster 为统计单位做联合单侧 max-T（或等价的预注册 Holm）校正。`NP-C0` 是必要的性能支持证据，不替代前两个配对归因检验。

### 4.2 Phase B：只在 Phase A 通过后做 N/P 消融

仅追加两个同预算 arms：

5. `N dynamic-direct`；
6. `P dynamic-direct`。

Phase B 回答 N/P 独立增量、互补或冲突。若 Phase A 未通过，不运行 Phase B，也不以 N/P 单臂的偶然结果挽救 NP。

不在首轮加入 CPG、ChemAware、额外层、额外学习率或新 action family。

## 5. 放行与停止门

### 5.1 工程门

任何训练前必须由一个短 preflight 完成：

- 输入文件存在、schema 与 SHA；
- fold-matched checkpoint 与 decision 对齐；
- clean fresh-forward rank replay；
- target/control payload replay；
- action multiplicity、no-op、family counts；
- outer-held outcome 未被读取；
- full candidate completeness；
- Phase A 四臂以及获准后的 Phase B 两臂 batch/sample/step equivalence；
- unique output directory；
- synthetic tiny overfit 和 all-true validator 测试。

任一失败则不加载 117M 模型、不启动正式训练。

### 5.2 首轮科学门

NP 进入第二 seed 必须同时满足：

1. 多重性校正后，`NP - C1` 的 full-list Recall@1 formula-cluster CI 下界严格大于 0；
2. 多重性校正后，`NP - C2` 的 full-list Recall@1 formula-cluster CI 下界严格大于 0；
3. `NP - C0` 点估计为正；
4. corrected > introduced，`corrected - 2×introduced > 0`；
5. near Recall@1 与 MRR 不下降；
6. preservation 不低于成熟协议门；
7. clip fraction 不出现系统性饱和。

Phase A 通过后才运行 N/P 消融；至少一个 N/P 分支相对其 matched-random counterfactual 有正增量，且两者组合不劣于较强单臂，才授权第二 seed。

第二 seed 重复通过后，才运行其余 outer folds。P3 在模型、阈值和配方全部冻结后只允许使用一次。

### 5.3 失败后的唯一解释路径

- `C1≈NP>C0`：只是一般增强/鲁棒性，定向语义未转移；停止该条件策略。
- `C2≈NP>C1`：target actions 有效，但动态权重没有增量；保留静态 target 基线并停止动态 weighting。
- `NP>C1` 但 `NP≤C2`：定向 payload 有用，动态校准无用或有害；不得把收益归因于动态选择。
- `N>C1`、`P≤C1`：保留 N；把 P 拆回 observed/predictable/unidentifiable，不调 N 学习率救 P。
- `P>C1`、`N≤C1`：保留 P；N 仅作安全/解释动作。
- `N、P 单独正而 NP 负`：机制冲突；分阶段 curriculum，不做线性混合。
- train action 改善、clean held 不改善：clean transfer 失败；不得扩大数据或改写成 embedding 成功。
- clean predictor 有效、直接训练仍只转移很小比例：此时才允许小规模 CPG residual fallback。

## 6. 为什么这条路线有最高现有依据，但不能承诺必然 +5 pp

内部证据链已经逐项成立：共享 encoder 可改善、成熟动作有容量、clean 输入可预测动作价值、旧全局梯度与静态硬选择失败。新路线只补最后缺口：把 action value 作为随当前 geometry 变化的、formula-crossfit 的连续训练权重，直接优化 clean full-list embedding。

近年方法学也指向同一方向：

- DreaMS 本身通过峰遮蔽、m/z 扰动和 contrastive triplet fine-tuning 直接塑造谱图 embedding，证明 end-to-end input augmentation 的基本范式成立：<https://www.nature.com/articles/s41587-025-02663-3>
- LESS 使用 optimizer-aware 低维梯度特征选择少量高价值微调数据，5% 数据常可超过全量：<https://proceedings.mlr.press/v235/xia24c.html>
- Adapt-infinity 指出静态 importance 在模型状态和任务分布变化时经常失效，采用分群与动态选择：<https://proceedings.iclr.cc/paper_files/paper/2025/hash/a6610efd6c767f63343a4ab28505212e-Abstract-Conference.html>
- Dynamic Loss-Based Sample Reweighting 支持随训练状态更新实例权重并压低冗余样本：<https://proceedings.iclr.cc/paper_files/paper/2025/hash/ded26b348d55953a4863d41540b7d5c4-Abstract-Conference.html>
- Train on Validation 提供了用少量目标分布变化筛选最有益训练样本的快速思路：<https://proceedings.iclr.cc/paper_files/paper/2026/hash/1c3d419b754cb4de0a67a453cb28d959-Abstract-Conference.html>
- 2026 augmentation model-selection 工作支持不再依赖固定 augmentation 网格，而将增强策略与模型状态共同优化：<https://proceedings.iclr.cc/paper_files/paper/2026/hash/a9168f1c54e5147027f1e8cf83e1a775-Abstract-Conference.html>
- TADA 只对训练早期尚未学会的样本做 targeted augmentation，而不是全量同剂量增强；其任务与模态不同，只支持“选择困难样本再增强”的设计原则：<https://proceedings.iclr.cc/paper_files/paper/2026/hash/98bf3b8505c611ac21055dd9d355c66e-Abstract-Conference.html>

外部工作只支持设计原则，不能替我们证明 MS/MS 上的 +5 pp。这里可以承诺的是：下一轮不重做已完成实验、每个增量有配对对照、失败能定位、正式大规模训练前能停止。不能承诺的是尚未观测到的性能数字。

## 7. 复用资产与待取得制品

本地代码已经明确引用、后续需要从服务器核实并按原目录取得的类别为：

1. frozen full candidate graph 与 official embedding cache；
2. fold-matched E4-A/E8 shared checkpoints、decision 与 held per-query ledgers；
3. R0 mature N actions 与 matched-control paths；
4. L0 full-candidate action ledger；
5. L1 formula-OOF clean-visible predictions 与 report；
6. E10-B/E11/E12 的 P recipe/report/matrix 制品：recipe 与 reference-selection 代码可复用；held per-action outcomes 只用于历史证据核对，禁止作为新 outer-train 标签；
7. contextual peak-token cache；
8. introduced-error/candidate-switch ledgers。

下载前必须在服务器用只读 listing 验证实际存在的绝对目录、文件名、大小和 SHA；本合同不凭本地残缺目录猜服务器路径。CPG0 大型失败中间缓存不属于首轮必需下载项。

## 8. 实施顺序

1. 服务器只读盘点并下载上述既有制品；
2. 写一个统一 artifact inventory，固定文件、SHA、shape、fold 与 provenance；
3. 复用 L1 特征代码，在 outer-train 物化 P actions，并扩展 P 的 clean-visible crossfit soft score；不重编码 contextual tokens，不导入历史 held outcome 标签；
4. 实现动态权重、full-list loss 和 Phase A 四臂共用 sampler；
5. 单元测试与 CPU synthetic preflight；
6. GPU 上仅做 replay + tiny overfit；
7. 通过后才生成一个包含 `#SBATCH --gpus=1` 的首轮 sbatch；
8. 首轮结果通过后才增加 seed/fold。

## 9. 最终冻结裁决

下一步不是重新发现动作、重新做梯度共识，也不是先做蒸馏。下一步是复用已验证的 N/P action 与 clean-visible 信号，完成一次 formula-crossfit、动态软加权、完整候选、直接共享 encoder 微调，并以同权重 matched-random arm 作唯一方向性对照。

除非本合同的首轮门失败且失败类型被明确记录，否则不得转向新算法；门失败时也只按第 5.3 节进入对应分支，不得重新回到随机噪声、P2b 或无边界超参扫描。
