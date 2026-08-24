# DreaMS 峰干预策略学习与专项噪声微调：系统性补充实验方案

日期：2026-08-24  
状态：讨论稿；在补充实验矩阵和新终测集封存前，不启动正式权重训练。

## 一、核心科学问题

我们要解决的不是“删多少峰最好”，而是：

> 对一个真实 strict-10 ppm 候选组，哪些查询峰在什么候选关系和化学背景下会净提高正确分子的相对排名，哪些干预会破坏原本正确的判断；能否学习一个可泛化的峰干预策略，并把干预后的正确候选分布迁移回未扰动谱图，从而真正改善 DreaMS 权重。

该问题包含三个互相独立的层次：

1. **干预头寸**：某个峰操作是否真的优于匹配随机对照；
2. **策略选择**：面对一个查询，应该选哪个峰、哪种算子和多大剂量，或选择不干预；
3. **表征迁移**：模型能否在不输入加噪谱图的情况下，对原始谱图复现纠错后的候选排序。

过去多次失败，主要因为三个层次被混在一个 loss 中同时训练。

## 二、已有证据总账

| 路线 | 数据和结果 | 真正得到的结论 | 不能推出的结论 |
|---|---|---|---|
| 规则重叠硬 triplet | 检索 AUC 从约 0.88 降到 0.59 | 规则重叠不能作为结构距离硬标签 | 规则没有任何价值 |
| 初代候选差异峰干预 | 误候选独有峰删除：Δmargin +0.0494；真实候选独有峰删除：−0.0558；n=156/184 | 候选特异峰能方向性改变 DreaMS 判断 | 小样本效应可以直接外推到全库 |
| 初代 head 微调 | 内部 Top-1 约 +0.57 pp，反事实指标没有同步改善 | head 可安全改变少量边界排序 | 增益来自反事实峰监督 |
| G5/G6/G7 | 全量 macro-AUC 下降 3.1–4.6 pp | 无差别训练会把同分子不同谱推散 | 困难负例分离方向错误 |
| P0 dropout | 关闭冻结 backbone 的 dropout 后 margin 压缩消失；head-only near 仍无改善 | dropout 是混杂；冻结 backbone 必须 eval | 关闭 dropout 本身能解决 near |
| directional-noise v2 | 整组条件峰删除与匹配随机删除持平 | “整组低频弱峰”不是有效错误特异动作 | 候选条件化单峰干预无效 |
| error atlas | 1,805 个官方错误；positive-deficit 1,439，negative-excess 385；共享主峰/中性丢失为高频筛查信号 | 错误不是单一机制，必须分臂 | 筛查标签已经是因果标签 |
| noise-v3 G1 gradient | 50% 单峰衰减在 1,605 个错误查询中修正 138；在 19,271 个正确查询中新增 113 | 梯度能在错误查询中找到高收益方向，但不能全局施加 | 梯度选择器应被整体废弃 |
| noise-v3 G1 role-only | 6,114 查询；50% 为 23/6；硬删除为 99/18；near 硬删除 75/15 | “匹配错误候选、不匹配正确候选”的峰是目前最干净的训练增强来源 | 硬删除就是最佳训练剂量 |
| P2b 局部排序 | P3 主面板 Recall@1 +1.07 pp；near-core −4.23 pp | 原始峰和中性丢失对主任务有互补信息，但不同候选区域需要不同策略 | 固定中性丢失融合可以解决 near |

总裁决：

- 通用随机遮峰只能作为鲁棒性基线，因为 DreaMS 预训练已经使用按强度抽样的 30% m/z masking；
- MS2DeepScore 使用过低强度峰删除、强度抖动和弱噪声峰添加，这些属于成熟的通用增强基线，不是本课题的主创新；
- 本课题的核心增量应是**候选条件化、因果对照验证、可选择 no-op 的峰干预策略学习**。

## 三、微调策略的系统分类

任何策略必须由四个字段唯一描述，禁止再用“噪声”一个词混称。

### 1. 错误臂

- **P-arm：positive deficit**。同分子真实候选得分不足，重点保护或恢复身份证据；
- **N-arm：negative excess**。错误候选得分过高，重点削弱候选特异混淆证据；
- **B-arm：both/boundary**。两条分数臂同时异常，只在 P/N 分臂验证后考虑；
- **S-arm：stable correct**。不追求纠错，只承担安全保持。

### 2. 峰选择器

- `random-native`：DreaMS 原生强度加权 m/z masking，通用基线；
- `random-ms2ds`：低强度删除、强度抖动和弱峰添加，文献基线；
- `condition-single`：同分子多谱中单个条件特异峰；
- `role-confounder`：错误候选匹配、正确候选不匹配；
- `role-identity`：正确候选匹配、错误候选不匹配，作为方向负对照并默认保护；
- `input-gradient`：完整候选组 margin 的输入梯度，保护 identity-only；
- `rule-stratified`：以上选择器产生峰后，用 CF/NL/ISO/HR/规则支持度做分层，不直接定义标签；
- `learned-policy`：使用干预前特征预测不同动作的净效用。

### 3. 干预算子

- no-op；
- 单峰强度衰减 25%、50%、75%；
- 单峰硬删除 100%，仅作为压力测试或被策略头明确选择的训练动作；
- DreaMS 原生 mask token，单独作为预训练一致性对照；
- 逐峰贪心软衰减：每次只处理一个峰，重新计算候选 margin/梯度，最多 3 步；
- 禁止未经单峰验证的一次性整组删除。

### 4. 训练目标

- 通用不变性；
- 干净候选组 listwise 身份排序；
- 干预视图 listwise 排序；
- 纠错视图到干净视图的 stop-gradient 分布迁移；
- baseline-correct 候选分布与真实正对关系保持；
- 化学概念解码只能作为后续独立消融。

## 四、无偏补充实验矩阵

## S0：数据与评价冻结

1. 训练开发池固定为 P3/P4 身份排除后的 23,876 个完整候选组；不截取前 N 行。
2. 以 IK14 为最小隔离单位，以分子式为外层五折分组单位。
3. 当前 P3 已被 P2b 消耗，只能作为固定回归面板；在训练非线性策略头前必须另锁 P4 或独立外部终测。
4. 每个查询保存：完整候选图、官方分数、正例谱、Top-K 负分子、峰角色、规则命中和所有干预结果。
5. 所有动作的随机种子、剂量和匹配对照在看结果前写入 manifest。

## S1：单峰选择器 × 剂量正交实验

第一轮只比较单峰，避免动作空间失控。

| 选择器 | 25% | 50% | 75% | 100% | 主要用途 |
|---|---:|---:|---:|---:|---|
| matched-random | ✓ | ✓ | ✓ | ✓ | 因果对照 |
| role-confounder | ✓ | ✓ | ✓ | ✓ | N-arm 主线 |
| role-identity | ✓ | ✓ | ✓ | ✓ | 方向负对照/峰保护检查 |
| input-gradient | ✓ | ✓ | ✓ | ✓ | 错误查询高覆盖补充 |
| condition-single | ✓ | ✓ | ✓ | ✓ | P-arm 候选 |
| native mask | 固定协议 | — | — | — | DreaMS 通用增强基线 |
| MS2DeepScore-style | 固定协议 | — | — | — | 文献通用增强基线 |

每个目标峰配至少三个强度、m/z、峰角色匹配对照。必须同时报告：

- Δs(q,p)；
- Δs(q,n_max)；
- Δmargin；
- Top-1/MRR 转换；
- corrected、introduced、neutral；
- baseline-wrong/correct；
- near/non-near；
- P2b corrected/introduced/persistent strata；
- identity 与 formula cluster bootstrap CI。

选择器不得在同一查询的干预结果上选最优动作后再报告该动作的收益。第一轮报告所有预注册动作。

## S2：逐峰贪心头寸实验

只让 S1 中通过因果门的“选择器 × 剂量”进入 S2。

1. 只在 baseline-wrong 或预注册低 margin 查询上执行；
2. 选择一个峰并软衰减后，重新前向、重新计算候选组 margin 和下一峰；
3. 最多 3 步，每一步都可选择 stop；
4. 对照为相同峰角色、相同强度/mz 分布、相同步数的随机序列；
5. 分别报告 step1、step2、step3 的新增纠错头寸，不能只报告最终 best-of-3；
6. 若第二或第三步没有带来 formula-cluster CI 为正的增量，正式训练退回单峰。

该阶段回答“是否存在足够大的可学习梯度”，而不是训练模型。

## S3：非线性干预策略头

### 3.1 任务定义

策略头不是新的检索模型，而是训练数据生成器。对每个查询和合法动作预测：

- `Y_margin`：目标干预相对匹配随机对照的 margin 增益；
- `Y_top1`：目标干预相对随机对照的 Top-1 增益；
- `P_harm`：baseline-correct 被改错的风险；
- 不确定度或分位数。

统一策略效用：

\[
U(a|q)=\widehat{Y}_{margin}
+\alpha\widehat{Y}_{top1}
-\beta\widehat{P}_{harm}
-\gamma\,\mathrm{size}(a), \qquad \beta>\alpha.
\]

只有保守下界

\[
LCB(a|q)=\widehat U(a|q)-\kappa\widehat\sigma(a|q)>0
\]

时才干预；否则选择 no-op。

### 3.2 输入必须全部来自干预前

- 官方 query/positive/negative embedding 与候选分差；
- query 全局 embedding、目标峰 token、m/z、强度、局部峰环境；
- identity/confounder/shared/unmatched 角色；
- 候选组大小、Top-K 负例集中度；
- 原始峰、熵和中性丢失证据；
- CF/NL/ISO/HR 类别、规则支持度和判别力；
- 仪器、加合物和可用的 CE 状态。

干预后的 margin、是否修正、P2b 最终对错不能作为输入，只能作为标签或分层结果。

### 3.3 模型与对照

先做两级模型，避免一上来堆复杂网络：

1. GBDT/小型 MLP：只用数值、角色和规则统计，作为可解释基线；
2. DeepSets/Set Transformer 策略头：读取目标峰 token 与候选组集合，作为主模型。

每个模型必须与以下策略比较：

- no-op；
- 全局固定 role-confounder；
- 全局固定 input-gradient；
- 随机合法动作；
- 不含规则特征的策略头；
- 含规则特征的策略头。

这样才能回答规则库是否真正提高策略选择，而不是只增加维度。

### 3.4 无偏评价

由于预注册动作会在同一查询上全部执行，本问题本质上是**同一查询内的多动作排序**，不需要把它伪装成观察性因果推断。正确做法是：

- 以分子式五折 cross-fitting 产生完全 OOF 的动作选择；
- 每个测试查询只根据训练折学到的模型选择动作；
- 选完动作后再读取该动作的真实干预结果评价；
- 报告 policy value、corrected/introduced、覆盖率和校准；
- 不能在同一 OOF 结果上反复调效用权重后继续当作确认结果。

## 五、策略进入正式微调的硬门

一个动作或策略头必须同时满足：

1. 至少覆盖 500 个 IK14、200 个分子式；
2. baseline-wrong 的 target-minus-random margin 和 Top-1 在 identity/formula CI 下界均大于 0；
3. baseline-correct 的 introduced rate 不超过预注册界，并明显低于 corrected rate；
4. near 与 non-near 分开报告，任一主结论不能靠另一层抵消；
5. 25/50/75% 存在可解释剂量关系，或策略头能稳定选择 no-op；
6. OOF learned-policy 显著超过固定 role-confounder；
7. 加规则特征的策略头必须显著超过不加规则的同结构模型，否则规则不进入微调，只保留解释用途；
8. 预估可修正头寸足以覆盖至少 10% 的官方错误；若做不到，不以“大幅提升模型性能”为目标包装该分支。

## 六、正式峰级微调结构

### 1. Rescue 流

- 只使用 baseline-wrong/低 margin 且策略 LCB>0 的样本；
- 生成策略选择的纠错视图；
- 冻结官方候选/reference embedding，得到纠错后的候选概率分布；
- 将该分布 stop-gradient 后蒸馏给学生对**原始未扰动查询**的预测。

### 2. Safety 流

- 大量 baseline-correct 查询；
- 官方候选分布蒸馏；
- 真实跨条件同分子正对保持；
- no-op 或经过验证的良性随机增强；
- 禁止对正确查询无条件执行 input-gradient 峰衰减。

### 3. 模型结构

首选零初始化峰贡献残差：

```text
official peak tokens
  -> peak gate / small token adapter
  -> residual correction to precursor embedding
  -> official projection and frozen reference library
```

它从官方 DreaMS 恒等映射开始，能改变峰贡献；比线性 head 更有表达力，又比全面解冻安全。只有峰 adapter 未通过时，才解冻最后一个 Transformer block。

### 4. 核心损失

\[
L=L_{clean-listwise}
+\lambda_t L_{corrected-view\rightarrow clean}
+\lambda_a L_{aug-listwise}
+\lambda_s L_{safe-distill}
+\lambda_r L_{real-positive-preserve}.
\]

禁止再使用“只要求加噪视图变好、却不迁移给 clean”的目标。

## 七、建议执行顺序

### P0：立即执行

1. 锁定 P4/外部终测身份，P3 只保留为已消费回归面板；
2. 完成 S1 尚缺的 75% 剂量、identity-only 方向负对照、condition-single 单峰和 native/random 文献基线；
3. 统一干预记录格式和候选组指标。

### P1：确认大梯度

4. 对 S1 通过的动作运行 S2 逐峰贪心，确认 step2/step3 是否提供额外头寸；
5. 固定动作集合和效用函数。

### P2：策略学习

6. 训练数值 MLP/GBDT 策略基线；
7. 训练 peak-token + candidate-set 非线性策略头；
8. 做 `without rules` vs `with rules` 的严格 OOF 消融。

### P3：专项微调

9. 先训练零初始化 query-side peak adapter；
10. 三 seed、固定超参数完成 P2 OOF；
11. 与官方 DreaMS、identity-only 微调、通用随机增强、固定 role-only 和 P2b 分别比较；
12. 全部冻结后一次性开启 P4/外部终测。

## 八、当前最值得讨论的三个决策

1. **主错误臂**：第一版策略头应同时学 P-arm 和 N-arm，还是先集中当前证据最强的 N-arm？建议先 N-arm，P-arm 优先使用真实跨条件正例而不是合成删峰。
2. **动作空间**：第一版是否允许硬删除？建议保留为候选动作，但加动作大小惩罚，且只有策略 LCB>0 时选择。
3. **规则接口**：规则是否进入策略头？建议作为预注册增量特征做 `with/without rules` 消融；绝不作为动作标签。

## 九、方法学依据与创新边界

- DreaMS 已使用 30% 强度加权 m/z masking，因此通用遮峰是基线而不是创新：<https://pmc.ncbi.nlm.nih.gov/articles/PMC13090125/>。
- MS2DeepScore 使用低强度峰删除、强度抖动和弱噪声峰添加，适合作为通用增强对照：<https://pmc.ncbi.nlm.nih.gov/articles/PMC8556919/>。
- MassSpecGym 强调结构互斥划分和统一评价，支持本方案的公式/身份隔离纪律：<https://proceedings.neurips.cc/paper_files/paper/2024/hash/c6c31413d5c53b7d1c343c1498734b0f-Abstract-Datasets_and_Benchmarks_Track.html>。
- 本课题可形成的核心创新不是“第一次对质谱加噪”，而是：**利用完整候选组、峰级候选角色、匹配随机干预和 OOF 策略学习，得到可选择 no-op 的个体化谱峰干预策略，再将纠错视图迁移到干净 DreaMS 表征。**

## 十、S1a 正交矩阵预注册（2026-08-24）

本轮只回答两个问题：**哪一种单峰选择方向有效，以及有效性如何随衰减剂量变化。** 不把整组删峰、通用随机 masking、条件峰共识或化学规则同时混入。

固定因素：

- 完整 23,876-query strict-10ppm 候选图；
- 官方 DreaMS 微调权重，backbone 固定 `eval()`；
- top-5 困难负候选、温度 0.10；
- 查询峰角色由 top-3 正谱与最难负分子的 top-3 谱联合确定；
- 每个 query-selector 的目标峰及三条强度/mz 匹配随机峰只选择一次，跨剂量完全复用；
- 所有干预重新在完整候选组中计算严格 rank、MRR 与 margin；
- identity/formula 双重 cluster bootstrap，2000 次；
- 全部矩阵单元报告，不根据结果删除不利单元。

正交因素：

| 因素 | 水平 |
|---|---|
| 单峰选择器 | candidate-gradient；role-confounder；role-identity 方向负对照 |
| 峰衰减剂量 | 25%；50%；75%；100% |
| 基线状态 | baseline-wrong；baseline-correct |
| 候选难度 | near；non-near |

本轮的 `role-identity` 不进入训练候选池，只验证实验是否具有正确方向敏感性：若衰减正确身份专属峰没有使 margin 相对匹配随机对照下降，则峰角色定义或干预实现需要回查。

S1a 只产生动作 headroom 和安全性证据，不直接选训练策略。候选动作至少需要同时具备：baseline-wrong 的 identity/formula CI 为正、baseline-correct 风险可控、25–75% 剂量关系可解释、near 不被另一层掩盖，才进入 S2。硬删除 100% 主要作为响应上界与风险压力测试。

执行入口：`tasks/run_noise_v3_s1a_single_peak_matrix.sbatch`。输出目录固定为 `data/validation/g8r_noise_v3_s1a_single_peak_matrix`，拒绝覆盖；运行结束必须通过 `matrix_validation.json` 的完整性门。

## 十一、S1c 动作空间扩展预注册（2026-08-24）

S1b 表明原12动作的 no-op-aware oracle 只能覆盖386个错误（+1.62 pp），不足以支撑2–4个百分点目标；同时只有386个独立正query，不足以直接训练高容量非线性策略。因此先扩展动作空间，不增加模型容量。

固定设计：

- 使用与S1a相同的23,876-query候选图、官方权重、top-5负候选和完整候选组评价；
- candidate-gradient 从每个query的单一最高峰扩展为按预测margin gain排序的前5个不同峰；
- role-confounder 从最强峰扩展为按强度排序的前3个不同峰；
- 每个峰分别执行25/50/75/100%单峰衰减；
- 每个query-selector-rank的目标与两条匹配随机峰跨剂量复用；
- identity-only峰仍受保护，不得进入candidate-gradient动作；
- 本轮不进行多峰累计删除，避免把“峰位置扩展”和“组合效应”混成一个因素；
- 全部动作报告，no-op-aware oracle仅作上限，不作模型成绩。

硬门：

1. 八个selector rank与四个剂量全部存在；
2. query集合、目标峰、对照峰跨剂量不漂移；
3. 非对照oracle至少达到+2.0 pp（478个独立错误），否则在进入策略网络前继续研究顺序组合动作；
4. 独立可修正query优先达到600以上；
5. candidate-gradient与role-confounder均保留独有修正；
6. 任何动作的训练价值必须同时报告baseline-correct风险。

执行入口：`tasks/run_noise_v3_s1c_topk_matrix.sbatch`。该作业依次执行单元测试、全图预检、扩展矩阵、完整性验证和no-op-aware oracle分析。

### S1c 结果裁决（2026-08-24）

正式矩阵得到408个独立可修正错误，对应no-op-aware oracle增益+1.709 pp；candidate-gradient与role-confounder分别保留295和38个独有修正。虽然正动作行扩展至2,176，但相较S1b只新增22个独立可修正query，说明继续增加单峰rank主要产生重复动作。S1c未达到预注册的478个错误（+2 pp）和600-query门槛，因此不得直接进入高容量策略学习；下一步只检验最多3步的顺序多峰组合是否产生新的独立纠错覆盖。旧headroom文件沿用了S1b的1 pp门槛，其`pass_to_policy_design=true`不作为S1c正式裁决。

## 十二、S2 动态顺序多峰预注册（2026-08-24）

S2只检验“当前谱图经过一次干预后，重新计算的下一峰是否产生S1c单峰并集之外的新修正”。禁止直接累加S1c预先排序的top-k峰。

固定设计：

- 使用相同的23,876-query、P3身份互斥候选图和官方DreaMS权重；
- 选择器固定为candidate-gradient与role-confounder；
- 剂量固定为50%软衰减和100%删除，不继续扩大剂量搜索；
- 每一步重新前向完整候选图，重新确定最佳正谱、最难负分子和候选峰角色；
- candidate-gradient每一步重新对当前完整候选组margin求输入梯度；
- role-confounder每一步从当前最难负候选定义的confounder-only峰中选择最强未使用峰；
- 最多3步，所有step1/2/3结果均报告；stop只通过no-op-aware oracle表达，不用真实结果训练或选择动作；
- 每条目标路径配置两条同角色、强度/mz匹配、相同步数和剂量的无重复随机路径；不完整对照路径显式排除；
- identity-only峰、前体token和已使用峰始终禁止进入后续动作；
- 先运行256-query smoke，再运行正式全图；输出目录拒绝覆盖。

进入策略学习的硬门：

1. S1c与S2联合no-op-aware oracle至少覆盖478个独立错误（+2 pp）；
2. 独立可修正query至少达到600；
3. step2或step3至少存在一个动作，其baseline-wrong target-minus-random Top-1在identity与formula聚类CI下界均大于0；
4. 该动作corrected必须大于introduced；
5. 若S2没有产生S1c之外的新修正，或联合头寸仍低于门槛，停止继续增加峰数，不训练高容量策略头。

执行入口：`tasks/run_noise_v3_s2_sequential.sbatch`。
