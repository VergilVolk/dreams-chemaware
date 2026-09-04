# ChemAware embedding 与定向噪声微调：最终大决战计划

> **2026-08-26 执行纠偏：** D1b 已确认只是 clean-only、query-side adapter 安全基线，不是噪声微调主结果；P2b/C2-C 暂停。后续唯一执行合同见 `docs/NOISE_FINETUNE_EXECUTION_CORRECTION_20260826.md`。本文件此前的实验总账继续保留，但“当前唯一下一步”由新合同替代。

**日期：** 2026-08-25  
**状态：** 架构重新对齐；停止 P2b 残差拟合；下一步只训练能够输出新 embedding 的模型  
**用途：** 本文件是后续噪声微调的唯一总账、上下文压缩和执行合同。旧文档保留为实验原始记录；发生口径冲突时，以本文件列出的结果口径和硬门为准。

---

## 0. 一页压缩状态

### 项目最终结构

```text
实验 MS/MS 谱图
  -> 模块1：ChemAware embedding 微调
       - 真实身份/候选组排序
       - 正例证据恢复
       - 候选条件化峰干预
       - 化学概念可解码约束
       输出：新的 global embedding + peak-token embedding
  -> 可选 P2b 局部专家
       输出：候选局部重排与峰级谱学证据
  -> 模块2：双重映射解释
       embedding/peak token -> 化学概念/局部结构环境
       化学概念/局部结构环境 -> 具体谱峰、质量差和规则来源
       输出：支持、冲突、证据不足与忠实性
  -> 生物学应用与错误反馈
```

### 不可再混淆的三条边界

1. **噪声微调和 ChemAware 概念微调都必须改变 embedding space。** 推理输入是原始谱图，输出是新的 embedding。
2. **P2b 是 embedding 后面的独立候选专家。** 它可以保底和提供谱学证据，但不是噪声微调教师、不是 ChemAware embedding，也不参与定义正负样本。
3. **模块2双重映射不是分类头装饰。** 它必须同时通过可解码、峰级定位和目标删峰忠实性，才能把 embedding 方向解释成化学证据。

### 当前最重要的证据

- A4 精确峰干预在 1,805 个官方错误中可恢复 776 个；联合历史动作的 outcome-oracle 覆盖 920 个查询，约 3.85 pp 的动作上限。
- A4-B0 真实同身份正例教师在 4,998-query 诊断面板上将准确率从 0.6389 提到 0.7351，修正/新增为 542/61。
- C1 在教师支持谱与评价正谱严格互斥后，仍在 80,250 个交叉拟合样本上提高 2.47 pp，near 提高 2.34 pp，修正/新增为 2,382/396。
- 上述数字是**动作空间或训练期教师证据**，不是微调后模型成绩。
- 当前真正缺失的环节是：用原始谱图峰 token 训练学生，把 P-arm 和 N-arm 教师方向迁移进新的 embedding，并在完整候选图上验证干净谱图性能。

### 立即停止的偏航

`P2b + RAW/token residual` 属于下游重排器增量，不输出新的 embedding。C2-C 的负结果只说明该残差不能继续提高 P2b，不能用于裁决噪声微调或峰 token 微调。该路线立即归档，不再消耗噪声微调预算。

---

## 1. 三个模块的科学关系

## 1.1 模块1A：错误引导与噪声微调

目标：让原始谱图经过 DreaMS 后得到更合理的 embedding，使：

- 同分子跨仪器、碰撞能和条件的谱图仍能聚集；
- strict-10ppm 质量近邻和 near 异构体在局部邻域中具有更合理顺序；
- 错误候选特异的混淆峰不再支配全局判断；
- 原本正确的细粒度异构体证据得到保护。

噪声不是标签。真实 IK14 身份和候选组关系才是标签；峰干预只生成一个训练期纠错观察视图。

## 1.2 模块1B：ChemAware 化学概念微调

目标：在不把规则重叠当距离的前提下，使新 embedding 能读取实验碎裂概念和局部结构环境。

当前已知：

- 266 个实验碎裂概念的冻结官方 embedding 线性探针 test macro-AUPRC 为 0.659，流行率基线为 0.200；
- 469 个数据驱动局部结构环境的 macro-AUPRC 为 0.240，流行率基线为 0.0447；
- 175 条“结构环境—embedding方向—谱图概念”桥梁在发现/确认数据中复现；
- 规则库的正确作用是概念监督、峰保护、策略分层和解释，不是身份标签或结构距离。

ChemAware 概念损失只能作为主身份排序之外的辅助目标，并必须比较加入概念前后检索性能、概念可解码性和忠实性。

## 1.3 模块2：可解释性双重映射

双重映射的可证伪链条为：

```text
第一层：global/peak embedding -> 化学概念或局部结构特征
第二层：化学概念或局部结构特征 -> 具体谱峰、峰对、质量差和规则来源
忠实性：删除被定位谱峰 -> 对应概念得分或候选证据应特异下降
```

只有同时满足以下三项的概念才能进入正式解释：

1. 在身份或骨架隔离测试集上可解码；
2. 能定位到具体谱峰或峰对；
3. 目标删峰效应显著强于 m/z、强度和删峰数匹配的随机对照。

模块1训练后必须重新校准全部概念探针，并比较官方 embedding 与 ChemAware embedding 的概念 AUPRC 和峰级忠实性，防止检索提升却损失化学可解释度。

---

## 2. 噪声微调完整结果总账

所有数字按四种证据级别标记：

- **D：描述性错误图谱**；
- **I：冻结模型的峰干预结果**；
- **T：使用身份或干预结果的训练期教师上限**；
- **M：真正更新模型后在干净谱图上的检索结果**。

只有 M 可以声明“微调模型性能提高”。

## 2.1 早期反事实与旧权重微调

| 实验 | 级别 | 主要结果 | 正确结论 | 局限 |
|---|---|---|---|---|
| 候选差异峰删除 pilot | I | 删除错误候选独有峰，margin 相对随机对照 +0.0494，95% CI [0.0330,0.0675]；删除正确候选独有峰为 -0.0558，CI [-0.0786,-0.0450] | 候选特异峰对排序有方向性因果作用 | 小样本，不能外推全库 |
| 早期 counterfactual head | M | 100-query hard 面板约 +2 pp；500-query 为 +1 pp，修正5/新增0，embedding保持约0.998 | 线性头能安全改变少数边界排序 | 规模小，未证明来自峰监督；不能据此声称整体空间改善 |
| G5/G6/G7 | M | 全量 macro-AUC 下降 3.1–4.6 pp | 无差别训练把同分子不同谱推散 | 训练池曾因排序截断只含516个分子，且损失/采样错位 |
| P0 dropout | M/诊断 | 关闭冻结 backbone dropout 后 margin 压缩消失，preservation约0.9987；near仍未改善 | 冻结模块训练时必须 eval；dropout是混杂 | dropout不是 near 错误根因 |

## 2.2 大规模错误图谱

23,876 个 strict-10ppm 真实训练查询、2,522 个身份：

- 官方错误：1,805；官方 Recall@1 = 0.9244；
- positive-deficit：1,439；
- negative-excess：385；
- 其中 both：197，comparative-boundary：178；
- 主要结论：错误空间以“正例证据不足”为主，单纯删峰只能处理一部分负例过高问题。

## 2.3 G1 / S1a：单峰选择器和剂量矩阵

### candidate-gradient

| 衰减 | 修正 | 新增 | 净修正 |
|---:|---:|---:|---:|
| 25% | 70 | 51 | +19 |
| 50% | 138 | 113 | +25 |
| 75% | 198 | 258 | -60 |
| 100% | 288 | 560 | -272 |

解释：梯度能在错误查询上找到高收益峰，但对正确查询同样会找到能改变排序的峰。它适合作为动作候选生成器，不能无条件执行。

### role-confounder

| 衰减 | 修正 | 新增 | 净修正 |
|---:|---:|---:|---:|
| 25% | 7 | 2 | +5 |
| 50% | 23 | 6 | +17 |
| 75% | 41 | 9 | +32 |
| 100% | 99 | 18 | +81 |

解释：错误候选匹配、正确候选不匹配的峰是最干净的 N-arm 动作；覆盖较低，但适合作为安全种子。

### role-identity 方向负对照

| 衰减 | 修正 | 新增 | 净修正 |
|---:|---:|---:|---:|
| 25% | 15 | 31 | -16 |
| 50% | 25 | 80 | -55 |
| 75% | 39 | 140 | -101 |
| 100% | 43 | 503 | -460 |

解释：真实身份专属峰携带正向身份证据，角色定义和实验方向有效；过去粗暴遮峰会破坏正确聚集。

## 2.4 S1b/S1c：单峰动作空间上限

- S1b：386 个独立错误至少可被一种非对照动作修正，no-op oracle 上限 +1.62 pp；
- S1c 完整口径：扩展多顺位峰后可恢复 553 个，而不是旧汇总器漏算后的408个；
- 增加单峰排名产生大量重复修正，独立错误覆盖增长有限。

结论：继续枚举更多相似单峰不是扩大动作空间的有效方式。

## 2.5 S2/S3A：动态多步动作矩阵

### 可靠固定动作

| 动作 | 结果 | 角色 |
|---|---|---|
| candidate-gradient 50%，动态3步 | 153/69，净+84 | 广覆盖主动作 |
| candidate-gradient 50%，动态6步 | 固定完整轨迹净收益继续为正；全图固定动作约+0.41 pp | 高覆盖，但仍需策略选择 |
| role-confounder 100%，动态3步 | 37/1，净+36 | 高精度小覆盖 |
| role-confounder 100%，最多5步 | step5约24/0；step6边际转负 | 安全专家，深度上限5 |
| candidate-gradient 100% | 修正同时制造更多错误 | 禁止默认动作 |
| shared-only | 各剂量/步数均明显净退化 | 禁止作为删除标签 |

联合 S1c+S2+S3A outcome-oracle 可恢复 799 个错误，理论上限 +3.346 pp；距离4 pp所需956个净纠正仍缺157个。

## 2.6 A4：全峰精确扫描

对1,805个官方错误和3,193个匹配正确对照，扫描206,288个峰动作、825,152个变体：

| 衰减 | 可恢复错误 | 受损正确对照 | 梯度与真实效应 Spearman |
|---:|---:|---:|---:|
| 25% | 138 | 183 | 0.794 |
| 50% | 289 | 375 | 0.732 |
| 75% | 461 | 652 | 0.652 |
| 100% | 738 | 1,078 | 0.374 |

关键结论：

- 精确扫描可恢复776个错误；联合历史动作覆盖920个，约3.853 pp；
- 梯度适合生成候选，不适合直接决定动作；
- 可恢复动作中 shared 峰最多，但 shared 同时也是伤害正确查询的主要来源；
- 动作必须依赖查询—正确候选—困难负候选关系，单谱峰标签不足以决定操作；
- 规则支持正例、规则不支持错误候选时可恢复率更高，但只属于描述性关联。

## 2.7 A4 非线性动作教师

formula-group OOF、4,998查询、175,605峰动作：

- 修正动作 ROC-AUC 0.849，AUPRC 0.132，约为阳性率7.59倍；
- 伤害动作 ROC-AUC 0.853，AUPRC 0.122，约为阳性率9.34倍；
- margin回归 Spearman 0.540；
- 40%覆盖：182修正、39新增，风险净收益104；
- 但相对 confounder-only 的风险净收益优势不显著；
- 在历史动作之外只新增19个修正。

结论：动作结果可学习，但旧动作空间的独立覆盖仍不足；教师适合作为筛选器，不能直接当模型性能。

## 2.8 A4-B0与C1：正例证据恢复

### A4-B0

- 4,998查询；
- baseline accuracy 0.6389；教师 accuracy 0.7351；
- corrected/introduced = 542/61；
- risk net = 420；
- 新增独立修正159；
- positive-deficit corrected 465；near corrected 374。

### C1严格支持谱互斥

- 80,250个支持互斥样本；
- 1,217个身份、627个分子式；
- baseline 0.5664，teacher 0.5912，+2.47 pp；
- corrected/introduced = 2,382/396；
- near +2.34 pp；
- formula-cluster risk-net CI严格为正。

结论：P-arm是真实、可规模化的教师方向。B0的9.62 pp被直接正例对齐放大，C1的2.47 pp是更可信的支持互斥上限。

## 2.9 C2峰token和错误的P2b残差路线

- query-only峰token专家只带来约+0.08 pp，说明单查询输出缺少候选条件化信息；
- candidate-token单特征在near有小正方向，但总体不稳定；
- C2-C以P2b为基线拟合RAW/token残差，结果为0或负；
- C2-C属于下游重排器实验，不能用于裁决峰token是否能通过adapter改变embedding。

正式裁决：归档C2-C；保留其工程教训——峰token必须在候选条件化训练中使用，不能先压成单一查询向量，也不能只用一个P2b-winning参考谱代表整个候选分子。

---

## 3. 为什么目前“梯度仍没有达到要求”

这里的“梯度”必须拆成四层，否则会再次误判。

## 3.1 动作效应梯度并不小，安全可学习梯度才小

A4硬删除能改变大量排序，但同时损害1,078个正确对照。大动作效应不等于可训练净收益。真正要最大化的是：

\[
G_{safe}=E[\Delta m\mid wrong]-\beta E[-\Delta m\mid correct],\quad \beta>1.
\]

过去偏重“能否改变margin”，没有同时把正确查询伤害作为同等重要的训练信号。

## 3.2 错误动作空间只处理了N-arm，主错误却是P-arm

1,805个错误中1,439个包含positive-deficit。删峰最多只能削弱错误候选，无法补回同身份跨条件谱缺失的正证据。A4未恢复的1,029个错误中，910个包含positive-deficit。这是动作空间迟迟达不到4 pp的首要原因。

## 3.3 峰动作的候选条件信息在聚合时被丢失

历史失败聚合包括：

- 把查询谱压成一个峰token方向；
- 把候选分子压成一张P2b-winning参考谱；
- 用全局线性head统一改变所有候选距离；
- 用单一峰角色或规则Jaccard代替查询内候选关系。

真正需要保留的是同一查询中的集合结构：多个真实正谱、Top-k负分子、每个候选的多张谱、峰—峰/中性丢失对齐和不确定度。

## 3.4 训练目标长期只优化增强视图，没有迁移到clean query

旧损失经常奖励“删峰后的谱图判断变好”，却没有迫使原始谱图产生相同候选分布。部署时输入仍是原始谱图，所以增强视图改善无法自动转化为clean embedding改善。

正确桥梁必须是：

\[
\pi_T(M\mid T(q))\xrightarrow{stopgrad}\pi_\theta(M\mid q).
\]

## 3.5 模型容量与作用位置错误

线性projection head只能进行全局PSD度量变换，难以改变具体峰对不同候选的贡献。全量解冻又会推散同分子谱。正确中间结构是零初始化的query-side peak-token adapter，必要时再解冻最后一个Transformer block。

## 3.6 独立样本量远小于动作行数量

几十万query-action行包含大量同query、同identity的重复动作。真正独立的化学覆盖在C1中只有1,217 identities、627 formulas。训练必须identity-equal采样、formula OOF，并防止同一查询的多个holdout产生伪样本量。

## 3.7 规则库有信息，但还没有证明提供独立策略增益

规则与可恢复性、动作风险存在关联；但尚未完成相同结构、相同训练预算下`with rules`对`without rules`的formula-OOF配对验证。把规则提前当动作标签会重新产生专家偏置和循环论证。

---

## 4. 最终训练问题：两臂教师蒸馏到同一个embedding

## 4.1 数据单位

一个训练单位必须是完整候选组：

- 原始查询谱图q；
- 同IK14、同adduct的多个真实正谱；
- strict-10ppm、同adduct的Top-k不同IK14负分子；
- 每个候选分子的多张参考谱；
- 官方global embedding与peak tokens；
- P-arm或N-arm教师分布；
- baseline-correct safety replay。

P2b可以用于训练图谱的探索性分层，但不得生成身份标签、不得进入主损失、不得成为学生目标。

## 4.2 学生结构

\[
z_\theta(q)=\operatorname{Norm}\left[z_0(q)+A_\theta(H_q,mz_q,I_q)\right].
\]

- `z0`：官方DreaMS global embedding；
- `Hq`：官方最后一层峰token；
- `Aθ`：零初始化peak gate/token adapter；
- 训练开始时严格复现官方embedding；
- 第一轮冻结官方参考库和绝大部分backbone，只训练query-side adapter；
- adapter通过后，才允许小学习率解冻最后一个block。

主结果必须使用`cos(zθ(q), z0(candidate))`评价，证明新query embedding本身改善。候选交互scorer只能作为额外结果，不能替代embedding指标。

## 4.3 P-arm：真实正例证据恢复

教师由同身份、同adduct、支持谱与评价正谱严格互斥的真实谱构造。教师只在训练期可见身份：

\[
z_T^P=\operatorname{Norm}[(1-\alpha)z_0(q)+\alpha z_{prototype}],\quad \alpha=0.25.
\]

学生输入仍是原始q。使用C1样本，按query/identity等权、formula-group OOF。

## 4.4 N-arm：候选条件化混淆证据抑制

优先级：

1. confounder-only安全动作；
2. A4动作教师LCB>0且harm低的candidate-gradient/shared/unmatched动作；
3. 低margin、低峰数、强shared删除默认no-op；
4. identity-only峰永久保护；
5. 25/50/75%软衰减优先，100%只在精确动作教师证明安全时进入。

教师分布来自经过验证的干预视图；学生仍输入clean query。

## 4.5 Safety流

- 大量baseline-correct查询；
- 官方候选分布蒸馏；
- 真实跨条件同分子正对保持；
- 对P2b引入错误、A4受损正确对照和near异构体进行过采样；
- 冻结backbone始终保持eval，关闭dropout混杂。

## 4.6 损失

\[
L=L_{clean-group}
+\lambda_P KL(\operatorname{sg}\pi_T^P\Vert\pi_\theta)
+\lambda_N KL(\operatorname{sg}\pi_T^N\Vert\pi_\theta)
+\lambda_S KL(\pi_0\Vert\pi_\theta)
+\lambda_C L_{cross-condition}
+\lambda_E L_{embedding-preserve}.
\]

第二阶段再加入：

\[
+\lambda_RL_{concept-decode}.
\]

第一阶段禁止把规则概念、P2b、候选交互网络和噪声教师同时全部加入。必须按消融逐步增加。

---

## 5. 大决战实验顺序

## D0：统一数据合同与上下文压缩

产物：一个不可变manifest，记录query、identity、formula、候选图、P/N/safety标签、教师来源和所有哈希。

硬门：

- P3/新封存测试身份重叠为0；
- 所有训练查询按identity等权；
- P-arm支持谱与评价正谱逐行互斥；
- N-arm动作全部来自训练折精确干预；
- 不含P2b分数作为教师字段；
- baseline rank与官方embedding逐query复现。

## D1：clean-only embedding adapter

目的：先证明同一架构能够吸收完整候选组身份监督。

模型：零初始化query-side peak adapter；只用clean groupwise + safety + preserve。

通过门：

- overall Recall@1相对官方不下降；
- near不下降；
- introduced不多于corrected；
- embedding保持度达到预注册门；
- 3 seed方向一致。

若D1失败，先修架构/数据，不进入噪声教师。

## D2：P-arm only

只增加C1支持互斥正例教师，其他完全不变。

主检验：positive-deficit、cross-condition、near和overall。要求P-arm相对D1的formula-cluster CI为正或至少在主目标positive-deficit上显著为正且overall不劣。

## D3：N-arm only

只增加A4安全动作教师。先跑confounder-only；再增加A4非线性安全扩展层。

必须比较：

- fixed confounder-only；
- learned safe action without rules；
- learned safe action with rules。

规则只有在最后一项相对前一项formula-cluster CI下界>0时进入正式模型。

## D4：P+N双臂

固定D2和D3权重，不在同一OOF结果上重新大扫超参数。检验两臂是否互补，尤其关注positive-deficit与negative-excess分别改善、near不退化。

## D5：ChemAware概念辅助

在D4通过后，只增加266个谱图概念与经过筛选的局部结构环境解码。比较：

- D4；
- D4+概念解码；
- D4+概念解码+规则峰保护。

主检索不下降、概念macro-AUPRC提高或保持、忠实性通过，才形成ChemAware embedding最终模型。

## D6：冻结后评价与P2b正交组合

按顺序评价：

1. 官方DreaMS；
2. ChemAware embedding cosine；
3. 官方DreaMS+P2b；
4. ChemAware embedding+P2b。

这样才能分离embedding微调贡献和专家模块贡献。

---

## 6. 评价矩阵与硬门

| 维度 | 必报指标 | 目的 |
|---|---|---|
| overall检索 | Recall@1、MRR、macro query AUC | 总体性能 |
| near | MCES 0-2 Recall@1、MRR | 细粒度结构分辨 |
| 错误臂 | positive-deficit、negative-excess、both、boundary | 防止一类改善掩盖另一类退化 |
| 转换 | corrected、introduced、persistent wrong、protected correct | 判断净收益和伤害 |
| 跨条件 | 同身份跨仪器/CE相似度与排名 | P-arm目标 |
| embedding | 官方与新embedding余弦、局部margin分布 | 证明空间改变且保持安全 |
| 化学概念 | macro/micro AUPRC、ROC-AUC、流行率基线 | 化学可解码性 |
| 忠实性 | 目标删峰 vs 匹配随机删峰概念logit/候选margin变化 | 双重映射因果支持 |
| 泛化 | formula/scaffold/identity隔离、外部生物学谱图 | 防开发集过拟合 |

最终通过条件：

1. 公式簇bootstrap的overall Recall@1差值CI下界>0，或预注册非劣且near/目标错误臂显著改善；
2. corrected>introduced，风险净收益`corrected-2*introduced>0`；
3. near不得出现显著退化；
4. 三seed方向一致；
5. ChemAware概念AUPRC不得因检索提升而显著下降；
6. 新增错误必须按峰角色、剂量、峰数、margin、MCES、规则证据和采集条件完整分解；
7. 最终盲测前模型、阈值、教师规则和权重全部冻结。

---

## 7. 当前唯一下一步

**不是继续训练P2b残差，不是继续枚举峰动作，也不是直接把全部损失一次加入。**

下一步只做D0+D1：

1. 把C1 P-arm、A4 N-arm和安全回放统一成不可变manifest；
2. 明确去掉所有P2b目标字段；
3. 实现零初始化query-side peak-token adapter；
4. 先运行clean-only候选组训练，证明架构能在干净谱图上输出安全的新embedding；
5. D1通过后，单变量加入P-arm；之后才加入N-arm。

这一步是过去所有动作空间审计与真正权重微调之间缺失的桥。若跳过D1直接加入两个教师、规则和P2b，任何结果都无法归因，项目会再次失去收敛能力。

---

## 8. 禁止事项

- 禁止把P2b得分当噪声微调标签或基础分数；
- 禁止把动作oracle、身份教师上限写成模型Recall提升；
- 禁止让规则Jaccard定义正负样本或embedding距离；
- 禁止只优化加噪谱图、不把教师分布迁移给clean query；
- 禁止全局执行candidate-gradient、shared删除或固定随机遮峰；
- 禁止只训线性projection head后宣称改变峰贡献；
- 禁止用同一查询的多个holdout冒充独立样本；
- 禁止在已消费P3上继续调参；
- 禁止承诺4 pp。4 pp是动作/教师头寸目标，最终模型必须由封存评价决定。

---

## 9. 最终科学故事

DreaMS的错误不是均匀随机噪声，而集中在两种可分机制：同分子跨条件正证据不足，以及少数峰对质量近邻错误候选贡献过高。候选条件化精确峰干预证明了负证据方向，支持互斥的真实同身份谱证明了正证据方向。我们不把规则重叠当距离，而让规则提供可解码概念、峰保护和动作风险信息。最终通过零初始化峰token adapter，把训练期纠错视图蒸馏到原始谱图的ChemAware embedding，再由P2b作正交局部专家，并用“embedding—化学概念—具体谱峰”的双重映射解释模型为什么改变判断。

这才是噪声微调、ChemAware规则微调、化学可解释性和生物学应用能够共处于同一个架构中的收敛版本。

---

## 10. 2026-08-25 首轮实现状态

已经落地：

- `tasks/noise_final_core.py`：严格候选图、并列排名、官方embedding缓存与零初始化峰token adapter；
- `tasks/build_noise_final_d0_manifest.py`：D0不可变数据合同，逐项核验P3隔离、C1支持谱互斥/同身份、A4公式OOF、身份等权和官方基线；
- `tasks/train_noise_final_d1_adapter.py`：D1 clean-only query-side峰token adapter；冻结backbone、官方head和候选参考embedding，禁止P2b、规则和噪声教师进入；
- `tasks/aggregate_noise_final_d1.py`：5折公式外推、3 seed 汇总、formula-cluster bootstrap与进入D2的硬门；
- `tasks/test_noise_final_d0_d1.py`：本地零初始化和D0合同smoke test；
- `tasks/run_noise_final_d0_manifest.sbatch`、`tasks/run_noise_final_d1_adapter.sbatch`、`tasks/run_noise_final_d1_aggregate.sbatch`：服务器执行入口。

D1的选择规则包含官方no-op epoch 0。若训练不能在inner公式组上产生风险加权净收益，则自动回退官方embedding；但三seed汇总另设“平均clean增益严格大于0、至少两个seed净修正为正”的非平凡门，避免把零变化误判为架构成功。

### D1先导折诊断与D1b修正

D1 seed 20260825/fold 0的实现链路通过：epoch 0逐查询复现官方模型，baseline mismatch=0，adapter确实产生非零峰级残差。epoch 2在inner公式组修正10例、引入7例，overall与near均小幅上升；但按“引入错误代价为2”的预注册选择函数仍为负，因此正确回退epoch 0，D1未通过。

诊断发现原安全项只在正确查询已经翻错后才产生梯度，训练均值约为10^-6，不能保护官方正负margin。D1b只做一个因果明确的修改：对每个官方正确查询，一旦新margin小于官方margin即处罚，并把该项权重提高到8；其他数据、结构、公式折、listwise目标和候选参考均不变。D1b仍是clean-only，不加入P/N教师。
