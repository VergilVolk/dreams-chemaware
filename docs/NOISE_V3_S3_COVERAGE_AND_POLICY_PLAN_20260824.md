# Noise-v3 S3：错误覆盖扩展、动作聚合与专项微调计划

## 一、当前样本账

完整 strict-10ppm 候选图包含 23,876 个查询，其中官方 DreaMS Top-1 正确 22,071 个、错误
1,805 个；near（MCES 0–2）错误 1,446 个。

S1c 单峰矩阵有 394,752 个查询-动作单元。其中 2,176 个动作单元能把错误查询修正，1,627 个
动作单元会把正确查询破坏。扩展单峰动作至少能修正 553 个不同错误查询。

S2 动态序列矩阵有 88,596 个配对动作单元、265,788 个实际干预变体。动作单元中共观察到
1,352 次修正和 1,577 次新增错误；至少一条序列能修正 473 个不同错误查询，其中 110 个是 S1c
没有修正的新错误。

S1c+S2 no-op-aware 联合上限为 663/1,805（36.7%，+2.7768 pp）；near 错误覆盖
574/1,446（39.7%）。仍有 1,142 个官方错误和 872 个 near 错误没有任何已测试动作可以修正。

这些数字是动作发现数据，不是独立训练样本数。用于策略学习时必须按 query identity/formula 分组，
不能把同一查询的多个动作当作独立样本；用于 DreaMS 微调的 triplet/contrastive manifest 尚未生成。

## 二、当前矩阵没有覆盖的错误机制

1. **正例证据缺失**：谱图受仪器、碰撞能或采集阈值影响，身份支持峰本来就没有出现。仅删除现有峰
   无法恢复缺失证据。
2. **成组碎裂模式**：当前主要逐峰操作，尚未处理同位素簇、中性丢失链和共现碎片组。
3. **非局部峰交互**：输入梯度是局部一阶近似，可能漏掉两个峰单独无效、联合才有效的作用。
4. **候选关系错误**：部分错误来自谱图—候选组的相对排序，需要 listwise 关系监督，单谱删峰不足。
5. **化学证据未进入选择器**：3486 条规则已缓存，但尚未检验规则差异能否增量预测哪个动作安全有效。
6. **不可辨识谱图**：部分立体异构体或高度近似异构体在当前 MS/MS 条件下没有可用差异证据，不能靠
   任意噪声强行分开，应标记不可判别或引入 RT/CCS 等正交信息。

## 三、S3 扩展动作矩阵

所有扩展仍以真实身份/MCES定义监督，规则只能作为特征和分层变量。

### A. 负例过近臂

- 动态 candidate-gradient：将路径从3步扩展至自适应1–6步，若风险调整效用不再增加则停止；
- exact leave-one-peak-out：在代表性大样本上计算真实单峰边际效应，用于校准局部梯度；
- group occlusion：对同位素簇、中性丢失链、共现碎片组整体衰减；
- differential-rule selector：只在规则证据支持“该峰更偏向错误候选”时作为增量动作，不作标签；
- hard-negative 分层：同分子式、MCES 0–2、3–5、共享主峰、共享中性丢失分别建层。

### B. 正例证据不足臂

- replicate-consensus view：同身份同加合物的核心峰保持，条件特异峰随机衰减；
- same-identity donor view：仅从同一分子的另一真实谱图构造条件互补视图，禁止跨身份合成峰；
- intensity-variance noise：根据同身份重复谱的峰强度方差扰动，而不是统一随机删峰；
- cross-condition positive：真实跨仪器/碰撞能同身份谱作为正例，结合严格10ppm困难负例；
- core-preservation control：所有增强必须保护身份核心峰，并与等数量、等强度随机扰动配对。

### C. 不可辨识与安全臂

- no-op 为显式动作；
- 记录候选谱图本身缺少差异峰的不可辨识组，不强制制造 margin；
- 原本正确查询进入 safety pool，任何动作新增错误均按高代价记录；
- 规则冲突、采集条件缺失、候选重复数过高分别报告，不混成一个标签。

## 四、策略聚合

训练查询级/动作级非线性策略头，而不是线性叠加多个动作。动作集合首版为：

`no-op`、candidate-gradient 50% 的1–6步、role-confounder 100%的1–3步、
consensus-dropout、same-identity donor view、group occlusion。

策略头输出每个动作的两个概率与一个连续效应：

- 修正错误的概率；
- 新增错误的概率；
- 相对匹配随机对照的 margin 增量。

风险调整效用定义为：

`U(q,a)=P(correct|q,a)-lambda*P(introduce|q,a)+eta*DeltaMargin_specific`

其中 `lambda > 1`，并以 `no-op` 为默认。只有效用下置信界大于零才允许干预。

训练必须使用 formula-group nested OOF；相同 IK14、分子式和动作路径不能跨折泄漏。首先训练不含规则的
数值策略，再加入规则类别、支持度、差异规则证据和规则—峰对应特征，做严格增量消融。

## 五、从动作策略到模型微调

策略通过交叉拟合门后，才生成专项微调 manifest。每个训练单元包含：干净查询、策略增强查询、同身份
真实正例、strict-10ppm 困难负例和官方教师 embedding。

建议损失：

`L = L_identity + lambda_r L_group-rank + lambda_a L_aug-consistency + lambda_cf L_aug-rank + lambda_p L_preserve`

- `L_identity`：干净谱图的同身份检索；
- `L_group-rank`：候选组内 listwise/ranking，而非随机二分类；
- `L_aug-consistency`：策略增强前后身份表征保持；
- `L_aug-rank`：增强后仍应胜过严格困难负例；
- `L_preserve`：保护官方 DreaMS 的干净检索能力。

不再只训练 projection head。先使用最后1–2层的低秩 adapter/LoRA；冻结主干时始终保持 eval，关闭隐式
dropout，只保留显式、可审计的谱图增强。先做小规模三seed门控，再扩大到全量。

## 六、进入正式微调前的硬门

1. 联合动作覆盖至少达到 1,000 个不同官方错误，或真实可实现的 OOF 净修正达到 +2 pp；
2. near 错误 OOF 净收益非负且公式簇 bootstrap 下界大于零；
3. 策略引入错误数小于修正数的一半；
4. `with-rules` 必须显著优于 `without-rules` 才能宣称化学先验贡献；
5. P3 封存测试集在策略、阈值和模型完全冻结后只开启一次。

## 七、S3A 已冻结的第一轮正式实现

2026-08-24 已将第一轮扩展矩阵冻结为以下 8 个动作、每个动作最多连续 6 步：

- candidate-gradient 50%；
- confounder-only 100%；
- shared peak 25%、50%、100%；
- unmatched peak 25%、50%、100%。

每一步都重新运行官方 DreaMS、重新确定当前最强正例/困难负例、重新划分峰角色；目标路径不得重复峰，
身份专属峰和 precursor token 受到保护。每条目标路径必须配有两条同角色、强度和 m/z 匹配且与目标路径
完全不重叠的随机对照，否则该路径不进入结果。

正式入口为 `tasks/run_noise_v3_s3a_extended_matrix.sbatch`。它依次执行核心测试、256-query smoke、
23,876-query 正式矩阵、fail-closed 验证、修正/新增错误与规则证据分析、矩阵可视化和最终产物断言。

正式产物必须保留：

- `paired_interventions.csv.gz`：全部查询-动作-步数结果；
- `selected_sequences.csv.gz`：目标峰路径和两条随机对照路径；
- `transition_audit.csv.gz`：每一条 corrected/introduced 转换及其错误候选、MCES和规则证据；
- `cell_summary.csv`：48 个预注册矩阵单元的修正、引入、净收益与新增覆盖；
- `decision.json`：联合 headroom、进入策略学习的硬门和声明边界；
- `s3a_action_matrix.png`、`s3a_transition_destinations.png`：用于持续审查矩阵与新增错误去向。

这里的任何 corrected 数字仍是动作干预结果，不是微调模型性能。只有完成 formula-group OOF 动作策略、
生成冻结 manifest、训练 adapter，并在未开启的 P3 上验证，才能声称模型性能提升。
