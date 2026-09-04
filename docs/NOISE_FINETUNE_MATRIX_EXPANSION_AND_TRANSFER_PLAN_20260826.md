# 噪声微调策略矩阵扩展与 clean embedding 迁移计划

**日期：** 2026-08-26  
**范围：** 仅包含原始 MS/MS 谱图到新 embedding 的噪声微调；不包含候选后处理、化学概念解码或解释模块。  
**目标：** 将已经验证的峰级干预信号迁移到同一共享编码器，使未扰动 clean query 和参考谱统一获得更强的检索 embedding。

---

## 1. 已有矩阵已经回答了什么

当前矩阵覆盖 23,876 个 strict-10 ppm 查询、2,522 个身份、1,805 个官方错误。已有 S1/S2/S3A/A4 结果已经证明：峰动作有强效应，但效应高度依赖峰角色、剂量、步数和查询状态。

| 策略 | 代表结果 | 当前定位 |
|---|---:|---|
| candidate-gradient 25% 单峰 | 70 修正 / 51 新增，净 +19 | 低剂量广覆盖候选 |
| candidate-gradient 50% 单峰 | 138 / 113，净 +25 | 有纠错效应，但全局执行风险较高 |
| candidate-gradient 50% 动态 3 步 | 153 / 69，净 +84 | 当前主要广覆盖 N-arm 原语 |
| candidate-gradient 50% 动态 6 步 | 固定动作约 +0.41 pp | 效应继续累积，但边际不稳定 |
| candidate-gradient 75%/100% | 明显净退化 | 禁止作为默认增强 |
| role-confounder 100% 单峰 | 99 / 18，净 +81 | 当前最干净固定方向 |
| role-confounder 100% 动态 3 步 | 37 / 1，净 +36 | 高精度低覆盖动作 |
| role-confounder 最多 5 步 | step5 约 24 / 0 | 安全深度上限为 5 |
| role-identity 衰减 | 剂量越高损害越大 | 身份峰必须保护 |
| shared-only 衰减 | 所有剂量、步数均净退化 | 禁止把 shared 角色直接当删除标签 |
| unmatched | 少数小剂量/特定步数为正 | 仅作探索性补充 |

已有可视化清楚显示：

1. candidate-gradient 50% 的净修正随步数增加，但新增错误仍然可观；
2. role-confounder 具有最好的修正/新增比例，但覆盖有限；
3. shared-only 的新增错误主要流向同分子式、MCES 0–2 的 near 候选；
4. 同一个 shared 峰既可能修正错误，也可能破坏正确异构体排序，因此峰角色不能脱离候选关系使用。

现有矩阵属于**动作效应矩阵**。它回答“如果已知该执行哪个动作，谱图会怎样变化”，尚未回答“该动作的规律能否被共享编码器从 clean 谱图中学会”。

---

## 2. 为什么动作上限进入模型后大幅缩水

### 2.1 outcome oracle 使用了事后答案

S1–A4 的 3.35–3.85 pp 是对每个查询观察全部动作结果后选择最佳动作：

\[
a^*(q)=\arg\max_a U(q,a).
\]

训练和部署时只能依据干预前信息预测动作。动作策略存在任何误选，都会把 oracle 修正转化为新增错误。该差距属于选择误差，不是优化器能够自动消除的误差。

### 2.2 动作是候选条件化的，embedding 是候选无关的

一个峰相对于负候选 A 可能是 confounder，相对于候选 B 可能是必要身份证据。动作矩阵使用正确候选与困难负候选定义峰角色；最终 encoder 推理时只接收一张谱图。将候选相关的多个最优方向压成单一 embedding 残差，会产生冲突梯度。

### 2.3 历史动作只改变增强视图，没有强迫 clean 视图继承

若损失只奖励：

\[
m(x_{aug})>m(x),
\]

模型可以只学会处理人工删除后的输入。部署输入仍为 clean 谱图，因而增强视图的纠错不必转化为：

\[
m_\theta(x_{clean})>m_0(x_{clean}).
\]

早期实验和当前 F1 的核心断点都位于这里。

### 2.4 positive-deficit 是主体，纯删除动作只能处理一条分数臂

1,805 个错误中，1,439 个包含 positive-deficit；A4 未恢复的 1,029 个错误中，910 个包含 positive-deficit。删除峰可以压低错误候选，却不能恢复跨仪器、碰撞条件下缺失的同身份正证据。仅使用 N-arm 的理论上限和可迁移增益都会受到硬限制。

### 2.5 强动作同时放大纠错和伤害

A4 从 25% 到 100% 的可恢复错误为 138、289、461、738；受损正确对照同时增长为 183、375、652、1,078。大 margin 梯度并不等于大安全梯度。真正需要优化的是：

\[
G_{safe}=E[\Delta m\mid wrong]-\beta E[-\Delta m\mid correct],\quad \beta>1.
\]

### 2.6 共享编码器会同时移动 query 与 reference

历史部分动作和教师审计在固定官方候选 embedding 下评估，只移动 query。统一 embedding 要求 query、positive、negative 都经过同一新编码器。若某一策略把同类峰在三者中同时放大或缩小，原先固定参考下的 margin 增益会被抵消。

### 2.7 当前学生结构和训练样本没有消费完整动作矩阵

F1 v1–v3读取 clean 峰 token 并蒸馏身份教师；它们没有把 identity-masked、confounder-masked、matched-random、动态 candidate-gradient 等真实扰动视图送入学生。v2 的通用 rank 项又比 teacher 项大约两个数量级；v3把大量“基线已正确、教师仅略增 margin”的样本作为主动监督，导致新增错误抵消修正。

因此，F1 失败不能裁决已有噪声矩阵无效。它只说明当前教师蒸馏实现没有完成动作效应到 clean embedding 的迁移。

---

## 3. 下一轮不再只有一张矩阵

下一轮建立三张用同一 query/action ID 对齐的矩阵。

### 3.1 矩阵 A：直接动作效应矩阵

回答动作是否真的改变正确方向。

每个单元保存：

- \(\Delta s_{pos}\)、\(\Delta s_{neg}\)、\(\Delta margin\)；
- corrected、introduced、neutral；
- target-minus-matched-random；
- overall、near、cross-condition、P/N/B/S arm；
- identity/formula cluster CI；
- 错误转移候选与 MCES 层级。

禁止以 best-of-action 作为模型性能；oracle 只保留为上限。

### 3.2 矩阵 B：梯度兼容性矩阵

回答不同策略能否在同一 encoder 内合并。

对每类损失计算共享可训练参数上的梯度：

\[
C_{ij}=E\left[\frac{g_i^Tg_j}{\lVert g_i\rVert\lVert g_j\rVert}\right].
\]

至少包括：

- clean identity ranking；
- P-arm cross-condition consistency；
- role-confounder；
- candidate-gradient 50%；
- matched-random consistency；
- baseline-correct safety；
- embedding preservation。

同时报告梯度范数、负余弦比例和按 near/non-near 分层结果。若两策略长期负相关，不允许简单加权求和；应使用分流 batch、PCGrad/梯度投影或独立 P/N 残差分支。

### 3.3 矩阵 C：clean-transfer 效率矩阵

回答动作潜力能否转化为干净谱图的新 embedding。

每个通过矩阵 A 的策略，使用完全相同的训练预算、adapter、fold 和 seed 单独训练一个 micro-model。评价只使用未扰动 clean 谱图，并定义：

\[
\eta_a=
\frac{(corrected-2\,introduced)_{clean\ student}}
{(corrected-2\,introduced)_{direct\ action}}.
\]

同时报告：

- clean Recall@1、MRR、near Recall@1；
- corrected/introduced 与公式簇 CI；
- train-rescue、inner-formula、outer-formula 三层迁移；
- query/reference 同时重编码后的真实结果；
- embedding preservation 与峰权重变化。

只有矩阵 A 为正、矩阵 B 不发生不可控冲突、矩阵 C 的 clean-transfer CI 为正的策略，才能进入正式联合微调。

---

## 4. 扩展噪声策略空间

### 4.1 错误臂和安全臂

| 分支 | 目标 | 样本来源 |
|---|---|---|
| P-arm | 恢复同身份跨条件正证据 | 同 IK14、同加合物、支持谱与评价正谱互斥 |
| N-arm | 降低错误候选的混淆证据 | baseline-wrong、negative-excess、低 margin |
| B-arm | 同时存在正例不足和负例过高 | P/N 单独过门后再组合 |
| S-arm | 防止正确查询被破坏 | baseline-correct、near、小 margin、历史 introduced |

### 4.2 峰选择器

| 选择器 | 角色 | 是否进入首轮 |
|---|---|---|
| matched-random | 严格因果对照 | 是 |
| role-confounder | 高精度 N-arm | 是 |
| candidate-gradient，identity峰保护 | 广覆盖 N-arm | 是 |
| role-identity | 方向负对照与保护标签 | 只作对照 |
| replicate-core | 同身份多谱稳定核心峰 | 是，P-arm保护 |
| replicate-conditional | 同身份条件特异峰 | 是，P-arm鲁棒性 |
| unmatched | 低覆盖探索 | 是，但低优先级 |
| shared | 仅作上下文特征 | 禁止直接定义删除 |
| token-attribution / integrated-gradient | 非线性峰贡献候选 | 第二轮 |

### 4.3 干预算子

原矩阵过度集中于衰减/删除。新矩阵加入按真实重复谱校准的扰动，避免拍脑袋设置噪声。

| 算子 | 水平 | 用途 |
|---|---|---|
| soft attenuation | 10%、25%、50%、75% | N-arm剂量曲线 |
| hard delete | 100% | 仅压力测试或高精度confounder |
| DreaMS mask token | 官方预训练协议 | 通用鲁棒基线 |
| empirical peak dropout | 按同身份重复谱峰出现率采样 | 条件缺峰鲁棒性 |
| empirical intensity jitter | 按匹配峰 log-intensity ratio 分位数采样 | 仪器/CE强度漂移 |
| empirical m/z jitter | 按仪器和 m/z 分层残差采样 | 质量误差鲁棒性 |
| low-intensity peak addition | 仅从同身份支持谱的条件峰分布采样 | P-arm缺失峰模拟；必须cross-fit |
| pair-preserving transform | 对同一中性丢失/峰对协同扰动 | 防止破坏化学成对关系 |

禁止从其他分子复制峰；任何 peak addition 的来源谱不得出现在对应评价正谱中。

### 4.4 剂量和步数

- 剂量优先使用重复谱经验分布的 q25/q50/q75，而非无限超参数扫描；
- candidate-gradient 固定比较 25%/50%，最多 1/3/5 步；
- role-confounder 比较 50%/75%/100%，最多 1/3/5 步；
- 每一步可 no-op，且必须重算候选关系；
- shared、identity峰不进入默认衰减动作。

### 4.5 分层轴

所有主要单元至少按以下层级报告：

- positive-deficit / negative-excess / both / boundary / correct；
- near MCES 0–2 / mid 3–5 / 其他；
- Orbitrap / QTOF / instrument unknown；
- 同仪器 / 跨仪器；
- 加合物；
- 峰数、候选数、baseline margin 分位数；
- core / conditional / identity / confounder / shared / unmatched 峰角色。

CE只在有限值子集中报告；缺失CE作为独立 unknown 层，不得误当作0。

---

## 5. 训练结构：把真实噪声送入共享编码器

每个 batch 由四条流构成。

### 5.1 Clean流

原始 query、真实正谱和困难负谱全部经过同一个新编码器，维持干净候选组排序。

### 5.2 P-arm流

使用真实跨条件同身份谱和经验校准的 conditional-dropout/intensity/mz 扰动。要求：

1. 同身份 clean/perturbed embedding 保持一致；
2. clean query 对同身份多谱 prototype 的相似度提高；
3. near负候选不得同步提高更多。

### 5.3 N-arm流

使用 role-confounder、candidate-gradient 50% 和通过 A4 安全门的动作视图。要求干预视图产生正确方向，并把候选分布 stop-gradient 迁移给 clean query。

### 5.4 Safety流

过采样 baseline-correct、near、小 margin 和历史 introduced 查询。使用 matched-random、官方分布保持和正确 margin floor，防止模型依靠统一收缩或扩张获得表面收益。

总损失：

\[
L=L_{clean}
+\lambda_P L_{P-invariance/rank}
+\lambda_N L_{N-counterfactual\rightarrow clean}
+\lambda_R L_{matched-random}
+\lambda_S L_{safety}
+\lambda_E L_{preserve}.
\]

每个损失先单独完成 micro-transfer，禁止第一次正式训练就全部相加。

---

## 6. 聚合策略

### 6.1 第一层：只组合梯度相容策略

若矩阵 B 显示 P/N 梯度总体相容，采用均衡多流 batch 和按identity等权的固定损失权重。

### 6.2 第二层：处理冲突策略

若 P/N 或 rescue/safety 梯度冲突明显，依次比较：

1. 交替优化、每步单独归一化梯度；
2. PCGrad或将冲突分量投影掉；
3. 两个零初始化残差分支：

\[
z=\operatorname{Norm}\left[z_0+g_P(x)\Delta_P(x)+g_N(x)\Delta_N(x)\right],
\]

其中 \(g_P,g_N\) 只读取当前谱图及采集元数据，不读取候选身份或结构答案。最终仍输出一个候选无关 embedding。

### 6.3 容量升级条件

- adapter连训练救援集都无法改善：实现/容量问题，先查梯度，再解冻最后一个Transformer block；
- 训练救援显著、formula-OOF失败：策略依赖身份/分子式，扩大化学覆盖或减少候选条件特征；
- formula-OOF改善但总体安全失败：加强S-arm和风险门，不增加模型容量；
- adapter通过三seed后，才允许last-block低学习率微调；禁止直接解冻全部7层。

---

## 7. 执行顺序和硬门

### E0：历史矩阵复现与统一manifest

1. 重新生成 S1/S2/S3A/A4 的统一 query-action 表；
2. 对齐corrected、introduced、winner candidate、MCES和随机对照；
3. 复现现有三张可视化，新增按错误臂和near分层的风险图；
4. 不训练模型。

### E1：经验噪声校准

从同身份、同加合物重复谱估计峰出现率、强度比和m/z偏差，冻结 q25/q50/q75 参数。必须报告支持身份数、分子式数、仪器层和跨条件覆盖。

**2026-08-26 执行补充：** E0 已在正式服务器数据上通过，统一了 1,625,901 条
query-action 证据。E1 的首轮真实数据审计发现，低重叠重复谱之间的原始缺峰率中位数可超过
70%，它是采集异质性的描述量，不能直接作为训练删峰比例。正式 E1 因此固定为三层输出：

1. 保留全部同身份重复谱对的原始描述分布；
2. 仅使用两侧峰簇数均不少于 5、共同峰不少于 3、峰 Jaccard 不低于 0.10 的谱图对估计剂量，且每个 IK14 在每个采集关系中等权；
3. E2 只筛查可靠分布的 q10/q25，并设置 30% 安全上限。该上限是动作筛查边界，不是经验最优值；q50/q75 禁止直接进入训练。

峰出现率仍只决定“优先扰动哪些峰”，重复谱对缺失率只决定“筛查多少峰”。二者不得再次混为一个删峰概率。

**E1 正式结果：** 3,412 个身份-加合物组、2,588 个身份、1,957 个分子式，覆盖
1,738 个跨条件组；共形成 402,458 个共识峰簇和 1,211,271 个匹配峰对。210,328 个
有向重复谱变异中有 104,952 个通过可靠性门，覆盖 2,449 个身份和 1,846 个分子式；
3 个采集关系具有足够支持，全部 E1 门通过。该结果授权进入 E2 动作筛查，不构成模型性能结论。

### E2-M0：动作矩阵冻结

E2 在任何新前向计算前拆成三臂并冻结全部格子：

- **corrective：** candidate-gradient、confounder、经验条件缺峰，以及后二者与条件缺峰的交集；
- **robustness：** 经验缺峰、强度抖动、m/z 抖动、低强度真实同身份峰添加；
- **negative control：** 历史上高风险的 shared-only 和通用随机删峰。

corrective 每个目标动作配 3 个峰数、强度、m/z 与角色匹配的随机对照。robustness 只作为第二观察视图，不能凭单次 Top-1 变化叫作纠错标签。shared-only 永久禁止进入训练。

**E2-M0 正式结果：** 已冻结 44 个单元，其中 corrective 28 个、robustness 12 个、
negative control 4 个；覆盖 cross-instrument、same-instrument cross-CE 和
same-instrument unknown-CE 三类经验采集关系。每个 corrective 目标固定配 3 个匹配随机
对照，所有格子必须完整报告，禁止依据结果删除或改名。该阶段只冻结实验家族，不含动作效果，
也不构成 embedding 提升。

### E2-M1：纠错动作的配对特异性扫描

M1 只执行 28 个 corrective 与 4 个 negative-control 单元。正式执行器固定以下口径：

1. 缓存的官方 embedding 只用于来源和复现审计；动作效应统一用同一次执行器产生的
   fresh-clean forward 与 perturbed forward 配对计算，避免把浮点/预处理差异误当作小效应；
2. 候选分子块和候选 embedding 每个 query 只构建一次，干净与扰动评分完全复用同一数组；
3. 随机对照优先匹配删除峰数、强度、m/z 和候选角色；同角色峰不足时允许透明降级为
   强度+m/z 匹配，并单独报告降级次数，禁止静默丢弃高阶动作；
4. 每个冻结单元均报告覆盖、corrected、introduced、risk-net、near-net，以及官方错误中
   target-minus-matched-random margin 的 identity/formula cluster bootstrap CI；
5. 只有覆盖不少于 100 个错误身份和 100 个错误分子式、两级 CI 下界均大于 0、
   corrected > introduced、corrected−2×introduced > 0 且 near-net 不为负的 corrective
   单元，才允许进入 E3；negative control 永不进入训练。

M1 仍然是冻结编码器上的动作发现，不是微调结果。它解决的是“哪些峰动作具有可复现的
特异梯度”，E4 才回答这些动作能否迁移为更好的 clean shared embedding。

**E2-M1 正式结果：** 4,998 个 query 上完成 303,638 次编码，32 个预注册单元全部有结果；
28 个 corrective 中有 18 个通过首轮 identity/formula CI、risk-net 和 near 安全门，4 个负
对照均未进入候选。通过模式高度结构化：candidate-gradient 的 step 3–6 和
confounder-only 的 step 1–6 全部通过；跨仪器与未知 CE 的“经验缺峰×候选方向”通过，
纯经验缺峰没有通过。该结果支持“噪声必须结合当前候选混淆方向”，不支持通用随机删峰。

首轮对照中约 28.9% 使用了角色匹配降级，因此 18 个单元不能直接全部进入训练。E2-M1b
固定执行完全角色匹配子集复核、动作家族一致性和28格联合 formula-cluster max-T 多重校正，
并输出新增错误图谱。只有通过 M1b 的单元才进入 E3。

**E2-M1b 正式结果：** 18 个首轮候选中有 14 个通过完全角色匹配、联合 max-T、多级 CI、
risk-net 与 near 安全复核。通过者收敛为四个机制家族：candidate-gradient（step 3–6）、
confounder-only（step 1–6）、cross-instrument missingness×positive-gradient（q10/q25）和
unknown-CE missingness×positive-gradient（q10/q25）。纯经验缺峰、缺峰×confounder 以及
已知 cross-CE 条件动作均未进入 E3。该收敛进一步说明：有效增强依赖候选margin方向，
通用采集噪声本身不构成纠错监督。

### E2：扩展动作矩阵

运行选择器×算子×经验剂量矩阵。每个目标动作配至少3个峰数、强度、m/z、角色匹配随机对照。固定动作只有在formula/identity CI下界为正且introduced受控时进入下一步。

### E3：梯度兼容性矩阵

在同一初始化和同一批数据上记录各损失梯度范数与余弦，确定哪些分支可以直接合并，哪些需要投影或分支化。

E3 使用架构无关的 embedding 切空间定义。对于单位 clean embedding \(z\) 和动作后官方
embedding \(z_a\)，蒸馏损失 \(1-\cos(z,\operatorname{sg}(z_a))\) 在单位球切空间中的目标
负梯度为

\[
g_a=z_a-(z_a^\top z)z.
\]

对14个单元报告 \(\|g_a\|\)、与真实候选margin方向的对齐度、逐query梯度余弦、负余弦比例
和formula-cluster CI；再在四个机制家族内先聚合嵌套剂量，形成4×4家族兼容矩阵。E3不训练
模型，它决定E4损失能否相加；候选信息只用于训练动作构造，最终共享编码器推理仍与候选无关。

**E3 正式结果：** 14 个单元覆盖 27,735 个动作、1,104 个身份和 408 个分子式。四个家族
的聚合方向在 formula-cluster 水平均未出现整体负 CI，但“全部兼容”不得解释为可以等权相加：

- cross-instrument 与 unknown-CE 两个 positive-gradient 缺峰家族几乎共线
  （平均余弦约 0.949），在 E4 合并为一个 acquisition-conditioned 分支，避免双重计权；
- candidate-gradient 与合并后的 acquisition 分支中度同向（约 0.34），可以共同训练但必须按
  梯度范数归一化；
- role-confounder 与 candidate-gradient 的平均余弦仅约 0.074，逐 query 约 39.5% 为负；
  它必须保留为独立分支，并在实际 batch 梯度为负时采用冲突投影，禁止简单求和；
- acquisition 动作的平均切向幅度约 0.34–0.37，是 candidate/role 的约 2.3 倍，故 E4
  必须做分支归一化，否则采集条件分支会仅凭数值尺度支配训练。

因此 E4 的正式结构从“四个损失相加”收敛为三个机制分支：candidate-gradient、合并后的
acquisition-conditioned positive-gradient、role-confounder。每个分支仅保留预注册的低/高
剂量视图，身份内等权；动作 embedding 只作为训练目标，推理时只输入未扰动原始谱图。

### E4：单策略clean-transfer micro-train

E4-M0 先冻结动作目标缓存：candidate 使用 E2-000/E2-003，role 使用 E2-004/E2-009，
acquisition 使用 E2-012/E2-015/E2-024/E2-027。跨仪器和 unknown-CE 是同一损失分支中的
不同数据层，而不是两个可重复加权的损失。缓存不按 corrected/introduced 对单样本加权，
避免把冻结编码器上的结果标签偷渡为训练 oracle。

E4-M1 使用一个零初始化、峰 token 条件化、范数受限的共享 residual adapter。查询谱图、
正例参考谱和困难负例均通过同一 adapter；训练时 clean 谱图被拉向 E4-M0 的动作目标，同时
以 clean 候选排序、安全 margin 和表征保持约束阻止新增错误。三个机制分支按身份等权、
按梯度范数归一化；仅在真实 batch 梯度冲突时做投影。P2b、候选重排分数及规则重叠标签
均禁止进入 E4。

**E4-M1 首轮工程审计（作废版本）：** 原实现直接相加原始损失，动作损失约 0.0018，
加权 clean-ranking 损失约 0.37，后者在数值上约大 200 倍。PCGrad 又是在三个均被
ranking 主导的混合目标之间执行，因此模型实际进行的是普通排序微调：inner clean Recall@1
在 epoch 1 上升约 0.118 pp（6 corrected / 1 introduced），但三个动作家族的 held-out
target-cosine gain 全为负，且随 epoch 单调恶化。该结果不能用于否定噪声动作，仅证明
raw-loss-scale 混合协议错误；best epoch 被安全门正确回退到 epoch 0。

E4-M1b 禁止原始损失尺度直接混合。先进行 1 epoch 三家族 action-only warm-up；随后把三个
action objective 与一个 clean ranking/safety/preservation objective 分别求梯度、各自单位化，
再做 PCGrad，最后恢复到原始梯度范数中位数。这样动作方向和 clean 安全方向各自拥有明确
表决权，训练日志必须同时报告原始梯度范数、冲突次数和各家族 held-out target gain。

**E4-M1b 联合训练结果与纠偏：** 梯度平衡消除了首轮约 200 倍的量纲压制。action-only
warm-up 后 acquisition 与 role 的 held-out gain 分别约为 +2.07e-5 和 +3.38e-5，但
candidate-gradient 为 -6.99e-6。加入 clean 目标后，inner clean Recall@1 最好提高约
0.118 pp（5 corrected / 0 introduced，near +0.164 pp），同时 candidate 和 acquisition
动作方向转负，因此安全门仍回退 epoch 0。该结果不能被解释为全部噪声机制失败：它证明
三个机制过早联合，且 candidate-gradient 依赖当前困难负候选，不一定能作为 query-only
蒸馏方向泛化。按“E4 单策略、E5 才组合”的原始纪律，下一步固定所有其余协议，分别运行
candidate、acquisition、role 三个单机制 pilot。candidate 单机制实验是机制诊断；若仍不能
实现 held-out action，它应转为候选排序/困难样本损失，而不是继续作为 query-only target。

每个策略使用相同训练预算；先1 fold×1 seed筛查，再对通过者做5 folds×3 seeds。硬门：

- train-rescue有明确改善；
- inner formula OOF 的 clean Recall/MRR/risk-net CI为正；
- near不退化；
- corrected > introduced，且 corrected−2×introduced >0；
- query和reference均由同一encoder重编码；
- zero-init严格复现官方排名。

### E5：有限组合

只组合E4通过且梯度互补的前2–3个策略。固定组合顺序：

1. P-arm经验扰动与跨条件一致性；
2. role-confounder；
3. candidate-gradient 50%；
4. matched-random safety。

每增加一个分支必须做配对增量检验；新增错误上升则回退。

### E6：容量升级和正式多折

adapter通过后，再比较last-block adapter。冻结单一配方后完成5 folds×3 seeds，随后才进入未消费测试集。

---

## 8. 性能目标与科学边界

目标是获得全面优于官方DreaMS的统一embedding，但当前不能预先保证4 pp。已有矩阵的3.35–3.85 pp是oracle上限，学生能达到多少取决于动作选择、梯度兼容性和clean-transfer效率。

本轮真正的成功标准不是“找到更多能改变margin的峰”，而是同时成立：

1. 动作相对匹配随机对照有特异效应；
2. 该效应在formula隔离下可预测；
3. 该效应可迁移到未扰动clean谱图；
4. 同一个共享encoder重编码参考谱后仍保持净收益；
5. overall、near、跨条件三项均不被单一亚组掩盖；
6. 新增错误有明确的候选与峰级解释，并随Safety流下降。

只有完成这六步，才能把“高潜力峰动作”变成“高性能embedding微调方法”。

---

## 9. E4 单机制结果后的强制审计（2026-08-27）

### 9.1 先纠正“教师有 4–5 pp”这一口径

噪声路线中没有一个可部署教师被证明具有 4–5 pp 增益。历史正式结果是：单个固定动作的
净收益远小于 1 pp；逐 query 读取动作结果后选择动作或停止的 no-op-aware oracle 最多覆盖
663/23,876 个错误，即约 +2.78 pp。它使用答案选择动作，只是上限，不是学生可观察的监督
函数。约 4 pp 的结果来自独立的 P2b 候选重排路线，严禁移植为噪声教师能力。

### 9.2 三个单机制 E4 pilot 的结果

- candidate-gradient：inner 一度 5 corrected / 0 introduced，但 held-out action gain 从 warm-up
  后转负；安全选择回退 epoch 0，outer 无变化。
- acquisition-positive-gradient：选择 epoch 1；outer action gain 为正，但 Top-1 0 corrected /
  0 introduced，仅 MRR 极小变化，未形成检索改进。
- role-confounder：inner 最好 5 corrected / 0 introduced；outer 2 corrected / 1 introduced，
  约 +0.017 pp，且 outer action gain 为负，不通过风险门。

这不是“三种噪声都没有信号”，而是当前 E4 没有正确实现预期的噪声微调任务。

### 9.3 E0–E4 的逐级责任审计

- E0 只整理历史证据，正确；但 oracle 上限与固定策略/可部署教师必须分列。
- E1 正确估计了真实重复谱中的采集扰动分布；它不是检索教师。
- E2 正确测量了“扰动查询、冻结候选参考”下的即时动作特异性；通过单元只说明动作有效，
  不说明动作可由干净查询预测，也不说明共享编码器更新后仍有效。
- E3 实际计算的是 embedding 切空间中的期望更新方向，不是当前 adapter/backbone 参数上的
  真实训练梯度。它可用于描述动作方向，但“gradient compatibility”不能作为可共同训练的充分证据。
- E4-M0 把多步、大剂量、可停止的动作压成固定长度 0.06 的单步方向，并丢失了历史策略的
  停止规则和动作收益大小。
- E4-M1/M1b 使用冻结 DreaMS 加一个 pooling 后 residual adapter；它没有微调 Transformer
  内部的峰 token 交互。更关键的是，它让 clean input 模仿 perturbed-output。真正的鲁棒噪声
  训练应当是 noisy input 回到 clean/identity target；候选纠错则应直接优化 clean/noisy 两个
  视图相对真阳性和困难负例的 margin。

### 9.4 当前实现的四个结构性错位

1. **oracle 到教师错位**：candidate/role 动作依赖当前真候选与错候选，query-only adapter
   看不到这一条件；同一类干净谱图可能收到相互抵消的目标方向。
2. **扰动方向反置**：采集鲁棒分支训练 clean→noisy target，而不是 noisy→clean/identity target，
   会把采集缺失写入干净 embedding。
3. **容量位置错位**：pooling 后 adapter 只能修正最终向量，不能改变峰 token 在 Transformer
   注意力中的贡献；它不是原计划中的 DreaMS 噪声微调。
4. **评估几何错位**：训练 hard negatives 固定于官方 embedding，正例取官方最相似谱；没有
   动态重挖当前模型的最难负例，也没有系统覆盖跨条件正例。E2 的 query-only 动作收益因此
   不能直接迁移到共享 query/reference encoder。

### 9.5 修正后的下一阶段（E4-R）

停止扩展当前 clean→perturbed residual 蒸馏。下一版必须分成两个语义清楚的训练分支：

1. **真实采集鲁棒分支**：输入 E1 采样的 noisy spectrum，目标为 stop-gradient clean spectrum
   或同身份跨条件 prototype；clean/noisy 均须保持对真阳性的排序，并共享同一编码器。
2. **特权峰纠错分支**：candidate/role 动作仅作为训练期 harmful-peak 标注；原谱与定向扰动谱
   都通过学生，直接优化对真阳性与困难负例的组内 margin，并训练 query-only 峰门控预测这些
   特权标注。推理时不需要候选身份或峰删除动作。

模型容量从 pooling 后 adapter 升级为最后一个 Transformer block 的轻量 adapter/LoRA 加共享
projection head。正例覆盖所有跨条件重复谱，负例按当前学生动态重挖；protected-correct 流继续
承担 preservation。正式训练前先做 formula-OOF harmful-peak 可预测性门：若干净峰 token 无法
预测动作标签，则该 selector 只能用于候选感知排序损失，不能作为 query-only 峰门控教师。

---

## 10. R0/R1：忠实恢复历史噪声策略并冻结特权教师（2026-08-27）

为避免再次把历史 S3A 动态动作压缩成静态 embedding 终点，R0 直接读取原始
`selected_sequences.csv.gz` 与 `paired_interventions.csv.gz`，保留：逐步峰路径、每一步重新挖掘的
困难负例，以及两条角色/强度/mz 匹配随机对照。固定策略只锁定两个经历史矩阵支持的家族：

- candidate-gradient，attenuation=0.50，step 3–6；
- role-confounder，attenuation=1.00，step 1–5。

R0 已在本地逐 cell 复现图中全部 corrected/introduced 数字，得到36,934条动作记录、1,991个身份、
877个分子式。动作结果单独存放，学生输入清单不含 corrected/introduced 等结果字段。

R1 明确拆开两类量：

1. 固定 S3A 策略效果，用于证明动作定义和历史代码被忠实复用；
2. 训练期 outcome-selected 特权教师，用于最大化未见分子式学生可学习的纠错梯度。

本地可完整物化的 S3A 全动作 oracle 为520个错误，A4 policy-eligible 精确动作 oracle 为776个，
两者并集882个，对应训练图谱上3.694 pp的可物化上限；历史 S1c/S2/S3A/A4 全并集为920个，
即3.853 pp，但其中38条旧轨迹当前未在本地动作文件中物化。3.853 pp仍是看过动作结果后选择动作或
不动作的训练图谱上限，禁止写成教师或学生模型性能。

R2不得再做clean embedding拟合perturbed embedding。正式学生采用一个共享DreaMS编码器：

- corrective分支同时编码clean谱与特权动作谱，直接优化它们相对真阳性和当前困难负例的候选组内margin；
- robustness分支使用安全的role-confounder/经验采集扰动，执行noisy-to-clean/identity consistency；
- query、正例和负例全部经过同一学生；困难负例按当前学生每epoch重挖；
- protected-correct查询保留margin floor与官方embedding preservation；
- 推理仅输入clean谱图，不输入候选身份、动作路径或oracle结果。

R2首先运行公式隔离的小规模可实现性门。只有学生在未见公式上同时满足clean Recall提升、near不降、
corrected大于introduced、preservation达标，并且动作相对匹配随机对照的margin增益为正，才允许进入
多折多seed全量训练。

### R2 工程冻结实现（2026-08-27）

R0/R1 服务器正式运行已经逐项复现历史 S3A 数字，并物化 882 个可纠错查询。需要特别注意：
882 个查询只来自 462 个身份与 248 个分子式，不能按 882 个独立分子处理。R2 因而固定采用
identity-equal epoch sampling 与 formula-held-out evaluation。

R2 的正式实现为 `train_noise_final_r2_shared_encoder.py`，边界如下：

1. query、positive 与 negative 使用同一个 DreaMS 编码器；推理只输入 clean spectrum；
2. 只解冻最后一个 Transformer block 与官方 projection head，前六层冻结；
3. 模型保持 eval mode 但允许梯度，关闭此前已证实会混杂细粒度 margin 的 dropout；
4. corrective 流同时优化 clean/action 的组内 margin，并将 clean 表征向 stop-gradient action 表征转移；
5. 对保留两条匹配对照的 S3A 动作，加入 action-vs-matched-control specificity；
6. 每个 batch 在固定的 top candidate shortlist 内按当前学生相似度重新选 hardest negative；
7. protected-correct 与 role-confounder robustness 流分别承担 margin floor、clean/action consistency
   和官方 embedding preservation；
8. P2b、RAW/token reranker、P3 测试集均禁止进入训练、样本权重和损失。

提交训练前必须先运行 `run_noise_final_r2_preflight.sbatch`，在真实 HDF5 上逐条重放全部动作并核验
五折公式隔离、样本覆盖和可用上限。预检通过后才运行三档低学习率 pilot；held fold 固定 epoch 后只评估
一次，不作 checkpoint 选择。3.853 pp 继续只作为训练图谱的历史 oracle 上限，R2 的有效性能只能由
held clean query 的 Recall@1、MRR、near、corrected/introduced 与 formula-cluster CI 决定。

### R2 首轮持出公式结果与强制排障（2026-08-27）

R2 的 FP16 零变化复现错误已经修正；正式复跑的两档学习率均在 FP32 下精确复现官方空间
（初始 preservation=1.0，rank mismatch=0）。因此后续结果不是数值基线漂移造成的。

两档模型均显示“优化信号存在、clean 迁移失败”的同一模式：

- `1e-6/5e-6`：训练 clean margin 从约 -0.052 改善至 -0.040，但 held clean Recall@1
  下降 0.084 pp，1 corrected / 6 introduced；
- `2e-6/1e-5`：训练 clean margin改善至约 -0.029，但 held clean Recall@1下降0.034 pp，
  5 corrected / 7 introduced；
- held action view 仍有157/171可正确排序，而clean view仅有1–2/171转正；
- action-clean margin差仍约0.079，未随四轮训练实质收缩；preservation仍为0.9990–0.9996。

这证明学生并非没有梯度，而是主要学到了训练动作/训练候选上的局部margin，未把特权峰动作迁移到
未见分子式的clean谱图。当前实现还存在三项必须在扩大训练前拆开的因素：

1. `clean -> stopgrad(current student action)`是随参数漂移的移动目标，不是冻结官方动作教师；
2. 每个身份每轮只抽一个动作，711条训练动作被压缩为约372个step-level样本，动态路径覆盖不足；
3. S3A/A4 corrective、role-confounder robustness、protected replay和最后一层解冻同时加入，无法区分
   固定教师失效、损失冲突和容量/覆盖不足。

因此R2不进入多折全量训练。先运行`audit_noise_final_r2_transfer.py`，在不调参、不读取P3的条件下，
分别量化训练/持出公式上的固定官方动作教师、学生action view和学生clean view，并比较clean表征是否
真的靠近固定教师候选分布。只有排障证明固定教师对clean可迁移后，下一版才允许：固定teacher logits、
全动作identity-equal加权、P/N单臂消融、inner-formula checkpoint选择，最后再组合并扩大。
