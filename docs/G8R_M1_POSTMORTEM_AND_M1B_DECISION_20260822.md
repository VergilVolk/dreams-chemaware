# G8R M1 首门复盘与 M1b 决策

**日期**：2026-08-22  
**结论**：G8R M1 当前配置未通过预注册门；这次结果否定的是“当前数据覆盖与损失配比下的这一套 head-only 配置”，并未否定 projection head、InfoNCE 或冻结表示本身。

## 1. 实测结果

| 指标 | 官方 DreaMS | G8R M1 | Δ |
|---|---:|---:|---:|
| 跨条件同分子 cosine | 0.7090 | 0.7382 | +0.0292 |
| hard-negative cosine | 0.5084 | 0.5387 | +0.0303 |
| strict-10ppm macro-AUC | 0.87325 | 0.87231 | −0.00094 |
| Recall@1 | 0.80806 | 0.80806 | 0 |
| preservation cosine | — | 0.99465 | 未达到0.995门 |
| corrected / introduced | — | 4 / 4 | 净改善0 |

按预注册标准，hard-negative 方向和 preservation 两门失败，因此 M1 **不通过**。总体检索处于非劣效范围，但没有改善。

## 2. 这次结果能说明什么

1. 当前配置显著提高了跨条件正例余弦；真实正例监督确实进入了模型。
2. 困难负例余弦也提高，正负间隔约变化为：

   \[
   \Delta[(s_{pos}-s_{neg})]\approx 0.02918-0.03026=-0.00108
   \]

   因此局部排序没有改善。
3. macro-AUC、Recall@1 和错误转换均显示模型基本保持原状，不能称为性能提升。
4. 当前结果是有价值的失败：正例保护已不再是 G5/G6/G7 的主要问题，新的瓶颈是困难负例没有获得足够、有效且相对化的监督。

## 3. DeepSeek 解释中的错误

### 3.1 “head-only 路线已死”没有证据

当前 head 是约 105 万参数的 1024×1024 全线性投影。归一化后的线性投影等价于学习一个全局 Mahalanobis 度量，可以各向异性地重排方向；它不是只能把所有 cosine 等幅放大。一次训练失败不能证明冻结 embedding 中不存在可分信号。

是否需要解冻 backbone，应由冻结 embedding 的线性/双线性可分性诊断决定，而不是由一次 head checkpoint 决定。

### 3.2 “InfoNCE 的 logsumexp 必然导致一起放大”不成立

单正例、单负例的 InfoNCE 可写成相对差值上的 softplus：

\[
L=\log\left(1+\exp\frac{s_n-s_p}{\tau}\right)
\]

它本身优化的就是正负相对顺序。直接把它换成无诊断的 triplet hinge，并不能自动解决问题。

当前实现还叠加了绝对正例损失：

\[
0.25(1-s_{real-pos})
\]

而这个绝对拉近项作用于所有有真实正例的 anchor。必须先分解各损失梯度，才能判断是 InfoNCE、绝对正例项、样本覆盖还是权重失衡。

### 3.3 “+0.029 与 +0.030 证明两簇一起放大”过度推断

两个均值来自不同规模的集合：1,000 个跨条件正对与446个 hard-negative pair；hard-negative 只涉及125个独特负谱。必须在**同一 anchor、同一正例、同一困难负例**上计算 paired margin，才能讨论“一起变化”。当前 pair-level bootstrap 还可能因重复负谱低估不确定性。

### 3.4 “4修正/4新增等于掷硬币”表述不严谨

它只能说明净错误数没有改善。8次翻转样本太少，不能推断随机机制，也不能据此判断具体损失形式。

## 4. 被忽略的首要混杂：困难负例覆盖严重不平衡

锁定训练集共有10,000 anchors，但：

- 真实跨条件正例覆盖率：100%；
- hard-negative anchor覆盖率：34.34%；
- 65.66%的anchor没有显式困难负例。

在 `hard_only` 模式中，没有困难负例的样本其 InfoNCE 排序项为零或近零，但仍接受绝对正例拉近和教师保持梯度。因此大多数更新更容易表现为“提高正例/整体局部相似度”，而不是推开困难负例。这是 M1b 必须首先修正的设计问题。

## 5. 先做诊断，不立即选择 A/B/C

### D0.1 同anchor配对间隔审计

只在同时具有真实正例和困难负例的anchor上计算：

\[
g_i=s(a_i,p_i)-\max_j s(a_i,n_{ij})
\]

输出 baseline、candidate、\(\Delta g_i\)、pairwise accuracy、margin violation rate。按 anchor IK14 或分子式簇 bootstrap，不能按重复 pair 当独立样本。

### D0.2 按训练监督覆盖分层

分别比较：

- 训练中具有 hard negative 的anchor；
- 只有真实正例、没有 hard negative 的anchor；
- near 与 mid hard negative；
- nucleoside/purine、amino acid/peptide 与 other。

若只有“无困难负例”组发生整体余弦上移，损失覆盖失衡得到直接验证。

### D0.3 损失与梯度分解

在固定batch分别记录：

- \(L_{NCE-hard}\)；
- 绝对真实正例损失；
- feature preservation；
- positive-relation preservation；
- 每项对head参数的梯度范数与梯度余弦。

若绝对正例梯度显著大于或持续抵消困难负例梯度，才能据此修改损失。

### D0.4 冻结表示可分性探针

用官方冻结embedding构建诊断探针：

- 输入：\(|z_a-z_b|\)、\(z_a\odot z_b\) 或受约束双线性分数；
- 标签：真实跨条件同分子 vs 固定10 ppm困难负例；
- 严格IK14/Murcko隔离；
- 只用于判断冻结表示是否含有可分信息，不作为最终模型结果。

若简单探针在独立验证集明显优于原始cosine，head-only仍有潜力；若探针也失败，才有依据进入adapter或最后一层解冻。

## 6. M1b 的正确训练设计

M1b 不是“把InfoNCE机械换成triplet”，而是把监督单元改成完整的局部排序三元组。

### 6.1 训练单元必须完整

每个用于排序更新的anchor必须同时具有：

1. 真实跨条件同分子正例；
2. 至少一个固定±10 ppm、同adduct优先的困难负例。

无困难负例的anchor不得进入局部排序损失；它们只能用于教师保持或后续独立噪声任务。采用双流batch：hard-ranking stream与preservation stream分开计数、分开记录。

### 6.2 去掉无条件的绝对余弦最大化

暂停 `0.25(1-s_real-pos)`。正例保护改为教师相对下界：

\[
L_{pos-floor}=[s^{official}_{pos}-s^{student}_{pos}]_+
\]

它只阻止真实正例退化，不要求把所有正例余弦继续推到1。

### 6.3 使用显式相对间隔，但不迷信hinge

首选平滑margin ranking：

\[
L_{rank}=\log\left(1+\exp\frac{m+s_n-s_p}{\tau}\right)
\]

其中 \(m\) 先固定为0.05；同时保留 hinge-triplet 作为单一消融，而不是直接替代所有设计。多负例时使用 hardest-negative 或 logsumexp negatives，但每个anchor等权，不能让负例多的分子主导。

### 6.4 当前不做的事情

- 不先加 shared-main-peak 反事实；当前普通困难负例监督尚未证明有效。
- 不解冻 Transformer；冻结表示可分性尚未测清。
- 不扫大规模LR/λ；M1b先跑一个锁定配置。
- 不改变当前M1验证集并继续称为同一预注册实验。新建M1b协议与输出目录，保留M1失败记录。

## 7. M1b 通过门

### 主门

- 同anchor paired margin \(s_p-\max s_n\) 的平均变化>0，cluster-bootstrap 95% CI下界>0；
- hard-panel pairwise accuracy提高至少1个百分点；
- strict-10ppm macro-AUC ≥ baseline−0.005；
- Recall@1 ≥ baseline−0.003；
- corrected > introduced；
- preservation cosine ≥0.995。

### 次门

- 跨条件正例不低于官方教师；
- near/mid及极性代谢物子组方向一致；
- 至少500个独特hard anchors用于正式困难面板；不足时只称pilot。

只有M1b通过，才进入共享主峰反事实和噪声消融；若冻结可分性探针显示信号存在、但M1b仍失败，再检查优化器/采样；若探针本身失败，才进入轻量adapter或最后一层解冻。

## 8. 最终决策

当前不选 A、B 或 C。正确顺序是：

> **D0配对诊断 → M1b平衡的局部相对排序 → 通过后做反事实噪声 → 仍无可分信号时才解冻。**

这条顺序既利用了本次“真实正例已经修好”的成果，也防止团队再次同时修改数据、损失和模型结构，无法知道性能变化来自哪里。

## 9. D0实测更新与M1b最终配置（2026-08-22）

### D0已经支持继续冻结backbone

- 同anchor完整三元组314个；官方margin均值0.143，IK14-cluster bootstrap 95% CI [0.107, 0.181]；pairwise accuracy 72.9%。
- near（MCES 0–2）margin仅0.081，mid（MCES 3–5）为0.234，主要违反集中在near。
- 冻结embedding上的Hadamard探针AUC=0.832、concat探针AUC=0.845，高于原始cosine AUC=0.758。

因此M1b继续采用冻结backbone。D0.3旧损失梯度分解降为非阻塞复盘：有checkpoint时补跑，但不等待它启动M1b。

### D0.4仍需补一个严谨性检查

当前5折按anchor IK14分组，但同一个负例IK14可能作为另一对的分子出现在不同折；Hadamard线性分类器也可能学习负权重，而线性head诱导的 \(W^TW\) 必须是PSD。因此“head具有潜力”成立，但“现有探针严格证明head一定能表达该判别器”尚未成立。

并行补做：

1. 直接以`g8r_locked/train.json`拟合、`val.json`测试；两套split已将anchor和负例IK14限制在各自partition；
2. CV时按正负关系图的connected component分折，而不是只按anchor IK14；
3. 增加非负对角Hadamard或低秩PSD metric probe，报告其相对cosine增益。

这些检查不阻塞M1b，但在论文中使用D0.4结论前必须完成。

### M1b回答1：采用双流，但排序epoch由3434个hard anchors定义

- **hard-ranking stream**：只含同时有真实跨条件正例和hard negative的3434个train anchors；每个step都计算rank、positive floor和preservation。
- **safety-preservation stream**：从完整10000 anchors轮换采样，只计算student-vs-official preservation；不计算绝对正例拉近。
- 每个optimizer step各取一个hard batch和一个safety batch；两个流的loss分别先取mean，再组合，不能因easy样本数量更多而扩大权重。
- “一个epoch”定义为hard stream完整遍历一次，约3434/32≈108 steps；不再让10000条easy样本决定313 steps。

这既使用完整10000作为安全覆盖，又保证每一次判别学习都有负例。不能把全部10000混进同一个ranking loader。

### M1b回答2：确认去掉全部合成噪声

M1b只回答“真实跨条件正例与真实困难负例能否形成更好的局部顺序”。关闭随机删峰、强度扰动、m/z扰动、加峰和合成positive。噪声在M1b通过后进入独立消融。

### 固定目标函数

\[
L=L_{rank}+\lambda_fL_{pos-floor}+\lambda_pL_{preserve}
\]

\[
L_{rank}=\operatorname{softplus}\left(\frac{m+s_{n^*}-s_p}{\tau}\right),
\quad m=0.05,\;\tau=0.1
\]

其中 \(n^*\) 为该anchor在当前固定候选中的最难负例。首个配置保持锁定数据中的自然near/mid分布，不额外重加权；同一anchor同时拥有near与mid时，由hardest-negative机制自然优先更难者。`L_pos-floor`只惩罚低于官方正例相似度的部分，首跑固定 \(\lambda_f=1\)。`L_preserve`同时覆盖hard流和safety流，两个流等权求均值；首跑保留 \(\lambda_p=5\)，不做数组扫描。

### M1b首门必须检查的是delta，不是candidate绝对值

主门改写为：

- 同anchor \(\Delta(s_p-\max s_n)\) 的IK14/关系图cluster-bootstrap 95% CI下界>0；不能只检查candidate margin本身>0，因为官方baseline已经为正；
- pairwise accuracy从72.9%至少提高到73.9%，同时报告McNemar/cluster CI；
- near组margin与违反率必须改善，mid组不得显著退化；
- macro-AUC与Recall@1保持原非劣界；
- corrected > introduced；
- preservation cosine≥0.995。

当前nucleoside/purine 174 anchors中没有hard negative，所以M1b不能声称改善核苷注释。极性代谢物hard panel必须另行构建，并在模型冻结后作为独立定向测试；不得把`other`上的提升外推到核苷。
