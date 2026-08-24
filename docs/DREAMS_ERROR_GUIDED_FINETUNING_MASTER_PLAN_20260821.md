# DreaMS 已知错误驱动的鲁棒微调课题执行总纲

**版本**：2026-08-21 v1.0  
**适用范围**：DreaMS 谱图表征与谱库检索微调；本文件暂不包含化学可解释性模块。  
**核心目标**：针对已经被真实数据确认的跨条件同分子分离、共享主峰误聚、质量近邻异构体混淆，以及核苷/嘌呤等极性代谢物错注，得到一个可重复、不会牺牲原有能力、能够接受外部盲评的改进模型。

---

## 0. 执行摘要：团队首先必须统一的判断

当前已经得到的最重要结论不是“噪声越强越好”，而是：

1. **DreaMS 的错误有结构**。一类是假阴性：同一分子的谱图因为仪器、碰撞能量和条件特异峰而分开；另一类是假阳性：不同分子因为少数共享高强度峰而过度接近。
2. **G5、G6、G7 是失败实验，不是候选模型**。它们把异构体推远了，但同时把真实同分子谱图推散，导致全量严格 10 ppm macro-AUC 比官方模型下降 3.1–4.6 个百分点。
3. **原来的合成噪声定义混淆了两种相反操作**。身份正例需要保护稳定主峰、扰动弱峰和条件特异峰；共享主峰困难负例才需要定向削弱已经证实导致误聚的主峰。
4. **短期目标不是宣称“全面超越所有分子检索方法”**。本阶段要先在“谱图—谱图的跨条件同分子检索与质量近邻区分”上超过官方 DreaMS。MassSpecGym 的候选分子检索已有 FLARE、MVP 等跨模态模型，任务输入和输出与当前谱图 embedding 检索不同，不能混用指标宣称 SOTA。
5. **SOTA 是最终评估结论，不是研发口号**。只有在固定测试协议、独立分子划分、强基线、置信区间和外部数据上同时成立，才能使用 SOTA 表述。

本课题的最短闭环为：

> 锁定错误类型 → 构造真实正例和局部困难负例 → 保守 head 微调 → 三门安全评估 → 峰级定向增强消融 → 必要时小范围解冻 → 极性代谢物挑战集 → 独立外部盲评 → 冻结模型。

---

## 1. 本课题究竟解决什么问题

### 1.1 主要科学问题

在不破坏 DreaMS 已有通用谱图表征能力的前提下，能否使 embedding 同时满足：

- 同一分子在仪器和碰撞能量变化后仍保持邻近；
- 前体质量接近但结构不同的分子保持足够间隔；
- 对核苷、嘌呤、氨基酸、含硫代谢物、泛酸和色氨酸代谢物等极性小分子减少真实错排；
- 对低质量或缺峰谱图具备鲁棒性，同时不对任意不完整谱图无条件不变；
- 新模型产生的每个提升都能追溯到“修正了哪些错误、又新增了哪些错误”。

### 1.2 本阶段不解决的内容

- 不以规则重叠度作为正负标签；
- 不用规则 Jaccard 回归 embedding 距离；
- 不在本阶段宣称从谱图直接生成未知分子结构；
- 不把 HNSCC 的 36 张外部谱图用于训练或调参；
- 不把训练 loss 下降当作检索性能提升；
- 不以单个 seed、单个小面板或 1 个百分点的偶然变化作为最终结论；
- 不把候选结构排序等同于 MSI Level 1 鉴定。

---

## 2. 已有证据与当前基线

### 2.1 官方 DreaMS 的原始任务

DreaMS 原论文的谱图相似度微调采用真实结构身份监督：同一 IK14 的另一张谱图为正例，不同结构且前体分子质量差不超过 0.05 Da 的谱图为负例，使用 margin 为 0.1 的 triplet loss。原论文检索评估中，正例为同 IK14，负例限定为前体 m/z 差不超过 10 ppm。原论文还使用 NIST20 作为与 MoNA 微调分子互斥的外部评估集。

因此，本课题与 DreaMS 的关系是**在官方正确的身份监督框架上修正残余错误**，而不是重新发明正负样本定义。

### 2.2 已经复核的失败模型

| 模型 | 全量严格 10 ppm macro-AUC | Recall@1 | 与官方差异 | 结论 |
|---|---:|---:|---:|---|
| 官方 DreaMS | 0.8676 | 0.9301 | 基线 | 保留 |
| G5 | 0.8216 | 0.9201 | −4.6 pp | 失败 |
| G6 | 0.8354 | 0.9228 | −3.2 pp | 失败 |
| G7 | 0.8365 | 0.9223 | −3.1 pp | 失败 |

G5–G7 的共同机制：异分子最高余弦下降，但同分子不同谱的最高余弦也下降；后者是检索退化的主因。后续实验必须直接保护真实正对关系，不能只锚定单个 clean embedding。

### 2.3 真实跨条件谱图审计

在 2,000 对同 IK14、同 adduct，且仪器不同或 `|ΔCE|≥10` 的真实谱图对上：

- 两侧谱图可匹配峰比例中位数约 0.33–0.38；
- 相对基峰强度 ≥0.20 的高强度峰匹配比例中位数为 1.00；
- 匹配峰绝对 m/z 误差中位数为 0.00047 Da；
- 匹配峰强度比的 `|log ratio|` 中位数为 0.45。

据此锁定增强原则：

- 身份正例：优先扰动弱峰/条件特异峰，保护主峰；
- 困难负例：只对经过反事实删除确认的共享主峰做定向处理；
- 单峰独立 m/z 位移：关闭；
- 任意 m/z 随机加峰：关闭，直至被真实条件特异峰分布替代。

### 2.4 外部极性代谢物错误

MTBLS1905 的固定外部盲评包含 36 张查询谱和 18 个已发表 connectivity 靶标。在相同的 ±10 ppm 候选池中：

| 方法 | 谱图级 Top-1 | Top-5 | 靶标级 Top-1 |
|---|---:|---:|---:|
| 官方 DreaMS | 75.0% | 94.4% | 77.8% |
| 经典高分辨峰匹配 | 86.1% | 100% | 94.4% |

DreaMS 的代表性错误包括：

- 1-methyladenosine 被 N6-methyladenosine 压过；
- guanosine 被 crotonoside 压过；
- guanine、asparagine、cystathionine、pantothenic acid、L-kynurenine 错排；
- 经典匹配也不是全对：DreaMS 在 carnosine、deoxyadenosine、N6-methyladenosine 和一张 kynurenine 谱上单独正确。

这组数据只作为最终外部门和错误机制说明，不能参与训练、阈值选择或 early stopping。

---

## 3. 四个固定任务面板

所有模型必须在同一权重、同一候选池下完成四个面板。任何模型只能在同时报告四个面板后讨论优劣。

### 面板 A：通用严格 10 ppm 检索

**问题**：总体谱库检索是否不退化？  
**数据**：MassSpecGym Murcko/连接分量互斥划分；候选限定同 adduct、前体 ±10 ppm。  
**主指标**：macro-AUC、Recall@1、MRR。  
**辅助指标**：positive max cosine、negative max cosine、margin、错误转换表。

### 面板 B：真实跨条件同分子

**问题**：仪器和碰撞能量变化后，同分子是否仍接近？  
**正例**：同 IK14、同 adduct、仪器不同或 `|ΔCE|≥10`。  
**负例**：同 adduct、质量近邻不同 IK14，并尽量匹配仪器/CE。  
**主指标**：triplet accuracy、正负 margin、同分子正对余弦。  
**分层报告**：跨仪器、同仪器高 ΔCE、峰数差、低质量谱、前体质量区间。

### 面板 C：共享主峰困难负例与同分异构体

**问题**：少数共享主峰是否仍会造成异分子误聚？  
**数据**：峰删除反事实已经确认的共享主峰困难负例；同分子式/同加合物异构体；质量近邻近结构和远结构分别报告。  
**主指标**：hard-negative cosine、pairwise ranking accuracy、flip-good/flip-bad。  
**禁止**：用规则 Jaccard 定义正负；用 Morgan Tanimoto 单独宣称 MCES 近异构体。

### 面板 D：极性代谢物定向挑战集

**问题**：与真实生物学应用相关的小分子是否改善？  
**类别**：核苷/修饰核苷、嘌呤、氨基酸/二肽、含硫代谢物、色氨酸—犬尿氨酸、泛酸/辅酶前体。  
**构造原则**：从训练库中挖掘同分子式、±10 ppm、相同 adduct 的结构竞争者；所有与 MTBLS1905 外部靶标相同的 IK14 从训练中排除。  
**主指标**：每类 Recall@1、MRR、macro-AUC、修正/新增错误数。  
**外部终测**：冻结后只运行一次 MTBLS1905 36 谱面板。

---

## 4. 数据划分与不可违反的防泄漏规则

### 4.1 当前已锁定的 G8R 数据

| 集合 | anchors | IK14 | 真实跨条件正例覆盖 | 与另一集合 IK14 overlap |
|---|---:|---:|---:|---:|
| train | 10,000 | 4,726 | 100% | 0 |
| inner validation | 2,000 | 957 | 100% | 0 |

每个 identity-adduct 仅保留一对真实跨条件谱图。锁定文件位于 `tasks/massspecgym_isomers/g8r_locked/`，哈希记录在 `audit.json`。

### 4.2 硬规则

1. 分组单位必须是 IK14，且跨所有 adduct；同一 IK14 不得跨 train/val/test。
2. 训练负例引用的谱图也必须留在训练分子集合；不能以 validation 分子作为显式负例。
3. 任何 `entries[:N]` 前缀采样一律禁止。
4. 每次运行必须记录数据文件 SHA256、有效 IK14、分子式数、跨条件正例覆盖率、hard-negative 覆盖率。
5. 历史 `eval` 已被 G5–G7 多次用于选择方向，只能作为开发证据，不再充当最终独立测试。
6. MTBLS1905 外部面板只能在模型、阈值和候选库全部冻结后运行一次。
7. 若将来使用 NIST20，必须确认与训练数据 IK14 互斥，并按原论文质量条件报告。

---

## 5. 微调目标函数：从 G8R 到正式模型

设 anchor 为 (a)，真实跨条件同分子谱为 (p)，质量近邻困难负例为 (n)，官方冻结模型为 (f_0)，待训练模型为 (f_\theta)，余弦为 (s_\theta(\cdot,\cdot))。

### 5.1 真实身份排序

\[
L_{\mathrm{rank}}=
\max\left(0, m-s_\theta(a,p)+s_\theta(a,n)\right)
\]

首轮 margin 使用 0.1，与 DreaMS 原论文一致；除非 validation 显示系统性欠分离，不先扫 margin。

### 5.2 真实正对一致性

\[
L_{\mathrm{real-pos}}=1-s_\theta(a,p)
\]

它保证真实谱图正例拥有直接梯度，避免多正例 InfoNCE 被容易的合成视图主导。

### 5.3 弱峰扰动一致性

\[
L_{\mathrm{aug}}=1-s_\theta(a,\tilde a)
\]

其中 \(\tilde a\) 只遮蔽弱峰/条件特异峰，保护相对基峰强度 ≥0.20 的主峰；合成正例权重必须低于真实正例。

### 5.4 正对关系保持

\[
L_{\mathrm{rel-preserve}}=
\max\left(0,s_0(a,p)-s_\theta(a,p)\right)
\]

该项直接针对 G5–G7 的失败机制：新模型不能把官方已经靠近的真实正对推散。

### 5.5 表征保持

\[
L_{\mathrm{feature-preserve}}=
1-\cos(f_\theta(a),f_0(a))
\]

该项防止整体 embedding 漂移，但不能代替正对关系保持。

### 5.6 共享主峰反事实困难负例

对已确认由共享主峰驱动的 \((a,n)\)，同时训练原谱和定向遮蔽视图的排序：

\[
L_{\mathrm{cf-hard}}=
L_{\mathrm{rank}}(a,p,n)+
L_{\mathrm{rank}}(\tilde a_{\mathrm{shared}},p,n)
\]

该视图只用于困难负例支路，不能混入一般同分子身份正例。

### 5.7 总损失

\[
L= L_{\mathrm{rank}}
+\lambda_rL_{\mathrm{real-pos}}
+\lambda_aL_{\mathrm{aug}}
+\lambda_cL_{\mathrm{cf-hard}}
+\lambda_{rp}L_{\mathrm{rel-preserve}}
+\lambda_pL_{\mathrm{feature-preserve}}
\]

参数调整顺序固定为：先锁定真实正例与负例 → 再确定 preservation → 最后消融 augmentation。严禁同时改变数据、解冻层数、噪声比例和多个 loss 权重。

---

## 6. 分阶段执行路线与指标门

## 阶段 M0：协议冻结与复现门

**目的**：确保所有人跑的是同一任务。  
**工作**：固定数据哈希、候选池、baseline 权重、指标实现和随机种子；重跑官方基线。  
**产物**：baseline JSON、逐 query 排名、候选池审计、环境记录。

**通过指标**：

- 官方模型的 macro-AUC、Recall@1 与已锁定结果差异 <0.001；
- 查询数和每 query 候选数完全一致；
- train/val IK14 overlap=0；
- 前体 ±10 ppm 和 adduct 筛选单元测试通过。

**停止条件**：任一指标对不上，禁止开始训练。

## 阶段 M1：G8R head-only 首门

> **2026-08-22 实测更新**：首个 M1 配置未通过。真实跨条件正例 cosine 增加0.0292，但 hard-negative cosine 增加0.0303；macro-AUC变化−0.00094、Recall@1不变、修正/新增错误为4/4、preservation=0.99465。该结果只否定当前配置。后续不得直接跳到loss替换、反事实负例或backbone解冻；先执行 `docs/G8R_M1_POSTMORTEM_AND_M1B_DECISION_20260822.md` 中的同anchor margin、监督覆盖、梯度分解和冻结可分性诊断。

**目的**：判断纠正后的目标能否在不动 Transformer 的情况下改善局部关系。  
**配置**：只训练官方 projection head，2 epoch，lr `1e-5`；真实跨条件谱为主正例；合成视图相对权重 0.10；保护主峰；关闭随机加峰与 m/z 位移；局部 hard negatives；加载官方教师。

**通过指标**：

- validation 跨条件正对余弦变化 ≥0；或 95% CI 下界不低于 −0.005；
- hard-negative cosine 下降，且配对 bootstrap 95% CI 上界 <0；
- strict-10ppm macro-AUC ≥ baseline−0.005；
- Recall@1 ≥ baseline−0.003；
- 新增错误数不多于修正错误数；
- embedding preservation cosine 均值 ≥0.995。

**决策**：

- 全部通过：进入 M2；
- 正例保护失败：增加关系保持或降低 hard-negative 权重，不增加 mask；
- hard negative 不改善：检查负例质量和峰反事实，不解冻 backbone；
- 总检索下降 >0.005：停止该配置。

**M1b追加纪律**：训练局部排序的anchor必须同时拥有真实跨条件正例和困难负例；无困难负例anchor不得接受无条件的绝对正例最大化。正式困难面板以同anchor的 \(s_{pos}-\max s_{neg}\) 为主指标，并按IK14/分子式簇bootstrap。

## 阶段 M2：噪声与真实条件变化消融

**目的**：证明收益来自哪一种增强，而不是“多加几个 loss”。  
**只允许四组**：

| 组 | 真实正例 | 弱峰遮蔽 | 强度扰动 | 共享主峰反事实 |
|---|---|---|---|---|
| A | ✓ | 关闭 | 关闭 | 关闭 |
| B | ✓ | ✓ | 关闭 | 关闭 |
| C | ✓ | ✓ | 实测范围 | 关闭 |
| D | ✓ | ✓ | 实测范围 | ✓ |

不允许加入随机 m/z 峰或单峰独立 m/z 位移。

**通过指标**：

- 三个 seed 的指标方向一致；
- 相对 A，B/C/D 至少有一组在 hard panel 或跨条件 panel 改善，且总检索不降；
- 每组报告修正/新增错误及类别分布；
- 若增强只提高 synthetic consistency、没有提高真实跨条件正对或检索，判定无效。

## 阶段 M3：困难负例课程学习

**目的**：把训练集中在 DreaMS 真正容易错的局部候选，而非全局随机负例。  
**负例优先级**：

1. 经峰删除反事实确认的共享主峰异分子；
2. 同分子式、同 adduct 的 connectivity 异构体；
3. 严格 10 ppm 同 adduct 的近质量负例；
4. 0.05 Da 内负例仅用于对齐 DreaMS 原始训练，不作为最终检索唯一负例；
5. 与 query 质量相差很大的 batch 随机分子不进入主要 loss。

**课程**：先真实 identity + 中等 hard negatives；再逐步加入最难反事实负例。每个 epoch 固定困难等级占比并记录。

**通过指标**：

- hard panel pairwise accuracy 至少提高 2 个百分点；
- 总 macro-AUC 不下降超过 0.005；
- corrected errors > introduced errors；
- 改善不应仅集中在一个分子式家族，前 10 个分子式贡献 <30%。

## 阶段 M4：参数高效解冻

只有 M1–M3 在 head-only 下形成稳定方向，才能改变 backbone。

**升级顺序**：

1. head only；
2. head + 最后一层 adapter/LoRA；
3. head + 最后一层 Transformer；
4. 最后两层；
5. 全模型微调仅在充足算力和更大训练集下考虑。

**首选**：adapter/LoRA 或峰 token 门控，因为它们比直接解冻两层更容易保留官方空间。  
**禁止**：head 失败后直接解冻更多层；这通常放大错误梯度。

**通过指标**：相对 head-only，在至少两个困难面板显著提高；总体检索和外部安全集不降；模型参数变化和训练成本可报告。

## 阶段 M5：极性代谢物定向微调

**目的**：解决核苷/嘌呤等实际注释问题，同时避免把 36 张外部谱训练进去。  
**训练数据**：从 MoNA/MassBank/NIST 可用训练部分挖掘同类别分子；排除外部靶标 IK14；按结构组、分子式、质量、adduct 分层。

**需要建立的挑战组**：

- 位置/甲基化异构核苷；
- 核糖/脱氧核糖差异；
- 核苷与非核苷同质量竞争者；
- 氨基酸与二肽近质量竞争者；
- 含硫中性丢失相近但骨架不同者；
- kynurenine 类与芳香含氮竞争者；
- pantothenate/辅酶前体类。

**训练方式**：类别不作为模型标签；它只用于分层采样和评价。主要标签仍为 IK14 身份与局部结构负例。

**通过指标**：

- 极性挑战集 macro Recall@1 相对官方提高 ≥2 pp；
- 至少 4/6 类方向改善，不允许靠单一核苷类拉高平均；
- 总体 strict-10ppm macro-AUC 不低于 baseline−0.005；
- 每类新增错误数和修正错误数公开。

## 阶段 M6：结构局部排序的可选支路

如果身份检索稳定后仍需要改善近结构排序，可加入 MCES 相对排序，但它是独立增量，不与 M1 同时上线。

\[
d_{\mathrm{MCES}}(a,p)+\delta < d_{\mathrm{MCES}}(a,n)
\Rightarrow s(a,p)>s(a,n)
\]

只学习相对顺序，不回归绝对 MCES；Morgan 仅作预筛，不替代 MCES 真值。  
**通过指标**：MCES@1/Rank correlation 改善且身份检索不降。若只改善 Pearson/Spearman、却降低 Recall@1，则不合并。

## 阶段 M7：冻结后的外部盲评

**数据**：MTBLS1905 36 张谱图/18 靶标；后续至少增加一个不同实验室外部库。  
**比较**：官方 DreaMS、新模型、modified cosine、spectral entropy、经典高分辨峰匹配；若进入候选分子检索，再比较 MIST/JESTR/MVP/FLARE 的同协议结果。

**外部门**：

- 在 MTBLS1905 中至少修正 2 张官方错误谱，新增错误 ≤1；
- 23 张 DreaMS/经典一致且正确的谱不能明显退化；
- 逐谱报告真实排名，不只报告平均数；
- 该面板运行前模型和阈值必须冻结，并保留时间戳/哈希。

小面板只能证明外部方向，不能单独证明通用 SOTA。

## 阶段 M8：正式测试与论文结果

正式结果至少包含：

- 三 seed 均值与 95% CI；
- 标准 benchmark 的同协议比较；
- 训练分子和测试分子严格互斥；
- 总体、跨条件、共享峰、极性代谢物四面板；
- 逐 query paired bootstrap 或 permutation test；
- corrected/introduced/persistent error atlas；
- 训练时间、参数量和推理成本；
- 完整消融与失败配置。

---

## 7. 指标口径与优先级

### 一级指标：决定模型是否可用

1. **macro-AUC**：每个 query 内区分同分子正例和质量近邻负例的 AUC，再对 query 等权平均；防止重复谱多的分子主导结果。
2. **Recall@1**：第一名是否为同 IK14；最符合真实检索体验。
3. **MRR**：正确分子的倒数排名，反映 Top-1 之外的整体排序。
4. **正对余弦与负对余弦**：必须分别报告，不能只看 separation。
5. **错误转换**：baseline wrong→model right、baseline right→model wrong、persistent wrong。

### 二级指标：说明改善发生在哪里

- 跨仪器/跨 CE triplet accuracy；
- hard-negative pairwise accuracy；
- 每分子式、结构类别、峰数、谱质量的 Recall@1；
- Top-5、Top-10；
- selective accuracy–coverage：仅输出高置信结果时的准确率和覆盖率。

### 三级指标：只能辅助解释

- Pearson/Spearman 与 Tanimoto/MCES 的相关性；
- UMAP/PCA 图；
- synthetic noise consistency；
- 训练 loss。

这些指标不能覆盖一级检索退化。

---

## 8. 统计与判定规则

1. 所有新旧模型必须对同一 query、同一候选池进行配对比较。
2. macro-AUC、Recall@1、margin 使用 query-level bootstrap，至少 2,000 次。
3. 小面板同时报告分子数和谱图数，避免把重复谱当独立分子扩大显著性。
4. 多类别分析报告置信区间和样本量；低于 30 个分子的类别只作探索。
5. 每个配置至少 3 个 seed；首门可单 seed，但不能进入论文结论。
6. 参数选择只看 inner validation；正式 test 和外部数据在模型冻结后运行。
7. 非劣效界预注册为 macro-AUC −0.005；不得在看到结果后放宽。

---

## 9. 团队运行纪律：千叮咛万嘱咐

### 每次开跑前必须打印

- git commit/工作树状态；
- 完整命令行和配置 JSON；
- checkpoint 起点与 SHA256；
- train/val 数据 SHA256；
- anchors、IK14、分子式、adduct、跨条件正例率；
- 每 batch 真实正例数、hard-negative 数；
- 实际可训练参数量；
- 输出目录，必须包含 seed 和配置短名。

### 严禁事项

- 严禁数组任务写同一输出目录；
- 严禁注释里的 λ 与运行参数不一致；
- 严禁把最后 epoch 命名为 best；best 必须由 validation gate 选择；
- 严禁只保存平均指标而不保存逐 query 结果；
- 严禁用测试集错误反向修改训练数据后继续把它称为独立测试；
- 严禁报告虚构、估计或未跑完的数；
- 严禁把 100/500 个样本的 CPU pilot 当正式证据；
- 严禁以“loss 下降、embedding preservation 很高”替代检索评价；
- 严禁为了挽救失败配置同时改五个参数；
- 严禁把规则相似当结构标签。

### 每次跑完必须回答五个问题

1. 总体指标是否过安全门？
2. 同分子正对发生了什么变化？
3. 困难负例发生了什么变化？
4. 修正了哪些具体分子，新增了哪些错误？
5. 改善是否与预设机制一致，还是由单一分子家族/数据泄漏造成？

无法回答这五个问题的实验不进入下一阶段。

---

## 10. 三个紧急迭代周期

### 周期 1：先获得可用模型方向

- 完成 M0；
- 运行 G8R head-only 首门；
- 运行独立三面板评估；
- 输出逐 query 错误转换；
- 只做一次修正迭代。

**交付**：一个通过/失败明确的 head checkpoint；不是排行榜。

### 周期 2：定位噪声收益并建立极性挑战集

- 完成 A/B/C/D 四组消融；
- 构建外部靶标互斥的核苷/嘌呤等训练挑战组；
- 形成按类别错误表；
- 三 seed 复核最佳配置。

**交付**：可证明“哪种噪声针对哪类错误有效”的结果。

### 周期 3：参数高效升级与外部冻结评估

- 仅对最佳 head 配置加 adapter/LoRA；
- 冻结模型和阈值；
- 运行标准 test 和 MTBLS1905；
- 与官方 DreaMS 和经典方法做配对统计；
- 决定论文主张是“鲁棒谱图表征”“极性代谢物检索改进”还是“保守共识注释”。

**交付**：论文级结果表、消融表、错误案例图和外部生物学接口。

---

## 11. 如何判断具备 SOTA 资格

### 可以争取的近期主张

> Error-guided, condition-robust fine-tuning improves DreaMS spectral-library retrieval for mass-near isomers and polar metabolites while preserving general retrieval performance.

其中文可表述为：

> 基于真实错误机制的跨条件与峰级反事实微调，在保持 DreaMS 通用检索能力的同时，提高质量近邻异构体及极性代谢物的谱库检索准确度。

### 不能直接主张的内容

- “全面超过 FLARE/MVP/MIST 的候选分子检索”；它们使用谱图—分子跨模态训练，当前模型只产生谱图 embedding；
- “提高所有未知代谢物的结构鉴定率”；没有结构真值和 FDR 控制不能这样写；
- “外部 36 谱达到 SOTA”；样本过小且已经被用于机制审计。

### 若要争取 MassSpecGym molecule retrieval SOTA

必须增加独立的 molecule encoder 或 candidate reranker，使输入成为“谱图 + 候选分子”，并在官方 MassSpecGym mass/formula candidate sets 上比较 Hit@1/5/20、MRR 和 MCES@1。该工作属于后续跨模态方向，不能与当前噪声微调同时展开，以免主线失焦。

---

## 12. 当前文件与立即执行入口

- 锁定训练/验证：`tasks/massspecgym_isomers/g8r_locked/`
- G8R 训练：`tasks/run_g8r_real_condition_head_gate.sbatch`
- 独立门评估：`tasks/eval_g8r_inner_gate.py`
- 真实噪声审计：`tasks/audit_empirical_cross_condition_noise.py`
- 全量错误转换提取：`tasks/extract_retrieval_errors.py`
- 红队审计：`docs/NOISE_FINETUNE_RED_TEAM_DECISION_20260821.md`

当前唯一允许启动的正式实验：

```bash
sbatch tasks/run_g8r_real_condition_head_gate.sbatch
```

在 `g8r_inner_gate.json` 出现并通过三门之前，不启动 G8R array、不扫 λ、不解冻 Transformer。

---

## 13. 方法依据

1. DreaMS：Bushuiev et al., *Nature Biotechnology* (2025), “Self-supervised learning of molecular representations from millions of tandem mass spectra using DreaMS”. https://www.nature.com/articles/s41587-025-02663-3
2. MassSpecGym：Bushuiev et al., NeurIPS 2024 Datasets and Benchmarks. https://papers.nips.cc/paper_files/paper/2024/hash/c6c31413d5c53b7d1c343c1498734b0f-Abstract-Datasets_and_Benchmarks_Track.html
3. FLARE：Fine-grained spectra–molecule alignment，在 MassSpecGym mass/formula candidate retrieval 上报告当前领先表现。https://pmc.ncbi.nlm.nih.gov/articles/PMC12873900/
4. MVP：多视图谱图—分子对比学习，并报告 consensus spectra 对候选排序的价值。https://pmc.ncbi.nlm.nih.gov/articles/PMC12642559/
5. DDA-BERT（相邻领域证据）：使用 spectrum dropout 增强跨数据集鲁棒性，但其任务为蛋白质组 peptide-spectrum matching，不能直接照搬扰动比例。https://doi.org/10.1038/s41467-026-72246-6
6. FDR-controlled spectral matching：Scheubert et al., *Nature Communications* (2017). https://www.nature.com/articles/s41467-017-01318-5

---

## 14. 最终总原则

> 先保护 DreaMS 已经学会的东西，再只修正被真实反事实证明的局部错误；先通过未污染验证门，再谈扩大训练；先报告修正和新增错误，再谈平均指标；先在同一任务协议下超过强基线，再谈 SOTA。
