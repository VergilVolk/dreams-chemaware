# Full23 v1.3.2：同轴 SOTA 对照与投稿定位

日期：2026-11-02  
算法：Full23 v1.3.2  
主结果：`data/validation/full23_v132_residual_scale_20261102`  
16 方法复算表：`deliverables/GLM_gnps_article_ladder/run16_full23_v132`

## 1. 一句话裁决

**Full23 v1.3.2 是当前 GNPS Gold/Silver 10-ppm `[M+H]+` 已消费开发基准上，15 个已实际运行冻结比较方法中的综合指标领先者；这些比较方法包括公开/经典方法与本项目的冻结基线。它尚不是经过未开封外部数据确认的全领域 SOTA。**

这个限定不是文字游戏：

- Full23 v1.3.2 在两个同轴面板的 Recall@1、MRR、near Recall@1、macro-query AUROC、micro-candidate AUROC 和 micro-candidate AUPRC 均为第一；
- 相对最强传统直接基线 WSE 的 Recall@1 增益为 `+1.0005/+1.3305 pp`，公式簇配对区间均严格为正；
- 但 GNPS 与 Enveda 均已消费，且 `0.9` 残差幅度是在查看 GNPS 后确定；
- GNPS 内拟合的 nested-OOF B 在该域的 Recall@1 仍略高，但它不是同等级的跨来源模型；
- MSBERT 尚未在同一候选轴运行；MS2Query 需要排除 GNPS 训练污染并忠实保留其原生邻域特征后才是公平比较；
- LSM-MS2 的最新谱库检索结果来自不同参考库、不同覆盖上限，且没有可独立同轴运行的公开 checkpoint。

因此，当前最准确的英文定位是：

> **benchmark-leading candidate-level evidence integration for closed-set reference-spectrum identity ranking**

而不是笼统的：

> state-of-the-art metabolite identification

## 2. 最终算法是什么

Full23 v1.3.2 不是简单投票。它由一个候选级标量证据父模型和一个完整表示残差读出组成：

\[
s_{\mathrm{v1.3.2}}
=s_{\mathrm{Full23\ v1}}
+0.9\left(s_{\mathrm{v1.3.1}}-s_{\mathrm{Full23\ v1}}\right).
\]

- **Full23 v1**：23 维候选级 Additive 父模型；
- **表示证据**：Noise V1、ChemAware Stage-1、ChemAware Phase-A、Noise–ChemAware FullMSG；
- **局部谱学证据**：WSE、P2b-on-Noise、Noise-MSG；
- **候选化学证据**：ChemAware V2 的连续 real-vs-null 证据；
- **v1.3.1 残差**：读取四套完整 encoder tensor-set 表示；
- **v1.3.2**：不重训网络，仅将已拟合残差幅度固定为 `0.9`；
- **BioAware**：不在该闭集谱库排序器内；只有真实样本事件上下文存在时才作为后置条件模块。

可执行发布包为 `deploy/unified_additive_full23_v132`。

## 3. 数据与证据等级

| 数据 | 分母 | 用途 | 当前证据资格 |
|---|---:|---|---|
| Enveda-180 identity | 25,000 queries | Full23 元开发与拟合 | consumed meta-development |
| Enveda-180 formula | 9,435 queries | Full23 元开发与拟合 | consumed meta-development |
| GNPS identity-disjoint | 10,995 queries；87,518 candidate molecules | 开发、架构选择、0.9 幅度校准 | consumed development |
| GNPS formula-disjoint | 5,261 queries；24,401 candidate molecules | 开发、架构选择、0.9 幅度校准 | consumed development |
| GNPS query-balanced union | 11,019 unique queries | 联合主效用；重复 query 总权重为 1 | consumed development |

两个 GNPS 面板存在 query 重叠，不能把 `10,995 + 5,261` 写成 16,256 个独立 query。联合端点必须使用 11,019 个 unique query 的 query-balanced 权重。

## 4. 同轴主结果

所有方法使用完全相同的 query、10-ppm candidate molecule 轴和并列规则。下面列出每个面板按 Recall@1 排名最靠前的方法。

### 4.1 Identity-disjoint，10,995 queries

| 方法 | R@1 | R@5 | MRR | near R@1 | macro AUROC | micro AUROC | micro AUPRC |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Full23 v1.3.2** | **88.367%** | **97.508%** | **92.452%** | **83.825%** | **94.925%** | **95.854%** | **84.414%** |
| weighted spectral entropy | 87.367% | 97.008% | 91.660% | 82.508% | 94.328% | 94.202% | 75.627% |
| entropy raw | 87.312% | 97.481% | 91.795% | 82.442% | 94.635% | 94.154% | 71.923% |
| P2b-on-Noise | 87.131% | 97.435% | 91.646% | 82.126% | 94.438% | 94.296% | 75.069% |
| Noise V1 | 86.558% | 97.035% | 91.274% | 81.441% | 94.129% | 94.358% | 75.230% |
| official fine-tuned DreaMS | 85.357% | 97.017% | 90.516% | 79.939% | 93.733% | 93.193% | 68.149% |
| Spec2Vec public | 86.294% | 96.826% | 91.020% | 81.138% | 93.834% | 93.673% | 73.004% |
| MS2DeepScore 2.x public | 79.118% | 93.752% | 86.365% | 72.405% | 90.731% | 90.427% | 68.902% |

### 4.2 Formula-disjoint，5,261 queries

| 方法 | R@1 | R@5 | MRR | near R@1 | macro AUROC | micro AUROC | micro AUPRC |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Full23 v1.3.2** | **89.603%** | 98.974% | **93.740%** | **82.303%** | **94.653%** | **95.384%** | **87.415%** |
| weighted spectral entropy | 88.272% | 98.688% | 92.796% | 80.095% | 93.896% | 92.837% | 78.086% |
| P2b-on-Noise | 88.253% | **99.012%** | 92.900% | 80.061% | 94.007% | 92.617% | 76.766% |
| entropy raw | 88.139% | 98.898% | 92.822% | 79.925% | 94.114% | 91.994% | 71.949% |
| Noise V1 | 88.063% | 98.764% | 92.833% | 79.959% | 93.805% | 93.145% | 76.925% |
| Spec2Vec public | 87.436% | 98.498% | 92.367% | 78.702% | 93.433% | 91.996% | 75.680% |
| official fine-tuned DreaMS | 86.809% | 98.118% | 92.024% | 77.989% | 93.186% | 90.722% | 68.012% |
| MS2DeepScore 2.x public | 81.467% | 95.571% | 88.737% | 70.618% | 90.062% | 88.616% | 72.065% |

### 4.3 不能省略的例外

“主要指标领先”不能写成“所有指标第一”。

- identity R@10：Full23 v1.3.2 与 entropy raw 同为 `98.881%`；
- identity R@20：Noise V1 为 `99.827%`，高于 Full23 的 `99.754%`；
- formula R@5：P2b-on-Noise 为 `99.012%`，高于 Full23 的 `98.974%`；
- formula R@10/R@20：Noise V1 为 `99.715%/100.000%`，高于 Full23 的 `99.696%/99.962%`；
- Full23 输出的是多参考谱聚合后的 candidate-molecule score，因此旧天梯的 spectrum-pair pooled AUROC/AUPRC 不能从当前发布账本合法重建，表中保持 NA。

宏平均 AP 在每个 query 恰有一个阳性且使用严格并列规则时等于 reciprocal rank，因此 macro AUPRC 与 MRR 不是两个独立胜利。

## 5. 最关键的配对比较

### 5.1 相对 WSE

| 面板 | R@1 增量 | corrected / introduced | 净新增正确 | formula-cluster 95% CI | MRR 增量 |
|---|---:|---:|---:|---:|---:|
| identity，n=10,995 | **+1.0005 pp** | 223 / 113 | +110 | **[+0.6306, +1.3677] pp** | +0.7919 pp |
| formula，n=5,261 | **+1.3305 pp** | 116 / 46 | +70 | **[+0.8024, +1.8581] pp** | +0.9439 pp |

这里的 `corrected−2×introduced` 只保留为历史描述量，不是模型去留门槛。主结论来自同一 query 上的净正确数、配对效应与公式簇区间。

### 5.2 相对 official fine-tuned DreaMS

| 面板 | R@1 增量 | corrected / introduced | 净新增正确 | formula-cluster 95% CI |
|---|---:|---:|---:|---:|
| identity，n=10,995 | **+3.0105 pp** | 533 / 202 | +331 | **[+2.4568, +3.5529] pp** |
| formula，n=5,261 | **+2.7941 pp** | 252 / 105 | +147 | **[+1.9570, +3.6413] pp** |

“约 +3 pp”只能表述为相对 official fine-tuned DreaMS；相对更强的 WSE 是约 `+1.00/+1.33 pp`。

## 6. 内部 B：必须单列，不能隐去

| 方法 | Identity R@1 | Formula R@1 | 训练/评估状态 |
|---|---:|---:|---|
| B | **88.595%** | **89.850%** | GNPS nested formula-OOF；直接使用 GNPS 标签拟合 |
| Full23 v1.3.2 | 88.367% | 89.603% | Enveda 拟合；GNPS 开发选型与 0.9 幅度校准 |

B 的点估计高 `0.227/0.247 pp`，约对应 25/13 个 query。因此不能写“Full23 超过了仓库里所有模型”。

另一方面，B 是域内交叉拟合开发模型，既有结果显示其跨 Enveda 迁移不如 WSE；Full23 是当前完整、可部署、跨来源证据系统。正确表述是：

> B is the highest GNPS-specific nested-OOF development reference, whereas Full23 v1.3.2 is the strongest released multi-source integrator evaluated against the frozen 15-method same-axis ladder.

## 7. 与当前业界方法的严格比较

| 方法 | 原始任务 | 是否已在同轴运行 | 投稿裁决 |
|---|---|---|---|
| [DreaMS](https://www.nature.com/articles/s41587-025-02663-3) | 10-ppm 内参考谱身份检索及多种下游任务 | **是** | 最直接的已发表 learned baseline；Full23 为 `+3.01/+2.79 pp` |
| [weighted spectral entropy](https://www.nature.com/articles/s41592-021-01331-z) | 谱库匹配、ROC/FDR | **是** | 最强传统直接基线；Full23 为 `+1.00/+1.33 pp` |
| [Spec2Vec](https://pmc.ncbi.nlm.nih.gov/articles/PMC7909622/) | learned spectral similarity / library matching | **是** | 原模型使用 GNPS 生态训练，必须披露潜在重叠；同轴结果显著低于 Full23 |
| [MS2DeepScore 2.0](https://www.nature.com/articles/s41467-026-69083-y) | 预测谱对所对应分子的结构相似度，含跨离子模式 | **公共 2.x 模型已运行** | 当前本地行为 package 为 2.11.0；投稿前应锁定并另报论文版 2.5.3 checkpoint，训练重叠需披露 |
| [MSBERT](https://pubs.acs.org/doi/10.1021/acs.analchem.4c02426) | GNPS/MoNA 谱库匹配和类似物检索 | **尚未** | 最重要的缺失同轴 embedding baseline；官方代码与数据公开，但 GNPS 训练重叠必须先审计 |
| [MS2Query](https://www.nature.com/articles/s41467-023-37446-4) | MS2DeepScore top-2000 召回后，以多证据 RF 排 exact/analogue matches | **尚未** | 与 Full23 方法定位最接近；必须保留其邻域特征并排除 GNPS panel 身份后重训，简化版不能冒充原方法 |
| [LSM-MS2](https://arxiv.org/html/2510.26715) | 1.8M 参考谱库上的谱身份检索和生物表征 | **不能独立同轴复现** | 目前最直接的最新 SOTA 宣称；MassSpecGym per-spectrum Top-1 为 0.739、DreaMS 0.726，但库覆盖上限仅 0.785，协议与本项目保证阳性的 10-ppm 候选轴不可横比 |
| JESTR / MIST / ESP / MVP / FLARE | spectrum-to-structure candidate ranking | **不同任务** | 只能放 related work 或独立结构检索表，不能把其 R@1 混入参考谱身份排名主表 |

MS2Query 的原生系统融合 MS2DeepScore、Spec2Vec、质量差与库邻域结构信息，本质上已经是学习式多证据重排器；因此“融合多个证据”本身不是 Full23 的新颖性。Full23 的可发表贡献必须落在：

1. 专门训练的 Noise/ChemAware 表示和条件化学证据是否带来不可替代的同轴增量；
2. 多来源训练后是否仍能迁移；
3. 新增身份连接是否带来独立支持的生物学响应恢复。

## 8. 当前可以写与不能写的句子

### 可以直接写

> We developed Full23, a candidate-level multi-evidence ranker for closed-set reference-spectrum identity search. On two previously consumed GNPS development panels comprising 10,995 identity-disjoint and 5,261 formula-disjoint queries, Full23 v1.3.2 achieved 88.37% and 89.60% Recall@1, exceeding weighted spectral entropy by 1.00 and 1.33 percentage points and the official fine-tuned DreaMS baseline by 3.01 and 2.79 points under an identical 10-ppm candidate protocol.

紧接着必须写：

> Because GNPS was used for architecture selection and residual-amplitude calibration, these values are post-selection development estimates rather than external confirmation.

可以写：

> Full23 v1.3.2 ranked first among the 15 comparator methods in the frozen same-axis GNPS development ladder for Recall@1, MRR, near-neighbor Recall@1, macro-query AUROC and micro-candidate AUROC/AUPRC.

### 当前不能写

- state-of-the-art metabolite identification；
- externally validated SOTA；
- MassSpecGym SOTA；
- structure elucidation SOTA；
- open-set annotation SOTA；
- independently validated on GNPS or Enveda；
- outperformed every model tested on GNPS；
- all metrics were best；
- all modules independently improved every data source；
- BioAware is part of the Full23 v1.3.2 closed-set ranker。

## 9. 投稿前的最小补齐清单

### P0：真正未开封外部源

最快的高质量选择是 NIST26 相对 NIST23 的新增记录。NIST 官方报告 NIST26 含 68,635 compounds、3,219,115 tandem spectra，化合物数比 NIST23 增加 33%。建议：

- 仅使用 NIST26 相对 NIST23 的新增身份；
- 与 MassSpecGym、GNPS、Enveda、MassBank、MoNA、MSnLib 及全部监督训练注册表做 IK14 和 formula 排除；
- 主分析先固定 `[M+H]+`、±10 ppm；其他极性/加合物作 OOD；
- query/reference 来自不同碰撞能，最好来自不同仪器；
- 每个真身份至少有两张独立谱；
- 预先固定候选聚合、并列规则、主要终点和所有模型后，一次性打开；
- identity-disjoint 目标至少 5,000 unique query identities，formula-disjoint 至少 2,000 formula clusters，near 至少 1,000 queries。

如果没有 NIST26 授权，则使用冻结日之后的新 MassBank 正式发布，或前瞻性盲法多碰撞能标准品谱。不能把 Enveda/GNPS 的新切片包装成独立外测。

### P1：补齐两个真正影响审稿的比较

1. **MSBERT**：在完全相同的候选轴直接运行；先报告训练身份/谱图重叠；
2. **MS2Query**：优先报告作者原生流程；只有完成无 panel 身份污染的忠实重训，才进入同轴主表。

现有 MS2DeepScore 行必须在稿件中写明具体 checkpoint MD5、package 版本与训练重叠边界；若以 2026 论文作引文，另补论文版 2.5.3 checkpoint 是最稳妥的做法。

### P2：实际可用性

- 固定一个独立 calibration partition，报告 1%/5% false-match rate 下 coverage；
- 区分输入缺失导致的 abstention 与真身份不在库导致的 no-match；
- 报告冷启动时间、缓存后速度、CPU/GPU 内存，以及参考谱数量扩展曲线；
- 在同一应用对象上比较 WSE、official DreaMS、Full23，并说明哪些实体判断改变、哪些获得独立支持。

## 10. 推荐论文表格

- **Table 1**：数据来源、开发/外测角色、queries、unique identities、formula clusters、candidate-count 分布、每候选参考谱数、V2 覆盖率；
- **Table 2**：未开封外部闭集身份排名主结果；
- **Table 3**：同一外部集的 corrected/introduced、near、仪器/碰撞能分层；
- **Table 4**：open-set coverage–false-match rate；
- **Supplementary Table 1**：本文件中的 GNPS 16 方法已消费开发天梯；
- **Supplementary Table 2**：Full23 strict-refit 模块消融；
- **Supplementary Table 3**：每个竞争者的版本、checkpoint、训练数据重叠、预处理与缺失率；
- **Supplementary Table 4**：内部 B 与其他历史开发模型，明确不同训练协议。

## 11. 最终判断

Full23 v1.3.2 已经达到“值得投稿的方法核心”水平：它在一个规模足够大的统一闭集身份排名基准上，不仅比 official DreaMS 高约 3 pp，也显著超过更强的 WSE，并同时改善困难 near 子集和候选判别。

现在不应继续在 GNPS 上追零点几。离 SOTA 主张最近的路径是：

1. 固定当前发布包；
2. 补 MSBERT 和忠实 MS2Query；
3. 在真正未消费的新来源一次性确认；
4. 把新增正确连接转化成一个有独立证据的生物学应用实例。

完成这四项后，稿件可以使用更强而仍准确的主张：

> Among reproducible methods evaluated on an identical post-cutoff 10-ppm candidate axis, frozen Full23 v1.3.2 achieved the highest closed-set reference-spectrum identity Recall@1.
