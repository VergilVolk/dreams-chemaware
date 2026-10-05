# 从前沿方法到 DreaMS 更强 embedding：文献对账与决策报告

日期：2026-09-14  
用途：决定下一轮 shared clean-spectrum embedding 工作，不授权直接开大训练  
核心边界：P2b、候选重排器、BioAware 后验专家与 shared embedding 训练严格分开

## 0. 执行结论

目前最值得继续的不是再发明一种孤立的删峰规则，也不是把多个分数再做一次线性融合，而是把已有约 3–5 pp 的**显式多通道/动作空间信号**升级为一个有文献依据的训练范式：

> **以真实重复谱和共识谱为条件视图，以分子结构相似度和同分子式近异构体为排序监督，以峰—子结构局部对齐为化学监督，以经验噪声为一致性增强，训练一个候选无关的 shared spectrum encoder。**

这条路线可以简称为 **condition-factorized multiview DreaMS（CF-MV-DreaMS）**。它不是放弃噪声微调，而是纠正噪声微调过去的监督表达：噪声不再被当成“应该追随的目标方向”，而被当成同一化学身份的一个受控视图；真正负责区分近异构体的信号来自结构连续监督、候选组内排序和峰—子结构局部对齐。

下一轮不应承诺必然获得 5 pp。能够严谨承诺的是：先用冻结表示证明这种多视图、局部交互的教师在相同候选协议上确实保留已有 3–5 pp 信号；只有教师过门，才训练 shared encoder；只有学生能转移至少一半教师增益，才解冻最后一层并扩大。这样可以避免再次把 oracle/action headroom 误写成可学习性能。

## 1. 先把“5 pp”说清楚

仓库中存在三种性质完全不同的“约 3–5 pp”，不能再混在一起。

| 信号 | 真实结果 | 它证明什么 | 它没有证明什么 |
|---|---:|---|---|
| A1 显式多通道关系坐标 | 相对强 `direct_multichannel`，NDCG@5 `+2.962 pp`，IK14 CI `[+1.317,+4.767] pp`；neg_rp `+4.824 pp`，pos_rp `+1.060 pp` | fragment、neutral-loss、coverage 等通道联合响应含关系型结构信息 | 不是 Recall@1；不是官方 DreaMS embedding 提升 |
| A1b 官方 DreaMS 单 cosine 关系坐标 | NDCG@5 `-0.590 pp`，CI `[-1.941,+0.822] pp`；两 panel 均下降 | 单一全局 cosine 没保留足够的局部多通道关系 | 不否定多视图或局部交互表示 |
| Noise-v3 动作/oracle 容量 | 历史上约 `4.93 pp` 的 outcome-aware action/no-op 容量 | 峰动作空间里存在可修正错误 | 不是 shared encoder 能达到的性能 |
| 成熟 E4 shared embedding | fold-0 相对 matching official DreaMS `+0.574 pp`，38 corrected / 4 introduced | 已经证明少量 clean embedding 改进可实现 | 尚未达到 3–5 pp，也不是多 fold/P3 SOTA |

共同模式非常清楚：**大信号出现在显式多通道仍被保留，或结果标签可以替每个 query 选动作时；一旦压缩成一个全局 cosine 或要求一个 shared student 无条件执行，增益大幅衰减。** 这更像表示与目标错配，而不是“没有有效信息”。

因此，A1 的约 5 pp panel 信号应被视为**表示充分性的证据和教师设计依据**，不是直接复用原公式就能得到 5 pp 的模型承诺。

## 2. 前沿文献给出的共同答案

### 2.1 DreaMS 本身已经告诉我们：质量、hard examples 和局部差异最重要

DreaMS 使用七层 Transformer、峰级 1024 维 contextual token、30% 强度加权 m/z mask 和保留时间顺序自监督；官方微调使用同结构正例与相近质量负例的 triplet 来增强微小结构差异。论文同时明确指出，zero-shot cosine 对小结构差异敏感度不足，而且低质量 GeMS-B 数据反而使性能下降，数据质量比盲目扩量更重要。[DreaMS](https://www.nature.com/articles/s41587-025-02663-3)

对本项目的直接含义：继续扩大无差别随机峰噪声，与 DreaMS 自己的成功经验相反；应扩大的是**高质量、分子均衡、条件覆盖明确的训练关系**。

### 2.2 MVP：不同条件谱不应被逐一拉向一个手工动作，而应共同对齐到共识视图

2026 年的 MVP 同时学习 molecular graph、fingerprint、individual spectrum 和 consensus spectrum 四个视图，对所有视图对做 contrastive alignment。其共识谱通过合并同分子不同谱的峰构建，实验证明 aggregate-then-rank 优于 rank-then-aggregate；MassSpecGym 上 individual spectrum 的 mass/formula Rank@1 为 26.4%/11.1%，consensus spectrum 为 36.0%/14.0%。[MVP](https://pmc.ncbi.nlm.nih.gov/articles/PMC12980492/)

这是与本项目最直接的正面先例：我们真实重复谱、跨仪器/CE 和经验 missingness 不应只产生删峰动作，而应形成**individual—consensus—structure 多视图训练图**。但 MVP 使用已知母分子式给峰分配子式，且主要是 spectrum-to-molecule retrieval；我们必须验证在候选无关的 spectrum embedding 上是否仍可转移。

### 2.3 FLARE：单一 pooled embedding 会丢掉峰—子结构细节

FLARE 不先把谱和结构各压成一个向量，而是在 contextual peak token 与 molecular node token 之间做双向 MaxSim late interaction；其核心论断正是谱—结构兼容性来自多个局部一致证据，而非单个 pooled global vector。在 MassSpecGym 预印本中报告 mass/formula Rank@1 43.15%/22.66%。[FLARE](https://pubmed.ncbi.nlm.nih.gov/41659479/)

这解释了 A1→A1b 的断裂：raw multichannel 中互补的局部证据被官方 precursor embedding 的单 cosine 压缩。FLARE 适合作为**训练期局部教师**，但它是候选结构感知方法、目前为预印本，不能直接冒充我们的 spectrum-to-spectrum shared embedding。

### 2.4 MS2DeepScore 2.0：最关键的可能不是新损失，而是分子均衡和相似度分层采样

MS2DeepScore 2.0 的采样使不同分子的出现频率差异低于 15%，并让十个 Tanimoto 区间等频；模型还训练 per-spectrum embedding evaluator 预测该谱的相似度误差。其消融显示 precursor m/z、ion mode、adduct 有帮助，而 instrument type one-hot 没有稳定帮助。[MS2DeepScore 2.0](https://www.nature.com/articles/s41467-026-69083-y)

这对我们极其重要：过去按 action row 或高重复分子曝光，容易让少数身份和动作家族主导梯度。新训练单位必须是 molecule/formula equal-weight，且覆盖从 same identity、near、mid 到 far 的连续结构区间；条件元数据不能未经消融就全部塞入。

### 2.5 MIST、FIORA、ModiFinder：化学规则的正确位置是局部表示和弱监督，不是一个静态总分

- MIST 把峰表示为化学子式 token，并使用峰/碎片级辅助目标；这是“化学规则进入表示”的成熟范式。[MIST](https://www.nature.com/articles/s42256-023-00708-3)
- FIORA 围绕局部键断裂及其邻域预测碎片和强度，并显式加入碰撞能、离子模式等 covariate；它说明局部碎裂单位可以比纯全局分子向量更可解释。[FIORA](https://www.nature.com/articles/s41467-025-57422-4)
- ModiFinder 只在已定义的结构修饰关系内解释 shifted/unshifted peak，并使用同 adduct、同 instrument 的 helper spectra；它说明 shared/unmatched peak 不能脱离结构关系被一概解释。[ModiFinder](https://pmc.ncbi.nlm.nih.gov/articles/PMC11540723/)

因此，现有 3,486 条 ChemAware 规则最值得做的不是再生成一个 rule-overlap score，而是产生训练期的局部弱标签：peak subformula、neutral-loss class、fragment-pair motif 和不确定性 mask。规则只有在峰级可定位、删峰忠实性和结构解码三项同时成立时才进入监督。

### 2.6 Denoising Search：有效“噪声”必须有化学模型和真实低质量场景

Spectral Denoising 在 240 个标准品上验证，并在高噪声和低浓度条件下提高识别；论文报告平均可在低 35 倍浓度下取得高置信鉴定，并在 Astral 血浆数据中得到 2.5 倍注释数。[Denoising Search](https://pubmed.ncbi.nlm.nih.gov/40155721/)

其启示不是“继续多删峰”，而是：augmentation 必须对应一个可解释的化学/电子噪声过程，并在该过程真正发生的低质量谱层验证。我们 E1 得到的重复谱 missingness 分布可用于条件视图生成，但必须保留 clean retrieval 与 matched-random 对照，不能把任何删峰后的分数变化自动当因果特异性。

### 2.7 DeepMet：多证据融合要由独立真值校准，而不是固定乘加

DeepMet 把模型置信、预测—实测 MS/MS、同位素和预测—实测 RT 交给交叉验证的 meta-learner，在 held-out 标准品上把注释准确率提高到 70%。[DeepMet](https://www.nature.com/articles/s41586-025-09969-x)

它支持多证据思想，但也给出严格边界：RT、同位素、网络上下文更适合作为注释可靠性层；没有真实标准/独立真值时，不能把这些后验信息蒸馏成“化学 embedding 正确”的证据。

### 2.8 StructureMASST 与 MetDNA3：适合下游发现，不应替代核心 embedding 学习

StructureMASST 对同一结构跨 ion form、CE 和 instrument 的多张标准谱进行 multi-MASST，显示单张参考谱会漏掉大量公共样本命中；其知识库包含 1.56M 参考谱和 200,258 个二维结构。[StructureMASST](https://www.nature.com/articles/s41587-026-03082-8)

MetDNA3 通过 MS1、反应关系和 MS2 约束连接知识层与数据层，报告超过 1,600 个标准品 seed 和 12,000 个传播注释；作者也明确承认预测反应对可能有假阳性，且 RT/CCS 等正交证据仍重要。[MetDNA3](https://www.nature.com/articles/s41467-025-63536-6)

这两类方法适合作为我们未来的 biological discovery/annotation layer，但仓库 B44–B46 已证明静态 catalogue membership、degree/topology 和聚合 context 在外部迁移中不成立。它们不能反向成为当前 shared embedding 的伪标签。

## 3. 2026 年文献同时发出的风险警报

### 3.1 不能相信未经审计的最高榜单数字

MassSpecGym in the Wild 审计了首年采用该基准的论文，发现 26 篇中至少 17 篇存在数据泄漏、shortcut learning、实现 bug 或指标分歧，并据此发布 v1.5。[MassSpecGym in the Wild](https://arxiv.org/abs/2606.19624)

更直接的例子是 SpecBridge：它曾报告相对强基线 20–25% 的 Top-1 提升，但目前已经撤稿，页面明确写明预处理/评测管线存在问题。[SpecBridge 撤稿页](https://arxiv.org/abs/2601.17204)

这意味着我们的公式隔离、身份隔离、候选协议、并列规则、冻结工件和 source-shift 审计不是“过度谨慎”，而是 2026 年该领域最需要的方法学竞争力。

### 3.2 排名性能与可靠性必须分开

2026 年的 conformal retrieval 工作显示，在分布漂移下，可靠候选集合会变大；conditional conformal prediction 可改善不同难度子组的覆盖校准。[Conformal retrieval](https://pubmed.ncbi.nlm.nih.gov/42113637/)

因此最终系统至少输出两样：新 embedding 的相似度/排名，以及独立的 spectrum reliability/abstention。不能用拒答提高后的子集准确率冒充全体 embedding 变强。

## 4. 为什么过去学生学不到 3–5 pp

文献和仓库结果共同支持五个原因。

1. **教师信号不是一个可实现的函数。** outcome-aware oracle 会为每个 query 看答案选动作；shared encoder 推理时看不到答案。4.93 pp 是动作集合覆盖率，不是可学习映射的下界。
2. **把“条件变化”与“结构差异”塞进同一个 cosine 目标。** 拉近跨条件同分子时，near isomer 也被一起拉近；MVP 用共识视图分离这个矛盾，MS2DeepScore 用连续结构监督和均衡采样处理它。
3. **全局 precursor token 压缩了局部互补通道。** A1 raw multichannel 为正而 A1b official cosine 为负，与 FLARE 的 late-interaction动机一致。
4. **旧 noise target 学的是 action embedding，而不是 clean ranking 的可实现边界。** target/control action 在冻结空间有效，不等于它们的梯度会改善未见 clean spectrum；仓库四臂实验已直接证明 dynamic、static、matched-random 几乎重合。
5. **数据暴露单位错误。** 按 spectrum/action row 计数，会让高重复身份和高覆盖动作统治优化；应按 identity/formula 和结构相似度区间等权。

最重要的修正是：**不再要求学生模仿某个删峰后 embedding 的位移；要求学生同时满足身份不变、结构连续、近异构体排序和局部化学对应。**

## 5. 推荐的模型：CF-MV-DreaMS

### 5.1 输入与输出

- 推理输入：一张 clean/raw MS/MS spectrum；不需要候选结构、P2b、表型或网络上下文。
- 推理输出：兼容现有检索的 global spectrum embedding `g`。
- 训练时额外保留：最后层 peak tokens `P={p_k}`；它们用于局部化学监督和教师比较，但不强制下游使用。
- 高潜力备选：输出 `g + K` 个 relation tokens，采用 multi-vector similarity；若坚持单向量，则先训练 multi-vector teacher，再蒸馏到 `g`。

### 5.2 四种训练视图

对训练分子 `m` 构建：

1. `x_m,c`：不同 instrument/CE/adduct 的 individual spectra；
2. `x_m,cons`：由真实重复谱构建的 condition-aware consensus，而不是简单全峰 union；
3. `s_m`：结构图/指纹，作为训练期 privileged view；
4. `a(x)`：按 E1 经验分布生成的受控 acquisition-noise view。

结构视图只在训练中提供几何监督，不进入 spectrum-to-spectrum 推理。

### 5.3 损失不是一项，而是职责互斥的五项

设 `g=f_theta(x)`，`P=peak_tokens(x)`：

1. **`L_view`：individual ↔ consensus 对齐。** 同身份跨条件谱靠近共识，不直接互相盲拉；避免一张异常谱成为所有正例的中心。
2. **`L_struct`：连续结构几何。** 预测 Morgan/Tanimoto、MCES 或分层结构相似度；按十个相似度区间和 molecule 等权采样，避免只学 identity 0/1。
3. **`L_rank`：同分子式候选组内 listwise。** 对每个 query，把不同条件同身份正例排在 same-formula near/mid negatives 之前；公式组等权，ties 计错。
4. **`L_local`：peak ↔ subformula/fragment 弱对齐。** 借鉴 MIST/FLARE，以规则库、子式枚举或 fragment model 产生带置信度的局部对应；使用 OT/MaxSim 或辅助分类，不把规则总分当身份标签。
5. **`L_noise`：经验噪声一致性。** clean 与 E1 acquisition view 一致，但其权重只由条件可靠度决定；它是鲁棒正则，不承担纠正 near isomer 的职责。

另加 `L_preserve` 对 protected/easy 关系做 relational distillation，防止新空间为攻 near 而破坏原本正确的全局邻域。损失权重不能一次大扫网格；先逐项证明每一项相对 matched control 的独立价值。

### 5.4 正负样本与批次

每批以 molecule 为单位，而不是 spectrum/action row：

- 一个 molecule 抽 2–4 个真实 condition views + 1 个 consensus；
- negatives 分成 same-formula near、same-formula mid、mass-window far 和跨 formula 四层；
- 各结构相似度 bin 等频，molecule 最大曝光差控制在 15% 左右；
- 重复谱多的分子不得获得更高总权重；
- query、positive、negative 的 condition distribution 尽量匹配，避免模型从 instrument/CE 捷径识别标签；
- 缺失 CE 视为 unknown stratum，不伪造为 cross-CE。

## 6. 成本受控、一步一门的实验顺序

### G0：冻结表示充分性审计，不更新 DreaMS

目的：确认已有 A1 多通道信号在扩大后的训练图上仍然存在，并判断信息位于 global 还是 local interaction。

固定比较：

1. official global cosine；
2. raw multichannel strong baseline；
3. frozen DreaMS peak-token late interaction；
4. individual→consensus similarity；
5. 3 与 4 的固定、nested-OOF 学习组合。

所有模型在 formula-group nested OOF 训练，candidate protocol 完全一致；结构仅用于训练/评价，不参与事后选子集。

进入 G1 的最低门：

- overall Recall@1 相对 official `>= +2.0 pp`；
- near Recall@1 `>= +2.0 pp`；
- formula-cluster CI 下界 `>0`；
- corrected `> 2 × introduced`；
- instrument/ion-mode 主要层均不为负；
- 随机 consensus 与 peak-token permutation 均不能复现主支路增益；landmark 支路另以 matched-random landmark 作独立诊断，不得因该辅助支路失败而否定已通过的 token+consensus 主支路。

若 G0 不过，说明当前 5 pp 只存在于小 A1 panel 或 outcome-aware oracle，禁止开 shared encoder 训练。

### G1：训练多视图教师，仍不解冻 116M backbone

只训练 consensus encoder、structure projection、local interaction 和 global projection；冻结 DreaMS peak tokens。目的不是追求最终 SOTA，而是证明一个**不看结果标签**的教师能稳定读出 raw 多通道信息。

除 G0 门外，要求五个 formula outer folds 全部方向非负；否则先修采样/协议，不解冻 backbone。

### G2：单向量 student 转移测试

学生只看一张 clean spectrum，蒸馏教师的：候选组内排序、连续结构几何和局部一致性；不蒸馏 oracle action identity。

先只训 projection + 一个小 adapter。进入解冻最后层的门：

- student 至少保留 G1 教师 `50%` 的 overall 和 near 增益；
- corrected/introduced 风险比不低于教师的 70%；
- protected correct degradation `<=0.2 pp`；
- 主要 condition strata 不出现显著负向。

若学生反复低于教师增益的 50%，应接受“单向量瓶颈”，转为 `g + K relation tokens` 的多向量 embedding，而不是继续调学习率。

### G3：shared encoder 微调

仅在 G2 过门后，解冻最后一个 Transformer block + projection/local heads；初始化与成熟 E4 对齐。学习率先复用成熟 E4 的 backbone `2e-6`、head `1e-5`，不与层数、epoch、loss weight 同时扫描。第一轮 one seed / one outer fold，比较：

- consensus + structure + local + empirical noise；
- 去掉 local；
- 去掉 consensus；
- empirical noise 换 matched random；
- official DreaMS continuation。

只有目标臂相对 official continuation 和 matched-random 都有严格正的 formula CI，才做第二 seed 和全 folds。

### G4：封存外部验证与真实生物学应用

开发结束后重新锁外部集，使用 MassSpecGym v1.5 规范审计 canonicalization、候选、ties 和 source shift。报告至少三套协议：

1. natural library retrieval；
2. cross-condition same-identity retrieval；
3. same-formula near-isomer stress test。

性能模型之外另训练/calibrate reliability evaluator；真实生物学注释仍需要 RT、同位素、EIC、标准品或其他正交证据分级。更好的 embedding 提高候选排序与谱库覆盖，但不能单独把 Level 2 变成 Level 1。

## 7. 哪些前沿想法现在值得借，哪些不值得

| 方法 | 借什么 | 不借什么 | 优先级 |
|---|---|---|---:|
| MVP | individual/consensus/structure 多视图共同对齐 | 已知母分子式依赖、只报告 candidate retrieval | 1 |
| FLARE | peak-node late interaction；先保留局部信息 | candidate-aware 推理直接冒充 spectrum embedding | 1 |
| MS2DeepScore 2.0 | molecule-balanced、十个相似度 bin、embedding evaluator | 只用全局 Tanimoto MSE 解决所有 near isomer | 1 |
| MIST | 子式 token 和 fragment auxiliary loss | 把规则命中当真值身份 | 1 |
| ModiFinder | 结构关系限定的 shifted/unshifted peak | 泛化为任意 shared/unmatched 删除 | 2 |
| Denoising Search | 化学噪声模型和真实低质量 benchmark | 无差别高比例删峰 | 2 |
| DeepMet | 真值校准的多证据 meta-learner | 把 RT/同位素/网络蒸馏成纯谱 embedding | 2（下游） |
| StructureMASST | 多标准谱、条件元数据、公共队列映射 | 把多次查询称为新 embedding | 2（应用） |
| MetDNA3 | 数据层—知识层双网络和反应约束 | 静态 catalogue membership/degree 捷径 | 3（需新 benchmark） |
| MSAlign | 冻结 foundation encoders 的轻量对齐 | 未经 v1.5/外部复核直接信榜单 | 观察 |
| SpecBridge | 不采用 | 已撤稿且承认评测管线问题 | 停止 |

## 8. 对当前项目的最终判断

1. **用户所说的“5 pp 是好方向”基本正确，但要精确定义。** 它支持“显式多通道关系信息有价值”，不支持“现有官方 DreaMS 单 cosine 已经能直接读取该信息”。
2. **下一步不是再试一个融合公式。** A1b 已经否定这种低容量读法；文献提示应在训练中保留多视图和峰级局部交互。
3. **噪声微调仍然保留，而且位置更清楚。** 它负责把真实 acquisition variation 变成 condition-invariant view；它不再独自承担 near-isomer discrimination。
4. **ChemAware 规则库没有白费。** 最有价值的用途是构建带置信度的 peak/subformula/neutral-loss 局部监督，与双重映射直接统一；静态 rule score 和 catalogue degree 已被仓库外部结果否定。
5. **最可能获得大幅提升的架构变化，是从 single pooled cosine 转到 multiview teacher + local interaction，再蒸馏回 global embedding；如果蒸馏失败，就承认 multi-vector embedding 是必要输出。**
6. **当前不应立刻大训练。** 先做 G0，成本低、裁决力高；G0 如果连 +2 pp 都不能在扩大、公式隔离的图上复现，就说明小 panel 的 5 pp 不足以支撑主线。

## 9. 论文层面的可形成创新

若 G0–G4 依次通过，方法学创新不应写成“给 DreaMS 加噪声”，而应写成：

> 一个以真实采集条件视图、共识谱和峰—结构弱对齐共同监督的谱图基础模型微调框架，在候选无关推理下同时改善跨条件身份稳定性与同分子式近异构体分辨，并通过局部峰证据和独立可靠性估计解释每次检索。

这比单一 noise augmentation 更硬，原因是它直接解决当前领域中两个相互冲突的问题：condition invariance 与 fine-grained structural sensitivity；同时把我们已有的错误图谱、经验 missingness、ChemAware 规则库、双重映射和外部严格评测统一到一条可证伪路线中。

但在 G4 之前，禁止使用“SOTA”“全面超过 DreaMS”或“5 pp shared embedding 提升”。

## 主要文献

1. Bushuiev et al. [Self-supervised learning of molecular representations from millions of tandem mass spectra using DreaMS](https://www.nature.com/articles/s41587-025-02663-3). Nature Biotechnology, 2025/2026.
2. Wang et al. [Learning from All Views: A Multiview Contrastive Framework for Metabolite Annotation](https://pmc.ncbi.nlm.nih.gov/articles/PMC12980492/). Analytical Chemistry, 2026.
3. de Jonge et al. [Cross ionization mode chemical similarity prediction between tandem mass spectra in metabolomics](https://www.nature.com/articles/s41467-026-69083-y). Nature Communications, 2026.
4. Geng et al. [MS2-SMILES AlignNet](https://pubmed.ncbi.nlm.nih.gov/42308958/). Talanta, 2026.
5. FLARE. [Fine-grained peak-node alignment preprint](https://pubmed.ncbi.nlm.nih.gov/41659479/), 2026.
6. Goldman et al. [MIST: domain-inspired chemical formula transformers](https://www.nature.com/articles/s42256-023-00708-3). Nature Machine Intelligence, 2023.
7. Ge et al. [FIORA](https://www.nature.com/articles/s41467-025-57422-4). Nature Communications, 2025.
8. ModiFinder. [Tandem Mass Spectral Alignment Enables Structural Modification Site Localization](https://pmc.ncbi.nlm.nih.gov/articles/PMC11540723/), 2024.
9. Kong et al. [Denoising Search](https://pubmed.ncbi.nlm.nih.gov/40155721/). Nature Methods, 2025.
10. Dhanasekaran et al. [DeepMet](https://www.nature.com/articles/s41586-025-09969-x). Nature, 2026.
11. Bittremieux et al. [Structure-centric searching enables global mapping of the public metabolome](https://www.nature.com/articles/s41587-026-03082-8). Nature Biotechnology, 2026.
12. Zhang et al. [MetDNA3 two-layer networking](https://www.nature.com/articles/s41467-025-63536-6). Nature Communications, 2025.
13. Liu et al. [MassSpecGym in the Wild](https://arxiv.org/abs/2606.19624), 2026 preprint.
14. Rakhshaninejad et al. [Reliable Molecular Retrieval Using Conformal Prediction](https://pubmed.ncbi.nlm.nih.gov/42113637/). JCIM, 2026.
15. Krzakala et al. [MSAlign](https://arxiv.org/abs/2605.19752), 2026 preprint.
16. Wang et al. [SpecBridge withdrawn record](https://arxiv.org/abs/2601.17204), 2026.
