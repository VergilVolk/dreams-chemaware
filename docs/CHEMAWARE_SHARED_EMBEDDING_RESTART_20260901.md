# ChemAware 共享 embedding 重启合同（2026-09-01）

## 1. 结论先行

旧 ChemAware 不再继续修补。新路线从官方 DreaMS 共享 encoder 出发，先回答一个更严格的问题：

> 在 query/reference 使用同一个、仅接收 clean spectrum 的编码器时，训练期化学结构与峰级概念监督能否在完整 strict-10ppm、same-adduct、按分子聚合的候选图上，带来超过普通 clean listwise continuation 的可复现增量？

只有这个因子问题成立，新增结果才可以称为 ChemAware shared embedding。规则数量、规则 loss、pair AUROC、候选后验分数、冻结动作 headroom 和谱图—分子 cross-encoder 分数都不能替代该结论。

## 2. 旧路线的失败判决

### 2.1 规则 attention bias 不是检索监督

历史 `ChemAwareDreaMS` 把观察到的质量规则转成 attention bias 或 Transformer 后残差，但训练目标仍主要是 masked-peak reconstruction。它没有直接优化完整候选组中的正确分子排名，因此“规则进入模型”不等于“规则改善身份几何”。

### 2.2 规则重叠不能充当身份距离

3,486 条规则是谱图层面的观察质量模式：

- core neutral loss 使用 precursor-minus-fragment；
- ISO/HR 使用 peak-pair；
- 79 条 MassBank mass-diff 实际是 precursor exact-mass offset；
- 两张谱都没有命中规则时是 missing，不是 Jaccard=1。

这些规则可用于概念监督、解释、困难样本分层和冲突挖掘，但不等于分子结构标签。历史 rule-overlap triplet/InfoNCE 永久退出主线。

### 2.3 projection-only 容量不足

冻结 116M backbone、只训练一个线性投影，容易同时移动正负相似度，却不能改变峰 token 的局部贡献。已有多轮结果证明该容量不足以解决 near/isomer 局部排序。

### 2.4 单正例、单负例与子图训练错配正式检索

正式任务按 query 的完整候选分子组评估，并对同一分子的多张参考谱取最大分数。单 triplet 或少量冻结 hard negative 无法覆盖新增错误，也不能为完整决策边界提供梯度。

### 2.5 化学辅助任务不能掩盖普通 continuation

即使“clean rank + chemical loss”优于官方权重，也不代表化学监督贡献了增量。必须至少同时训练：

1. official frozen baseline；
2. clean listwise continuation；
3. clean listwise + frozen-chemistry candidate hardness；
4. clean listwise + frozen-chemistry candidate hardness + peak concept；
5. matched-capacity random/mismatched concept control。

ChemAware 的正式增量是 3/4 相对 2 和 5 的配对差，不是相对官方权重的总差。

## 3. 最新方法给出的可迁移结论

### 3.1 DreaMS

DreaMS 证明大规模无标签谱图预训练能形成强共享谱图表示，但其论文也明确指出 zero-shot embedding 对细小结构差异不够敏感，后续采用近质量 hard triplet 微调。

来源：<https://www.nature.com/articles/s41587-025-02663-3>

### 3.2 FLARE

FLARE 的核心增量来自双向 peak-to-atom / atom-to-peak MaxSim，而不是只比较 pooled global embedding。可迁移结论是：化学监督应触达局部 peak token，并用局部证据支持全局匹配。

不能直接照搬之处：FLARE 部署时使用候选分子图，是 spectrum-to-molecule scorer；本项目要证明的是 spectrum-to-spectrum shared embedding。因此 peak-node 模块只能作为训练期教师或辅助目标，不能成为正式共享 embedding 的推理依赖。

来源：<https://pmc.ncbi.nlm.nih.gov/articles/PMC12873900/>

### 3.3 SpecBridge 与 MSAlign

两者都表明：将 DreaMS 与冻结分子 foundation model 对齐，并用候选组 contrastive objective，比从零训练跨模态模型更稳定。MSAlign 进一步把 candidate-based contrastive learning 作为关键设计。

可迁移结论：

- molecule encoder 先冻结；
- candidate group 而不是随机 batch negatives 决定主梯度；
- 若使用 spectrum projector，它必须成为 query/reference 共享推理路径的一部分并接受 matched clean control；训练后丢弃的 projector 不能证明共享 embedding 获得化学能力；
- molecule teacher 只在训练期存在，最终谱图共享 encoder 不依赖候选。

来源：

- <https://arxiv.org/abs/2601.17204>
- <https://arxiv.org/abs/2605.19752>

### 3.4 “What are we optimizing for?” 与 MassSpecGym v1.5

2026 年检索目标研究指出，更准确的 fingerprint prediction 不一定转化为更好的 candidate retrieval。因此 fingerprint/substructure loss 只能是辅助，完整 molecule-listwise loss 必须是主目标。

MassSpecGym in the Wild 审计又表明，预训练泄漏、SMILES/candidate shortcut、实现 bug 和指标漂移足以制造虚假结论。正式外评必须锁定 v1.5 数据、候选集、canonicalization 和指标实现；项目内部全图比较也必须固定同一 query/candidate protocol。

来源：

- <https://openreview.net/pdf?id=hiD1bo11Mm>
- <https://huggingface.co/datasets/roman-bushuiev/MassSpecGym/blob/main/README.md>
- <https://openreview.net/pdf?id=I1PUDXXYot>

## 4. 新架构：ChemAware Shared Encoder v2

实现入口：`dreams/models/chem_aware/shared_embedding_v2.py`

### 4.1 部署合同

```text
clean spectrum
  -> official DreaMS backbone contextual peak tokens
  -> official projection head -> z_official
  -> signed peak residual adapter(tokens, m/z, intensity, neutral loss)
  -> z_chem = normalize(z_official + bounded_delta)
```

推理时：

- 只输入一张 clean spectrum；
- query/reference 使用同一权重；
- 不输入 candidate、SMILES、正确结构、P2b、反应网络或 phenotype；
- 输出一个归一化共享 embedding；
- 化学教师和 concept decoder 可从发布 checkpoint 中移除。

### 4.2 signed peak residual

adapter 使用两个独立峰通道：

- support channel：学习可能不足的正例证据；
- conflict channel：学习可能造成错误近邻的混淆证据。

两者都只读取单谱的 contextual peak token、fragment m/z、intensity 和 precursor-minus-fragment neutral loss。最终 residual 有范数上界并零初始化，因此 epoch 0 必须逐位复现官方 embedding。

这两个通道不是预先指定“好峰/坏峰”的硬标签。其语义必须通过训练后概念 probe、峰干预和 matched-random control 验证。

### 4.3 主目标：完整分子候选组 listwise

对 query `q`、候选分子 `m` 的多张参考谱 `r`：

```text
s(q,m) = max_r cosine(z_chem(q), z_chem(r))
L_rank = -log softmax_m(s(q,m)/temperature)[true_molecule]
```

硬要求：

- strict 10 ppm；
- same adduct；
- self-row excluded；
- 正确分子唯一且排在候选组第一槽；
- 同分子多参考谱 max 聚合；
- 每个 identity 总训练权重相等；
- 每轮使用全部候选或动态覆盖完整负例边界，不能永久冻结一个 negative。

### 4.4 冻结化学教师直接重加权候选边界

最初的 v2 草案曾使用可训练的 `spectrum -> molecule` projector，并准备在部署时丢弃。二次审查判定这是捷径：projector 可以独自吸收 molecule loss，而共享谱图几何不发生有效改变。因此正式 G2 禁止训练期可丢弃 projector。

已知训练分子的冻结表示 `u(m)` 只用于计算完整候选组中负分子的相对化学困难度：

```text
h(q,m) = cosine(u(true_molecule), u(m))
log w(q,m) = beta * h(q,m) - mean_negative(beta * h)

L_chem = -log [ exp(s_pos/T) /
                (exp(s_pos/T) + sum_m w(q,m) * exp(s_neg_m/T)) ]

L_retrieval = (1 - lambda_mol) * L_rank + lambda_mol * L_chem
```

因此：

- 化学梯度直接作用于部署时使用的 spectrum-spectrum candidate scores；
- 负例权重在组内做几何均值归一化，改变“哪些负例重要”，不偷偷放大整个 loss；
- 少于两个可观测负例时化学权重退化为 1，不伪造化学监督；
- `lambda_mol` 被限制在 `(0,1]`，总检索项是 clean 与 chemical listwise 的凸组合；
- teacher 只读取训练折允许的分子，held-out formula 的结构表示即使预先缓存也不能进入训练 batch。

第一轮使用 frozen MolFormer connectivity geometry；后续可以在同一接口比较 ECFP4/Tanimoto-preserving embedding 和 train-fold-only graph teacher。选择依据始终是共享谱图检索相对 G1 与伪教师的配对增量，不是 cross-modal 分数。

### 4.5 peak-level concept auxiliary

概念分为两组并独立报告：

- observed rule concepts：3,486 条质量模式及压缩后的语义类别；
- structure concepts：训练结构的 ECFP/substructure/formula environment。

每个标签必须有 `positive / negative / unobservable` 三态。`unobservable` 不进入 BCE，禁止把未命中规则解释成化学结构不存在。

局部版本使用 FLARE 式 peak-to-substructure teacher，但只蒸馏到 peak token 与共享 residual；正式推理不保留 molecule nodes。

### 4.6 安全损失

- `L_protect`：对官方已正确 query 保留其 positive-vs-hardest-negative margin；
- `L_preserve`：限制 `cos(z_chem, z_official)`；
- introduced replay：新引入错误在后续 epoch 进入风险 replay；
- near/isomer replay：同分子式与 MCES-near 组必须持续出现。

主损失：

```text
L = L_retrieval
  + lambda_concept * L_concept
  + lambda_protect * L_protect
  + lambda_preserve * L_preserve
```

各分支先审计真实参数梯度范数和余弦。loss 数值非零不等于对 adapter/backbone 有有效梯度。

## 5. 实验序列与裁决

### G1：clean full-list baseline

目的：测量普通 listwise continuation 的可得增量。

- 零初始化 signed adapter；
- backbone/head 冻结；
- 只训练 `L_rank + L_protect + L_preserve`；
- 5 formula folds × 至少 3 seeds；
- 每折动态重算当前学生 hard negatives；
- 报 overall、near、MRR、corrected、introduced、preservation。

若 G1 已经获得大部分增量，后续必须把它作为新的 clean baseline，不能把总增量归入 ChemAware。

### G2：frozen molecule teacher

在 G1 完全相同的 sampler、步数、adapter 容量、初始化、学习率与选模规则上，将一部分 `L_rank` 替换为 frozen-teacher chemical-weighted listwise；不增加可训练 projector。

必须同时运行：

- correct molecule teacher；
- identity-permuted teacher；
- matched-random-marginal teacher；
- correct teacher restricted to the exactly observable same-formula scope；
- same-formula mismatched teacher on that identical scope。

最后两臂必须共享同一个 observable mask。某分子式在训练折内只有一个身份时，该身份标记为 unobservable 并跳过 molecule loss；禁止用全局随机身份填补后仍称为 same-formula control。

只有 correct teacher 显著优于所有对照，且 near 不退化，才进入 G3。

### G3：peak-level chemical concept

增加 structure concept 和 observed-rule concept，先分别加入，再联合。

通过条件：

- G3 相对 G2 有配对净增量；
- peak intervention 对目标概念显著优于强度/频率匹配随机峰；
- concept branch 对 adapter/最后 block 有稳定非零梯度；
- rule-only 组不能把空命中当负标签；
- introduced 不集中在高规则覆盖或特定 formula 类。

### G4：容量扩展

仅在 G2 或 G3 通过后比较：

1. adapter only；
2. adapter + official projection head；
3. adapter + last Transformer block；
4. adapter + last two blocks / LoRA。

使用分层学习率，且 dropout 在冻结 backbone 中关闭。容量扩展若只提升 development、不跨 fold/seed，回退到前一配置。

### G5：正式外评

- 固定所有超参数和 checkpoint；
- 冻结开发图后使用一个未消费的 identity/formula 隔离外部面板；
- 或按 MassSpecGym v1.5 data-safe 协议一次性评估；
- 使用同一个 shared checkpoint 重新编码 query/reference；
- 不允许复用官方 query embedding 再只改 reference；
- 不与 P2b/BioAware 相加后冒充 shared-weight 提升。

## 6. 性能门与“很多 pp”的真实含义

项目目标仍是多个百分点，但工程裁决不能预设结果。

每一轮必须先报告：

- baseline Recall@1；
- 剩余错误数；
- oracle/headroom；
- 达到 +4 pp 需要净修正多少 query；
- 当前训练池覆盖其中多少 identity/formula/near query；
- introduced 风险预算。

若 formal G0/G1 证明净 headroom 或可学习覆盖不足 4 pp，则“多 pp”需要改变数据/任务或引入新的结构教师，不能通过换评价子集、放宽候选或相加后处理成绩实现。

## 7. 当前本地与集群状态

### 已完成

- 旧规则 bias、rule-overlap contrastive、projection-only、单 triplet 路线已审计并退出主线；
- G0 核心语义测试 5 项通过；
- v2 shared encoder/loss 测试 6 项通过；
- 本地 MassSpecGym HDF5 可读：231,104 spectra、28,929 IK14，其中 24,213 个身份至少有两张谱；
- v2 零初始化、候选无关接口、分子级 max/listwise、concept missing mask 和 identity-equal weighting 已代码化；
- 新增独立 v2 token cache 格式，强制保存 precursor m/z、官方 contextual peak tokens 与同批官方 embedding，拒绝把缺少 precursor 的历史 noise cache 偷换为新缓存；
- 新增 fail-closed 预检：graph SHA256、official checkpoint SHA256、全图 spectrum-row 精确覆盖、1024 维一致性、官方 embedding 单位范数、以及每个 graph pair 的官方 cosine 逐项对账；
- formal preflight 进一步硬编码冻结账本的 graph/HDF5/official/raw 四个 SHA256，要求 G0 rule report 与 full-audit 均为 formal pass、3,486 rules、23,876 queries、25,275 spectra、3,472 identities 且 P3 overlap=0；token 与 molecule cache 都记录同一 preflight SHA256，训练时再次对账；
- G1 trainer 已实现 formula 外折/内折隔离，并从训练候选库中同时剔除两个 held-out formula fold，避免 held-out 结构以负例形式泄漏；
- G1 每批使用全部 split-eligible 候选分子，query/reference 共用同一 adapter，并按部署协议对同分子参考谱取最大值；
- G1 正式入口已拆成 cache、5 folds × 3 seeds array、paired summary 三个 sbatch；summary 会拒绝缺折、重复/遗漏 query、非 formal 产物与跨 seed 官方基线不一致；
- G2 teacher cache、5-arm 对照矩阵与配对汇总已实现；MolFormer 只保留为被对照否定的历史教师，正式 G2 直接从冻结 graph 与 HDF5 结构字段重建身份—SMILES ledger，再构造 Morgan connectivity teacher，不再为了取得 SMILES 而依赖或运行 MolFormer；cache 锁定 graph/HDF5/preflight，并显式审计非手性 SMILES 坍缩和 fingerprint 碰撞；
- G2 五臂为 correct、identity-permuted、random-marginal、correct-same-formula-scope、same-formula-mismatched；后两臂共享完全相同的可观测 mask；
- G1/G2 每个 fold 会对账 training query、allowed molecule、adapter 初始权重和所有共同训练超参数的哈希，避免把初始化或采样差异误报为化学增量；
- G2 只有在三个 seeds 上同时配对胜过 G1、全局伪教师和同分子式错配教师，并通过 near/risk 门，才设置 `pass_to_G3=true`；
- G2 每个 epoch 必须记录 `gradient(adapter, L_chem - L_clean)` 的范数、非零参数张量数及其与 clean gradient 的余弦；十个 batch 内找不到非零增量梯度则训练立即失败，正式汇总也拒绝缺失该证据的 fold；
- G2 单折五臂 pilot 已有独立 fail-closed 裁决器：除排序风险门和 128 维 CountSketch 方向门外，还逐臂对账训练查询、allowed molecule、PEFT 初始化/容量、共同训练合同和可观测候选 mask SHA256；完整 75-task 数组启动前会重新哈希 pilot 的 decision、预测与 checkpoint，任一工件漂移即拒绝运行；
- 本地已完成缓存预检—训练—选模—外折预测—checkpoint—多 seed 汇总的端到端 smoke；移除会吸收化学 loss 的可丢弃 projector、加入增量梯度门与 G2b 冻结 probe 合同后，当前 ChemAware 核心、管线、PEFT 与推理测试合计 36 项通过。
- 已直接加载本地官方 `official_embedding_slim.pt`、116M DreaMS backbone 和一张真实 MassSpecGym 谱图做部署路径 smoke：117,101,029 参数模型上 v2 epoch-0 embedding 的最大绝对误差为 0、delta norm 为 0、输出 norm 为 1；因此零初始化合同不只在 mock 网络成立。
- 统一 `shared_dreams_inference.py` 已支持 G1/G2 checkpoint：加载时强制核对 official-checkpoint SHA256、shared/P2b/candidate-input 合同，只挂载 adapter；任何声明使用或携带可丢弃 molecule projector 的 checkpoint 会 fail-closed。
- bounded-capacity reachability smoke 已构造一个官方 positive=0.78、hardest negative=0.81 的真实局部排序错误；共享 adapter 在 `delta_bound=0.12` 下把正确分数提升到错误负例之上，同时所有 residual norm `<0.12`、平均 official preservation `>0.99`。这只证明容量可修正小 margin，不代表真实图泛化。

### 2026-09-02 本地真实谱图诊断与容量转向

本地从已有官方 DreaMS embedding/token 缓存中构造了两个完全分子身份隔离的诊断图。发现图含 922 queries、457 query identities、927 reachable spectra，官方 Recall@1 为 74.84%；确认图含 921 queries、457 query identities、927 reachable spectra，官方 Recall@1 为 76.87%。两图各有 460 个候选身份，交集为 0。它们来自 mass-dense 预选队列，因此只用于机制诊断，不是正式 benchmark。

在发现图上，原始 v2 adapter 的五折净结果只有 3 corrected / 1 introduced，即 +0.217 pp；contextual gate 为 4 corrected / 2 introduced，净值仍是 +0.217 pp。将发现折模型按 formula cross-fit 到身份完全不重叠的确认图后：

- 未缩放模型为 3 corrected / 3 introduced，Recall@1 净变化 0；
- 只用发现内折选择 residual scale 后为 3 corrected / 2 introduced，即 +0.109 pp，但风险效用 `corrected - 2 × introduced = -1`；
- contextual gate 为 2 corrected / 4 introduced，即 -0.217 pp，风险效用 -6。

峰门控几何解释了这一现象：support/conflict gate 的归一化熵中位数分别为 0.9939/0.9933，两通道 gate cosine 中位数为 0.9762，说明权重几乎均匀且两通道高度重合。增加 top-k、低温度、contextual gate 和 global residual 都没有形成可迁移优势，故停止继续调 adapter 形状。

MolFormer 的五种教师/伪教师对照也未形成化学归因：correct teacher 的收益不能稳定胜过 identity-permuted、random-marginal 或 same-formula-mismatched；correct 与 same-formula-mismatched 的最终 adapted embedding 平均 cosine 为 0.999981。当前结论不是“MolFormer 无用”，而是现有冻结 embedding 上的候选权重机制没有把身份特异的化学几何转成可泛化的谱图几何。

因此新增 v3 G1-PEFT 容量控制：对官方 DreaMS 最后一层 fused Q/K/V/O、FFN in/out 与官方 projection head 安装零初始化低秩增量；base 116M 参数全部冻结，query/reference 仍使用同一 raw-spectrum encoder，主目标仍为完整 split-eligible molecule-listwise。正式默认 rank=8；真实模型 rank=4 smoke 中仅 69,632 个参数可训练，重算缓存的逐坐标最大误差为 `9.87e-8`，严格排名完全一致；首步梯度 L2 为 0.2535、参数更新 L2 为 0.0341，共 4 个低秩参数张量发生改变。该 smoke 只用了一个训练 query，best epoch 为 0，因此仅证明加载、反向、保存与推理链路，不是性能证据。

v3 G2 已按同一 trainer 预先实现，但不自动提交：G1/G2 在同一 seed/fold 下共享初始 PEFT 哈希、容量、训练 query、allowed molecule、完整候选 sampler、epoch 数和选模规则；唯一变化是 chemical-weighted listwise 及其五种 teacher/control。每个 epoch 另行审计 `gradient(L_chem - L_clean)` 是否到达可部署 PEFT 参数。五臂单折 pilot 完成后必须运行独立裁决器；只有正确教师同时产生身份特异的梯度方向、相对 G1/全局伪教师/同分子式错配教师具有正的 `corrected - 2×introduced`，且 near 不退化，才生成 `pass_to_full_matrix=true`。完整 5 folds × 3 seeds × 5 arms 数组只接受该已通过且工件哈希未漂移的裁决文件。

同一批 4 个查询、每个 8 个候选分子的梯度因子审计进一步筛选了结构教师。正式签名使用覆盖全部梯度坐标的固定 128 维 CountSketch，而不是稀疏抽点。MolFormer 下 correct-vs-identity-permuted / random-marginal 的梯度签名 cosine 分别为 0.9803 / 0.9582，但 correct-same-formula-scope vs same-formula-mismatched 高达 0.999967，故淘汰为正式 G2 教师。随后从同一已审计 SMILES ledger 构建 radius-2、2048-bit、非手性 Morgan connectivity teacher；明确使用 L2-normalized binary-bit cosine（Ochiai），不冒称 Tanimoto。Morgan 的对应 cosine 为 0.9395 / 0.9579 / 0.9397，尤其 same-formula correct-vs-mismatch 已从近乎同向降到 0.9397，说明它至少在优化层面提供了身份特异方向。该结果仍只是单 batch 机制证据；正式 G2 pilot 改用 Morgan，最终仍由配对外折排名与伪教师控制裁决。

Morgan cache 也显式审计折叠碰撞：本地 460 个身份中有 2 组、4 个身份得到完全相同的 2048-bit fingerprint；291 个同分子式身份对中仅 1 对完全相同。一个碰撞来自半径有限导致长脂肪链长度差异不可见，另一个继承自非手性 SMILES ledger 的跨身份坍缩。因此 Morgan 也不是“真实结构距离”，只是比当前 MolFormer 更适合这一局部身份边界的冻结教师；碰撞身份不能支持可分辨性主张。

随后对“冻结化学读出器是否可达”做了 nested formula-held-out 诊断：每个外折只用允许身份的官方 DreaMS 谱图质心拟合 ridge map，alpha 在独立内折选择；候选评分使用 Morgan fingerprint，因此这只是训练机制诊断，不是可部署结果。正确 Morgan 标签在 922 queries 上的 Recall@1 为 46.96%、near 为 48.07%；三个严格在当前允许身份集合内部置乱的教师分别为 39.15%、42.30%、39.37%，正确标签平均高 6.69 pp。三次 formula-cluster bootstrap 的 Recall@1 差异下界均大于 0，说明官方 embedding 中确有跨分子式可线性解码的连接结构信号；但相对置乱臂的 `corrected - 2×introduced` 均为负，说明直接用 probe 排序会交换大量原有正确/错误，不能部署，也不能单独作为训练目标。下一步只能把冻结 probe 作为小权重、错误/低 margin 定向的训练约束，并继续保留 official-correct margin protection 与五种伪教师归因。

该风险约束现已实现为独立 G2b，而不是静默替换原 G2。每折只用允许的训练身份及其官方 reference-spectrum 质心拟合 centered ridge probe；probe 仅含 buffer、可训练参数为 0，训练结束后丢弃。化学 listwise 只选择官方 positive-vs-hardest-negative margin `<=0.01` 且正例和至少一个负例具有教师目标的训练 query；官方高 margin 正确 query 的化学梯度严格为 0，低 margin 正确 query 仍受原有 margin-protection loss 约束。选择条件只依赖冻结官方分数和可观测 mask，不依赖正确/伪教师的目标值；五臂 pilot 与多折 summary 都会分别要求全局三臂、同分子式两臂具有完全相同的 selection-query ledger SHA256，否则 fail-closed。

真实 117,101,029 参数 CPU smoke 使用 8 个训练 query、2 个内/外 query 和 rank-4 PEFT。冻结 probe 由 301 个允许身份拟合，1024→2048，0 个可训练参数；首批 8 个 query 中仅 1 个被化学目标选择，且它是官方错误，没有高 margin 正确 query 被选。clean gradient norm 为 0.19820，化学增量 gradient norm 为 0.17445，两者 cosine 为 0.22367；4 个 PEFT 参数张量收到非零化学增量，首步总 gradient norm 为 0.22021、update norm 为 0.03789。该 smoke 的 best epoch 仍为 0，因此只证明信号确实到达可部署 PEFT，并非性能证据。保存包只含 8 个 PEFT 张量；真实 116M backbone 经统一推理入口重建后共有 117,170,661 个参数、全部冻结，明确报告 `training_only_frozen_probe_loaded=false`，没有 probe/teacher state。

G2b 使用独立的 pilot、pilot-decision、75-task full matrix、summary 目录和提交脚本，不会覆盖 G2 candidate-hardness 产物。单折 pilot 仍必须同时通过相对 G1 的正风险净值、near 非退化、相对两个全局伪教师和同分子式错配教师的正风险净值，以及 128 维化学增量梯度方向非同一；未通过时 full matrix 不会启动。G2 与 G2b 都只是待正式外折裁决的候选机制，当前不选择胜者，也不合并其结果。

最新方法审查也检查了候选式 cross-modal alignment：MSAlign（arXiv:2605.19752）报告在冻结 DreaMS/ChemBERTa 上，candidate-based InfoNCE 稳定优于 fingerprint regression 和普通 in-batch contrastive；这支持我们继续使用完整候选 hard negatives，但 MSAlign 的 molecule projection 是其部署评分器，不能直接作为本项目“只部署 shared spectrum encoder”的结果。DreaMS 原论文使用同一分子不同谱图正对和近质量负例，FLARE 则强调局部 peak-node alignment；后者依赖 formula/subformula 与 molecule graph，在正式 G0 规则工件尚未到本地前不适合直接移植。

据此实现过一个只限机制诊断的 multiview probe：在被 G2b 选中的 query 内，同时让 query 与正分子的 reference 谱图接受同一冻结 Morgan probe listwise；各 view 先在 query 内平均，避免 reference 多的身份超权。真实 rank-4、8-query smoke 中，multiview 化学增量梯度 norm 为 0.09575，仅为 query-only 0.17445 的 54.9%；两者 128 维签名 cosine 为 0.9860，总首步梯度也几乎不变（0.21926 vs 0.22021）。这说明 reference view 主要重复并稀释了现有方向，尚无理由增加正式矩阵。代码保留用于复现实验负结果，但 formal trainer 会明确拒绝该 objective，G2b 正式 recipe 继续是 query-only。

随后在同一本地诊断图上扩大到 64 个训练 query、100 个内折和 100 个外折 query，并按正式 schedule 跑 5 epochs。纯 G1 在第 5 轮内折修正 1/100、introduced=0（Recall@1 +1 pp，near +1.25 pp），但公式隔离外折为 0 corrected/0 introduced，仅有 2 个非首位 rank 改善、MRR +0.25 pp。G2b correct 把 `lambda_probe` 提到 0.25；33/64 个训练 query 被选择且全部为官方错误，高 margin 正确仍为 0 个，但五轮后内折没有任何 rank 改变，未能复现、更未超过 G1 的一个修正，best epoch 回退为 0。因此不启动该权重的伪教师臂，也不提高正式 G2b 默认 `lambda_probe=0.05`。此结果仍是 mass-dense 本地机制诊断，不替代正式图 pilot；它只说明“加大 probe 权重”当前没有局部正证据。

同规模的原始 G2 candidate-hardness 也做了 correct-Morgan、`absolute_bounded`、`lambda_molecule=0.25` 诊断。其每轮化学增量梯度 norm 仅为 0.0086–0.0204，而 clean gradient norm 为 0.2440–1.2209；前四轮没有 rank 变化，第五轮内折和外折结果与 G1 完全相同。逐查询核验表明 100 条外折查询的 `old_rank/new_rank` 数组逐项一致；两个最终 PEFT 参数向量 cosine=0.999989，相对 L2 差仅 0.460%。因此该结果不能归因为化学教师：当前 bounded hardness 在实际优化中近似复刻 G1，停止为它运行伪教师矩阵或扩大算力。正式 G2 pilot 代码仍保留为 fail-closed 的可复现实证入口，但在新证据出现前不再视为优先候选。

这一否定结论也符合 2026 年的目标错配警告：[Small molecule retrieval from tandem mass spectrometry: what are we optimizing for?](https://arxiv.org/abs/2602.16507) 指出，优化 fingerprint 相似性预测并不等价于优化候选分子检索，前者变好时后者甚至可能变差。它不能替代本项目的对照实验，但要求下一分支直接优化同候选集的排序错误，并用冻结结构只定义训练难度或 margin；任何收益都必须先胜过完全不看结构、但使用相同错误聚焦策略的 G1 对照。

该长 smoke 还暴露了 preservation 均值掩盖尾部漂移：G1/G2b 第 5 轮的全体均值仍约 0.9981，但最差单谱分别降至 0.9639/0.9646。训练器现同时报告 mean/min/q01/q05，并增加独立 `minimum_single_spectrum_preservation=0.95` 选模门；平均 cosine 门不再被解释为每张谱都受到保护。正式 G1/G2 共用这一新增合同字段，任一臂改变门值都会被 matched-capacity 审计拒绝。

真实 G2-PEFT smoke checkpoint 已经通过统一推理入口重新加载：loader 对账 official/raw checkpoint 哈希，只恢复 PEFT 增量，不加载 Morgan/MolFormer、SMILES 或候选；单张原始 HDF5 谱图输出为 1024 维、norm=1、全部有限。该检查确认结构教师严格停留在训练期。

### 2026-09-02 后续：真实 attention、margin 选模与容量平台

进一步代码审计确认历史 `chem_bias` 路径并未真正进入 attention logits。当前实现已在 softmax 前把化学 bias 加到每头注意力 logits，并加入形状、有限性和 mask 合同。真实官方模型审计表明：rule scale 为 0 时逐元素恢复官方输出；非零 scale 会改变输出；梯度能够到达 rule scale。该修复只证明干预路径真实，不把规则视图的收益预先当真。全谱 IDF precursor bias、metadata bonus、whole-molecule Morgan hardness、frozen probe、candidate margin、warm start 和 error-focus 等分支均未稳定超过 clean G1；匹配 peak-permuted 的规则证据只出现轻微 MRR 变化，没有 Recall@1 纠正增量。

本地 927 张可达真实谱、formula outer fold 0、seed 17 的完整训练诊断中，原始一层 rank-4 G1 在内层选出的 epoch 2 将 198-query 外层 Recall@1 提高 `0.50505 pp`。随后选模效用加入只依赖内层的连续 positive-vs-hardest-negative mean margin，同时保留 Recall、near、mean preservation 和 single-spectrum preservation 安全门；一层模型在 epoch 3 达到外层 `2 corrected / 0 introduced`，Recall@1 `+1.010101 pp`、MRR `+0.631313 pp`、near Recall@1 `+1.492537 pp`。该结果首次超过半个百分点，但仍是单折本地诊断，不是多个百分点，也不是正式外部结论。

更强 preservation（`lambda_preserve=20`）、12 轮慢学习率和最后两层 rank-4 PEFT 均未增加纠错数：三者外层都保持 `2 corrected / 0 introduced`、Recall@1 `+1.010101 pp`。两层模型共有 131,072 个可训练参数；best epoch 8 的 MRR 增益为 `+0.614478 pp`，低于最佳一层的 `+0.631313 pp`，但外层 mean margin 增量为 `+0.00222451`，是当前连续几何改善最大的配置。结论是容量与训练时长可以改善 margin，却尚未突破离散纠错平台；当前正从该安全两层 checkpoint 以更小学习率继续，仍由同一内层效用选择。

候选特异结构—峰证据也经过两道负审计。基于结构单键切割和精确质量匹配的内层统计中，truth 相对 DreaMS hard negative 的 union evidence 均值为 0.2748 对 0.0923，整体 `truth > hard` 为 62.1%；但在 21 个官方错误中只有 4 个（19.05%）满足该方向，不能无条件注入残余错误。MIST 启发的局部 peak-substructure 冻结 probe 在 508 个内层标注峰上的 macro AUPRC 为 0.3770，peak-permuted control 为 0.3691，差仅 0.00792；信号太弱，停止进入完整训练。

对“多正样本一致性”的数据前提进行独立审计后撤回了该表述。当前图 922 个 query 的直接正候选组全部只有 1 张参考谱；即使在全图按分子身份合并并排除 query 本身，也只有 16/922（1.735%）拥有至少两张参考谱。因此新辅助项只能称为“官方几何锚定的正配对相似度增量”，不能称为多正样本目标；`mean` 与 `worst` 在当前直接候选图上也完全等价。烟雾实验确认该项对可部署 PEFT 的独立梯度范数为 0.03489，约为 clean gradient 0.25623 的 13.6%，方向余弦 0.7785；但 16-query 一轮诊断没有 margin 改善。完整同协议实验运行到第 4 轮时仍只有 1 个修正且 margin 为负，已在化学库合同重审时主动停止；不得把辅助 loss 下降解释为性能证据。

### 2026-09-02 化学库合同重审

进一步核验并完成字段语义纠偏：MassSpecGym HDF5 共 231,104 条已标注谱，覆盖 28,929 个 IK14 和 34,259 个“IK14 × 加合物”组，其中 27,998 个组至少两谱；训练折有 129,261 条查询同时具备同身份参考和 strict-10ppm 异身份候选（尚未应用 P3 allow-list，也未去除完全相同峰表，只是本地上界）。`SIMULATION_CHALLENGE` 只表示是否进入 spectrum-simulation benchmark 子集，不是实验谱/模拟谱来源标签。旧构建器因误读该字段丢弃了 119,029 条合法谱，因此 `unified_v2` 只能作为历史回归工件；其一键一谱和元数据缺失问题仍然成立，但不得再用“真实谱/模拟谱”解释。

同时确认 3,151 条 MassBank 派生项全部 support=1 且默认启用：79 条所谓 `NL` 实为 `abs(precursor_mz-exact_mass)` 前体偏移，3,072 条 `CF` 来自各记录按 m/z 排序后的最低三峰，生成时不使用强度。它们只能作为观测模体做冲突/QC，禁止作为身份或机制碎片监督。完整证据和下一步合同见 `docs/CHEMAWARE_CHEMICAL_LIBRARY_REASSESSMENT_20260902.md`。

修正后的训练谱系也改变了 E1 的解释。P3-disjoint 的全部 train 身份并集为 19,403 个身份、137,830 条谱；此前按该字段拆成所谓 `real_train_primary` 与 `simulation_train_optional` 的 9,610 身份 allow-list 是语义错误，相关旧工件只能封存复现，不能作为新主张的数据入口。历史 E1 中包含 `True` 行本身并不构成污染；真正的问题是官方 DreaMS 已经用同结构正对和 0.05 Da 近质量负例做过端到端 triplet 微调，所以普通同身份 triplet 只能作为域内 continuation 控制，不能定义 ChemAware 的新增科学问题。

P2b 的共享向量可表示性审计也已完成：exact neutral-loss 集合相似度可被 PSD Gaussian set kernel 高精度近似，但 P2b 排序本身弱于 DreaMS，且在冻结 P3 near-core 上退化。因此瓶颈不是“无法塞进一个共享向量”，而是教师没有对准 DreaMS 的残余候选错误；P2b 不进入全局蒸馏。

修正后的 `unified_v3` 已全量构建：297,899 条代表谱、210,813 个完整 InChIKey、184,379 个 IK14。相对 v2 增加 3,324 个代表键和 2,221 个 IK14，且没有移除旧键；19,371 条最终 MassSpecGym 代表谱全部保留 fold、碰撞能和 benchmark 标记。该结果只修复部署 gallery，不恢复被一键一谱去重掉的重复条件，因此重复谱训练清单仍须直接从 HDF5 独立构建。

原 formal shared-v3 提交链现已默认 fail-closed：其 23,876-query 图由旧 `real_train_primary` 子集生成，可以复现为受限 cohort 诊断，但不能继续标记为全量 formal ChemAware。只有从修正后的 `train_primary_all` 重建候选图、重新生成规则/峰 token 缓存并通过新的哈希审计后，才允许恢复提交；此处不是因为 `True` 行“污染”了旧图，而是旧图的数据选择覆盖不完整且选择理由无效。

修正后的 identity-continuation 控制池已经按加合物独立冻结：`[M+H]+` 为 78,039 个 anchors、9,779 个 identities、2,938,393 条正边和 2,992,814 条 strict-10ppm 负边；`[M+Na]+` 为 5,580 / 600 / 143,583 / 145,871。所有 anchors 均来自 corrected P3-disjoint `train_primary_all`，并排除了容差量化后峰哈希相同的正谱。它们只定义 matched clean control，不构成 ChemAware 新贡献。

旧 observability cohort 进一步补做了真值盲谱学一致性审计。五个传统谱相似度的 3/5 多数在 formula-disjoint confirmation 上是 116 corrected / 165 introduced，净降 1.39 pp，故“传统相似度一致”不能直接当教师。只用可观察 top-2 分差、一致度和候选规模拟合的高精度适用域门，在 discovery formula-grouped OOF 后冻结阈值 0.75；confirmation 为 4/0。冻结全部参数与哈希后一次性消费此前未读 test，3,539 queries 上只触发 9 次，但为 9 corrected / 0 introduced，Recall@1 +0.254 pp，公式簇 bootstrap 区间 +0.086 至 +0.448 pp。九例峰哈希均不同；5 例有已知跨仪器正参考，2 例有明确不同碰撞能正参考。

这个信号极稀疏，而且门控本身需要候选参考谱，不能部署为共享单谱 embedding；100 个标签置乱门中仍有 1 个在 confirmation 得到更高风险净值。因此下一科学问题被收窄为：这种稀疏、高置信、候选判别的局部谱学信号能否在训练期压缩进 shared encoder，并在推理时不读取候选结构/候选谱，仍相对“相同稀疏度、相同错误聚焦但不看局部证据”的 clean control 稳定获益。已消费 internal test 不得再用于选择；在新图和新的外部 sealed panel 冻结前仍不启动训练。

对五种评分的依赖性复核又进一步收窄了措辞：discovery/confirmation/test 中约 77% 查询五者赢家完全一致，约 99% 最多两个不同赢家；错误查询上的 truth-vs-wrong 优势相关矩阵有效秩只有 1.56–1.60，所有冻结门触发案例更是五票一致。因此它们不能再被描述为“五种独立局部证据”，只是一类峰匹配信息的相关汇总。现阶段实验若成功，证明的是 conventional spectral evidence 可以被 shared embedding 更好保留，不足以单独证明 chemical awareness；项目名继续沿用，但化学主张必须等待结构—碎片监督相对 matched spectral control 的新增量。

### 正式训练前仍缺

本地工作站没有以下集群工件：

- `g8r_error_atlas_listwise_cache.npz`；
- `g8r_chemaware_g0_rule_cache.npz`；
- `g8r_chemaware_g0_full_audit.json`；
- full contextual peak-token cache；
- 未消费的最终 sealed panel。

已有 G0 JSON 证明旧远端 rule cache 覆盖 23,876 queries、25,275 spectra、3,472 identities、3,486 rules，但它绑定的是旧 `real_train_primary` 图，只能复现历史 restricted-cohort 诊断，不能代替修正后的训练输入。

## 8. 立即执行顺序

`tasks/submit_chemaware_shared_v3_formal.sh` 已主动 fail-closed，当前不得提交旧依赖链。恢复训练前的唯一顺序是：

1. 在集群从 corrected `train_primary_all` 按 `[M+H]+` 和 `[M+Na]+` 分层重建完整 strict-10ppm candidate graph，不以 `SIMULATION_CHALLENGE` 选择或命名 cohort；
2. 对新图重新生成官方 embedding、峰 token、G0 观测模体和所有派生缓存；旧图哈希及旧 preflight 不得复用；
3. 冻结两个 matched 训练臂：G1-sparse-control 与 G1+frozen-local-evidence。二者必须具有相同 query 数、错误聚焦强度、PEFT 容量、步数、采样、选模门和保护项，唯一差异是局部谱学证据；
4. 先证明化学增量梯度与 control 不同，并在未消费的内部开发折上通过 risk/near/preservation 门；
5. 只有多折多 seed 配对结果稳定，才在新的、尚未使用的外部数据集上做一次性 sealed evaluation。当前 observability test 已消费，不能再承担这一步；
6. 普通 identity triplet、P2b 全局蒸馏、Morgan whole-molecule 教师和旧 G2/G2b 矩阵均不进入首轮新实验。

## 9. 当前主张边界

目前只能主张：

> 已完成 ChemAware 数据语义纠偏、代表谱库 v3、P3-disjoint identity-continuation 控制池、旧训练谱系封存和局部谱学教师适用域审计。普通传统谱学多数全局弱于 DreaMS；但一个在 discovery 内交叉拟合并在 test 前冻结的高精度门，在 formula-disjoint confirmation/test 上分别得到 `4/0` 与 `9/0` corrected/introduced，test Recall@1 `+0.254 pp`。这只证明存在极稀疏、候选判别的局部谱学方向，不是共享 embedding 增益、不是外部验证，也不证明普适化学规律。旧 shared-v3 formal 图因字段误读导致覆盖不完整，所有新训练继续暂停。
