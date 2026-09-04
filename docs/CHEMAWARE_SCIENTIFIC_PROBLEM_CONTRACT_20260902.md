# ChemAware 科学问题合同（内部冻结稿）

**日期**：2026-09-02  
**状态**：问题已收窄；数据入口修正完成；新训练仍暂停  
**边界**：只讨论 MS/MS 谱图共享表示与代谢物候选检索

## 1. 一句话科学问题

> 对于前体质量几乎相同、官方 DreaMS 容易混淆的候选分子，常规峰匹配相似度有时会高度一致地支持正确候选；这种被全局 DreaMS embedding 丢失的稀疏谱图判别信息，能否在训练期被压缩进 query/reference 共用的单谱 embedding，使部署时只输入 MS/MS 谱和前体信息，就能比同数据、同容量、同错误聚焦的普通 DreaMS continuation 更稳定地找回正确代谢物，同时不破坏近异构体和原有高置信正确结果？

这才是当前 ChemAware。它不是“再做一次同分子 triplet”，不是“把 3,486 个质量数塞进 attention”，也不是“用 Morgan 或传统谱相似度替代 DreaMS”。

## 2. 为什么这是一个独立问题

官方 DreaMS 已经使用同结构正谱和 0.05 Da 内异结构负谱做过端到端 triplet 微调。因此，下列任务只能是控制组：

- 同一 IK14 的不同谱拉近；
- 严格质量窗口内不同 IK14 的谱拉远；
- 在 MassSpecGym 上继续训练官方目标。

当前可被数据支持的新增变量是“传统峰匹配信息对候选边界的补充贡献”。内部 held-out 结果说明该信号存在但极稀疏：冻结门在 formula-disjoint confirmation/test 上分别只触发 4/9 次，却均为 corrected 且未 introduced。这个门本身读取候选参考谱，不能作为最终系统；它只提供待压缩的训练期信号。

这里必须主动降格命名：entropy、sqrt cosine、linear cosine、top-10 match 和 intensity coverage 都来自同一批峰匹配关系，不含分子结构、反应或子结构真值。它们在三套分割上的 truth-vs-wrong 优势相关矩阵有效秩仅为 1.56–1.60；冻结门触发的 34/4/9 个案例中五者全部同票。因此当前证据支持的是 **spectral-evidence-aware** 问题，不足以声称已经获得“chemical-aware”机制。`ChemAware` 只能暂作项目代号；只有未来引入可审计的结构—碎片关系并超过匹配谱学控制，才能恢复化学感知主张。

## 3. 输入合同

### 3.1 训练数据

- 数据母库：`MassSpecGym_MurckoHist_split.hdf5`，231,104 条已标注 MS/MS；
- 身份：IK14 connectivity identity；完整 InChIKey 另作立体层审计；
- P3-disjoint train：19,403 identities / 137,830 rows；
- `[M+H]+` 与 `[M+Na]+` 分层，不跨加合物采正负；
- query/positive：同 IK14、同加合物、峰哈希不同；
- hard negative：不同 IK14、同加合物、query-centered strict 10 ppm；
- `SIMULATION_CHALLENGE` 仅作为 benchmark membership 报告，不参与筛选；
- HDF5 缺少来源库 provenance，当前不得声称跨来源训练。

修正后的 identity-continuation 控制池规模：

| 加合物 | Eligible anchors | Identities | 正边 | 负边 |
|---|---:|---:|---:|---:|
| `[M+H]+` | 78,039 | 9,779 | 2,938,393 | 2,992,814 |
| `[M+Na]+` | 5,580 | 600 | 143,583 | 145,871 |

把两个控制池按候选分子分组后得到不含模型分数的完整问题清单：83,619 个 query、9,854 个 query 身份、6,220 个 query 分子式、392,229 个候选分子节点和 6,220,661 条 query–reference 谱边。其中 308,610 条为负候选分子边；75.08% 与真分子同分子式，78.06% 的 query 至少面对一个同分子式干扰物，33.71% 至少面对一个已有 MCES-near 标注的干扰物。这说明主问题确实集中在近质量与异构体候选判别，而不是无限制全库相似搜索。

数据严重不均衡：每个身份对应 1–507 个 eligible query（中位数 3），每个分子式对应 1–692 个 query（中位数 3）。训练需做 identity-equal 采样或权重；统计推断必须按 identity 等权并按 formula 聚类，禁止让少数高重复分子主导结果。

### 3.2 训练期谱图证据

对 query 与候选参考谱计算五个冻结、可复现的谱学评分：entropy similarity、sqrt cosine、linear cosine、top-10 peak match 和 intensity coverage。它们是同一峰匹配信息的不同汇总，不得称为五种独立机制。只有在这些相关评分高度一致、且 DreaMS 排序几何显示处于适用域时，才形成稀疏训练信号。

这些证据只能在训练期定义样本或约束；最终 checkpoint 不得携带候选结构、候选谱、SMILES、Morgan、规则表或教师模型。

### 3.3 部署输入

每张 query/reference 谱只允许输入：

- 清洗后的 MS/MS 峰对 `(m/z, intensity)`；
- precursor m/z；
- DreaMS 原本允许的基础谱图预处理信息。

query 和 reference 必须调用同一个 encoder，输出同一空间的 L2-normalized embedding。候选谱只离线编码一次，在线检索使用向量相似度和明确的前体/加合物过滤。

## 4. 输出与最终目标

输出不是分子名称文本，而是一个共享谱图 embedding 及其诱导的候选分子排序。对每个 query：

1. 在同加合物、strict-10ppm gallery 中检索参考谱；
2. 同一候选分子的多张参考谱按预注册规则聚合，默认取最大相似度；
3. 返回候选分子排名、分数、置信度和审计信息；
4. 只有在独立校准后才把 rank-1 转成可报告的注释等级。

科学目标是提高严格候选检索，而不是让某个辅助 loss 下降、让 embedding 更像结构 fingerprint，或让少量训练 pair 的 cosine 增大。

## 5. 必须同时存在的三个训练臂

### G1-all-error-control

使用完整修正候选图和普通身份标签，对所有预注册的 DreaMS 错误/低 margin 查询做 clean continuation。它回答“只要更集中地继续训练 DreaMS 是否已经足够”。

### G1-sparse-ledger-control

与实验臂使用完全相同的稀疏 query ledger、candidate graph、PEFT 容量、初始化、采样、训练步数、选模规则和 preservation 门，但训练目标只读取正确身份，不读取任何传统谱相似度数值。允许共享由谱学教师冻结出的 query ledger，是为了让“选了哪些样本”在两臂中完全相同；否则无法区分 curriculum 收益和谱学监督收益。

### G1+spectral-evidence

与 G1-sparse-ledger-control 使用同一批 query 和候选；唯一新增因素是冻结的、逐 query–candidate-reference 的峰匹配评分或排序约束。它必须提供超出 one-hot 身份标签的分级信息，产生不同于 sparse-ledger control 的非零增量梯度，并且 checkpoint 仍只包含共享 encoder 的可部署参数。若所谓实验臂只是用谱学门挑样本、随后仍训练普通身份 loss，它只能叫谱学 curriculum，不能证明局部证据被压缩进 embedding。

如果后两臂的 query ledger、候选 mask、初始化哈希或训练合同不同，实验不能归因于谱学分级监督。实验臂还必须同时超过 G1-all-error-control，才能排除普通错误聚焦 continuation 已经足够。

## 6. 如何界定“做对了”

### 6.1 评价单位与候选合同

- 评价单位：query spectrum；另做 identity-equal 与 formula-clustered 统计；
- 正确身份：IK14 相同；立体异构问题单列，不混入主指标；
- 候选：同加合物、strict 10 ppm、排除 query 自身；
- 分子聚合：候选 IK14 内参考谱分数取最大值；
- 并列：任何负候选分数大于等于正候选时，正例不算 rank-1。

### 6.2 数据隔离

- 训练、选模和评价至少按 identity 隔离；
- 主要泛化结论按 formula 隔离；
- scaffold 隔离作为更强压力测试；
- 已消费 P3 和 observability test 永不再用于调参或机制选择；
- 最终主张必须来自新的未消费外部谱库或公开独立数据集。

### 6.3 主要指标与安全指标

- 主指标：Recall@1；
- 次指标：MRR、pairwise ranking accuracy；
- 近邻安全：same-formula / near-core 分层 Recall@1；
- 配对变化：corrected、introduced、persistent wrong；
- 风险效用：`corrected - 2 × introduced`；
- 表示保护：全体与尾部的 official-embedding cosine preservation；
- 不确定性：按 formula 聚类 bootstrap，并报告配对置信区间。

### 6.4 成功条件

只有同时满足以下条件才算 ChemAware 变强：

1. 相对官方 DreaMS 的主要候选检索指标提高；
2. 相对 matched G1-sparse-ledger-control 仍有独立正增量，并且不弱于 G1-all-error-control；
3. 多折、多 seed 方向一致，不由单折少数样本决定；
4. near-core 不退化，introduced 受控，风险效用为正；
5. 新的 sealed external evaluation 复现方向；
6. 部署 checkpoint 不需要训练期教师或候选输入。

## 7. 当前证据与否证边界

当前已知：

- P2b/neutral-loss 可以近似为共享核，但其排序弱于 DreaMS且在 P3 near-core 退化，不能全局蒸馏；
- 五视图简单多数在 confirmation 净降 1.39 pp，不能直接替代 DreaMS；
- 五个评分高度依赖：三套分割约 77% 查询五者赢家完全一致，约 99% 最多两个不同赢家；truth-vs-DreaMS-wrong 优势相关矩阵的五维有效秩仅 1.56–1.60；
- 冻结高精度适用域门在 internal confirmation/test 上为 4/0 与 9/0 corrected/introduced，说明存在极稀疏方向；
- 冻结门触发的 discovery/confirmation/test 案例中五个评分均全票支持同一候选，所以现阶段真正待压缩的是一类强峰匹配信号，不是五类独立化学知识；
- 事后 feature ablation 中，仅用 DreaMS margin/候选规模的 confidence 模型在 confirmation 为 0/0；不含 DreaMS margin 的 raw-confidence 为 2/0，raw-margin-only 为 4/0。这支持峰匹配强度含有额外信息，但因消融是在 confirmation 已见后完成，只能用于设计下一实验，不能当作新的确认性结果；
- 18,810 张 observability 谱中有 97 个容差取整峰哈希跨 IK14，但没有任何同哈希异身份边进入实际评价候选；test 与 discovery/confirmation 的谱哈希重合为 0。discovery 与 confirmation 有 2 个跨公式同哈希，故 confirmation 隔离并非峰哈希层绝对隔离，但冻结门结果未被这些候选影响；
- 该门仍读取候选参考谱，尚未证明信号能压进共享 embedding；
- 100 个标签置乱门中仍有 1 个在 confirmation 获得更高风险净值，当前证据不能支持普适规律主张。
- 历史 observability cohort 的实际三份 manifest 行、身份和公式完全互斥且并集完整，但当前同名 cohort builder 已发生合同漂移；结果只能按冻结 manifest/pair table 复现，不能声称当前 builder 可字节重建。

因此当前应接受的零假设是：互相高度相关的传统峰匹配证据不能在 matched control 之外稳定改善共享 DreaMS embedding。后续实验的任务是尝试拒绝这个零假设；若多折或外部评价不成立，就停止该路线，而不是继续扩大模型或搜索 loss。即使拒绝该零假设，也只能先得出“谱图信息压缩有效”；化学机制仍需另设结构—碎片监督及其匹配控制。

## 8. 当前执行状态

- `unified_v3` 代表谱检索库：已完成；
- corrected P3-disjoint identity-continuation 控制池：已完成；
- 局部证据 discovery/confirmation/internal-test 审计：已完成，internal test 已消费；
- corrected full candidate graph 与依赖缓存：未完成；
- corrected metadata-only candidate manifest：已完成并审计；模型分数与训练缓存仍未构建；
- matched G1-all-error-control / G1-sparse-ledger-control / G1+spectral-evidence：未运行；
- 新 external sealed evaluation：未建立。

在最后三项完成前，不启动或恢复旧 formal shared-v3 提交链。
