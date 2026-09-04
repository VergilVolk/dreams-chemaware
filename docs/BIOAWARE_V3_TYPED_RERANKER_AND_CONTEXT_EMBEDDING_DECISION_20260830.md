# BioAware v3：类型化重排器与生物上下文 embedding 决策记录

日期：2026-08-30

## 1. 不再混淆的边界

1. **BioAware Track A 是 embedding 之后的候选组内重排器。** 它不改变 DreaMS 权重。
2. **BioAware Track B 才是 embedding 微调。** 生物网络只在训练期构造监督；推理时原始谱图必须独立输出新的通用 embedding。
3. **P2b 是独立的谱学候选专家。** 它可以成为最终系统的谱学基座，但不能冒充 BioAware 或 embedding 微调。
4. **反应相邻分子不是同分子正样本。** 禁止直接把反应邻居 embedding 拉近；这会破坏异构体与相邻代谢物分辨能力。

## 2. 最新代码复核后的方法学定位

MetDNA、KGMN、MetDNA3/MrnAnnoAlgo3 的共同核心并不是“给网络邻居加分”，而是：

- 用 MS/MS 相似性和峰相关构建数据层；
- 用反应网络约束候选传播；
- 递归传播时控制种子置信度；
- 显式处理一特征多候选和多特征一分子的冗余；
- 在冲突或证据不足时降低置信或停止传播。

BioAware 的方法创新不能声称发明了网络传播。可成立的方向是：

> 以 DreaMS 候选组为谱学基座，把反应超边完整度、方向、候选特异性、独立种子、数据层谱学一致性和冗余冲突编码为类型化上下文，由组内排序器学习何时使用、何时降权和何时弃权，并把每次排序改变还原为可审计证据路径。

## 3. 已发现并修复的关键工程错误

Rhea 中存在左右两侧非货币分子完全相同的转运/状态变化记录。Rhea 70859、70891 即属于这种“分子身份零净变化”反应。此前它们被错误地当作两个分子之间的代谢边，贡献了 MTBLS1905 两个纠正案例中的一个。

修复规则：

- 对每个反应分别构建左右侧非货币化合物的 `compound_id × stoichiometry` 签名；
- 两侧签名完全相同时，标记为 `identity_noop_reaction`；
- 在候选竞争计数和网络支持聚合之前删除这类路径；
- 输出 `excluded_identity_noop_path_count` 供审计。

修复后的事实：

- 117 个 Level-1 开发查询的 10 个种子轮换中，共排除 520 条零净变化路径；
- MTBLS1905 已暴露开发集的固定策略从 `+5.56pp（2/0）` 降为 `+2.78pp（1/0）`；
- 保留的纠正是核苷相关候选：Rhea 13233/30771 提供有效跨身份反应证据；
- 因为共底物没有全部观测到，该证据应连续降权，不能用“超边不完整即删除”的硬门。

## 4. 当前证据强度

### 4.1 固定 one-hop 策略

在 117 个 Level-1 查询、10 个冻结的 30/70 身份轮换中：

- 官方 DreaMS Recall@1：81.20%；
- dependency-corrected one-hop：4 修正、2 新增，净增益约 0.24pp（按 819 个轮换实例）；
- formula-cluster bootstrap CI 跨 0；
- 不能声称显著、不能声称 SOTA。

### 4.2 类型化候选账本

已构建：

- 117 个真实查询；
- 466 个候选；
- 88 个分子式；
- 3,262 个“种子轮换 × 候选”实例；
- 70 个类型化聚合特征；
- 真值身份只作标签，从不进入种子或上下文。

### 4.3 聚合后 listwise v3

严格 nested formula-OOF：

- Recall@1：81.20% → 82.05%，`+0.85pp`；
- 2 修正、1 新增；
- formula-cluster CI 跨 0；
- 风险加权净值 `corrected - 2 × introduced = 0`；
- 未过门。

原因：先把轮换证据求均值，再训练模型，丢失了“同一候选在不同种子上下文下的稳定性”。

### 4.4 轮换级 headroom

保留 819 个上下文候选组进行 formula-OOF 训练，最后聚合到 117 个查询。固定审计矩阵的最佳开发单元为：

- Recall@1 `+1.71pp`；
- 2 修正、0 新增；
- 这是从固定矩阵中事后挑出的 headroom，不是可部署结果；
- 必须由 nested formula-OOF v4 重新裁决。

## 5. Track A 最终架构

### 5.1 两层专家

第一层是大样本谱学专家：

\[
S_{spec}(q,c) = f(\text{DreaMS},\ \text{raw peak matching},\ \text{neutral loss},\ \text{peak token})
\]

它负责通用候选排序，训练样本来自大规模真实候选图。

第二层是小样本生物上下文残差：

\[
\Delta S_{bio}(q,c,O)=g(\text{typed reaction context},\ \text{data layer},\ \text{conflict})
\]

最终分数：

\[
S(q,c)=S_{spec}(q,c)+\alpha(q,c)\,\Delta S_{bio}(q,c,O)
\]

其中 `alpha` 是风险门；证据不足时严格退化为谱学专家。

### 5.2 类型化上下文

必须至少包含：

- 反应方向是否已知、支持或冲突；
- 反应超边源侧/目标侧完整度；
- 缺失共底物签名；
- 独立种子数和独立反应数；
- 候选特异性（同一路径支持多少竞争候选）；
- 反应大小、节点度、货币代谢物过滤；
- 数据层 MS2、峰相关、RT 和离子家族一致性；
- 1→N、N→1 冗余与冲突。

### 5.3 训练与验证

- 排序损失：LambdaRank/listwise，不再做全局二分类；
- 外层：truth formula 隔离；
- 内层：模型、融合强度和弃权门选择；
- 每个真实查询等权，种子轮换只作上下文增强；
- 新增错误惩罚至少为修正收益的 2 倍；
- 报告 corrected、introduced、自然错误 near 亚组、公式聚类 CI 和 McNemar；
- 冻结后才允许打开新的外部队列。

## 6. Track B：生物上下文微调 embedding

### 6.1 目标

输入原始谱图 `x`，共享编码器直接输出新表示：

\[
z=E_\theta(x)
\]

推理不需要样本网络、候选身份或疾病标签。BioAware 上下文只在训练期提供“哪一个困难候选排序关系可信”的教师信号。

### 6.2 禁止的做法

- 禁止把反应邻居当同分子正样本；
- 禁止让查询编码器和参考编码器使用不同权重；
- 禁止只训练 embedding 后面的 reranker 后称为 embedding 微调；
- 禁止把表型标签、组别差异或 P2b 分数写进通用 embedding 的推理输入；
- 禁止从已打开的 P3 调参。

### 6.3 推荐训练结构

在 DreaMS 最后两个 Transformer block 内加入零初始化 residual adapter，查询谱和参考谱共享同一编码器。训练目标：

\[
\mathcal L = \mathcal L_{clean-rank}
+ \lambda_{ctx}\mathcal L_{context-distill}
+ \lambda_{rel}\mathcal L_{typed-relation}
+ \lambda_{pres}\mathcal L_{preserve}
\]

其中：

- `clean-rank`：同分子跨条件正例与真实 near 异构体负例；
- `context-distill`：仅使用 Track A 在外层隔离、decoy 校准后可靠的候选 margin；
- `typed-relation`：预测反应类型/方向的辅助任务，不要求反应邻居 embedding 接近；
- `preserve`：约束新旧 embedding，保护原本正确查询。

教师只定义排序方向：

\[
\mathcal L_{context-distill}=w_q\,\operatorname{softplus}
\left(m+s_\theta(q,c^-)-s_\theta(q,c^+)\right)
\]

只有在 BioAware 的证据优于度保持、反应大小匹配和质量差匹配 decoy，且交叉拟合中 `corrected - 2×introduced > 0` 时，`w_q` 才非零。

### 6.4 进入 embedding 微调的硬门

Track A 必须先满足：

1. 外层 formula-OOF 风险净值为正；
2. 至少两个独立机制/队列方向一致；
3. degree-preserving 和 reaction-size matched decoy 下仍有优势；
4. 教师覆盖不少于 100 个身份，避免再次发生小教师无法迁移；
5. 每条教师关系可还原到具体种子、反应、候选和冲突信息。

若不满足，BioAware 可以保留为后验专家，但不得作为 shared embedding 教师。

## 7. 当前裁决

- BioAware v1 的“一步反应加分”创新不足，且存在真实图语义错误，已停止。
- 零净变化过滤和类型化上下文是必要修复，但本身不是 SOTA。
- 聚合 listwise v3 未过性能门。
- 轮换级 listwise 是当前最合理的 Track A 学习方式，正在由 nested formula-OOF v4 裁决。
- 在 v4 或后续两层专家获得可复制风险净增益前，不启动 BioAware shared-embedding 大训练；可以完成代码预检和数据契约，但不得浪费 GPU 做弱教师蒸馏。

## 8. 2026-08-30 冻结 V3 与内部 RPLC 验证更新

### 8.1 已修复的 context-adapter 工程错误

旧版 `BioContextAdapter` 在零初始化时使用 `torch.where` 直接返回候选原始
embedding，导致 adapter 分支梯度被完全切断。现已改为“数值上严格等于原表示、
反向传播走归一化残差分支”的 straight-through fallback，并增加零初始化梯度测试。
修复后模型确实产生约 `0.003` 的候选分数变化，证明“完全学不到”曾包含实现问题；
但 MetDNA3 formula-OOF 的平均 margin 变化仍为负，说明一跳 Rhea tensor 本身信息不足，
不能把后续失败继续归因于优化器。

### 8.2 一跳 Rhea 的信息瓶颈

在 819 个身份隔离上下文轮换中，正确候选有上下文的比例高于错误候选，但在 154 个
DreaMS 错误轮换上，正确候选相对最难错误候选的上下文优势为负；实际填充的关系类型
几乎只有 `complete_direction_unknown`。逐特征 truth-oracle 也只能挽救 5/154 个错误。
因此下一代 BioAware 不再把“是否有一步 Rhea 邻居”当作充分证据，而必须依赖候选特异的
完整 MS2 路径、预测反应、结构网络、规则与 RT 的类型化共识。

### 8.3 冻结 V3 共识路由器

当前冻结路由器由两个互补专家组成：

1. 深度 3 的完整 raw-MS2 路径专家；
2. 六类证据组内归一化的 rank-consensus 专家。

只接受单专家提议或双专家一致提议；冲突严格回退官方 DreaMS。已消耗的 117-query
开发集上，冻结 deploy recipe（不是 OOF 代理）精确重放为：

- Recall@1 `81.20% -> 85.47%`，`+4.27pp`；
- `5` 修正、`0` 新增；
- MRR `+2.68pp`；
- 冻结 deploy 专家出现 `1` 次冲突并按规则回退，最终结果不受破坏。

旧 artifact 中的 `expert_conflicts=0` 描述的是 OOF 专家预测，而不是全开发权重的 deploy
重放。该元数据边界现已显式记录；内部验证必须先通过 deploy replay 门。开发数据已被用于
构造权重和门控，因此即使其重放 bootstrap 下界为正，也仍只属于实现身份检查，不能作为
独立性能证据。

### 8.4 内部 RPLC 与外部测试边界

已只打开 NIST urine RPLC 内部清单，不打开外部 16-panel 排名结果：

- 764 条 Level-1 行，515 个身份；
- 16 个正/负离子 targeted-MS2 mzML；
- 当前 MassSpecGym 参考库可覆盖 203 行、156 个身份；
- 其中 138 个身份具有同加合物、10 ppm 内的真实候选竞争。

原始数据随后已完成逐文件校验和正式预检：16 个 mzML 共 `864,923,716` bytes，含
`83,017` 张可用 MS2；764/764 条 Level-1 行均能在发布的 `15 ppm / 25 s` 窗口内
匹配。排除一张谱可能对应多个真值身份后有 341 个身份；再要求当前参考库包含真值且
同加合物 `10 ppm` 内至少有两个候选，最终冻结为 95 个查询、79 个身份/分子式，候选
身份中位数为 4。该 n 低于理想确认规模，但与 91 身份的开发协议同量级；不得通过保留
真值歧义谱来扩充。内部结果必须同时报告效应量、纠正/新增、正负离子分层和功效边界。

内部 RPLC 管线只能重新构造样本数据层；禁止重新拟合 V3 权重、阈值和冲突规则。内部门为
`corrected > introduced`、风险净值为正、正负离子子面板均不退化、MRR 不退化。只有全部
通过，才允许一次性打开外部 16-panel；SOTA 判断只在外部预注册协议上作出。

### 8.5 embedding 微调的当前资格

V3 的 5 个开发纠正仍不足以满足 shared-embedding 教师覆盖至少 100 个身份的硬门。
因此当前可以实现和测试两类 adapter 的执行契约，但不能启动声称可泛化的 BioAware
shared-embedding 大训练：

- **B1 通用共享 embedding**：生物上下文仅在训练期给出可信候选 margin；推理只输入原始谱图；
- **B2 样本上下文候选 embedding**：推理允许同一样本的已观测代谢网络，但无上下文时必须逐位回退原 embedding。

无论 B1/B2，都禁止把反应相邻的不同代谢物当作同分子正样本。只有内部 RPLC 和外部
多面板证明教师的风险净增益及身份覆盖后，才允许将其排序方向蒸馏进共享 encoder。

## 9. 内部 RPLC 冻结复现与外部 16 面板执行更新

### 9.1 内部 RPLC 是正向保底，不是 SOTA 结论

冻结 V3 在 95 个内部 RPLC 查询上一次性复现为：

- 官方 DreaMS Recall@1 `82.11%`；
- 冻结 V3 Recall@1 `85.26%`，提升 `+3.16pp`；
- `5` 修正、`2` 新增，风险净值 `corrected - 2×introduced = 1`；
- MRR 提升约 `+1.93pp`；
- 公式簇 bootstrap CI 跨 0，原因是 95-query 样本量不足。

8 次实际干预全部来自 rank-consensus 专家。5 个纠正主要由结构网络与 RT 的联合优势
驱动；2 个新增错误没有反应路径或结构网络优势，却被规则与 RT 误导。这说明 V3 已构成
可保留的候选重排器，但仍需要外部多面板与网络伪图来证明其生物网络贡献，而不能用内部
点估计声称 SOTA。

### 9.2 外部协议与统计单元

外部源冻结为 8 个样本/分离单元、正负离子共 16 个 panel：BV2 cell、mouse brain、
mouse liver、NIST plasma，各含 HILIC 和 RPLC。共 6,004 条 Level-1 行、1,240 个身份、
127 个 targeted-MS2 mzML。全部原始文件已按清单完成下载恢复、字节数和 SHA256 校验，
总计 `6,875,458,900` bytes。

外部评价遵循：

- 每个单元独立构建 MS1 数据层和候选图；
- MS1 噪声阈值固定为内部两套数据独立选出的 `100000`，外部不再选择；
- V3 artifact、权重、门控和冲突回退全部冻结；
- 单 panel 只报告方向，主推断使用 16 panel 合并后的公式簇 bootstrap；
- 要求 pooled CI 下界大于 0、MRR 不退化、风险净值为正，且至少 12/16 panel 不退化；
- 外部结果不得反向修改 V3。

### 9.3 度保持网络伪图不是分数乱序

已构建 20 套正式负对照图。对真实 eMRN 的 `minimum_step=0` 和 `minimum_step=1`
分别进行双边交换，逐节点度数和逐层边数保持完全一致，但生物配对被破坏：首个伪图的
真实边残留率分别约 `0.42%` 和 `0.017%`。每套伪图必须重新运行：候选路径、raw-MS2
边验证、深度专家、证据账本和冻结路由器；禁止直接打乱最终分数。只有真实 V3 的外部
增益同时超过伪图 95 百分位且经验 `p<=0.05`，网络特异性门才通过。

## 10. 生物上下文 embedding 的两种不同问题

### 10.1 B1：候选无关的通用共享 embedding

若推理时只允许输入一张干净谱图，样本网络不能决定同一谱图在不同样本中的不同候选。
因此 B1 只能学习可跨样本复现的化学关系结构：

- 同分子跨条件谱是唯一检索正例；
- near 异构体是真实负例；
- 反应类型、方向和缺失共底物由独立辅助头预测，不要求反应邻居互相接近；
- 只有在多个独立样本上下文中方向一致、且优于度保持伪图的候选 margin，才可作为
  shared encoder 的排序蒸馏信号。

这一路线输出新的通用 embedding，但不可能吸收只对某个样本成立的上下文选择。

### 10.2 B2：样本上下文候选 embedding

若推理时允许同一样本已观测代谢物作为输入，则应采用候选特异上下文 adapter：

\[
z_c^{ctx}=\operatorname{norm}(z_c+\alpha_c A(z_c,\{z_s,r_{sc},e_{sc}\}))
\]

其中 `r` 是类型化反应关系，`e` 包括路径置信、数据层 MS2 支持、反应完整度和冲突。
没有上下文或存在冲突时 `alpha=0`，逐位回退通用 embedding。该模型必须用候选组内
listwise 损失训练，同时对新增错误施加至少 2 倍代价，并报告无上下文回退、伪图对照、
公式/数据集隔离和候选级证据路径。

B2 才能真正表达“同一候选在不同生物样本中的上下文不同”，但它是 contextual embedding，
不能冒充候选无关的 DreaMS 权重升级。当前代码已实现零初始化、严格回退和 listwise 损失；
大训练仍受教师身份数至少 100 与外部伪图门约束。

## 11. 首个外部面板与 V4 高精度冻结

BV2cell-HILIC 是 V3 冻结后的第一个外部面板。该面板包含 102 个查询、
90 个分子身份和 86 个分子式。冻结 V3 在零重训条件下得到 3 个修正和
3 个新增错误，Recall@1 净变化为 0。这个结果否定了“V3 已经是外部
SOTA”的说法，也说明开发集的 `+4.27pp` 不能直接外推。

低融合优势的 rank 干预在三个已消耗集合中同时包含修正和新增，因此冻结
一个更窄的 V4：

- 完全复用 V3 的六类证据定义和非负权重，不重新拟合证据权重；
- 关闭 path-only depth-3 专家，只保留候选组内 rank-consensus 专家；
- 仅在官方 DreaMS Top1-Top2 margin 不超过 `0.05`、至少两类证据支持、
  fusion advantage 不低于 `0.10` 时干预；
- 任何条件不满足时逐位回退到官方 DreaMS 排序。

`0.10` 来自固定候选网格，并以“每个已消耗集合均零新增、lambda=2 风险
净收益为正”为硬约束。三组已消耗结果分别为：开发 `1/0`、内部 RPLC
`2/0`、BV2cell-HILIC `1/0`（修正/新增）。这些数字仅用于冻结 V4，
不构成验证结果；BV2cell-HILIC 从 V4 确认性统计中永久排除。

V4 的确认集是尚未用于门控选择的其余七个外部面板。通过条件同时要求：

1. 七面板合并的 formula-cluster bootstrap Recall@1 差值下界大于 0；
2. `corrected > introduced` 且 `corrected - 2*introduced > 0`；
3. 任一面板 Recall@1 不下降；
4. 真实反应图优于 20 个度保持重连伪图；
5. 所有结果来自同一冻结工件、同一查询/候选协议、零现场重训。

只有上述门全部通过，才可称为“高精度外部改进”；要称 SOTA，还必须与
MetDNA3/NetID 等在相同输入证据和候选协议下比较。

## 12. 最新公开实现带来的约束，而不是照搬

对 MetDNA3 (`MrnAnnoAlgo3`) 与 NetID 的代码审计确认：前者在数据层构建
知识约束的 feature network，计算实验 MS2 相似性并递归传播；后者把质量、
RT、MS2、同位素/加合物及生化转化统一为全局优化问题。BioAware 后续不能
退化成一步 Rhea 邻居加分，至少需要同时保留：

- 知识层候选路径和数据层真实峰/MS2 支持；
- 每种机制一票的类型化证据，避免密集规则重复计权；
- 候选组内全局一致性，而非逐候选独立二分类；
- 可审计的冲突与精确回退；
- 真实图对度保持伪图的增量检验。

我们的创新目标不是复现递归传播，而是把这些证据变成可学习、可门控、
候选特异的上下文表示，并证明它在严格外部协议上增加准确率而非仅增加覆盖率。

## 13. V3/V4 前两个未触碰外部面板的裁决

截至 2026-08-30，两个没有参与 V4 阈值选择的外部面板已完成零重训评价：

| 面板 | 官方 DreaMS Recall@1 | V3 变化 | V3 修正/新增 | V4 变化 | V4 修正/新增 |
|---|---:|---:|---:|---:|---:|
| BV2cell RPLC | 0.7945 | +0.0137 | 2 / 1 | 0.0000 | 0 / 0 |
| Mouse brain HILIC | 0.8012 | -0.0124 | 2 / 4 | 0.0000 | 1 / 1 |

这两项结果已经排除“V3/V4 当前可称 SOTA”：

1. V3 在不同生物基质之间方向不稳定，固定权重没有稳定校准证据可靠度；
2. V4 减少干预后降低了伤害，但两面板合并的净增益仍为零；
3. “V4 零新增”只在 BV2cell RPLC 成立，Mouse brain HILIC 已出现 1 个新增；
4. 禁止根据这两个面板重新调 V4，仍须完成全部七面板和度保持伪图。

## 14. V5：嵌套按研究留一，而不是继续调固定阈值

已新增 `develop_bioaware_v5_leave_study_out.py`。它回答的是固定权重失败后的核心问题：
证据可靠度能否跨研究学习。

- 外层按四个研究留一：BV2cell、Mouse brain、Mouse liver、NIST plasma；
- 被留出的研究不参与权重、门控或阈值；
- 其余三个研究内部再次按研究留一，生成门控选择数据；
- 候选组等权训练并维持同一官方 DreaMS query/candidate/tie 协议；
- 分开报告 seen-formula 与 unseen-formula；
- 只有 pooled formula-cluster CI 下界大于零、风险净值为正、四个外层研究均不退化，
  才认为类型化上下文具备跨研究可学习性。

V5 是跨研究 OOF 开发证据，不是新的盲测。若通过，仍需冻结工件并在一个完整新研究上
一次性验证。

## 15. 当前 B2 上下文 embedding 的负结果与扩展原则

现有小样本 B2 pilot 在 117-query 消费开发集上为 0 修正/0 新增。它证明了零初始化、
精确回退和身份隔离可以实现，但不证明上下文 embedding 有效。主要限制是：

- 只有 60 个 query 有上下文；
- 只有 258 个 rotation instance 含有效上下文；
- 平均候选分数改变量仅约 0.0031；
- 单一 HILIC 研究无法学习跨基质的证据可靠度。

因此下一版 B2 必须使用八个外部面板形成按研究留一训练。`g_c` 必须保留反应类型、
方向、独立种子数、路径完整度、真实 MS2/峰支持、冲突与 unknown，不能退化为 Rhea
一步邻居分数。其输出仍为有界候选上下文表示：

\[
z_c^{ctx}=\operatorname{norm}(z_c+\alpha_c\Delta(z_c,g_c)).
\]

没有上下文或冲突时精确回退到 `z_c`；Rhea 邻居永远不是同分子正例；P2b 永远不作为
embedding 教师。

## 16. V6：证据可辨识性门与协议修复

Mouse brain HILIC 的逐候选审计暴露了固定数值门无法解决的问题：一个正确修正拥有唯一、
强反应/结构网络支持；一个新增错误则是两个糖异构体具有几乎完全相同的反应网络和 SMN
支持，最终仅由极小 RT 差翻转。后者不是“阈值略松”，而是生物证据本身无法辨识候选。

V6 因此不再新增数值阈值，而加入一个无量纲的候选组内机制门：拟替换候选必须在至少一个
类型化机制中成为严格、唯一、非零的赢家。当前机制分为反应网络（known/predicted）与
结构网络（SMN）；RT、规则命中和 decoder 只能参与已有融合分数，不能单独激活干预。
不满足时逐位回退官方排序。该规则在五个已打开集合上的回放为：开发 `1/0`、内部 RPLC
`2/0`、BV2cell HILIC `1/0`、BV2cell RPLC `0/0`、Mouse brain HILIC `1/0`。这些仅是
回放，不是验证；后三个外部面板永久排除。V6 的确认集只剩 Mouse brain RPLC、Mouse
liver HILIC/RPLC、NIST plasma HILIC/RPLC 五个面板。

同时发现 V3/V4 评价器曾把严格并列基线覆盖为“按字典序展示的候选 ID 等于 truth 即正确”。
BV2cell HILIC 恰有 1 个这种 query，使旧报告的官方基线虚高约 0.98pp。正式 baseline builder
一直采用“任一负候选分数大于等于正候选即非 Top-1”的严格口径；V6 核心及 V3/V4 后续评价器
已统一到该口径，并新增全外部面板一致性审计。旧 V4 外部数字只能作为失效协议下的历史结果，
不得进入最终汇总。

## 17. 生物上下文 embedding 的正式双对照

共享 embedding 训练不再只跑一个混合模型。正式设计固定为两个除辅助损失外完全相同的
formula-OOF 对照：

1. `spectral_only`：同分子跨条件谱为正、near/质量匹配异分子为负，训练共享峰 token adapter；
2. `biology_relation`：在同一排序、安全和保持损失上增加 Rhea 类型关系辅助分类，反应邻居仍然
   是不同分子，绝不当正例。

两臂使用同一 5 折、同一 seed、同一学习率和训练步数。只有 biology 相对 spectral-only 的
paired formula-cluster CI 下界大于零、near 不下降且修正多于新增，才能说生物关系改善了
embedding；否则只能把谱学微调的结果归给谱学损失。P3 身份由已消耗 P3 审计按原 builder
定义重建为训练排除集，真实 train 身份 9,610、行 57,274、P3 重叠 0；本地不宣称与服务器
seal 字节一致，服务器正式运行必须使用原 seal。

## 18. V6 首两个确认面板：正方向尚不足以确认

截至当前，V6 的五个预注册确认面板中已完成两个：

| 面板 | query | 官方 Recall@1 | V6 Recall@1 | 修正/新增 | 净变化 |
|---|---:|---:|---:|---:|---:|
| Mouse brain RPLC | 97 | 0.8763 | 0.8763 | 0 / 0 | 0.0000 |
| Mouse liver HILIC | 146 | 0.8767 | 0.8836 | 2 / 1 | +0.00685 |

Mouse liver HILIC 最初由旧 V6 工件执行；该目录已封存为
`result_v6_invalid_artifact_v1`，并用 v2 工件重新评估。两次点估计相同，但正式入账的
工件 SHA256 为 `b5ec9700...f2421b`。Mouse liver HILIC 没有 strict-tie 异常。

两个面板合计只能说明 V6 当前没有出现净负方向，不能证明显著、不能证明图特异性、也不能
称 SOTA。必须完成余下 Mouse liver RPLC、NIST plasma HILIC/RPLC，以及 20 个逐面板
度保持重连图；主门仍是五面板 pooled 公式簇 CI、风险净值和真实图超过伪图 95 百分位。

## 19. V5 tie 漏洞及修复

对跨研究 V5 代码的运行前审计发现：旧 `apply_gate` 在无干预时用按字典序展示的候选 ID
重新计算正确性。若 truth 恰好在谱学并列中被显示为第一，它会把严格 baseline 的错误免费
变成正确。现已改为语义级精确回退：

- 无干预：`final_correct = baseline_correct`；
- 有干预：`final_correct = proposed_unique AND proposed_id == truth`；
- 并列一律计入错误。

新增回归测试专门构造“truth 在并列中按字典序第一”的样本，确认 abstention 不产生修正。
修复前的任何 V5 数字无效；正式 V5 只允许使用修复后的 nested leave-study-out 实现。

## 20. B1 生物归因再加一门

仅比较 `biology_relation` 与 `spectral_only` 的检索差值仍可能把随机正则化误写成生物机制。
正式 B1 因此额外保存双方分子式均未见的 relation readout：同一个训练期 relation head 分别
读取官方 embedding 与 adapter 后 embedding，报告总体 accuracy、macro-F1 和 reaction
precision/recall。该读出只作诊断、不参与选参，但生物关系归因要求：

1. biology arm 相对 spectral-only 的 paired formula-cluster CI 下界大于零；
2. near 不下降、修正多于新增；
3. adapter 后严格留出 relation accuracy 与 reaction recall 均不低于官方 embedding；
4. 所有 fold 都有足够严格留出反应对，预检下限为每折 10 对。

若 1–2 通过而 3 不通过，只能称“辅助任务正则化改善谱学检索”，不能称“模型学到了生物关系”。

## 21. Rhea 方向语义修正与保守补全

冻结的 Rhea participant cache 必须称为“反应邻接图”，不能称为生理有向图。Rhea 的
canonical left/right 序列化没有底物/产物的生物学方向含义，因此现有 B1 manifest 中
891 个 Rhea 对全部标为 `reaction_direction_unknown` 是正确的，而不是缺失值 bug。

现在新增一个独立、保守的 Reactome 方向缓存：只在同一 Rhea master reaction 的全部
Reactome cross-reference 对方向达成严格共识时，才标成 LR 或 RL；混合、undefined 和
未映射记录继续保持 unknown，原 participant side 从不改写。全库实测 17,656 个反应中
只有 619 个具备单向共识（537 LR、82 RL，3.51%），另有 40 个 LR/RL 双向支持。映射到
当前 B1 身份空间后，891 个反应对中只有 114 个具有任一有向支持，且只有 61 个具有单一、
不混合的方向语义。因此方向必须作为稀疏第三消融臂，不能把所有 Rhea 对重新标成有向，
更不能宣称组织、区室或疾病状态下的通量方向。

正式主对照仍是 `spectral_only` 对 undirected `biology_relation`。Reactome-consensus
direction 只有在逐折支持量通过预检后才允许单独加入。无论方向是否已知，反应邻居都不是
same-identity positive。
