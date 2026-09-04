# ChemAware 化学先验 v2 与共享谱图 embedding 注入方案

**日期**：2026-09-03  
**状态**：机制设计与数据合同；不授权正式训练  
**部署边界**：query/reference 共用同一编码器；推理只输入单张 MS/MS 谱及其前体元数据；结构、候选和碎裂教师只存在于训练期

## 1. 当前裁决

ChemAware 的主要瓶颈不是 PEFT 容量，而是化学监督的语义和粒度。现有 335 项核心库把中性丢失、固定碎片质量、同位素间隔和氢重排混在一个平面列表中；3,151 项 MassBank 派生库又把单条记录的观测峰直接提升为规则。两者都缺少候选结构适用条件、可执行的键变化、碎裂父子关系和经过校准的不确定性。

因此 v2 不再把“一个质量值”定义为一条化学规则。新的最小语义单元是：

> 在明确的前体结构、加合物、极性和碰撞条件下，一个满足原子守恒、价态/电荷约束和碎裂路径约束的候选变换，对某个观测峰或中性丢失产生一个带不确定性的解释。

现有规则库只保留为历史兼容和 QC 输入，不再直接生成训练梯度。

## 2. 文献方法给出的共同约束

1. SIRIUS 的关键对象是带峰公式和父子关系的 fragmentation tree，而非无结构质量差列表；CSI:FingerID 再从碎裂树预测结构指纹。
2. MS-FINDER 的氢重排规则只在候选结构的组合断键、BDE、碎裂链接和质量误差共同约束下用于排序，不能把氢质量偏移广播为任意峰对规则。
3. MIST 把峰表示为子式，同时显式建模前体—峰差并预测分子子结构；其增益来自化学化的峰表示和辅助任务。
4. CFM-ID、ICEBERG 和 FraGNNet 都把候选结构上的碎裂图或碎片集合与发生概率/峰强度分开建模；“可生成”不等于“高概率发生”。
5. FLARE 的增量来自峰—原子/子结构的局部对齐，而不是只对全谱和整分子做全局向量对齐。
6. 已有光谱学研究表明谱库中的碎片离子结构注释可能出错，因此 v2 必须保存多解、来源和置信度，禁止把自动注释当作单一真值。

这些方法不能原样搬入 ChemAware：FLARE/JESTR 一类方法推理时读取候选分子；我们的最终产物必须仍是 spectrum-to-spectrum 的共享单谱 embedding。可取之处只能作为训练期教师或辅助标签。

## 3. v2 不是一个表，而是四层系统

### L0：物理硬约束

- 元素与同位素守恒；
- 前体、加合物、极性、电荷和电子态一致；
- 质量由分子式实时计算，浮点质量不是主键；
- 子式不能超过前体元素计数；
- 断键、环裂解、氢迁移和自由基变化必须显式记录；
- 仪器质量容差按 instrument/resolution 配置。

L0 只负责拒绝不可能解释，不直接给身份加分。

### L1：可执行碎裂变换

每条定义必须包含：前体结构适用域、SMARTS/局部键环境、键删除/形成、保留电荷的一侧、氢迁移范围、允许的路径深度、产物结构或子式约束。通用断键、环裂解和已知类特异变换均属于此层。

L1 的输出是候选碎片图，不是观测峰标签。

### L2：条件化概率模型

对 L1 枚举的事件学习：

`P(event, fragment, intensity | local_structure, adduct, polarity, collision_energy, instrument)`。

这里允许使用 ICEBERG/CFM-ID/FraGNNet 类模型，或在本地重复谱上训练轻量校准器。模型输出必须保留 top-k 解释和概率，不能硬选唯一碎片。

### L3：经验观测与晋级规则

MassBank/MoNA/GNPS 的单谱峰首先进入 observation 表。只有同时满足以下条件的事件才能晋级为经验先验：

- 支持来自多个唯一分子，而非多张重复谱；
- 报告唯一骨架、实验室/来源和仪器条件数量；
- 有明确的负支持分母；
- 在公式或骨架隔离的校准集上仍有效；
- 概率经过校准，并报告区间而非“medium/high”文字；
- 不与 L0 冲突。

现有 support=1 的 3,151 个 MassBank 项全部停留在 L3-observation，不允许 `enabled_by_default=true`。

## 4. 三类工件必须分离

### 4.1 `prior_definitions.json`

小而稳定、人工可审计的定义库。记录变换语义、适用域、证据摘要、版本和来源。它不包含每张谱的峰。

### 4.2 `fragment_evidence.parquet`

长表，一行是一条原始或自动证据：spectrum、molecule、peak、候选 fragment、路径、质量误差、条件、来源和原始置信度。MassBank 的记录级信息进入这里。

### 4.3 `candidate_peak_teacher.parquet`

训练期编译账本，一行是 `query × candidate × peak × annotation`：

- query/candidate/peak 标识；
- fragment formula、atom-subset hash、bond-edit/path hash；
- direct ion 或 neutral loss；
- mass error；
- teacher posterior；
- 同候选集解释数和候选排他性；
- teacher、规则库、源数据和 split 的哈希。

模型训练只读取编译账本，不在线运行碎片枚举器。这样既快，也能冻结监督语义。

## 5. 化学先验如何进入共享 embedding

### 5.1 不采用的方式

- 不把规则命中矩阵直接加到 self-attention；
- 不给所有匹配某质量差的峰统一加正权重；
- 不把候选结构拼进部署 encoder；
- 不用 whole-molecule Morgan 相似度替代峰级监督；
- 不把“规则选择过的样本 + identity-only loss”称为化学增益；
- 不训练一个最终仍需候选分子输入的 cross-modal 检索器。

### 5.2 先补回局部化学信息，再做两级蒸馏

冻结探针已经否定了一个关键前提：官方 DreaMS **最终层**峰 token 对精确子式的可读性没有胜过只使用原始 m/z、前体 m/z、中性丢失和强度的同容量基线；正确标签相对峰置乱的优势也接近零。因此，不能继续假设“在最终峰 token 上接一个化学头”就是高效注入点。

新的第一注入点是候选无关的原始谱图化学分支 `Cψ(s)`：它只读取部署时本来就有的峰 m/z、强度、前体 m/z、极性/加合物等元数据，显式编码 fragment m/z 和 neutral loss，并通过小型 set/Transformer encoder 产生峰状态 `c_i(s)` 和谱级状态 `c_0(s)`。结构和候选只生成训练标签，不进入该分支前向计算。

设冻结官方向量为 `e_0(s)`，零初始化的有界残差为 `Δψ(s)`，最终共享向量为：

`E(s) = normalize(e_0(s) + Δψ(s))`。

当前实现默认 96 维、2 层、4 头，部署参数 259,270（约 0.22% 的官方模型规模），初始化时逐元素严格复现官方 embedding。

训练期结构教师对峰 i 的多个解释给出后验 `q_i(f)`。

**峰级辅助目标**：

`L_peak = Σ_i w_i · CE(q_i(f), Pφ(f | c_i))`

或使用 fragment/subformula embedding 的 posterior-weighted InfoNCE。`w_i` 由质量误差、教师校准度、解释歧义和跨重复谱复现共同决定。训练期峰头 `Pφ` 可丢弃，但必须证明 `L_peak` 更新了可部署化学分支，而不只是训练头。

**候选级分布蒸馏**：

教师把同分子式候选集合转成软分布 `π_T(m | s)`；学生仍只通过 query/reference 谱 embedding 的 cosine 和同身份参考聚合得到 `π_θ(m | s)`：

`L_distill = KL(stopgrad(π_T) || π_θ)`。

教师分布必须来自峰级解释的聚合，而非整分子指纹相似度。候选只用于产生训练标签，不进入 `Eθ`。

### 5.3 总损失

`L = L_clean_listwise + λ_peak L_peak + λ_distill L_distill + λ_preserve L_preserve`

- `L_clean_listwise`：完整同分子式/近质量候选上的身份目标；
- `L_peak`：让峰 token 学到可迁移的局部化学语义；
- `L_distill`：把候选差异压进最终单谱向量；
- `L_preserve`：约束高置信原始正确谱和全局 embedding 漂移。

只做 `L_peak` 可能让训练头变好而最终向量不变；只做 `L_distill` 又容易让全局弱教师覆盖局部信息。两级目标缺一不可，但必须先分别做增量梯度和 formula-OOF 可学习性预检。

## 6. 最高效的微调路径

### Stage A：零 backbone 的可学习性检查（已执行，否定最终 token 注入）

冻结官方 DreaMS，只读取最终峰 token 和 spectrum embedding：

1. 训练一个很小的 peak head 预测 top-k fragment/subformula 后验；
2. 训练一个固定容量的 candidate teacher probe；
3. 与结构置乱、峰置乱、posterior 置乱和零特征对照比较；
4. 按 formula/scaffold OOF，不能按谱随机拆分。

结果显示：MAGMa 多解碎片指纹在最终峰 token 上只比峰置乱高约 0.009 AUPRC，且相对同分子式结构置换只高约 0.001；精确子式任务中，最终峰 token 的元素 AUPRC 约 0.499，低于原始质量特征约 0.526，正确标签相对峰置乱的 MAE 改善约 0.00025。故最终 token 化学头未过门，原定“最后一层 LoRA 优先”暂停。

### Stage B：原始谱图化学残差分支（当前首选）

1. 冻结官方 DreaMS，并缓存官方 1024 维向量；
2. 用 formula-OOF 的软子式/中性丢失标签预训练小型原始谱图分支；
3. 接入零初始化、有范数上界的 1024 维残差头；
4. 在完全相同的 query ledger 上比较 clean、正确化学标签、峰置乱标签和同分子式结构置换标签；
5. 只有正确化学臂显著胜过全部 matched controls，才允许把该支路称为化学增益。

这一阶段的好处是：绕开已经丢失局部组成信息的最终峰 token；训练时无需重算 117M backbone；部署仍然是单谱输入、query/reference 权重共享，额外参数仅约 26 万。

### Stage C：最后一层 rank-4 LoRA 与分支蒸馏

只有 Stage B 证明化学残差有效后，才使用仓库现有零初始化 PEFT：

- 冻结约 117M 官方参数；
- 只在最后一个 Transformer block 的 fused Q/K/V/O、FFN in/out 和官方 projection head 安装 rank-4 低秩增量；
- 可训练参数约 69,632，约占总参数 0.06%；
- 缓存前面所有冻结层的 token 输出，训练只重算最后一层和两个小头；
- query/reference 严格共享权重；
- 保存包只含 PEFT 参数，不含结构教师或候选数据。

LoRA 不再承担“从已丢失信息中重新解码化学”的任务，而用于把已验证的原始谱图化学分支蒸馏回官方最后一层，或联合做小幅适配。若蒸馏成功，可以移除额外分支；若失败，保留已验证的小分支仍满足部署合同。

### Stage D：条件升级

只有 rank-4 一层在多折多 seed 上通过后，才依次尝试：

1. rank-8 一层；
2. rank-4 最后两层；
3. 低学习率短程解冻 projection head。

每次只改变一个容量因素。禁止通过延长训练或扩大参数来挽救不胜对照的化学监督。

## 7. 数据采样与防泄漏

- 使用修正后的 P3-disjoint pool；
- identity 等权，避免高重复分子支配规则概率和 loss；
- batch 以同分子式候选组为核心，优先包含 DreaMS 低 margin 和错误查询；
- 同一 identity/formula/scaffold 不跨训练、开发和确认；
- learned fragmentation teacher 必须报告其预训练身份并从评价集合剔除重叠；
- 若使用 MassSpecGym 训练过的 ICEBERG 权重评价 MassSpecGym，不得称独立泛化；优先使用训练折内 cross-fit 教师或无训练的物理枚举器加 OOF 校准；
- 所有结构选择过的 query ledger 必须同时提供给 identity-only 控制，隔离 curriculum 收益。

## 8. 最小严格实验矩阵

固定相同 query ledger、candidate graph、PEFT 初值、容量、步数和选模规则：

1. `C0 clean`：identity-only full-listwise；
2. `C1 selected-clean`：使用相同化学筛选样本，但仍 identity-only；
3. `C2 peak-permuted`：相同 posterior、稀疏度和熵，峰位置在 query 内置乱；
4. `C3 structure-swapped`：同分子式候选结构映射置换；
5. `C4 correct-peak-only`：只加 `L_peak`；
6. `C5 correct-distill-only`：只加 `L_distill`；
7. `C6 full`：同时加两级目标。

只有 C6 稳定优于 C0、C1、C2、C3，且 C4/C5 的消融符合预期，才能把增益归因于化学先验。

## 9. 进入训练前的硬门

### 规则库门

- schema 100% 可解析；
- 质量全部由公式复算并通过容差；
- active 规则 100% 有结构适用域、极性/加合物域和来源；
- record-level observation 的默认启用率必须为 0；
- empirical prior 报告 unique molecule/scaffold/source 支持及负分母；
- 所有自动峰注释保留多解和概率。

### 信号门

- formula-OOF peak label probe 显著优于所有 matched controls；
- 在原始 DreaMS error/low-margin 子集上存在独立纠错 headroom；
- 正确教师相对置乱教师的 cluster-bootstrap 下界大于 0；
- 化学增量梯度到达最后一层 LoRA，而不只停在可丢弃 head；
- 高置信原始正确查询的 introduced error 受预注册风险预算约束。

未通过这些门之前，`formal_training_authorized=false`。

## 10. 下一步实施顺序

1. 创建 v2 JSON schema 和最小示例，不改旧消费者；
2. 将 335/3,151 项迁移成 `legacy_observation`，不自动晋级；
3. 用 MAGMa 生成多解候选碎片图，加入路径/氢迁移/电荷字段；
4. 用重复谱学习事件概率和条件稳定性，先做 formula-OOF 校准；
5. 生成稀疏 candidate-peak teacher ledger；
6. 运行冻结 DreaMS peak-token 可学习性审计；
7. 只有通过后才启动最后一层 rank-4 PEFT 的七臂小矩阵。

## 参考实现与方法依据

- DreaMS: https://www.nature.com/articles/s41587-025-02663-3
- SIRIUS 4: https://www.nature.com/articles/s41592-019-0344-8
- MS-FINDER HR rules: https://pmc.ncbi.nlm.nih.gov/articles/PMC7063832/
- MIST: https://www.nature.com/articles/s42256-023-00708-3
- CFM-ID: https://academic.oup.com/nar/article/42/W1/W94/2437594
- ICEBERG/ms-pred: https://github.com/coleygroup/ms-pred
- FLARE: https://pmc.ncbi.nlm.nih.gov/articles/PMC12873900/
- Fragment annotation reliability: https://www.nature.com/articles/s42004-024-01112-7
