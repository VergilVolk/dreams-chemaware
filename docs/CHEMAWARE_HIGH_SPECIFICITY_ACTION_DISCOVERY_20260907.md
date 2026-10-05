# ChemAware 高特异性动作发现合同（B1–B6）

**日期**：2026-09-07  
**状态**：B1、B1J、B2 已实现但尚未读取新结果；B3、B4、B5 受前置证据门约束  
**边界**：本文只定义冻结动作及其证据门，不授权 embedding 微调、候选重排或正式外评

## 1. 这次修的不是超参数，而是动作的科学对象

第一代 ICEBERG differential action 有正向冻结效应，但它存在三个会破坏化学归因的结构问题：

1. `correct`、`candidate_swapped`、`peak_permuted` 各自按各自的 ICEBERG 分数重新选择 hard negative，三臂不一定作用于同一条官方 DreaMS 错误边界；
2. 每个 query 固定修改 top-3 峰，即使峰级证据非常弱也不弃权；
3. 候选交换与峰置乱控制只保证了大致操作形式，没有逐 query 保证相同的候选集合、boost/attenuation 数量和乘法剂量。

因此旧筛选可能证明“某种稀疏强度扰动有效”，却不能充分证明“正确候选的化学信息比等容量伪动作有效”。继续改学习率、epoch、PCGrad 或 gradient cap 不能修复这个识别问题。

## 2. B1：同边界、多负例共识动作（当前第一优先级）

实现：

- `tasks/chemaware_boundary_consensus_action_core.py`
- `tasks/audit_chemaware_boundary_consensus_action_atlas.py`
- `tasks/run_chemaware_boundary_consensus_action_atlas.sbatch`

### 2.1 候选边界

对 query `q`，先用冻结 official DreaMS 的真实 reference 聚合分数选取同分子式负候选：

```text
N_K(q) = top-K { s_official(q, c) : c != c+, formula(c) = formula(c+) }
c*     = argmax_{c in N_K(q)} s_official(q, c)
```

候选少于两个同分子式负例时直接弃权。ICEBERG 只负责给已经锁定的候选预测峰证据，不再负责改变检索边界。

### 2.2 峰级共识

对每个已观测峰 `i`，将各候选预测谱分别做 max-normalize 和平方根变换：

```text
d_ij = sqrt(p(c+, i)) - sqrt(p(c_j, i))
d_i* = sqrt(p(c+, i)) - sqrt(p(c*, i))
```

- support：`d_i* > 0`，且至少 75% 的 top-K 负例方向一致；
- conflict：`d_i* < 0`，且至少 75% 的 top-K 负例方向一致；
- 预测幅度低于 0.10、证据绝对值低于 0.05/0.10、实验峰强度低于 0.01时不动作；
- 前体及其 1.1 Da 邻域不参与动作；
- 没有合格峰时返回严格 clean duplicate，而不是强行凑 top-k。

动作族为：

1. `conflict_attenuate`：只衰减高置信 conflict 峰；
2. `support_boost`：只增强高置信 support 峰；
3. `bidirectional_sharpen`：同一 query 必须同时存在 support 与 conflict，成对增强/衰减；
4. `pair_logratio_sharpen`：比较真实结构与每个同式负候选在两个已观测峰之间的预测对数强度比；只有 official boundary 与负例分位数同向时才选择该峰对，并以 `exp(-δ/2)`、`exp(+δ/2)` 的对称对数剂量衰减/增强两个峰。

第四类不是逐峰分数的简单组合。若 `r_c(i,j)=log(p_c,i+ε)-log(p_c,j+ε)`，其证据是 `r_true(i,j)-r_negative(i,j)` 在 top-K 负例中的稳健同向部分；因此每个候选的全局预测强度缩放被严格消去。四类动作都只改变已观测峰的相对强度，不增加 m/z；base peak 永不编辑，而且 target 与反向剂量均不得把其他峰推过 base peak，因此执行前后的 max-normalization 标尺严格不变。

### 2.3 严格控制

每个 target query 同时生成：

- `candidate_role_reversed`：固定同一个 `c+ ∪ N_K(q)`，交换 true 与官方边界的角色；
- `intensity_rank_permuted`：在相近强度等级内置乱峰证据，打断正确 m/z—证据对应；
- `direction_reversed`：固定 target 的同一组峰位置，把每个乘法因子换成其倒数；它与 target 具有完全相同的绝对对数剂量，只反转化学方向；
- `clean duplicate`：无动作基线直接由 official embedding 给出。

前两个伪化学控制逐 query 强制拥有与 target 完全相同的 attenuated 数、boosted 数和乘法因子；不能独立选 hard negative，也不能拥有更少的动作容量。方向控制则逐 query 固定完全相同的峰位置，并保持每个峰的绝对 log-dose 不变。

### 2.4 发现与确认门

- formula folds 0–1：比较 32 个预声明设置，其中 24 个逐峰设置、8 个峰对 log-ratio 设置；
- fold 2：只允许一次确认，且仅在 discovery 所有门都通过后运行；
- fold 3：保留给后续 shared-embedding development；
- fold 4：继续封存。

设置必须同时满足：

1. target 的全 query formula-cluster margin CI 下界大于 0；
2. target 减 candidate-role-reversed 的 CI 下界大于 0；
3. target 减 intensity-rank-permuted 的 CI 下界大于 0；
4. target 减同峰位 direction-reversed 的 CI 下界大于 0；
5. 覆盖至少 20 queries、10 formulas、所在折 5%；
6. Recall@1 不下降且 corrected 不少于 introduced。

报告同时包含 Recall@1/5/10/20/50、MRR、pairwise micro AUC、query-macro AUC、margin、corrected 和 introduced。深预检在加载 DreaMS 前，已用真实谱图构造 discovery 与 confirmation 上全部 13,088 个 setting-query 动作计划，并逐一核验对照容量；这仍只是可执行性证明，不是效果结果。通过 B1 只允许进入新的 A2 transfer audit，仍不直接授权训练。

### 2.5 B1J：化学证据 × 输入 Jacobian 交集动作

实现：

- `tasks/chemaware_jacobian_intersection_action_core.py`
- `tasks/audit_chemaware_jacobian_intersection_actions.py`
- `tasks/run_chemaware_jacobian_intersection_actions.sbatch`

B1 只回答“教师认为哪个峰方向有利”，没有回答 DreaMS 是否对该峰方向敏感。B1J 固定 official DreaMS 的真实 molecule 与同式 boundary molecule，并固定各自当前最高分的 reference spectrum，定义 clean query 的部署边界：

```text
m(x) = <E(x), r+> - <E(x), r*>
J_i  = ∂m(x) / ∂log(I_i)
```

只有峰级化学方向 `a_i = sign(d_i)` 与局部模型方向同向，即 `J_i × a_i × δ > 0`，该峰才可进入动作。排序效用是化学证据幅度与正的一阶边界增益的几何平均；任一项为零都不能由另一项补偿。动作采用对称 log-dose `exp(-δ)` 或 `exp(+δ)`，预声明 12 个设置，不扫描学习率、epoch 或网络结构。

此路线新增了比“等峰数”更严格的控制：candidate-role-reversed 与 intensity-rank-permuted 不仅匹配 boost/attenuate 数和乘法因子，还逐角色贪心匹配实验峰强度与 `|J_i|`；若任一控制缺少足量同向峰，则该 query 四臂共同弃权。direction-reversed 继续使用 target 的同一峰位和相反 log-dose，且其一阶增益必须与 target 严格反对称。

B1J 只计算 276 个 discovery query 的 clean-input backward；仅当 discovery 的 formula-cluster CI、覆盖和控制匹配门全部通过，才计算 fold 2。它不编码任何扰动谱、不更新权重，PASS 也只允许进入非线性冻结动作验证，不能直接作为 embedding 微调结论。

## 3. B2：实验域条件化经验规则（当前第二优先级）

实现：

- `tasks/mine_chemaware_domain_conditioned_action_rules.py`
- `tasks/run_chemaware_domain_conditioned_action_rules.sbatch`

旧规则矿工把 `[M+H]+` 与 `[M+Na]+`、Orbitrap 与 QTOF、已知与缺失碰撞能量全部混在一个身份均值中。这会把实验域偏差误当成结构—碎裂规律。

B2 的统计单元改为：

```text
identity × adduct × instrument_family × collision_energy_band
```

预声明能量带细化为 `<=10`、`10–20`、`20–30`、`30–40`、`40–60`、`>60`；仪器和加合物从不跨域池化。`missing CE` 与 `unknown instrument` 只进入覆盖审计，永远不能产生规则。每个合格域内仍采用同分子式、有/无父结构谓词的身份对照，folds 0–1 发现、fold 2 确认、folds 3–4 不看。主数据预检已经走通 83,619 张 query 谱，形成 28,222 个身份—实验域单元；这只证明输入与分层可执行，不是效应或性能结果。

B2 只编译 `support_boost_observed_match`：规则峰必须真实存在，且 positive association 通过最少 12 formulas、24 positive identities、formula-cluster CI、单侧 sign-flip 与全局 BH-FDR。负相关或峰缺失只保留为统计现象，明确禁止直接变成 conflict/attenuation 动作。

## 4. B3：FIORA 局部断键教师（高潜力，但当前不应盲接）

FIORA 的科学对象比 binned ICEBERG 更适合下一代 ChemAware 动作：它在键级局部邻域上预测断裂，显式输出带电碎片、中性丢失与多种氢迁移，并将 CE、前体模式和 instrument 作为协变量。论文与官方代码：

- <https://www.nature.com/articles/s41467-025-57422-4>
- <https://github.com/BAMeScience/fiora>

但官方开源 `FIORA OS v1.0.0` checkpoint 的已声明训练域只有 `instrument=[HCD]`，前体模式为 `[M+H]+/[M-H]-/[M]+/[M]-`；当前 MassSpecGym HDF5 给的是分析器类别 `Orbitrap/QTOF`，不是 fragmentation method，而且包含 `[M+Na]+`。把 `Orbitrap` 或 `QTOF` 手工改写成 `HCD` 会制造伪条件一致性，不能做。

B3 只有在以下三门通过后才实现正式 teacher ledger：

1. 明确哪些本地谱可由元数据证明采用 HCD；未知者不得填充；
2. 对 FIORA 的 MSnLib v7 训练身份与所有 ChemAware folds 做 InChIKey/formula 重叠审计；
3. 在完全相同的 official boundary 上，FIORA 与 ICEBERG 的峰级方向一致性显著超过 candidate-role-reversed 和 peak-permuted 控制。

因此 B3 当前是“高潜力、未授权”，不是为了追新模型立即安装并跑全库。

## 5. B4：规则 × 碎裂教师交集动作（等待 B1/B2 门）

若 B2 获得足够覆盖，下一项最强的高精度动作不是把规则与 ICEBERG loss 相加，而是取可举证交集：

```text
父结构谓词成立
AND query 实验域匹配
AND 规则峰真实可见
AND B1 true-vs-top-K-negative 峰方向为 support
AND official boundary negative 不满足该父结构谓词
```

只有同时满足五项才 boost；任一项缺失就弃权。其控制为同域 predicate-swapped、同剂量强度/mass 匹配峰和 clean duplicate。B4 预计覆盖低但因果归因最强，适合作为高精度 action seed；只有 B1/B2 的确认结果证明有足够 query/formula 覆盖后才值得编码，避免先写一个注定只有个位数样本的 trainer。

## 6. B5：受控多碰撞能轨迹动作（当前冻结）

文献提示，单个碎片随多个碰撞能变化的纵向轨迹可补足单张谱的横截面相似度（<https://pubmed.ncbi.nlm.nih.gov/40592760/>）。但当前 MassSpecGym HDF5 只有 identity、adduct、`Orbitrap/QTOF` 与标准化 collision-energy 数值，没有原始采集批次、具体 fragmentation method 或可验证的同一实验 CE series。虽然本地只读盘点显示 8,927 个 identity×adduct×instrument-family 组具有至少 10 个单位的 CE 跨度，这不能证明它们来自受控纵向采集。

因此当前禁止把这些跨来源重复谱拟合成“随 CE 单调增强/衰减”的化学动作。B5 只有在外部记录同时提供同一原始采集系列、统一 CE 标尺、相同 adduct/fragmentation method，并能做 source-disjoint 确认时才解冻；届时只允许编译可复现的 fragment trajectory，而不是对当前 CE 数字做排序后强行拟合。

## 7. 执行边界

推荐只提交一个整合入口；它在同一个单 GPU 作业中依次执行 B1/J、B1 和 B2，
并将三类产物写入同一 `run_${SLURM_JOB_ID}` 下的独立子目录：

```bash
sbatch tasks/run_chemaware_high_specificity_action_discovery.sbatch
```

三个单项 sbatch 仍保留，仅用于某一阶段失败后的隔离诊断；常规发现流程不再分别提交。

三者均固定 `gpu` partition、恰好一张 GPU、无 `--mem`/`--mem-per-cpu`，输出使用 `run_${SLURM_JOB_ID}` 防覆盖。B1、B1J 不更新权重；B2 不加载或更新 DreaMS。任何一个结果未过独立确认门，都不得直接转成 embedding 微调动作。
