# Noise × ChemAware 单编码器整合：负结果、模型合并调研与执行方案

日期：2026-10-09  
状态：当前参数层整合的权威研究说明。本文替代“继续扫描 Noise→ChemAware 顺序微调早期 checkpoint”的建议；该建议已撤回。

## 1. 结论先行

Noise V1 与 ChemAware Phase-A 不应再通过无约束的顺序全参数微调相连。作业 `2356060` 已经足以说明：从强 Noise V1 初始化后继续执行原 Phase-A 的 2,000-step 全 backbone 更新，会在只留下极小或不可确认 Top-1 波动的同时，系统性压缩候选间隔并降低 GNPS pooled AUROC。这个实验不进入论文主结果，也不值得继续在 MassSpecGym 上扫描训练步数。

参数层下一步采用成熟的单模型合并问题定义：

\[
\theta_0=\text{official DreaMS},\qquad
\theta_N=\text{Noise V1},\qquad
\theta_C=\text{ChemAware Phase-A}.
\]

主方法候选为 **Fisher-weighted model merging**；reference-faithful TIES 与简单 model soup 只作为必要强对照。合并后部署对象仍是一套 DreaMS encoder、一次前向和一个 1,024 维 embedding，不使用预测时“同意/不同意”规则，不使用双 encoder ensemble，也不在参数层引入候选路由。

开发与最终评测边界从本文起固定为：

- MassSpecGym：`development / consumed / leaked-for-selection`，禁止再用于模块选择、checkpoint 选择、早停、继续/停止判据或性能确认；
- GNPS Gold/Silver：`development / consumed`，只用于选择唯一合并算法及其开发参数；
- Enveda-180：`final / unopened-for-model-scores`，统一模型固定后只打开一次；
- 下游 WSE、P2b、ChemAware V2 等使用已有最佳实现，不因参数合并重新发明重排器。

## 2. 数据边界为什么必须立刻纠正

MassSpecGym 已被 Noise、ChemAware、P2b、重排器、共享 embedding 和多轮融合反复用于训练、筛选、错误分析与路线设计。formula role 2/3、fold-0、candidate graph 或新的分组名称都不能恢复独立性。今后 MassSpecGym 上的新数字最多是历史诊断，不能决定任何待发表组件。

GNPS Gold/Silver 同样已经被完整开发流程消费，因此也不是最终外测，但它仍可承担统一且透明的开发选择：

| 面板 | queries | candidate molecules | directed spectrum pairs | 角色 |
|---|---:|---:|---:|---|
| identity-disjoint | 10,995 | 87,518 | 175,171 | development / consumed |
| formula-disjoint | 5,261 | 24,401 | 47,724 | development / consumed |

Enveda-180 是当前唯一计划中的最终外测。任何合并系数、Fisher 归一化、TIES density、下游阈值或 open-set calibration 都不得在看到 Enveda 模型结果后修改。

## 3. 作业 2356060 到底做了什么

入口：`tasks/run_noise_v1_chemaware_phasea_native.sbatch`。输出目录：

```text
data/validation/noise_v1_chemaware_phasea_exact/run_2356060
```

它没有另写 ChemAware 模型，而是复用了原有：

- `build_chemaware_max_boundary_native_triplets.py`；
- `train_chemaware_specific_replay_native.py`；
- DreaMS `ContrastiveSpectraDataset`；
- DreaMS `ContrastiveHead`；
- cosine triplet margin loss；
- Adam、`lr=5e-6`、`batch_size=4`、margin `0.1`、seed `3407`；
- 全 backbone 从第一步可训练。

唯一核心干预是将初始化 checkpoint 从 official DreaMS 换成 Noise V1。训练池仍按 original official geometry 构造，报告明确为 `current_geometry_remine=false`。早先的 `official-error query 182 has no active max boundary` 来自试图按 Noise 当前几何重新挖掘原 Phase-A 边界；成功作业没有删除该 query，而是恢复“原 Phase-A triplets，只更换初始化”的正确单变量设计。

构造合同：

- 4,032 个训练 queries；
- 427 个 official-error queries 全覆盖；
- 5,956 个总 events，其中 1,327 个 error events；
- 1,024 个未修改 DreaMS replay events；
- 52,048 条 positive identity edges、49,221 条 negative identity edges；
- validation pool 未改变；
- formula role 4 未访问。

本实验固定使用原 Phase-A 赢家 endpoint `step-002000`，没有重做旧流程中每 500 step 到 3,000 step 的 role-2 checkpoint 搜索。因此它忠实复用了核心训练方法，但不是对旧选择过程的逐步骤复刻。

## 4. 作业 2356060 的结果与资格

以下 MassSpecGym 数字仅用于解释失败机制，状态一律为 `development / consumed / not for model selection / not for article claims`。

### 4.1 ChemAware role 2，1,975 queries

| 方法 | Recall@1 | MRR | mean positive margin | micro AUROC |
|---|---:|---:|---:|---:|
| official DreaMS | 89.6203% | 0.939888 | 0.342564 | 0.937444 |
| Noise V1 | **94.4810%** | **0.969549** | **0.475809** | **0.972289** |
| Noise→Chem step-2000 | 94.3797% | 0.969516 | 0.434116 | 0.969345 |

Noise→Chem 相对 Noise：Recall@1 `-0.1013 pp`（净少 2 个正确 query），MRR `-0.0032 pp`，margin `-4.1692 pp`，micro AUROC `-0.2944 pp`。Recall@3 增加 `+0.2025 pp`，但不足以抵消排序几何的整体收缩。

### 4.2 ChemAware role 3，1,929 queries

| 方法 | Recall@1 | MRR | mean positive margin | micro AUROC |
|---|---:|---:|---:|---:|
| official DreaMS | 90.4614% | 0.945870 | 0.332703 | 0.941701 |
| Noise V1 | **95.2825%** | **0.974457** | **0.469182** | **0.973611** |
| Noise→Chem step-2000 | 94.9196% | 0.972363 | 0.428457 | 0.969525 |

Noise→Chem 相对 Noise：Recall@1 `-0.3629 pp`（少 7 个正确 query），MRR `-0.2095 pp`，margin `-4.0725 pp`，micro AUROC `-0.4086 pp`。相对 official 的 corrected/introduced 从 Noise 的 `105/12` 变成 `106/20`：只多纠正 1 个，却多引入 8 个错误。

### 4.3 GNPS identity-disjoint，10,995 queries

状态：`development / consumed`。

| 方法 | Recall@1 | pooled pairwise AUROC |
|---|---:|---:|
| weighted spectral entropy | **87.3670%** | 0.940240 |
| P2b on Noise | 87.1305% | **0.941574** |
| Noise V1 | 86.5575% | 0.941279 |
| Noise→Chem step-2000 | 86.1028% | 0.934358 |
| official DreaMS | 85.3570% | 0.928697 |

Noise→Chem 相对 Noise：

- Recall@1 `-0.4548 pp`；
- corrected/introduced `127/177`，净少 50 个 Top-1；
- risk-net `corrected - 2*introduced = -227`；
- formula-cluster CI `[-0.9815,+0.0452] pp`；
- MRR `-0.3246 pp`，multiplicity-corrected CI 严格为负；
- margin `-4.7209 pp`，CI `[-5.0723,-4.3897] pp`；
- pooled AUROC `-0.6921 pp`。

### 4.4 GNPS formula-disjoint，5,261 queries

状态：`development / consumed`。

| 方法 | Recall@1 | pooled pairwise AUROC |
|---|---:|---:|
| weighted spectral entropy | **88.2722%** | 0.926509 |
| P2b on Noise | 88.2532% | 0.924834 |
| Noise→Chem step-2000 | 88.1201% | 0.922763 |
| Noise V1 | 88.0631% | **0.929902** |
| official DreaMS | 86.8086% | 0.903277 |

Noise→Chem 相对 Noise：

- Recall@1 `+0.0570 pp`，仅净增加 3 个 query；
- corrected/introduced `72/69`；
- formula-cluster CI `[-0.6685,+0.7829] pp`；
- risk-net `-66`；
- MRR `-0.0042 pp`；
- margin `-4.9142 pp`，CI `[-5.4274,-4.4126] pp`；
- pooled AUROC `-0.7139 pp`。

因此 formula 面板的三个净正确 query 不能称为新 embedding 提升。跨两个 GNPS 面板的一致信号是候选分离间隔和 pooled ranking quality 下降。

## 5. 失败机制的科学解释

原 ChemAware Phase-A 从 official DreaMS 出发，在 MassSpecGym role 2 得到 `+2.1266 pp`。但 Noise V1 在同一角色开始训练前已经相对 official 达到约 `+4.86 pp`，在 role 3 达到约 `+4.82 pp`。这说明 Noise 已经覆盖了大量 Phase-A 原本负责修复的边界。

从这个强起点继续执行原剂量的全参数化学 triplet 更新，新增有效监督很少，但每一步仍移动整个 117M 参数 backbone。role 2/3 中，新模型与 official embedding 的平均 cosine 分别从 Noise 的 `0.47539/0.47779` 降为 `0.43897/0.44251`；GNPS 两面板 margin 又同时下降约 4.7--4.9 pp。最符合证据的解释是 stability-plasticity 失衡，而不是“两个 embedding 不存在互补性”。

这也说明简单的继续训练、早停扫描或反向 Chem→Noise 都不是优先解。它们仍把一个目标当成后来者覆盖另一个目标，而且需要在已经泄漏的开发数据上不断选择训练剂量。

## 6. 成熟方法调研

### 6.1 Model Soup 与 WiSE-FT

Model Soups 发现，从共同预训练起点 fine-tune 的模型经常位于相连的低误差区域，权重平均可以在不增加推理成本的情况下改善泛化。WiSE-FT 进一步用预训练模型与 fine-tuned 模型的权重插值缓解分布外鲁棒性损失。

适用性：Noise 与 ChemAware 架构相同、祖先相同，满足基本条件。局限：普通平均假设每个参数更新同等可信，不解决两个任务向量的符号冲突和重要性差异。

参考：

- Wortsman et al. Model soups, ICML 2022: https://proceedings.mlr.press/v162/wortsman22a.html
- Wortsman et al. WiSE-FT, CVPR 2022: https://openaccess.thecvf.com/content/CVPR2022/html/Wortsman_Robust_Fine-Tuning_of_Zero-Shot_Models_CVPR_2022_paper.html

### 6.2 Task arithmetic 与 TIES-Merging

Task arithmetic 以共同基座为原点：

\[
\Delta_N=\theta_N-\theta_0,\qquad
\Delta_C=\theta_C-\theta_0,
\]

然后组合任务向量。TIES 针对普通相加的两个主要干扰源提出 trim-elect-merge：删除小幅冗余更新、按坐标决定主导符号、只合并与该符号一致的任务更新。

适用性：无需重新训练、无需访问评测真值，最终仍是一套 encoder。局限：TIES 只通过更新幅度和符号推断重要性；两个专家时，冲突坐标的处理可能近似为选择幅度较大的一方，未必等于保留真正的谱学功能。

参考：

- Ilharco et al. Editing Models with Task Arithmetic, ICLR 2023: https://openreview.net/pdf/0776550849b74d70586738db037bf1c9e9707c63.pdf
- Yadav et al. TIES-Merging, NeurIPS 2023: https://openreview.net/pdf?id=xtaX3WyCj1

### 6.3 Fisher-weighted model merging

Fisher merging 把每个 fine-tuned checkpoint 视为局部参数后验，用 Fisher 信息近似参数精度。对角近似下：

\[
\theta_{M,j}=
\frac{F_{N,j}\theta_{N,j}+F_{C,j}\theta_{C,j}}
{F_{N,j}+F_{C,j}+\epsilon}.
\]

这直接对应当前问题：一个参数若对 Noise 原目标敏感，就不应被 ChemAware 任意覆盖；若对 ChemAware 的困难化学边界更关键，就应保留 ChemAware 值。它利用两个已经完成的专项模型，不要求重新进行 2,000-step full-backbone 训练，推理成本仍为一个模型。

参考：

- Matena and Raffel. Merging Models with Fisher-Weighted Averaging, NeurIPS 2022: https://proceedings.neurips.cc/paper_files/paper/2022/hash/70c26937fbf3d4600b69a129031b66ec-Abstract-Conference.html

### 6.4 EWC、L2-SP、PCGrad 与 CAGrad 为什么不是第一选择

EWC 在学习新任务时用旧任务 Fisher 加权的二次惩罚保护重要参数；L2-SP 用较简单的起点正则限制 fine-tuning 漂移。二者都比无约束顺序训练合理，但仍需重新训练并选择正则强度。它们作为 Fisher merge 失败后的训练型备选，而不是当前第一步。

PCGrad 与 CAGrad在联合多任务训练中直接处理梯度冲突，但需要同时读取 Noise 与 ChemAware 两套训练流、统一 batch 和损失尺度并重新训练整套 encoder。现阶段已有两个成熟 checkpoint，先做 post-hoc merge 更直接、便宜且可解释。

参考：

- Kirkpatrick et al. Elastic Weight Consolidation, PNAS 2017: https://doi.org/10.1073/pnas.1611835114
- Li et al. L2-SP, ICML 2018: https://proceedings.mlr.press/v80/li18a.html
- Yu et al. PCGrad, NeurIPS 2020: https://proceedings.neurips.cc/paper_files/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html
- Liu et al. CAGrad, NeurIPS 2021: https://proceedings.neurips.cc/paper/2021/hash/9d27fdf2477ffbff837d73ef7ae23db9-Abstract.html

## 7. 现有 task-vector 工件为何不能直接当作答案

仓库已有：

- `tasks/merge_dreams_task_vectors.py`；
- `tasks/analyze_dreams_task_vector_interference.py`；
- `tasks/run_grand_fusion_task_vector_scan_2gpu.sbatch`；
- 服务器 run `2349962` 的七个 linear/TIES checkpoint；
- `tasks/run_GLM_taskvec_internal_gate.sbatch`；
- `tasks/run_tv_entity_recovery.sbatch`。

这些资产可以复用，但既有结果不能作为当前结论：

1. run `2349962` 的 ChemAware constituent 是 Stage-1 normalized checkpoint，不是目标中的最佳 Phase-A checkpoint；
2. `run_GLM_taskvec_internal_gate.sbatch` 使用 MassSpecGym corrected-graph fold-0 选择合并臂，当前已失去证据资格；
3. 当前 TIES 实现按单个 tensor 分别保留 top-density，而 reference TIES 通常先把 task vector 视为统一坐标集合再 trim；它应明确标成 local per-tensor variant，不能直接冒充 reference-faithful TIES；
4. `run_tv_entity_recovery.sbatch` 在同一 GNPS 实例上用真值遍历 FDR threshold并报告最优 coverage，calibration 与 evaluation 未隔离；它只能作开发诊断，不能提供独立 open-set 性能；
5. 服务器记录中的 task-vector global cosine 约 `0.037` 来自 Stage-1 组合，并不等于 Phase-A 与 Noise 的功能互补，也不能替代检索结果。

因此，正确做法不是重写整个 merge 框架，而是保留 checkpoint normalization、键/形状检查和部署转换；替换 constituent、实现 reference-faithful merge，并取消 MassSpecGym 内部门。

## 8. 选定的最小而完整实验

### 8.1 固定输入

| 角色 | checkpoint | 已知 SHA-256 | 资格 |
|---|---|---|---|
| common base | `data/e1/official_embedding_slim.pt` | `8928f908606c0bd652c5a4107d3c35102f660622958c225a1f625abe4b1ba245` | 共同祖先/基线 |
| Noise constituent | `noise_relation_t1_t3_run_2347055/checkpoint/primary_seed_3407_slim.pt` | `01ac8c7bfa75d89862e0b1eb2b5ffb297955cffae471fd483ad6c38d13f33beb` | 当前最佳共享 encoder |
| Chem constituent | protected Phase-A step-2000 | `a8428329ca1d12bfe735f3f8b848ed020a8db18d0a654d4f62cdbeb006e61135` | 最佳 ChemAware Phase-A 开发 embedding |

开始前必须确认三者 state-dict keys、shape、dtype、projection head 和 normalization 完全兼容。该检查是模型运算的必要条件，不是新增评测门。

### 8.2 只比较三个合并构造

1. **Model soup control**：`0.5*theta_N + 0.5*theta_C`；
2. **Reference TIES control**：全 task-vector trimming、elect sign、disjoint mean merge；
3. **Fisher merge primary**：在两套原训练流上分别估计 diagonal Fisher，再闭式合并。

不做七臂或几十个系数的开发扫描。Fisher 统计只读取已用于训练的 role 0/1 或 Noise 原训练样本与其原损失，不读取 MassSpecGym role 2/3、GNPS truth 或 Enveda truth。不同目标的 Fisher 在 tensor/layer 内归一化，并公开 epsilon、clipping 和归一化规则，防止仅因损失尺度让一个模型全局压倒另一个。

### 8.3 GNPS 开发选择

所有 constituent、merge 和下游对照必须在完全相同的 GNPS candidate graph 上编码和计分。唯一主选择量为两个面板 query-weighted Recall@1 的等权平均：

\[
S=\tfrac12(R1_{identity}+R1_{formula}).
\]

若主量完全相同，以两面板 pooled pairwise AUROC 的等权平均作为预先定义的 tie-break。MRR、Recall@2/3/5/10/20、macro/micro AUROC/AUPRC、margin、near subset 和 corrected/introduced 全部报告，但不在看结果后改写选择函数。

GNPS 仅选择一个最终 encoder。这里的数字一律标记 `development / consumed`，不写成外部确认。

### 8.4 Open-set annotation calibration

GNPS 上任何 coverage@5%FDR 必须使用互斥的 formula-cluster calibration/evaluation folds，或使用完整 cross-fitting：每个 query 的阈值只能由不含该 query 及其 formula cluster 的 folds 决定。禁止在同一批标签上遍历阈值并报告最佳 coverage。Enveda 使用 GNPS 冻结的阈值，不重新校准。

### 8.5 Enveda-180 最终一次评测

只有唯一 encoder、P2b/WSE/ChemAware V2 接法和 open-set threshold 全部由 GNPS 固定后，才编码 Enveda。最终表至少包括：

- Recall@1/2/3/5/10/20；
- MRR、mean/median rank；
- macro-query、micro-candidate、pooled pairwise AUROC/AUPRC；
- positive-vs-best-negative margin 与 signed Top1-Top2 gap；
- corrected/introduced/risk-net；
- near-structure subset；
- 使用 GNPS 固定阈值的 annotation coverage 与 empirical FDR；
- official、Noise V1、ChemAware Phase-A、WSE、P2b、model soup、TIES、Fisher merge 的同候选图比较。

每个数字都携带数据集、分母、数据角色、泄漏状态和 CI。Enveda 结果一旦打开，不允许重新选择 merge 或阈值。

## 9. 下游重排器的正确接法

参数合并只负责产生更好的共享 spectrum embedding。下游不重新实现：

- WSE 保持原始经典谱图相似度；
- P2b 使用已有最佳冻结定义，不把 raw sqrt cosine、entropy、neutral-loss 与 composite 重复计票；
- ChemAware V2 保持原有候选条件化残差语义，不改成全局 dense embedding score；
- RRF 继续作为 retired/audit-only；
- BioAware 只在存在真正事件级样本上下文时启动，不进入 GNPS/Enveda 普通谱库检索。

需要重算的只有那些数学上依赖新 embedding 相似度的输入通道；其余实现与参数原封不动复用。参数层不会用“专家同意/不同意”决定回退，也不会把多个候选分数硬加成所谓大一统。

## 10. 文章中的允许表述

如果 Fisher/TIES 在 GNPS 开发上胜出、并在 Enveda-180 一次性外测中稳定超过 Noise、ChemAware、WSE 与 P2b，可表述为：

> Two condition- and chemistry-specialized encoders sharing the same DreaMS ancestor were consolidated into a single spectrum encoder through parameter-importance-aware model merging, retaining single-pass inference while improving cross-condition molecular retrieval.

不能预先声称：

- 两个 embedding 已经指数叠加；
- MassSpecGym 的内部增益是外部确认；
- 参数向量近正交等于功能互补；
- model merging 本身是本论文原创算法；
- Enveda 打开前已经获得最终 SOTA。

真正的方法增量必须来自质谱场景特有的、可证伪的统一结果：条件稳健性、近结构边界、候选排序与 open-set 注释度在同一最终外测上同时改善，而不是把历史开发百分点相加。

## 11. 当前执行决定

1. 停止 MassSpecGym 模块评测与 Noise→ChemAware early-step 扫描；
2. 作业 `2356060` 保存为失败机制记录，不进入论文主结果；
3. 不再提交现有 Stage-1 七臂内部 MassSpecGym gate；
4. 修正并复用现有 model-merging 工具，输入换成 Noise V1 与 protected Phase-A；
5. 实现一个 reference TIES control 和一个 Fisher merge primary；
6. 仅在 GNPS 双开发面板上选择唯一 encoder；
7. 接回已有最佳下游实现后，一次性进入 Enveda-180。

这条路线把“两个 embedding 的优势如何进入同一个 encoder”转化为成熟、低成本、可证伪的参数合并问题，并彻底取消预测时的低级同意/回退逻辑。
