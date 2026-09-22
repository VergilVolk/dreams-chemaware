# ChemAware 第二阶段：残差化学 Triplet 课程

日期：2026-09-22

状态：`IMPLEMENTED / LOCAL_CONTRACT_PASS / GPU_RESULT_PENDING`

## 1. 已成立的起点

本路线不是从头重做。冻结起点是 2026-09-21 的 ChemAware native-triplet `best.ckpt`：

- role-3 Recall@1：`0.9046137895 -> 0.9227579057`；
- 增量：`+1.8144116122 pp`；
- formula-cluster bootstrap 95% CI：`[+0.8154763981,+2.8703289304] pp`；
- corrected / introduced：`57 / 22`；
- 最佳点：global step `3,000`；
- 原生组件：DreaMS `ContrastiveSpectraDataset`、`ContrastiveHead`、cosine triplet-margin loss、Adam；
- ChemAware 唯一自定义部分：triplet 中错误候选分子的选择。

第二阶段不得把这项结果写成待验证设想，也不得更换已经证明有效的原生训练栈。

## 2. 第一阶段 Triplet 的剩余问题

第一阶段共有 7,688 个唯一 query-negative 分子事件、4,032 个 query、2,518 个公式。其不足不是规模太小，而是课程在 checkpoint 更新后变旧：

1. 负例是按 official embedding 的边界构造的；在 step 3,000 后，一部分已经成为零梯度容易负例。
2. 不同 query 可贡献不同数量的候选事件，训练权重不完全均衡。
3. correct 与三个 content-permuted null 的 role-2 pair Jaccard 仍为 `0.7247--0.7352`，普通困难负例和化学特异负例尚未严格拆开。
4. 旧 trainer 用 batch-level train loss 每 1,000 步选 best；这不是正式检索目标，且 best 与 last 的巨大分化证明不能继续长训。

## 3. 新方法：Checkpoint-Adaptive Two-Slot Curriculum

对每个训练 query `q`，先用第一阶段 `best.ckpt` 对候选分子重新评分：

```text
s(q,m) = max_{r in references(m)} cos(f(q), f(r))
s+(q)  = max_{m in true identities(q)} s(q,m)
v(q,m) = margin + s(q,m) - s+(q)
```

其中 `margin=0.1`，分子内参考谱聚合与正式检索完全相同。

每个 query 最多保留两个不同负分子：

1. **Adaptive-hard slot**：当前 checkpoint 下分数最高的错误候选；
2. **Chemical slot**：在距离当前最难错误候选不超过冻结 hardness window 的候选中，优先选择 correct-vs-three-null 严格特异候选，其次按重复化学臂之间的候选级差异排序。

若没有合格的不同化学候选，第二槽才由当前第二难错误候选补齐。绝不复制 query-negative pair 凑数量。

化学只决定负例课程，监督关系仍是：

```text
positive = 同一真实分子身份的其它谱图
negative = 不同真实分子身份的候选谱图
L = max(0, 0.1 - cos(q,p) + cos(q,n))
```

训练时仍由 DreaMS 原生 dataset 动态抽取一张 positive reference 和一张 negative reference。

## 4. 为什么不是只用 293 个严格化学动作

role-2 冻结的 correct-vs-null recipe 在 roles 0--1 上只产生：

- 293 个 candidate directions；
- 290 个 anchor queries；
- 239 个公式。

其 role-2 correct/null specificity ratio 为 `3.5`，说明特异性真实存在；但覆盖不足以稳定微调 116M 参数模型。第二阶段因此采用：

```text
全 query 的 checkpoint-adaptive hard 边界
+ 位于相同局部困难带中的化学候选
+ 小规模严格特异候选优先级
```

而不是在“高特异但极稀疏”和“高覆盖但被普通难例稀释”之间二选一。

## 5. 本地只读审计结果

本地用 official embedding cache 检查工程与统计合同，不把这些数字当作第二阶段性能：

| hardness window | events | queries | formulas | chemical events | strict events | margin-violating | role-2 overall max Jaccard | second-slot max Jaccard |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.30 | 6,792 | 4,032 | 2,518 | 2,505 | 291 | 15.95% | 0.8497 | `0.6669` |
| 0.50 | 6,792 | 4,032 | 2,518 | 2,702 | 292 | 15.40% | 0.8239 | `0.6162` |

在预先规定“至少保留最佳 active-boundary 比例的 95%”后，0.50 保留 `97.88%`，同时有更低 correct/null pair overlap，因此本地选择 0.50。整体 overlap 包含每个 query 按设计共享的 adaptive-hard 第一槽；扣除该共同槽后，0.50 的三个 second-slot Jaccard 为约 `0.6009/0.6009/0.6162`，每个 null 对应 `322--338` 个 correct-only pair。正式服务器运行必须基于第一阶段 `best.ckpt` 的新 cache 重做 0.30/0.50 选择，不能沿用本地答案。

所有本地池通过：

- positive identity edge 正确；
- negative identity edge 正确；
- query-negative pair 无重复；
- 每 query 最多两个事件；
- roles 0--1 / role 2 / role 3 公式互斥；
- role 4 不读取。

## 6. 冻结的训练与选择流程

1. 以第一阶段 `best.ckpt` 仅编码 roles 0--3 evidence 可达的唯一谱图行；role 4 谱图不编码；
2. 构造 hardness window 0.30 和 0.50 两个池；
3. role 2 在 active-boundary 保留门内选择 pair overlap 更低的池；
4. 从第一阶段 `best.ckpt` 初始化；
5. 保持昨天成功的 `lr=5e-6`、margin `0.1`、batch size `4`、全 backbone 可训练；
6. 只训练 1,000 optimizer steps，固定保存 250/500/750/1,000；
7. role 2 选择 checkpoint，不再按 train loss 选 best；
8. 选择门：相对第一阶段 Recall@1 严格增加，`corrected-2*introduced>0`，MRR/Recall@3/micro-AUC/macro-AUC 均不下降；
9. 若没有 checkpoint 通过，保留第一阶段模型并停止；
10. 若通过，三个 content-permuted null 从同一第一阶段 checkpoint、同一 seed、同一学习率训练完全相同的 selected steps；
11. role 3 最后一次比较 official、第一阶段、correct residual 和三个 null；role 4 不碰。

## 7. 运行入口

```bash
sbatch tasks/run_chemaware_residual_native_stage2.sbatch
```

约束：一个 GPU；不手动指定内存；所有 GPU Python 动作用 `srun`。

## 8. 当前允许和禁止的主张

允许：

> 第二阶段 residual triplet 课程已完成本地构造、身份边、覆盖、角色隔离和单 GPU 运行合同；其出发点是已成立的 `+1.8144 pp` 第一阶段共享 embedding。

禁止：

- 尚不能说第二阶段已经超过 `+1.8144 pp`；
- 尚不能说第一阶段 `+1.8144 pp` 已被完全归因于正确化学语义；
- role-3 仍是开发面板，不是新的 outer；
- 本地 official-cache window 审计不是第一阶段 checkpoint 下的正式 triplet 统计。

## 9. 方法依据与创新边界

DreaMS 原论文明确把短程 hard-example contrastive fine-tuning 用于增强相近分子结构的区分能力，并使用同身份 positive、近质量异身份 negative 与 margin 0.1。本路线继承这一训练定义。Deep metric learning 文献也指出，大量容易 triplet 产生接近零的梯度，因而需要随 embedding 状态更新的 hard-example mining。

因此，“checkpoint 后重新挖 hard negatives”本身不是新概念。本文可主张的算法贡献必须是其面向候选分子谱库检索的具体组合：

1. 分子内 max-reference、与部署检索同构的残差边界；
2. correct-vs-three-null 化学语义用于第二槽候选选择；
3. query 固定预算，避免候选数造成隐式权重；
4. role-2 冻结窗口与步数、matched semantic-null 归因；
5. 保持部署时完全 spectrum-only 的共享 embedding。

参考：

- DreaMS: https://www.nature.com/articles/s41587-025-02663-3
- Smart Mining for Deep Metric Learning: https://arxiv.org/abs/1704.01285
