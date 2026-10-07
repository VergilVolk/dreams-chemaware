# Gate 1 后统一算法重建：对抗性审计与开发合同

**日期：** 2026-10-07  
**性质：** Gate 1 否证后的路线裁决；不是结果报告，也不是性能承诺。  
**目标：** 把 Noise、ChemAware、P2b、WSE、ChemAware V2 及历史负结果组织成一个审稿人可理解、可归因、可证伪的统一算法，而不是继续做分数拼装。

---

## 0. 裁决先行

原“候选差分排他证据 + 多路权重 + HodgeRank”方案不能再作为论文核心。Gate 1 在预注册机会平衡后得到：

- 营救集方向正确率 `28.53% [24.08, 32.98]`；
- `corrected - 2 * introduced = 660 - 2 * 1120 = -1580`；
- 该失败不是权重未调好，而是核心方向在 Noise 排错候选上系统性指向错误方。

因此永久关闭：

1. 以候选排他峰计数为主证据的统一打分；
2. 为该证据补化学 null、漂移 null 后继续训练的路线；
3. 把 HodgeRank 当作方法学创新或挽救器。Hodge 投影最多是排序一致性工具，不能修复错误证据方向。

保留但降级：Noise、WSE、P2b 等已验证分数仍是性能基线；候选图和统一评测协议仍是基础设施。

---

## 1. 不能假装没有做过的近邻方案

### 1.1 局部 Fisher、白化子空间或“候选局部几何”不是新出口

仓库已经存在三个近义实验，结论分别封死了三种常见包装：

| 方案 | 已有读数 | 审计结论 |
|---|---:|---|
| observable tangent metric | 全体 `3 corrected / 23 introduced`，Recall@1 `-0.9766 pp`，CI 全负，risk utility `-43` | 局部切空间会伤害原本正确查询；不能换名重启 |
| boundary-aligned centered rule metric | 绝对 `+2.3328 pp`，但 prevalence-matched 权重置换达到 `+2.3847 pp` | 增益可被无化学语义的同分布权重复制；化学特异性失败 |
| whitened rule interaction kernel | 绝对 `+2.5402 pp`，但相对线性父模型仅 `+0.0518 pp`，CI 跨零 | 交互核没有超过线性父模型；不能声称非线性化学机制 |

所以不再提出“局部判别子空间”“白化化学度量”“Fisher 候选几何”等近义方案。

### 1.2 简单顺序训练已经有实现，但尚不是统一方法

`GLM_build_compose_axis_curriculum.py` 和 `run_GLM_compose_axis_curriculum.sbatch` 已实现从 ChemAware Phase-A 出发继续 Noise 课程，并设置 matched continuation control。它回答的是 **Chem→Noise 顺序续训是否增加性能**，但仍有四个结构缺陷：

1. 训练目标仍是静态 triplet，不直接对应 molecule-level candidate ranking；
2. Noise 与 ChemAware 不是同时出现的两个可归因因素；
3. 没有显式保护已外部确认的 Noise 能力；
4. P2b、WSE、V2 的局部证据没有进入统一学习或受控决策。

它应保留为顺序训练基线，而不是主方法。

### 1.3 task-vector/TIES 是必要基线，不是论文核心

服务器记录称七个 task-vector 臂已经生成，Noise 与 ChemAware 参数更新近正交（cosine 约 `0.037`）；本地尚无该 run bundle，内部保护门也尚待评价。该读数在工件回收前只能作为待复核状态。线性合并或 TIES 如果成功，可以成为低成本强基线；但模型合并本身不是本项目的新方法，且参数近正交不等于功能互补。

### 1.4 PCGrad、投影保护和 preservation loss 已经出现过

仓库已有 `direct_projected_guarded`、梯度投影和 embedding preservation。它们没有创造化学特异性，尤其 A2 action-to-embedding 筛选未找到超过匹配对照的正确化学动作。因此新路线不能写成“使用 PCGrad 解决冲突”；优化器只配做实现工具。

### 1.5 learned router 已被证据否决

无监督融合几乎无增益；清除泄漏后的监督 router 只有约 `+0.5--0.6 pp`，谱图特征通道没有增量。最终系统不得再把“学习何时选哪个专家”作为主要创新。

---

## 2. 仓库真正拥有的、可以统一的正资产

| 资产 | 最硬结果 | 在新系统中的角色 |
|---|---|---|
| Noise V1/T1-T3 encoder | GNPS identity `85.36→86.56`；formula `86.81→88.06`，两者均外部正向 | **唯一合理的统一训练起点**；提供条件变化鲁棒性 |
| ChemAware native encoder | role-3 `+1.8144 pp`；Phase-A role-2 `+2.1266 pp` | 提供近结构候选边界课程；尚需外部归因 |
| candidate-list listwise objective | 已有完整两臂代码和预注册，但尚未正式运行 | 把训练目标对齐 molecule-level 检索决策 |
| WSE | GNPS Top-1 强基线，identity `87.37%`、formula `88.27%` | 最强非学习谱学基线与回退专家 |
| P2b | sealed main `+1.07 pp`，但 near-core `-4.23 pp` | **有边界的局部谱学专家**；严禁全局启用 |
| ChemAware V2 reranker | role-3 dev `+3.9399 pp`、93/17 | 候选化学残差专家；必须先独立确认 |
| Noise RRF | official 几何探索正向，但 Stage-1/GNPS 确认失败 | 永久负对照，不进入部署分数 |
| BioAware B44--B46 | 外部反转 / null / harmful | 不进入统一谱图算法；只作为为什么不能加入静态生物先验的证据 |

“使用全部开发”不等于让每个模块都投票。正模块进入模型或受控残差层；失败模块进入消融和否决链。把负结果重新启用才是真正浪费此前开发。

---

## 3. 新的核心对象：条件轨道与候选边界的联合检索

### 3.1 一句话定义

> 用同一个共享谱图编码器，在**分子候选集合**这一真实决策单位上，同时保持同一分子跨实验条件的排序一致性，并扩大同分子式/近结构错误候选的决策边界；两类监督分别用等剂量 matched null 证明其信息增量。

这不是“Noise 分数 + ChemAware 分数”，而是把两条线放回同一个几何问题：

- **条件轨道（orbit）**：同一分子在仪器、碰撞能、离子形式和谱图质量变化下形成一组观测。它们的候选排序应保持一致；
- **候选边界（boundary）**：同分子式或近结构候选在谱图空间中过近。它们必须被化学 hard-negative 课程推开。

Noise 解决轨道内不变性，ChemAware 解决轨道间边界分辨率。二者从此不是两个独立项目，而是同一检索几何的两条轴。

### 3.2 决策对齐的分子级分数

对查询谱 `q`、候选分子 `c` 及其参考谱集合 `R(c)`：

```text
S_theta(q,c) = tau * [logsumexp_{r in R(c)} cos(z_theta(q),z_theta(r))/tau
                      - log |R(c)|]
```

该 score 是 reference-count-corrected smooth maximum，与真实评测的 molecule-max 决策对齐，同时避免参考谱越多天然占优。现有 ChemAware listwise 代码已经实现这一核心，不需要重写模型架构。

### 3.3 三个训练项，只有一个主模型

#### A. 基础候选检索项

```text
L_rank = CE( S_theta(q, candidates) / T, true_molecule )
```

它校正 triplet 训练与候选集合评测不一致的问题。

#### B. 条件轨道一致性项

对同一分子的两个真实条件视图 `q` 与 `q'`，不强迫两个 embedding 完全相同，而约束它们在**同一候选集合上的分布**一致：

```text
p_theta(c|q)  = softmax(S_theta(q,c)/T)
L_orbit       = symmetric_KL[p_theta(.|q), p_theta(.|q')]
                + hinge(true candidate remains above hard negatives)
```

这样保留可能有用的条件信息，同时直接保护最终排序。Noise 的多条件真实谱图和已验证 action ledger提供 `q,q'`；same-query、same-dose 的 registered control 构成等剂量空模型。

#### C. 化学边界项

对 ChemAware 冻结关系账本标出的错误候选，在 listwise logits 上加入候选特异 margin：

```text
L_boundary = CE( [S_theta(q,c) + delta_chem(q,c)] / T, true_molecule )
```

必须同时训练/评估一个候选数量、基础难度、reference multiplicity、训练剂量完全匹配的 candidate-rotated 或 structure-swapped null。只有 `real chemistry - matched null` 的配对 CI 下界大于 0，才允许说化学内容起作用。

### 3.4 不是固定损失相加，而是带保护约束的优化

主问题写为：

```text
min_theta  L_rank + lambda_O * L_orbit + lambda_C * L_boundary

subject to
  L_noise_replay(theta) <= L_noise_replay(theta_N) + epsilon_N
  L_global(theta)       <= L_global(theta_N)       + epsilon_G
  L_near(theta)         <= L_near(theta_N)         + epsilon_near
```

其中 `theta_N` 是外部确认过的 Noise V1。用 primal-dual/Lagrange 或 GEM 式约束实现均可；它们是优化工具，不是创新主张。关键是：**ChemAware 获得新分辨率时不得拿 Noise 已确认的跨条件能力和 near-core 安全换取。**

固定 `lambda` 的普通多任务损失、PCGrad、naive sequential、task-vector/TIES 都必须作为基线同场比较。

---

## 4. 为什么这个设计比“先后继续训练”更硬

1. **决策单位正确。** 直接优化候选分子集合，不再把一条 triplet 当成最终问题。
2. **两条开发线有共同数学对象。** Noise 约束同分子观测轨道，ChemAware扩大近结构候选边界。
3. **信息内容可归因。** 真实条件视图对 matched condition control；真实化学关系对 difficulty-matched candidate null。
4. **不是靠选最好顺序讲故事。** 用同一训练宇宙的 `2×2` 因子实验估计 Noise 主效应、Chem 主效应和交互项。
5. **已有强父模型受到约束。** 从 Noise V1 起步并保护其外部已确认能力，而不是从证据更弱的开发冠军起步。
6. **不要求新多模态架构。** 模型仍是现有 DreaMS shared encoder；只改变候选级训练组织、损失和约束。

---

## 5. 第一轮必须是同宇宙 `2×2` 因子实验

所有臂使用同一 Noise V1 起点、同一 query、候选集合、参考谱、总步数、batch 谱数、优化器和 checkpoint 时刻。候选难度排序、reference 截断和训练组构造全部由一个冻结的 Noise V1 几何产生，禁止每个臂在自己的中途几何上重新挖样本。

| 臂 | 条件轨道 | 化学边界 | 目的 |
|---|---|---|---|
| R | 无 | 无 | candidate-list objective 本身的增益 |
| O-null | matched control | 无 | 共同续训和剂量对照 |
| O-real | 真实多条件视图 | 无 | Noise 内容增量 |
| C-null | 无 | candidate-rotated matched null | hard-negative 难度/数量对照 |
| C-real | 无 | 真实 ChemAware margin | 化学内容增量 |
| OC | 真实多条件视图 | 真实 ChemAware margin | 完整统一模型 |

配对效应：

```text
Delta_orbit = O-real - O-null
Delta_chem  = C-real - C-null
Interaction = OC - O-real - C-real + R
```

`Interaction > 0` 是“1+1>2”的证据，但**不是统一模型成立的必要条件**。只要两个内容对照成立、OC 胜最强父模型且外部迁移，联合模型已经有价值；不得把统计噪声包装成“1+1>10”。

### 必须同场的非主方法基线

- official DreaMS；
- Noise V1；
- ChemAware released encoder / Phase-A；
- naive Noise→Chem；
- 已有 Chem→Noise compose-axis；
- ordinary mixed-loss multitask；
- PCGrad/GEM-style protection；
- linear task-vector 与 TIES；
- WSE（分数基线，不冒充 encoder）。

---

## 6. 第二轮：把重排器变成有边界的残差专家

统一 encoder 通过后，才评价后融合。不能把弱 router 再包装成主模型。

### 6.1 P2b

- 在 sealed main 上有 `+1.07 pp`，因此必须保留；
- 在 near-core 上 `-4.23 pp`，因此 gate 必须在读 truth 前由候选结构距离/预定义 near 条件决定；
- 用新的统一 encoder 重算 P2b-on-unified，重新冻结权重；不能假定 P2b-on-official 或 P2b-on-Noise 的系数可直接迁移；
- 若外部 safe regime 的配对 CI 不正，P2b只留作消融。

### 6.2 WSE

WSE 是强默认专家和回退分数。它不需要被“吸收”为神经网络才能证明统一；必须与 encoder 分开报告，防止把经典谱学优势写成模型增益。

### 6.3 ChemAware V2

V2 当前只有 post-outer development 资格。先在 GNPS identity/formula 或新外部谱库做一次冻结确认；未确认前不得进入最终分数。通过后，采用零初始化、有界 residual：

```text
score_final = S_unified
              + gate_P * beta_P * residual_P2b
              + gate_C * beta_C * residual_V2
```

`beta >= 0`、残差有绝对上界；证据缺失、near-core、冲突或校准外分布时严格回退 `S_unified`。

### 6.4 Noise RRF 与其他历史重排器

RRF 已外部确认失败，只作为“为什么不能无条件融合”的负对照。RAW/A0/A1 等开发探针进入补充消融，不进入生产分数，除非通过同一冻结外部门。

### 6.5 是否把专家蒸馏进 encoder

只在第二轮残差层外部成立后，才允许做 teacher compilation：把 P2b/V2 在交叉拟合训练折上稳定纠正的 candidate pair 编译成新的 listwise margin。必须有：

- teacher-real vs teacher-score-permuted matched arm；
- near-core 排除；
- 独立测试不读取 teacher truth；
- student 胜统一 encoder 且不低于显式 residual 系统。

这一步可能得到单 checkpoint 部署，但不是第一轮必需项。

---

## 7. 三位审稿人的最强攻击与必须预埋的回答

### 7.1 计算/机器学习审稿人

**攻击：** 这只是 continual learning、多任务损失或 GEM/PCGrad 的领域应用。  
**正确回答：** 不把优化器说成创新。方法贡献是 molecule-level retrieval 上的 orbit/boundary 因子化、两个 matched-null 因果对照和严格外部迁移；PCGrad/GEM/sequence/task-vector 均是同场基线。若 OC 不能胜这些基线，方法主张失败。

**攻击：** 六个臂和多个 checkpoint 造成 selection bias。  
**回答：** 训练折固定，role-2 只选一个 checkpoint 时刻，所有臂共享该时刻；role-3 一次确认；GNPS/MoNA 只运行已冻结模型。不得各臂各选最好点。

**攻击：** 数据重复与谱库泄漏。  
**回答：** 同一 query spectrum、InChIKey14、分子式、来源和结构簇的重叠分别审计；主区间按分子式簇 bootstrap；报告 identity-disjoint 与 formula-disjoint 两个外部面板。

### 7.2 质谱/化学审稿人

**攻击：** ChemAware 只是在选择更难负例，化学规则本身没有价值。  
**回答：** C-real 与 C-null 拥有完全相同 query、候选数量、基础分数难度、reference multiplicity、剂量和优化器；只改变 chemical relation 与 candidate 的对应。没有 real-null 正差就禁止“化学”表述。

**攻击：** 强行把跨条件谱图拉近会抹掉真实碎裂信息。  
**回答：** 约束候选分布一致，而非 embedding 完全相等；并保留条件分层、source/polarity 和 near-core 非劣门。

**攻击：** neutral loss / rule evidence 对近异构体会误导。  
**回答：** P2b 的 near-core 负结果显式决定 gate；V2 在外部确认前禁用；冲突时回退统一 encoder，而不是强制改 Top-1。

### 7.3 通用审稿人

**攻击：** 模块过多、名字太多，像工程拼盘。  
**回答：** 正文只保留三个对象：`condition-robust encoder`、`chemically specialized candidate training`、`bounded evidence resolver`。Noise、ChemAware、P2b、V2 均放 Methods/ablation，不在摘要堆名字。

**攻击：** 不同数据集百分点被相加。  
**回答：** 所有协同和增量只在同一 query 分母上计算；旧数字只用于资产资格，不进入新系统总增益。

---

## 8. 预注册门：什么值得跑，什么必须停止

### Gate 0：互补性与资产就绪

服务器只先完成两项：

1. ChemAware encoder 在两个冻结 GNPS 面板的 molecule scores；
2. task-vector 七臂内部保护门。

然后在同一 GNPS query 上报告 Noise/Chem 的错误交集、条件 oracle headroom、near/source/polarity 分层。若 ChemAware 不提供任何 Noise 独有纠正，或外部全面反向，则不进入联合训练；只保留 Noise + P2b/WSE 系统。

### Gate 1：2×2 因子内部确认

OC 必须同时满足：

- 胜 Noise V1，公式簇 CI 下界 `>0`；
- `Delta_orbit` 与 `Delta_chem` 分别 `>0` 且 CI 下界 `>0`；
- `corrected > 2 * introduced`；
- near-core 不劣于 Noise 超过预注册容忍度；
- role-3 一次确认通过。

任一内容对照失败，就删除对应损失，不用总性能掩盖归因失败。

### Gate 2：外部确认

冻结一个 OC checkpoint 后一次性跑：

- GNPS identity-disjoint 10,995；
- GNPS formula-disjoint 5,261；
- MoNA/新来源与 polarity/source 分层（资产就绪后）；
- 15+ 方法同候选图天梯。

最低发表资格：两个 GNPS 面板相对 Noise V1 均为正、至少一个 CI 下界 `>0`、另一面板通过严格非劣；更强主张要求两个 CI 均正。若只在内部 role 面板赢，统一模型降级为负/诊断结果。

### Gate 3：残差专家

- P2b 只在预定义 safe regime 上相对 OC 有正 CI；
- V2 必须先有独立确认，再测试 OC+V2；
- final system 必须胜 `max(OC, WSE, P2b-safe)`，而不是只胜 official；
- near-core、source、polarity 任何关键层显著伤害则回退。

### Gate 4：可靠性

输出 Top-1 的同时给 conformal candidate set 或已校准的 risk-coverage 曲线。分布漂移导致候选集膨胀是诚实结果；不得用一个全局置信阈值冒充跨库校准。

---

## 9. 统一评价表，不能少的指标

每个方法、每个面板统一报告：

- Recall@1/3/5/10/20/50；
- MRR；
- pooled pairwise AUROC、query-macro AUROC；
- corrected / introduced / `corrected - 2*introduced`；
- formula-cluster bootstrap CI 与普通 query bootstrap CI；
- candidate count、source、polarity、instrument、near-core 分层；
- risk-coverage / abstention；
- 与最强父模型、WSE、P2b-safe 的逐 query 配对差；
- 参数量、训练谱数、GPU 时长与推理成本。

主比较必须是最强父模型或最强可部署系统，而不是 official DreaMS。

---

## 10. 论文能诚实主张什么

若全部门通过，建议定位为：

> A condition-robust and chemically specialized MS/MS representation with bounded evidence resolution.

中文含义：面向实验条件变化且具候选化学分辨能力的统一 MS/MS 表示与受控证据解析。

可主张的贡献：

1. 在同一 shared encoder 中联合建模同分子条件轨道与近结构候选边界；
2. 用两个 matched-null 因子对照分别验证条件信息和化学信息的独立增量；
3. 在同一冻结候选图上完成 15+ 方法、跨库、跨身份/分子式、近结构与风险覆盖评价；
4. 用有边界的局部谱学/化学残差层提高性能，并在证据冲突时严格回退。

不能主张：

- 新的通用多任务优化理论；
- 从碎片规则推断了真实碎裂机制；
- BioAware 已进入统一算法；
- 旧模块百分点可相加；
- 没有外部结果前已经形成“新范式”。

---

## 11. 实现顺序与文件边界

### Phase 0：只回收现有结果，不再发明模块

1. 回收 `run 2349962` task-vector 七臂并运行内部保护门；
2. 生成 ChemAware GNPS scores 和错误互补账；
3. 对已有 Chem→Noise compose-axis 只做基线裁决，不改门重跑。

### Phase 1：统一训练最小实现

建议新增：

- `tasks/GLM_build_orbit_boundary_groups.py`：同一候选组内写入真实/空模型轨道视图和真实/旋转化学 margin；
- `tasks/GLM_orbit_boundary_loss.py`：listwise score、orbit distribution consistency、chemical margin、保护约束；
- `tasks/GLM_train_orbit_boundary_encoder.py`：同一 trainer 支持 R/O-null/O-real/C-null/C-real/OC；
- `tasks/GLM_select_shared_factorial_step.py`：只用 R 臂选择一个 checkpoint 时刻，并把同一时刻机械地应用到其余五臂；O/C 实臂和空模型臂均不得各自挑最佳点；
- `tasks/run_GLM_orbit_boundary_factorial.sbatch`：严格 GPU 资源、hash pins、六臂同剂量、一次 role-3 和外部冻结链；
- 对应纯逻辑单元测试、静态 sbatch 测试和 synthetic loss tests。

不要复制六套训练代码；臂差异只能来自两个布尔因子及固定的 payload arrays。

### Phase 2：候选解析器

只有 Phase 1 通过后，才实现 `OC`、`OC+P2b-safe`、`OC+V2`、`OC+P2b-safe+V2` 四行消融。失败专家自动回退为零残差。

---

## 12. 最终判断

此前方案差的根本原因不是“不够复杂”，而是把多个分数在决策末端拼接，却没有说明每个模块改变同一检索问题的哪一个可识别因素。新方案的核心不是再加模块，而是重写问题：

> **同一分子的实验条件变化形成轨道；相似候选形成边界。统一模型必须收紧前者、推开后者，并分别通过 matched null 证明信息内容。**

这个方案对得起已有开发，因为：

- Noise 的外部正结果成为受保护的父模型和轨道监督；
- ChemAware 的 hard-negative 与 listwise 资产成为候选边界监督；
- P2b/WSE/V2 被放到其证据等级允许的位置；
- RRF、BioAware、候选排他证据和白化几何的失败成为方法边界而不是被遗忘；
- 旧的顺序训练、PCGrad、task-vector 全部变成必须击败的强基线。

它仍然不是已经成立的成果。审稿人是否认可，只由 `2×2` 内容对照、外部冻结验证和 near-core 安全三个结果决定。没有这三项，任何“集大成”都是包装；三项都过，即使没有新优化定理，也足以形成一篇方法严谨、社区有用的统一算法论文。

---

## 13. 文献边界（用于防止错误新颖性主张）

- DreaMS 已包含 self-supervised spectrum representation 与 hard-negative contrastive fine-tuning；本项目不能把“用 hard negatives 微调 DreaMS”本身写成创新。
- MSAlign、CSU-MS2、SpecBridge 等已覆盖谱图—结构跨模态对齐；本项目当前不做新多模态模型。
- PCGrad、GEM 已覆盖梯度冲突与旧任务保护；优化约束只属于实现。
- conformal molecular retrieval 已覆盖候选集合可靠性；本项目可把它作为输出校准，而不能声称首次引入可靠候选集。
- MCheM、ModiFinder 等已证明局部化学/碎片证据可补充全谱相似性；本项目的新颖性必须来自双轴统一、matched-null 归因和外部证据，而不是“使用化学规则”。

