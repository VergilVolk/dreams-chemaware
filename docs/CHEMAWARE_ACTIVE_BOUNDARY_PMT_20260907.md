# ChemAware 当前成果、失败定位与主动边界配对迁移

**日期**：2026-09-07  
**当前裁决**：化学信息、冻结动作增益和 shared-embedding 可训练性都已经分别被证明；尚未被证明的是“正确化学动作相对等预算伪动作，能在完整留出候选图上产生严格正的 shared-embedding 增量”。本轮不把这个缺口表述成 0 pp，也不把训练损失下降表述成检索性能提升。

## 1. 已成立的成果按证据层级整理

1. **结构条件教师 headroom**：700-query 错误富集 panel 上，ICEBERG correct 为 Recall@1 0.5886，candidate-swapped 为 0.1329，peak-permuted 为 0.2986。它证明候选结构条件谱预测有判别信息，不是部署 embedding 结果。
2. **冻结 ChemAware 动作 E1**：旧 top-3 conflict attenuation 在 formula-disjoint confirmation 上为 3 corrected / 2 introduced，Recall@1 `+0.7519 pp`；相对两个置乱对照的 margin 聚类区间为正。它证明动作可纠错，但 2:1 风险效用为负，且不能直接外推 shared embedding。
3. **clean-visible 规则表示**：rule-mass kernel 在 1,929-query held inner 上为 `+1.2442 pp`、32/8，绝对区间以及相对 mass、mass-shifted control 的配对区间均为正。它是 shared embedding 的开发结果，但后续 576-query 新表示 screen 未复现同等稳定性。
4. **可靠的非化学训练骨架**：error-conditioned full-candidate shared adapter 在同一 1,929-query inner 上为 `+0.3629 pp`、9/2，formula-cluster CI 下界大于 0；另一个 matched Phase A 的共同 continuation 为 `+0.4666 pp`、12/3。新化学注入必须在这个骨架上证明额外增量。
5. **direct shared proof-of-effect**：旧 ICEBERG direct 在 180-query inner-all 上为 `+0.5556 pp`、1/0，在 49-query selected 子集上为 `+2.0408 pp`。它证明动作训练能够改变 clean embedding，但单臂、小样本、CI 下界为 0，并且 preservation/clip 门失败，不能称为化学特异结果。
6. **A2 失败机制已定位**：24-action 梯度审计中，correct/control 选择完全相同参考 pair 的比例为 candidate-swapped 16/24、peak-permuted 19/24。三条路由传递的主要是共同 identity 梯度，而不是动作差异；调 LR、epoch、PCGrad 或 gradient cap 都不能恢复已经丢掉的信息。

## 2. 本轮新增的真实动作成果

### 2.1 域内规则

严格限制在 `[M+H]+|Orbitrap|high_40_60` 后，确认一条规则：父谓词 `fr_NH0`、已观察 C2H6N fragment `m/z 44.0495 ± 0.005`。确认集含 29 个独立公式，effect 0.2478，95% CI `[0.0961, 0.4103]`，BH q=0.0440。

### 2.2 分层跨域 meta-action

为避免逐域切分损失统计功效，同时禁止把仪器/加合物/碰撞能组成差异误当化学效应，本轮采用：

1. 在每个 `formula × acquisition-domain` 内先比较有/无父结构谓词的 identity；
2. 在 formula 内等权合并各采集域效应；
3. 再令各 formula 等权；
4. discovery folds 0–1 选择，fold 2 一次确认，folds 3–4 不参与；
5. 要求至少两个严格正向采集域、至少 75% 可评价域非负、formula bootstrap CI 下界大于 0、sign-flip 与全局 BH-FDR 同时通过。

该方法从 14 个 discovery candidates 中确认一条 meta-action：仍为 `fr_NH0 → observed C2H6N fragment`，但在 4 个严格正向采集域复现。确认集为 44 个独立公式、67 个正 identity，effect 0.11157，95% CI `[0.04512, 0.18408]`，sign-flip p=0.000700，BH q=0.00980。`piperazine` 只有一个严格正向确认域，因此被正确拒绝。

这是一条比单域规则更稳健的化学动作，但仍只授权“增强已经观察到的峰支持”；不得新增 m/z，不得把峰缺失或负相关编译成 attenuation。

## 3. 本地直接注入实验给出的教训

第一版 rule-selected clean-boundary adapter 使用 86 个 control-hardness≤0.05 的三元组，训练 53、fold-3 评价 20，并跑了 3 个 seed。正确化学臂的训练 pair loss 从 0.002506 降到约 0.00092–0.00094，平均相对下降约 63%；但是：

- fold-3 R@1 保持 0.8000，R@5/10/20/50 均保持 1.0000；
- correct-control 的 R@1 与 MRR 差均为 0；
- correct-control margin 为 `-4.85e-5`，95% CI `[-0.001024, 0.000643]`；
- preservation cosine 约 0.99996–0.99997。

因此它证明的是“优化器能拟合所给三元组”，不是“化学知识迁移成功”。根因由新的主动边界预检直接证明：跨域 meta-action 虽形成 99 个 hardness-matched 三元组，但 62 个训练三元组中仅 2 个是当前错误、3 个位于 0.05 边界内；fold-3 的 21 个三元组中只有 1 个位于 0.05 内。训练间隔中位数为 0.5074。旧目标的大多数梯度在加固已经很容易的排序。

代码现已 fail-closed：主动训练至少要求 12 queries / 10 formulas，fold-3 至少要求 5 queries / 5 formulas；当前规则输出 `CHEMAWARE_RULE_ACTIVE_BOUNDARY_ACTION_COVERAGE_FAIL`，禁止继续训练。这不是性能失败后调参，而是在训练前取消一个统计上不具可评价性的实验。

## 4. 新方法：Boundary Paired-Margin Transfer（B-PMT）

### 4.1 监督量

对 query `i`，记 official clean full-candidate margin 为

```text
m_i^0 = s(q_i, c_i+) - max_{c != c_i+} s(q_i, c)
```

B1 在同一个 official 候选边界上产生正确动作 `C` 与三个等容量对照：候选角色置换 `R`、峰强度秩置换 `P`、同峰位反方向 `D`。定义严格化学优势：

```text
A_i = max(0, min(
    m_i^C - m_i^0,
    m_i^C - m_i^R,
    m_i^C - m_i^P,
    m_i^C - m_i^D
))
```

取最弱的一项，是为了禁止某一容易对照掩盖另一项失败。只有动作真实存在、`A_i>0` 且 `m_i^0≤0.05` 时才进入 corrective 集；其余样本的化学权重严格为 0。

### 4.2 写入 clean embedding 的目标

动作谱图不进入部署，也不做 embedding 蒸馏。化学优势只设定 clean margin 的增量目标：

```text
t_i(alpha) = m_i^0 + alpha * clip(A_i, 0, 0.05)
L_BPMT = sum_i w_i * softplus((t_i - m_i(theta)) / T) / sum_i w_i
```

其中 `alpha∈{0, 0.25, 0.50}`，`w_i` 为 formula-equal 权重乘以 capped corrective strength。正、负参考来自 official full-candidate 真实边界，并在 B-PMT 项中 detach；所以这个特权小样本项只推动 clean query 路径，不能靠移动高复用候选制造假提升。shared adapter 仍对所有 query/reference 使用同一函数，推理只输入 clean spectrum embedding。

对单个 active 样本，令 `z=(t-m)/T`，则：

```text
dL/dm = -sigmoid(z)/T < 0
```

忽略样本间梯度干扰时，一步梯度下降给出：

```text
Delta m ≈ eta * w_i * sigmoid(z)/T * ||grad_theta m_i||^2 / sum_j w_j > 0
```

这说明新项对目标 clean margin 的一阶方向严格正确。它没有证明跨样本泛化；泛化必须由 fold-3 的配对四臂结果判断。

### 4.3 四臂因果比较

四臂共享初始化、500 个 optimizer steps、全候选 clean listwise、query order、候选采样、preservation、margin floor 和随机种子：

- `clean_duplicate`：同一 active membership，`alpha=0`；
- `matched_formula_deranged`：同一 membership、同一优势剂量多重集、按 official hardness 匹配，但剂量来源 formula 必须不同；
- `alpha025`：正确化学优势的 25% 继承；
- `alpha050`：正确化学优势的 50% 继承。

对照是独立训练臂，不做 `g_correct-g_control`，也不给任何负权重。所有动作在回收采样前至少曝光一次。

### 4.4 晋级门

`alpha025/alpha050` 必须同时严格胜过 clean duplicate 和 matched deranged：

1. formula-cluster R@1 差的 95% CI 下界大于 0；
2. corrected > introduced；
3. R@5/10/20/50、MRR、macro AUC、micro AUC 全部不下降；
4. mean preservation ≥0.995；
5. fold 4 继续封存。

任何一点失败，都只否决当前 B-PMT 表达，不撤销 E1 动作或 rule-mass 开发结果。

## 5. 可量化的预期，而不是承诺

目前没有证据支持把 ChemAware shared-embedding 增量承诺为 3–5 pp。已有可复核量级是：非化学训练骨架 `+0.3629` 至 `+0.4666 pp`，旧冻结化学动作 E1 `+0.7519 pp`，rule-mass clean-visible 开发上限 `+1.2442 pp`，旧 direct shared 全体 proof-of-effect `+0.5556 pp`。

如果在完全匹配的同一 cohort 上，B-PMT 能把 E1 特异动作收益的 20%–50% 写入 clean embedding，则化学增量的规划区间约为 `0.15–0.38 pp`；它必须是相对 continuation 和 matched control 的额外增量，不能把来自不同 cohort 的历史数字直接相加。若 B1 新动作在同边界、多负例和四对照门下获得更大的确认效应，再依据其冻结结果更新这个区间。

要达到 3–5 pp，必须发现多个高覆盖、错误互补且在当前边界上严格特异的动作族；单条 C2H6N 规则的 active train/eval 只有 3/1，数学上不可能承担该目标。

## 6. 当前可执行入口

动作发现、B1/B1J、域内规则、分层 meta-action 与 B-PMT manifest 已整合到一个单 GPU、无手工内存参数的作业：

```bash
sbatch tasks/run_chemaware_high_specificity_action_discovery.sbatch
```

若且仅若该作业自动生成 `CHEMAWARE_BOUNDARY_PMT_MANIFEST_ADMITTED`，再运行四臂 shared-embedding Phase A：

```bash
sbatch tasks/run_chemaware_boundary_pmt_phase_a.sbatch
```

第二个入口自动寻找最新 admitted manifest，不要求手填 job id；同样只申请一张 GPU，不指定 `--mem` 或 `--mem-per-cpu`。

