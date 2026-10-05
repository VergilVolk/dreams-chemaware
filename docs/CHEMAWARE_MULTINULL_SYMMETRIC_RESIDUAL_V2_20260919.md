# ChemAware Multi-Null Symmetric Residual V2

日期：2026-09-19  
状态：`POST-OUTER DEVELOPMENT / ROLES 0-3 COMPLETE / NEW INDEPENDENT CONFIRMATION REQUIRED`  
禁止事项：不得用已经消费的 formula role 4 重新选择、修改或确认本方法。

## 1. 科学问题纠偏

旧方法只证明“正确化学视图可以产生高分”，没有强制证明该动作依赖正确化学语义；
单一 content-permuted 对照也不足以区分真实规则响应、偶然的候选几何和空对照特异模式。

V2 将问题改写为：在 official DreaMS 与 mass 几何已经解释的风险之外，正确化学中心相对
多个交换空中心是否仍能稳定预测某候选干预的收益和伤害，并且该优势能否同时击败可部署的
错误方向、零化学和候选错位对照。

## 2. 表征与目标

对每个 query-candidate 对，记非规则特征为 `B`。正确规则中心为 `C0`，三套独立内容置换
中心为 `C1,C2,C3`。对每个化学特征计算 `D_j = C0 - C_j`，使用对中心次序不敏感的表示：

```text
Phi = concat(mean(D), min(D), max(D), std(D), P(D>0), P(D<0)).
```

在 formula roles 0/1 内交叉拟合不含规则的 nuisance 概率：

```text
m_t(B) = P(Y_t=1 | B),  t in {benefit, harmful}
R_t = Y_t - m_t(B)
r_t(Phi) ~= E[R_t | Phi]
U = (m_b + lambda*r_b) - 2*(m_h + lambda*r_h).
```

role 2 同时冻结 `lambda` 与动作阈值。选择目标不是正确臂自身的最好点估计，而是最大化它
相对全部选择空对照的最小风险效用优势，并要求 corrected 至少为 introduced 的两倍、覆盖
至少50个公式。role 3 只做一次不调参确认。

## 3. 可部署的化学对照

选择阶段使用：

1. `zero_contrast`：移除全部化学增量；
2. `reversed_contrast`：反转化学方向；
3. `candidate_rotated_truthblind`：在每个查询内部循环错配候选化学证据，不读取标签、身份、
   公式或旧排名；
4. `null_semantics_0/1/2`：分别把三个内容置换中心当作伪正确中心。

旧 `alignment_permuted` 使用 baseline 是否正确进行分层，只保留为开发诊断，不进入 V2 的
选择或部署门。

## 4. 冻结设置

- representation：`symmetric_summary`
- rule nulls：`content_permuted / content_permuted_b / content_permuted_c`
- residual dose：`2.0`
- candidate threshold：`0.3565609484787261`
- risk penalty：`2.0`
- role 2 selected actions：119，覆盖82个公式
- role 4：本次运行未评分；但 role 4 已被旧实验消费，不能再充当新方法确认集。

## 5. 当前开发证据

| 数据角色 | Queries | Corrected | Introduced | Recall@1增量 | MRR增量 |
|---|---:|---:|---:|---:|---:|
| role 2 选择集 | 1,978 | 78 | 17 | +3.0839 pp | +0.01736 |
| role 3 固定确认 | 1,929 | 93 | 17 | +3.9399 pp | +0.02268 |

role 3 的 formula-cluster 95% CI 为 `[+2.8191,+5.1921] pp`。相对可部署的
candidate-rotated 空对照，Recall@1 仍增加 `+3.3178 pp`，formula-cluster 95% CI 为
`[+2.3060,+4.3776] pp`。相对三套 null-semantics 的配对区间也全部严格高于0。

相对旧 canonical residual 的 role-3 `+3.1104 pp`，V2 在同一已用开发集上的点估计增加
`+0.8295 pp`。这是同面板方法比较，不是新的独立泛化证据。

## 6. 已否定的组合

新表征下 same-feature direct 在 role 3 为102 corrected / 50 introduced，风险明显高于 residual。
因此旧的 direct-first 经验不能机械迁移。counterfactual-exclusive direct-first 仅为
`+2.7475 pp`，相对 direct 的增量区间跨零，已停止。

residual-first 再接 nuisance 在 role 3 点估计为 `+4.0954 pp`，但 role 2 比 residual-only 更差，
且相对 residual 的区间跨零。按 role-2 冻结顺序，正式 V2 保留 residual-only，不追逐该点估计。

## 7. 结论边界与下一门

目前可以说：多空对照、对称化学残差在两个开发角色上形成了约3.1--3.9 pp 的候选排序增益，
且正确化学语义相对多类空对照有严格正的配对优势。

目前不能说：V2 已在新 outer 复现、已提升共享 embedding、或已经达到外部数据泛化。

下一步只能在新的独立来源/新冻结面板上一次性比较 official、旧 residual、V2 residual 及全部
空对照。若 V2 通过，再把其候选级效用差作为教师，蒸馏到共享 embedding 的相似度差；不得先
用已消费的 role 4 选择蒸馏权重。

## 8. 本地产物

- 主开发报告：`data/validation/chemaware_multinull_deployment_safe_full_20260919/report.json`
- role-2 冻结账本：`data/validation/chemaware_multinull_deployment_safe_full_20260919/validation_policy.npz`
- role-3 冻结账本：`data/validation/chemaware_multinull_deployment_safe_full_20260919/inner_policy.npz`
- truth-blind bundle：`data/validation/chemaware_multinull_deployment_safe_full_20260919/truthblind_policy.joblib`
- 组合否定审计：`data/validation/chemaware_multinull_exclusive_backoff_full_20260919/report.json`
