# BioAware B39-M2 内部固定原子动作结果（2026-09-13）

## 裁决

B39-M2 严格通过了工程复现门，但 6 个预注册核心原子动作均未在 B37 `catalog_topology` 之上产生正增量。不得进入复杂模型，不得进入 context representation，也不得作为 shared embedding 教师。

该结果不否定 B37 的工程增益。相反，在完全一致的 548 个内部真实查询上：

- 官方 DreaMS Recall@1：`0.70255`；
- 冻结 B37 topology Recall@1：`0.75365`；
- topology 相对 DreaMS：`+5.1095 pp`；
- topology 转换：31 corrected / 3 introduced。

它否定的是一个更窄的命题：当前 M1 能恢复的“方向明确 + Rhea 超边完整 + 候选特异 + 匹配实验层为正”的原子事件，不能在 topology 之上进一步纠错。

## 复现与实现审计

- B37 候选图与 topology transition 完整复现：548/548 query；候选数、baseline gap 和所有候选身份零漂移。
- 所有 cell 使用同一 topology 起点。
- 所有 cell 使用同一风险层：`DreaMS baseline_gap <= 冻结 outer-domain topology gate_margin`。
- 没有按 cell 重训模型、重新选择阈值或改变 gate。
- 两个未构建的 null 明确报告为 unavailable；没有填 0。
- `candidate specificity` 消融明确标记为不可识别，因为当前 ledger 中该变量恒为 1。

## 核心格结果

| cell | 动作 | 对 topology 的 corrected / introduced | Recall@1 增量 | 实际干预 |
|---|---|---:|---:|---:|
| B39-00 | 有向完整 Rhea + spectral | 1 / 3 | -0.365 pp | 4 |
| B39-01 | 有向完整 Rhea + coabundance | 1 / 4 | -0.547 pp | 6 |
| B39-02 | 有向完整 Rhea + 两层同时为正 | 1 / 2 | -0.182 pp | 3 |
| B39-03 | 双向完整 Rhea + spectral | 0 / 0 | 0 | 0 |
| B39-04 | 双向完整 Rhea + coabundance | 0 / 1 | -0.182 pp | 2 |
| B39-05 | 双向完整 Rhea + 两层同时为正 | 0 / 0 | 0 | 0 |

知识边单独覆盖的消融 B39-08 为 1 corrected / 7 introduced，相对 topology 为 `-1.095 pp`。这进一步说明不能把“有 Rhea 边”直接升级成推翻 topology 的硬动作。

## 覆盖漏斗揭示的真正瓶颈

548 个 query、2,003 个候选对中：

| 条件 | event rows | 有覆盖 query | 真值获支持 query | 错候选获支持 query |
|---|---:|---:|---:|---:|
| 任意原子目录事件 | 7,789 | 362 | 288 | 199 |
| Rhea event | 5,497 | 239 | 196 | 117 |
| 方向明确 | 581 | 68 | 46 | 23 |
| Rhea 超边完整 | 1,330 | 130 | 81 | 62 |
| spectral excess > 0 | 170 | 50 | 32 | 21 |
| coabundance excess > 0 | 2,056 | 171 | 136 | 80 |
| 两个实验层同时 > 0 | 84 | 27 | 18 | 10 |
| 方向明确 + 超边完整 + 两层同时 > 0 | **7** | **4** | **2** | **2** |

严格动作最终只覆盖 `4/548 = 0.73%` query；其中真值与错候选各 2 个。此处没有一个有足够样本量的“弱正信号”等待扩大，只有严重的交集稀疏和失去判别性。

B37 的 31 个 topology 修正中：

- 21 个真值有任意目录事件；
- 14 个真值有 Rhea event；
- 3 个真值方向明确；
- 7 个真值超边完整；
- 只有 1 个真值有正 spectral evidence；
- 最严格交集为 0。

因此严格事件动作不是 topology 的增量确认器，而是把 topology 已经正确修复的绝大多数样本排除掉。

## 科学解释

1. B37 的约 5–6 pp 信号主要属于 catalog/network observability prior：真值位于可信代谢目录，而 DreaMS 错候选往往不在。
2. 当前内部实验层不足以证明某一条具体 reaction event；它在 candidate–seed pair 层可重放，但在 reaction-ID、方向、超边完整性和两层实验证据取交集后几乎为空。
3. “再放宽一个阈值”不是答案。任何依据本结果放宽条件并重新报告的动作都会产生 post-outcome selection。
4. 短期工程主线应保留 B37/B30 catalog-risk router；反应特异创新主线转为预注册的图补全/软上下文动作，并要求 topology 增量及强 null。

## 下一步

### 工程保底

冻结 B37/B30，完成未参与开发的真实外部盲测和项目级 target-decoy/FDR。它解决的是强谱图模型的最后一公里 Top-1 风险路由，不应被改名为反应传播。

### B40 方法创新

不再要求每个 query 都观测到完整的一步反应超边。参考 MetDNA3/KGMN 的两层图思想，构建：

- 知识层：Rhea/KEGG hypergraph 的方向、反应族、组分和度数；
- 数据层：样本内 seed 可见性、共丰度和 query–seed 谱学边；
- 映射层：候选–seed 的软多跳图完成分数；
- 安全层：B37 catalog risk + B30 sink veto + abstention/FDR；
- 强 null：component/degree/arity/direction-matched rewire、sample-local seed permutation、matched non-neighbour。

B40 首先只生成 outcome-blind 图完成分数。只有相对 topology 至少 +3 pp、对 null 显著且跨来源保持，才进入 context-conditioned candidate embedding。当前没有依据把该上下文蒸馏进 candidate-independent clean-spectrum encoder。

## 产物

- `tasks/freeze_bioaware_b39_m2_fixed_action_manifest.py`
- `tasks/evaluate_bioaware_b39_m2_fixed_action.py`
- `tasks/validate_bioaware_b39_m2_fixed_action.py`
- `tasks/audit_bioaware_b39_m2_coverage_funnel.py`
- `tasks/run_bioaware_b39_m2_fixed_action.sbatch`
- `data/validation/bioaware_b39_m2_manifest_localcheck_20260913_v1/`
- `data/validation/bioaware_b39_m2_localcheck_20260913_v1/`
