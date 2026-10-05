# GLM 整合实验预注册：化学冠军 × 噪声课程（1+1 账）

**日期：** 2026-09-30
**性质：** 预注册（运行前冻结）。任何判据、楼层、臂定义在提交后不得修改；结果无论正负都按本文件裁决。
**入口：** `tasks/run_GLM_compose_axis_curriculum.sbatch`（集群策略：`--gpus=3`、不指定分区、不指定内存、全部阶段在计算节点内运行）
**实现：** `tasks/GLM_build_compose_axis_curriculum.py`（构建）、`tasks/GLM_compose_axis_core.py`（纯逻辑）、`tasks/GLM_summarize_compose_additivity.py`（判据）、三套合同测试；训练与评测**全部复用**既有已验证代码（`train_noise_dreams_native_residual_stage2.py`、`evaluate_chemaware_v2_direct_triplet.py`、`select_chemaware_residual_checkpoint.py`、GNPS 编解码评测链）。

---

## 1. 要回答的问题

> 把**化学证据选出的课程**与**噪声动作课程**放在同一条共享 encoder 上先后作用，收益是**相加**、**超加**，还是**互相抵消**？

当前已知：化学冠军 Phase-A role-2 `+2.1266pp`；噪声 Stage-1（绝对）`+0.49637pp`、因果对照 `+1.26548pp`。两线都从 official 独立初始化，**组合从未实验**。因此本实验的核心不是"再训一个模型"，而是**在同一个面板上做 1+1 账**。

## 2. 设计（三臂 + 基线，单一宇宙）

| 角色 | 定义 | 含义 |
|---|---|---|
| **base** | Phase-A step-2000（化学冠军，SHA `a8428329…` 钉死） | 不训练；其 role-2 选择面板即本实验的**唯一计分宇宙** |
| **arm A**（轴平衡） | 在 Phase-A 几何上，从噪声 Stage-1 动作库中每 query 选**一条**关系，且 PD/NE 两轴交错取样 | 机制①：误差轴分工（PD 正缺失 / NE 负过剩） |
| **arm N**（未分层） | 同一几何、同一剂量，按既有 Stage-2 配方选"优势最大"关系 | 机制①的**对照**：分层是否真的有用 |
| **arm C**（配对对照） | 与 A 完全相同的 query、关系、剂量、种子，仅把动作谱换成注册对照谱 | 因果对照：排除"共同续训损失/收益"被误读为课程收益 |

- 三臂**同一预热起点**（Phase-A）、**同一 lr `1e-6`**、**同一 `--max-epochs 1`**、**同一剂量法则**（每 query 至多一条动作关系 + 一条活跃干净边界事件 + 保护哨兵）。
- 不改动任何模型/loss/优化器/预处理器/检查点格式；只改**哪三条谱占据 anchor/positive/negative**。
- 关系准入（在 Phase-A 自身几何上）：动作靶向边距 `< 0.1`、且优于同 query 注册对照 `≥5e-6`、且优于干净同边界 `≥5e-6`；outer-held 公式一律排除。
- 轴判定（训练前固定、无随机）：以处理总体的干净边界中位数为基准，`正缺失 = median(s_true) − s_true`、`负过剩 = s_false − median(s_false)`，取较大者定轴。

## 3. 预注册楼层（GPU 训练前 fail-closed）

- **每轴 ≥ 60 个 query**、**总 ≥ 200 个 query**（含 arm N 的总数）——依据：role-2 面板 1,975 queries 上 +1pp ≈ 20 个净纠正，而受教关系的迁移历来是部分的；低于此楼层任何"+1pp"都不可判读。
- 未过楼层 → `GLM_COMPOSE_FLOOR_GATE: STOP`，**不训练**、不改判据、不退而求其次（楼层是资源门，但不是可以被观测值倒推的门）。
- 预热起点 SHA 必须等于钉死值；契约测试（三套）必须先通过。

## 4. 预注册判据（全部在 role-2 单宇宙上；配对参考 = Phase-A）

必需门（全部满足才算组合成功）：
1. `role2_rows_are_distinct_checkpoints`（六行必须来自六个不同检查点——防止拼接/陈旧报告）；
2. `arm_paired_deltas_match_absolute_recall1`（配对 delta 必须等于绝对 R@1 之差，容差 1e-9）；
3. **组合胜化学冠军**：arm A 对 Phase-A 的配对 `ΔR@1 > 0` 且公式簇 CI 下界 `> 0`；
4. **组合胜配对续训对照**：`R@1(A) > R@1(C)` 且 `corrected > introduced`；
5. `compose_risk_net_positive`（λ=2 风险净 > 0）。
附加门（失败即降级，不阻断记账）：role-3 非回归（保护基线 Phase-A）、GNPS 双面板非回退。

裁决词表（互斥）：
- `GLM_COMPOSE_SUPER_ADDITIVE`：全门过 **且** `Δ_compose > Δ_chem + Δ_noise`（同一面板、同一 baseline）；
- `GLM_COMPOSE_ADDITIVE_ONLY`：全门过但仅为相加，**无协同**；
- `GLM_COMPOSE_SIGNAL_UNSAFE_CONTROL`：胜冠军但**不胜对照** → 增益来自共同续训，不是课程内容；
- `GLM_COMPOSE_INCUMBENT_RETAINED`：组合无增益 → 保留化学冠军。

机制①的独立读数：`role2_axis_contrast.json`（配对参考 = arm N）中 arm A 对 arm N 的配对 CI 下界 `> 0` 才算**轴分层有效**；否则机制①被证伪（记账仍成立）。

## 5. 可证伪映射

| 观测 | 结论（不可事后改写） |
|---|---|
| A > N > C，A 对 Phase-A CI>0 | 课程组合有效 **且** 轴分层有效 → 1+1 至少相加 |
| A ≈ N > C | 组合有效但**分层无用** → 机制①证伪，组合收益来自课程内容而非轴分工 |
| A ≈ C | 收益来自共同续训 → **不能**声称课程贡献 |
| A < 基线 | 化学冠军上的噪声课程**有害** → 顺序组合方向证伪 |
| 楼层未过 | 本设计在当前证据量下**不可判读**，不改门 |

## 6. 次级（不阻断主结果）

- 修正图 fold-0 18,333 held queries：对 Phase-A / arm A / arm C 各出一份噪声侧完整评测（Recall@k、MRR、macro/micro AUROC、pooled pairwise、near、corrected/introduced）。该阶段以 `set +e` 包裹：失败只记录 `fold0_secondary` 状态，**绝不撤销**已写入的 role-2/GNPS 主结果（避免"次级失败毁掉主结果"的事故模式）。
- GNPS Gold/Silver 双面板（identity-disjoint 10,995 / formula-disjoint 5,261）：official、Phase-A、Noise Stage-1、A、N、C 六者编码后按六个配对评测，其中 `phasea_vs_official` 与 `noise_stage1_vs_official` 提供同一外部面板上的 Δ_chem / Δ_noise，可用于交叉核对角色面板上的 1+1 账。

## 7. 成本与产出

- 一个作业；构建 2 次（每 GPU 一次，并行）、训练 3 次（并行）、role-2 评测 2 次（含轴对照）、role-3 评测 1 次（条件）、GNPS 编码 6 次 + 评测 6 次、fold-0 评测 3 次（次级）。
- 产出：`$OUT/additivity/report.json`（裁决 + 完整记账）、`$OUT/run_status.json`、`$OUT/SHA256SUMS`、两套构建报告（含轴计数/每轴 margin 分布/来源分布/哈希溯源）。

## 8. 明确的非主张

- 本实验**不**产生 outer/fold-4 结论，也不消费 role-3 作为旧主张的确认（role-3 仅作非回归保护）。
- 本实验**不**修改 1,000/500 的独立化学线覆盖门；60/200 是本**组合实验**的判读楼层，不构成对化学线独立成池门的重定义。
- 实验前不得宣称任何协同；`SUPER_ADDITIVE` 即使成立也仅限 role-2 选择面板这一个宇宙，外部效度须由 GNPS 面板与后续独立源确认。
- GLM_ 前缀为命名空间约定；论文禁用 "GLM analysis" 表述。

## 9. 提交

```bash
sbatch tasks/run_GLM_compose_axis_curriculum.sbatch
```

作业内的契约测试与楼层门会在任何 GPU 训练之前先跑；构建报告里的 `axis_selected_queries` / `axis_qualified_queries` / `compose_axis` 列是复核轴分配是否诚实的入口。
