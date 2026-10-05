# Noise 直接微调瓶颈重审（2026-09-05）

## 裁决摘要

现有证据不支持“动作本身无效”或“必须先蒸馏 teacher 才能微调”。更准确的结论是：

1. E4 及其后继把好动作编码成一个更容易的正样本，却没有把动作揭示的候选决策边界直接施加到 clean query；动作越成功，单调 rank loss 给它的梯度反而越小。
2. E4 的 clean-duplicate 本身把 clean rank 算了两次；targeted 的绝大多数 held 增益可由这份普通 continuation 解释。
3. dynamic-direct 仍然只是用 action loss 替换一部分 clean-duplicate loss。好动作的 loss 更低，因而会减少而不是增加有效更新剂量。
4. 后期 DEB 虽声称保留 candidate matrix，但训练聚合与部署检索不一致：训练重加权所有 positive-reference × negative-molecule edge，部署却取每个 molecule 的最佳 reference。它优化的不是最终 Top-1 边界。
5. E14/E15 又把本来多来源、多动作的结构过滤成小而偏的单动作/少 identity 训练池。动作容量没有消失，是进入训练器之前被数据结构和采样器压扁了。

因此，真正要修的是 **action -> clean candidate boundary -> shared encoder update**，而不是先假定需要 teacher embedding 蒸馏，也不是推翻 noise 微调路线。

## 一、E4 为什么“动作很好，改错效率却低”

历史 E4 `direct_action_loss` 为：

```text
clean_rank + action_rank + 0.25 * consistency
+ 2 * clean_margin_floor + 5 * embedding_preserve
```

它没有 target-vs-control 的 action advantage，也没有把 action 揭示的真实 hard-negative boundary 变成 clean-query 的直接目标。

对不可变 E4 causal 日志的复算结果：

| 项目 | clean-duplicate | targeted | targeted 独立部分 |
|---|---:|---:|---:|
| held delta | +0.5065 pp | +0.5403 pp | **+0.0338 pp** |
| corrected / introduced | 37 / 7 | 38 / 6 | 仅多 1 修复、少 1 引入 |
| net corrections | 30 | 32 | **2** |

即 targeted 的 32 个净纠错中，30 个（93.75%）已经由 clean-duplicate 解释。

更关键的是梯度方向反了：targeted action 的平均 margin 比其 clean view 高约 0.0123，但在 E4 的 softplus rank loss 下，action 的 margin 梯度压力只有同一 query clean view 的约 0.89。动作越成功、margin 越大，loss 与梯度越小。与此同时 targeted 每 epoch 约 89%–93% 的 step 触发 global gradient clipping。clipping 只能缩放已有梯度，不能补回目标函数里不存在的 action-to-clean 信息。

clean-duplicate 的空 action path 会重放原谱，因此 `clean_rank + aug_rank` 实际是同一个 clean rank 的双计数。这不是合格的 action control；它是一份额外 continuation 剂量。

## 二、dynamic-direct 为什么仍没有修好

dynamic-direct 的核心仍是：

```text
augmented = no_op * clean_loss + sum(action_weight * action_loss)
total = clean_loss + augmented + safety/preserve
```

它匹配了 action mass，却没有建立 action-to-clean transfer term。不可变正式日志显示：

| arm（epoch 4） | action mixture - clean loss | mean grad norm | clip fraction | held delta vs mature init |
|---|---:|---:|---:|---:|
| clean continuation | 0 | 2.604 | 0.443 | +0.1182 pp |
| matched random | +0.0783 | 3.292 | 0.552 | +0.1182 pp |
| static target | -0.0147 | 2.458 | 0.437 | +0.1182 pp |
| dynamic N+P | -0.0170 | 2.437 | 0.437 | +0.1013 pp |

dynamic 找到了更容易的 action，于是 action branch 的 loss 和梯度都更低；最终它比 clean/static 少一个净 query。这里失败的不是动作质量，而是“用更容易的 action loss 替换 clean loss”这一注入语义。

## 三、对既有记录的系统性批判

### 3.1 E8 并没有证明动作已经迁移

stop-gradient、symmetric 和 control 的 Top-1 近似，只能说明 consistency 不是主要增益来源。它不能证明 action direction 已进入 clean encoder；三臂近似同样支持“action branch 基本无关，主要是 continuation”的解释。

### 3.2 action 可预测性不等于可注入性

M2/Brier、clean-visible router 或 teacher replay 可以回答“能否预测哪个动作好”，不能回答“现有 loss 是否会把这个优势写入 clean embedding”。以前把 learnability gate 当作 transfer gate，混淆了选择问题与优化问题。

### 3.3 全局 gradient compatibility 不是必要条件

不同 formula、identity 和 candidate boundary 的逐样本高维梯度不需要指向同一个全局方向。把低 cosine 当成动作不可迁移，会错误拒绝本应由条件模型分别吸收的局部方向。真正应测的是每个来源在自身边界上的 held influence，而不是先求全局共线。

### 3.4 DEB 的“candidate matrix preserved”表述过强

DEB 对每个 positive reference 都产生 edge，并用 `clean_margin.min(dim=0)` 选 hard molecule 后对所有 positive edges 归一化加权；真实检索对一个 molecule 取最佳 reference。弱 positive reference 会在训练中得到指数级 hardness，却可能完全不决定部署时的 molecule score。micro edge AUC 上升、hardest margin/Top-1 下降与此聚合错位一致。

DEB 的梯度校准还有独立问题：`max(1.0, requested_scale)` 禁止减小 action branch。实际 action norm 约 25.23、safety norm 约 4.29e-6，所需 scale 约 1.7e-7，却被固定为 1.0，随后 100% clipping。所谓 calibration 没有执行它测得的缩放。

### 3.5 E14/E15 的负结果不能裁决路线

E14 每 query 单动作坍缩、global safe filter 和把 risk 当正向增强；E15 从数千 action 缩到 281 actions / 198 queries / 133 identities，并丢弃参考排除后无合法候选的 query。M2 又能在小 panel 上做到 27 corrected / 0 introduced。合在一起说明 payload 与网络有容量，正式 held 失败来自训练池和监督实现，不是 noise action 没有潜力。

### 3.6 teacher -> peak gate -> encoder 何时算蒸馏

若 peak gate 或 encoder 以 teacher score、teacher margin、teacher residual 或 teacher embedding 为回归目标，这条链就是蒸馏/教师迁移。若 action 只在 outer-train 内负责选择真实 identity boundary，而 clean query 直接以真实候选身份监督更新，则不是必须依赖 teacher target 的蒸馏。现阶段后者更直接，也更符合部署目标。

### 3.7 第一版 clean-visible gate 本地正结果无效

`policy_candidate_actions.csv.gz` 不是完整候选集合；其生成脚本先按真实 corrected / margin outcome 排序再截取候选。用它训练或评估 gate 会把结果筛选泄漏成“可学习性”。因此 `noise_clean_visible_peak_gate_audit_20260905` 必须标为无效，不得引用其正结果。改用 `exact_peak_scan.h5` 全动作集合后的 gate 与 residual-injection 结果为负；这些负结果只否定对应窄接口，不否定直接 clean-boundary 微调。

## 四、本地非蒸馏直接注入实验

### 4.1 合同

- 训练标签只有真实 molecular identity。
- S3A action outcome 只在训练折内决定 clean boundary 的剂量；不回归 teacher embedding、margin 或 residual。
- 每个 outer fold 完整排除 evaluation formula。
- 同一个 identity-initialized bounded map 同时作用于 clean query 与 candidate reference。
- 训练和评估都使用 molecule-level candidate retrieval；不把 edge AUC 当主终点。
- 这是 frozen official embedding 上的 shared-map reachability proxy，不是 raw-spectrum encoder checkpoint。

### 4.2 结果

S3A 训练池在 candidate manifest 交集后有 15,411 queries、945 formulas、520 action-correctable queries。最佳冻结配置结果：

| 配置 | delta | formula-cluster CI | corrected / introduced | mean query cosine |
|---|---:|---:|---:|---:|
| clean-uniform | +2.476 pp | [+1.182, +3.970] | 298 / 192 | 0.939 |
| action-routed | **+3.808 pp** | **[+2.438, +5.338]** | **315 / 152** | 0.936 |

动作路由相对同剂量 clean control 的独立差值约 **+1.33 pp**，远大于历史 E4 targeted 相对 clean-duplicate 的 +0.034 pp。

最佳 action-routed 的分折结果：

| formula fold | delta |
|---|---:|
| 0 | +5.577 pp |
| 1 | +5.586 pp |
| 2 | +0.649 pp |

query-weighted 总体为 +3.808 pp；formula-macro 为 +7.202 pp。相邻 seed 总体为 +2.359 与 +3.060 pp。因此已经出现明确的 4–6 pp 局部/公式簇注入希望，但尚未证明稳定、多 seed、全体 query 加权的 >=4 pp。

低秩宽度 32/64/128、对角度量、correctable 权重 5/6/7、epoch 12/14/16 的窄审计均表明：64 维、权重 6 是一个真实但 seed-sensitive 的窗口。继续在同一 OOF panel 上搜索会产生模型选择偏差，应停止。

## 五、下一版直接微调的强制合同

1. **动作不再作为第二个容易正样本。** target/control 的作用是训练折内选择和分配 clean candidate boundary 的预算；主梯度直接落在 clean query 的真实 identity Top-1 边界。
2. **完全匹配 molecule-max 聚合。** 对每个 positive molecule 与 negative molecule 使用 live best-reference score；训练 hard-negative refresh 与评估使用同一聚合。
3. **多动作结构不可压扁。** ledger 主键为 `(query_row, source, action_recipe, reference_strategy, control)`；每 epoch 按 query 归一化总剂量，保留 N/P/source multiplicity，不能用单一 best action 代替。
4. **harmful/uncertain 为约束，不是正增强。** harmful 动作只产生 clean/reference preservation 或反事实不变性约束；不得进入 action rank positive branch。
5. **分支先校准再相加。** action、clean、safety 各自记录未裁剪 norm、cosine 和 influence；允许 action scale 小于 1，禁止 `max(1, requested)`；global clip fraction 若长期高于 10% 就 fail closed。
6. **结构化 trust region。** 不仅在已见 batch 上罚 cosine，还必须限制每个谱的最大 residual norm；防止未见 formula 无界外推。
7. **对照必须是等预算 clean-boundary continuation。** clean-duplicate 不能再冒充动作对照。promotion 需要 action-routed 相对该对照有严格正的 formula-cluster CI。
8. **先恢复后期 P 动作制品。** 本地缺少 E10B/E11/E12B 的逐 query action matrices；在服务器只读核对 hash、candidate geometry 与引用行后，才允许把其 reference-diversity/P directions 加入训练。不能从文档汇总数字反造训练标签。
9. **门槛不变。** 正式 raw-spectrum encoder 必须在未消费的 formula panel、多 seed 上达到 query-weighted >=4 pp、严格正的 multiplicity-corrected formula CI，并保持官方全图 safety；本地 +3.808 pp 只放行实现，不放行结论。

## 六、当前决定

- 不再推进 teacher -> peak-gate -> encoder 作为默认主线。
- 不再修补 historical E4、dynamic-direct、DEB 或 E15-M3 trainer。
- 保留动作科学逻辑，重写 action-routed clean molecule-boundary trainer。
- 当前不提交正式 sbatch：本地已看到 4–6 pp 的分折/公式宏观希望，但多 seed query-weighted >=4 pp 尚未成立，且后期 P action 原始制品尚未在本地核验。
- 下一步是服务器只读恢复 E10B/E11/E12B 逐动作矩阵，然后把 N/P 多来源方向加入同一直接 boundary 合同；不是再训练一个 teacher。

## 七、2026-09-06 继续审计：反证、确定性代码错误与 v2 实现

### 7.1 不能把 `molecule-max` 当作单独解药

在完全相同的 S3A formula-OOF proxy 上，只把训练候选从每分子一张冻结代表谱改成每分子两张谱、每步重新取 max，结果从 `+3.8075 pp` 降至 `+2.1023 pp`，corrected/introduced 从 `315/152` 变为 `296/206`。动态 reference switch 增加了断档；因此 evaluator-aligned 聚合是必要合同，但不是充分方法。

另外三项有明确假说的反证也均失败：

| 改动 | delta pp | corrected / introduced | 裁决 |
|---|---:|---:|---|
| action-introduced control 定向加权 | +2.5695 | 279 / 169 | 记忆训练折 harmful query 会稀释 corrective budget |
| 按 correcting-action 复发数连续加权 | +1.8454 | 250 / 171 | 多 action 同时成功不等于更可迁移；单次成熟动作不能降为噪声 |
| clean hardest-molecule `0.05` margin | +2.3359 | 297 / 197 | 增强 clean 压力同时扩大 shared-geometry 漂移 |

这些结果否定继续扫描 gate、safety weight 或 rank margin。旧 `+3.8075 pp` 是有用的可达性证据，但本身仍是脆弱的 query reweighting proxy。

### 7.2 `--include-a4-training` 历史上从未加入 A4

统一 E0 ledger 的 672,256 条 A4 action 行中，`query_row` 全为空；旧 loader 却直接按 `query_row` 分组，导致 A4 全部被 pandas 静默丢弃。因此旧的 S3A+A4 报告虽然配置写着 `include_a4_training=true`，训练池仍只有 S3A 的 520 个 action-correctable query。

现已用冻结 `scan_queries.csv.gz` 的 `query_index -> query_row` 一对一映射修复，并加入 identity/formula 一致性及零映射 fail-closed。修复后训练池 action-correctable 从 520 增至 882。但同配置复测只有 `+1.0278 pp`、`267/223`，formula CI 跨零。这一结果同时说明：

1. 旧 A4 开关是确定性的实现错误；
2. A4 把 oracle 覆盖扩至 882，不代表它的逐 query outcome-selected 单峰动作可通过 clean-query 重采样迁移；
3. 后续不能继续把 882 当作“不携带峰方向也能训练”的证据。

### 7.3 真正缺失的是 action payload，而不是更多 query gate

本地 `selected_sequences.csv.gz` 完整保存 66,735 条 S3A ordered target trajectories 及每条轨迹的两条 matched-control paths。重新运行 faithful R0 builder 后得到：

- 36,934 条成熟 N action；
- 1,991 identities、877 formulas；
- candidate-gradient `a=0.50, step=3..6`；
- role-confounder `a=1.00, step=1..5`；
- outcome 字段与 student manifest 物理分离。

在 official 初始化几何上，严格 `target - matched-random > 0.01` 且 clean 为错误的本地 PMT 开发清单包含 2,284 actions、589 queries、199 formulas，并覆盖全部 9 个成熟 N cells。它只用于 CPU 接线测试；mature-E8 正式训练前仍必须在该当前几何重放，不能沿用 official outcome。

### 7.4 direct-boundary v2 已实现并通过的合同

新增 v2 不再对每个弱 positive reference 建立伪难 edge，而使用：

```text
max(score(query, true-molecule spectra))
- score(query, each live negative-molecule representative)
```

target/control 仅以 detached 方式分配一个固定的 per-query action budget；primary corrective loss 始终落到 clean molecule boundary。同一 query 的全部 action 必须共享完全相同的正负候选、同一 epoch、同一完整 batch，并在 query 内归一化；重复 action 不得放大 clean、action 或 preserve 剂量。harmful/uncertain action 不进 positive rank，只能进单边 action-safety。matched control 不反传。梯度校准允许 scale 小于 1；但 safety hinge 在初始化为零时回退 scale=1，禁止把 action stream 静默归零。

当前通过：8 个 v2 数值不变量、2 个 raw/multi-action CPU smoke、E4-A/PMT 既有回归测试。正式 trainer 已增加 `candidate_boundary_version=v2_molecule_max`，并保留 v1 仅作历史反证。

### 7.5 P recipe 已冻结，P outcome 尚未伪造

本地 recipe registry 已从代码而非结果报告提取：E10B 19 个 positive recipes/38 个方向列，E11 16 个 reference-diversity cells，E12B 25 个 relaxed-recurrence cells；同时固定 mature E8 checkpoint SHA。registry 明确声明未读取 historical outcome matrix、best cell 或 outer-held label。

因此当前“超过 4 pp 的大量注入希望”有具体结构，但必须分清已接通与待接通：36,934 条成熟 N raw actions 是五折前的源动作；fold 0 排除 held formula 后有 28,509 条，现已接入 v2。E10B/E11/E12B 的 60 个 P recipe/reference cells 目前只完成无 outcome 的 registry，尚未接入 v2，不能写成已经共享训练。历史成熟几何的 N/P action union 曾达到 `+4.457 pp`，但它仍只是容量；本地尚未得到 query-weighted、multi-seed 的 raw encoder `>=4 pp`，不得冒充完成。

## 八、2026-09-06 第二次实现批判：全 routed 注入链真正打通

继续沿 manifest -> loader -> sampler -> loss 审查后，又确认并修复四个会制造“配置已开、训练未吃到”的断档：

1. **loader 只读正动作**：v2 被 PMT 入口约束，但旧 loader 只读取 2,284 条 `corrective_actions.csv.gz`，因此 28,509 条 fold-0 routed action 从未贯通。v2 现强制读取 `all_routed_actions.csv.gz`。
2. **scope 只过滤旁路文件**：`corrective-query-scope=errors` 过去只过滤 corrective 导出表，没有同步清零 all-routed 表的 clean-correct 正权重。scope 现下沉到 routing core；clean-correct action 被降为 robustness/uncertain 且 corrective weight 精确为零。
3. **逐 action schedule 偷乘 query 剂量**：旧 coverage-first 按 action 分 epoch，同一 query 的 1--9 条 action 可跨 epoch、拆 batch。v2 现以完整 query action-set 为不可分割单元，全部动作只曝光一次，不 recycle。
4. **负候选张量同形异义**：各 action 的 `hard_negative_row` 会使同一 query 的 margin 向量对应不同负分子。v2 现统一 query 级 candidate references，并在 loss 内发现任何语义漂移即 fail-closed。

修复后的 official-geometry 本地开发 manifest（不是 mature-E4 encoder 结果）为：

- 28,509 routed actions；
- 7,766 queries、1,562 identities、689 formulas；
- 2,284 corrective、22,018 robustness-only、2,534 harmful、1,673 uncertain；
- 每 query 1--9 actions，中位数 4；
- 四个 epoch 为 7,127 / 7,128 / 7,128 / 7,126 actions；
- `batch-actions=9` 下为 7,766 个 query-equal/query-complete optimizer steps，单 step 一个 query、最多 9 actions；所有动作恰好一次，零 query 跨 epoch。

机器审计 `noise_direct_v2_injection_manifest_audit_query_equal_v2_20260906/report.json` 将当前结论拆成两个不可混淆的字段：

- `credible_large_injection_ge4pp_hypothesis=true`：28,509 条真实 raw action 已具备完整注入 schedule，且冻结共享映射的最佳本地点估计为 `+3.8542 pp`、formula-cluster CI `[+2.5470,+5.3438] pp`；因此超过 4 pp 是可检验假说，不再是拍脑袋。
- `ge4pp_encoder_result_achieved=false`：本机缺正式 candidate graph 与 official embedding cache，尚不能运行完整 117M raw encoder；任何文档不得将结构性希望写成已完成性能。

正式但尚未提交的入口为 `tasks/run_noise_final_direct_boundary_v2_phase_a.sbatch`。它只用一张 GPU，在唯一 job-ID 根目录内运行 fold-0 三 seed；每个 seed 必须相对 official 总增量 `>=4.0 pp`、相对 mature E4 增量为正、Bonferroni 修正公式簇 CI 为正，并且完整候选 MRR/AUC/AUPRC/Recall@2..20 等面板全部不退化，才允许进入 multifold。失败则冻结该配置，不再用 P 或 teacher 掩盖 N-transfer 失败。
