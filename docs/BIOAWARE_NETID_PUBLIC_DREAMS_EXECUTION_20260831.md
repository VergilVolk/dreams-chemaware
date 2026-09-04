# BioAware × NetID 公开数据执行与裁决记录（2026-08-31）

## 1. 当前结论

原定“完整复现 KGMN/MetDNA2，再只替换 MS2 边”的路线被真实外部依赖阻塞：作者 OEP003284 原始 LC–MS 数据需要 NODE 认证下载，完整 zhuMetlib 也未公开随仓库提供。缺少这些输入时，不能把小型补充表或内部数据伪装成完整 KGMN 复现。

因此当前采用公开、可审计的 NetID v1.0 release 建立较小但真实的下一步：

1. 复核作者公开输出及其评价映射；
2. 从 mouse-liver 公开 targeted MS2 工作簿恢复 feature 级真实谱图；
3. 用冻结官方 DreaMS 编码这些谱图；
4. 仅检验 DreaMS 相似度是否能识别 NetID 作者图中的 feature-feature 边；
5. 只有该信号通过 component-cluster 统计门，才进入交叉拟合的边可靠度校准和固定 FDR 网络增量。

这不是完整 NetID 重跑，不是新的盲测，也不是 SOTA 结果。

## 2. 已完成的公开作者结果审计

冻结输入：

- NetID v1.0 Zenodo/GitHub release；
- archive SHA256：`46d6aab26f980e228559cac98d3cbf7aa09d10e97c8dd2782fdb374b4e4fab5a`；
- DOI：`10.5281/zenodo.5508337`。

正式报告：

- `data/validation/netid_public_release_audit_v2_20260831/report.json`

关键复核结果：

- yeast 高置信手工结果通过 `medMz + medRt -> raw row -> NetID_output row` 对齐；禁止把 `manual_curate.id` 当输出行号；
- 映射到 314 个高置信 feature；
- 作者 formula accuracy = 292/314 = 0.92994；
- class accuracy = 281/314 = 0.89490；
- formula + class joint accuracy = 262/314 = 0.83439；
- 这些是已公开且已消费的作者结果，只能作实现审计，不能再称独立盲测。

## 3. 为什么不能声称精确复现 NetID 求解器

作者流程调用 `cplexAPI` / IBM CPLEX，但公开 release 仅给出求解后的 Cytoscape nodes/edges，没有完整未选择变量、目标系数和约束状态。换用 SciPy MILP 或其他求解器也无法还原一个没有被导出的 ILP。

因此：

- 可复核作者输出；
- 可使用公开 post-solution graph 做组件级信号研究；
- 不可声称精确重跑作者全局优化；
- 不可把作者预测 assignment 当独立结构真值。

## 4. 可用真实 MS2 规模

mouse-liver 公开工作簿审计结果：

- 123 个 xlsx；
- 1,552 个 targeted requests；
- 536 个 sheet 是明确的缺失 MS2 占位符；
- 1,016 张非空谱；
- 810 张谱至少有 3 个 fragment peaks，对应 808 个唯一 features；
- 11,713 个 fragment peaks；
- 公开表中只有 3 个 manual metabolite 带 SMILES，故不具备结构检索真值规模。

公开 post-solution graph 中，预计有约 188 条两端均有至少 3 峰谱图的作者边，其中包含 biotransformation 与 isotope/adduct/fragment 等 ion-phenomenon 类型。正式数目以 GPU 阶段输出为准。

## 5. 当前单变量问题

问题不是“DreaMS 能否直接重排 NetID 的结构候选”，因为该公开 release 没有足够独立结构真值。当前问题严格限定为：

> 在不使用 DreaMS 分数做匹配、不训练模型、不使用 phenotype 的情况下，作者 post-solution feature edges 的官方 DreaMS centroid cosine 是否高于 precursor-mass、RT、graph-degree 和 peak-count 匹配的 nonedges？

推断以作者图 connected component 为 cluster bootstrap 单位，避免把同一网络组件中的大量边当独立样本。

通过门：

- 至少 150 条可评估作者边；
- overall component-cluster bootstrap CI 下界大于 0；
- biotransformation 与 ion-phenomenon 两类的 mean delta 均不为负。

负结果也必须保留：若门失败，停止从该公开数据源向 NetID 注入 DreaMS edge，不再靠调阈值挽救。

## 6. 唯一服务器入口

同步本轮新增/更新的 `tasks/`、`tests/` 文件后，仅提交：

```bash
sbatch tasks/run_netid_public_dreams_edge_stage.sbatch
```

脚本将依次：

1. 检查 Python 依赖；
2. 运行 12 项实现和 sbatch 合同测试；
3. 校验本地 `NetID-v1.0.zip`，若不存在则从冻结 Zenodo URL 断点式下载；
4. 原子安装并校验公开 NetID source；
5. 重建并复核作者公开 audit；
6. 用冻结官方 DreaMS checkpoint 编码公开 mouse-liver MS2；
7. 执行 component-isolated edge-signal audit。

服务器日志固定写到仓库根目录：

- `/data02/run01/scv7tsl/DreaMS/netid_dreams_edge_<jobid>.out`
- `/data02/run01/scv7tsl/DreaMS/netid_dreams_edge_<jobid>.err`

核心结果：

- `data/validation/netid_dreams_edge_signal_20260831/report.json`
- `data/validation/netid_dreams_edge_signal_20260831/edge_matched_nonedges.csv.gz`

若服务器不能访问 Zenodo，需手动同步本地冻结 archive：

- `data/external/netid_v1/NetID-v1.0.zip`（155,278,201 bytes）

不需要上传 `data/reference/OEP003284_raw/`；该目录只属于当前被阻塞的完整 KGMN 路线。

## 7. 通过后的下一步（尚未执行）

通过只说明 DreaMS 在 NetID feature graph 中存在独立谱学边信号。下一步仍必须：

1. 以 graph component 完全隔离做 cross-fitting；
2. 用 degree-preserving / mass-RT-matched decoys 校准 edge probability；
3. 固定阈值后比较三臂：作者图、DreaMS edge、作者证据与 DreaMS intersection；
4. 在固定 target-decoy/FDR 下报告 coverage、edge precision proxy、组件冲突与 ion-family 冲突；
5. 获得独立标注数据后，才能报告 annotation accuracy；没有独立结构真值时不能声称超过 NetID 或达到 SOTA。

该阶段的潜在创新点只能是“component-isolated、FDR-calibrated DreaMS edge reliability 接入成熟全局网络”，不是简单把 DreaMS cosine 加权到旧网络分数上。

## 8. 2026-09-01 本地正式结果

本地 CPU 已完成全部 810 张谱的官方 DreaMS 编码：

- 808 个唯一 NetID features；
- embedding 维度 1,024；
- 最大单位范数误差 `1.788e-07`；
- official checkpoint SHA256 保持为 `8928f908606c0bd652c5a4107d3c35102f660622958c225a1f625abe4b1ba245`；
- 编码未使用作者 annotation、identity label、phenotype 或 P2b。

正式 edge gate 失败：

| 分层 | 边数 | DreaMS 边均值 | matched nonedge 均值 | Δ | AUC | component-bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| overall | 188 | 0.2299 | 0.2186 | +0.0113 | 0.5267 | [-0.0064, 0.0324] |
| biotransform | 111 | 0.1907 | 0.1767 | +0.0140 | 0.5287 | [-0.0126, 0.0475] |
| ion phenomenon | 77 | 0.2863 | 0.2788 | +0.0075 | 0.5227 | [-0.0243, 0.0407] |

随后在完全相同的边与 matched nonedges 上进行冻结的失败机制诊断。188 条作者边中仅 7 条带作者 `score_MS2_similarity`；181 条主要来自质量差、RT、同位素/加合物/碎片规则和全局求解。结果显示：

- biotransform 的预注册 modified cosine：Δ = -0.0032，AUC = 0.5197，CI = [-0.0481, 0.0357]；
- ion-phenomenon 的预注册 direct peak cosine：Δ = -0.0118，AUC = 0.4916，CI = [-0.0540, 0.0287]；
- direct、neutral-loss、modified 三种 raw peak score 在 overall 层均无正向可识别信号。

正式诊断是：

> `public_post_solution_edges_not_identifiable_by_preregistered_raw_ms2_scores`

因此禁止进入 component-cross-fitted DreaMS edge calibrator。该结果否定的是“把 NetID post-solution 全局关系边当作 MS2 相似度标签”的接口，不是否定 NetID 的 MS1/ion-family 全局一致性价值，也不否定 DreaMS 在独立结构候选检索任务中的价值。

下一步架构边界调整为：

1. NetID 只承担 MS1 feature 的 isotope/adduct/fragment/biotransformation 全局一致性与去冗余；
2. DreaMS 只在存在真实 MS2 且有独立结构候选与真值的 feature 上承担结构级排序；
3. 两模块只能通过经过独立标注校准的候选置信度接口衔接，不能用 NetID 自己的 post-solution edge 反向监督 DreaMS；
4. 没有公开完整 pre-solution ILP 状态时，不宣称完整 NetID 复现或 SOTA。

冻结本地结果：

- `data/validation/netid_mouse_liver_dreams_20260831/report.json`
- `data/validation/netid_dreams_edge_signal_20260831/report.json`
- `data/validation/netid_edge_signal_modalities_20260901/report.json`

## 9. Positive-mode 独立补充与最终裁决（2026-09-01）

原审计只覆盖 `Mouse_liver_neg`。公开 release 还包含完整的 `Mouse_liver_pos`，因此在不改变任何阈值和匹配规则的前提下进行了独立极性复现。

Positive-mode 数据：

- 169 个 targeted-MS2 workbooks；
- 2,029 个 targeted requests；
- 1,336 张至少 3 个 fragment peaks 的谱；
- 1,316 个唯一 features；
- 31,349 个 fragment peaks；
- official DreaMS embedding 最大 norm 误差 `1.192e-07`。

Positive-mode edge gate：

| 分层 | 边数 | DreaMS 边均值 | matched nonedge 均值 | Δ | AUC | component-bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| overall | 2,965 | 0.3665 | 0.2418 | +0.1247 | 0.6793 | [0.1195, 0.2821] |
| biotransform | 2,835 | 0.3657 | 0.2406 | +0.1251 | 0.6792 | [0.1201, 0.3143] |
| ion phenomenon | 130 | 0.3845 | 0.2668 | +0.1177 | 0.6816 | [0.0959, 0.2272] |

Positive-mode 通过初始 edge gate，但不能外推到 negative mode。

### 9.1 组件与作者 MS2 证据鲁棒性

- 最大组件包含 2,776/2,965 条边（93.6%），其 Δ 仍为 +0.1178；
- 删除最大组件后剩余 189 条边，Δ = +0.2264；
- 65 个组件中 84.6% 的组件均值为正；
- equal-component bootstrap mean = +0.2180，95% CI = [0.1646, 0.2705]；
- 1,053 条边已有作者 `score_MS2_similarity`，其 Δ = +0.2536；
- 其余 1,912 条没有作者显式 MS2 score 的边仍有 Δ = +0.0538，95% CI = [0.0494, 0.1734]；
- quartet 内随机化的 overall 与 no-author-MS2 empirical p 均为 `9.999e-05`；
- 8,895 个 decoy rows 对应 6,735 个唯一 decoy pairs，最大复用 9 次。

该结果允许进入 component-cross-fitted calibration，但仍不是独立 edge truth。

### 9.2 Component-cross-fitted fixed-FDR 校准失败

严格外层组件隔离后，每折在训练 feature 宇宙内重新生成 matched decoys；held-out 组件的 feature 不允许进入训练正边或负边。阈值只在训练折选择。

结果：

- OOF AUC = 0.6760；
- 所有 fold 的 DreaMS coefficient 均为正；
- 训练目标 FDR=5% 时，外折 coverage=1.38%，FDR proxy=12.2%；
- 训练目标 FDR=10% 时，外折 coverage=10.56%，FDR proxy=24.5%；
- 训练目标 FDR=20% 时，外折 coverage=21.21%，FDR proxy=27.2%；
- 在没有作者显式 MS2 score 的 1,912 条边中，10% 目标只保留 16 条（0.84%），FDR proxy=137.5%；即使目标放宽到 20%，coverage 也只有 5.13%，FDR proxy=58.5%。

因此：

> Positive-mode DreaMS cosine 与作者 post-solution edge membership 有强相关，但该相关无法在 graph-component distribution shift 下形成可靠的 fixed-FDR edge calibrator。它不能进入 NetID overlay，更不能成为注释性能或 SOTA 结果。

该失败说明作者 post-solution 图包含大量组件特异和已有 MS2 选择信息。用它训练复杂模型只会更强地复刻作者求解结果，而不会提供可信的新 edge probability。

最终架构裁决：

1. `Mouse_liver_neg`：edge signal gate 失败；
2. `Mouse_liver_pos`：相关性 gate 通过、robustness 通过，但 component-cross-fitted fixed-FDR calibration 失败；
3. DreaMS-to-NetID post-solution-edge overlay 分支正式终止；
4. NetID 仍可作为正交的 MS1/ion-family 全局一致性模块，但当前公开 release 缺完整 pre-solution ILP state，不能本地生成新的严格 NetID 解；
5. DreaMS 的结构排序增量必须在独立结构真值、真实候选组和未消费测试集上评价，不能由 NetID 自身预测图提供监督。

Positive-mode 冻结结果：

- `data/validation/netid_public_positive_ms2_20260901/report.json`
- `data/validation/netid_mouse_liver_positive_dreams_20260901/report.json`
- `data/validation/netid_positive_dreams_edge_signal_20260901/report.json`
- `data/validation/netid_positive_edge_robustness_20260901/report.json`
- `data/validation/netid_positive_edge_calibration_20260901/report.json`
