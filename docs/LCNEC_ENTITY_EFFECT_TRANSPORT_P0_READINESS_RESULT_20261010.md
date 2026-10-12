# LCNEC 实体效应迁移 P0：首次执行与就绪审计

日期：2026-10-10  
裁决：**分析实现通过；LIPn 内部实体可行性门通过；HSST3n→LIPn 全局运输覆盖门失败。只保留窄技术验证，不再作为广覆盖生物学主线。**

## 已经实际完成

- 实现 `tasks/analyze_entity_effect_transport.py`；
- 实现并运行 `tasks/test_entity_effect_transport.py`；
- 合成合同测试通过：唯一强链接同向效应可复现；反向效应和 no-match 不复现；歧义链接受到冻结权重惩罚；链接表出现 phenotype 字段或冻结文件哈希变化时强制失败；
- 实现通用 CPU sbatch `tasks/run_entity_effect_transport.sbatch`，所有输入必须显式提供，输出按 Slurm job ID 隔离；
- 默认在完整发现实体宇宙使用 BY，而不是在事后筛选子集上 BH。
- 对 LIPn 85/85 个 mzML 完成流式 acquisition audit；采集门正式通过，仅授权表型盲实体构建。
- 在仅含 17 个技术对照注射、完全不读取 68 个 study 文件标签的条件下完成 LIPn QC headroom：1,586 个前体-RT family 中 382 个同时通过 QC 重现性、blank 洁净度和 dilution 响应硬门。
- 冻结 HSST3n 263/263 和 LIPn 382/382 个合格实体的代表 QC MS/MS；5 ppm 表型盲候选审计仅覆盖 6/263 个发现实体（2.28%），未达到全局运输 headroom。

### LIPn 正式采集审计

结果目录：`data/validation/lcnec_lipn_acquisition_gate_20261010/`

| 分区 | 文件数 | MS2 总数 | 每文件 MS2 中位数 | 每文件四位小数唯一前体中位数 |
|---|---:|---:|---:|---:|
| study | 68 | 184,116 | 2,708 | 1,803.5 |
| pooled QC | 9 | 24,362 | 2,707 | 1,859 |
| blank | 2 | 5,430 | 2,715 | 1,552 |
| QC dilution | 6 | 16,256 | 2,709 | 1,784 |

`pass_to_feature_headroom=true` 只说明 LIPn 有足够 DDA/MS2 进入独立 feature/ion-family 构建。blank 同样有大量 MS2，因此 blank/QC/dilution 过滤是必要硬门；不能把 DDA 数量直接解释为生物代谢物数量。

### LIPn 表型盲 QC headroom

结果目录：`data/validation/lcnec_lipn_qc_headroom_gate_20261010/`

- 输入：9 个 pooled QC、2 个 blank、6 个 QC dilution，共 17 个技术注射；`phenotype_labels_read=false`；
- 观测到 46,048 条 MS2 event，聚合为 1,586 个前体-RT family；
- 714 个满足 QC 重现性，1,077 个满足 blank 洁净度，508 个满足 dilution 单调响应；
- 382/1,586 个（24.09%）同时通过三门；参数为 5 ppm、15 s、至少 6 个 QC、blank/QC≤0.20、dilution Spearman rho≥0.70；
- `pass_to_author_overlap_audit=true` 仅允许继续做表型盲代表谱图与 open-set 链接，不是效应复现或生物学阳性。

### 跨平台候选空间审计

结果目录：`data/validation/lcnec_entity_candidate_headroom_20261010/`

- HSST3n：263/263 个 QC 合格实体成功冻结代表 MS/MS；LIPn：382/382；
- 5 ppm 下只有 6 个候选对，覆盖 6/263 个 HSST3n 实体（2.28%）；每个发现实体仅一个质量候选，但只到达 5 个 LIPn 实体，其中一个验证实体被两个发现 family 命中；
- 未校准 sqrt-intensity cosine 中位数 0.711，3/6 ≥0.70；这些值没有 null calibration，不能转写为 `match_p` 或身份；
- `pass_to_global_transport=false`。即使后续 6 对均复现，也只能证明少量同离子技术迁移，不能证明该方法在两个正交平台间具有广覆盖增量。

## 本地真实数据审计

| 工件 | 当前状态 | 角色 |
|---|---|---|
| LIPn raw zip | 完整存在；336,864,469 bytes；SHA-256 `6badd39a687d2b646f57c21a90115ef3f2fe6c58b439726c579bca15d97acfd1` | 原始技术验证平台 |
| LIPn overview | 存在 | 85 injections 的样本/QC设计入口 |
| Stage-1 `primary_decomposition.npz` | 本地缺失 | 必需发现端 34×263 患者效应与实体轴 |
| Stage-1 EIC matrix | 只有 32,752-byte `.filepart` | 不可读取、不可作为结果 |
| LIPn acquisition gate | 85/85 文件通过；QC 与 study 均有充分 DDA | 只授权表型盲 feature/ion-family 构建 |
| LIPn 独立 ion-family ledger | 1,586 个 family；382 个通过三项技术门 | 表型盲验证实体宇宙已具备 |
| LIPn representative MS/MS manifest | 382/382 已冻结 | 跨平台链接的谱学输入 |
| HSST3n representative MS/MS manifest | 263/263 已冻结 | 发现端谱学输入 |
| 5 ppm candidate headroom | 6/263（2.28%）；3/6 cosine≥0.70；未校准 | 全局覆盖门失败，仅窄技术审计 |
| HSST3n↔LIPn calibrated link p/weights | 不存在 | 必需 open-set 对应证据 |
| LIPn patient-level effect table | 不存在 | 必需验证效应 |

因此，实体发现阶段已由 `INPUT_NOT_EVALUABLE` 前进到 `PHENOTYPE_BLIND_ENTITY_UNIVERSE_READY`；但跨平台全局运输路线得到明确的 headroom 阴性。6 对候选的效应迁移仍是 `INPUT_NOT_EVALUABLE`，既不能当阳性，也没有必要为制造大数字而放宽质量窗或引入未经校准的加合物转换。

## 为什么不能沿用旧 Stage-2

旧量化脚本把 HSST3n 冻结的绝对 RT 和 263 个 target 直接应用到 LIPn。跨色谱平台绝对 RT 不具可交换性；这会把“在预设窗口是否出现信号”当成实体迁移。旧分析还允许 LIPn 缺失时跳过 2A 后继续总裁决，并用 Top-1 cosine 0.80 定义 dark。三点均违反新合同。

## 旧 12 个跨平台阳性对照的用途

仓库有 12 个作者 HILICp 聚合层阳性对照：12/12 方向一致，历史报告 Spearman rho=0.902。它们是已命名的聚合效应对照，不具备：

- calibrated `match_p`；
- 一对多/no-match 候选集合；
- LIPn 患者级验证效应；
- unresolved entity 分区。

所以它们可以检查效应单位和方向转换，却不能冒充新方法的 formal link-and-replicate 结果，也不能证明 unknown 增量。

## 后续合法工作与路线降级

LCNEC 只允许对这 6 对候选做带外部 null calibration 的窄验证，再决定是否读取 34 对患者标签；不允许为扩大覆盖而事后放宽窗口。广覆盖的方法开发应迁移到采集化学相容、且有事件级 perturbation/tracer 终点的外部数据，而不是继续强迫 HSST3n 与 LIPn 两个正交平台共享实体宇宙。

