# Enveda-180 冻结外部基准构建结果

**日期：** 2026-10-09  
**服务器作业：** `2354608`（`prep_gnps_e180`）  
**状态：** `ENVEDA180_CROSS_CONDITION_BENCHMARK_FROZEN`（构建审计；已被更严格的 v2 重建要求取代）
**证据来源：** 服务器标准输出；本地尚未回传并重新计算服务器产物哈希。

> 2026-10-10 更新：该作业未载入后来登记为 required 的 MSnLib 排除源，因此不得作为最终一次性外测面板。正式面板必须由当前 `tasks/unified_consumed_sources_v1.json` 的 fail-closed 六源合同重新构建到 `enveda180_*_v2_20261010`，缺任一实际登记来源即停止。MassBank 以已消费面板的完整身份/分子式元数据排除；仓库不存在的全量 MGF 不再作为硬依赖，其谱图哈希缺口列为残余边界。本文以下数字只描述作业 2354608 的构建审计，不授权模型评分。

## 1. 数据角色裁决

- GNPS Gold/Silver 10 ppm 清洗集继续登记为开发资源：
  - `evaluation_role = development`
  - `truth_status = consumed`
  - `allow_final_claim = false`
- Enveda-180 登记为尚未评分的封存外部测试资源：
  - `evaluation_role = sealed_external_test`
  - `truth_status = frozen_unscored`
  - `allow_final_claim = true`
  - `construction_model_blind = true`
  - `performance_scores_opened = false`

本次作业没有加载模型，也没有打开任何方法的性能结果。后续不得依据 Enveda-180 的得分重新选择模型、权重、模块或阈值。

## 2. 原始文件与完整性

Enveda-180 输入文件：

```text
data/external/enveda180_20260713/enveda-180-filtered.mgf.gz
bytes = 285632591
md5   = 6d06a2c916aefe1c0b2d86bf40628441
```

原始 MGF 共含 **1,175,202** 条谱图记录。

## 3. 无模型元数据审计

审计产物：

```text
data/validation/enveda180_scoreblind_manifest_v1/
```

主要计数如下：

| 项目 | 数量 |
|---|---:|
| 原始谱图记录 | 1,175,202 |
| 元数据合格且去除同身份完全重复后的记录 | 569,890 |
| 主加合物记录（`[M+H]+`、`[M-H]-`） | 417,865 |
| 前体质量通过结构一致性验证的主加合物记录 | 417,865 |
| 缺少可解析碰撞能 | 236,640 |
| 峰数不足 | 368,586 |
| 同一身份完全重复谱 | 86 |
| 跨身份共享相同谱图哈希 | 20,150 |
| 与已消费身份重叠的记录 | 6,708 |
| 与已消费分子式重叠的记录 | 444,477 |

碰撞能分布为 20、40、60 三档，合格记录分别为 105,504、226,212、238,174；正、负离子记录分别为 462,384、107,506。

### 实际载入的排除源

| 来源 | 状态 | 新增身份数 | 新增分子式数 |
|---|---|---:|---:|
| MassSpecGym HDF5 | 已载入 | 28,929 | 17,634 |
| GNPS Gold/Silver manifest | 已载入 | 124,335 | 18,316 |
| MoNA positive MGF | 已载入 | 993 | 0 |
| MoNA negative MGF | 已载入 | 2,045 | 0 |

合并后排除账本包含 **156,302 个身份**和 **35,950 个分子式**。

以下两个可选来源在服务器上不存在，因而本次没有用于重叠排除：

```text
data/validation/bioaware_b44_massbank_panel_localcheck_20260913_v4/reference_library.csv.gz
data/validation/chemaware_msnlib_external_manifest_v2/eligible_records.csv.gz
```

因此，“封存外部测试”的严格去重范围只覆盖本次实际载入的 MassSpecGym、GNPS 和 MoNA 资产；不能在论文中无条件写成“与项目使用过的所有公共谱库完全零重叠”。

## 4. 冻结跨碰撞能基准

最终目录：

```text
data/validation/enveda180_cross_condition_benchmark_v1/
```

构建规则：

- 查询谱取同一身份的最高碰撞能记录；
- 正确参考谱来自不同且更低的碰撞能；
- 候选窗口为前体质量 **10 ppm**；
- 每个面板最多 25,000 个查询；
- 查询通过固定哈希种子 `20261008` 进行模型盲抽样；
- 每个候选分子最多保留 3 张参考谱；
- 单查询候选分子数上限为 256；
- 跨身份相同谱图哈希被排除；
- 正确候选固定置于第一块，后续评价必须继续执行“并列对正例不利”。

### Identity-disjoint 面板

| 项目 | 数量 |
|---|---:|
| 查询 | 25,000 |
| 查询身份 | 24,421 |
| 查询分子式 | 8,965 |
| 候选分子 | 1,202,361 |
| query-reference 谱图关系 | 2,847,978 |
| 含同分子式近结构干扰的查询 | 23,238（92.95%） |

平均每个查询约含 **48.09 个候选分子**和 **113.92 条候选谱关系**。

### Formula-disjoint 面板

| 项目 | 数量 |
|---|---:|
| 查询 | 25,000 |
| 查询身份 | 22,253 |
| 查询分子式 | 9,128 |
| 候选分子 | 293,785 |
| query-reference 谱图关系 | 677,501 |
| 含同分子式近结构干扰的查询 | 19,820（79.28%） |

平均每个查询约含 **11.75 个候选分子**和 **27.10 条候选谱关系**。

两个面板合并后实际裁剪保留 **340,765 张谱图**。

这里的 `identity_disjoint` 和 `formula_disjoint` 指相对已消费排除账本的身份/分子式隔离，不表示同一查询的候选池内不存在同分子式干扰；相反，近结构查询比例是本基准的重要设计特征。

## 5. 冻结产物哈希

```text
97834b68950d4edad5bf83ebaf6a7866f247fef37ca4eca0a1962cde092b0c83  spectra.mgf
a74bc737c1ed7da9145029dd50228978dac28f429ae718e208ed9a6704b410d6  manifest.csv.gz
ae65994411c957073d07eeefccf54f34500b5292bf1c89fccac9ca6f34806ffd  panel_identity_disjoint.npz
d70e63081d4eeedbac44108cab669eec22825f03d1cb263e973993fd3b4fc940  panel_formula_disjoint.npz
aa69cab248d337dd4be3e097a5ff1c15e8a4b571f02312567790dc35e794e932  pairs_identity_disjoint.npz
4a39b5920c6114d684c969c132519c6620db9fdad5966b2e51e9a2c3a5936cfa  pairs_formula_disjoint.npz
```

## 6. 科学解释与边界

本次结果完成的是一个大规模、有结构真值、跨碰撞能、近结构富集且尚未评分的候选检索基准。它明显比现有 GNPS 开发面板更大：identity 面板候选分子数超过 120 万，且 92.95% 查询包含同分子式干扰，适合检验 Noise、ChemAware、P2b、WSE及统一模型在真实困难候选空间中的表现。

但本次没有产生任何算法性能结果，也没有证明统一模型优于现有方法。`allow_final_claim = true` 表示该资源具备最终评价角色，不等于未来任意结果都自动具备最终声明资格。最终评价前仍须先冻结模型、模块清单、权重、阈值、缺失模块处理和主次终点；打开结果后不得回调。

另需保留以下限制：

1. 本次未纳入缺失的 MassBank 和 MSnLib 排除源；
2. 数据来自 Enveda-180 的特定采集与过滤流程，外推到公共实测谱库、不同仪器和真实队列仍需谨慎；
3. 本基准验证的是谱库候选检索和跨碰撞能迁移，不包含真实样本内 BioAware 上下文；
4. 25,000-query 上限和固定哈希抽样属于预评分计算约束，必须与结果共同报告。

## 7. 本轮终止点

服务器最终输出为：

```text
GNPS_DEVELOPMENT_READY: data/validation/unified_benchmark_assets/gnps_development_v1.json
ENVEDA180_EXTERNAL_FROZEN_UNSCORED: data/validation/enveda180_cross_condition_benchmark_v1
```

按用户要求，本记录止于基准构建与证据边界归档；不在本轮继续评分、训练、融合或启动后续作业。
