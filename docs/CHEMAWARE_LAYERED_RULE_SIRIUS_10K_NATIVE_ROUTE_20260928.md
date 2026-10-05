# ChemAware 分层化学规则、SIRIUS 候选证据与 10k 原生 triplet 路线

日期：2026-09-28

## 1. 结论先行

本轮不改 DreaMS encoder、预处理、cosine triplet-margin loss、Adam 或单正/单负采样。化学知识只做一件事：在冻结候选图上选择最值得纠正的错误候选，然后使用当前共享 embedding 的精确最大参考谱构造 `(q, p*, n*)`。

已完成两项本地正式资产：

- 分层规则语料由旧 335 条扩充为 8,132 条；新增 4,638 条固定版本 MS-FINDER 经验规则。
- 两个通过独立公式簇确认的静态来源形成 33,574 条候选级关系记录，覆盖 4,032 个训练 query；这些是可复用关系语料，不冒充新增 triplet。
- SIRIUS 真值盲三臂输入面板覆盖 4,032 个训练 query、7,939 个 query–formula 单元、16,787 个候选、6,977 个唯一结构。正确谱、强度置乱谱、质量平移谱各 7,939 份。

修正 query-row 映射后，静态规则中的旧 curated formula 与 MS-FINDER long-tail 两个来源通过公式簇资格门；MS-FINDER recurrent 和两个 confirmed-structure 规则未通过。通过来源在 official-geometry 容量审计中仅找到 3 个已被 Phase-A 覆盖的事件，新增为 0，因此不会被复制或放宽条件来凑数。新增化学 triplet 的主力仍是 SIRIUS fragmentation-tree 与 CSI:FingerID；两者分开评分、分开确认，且必须分别战胜强度置乱、质量平移和候选角色置换三条对照。

## 2. 规则库的层级与含义

正式语料：`data/validation/chemaware_layered_rule_corpus_v5_20260928/`

| 层级 | 数量 | 用途 | 训练资格 |
|---|---:|---|---|
| A：物理约束或独立确认 | 18 | 元素守恒、氮规则、偶电子偏好、同位素、adduct 一致性和已确认结构规则 | 仍需候选特异观测与匹配对照 |
| B：条件化人工/经验规则 | 438 | 旧 curated 规则、MS-FINDER 高频复现规则、9 类 HR 模板 | 只能作为候选选择先验 |
| C：需要校准 | 4,525 | 旧无条件质量观察与 MS-FINDER 长尾规则 | 未通过公式外校准前禁用 |
| Q：隔离观察 | 3,151 | 单条 MassBank 谱观察 | 权重为 0，永不直接训练 |

置信度权重是准入优先级，不是 loss multiplier。所有实际进入训练的 triplet 权重恒为 1。

### 2.1 MS-FINDER 扩充

固定来源为 RECETOX/MS-FINDER commit `78c7ef9697240da857b1fff643d9e53afb7308d0`：

- `NeutralLossDB_vs2.ndb`：1,643 条；
- `ProductIonLib_vs1.pid`：2,995 条；
- 合计 4,638 条，全部重新核对元素式与单同位素精确质量；最大误差 `4.02863e-6 Da`；
- 279 条满足“经验频率至少 1 且关联短 InChIKey 至少 2 个”，先验置于 B 层；其余 4,359 条置于 C 层；
- 所有规则在公式簇确认前均保持关闭。

MS-FINDER 的 HR、偶电子、同位素、产物离子、中性丢失和价态逻辑均作为可追溯来源保留。仓库 README 对 assembly 标注 CC-BY-4.0，同时 third-party notice 对源码标注 LGPL-3.0；内嵌经验表的再分发许可仍需项目层面复核，所以当前限定为内部研究资产并保留完整来源哈希。

## 3. 静态规则的修正确认结果与增量边界

在 role-0/1 内部按分子式划分的确认折上，正确规则必须同时战胜质量置乱和分子式置乱，且 top-1 差与 margin 差的 formula-cluster bootstrap 95% CI 下界都严格大于 0。

| 证据族 | 正确 top-1 | 对照 top-1 | 裁决 |
|---|---:|---:|---|
| 旧 curated formula | 8.563% | 6.422% / 5.199% | 两个对照的 top-1 与 margin CI 下界均大于 0，通过 |
| MS-FINDER 长尾 | 35.372% | 29.358% / 28.950% | 两个对照的 top-1 与 margin CI 下界均大于 0，通过 |
| MS-FINDER 高频复现 | 13.761% | 9.582% / 16.208% | 第二对照优于正确规则，失败 |
| 已确认结构规则 | 0.706% | 0 / 0.706% | margin 与第二对照无增量，失败 |

上述数字来自最终 profiled/collision-corrected v9：`data/validation/chemaware_layered_rule_source_qualified_v9_profiled_collision_20260928/`。旧 v4/v5 将 manifest query 序号误当成 HDF5 spectrum row，其百分比全部作废，不得引用。

“来源通过”只证明该来源具有总体候选特异性，不等于它能提供新的 active 边界。在 official-geometry 工程审计中，共有 1,478 条 active 边界，但只有 3 个候选对同时满足成对正确方向及两个 null 优势，而且三者都已存在于 Phase-A，故新增事件为 0。这一审计不是 Phase-A 性能评估，只用于提前否决“靠静态规则凑 1,000 个新增化学 triplet”的方案。静态规则继续作为分层证据和 ChemAware-RSI 语料；真正的结构—碎裂增量需由 SIRIUS 提供。

## 4. SIRIUS 模式

正式本地面板：`data/validation/chemaware_sirius_source_panel_v9_profiled_collision_20260928/`

每个训练 query 对候选图中的每个不同分子式生成一份固定分子式 `.ms`。真实分子式没有单独暴露；所有候选分子式都被完整枚举。每个单元同时生成：

1. 正确观测谱；
2. 保留 m/z 和强度多重集、仅置乱强度秩的谱；
3. 保留峰数和强度、对 m/z 做确定性正负平移的谱。

候选角色置换在结果导入后、完全不读取 truth 的情况下构造。TreeScore 只比较跨分子式候选；CSI:FingerID 只比较同分子式结构候选。两类分数不相加。

SIRIUS 输入按仪器分片：3,204 个 query 为 Orbitrap、750 个为 QTOF、78 个仪器缺失并使用官方默认 QTOF profile；对应 query–formula 单元为 6,665 / 1,136 / 138。正确谱与两个谱级对照在每个 profile 内基数完全一致。有碰撞能量时使用 SIRIUS 正式 `>collision` 字段，无碰撞能量时使用 `>ms2`。precursor 容差统一为 10 ppm以保持 DreaMS 候选图，MS2 容差分别为 Orbitrap 5 ppm、QTOF/default 10 ppm。

custom DB 全局同一分子式最多 19 个唯一结构，小于 `write-summaries --top-k-summary 50`，因此输出不会截断正式候选图。该门按全库分子式计数，不只看单个 query。SIRIUS 当前只支持单电荷离子；面板已对 adduct 做单电荷硬校验。

## 5. 资格门与 triplet 构造

一个证据族只有满足以下全部条件才可挖 triplet：

1. role-0/1 内部分子式不交叉确认；
2. 至少 100 个适用 query；
3. 正确证据相对强度置乱、质量平移、候选角色置换三条对照的 top-1 差和 margin 差，formula-cluster bootstrap 95% CI 下界均严格大于 0；
4. 对具体 true/false 候选，正确证据严格偏向 true，且该 pair 优势必须大于每一个对应 null 的 pair 优势；不要求该来源独自把 true 排到整张候选表第一；
5. 同级或更高置信层的反对证据否决该 pair；低于最强支持层的反对只写入 ledger，不得否决更强证据。A（SIRIUS）高于 B（curated），B 高于 C（long-tail）；层级不乘入 loss；
6. false 候选在受保护 Phase-A embedding 下仍位于激活的 margin 边界；
7. 对每个候选只取当前最大相似参考谱：`p* = argmax positive reference`，`n* = argmax selected false-candidate reference`。

化学源最低覆盖门冻结为 1,000 个不同候选边界、500 个 query。达不到就停止；不靠复制、reference 乘法或 replay 冒充化学证据。

## 6. 10k+ 语料的准确含义

最终目标池为 12,000 个原生 DreaMS event，至少 10,000。三层分别计数：

- 完整保留的 5,956 个 Phase-A event；
- 至少 1,000 个通过资格门的新化学 max-boundary event；
- 为覆盖与抗遗忘补足的、未经改写的 DreaMS 10-ppm replay。

只有第二层称为“新增化学监督”。总池达到 10k 不等于发现 10k 条独立化学规则。`corpus_event_manifest.tsv` 为每个 event 保留层级、谱行、query、candidate、source tag、curriculum role 和恒定 loss weight；`chemical_source_event_ledger.tsv` 额外保留支持该候选关系的证据族、delta 与低置信反对项。保护目录同时保存 static/SIRIUS `candidate_scores.tsv`、候选结构表、query-to-spectrum-row 注册表和完整规则库，供 ChemAware-RSI 使用。

## 7. 微调与评估

初始化为受保护的 `+2.1266 pp` Phase-A checkpoint。训练入口仍为：

- `dreams.utils.data.ContrastiveSpectraDataset`；
- `dreams.models.heads.heads.ContrastiveHead`；
- 单正/单负动态采样；
- cosine triplet-margin loss，margin `0.1`；
- Adam、全 backbone 更新；
- 1 张 GPU，无手动内存请求。

固定保存 250、500、750、1,000 step。只在 role-2 选择，并要求相对 Phase-A 的 formula-cluster CI 下界为正；未通过则保留 Phase-A 且不打开 role-3。通过后只对选定 checkpoint 打开一次 role-3。role-3 还必须满足 Recall@1 为正、公式簇 CI 下界大于 0、`corrected > 2 × introduced`，且 MRR、Recall@3、micro-AUC、macro-AUC 均不下降。失败结果进入 `protected_negative_result`，只有通过者进入 `protected_candidate`。role-4 始终不触碰。

## 8. 唯一提交入口

服务器需已有 SIRIUS 6 launcher，并已完成合法账号/许可配置。同步本轮代码和以下内部规则资产后，只提交：

```bash
sbatch tasks/run_chemaware_layered_sirius_10k_native.sbatch
```

该 sbatch 在同一个 Slurm allocation 中依次完成静态规则重建与确认、三臂 SIRIUS、公式簇资格、Phase-A 几何编码、多来源化学 triplet 挖掘、12k 分层池构造、原生微调、role-2 选择和有条件 role-3 确认。它申请恰好一张 GPU，没有 `#SBATCH --mem`。

## 9. 主要实现与资产

- `tasks/import_chemaware_msfinder_rule_tables.py`
- `tasks/build_chemaware_layered_rule_corpus.py`
- `tasks/build_chemaware_layered_rule_source_ledger.py`
- `tasks/export_chemaware_sirius_source_panel.py`
- `tasks/import_chemaware_sirius_controlled_source_ledger.py`
- `tasks/qualify_chemaware_candidate_source_ledger.py`
- `tasks/build_chemaware_multisource_native_triplets.py`
- `tasks/build_chemaware_layered_10k_native_pool.py`
- `tasks/run_chemaware_layered_sirius_10k_native.sbatch`
- `data/validation/chemaware_msfinder_rule_tables_v1_20260928/`
- `data/validation/chemaware_layered_rule_corpus_v5_20260928/`
- `data/validation/chemaware_layered_rule_source_qualified_v9_profiled_collision_20260928/`
- `data/validation/chemaware_sirius_source_panel_v9_profiled_collision_20260928/`

## 10. 参考边界

- MS-FINDER HR 论文：<https://pmc.ncbi.nlm.nih.gov/articles/PMC7063832/>
- RECETOX/MS-FINDER 固定源码：<https://github.com/RECETOX/recetox-msfinder>
- SIRIUS 6 CLI：<https://v6.docs.sirius-ms.io/cli/>
- SIRIUS 方法说明：<https://v6.docs.sirius-ms.io/methods-background/>
- SIRIUS custom database：<https://v6.docs.sirius-ms.io/cli-standalone/>
- SIRIUS 单电荷限制：<https://v6.docs.sirius-ms.io/faq/>
- Seven Golden Rules：<https://pmc.ncbi.nlm.nih.gov/articles/PMC1851972/>
- MZmine ion identity networking：<https://mzmine.github.io/mzmine_documentation/latest/module_docs/id_ion_networking/iin/iin.html>
