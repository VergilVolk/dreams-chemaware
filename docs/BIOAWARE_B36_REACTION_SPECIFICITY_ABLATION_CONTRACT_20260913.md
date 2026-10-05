# BioAware B36：反应网络特异性决定性消融合同

日期：2026-09-13  
状态：实现完成，等待服务器正式运行

## 1. 唯一科学问题

当前 BioAware 在打开的开发资源和 B35 真实谱库协议上有正向检索结果，但历史六域外层折全部选择了 `catalog_pairwise_hgb`，而没有选择 `reaction_context_pairwise_hgb`。因此不能把现有增益直接归因于 Rhea 反应关系。

B36 只回答：在完全相同的候选图、模型容量、外层隔离和内层门控协议下，候选特异反应上下文是否提供超过 DreaMS 谱学分数和目录覆盖先验的独立增益？

## 2. 四个真实证据臂

1. `spectral_only`：只读 DreaMS 谱学分数；
2. `spectral_plus_catalog`：谱学分数加网络成员、网络度数、质量窗口目录覆盖率和参考谱数量；
3. `spectral_plus_reaction`：谱学分数加真实反应路径、种子支持、一步/两步路径完整性与瓶颈证据；
4. `spectral_plus_catalog_plus_reaction`：目录与反应证据同时存在。

四臂都使用同一个浅层 pairwise HGB、相同超参数、相同候选差分训练和相同门控网格。每个臂独立在内层留来源 OOF 上选门，外层结果不参与选择，并允许显式 `no_op` 回退。

所有证据臂共用 B16 的同一模型随机流；正式统计前，`spectral_plus_catalog` 必须精确复现冻结 B16 的 860 queries、44 corrected、3 introduced、Recall@1 增益 4.7674 pp。这个门验证实现身份，不是新性能结果。若旧结果缺少 scikit-learn 版本而当前环境发生数值漂移，脚本保存完整结果并令科学门失败，而不是在长任务末尾抛异常；只有固定回原软件环境后才可恢复历史数值比较。

## 3. 两类反应证据阴性对照

- `within_query`：在每个候选组内循环错配完整反应特征向量；严格保持该 query 的反应特征多重集合，但打断“候选身份—反应上下文”的对应。
- `catalog_stratum`：在来源、网络成员、粗粒度网络度数、目录覆盖率和参考谱数量分层内错配反应特征向量；保持目录先验的粗粒度结构，但打断候选特异反应关联。

每类正式运行十个预固定随机重复。错配不读取 truth、正确性、干预结果或表型。它们是特征关联阴性对照，不冒充原始 Rhea 图的 degree-preserving edge rewire。

## 4. 隔离与统计

- 六个打开开发来源整体留一；
- 外层来源对应的 truth identity 和 formula 从训练中清除；
- 内层再做留来源门控选择；
- identity-cluster 与 formula-cluster 成对 bootstrap；
- corrected/introduced、风险净收益 `corrected - 2*introduced` 和 exact McNemar 同时报告；
- 所有固定臂全部报告，不按结果删除或改名；
- 科学门失败仍保存完整有效工件，不以异常退出伪装成工程崩溃。

## 5. 预注册通过门

只有同时满足以下条件，才进入原始 Rhea 图重连确认和共享 embedding 微调：

1. 完整真实反应臂相对 DreaMS 至少增加 3 pp；
2. corrected 大于两倍 introduced；
3. 六个外层来源均不退化；
4. 完整臂相对 catalog-only 的 identity/formula 成对 CI 下界都大于零；
5. reaction-only 相对 spectral-only 的 identity/formula 成对 CI 下界都大于零；
6. 完整臂超过两类错配对照的全部重复，且两类经验单侧 p 均不高于 0.10。

若失败，结论不是“BioAware 无价值”，而是：当前 5.9–6.2 pp 行为主要属于谱学/目录覆盖专家，尚无证据支持把反应上下文注入共享 embedding。此时保留保守重排器，但停止用“生物反应网络微调”包装它。

## 6. 执行工件

- `tasks/audit_bioaware_b36_reaction_specificity_ablation.py`
- `tasks/validate_bioaware_b36_reaction_specificity_ablation.py`
- `tasks/test_bioaware_b36_reaction_specificity_ablation.py`
- `tasks/run_bioaware_b36_reaction_specificity_ablation.sbatch`

正式输出目录由 Slurm job ID 唯一命名，不覆盖旧结果。服务器只需提交上述 sbatch；脚本包含依赖 fail-closed、单元检查、正式运行和独立算术验证。

## 7. 本地端到端工程预演

本地已用完整 860-query 六域候选图跑完四个真实臂、每类一个错配重复和 100 次 bootstrap，并由独立验证器通过覆盖、算术、门控数量、基线一致性、多重集合保持和工件哈希检查。该运行只是工程预演，不替代服务器的十重复、10,000 次 bootstrap 正式结果。

预演点估计为：spectral-only 0；catalog-only +5.581 pp（50/2）；reaction-only +1.395 pp（19/7）；full +5.000 pp（47/4）。full 相对 catalog-only 为 -0.581 pp，reaction-only 的 identity/formula CI 都未严格大于零。这是对“现有增益可能主要来自目录覆盖”的预警，不是正式科学裁决。

预演还发现冻结 B16 报告没有记录 scikit-learn 版本；当前 1.7.2 环境重训 catalog 臂得到 50/2，而冻结 B16 为 44/3。输入文件和脚本哈希一致，说明历史模型数值环境仍需在服务器复核。B36 因此将这一项作为显式科学门报告；不匹配时保存有效工件并阻止跨版本历史归因。

## 8. 正式结果（job 2336381）

工程验证通过，结果工件有效；科学门失败。

| 证据臂 | Recall@1 增益 | corrected / introduced | risk net（lambda=2） |
|---|---:|---:|---:|
| spectral-only | 0.000 pp | 0 / 0 | 0 |
| spectral + catalog | **+5.581 pp** | **50 / 2** | **46** |
| spectral + reaction | +1.395 pp | 19 / 7 | 5 |
| spectral + catalog + reaction | +5.000 pp | 47 / 4 | 39 |

关键成对结果：

- full 相对 catalog-only 为 **-0.581 pp**；identity-cluster CI `[-1.537, +0.355]` pp，formula-cluster CI `[-1.517, +0.346]` pp，McNemar `p=0.332`。当前反应上下文没有在 catalog 之上提供增量。
- reaction-only 相对 spectral-only 为 `+1.395 pp`，但 identity-cluster CI `[-0.116, +3.153]` pp、formula-cluster CI `[-0.110, +3.053]` pp 均跨零。query-level McNemar `p=0.029` 不能覆盖身份/分子式聚类依赖，因而不能作为正式通过证据。
- full 相对 spectral-only 为 `+5.000 pp`，两个聚类 CI 均为正；但 catalog-only 单独已经达到 `+5.581 pp`，所以该增益不能归因于反应网络。
- within-query 错配对照增益范围 `+2.791` 至 `+5.116 pp`，经验单侧 `p=0.273`；catalog-stratum 错配范围 `+2.442` 至 `+5.465 pp`，经验单侧 `p=0.273`。真实反应上下文没有超过错配上下文分布。
- 六个来源中，reaction-only 在 BV2cell 和 ST001154 上为负，在 KGMN200STD 上回退 no-op；反应信号缺乏跨来源一致性。

冻结 B16 复现门也失败：历史 B16 为 44/3、`+4.767 pp`，当前同输入重训为 50/2、`+5.581 pp`。这不推翻 B36 同一运行内部的证据臂比较，但禁止把当前 catalog 数值直接冒充历史 B16/B17 的精确复现；后续必须冻结模型工件和软件版本。

正式裁决：现有约 5--6 pp 的可部署候选修正主要是谱学分数加目录覆盖/候选熟悉度专家。当前聚合后的 Rhea 路径、种子支持和一步/两步路径标量没有证明候选特异反应增益，因此不得直接进入共享 embedding 微调，也不得包装成“反应网络改进 DreaMS 表示”。下一步只允许先做反应证据可观测性与信息损失审计，判断问题位于数据覆盖、标量聚合还是学习器。
