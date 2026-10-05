# BioAware B44：MassBank 冻结外部确认协议

日期：2026-09-13

## 1. 这一步回答什么

B42/B43 已在六个打开的开发域中证明：严格 KEGG 与独立、去 currency 的 Rhea
目录拓扑都能为 DreaMS 候选排序提供信号。B43 进一步表明，整体增益并非全部来自
“真值有收录、错误候选未收录”；Rhea/KEGG 共识在目录内竞争子集仍有正向信号。

这些结论仍属于打开开发集。B44 的唯一目的，是在完全独立的 MassBank 身份和分子式
上，对冻结工件做一次性外部检验。B44 不训练共享 encoder，也不评价样本内代谢传播。

## 2. 重要语义修正

历史 B37 的 `network_member`/`known_log_degree` 实际来自 MetDNA2 的 strict-KEGG
派生图，不是 Rhea。B42 已重新独立构建 Rhea 图并完成对账。因此后续名称统一为：

- `strict_kegg`：MetDNA2 strict-KEGG 派生图；
- `rhea`：从离线 Rhea participants 独立构建、去 currency、排除超大反应后的图；
- `BioAware-Catalogue v1`：strict-KEGG 与 Rhea 候选静态目录特征的非线性候选专家。

不得把 BioAware-Catalogue v1 写成 reaction propagation、sample-context 或 embedding
微调。

## 3. 已封存的 MassBank 面板

本地正式封存目录：
`data/validation/bioaware_b44_massbank_panel_localcheck_20260913_v4`

- 893 个 query / 893 个真值身份；
- 395 个真值分子式；
- 345 个负离子、548 个正离子 query；
- 14,867 条 query-reference 关系；
- 每个 query 至少两个同分子式、同离子模式、同 adduct 的候选身份；
- query 行在全局 reference library 中移除；
- query 身份、候选身份及真值分子式与 B42 打开宇宙隔离；
- 构建阶段没有读取 embedding、模型或结果分数。

MassBank HDF5 与配套 MGF 被逐行验证：37,082 条记录完全同序，InChIKey 前 14 位和
precursor m/z 均一致。10 条空峰谱被显式记录并排除。adduct 通过 full InChIKey、离子
模式和四位 precursor m/z 与 2024 元数据表唯一匹配；歧义或无映射记录不进入面板。

## 4. 已冻结的模型工件

本地工件目录：
`data/validation/bioaware_b44_catalogue_v1_localcheck_20260913_v1`

模型输入固定为：

1. DreaMS candidate molecule 最大 reference cosine；
2. strict-KEGG/Rhea 目录收录数；
3. 两目录共同收录指示；
4. 两目录 log-degree 均值；
5. 两目录 log-degree 最小值。

模型为 pairwise `HistGradientBoostingClassifier`。训练只使用六个 B42 开发域；部署门
由 B42 六域 OOF 预测一次性选择：baseline gap 不超过 0.05 且 proposal probability
不低于 0.55。B44 在冻结过程中没有被读取。

开发 OOF 在该全局冻结门下为 +6.98 pp（65 corrected / 5 introduced）。这是调门后的
开发指标，不是外部性能声明。

## 5. 一次性评价口径

因为部署门由负离子 B42 OOF 校准，B44 的正式 primary panel 是 345 个负离子 query；
正离子结果只作为 polarity-transfer 次级审计。每个候选分子的谱学分数为该 query 对
该分子所有剩余 reference spectra 的最大 cosine。

并列严格对真值不利。BioAware 只把一个冻结 proposal 移到 rank 1，其余 DreaMS 顺序
保持不变。报告 Recall@1/2/5/10/20、MRR、严格 rank 诱导的 macro query AUC、
corrected/introduced、formula-cluster bootstrap CI 和 McNemar exact p。

同时固定五个 query 内目录特征置换负对照；置换保留每个候选组的谱学分数和目录特征
多重集合，但破坏“哪个候选拥有哪个目录属性”的对应。

外部确认必须同时满足：

- 负离子 primary Recall@1 formula-cluster CI 下界大于 0；
- corrected 大于 2 倍 introduced；
- MRR formula-cluster CI 下界不小于 0；
- 真实目录特征优于五个置换负对照；
- 真实减平均负对照的 formula-cluster CI 下界大于 0。

## 6. 运行入口

单作业入口：`tasks/run_bioaware_b44_full_pipeline.sbatch`。B42 路径已经固定在脚本中，
无需提交环境变量；完整上传清单见 `tasks/bioaware_b44_upload_manifest.txt`。

它依次完成：metadata-only 面板封存、development-only 模型冻结、GPU 编码与一次性
外部评价。作业申请一张 GPU、不手工申请内存，输出/错误日志位于提交目录根部，所有
产物目录包含 Slurm job ID，禁止覆盖。

## 7. 决策边界

- 若 B44 primary 全部门通过：可称为“冻结候选目录专家在独立 MassBank
  同分子式检索中的外部确认”，仍不能称共享 embedding 改进或样本生物上下文算法。
- 若真实增益为正但特异性负对照不过：保留工程候选专家，但不能把增益归因于正确的
  KEGG/Rhea 身份拓扑。
- 若 primary 不显著或 corrected/introduced 风险门失败：停止该固定工件，不在 B44
  上调参；回到新的开发集研究候选覆盖与校准。
- 任何结果都不能直接证明反应机制、代谢通量、疾病机制或 SOTA。
