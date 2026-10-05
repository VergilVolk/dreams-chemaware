# DreaMS服务器存储清理裁决

日期：2026-09-22  
状态：本地证据驱动的预清理清单；服务器只读盘点前不授权删除  
原则：负结果的结论资产保留，负结果产生的可再生重型训练状态可以瘦身。

## 1. 结论

当前最可能释放空间的是旧 `.pt`、`.ckpt`、optimizer状态、重复embedding/token cache、smoke输出和中断传输文件，而不是 `report.json`、逐query评价、manifest、哈希或负结果账本。

本地 `data/validation` 镜像显示：

- `.pt` 约39.85 GiB；
- `.ckpt` 约6.94 GiB；
- `.npy` 约20.09 GiB，其中16.76 GiB是仍被当前ChemAware入口使用的`tokens_f16.npy`；
- B47两套原始MS/MS约3.22 GiB，属于当前prospective资产。

这些数字不能代替服务器实际盘点，但足以确定清理顺序。

## 2. 永久保留或当前禁止删除

### 2.1 全项目基础资产

- `data/e1/official_embedding_slim.pt`及其SHA256账本；
- native DreaMS官方初始化checkpoint；
- corrected candidate graph、official baseline ledger、formula/split/tie manifests；
- 所有`report.json`、`decision.json`、逐query评价表、claim ledger、artifact manifest、SHA256、seal、lock和关键Slurm stdout/stderr；
- 任何已经打开outer或承担one-time语义的seal/lock，无论作业成功或失败都不得删除。

### 2.2 当前ChemAware生产依赖

- `data/validation/chemaware_corrected_manifest_tokens_v1/`。其中`tokens_f16.npy`约16.76 GiB，但仍被canonical outer、native triplet、residual stage-2和多个正式评价入口引用；当前不能删除。
- `data/validation/chemaware_truthblind_candidate_policy/run_2338337/`；
- `data/validation/chemaware_high_coverage_native/run_2340524/evidence/`；
- `data/validation/chemaware_high_coverage_native/run_2340524/triplets/`；
- `data/validation/chemaware_high_coverage_native/run_2340524/official_role3_preflight.json`；
- `data/validation/chemaware_high_coverage_native/run_2340721_resume_2340524/training/best.ckpt`；
- `data/validation/chemaware_high_coverage_native/run_2340721_resume_2340524/role3_evaluation_after_stop.json`；
- 2026-09-22 residual native triplet Stage-2的official cache、evidence、训练结果和null controls，直到该阶段正式封存。

### 2.3 当前Noise生产依赖

- native Noise合同所需的两个官方native checkpoints；
- mature-E8的canonical comparator：`g8r_noise_final_e8_direct_transfer/...e8_baseline_symmetric_shared/seed_20260830/fold_0/final_shared_encoder.pt`；
- 七来源32,114-row正式action provenance ledger及其alias/routing manifests；
- corrected graph与最终held评价分母；
- 若旧exact-injection V3仍需作历史复算，则只保留其明确列出的`n_bank`和best-v5 ledger；不得以此为理由保留全部旧sweep checkpoint。

### 2.4 当前BioAware生产依赖

- B47 18工件、638,372,014-byte provenance闭环；
- ST001122与ST003356原始MS/MS、truth-blind joins、candidate graph、official embeddings、seed artifact；
- U0、U1c、U2和U3的冻结ledger、report与哈希；
- B44--B46最终报告和最小逐query负结果证据。它们是论文的shortcut/negative-evidence资产，不能因结果为负而整目录删除。

## 3. 第一批高置信瘦身对象

以下对象可以进入服务器核对清单，但删除前仍须确认服务器路径、大小、mtime、是否被运行中作业打开，以及本地或归档副本SHA256。

### 3.1 ChemAware high-coverage native重复checkpoint

- `run_2340721_resume_2340524/training/last.ckpt`：已被role-3 seal明确判为late drift，不是release candidate。本地已恢复并登记哈希后，服务器副本可删除；保留评价JSON和哈希。
- `run_2340524/training/best.ckpt`与`last.ckpt`：当前正式模型来自resume run，stage-2只依赖resume best。若服务器端与本地SHA256一致，可删除服务器上的这两个早期训练checkpoint；保留该run的evidence、triplets和preflight。
- `run_2340560/training/`：不在canonical seal或当前stage-2依赖链中。先保留其日志、配置、summary和hash，再删除`best.ckpt`、`last.ckpt`及optimizer状态。
- 任意`.filepart`：只有在同名完成文件存在、大小与SHA256已核对且没有活跃传输时才删除。

按本地镜像估计，上述ChemAware瘦身可释放约3.5--5.8 GiB，同时保留canonical `best.ckpt`。

### 3.2 旧Noise checkpoint矩阵

- `cross_condition_m3/seed_*/best_m3.pt`；
- `noise_consistency_pilot/seed_*/best_noise_consistency.pt`；
- `noise_isomer_infonce*/**/best_infonce.pt`与`last.pt`；
- `g8r_noise_final_e4a_direct/`中非canonical超参数、optimizer、seed和fold checkpoint；
- `g8r_noise_final_e8_direct_transfer/`中除`e8_baseline_symmetric_shared/seed_20260830/fold_0`外的candidate/combined/official-reference/stopgrad训练checkpoint；
- 旧23,876-query图上的direct-boundary、E13--E16、PN scan与smoke checkpoint。

这些路线已经被corrected-graph与2026-09-20 native-triplet合同替代。保留：summary、gate evaluation、逐query结果、artifact manifest、关键日志，以及论文中仍引用结果所需的最小canonical checkpoint。删除前必须生成“保留文件清单”，不能整棵删除`g8r_noise_final_e4a_direct`，因为其中仍混有历史canonical checkpoint和报告。

仅本地显式大目录估计：旧E4-A约11.78 GiB、E8矩阵约3.05 GiB、cross-condition/consistency/InfoNCE约7.7 GiB；多数空间来自模型权重。

### 3.3 旧ChemAware训练扫描与smoke模型

旧candidate-residual、ICEBERG direct/shared、shared-v2/v3、PEFT、adapter、prefix、mass-teacher与各类`*_smoke*`路线中的模型权重可进入清理候选。保留每条关闭路线的：

- 最终report/decision；
- 选择与对照所需的小型NPZ/CSV；
- 数据、代码和checkpoint哈希；
- 若主文或补充材料直接引用，保留对应逐query评价。

不保留普通smoke checkpoint、optimizer状态、重复prefix cache或失败训练的`last`权重。`rule_mass`正式inner、canonical candidate policy、native-triplet和当前residual-stage2不在此清理范围。

### 3.4 临时与不完整工件

- 空目录；
- 已有完整同名文件且哈希一致的`.filepart/.partial/.part`；
- seal前失败、无report/decision、无独立科学信息的随机staging目录；
- Slurm临时scratch副本，前提是正式output已原子发布并校验；
- pytest/smoke临时目录中不含唯一失败证据者。

## 4. 负结果应怎样保留

一个已关闭实验至少保留：

1. `report.json`或`decision.json`；
2. 冻结配置、输入路径和代码哈希；
3. paired/per-query评价或足以重新计算结论的小型表；
4. corrected/introduced、CI、停止门与失败类型；
5. stdout/stderr或关键错误摘要；
6. 若checkpoint不是后续依赖，只保留checkpoint SHA256、大小、训练配置和结果摘要，不必在服务器保留多份400 MiB--1.2 GiB权重。

负结果的checkpoint通常可删，负结果的证据账本不能删。

## 5. 服务器执行顺序

1. 在服务器仓库根目录运行：

   `bash tasks/audit_server_storage_readonly.sh data/validation > server_storage_inventory.tsv`

2. 同时记录运行中/排队作业，任何被活跃作业引用的路径暂不清理。
3. 将盘点文件带回本地，根据真实绝对路径、大小和mtime生成逐路径删除manifest。
4. 每个候选先生成最小保留包与SHA256；至少保留两个独立副本中的一个。
5. 先清`.filepart`、staging、空目录和明确非canonical checkpoint；观察空间回收后再决定是否清旧大缓存。
6. 删除只使用逐条绝对路径，不使用`*`、`find ... -delete`或整个`data/validation`级别的递归操作。

## 6. 当前不能凭本地直接裁决的对象

- 服务器上是否存在本地没有同步的run、恢复目录或唯一日志；
- 哪些checkpoint仍被排队作业引用；
- scratch与project filesystem是否存在重复副本；
- 文件是否为硬链接/稀疏文件；
- server-only B47 registry和Noise七来源ledger的真实路径与大小。

因此，本文件给出科学资格与优先级，但真正删除必须等服务器只读inventory返回后再冻结精确manifest。
