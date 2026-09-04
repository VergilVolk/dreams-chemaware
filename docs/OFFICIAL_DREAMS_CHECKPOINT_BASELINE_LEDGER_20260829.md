# 官方 DreaMS 微调 checkpoint：跨任务基线总账

**冻结权重：** `data/e1/official_embedding_slim.pt`  
**SHA256：** `8928f908606c0bd652c5a4107d3c35102f660622958c225a1f625abe4b1ba245`  
**用途：** 统一后续噪声微调、P2b、BioAware、解释性与生物学应用的官方基线口径。  

> 不同表格行采用不同 query、候选图、加合物和难度分层，数值不可横向当作同一测试集，也不可相互拼接。比较新模型时必须在同一行、同一冻结协议内成对比较。

## 1. 核心谱库检索任务

| 任务/协议 | Query数 | Recall@1 | MRR | Macro-AUC |
|---|---:|---:|---:|---:|
| G5/G6/G7全量gate | 9,289 | 0.930132 | 0.960120 | 0.867554 |
| G8R locked困难跨条件检索 | 620 | 0.808065 | 0.895565 | 0.873253 |
| P2/P2b formula-OOF开发图 | 5,037 | 0.860631 | 0.916099 | — |
| P2/P2b near子集 | 2,094 | 0.761223 | — | — |
| 全错误图谱 | 23,876 | 0.924401 | 0.956134 | — |
| 全错误图谱near子集 | 13,784 | 0.895096 | — | — |

全错误图谱包含1,805个官方Top-1错误，其中1,446个属于near子集。

G8R locked额外基线：

- 跨条件正例余弦：0.709022；
- 最难硬负余弦：0.508440；
- 两者均值差：0.200581。

## 2. P3冻结测试面板

| P3面板 | Query数 | Recall@1 | MRR | Macro query AUC |
|---|---:|---:|---:|---:|
| Main real pristine（主面板） | 3,000 | 0.879333 | 0.930432 | 0.920927 |
| Isomer real pristine | 1,989 | 0.794872 | 0.881889 | 0.865622 |
| Near-core real pristine | 496 | 0.487903 | 0.698998 | 0.675567 |
| Near+mid real pristine | 661 | 0.544629 | 0.732985 | 0.713822 |
| Isomer exposed extension | 851 | 0.773208 | 0.871052 | 0.864140 |
| Sim-to-real secondary | 609 | 0.898194 | 0.942419 | 0.941279 |

P3只有`P3-main-real-pristine`是预注册主面板；其余为困难或次级重叠面板，不应合并计算一个总体置信区间。

## 3. 结构残差与局部Top-1任务

| 任务 | 样本量 | 官方结果 |
|---|---:|---:|
| Structure residual discovery Top-1 | 377 queries | Recall@1 0.814324；70错误 |
| Structure residual confirmation Top-1 | 126 queries | Recall@1 0.865079；17错误 |
| Confirmation全部结构对 | 15,652 pairs | Pearson 0.765856；Spearman 0.571325 |
| Confirmation不同身份分子对 | 14,456 pairs | Pearson 0.535535；Spearman 0.460938 |
| 同分子式、不同身份对 | 499 pairs | Pearson 0.322848；Spearman 0.324478 |
| 同身份谱图 | 1,196 identity pairs | 平均余弦0.751770 |

采集条件分层：

| 同身份谱图条件 | N | 平均余弦 | P10 |
|---|---:|---:|---:|
| 同仪器 | 930 | 0.769544 | 0.534609 |
| 跨仪器或仪器未知 | 266 | 0.689629 | 0.476687 |
| CE差≤10 | 201 | 0.857555 | 0.695684 |
| CE差>10 | 452 | 0.701272 | 0.491283 |

## 4. 生物学/BioAware已知身份任务

| 数据集 | Query数 | 官方Recall@1 | 补充指标 |
|---|---:|---:|---:|
| MTBLS13729已知身份小面板 | 21 | 20/21 = 0.952381 | 1个错误 |
| MTBLS1905外部已知身份面板 | 36 | 27/36 = 0.750000 | Top-5 34/36 = 0.944444 |
| MetDNA3 HILIC开发面板 | 117 | 95/117 = 0.811966 | 22个错误 |

MTBLS13729完整非靶向队列没有大规模已知身份真值，不能报告Recall@1；该队列评价的是冻结候选覆盖、谱学证据层与配对MS1丰度结果。

## 5. 汇报用最小基线表

| 代表任务 | 官方Recall@1 |
|---|---:|
| 全量gate检索 | 93.01% |
| 全错误图谱 | 92.44% |
| P3-main | 87.93% |
| P3-isomer | 79.49% |
| P3-near-core | 48.79% |
| MTBLS1905真实外部代谢物面板 | 75.00% |

## 6. 统一解释

官方DreaMS在普通谱库检索上约88%–93%，在一般异构体或困难跨条件任务上约76%–81%，在严格near-core异构体任务上下降到48.8%；在当前真实外部已知身份代谢物任务上为75%–81%。后续任何提升必须注明属于共享embedding、embedding后专家还是生化网络专家，并在完全相同的协议内与上述官方值成对比较。

## 7. 主要证据工件

- `data/validation/noise_isomer_infonce_g5/seed_0/gate_eval.json`
- `data/validation/m1b_head_expressibility/expressibility_sweep.json`
- `data/validation/g8r_p2b_rank_fusion.json`
- `data/validation/g8r_p2b_p3_final.json`
- `data/validation/g8r_noise_v3_s3a_extended_matrix/report.json`
- `data/validation/dreams_structure_residual_atlas_large/summary.json`
- `data/validation/bioaware_metdna3_dreams_official_v1/report.json`
- `data/validation/bioaware_reaction_context_development_v2_20260828/report.json`
- `docs/PROJECT_RESEARCH_MASTER_SUMMARY_20260829.md`
