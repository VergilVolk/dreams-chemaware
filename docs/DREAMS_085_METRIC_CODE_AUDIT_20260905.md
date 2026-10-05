# DreaMS 0.85 谱库检索指标代码审计

日期：2026-09-05  
状态：已核对 DreaMS 官方论文、官方仓库与本地评估代码

## 结论

DreaMS 原文图 4b 的约 `0.85` 是谱图对二分类的 pooled ROC-AUC，不是 Top-1、
Recall@k、macro-query AUC 或候选结构检索准确率。评估分数是两张谱图 DreaMS
embedding 的 cosine；标签是两张谱图是否具有相同 14 字符 InChIKey connectivity
block。

## 官方论文协议

- 数据：NIST20 高质量子集，仅 `[M+H]+`。
- 训练隔离：排除与 MoNA contrastive 微调集 IK14 重叠的 NIST20 分子。
- 正例：同 IK14 的谱图对。
- 负例：IK14 不同且前体质量差不超过 10 ppm 的谱图对。
- 规模：论文表述为约 750,000 个二分类 examples。
- 指标：`roc_curve(label, cosine)` 后计算曲线下面积；官方图例显示 fine-tuned
  `DreaMS (AUC = 0.85)`，zero-shot `DreaMS (AUC = 0.71)`。

## 官方代码链

官方仓库 commit：`dbec3a0b514a99e5056cfccde4559fda8cfe8129`。

1. `experiments/spec_sim/spec_retrieval.ipynb`
   - 从 `nist20_clean_A.pkl` 过滤 `[M+H]+`，得到 153,236 张谱；
   - 用 `EXACT MASS ± 10 ppm` 构造候选，排除 self；
   - 将 `(i,j)` 排序并 `np.unique`，构造无向 pair；
   - 以 IK14 相等生成 `inchi14 label`；
   - 原始候选构造得到 2,634,504 个无向 pair；
   - 最终绘图读取 `data/spec_retrieval_1M_dreams_preds_60peaks_epoch=0-step=2500.pkl`
     与 fixed DreaMS predictions，再按
     `nist_inchi14s_disjoint_from_mona_contrastive.pkl` 过滤，输出显示 750,535 行；
   - 对每个方法执行 `metrics.roc_curve` 和 `metrics.auc`。
2. `dreams/utils/data.py::SpecRetrievalValidation`
   - 只读取已经存在的 pair 表；
   - 计算 embedding cosine 和 `label` 的 ROC-AUC；
   - 不负责构造、采样或做 MoNA-disjoint 过滤。
3. `dreams/models/dreams/dreams.py::on_validation_epoch_end`
   - 训练期 callback 读取
     `nist20_clean_spec_entropy_[M+H]+_retrieval.pkl` 和
     `nist20_clean_spec_entropy_[M+H]+_50k_pairs_retrieval.pkl`；
   - 公开仓库没有找到后一个 50k pair 表的生成代码；它不能被当成论文最终约
     750k pair ledger 的已验证替代品。

## 本项目应如何评估

本项目的主要科学问题是共享 clean-spectrum embedding 在实际 MassSpecGym 候选谱库
检索上的改善。因此，每个冻结 checkpoint 必须在同一个 query、candidate graph、
正例定义和 tie policy 下与 official DreaMS 配对比较：

- Recall@1/2/3/5/10/20；
- MRR、mean/median rank；
- macro-query AUROC/AUPRC；
- micro-candidate AUROC/AUPRC；
- positive-vs-best-negative margin、Top1-Top2 gap；
- selective risk/coverage；
- corrected、introduced、risk-net 和 formula-cluster paired CI。

此外，在一次冻结且带 SHA256 的 MassSpecGym `[M+H]+` 10-ppm spectrum-pair ledger
上报告 pooled ROC-AUC。它与原文 0.85 使用相同的数学指标和标签思想，但数据域、采样
与依赖结构不同，必须称为“MassSpecGym 10-ppm pooled pairwise AUROC”，不得称为
“论文 0.85 精确复现”。

NIST20 只有在合法获得数据、原论文 pair ledger 或可核验的等价重建参数、以及
MoNA-disjoint IK14 清单后才作为外部复现；其缺失不得阻塞实际任务模型比较。

## 对 E4-DEB 的更正

`tasks/evaluate_noise_shared_nist20_pairwise.py` 仅保留为可选 legacy callback pair-ledger
评估器，并明确 `exact_paper_replication=false`。`tasks/summarize_noise_final_e4_deb.py`
不再把 NIST20 文件是否存在设为晋级门；E4-DEB 仍因其在同一 MassSpecGym held graph
上没有超过 clean duplicate 与 matched random 而失败，这一科学裁决不受本更正影响。
