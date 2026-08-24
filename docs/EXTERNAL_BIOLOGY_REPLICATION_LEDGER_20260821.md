# 外部生物学复现台账

日期：2026-08-21

目标：把 MTBLS13729 发现队列中的“Rmu 相对配对癌旁长链酰基肉碱候选积累”拆成可证伪的命题，而不是用另一个队列的任意阳性结果包装验证。

## 已完成：MTBLS8090，泛 CRC 组织代谢表型边界检验

- 公开队列：MTBLS8090；35 对 CRC 肿瘤/配对癌旁组织，反相与 HILIC LC-MS；原始研究为 Kang et al. 2023（PMID 37772396）。
- 可用性：反相 MAF 含 42 个 acylcarnitine 条目，24 个满足预冻结规则“明确 C 数且 `C>=12`”。
- 方法：每位患者计算 24 个条目的 `median(log2 tumor/normal)`；特征纳入不使用丰度、疾病标签或检验结果。
- 结果：平均类别效应 `-0.196 log2FC`、Wilcoxon `p=0.273`、20 万次符号置换 `p=0.429`。
- 结论：**不复现泛 CRC 中 LCAC 上升。** 因为无黏液型/Rtu 标签和原始 DDA MS2，它不能否定 Rmu 特异表型，也不能评价 DreaMS；但明确否定了将发现集结果泛化为普适 CRC 代谢规律的做法。

复现脚本：`tasks/replicate_lcac_mtbls8090.py`；结果目录：`data/external/MTBLS8090/lcac_replication/`。

## 候选：GSE236696，Rmu 配对单细胞转录组

- 数据：6 对黏液型 CRC 肿瘤/癌旁组织的 scRNA-seq，GEO GSE236696，公开原始矩阵约 445 MB。
- 能回答：在上皮细胞层面，肉碱穿梭/FAO 基因集是否与 Rmu 肿瘤状态相关；可用于正交机制线索。
- 不能回答：代谢物丰度、LCAC 结构身份、DreaMS 注释性能、通量或 Rmu 相对 Rtu 的差异。
- 决策：在继续下载前，先完成预定义基因集与细胞类型/患者级统计方案；避免把单细胞的细胞组成差异错当成代谢重编程。

## 下一门槛

真正的外部生物学验证必须满足至少一项：

1. 带 Rmu/Rtu 病理标签的组织代谢组，复现预冻结 LCAC 规则下的 Rmu-RN 方向；或
2. 至少两个独立的 Rmu 转录组队列，在患者/样本级、控制 MSI/MMR 后一致支持预定义肉碱穿梭/FAO 基因集。

第二项只能提供机制线索，不能把静态转录替代为代谢通量。

## 参考

- MTBLS8090: <https://www.ebi.ac.uk/metabolights/MTBLS8090>
- GSE236696: <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE236696>
- 发现集原始研究: <https://pubmed.ncbi.nlm.nih.gov/42366730/>
