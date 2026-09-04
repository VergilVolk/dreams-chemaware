# MTBLS13729 feature 1717 多胺轴逆向审计（2026-08-30）

## 结论先行

feature 1717 是一个**丰度效应强、原始 DDA MS2 可复核、跨色谱定量高度一致，但仍缺标准品终证**的候选。当前可升级为“m/z 230.185931、N1,N8-diacetylspermidine-like / acetylated-polyamine ion-family candidate”，仍不能写成已确认的 N1,N8-diacetylspermidine 或 MSI Level 2/Level 1。

## 1. 本地定量证据

- 坐标：positive RP，m/z `230.185931`，RT `89.57 s`。
- 59 个样本检出率 `94.9%`，中位 apex 偏差 `1.76 s`，中位 SNR `177,171`。
- Rmu 有值的 9 对患者全部为正；raw 平均 `+3.009 log2FC`，三档 PQN 为 `+2.859` 至 `+2.875 log2FC`。
- 四种归一化的精确 sign-flip p 均为 `0.00390625`；留一患者最小平均效应仍为 `+2.38 log2FC` 以上。
- Rtu 的平均效应接近零，目标子集中的 Rmu-Rtu 交互为 `+2.92` 至 `+3.09 log2FC`，精确置换 p 为 `0.0096–0.0143`。

这里最后一项仍不能被称为“全发现空间校正后的黏液型特异性”，因为它来自预先选择的 8 候选目标集，而且完整冻结全特征审计中没有 interaction FDR 通过的代谢物。

## 2. 身份证据审计

- 候选中性式为 `C11H23N3O2`，本地 HMDB 表中该分子式只有一个命名匹配：N1,N8-diacetylspermidine。
- 理论 `[M+H]+` 为 `230.186303`，实测质量误差 `-1.62 ppm`。
- 旧通用 MS1–MS2 桥接表给出 `0` 条 accepted link，但回到原始 positive-RP mzML 并按峰界重新审计后，找到 `73` 张峰界内 DDA MS2，覆盖 `45` 个样本。说明旧桥接表漏掉了真实采集证据，不能再用“0 link”推断“没有 MS2”。
- `m/z 100.0759` 在 `73/73` 张谱中出现并均为基峰；`114.0916`、`72.0445` 和 `213.1607` 也呈复现性。文献中的 N1,N8-diacetylspermidine CRC MRM 使用 `230.2 -> 100.0`，与本地最稳定产物离子一致。
- 同一研究的 independent positive-HILIC MAF 含 HMDB0041947/N1,N8-diacetylspermidine 候选。RP feature 1717 与该 HILIC 峰在 `59` 个共同样本中的 Spearman `rho=0.860`，在相同组织标签内残差相关 `rho=0.756`；`29` 对肿瘤-癌旁差值相关 `rho=0.719`。在 `85` 个可评估 HILIC 注释峰中，该候选相关性排名第 `1`，分层置换和配对差值置换均为 `p=4.99975e-5`。
- 这些结果把证据从“单一精确质量命中”升级为“原始 MS2 + 诊断产物离子 + 跨色谱同样本定量一致”。但 HILIC MAF 未报告碎片、可靠性或检索分数，且其 m/z 相对理论 `[M+H]+` 偏差约 `-13.87 ppm`；仍没有 authentic-standard RT 和加标共洗脱，所以不能升级为确切位置异构体结构确认。

最终身份写法：

> an m/z 230.185931 N1,N8-diacetylspermidine-like feature supported by recurrent product-ion evidence and orthogonal cross-chromatography abundance concordance

禁止写法：

> N1,N8-diacetylspermidine was identified at MSI Level 2

## 3. 独立 MTBLS8090 通路复核

MTBLS8090 提供 35 对 CRC 肿瘤/癌旁，但没有 N1,N8-diacetylspermidine 本身，也没有黏液型标签。可用的三项多胺为：

| 代谢物 | 平均 log2FC | 中位 log2FC | Wilcoxon p | 三项 BH q |
|---|---:|---:|---:|---:|
| N1-acetylspermidine | +0.995 | +0.042 | 0.512 | 0.522 |
| N1-acetylspermine | -0.194 | -0.086 | 0.168 | 0.505 |
| Spermidine | +0.326 | -0.157 | 0.522 | 0.522 |

没有一项显著，方向也不构成一致的多胺乙酰化升高。因此 MTBLS8090 **不提供泛 CRC 外部复现**。它也不能直接否定 feature 1717，因为缺少同一代谢物、同一色谱协议和 Rmu 分层。

## 4. 文献能支持什么

既往 CRC 研究报告尿液 N1,N8-diacetylspermidine/N1,N12-diacetylspermine 可升高，也有原发结直肠腺癌组织 N1-acetylspermidine 升高的研究。这些结果只建立“乙酰化多胺与 CRC 有生物学先验”，不能确认本地峰身份，也不能证明 Rmu 特异性。

- Urinary DiAcSpd/DiAcSpm in CRC: https://pubmed.ncbi.nlm.nih.gov/20655890/
- N1-acetylspermidine in colorectal adenocarcinoma tissue: https://pubmed.ncbi.nlm.nih.gov/6692383/
- HMDB candidate entry: https://www.hmdb.ca/metabolites/HMDB0041947

## 5. 决策与最小闭环

当前论文角色：**最强的新候选代谢轴**，可与 feature 3222 的酰基肉碱类别锚点并列展示；但具体名称仍必须保留 `-like` 或 `candidate`。

最低升级实验：

1. 购买 N1,N8-diacetylspermidine 标准品；
2. 同法 RT；
3. 完整正离子 MS2 与诊断离子比；
4. 样本加标共洗脱；
5. 若可能，用同位素内标做半定量/绝对定量。

只有 1–4 全部通过，才恢复命名代谢物表述。若无法购买标准品，则保留为 m/z 230.185931/C11H23N3O2 candidate，并将具体名称放在候选列而非结果标题。

## 6. 可复核产物

- `tasks/audit_mtbls13729_polyamine_axis.py`
- `data/mtbls13729/polyamine_axis_adversarial_audit_v1/summary.json`
- `data/mtbls13729/polyamine_axis_adversarial_audit_v1/local_normalization_audit.csv`
- `data/mtbls13729/polyamine_axis_adversarial_audit_v1/mtbls8090_polyamine_context.csv`
- `tasks/audit_mtbls13729_candidate_ms2_coverage.py`
- `data/mtbls13729/frozen_candidate_ms2_coverage_v1/candidate_ms2_coverage.csv`
- `tasks/summarize_mtbls13729_candidate_ms2_consensus.py`
- `data/mtbls13729/frozen_candidate_ms2_consensus_v1/report.json`
- `tasks/audit_mtbls13729_polyamine_crosschrom.py`
- `data/mtbls13729/polyamine_crosschrom_audit_v1/summary.json`
