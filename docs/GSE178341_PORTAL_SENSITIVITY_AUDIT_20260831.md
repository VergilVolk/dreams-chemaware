# GSE178341 官方门户归一化表达敏感性审计（2026-08-31）

## 定位

这是等待官方 10x 原始计数矩阵期间执行的**次级敏感性分析**，不是预注册的原始计数患者级
pseudobulk 主结果。Single Cell Portal 只返回 epithelial tSNE 中固定的 100,000 细胞子样本及
归一化表达值，因此本结果只能用于方向预检和工程排障。

## 数据审计

- 固定基因：NXPE1；MUC2/TFF3/SPDEF/FCGBP/AGR2；GNE/NANS/CMAS/SLC35A1；CASD1/SIAE。
- 门户 100,000 个细胞中 98,438 个可与 GEO 作者元数据按完整 cell ID 精确连接。
- 1,562 个门户细胞不在 GEO submit 元数据：C121 为 GEO 中完全不存在的门户患者，C144 为门户/GEO
  aliquot 版本差异；两者均不属于 6 位 GEO 黏液型患者。分析只保留精确交集并完整披露，不进行
  模糊条形码匹配。
- 统计单位始终是患者 PID：6 位纯黏液型、53 位纯普通腺癌。每位黏液型使用预先冻结的 3 位
  同部位、同 MMR 对照；不使用细胞级 p 值。

## 主要结果

### NXPE1：独立验证暂未成立

- 广义上皮患者均值差 `+0.0279`，患者 bootstrap 95% CI `[-0.0369, +0.1115]`，置换
  `p=0.547`。
- 冻结匹配均值差 `+0.0444`，6 位患者差异为
  `+0.254,+0.085,-0.002,-0.042,+0.062,-0.090`，精确 sign-flip `p=0.500`。
- 预设 goblet-family 中均值差仅 `+0.0112`，`p=0.867`。

因此，current-GDC 中的 NXPE1 mucinous-relative lineage 信号**没有在该官方门户子样本中获得
患者级独立确认**。不得用 GSE178341 包装 NXPE1 驱动 Neu5Ac、O-acetylation 或黏液型特异机制。

### 固定面板：出现更稳的分泌—高尔基体转运背景

在全部肿瘤上皮患者均值中：

| 固定终点 | Mucinous - conventional | 置换 p | 面板内 BH q | 冻结匹配均值 | 精确 sign-flip p |
|---|---:|---:|---:|---:|---:|
| AGR2 | +1.6384 | 0.00040 | 0.00480 | +1.4385 | 0.03125 |
| SLC35A1 | +0.1361 | 0.00572 | 0.03432 | +0.1379 | 0.06250 |
| MUC2 | +1.1404 | 0.01586 | 0.06344 | +1.0878 | 0.09375 |
| TFF3 | +1.4078 | 0.03372 | 0.08093 | +1.2672 | 0.09375 |
| SPDEF | +0.2620 | 0.02849 | 0.08093 | +0.2049 | 0.15625 |
| GNE | +0.0690 | 0.383 | 0.657 | +0.0969 | 0.21875 |
| NANS | +0.0357 | 0.687 | 0.824 | -0.0012 | 0.96875 |
| CMAS | +0.0150 | 0.855 | 0.933 | +0.0894 | 0.46875 |

AGR2 的 6 个冻结匹配差异全部为正；SLC35A1 为 5/6 正。预设 goblet-family 中 AGR2 仍同向
（均值差 `+1.1609`），但该窄区室不是冻结匹配的适用人群，且面板校正后 `q=0.084`。

## 当前生物学解释

该固定面板更支持一个有边界的工作模型：

1. 黏液型上皮具有增强的 mucin folding/secretory program（AGR2，并伴随 MUC2/TFF3/SPDEF）。
2. Golgi CMP-sialic-acid transporter SLC35A1 相对升高。
3. GNE/NANS/CMAS 没有同步患者级升高，因此不是“整个 de novo sialic-acid pathway 全面增强”。
4. 结合 MTBLS13729 中 free Neu5Ac 升幅大于 CMP-Neu5Ac/UDP-GlcNAc，最稳妥的表述是
   **free-pool 与 mucin carrier/transport program 解耦的 selective remodeling**，而不是通量增强。

这仍可能受 epithelial subtype composition 影响。原始计数 pseudobulk、右侧/MMR 敏感性、固定
临床技术协变量与 secretory-state adjustment 才是裁决层。

## 决策

- 不再以 NXPE1 作为需要“救回”的主机制；保留为预注册负/不稳定结果。
- 不新增候选基因；原始计数分析仍严格使用已冻结 12 基因面板。
- 下一项唯一高价值动作：完成官方 10x 原始计数患者 pseudobulk，验证 AGR2/SLC35A1 方向及
  NXPE1 失败是否复现。
- 无论原始计数结果正负，不能从 RNA 推断 Neu5Ac 来源、糖链连接、酶活性或代谢通量。

## 工件

- `data/external/GSE178341_mucinous_secretory_audit/scp_portal_normalized_sensitivity_v1/report.json`
- `data/external/GSE178341_mucinous_secretory_audit/scp_portal_normalized_sensitivity_v1/fixed_panel_patient_results.csv`
- `data/external/GSE178341_mucinous_secretory_audit/scp_portal_normalized_sensitivity_v1/patient_normalized_expression.csv`
- `data/external/GSE178341_mucinous_secretory_audit/scp_portal_normalized_sensitivity_v1/fixed_panel_patient_directions.png`

