# 独立黏液型 CRC 代谢组公开数据系统检索审计（2026-08-30）

## 结论

截至本次冻结检索，没有找到一个比 MTBLS13729 更合适、同时满足以下条件的独立公开队列：

1. 人体 CRC 肿瘤组织或配对癌旁组织；
2. 患者级 mucinous / conventional histology 标签；
3. 可下载的代谢物矩阵或原始 LC-MS 数据；
4. 足以对修饰鸟苷、多胺或长链 acylcarnitine 轴做患者级复算。

这不是“互联网上绝对不存在”的证明，而是对当前公开数据库中**可访问且可患者级对账的 metadata** 的冻结审计结论。

## 数据库级检索，而非普通关键词搜索

使用 OmicsDI API 检索 `"colorectal cancer" AND omics_type:"Metabolomics"`，冻结得到 88 条结果：

- 53 个 MetaboLights accession；
- 15 个 Metabolomics Workbench `ST` accession；
- 20 个其他条目，主要是 GNPS/MassIVE 镜像、OEX00031348/49/52/56 MSI/MSS 组织代谢组，以及若干细胞、血浆、微生物或重复收录数据。

对 53 个 MetaboLights 条目逐一调用 public files API，只下载 `s_*.txt` 样本元数据；53/53 成功，无静默失败。对 15 个 Workbench 条目逐一下载 public factor metadata；15/15 成功。

## 审计结果

### MetaboLights

- 53/53 个 accession 完成样本表审计；
- 公开 ISA header 中没有任何研究提供标准化的 histology/mucinous 字段；
- 唯一出现 mucinous 患者级编码的是 MTBLS13729 自身：10 条 `Rmu` 样本名；
- MTBLS13729 的 ISA 表并没有显式写 “mucinous”，需要结合论文设计把 `Rmu` 解码；因此脚本将其标为 `encoded_mucinous_rows=10`、`explicit_mucinous_rows=0`。

### Metabolomics Workbench

- 15/15 个 accession 完成 factor metadata 审计；
- 没有任何条目公开 mucinous 患者级标签。

### 其他 20 条 OmicsDI 结果

- 9 条以上是已经审计过的 MetaboLights/Workbench 的 GNPS 镜像或 proteomics/cell-line 数据，不构成新临床代谢组；
- OEX00031348/49/52/56 是 MSS/MSI CRC 组织代谢组，已经用于 Level-1 modified-guanosine 反证，但没有 mucinous histology；
- MSV000092836 是 CRC plasma/微生物关联数据，不是配对组织且公开分型不是 mucinous；
- MSV000096806 是 CRC stage/淋巴结分类研究，公开资料未提供可用于 mucinous 复算的组织学标签；
- MSV000092468 是 AHCY 机制研究的模型/多模态数据，不是独立黏液型患者代谢组。

## 这项负结果如何影响论文

允许写：

> A systematic audit of 88 public CRC metabolomics records, including patient-level metadata from 53 MetaboLights and 15 Metabolomics Workbench studies, did not identify an independent publicly accessible tissue cohort with mucinous histology labels suitable for patient-level replication. MTBLS13729 itself was the only deposition containing an encoded Rmu subgroup.

禁止写：

> No independent mucinous CRC metabolomics cohort exists.

正确含义是“在冻结时间点的公开 metadata 中未找到可复算队列”。未公开临床字段、受控访问数据或将来发布的数据仍可能存在。

## 对后续验证优先级的影响

1. 不再反复下载普通 CRC 队列并把一般 tumour-normal 方向包装成 mucinous replication；
2. 将外部一般 CRC 队列用于 pathway context、异质性和反证；
3. 黏液型特异性只能靠新的独立组织队列、受控访问临床 metadata 或标准品/靶向验证升级；
4. 当前主文把 Rmu 明确称为 `10-pair discovery subgroup`，不称独立确认的人群效应。

## 可复核工件

- OmicsDI 冻结搜索：`data/external/omicsdi_crc_metabolomics_search_20260830.json`
- MetaboLights 审计：`data/external/public_crc_metabolomics_histology_audit_v1/report.json`
- MetaboLights 逐研究表：`data/external/public_crc_metabolomics_histology_audit_v1/study_audit.csv`
- Workbench 审计：`data/external/public_crc_mw_histology_audit_v1/report.json`
- Workbench 逐研究表：`data/external/public_crc_mw_histology_audit_v1/study_audit.csv`
- 执行脚本：
  - `tasks/audit_public_crc_metabolomics_histology.py`
  - `tasks/audit_public_crc_mw_histology.py`

## 主要接口来源

- OmicsDI API: https://www.omicsdi.org/help/api
- MetaboLights public studies/files API: https://www.ebi.ac.uk/metabolights/
- Metabolomics Workbench REST API: https://www.metabolomicsworkbench.org/tools/MWRestAPIv1.2.pdf
