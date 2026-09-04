# 外部 CRC Neu5Ac 生物地理审计（PMC11438248，2026-08-31）

## 1. 审计目的

本审计检验一个严格限定的问题：是否存在独立患者组织队列支持 Neu5Ac 在 CRC 中受到疾病状态与
解剖位置共同调节？它不把解剖位置证据替换成黏液型组织学复现，也不把另一个平台的标准品身份
替换成本项目当前 LC 方法下的同法 spike-in。

来源研究为 Jain 等发表于 *Molecular Cancer* 的 2024 年研究，纳入 372 对患者匹配的原发 CRC
肿瘤与正常黏膜，覆盖盲肠、升结肠、横结肠、降结肠、乙状结肠、直乙交界和直肠七个亚部位。
官方补充方法与表格已下载并按 SHA256 冻结。

## 2. Neu5Ac 身份证据

官方补充 Table S1 将 `N-Acetylneuraminic acid` 列为：

- ID Level：`1`；
- acquisition：`HILIC (-)`；
- ion m/z：`308.0980`；
- RT：`355.7 s`。

补充方法说明 Level 1 来自 in-house 标准品的 m/z、RT 与 CRC 样本 MS/MS 对照。因此该队列提供
独立的 CRC 组织、标准品支持 Neu5Ac 身份背景。它不是 MTBLS13729 当前正相 RPLC feature703 的
同法标准终证；两者的色谱、极性、仪器和样本队列不同。

补充 Table S3 还列出 `N-Acetyl-9-O-acetylneuraminic acid`，但其 fragmentation similarity
仅 `42.6`，属于不同的修饰分子与不同证据层，不能与 Neu5Ac Level-1 行合并。

## 3. 疾病依赖空间梯度

补充线性异质性表报告：

| 组织 | cecum-to-rectum slope | p值 | 判读 |
|---|---:|---:|---|
| 匹配正常黏膜 | +0.349 | <0.001 | 显著正向解剖梯度 |
| CRC 肿瘤 | +0.088 | 0.091 | 梯度明显衰减，名义不显著 |

这表明 Neu5Ac 不是一个在所有结肠部位可直接互换的静态背景物：正常黏膜具有强烈空间梯度，而
肿瘤状态改变了该梯度。主文也将 N-acetylneuraminic acid 归入正常与肿瘤浓度梯度不同的代表性
代谢物，并报告右侧肿瘤中的 Neu5Ac 低于左侧肿瘤。

对 MTBLS13729 的直接意义是：10 个 Rmu 全部来自右侧，组织学与位置存在完全混杂；因此
`Rmu-RN` 的患者内配对是必要主终点，`(Rmu-RN)-(Rtu-RN)` 是必要的亚型敏感性终点。外部数据
提高了“必须位置敏感”的可信度，却不能单独证明本地 interaction 完全由组织学驱动。

## 4. 不能宣称的内容

字段级搜索得到 `mucinous` 关键词命中数为 0。公开补充患者特征只提供 sex、age、stage 和
anatomical subsite，未提供可复算的黏液型亚组。因此：

- 不能称“372 对队列独立复现 Rmu Neu5Ac 升高”；
- 不能把右侧/左侧差异当成黏液型/常规型差异；
- 不能从空间梯度推断 Neu5Ac flux、合成、降解或糖链去向；
- Neu5Ac 未出现在补充的“各亚部位共同 tumour-vs-normal 差异”汇总表中，因此该外部证据是
  spatial-context support，而不是泛 CRC tumour-normal 主效应确认。

补充分期表还存在一处源文件内部不一致：方法报告 372 对，但分期单元格相加为 374。本项目保留
论文级 372 对队列描述，同时显式记录表格不一致，不用该表反推患者数。

## 5. 对当前主线的证据升级

该队列使独立外部证据从“只有转录和 n=2 结构糖组”增加了一层患者组织代谢物背景：

1. 独立标准品支持 Neu5Ac 确实可在 CRC 组织中可靠测得；
2. 大样本患者配对队列显示 Neu5Ac 空间调控具有疾病依赖性；
3. 它解释了为什么本项目必须同时建模配对、解剖位置与组织学；
4. 它仍未补上独立黏液型丰度复现、同法 spike-in 或 glycan destination。

因此完成度总账应把“独立 Neu5Ac 患者组织背景”从缺失升级为 `PASS_CONTEXT`，但
“独立 Rmu abundance replication”继续保持 `FAIL_MISSING`。

## 6. 可复核工件

- 官方补充方法：`data/external/CRC_metabolic_biogeography_PMC11438248_20260831/supplementary_methods.docx`
  - SHA256：`1cbce8ed71d352165834d5865dfac86287165c905a85b3e55dbc924f2578f96a`
- 官方补充表：`data/external/CRC_metabolic_biogeography_PMC11438248_20260831/supplementary_tables.docx`
  - SHA256：`edc6f11d11e9b74f11d599aeadc64466bf2e2dc73fb9313edaabde71b8d9e748`
- 审计脚本：`tasks/audit_external_crc_neu5ac_biogeography.py`
- 机器可读结果：
  `data/external/CRC_metabolic_biogeography_PMC11438248_20260831/neu5ac_biogeography_audit_v1/`

## 7. 投稿用一句话

> In an independent series of 372 patient-matched colorectal tumours and normal mucosae,
> standard-supported Neu5Ac showed a strong cecum-to-rectum gradient in normal tissue that was
> attenuated in tumours, providing disease-dependent spatial context rather than a mucinous-specific
> replication of the MTBLS13729 phenotype.

