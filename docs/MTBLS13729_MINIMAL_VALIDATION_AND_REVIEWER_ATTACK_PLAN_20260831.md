# MTBLS13729 最小验证与审稿攻击预案（2026-08-31）

## 一、目的

本文件不再增加候选故事，而是针对当前主轴
`mucinous-relative hybrid mucin glycome with donor–carrier–core–linkage decoupling`
列出最可能击穿论文的审稿问题、已有回答和最小补强。

## 二、五个最高风险问题

| 审稿攻击 | 当前回答 | 当前等级 | 最小补强 | 通过标准 |
|---|---|---|---|---|
| Neu5Ac身份是否只是同质量异构体？ | source Level-1跨面板桥、患者变化rho约0.959、锁定EIC/MS2 | 强桥接，非同法终证 | 普通Neu5Ac authentic standard，同梯度RT+同CE MS2+pooled spike-in | RT共洗脱、碎片一致且spike-in不产生独立峰肩 |
| Rmu效应是否只是10例小样本/选择后结果？ | 10/10同向、三套丰度协议、五模块interaction唯一通过；full-space FDR10=0已披露 | discovery | 独立带mucinous标签组织；若不可得则预注册靶向面板 | 独立队列方向同向并给出配对效应和CI；不能以TCGA代替 |
| free Neu5Ac是否等于糖链高唾液酸化？ | donor、carrier、core、linkage及O-acetyl-like负结果均不支持统一上升 | 已反驳简单模型 | linkage-aware O-glycan或MUC2 glycopeptide | 同一样本报告free pool和至少alpha2-3/alpha2-6/core-2/core-3载体结果 |
| 外部MUC2证据是否被过度使用？ | PXD055865三标本仅两患者；鉴定数不当丰度；不称复制 | 边界清楚 | 无计算补丁可提高独立性 | 主文和图注始终写2 patients、structural context、not abundance replication |
| 是否证明了通量、酶或治疗靶点？ | 明确没有；静态丰度与bulk RNA只生成竞争机制 | 未通过 | isotope tracing + node perturbation + rescue | 没有实验就禁止drives/flux/activation/target措辞 |

## 三、最小资源分配顺序

### P0：最少投入、最大论文增益

1. Neu5Ac authentic standard；
2. 能获得时加入稳定同位素内标；
3. 在当前同一LC方法下完成RT、MS2和pooled extract spike-in；
4. 冻结操作前的接受标准，不能看完结果再调RT/碎片阈值。

这一步直接将主阳性锚点从“强跨面板身份桥”推进到“同法标准支持”。

### P1：回答主机制而非扩候选

选择一种可实现的carrier/destination readout：

- MUC2/StcE glycopeptide；或
- PGC-LC-MS/MS O-glycomics；或
- 至少alpha2-3/alpha2-6/core-2/core-3的lectin/抗体面板并与MUC2共定位。

优先同一样本；若只能使用替代组织，必须保持“orthogonal context”而不是“same-sample proof”。

### P2：只有要做因果机制稿时才投入

- 同位素前体进入free Neu5Ac、CMP-Neu5Ac和糖链的时间序列；
- CMAS/SLC35A1、候选sialidase或carrier/linkage节点扰动；
- glycan readout、MUC2 readout、细胞表型和rescue。

## 四、不应继续投入的方向

- 无新真值的BioAware阈值扫描；
- 再堆bulk RNA队列；
- 把m/z 350.109峰强行命名为4-O或9-O-acetyl-Neu5Ac；
- 用PXD055865鉴定条目数做tumour-normal abundance统计；
- 在没有标准或独立队列时继续扩展更多低覆盖代谢物故事；
- 用“一个显著、一个不显著”替代患者内直接差值检验。

## 五、投稿层级裁决

- **现在：** algorithm-enabled, evidence-calibrated clinical discovery；
- **P0+P1通过后：** identity- and structure-supported clinical glycome-remodeling study；
- **P2通过后：** causal glycan-metabolic mechanism。

任何阶段都不能把更高一级的措辞提前借用。

## 六、当前权威工件

- 完成度总账：`data/mtbls13729/mechanism_paper_completion_audit_v9_final/`
- 投稿决策：`docs/MTBLS13729_BIOLOGY_PUBLICATION_DECISION_20260831.md`
- MUC2外部载体审计：`docs/MTBLS13729_PXD055865_MUC2_GLYCOPEPTIDE_AUDIT_20260831.md`
- O-acetyl标准谱库审计：`docs/MTBLS13729_OACETYL_NEU5AC_STANDARD_RESOURCE_AUDIT_20260831.md`
- 主结果：`docs/MTBLS13729_BIOLOGY_MANUSCRIPT_RESULTS_V2_20260830.md`
