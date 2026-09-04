# LCNEC 暗代谢物生物学结果（2026-08-31）

## 结论先行

LCNEC 已从候选数据集升级为当前 **主生物学论文候选**。本项目没有简单重复原文的
2-hydroxyglutarate、N-lactoyl amino acid 和脂质重塑叙事，而是从公开 HSST3n 原始数据中，
在表型盲质量控制和冻结注释协议下得到两个新的、彼此相容但必须分开表述的丰度模式：

1. **phosphorylated-nucleotide / nucleotide-sugar / NAD-related pool redistribution：**
   AMP、GMP、ADP、ADP-ribose 与 UDP-HexNAc 升高，而 guanosine、guanine 降低；
2. **expanded antioxidant pools：** GSH、GSSG 与 ascorbate 升高，而 ophthalmate 降低。

这些结果支持静态代谢物池的重排，不证明 ATP energy charge、代谢通量、酶活或因果适应。

## 证据链

### 1. 数据与发现空间

- 34 对 LCNEC tumor/adjacent tissue；HSST3n 平台含 68 study、9 pooled QC、2 blank、
  6 dilution injections。
- study 中 133,925 张 MS2，pooled QC 中 17,727 张 MS2。
- 263 个 precursor-RT 家族通过 QC/blank/dilution 门；42 个与作者 HSST3n 表匹配，221 个
  作者表外。
- 221 个目标经原始 MS1 EIC 重定量后，100 个在四种归一化口径下方向稳健；去冗余后冻结
  为 81 个模块。
- 42 个作者已知特征的 raw-EIC 效应与作者 beta Spearman rho=0.943，方向一致率 90.5%。

### 2. 冻结谱学注释

- 81/81 模块均有 pooled-QC MS2，并完成官方 DreaMS + 冻结 P2b + classical spectral
  evidence 的 m/z 约束候选排序。
- 22 个 feature 通过跨窗口/模型一致性门，对应 21 个 connectivity hypotheses。
- 其中 12 个与作者另一 LC-MS 平台重叠：方向一致 12/12，效应 Spearman rho=0.902，
  10 个在作者表中同向 FDR<0.05。这是流程的正交阳性对照，不是新发现。
- 另有 9 个作者表外谱学假说，其中 4 个进入优先结构与患者一致性确认。

### 3. 四个作者表外优先候选

| 候选 | m/z / RT | 34 对平均 log2FC | 同向患者 | q | 分子式误差 | 直接碎片证据 | 当前身份层级 |
|---|---|---:|---:|---:|---:|---|---|
| ADP family | 426.022781 / 34.93 s | +2.400 | 33/34 | 3.02e-12 | 1.509 ppm | 13 peaks；coverage 0.895；sqrt cosine 0.890 | connectivity-family hypothesis |
| ADP-ribose family | 558.064879 / 34.58 s | +1.556 | 31/34 | 1.47e-7 | 0.863 ppm | 20 peaks；coverage 0.962；sqrt cosine 0.926 | connectivity-family hypothesis |
| ascorbate | 175.024979 / 39.74 s | +5.407 | 32/34 | 2.92e-7 | 0.955 ppm | 15 peaks；coverage 0.973；sqrt cosine 0.975 | compound Level-2 hypothesis |
| quinolinate | 166.014725 / 54.79 s | +2.047 | 28/34 | 3.17e-6 | 0.865 ppm | 5 peaks；coverage 0.846；sqrt cosine 0.896 | compound Level-2 hypothesis |

四者的 leave-one-patient-pair-out 平均效应方向全部稳定，Wilcoxon p 均小于 3e-6。仍然缺
authentic-standard RT，因此不得称 Level 1。

四者在去冗余账本中均为 singleton module；对全部 quality-passed feature families 以 ±5 s
窗口审计 C13、Na-H、chloride、formate 与 acetate 常见质量差，没有发现 0.01 Da 内的共洗脱
替代峰。该阴性审计只排除预定义的明显同位素/加合物混淆，不能排除未被 feature picker 捕获的
离子、in-source multimer 或同分异构体。

### 4. BioAware 的正确作用

BioAware 没有修改谱学身份，也没有使用表型选择候选；它只做 reaction-hypergraph context
与 hub abstention：

- ADP 命中 881 条 Rhea reactions，是 currency hub；保留丰度结果，但禁止用它激活特异
  通路结论。
- ADP-ribose（21 reactions）、ascorbate（37）和 quinolinate（4）为非 hub context anchors。
- quinolinate 可定位到 de novo NAD biosynthesis 的 QPRT 反应；ADP-ribose 可定位到 NUDT5
  水解生成 AMP 与 ribose-5-phosphate；这些只表示反应成员关系，不表示反应方向或酶活。

## 论文中“复现”与“新发现”必须分开

- **复现层：** 12 个作者跨平台候选、12/12 同方向、rho=0.902，用于证明我们的 dark-feature
  pipeline 能从原始数据找回真实生物学效应。
- **新增层：** 4 个作者表外优先候选及 nucleotide/NAD/redox abundance patterns。
- **不可混写：** 作者表外不等于首次在 LCNEC 中存在；Level-2 不等于精确立体异构体；
  pathway membership 不等于机制因果。

## 主图建议

1. 原始数据到 81 modules、22 hypotheses、4 priorities 的表型盲漏斗图；
2. 12 个作者跨平台重叠物的效应相关图；
3. 四个优先候选的 34 对患者 log2FC 分布与 leave-one-pair-out 稳定性；
4. nucleotide/NAD/redox 证据图：测量节点用实线框，BioAware context 用虚线，ADP 标出
   hub abstention；
5. 四个优先候选的 query/reference mirror MS2 与关键匹配碎片。

上述跨平台图、患者配对图、丰度证据图和四候选完整 mirror spectra 已输出到
`data/validation/lcnec_hsst3n_manuscript_figures/`；图中未匹配峰、网络 context 与 hub
abstention 均保持可见或明确标注，没有把 target-only evidence 画成完整谱图。

## 独立 LCNEC proteogenomic 背景（不是代谢物复制）

2026 年独立 Science Advances 队列包含 107 对 LCNEC tumor/NAT，并在 103 对中量化蛋白组；
文章报告 KEAP1 mutation 与 metabolic reprogramming 相关，并在 combined LCNEC with NSCLC
中观察到 NRF2/pentose-phosphate-program enrichment。它为本项目 antioxidant-pool 结果提供了
一个可检验的分层方向，但不测本项目四个代谢物，患者也不重叠，因此不能记作 abundance
replication。后续若取得其患者级 protein table，应预先冻结 QPRT、NUDT5、G6PD/PGD/TKT/
TALDO1 和 ascorbate-handling genes 的小面板，并将 pure/combined histology 与 KEAP1 状态分开；
不得在看到结果后扩基因集。

## 允许与禁止结论

允许：

- LCNEC paired tissues show a reproducible redistribution of phosphorylated nucleotide/NAD-related
  and antioxidant metabolite pools.
- Four author-unreported Level-2/connectivity-family hypotheses survived exact-mass, direct-fragment,
  cross-normalization, and patient-pair consistency checks.

禁止：

- ATP energy charge increased；QPRT/NUDT5/PARP/CD38 activity increased；NAD flux increased；
- ascorbate accumulation is tumor-protective or therapeutic；
- exact stereoisomer identity、Level 1、causal dependency、clinical biomarker。

## 最小闭环与停止规则

1. 先完成上述 5 张图和 feature-level supplement，不再扩散到新数据集。
2. 手工审计四者的 isotope/adduct/coelution 和 mirror spectra；任何 identity conflict 都降级。
3. 如只能购买少量标准，优先 ascorbate 与 quinolinate；ADP/ADP-ribose 在标准确认前保持
   connectivity-family。
4. 独立 LCNEC proteogenomic 数据可验证 QPRT/NUDT5/redox enzyme context，但不能当作
   代谢物丰度复制。
5. 没有标准或独立代谢组时，论文定位为 evidence-calibrated algorithm-enabled biological
   discovery，不包装成 causal metabolism paper。

## 机器可审计的投稿就绪裁决

`data/validation/lcnec_hsst3n_manuscript_readiness/readiness_report.json` 已将整个证据链锁成
fail-closed 论文门。它不重新拟合、不重新选择候选，只验证冻结工件及其哈希：原始采集与
pooled-QC/blank/dilution、dark EIC、跨归一化稳健性、已知特征正对照、81-module 全量注释、
12 个跨平台复现、四候选 formula/fragment 与 34 对患者一致性、BioAware hub abstention、
adduct-spacing 阴性审计和成套论文图均通过。

机器裁决为：

- `ready_for_algorithm_enabled_level2_biology_manuscript = true`；
- `ready_for_level1_identity = false`；
- `ready_for_independent_metabolite_replication_claim = false`；
- `ready_for_causal_metabolism_claim = false`。

因此当前不是“每条线都失败”，而是已经形成一条可投稿的算法赋能 Level-2 生物发现主线；
它仍不是标准品确认或因果代谢机制论文。缺口不得通过文字包装消除，只能通过 authentic
standards、独立 LCNEC 代谢组或 perturbation/tracing 补足。

## 关键工件

- `data/validation/lcnec_hsst3n_annotation_biology/`
- `data/validation/lcnec_hsst3n_priority_structure/`
- `data/validation/lcnec_hsst3n_priority_pair_consistency/`
- `data/validation/lcnec_hsst3n_priority_adduct_audit/`
- `data/validation/lcnec_hsst3n_bioaware_context/`
- `data/validation/lcnec_hsst3n_all_robust_annotation/`
- `data/validation/lcnec_hsst3n_manuscript_readiness/readiness_report.json`
- `data/validation/lcnec_hsst3n_manuscript_supplement/supplement_manifest.json`
- `docs/LCNEC_MANUSCRIPT_RESULTS_DRAFT_20260831.md`
