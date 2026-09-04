# 生物学数据集换题硬门结果（2026-08-31）

## 最终裁决（LCNEC 81 模块全量注释与正交确认后更新）

LCNEC 已经跨过替代门槛，升级为 **主生物学论文候选**；MTBLS13729 冻结为独立保底和
方法迁移案例。依据不再只是发现容量，而是一个已经闭合到“公开原始数据—表型盲筛选—
冻结谱库注释—作者另一平台复现—精确分子式/碎片—34 对患者一致性—BioAware 网络弃权”
的证据链。LCNEC 仍然是静态丰度和 MSI Level-2/连接性家族假说，不是 Level-1、通量、
酶活或因果机制。

因此：

1. **LCNEC 升级为当前主生物学论文候选。** 81 个冻结模块已全部完成 m/z 约束的官方
   DreaMS + 冻结 P2b + 经典谱学注释；22 个通过一致性门，12 个与作者另一平台重叠且
   效应方向 12/12 一致（Spearman rho=0.902），另有 9 个作者表外谱学假说。
2. **MTBLS13729 保留为冻结保底，不因换题而废弃。** 当前保底不是
   单个弱峰，而是患者配对的 free-Neu5Ac 主轴、同患者 donor/pool 解耦、外部糖组和
   转录背景共同限定出的 `mucinous-relative hybrid mucin glycome`。它仍是发现级，不是
   通量或因果机制。
3. **MTBLS13432 / SgME-HCC 的直接组织 MS/MS 重注释路线停止。** 公开 tissue mzML
   的所谓 MS2 是固定前体周期事件，不是覆盖暗峰的 DDA。不得再下载 88/218 个同协议
   文件试图用样本数弥补采集协议缺失。
4. **四个作者表外候选进入主结果。** ADP 家族、ADP-ribose 家族、ascorbate 和 quinolinate
   均通过 5 ppm 分子式与直接碎片门，并在 34 对患者中方向一致率为 82.4%--97.1%；其中
   ADP-ribose、ascorbate、quinolinate 是非 hub 生化网络锚点，ADP 因 881 条 Rhea 反应被
   BioAware 主动弃权为通路证据。
5. **2026 Science lysosomal-aging atlas 暂不进入下载。** 其年龄梯度、多组织、热量限制和
   储积病对照非常强，但原论文已经围绕 glycerophosphodiesters/cystine 完成主要机制闭环；
   公开包合计 132.89 GB，当前 README 只给原始 Thermo RAW 与粗粒度组别说明，尚未证明
   有足够的未解释 MS/MS 模块可产生独立故事。
6. **2026 Nature Communications PDAC mutation–metabolome 项目不作为替代主线。** 它有
   41 例 WES–血清配对、151/79/79 三个预后队列、QC 双碰撞能 MS/MS、九代谢物标准、
   GRPEL1 类器官/动物/雷帕霉素验证；正因为闭环已经很强，我们的增量更可能是注释复核，
   而不是形成明显强于原文的独立生物学发现。

## 证据对比

| 数据集 | 真实优势 | 关键缺口 | 相对 MTBLS13729 的明确增益 | 裁决 |
|---|---|---|---|---|
| MTBLS13729 | 30 对 CRC；四面板 DDA；已完成统一注释、targeted-EIC、患者配对与外部结构/转录边界审计 | 无 pooled QC/blank；Rmu 仅 10 对；缺独立 Rmu 丰度复制、同法标准、同一样本 glycan destination 和因果实验 | 已有可投稿的、证据边界清楚的 biology package | **冻结保底/迁移验证** |
| SgME-HCC | 空间 MER、组织学、bulk 与 RNA 设计很强 | 公共组织文件不含目标化 DDA；关键 pooled-QC 多碰撞能碎裂及中间工件未公开 | 空间设计更强，但身份层无法由公开 raw 支撑 | **NO-GO** |
| LCNEC | 34 对肿瘤/癌旁；完整 QC/blank/dilution；81 个模块全量注释；作者另一平台效应复现 | 尚无标准品 RT、独立 LCNEC 代谢组复制或因果扰动 | 22 个一致性候选；12 个跨平台重现（方向 100%，rho=0.902）；4 个作者表外优先候选均过分子式、碎片和患者一致性门 | **主生物学论文候选** |
| Lysosomal aging | 多组织、年龄梯度、mock lysosome、热量限制、跨物种及疾病参照；原始 MS/MS/MS3 完整 | 132.89 GB；原文闭环强；README 无 feature/annotation/unknown ledger；人类疾病转化仍间接 | 因果设计明显更强，但不是明显的未解注释故事 | **暂缓** |
| PDAC mutation–metabolome | 41 例 WES–血清配对；309 例多队列代谢组；QC 25/50 NCE；预后、类器官、动物与药敏验证 | 原文已有标准支持的九代谢物面板、GRPEL1 因果链和独立验证 | 数据与机制都强，但可新增的独立问题不明显 | **不换题；可作旁证/算法外测** |

## SgME 的决定性否决证据

### 目标账本纠错

- 旧“59 个高价值 gap”只是上限，不是 59 个真实未注释目标。
- 真实未注释 gap 为 **51**；其中 **34** 个具有可直接匹配的组织前体 m/z，另 **17** 个
  只有 neutral mass，不能直接在 tissue DDA 中寻找。
- 其余 8 个已被作者赋过身份，只是没有进入最终 retained 表，不能重新包装成新增注释。

### 原始 mzML 流式审计

- 3 位患者、6 个 normal/tumour section、12 个 HPOS/LPOS 文件；共读到 162 个 MS2。
- 34 个可匹配 gap 在 20 ppm、30 s 窗口内命中 **0**。
- 更关键：全部 78 个 HPOS MS2 的 precursor m/z 均为 **625.0**，全部 84 个 LPOS
  MS2 均为 **1025.0**，碰撞能均为 4，并按约 60 s 周期重复。
- 因此这些记录属于固定扫描/校准样事件，而不是样本依赖的 DDA。增加同协议文件只会重复
  625/1025，不会使暗峰获得碎裂证据。

对应工件：

- `data/validation/mtbls13432_sgme_coverage_probe/coverage_probe.json`
- `data/validation/mtbls13432_sgme_coverage_probe/ms2_spectra.csv`
- `data/validation/mtbls13432_sgme_coverage_probe/target_ledger.csv`

## 当前保底为何仍然是 MTBLS13729

新数据集必须超过的不是早期“3222 单峰”，而是后续冻结的完整结果：

- free Neu5Ac targeted-EIC 在 Rmu–RN **10/10 正向**，均值约 **+1.935 log2**（约
  3.82 倍）；
- subtype-sensitivity `(Rmu-RN)-(Rtu-RN)` 在 raw/PQN 下约 **+2.209/+2.142 log2**，
  五模块 BH q 约 **0.00179/0.00162**；
- feature 703 与 source Level-1 Neu5Ac 的同样本/患者桥接相关约 **0.959**；
- 同患者 free Neu5Ac 升幅超过 CMP-Neu5Ac 与 UDP-GlcNAc，Holm p 均约 **0.0273**；
- 外部糖组、MUC2 glycopeptide、372 对 CRC 空间背景和 980 例转录背景用于限定
  carrier/core/linkage 与 free-pool 的关系，并保留反证。

这仍缺独立 Rmu 丰度复现、同法标准和因果实验，但已经比“换到一个更大 raw 数据后重新
找显著峰”更接近可投稿闭环。

## LCNEC 暗特征淘汰门（已完成）

本阶段不得直接宣称换题，也不得调参后碰完整四平台。先冻结一套平台内淘汰门：

1. 仅使用一个 untargeted 平台；保留其 68 study、9 pooled QC、2 blank、6 dilution 的完整
   注射结构。
2. 表型盲完成 MS1 feature、DDA linkage、同位素/加合物归并和 QC/blank/dilution 审计。
3. 只有同时满足下列条件，才允许扩到四平台：
   - 至少 100 个非冗余、QC CV 合格、blank 排除、dilution 响应合理且具有 DDA MS2 的
     作者列表外 feature family；
   - 至少 20 个具有可审计 Level-2 候选，而不是只靠 exact mass；
   - 至少 5 个形成患者配对 FDR<0.10、跨合理归一化与协变量模型方向稳定的模块；
   - 其中至少 1 个模块不属于原文已主讲的 2-HG、N-lactoyl amino acids、PC/ether lipid、
     PUFA-BMP、TAG/CE、acylcarnitine 或 cotinine；
   - 新增候选的人工审计冲突率不高于 5%。
4. 任一关键门失败，LCNEC 只保留为算法外测，不升级为主生物学课题。

这些阈值是进入四平台的淘汰门，不是论文成功承诺，也不得用单次 OOF/开发集结果替代外部
验证。

## LCNEC 单平台硬门实测结果

### 采集真实性

- Zenodo HSST3n ZIP 的 MD5 为 `b843e95f4e3ec6382e9133910adfc1b6`，与公开记录一致。
- 85/85 文件全部解析：68 study、9 pooled QC、2 blank、6 serial dilution。
- 研究样本累计 133,925 张 MS2，单文件中位 1,969 张；独立前体中位 958 个。
- pooled QC 累计 17,727 张 MS2，单文件独立前体中位 959 个。

### 表型盲 headroom 与作者表对账

- 17 个 QC/blank/dilution 文件共 33,476 张 MS2，聚成 1,138 个 precursor-RT 家族。
- 607 个家族在至少 6/9 pooled QC 中复现，359 个在 6 点稀释序列中 Spearman rho>=0.70。
- 263 个同时满足 QC 复现、blank/QC<=0.20 和稀释响应。
- 与作者 Table S2 的 97 个 HSST3n 条目按 5 ppm/15 s 对账，仅 42 个匹配；221 个为作者
  HSST3n 列表外、采集质量合格的 dark-feature 家族。`unmatched` 仅表示作者表外，不代表新分子。

### 配对丰度与反证审计

- 221 个 dark targets 直接从原始 MS1 做 5 ppm/15 s EIC 重定量；207 个通过独立 MS1
  QC CV、blank、dilution 和 study detection 门。
- raw、per-mg、per-mg+PQN、per-mg+QC-drift+PQN 四种口径分别有 111/112/115/114 个
  稳健配对特征；100 个在四种口径中同向通过。
- 按 RT<=5 s 且跨样本 Pearson r>=0.95 去除明显共洗脱冗余后，仍有 81 个模块，最大模块
  仅 4 个特征。
- 42 个作者已知匹配特征作为阳性对照：本流程的配对效应与作者 beta 的 Spearman rho=0.943，
  方向一致率 90.5%，说明 dark-feature 配对效应不是简单的 EIC 工程伪影。

### 全量注释后的允许与禁止

- **允许：** 将 12 个作者跨平台复现候选作为流程阳性对照；将 ADP/ADP-ribose/
  ascorbate/quinolinate 作为作者表外优先 Level-2/连接性家族假说；写“phosphorylated
  nucleotide/NAD-related pool redistribution”和“expanded antioxidant pools”。
- **禁止：** 把 81 个模块写成 81 个新代谢物；把 ADP hub 写成特异通路锚点；声称 ATP
  energy charge、PARP/CD38/NUDT5/QPRT 活性、代谢通量、肿瘤依赖或临床标志物。
- **下一门：** 优先完成论文图、同位素/加合物与人工镜像谱总审计；若可采购，ascorbate 与
  quinolinate 标准最具性价比。独立 proteogenomic 队列只用于酶/通路背景，不可冒充代谢物丰度复制。

对应工件：

- `data/validation/lcnec_hsst3n_acquisition_gate/acquisition_gate.json`
- `data/validation/lcnec_hsst3n_qc_headroom_gate/qc_headroom_gate.json`
- `data/validation/lcnec_hsst3n_author_overlap_gate/author_overlap_gate.json`
- `data/validation/lcnec_hsst3n_dark_eic_gate/dark_eic_gate.json`
- `data/validation/lcnec_hsst3n_dark_robustness_gate/robustness_gate.json`
- `data/validation/lcnec_hsst3n_known_eic_validation/known_eic_validation.json`

## 停止条件与重新开放条件

- **SgME 重新开放：** 作者补充 pooled-QC 多碰撞能 MS/MS 原始谱及缺失中间工件，且 51 个
  真实 gap 中至少 12 个有可链接碎裂。
- **LCNEC 已升级：** 单平台暗特征门、全量注释门、作者跨平台复现门、四候选结构门和患者
  一致性门均通过。
- **Lysosomal atlas 升级：** 先获得 processed feature/annotation ledger 或从小型组织包证明存在
  大量原文未解释且可跨组织/干预复现的 MS/MS family；否则不下载 132.89 GB。

## 当前一句话战略

**主线正式迁移至 LCNEC：其样本量、质量控制、跨平台复现和作者表外候选数均明显高于
MTBLS13729；MTBLS13729 保留为冻结保底。接下来不再继续换数据集，而是把 LCNEC 的
四个优先候选与两组丰度模式收束成可复核、不过度因果化的论文结果。**
