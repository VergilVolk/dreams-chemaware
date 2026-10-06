# 任务 2 终版覆命报告（2026-10-06）

**状态：队列完成；两项外部核实受网络权限所阻，转 PI 侧一次点击。**

## 一、本轮四项完成情况

1. **扰动研究 Data availability 裁决**
   - Schuhknecht 2025：**FAIL**（仅处理矩阵 BioStudies S-BSST1724，无原始谱图，已按杀死规则出局）；
   - CRISPRi 2022（Anglada-Girotto）：数据登记号**已定位 PRIDE PXD024133**（文件级核验被 PRIDE JS 壳挡住，PI 侧一次打开即可定原始/矩阵）；
   - 2026 药筛（s41467-026-78024-8）：文章在 Nature 付费墙后，声明段待 PI 有权限时一瞥；
   - **普查中间结论**：旗舰资源不存原始谱图的模式下，"≥3 个原始 DDA 扰动研究"规则的存活概率低；若 PXD024133 与 2026 药筛也出局，扰动指纹线按预注册降级为单研究可行性注记。
2. **GNPS 天梯终版** ✅：`deliverables/GLM_gnps_article_ladder/.../ladder_full.csv`
   （20 行 × 35 列：R@1/2/3/5/10/20 + near 全套 + MRR + macro/micro AUROC/AUPRC + 双口径 pooled pairwise + corrected/introduced/risk-net）+ PDF/PNG/SVG 三格式图。S4 缺口（MS2DeepScore/Spec2Vec 权重）在 JSON 中显式登记。
3. **含氟命中用药史核查规格** ✅：`docs/GLM_LCNEC_FLUORINATED_HITS_MEDICATION_CROSSCHECK_SPEC_20261006.md`
   （五模块表型盲核查步骤 + 预注册判定规则：≥2 模块被用药史解释 → 改标药物暴露标志）。
4. **提交** ✅：本轮 `b0ee3db` 等共 10+ 笔全部推送 `module1-chem-attn-v6-0717`。

## 二、任务 2 迄今全部资产

- Gate-1 迁移上界卡（top-10 弃权式 98.7%）；
- LCNEC 反向搜索**修正权威表**：30 模块、中位 0.270、5 类级命中（273.08→C14H10F2N4 与 313.12→C14H15F3N4O 两个近前体候选；3/5 含氟）；
- 含氟用药史核查规格；天梯终版；普查文档（增补 1-7）。

## 三、PI 侧遗留（各一次点击）

1. `www.synapse.org/Synapse:syn53184679` 与 `…syn53190805`（Quartet 原始数据核验；我这侧 NIH 地域墙）；
2. PRIDE `PXD024133` 文件页（CRISPRi 原始/矩阵核验）；
3. Nature s41467-026-78024-8 的 Data availability 段（机构权限）。

## 四、诚实边界

- 反向搜索邻居均为**类级类似物**，非身份确认；结构主张永远等标准品/正交证据；
- 含氟→药物暴露是假说，用药史核查前不得写进任何结论；
- v1 试点的匹配 bug 教训已入档：手写匹配必须 mz 排序 + 双指标复核。
