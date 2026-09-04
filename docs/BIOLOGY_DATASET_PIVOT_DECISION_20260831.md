# 生物学数据集换题裁决（2026-08-31）

> **状态更正（2026-08-31，硬门实测后）：本文最初的“SgME 主线首选”已被后续原始
> mzML 采集审计否决，不再作为当前决策。最终裁决见
> `docs/BIOLOGY_DATASET_PIVOT_HARD_GATE_RESULT_20260831.md`。未经新的 pooled-QC
> 多碰撞能 MS/MS 或缺失的 `PM_groups.rds` 补齐，不得启动 88/218 文件扩展下载。**

## 结论先行

**主生物学开发应从 MTBLS13729 转出，但不删除 MTBLS13729。**

- MTBLS13729 冻结为“算法落地与证据校准病例”：保留 Level-1 free Neu5Ac、同患者 free-pool/activated-donor 解耦、hybrid mucin-glycome 解释及其全部限制。
- 新主线首选 **MTBLS13432 / SgME-HCC（Molecular Systems Biology, 2026）**。
- **LCNEC atlas（Zenodo 19005638；Biomedicine & Pharmacotherapy, 2026）** 作为并行的高质量 DDA 注释/泛化验证集，不首先承担主机制故事。
- MSV000100574/100560 暂不作为主线：规模大，但当前沉积仍不完整、样本映射和可引用论文关系不够清楚。

这不是因为 MTBLS13729 “完全失败”，而是它已达到无新湿实验条件下的证据上限。继续增加相关 RNA 队列只能增加背景一致性，不能补上独立黏液型丰度复现、同法标准、糖链去向和因果实验。

## 硬门比较

| 候选 | 设计与数据质量 | 可用 MS2 | 未解决生物学问题 | 注释增量空间 | 与本项目直接兼容 | 裁决 |
|---|---|---|---|---|---|---|
| MTBLS13432 SgME-HCC | 26 位患者、117 个肿瘤/邻近组织；bulk LC-MS/MS + DESI-MSI/H&E + RNA；作者代码与处理数据公开 | pooled-QC 上对 265 个高丰度离子做多碰撞能 MS/MS；230 个被 MS/MS 支持 | 作者明确提出 ME-low-grade+/ME-necrotic- 代谢簇的突变式变化机制未知 | 8,742 个 LC-MS features 中作者仅把 265 个送入核心 MS/MS 流程；230 个高丰度 PM 中 24 个标为 Unannotated，另有 35/265 未确认 | 很高：DreaMS/shared embedding/P2b 做身份候选，BioAware 做冻结身份后的网络解释，空间 MER 是预先存在的生物学终点 | **主线首选，但先过 72 h 注释头空间门** |
| LCNEC atlas | 34 对肿瘤/癌旁；四个 untargeted 平台；pooled QC、blank、QC dilution、注射顺序和临床协变量齐全 | 四个平台均为 MS1 + DDA，raw mzML 全公开 | 稀有肺神经内分泌癌代谢图谱仍有拓展空间 | 原文已报告 1,052 个代谢物；暗特征空间需从 raw 重新提取后量化 | 很高，尤其适合检验算法能否在真实 DDA 中提高身份覆盖且不制造假阳性 | **外部注释/泛化验证首选；主机制备选** |
| MTBLS13729 | 30 对组织，但无 pooled QC/blank；Rmu 只有 10 对 | 四面板 DDA 可用 | 已形成 discovery-level hybrid mucin-glycome | 新身份和独立验证空间已被反复深挖，关键升级依赖标准/独立队列/湿实验 | 已完全接入 | **冻结，不再作为主机制开发集** |
| MSV000100574/100560 | 约 2.6k 文件、约 347 GB、约 4.1M spectra | Q Exactive LC-MS/MS | 胰腺癌方向潜在价值高 | 可能很大 | 技术上可接 | **沉积 Partial、元数据/论文关系未闭合，暂缓** |

## 为什么 MTBLS13432 值得优先，但不能盲目押注

### 真正优势

1. **作者预先定义了一个尚未解释的现象。** 约 21% 的 175 个空间可检测高丰度代谢物属于 ME-low-grade+/ME-necrotic- 簇：低级别肿瘤区升高，在坏死区骤降。作者明确称其可能反映“currently unknown mechanism”。我们不是事后从几千个峰里挑故事。
2. **身份错误会直接改变空间生物学解释。** 原文也强调准确代谢物注释是空间代谢组的关键限制，且所有身份仅为 MSI Level 2。
3. **所有关键中间层可复算。** 作者公开 8,742-feature bulk 表、230-PM 表、MER 映射、RNA 表和完整 R 代码；不必重建 300 GB DESI 原始图像才能先做试验。
4. **本项目模块有清晰分工。** 新 embedding 和 P2b 只处理谱学身份；BioAware 在身份冻结后提供反应/通路一致性；MER 和 RNA 只用于生物学解释，不反向泄漏进身份选择。

### 硬风险

1. 8,742 个 feature 并不等于 8,742 个带 MS2 的 feature。原文核心碎裂池只有 265 个高丰度离子；因此不得宣传“将几千个暗峰全部注释”。
2. 作者已经对 230 个高丰度峰做了较深的手工 HMDB/LIPID MAPS 注释。我们必须证明身份纠错或新增身份改变了 MER 解释，而不是只换一个候选名。
3. 无标准品时仍是 Level 2。算法分数、网络支持和空间相关不能升级为 Level 1。
4. 原文已有 COMET's Path；我们的创新不能是重复相关分析，而应是“更可信身份使未知 MER 簇出现新的、可审计的代谢反应结构”。

## 72 小时淘汰式预实验

### 阶段 A：不下载大型 MSI，先用作者处理数据和少量 LC-MS 原始文件

1. 完整复现 8,742 features、265 tested ions、230 MS/MS-supported PMs、206 database-matched PMs及六个 MER 簇。
2. 下载三个 LC 模式的 assay metadata、pooled-QC MS/MS 文件和每模式 2–4 个代表样本；统计可链接 MS2 数、碰撞能、precursor/adduct 与 feature 命中率。
3. 用完全冻结的官方 DreaMS、当前 shared embedding、P2b 和经典谱库生成候选证据账本；表型、MER、grade 和 RNA 禁止进入身份打分。

### 阶段 B：Go / No-Go 门

只有同时满足以下条件，MTBLS13432 才升级为主生物学课题：

1. **协议复现：** 作者 230 个 PM 的 feature/RT/mode 链接复现率至少 95%，已命名高置信身份的方向性一致率至少 90%。
2. **真实增量：** 在 59 个“未确认或未注释”高丰度离子中，至少 12 个获得谱学可审计的 Level-2 候选；其中至少 6 个在作者 175-feature 空间聚类池内。
3. **机制落点：** ME-low-grade+/ME-necrotic- 簇中至少新增/纠正 4 个身份，并形成一个不依赖单峰的化学家族或相邻反应模块。
4. **安全门：** 对作者已有强谱学证据的身份，高置信冲突率不超过 5%；所有冲突必须逐峰解释。
5. **生物学稳健性：** 新模块的 MER 趋势不能由单一患者或单一 section 驱动；必须做 patient/block clustered bootstrap 或 leave-one-patient-out。

若第 2 或第 3 门失败，立即停止把 MTBLS13432 当主机制故事，转而把它作为空间注释 benchmark；主数据切到 LCNEC raw DDA。

## LCNEC 的并行价值

LCNEC 数据是当前最干净的真实 DDA 外测：34 对肿瘤/癌旁，四个 untargeted LC-MS 平台均包含 MS1+DDA，另有 PRM，且 QC、blank、serial dilution 和注射顺序完整。它适合回答两个硬问题：

1. 我们是否能在作者 1,052 个已报告代谢物之外，从 raw feature space 增加可复核的 MS2 身份？
2. 新 embedding/P2b 的增量能否在严格 QC 和患者配对统计下形成新的代谢模块，同时不破坏作者已确认的主结论？

但原文已经把 D/L-2HG、N-lactoyl amino acids、脂质重塑和吸烟暴露讲得较完整。因此在没有发现新的、成组且稳健的暗特征模块前，它更适合做算法泛化证明，不适合直接承诺一篇新机制论文。

## 资源与来源

- SgME-HCC article: https://doi.org/10.1038/s44320-026-00205-w
- SgME-HCC code and processed data: https://github.com/ccpagroup/sgme-hcc
- SgME-HCC raw data: MetaboLights MTBLS13432
- LCNEC article: https://doi.org/10.1016/j.biopha.2026.119327
- LCNEC raw mzML and statistics code: https://doi.org/10.5281/zenodo.19005638

## 最终项目定位

建议形成“一个主发现 + 两个外部应用”的结构：

1. **主方法：** 改进 shared DreaMS embedding + 安全候选专家 + 可审计解释层；
2. **主生物学：** MTBLS13432 中未知 MER 转换簇的身份重解析与反应模块；
3. **高质量外测：** LCNEC 四平台 DDA 中的注释覆盖和患者配对模块；
4. **临床证据校准案例：** MTBLS13729 hybrid mucin-glycome，保持 discovery-level 边界。

这比继续只在 MTBLS13729 上增加间接文献证据更有发表效率，也比直接下载数百 GB 新数据后再寻找问题更稳健。
