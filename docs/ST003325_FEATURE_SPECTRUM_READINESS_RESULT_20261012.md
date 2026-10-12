# ST003325 事件级扰动实验：谱学就绪结果

日期：2026-10-12  
状态：正式 readiness 结果；不是生物学阳性结果

## 裁决

ST003325 已通过进入结构隐藏扰动开发实验的最低数据门，但尚未证明任何 STS、DHEA-S 或未知化学响应。

## 数据、分母与泄漏状态

- 数据集：Metabolomics Workbench ST003325 / PR001844。
- 角色：`development/consumed`；不是封存外测。
- 效应泄漏：feature QC、DDA 探针、feature--scan linkage 和 MS/MS 查询集均未使用结构身份、扰动效应值或结果方向；随后设计审计读取了 GROUP 标签但未读取 feature effect。
- remote raw archive：24,202,566,631 bytes；官方 MD5 `c36568f353b09042ffa614db5af381a1`。

## 真实数据结果

定向 Range 读取只抽取 6 个 pooled-QC DDA mzML，共 327,608,125 bytes、44,307 spectra 和 39,043 MS2；其中 positive 21,101、negative 17,942，全部 MS2 具有 selected-ion m/z。

| 面板 | 全部 features | QC 合格 | QC 合格且在 >=2/3 DDA 中链接 | 分母内比例 |
|---|---:|---:|---:|---:|
| AN005444 HILIC negative | 12,265 | 1,029 | 364 | 35.37% |
| AN005445 RP positive | 42,345 | 2,051 | 440 | 21.45% |
| 合计 | 54,610 | 3,080 | 804 | 26.10% |

链接使用固定 5 ppm、10 s，QC detection >=0.75、study detection >=0.10、QC CV <=0.30、blank/QC median <=0.20。`804/3,080` 是 entity--scan link 覆盖，不是身份准确率。

804 个链接只形成 760 个唯一 MS/MS peak-list hashes；44 个谱学单元各映射到两个 abundance features，最大 multiplicity=2。因此后续保留 804 个 abundance entities，但只允许 760 个独立 spectral units 参与图或证据聚合。谱图峰数中位数 33.5，范围 6--192。

## 设计审计

每面板 72 个研究进样来自 12 个 clone（CE、KO、PTC、WT 各 3 个），每 clone 6 个 plate。Basal 固定在 plate 1--3，Basal+DHEAS 固定在 plate 4--6，条件与 plate number 完全绑定。两个面板测量同一组 72 个物理条件，不能合计成 144 个独立重复。

这仍允许一个受限但真实的验证：学习两种 STS 编辑机制相对 WT 的 plate-set response，并检验其是否迁移到第三种独立编辑机制及留出 WT clone。它不允许无条件声称纯 DHEA-S 因果效应。

## 科学意义与边界

这个结果的意义不是“我们已经找到了 804 个未知代谢物”，而是已经形成一个不依赖结构身份、具有真实 clone/edit-class 复现轴的事件级测试集。下一步必须先证明完整 MS1 响应本身能跨编辑机制复现，再问 WSE、Noise、P2b 或其他冻结谱学证据是否提供条件增量。

公共数据和 Atlas 在这里都只是可选消融。即使完全不使用公共参考，这个实验仍然回答核心科学问题；若公共证据不能超过研究内谱学和 matched shuffle，它应被删除而不是保留为故事。

## 机器可读证据

- `data/validation/st003325_targeted_dda_probe_20261011/report.json`
- `data/validation/st003325_feature_spectrum_linkage_20261011/report.json`
- `data/validation/st003325_perturbation_design_audit_20261011/report.json`
- `data/validation/st003325_structure_hidden_query_set_20261012_v2/report.json`

