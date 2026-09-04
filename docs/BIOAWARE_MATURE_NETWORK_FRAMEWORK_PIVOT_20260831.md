# BioAware 转向成熟代谢网络框架的正式决策

**日期：2026-08-31**  
**目标：** 保留严格 BioAware 作为 DreaMS 接口与风险控制层，不再自研一套完整代谢网络注释算法；先完整复现一个成熟框架，再只针对一个已证实瓶颈做最小改进，最后研究网络证据能否指导共享谱图 embedding 微调。

## 1. 先纠正一个关键前提

历史 BioAware G3-v2 的“2 个修正、0 个新增”不能作为当前算法起点：其中一个修正依赖“零权重证据仍被计入 gate”，两个嵌套网络证据又被当成独立票。严格重放后先变为 1/0，再合并相关网络证据后变为 0/0。

因此可以复用的是：

- DreaMS 候选谱学分数；
- 种子可靠度、路径追踪、冲突、缺失和弃权接口；
- formula/scaffold/study 隔离、degree-preserving decoy、corrected/introduced 与固定 FDR 评价；

不能复用的是“BioAware 已经稳定改正 2 个错误”的性能结论。

## 2. 代表算法已经怎样处理我们的瓶颈

| 我们的瓶颈 | 成熟算法的已有措施 | 能解决多少 | 仍未解决的部分 |
|---|---|---|---|
| 一步 Rhea 覆盖低、种子少 | MetDNA/MetDNA3 递归传播，新注释成为下一轮种子 | 显著扩大网络可达范围 | 错误种子也会递归放大 |
| 反应网络过密、共同底物导致假边 | MetDNA3 用实验 feature 层约束知识 MRN，并用 MS2 edge 过滤 | 比纯知识图传播更候选特异 | 固定 modified-dot-product 对跨仪器、碰撞能和共享主峰仍敏感 |
| 谱图、反应、峰相关证据割裂 | KGMN/MetDNA2 联合 MRN、谱图相似与峰相关层 | 提供成熟的多层传播与去冗余主体 | 不代表这些证据在统计上独立，也缺少现代 embedding 校准 |
| 同位素、加合物、源内碎片和生化边互相冲突 | NetID 做全局网络优化，并区分质谱现象边与生化边 | 很适合解决离子家族与全局一致性 | 不是面向同分子式近异构体的主结构排序器 |
| 大量 feature 无法定结构 | TidyMass2 通过 feature-level 功能模块绕开完整身份注释 | 很适合生物学解释和模块发现 | 它明确不是新的结构身份注释算法，不能替代候选判别 |
| 多种谱学相似度各有偏差 | MS-Net 用 multi-similarity network 做候选缩减和网络注释 | 值得作为谱学网络消融 | 当前为预印本/KNIME 工作流，且不是反应网络先验本身 |
| 网络先验没有进入 DreaMS embedding | 上述方法基本都停留在传播、全局分配或功能模块层 | 没有现成答案 | 这是我们可以做的真正新增点，但必须由严格网络教师产生监督 |

## 3. 主框架选择

### 3.1 理想主框架：MetDNA3

MetDNA3 最贴合当前问题，因为它已经实现：

1. MS1 预映射；
2. 知识 MRN 与实验 feature 网络双层拓扑；
3. raw-MS2 feature-edge 约束；
4. 递归传播；
5. 候选去冗余；
6. decoy/FDR 评价。

我们不需要重写这些部分。最小创新只应放在一个位置：**用校准后的 DreaMS edge reliability 改善作者 feature-edge 的判定，并用保守交集降低错误传播。**

### 3.2 当前现实：公开 MetDNA3 不是完整可运行包

机器审计已确认：公开 `MrnAnnoAlgo3` 有核心 R 函数，但缺少正式运行所需的 `obj_mrn*.rda`、`info_mrn*.rda`、`md_mrn*.rda`；公开 release 也没有这些资产。作者 README 把完整功能和示例数据放在需注册的网站。

所以当前不能诚实地说“已复现 MetDNA3”。合法路径只有两条：

- 得到作者完整包、作者 webserver 输出或工作目录后，走精确 MetDNA3 桥接；
- 在此之前，以公共代码和网络资产齐全的 KGMN/MetDNA2 建立可复现成熟基线。

当前自动裁决文件为 `data/validation/metabolic_network_framework_reproducibility.json`，结果是 `kgmn_metdna2_reproducible_fallback`。

## 4. 最小且有价值的集成

### 4.1 不再做“多票加分器”

严格 BioAware 只输出四类信息：

- `spectral_unary`：DreaMS 对候选参考谱的校准证据；
- `seed_reliability`：种子是否可进入递归传播；
- `path_state`：available/support/conflict/unknown；
- `risk_state`：是否应弃权、是否回退到 DreaMS。

KGMN/MetDNA2 或 MetDNA3 负责候选生成、网络传播和去冗余。两者职责不交叉。

### 4.2 唯一首轮改进：DreaMS 约束实验谱图边

固定同一批作者候选边，比较：

1. 作者原始谱图相似度；
2. 校准后的官方 DreaMS 相似度；
3. 校准后的 noise-tuned DreaMS 相似度；
4. 作者分数与 DreaMS 的保守交集。

所有臂共享同一种子、MRN、MS1 候选、传播轮数和去冗余逻辑。只有边可靠度不同。这样才能把增益归因到 DreaMS，而不是改了整套网络流程。

### 4.3 风险控制不是第二个创新

degree-preserving decoy、reaction-size matched decoy、target-decoy FDR、unknown 状态和保守回退都属于必要评价纪律，不应包装成多个创新点。正式主指标是：

- 固定 1%、5%、10% FDR 的注释覆盖；
- Top-1/Top-3；
- corrected/introduced；
- coverage-risk；
- 传播深度、节点度数、seen/unseen reaction、formula/scaffold/study 分层。

## 5. 网络先验怎样指导 embedding 微调

网络相邻代谢物是不同身份，不能直接当正例拉近。正确做法是把网络当作**训练关系可靠度教师**：

1. 在训练 study 内交叉拟合成熟网络算法；
2. 只保留真实网络显著优于 degree-preserving decoy、跨折方向一致的候选 margin；
3. 每条训练样本仍由同身份谱图构成正例，由真实困难候选构成负例；
4. 网络只决定该 ranking triple 的可信度和权重，不定义 identity；
5. 训练 query/reference 完全共享的 DreaMS encoder；
6. 推理时只输入干净谱图，不再需要网络、候选或表型。

目标函数：

`L = w_network * softplus((m + s_neg - s_pos) / tau) + lambda_preserve * (1 - cos(z, z_official))`

必须比较四个教师：随机网络权重、仅身份 ranking、原成熟网络教师、DreaMS-constrained 网络教师。只有最后一项在外部数据上更好，才能说生物先验改善了 embedding。

## 6. 论文创新应怎样表述

如果实验通过，创新不是“我们发明了网络传播”，而是：

> 在成熟的代谢网络递归注释框架中，以校准的共享质谱表示约束实验 feature 网络边，降低知识网络扩张产生的错误传播；进一步将交叉拟合且通过 decoy 的网络候选 margin 蒸馏到候选无关的共享谱图 embedding，使网络先验只塑造困难候选的判别方向而不改变分子身份定义。

这个表述同时保留三条清晰边界：成熟传播归作者；DreaMS edge reliability 是我们的最小算法改进；网络到 shared embedding 的安全蒸馏是第二阶段研究问题。

## 7. 执行顺序

1. 完成 KGMN/MetDNA2 作者原协议复现；
2. 冻结作者基线、输入、随机种子和输出哈希；
3. 用严格 BioAware/DreaMS 生成校准 edge reliability；
4. 运行四臂对照，不改传播主体；
5. 外部 study 通过 fixed-FDR 与 corrected/introduced 门后，才进入 embedding 蒸馏；
6. 若获得完整 MetDNA3 工件，用相同接口替换 KGMN 主体，保留全部对照协议。

禁止跳过第1步直接宣称“DreaMS 改进 MetDNA3”，也禁止用历史 2/0 作为外部性能基线。

## 8. 当前冻结进展（2026-08-31）

### 8.1 作者基线协议已纠偏

首次审计发现基线脚本错误地关闭了 `is_credential`，同时开启了 200STD 对照不应使用的 RT 校准；该组合只复现 MRN 递归的一部分，不是完整 KGMN/MetDNA2。现已修为：

- `is_credential = TRUE`；
- `is_cred_pg_filter = TRUE`；
- `is_cred_formula_filter = FALSE`；
- `is_pred_formula_all = FALSE`；
- `is_rt_calibration = FALSE`；
- 使用仓库自带 Linux ELF GenForm，并复制到只读运行工件目录；
- 源仓库必须 clean，commit、GenForm、输入、参数和结果全部哈希冻结；
- 冻结器必须看到真实 credential 产物，不能只凭日志判断运行完成。

因此服务器重新运行前必须先同步最新脚本；旧的 credential-off 结果即使成功也不得作为作者基线。

### 8.2 outcome-free 边校准清单已冻结

已构建 `data/validation/kgmn_dreams_edge_calibration_manifest_20260831/`：

- 448 个“真实反应边 vs exact-formula 非网络诱饵”配对三元组；
- 154 个 acquisition-matched 唯一反应边；
- 397 个身份、169 个分子式、44 个网络组件；
- 五折按网络 component 隔离，P3 身份重叠为 0；
- 425 个严格 instrument+adduct 匹配，23 个 instrument+polarity 回退；
- 清单中无任何 action outcome 或测试真值列。

这只是校准与机制评价样本，不等同于最终注释性能。下一步是在该清单上冻结作者 DP、官方 DreaMS、noise-tuned DreaMS 和保守交集四臂的校准器，然后才允许注入同一作者候选边进行传播评价。

### 8.3 作者边分数与动态传播接口已改为严格复现

边校准不再使用 Python 近似 reverse-dot 作为“作者分数”。正式作者臂现在：

- 将冻结三元组涉及的全部有效碎片和 precursor metadata 导出；
- 在隔离的 MetDNA2 1.2.10 R 运行时内直接调用作者 `convertSpectraData` 与 `runSpecMatch(..., scoring_approach="dp", mz_tol_ms2=25)`；
- 按作者递归代码的规则，总让较小 precursor 的谱图作为 reference；
- 输出顺序、输入及分数文件均哈希绑定，Python 只读取和校验，不再重写作者相似度；
- 校准性能完全由 network-component 外折产生；
- 外部部署使用“全训练校准器 + 同一全训练概率尺度上冻结的阈值”，不把 OOF 阈值错套到另一个概率尺度；
- KGMN overlay 只在动态 feature-feature MS2 edge 处插入 hook，并显式将闭包导出到 PSOCK workers；
- `noop_author` 必须在 credential、table1 和 table3 三张表上逐行多集等价复现作者结果，否则两个实验臂不会启动。

因此当前能检验的科学问题被严格收窄为：在完全相同的候选生成、MRN、递归层数、峰相关 credential 与去冗余流程中，校准后的 DreaMS 是否比作者 DP 更可靠地决定动态谱图边。它还不是共享 embedding 改进，也不是独立队列 SOTA 结论。

## 9. 代表文献与代码

- MetDNA3 (Nature Communications, 2025): https://www.nature.com/articles/s41467-025-63536-6
- MrnAnnoAlgo3: https://github.com/ZhuMetLab/MrnAnnoAlgo3
- KGMN/MetDNA2 (Nature Communications, 2022): https://www.nature.com/articles/s41467-022-34537-6
- NetID (Nature Methods, 2021): https://www.nature.com/articles/s41592-021-01303-3
- TidyMass2 (Nature Communications, 2026): https://doi.org/10.1038/s41467-026-68464-7
- MS-Net workflow (2025 preprint artifact): https://zenodo.org/records/17669288

## 10. 当前目标替换与近期验收门

### 10.1 目标替换

原来的远期目标——从零构建覆盖候选生成、网络传播、全局一致性、风险控制和上下文 embedding 的完整 BioAware——降为后续研究储备。当前唯一主目标改为：

> 以严格修复后的 BioAware 作为 DreaMS 接口与安全层，完整复现 KGMN/MetDNA2；保持作者候选生成、KMRN、递归传播和 peak credential 不变，只用校准后的 DreaMS 改写动态谱图边可靠度，并在匹配协议与独立 hidden-seed 协议上检验是否减少错误传播。

历史 G3-v2 的路径追踪、冲突状态和弃权接口可以复用；历史“2 corrected / 0 introduced”不进入性能总账、模型选择或论文主张。

### 10.2 为什么不是直接移植 MetDNA3、NetID 或 TidyMass2

- **MetDNA3** 是最合适的最终主体，但公开仓库缺少正式 MRN 工件，当前无法完整、独立复现；获得作者工件后再用同一 edge-reliability 接口替换 KGMN。
- **NetID** 的强项是离子形式、同位素、源内碎片与生化转换之间的全局一致性；它可作为后续 ion-family 冲突消解模块，但不能替代近异构体的结构候选判别。
- **TidyMass2** 的强项是让未定结构 feature 进入功能模块分析；它应服务生物学解释和注释率口径，不应被误写为结构身份重排器。
- **KGMN/MetDNA2** 同时具备公开代码、网络资产、递归传播、谱图边和全局 peak credential，是目前唯一能用 no-op 严格验证“只改一处”的成熟主体。

### 10.3 第一阶段唯一创新变量

第一阶段只允许改变递归传播时 feature-feature MS2 edge 的可靠度：

1. `noop_author`：作者 DP，验证完整流程等价复现；
2. `official_dreams`：同一边使用 component-isolated 校准后的官方 DreaMS；
3. `author_official_intersection`：只有作者 DP 与 DreaMS 同时支持才开放传播边；这是预注册主模型；
4. noise-tuned DreaMS 只有在独立 eligibility 报告通过后才允许加入，不得自动冒充正式臂。

不允许同时改 RT、候选库、种子、传播深度、MRN、credential 或去冗余规则。否则任何提升都不能归因于 DreaMS edge reliability。

### 10.4 近期成功标准

开发基准只能用于方法调试。进入外部验证必须同时满足：

- no-op 在 credential、候选表和最终注释表上与作者结果多集等价；
- 预注册主模型相对作者基线 `corrected > introduced`，且 Top-1、Top-3/Top-5、coverage 不下降；
- component/formula cluster bootstrap 的主要效应下界大于 0；
- 增益不只来自低度数、已见 reaction 或单一分子式；
- 传播深度增加时错误率不单调恶化。

独立 hidden-seed 生物数据采用 study 内保留 30% Level-1 代谢物为种子、隐藏 70% 作验证的协议；所有阈值只在训练 study 内确定。该阶段通过后，才允许把网络教师产生的可信候选 margin 蒸馏到共享 DreaMS embedding。

## 11. 外部验证数据合同已经冻结

KGMN 作者 Supplementary Data 1--3 已按 Zenodo 7089991 的 MD5 校验并解析。这里把三个评价问题严格拆开，避免再次用一个小面板同时声称覆盖率、结构准确率和生物学机制：

### 11.1 主评价：46STD/S9 隐藏种子传播

- 作者数据中实际观测到 42 个 Level-1 身份、80 个正负模式 peak rows；
- 采用 10 次固定重复，每次 13 个种子、29 个隐藏身份；
- 按“仅负、正负均有、仅正”三层分层；
- 使用一次随机置换后的循环窗口，而不是十次独立随机抽样；
- 因而每个身份都至少一次为种子、至少一次为隐藏验证目标；
- 各极性层内种子曝光次数最多相差 1：仅负固定 4 次，正负均有 2--3 次，仅正 3--4 次。

主指标为 hidden-identity coverage、Top-1、Top-3、相对作者 DP 的 corrected/introduced、传播深度错误率。所有 edge calibration 和 gate 必须在 OEP003284 之外冻结。

### 11.2 次评价：人工核验的 peak/ion-form 准确率

Supplementary Data 1 给出五类生物样本、两种极性下共 3451 个人工核验 peak，覆盖 242 个身份。该面板用于评价：

- Top-3 身份；
- isotope/adduct/in-source-fragment 归属；
- global peak credential 是否降低错误 peak assignment。

它不能替代 46STD/S9 的隐藏种子传播评价，因为同一真实代谢物对应大量 ion forms，统计单位和问题完全不同。

### 11.3 仅机制核对：标准品确认的产物

去掉多候选歧义后只有 20 个 feature rows、9 个独立化合物。该面板只允许展示具体反应路径和错误案例，不允许调参，也不能承担独立性能结论。

### 11.4 当前工件

- 来源下载与哈希：`tasks/fetch_kgmn_external_validation_sources.py`；
- 外测协议冻结：`tasks/freeze_kgmn_external_validation_contract.py`；
- 服务器入口：`tasks/run_kgmn_external_validation_contract.sbatch`；
- 单元测试：`tests/test_freeze_kgmn_external_validation_contract.py`。

本地已用作者 Supplementary Data 1--3 完整重放并通过测试。该合同仍不等同于算法结果；完整主评价还需要 OEP003284 原始 LC-MS 和作者 KGMN runtime。

外测评价器已另行实现为 `tasks/evaluate_kgmn_hidden_seed_recovery.py`。作者臂和候选臂必须提交相同长表字段，评价器自行重算候选身份排序；真候选同分并列计错、缺预测仍留在分母、重复路径按候选身份取最强证据。主效应按身份聚类 bootstrap，而不是把同一代谢物的十次重复当成十个独立样本。

## 12. 当前真正的瓶颈与对应成熟措施

1. **不是“一步 Rhea 不够深”这么简单。** 递归传播确实能提高覆盖，但 MetDNA/KGMN 已证明错误会随递归扩散。我们的首要变量不是继续加 hop，而是让每次传播边的开放更可信。
2. **不是把反应邻居直接拉进同一 embedding。** 反应邻居是不同分子，直接作正例会损害结构区分。成熟方法把 MRN 用于候选生成与传播；我们的 embedding 阶段只蒸馏通过外折和 decoy 的排序 margin。
3. **不是只优化候选 Top-1。** KGMN 的主要强项之一是 global peak correlation，把同一代谢物的 isotope/adduct/ISF 统一起来；因此主结构排序和 ion-family assignment 必须分别报告。
4. **不是让网络覆盖替代身份置信度。** TidyMass2 说明未注释 feature 也能支撑功能模块，但这种“可解释覆盖”不能冒充 MSI Level 2/1 身份。
5. **不是再造一个加权投票器。** MetDNA3 已用知识层与数据层的拓扑交集限制传播，NetID 已用全局约束解决峰注释冲突。我们只需要给实验谱图边增加经过 component-isolated calibration 的 DreaMS 可靠度，并保留作者全局去冗余。

因此近期目标不是承诺 10 pp，也不是宣布 SOTA，而是得到一条可审计因果链：作者 no-op 完整复现 -> 只替换 edge reliability -> 外部隐藏种子显著减少 introduced -> 再把这些外折可信 margin 蒸馏到共享 embedding。任何一环失败，就停在对应模块，不用下游结果替上游背书。

## 13. 本轮工程审计修复

1. 作者 baseline 冻结器要求 `run.log.txt`，但原 sbatch 没有保存 R 输出，导致算法即使跑完也必在冻结阶段失败。现已使用临时日志、`pipefail` 和完成后原子移动保存日志，再启动冻结器。
2. 三臂传播脚本原先只验证校准工件哈希，没有检查 `official_dreams_eligible_for_dynamic_propagation_test`。这会让未通过 component/formula 隔离门的 DreaMS 继续进入传播。现已改成 fail-closed，门未过则三个正式传播臂均不启动。
3. hidden-seed 原先十次独立随机抽样会让 3 个身份从未充当种子。现已改为分极性的一次置乱加循环窗口，保证每个身份既被隐藏也被用作种子。
4. 新增 sbatch 合同测试，统一要求 `--gpus=1`、禁止显式超额内存申请，并检查日志冻结和校准门发生在正式传播之前。

这些修复不增加算法性能，却是后续任何“改进来自 DreaMS 边”的必要可复现基础。

## 14. 成熟算法对各瓶颈的直接处方与我们的取舍

### 14.1 种子稀少、一步网络覆盖低

- **MetDNA 的处方：** 已注释种子沿 MRN 递归传播，新注释继续成为种子。
- **KGMN/MetDNA2 的增强：** 每条传播同时受代谢反应、MS1、RT 和 MS2 约束，并在传播后用全局 peak correlation/credential 清理冲突。
- **MetDNA3 的增强：** 先构建知识层与实验 feature 层，再只沿两层预映射一致的边递归，避免每轮重复搜索完整网络。
- **我们的取舍：** 完整复用成熟递归主体，不再自己写 hop 加分。创新只检验 DreaMS 是否能提高实验 feature-edge 的可靠度。

### 14.2 网络枢纽、共同底物和错误种子的放大

- **成熟措施：** 实验层约束、传播深度限制、候选去冗余、全局 peak credential，以及 target-decoy/FDR。
- **仍存在的空位：** 作者固定 DP 阈值没有针对现代 embedding 的跨仪器、碰撞能漂移和共享主峰行为做校准。
- **我们的取舍：** 不扩大网络，不增加 hop；按 reaction component 外折校准 DreaMS，预注册作者 DP 与 DreaMS 的保守交集为主模型，并逐深度报告 introduced。

### 14.3 同位素、加合物、源内碎片被误当作多个代谢物

- **NetID 的处方：** 用质量差规则和全局优化统一离子现象边、生化转换边、RT 和 MS2，寻找全局自洽解。
- **KGMN 的处方：** global peak correlation network 对代谢物子网络进行 credential，过滤冲突与冗余 peak assignments。
- **我们的取舍：** 第一阶段保留 KGMN credential，不自研第二套全局求解器；若外测显示主要剩余错误是 ion-form 冲突，再把 NetID 作为第二个正交模块，而不是把它混入结构候选排序。

### 14.4 近异构体谱图相似、固定相似度边不可靠

- **成熟算法的局限：** MRN 可以缩小候选空间，却不能保证相邻结构在 MS2 上可区分；固定 DP 和单一阈值容易被共享碎片、碰撞能与仪器差异影响。
- **我们的单点改进：** 在作者候选边完全不变的前提下，用 component-isolated、exact-formula decoy 校准的 DreaMS edge probability 替换或收紧作者 DP 边。
- **验收方式：** 不看训练边 AUC，而看 hidden-seed 身份恢复的 corrected/introduced、fixed-FDR coverage、传播深度错误率，以及 unseen reaction/formula/scaffold 分层。

### 14.5 网络先验如何进入 embedding 而不破坏身份空间

- **禁止做法：** 把一跳反应邻居当成正例直接拉近。反应邻居是不同身份，这会重演“正负一起抬高”的检索退化。
- **允许做法：** 网络只提供候选排序关系的可信度；正例仍是同身份跨条件谱图，负例仍是真实困难候选。
- **训练监督：** 只蒸馏外折网络中相对 degree-preserving decoy 显著、跨折方向一致、且作者/DreaMS 交集支持的 margin；所有其他样本权重为零或只做官方表示保持。
- **推理合同：** query/reference 使用同一个共享 encoder，输入只有干净谱图；网络、候选和表型均不得成为推理输入。

## 15. 当前落地状态与唯一外部阻塞

已完成并通过本地合同测试的部分：

1. KGMN/MetDNA2 作者 200STD 基线、DreaMS edge overlay、三臂传播和冻结器；
2. 448 个 paired triples、154 个唯一反应边的 outcome-free 校准清单；
3. 46STD/S9 的 10 次隐藏种子合同与严格评价器；
4. hidden-seed 状态净化：同时过滤 CSV seed 表与 S4 `result_annotation`，不复制带身份的 library cache，避免“隐藏身份仍通过缓存泄漏”；
5. OEP003284 输入 fail-closed 预检；
6. 相关 Python、R 和 sbatch 合同测试共 14 项通过。

当前唯一外部数据阻塞是：本地尚无 OEP003284 正负模式的完整 KGMN 输入（MS1 feature table、sample metadata 和 MS2 MGF/MSP/MzXML/CEF）。这不阻塞 200STD 作者基线与 edge calibration，但阻塞 46STD/S9 的真实 hidden-seed 外测。缺数据时不得拿内部清单或 MTBLS13729 代替外部性能结论。

服务器近期执行顺序固定为：

1. 作者 200STD no-op 完整复现并冻结；
2. 在冻结边清单上完成作者 DP 与官方 DreaMS component-OOF 校准；
3. 只有 DreaMS eligibility gate 通过，才运行 `noop_author / official_dreams / author_official_intersection`；
4. 同步 OEP003284 后运行 hidden-seed 外测；
5. 只有外测显著减少 introduced，才构建 network-teacher-to-shared-embedding 数据。

## 16. OEP003284 隐藏种子执行桥已经补齐

截至本轮，不再只停留在“有合同、无执行器”。外部主评价已经形成如下闭环：

1. `run_kgmn_metdna2_46std_initial_seed.R` 在作者原版 MetDNA2 1.2.10 中分别生成正、负模式完整 Level-1 初始种子状态；此时关闭递归、credential 和生物解释，只形成后续白名单过滤所需的执行缓存；
2. `prepare_kgmn_hidden_seed_state.R` 按冻结的十次循环窗口同时过滤 CSV 与 S4 初始注释对象，只保留当次允许的种子，隐藏身份仍保留在 MRN 候选空间和无标签谱图中；
3. `export_kgmn_feature_dreams_embeddings.py` 只接受能够一一映射回 MS1 feature name 的 MSP/MGF，使用官方共享 DreaMS encoder、eval mode 和固定 checkpoint；CEF/MzXML 在没有 identifier-preserving 转换前只允许作者臂，不允许冒充 DreaMS 实验臂；
4. `run_kgmn_metdna2_46std_hidden_seed_arm.R` 在相同准备状态上运行 no-op、official DreaMS 和作者-DreaMS 交集臂；候选生成、MRN、递归深度、credential、formula filter 和最终计分均保持作者实现；
5. `export_kgmn_hidden_seed_predictions.R` 从 MetDNA2 内部 `table_identification` 取得最终 `total_score`，从 `list_identification` 取得公开 CSV 已丢弃的 `round`，按 peak、候选身份和 adduct 严格对账；
6. `combine_kgmn_hidden_seed_predictions.py` 要求完整 10 repeat x 2 polarity 的 20 个分片，缺任何分片即失败；
7. `evaluate_kgmn_hidden_seed_recovery.py` 将缺预测保留在分母，候选同分并列计错，重复 ion form/路径按候选身份取最强证据，按身份聚类 bootstrap，并报告传播深度分层。

服务器总入口为 `tasks/run_kgmn_oep003284_hidden_seed.sbatch`。它首先要求冻结校准器已经通过 dynamic-propagation eligibility gate，然后安装相互隔离的作者包和 overlay 包。两个极性各运行一次外部 no-op，并要求作者与 no-op 的 table1、table3 字节完全一致；只有该技术门通过，才启动十次作者臂和两个实验臂。

本地目前通过 25 项相关合同与单元测试，包括输入格式边界、隐藏标签过滤、MSP/MGF identifier 对账、空预测/缺分片 fail-closed、冻结校准门、no-op 顺序和正式 sbatch 的 GPU/内存合同。该数字只代表工程合同覆盖，不代表任何外部性能结果。

### 16.1 当前仍然不能宣称的内容

- 没有 OEP003284 的规范化正负模式 MS1 表、sample metadata 和 identifier-preserving MSP/MGF，就不能运行 DreaMS edge 外测；
- 200STD 的一致性和校准结果不能替代 OEP hidden-seed 性能；
- hidden-seed 通过也只说明网络传播边改进，不说明 shared DreaMS embedding 已经改善；
- 第二阶段 embedding 蒸馏仍须使用同身份跨条件正例和真实困难负例，反应邻居只提供 margin 可信度，不成为身份正例。

## 17. OEP003284 正式输入与隐藏种子终审（2026-08-31）

本轮发现并修正了一个会直接改变结论的协议问题：Supplementary Data 3 已经公开作者用于 46STD/S9 的完整正负峰表，分别为 15,942 和 16,760 个 feature。因此正式主线不再使用自建 XCMS 峰表；作者峰表原样保留缺失值、强度、m/z、RT 和 feature name。XCMS 重建只允许作为敏感性分析，不能替代正式输入。

新增的作者 identifier contract 实测得到：

- 80/80 个 Level-1 peak rows 全部存在于对应极性的作者峰表；
- 42 个 Level-1 身份对应 42 个分子式，身份聚类与分子式聚类在这个小面板中完全等价；
- 20/20 个标准品确认 feature rows 全部存在；
- 真值表与原始峰表的最大 m/z 差为 4.93e-5 Da、最大 RT 差为 0.0495 s，均是作者汇总表四舍五入量级；
- 正式真值键冻结为 `polarity + peak_name`，而不是重新做 m/z/RT 最近邻匹配。

原始 mzXML 到 DreaMS 的桥也不再由我们另写一次 feature mapping。正式执行先让作者 MetDNA2 `combineMs1Ms2` 完成 MS1--MS2 对接及代表谱选择，再把该无标签 cache 导出为保留原 feature name 的 MSP，最后用固定官方 DreaMS checkpoint 编码。这样作者 DP 和 DreaMS 比较的是同一批作者映射谱，避免 mapper 差异冒充 embedding 增益。

隐藏标签终审增加两道 fail-closed 门：

1. 每次 hidden-seed 状态同时白名单过滤 seed CSV 和 S4 `result_annotation`；随后递归扫描 `ms2` 与 `ms1_data` cache 的值、名称和 S4 slot，任何隐藏 InChIKey 块出现即终止；
2. 外部 no-op 必须同时复现 credential 长表、最终宽表和候选长表的完整行多集；不再只比较 table1/table3，更不能依赖文件字节序偶然一致。

正式统计口径也被冻结：缺预测保留在分母、同分并列计错、重复路径按候选身份最大分聚合；主模型是预注册的作者-DreaMS edge 交集。通过要求 Top-1 身份聚类 bootstrap 下界大于零，Top-3 与覆盖率的身份聚类下界非负，corrected 大于 introduced，并且单侧 McNemar p 不高于 0.05。传播深度分层的 corrected/introduced 另行完整报告。

必须明确，这仍是作者 `test_evaluation=46STD` 的 closed-world 候选宇宙：隐藏的是初始种子标签，不是把该化合物从候选 MRN 中删除。它适合检验“同一成熟 KGMN 主体内，DreaMS edge reliability 是否减少错误传播”，不等于 open-world 未知物注释，也不能直接宣称 SOTA。

服务器执行固定为三步：

1. `sbatch tasks/run_kgmn_metdna2_200std_baseline.sbatch`
2. `sbatch tasks/run_kgmn_dreams_edge_calibration.sbatch`
3. OEP003284 的 24 个 NODE mzXML 通过 MD5 验证后，依次运行 `sbatch tasks/run_kgmn_oep003284_author_inputs.sbatch` 和 `sbatch tasks/run_kgmn_oep003284_hidden_seed.sbatch`

最终只读取 `data/validation/kgmn_oep003284_hidden_seed_20260831/final_decision.json`。若预注册交集臂失败，官方 DreaMS 次要臂即使更好也不能事后替代主结果；若通过，才允许构建 network-teacher-to-shared-embedding 数据，仍不能把反应邻居直接作为身份正例。
