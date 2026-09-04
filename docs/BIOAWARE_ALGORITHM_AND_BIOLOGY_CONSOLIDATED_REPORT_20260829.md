# BioAware 算法与 MTBLS13729 生物学课题综合成果报告

**版本日期：2026-08-29**  
**用途：** 组会汇报、论文框架、阶段决策与后续开发共同事实底稿。  
**证据原则：** 区分模型权重、embedding 后专家、oracle/headroom、开发 OOF、封存/外部验证；正结果、负结果和工程边界同时保留。

---

## 1. 结论先行

BioAware 已经完成从“一跳代谢网络加分器”到“候选特异、多证据分解、风险受控、允许弃权”的方法学演进，并形成了可复现的数据层、证据层和评价层。原始 G3-v2 在 117-query、formula-group nested OOF 的已消耗开发任务上报告 2 个修正、无新增错误，Recall@1 从 `0.8120` 提升至 `0.8291`（表面 `+1.71 pp`）。但 2026-08-30 的实现审计发现：gate 的支持计数包含了拟合权重为零的证据族，同时 reported-reaction 与 reported-plus-predicted-reaction 是嵌套且高度相关的证据，却被当成独立票。只修复零权重支持计数后，诊断重放降为 `1/0（+0.85 pp）`；再合并相关网络证据后为 `0/0`。因此 `+1.71 pp` 现仅保留为历史开发结果，不能作为当前有效性能主张。详见 `docs/BIOAWARE_STRATEGY_CRITICAL_AUDIT_AND_REDESIGN_20260830.md`。

BioAware 在 MTBLS13729 生物学应用中的最大已实现贡献，不是用反应网络强行更改身份，而是：

1. 用表型盲全局 MS1 峰图识别同位素、加合物和源内碎片关系，避免把同一代谢物写成多个发现；
2. 将 DreaMS、原始峰证据、离子家族、反应关系和样本层证据严格分层，冲突时保留歧义而非强行融合；
3. 在冻结注释之后独立进行患者内 MS1 定量，从而得到可审计的生物学发现。

目前已经形成两项主要生物学结果：

- **修饰鸟苷样模块：** Rmu 肿瘤相对匹配癌旁 10/10 同向升高；全局 PQN 后平均约 `+2.85 log2`（约 `7.2` 倍），精确 sign-flip `p=0.001953`，技术匹配背景经验 `p=0.000708`。
- **长链酰基肉碱类别：** Rmu 相对匹配癌旁平均 `+1.382 log2`（约 `2.60` 倍），精确 sign-flip `p=0.00586`；链级去冗余后仍为 `+1.368 log2`、`p=0.00586`。C20:4 锚点在 42 个样本中找到 59 条匹配 MS2，25 个样本具有强肉碱诊断模式。

这些结果支持“Rmu 中存在修饰核苷/嘌呤周转、长链脂酰基处理和多胺乙酰化的多轴代谢异常”。它们仍属于发现级静态丰度证据，不能推出代谢通量、具体酶、位置异构体或已确认的黏液型特异机制。

---

## 2. BioAware 的方法定义与边界

### 2.1 BioAware 是什么

BioAware 当前是位于 DreaMS embedding 之后的候选证据专家。其目标不是用“代谢网络相邻”代替谱图匹配，而是为每个候选分别构建可追溯证据：

- DreaMS 谱图相似度作为 unary evidence；
- 已知或预测反应关系；
- 样本中实际观测的 MS1 feature 节点；
- raw-MS2 feature-edge 支持；
- 同位素、加合物、共洗脱及跨样本丰度关系；
- 候选结构到碎片/中性丢失的特异 likelihood；
- RT 和结构网络作为辅助证据；
- 冲突、货币代谢物和高度数节点触发降权或弃权。

BioAware 不改变 DreaMS 权重，因此不能称为“更好的 embedding”。共享 embedding 的噪声微调、P2b 谱学重排和 BioAware 网络专家是三个正交模块：

| 模块 | 位置 | 是否改变 embedding | 当前作用 |
|---|---|---:|---|
| 噪声/ChemAware 微调 | DreaMS 编码器内部 | 是 | 改善共享表示空间 |
| P2b | embedding 后 | 否 | 通用谱学 rank fusion；常规检索保底 |
| BioAware | embedding 后 | 否 | 样本/反应/离子家族/候选特异证据与弃权 |

### 2.2 借鉴与原创边界

借鉴或对齐的部分包括：

- MetDNA/MetDNA3 的代谢反应网络递归传播；
- MetDNA3 的 MS1 预映射与 raw-MS2 feature-edge 约束；
- MetDNA3 的 SMN 与 RT 辅助证据；
- NetID 类方法的全局一致性思想。

本项目的主要方法学增量是：

1. 将 DreaMS embedding 检索分数作为候选 unary，而不是另建孤立网络系统；
2. 将网络可达性、真实观测 feature、raw-MS2、离子家族和候选结构证据分别保存，禁止把任一层冒充身份真值；
3. 每个 query 进行 truth-identity-held-out seed rotation，避免真值从种子侧泄漏；
4. 候选组内归一化后，每个独立机制族最多贡献一票，避免大量同源规则重复计票；
5. 使用 `corrected - 2×introduced`、formula-group OOF、公式簇 bootstrap、degree-preserving decoy 和 conflict abstention 控制风险；
6. 在真实队列中构建表型盲全局离子图，再进行冻结身份与患者内定量，避免循环论证。

可形成的创新定位是：

> **DreaMS-guided, evidence-factorized and risk-controlled global metabolite assignment**：将谱图表示、候选特异碎裂、观测 MS1 节点、反应 hyperedge、离子形式和跨样本关系作为独立因子联合求解，并在证据冲突时弃权。

当前实现已经完成大部分证据层和安全门，但尚未完成一个在外部 Level-1 大型任务上显著优于基线的最终全局分配器。

---

## 3. BioAware 算法成果总账

### 3.1 工程与数据资产

- Rhea 离线缓存：17,656 个反应、10,152 个化合物、78,843 条参与物记录，并单独标记 39,349 条货币代谢物记录。
- MTBLS13729 v2 feature graph：neg-RP 965 个节点、13,593 条边；pos-RP 5,291 个节点、78,468 条边。
- 全局 MS1 峰网络覆盖冻结重定量目标：neg-RP 3,798 个、pos-RP 13,155 个；正式版本由统一 EIC 构建且不读取表型或候选身份。
- MetDNA3 HILIC 开发任务：117 个 query、91 个身份、88 个分子式、466 个候选；官方 DreaMS Recall@1 为 `95/117=0.8120`，共有 22 个 Top-1 错误。
- 统一候选证据账本包含 12 类候选级证据，所有输入、拆分、候选表和输出均保存 SHA256。

### 3.2 BioAware v1：一跳 Rhea 失败，但确定了错误机制

在 MTBLS13729 的 21-query 冻结 pseudo-truth 小面板上：

- DreaMS：20/21；
- BioAware v1：19/21；
- corrected/introduced：`0/1`；
- 公式簇 bootstrap ΔCI：`[-0.1667, 0]`；
- gate 失败。

唯一新增错误来自不完整转氨反应路径：网络支持 2-aminobutanoate 覆盖了谱学更强的 GABA，但共同需要的 α-酮丁酸并未被要求观测。该结果证明全局一跳邻接不足以做身份覆盖，推动了 reaction-complete hyperedge 和弃权机制。

### 3.3 BioAware v2：实验 feature 图提高安全性，但覆盖不足

冻结四对照结果：

- archived-v1：`0/1`，精确复现旧失败；
- expanded-Rhea-only：`1/1`，净变化为 0；
- two-layer feature graph：`0/0`，只对 2/21 query 形成网络证据，最终完全弃权；
- 正式门未通过。

事后 hyperedge-completeness 规则在已看过的 21 例上得到 `1/0`（表面 `+4.76 pp`），但该规则由本批错误启发，而且会删除 MTBLS1905 的一个合理修正，因此只能作为机制拟合，不能作为性能结果。

### 3.4 MetDNA3 对齐开发：网络层逐项裁决

在 117-query HILIC consumed-development 任务上：

| 证据臂 | corrected / introduced | ΔRecall@1 | 裁决 |
|---|---:|---:|---|
| Rhea dependency-corrected 一跳融合 | 4 / 2 | `+0.244 pp` | CI 跨 0；未开锁 |
| known MRN + observed MS1 + full raw-MS2，depth-2 | 0 / 0 | 0 | 预注册主分析无效 |
| 同上，depth-3 消融 | 3 / 0 | `+2.56 pp` | 仅2个独立身份，公式簇CI下界0 |
| SMN（Dice≥0.4） | 4 / 8 | `−3.42 pp` | 网络过密，负对照 |
| identity-isolated RT | 0 / 3 | `−2.56 pp` | RT不能单独覆盖 |
| reported + predicted eMRN raw path | 5 / 6 | `−0.85 pp` | 扩边增加风险 |

递归 MS1 feature headroom 显示：step0 一跳没有稳定救援；两跳有 5 个 query/3 个身份，三跳有 6 个 query/4 个身份；预测 step1 没有跨阈值稳健独立增量。因此目前只保留已知 reaction pair、两跳为主、三跳作消融。

### 3.5 候选特异结构与规则证据

冻结 embedding 到 Morgan 指纹的公式隔离解码器：

- 单独使用时 Recall@1 仅 `0.487`，8 修正、46 新增，不能部署；
- 但在原先 11 个完全未解决错误中新增 2 个独立 headroom。

335 条核心规则形成的候选参考谱 likelihood：

- 各单臂会修正 5–6 个，同时新增 40–54 个，不能直接覆盖；
- 三种固定规则读法分别补充少量未解决候选；
- 规则信号正确用途是候选特异证据与冲突解释，不是身份标签。

旧 reaction/SMN/RT/predicted-edge 模块的实际修正并集为 8 个错误；加入 decoder 与规则头寸后，consumed-development 可触达并集达到 13，数学上超过“提升10pp所需净修正12个”的门槛。但这是看过真值后的 oracle/headroom，不是模型性能。

### 3.6 多证据融合：历史最好输出与审计后的失败边界

| 版本 | 设计 | 结果 | 裁决 |
|---|---|---|---|
| G3-v1 | 原始证据值直接非负融合 | 0/0 | 量纲不一致，安全但无效 |
| G3-v2（原实现） | 候选组内归一化；机制族投票；consensus gate | **2/0，表面 +1.71pp** | 审计发现零权重计票和相关网络票重复计数；不可继续作为有效最佳结果 |
| G3-v3 | 加入同分子式 leave-query-out peer context | 0/0 | peer residual 无稳定增量 |

G3-v2 仍支持“证据分解和弃权值得保留”，但不能证明当前融合器有效。修复零权重支持逻辑后只剩1个修正；将嵌套且高度相关的网络证据合并后无净提升。RP/外部测试继续封存，必须先完成证据依赖建模和新开发验证，当前不能声称 BioAware 优于 DreaMS，更不能声称 SOTA。

---

## 4. BioAware 当前科学认识

### 4.1 已经回答的问题

1. **代谢网络是否可直接提升谱库检索？** 不可以。一跳网络会被高度数、缺共同底物和方向不确定路径误导。
2. **增加更多预测边是否自动改善性能？** 不会。eMRN 扩边增加覆盖，但没有给现有错误增加稳健的一跳 truth seed，并可能同时支持错误候选。
3. **真实样本 feature 是否不可或缺？** 是。稳定救援只在实际观测 MS1 中间节点和递归路径出现。
4. **raw-MS2 是否能直接作为候选身份加分？** 不宜。它更适合作为 feature-edge 是否可信的约束；最难错误中可能反而支持错误路径。
5. **当前安全融合是否已经成立？** 尚未成立。原 G3-v2 的2修正/0新增受零权重计票与网络证据重复计数影响；严格审计后没有独立共识净增益。

### 4.2 当前真正瓶颈

瓶颈不是“图不够大”，而是：

- 真候选不在网络或无合格 Level-1 邻居；
- 样本内实际观测中间节点不足；
- 同一谱学/网络证据同时支持真候选和异构错误候选；
- 小型开发集只有22个错误，独立化学身份更少；
- 缺乏大型 Level-1、样本矩阵配套的外部锁定 benchmark；
- 证据覆盖增长快于候选特异性，导致 corrected 与 introduced 同时增加。

---

## 5. MTBLS13729 生物学课题成果

### 5.1 数据与设计

- 30 位患者，每位肿瘤与癌旁配对，共 60 个生物样本；4 个 LC-MS panel，共 240 个沉积 mzML。
- 原始 mzML 含完整 MS1，可进行峰面积定量；原管线此前仅处理 MS2。
- 无 pooled QC、blank 或参考样，因此主分析采用患者内配对、多归一化敏感性、注入顺序审计和表型盲候选冻结。
- Rmu 只有10对，定位为发现队列，不作为确认队列。
- 正离子 `P06-Ltu.mzML` 损坏，但不属于 Rmu/RN 主终点或 Rtu/RN 次要交互队列。

冻结注释面板包括 neg-RP 62 个、pos-RP 555 个目标。pos-RP 有 6 个丰度 feature 跨归一化达到探索性 FDR10，3 个达到 FDR5；表型盲 ion-family 去冗余后，6 个 feature 对应 5 个描述性中性家族。

### 5.2 BioAware 对生物学结果的真实贡献

反应网络没有直接覆盖八个核心候选，因此没有资格提高它们的身份等级。BioAware 的有效贡献来自表型盲全局峰图：

| ion family | 证据 | 生物学处理 |
|---|---|---|
| `1597/7489` | `[M+H]+/[M+Na]+`；质量残差约0.0010 Da；跨样本 Pearson 0.730；分半复现 | 合并为 methylguanosine-like isomer family；禁止写 Nelarabine |
| `3019/8481` | `[M+H]+/[M+Na]+`；质量残差0.000465 Da；跨样本 Pearson 0.571；分半复现 | 合并为 dimethylguanosine-like isomer family；8481不作独立发现 |

这一步避免把同一分析物的不同离子形式重复写成不同代谢物。去冗余改善的是计数与模块统计可信度，不是 MSI Level 1 身份。

### 5.3 主结果一：修饰鸟苷样模块

完全折叠两个离子家族后：

| 归一化 | Rmu/RN 平均 log2FC | 倍数 | 同向 | 精确 p | leave-one-patient 最低均值 |
|---|---:|---:|---:|---:|---:|
| raw | 2.953 | 7.74 | 10/10 | 0.001953 | 2.594 |
| global PQN ≥60% | 2.856 | 7.24 | 10/10 | 0.001953 | 2.500 |
| global PQN ≥80% | 2.852 | 7.22 | 10/10 | 0.001953 | 2.509 |
| global PQN ≥90% | 2.860 | 7.26 | 10/10 | 0.001953 | 2.529 |

技术匹配负对照从完整 feature 空间预冻结2,000个随机双家族模块，其中1,412个具有完整10对患者值。真实模块在四种归一化下均超过全部可比随机模块，经验单侧 `p=1/1413=0.000708`。但随机面板完整率为70.6%，低于预设75%门，因此综合 gate 仍为 false：这是强技术背景特异性证据，不是外部确认。

修饰鸟苷模块与 purine-like `4966` 的患者内效应高度相关：Pearson `r=0.877, p=0.00191`，Spearman `rho=0.933, p=0.000236`。它支持修饰核苷/RNA 周转或嘌呤回收异常的假设，但不能锁定位置异构体或具体 RNA 修饰酶。

### 5.4 主结果二：长链酰基肉碱类别与 C20:4 锚点

从当前统一 EIC 与原始 MS2 重新构建的表型盲类别分析得到：

| 终点 | 平均 log2FC | 精确 p | 结论 |
|---|---:|---:|---|
| Rmu/RN，PQN | 1.382 | 0.00586 | 约2.60倍升高 |
| Rtu/RN，PQN | 0.260 | 0.512 | 无一致改变 |
| Rmu–Rtu 次要交互 | 1.122 | 0.0395 | 探索性名义正向 |
| 链级去冗余 Rmu/RN | 1.368 | 0.00586 | 主信号保持 |
| 链级去冗余 Rmu–Rtu | 1.034 | 0.0659 | 仅保留亚型趋势 |

每个 Rmu 患者对至少覆盖16个、通常覆盖20个 LCAC 特征。C20:4 acylcarnitine-like `feature 3222` 的谱学证据为：

- 42 个样本有质量/RT匹配 MS2；
- 59 条匹配 MS2；
- 25 条谱、来自25个样本，通过预先固定的强肉碱诊断模式；
- 身份保持 MSI Level 2，尚不能确定双键位置、立体化学或标准品共洗脱。

该结果支持 Rmu 中长链脂酰基输入、线粒体转运与后续氧化之间存在稳态失衡，但不能判断是输入增强还是下游氧化瓶颈，也不能锁定 CPT1A、CPT2、SLC25A20 或具体 β-氧化酶。

### 5.5 第三条轴：多胺乙酰化

`feature 1717` 为与 N1,N8-diacetylspermidine 精确质量/分子式一致的候选，在 Rmu/RN 中稳健升高。2026-08-30 逆向审计确认当前 accepted experimental MS2 link 为0，因此旧版“Level 2”措辞撤回。它与修饰鸟苷模块的患者效应不相关（Pearson `r=0.095, p=0.823`），可保留为相对独立的多胺乙酰化探索轴，但命名身份必须等待标准品和实验 MS2。

### 5.6 三条代谢轴的统一解释

当前数据更支持三条相对独立、可能共同反映肿瘤组织重塑的轴：

1. **修饰鸟苷/嘌呤周转轴：** 效应最大、10/10同向、技术背景尾部显著；
2. **长链酰基肉碱/脂肪酸处理轴：** 类别级显著，C20:4具有多样本诊断 MS2；
3. **多胺乙酰化轴：** N1,N8-diacetylspermidine-like 独立升高。

现有 n=10 不支持把三者拟合为一个共同因果机制。更合理的论文叙事是“算法驱动的证据分层发现了三个可复核代谢模块”，而不是声称一个已证明的代谢通量网络。

### 5.7 外部复核边界

MTBLS8090 的35对 CRC 肿瘤/癌旁未复现“泛 CRC 长链酰基肉碱升高”，总体方向略相反。该队列没有黏液型标签和可复核原始 DDA MS2，因此不能检验 Rmu 特异性，但它明确否定了把本结果写成普适 CRC 规律。

---

## 6. 当前可以与不可以主张什么

### 6.1 可以主张

- BioAware 已形成 DreaMS 引导、候选特异、证据分解、风险受控且允许弃权的完整工程框架。
- 原 G3-v2 曾得到2修正/0新增的表面正向输出，但审计后该结果不能作为有效性能证据；其价值仅在于暴露了证据依赖与 gate 设计问题。
- 一跳网络、预测扩边、SMN和RT单独覆盖均已被严格证明不足，BioAware 的方法学创新正在解决真实失败模式。
- 表型盲 ion-family 图有效去除了重复加合物，直接提高非靶向发现的计数可信度。
- Rmu 相对匹配癌旁存在稳健的修饰鸟苷样模块和长链酰基肉碱类别积累；C20:4具有多样本类别诊断 MS2。
- N1,N8-diacetylspermidine-like 提供第三条相对独立的多胺代谢候选轴。

### 6.2 不可以主张

- BioAware 已经显著或外部验证地超过 DreaMS；
- BioAware 已经达到 SOTA；
- 13-query headroom 等于实际提升10pp；
- Rhea/KEGG 网络确认了 MTBLS13729 核心候选身份；
- methylguanosine/dimethylguanosine 的具体位置异构体已经确认；
- C20:4 已达到 MSI Level 1；
- 已证明 Rmu 相对 Rtu 特异；
- 已证明 RNA 修饰酶、脂肪酸氧化通量、CPT节点或多胺酶的因果改变。

---

## 7. 当前论文价值判断

### 7.1 算法论文价值

原 G3-v2 的 `+1.71 pp` 已因实现审计降级为历史开发输出，更不足以支持 SOTA 算法论文。真正有价值的方法学故事是：

1. 系统证明“更大的代谢网络”不等于更好的注释；
2. 定位覆盖、候选特异性、共同底物缺失和证据冲突四类失败；
3. 提出 DreaMS unary + observed feature + reaction hyperedge + raw-MS2 edge + ion-family + abstention 的证据因子化框架；
4. 在外部大型 Level-1 样本矩阵上证明显著净增益。

缺少第4项时，BioAware 应作为整篇平台论文的创新模块，而不是单独声称性能 SOTA。

### 7.2 生物学应用论文价值

当前结果已经足以形成高质量“算法驱动再注释与生物学发现”案例：

- 身份在表型之前冻结；
- ion-family 去冗余；
- 患者内配对与多归一化；
- 修饰鸟苷模块的大效应与技术匹配负对照；
- LCAC 类别与 C20:4 多样本 MS2；
- 阴性外部结果和亚型交互边界均如实保留。

但要形成一区级强主张，仍至少需要以下一项：

- 独立含 mucinous/MMR 标签的 CRC 组织队列复现模块方向；
- 少量标准品将关键修饰鸟苷和 C20:4 升级至 Level 1；
- 与大型真值任务上的算法显著增益结合，使生物学部分承担真实应用证明而不是单独机制确认。

---

## 8. 下一步优先级

### P0：立即完成现有生物学证据加固

1. 完成长链酰基肉碱 leave-one-feature / leave-one-patient 广度审计，确认类别效应不是少数峰或单个患者驱动；
2. 对 `1597/7489/3019/8481/1717/3222` 输出逐峰诊断碎片、共洗脱、加合物和候选异构体审计表；
3. 固化“5个中性家族而非6个独立代谢物”的论文口径；
4. 把作者 MAF、官方 DreaMS、E6、P2b 与 BioAware ion-family 证据逐候选对照，明确新增信息来自哪里。

### P1：BioAware 算法主攻

1. 不再扩大无约束网络；优先完成 candidate-specific fragmentation likelihood；
2. 在真实观测 MS1 节点上实现 reaction-complete hyperedge 与 ion-family 一致性；
3. 用可解释的 ILP/因子图作为首版全局分配器，再考虑 GNN；
4. 证据族统一为候选组内 calibrated votes，禁止原始量纲直接叠加；
5. 开发门保持 `corrected-2×introduced>0`、formula CI下界>0、每fold非负和随机网络不复现。

### P2：外部裁决

1. 建立更大的 Level-1、带样本矩阵外部 benchmark；
2. 所有模型、阈值、seed和候选协议冻结后一次性评估；
3. 只有外部显著增益后，才讨论 BioAware 超越 DreaMS 或 SOTA；
4. 生物学外部验证优先寻找带 mucinous histology/MMR 的 CRC 组织队列。

---

## 9. 核心复现工件

下列路径按当前本地工作区核验。标注“服务器待同步”的结果已在服务器日志与阶段文档中完成，但原始 JSON 尚未同步到本地；在同步并核对哈希前，不应把它们描述为“本地完整复现包”。

### BioAware 算法

- `data/validation/bioaware_metdna3_development_eval_v1/report.json`
- `data/validation/bioaware_metdna3_candidate_edge_decision_v1/report.json`
- `data/validation/bioaware_candidate_fragment_decoder_v1/report.json`
- `data/validation/bioaware_candidate_rule_likelihood_v1/report.json`
- `data/validation/bioaware_candidate_evidence_ledger_v1/report.json`
- `data/validation/bioaware_rank_consensus_fusion_v2/report.json`
- `data/validation/bioaware_global_consensus_fusion_v3/report.json`
- `data/validation/bioaware_10pp_headroom_v1/report.json`

### MTBLS13729 生物学

- `data/mtbls13729/biology_closure_analysis_v1/report.json`
- `data/mtbls13729/modified_guanosine_matched_background_v1/matched_background_report.json`
- `data/mtbls13729/biology_closure_20260829/report.json`（服务器待同步）
- `data/mtbls13729/frozen_ion_family_audit_20260829/report.json`（服务器待同步）
- `data/mtbls13729/c20_4_anchor_ms2_audit_20260829/report.json`（服务器待同步）
- `data/mtbls13729/acylcarnitine_panel_20260829/class_score_report.json`（服务器待同步）
- `docs/MTBLS13729_BIOAWARE_BIOLOGY_CLOSURE_20260829.md`
- `docs/MTBLS13729_FROZEN_BIOLOGY_RESULT_20260829.md`

---

## 10. 最终统一表述

> 本项目构建了一个由 DreaMS 谱图表示引导、将反应网络、真实 MS1 节点、raw-MS2 feature-edge、候选特异碎裂和离子家族关系分解建模、并允许弃权的 BioAware 原型。严格审计表明，当前线性多票融合存在零权重计票和相关网络证据重复计数，尚未建立相对 DreaMS 的有效性能增益；下一版必须改为校准证据与显式依赖的全局因子图。其离子家族和证据分层基础设施已在 MTBLS13729 中产生直接生物学价值：去除重复加合物后，识别出 Rmu 中稳健积累的修饰鸟苷样模块、长链酰基肉碱类别和独立多胺乙酰化轴。上述结果构成算法驱动非靶向代谢组学应用的发现闭环，但身份仍主要为 Level 2，亚型特异性、通量和具体酶机制仍需外部队列或标准品验证。
