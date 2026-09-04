# MTBLS13729 BioAware 生物学闭环：修饰鸟苷模块与三条代谢轴

## 结论先行

当前最扎实的新生物学结果不再是某一个数据库名称，而是一个经离子家族去冗余、定向 EIC 重定量和多归一化复核的**修饰鸟苷样代谢模块**。

- 在 10 对 Rmu 肿瘤/匹配癌旁中，完全按离子家族合并后的模块为 **10/10 同向升高**。
- 原始丰度的患者内平均变化为 `+2.953 log2`（约 `7.74` 倍），精确 sign-flip `p=0.001953`。
- 三档表型盲全局 PQN 后仍为 `+2.852` 至 `+2.860 log2`（约 `7.22–7.26` 倍），每档精确 `p=0.001953`；leave-one-patient-out 最小均值仍高于 `+2.50 log2`。
- Rtu 对匹配癌旁的模块变化约 `+0.83–0.91 log2`；Rmu 与 Rtu 的效应差约 `+2.01–2.05 log2`，探索性精确置换 `p=0.0197–0.0234`。

这是一项**发现级、组织静态丰度**结果。它支持 Rmu 中修饰鸟苷/嘌呤周转异常，但暂不证明具体位置异构体、RNA 修饰酶、代谢通量或黏液型特异性。

## 1. 冻结与分析合同

1. 候选身份和三路谱学证据在查看肿瘤/癌旁丰度前冻结。
2. 主终点是 10 对 Rmu 与匹配 RN 的患者内丰度差；Rmu 与 Rtu 的效应差为次要、探索性终点。
3. BioAware/Rhea/HMDB 只用于离子家族、候选覆盖和生化关系描述，不参与重新定义身份真值，也不参与筛选丰度阳性。
4. 定量使用原始 mzML 的定向 EIC：`5 ppm`、目标 RT 附近 `±20 s`、局部峰解析、最大 apex 偏移 `12 s`。
5. 只计算肿瘤与匹配癌旁均检出的患者对，不对缺失值做表型相关填补。
6. 全局归一化因子由完整 `126,082` 个共识 MS1 feature 中高检出背景特征计算，分别冻结为检出率 `>=60%`、`>=80%`、`>=90%` 三档；不使用候选身份或表型效应选择背景。
7. 当前闭环属于对已冻结八候选的定向巩固，`formal=false`：它不能替代原完整 feature 空间的多重检验，也不能把事后模块整合写成新的全发现 FDR。

## 2. BioAware 在本结果中的真实贡献

BioAware v1 的 Rhea 一跳传播在 MTBLS13729 上没有带来准确率增益：21 个可评估 query 中 `0` 修正、`1` 新增。更重要的是，当前八个核心候选中没有一个可被 Rhea 直接节点覆盖。因此，**反应网络没有资格为这些候选提升身份等级**。

BioAware 的有效贡献来自表型盲的全局峰图：

| 离子家族 | 证据 | 处理 |
|---|---|---|
| `1597/7489` | `[M+H]+/[M+Na]+` 质量残差 `0.001003 Da`；跨样本丰度 Pearson `r=0.730`；奇偶样本分半均复现；本地 apex 中位差 `0 s` | 合并为一个 methylguanosine isomer family，禁止把 Nelarabine 名称作为内源身份 |
| `3019/8481` | `[M+H]+/[M+Na]+` 质量残差 `0.000465 Da`；全局峰图跨样本 Pearson `r=0.571`，分半为 `0.658/0.447`；本地 apex 中位差 `0 s` | `8481` 仅作为 dimethylguanosine family 的表型盲加合物支持，不作为独立发现 |

这一步解决的是非靶向代谢组学中常见的“多个 feature 被误写成多个代谢物”的问题。它提高了生物学计数和模块统计的可信度，但不等价于标准品确认。

## 3. 身份与定量联合证据

| feature/家族 | 当前最严谨身份 | 身份边界 | Rmu 主要证据 |
|---|---|---|---|
| `4966` | `C7H9N5O` purine-like/deazaguanine-like isomer family | HMDB 名称与谱库结构不唯一；保持异构体层 | PQN80 `+2.339 log2` |
| `3019/8481` | dimethylguanosine isomer family | 位置异构体未定；`8481` 为 Na 加合物支持 | 合并进入修饰鸟苷模块 |
| `1597/7489` | methylguanosine isomer family | 位置异构体未定；两个加合物只计一个中性家族 | PQN80 单家族约 `+3.6 log2`；两个离子患者效应相关 `r=0.915, p=0.00055` |
| `1717` | N1,N8-diacetylspermidine-like | 仅精确质量/分子式一致；accepted实验MS2=0，无标准品，不是Level 2 | PQN80 `+2.866 log2` |
| `3222` | C20:4 acylcarnitine-like | Level 2；双键位置/立体化学未定 | PQN80 `+1.675 log2`；59/59 检出，稳定单分子锚点 |
| `3180` | chlorinated/exogenous-like | 内源合理性差，作为误注释/污染控制 | 不进入机制主线 |
| `16425` | LPE-like | 低检出、身份和局部峰稳定性较弱 | 仅保留次级候选 |

峰质量审计支持把 `3019/1597/7489/1717/3222/4966` 置于前列：这些 feature 的检出率为 `86.4–100%`，中位 apex 偏移约 `0.90–3.55 s`。`3180/16425` 检出率仅 `79.7%`，且 apex 离散更大，应降级。

## 4. 修饰鸟苷模块的稳健性

完全合并两个离子家族后：

| 归一化 | Rmu n | 平均 log2FC | 倍数 | 同向比例 | 精确 sign-flip p | LOO 最小均值 |
|---|---:|---:|---:|---:|---:|---:|
| raw | 10 | 2.953 | 7.74 | 10/10 | 0.001953 | 2.594 |
| global PQN, prevalence >=60% | 10 | 2.856 | 7.24 | 10/10 | 0.001953 | 2.500 |
| global PQN, prevalence >=80% | 10 | 2.852 | 7.22 | 10/10 | 0.001953 | 2.509 |
| global PQN, prevalence >=90% | 10 | 2.860 | 7.26 | 10/10 | 0.001953 | 2.529 |

因此，这个结果不是由某一个患者、某一个加合物或全局信号尺度造成。它也不是 BioAware 网络传播“制造”的：网络只帮助完成离子家族合并，丰度效应来自原始 mzML 的独立 EIC。

### 4.1 严格技术匹配随机模块

为检验模块是否只是低 m/z、早 RT 或特定检出率区域的普遍升高，从完整正离子 feature 空间
预冻结 2,000 个随机双家族模块。四个技术位点分别与 `1597/7489/3019/8481` 在 m/z
`±25 Da`、RT `±45 s`、全局检出率 `±0.15` 内匹配；匹配不读取肿瘤/癌旁标签。95 个唯一
背景 feature 随后用与真实模块完全相同的 `5 ppm`、局部峰、`12 s` apex 规则从 59 个原始
mzML 重定量，并按两组加合物家族进行同样折叠。

- 1,412/2,000 个随机模块具有全部 10 对患者的完整值；
- 四种归一化下，真实模块平均效应均超过全部 1,412 个可比随机模块，经验单侧
  `p=1/1413=0.000708`；
- 随机模块均值的 99% 分位为 `1.86–1.97 log2`，真实模块为 `2.85–2.95 log2`；
- 单看“10/10 同向”并不罕见到 0.05 以下（随机概率约 `0.051–0.068`），真正有区分力的是
  **同向性与大效应量同时出现**，联合经验 `p=0.000708`。

预设完整性门要求至少 75% 的随机面板具有 10 对完整值，实际为 70.6%，因此综合 gate 保持
`false`，不作事后改门。这个失败不推翻条件于 1,412 个完全可比面板的强尾部证据，但要求将
结果表述为“技术匹配背景中的强模块特异性”，而非无条件外部确认。

## 5. 不是一个强行拼接的单通路，而是三条相对独立的代谢轴

### 5.1 轴一：修饰鸟苷/嘌呤周转

- 修饰鸟苷模块与 purine-like `4966` 的 Rmu 患者内效应高度相关：Pearson `r=0.877, p=0.00191`，Spearman `rho=0.933, p=0.000236`。
- methylguanosine 与 dimethylguanosine 两家族效应相关：Pearson `r=0.833, p=0.00531`。
- 这支持“同一患者中修饰鸟苷与相关嘌呤代谢物协同积累”，可提出 RNA 周转、修饰核苷降解/输出或嘌呤回收异常的机制假设。
- 但现有静态组织数据不能区分 RNA 降解增加、核苷输出受阻、细胞组成改变或肾/循环清除因素，也不能锁定具体 RNA 修饰酶。

既往 CRC 靶向血清核苷研究已经观察到多种甲基鸟苷异构体发生改变，证明这类分子具有 CRC 生物标志物潜力；但血清方向依赖异构体，不能直接替代本研究的组织和亚型结论（[2023 targeted serum nucleosides](https://pmc.ncbi.nlm.nih.gov/articles/PMC10334214/)；[2026 targeted serum nucleosides](https://pubmed.ncbi.nlm.nih.gov/41925117/)）。细胞研究也表明 RNA 来源修饰核苷可以积累并输出至细胞外，为“RNA turnover/export”提供机制可行性，但不是本队列的因果证明（[RNA-derived modified nucleoside export](https://doi.org/10.1080/15476286.2021.1960689)）。

### 5.2 轴二：多胺乙酰化

`1717` 的 m/z 230.185931 多胺样信号稳健升高，但与修饰鸟苷模块的患者效应不相关（Pearson `r=0.095, p=0.823`）。因此它更适合作为独立的多胺代谢候选轴，而不是硬并入核苷模块。逆向审计确认当前 accepted experimental MS2 link 为0，所以 N1,N8-diacetylspermidine 只能是精确质量/分子式候选。CRC 文献已有乙酰化多胺升高先验（[urinary DiAcSpd/DiAcSpm CRC study](https://pubmed.ncbi.nlm.nih.gov/20655890/)；[primary colorectal tumor N1-acetylspermidine study](https://pubmed.ncbi.nlm.nih.gov/6692383/)），但独立 MTBLS8090 的三项可用相关多胺均未显著复现，文献和通路一致性都不能替代本地标准品与实验 MS2。

### 5.3 轴三：长链酰基肉碱/线粒体脂肪酸处理

`3222` 是当前最稳定的单分子锚点，C20:4 acylcarnitine-like 在三种归一化下升高约 `2.34–3.39` 倍；此前类别分析还显示多个长链酰基肉碱协同积累。它与修饰鸟苷模块不相关（Pearson `r=0.094, p=0.809`），应作为独立的脂肪酸处理轴。CRC 中酰基肉碱改变已有靶向研究支持，但静态丰度不能推出 beta-氧化通量或具体 CPT 酶改变（[targeted CRC acylcarnitines](https://pmc.ncbi.nlm.nih.gov/articles/PMC11816151/)）。

## 6. 临床敏感性：MMR 不能解释全部信号，但也尚未排除混杂

- 完整模块值可用于 5 个 dMMR Rmu 和 4 个 pMMR Rmu；两组均为所有患者正向。
- dMMR Rmu 平均 `+3.959 log2`，pMMR Rmu 平均 `+1.848 log2`。
- dMMR 与 pMMR 的效应差为 `+2.111 log2`，精确置换 `p=0.103`。
- pMMR Rmu 与 pMMR Rtu 的效应差为 `+0.911 log2`，`p=0.415`。

因此，修饰鸟苷模块并非仅由 dMMR 个体产生，但样本量不足以证明其独立于 MMR，也不足以确认黏液型特异性。临床 S2 原表只用于此敏感性分层，没有用于候选或模块选择。

## 7. 论文主轴与证据等级

建议的生物学主标题：

> **表型盲谱学再注释与离子家族整合揭示黏液型结直肠癌组织中的修饰鸟苷样代谢模块积累，并伴随独立的多胺乙酰化和长链酰基肉碱异常。**

可写：

- 算法/谱学框架比原始 MAF 提供更完整的结构候选、离子家族和证据分层；
- 修饰鸟苷模块在 10 对 Rmu/RN 中 10/10 同向，且对全局 PQN、留一患者和加合物合并稳健；
- Rmu 相对 Rtu 的效应更强是探索性证据；
- C20:4 acylcarnitine-like 与 N1,N8-diacetylspermidine-like 是相对独立的正交代谢轴。

不可写：

- “已确认 methylguanosine/dimethylguanosine 的具体位置异构体”；
- “BioAware 反应网络提高了这些身份的准确率”；
- “已证明 Rmu 特异、RNA 修饰酶异常、代谢通量变化或具体酶机制”；
- “八个 feature 等于八个新代谢物”；
- “局部模块 p 值等于完整非靶向发现的 FDR”。

## 8. 最小成本的下一步闭环

1. **优先身份复核**：人工检查 `1597/7489/3019/8481` 的 MS2 诊断碎片、加合物和共洗脱；如仅能购买少量标准品，修饰鸟苷位置异构体优先级高于低可信 `3180/16425`。
2. **模块特异性负对照**：从完整 feature 空间预冻结 m/z、RT、检出率匹配的随机二家族模块，比较 10/10 同向和平均效应的经验尾概率；这是当前最需要补的计算验证。
3. **外部组织复核**：优先找含 mucinous histology 或 MMR 注释的 CRC 组织代谢组；无亚型标签的数据只能检验“泛 CRC 是否出现同类修饰核苷变化”，不能确认亚型。
4. **转录组旁证**：若同病例或可比队列有 RNA-seq，预注册 RNA turnover、purine salvage、modified-nucleoside transport 和 polyamine acetylation 基因集；只做模块级关联，不倒推代谢物身份。
5. **保留阴性结果**：Rhea direct coverage=0 和 BioAware v1 的 0修正/1新增应写入方法边界，说明网络证据采用弃权设计，而不是为了故事强行传播。

## 9. 复现工件

- 主结果：`data/mtbls13729/biology_closure_analysis_v1/report.json`
- 联合证据表：`data/mtbls13729/biology_closure_analysis_v1/candidate_identity_and_abundance.csv`
- 完全离子家族合并患者效应：`data/mtbls13729/biology_closure_analysis_v1/fully_ion_family_collapsed_module_patient_effects.csv`
- 峰质量审计：`data/mtbls13729/biology_closure_analysis_v1/peak_quality_audit.csv`
- 临床 S2 敏感性：`data/mtbls13729/biology_closure_analysis_v1/module_patient_effects_clinical.csv`
- 主图：`data/mtbls13729/biology_closure_analysis_v1/modified_guanosine_module_summary.png`
- 候选效应热图：`data/mtbls13729/biology_closure_analysis_v1/biology_candidate_effect_heatmap.png`
- 技术匹配随机模块报告：`data/mtbls13729/modified_guanosine_matched_background_v1/matched_background_report.json`
- 技术匹配随机模块图：`data/mtbls13729/modified_guanosine_matched_background_v1/modified_guanosine_matched_background.png`
