# Contrast Atlas v10：WSE 全局响应检索与 MSV000080918 生物学结果

## 结论先行

本轮完成了真实的系统级计算，但结果**不支持把当前 WSE 响应检索写成一个有效的 Atlas 发现引擎**。

- 16 个来源 ID；并不等于 16 项彼此独立的研究；
- 53 个研究级、有方向、有 `beta/SE` 的 contrasts；
- 55,127 个研究内实体；
- 294,236 条模型无关、跨来源、同观测前体 10-ppm 候选对；
- 122,795 个 query×target-source block 具有正 WSE 和至少一个匹配碎片；
- 35,260 个不重复的 reciprocal 实体对；跨 all/known/unresolved 三个 world 展开后为 68,586 行；
- 2,490 个 source-disjoint contrast 查询—候选方向。

在该完整分母上，all-world 对 53 个查询中的 **51 个**给出“相似响应”，对 **49 个**同时给出“相反响应”。34 个 eligible-primary 登记方向只有 `3/34` 在最佳响应面 Top-1、`20/34` 在 Top-5。如此高的回答率和很低的登记关系 Top-1 命中表明：当前 response operator 与置换/max-T null 缺乏实际生物响应特异性。

因此：

1. `51/53` 和 `49/53` 不是 51 或 49 条正确生物学联系；
2. permutation/max-T 显著只表示结果偏离当前 null，不表示候选联系正确；
3. 这套 WSE 结果可继续用于候选生成与失败机制研究，不能用于发现或机制声明；
4. 本轮没有证明 unresolved 信息增量，也没有检验 Full23 增量。

机器可读结果见：

- `data/validation/contrast_atlas_v10_wse_scientific_result_v1_20261031/report.json`
- `data/validation/contrast_atlas_v10_wse_scientific_result_v1_20261031/new_source_open_top_hits.tsv`
- `data/validation/contrast_atlas_v10_wse_scientific_result_v1_20261031/iga_ko_to_ad_skin_candidate_evidence.tsv`

## 1. 本轮真正完成了什么

### 1.1 模型无关候选轴

候选纳入只使用来源、观测前体质量、显式 charge/polarity/adduct 冲突；没有读取 WSE、响应方向、疾病名称或登记关系。零候选、单候选和多候选 block 全部保留。

新增来源 `MSV000080918` 后：

| 项目 | 分母 |
|---|---:|
| 来源 ID | 16 |
| contrasts | 53 |
| 实体 | 55,127 |
| query×target-source blocks | 826,905 |
| 非空 blocks | 141,332 |
| 有正谱学证据的 WSE answers | 122,795 |
| directed candidate pairs | 294,236 |
| world-specific reciprocal rows | 68,586 |
| 不重复 reciprocal entity pairs | 35,260 |

旧 15 来源轴上的 291,330 个 pair ID 全部保留；新增 2,906 个方向对严格由 1,453 个新→旧与 1,453 个旧→新组成。

### 1.2 新来源并非“再加一个 accession”

`MSV000080918/MSV000080558` 来自同一个 Shalapour 研究家族。经过样本日期、年龄、笼位与谱图资格审查后，本轮只保留：

- `IgA-KO − IgA-intact`：primary development positive control；15 vs 11 个 cage，日期/年龄调整，cage-clustered CR1；
- `CD8-KO − intact`：sensitivity development contrast；只有 4 个 intact cages，且存在共享 cage 与基因型—住房问题。

它们都是 `development/consumed`，不是外部验证。尤其 genotype 与长期住房/微生物生态不能完全分离；clustered SE 不能消除这种混杂。

## 2. 为什么当前全局检索失败

### 2.1 回答率不等于正确率

| World | 相似响应 | 相反响应 |
|---|---:|---:|
| all | 51/53 | 49/53 |
| known | 16/53 | 17/53 |
| unresolved | 51/53 | 46/53 |

同一个查询可以分别在相似面与相反面选择不同候选，因此这不是旧版字典序二选一错误。但当几乎所有异质查询在两面都能找到 max-T 显著候选时，显著性不再提供足够的生物选择性。

可能原因包括广泛的来源级变化、稠密连接、跨 contrast 共享的非特异响应，以及当前 within-study profile reassignment null 没有充分模拟这些结构。现有结果只能确定“当前算子/null 不够特异”，尚不能在这些原因中作唯一归因。

### 2.2 登记关系也没有形成可靠开放检索

34 个 eligible-primary 登记方向没有 signed entity-level truth，且多个方向共享母项目，因此它们不是 34 个独立真值。不过，即使按最宽松的描述性用途，all-world 的最佳响应面也只有 `3/34` Top-1、`20/34` Top-5。这与 `51/53` 的高调用率合在一起，排除了“只要增加规模，Atlas 就自然有效”的解释。

## 3. MSV000080918 提供了什么新的科学诊断

### 3.1 同一化学组成、不同生物响应的压力测试

IgA 与 CD8 contrasts 共有 185 个响应实体：

| 指标 | 结果 |
|---|---:|
| beta Pearson | 0.4125 |
| beta Spearman | 0.3173 |
| 方向一致率 | 0.6000 |
| IgA `q≤0.05` 且 date-LOSO 稳定 | 25 |
| CD8 `q≤0.05` 且 date-LOSO 稳定 | 3 |
| 稳定集合交集/并集 | 2/26 |

两条响应共享谱图组成与采集背景，却不相同。这正适合检查检索器是否使用 contrast-specific response，而不是只识别来源化学组成。它们不是预先成立的生物学正负对照，因此“检索到不同目标”本身仍不能证明正确。

实际 all-world Top-1 更暴露了问题：

- CD8 相似面 → 人尿液 T1D normal-weight；
- IgA 相似面 → 微生物 lanthanophore overexpression；
- IgA 相反面 → 甜瓜根部 *B. subtilis* 处理；
- CD8 相反面 → montelukast 处理。

这些都是未裁决假说，不能依赖 `p≤0.05` 直接作生物学解释。

## 4. 一个值得保留、但不能夸大的候选

### 4.1 IgA-KO feces 与 AD lesional skin

在规范重跑中，`IgA-KO − intact` 对 `MSV000090788 lesion − nonlesion` 是 all-world 相似面的第 **2/51** 名，query max-T `p=0.007782`。该候选含 27 条 reciprocal 边；其中 3 条满足两端 normal-approximation `q≤0.05`，但该 q 仅是检索诊断，不能替代原研究的精确推断。

贡献最高的两条边为：

| 新来源实体 | AD 实体 | WSE | 新来源 beta±SE | AD beta±SE | source-native 标签 |
|---|---|---:|---:|---:|---|
| 166.0863 | 166.0863 | 0.8148 | +1.411±0.183 | +1.150±0.263 | L-/DL-phenylalanine |
| 205.0978 | 205.0973 | 0.7487 | +1.341±0.227 | +0.823±0.160 | tryptophan |

效应的正确含义是：这些离子分别在 IgA 缺失小鼠粪便相对 intact 更高，并在同一 AD 患者的病灶皮肤相对非病灶皮肤更高。不同组织、物种和模型的 beta 不能直接比较大小。

### 4.2 身份仍未确认

这两条不是标准品确认：

- m/z 166.086 两端 source-native 库名支持 phenylalanine，但共同严格库策略在 AD 端选择了异构体 `3-amino-3-phenylpropanoic acid`；应写作 phenylalanine/beta-phenylalanine-like；
- m/z 205.097 两端 source-native 库名支持 tryptophan，但共同策略返回另一个 indole-containing candidate；应写作 tryptophan/isomer-like；
- m/z 188.071 两端都带 tryptophan 类注释，且 AD 端为 `[M+H−NH3]+`，很可能是同一 ion family，不能当第三个独立代谢物。

因此，这一结果是**芳香族氨基酸样离子的跨背景同向候选**，不是“同一两个代谢物已复现”。

### 4.3 原始论文没有验证我们的机制

Shalapour 等原研究证明的是肝 IgA+ 细胞、PD-L1/IL-10、CTL 与 NASH-HCC 的免疫机制，并显示 immune status 与整体粪便代谢组相关；它没有报告 phenylalanine/tryptophan，也没有证明这些粪便离子介导 CTL 或肿瘤表型。[Nature/PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC5884449/)

AD 原研究比较的是 30 名轻中度成人 AD 患者的配对病灶与非病灶皮肤。正文报告病灶中 aspartyl-phenylalanine 等二肽更高；这只能提供有限的“氨基酸相关皮肤化学改变”合理性，不能验证游离 phenylalanine/tryptophan 身份或方向。[mSystems 原文](https://journals.asm.org/doi/full/10.1128/msystems.01119-23)

一项独立 AD 血浆研究报告 AD 相对健康者 phenylalanine/tryptophan 降低；由于样本与对比定义不同，它不能验证、也不能直接推翻病灶−非病灶皮肤的正方向。[独立血浆研究](https://pmc.ncbi.nlm.nih.gov/articles/PMC10978712/)

允许的表述仅为：

> 在 development/consumed 重分析中，两条与芳香族氨基酸或其异构体一致的同前体质量谱学对应，在 IgA 缺失小鼠粪便和配对 AD 病灶皮肤中均呈正向效应；精确身份、跨组织联系及机制尚未验证。

禁止写成 IgA–AD 共同机制、gut–skin axis、精确代谢物复现、unknown 增量或 Full23 增量。

## 5. 这轮对 Atlas 方法的实质推进

这轮不是“Atlas 已成功”，而是得到三个可证伪结论：

1. **规模已不再是主要借口。** 53 个真实 contrasts 与完整候选轴仍产生近乎全调用，说明当前核心瓶颈是 response specificity，不是数据条目太少。
2. **谱学对应与研究级推断是两个不同门。** WSE 可以提供可追溯候选，但不能把一个宽松的 response null 自动变成正确的跨研究生物联系。
3. **身份冲突本身是重要应用结果。** source-native 标签的一致并未消除共同策略中的同分异构体竞争；若不保留这一不确定性，Atlas 会把化学家族线索错误写成同一分子复现。

## 6. 下一项决定性科学实验

下一步不再扩研究来源，也不再新增一般性审计。使用现有 53 contrasts，直接检验：

> 在同一来源内有多个响应轴时，去除跨 contrast 共享的来源级响应后，剩余的 contrast-specific 化学响应能否区分 IgA/CD8、药物、疾病亚轴，并在固定候选宇宙中降低近乎全调用，同时保留有预先依据的登记关系？

成功标准不是 p 值数量，而是同时满足：

- heterogeneous query 的调用率显著下降；
- 同源不同响应轴得到可区分结果；
- eligible-primary 登记关系的 Top-1/Top-5 不因普遍弃权而虚增；
- known-only、known+unresolved 在机会匹配后分别报告；
- WSE 与 Full23 使用完全相同的候选、response operator 和分母；
- 至少一条新增联系通过未参与检索构造的独立证据确认。

在这些门通过前，当前 Atlas 只能称为大型、可追溯的研究级响应候选系统，不能称为已经建立的生物发现方法。
