# Metabolomic Contrast Atlas：系统公共响应集合与开放检索结果（2026-10-22）

## 固定科学目标

本工作不再把某个疾病、两研究配对或未知 feature matching 重新定义为整个课题。
固定目标仍是：把公共非靶向 LC-MS/MS 中**有方向、有幅度、有不确定性**的研究级响应，组织成可以开放检索的生物学知识单元；身份层回答同一实体是否复现，关系层回答不同但有化学关系的实体是否支持可验证的响应关联。关系层不得把不同实体的 beta/SE 合并成同一分子。

所有本页数据均为 `development/consumed`。`unresolved_entity` 仅表示未获严格直接谱库支持，不表示新的、唯一的未知代谢物。

## 从系统普查到真实面板

公开 GNPS 文献映射普查共有 629 个任务；132 个为完成的 FBMN；95 个同时具有可读标准化定量表和代表 MS/MS；82 个为单 accession。低基数元数据仅作为候选设计，随后逐项核对生物单位、重复测量、批次和论文语义。

首轮纳入三项异质而真实的研究：

| 来源 | 合法研究对象 | 进入对比的生物单位 | 响应对象 | 资格边界 |
|---|---:|---:|---:|---|
| MSV000090804 | 镧载体簇 KO、过表达、镧限制 | 19 | 3 | 80 个 LC-MS 文件先按样本名合并为生物单位；media 不进入对比 |
| MSV000086360 | B. subtilis 3610 处理 × 根/茎/叶 | 96 | 3 | 每个组织内平衡比较：14/14、15/15、19/19 |
| MSV000084794 | 人群工业化梯度 | 90 | 1 | 描述性 context contrast；人群、地域、储存等不能解释为单一因果干预 |

MSV000090804 的原论文表明 methylolanthanin 产生对正常镧积累是必需的，过表达提高生物积累和吸附；MSV000086360 的原论文报告 B. subtilis 处理影响种子和成株的脂质、谷胱甘肽相关代谢及生长/抗真菌表型；MSV000084794 是六个人群、90 名参与者的工业化梯度研究。因此三者分别承担受控遗传/营养、受控宿主-微生物干预和非因果人群 context 三种角色，不能互相冒充。

两项大数据候选被明确排除于响应层：

- MSV000087364 有 78 个痰样本，但论文设计是 24 人 ETI 治疗前后；公开 FBMN 元数据缺少冻结 participant/pair ID，直接当 78 个独立样本会产生伪重复。
- MSV000083475 有 665 个非控制 Earth Microbiome 样本，但 host-associated/free-living 几乎被 LC-MS 批次和提取协议分离；不能用样本量掩盖不可识别设计。

相关主文献：

- [methylolanthanin 与镧代谢](https://pmc.ncbi.nlm.nih.gov/articles/PMC11317620/)
- [B. subtilis–melon 生长促进与代谢响应](https://www.nature.com/articles/s41564-022-01134-8)
- [90 人工业化梯度粪便代谢组](https://pmc.ncbi.nlm.nih.gov/articles/PMC9765122/)

## 真实响应对象结果

最终冻结目录：`data/validation/public_fbmn_systematic_response_panel_v6_20261022`。

构建合同：

1. 先冻结 task、文件名连接、真正生物单位、组别和预期分母；
2. 正值取 log2，样本内中位数中心化；零/非正值保持缺失，不当作丰度零且不插补；
3. 同一生物单位的多文件先取中位数，再估计效应；
4. beta 为组均值的线性对比，SE 为独立组 Welch 标准误；
5. 空 MS/MS block 弃权；
6. identity partition 在不读取 beta/SE 的情况下，由 GNPS 直接 Gold/Silver/Bronze、合法 planar InChIKey 且非 suspect-related 命中定义。

结果分母：

- 3 个独立来源；
- 205 个进入合法对比的唯一生物单位；
- 7 个 response objects；
- 20,537 条效应合格记录，其中 20,535 条同时具有非空 MS/MS；
- 严格 library-supported 全实体仅为 30/4,612、89/4,316、1,645/36,234；
- 各 response object 的合格 unresolved 记录分别为 4,218、3,439、4,113、1,414、2,229、2,671、2,114。

这一步实现了“结构未知信息可以合法进入响应对象”，但尚未证明它改善跨研究检索。

## 并入开放公共响应集合

新来源以 append-only contract 并入已有 7-source 集合，冻结目录为：
`data/validation/contrast_atlas_open_response_collection_v5_fbmn_20261022`。

扩展后：

- 10 个独立 study/source family；
- 25 个研究级 contrasts；
- 651 个生物单位；
- 99,265 张非空代表 MS/MS；
- 87,151 条合格响应记录；
- 1,941 条 library-supported 与 85,210 条 unresolved 响应记录；
- 每个查询面对 21–24 个 source-disjoint 候选。

这不再是二选一或同一项目内的小候选池，但仍是开发集合，不是独立封存外测。

## 冻结关系层算子的开放检索结果

没有根据新增研究调整参数。直接复用此前冻结的 centered spectral-response kernel：

- 响应权重：`beta / sqrt(beta^2 + SE^2)`；
- 每个 contrast/world 内先中心化；
- 以 0.1-Da fragment measure 聚合后归一化；
- all、known、unresolved 三世界；
- 完整候选得分表先写出，再打开既有 6 条登记设计关系；
- 31 次 identity-stratified response reassignment 仅作敏感性分析，不解释为随机试验 p 值。

结果目录：`data/validation/contrast_atlas_centered_response_kernel_v5_fbmn_20261022`。

| 世界 | 可评分登记关系 | signed-similar Top-1 | alignment-strength Top-1 | composition Top-1 |
|---|---:|---:|---:|---:|
| all | 6/6 | 1/6 | 1/6 | 0/6 |
| known | 2/6 | 2/2 | 2/2 | 2/2 |
| unresolved | 6/6 | 0/6 | 0/6 | 0/6 |

扩展候选宇宙后，成人 IBD → 儿童 VEO-IBD 在 all/known 的 alignment rank 为 1/1，反向为 6/1；unresolved 为 4 和 8。HFD 登记关系的 unresolved alignment rank 为 6–14。31 次 response reassignment 中，IBD unresolved 达到同样好或更好排名的比例为 0.161/0.226，HFD 为 0.323–0.548。

因此必须给出负裁决：

> 当前全局中心化 fragment-response kernel 没有证明 unresolved 化学信息在扩大后的开放候选集合中产生增量。known-only 的 IBD 正控仍强，但 composition 同样 Top-1，不能把它包装成新的未知化学发现。

这项负结果不会推翻 Contrast Atlas 总目标；它否定的是“把整个响应压成一个全局平均 fragment vector 就足够”的具体算子。

## Signed fragment/neutral-loss programs：表面改善被零模型完全解释

在不改变 10-source/25-contrast 候选宇宙的前提下，又完成了一次预先固定的关系层实验：

1. 响应仍取 `beta / sqrt(beta^2 + SE^2)` 并在 contrast/world 内中心化；
2. 正权和负权分别形成 up/down response programs，不再先压成一个正负相消的向量；
3. 同时比较 fragment-only 与等范数 fragment + neutral-loss 双通道；
4. 先写出全部 3,276 行固定候选分数，再打开 6 条既有登记关系；
5. 用 31 次 identity-stratified response reassignment 检查排名究竟来自响应归属还是研究化学组成。

冻结目录：`data/validation/contrast_atlas_signed_fragment_loss_programs_v2_20261022`。

| 表示 / 世界 | 可评分登记关系 | alignment Top-1 | composition Top-1 |
|---|---:|---:|---:|
| fragment / all | 6/6 | 0/6 | 0/6 |
| fragment / known | 2/6 | 1/2 | 2/2 |
| fragment / unresolved | 6/6 | 1/6 | 0/6 |
| fragment+loss / all | 6/6 | 3/6 | 0/6 |
| fragment+loss / known | 2/6 | 1/2 | 2/2 |
| fragment+loss / unresolved | 6/6 | 3/6 | 0/6 |

仅看观察排名，neutral-loss 似乎将 unresolved 的 Top-1 从 1/6 提高到 3/6。但这是误导性的：在 fragment+loss/unresolved 主比较中，31/31 次响应重排都同样得到恰好 3/6 Top-1；零模型的最小值、中位数、最大值均为 3，`plus-one fraction(null >= observed)=1.0`。成人 IBD→儿童 IBD、女性 HFD→mouse HFD、男性 HFD→mouse HFD 在响应被打乱后仍始终为第一，说明排名由来源化学组成/覆盖主导，而不是由真实响应归属产生。

因此该算子正式判为 `FAIL_UNKNOWN_RESPONSE_SPECIFICITY`。这同时否定两种全局压缩：单 signed centroid 和分离 up/down 后的全局 fragment/loss programs。不得再在这 6 条已消耗关系上调整 bin、权重、阈值或通道比例。

下一阶段不再继续榨取这 6 个答案。最高性价比工作转为扩大**独立、预先成立、带明确方向语义的跨来源响应真值**：优先从 95 个具备定量表和代表 MS/MS 的 FBMN 任务中，寻找相同干预/宿主状态在独立项目中的复现，以及明确的 no-match 查询。新关系必须在看检索分数前登记；不同疾病名不自动作为阴性。只有扩大的真值集合冻结后，才能检验局部 relation edges 或 Noise/统一算法替代谱学接口，避免继续用 6 条已消耗关系选方法。

## 第三个独立 IBD 来源：先扩真值，再检验冻结算子

系统筛选 MassIVE 公开资产后，`MSV000096564` 提供了一个低成本而设计可恢复的新来源：公开 MZmine 定量表 25,067,892 bytes、代表 MGF 3,305,469 bytes 和参与者元数据 121,094 bytes。试验登记 `NCT05973500` 与公开研究资料表明，研究为 14 天地中海饮食干预；因此本次只取干预前 T1，完全排除后续时间点。

冻结规则：

1. T1 有 22 名独立参与者，11 IBD（活动期或缓解期 Crohn's）和 11 control；
2. 每人恰有 3 个粪便 LC-MS 技术文件，先在参与者内取中位数，不把 66 个文件当独立样本；
3. 正值 log2 后按技术文件中位数中心化，非正值保持缺失；
4. beta 为参与者级 IBD-minus-control，SE 为独立组 Welch 标准误；
5. baseline 疾病对比是观察性 condition contrast，不解释为饮食干预效应；
6. deposited MZmine matched-lipid 仅有 16 条，全部保留为 provisional；因没有完整严格 direct-library search，不能将任何实体转为 known。

冻结结果：`data/validation/msv000096564_baseline_ibd_response_v1_20261022`。

- 7,561 个定量 features；
- 7,267 个非空代表 MS/MS blocks；
- 2,825 个 response-QC 合格 features；
- 2,817 个同时有非空 MS/MS；
- strict library-supported 0/7,561，故本来源在主分区中全部为 unresolved，而不是声称 7,561 个新代谢物。

在读取检索分数前，登记了它与 adult IBD200 和 pediatric VEO-IBD 的两条近条件家族关系，并自动加入反向关系。两条关系都明确不预设 signed direction；年龄、IBD 亚型、治疗暴露差异保留在 evidence boundary 中。

扩展集合 `data/validation/contrast_atlas_open_response_collection_v6_ibd3_20261022` 现包含：

- 11 个独立 source families；
- 26 个 study-level contrasts；
- 673 个生物单位；
- 106,532 张代表 MS/MS；
- 89,968 条合格响应记录；
- 1,941 条 library-supported 与 88,027 条 unresolved 响应记录；
- 10 条登记的有向关系；每个查询面对 22–25 个来源独立候选。

两个已经冻结的全局算子随后原样重跑，没有任何调参：

- centered kernel 的 unresolved 仍为 0/10 Top-1；
- signed fragment+loss 的 unresolved 为 3/10 Top-1，但 31/31 次响应重排仍全部恰为 3/10，主 gate 继续为 `FAIL_UNKNOWN_RESPONSE_SPECIFICITY`；
- 新关系中，pediatric VEO-IBD → baseline IBD 的 fragment+loss/unresolved rank=2，但 31/31 次重排均达到同样好或更好排名；adult IBD200 → baseline IBD rank=8；反向分别 rank=9 和 14。

因此新增来源不是为了“救活”旧方法，而是提供了真正的新反例：即使条件标签更接近，当前全局 fragment/loss programs 仍不能区分响应归属与来源化学组成。两类全局算子应正式退休，不再在这 10 条 development/consumed 关系上继续调参。

公开依据：[MassIVE 数据记录](https://massive.ucsd.edu/ProteoSAFe/dataset.jsp?task=f6e02685d3904466a472aa0d22d3f195)、[NCT05973500](https://clinicaltrials.gov/ct2/show/NCT05973500?draw=2&rank=1)、[公开研究资料](https://pmc.ncbi.nlm.nih.gov/articles/PMC13409120/)。

## 当前科学结论

已经完成的实质进展不是又一个数据集脚本，而是：

1. 从 95 个具备定量+MS/MS 的公共 FBMN 任务中，实际建立了可扩展的、设计感知的响应对象入口；
2. 将开放候选宇宙扩展到 11 来源、26 contrasts、673 生物单位和 106,532 张 MS/MS；
3. 证明大量 unresolved feature entities 可以进入合法研究级效应估计；
4. 同时证伪了 centered kernel 和 signed fragment/loss programs 两种全局关系核的 unknown-information 增量；后者在 10 条登记关系上观察到的 3/10 Top-1 在 31/31 个响应重排零模型中原样出现；
5. 新增的第三个 IBD 来源不是小候选池：它在 22–25 个来源独立候选中对冻结算子提供了新的负证据。

所以现在可以继续开发 Contrast Atlas，但不能写“unbiased 方法已经成功”。下一阶段优先把跨来源真值扩到多个独立生物学家族及明确 no-match 查询；新的局部 relation-layer 方法只能在尚未参与算子选择的来源上检验。最终成败标准仍是：响应特异的关系证据能否在固定错误风险下提供 measurement-matched known-only 没有的正确答案，并能对无对应查询可靠弃权。
