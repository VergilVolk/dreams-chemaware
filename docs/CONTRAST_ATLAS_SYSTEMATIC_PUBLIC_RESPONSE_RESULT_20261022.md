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

## 下一项唯一高性价比方法实验

下一项关系层算子不再继续调全局均值，而是冻结为**局部、多尺度的 signed chemical response programs**：

1. 每个研究内保留多个局部响应程序，而不是把相反的脂质、肽和小极性分子变化相互抵消；
2. 同时使用 fragment 和 neutral-loss 证据，分别对应共享子结构和局部碎裂差异；
3. identity track 只累积同实体证据；relation track 只比较局部程序，不合并不同实体的 beta/SE；
4. 先在新加入的、设计清楚的 B. subtilis 三组织响应中检验同一干预能否跨组织恢复，再一次性回到 10-source/25-contrast 开放候选集合；
5. all、known、measurement-matched known、unresolved-shuffled-MS2 对照必须保留；只有 unresolved/all 在固定错误风险下增加正确回答覆盖，才称 unknown increment。

Noise、ChemAware、P2b 或统一算法资产只通过冻结谱学接口替换 fragment/neutral-loss 相似性；不得为本生物学面板重新训练，也不得把模块名称本身当创新。

## 当前科学结论

已经完成的实质进展不是又一个数据集脚本，而是：

1. 从 95 个具备定量+MS/MS 的公共 FBMN 任务中，实际建立了可扩展的、设计感知的响应对象入口；
2. 将开放候选宇宙扩展到 10 来源、25 contrasts、651 生物单位和 99,265 张 MS/MS；
3. 证明大量 unresolved feature entities 可以进入合法研究级效应估计；
4. 同时证伪了当前全局关系核的 unknown-information 增量。

所以现在可以继续开发 Contrast Atlas，但不能写“unbiased 方法已经成功”。下一阶段的成败标准只有一个：局部化学响应程序是否能在未调参的新来源中恢复正确关系，并在开放 source-disjoint 检索中提供 known-only 没有的正确答案。
