# 反向代谢组学相关思路：事实重置与未决问题

**日期：2026-09-12**  
**状态：深度调研中的事实台账；不提出新方法，不构成实施路线**

## 0. 为什么重置

上一份报告犯了方法论错误：先提出“化学探针场”，再用文献给它寻找位置；同时把 DeepMet 的部分流程误归入反向代谢组学。正确顺序应是先逐篇固定输入、输出、验证与真正贡献，再判断师兄的启发究竟能导向什么新问题。

## 1. 已核实的工作边界

| 工作 | 真正输入 | 真正输出 | 关键验证 | 不能被误说成什么 |
|---|---|---|---|---|
| [Reverse metabolomics, Nature 2024](https://www.nature.com/articles/s41586-023-06906-8) | 新合成化合物得到的 MS/MS 谱 | 公共仓库中的谱图匹配及其物种、组织、疾病等元数据 | 独立 IBD 队列，同实验 RT+MS/MS，定量、免疫/PXR实验、细菌培养 | 不是把 query/library 参数交换；其创新包含组合合成、仓库检索、表型关联和实验闭环 |
| [Reverse metabolomics protocol, 2025](https://www.nature.com/articles/s41596-024-01136-2) | 已知或未知分子的 MS/MS/USI | MASST 命中文件 + ReDU 元数据 | 要求返回原始队列做 feature extraction 和验证 | 不是自动给出定量、身份真值或机制 |
| [DreaMS, 2025/2026](https://www.nature.com/articles/s41587-025-02663-3) | 大规模未标注 MS/MS；下游任务另用标注数据微调 | 通用谱图表示、相似度/指纹/性质/含氟任务；201M 谱 Atlas | 多种下游基准；Atlas 个案是研究假设生成 | Atlas 不是标准品反向代谢组学，也不是已经验证的疾病机制平台 |
| DeepMet, Nature 2026（本地全文已核） | 2,046 个已知人类代谢物结构训练 LSTM；结构字符串生成 | 生成代谢物样结构；两条核心应用及后续仓库检索见下文 | 购买/合成标准，同平台 RT+MS/MS；held-out 结构注释；新参考谱还被用于搜索 35,460 个人体组织/细胞样本 | 不是 DeepMind；不是 Transformer；其核心模型不是反向代谢组学，但论文确实包含标准谱先行的公共数据搜索应用 |
| [StructureMASST, 2026](https://www.nature.com/articles/s41587-026-03082-8) | 名称、SMILES、SMARTS、结构/亚结构、质量修饰 | 参考谱及公共仓库样本分布；精确、类似物、指定/盲修饰搜索 | Pan-ReDU 元数据、同文件共现、ModiFinder 等 | “结构/标准→全仓库生物分布”已经不能单独算新意 |
| [N-acyl lipid resource, Cell 2025](https://www.sciencedirect.com/science/article/pii/S0092867425005653) | N-acyl lipid 诊断模式、MassQL、标准谱 | 2,700 项目中的家族资源：851 个 N-acyl lipids，777 个不在结构数据库 | 13 个标准 RT+MS/MS；HIV/认知关联；T 细胞功能 | “从一个化学类扩展未知家族并联系疾病”也已有强先例 |
| [Drug exposure records, Nat Commun 2025](https://www.nature.com/articles/s41467-025-65993-5) | 103,209 个药物/代谢物参考谱 | 公共数据中的药物及未知类似物暴露记录 | 仓库网络+fastMASST；代谢质量差过滤；ModiFinder | “已知物→修饰类似物→来源/暴露”已有系统实现 |
| [MetSummarizer, 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8050247/) | 每个样本的已知谱特征 presence/absence | 跨仓库表型分类、复杂混合物成分分解 | 跨项目评估并揭示内标/批次伪关联 | “把样本投影到固定谱特征后预测表型”不是空白，而且极易学到实验室混杂 |
| [Pseudo-targeted metabolomics, Nat Protoc 2020](https://www.nature.com/articles/s41596-020-0341-5) | pooled reference 的 untargeted 数据生成 MRM transition | 在大批样本中稳定半定量约 800–1,300 个目标 | RT 校正、靶向采集流程 | “固定一组谱学传感器再扫全队列”在分析化学上已有成熟近邻 |

## 2. DeepMet 必须单独拆开的两条路线

### 2.1 Application 1：结构先行的定向发现

1. 用 ChEMBL 预训练、2,046 个已知人类代谢物微调的 LSTM 生成 SMILES；
2. 以生成频次等指标优先排序代谢物样结构；
3. 从高优先级结构中购买 106 个标准并定制合成 2 个；80 个获得合格 MS/MS；
4. 用同一色谱与质谱平台在尿液和血液数据中，以 RT 和 MS/MS 检查这些预测物；
5. 在该部分确认 17 个预测物，但其中部分后来发现是 HMDB 漏收的已知物。

这是一种**结构生成驱动的定向代谢物发现**。它和反向代谢组学共享“候选结构/标准先行”的表面方向，但没有以 MASST/ReDU 搜索公共仓库为中心。

需要补充一个容易被过度简化的事实：DeepMet 论文随后把本研究获得的参考 MS/MS 谱搜索了来自 MetaboLights 和 Metabolomics Workbench 的 35,460 个人体组织和细胞样本。因此，DeepMet 论文包含“新参考谱→公共生物样本分布”的应用层；但生成模型的训练目标仍是学习代谢物样结构分布，不能把这个下游应用反过来定义成 DeepMet 的模型本体。

### 2.2 Application 2：未知实验谱的结构注释

1. 输入是实验中未注释的 MS/MS/精确质量；
2. DeepMet 生成或优先排序候选结构；
3. CFM-ID 从结构预测 MS/MS；
4. meta-learner 融合生成先验、预测谱相似度、碎片数、质量误差、同位素和 RT 等正交证据；
5. 购买/合成标准验证未知峰的完整结构。

这仍然属于“未知谱→结构候选”的常规方向，只是候选空间与融合方式更强。不能拿它证明“反向样本搜索已经做完”，也不能把其融合增益移植成 DreaMS embedding 增益。

## 3. 现在可以排除的伪创新

以下命题均已被现有工作直接或近似覆盖，不能作为我们跳出原范式的核心：

1. 用 DreaMS 相似度替换 cosine 做 MASST；
2. 用标准谱扫描一个病例—对照队列并比较检出率；
3. 把所有标准谱作为固定特征，再训练疾病分类器；
4. 从标准谱做类似物/质量修饰开放搜索；
5. 用 Rhea/KEGG 图把标准命中传播到相邻代谢物；
6. 把以上模块串联后称为新范式；
7. 把 DeepMet 的结构先验或随机森林融合说成反向代谢组学。

## 4. 师兄这句话中仍未被我们理解的部分

“把标准品谱图输入，检测不同生物、疾病组织中有没有、以及有多少”至少有四种不同科学含义，目前不能擅自选一个：

1. **检测方向反转：**标准作 query，样本作库——这就是 reverse metabolomics/StructureMASST；
2. **测量坐标反转：**不再先做 feature finding，而把样本直接表示成对一组标准的响应向量——接近谱库引导/伪靶向分析；
3. **预测方向反转：**不再从样本预测化学身份，而从化学标准预测其可能出现的生物上下文——接近化学到表型/来源建模；
4. **研究设计反转：**不是从队列显著峰出发，而是先提出一个完整化学空间，再用多队列决定其生物学问题——最接近 Nature 原作的真正价值。

师兄可能指其中一个，也可能是把它们进一步抽象后产生第五种问题。当前证据不足，不能把第 4 种自动改名为“化学探针场”。

## 5. 下一轮调研必须回答，而不是先设计模型

1. 成功的 reverse-metabolomics 后续论文究竟增加了哪一步：更大合成库、更好的搜索、跨队列统计、来源追踪，还是功能验证？
2. “标准谱集合→样本表示→生物问题”在哪些领域已经以 pseudo-targeted、DIA library、spectral barcode、phenotype atlas 等名字存在？
3. 是否存在直接学习 $P(\text{biological context}\mid\text{spectrum or structure})$ 的仓库模型？它如何避免实验室、仪器、内标和样本类型泄漏？
4. 在不逐个购买标准的情况下，哪些结论仍然可由 reference-anchored evidence 合法支持？哪些仍必须同平台 RT+MS/MS？
5. 我们的独特能力必须依赖哪一项现有方法做不到的事实：跨条件 embedding、近异构体纠错、峰级双重映射，还是化学规则的可干预性？若去掉 DreaMS 改进仍能完成，便不是我们的核心创新。

## 6. 当前裁决

现在不能给师兄的思路下最终定义，也不能启动“BioAware + reverse metabolomics”或“化学探针场”实现。正确动作是继续建立逐篇任务图，并在所有最近竞争方法上执行同一套排除测试。只有发现一个**无法被 StructureMASST、Cell 2025 N-acyl resource、MetSummarizer、pseudo-targeted workflow 或普通 DreaMS Atlas 等价实现**的科学问题后，才进入设计阶段。
