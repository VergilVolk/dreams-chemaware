# DreaMS 真实错误图谱与因果峰干预协议

**状态**：代码完成，等待服务器全量构建  
**日期**：2026-08-23  
**作用**：为下一轮 ChemAware 噪声微调建立真实、可审计、与封存 P3 隔离的训练依据。

## 1. 为什么重新建图

过去的反事实微调只覆盖约 1,163 个分子，而且把两种不同问题混在一起：

1. **正例缺失**：同一分子的不同实验谱图被条件差异推远；
2. **负例混淆**：不同分子因共享强峰或相似中性丢失被错误拉近。

两种错误需要相反的峰干预。正例缺失要检验“删除条件特异峰能否拉近同分子”；负例混淆要检验“删除共享主峰能否拉远错误候选”。在没有分开这两个轴之前，统一随机删峰会相互抵消，也无法说明学到了什么。

## 2. 数据边界

- 数据：P3 封存后允许用于开发的全部真实 MassSpecGym 训练谱图。
- 查询：所有满足 strict 10 ppm、同加合物、同时存在同分子正例和异分子负例的真实谱图；不再限制每个分子只取 2 张谱。
- 候选：同一查询的完整 strict-10ppm、同加合物候选组；按 IK14 聚合，同一候选分子的多张谱取最大分数。
- 正例标签：仅由 IK14 同一性定义。
- P3：任何 P3 查询身份与错误图查询身份重叠均立即失败。
- 规则：335 条规则只用于描述、分层和解释，不参与正负标签定义。

## 3. 固定比较对象

### 官方 DreaMS

官方 embedding 的余弦相似度，候选分子内取最大谱图相似度。

### 冻结 P2b 排序器

严格复用已经封存的 P2b：

\[
S_{P2b}=0.1S_{DreaMS}+0.1S_{entropy}+0.8S_{neutral-loss}
\]

sqrt cosine 权重为 0；规则与结构标签均不进入分数。融合发生在谱图对层面，之后才按候选分子取最大值，禁止把不同参考谱的最佳特征拼成“Frankenstein candidate”。

## 4. 四种真实转移

每个查询同时记录官方 DreaMS 与 P2b 的结果：

| DreaMS | P2b | 转移类型 | 含义 |
|---|---|---|---|
| 对 | 对 | protected_correct | P2b 没有破坏原本正确结果 |
| 错 | 对 | corrected | 峰级/中性丢失证据修正了 DreaMS |
| 对 | 错 | introduced | P2b 新增错误，必须重点分析 |
| 错 | 错 | persistent_wrong | 两者都未解决的真实残余错误 |

这四类都保留，不能只看 corrected，也不能只挑成功案例训练。

## 5. 错误的双轴分解

错误条件为：

\[
s(q,p) \le \max_n s(q,n)
\]

它可能由两个独立原因造成：

- **positive deficit**：\(s(q,p)\) 异常低；
- **negative excess**：\(\max_n s(q,n)\) 异常高；
- 两者可同时发生。

为了避免把候选数和 near 难度误当成机制，参考分布只来自同一难度层内 DreaMS 正确的查询。分层变量为：

- 是否存在 MCES 0–2 的 near 候选；
- 候选分子数：2、3–4、5–8、9+。

使用组内中位数与 IQR 构造稳健标准分数。阈值只用于筛选后续峰遮蔽队列，不作为训练标签或机制定论。

## 6. 错误图的三层边

### 查询层

一行一个查询：身份、分子式、骨架、采集条件、候选规模、DreaMS/P2b 排名与间隔、四类转移、规则证据。

### 候选分子层

一条边表示 query→candidate molecule：MCES 层、DreaMS/P2b 分数与排名、候选结构、获胜参考谱。

### 谱图对层

一条边表示 query spectrum→candidate spectrum：DreaMS cosine、sqrt cosine、entropy、强度覆盖、Top10 匹配、中性丢失相似度等全部峰级统计。

保留完整边表，后续可以更换聚合与训练目标，不需要重新计算全部特征。

## 7. 化学规则的正确角色

对查询谱、DreaMS Top-1、P2b Top-1、最佳同分子正例计算 335 维规则向量，并报告 CF/NL/ISO/HR/NR/EE 分类重叠。

- 若查询和正例的规则一致性高于错误候选：规则可能提供救援证据；
- 若查询和错误候选的规则一致性更高：规则解释了为什么会混淆，但不能证明结构相似；
- 两张谱均无规则命中时，Jaccard 记为缺失，不能人为记为 1。

规则始终是观测到的质量模式，不等同于唯一碎片结构或断键机理。

## 8. 输出文件

### 图谱构建

目录：`data/validation/g8r_real_error_atlas/`

- `query_summary.csv.gz`
- `candidate_edges.csv.gz`
- `spectrum_pair_edges.csv.gz`
- `top_spectrum_rule_vectors.npz`
- `report.json`

### 深度分析

目录：`data/validation/g8r_real_error_analysis/`

- `query_error_signatures.csv.gz`
- `occlusion_priority_cases.csv.gz`
- `formula_error_enrichment.csv`
- `scaffold_error_enrichment.csv`
- `positive_score_reference.csv`
- `negative_score_reference.csv`
- `report.json`

## 9. 三步服务器流程

```bash
sbatch tasks/run_g8r_real_error_pipeline.sbatch
```

一个单卡任务按顺序完成三个阶段。峰匹配特征由 8 个 CPU 进程并行计算，每个前置阶段都会立即打印进度；任一阶段失败时整个任务停止。所有输出 fail-closed，已有目录时拒绝覆盖。

## 10. 图谱之后才能做的因果实验

### 负例混淆臂

- 对象：negative-excess / shared-major-peak screen；
- 干预：删除查询与错误候选共享的峰；
- 对照：删除数量、强度、m/z 分布匹配的随机峰；
- 方向：错误候选相似度应比随机删除下降更多。

### 正例缺失臂

- 对象：positive-deficit / cross-condition screen；
- 干预：删除同分子谱之间不共享的条件特异峰；
- 对照：同样采用匹配随机删除；
- 方向：同分子相似度应比随机删除上升更多。

只有在独立身份/分子式分组上方向复现、置信区间不跨 0 的样本，才进入噪声微调池。随机 20–30% 删峰保留为通用增强对照，不再承担“纠正特定错误机制”的主假设。

## 11. 微调前的硬门

1. P3 查询身份重叠必须为 0；
2. DreaMS 与 P2b 的实现必须复用同一共享评分函数；
3. 错误图必须覆盖全部有效真实查询，报告谱图加权和身份等权两套结果；
4. corrected、introduced、persistent wrong 全部必须有具体谱图实例；
5. 峰干预必须有匹配随机对照；
6. 规则不能用于定义正负样本；
7. 因果峰证据未经方向复现，不得进入正式微调。

## 12. 当前结论边界

该错误图将回答“错误集中在哪里、比较式错误的哪一侧出了问题、哪些峰/规则证据值得做因果干预”。它本身不证明某个峰是唯一致因，也不保证微调提升。它的价值是阻止团队再次在样本定义、方向相反的噪声、规则循环标签和开发集过拟合上消耗预算。
