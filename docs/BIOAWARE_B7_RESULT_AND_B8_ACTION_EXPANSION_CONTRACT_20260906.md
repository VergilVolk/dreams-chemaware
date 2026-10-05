# BioAware B7 结果与 B8 严格匹配对照契约（2026-09-06）

> 纠错记录：最初草拟的“把 Rhea 全局收录/度数移植到 23,876-query
> MassSpecGym 图后再训练”的 B8 已在执行前撤回。它改变了 B7 数据域与知识源，
> 还错误地把外部面板上的常数特征推广到新图；更根本地，它没有先证明
> BioAware 选样比同难度普通错误更可学习。对应可执行脚本已删除，不得提交。

## B7 已观察结果

B7 首次证明了候选图先验可以通过直接 listwise 损失向共享 DreaMS encoder
传入非零梯度，而不是只停留在 embedding 后重排。工程复现门通过：官方 query、
reference 与候选分数的最大误差均低于 `9e-7`，严格 Top-1 排名错配为 0。

候选动作的开发头空间为：

- 548 个负离子 query；
- baseline Recall@1 = 0.7026；
- action Recall@1 = 0.7509，差值 +4.84 pp；
- corrected / introduced = 191 / 85（这些是五个 outer view 的重复计数）；
- 去重后 corrected identities / formulas = 27 / 25；
- 一步直接梯度对 baseline-wrong margin 的平均改善为 +0.00469，882/882 支持。

共享 encoder 的 fold-0 留出结果为：

- Recall@1：0.6514 -> 0.6606，+0.92 pp；
- MRR：0.77245 -> 0.77092，下降 0.00153；
- corrected / introduced = 3 / 2；
- `corrected - 2*introduced = -1`；
- preservation = 0.98845；
- 训练四轮中梯度裁剪比例依次为 1.00、1.00、0.961、0.75。

因此 B7 是“梯度注入可行”的阳性工程结果，但不是性能通过：只有一个 outer fold，
净改正一个 query，McNemar 不显著，MRR 与风险净收益为负，而且表示漂移过大。

## 必须纠正的科学身份

B7 的 +4.84 pp 动作采用 `catalog_opportunity`：官方 DreaMS 相似度、Rhea
是否收录、Rhea 度数，以及在外部测试中为常数的 mass-candidate fraction。
它不是反应路径证据，也不是同一样本代谢上下文。此前真正的
`reaction_strength` 相对 `catalog_opportunity` 仅有 6 个修正和 6 个新增，净增益为
0；反应共丰度相对匹配随机边同样未通过。

所以不能把 B7 写成“反应网络微调成功”。准确表述是：

> Rhea catalog opportunity 发现了一组能够产生直接共享-embedding 梯度的困难
> 候选边界，但目前尚不能区分可泛化的谱学结构信号与数据库收录偏置。

## 修正后的 B8：同协议 matched-hard-error 对照

B8 不扩大数据，也不重算任何 Rhea 特征。它固定使用 B7 已经验证过的：

- 548 个 Full16 负离子 query；
- 同一个 B4 manifest 与五个 truth-formula fold；
- 同一个 B7 nested formula-OOF graph action ledger；
- 同一个共享 encoder、loss、seed、步数、学习率和 held fold。

训练器按 truth identity 等权抽样，因此 query 级匹配仍会改变有效训练质量。对每个
outer fold，B8 先把 B7 graph-prior 选中的 baseline-wrong corrective query 按 truth
identity 聚簇，再与普通 baseline-wrong identity 无放回一对一匹配；簇内 query 必须
在相同 biological unit 中配对。两臂保留完全相同的 identity 数、每个匹配 identity
的 query 数和 unit 分布。匹配不得使用训练后 embedding 或 held outcome，距离只使用
baseline rank、候选数量、DreaMS truth margin、baseline top gap、正例参考谱数和
truth identity 的原始重复次数；配对对象不得共享 truth identity 或 formula。原 graph
action 会破坏的 query 在两臂中使用相同 safety 权重。

若完整 graph corrective 集不能严格匹配，则 graph 与 generic 两臂同时裁到相同的
可匹配 identity 子集，并重新运行两个 fold-0；不得拿完整 B7 graph 结果与裁剪后的
generic 结果比较。平衡统计必须与训练器一致，按每个 truth identity 一个均值向量计算，
不得按 query 数给重复 identity 额外权重。共同支持采用相同 unit、相同 baseline-rank/
候选数/正例参考谱数分箱以及基础难度 caliper；若仍不平衡，只能成对删除 graph-generic
identity，不得放宽 `|SMD| <= 0.25`。两臂必须使用完全相同训练配置。这个实验只回答：

> graph-prior 选中的边界，是否比相同难度的普通 DreaMS 错误包含更多可由共享
> spectrum encoder 学到的信号？

如果 matched-generic 与 B7 相当或更好，B7 的 +0.92 pp 只能解释为一般 hard-error
微调，graph prior 不具有 embedding 选样增量，停止该 embedding 路线。只有 B7 在
相同 held fold 上同时优于 matched-generic 的 Recall@1、MRR 与风险净收益，才允许
扩展剩余公式折；即便如此，仍只能称 catalog-prior-guided hard mining，不得称为
reaction-specific BioAware embedding。

## 当前不允许的结论

- B7/B8 不是 SOTA 证据；
- catalog membership/degree 不是反应机制；
- 动作头空间不是共享 embedding 的实际增益；
- 不能用 P2b、表型或测试集结果作为 B8 输入；
- B8 未通过前不得扩大解冻层数或运行多 seed 大训练。

## B8 首轮共同支持失败与修订（2026-09-06）

首轮 identity-cluster matcher 要求一个 generic identity 逐条复现 graph identity
的全部 query 与 acquisition unit，并同时匹配候选数、正例参考谱数和原始 identity
重复次数。fold 0 的 22 个 graph-corrective identities 最终只保留 2 个，最大
绝对 SMD 为 1.699；其余折同样只有 2--3 个 identity。该结果说明首轮 B8
estimand 不可识别，不能通过放宽 SMD 门或直接训练来修补。

修订后的 B8 与训练器实际的 identity-uniform estimand 对齐：每个 matched
identity 只保留一个代表 query；graph/generic query 必须来自同一 biological
unit、完全相同的 baseline rank、候选数、正例参考谱数及 identity query 数，
并满足冻结的 truth-margin 与 Top1--Top2 gap caliper。所有 rank、候选数与 margin
均须在排除 held-formula 候选后的真实反传候选图上重算；过滤后已不再错误的 query
不得进入任一 corrective arm。正式训练每个候选固定一张
reference spectrum，因此两个臂的 identity、query 和 reference 训练质量完全
相同。若修订后 fold 0 仍少于 12 个 identity 或最大绝对 SMD 超过 0.25，程序
写出 `not_identifiable` 报告后正常停止，不启动任何 GPU 训练。

该修订仍只检验 `catalog-opportunity-guided hard mining` 是否比同难度普通错误
更可学习。B7 使用的 membership/degree prior 不是反应路径证据；无论 B8 结果
如何，都不得据此声称 reaction-specific BioAware embedding。
