# BioAware 当前状态与动作准入门（2026-09-06）

## 结论

BioAware 尚未得到可以宣称为“反应特异、可泛化、可注入共享 embedding”的动作。

当前最强的已打开开发集结果是同分子式候选任务上的 `+5.81 pp`（39 corrected / 11 introduced）。重新拆解后，`DreaMS + known network degree` 单独得到 `+6.02 pp`（38 / 9）；旧 BioAware 路径模型相对 degree-only 为 `-0.21 pp`，formula-cluster CI 跨零。固定 degree 与路径可得性后的联合置换检验也不显著（Recall@1 `p=0.624`，risk-net `p=0.574`）。因此，旧增益主要可由网络数据库覆盖度/节点流行度解释，不能解释为真实反应路径证据。

现阶段没有任何共享 embedding 实验建立了稳定增益。B4--B6 同时存在动作覆盖不足和训练协议/梯度注入缺陷；这些结果既不能支持 BioAware embedding，也不能否定生物上下文本身。

## 当前可以保留的两类资产

1. **工程先验基线**：`DreaMS + network degree`。它可作为候选排序的 catalog-prior comparator，但不得命名为 reaction-aware，也不得直接作为共享 embedding 教师。
2. **反应证据原始资产**：已知 Rhea/KEGG 边、raw path nodes、complete path、path multiplicity、bottleneck、谱图/中性丢失证据。这些是下一轮动作发现的输入，不是已经通过的动作。

## 当前执行任务

首先扩展并审计 MetDNA3 已下载的完整 16-panel Level-1 数据，而不是继续只使用负离子小子集：

- 4 个生物来源；
- HILIC/RPLC；
- 正/负离子；
- 6004 个 Level-1 行、1240 个 Level-1 identity；
- 严格 10 ppm、同 adduct 的真实候选组。

本阶段不读取 embedding、不拟合模型、不选择阈值，且不使用 P2b。输出必须回答：

- 精确可评估的歧义 identity/formula 数；
- 每个 biological source 的有效覆盖；
- leave-one-biological-source-out 且 identity/formula purge 后的训练规模；
- Rhea、KEGG 及其并集的候选级覆盖与区分机会；
- 是否达到至少 600 identity / 450 formula、每个 held-out source 至少约 100 个 edge-bearing identity 的功效要求。

## 下一条允许检验的 BioAware 动作

只有规模门通过后，检验“机会校准的反应特异支持”：

1. 主对照锁为 `DreaMS + 全部 network opportunity main effects`，包括 degree、path count、node-combination count、reference multiplicity 和 path availability。
2. 真实已知反应路径与 path-length、degree、谱图数量、质量差、元素变化和结构相似度匹配的非反应路径比较。
3. 缺失严格拆为 `no_path`、`path_unobservable`、`complete_nonspecific`、`complete_specific`，禁止以 0 填充并混为一类。
4. 外层按 4 个 biological source 留一，内层按 identity/formula/reaction component cross-fit；禁止同一来源的另一 LC 模式进入训练。
5. 主终点是该动作相对 full-opportunity baseline 的增量，而不是相对裸 DreaMS 的总增益。

若机会校准路径仍失败，立即转向更接近化学机理的 `reaction-transform x fragment/neutral-loss concordance`，不再继续堆路径计数或调 logistic gate。

## 冻结门

一个动作只有同时满足以下条件，才允许进入重排器或 embedding：

- 相对 full-opportunity baseline 的 Recall@1 增量至少 `+3.0 pp`；
- identity-cluster 与 formula-cluster CI 下界均大于 0；
- corrected identity 大于 `2 x introduced identity`；
- 至少 25 个增量 corrected identity、净增至少 15、至少 50 个 switched identity、覆盖至少 30 个 formula；
- 4 个 biological-source fold 风险净收益均非负、至少 3/4 的 Recall@1 增量为正、最差 fold 不低于 `-1 pp`；
- matched-control 覆盖至少 90%，无 fallback；
- degree/path/opportunity-matched null、graph rewiring 和 seed shuffle 经多重校正后均通过。

达到上述门后，最高性价比顺序是：

1. 冻结轻量候选组内重排器，先验证真实部署增益；
2. 检验动作能否由单张谱图预测；
3. 只有谱图可恢复成分才直接微调共享 embedding；依赖样本 feature graph 的成分必须保留为 context-conditioned ranker/adapter。

任何 `3--5 pp` 都是准入标准，不是预先保证的结果。未通过上述反证时，不得称为 BioAware 方法学增益或 SOTA。
