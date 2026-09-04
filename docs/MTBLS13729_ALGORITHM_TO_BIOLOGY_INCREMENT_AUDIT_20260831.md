# MTBLS13729 算法—生物学增量审计（2026-08-31）

## 1. 审计问题

本审计回答一个投稿时不可回避的问题：MTBLS13729 的生物学主轴究竟是新算法发现的，还是原论文身份经过新的证据流程被重新找回？官方 DreaMS、实验性 E6 shared embedding 与冻结 P2b 候选专家分别贡献了什么？

结论必须区分四件事：结构身份、候选覆盖、谱学证据一致性与患者丰度。没有标准品真值时，模型之间只能报告 `retained/changed/abstained`，不能把候选变化称为修正。

## 2. 三路真实应用的总体增量

三路推理使用相同的 10 ppm precursor 候选图、相同的一谱/feature/sample 选择和相同的 feature 汇总规则。

| panel | 有质量候选的 feature | 官方 DreaMS 已分配 | E6 已分配 | P2b 已分配 | 官方 / E6 / P2b Level2a-supported |
|---|---:|---:|---:|---:|---:|
| neg_rp | 346 | 345 | 345 | 345 | 30 / 31 / 29 |
| pos_rp | 3,571 | 3,072 | 3,081 | 3,243 | 224 / 245 / 201 |

由此可得两类不同增量：

- E6 是保守的 evidence stabilizer。正相 RPLC 仅增加 9 个已分配 feature，却增加 21 个 Level2a-supported feature。它改变共享 embedding，但当前仅为 seed 20260828、formula fold 0 的实验工件，不能代表最终多折模型。
- P2b 是 coverage expander。正相 RPLC 增加 171 个候选分配，即相对有质量候选池约增加 4.79 个百分点覆盖；但 Level2a-supported 从 224 降至 201。因此新增覆盖只能作为候选 lead，不能称高置信注释增加。

query 层也显示两者行为不同。官方—E6 在 neg_rp/pos_rp 分别改变 366/2,185 个 query；官方—P2b 分别改变 2,720/16,429 个 query。E6 的改变明显更保守，P2b 则广泛重排候选。

## 3. feature 703 Neu5Ac 的来源归属

feature 703 是当前生物学主轴，但不是新模型首次发现的新化学身份。

| 证据层 | 当前事实 | 对论文贡献 | 禁止外推 |
|---|---|---|---|
| 原论文 | negative-HILIC 中为 Level-1 Neu5Ac，且原文已点名 | 提供同队列身份锚 | 不能称首次发现 Neu5Ac |
| 正相 RPLC 正交找回 | m/z 310.11336、RT 49.27 s；33 张 peak-resolved MS2 | 证明另一色谱/极性面板存在同一身份桥 | 不是独立患者队列复制 |
| 经典谱库 | Aceneuramic acid `[M+H]+`，median cosine 0.9054，31 个强支持样本 | 提供候选特异谱学支持 | 不等同本实验同法标准品 Level 1 |
| 官方 DreaMS 候选字段 | 冻结 integrated ledger 中 feature703 的 DreaMS 身份字段为空 | 证明主身份不能归功于官方 DreaMS 自动命名 | 不能据空字段推断模型“识别错误” |
| E6 / P2b 候选级归属 | 当前本地缺少服务器 `threeway_application_v1/*__threeway_features.csv.gz` 明细 | 只能报告总体覆盖/证据增量 | 不能从汇总日志反推 feature703 被哪一路命名 |
| 患者丰度 | 锁定 targeted-EIC：Rmu n=10，10/10 上升，均值 +1.935 log2 | 建立黏液型相对丰度主轴 | 不证明通量、来源或酶活性 |
| 亚型敏感性 | 候选面板最大 q=0.00607；五模块 raw/PQN q=0.00179/0.00162 | 支持 Rmu-relative remodeling | 完整 13,155-target exact-FDR10 仍未通过 |

因此最严格、同时最有价值的表述是：

> 原论文 Level-1 Neu5Ac 身份通过新的 raw-MS2、经典谱库和同患者跨面板证据在正相 RPLC 中被正交找回；统一算法—定量流程使该身份进入可复核的患者配对和亚型敏感性分析。当前证据不支持把 Neu5Ac 身份本身归为 E6 或 P2b 的新发现。

## 4. 算法是否真正帮助了生物学发现

答案是帮助了，但贡献必须写对。

1. **统一候选协议。** 官方 DreaMS、E6 与 P2b 在相同候选图上比较，消除了不同候选池造成的伪增益。
2. **覆盖与证据分离。** P2b 证明可扩大候选覆盖，但同时暴露 Level2a 下降；E6 则证明 shared embedding 的主要应用增量可能是证据一致性，而不是候选数量。
3. **从 MS2 到 MS1 患者终点。** 算法管线把 MS2 身份候选连接到锁定 targeted EIC，使 identity、丰度和亚型终点可以分别审计。
4. **反例和弃权。** proline-sodium-like、leucine/isoleucine-like 等候选被质量、配对或竞争谱库证据降级，说明流程不是只接受正故事。
5. **机制边界校准。** 下游 TCGA、O-glycomics 和 MUC2 glycopeptide 证据把 free Neu5Ac 与 donor、carrier、core、linkage 拆开，防止算法候选被过度包装成全局高唾液酸化。

所以本项目当前的算法—生物学定位是 `algorithm-enabled, evidence-calibrated clinical discovery`，而不是 `AI discovered a new metabolite`。

## 5. 仍缺的决定性候选级审计

服务器三路运行日志已经冻结总体数字，但当前本地没有以下两个 feature-level 表：

- `data/mtbls13729/threeway_application_v1/neg_rp__threeway_features.csv.gz`
- `data/mtbls13729/threeway_application_v1/pos_rp__threeway_features.csv.gz`

在这两个工件同步前，feature703 以及其他重点候选的 official/E6/P2b 身份一致性只能标记为 `not locally verified`。同步后应运行 `tasks/audit_mtbls13729_integrated_candidate_algorithm_value.py`，冻结逐候选矩阵；不得用汇总计数代替。

## 6. 投稿用一句话

> A unified DreaMS-based annotation and evidence-calibration workflow recovered a source-anchored Neu5Ac feature in an orthogonal LC-MS panel and connected it to a reproducible mucinous-relative abundance phenotype; transcriptomic and glycomic evidence then resolved the phenotype as a hybrid mucin-glycome remodeling program rather than global hypersialylation.

## 7. 证据来源

- 三路应用冻结日志：`mtbls13729_p2b_2326596.out`
- 生物学候选总账：`data/mtbls13729/integrated_biology_ledger_v2/integrated_candidate_ledger_v2.csv`
- 候选声明计分卡：`data/mtbls13729/candidate_claim_scorecard_v3/candidate_claim_scorecard_v3.csv`
- Neu5Ac 主图与源数据：`data/mtbls13729/neu5ac_glycan_publication_figure_v2_final/`
- 完成度总账：`data/mtbls13729/mechanism_paper_completion_audit_v2_final/`

