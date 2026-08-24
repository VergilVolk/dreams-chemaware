# G8R RAW reranker：冻结结论与 P1–P3 路线（2026-08-22）

## 1. RAW-v1 已冻结的事实

RAW-v1 是冻结 DreaMS 之后的候选组内二阶段重排器，不是 DreaMS 权重微调，
也没有证明 embedding 得到改善。

| 数据/协议 | DreaMS Recall@1 | RAW-v1 Recall@1 | 变化 | corrected / introduced | 结论 |
|---|---:|---:|---:|---:|---|
| g8r_val（已消耗开发集） | 0.8081 | 0.8516 | +4.35 pp | 44 / 17 | 强开发信号，不能再用于调参后的外部证明 |
| 原协议 Test-A | 0.9225 | 0.9270 | +0.45 pp | 59 / 50 | CI 跨 0，未显著 |
| 原协议 Test-B | 0.8956 | 0.9029 | +0.73 pp | 77 / 64 | CI 跨 0，未显著 |

Test-A 与 Test-B 有 1,427 个 IK14 重叠。若 Test-B 定义为异构体困难视图，
这种重叠不使各面板内部结果失效，但禁止将二者作为两次独立复现或合并 p 值。

允许的阶段性结论：RAW 峰级证据在开发集上具有明显补充信号，在原协议冻结测试
中保持弱正向点估计，但尚未达到显著超越官方 DreaMS 的证据标准。

## 2. P1：风险受控的干预策略

目标不是让重排器改动更多 query，而是识别重排的预期收益：

\[
u(q)=\operatorname{Top1}_{\mathrm{rerank}}(q)
-\operatorname{Top1}_{\mathrm{DreaMS}}(q)\in\{-1,0,+1\}.
\]

- `+1`：修正错误；
- `-1`：引入错误；
- `0`：中性干预。

P1 必须使用完全 OOF 的 RAW-ranker 预测产生门控特征和效用标签；门控外层选择也
必须按 formula/IK14 分组。不能先在全训练集拟合 RAW ranker，再把其训练内预测
伪装成门控 OOF。

风险选择应使用固定、预注册的效用或约束，例如：

\[
U=\mathrm{corrected}-\lambda\,\mathrm{introduced},\quad \lambda>1,
\]

并在所有候选超参数之间使用同一个 \(\lambda\)。更稳妥的选择是：在 introduced
上限或 introduced/corrected 比例约束下最大化 corrected，并报告完整
risk–coverage 曲线。

P1 输出至少包括：corrected、introduced、net、干预精确率、修正召回率、gate
coverage、Recall@1/MRR、formula-cluster CI，以及 seen/unseen formula/scaffold
和候选数分层。

`g8r_val`、Test-A、Test-B 均已消耗，只能用于兼容性和描述性审计，不能继续选择
P1 阈值或宣称新的外部提升。

## 3. P2：候选组内 listwise 排序

P2 直接对同一 query 的完整候选组训练：

\[
L_{\mathrm{list}}=-\log\frac{\exp S(q,p)}
{\exp S(q,p)+\sum_{n\in C(q)}\exp S(q,n)}.
\]

最终分数采用受保护的残差形式：

\[
S_{\mathrm{final}}=S_{\mathrm{DreaMS}}+\alpha(q)\,\Delta S_{\mathrm{peak}},
\quad 0\leq\alpha(q)\leq1.
\]

每个 query 等权；候选生成必须与部署协议一致；同时覆盖自然负样本、MCES 0–2
困难负样本和跨条件同分子正例。规则只可作为候选特征或解释证据，不可充当标签。

## 4. P3：新的锁定验证集

MassSpecGym HDF5 当前只有 train/val，已有 val 已被反复查看。P3 必须在 P2 开发前
从尚未用于 P2 训练的数据中预先锁定新测试集，并审计：

- spectrum row、IK14、完整 InChIKey；
- formula 与 Murcko scaffold；
- 仪器、碰撞能量、加合物和候选数量；
- 一般检索与 near-isomer 困难视图的层级关系。

模型、特征、门控、阈值和容差冻结后只评价一次。若一般面板与困难面板重叠，
困难面板应明确称为亚组，而非第二个独立复现。

## 5. 当前 P1 脚本的阻塞项

1. RAW ranker 先在全训练缓存上拟合，门控 OOF 使用了训练内 RAW 预测，存在泄漏。
2. 门控标签把“修正”设为 1，将“新增错误、双方正确、双方错误”全部设为 0；
   `class_weight` 因而没有专门惩罚新增错误。
3. 每个 cost ratio 又用自己的 ratio 计算选择分数，候选超参数之间的目标不一致。
4. `raw_v1_recall1` 实际是重新拟合的无门控 ranker，不是冻结 RAW-v1 系统。
5. 文档声称采用加性残差分数，代码实际在 DreaMS 排序与 RAW 排序之间硬切换。
6. 只使用 `predict()` 的 0.5 阈值，没有在固定效用下选择风险阈值。
7. 输出缺少 CI、风险—覆盖曲线和干预精确率，当前成功标准可由缩小 gate 平凡满足。
8. formula OOF 目前只按 query formula 切分，必须审计 held-out IK14/谱图是否仍作为
   candidate 出现在 ranker 训练折；若存在，应从训练折候选角色中一并剔除。
9. 当前任务只读取缓存并训练小型 sklearn 模型，不使用 GPU；正式脚本不应申请 GPU。

以上阻塞清除前，不提交 `run_g8r_p1_risk_gate.sbatch` 作为正式实验。

## 6. P1 开发结果与复审（2026-08-22）

已报告的 g8r_val 结果如下；该集合已经被反复查看，结果仅用于开发诊断：

| 系统 | Recall@1 | corrected / introduced | gate coverage | 固定 \(\lambda=2\) 效用 |
|---|---:|---:|---:|---:|
| DreaMS | 0.8081 | — | — | — |
| 冻结 RAW-v1 | 0.8516 | 44 / 17 | 46.45% | 10 |
| P1 selective switch | 0.8435 | 29 / 7 | 8.06% | 15 |

P1 相对 DreaMS 的 Recall@1 增量为 +0.0355，已报告 formula-cluster CI
[+0.0165,+0.0563]。该区间不回答 P1 是否优于 RAW-v1；还需直接对
`P1 - frozen RAW-v1` 做逐 query 配对区间和转换表。

8.06% × 620 约为 50 次开门，其中 29 次修正、7 次引入错误、14 次对 Top-1
正确性中性。因此必须区分：

- 所有开门中的修正率：29/50 = 58%；
- 所有开门中的破坏率：7/50 = 14%；
- 非中性变化中的条件胜率：29/(29+7) = 80.6%。

当前脚本把第三项命名为 `intervention_precision`，容易误读成第一项，应更名并同时
报告三项。RAW-v1 同理：44/288=15.3% 是所有开门中的修正率，44/(44+17)=72.1%
才是非中性变化中的条件胜率。

“风险曲线在较宽覆盖率范围内 corrected/introduced 不变”不能直接证明门控无判别力。
它可能表示新增开门全部是中性切换，也可能由 `agree` 二值特征造成大量效用分数并列。
必须保存每个 query 的 `p_correct`、`p_introduce`、`U`、gate、两个 Top-1、utility，
并报告唯一 U 值数量、并列块大小和基于唯一阈值的 risk–coverage 曲线。

正式进入 P2 前，P1 还需完成四项判别性对照：

1. 同覆盖率随机 gate；
2. 仅 `DreaMS-vs-RAW Top-1 disagreement` gate；
3. 仅 DreaMS 低置信度 gate；
4. 完整学习 gate。

若完整 gate 不能在相同覆盖率下显著优于这些简单对照，就不能宣称学到了 query-level
风险，只能称为选择性过滤。另需把 RAW-ranker 与 gate 做严格嵌套 formula OOF，
并清除 held-out IK14/谱图以 candidate 身份进入上游训练折的可能性。

## 7. P1 四门控对照复审（2026-08-23）

同为 50/620（8.06%）开门时，已报告：

| gate | corrected | introduced | neutral | \(corrected-2\,introduced\) | 未加权 net |
|---|---:|---:|---:|---:|---:|
| disagreement-only | 37 | 11 | 2 | 15 | 26 |
| learned gate | 29 | 7 | 14 | 15 | 22 |
| low-confidence | 14 | 3 | 33 | 8 | 11 |
| random（单次报告） | 4 | 1 | 45 | 2 | 3 |

按预注册的 \(\lambda=2\) 风险效用，disagreement-only 与 learned gate **并列**，
不能再使用未加权 net 将 disagreement-only 宣布为胜者。learned gate 相对前者少修正
8 个，同时少引入 4 个错误，恰好是 \(8-2\times4=0\) 的等价交换。

当前 `disagreement-only=50` 仍需审计：若共有 66 个 disagreement，纯二值 gate
无法自然选择其中 50 个，必须公开其 tie-break（随机、行顺序、DreaMS margin 或 U）。
任何使用标签或 learned-U 选择这 50 个的实现都不再是 disagreement-only。

进入 P2 前只补三项廉价裁决：

1. 报告全部 disagreement query 的 corrected/introduced/neutral；
2. 在 disagreement 子集中按多个固定 k 比较 learned-U 与 1,000 次随机抽样分布；
3. 保存逐 query 的两套 Top-1、U、gate、utility，计算 learned-U 区分 `+1` 与 `-1`
   的组内 AUC/AUPRC和 formula-cluster CI。

若 learned-U 在 disagreement 子集中不能优于随机选择，则冻结最简单的
`disagreement-only` 工程基线并终止 P1；若能够显著降低同覆盖率 introduced，P1
可作为 P2 后的安全门保留。无论结果如何，P3 测试集必须在开发任何 P2 模型或新增
门控特征之前先构建、哈希并封存。

## 8. P3 数据不足裁决与锁集设计（2026-08-23）

严格排除 g8r、RAW cache、Test-A/B/C 和历史 large OOD 审计后，当前可获得约
1,914 个 pristine near/isomer IK14。裁决如下：

1. 不为凑数量而把历史 large OOD 暴露分子混入主盲测；它们未参与训练，但其结果
   已影响过算法决策，属于 adaptive evaluation exposure。
2. 1,914 个独立 IK14 足以作为 near 挑战视图；near 不要求与 main 独立。它可以与
   main 重叠，但必须明确称为困难亚组/挑战视图，禁止合并 p 值或称作第二次独立复现。
3. 解除 large-audit 排除后新增的分子单列为 `P3-near-exposed-extension`，只用于次级
   稳健性分析，不能进入主 CI/p 值。
4. main 与 near 不再强制互斥。main 保持代表性抽样；near 使用全部合格 pristine
   near IK14，最大化困难层统计功效。

P3 必须把“query 是否干净”与“部署候选库”分开：

- query pool：严格排除所有曾用于训练、调参或自适应评价的 IK14；
- reference candidate library：按最终部署协议单独冻结，采用同一完整参考库、
  strict 10 ppm、same adduct、exclude self、per-IK14 molecule aggregation；
- 不得只在 pristine query 子集内部寻找候选，否则会再次改变候选数和任务难度。

建议同时保留一个完全未暴露 candidate 的严格 inductive stress view，但它是次级压力
测试，不能替代部署对齐的主候选图。

当前 `build_g8r_p3_test.py` 锁集前必须修复：

1. query mask 与 candidate-library mask 分离；
2. near 不再从 main 中剔除，删除人为 disjoint 约束；
3. `~isfinite(dCE)` 不能算作跨碰撞能证据，缺失值应标记 unknown；
4. `--no-exclude-large` 时 provenance/source 字段必须随配置改变；
5. 记录 HDF5、pairs.json、所有排除清单及候选库的 SHA256；
6. 输出目录存在时 fail-closed，防止覆盖已封存 manifest；
7. 同时输出 P2 允许训练的 IK14 清单，并让 P2 loader 强制排除 P3 query IK14；
8. 明确候选谱重复到 molecule score 的聚合规则，避免谱图数量改变排序权重。

## 9. P3 重写后的二次审计（2026-08-23）

重写版本已做到 query/candidate 分离、main/near 允许重叠、CE 缺失修复及基本哈希，
但 formal seal 仍被以下问题阻塞。

### 9.1 真实谱与模拟谱被混入同一“部署库”

HDF5 共 231,104 条，其中 `SIMULATION_CHALLENGE=True` 模拟谱 119,029 条，真实实验谱
112,075 条。当前 P3 query pool 和 full-HDF5 candidate library 均未过滤该字段。
仓库既有参考库规范也明确丢弃模拟谱。主 P3 必须定义为：

- query：`SIMULATION_CHALLENGE=False`；
- deployment candidate library：`SIMULATION_CHALLENGE=False`；
- 模拟谱如需保留，单建 `P3-sim-to-real` 次级压力面板，不能进入主结果。

已有 g8r train 的 10,000 anchors 中 6,612 条为模拟谱，g8r val 的 2,000 anchors 中
1,304 条为模拟谱。因此 RAW/P1 的开发提升可能包含明显的 simulation-specific 成分，
不能直接表述成真实实验 LC-MS/MS 的提升。

### 9.2 面板命名与MCES层不一致

报告的 `near-pristine=2,788` 实际定义为“存在任意同分子式异构体负候选”；其中已知
MCES 0–2 只有 581，MCES 0–5 为 778。应拆成：

- `P3-isomer-pristine`：全部同分子式异构体 query；
- `P3-near-core-pristine`：MCES 0–2；
- 可选 `P3-nearmid-pristine`：MCES 0–5。

不能用 2,788 作为 near 样本量或 near 统计功效。

### 9.3 MCES映射没有覆盖完整候选库

`pairs.json` 只枚举 train fold 分子对，而 P3 candidate library 横跨 train+val。
query 与 val-fold 同分子式候选之间没有 MCES 记录会被误记为“非near/未知”。锁集前需对
P3 query 的全部唯一同分子式候选对补算或明确标记 `MCES unknown`，随后再生成 near-core。

### 9.4 P2训练池被错误缩回旧g8r

当前 `P2-allowed=sorted(g8r_anchor)` 只有 5,683 IK14，丢弃了大量未进入P3的train-fold
分子。正确清单应为：

\[
\mathrm{P2Allowed}=\{\mathrm{train\ fold,\ real\ spectra\ IK14}\}
\setminus\{\mathrm{all\ P3\ query\ IK14}\}.
\]

并在P2 pair/candidate构建时禁止P3 IK14以query或训练candidate身份出现。这样才能
真正扩大训练覆盖，而不是继续困在旧g8r小池。

### 9.5 其余封存门

- RDKit不可用时必须fail-closed，不能静默输出空scaffold；
- main少于5,000、near-core低于预注册下限时必须fail，而非只打印；
- 本地预期HDF5/pairs哈希应由sbatch自动断言，不依赖人工看日志；
- formal lock前先锁定评价器：positive=max同IK14谱分数、negative=max每个异IK14谱
  分数、严格并列 `rank=1+#(s_neg>=s_pos)`、每query等权、formula-cluster CI；
- P2 allowed清单和各manifest自身也应记录SHA256。

## 10. P3 v3 已实现与本地全量容量审计（2026-08-23）

`tasks/build_g8r_p3_test.py` 已按第9节重写，当前正式协议为：

1. 主查询和参考库均只使用 `SIMULATION_CHALLENGE=False` 的真实实验谱；
2. query 可来自尚未被消耗的 train/val 身份，P2 仍只允许 train 身份，且减去全部
   P3 query IK14；
3. 参考候选库固定为全 HDF5 中 112,075 条真实实验谱，strict 10 ppm、same adduct、
   exclude self，评价时按候选 IK14 取最大谱图分数；
4. `pairs.json` 未覆盖、但在 P3 候选图中可达的同分子式分子对全部补算 MCES；超过
   50 键或解析失败的分子对显式标为 unknown，不得偷归为远结构；
5. 输出 main、isomer、near-core(MCES 0–2)、nearmid(MCES 0–5)、exposed-extension、
   sim-to-real 六个视图，仅 main 是主统计面板；
6. HDF5、pairs.json、排除源、真实参考库、候选图、MCES supplement、评价协议和各
   query manifest 均写入 SHA256；输出目录存在即拒绝覆盖。

本机缺少服务器上的两个 RAW cache，因此以下为“少排除两个缓存”的容量上界审计，
不能替代服务器 formal seal 数字；缓存身份理论上主要来自已排除的 g8r 身份，但仍须由
服务器 dry-run 复核：

| 视图 | 独立 IK14 | 说明 |
|---|---:|---|
| 可用真实 pristine 主查询 | 3,285 | 同时具有同IK14正例和10 ppm异IK14负候选 |
| 真实 pristine 同分子式异构体 | 1,989 | 不能称为 near |
| 真实 pristine MCES 0–2 | 497 | 真正 near 核心 |
| 真实 pristine MCES 0–5 | 664 | near+mid 挑战层 |
| 历史 large-audit 暴露扩展 | 851 | 只能作次级稳健性审计 |
| simulated-query to real-library | 609 | 只能作次级域迁移审计 |

候选图新增发现 2,972 个 `pairs.json` 未覆盖的可达同分子式分子对，其中 2,296 对可按
旧协议补算，676 对因结构规模/解析限制保留 unknown。旧版直接复用 train-only
`pairs.json` 会系统性低估 near/mid。

在严格真实谱协议下，5,000 个独立主查询不可得。继续凑5,000只能放回模拟谱、重复
IK14或放松历史暴露排除，均不可接受。因此 formal 门改为：

- `P3-main-real-pristine = 3,000` 个IK14等权查询；
- `P3-near-core-real-pristine >= 450`；
- 主结果只报告main；near/isomer为重叠困难亚组，不作为第二次独立复现。

本地固定源哈希：

- HDF5: `ccda2c4114d9b21413977df03376ca0fc097956a7fa304b861a3154a2b81e64f`
- pairs.json: `0461e95b7e98d9d7fa0f03c884b7b8b7996e58bbdafecdb95de31e02c6cb0d9a`

`run_g8r_p3_lock.sbatch` 已自动断言上述哈希，CPU-only，不再错误申请GPU。formal lock
前必须先在服务器运行同脚本 `--dry-run`；若真实缓存排除后 main<3000 或 near<450，
禁止降低门槛后悄悄封存，应先回报实际损失来源再做设计裁决。
