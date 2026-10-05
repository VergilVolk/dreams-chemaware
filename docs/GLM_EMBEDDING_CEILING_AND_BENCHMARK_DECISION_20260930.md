# GLM：Embedding 天花板、越改越差的结构成因与基准更换决策

日期：2026-09-30
性质：裁决性分析文档（数字全部引自冻结工件：`GLM_PROJECT_OVERVIEW_20260930.md`、`NOISE_NATIVE_TRIPLET_AND_2PP_CEILING_AUDIT_20260927.md`、`NOISE_ENCODER_P2B_AND_RERANKER_POSITIVE_RESULTS_20260930.md`、`NOISE_FIRST_PRINCIPLES_METHOD_AUDIT_20260930.md`）。

---

## 1. 我们到底在什么数据集上测什么（先把这个钉死）

| 评估面 | 规模 | official Recall@1 | 错误率 | 用途 |
|---|---:|---:|---:|---|
| MassSpecGym corrected held fold-0 | 18,333 queries | 93.1875%（1,249 错） | 6.8pp | encoder 线内部主端点 |
| corrected development graph | 83,619 queries | 92.8760%（5,957 错） | 7.1pp | 开发面；near 28,188/4,036 错 |
| P3-main（封存） | 3,000 | 87.9333% | 12.1pp | 重排器封存面板 |
| **P3-isomer（封存）** | 1,989 | **79.4872%** | **20.5pp** | 异构体难度面板 |
| P3-near-core（封存） | 496 | 48.7903% | **51.2pp** | 最难核（MCES 0–2） |
| role-2 / role-3（化学线） | 1,975 / 1,929 | ≈89.62% / — | 10.4pp / — | 化学线选择/确认 |
| role-4 outer | 16,198 | 从未打开 | — | 封存 |
| **GNPS Gold/Silver 10ppm（自建，9/28 封存，模型盲）** | identity-disjoint 10,995 queries/175,171 对；formula-disjoint 5,261/47,724；52,871 谱 | **从未产出** | — | 外部基准，全部就绪未消费 |

两个必须承认的事实：

1. **"93.19%/94.24%"不是全库检索**。它是 ±10ppm 公式候选图内、molecule-max 聚合后的排名任务——比 MassSpecGym 原始 leaderboard 协议容易。我们从未在本项目里跑过原始全库协议，所以对外不可比。
2. **GNPS 基准建好了但没有任何模型的绝对数字**。official、Stage-1、V1 在两个 GNPS 面板上的 Recall@1 至今未产出。

## 2. "卡在 20 几个 pp" 的定位与结构成因

唯一对得上 20 几 pp 的是 **P3-isomer 面板错误率 20.51pp**（near-core 更是 51.2pp）。所有 encoder 迭代在这个难度段上基本没动过。结构成因有四层，按重要性排序：

1. **信息方向反了**。Noise 动作教的是"扰动后仍是同一分子"的**不变性**；hard positive 教的也是同身份不变性。而异构体判别需要的是**敏感性**——识别区分结构异构体的诊断碎片/强度模式。我们手里全部监督信号都在加强不变性，没有一个字节在教"这个诊断峰区分 A 和 B"。这是最根本的错配，调剂量调不出来。
2. **单向量表示的信息瓶颈**。共享 encoder 必须在看到候选之前把谱图压进一个 1024 维向量。诊断信息是 query-candidate 相关的（"这个 query 的这些 rival 之间该看哪个碎片"），固定向量原则上装不下全部 query 特异的判别位。预训练目标（全局对比几何）进一步平均掉了罕见诊断碎片。
3. **molecule-max + 参考多重性膨胀**。rival 分子参考谱越多，max 分数越被抬高。错误向参考多的 rival 系统性偏移，这不是任何 encoder 训练能直接消除的系统性偏置。
4. **覆盖天花板**。动作监督只覆盖 ~4.6% 训练 query；旧 cohort 的零风险修正天花板 +1.735pp（仅覆盖 24.36% 官方错误）。在 18,333 held 上 +5pp = 917 个净纠正 = 官方错误的 74.6%；从 V1 当前 1,056 错出发也要 724 个净纠正，而 724 ≈ 全部 near 错误数（6,781×10.68%）。**+5pp 等价于把异构体问题基本解光**——恰好是第 1 条说我们没有信息去解的那块。

## 3. 为什么越改越差（逐次记账）

| 迭代 | 对部署端点的净效果 | 为什么 |
|---|---|---|
| Stage-1 | +0.496pp（因果 +1.265） | 真新信息（噪声不变性），唯一干净正果 |
| Stage-2 | 仅 +0.104pp 增量 | 同一池子捡残差，饱和（9/27 审计已判） |
| T1/T3 v1 | +0.556pp（CI 正） | 收益主要来自 molecule-level 关系课程，非动作（剂量 2.81%） |
| v2 | fold-1 三臂全降 | 剂量按动作数，7.29 次/query；被 NO-GO 拦住，held 未花 |
| v3 | fold-1 双臂降（targeted−control +0.215pp） | query 剂量修好，但 97% 更新仍是共同漂移；且从 Stage-1 而非 V1 出发 |

模式很清楚：**每次迭代都在重新包装同一个信息池（不变性监督），包装越复杂，共同漂移/see-saw 的损伤越超过边际信号**——margin 涨、排名跌三次实证。9/27 天花板审计的原话早已判死这条路："more epochs or merely more extremely hard triplets cannot by themselves bridge this gap"。v2/v3/v4 全部属于这句话的实例。

## 4. 为什么 embedding 不如重排器

自家证据：P2b OOF +3.91pp（near +5.83）/ 封存 P3-main +1.07pp；V2 多残差重排 role-3 dev +3.94pp；RRF 探索 +1.77pp——全部比任何 encoder 续训（+0.5pp 量级）大一个数量级。结构原因：

1. **信息接入时机**：重排器在决策时读原始谱证据（碎片存在性、中性丢失、峰支持度）；embedding 在见到候选之前就得承诺一个向量。P2b 的证据正是 encoder"看得见但没装进向量"的那部分信息。
2. **目标对齐**：重排器直接优化局部候选次序（就是部署决策）；encoder 优化全局相似几何（平均情形），对 Top-1 边界尾部几乎无梯度分辨率。
3. **作用范围**：重排器只动近边界候选的分数；encoder 动一次权重全体 query 一起动——see-saw 和 introduced 的根源。分数空间的 trust-region 是平凡的，权重空间的 trust-region 是我们一直做不出来的。
4. **重排器的边界同样要直视**：P2b 在 near-core **−4.23pp**（最难核上有害）；RRF 叠 Stage-1 确认失败（修正高度重叠）。重排器不是免费大增益，是"信息在原始谱证据里"的证明。

推论：**大增益的信息在原始谱证据池里，重排器是现在能接住它的唯一形态**。encoder 线要吃到它，需要的是把判别性证据蒸馏进表示——这是一条未走的路，不是再调一次剂量。

## 5. 基准决策：GNPS 升格为主基准（回答"必须选更大更好的数据集"）

**结论：是，GNPS Gold/Silver 应升格为外部主基准，且基础设施全部就绪，唯一缺口是绝对数字未产出。**

理由：
- 规模：identity-disjoint 10,995 queries（P3-main 的 3.7 倍）+ formula-disjoint 5,261；
- 结构直接命中弱点：identity-disjoint 测泛化（训练身份与查询不相交），formula-disjoint 就是我们自建的异构体面板（同公式 rival）——"20 几 pp"问题在 GNPS 上有独立复制；
- 模型盲、封存、与任何训练折无重叠；52,871 谱、[M+H]+、严格 10ppm，与部署数学一致。

必须同时说明的边界：仅 [M+H]+ 单加合物；pairwise/panel 协议不等于全库 leaderboard；无 MCES 分层（formula-disjoint 是近似代理）；32k MSG/MoNA IK14 重叠被排除，所以对 MoNA 严格外推仍需专门面板。

**立即行动（成本 ~1 GPU 小时，零新代码）**：用现有 encoder+evaluator 产出 official、Stage-1、V1 三个 checkpoint 在两个 GNPS 面板上的绝对 Recall@1/MRR/AUC + 配对差。这一步之后：(a) "20 几 pp 是否跨数据集复制"有答案；(b) 后续任何声明获得外部主基准；(c) V1 的 GNPS 表现直接决定 encoder 线还值不值得投。

**保留分工**：MassSpecGym corrected fold-0 继续做内部优化端点（它有公式折和因果对照基础设施）；GNPS 做对外声明与模型盲复检；两者都过才算数。若要对外可比，另补一次原始 MassSpecGym leaderboard 协议（从未跑过）。

## 6. 对 5pp 目标的最终含义

- 在现有信息池（不变性监督 + 原始谱证据重排）里，**encoder 单路线没有 +5pp 的头寸**：动作通道因果量级 0.2pp、覆盖天花板 1.7pp、化学线 58% 残余错误无化学方向、oracle 满修也只有 +8.25pp（role-2）且现实转化率 +0.3~1pp。
- 大增益若存在，在**原始谱判别证据**里：短期以重排器/混合系统兑现（P2b 证据），长期以判别证据蒸馏进 encoder 兑现（未走的路）。
- 基准换到 GNPS 不会创造增益，但会创造诚实：它把"我们在 18k fold 上反复迭代的收益"与"真实泛化"分开计量。这正是过去两个月最缺的那把尺子。
