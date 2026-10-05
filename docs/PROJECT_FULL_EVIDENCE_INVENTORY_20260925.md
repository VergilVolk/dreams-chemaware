# DreaMS 扩展项目全量证据总账（工作底稿）

**日期：** 2026-09-25  
**用途：** 在撰写论文、PPT 或项目总结之前，先统一项目从数据、评价、算法、解释、平台到生物学应用的完整证据链。  
**状态：** V2 全量审计工作底稿，不是论文结果稿；若与旧文档冲突，以更晚的专项纠错、冻结报告和机器工件为准。V1 曾遗漏开放世界拒答、早期因子挖掘、峰级因果链、Noise 完整失败谱系、跨组学生物学验证和 Codex/PPT 工程资产，现于第18节以后补齐。  

## 0. 总体裁决

本项目已经不再是“给 DreaMS 加一组化学规则”的单点尝试，而是形成了一个围绕可信 LC–MS/MS 代谢物注释的完整研究体系：

1. 建立严格、可复算的候选检索和评价协议，并系统定位 DreaMS 的局部失败边界；
2. 证明粗粒度化学规则不能直接充当分子结构距离标签，但可以成为错误发现、候选证据和表示解释工具；
3. 分别从共享 embedding 微调和 embedding 后候选证据融合两条路径改善检索；
4. 证明静态代谢网络目录、度数和简单传播不能稳定提高外部注释，进而把 BioAware 收敛到真值盲、样本特异、原子反应事件的严格路线；
5. 建成可处理真实非靶向数据、保留置信度与证据等级的注释平台，并提供可演示的网页界面；
6. 在 MTBLS13729 和 LCNEC 公共队列中完成算法到生物学的应用闭环，同时保留标准品、独立复现和因果实验尚未完成的边界。

项目目前最强的算法结果不是一个可以相加的“总提升”，而是三个不同层级的结果：

| 层级 | 当前最强结果 | 证据资格 | 必须同时说明的边界 |
|---|---:|---|---|
| 共享 DreaMS embedding 微调 | ChemAware native triplet：role-3 开发图 Recall@1 `+1.8144 pp`，57/22，公式簇 CI `[+0.8155,+2.8703] pp` | E2，直接更新权重 | role-3 已被开发使用；化学语义对普通 hard-negative continuation 的独立贡献尚未闭合；无 outer/external 结论 |
| embedding 后候选融合 | P2b：封存 P3-main Recall@1 `+1.07 pp`，89/57，CI `[+0.24,+1.89] pp` | E3 | P3 near-core 为 `-4.23 pp`，不能宣称解决极近异构体 |
| 生物上下文 | 尚无可迁移正增益 | 冻结负结果链 | B44 外部 `-0.56 pp`，B45 拓扑 `+0.23 pp` 且不显著，B46 双上下文 `-0.77 pp`；B47 原子事件仍未完成结果裁决 |

因此，当前最准确的项目定位是：

> 以 DreaMS 为谱图表示基础，通过严格错误图谱、困难负例课程、局部谱学证据、可解释化学概念和样本特异生化事件，构建性能可验证、证据可追溯、风险可控制并能够在证据不足时拒答的 LC–MS/MS 代谢物注释方法。

## 1. 证据分级与统一语言

| 等级 | 含义 | 当前代表 |
|---|---|---|
| E0 | 来源未绑定或尚未运行，禁止作为结果 | Peak-token `+0.842/+1.001 pp`；BioAware U3-v3；Noise native triplet；ChemAware specific replay 的未运行部分 |
| E1 | 历史、探索性或协议后来被纠正 | 旧 23,876-query atlas；Noise E4-A `+0.635 pp`；早期 BioAware 开发增益 |
| E2 | 协议明确的内部开发/确认证据 | corrected atlas；ChemAware `+1.8144 pp`；A1 多标准多通道结果；BioAware B44–B47 诊断链 |
| E3 | 预冻结内部封存评价 | P2b P3-main 与 P3 near-core |
| E4 | 独立外部或同法标准品确认 | 当前没有正向算法条目；B44 是外部负结果，真实候选仍待同法标准品终证 |

所有结果还必须注明其角色：

- **W：shared weight**，改变统一 DreaMS 权重；
- **X：post-embedding expert**，冻结 embedding 后重排候选；
- **A：action/oracle/headroom**，看过动作后测量上限；
- **D：development**，允许模型选择但不是最终盲测；
- **B：sealed/blind**，冻结后一次性评价。

禁止把 A 当 W、X 当 W、D 当 B，也禁止把不同协议、不同分母的百分点相加。

## 2. 数据与评价基础设施

### 2.1 大规模数据资产

- `annotated01` 汇总 3,263,214 张谱、76,157 个 IK14，中位每分子 33 张谱；它是规则统计和困难样本发现池，不是独立主评测集，因为缺少部分采集条件字段且与 MassSpecGym 高度重叠。
- 统一真实参考谱库包含正离子 265,011 张、负离子 29,564 张，约 207,787 个唯一 IK14；同时剔除前体信息不一致的 MassSpecGym 和 GNPS 记录。
- corrected development graph 包含 83,619 queries、9,854 identities、6,220 formulas、392,229 candidate molecules 和 6,220,661 query–candidate spectrum edges。

### 2.2 当前正式候选协议

- query-centred strict 10 ppm；
- same-adduct；
- 候选按分子身份聚合，同一分子的参考谱取最大分数；
- positive molecule first 仅用于保存，不允许利用候选顺序解除并列；
- ties 对正例不利；
- 训练、选择和置信区间优先按 formula/identity/source 聚类隔离；
- 同时报告 Recall@1/3/5/10/20、MRR、macro/micro AUROC/AUPRC、near 子集、corrected/introduced、风险净收益和 preservation。

### 2.3 官方 DreaMS 基线

冻结 checkpoint：`official_embedding_slim.pt`，SHA256 `8928f908606c0bd652c5a4107d3c35102f660622958c225a1f625abe4b1ba245`。

| 代表任务 | Query 数 | 官方 Recall@1 | 解释 |
|---|---:|---:|---|
| corrected full graph | 83,619 | 92.876% | 当前最大开发分母；5,957 个 Top-1 错误 |
| corrected near subset | 28,188 | 85.682% | 4,036 个错误 |
| P3-main | 3,000 | 87.933% | 封存主面板 |
| P3-isomer | 1,989 | 79.487% | 异构体压力面板 |
| P3-near-core | 496 | 48.790% | MCES 0–2 极近结构压力面板 |
| MTBLS1905 已知身份面板 | 36 | 75.000% | 外部小面板；Top-5 94.44% |
| MetDNA3 HILIC 开发面板 | 117 | 81.197% | BioAware 历史开发面板 |

corrected full graph 还给出 macro-query AUROC/AUPRC `0.968687/0.959622`，micro-candidate AUROC/AUPRC `0.956287/0.864155`，MassSpecGym-derived 10-ppm pooled pairwise AUROC `0.847501`。这些不是 DreaMS 论文 NIST20 的原样复现。

### 2.4 评价体系本身的成果

项目先后发现并纠正了：

- 随机负样本显著抬高 AUROC；
- 早期训练池实际只有 516 个分子；
- 同分子定义、candidate canonicalization、reference multiplicity、tie policy 和 dropout train/eval 可改变结论；
- 旧 23,876-query cohort 来自对 `SIMULATION_CHALLENGE=False` 的错误语义解释，不能继续承担正式主张；
- 真实应用中必须先按前体质量构造候选图，再在窗口内排序，不能先做全库 Top-k 再过滤质量。

最后一项修复在相同 `neg_rp` 试点中把可信谱从 929 提高到 1,023，覆盖率由 13.43% 提高到 14.79%，新增 94 张谱；这是协议修复，不是模型提升。

## 3. DreaMS 错误空间：已经知道什么，仍缺什么

### 3.1 当前正式结论

- corrected graph 的总体错误分母已经可靠冻结：5,957 个 Top-1 错误，其中 near 子集 4,036 个。
- 极近结构是明确困难区：P3-near-core 官方 Recall@1 仅 48.79%。
- 同身份谱图的 acquisition mismatch 会降低相似度：跨仪器/未知仪器平均余弦 0.6896，同仪器 0.7695；碰撞能差大于 10 时平均 0.7013，小于等于 10 时为 0.8576。
- 峰级因果审计支持两个可复现方向：不同分子共享的高强度峰可造成错误聚合；条件特异峰可造成同分子谱图分离。

### 3.2 旧 atlas 只保留为机制假设词表

旧 23,876-query 图曾把 1,805 个错误拆为 positive deficit、negative excess、shared-major-peak、neutral-loss convergence、cross-condition 和 RAW-rescue 等表型。该 cohort 后来因语义错误失去正式资格。这些名称可以指导 corrected graph 的重新审计，但旧计数不得作为当前错误机制比例。

因此，现在可以说“错误并非均匀随机，集中于近结构边界、跨采集条件正例和峰证据混淆”，但不能说 corrected graph 的 5,957 个错误已经完成正式因果分类。

## 4. 化学规则与可解释性：从错误的标签角色转向证据角色

### 4.1 规则库资产

- 335 条核心可解释规则：214 NL、102 CF、8 ISO、9 HR、1 NR、1 EE；
- 3,151 条 MassBank 记录衍生经验模式；
- 合计 3,486 条可用于统计和审计的规则条目。

平台注释时只把 335 条核心规则视为主要语义证据；3,151 条扩展记录模式不能包装为人工验证机制规则。

### 4.2 规则不能直接替代结构标签

在 24,333 个分子对上，规则重叠与 MCES-style 结构距离的相关性较弱：核心规则 Pearson/Spearman 约 `-0.167/-0.122`，全规则约 `-0.194/-0.144`。规则可以检测某些错误风险，全规则不同分子 error-detection AUC 约 0.647，但不足以定义全局 embedding 距离或正负标签。

这一负结果决定了后续 ChemAware 的方向：化学信息不再直接规定“两个分子应相距多远”，而只用于选择有信息的困难边界、提供候选证据和解释峰。

### 4.3 双重映射的正结果

| 解释任务 | 结果 | 严格含义 |
|---|---:|---|
| 266 个谱图碎裂概念 | test macro-AUPRC 0.659，基线 0.200；254/266 达到各自基线至少 2 倍 | 官方 DreaMS 全局 embedding 存在可线性读取的实验碎裂概念方向 |
| 469 个数据驱动局部结构环境 | macro-AUPRC 0.240，基线 0.0447；396/469 达到至少 2 倍 | 分子平均 embedding 中存在超出规则库的局部结构方向 |
| 结构环境—谱图概念桥 | 175 个复现且方向一致的候选桥 | 建立结构环境—embedding 方向—谱图质量概念的候选联系 |
| 峰删除忠实性 | 2 个候选的定向删峰可推动对应 embedding 方向 | 峰是表示方向的真实输入来源；尚未证明这些方向控制 Top-1 检索 |

MIL、注意力权重直接解释和单一 peak-token reranker 都没有形成可靠性能增益。未绑定来源的 Peak-token `+0.842/+1.001 pp` 必须永久保持 E0，除非找到 canonical report。

## 5. Noise：针对峰级错误的共享 embedding 路线

### 5.1 历史正结果与其降级

旧 cohort 上 E4-A 在 5 formula folds × 3 seeds 的平均结果为 overall `+0.635 pp`、near `+0.522 pp`，证明峰扰动课程能够更新统一 query/reference encoder。但旧 cohort 后来失去当前正式资格；更关键的是，同预算因果对照显示 targeted 相对 matched random 仅 `+0.03377 pp`，CI 下界为 0，约 94% 的表面增益可由普通 clean continuation 解释。

因此 E4-A 是重要的工程里程碑和历史证据，不再是当前主性能结果。

### 5.2 动作矩阵的真实价值

candidate-gradient、role-confounder、role-shared、role-unmatched 和逐步峰干预建立了错误—动作矩阵，回答“哪些峰级变化可能修正哪些局部排序”。这些动作的 oracle/headroom 曾达到数个百分点，但 headroom 不是模型成绩；它的科学价值是提供有方向的困难样本和课程候选。

### 5.3 当前路线

Noise 已重置为原生 DreaMS triplet：七来源 32,114 条 qualified action 只用于选择 hard negative / curriculum，训练仍使用同身份 positive、异身份 negative 和原生 cosine triplet hinge。冻结合同包含 113,796 events / 28,449 steps 的大规模课程设计，并要求 targeted 必须胜过 matched shuffled。

**当前状态：合同与接口已冻结，但 corrected graph 上尚无正式训练结果。** ChemAware 的 `+1.8144 pp` 只能证明这套原生运行时可行，不能代替 Noise 结果。

## 6. embedding 后谱学重排器

### 6.1 RAW-v1：重要的开发—测试落差

- 开发：`+4.35 pp`，44/17；
- 原协议 Test-A：`+0.45 pp`，59/50，CI 跨 0；
- Test-B：`+0.73 pp`，77/64，CI 跨 0。

RAW-v1 证明局部原始谱学特征有候选排序信号，也证明开发增益不能直接外推。它是历史基线，不是 embedding 改进。

### 6.2 P2b：当前最可靠的封存下游增益

冻结分数为 `0.1*DreaMS + 0.1*entropy + 0.8*neutral-loss sqrt cosine`；普通 sqrt cosine 的冻结权重为 0。

| 面板 | DreaMS | P2b | 变化 | corrected/introduced |
|---|---:|---:|---:|---:|
| formula-OOF 开发，n=5,037 | 86.06% | 89.97% | `+3.91 pp` | 280/83 |
| sealed P3-main，n=3,000 | 87.93% | 89.00% | `+1.07 pp` | 89/57 |
| P3 near-core，n=496 | 48.79% | 44.56% | `-4.23 pp` | 20/41 |

P2b 的增益主要来自中性丢失证据；它证明 DreaMS 全局表征与局部峰/中性丢失证据互补。极近结构退化是硬边界，使用 MCES 真值做回退只能作为 consumed-P3 机制诊断，不能作为可部署安全门。

### 6.3 化学候选侧重排器

在同一 1,929-query role-3 开发图上：

- canonical truth-blind candidate policy：`+3.1104 pp`，64/4，CI `[+2.2199,+4.0683] pp`；
- multi-null symmetric residual V2：`+3.9399 pp`，93/17，CI `[+2.8191,+5.1921] pp`；相对 candidate-rotated control `+3.3178 pp`。

这些结果证明候选条件化化学证据具有强边界识别能力，但仍是已消费开发图上的 reranking，不是共享 embedding 或 outer 结果。

## 7. ChemAware：从直接注入失败到原生 triplet 成功

### 7.1 被否定的旧范式

历史路线包括注意力注入、规则 Jaccard 距离监督、候选 residual 蒸馏、动作梯度 transfer、ChemBERTa/MolFormer 跨模态教师、learned PSD 特征和多视角 probe。决定性负结果包括：

- action-gradient transfer 未胜过 candidate-swapped / peak-permuted 控制；
- correct 与 candidate-swapped 在 16/24 动作上选择相同 reference pair，peak-permuted 为 19/24，说明 pair selection 会丢失化学差异；
- rule-selected boundary loss 下降约 63%，但 Recall@1 不变；
- direct prior 在 50,397 queries 中仅 258 active，增益约 `+0.004 pp`；
- 普通 continuation 的增益可与若干“化学臂”相同。

这些失败确立了一个核心原则：候选特权化学信息不能被当作连续目标强塞进候选无关的共享 embedding。

### 7.2 中间正资产

- `rule_mass` 显式共享核：1,929-query inner 开发图 `+1.2442 pp`，39/15，公式簇 CI 为正；它没有更新 DreaMS 权重。
- shared-Gram 几何审计：自由逐谱投影在 2,048-query 图上 85/3，净 `+4.0039 pp`，correct residual 能量解释 95.30%；它是 oracle 几何上限，不是模型性能。

这两项结果排除了“化学信号完全不能进入共享几何”的悲观解释，并把真正瓶颈定位到干净谱图可观测性、训练目标错位和课程设计。

### 7.3 当前最强 learned shared-embedding 结果

化学信息只选择 hard negative，训练完全复用 DreaMS 原生 `ContrastiveSpectraDataset`、共享 encoder、cosine triplet margin 和 Adam：

```text
chemical evidence -> select informative wrong candidate
clean query + same-identity positive + different-identity negative
-> native DreaMS triplet hinge
```

训练池包含 7,688 query-negative molecule events、4,032 anchors、2,518 formulas、49,980 positive reference edges、55,663 negative reference edges、6,658 action-hard events 和 2,295 correct-vs-multinull specific-hard events。

在完整 role-3 开发图上：

- Recall@1 `0.904614 -> 0.922758`，`+1.8144 pp`；
- MRR `+1.0789 pp`；micro-AUC `+0.8079 pp`；macro-AUC `+0.9342 pp`；
- corrected/introduced `57/22`，风险净收益 13；
- best 出现在 step 3,000，约 1.56 epochs，与 official embedding 平均 cosine 0.7212；
- 训练到 step 412,000 后只剩 `+0.4147 pp` 且 CI 跨 0，与 official cosine 降到 0.3057，证明过度训练破坏全局几何。

### 7.4 后续 refinement 的结果与状态

- reference-aligned bank 将 active events 从 3,046 提高到 5,212；这是课程质量审计，不是性能结果。
- pair-expanded bank 在服务器产生 9,957 events、7,346 active events、1,516 chemical spectrum events。role-2 step 2,000 相对 confirmed stage-1 仅 `+0.759 pp`，38/23，CI 跨 0，Recall@3 下降，风险净收益 -8；没有 checkpoint 通过门，因此 role-3 和 null 未打开。
- specific-replay 把问题收敛为“低剂量、反事实特异化学 residual + 安全 replay”，选择每 query 最多 2 个化学事件、LR `3e-6`、250–1,000 steps。2026-09-24 的本地 dose audit 选择 cap=2；截至本底稿，没有正式性能结果。

### 7.5 当前主张边界

可以说 ChemAware 已经得到目前最强的 learned shared-embedding 开发结果；不能说全部 `+1.8144 pp` 已归因于正确化学语义，不能说通过 outer/external，也不能把 candidate reranker 的 `+3.94 pp` 与 embedding 的 `+1.81 pp` 相加。

## 8. BioAware：静态目录失败后转向样本特异事件

### 8.1 历史开发增益的真实来源

B30/B37 曾在已打开开发域上达到约 `+5.93 pp`，但主要来自候选是否进入知识目录及目录中心度。随后严格对照否定了它的可迁移性：

| 实验 | 结果 | 裁决 |
|---|---:|---|
| B44 独立 MassBank，n=893 | `-0.56 pp`，8/13 | 静态 catalogue prior 跨库反转 |
| B45 coverage-neutral topology，n=860 | `+0.2326 pp`，4/2，CI 跨 0 | 匹配覆盖和度数后，拓扑没有独立材料效应 |
| B46 real transform + coabundance，n=1,426 | `-0.7714 pp`，10/21，公式 CI 上界低于 0 | 聚合变换与共丰度不是可迁移的候选特异事件证据 |

这些是高价值负结果：它们关闭了静态目录 membership、degree/topology、简单 path/diffusion 和聚合 context ledger 的继续调参路线。

### 8.2 B47 外部真值盲基础设施

- 冻结 51,976 queries、10,578 candidate identities、2,078,709 query-reference edges；
- 18 个工件、638,372,014 bytes 完成哈希闭环；
- U0 显示 molecule-max 与 expected-single 在 both-unique query 中造成 22.24% Top-1 身份翻转，两个来源分别 22.16% 和 22.39%，证明 reference multiplicity 是重大混杂；
- U1 小剂量谱学 unary 在 corrected graph 上 `+0.0682 pp`，CI 为正，但 131/74、风险净收益 -17 且一折反向，故 STOPPED；
- U1b v1 的 45/0 来自候选顺序并列泄漏，撤销；v2 为 0/0 no-op；
- U1c seed identity 漏斗为 `4300 -> 4300 -> 379 -> 238 -> 127 -> 127`，瓶颈发生在 Rhea 覆盖/非 currency 安全层；
- U2 在 238 个谱学共识身份中得到 Rhea-covered 129、Rhea-safe 127、Rhea+strict-KEGG provisional 141、宽松 edge headroom 149，另有 89 个身份无目录边；127 个安全身份可产生 3,243 个 sample-collapsed events。

U3-v1/v2 因 null 和反应变换定义错误被判废。U3-v3 已冻结原始 reaction-slot 非氢元素变换签名、degree-preserving rewiring、跨样本完整 seed-event 置换和 wrong-identity 非方向诊断，但尚未运行正式外部事件产量。因此 BioAware 当前拥有严格的 prospective 基础设施和清晰负结果链，尚无新的性能正结果。

## 9. 反向代谢组学与标准锚定探索

### 9.1 已关闭的宽泛主张

“把标准谱输入再去公共样本中搜索”已被 MASST、StructureMASST、reverse metabolomics、microbeMASST 和 RDD 覆盖，不能作为本项目新算法。A0–A2 公共语境模型主要学习技术可检测性和数据库覆盖偏差，已撤回主线资格。

### 9.2 多标准关系坐标

- A1 显式 fragment/neutral-loss/coverage 多通道固定融合，在 153 张参考谱、69 个独立 IK14 上相对 direct-multichannel NDCG@5 `+2.962 pp`，CI `[+1.317,+4.767] pp`；
- A1b 把同一思想换成 official DreaMS 单 cosine 后为 `-0.590 pp`，CI 跨 0，两个离子面板均下降，停止进入样本级。

结论是：第三方标准响应在显式多通道谱学空间中存在关系信息，但没有迁移到 official single-cosine 表达。

### 9.3 可保留的未来方向

真正值得验证的是“主动化学消歧”：在真实候选歧义图上，以最少标准品、加标或定向再采集实验最大化身份和机制结论的消歧。该方向目前只有方法合同，没有结果。

## 10. 注释平台与展示系统

### 10.1 端到端能力

平台已经覆盖：

- mzML/HDF5/MGF 等输入与 DreaMS embedding；
- strict mass-window 候选图和候选分子排序；
- FDR/decoy、分数校准和 Schymanski 分层；
- 规则证据、暗物质聚类与候选 lead；
- MS1–MS2 连接、冻结 EIC 重定量、差异分析接口和通路接口；
- 逐 query 候选、分数、干预、证据等级与审计记录。

平台不会自动输出需要同法标准品的 Level 1，也不把精确质量/库谱候选直接包装为确认结构。真实试点 confident 注释覆盖约 5.9%，约 94.1% 谱作为暗物质进入聚类和 lead 流程。

### 10.2 网页展示

远端 `feature/showspace` 分支已包含 Gradio 应用、Windows 桌面启动/停止/健康检查、回归测试和示例谱图。支持 MGF、mzML、mzXML、HDF5、JSON 上传，多谱图浏览、stick plot、官方 1024 维 DreaMS embedding、MassSpecGym 候选检索、结构去重、SMILES/Formula/RDKit 属性展示及 JSON/CSV 导出。

当前网页使用官方 DreaMS checkpoint；启发式规则只作为提示，不是结构鉴定置信度，也尚未部署最新 ChemAware checkpoint。

## 11. MTBLS13729：深度证据校准的 CRC 应用

### 11.1 原文与本项目不可直接相除的分母

原文从 9,766 个 feature 中注释 345 个，原始注释率 3.53%；在 6,054 个带 MS2 的 feature 中为 5.70%，其中 Level 1 为 157、Level 2 为 188。

本项目在 16,953 个 RPLC 重定量 target 上：

- 官方 DreaMS 分配 3,417 个候选，20.16%；
- 实验 E6 3,426 个，20.21%；
- 冻结 P2b 3,588 个，21.16%；
- 三路候选并集 3,599 个，21.23%。

这只能称候选覆盖扩展，不能与原文含标准品的注释率相除后宣称准确率提高六倍。E6 只增加 9 个候选但 Level2a-supported 增加 22；P2b 增加 171 个候选但强证据减少 24，说明“覆盖”和“证据质量”不是同一终点。

### 11.2 生物学结果的演进

早期发现包括修饰鸟苷样模块、长链酰基肉碱和乙酰化多胺平行轴。严格再审计后，当前主轴收敛为 mucinous-relative free Neu5Ac pool expansion 与 donor/destination decoupling：

- positive-RP feature 703 与 source Level-1 Neu5Ac 构成强同队列正交桥；
- Rmu–RN 10/10 正向，锁定 targeted-EIC 平均 `+1.935 log2`；
- Rmu-vs-Rtu interaction 在两种归一化下约 `+2.21/+2.14 log2`，五模块 BH q 约 `0.0018/0.0016`；
- 同患者 free Neu5Ac 升幅显著高于 CMP-Neu5Ac 和 UDP-GlcNAc，差值 `+1.693/+1.922 log2`，Holm p 均 0.0273；
- 外部 O-glycomics、患者级 raw-UMI、TCGA 和空间/糖肽数据支持选择性的 hybrid mucin glycome，而不是全局 hypersialylation；
- 两个 mono-O-acetyl-Neu5Ac-like 精确质量峰没有随 free Neu5Ac 升高，是保留的重要负结果。

修饰鸟苷、长链酰基肉碱和多胺轴仍是有效发现级候选，但位置异构体、双键位置和精确身份没有标准品终证。BioAware 在该队列中的直接贡献主要是离子家族折叠、网络冲突弃权和证据分层，不是提高身份率。

### 11.3 结论边界

MTBLS13729 已形成高质量的 evidence-calibrated discovery 包，但 full 13,155-target exact-FDR10 为 0；仍缺同法 authentic standard、spike-in、独立 Rmu 代谢组、同一样本 glycan destination、同位素 tracing、酶扰动和 rescue。

## 12. LCNEC：当前更强的主生物学论文候选

34 对 LCNEC 肿瘤/邻近组织的公开原始数据拥有 pooled-QC MS2 和跨平台来源信息。表型盲冻结流程从 81 个 dark modules 得到 22 个多证据 feature、21 个 connectivity hypotheses、12 个跨平台同方向复现和 4 个作者表外优先候选：

| 候选 | 34 对平均 log2FC | 同向患者 | q | 当前等级 |
|---|---:|---:|---:|---|
| ADP family | +2.400 | 33/34 | `3.02e-12` | connectivity-family hypothesis |
| ADP-ribose family | +1.556 | 31/34 | `1.47e-7` | connectivity-family hypothesis |
| ascorbate | +5.407 | 32/34 | `2.92e-7` | compound Level-2 hypothesis |
| quinolinate | +2.047 | 28/34 | `3.17e-6` | compound Level-2 hypothesis |

四者均通过 exact-mass、直接碎片、患者内配对、跨归一化和去冗余审计，但没有 authentic-standard RT，因此精确新代谢物声明数仍为 0。允许的生物学结论是 phosphorylated-nucleotide/NAD-related pool redistribution 与 antioxidant-pool remodeling；不允许写 ATP energy charge、通量、酶活或因果依赖。

BioAware 对 ADP 的 881 条 Rhea reactions 主动识别为 currency hub 并弃权；对 ADP-ribose、ascorbate 和 quinolinate只提供 context anchors，不修改谱学身份。独立 107 例 LCNEC 蛋白组提供 QPRT/NAD 与 redox 方向背景，但不是代谢物身份或通量复现。

## 13. 项目最重要的负结果与工程纠错

这些内容不能从最终故事中删去，因为它们决定当前方法为何可信：

1. 规则 Jaccard 不能当结构距离；
2. MIL/注意力聚合没有超过简单基线；
3. 早期随机负样本和 516 分子训练池造成虚高；
4. 旧 23,876-query cohort 失去正式资格；
5. Noise E4-A 的定向动作独立归因未通过；
6. RAW-v1 开发增益未在冻结测试中显著泛化；
7. P2b 在 near-core 明确有害；
8. ChemAware 多条直接注入/蒸馏路线只拟合 loss、不改善排序；
9. ChemAware native triplet 的过度训练会破坏全局几何；
10. BioAware 静态目录增益在 MassBank 外部反转；
11. B47 U1b 的 45/0 来自 tie leakage，已撤销；
12. reverse context A0–A2 主要学习技术可检测性，已撤回创新主张；
13. 真实应用中的候选覆盖、证据等级、结构准确率和生物学发现必须分开；
14. 多个看似合理的 MTBLS13729 候选在 targeted-EIC、竞争身份或外部复核中被降级；
15. LCNEC 四个优先候选仍然只是 Level-2/connectivity hypotheses。

## 14. 完整科研故事的推荐主线

### 第一幕：定义真正问题

非靶向代谢组学的瓶颈不是“给每张谱一个名字”，而是在开放候选世界中输出证据可追溯、置信度可校准的分层结构结论，并在真结构不在库中或证据不足时可靠拒答。

### 第二幕：建立可信零点并定位错误

DreaMS 已具有很强的总体检索能力，但近结构、跨采集条件和峰证据混淆仍形成系统性失败。项目首先修复候选协议、数据泄漏和评价口径，再建立 corrected graph、P3 以及公式簇统计，而不是从宏观化学先验直接跳到训练。

### 第三幕：否定粗暴化学注入

规则库和结构分析证明“碎裂现象相似”不等于“完整结构相近”；直接注意力注入、规则距离监督、MIL 和候选 residual 蒸馏均暴露粒度或目标错位。负结果把规则重新定位为困难边界选择器和可解释证据。

### 第四幕：形成两条互补算法路线

一条路线改变共享表示：峰级 Noise 课程和 ChemAware 困难负例课程。当前最强结果是 ChemAware 原生 triplet `+1.8144 pp`。另一条路线不强迫所有信息进入单一 embedding，而是在候选阶段融合局部谱峰和中性丢失证据；P2b 在封存主面板获得 `+1.07 pp`，同时揭示 near-core 风险。

### 第五幕：把生物上下文放回正确位置

BioAware 的静态目录与简单传播被外部反证。真正可能成立的创新必须基于同一样本真实观测 seed、候选—seed—reaction 原子事件、离子实体、结构化 null 和风险受控全局 assignment；当前 B47 已建立严谨分母，但结果门尚未开启。

### 第六幕：从算法走向真实发现

注释平台将候选检索、证据等级、FDR、暗物质和定量接通。MTBLS13729 展示深度证据校准和 hybrid mucin glycome；LCNEC 展示 34 对患者、跨平台复现和四个作者表外 Level-2/connectivity 候选。二者共同证明算法价值不只在单一 Recall 数字，也在于扩大可审计候选、过滤错误身份、组织验证优先级并生成可检验生物学假说。

## 15. 当前未完成但必须追踪的决定性实验

1. ChemAware：同预算 official-hard-only 与三组 content-permuted triplet 对照，闭合化学归因；随后建立真正未消费的 outer/external 面板。
2. ChemAware specific replay：运行 cap=2 的低剂量残差课程；只有 role-2 通过完整安全门才允许打开 role-3/null。
3. Noise：在 corrected graph 上运行 native triplet targeted 与 matched-shuffled，不能继承旧 E4 数字。
4. Failure Atlas：在 corrected 83,619-query graph 上重建错误机制分解，旧 1,805-error 表只作假设词表。
5. BioAware：运行 U3-v3 truth-blind atomic-event yield；通过后才允许一次性 outcome 评价。
6. P2b：开发仅依赖推理时可见证据的 near 安全门，并用新封存面板验证。
7. 平台：接入最新 ChemAware checkpoint、校准拒答/候选集合，并建立正式公开部署的资源和许可清单。
8. 生物学：优先做 Neu5Ac 同法标准与 spike-in、modified-guanosine 竞争标准包、LCNEC quinolinate/3-nitrobenzoate 竞争标准；随后才讨论机制升级。
9. 外部验证：建立新的 Level-1、带样本矩阵和采集元数据的独立检索面板，避免继续消费 role-3、P3 或旧 BioAware 开发域。

## 16. 不能遗漏的成果清单

- [x] 官方 checkpoint 和跨任务基线总账；
- [x] candidate protocol、tie、reference multiplicity 和 corrected graph；
- [x] 旧 atlas 的撤销及其仍可保留的机制假设；
- [x] 335+3,151 规则库与规则角色重置；
- [x] embedding—化学概念—峰证据双重映射；
- [x] Noise 动作矩阵、历史 E4 正结果和因果归因失败；
- [x] RAW-v1 与 P2b 的开发/封存/near 三层结果；
- [x] ChemAware candidate reranker、shared kernel、Gram headroom、native triplet 正结果及后续失败/refinement；
- [x] BioAware B30–B47 的开发捷径、外部反转、负结果和 prospective 基础设施；
- [x] reverse metabolomics 查重、A1/A1b 正负对照和主动标准品方向；
- [x] 端到端注释平台、质量窗口修复、统一谱库、暗物质处理与 showspace；
- [x] MTBLS13729 从早期三轴到 Neu5Ac/hybrid mucin glycome 的证据演进；
- [x] LCNEC 81 modules、22 hypotheses、12 跨平台复现和 4 个优先候选；
- [x] 所有关键负结果、泄漏纠错、停止门和仍缺的标准品/外部验证。

## 17. 权威来源顺序

1. 机器工件 `data/validation/**/report.json`、冻结 checkpoint 和 SHA256；
2. `PROJECT_CLAIM_REGISTRY_20260919.md`、`PROJECT_AGENT_EXECUTION_REGISTRY_20260919.md`；
3. 2026-09-21 至 2026-09-24 的 ChemAware/BioAware 专项结果与协议；
4. `SYSTEMATIC_REPOSITORY_REVIEW_20260922.md`；
5. `PROJECT_RESEARCH_MASTER_SUMMARY_20260829.md` 及其后续增补；
6. 更早的阶段报告只用于解释研究演进，不能覆盖后续勘误。

---

## 18. V2补充：课题真正的最高层科学问题

V1把课题压缩成了“提高检索性能”，这不完整。对外科学问题应当是：

> 给定未知小分子的LC–MS/MS谱图及采集条件，能否输出证据可追溯、置信度可校准的分层结构结论；当正确结构不在候选库中、谱学证据不足或多类证据冲突时，能否可靠拒答？

由此得到两种不能混淆的运行模式：

| 模式 | 可用输入 | 目标 | 代表模块 |
|---|---|---|---|
| Spectrum-only | 查询谱图、前体质量、参考谱库或候选结构库 | 在不读取样本表型和代谢网络上下文的条件下完成候选检索、排序、校准与拒答 | official DreaMS、Noise、ChemAware shared embedding、RAW/P2b、规则证据 |
| Sample-aware | 上述输入，加同一样本的离子家族、候选种子和精确反应事件 | 在强谱学unary之上使用样本特异事件修正候选，并在证据不足时回退 | BioAware B47/ECSR、离子实体、exact event、global assignment |

三层验收阶梯也必须补回：

1. 封闭计算真值：统一候选协议、公式/身份隔离、平分处理和聚类置信区间；
2. 盲标准品跨条件验证：同法RT、完整MS/MS、竞争异构体与必要的co-injection；
3. 真实队列：只能报告候选覆盖、证据等级、拒答与待验证发现，不能把没有标准品的候选覆盖写成准确率。

此外还缺一个正式开放世界终点：移除正确候选后测量高置信错误接纳率与候选集合覆盖。现有工作已经为校准和弃权建立基础，但尚未完成统一开放世界基准。

## 19. V2补充：完整时间线与方法收敛

### 19.1 规则去标签化与严格基线阶段

- 从最初127条规则扩展到335条核心记录，并引入3,151条MassBank记录派生模式；随后证明这些主要是观测质量模式，而不是可以直接给分子身份贴标签的机制规则。
- 自建严格10 ppm队列中，原始SSL embedding的pooled AUC/Top-1为`0.615/0.559`，官方微调为`0.799/0.763`。这一历史比较说明项目面对的是强基线上的残余局部错误，而不是修复一个弱模型。
- P0早期错误审计发现错误富集于同/近分子式、高Morgan相似和MCES 0–2区域；但该阶段分母和后续正式图不同，只保留为路线起点。

### 19.2 隐式因子与双重映射阶段

- PCA/varimax前32个方向解释42.3%方差，但`0/32`同时通过结构、谱峰、跨集合稳定与混杂门，否定了“用几个主成分直接命名化学因子”的简单路线。
- precursor SAE获得5个跨种子稳定方向；S0277与芳香甲基环境`cc(c)C`相关，但不足以形成稳定碎裂机制。
- 多层crosscoder的重建能够泛化，但第7层precursor 512/128因子和第6层peak 512因子均为`0`个三种子全门通过，说明“共享稀疏子空间存在”不等于“单一化学因子可识别”。
- 原始—官方微调层差异、同分子条件不变性和身份判别的部分信号可以复现；单独仪器变化效应没有稳定复现。
- 最终正资产不是PCA图，而是266个实验碎裂概念、469个局部结构环境、175条跨层桥和两个通过输入忠实性检验的峰因子。

### 19.3 峰级反事实与下游重排阶段

- discovery/confirmation隔离后，删除全部混淆峰纠正率`28.1%`，匹配随机删峰`7.6%`；删除身份特异峰产生相反方向。
- 冻结8项机制面板一次性测试中，定向删除纠正`10.74%`，匹配随机`4.52%`；公式平衡margin净改善`0.0305`，95% CI `[0.0229,0.0386]`。
- 冻结主干CPU反事实头出现正MRR区间，但Top-1区间下界为0；它证明局部监督可学习，不是正式共享embedding结果。
- RAW-v1与P2b把“峰级证据可能有用”进一步转换为候选排序算法，并暴露开发—封存及main—near的边界。

### 19.4 Noise动作、直接传输与归因清算阶段

- S1a/S1b/S1c、S2、S3A建立单峰、角色峰与顺序干预矩阵；A4完成全峰精确扫描、非线性教师和正例证据路线。
- E1–E11建立动作筛选、梯度兼容、共享几何和headroom链；旧E4-A给出历史正结果，但后续归因和corrected graph推翻了其正式资格。
- V5–V11、Injector V1、Hybrid V1/V2及live-shared V3系统检验了动作是否真正进入共享权重。结果表明动作方向存在，但训练连续效应、更新幅度衰减、候选参考梯度和优化器状态可以吞没语义残差。
- 2026-09-06图谱纠错把主分母从错误的23,876-query图切换到83,619-query corrected graph；此后所有旧动作计数必须降为历史。
- 最终Noise与ChemAware在同一范式会师：知识不再回归连续分数或直接注入峰，而只选择hard negative；优化仍使用DreaMS原生身份triplet。

### 19.5 ChemAware范式重置与原生triplet突破阶段

- 旧335/3,151规则被重新定义为观测记录；unified_v2曾错误丢弃119,029张谱，已勘误。
- KL、PMT、candidate residual、attention/gradient injection等路线分别暴露标签语义、候选特权信息、共享Gram几何和干净谱可观测性问题。
- rule-mass显式共享核证明部署可见的干净谱关系有用；shared-Gram审计证明化学target在几何上可容纳，但自由逐谱oracle不等于可学习函数。
- candidate-side策略取得强开发增益；随后ChemAware只用化学证据组织原生hard-negative课程，获得`+1.8144 pp`直接权重结果。
- reference-aligned、pair-expanded、residual stage2与specific replay继续研究覆盖、参考谱对齐和低剂量化学特异性；截至本总账日期，尚无新结果超过已封存的role-3 best。

### 19.6 BioAware从内部大增益到外部否定阶段

- 早期MetDNA3/NetID式候选网络、候选证据账本、typed listwise、rotation和context adapter完成工程闭环。
- B30/B37内部约`+5.93 pp`后来被识别为catalogue membership/degree和跨来源失败历史信号，而不是反应机制。
- B38–B40逐步否定显式路径摘要、六类原子动作和四跳扩散；B44–B46完成外部反转、覆盖中性拓扑null和双上下文负结果。
- B47转向两套真实外部数据、真值盲候选图、reference multiplicity审计、强谱学unary、Rhea安全种子和原子事件产量门。

### 19.7 平台与真实生物学阶段

- 候选质量顺序修复、统一谱库、FDR/校准、Schymanski分级、暗物质聚类、MS1–MS2连接、患者配对定量和网页演示形成可交付系统。
- MTBLS13729经历C20:4、修饰鸟苷、多胺和Neu5Ac多轴审计，最终把主现象收敛为free-Neu5Ac pool expansion与donor/destination decoupling，同时保留多个平行轴和反证。
- LCNEC从34对公开组织、263个统一feature family、81个dark modules出发，形成4个优先Level-2/连接性家族候选，并通过独立蛋白组和外部转录组做机制背景三角化。

## 20. V2补充：Noise完整版本谱系，不能只写E4

| 阶段 | 做了什么 | 结果或失败 | 保留的科学认识 |
|---|---|---|---|
| S1a/S1b/S1c | 单峰和角色峰干预 | 建立纠正/新增错误矩阵及动作差异 | 峰角色不同，不能把随机删峰当全部噪声基线 |
| S2/S3A | 顺序多步干预 | candidate-gradient逐步累积净收益；shared强删峰持续制造错误 | 干预必须小步、重算排序并保留原身份信息 |
| A4/C1 | 全峰精确扫描、动作教师和cross-fit | 教师可发现方向，但不能自动转为部署动作 | oracle/action headroom与可学习权重严格分开 |
| E4-A历史 | 5 folds×3 seeds | overall约`+0.635 pp`、near`+0.522 pp` | 曾证明训练栈可移动共享表示；当前因旧cohort失去正式资格 |
| E4-A归因 | targeted vs matched-random/common continuation | targeted额外仅约`+0.0338 pp`，CI触及0；约94%增益来自continuation | 不能把全部E4增益归因于定向噪声 |
| dynamic N+P | 四臂Phase A | 不胜matched-random/static target；第二seed门关闭 | dynamic-direct kernel停止 |
| V5/V6 | best-action direct | 动作回收率过高；真实优化器占比约5–7% | 行数、身份覆盖和优化器剂量必须分账 |
| V7 | corrective restoration | 相对E8 `+0.1309 pp`，67/43，risk-net `-19`；低于shuffled | 有微弱信号但不安全，参考侧梯度主导 |
| V8/V9 | query-local/functional direct | 未建立新的合格增益 | query局部化仍不能替代完整安全门 |
| V10 | safe exact direct | 计算后checkpoint写入受磁盘耗尽影响 | 需要磁盘预检、原子发布和结果恢复 |
| V11 | historical champion replay | 未恢复历史收益 | “E4形状的loss”不等于完整E4科学流 |
| Hybrid V1 | 七源动作+Injector V1 | 相对official约`-0.0927 pp`，更新绝对幅度缩水约96% | 动作可区分，但传输幅度丢失 |
| Hybrid V2 | 精确0.25残差剂量、30,496步/臂 | targeted相对E8`+0.0927 pp`，低于matched shuffled`0.0273 pp`；risk-net负 | 精确传输仍不足以证明语义增量 |
| live-shared V3 | 恢复完整live E4目标 | 首轮实现把精确0.25改成上限并放松传输门，结果作废 | 属实现错误，不是新的科学阴性 |
| native triplet | 32,114条七源动作只选hard negative | 113,796 events、28,449 steps；合同冻结、未出正式结果 | 当前唯一合法Noise后继；必须targeted对matched-shuffled |

Noise真正完成的贡献包括：动作矩阵、因果峰证据、完整失败总账、精确注入器审计、optimizer-space归因、corrected graph重置及原生triplet合同。不能只用一个已降级的`+0.635 pp`代表整条路线。

## 21. V2补充：ChemAware完整成果层级

### 21.1 规则库与语义纠错

- 335条核心记录实际由214个neutral loss、102个fixed fragment、8个isotope、9个hydrogen rearrangement及少量NR/EE构成；其中316条可读，但可执行母体结构前提为0。
- 3,151条MassBank派生项全部support=1；其中79条所谓NL实为前体m/z与exact mass偏移，3,072条CF来自每条记录最低三峰。它们只可用于覆盖、冲突、QC和不确定性，不能称3,151条机制规则。
- 正式G0历史缓存覆盖25,275张谱、3,472个身份和3,486维观测向量；它绑定旧图，只能复现历史诊断。

### 21.2 可以保留的中间正结果

| 资产 | 结果 | 正确资格 |
|---|---:|---|
| error-conditioned spectrum-only adapter | `+0.3629 pp` | 训练骨架开发结果，不是化学归因 |
| rule-kernel Phase A共同continuation | `+0.4666 pp` | none/mass/rule-mass相近，说明公共训练有效 |
| 旧ICEBERG direct proof | 180-query，`+0.5556 pp`，1/0 | 小样本proof-of-effect，CI下界0 |
| ICEBERG E1动作 | 约`+0.7519 pp` | 候选条件动作开发证据 |
| mass显式核 | `+0.8813 pp` | 干净谱显式共享核开发结果 |
| rule-mass显式核 | `+1.2442 pp`，32/8或重算39/15；绝对和对mass/shifted-rule区间为正 | 没更新DreaMS权重；新576-query表示未稳定复现 |
| strict ICEBERG residual headroom | dose 0.25/0.50/1.0为`+2.7344/+4.9805/+7.6172 pp` | oracle-routed候选分数上限，不是模型 |
| shared-Gram oracle | 2,048-query，85/3，`+4.0039 pp`，95.30% residual能量可解释 | 证明几何可容纳，不证明干净谱函数可学 |
| canonical candidate policy | `+3.1104 pp`，64/4 | candidate-side强开发结果 |
| multi-null V2 | `+3.9399 pp`，93/17 | 当前最强candidate-side开发结果，非embedding |
| native triplet best | `+1.8144 pp`，57/22，CI严格为正 | 当前最强learned shared-embedding开发结果 |

### 21.3 失败结果为何重要

- A2 action-transfer的24动作审计中，candidate-swapped与正确动作有16/24选择同一参考pair，peak-permuted为19/24；单位更新归一化后化学区别结构性消失。
- direct prior在50,397 query中只有258个active，增益约`+0.004 pp`，说明稀疏先验覆盖不足。
- 新576-query observation screen中mass约`+0.3472 pp`且区间跨0，learned 120-channel observation embedding为`-0.5208 pp`，说明旧rule-mass结果对规则定义和panel敏感。
- 长程native training从best step 3,000退化到last step 412,000；Recall@1增益降到`+0.4147 pp`且CI跨0，embedding与official cosine由0.7212降到0.3057。

### 21.4 2026-09-22至09-24的最新后续

- reference-aligned把active events从3,046增至5,212，解决训练positive与部署molecule-max reference不对齐问题；这是容量审计和训练合同，不是新的性能结果。
- pair-expanded服务器池达到9,957 events、7,346 active、1,516 chemical events。role-2 step 2,000为`+0.759 pp`、38/23，但CI跨0、Recall@3下降、risk-net为`-8`，因此没有checkpoint通过，role-3和null未打开。
- specific replay根据本地dose audit选择cap=2、学习率`3e-6`、250–1,000步低剂量课程；截至当前只有协议和剂量审计，没有可声称的新模型结果。

## 22. V2补充：BioAware全阶段，而不是只写B44–B47

| 阶段 | 实际工作 | 裁决 |
|---|---|---|
| v1工程 | Rhea/KEGG映射、候选网络、保守gate、审计账本 | 工程与可解释性闭环成立；MTBLS13729为0/1，未提高准确率 |
| MTBLS1905小面板 | 36谱/18目标外部已知身份 | evaluation-only为1/0但CI下界0；deployable为0变化 |
| MetDNA3开发 | HILIC候选、MS1预映射、递归headroom、候选路径、SMN/RT与candidate-edge审计 | 建立完整证据图；多数是headroom，不是最终准确率 |
| B30 | 六来源nested LOSO catalogue-risk路由 | `+5.930 pp`，54/3；真实内部工程阳性，但机制不是reaction reasoning |
| B36/B37 | reaction与catalogue拆解 | catalogue贡献约`+5.58 pp`，reaction约`+1.40 pp`且CI跨0；主要修正来自目录覆盖差异 |
| B38 | 显式路径与事件摘要 | 不胜简单topology | 丰富路径摘要不自动带来机制性 |
| B39 | 六类一步Rhea原子动作 | 均未超过topology | 原子动作缺少样本和谱学可观测性 |
| B40 | 四跳扩散和局部传播 | 显著破坏收益 | 简单多跳传播停止 |
| B42/B43 | 独立catalogue/topology开发 | 部分子集出现强增益 | 后被B44/B45证明依赖coverage/degree捷径 |
| B44 | MassBank独立外部 | `-0.56 pp`，8/13 | 外部反转，static catalogue不可部署 |
| B45 | coverage/degree-neutral topology | `+0.2326 pp`，4/2，CI跨0，不胜三个matched permutation | 独立拓扑没有实质作用 |
| B46 | reaction transform+coabundance | `-0.7714 pp`，10/21，公式CI上界<0 | 聚合双上下文有害 |
| B47-B1/U0 | 两来源真值盲候选图与工件注册 | 51,976 queries、10,578 identities、2,078,709 edges；max聚合导致22.24% unique-top1翻转 | reference multiplicity是重大混杂 |
| U1/U1b | formula-OOF spectrum unary/safe veto | U1`+0.0682 pp`但131/74、risk-net`-17`；U1b 45/0为tie leakage，修正后0/0 | unary停止，不得用于B47 seeds |
| U1c/U2 | seed漏斗与目录覆盖 | 4,300→4,300→379→238→127→127；127 Rhea-safe对应3,243 events | 主瓶颈是Rhea覆盖/枢纽，不是sample collapse |
| U3 | truth-blind exact atomic-event yield | v1/v2逻辑错误判废；v3合同ready未运行 | 尚无event性能或BioAware增益 |

BioAware的重要成果不仅是“失败”：它识别并排除了catalogue membership、degree、path/diffusion、reference multiplicity、tie ordering和aggregate context等捷径，建立了进入真正sample-aware算法前必须满足的event provenance与结构化null标准。

## 23. V2补充：注释平台、软件与网页部署

### 23.1 真实注释管线模块

1. 谱图预处理和官方DreaMS embedding；
2. 先按前体质量建立候选图，再在窗口内按embedding相似度排序；
3. target–decoy q-value/FDR与Platt/isotonic校准；
4. Schymanski/MSI式分级，自动输出封顶在Level 2a/3，绝不自动产生Level 1；
5. 335核心观测规则作为诊断证据，不把规则命中当身份概率；
6. 暗物质谱聚类、候选lead与未知簇管理；
7. MS1–MS2连接、患者内配对差异和通路接口；
8. 每个中间产物、参数、候选、rank、弃权和哈希落盘。

### 23.2 已完成的关键工程结果

- 修复“全谱库先Top-k、再质量过滤”的错误顺序后，同一neg_rp smoke中可信谱由929增至1,023，覆盖率13.43%增至14.79%，唯一IK14由87增至89。
- 统一谱库正离子265,011张、负离子29,564张，约207,787个唯一IK14；剔除619条MassSpecGym和129,929条GNPS前体信息不一致记录。
- MTBLS13729实际处理neg_rp 374,232张、pos_rp 419,676张MS2；进入统一注释协议的MS1-linked谱分别为24,846和86,646张。
- 真实试点confident覆盖约5.9%，约94.1%保留为暗物质，未包装成结构鉴定。
- 13,770个Top-1中，规则证据仅翻转1个Schymanski等级，直接证明“命中规则”不能替代置信度。

### 23.3 网页演示分支

远端`feature/showspace`包含Gradio应用、桌面工作流、回归测试和健康检查，支持MGF、mzML、mzXML、HDF5和JSON输入，使用官方DreaMS embedding与MassSpecGym候选，展示结构、分数和导出结果。它是可演示产品资产，但当前尚未接入最新ChemAware native-triplet checkpoint，也不能宣称已经部署BioAware事件模型。

## 24. V2补充：MTBLS13729完整生物学证据链

### 24.1 原始队列与作者基线

- 240个mzML，4个panel×60样本；没有pooled QC、blank和外部参考。
- 作者原生注释345/9,766 features=`3.53%`，或345/6,054个MS2-bearing features=`5.70%`；其中Level 1为157、Level 2为188。
- 当前16,953个RPLC重定量targets中，official DreaMS/E6/P2b候选覆盖为20.16%/20.21%/21.16%，三路稳定共识为12.75%。这些是不同分析宇宙上的覆盖，不能声称比作者准确率提高数倍。
- E6只比official多9个候选，却多22个Level2a-supported；P2b多171个候选，却少24个强证据候选，揭示“覆盖扩张”和“证据质量”不是同一终点。

### 24.2 早期三轴及其边界

- modified-guanosine模块：1597/7489与3019/8481完成加合物去冗余；10对Rmu/RN为10/10同向，raw `+2.953 log2`，PQN `+2.852–2.860`，精确sign-flip `p=0.001953`；随机双家族模块背景经验单侧`p=0.000708`，但随机面板覆盖70.6%未达75%门。
- acetylated-polyamine：f1717保留为N1,N8-diacetylspermidine-like或acetylated-polyamine family，未排除位置异构体。
- long-chain acylcarnitine：f3222为C20:4 acylcarnitine-like，Rmu/RN约`+1.228`至`+1.760 log2`，但亚型交互不显著，独立MTBLS8090未支持普适CRC外推。
- 三轴之间相关性不足，支持平行代谢轴，而不是强行拼成单一通路。

### 24.3 后续收敛的Neu5Ac主现象

- f703与source Level-1 negative-HILIC Neu5Ac在59个共同样本相关`0.959`；positive-RP仍需同法标准品升级。
- Rmu-RN为10/10同向，targeted EIC约`+1.935 log2`；亚型交互约`+2.209/+2.142`，q约`0.0018/0.0016`。
- free Neu5Ac相对CMP-Neu5Ac和UDP-GlcNAc的差异为`+1.693/+1.922`，Holm p=`0.0273`，支持pool-to-donor decoupling。
- 外部O-glycomics两例MUC显示core-2和sialyl-Lewis X/A扩张、alpha2-6下降；不支持“全局高唾液酸化”。
- O-acetyl-Neu5Ac-like、NXPE1、MUC2 carrier、蛋白组和单细胞审计提供来源/载体/细胞组成边界，多项没有通过正式门。最终语言必须是“hybrid mucin glycome重塑假说”，不能写成通量或酶活已证实。

### 24.4 当前标准品优先级

1. f703 Neu5Ac同法RT/MS2与spike-in；
2. f1597/f3019的m7G、m2G、Gm、m2²G及可得二甲基鸟苷竞争标准；
3. f1717 acetylated-polyamine异构体；
4. f3222 C20:4/C18/C16 acylcarnitine组合。

## 25. V2补充：LCNEC完整算法—生物学闭环

### 25.1 同一分析宇宙上的注释漏斗

- 34对肿瘤/癌旁样本；独立重建263个QC/blank/dilution-qualified HSST3n precursor-RT families。
- source table重叠42/263=`16.0%`；official DreaMS候选覆盖158/263=`60.1%`；DreaMS-P2b top一致136/263=`51.7%`；完整high/moderate证据保留66/263=`25.1%`。
- 42个source-matched正对照中，DreaMS有候选38，完整证据保留31；221个source-table-absent families中DreaMS有候选120，完整证据保留35，最终9个通过生物学稳健筛选、4个优先。
- 可唯一解析的19个source正对照中，official DreaMS和完整工具均为17/19结构一致=`89.5%`；完整证据门没有修复两个同分子式异构体错误，诚实暴露当前性能边界。

### 25.2 四个优先候选

| 候选 | 患者内效应 | 当前身份 | 独立背景 |
|---|---:|---|---|
| ADP family | `+2.400 log2`，33/34 | connectivity family | currency hub，BioAware主动弃权 |
| ADP-ribose family | `+1.556 log2`，31/34 | connectivity family | 独立PARP1/2上升，支持PARP-turnover背景，不确认异构体/通量 |
| ascorbate | `+5.407 log2`，32/34 | MSI Level-2 hypothesis | redox蛋白呈混合补偿；需ascorbate与D-glucuronolactone标准 |
| quinolinate | `+2.047 log2`，28/34 | MSI Level-2 hypothesis | QPRT/HAAO/IDO1等下降、NMNAT3上升，支持利用受限/重分配假说 |

### 25.3 跨组学验证与负边界

- 独立蛋白组实际为103对，其中80对pure、23对combined；预注册22蛋白中18个可测、13个过主门。
- PARP1 `+1.319 log2`、PARP2 `+0.868`；QPRT `-0.853`、HAAO `-1.103`、IDO1 `-1.063`。这些提供通路背景，不是代谢物复制。
- pure LCNEC redox轴均值方向混合，但46个蛋白对中12对通过患者级协变门，全部位于redox轴，支持异质性补偿重塑。
- 外部George队列的基因组事件分层中，NAD、ADP-ribose和redox三个预冻结轴均通过；该队列没有匹配癌旁和代谢物，只能提供表达背景。
- 四优先候选6个患者级两两关系为`0/6`通过；技术混杂16项为`0/16`通过；cotinine/age/sex/BMI/stage敏感性也没有候选通过联合门。
- 精确新代谢物声明仍为0；正确创新是“从暗物质中形成证据校准、可验证的候选与机制假说”，不是未经标准品确认的化学新发现。

## 26. V2补充：Reverse、标准锚定与主动验证

- Reverse metabolomics检索本身已有成熟工作，不能作为项目新范式。
- A0–A2最初的context joint model没有超过技术基线，后被重置为可检测性偏倚和主动标准品选择问题。
- A1在153 spectra、69 identities的显式多通道表示中NDCG@5提高`+2.962 pp`，CI `[+1.317,+4.767]`；A1b切回official single-cosine后为`-0.590 pp`且CI跨0，路线停止。
- 可以保留的未来价值是：利用参考锚定峰算子选择最有信息量的标准品、碰撞能和验证顺序，而不是重复开发通用reverse search。

## 27. V2补充：复现、防错、Agent与PPT工程资产

### 27.1 科研复现基础设施

- checkpoint、HDF5、候选图、参考谱库、缓存、panel和source manifest均逐步引入SHA256绑定；缺行、行顺序不一致、候选无正例和哈希漂移均fail-closed。
- 每个query保存候选数、rank、Top-1、动作、margin、corrected/introduced；真实无真值应用只允许`retained/changed/abstained`。
- 公式簇bootstrap、identity cluster、McNemar、near/mid/main分层和Bonferroni门用于阻止重复谱伪显著。
- 这些合同真实抓住过516分子训练池、A/B面板重叠、candidate ordering、dropout train/eval混杂、tie leakage、reference multiplicity、磁盘写入失败和服务器/本地工件不同步。

### 27.2 Agent化科研组织

- 项目已经形成claim registry、agent execution registry、validation candidate board和integrated evidence adjudication route。
- 每条结论都有允许/禁止措辞、数据、候选协议、split、baseline、artifact、hash、证据等级和图位；`PASS/complete`不自动等于科学主张通过。
- Agent可以并行提出、执行和反驳假设，但所有结论必须回到统一裁判协议；这是项目方法论资产，而非只是一组聊天记录。

### 27.3 汇报与可视化工程

- `.codex_tmp/report2*`保留了从44页清华模板审计、布局JSON、38页重建PPT到montage检查的完整流水线；`deliverables/`中有0906/0911 V7/V8汇报版本。
- 该流水线曾执行字体、字号、色卡、图表和退出模块审计，但尚未正式归档为项目资产；旧PPT中的数字仍必须以本总账和claim registry重新校验。

## 28. V2补充：仓库覆盖与尚未逐项闭合的问题

### 28.1 已完成的覆盖范围

- `docs/`：机器清单共366个文件，其中BioAware 67、biology 60、ChemAware 42、Noise 85、reranker 7、reverse 14、explainability 12、platform 6、其他73；重点全文核查MASTER_SUMMARY、CLAIM_REGISTRY、AGENT_REGISTRY、SYSTEMATIC_REPOSITORY_REVIEW、INTEGRATED_ROUTE、VALIDATION_BOARD，并按主线追溯早期和最新专项报告。
- `tasks/`：机器清单共4,967个文件，其中排除`__pycache__`后可执行源码/SBatch/脚本2,113个；已按Noise、ChemAware、BioAware、biology、reranker、reverse和explainability分类，并核对近期正式入口、单元测试、validator和SBATCH合同。
- `data/validation/`：1,079个一级实验目录；1,066个可递归读取、13个不可读、19个空目录、877个目录至少含一个`report.json`；已核对权威机器报告、近期ChemAware/B47工件及claim registry绑定路径。
- git：核对2026-09-21至09-24的native triplet、reference-aligned、pair-expanded和specific-replay提交；核对远端`feature/showspace`部署分支。

机器清单位置：`data/validation/project_repository_artifact_manifest_20260925/`，包含`docs_manifest.csv`、`tasks_manifest.csv`、`validation_manifest.csv`和`summary.json`；每个文档/脚本均记录相对路径、类别、字节数、修改时间和SHA256，文档另保存标题/状态/一级标题摘要，validation另保存可读性、文件数、report数、formal和status提示。

### 28.2 当前仍存在的物理审计缺口

- 若干临时validation目录在Windows本地显示访问拒绝；它们主要是pytest/localcheck目录，不能在未取得读取权限前声称逐文件内容已核验。
- 部分正式工件只在服务器存在，本地只有报告或哈希；B47曾出现registry不同步，Noise也有服务器侧ledger依赖。
- 约69个validation目录为空，另有少量目录无JSON/CSV结果；它们应登记为中断或空壳，不能被“目录存在”误算为完成实验。
- `tasks/`中大量脚本属于测试、旧版本、smoke和生成器；完整性要求应是“每条科学结论绑定唯一canonical入口和工件”，不是把每个旧脚本都当独立科研成果。

因此，本V2可以称为“主线、正结果、关键负结果、纠错和当前活动合同的完整证据总账”，但不能虚假声称已经人工逐行阅读2,147个脚本或打开所有受限目录。下一步应生成机器可读的逐文件manifest，并把每个文件标为canonical、superseded、supporting、smoke、empty或unreadable，才能实现真正的文件级零遗漏。
