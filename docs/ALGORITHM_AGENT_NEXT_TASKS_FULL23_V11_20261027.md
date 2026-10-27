# 算法 agent 下一阶段任务：冻结交付、v1.1 有界验证、直接竞争比较

下达日期：2026-10-27。依据：最新 `5d17fc12a47246e9e37df3a8b330caee8cc11000`（包含 `d05f555` 有界残差修订）、Full23 v1 发布包及同日竞争研究报告。本文给出下一阶段执行顺序；不改变既有数据轴或历史结果。

## 目标与优先级

P0 是把可运行的 Full23 v1 接到生物任务，并完成已实现 v1.1 的有界实验；二者并行，不让应用等待实验赢家。P1 是补最接近的外部方法与可靠性评估。不要另开 encoder 训练、cosine/哈希搜索或第四套融合器网格。

## A0：版本与完整交付清单

1. 正式算法仍标记 `Unified Additive Full23 v1`；checkpoint SHA 为 `7f1309e2a89ef3640d9b6871a821df6866f2528e629547075d5305ab6707d717`。v1.1 在结果通过预设晋级规则前标记 experimental。
2. 交付 `model_status.json`：版本、commit、checkpoint、特征定义、上游权重、训练来源与已消费评估集、缺失语义、可运行命令。大型上游 encoder 不在小包内是明确依赖，不能把“决策头可运行”写成“原始谱到全部分数已一键完成”。
3. 保留官方微调 DreaMS、Noise V1、P2b、Stage-1、Phase-A、FullMSG、Noise-MSG、V2 名称与 hash。不要把 FullMSG 续训等同于 Noise V1，也不要把 family ablation 写成其中每个模型各自有效。

## A1：完成现有 v1.1，回答两个不同问题

入口：`tasks/run_full23_multireference_readout_1gpu.sbatch`；实现与规则以 `docs/FULL23_V11_MULTIREFERENCE_EMBEDDING_INCREMENT_20261027.md` 为准。

执行最新固定配置：冻结 v1 + `B*tanh(raw/B)` 表示残差；参数匹配 Top-1-reference / Top-3-reference 两臂；reference 置换证伪。复用四套 embedding 缓存；preflight 必须重现已有候选轴与相似度，并通过 `full23_anchored_multireference_readout_v2_bounded` 检查。确认每候选实际 reference 数、排序/去重规则与 mask，不能静默减少为一个 reference。

保存 `run_2419582` 的无界版本及其负结果，禁止覆盖。交代 `fd1575d → d05f555 → 5d17fc1` 的时间线：初版公式和 metadata 为 unbounded，后续加入 Enveda 推导的 B。若声称这是恢复事先规定的精确 bounded contract，必须附更早可核查记录；否则称开发中的有界残差修订。不要把“B 的计算没用 GNPS”扩大为“设计从未受 GNPS 反馈影响”。B、2B 与训练 parent 的保护/可达比例是容量诊断，不是准确率和迁移保证。

交付分开回答：

- **性能问题**：哪一臂满足文档预设晋级条件并超过冻结 v1？保留 identity、formula、near 子集、corrected/introduced、公式簇配对 CI、MRR 和资源成本。
- **机制问题**：Top-3 是否超过参数匹配 Top-1，且收益是否依赖真实 reference 内容？只有胜过 v1 不足以证明多参考机制。

不得看 GNPS 后追加超参数网格、改晋级标准或删失败面板。GNPS 在本轮仍是 consumed development。Enveda OOF parent score 防止 parent stacking 的一种泄漏，并不使残差在同批训练数据上的得分成为独立泛化估计。若均不晋级，冻结 v1，记录实验失败及机制边界；不撤销上游模块已有结果。若晋级，另发版本包和 hash，仍保留 v1 主对照。

产物：`v11_promotion_verdict.json`、逐 query 预测/变化表、两臂/置换报告与一页机制判断。报告应由原始预测生成，不只粘贴训练日志。

## A2：补直接竞争者，先做适配清单再批量跑

最高优先级：**MS2Query、MSBERT、MS2DeepScore 2.0**。保留 WSE、官方微调 DreaMS、Noise 和 Official+WSE learned control；Spec2Vec 是较低成本补充。

`competitor_manifest.json` 至少登记：官方 paper/code URL、commit、checkpoint hash、许可证/下载可用性、预训练库和可确认的重叠、极性/仪器适用性、输入要求、候选生成及聚合方式。不得把拿不到权重的随机初始化模型作为作者方法；遇到资源阻碍报告具体阻碍及可复现替代，不无限重训。

同轴比较要求：

1. 同 query、候选分子、参考谱集合、去重身份定义与折；不准让某方法阈值改变其他方法分母。保留无输出/弃权与失败原因，同时报告全分母和共同可运行子集。
2. 各方法采用其规定的预处理和聚合；适配产生的差异逐项登记。不能为“完全相同预处理”损坏作者方法，也不能给某方法额外真值信息。
3. MS2Query 原生是候选检索加多证据 RF，并非一个简单 cosine。将作者原生流程的结果与固定候选轴上的适配排序分开标记，披露库邻域信息和候选召回上限；不可将裁掉其关键输入的简化版称原方法。
4. 当前 Enveda/GNPS 只用于开发比较。另建 prospective validation manifest：依据仪器、化学覆盖、独立采集及训练重叠条件选择最小必要外部集合，先锁模型/轴/指标，再打开结果。未知的预训练重叠必须标为 unknown；仅来源文件不同不足以保证分子/谱隔离。
5. 保存逐 query 成败交换与候选分数；报告 R@1、MRR、near 难例、公式簇 CI、候选数分层与运行成本。micro-AUC/AUPRC 是辅助，不能替代 ranking。

不用把 SIRIUS、MVP、FLARE 的结构候选任务硬塞同表；只有新增相同任务时才比较。不要花主要算力再穷举 cosine 替换。

## A3：部署可靠性与 V2 证据

先复核已有结果是否已覆盖下列问题，只补缺项：

- 输入缺失触发 whole-query abstention 与“真值不在库”的 no-match 是不同问题。加带已知真值的库外查询/移除真实候选设计及质量相近困难负例；阈值用独立 calibration partition 锁定，评估 false-match rate 与 coverage-risk。合成移除真值只是压力测试，不能替代真实库外外部评价。
- 原始 Additive logit、percentile 和任意 sigmoid 不能直接声称身份概率。必要时做单独校准并登记使用数据；对 atlas 无校准连接就输出 rank/evidence，禁止把负 logit 直接作为图权重或能量。
- V2 仅用实际可用且身份来源合法的候选结构，不读药物名、响应或真值来补 SMILES。对 unknown candidate 采用发布包支持的 unavailable profile，清楚报告其覆盖和适用性能；不能整批假装 real-V2，也不能把 unavailable 默认为高置信无关。
- V2 的组消融已成立，但若论文强调化学一致性机制，检查现有 null 是否排除了缺失/来源模式。补验证预算内的 availability/source-matched 内容错配，区分内容效应与可用性代理；不要因该检查事后修改主模型。
- 同硬件记录 raw preprocessing、四 encoder 编码、缓存、局部评分、V2、head 的时间/峰值显存/内存。分别报告冷启动、缓存后与随 reference 数扩展成本。

## A4：给生物 agent 的接口，立即以 v1 实现

与生物 agent 先对齐 `query_id / candidate_id / reference_id / pair_id / study_id / axis_version`。保留旧固定边重加权和新身份候选实验两个命名空间。旧 22,046 对 identity handoff、最新 41,082 evidence-world 行和 20,485 all-world unique edges 不是可混用的轴。

关键防错：Full23 的 query percentile 等特征要求完整 query 候选块。不能把 mutual Top-1 边各自当一个 singleton query 再跑模型，也不能在看响应后临时删低分候选。新轴由模型无关物理条件生成并冻结。对不同前体的 relation 轨不要直接宣称身份模型已有效。

交付：

- 完整候选宽/长表、所有必需通道与 V2 applicability/status；
- 冻结 v1 输出与各基线输出，query 级弃权、原因及完整覆盖分母；
- 逐实体 retained/changed/gained/lost，按轴版本严格 join；
- 缺上游资产时 fail-fast，失败不悄悄退化为 WSE 或 Official；
- 至少复现一批真实 query 的端到端结果，与 release golden replay 的合成合同检查分开描述。

## 下一次汇报格式

只汇报：已产生的新文件/运行 ID；v1.1 两问题结论；竞争方法覆盖与阻碍；v1 atlas 真实输入/输出覆盖；no-match 与成本仍缺什么。没有原始输出就写未完成，不以脚本提交替代实验完成。所有已消费结果允许用于诚实开发，不重新命名为未开封外测。
