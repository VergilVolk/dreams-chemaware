# 批次 3 摘要（2026-08-29 → 2026-08-31 密集产出期）

来源：`tasks/_history_review/batch_3_files.txt`（52 个文件，全部用 read 工具逐字读完，长文件分段读毕）。

---

## 一、逐文件摘要

MTBLS13729_MODIFIED_GUANOSINE_EXTERNAL_VALIDATION_20260830.md | 2026-08-30 | 结果+审计 | 修饰鸟苷模块在 Rmu 10/10 同向 +2.953 log2（7.74x，精确 p=0.001953，匹配背景 p=1/1413=0.000708 但完整性门 70.6%<75% 故 gate=false）；外部 40 对 OEP00006137 Level-1 队列反向：MSI -0.778 log2（15/20 降，Wilcoxon p=0.0240）、MSS -0.122 不显著 | 本地 PASS（条件性）/ 泛 CRC 主张 FAIL / 改写为"背景依赖" | 把疾病端主叙事从"泛 CRC 标志物"反证式收敛为"黏液型或异构体特异"。

BIOAWARE_DUAL_TRACK_ALGORITHM_PLAN_20260830.md | 2026-08-30 | 计划 | 定义 Track A（候选上下文重排器，受约束残差 S=S0+g·Δ）与 Track B（BioChem-Embed 共享谱图 embedding 微调）双轨及其通过门（corrected−2×introduced>0、CI 下界>0）；复核发现 MTBLS13729 50 候选仅 22 个有一跳证据、Rhea 方向语义全部 canonical_lr_not_physiological | 仅计划（A0/B0/B1 三步未执行完）| BioAware 算法从"一跳加分"转向"残差+微调"双轨的路线图。

MTBLS13729_GSE236696_EPITHELIAL_VALIDATION_20260830.md | 2026-08-30 | 结果 | 6 对黏液型单细胞上皮 pseudobulk：嘌呤轴 6/6 上升 +0.634、FAO 6/6 下降 -0.567（原计分精确 p 均 0.03125，匹配随机轴幅度 p=0.0113/0.00020）；两主终点 Holm 后均 0.0625；TCGA 32 对支持一般 CRC 程序而非黏液型特异（FAO beta=+0.010 p=0.911、嘌呤 -0.232 p=0.0091）| 强方向性复核 PASS / 独立显著复现与黏液型特异 FAIL | 为两条代谢轴提供患者配对、细胞分辨率的方向性正交支撑。

MTBLS13729_BIOAWARE_BIOLOGY_CLOSURE_20260829.md | 2026-08-29 | 结果 | 修饰鸟苷模块闭环：10/10 同向，raw +2.953、三档 PQN +2.852~2.860 log2，全部精确 p=0.001953，LOO 最小 >+2.50；Rmu-Rtu 差 +2.01~2.05（探索性 p=0.0197-0.0234）；与 4966 相关 Pearson r=0.877；BioAware v1 为 0 修正/1 新增（网络不覆盖八候选）| 发现级 PASS / 随机面板 gate=false（70.6%<75%）| 疾病端核心结果：三轴（修饰鸟苷/多胺/酰基肉碱）并列结构的奠基文档。

BIOAWARE_ALGORITHM_AND_BIOLOGY_CONSOLIDATED_REPORT_20260829.md | 2026-08-29 | 综合（结果+审计）| G3-v2 原报 2 修正/0 新增（Recall@1 0.8120→0.8291，+1.71pp）被实现审计推翻：零权重证据计票+嵌套网络证据重复计数，修复后 1/0 再 0/0；生物学端修饰鸟苷模块（+2.85 log2）与 LCAC 类别（+1.382 log2，p=0.00586）为两大主结果 | +1.71pp 被撤回/仅历史；RP/外部测试封存 | 算法线与生物线共用的事实底稿，确立"BioAware 是证据组织层不是身份真值"。

MTBLS13729_FROZEN_BIOLOGY_RESULT_20260829.md | 2026-08-29 | 结果 | 冻结注释面板：neg-RP 62/pos-RP 555 目标，8 个发现候选中 6 过主终点 FDR10、3 过 FDR5；C20:4 feature 3222 三归一化 +1.228~+1.760 log2（q=0.0861-0.0925）；离子家族去冗余把 6 feature 折为 5 中性家族 | 发现级；身份层 A/B/C/D 分层 | 生物学结果的最初冻结版本，确立"表型盲冻结→定向 EIC 重定量"协议。

MTBLS13729_POLYAMINE_AXIS_AUDIT_20260830.md | 2026-08-30 | 审计 | feature 1717：Rmu 9/9 正 +3.009 log2（四种归一化 p=0.00390625）；重审计找到 73 张峰界内 DDA MS2/45 样本（推翻旧"0 accepted link"），m/z 100.0759 为 73/73 基峰；HILIC 跨色谱 rho=0.860、85 注释中 rank 1；MTBLS8090 三项多胺均不显著（BH q≥0.505）| 升级为"最强新候选轴"但保持 N1,N8-diacetylspermidine-like；外部复现 FAIL | 多胺轴从"精确质量命中"升级为"MS2+跨色谱+配对定量"证据链。

MTBLS13729_BIOLOGY_ADVERSARIAL_REAUDIT_20260830.md | 2026-08-30 | 审计（逆向纠偏）| 把跨组学证据裁为三层：稳固（TCGA 一般 CRC 嘌呤 +0.779 z q=3.34e-8、FAO -1.669 z q=1.16e-9）、中等（GSE 方向一致但 Holm=0.0625）、探索（修饰核苷 5/6 p=0.125，留一基因后仅 2-3/6）；逐项纠偏表删除"独立显著复现/黏液型特异"等表述 | 黏液型特异 FAIL / 主结论收敛为"一般 CRC 程序一致+Rmu 幅度待验证" | 防止跨组学方向一致被写成因果或亚型特异的纪律性纠偏。

MTBLS13729_ACETYLATED_POLYAMINE_MECHANISM_AUDIT_20260830.md | 2026-08-30 | 审计 | 多胺机制链复核：GSE236696 六对上皮中多胺乙酰化/分解 +0.690、酸性应答 +1.208、趋化 +2.378（均 6/6、p=0.03125）；多胺乙酰化与酸性应答患者间 rho=0.886（p=0.0333）；GSE236697 单病例空间数据不支持 SAT1-多胺与酸性/趋化区共定位（深度校正后 partial rho 仅 0.010/0.040）| 空间层阴性结果保留；机制链仅假设 | 用空间阴性结果限制"酸性-多胺-免疫"因果叙事的强度。

MTBLS13729_BIOLOGY_CLOSURE_AND_MINIMAL_VALIDATION_20260830.md | 2026-08-30 | 结果+计划 | 独立黏液型 TMT 蛋白组（29 患者，LMC6/LNMC8/RMC7/RNMC8+25 正常）：嘌呤面板 9 蛋白四类肿瘤中位 +0.25~+0.42、FAO 面板 12 蛋白 -0.04~-0.43（LMC/LNMC 与 RMC/RNMC 12/12 同向负）；GSE236696 上皮嘌呤 +0.634（6/6）、FAO -0.567（6/6）；MassBank 审计后标准品采购不得缩水（1597 对 m7G/m2G 最佳 cosine 差仅 0.00450）| 跨组学方向支撑 PASS（池化 TMT 限制）/ 身份终证未完成 | 生物学从代谢组推进到蛋白组+单细胞双正交层，并冻结最小标准品集合。

MTBLS13729_THREE_AXIS_MECHANISM_BENCHMARK_20260830.md | 2026-08-30 | 审计（对标）| 三轴对标机制论文五道门（结构终证/来源通量/候选酶/表型/rescue）：三轴均未过 Level 1，无 tracer/扰动/rescue；临床分层 dMMR vs pMMR 最小 p=0.103、BH q=0.324 不显著 | 全部轴仅"abundance programs"；黏液型特异未确认 | 明确项目与 Nature Metabolism 级机制论文的证据差距与最小升级动作。

MTBLS13729_MUCINOUS_RISK_TRANSCRIPTOMIC_AUDIT_20260830.md | 2026-08-30 | 审计 | GSE281917（140 例黏液型，组织学与平台完全共线故禁止跨系列比较）：MuC23 高风险与 purine rho=-0.498（q=2.37e-9）、多胺乙酰化 +0.262（q=0.00267）相关；六类组成校正后仅 purine 保留（-0.254，q=0.0139）；TCGA 42 例黏液型方向复现但组成校正 CI 跨 0 | 组成独立复现 FAIL；仅"risk-associated bulk transcript state" | 把代谢轴挂到风险分层背景，同时暴露 MuC23 与基质组成（fibroblast rho=0.738）强混杂。

NOISE_ROUTE_VISUAL_ACHIEVEMENT_LEDGER_20260829.md | 2026-08-29 | 结果总账 | 噪声微调：E4-A 共享 embedding 5 fold×3 seed OOF 平均 +0.635 pp overall/+0.522 pp near，15/15 fold CI 下界>0；动作空间上限 3.35~4.93 pp 与模型结果严格分开；candidate-gradient 为稳定正轴、role-shared 大剂量灾难 | 权重级 PASS / 3-5 pp 仅 headroom 非模型 | 噪声路线成果与可视化索引的权威总账，防止上限冒充增益。

NOISE_TRANSFER_BOTTLENECK_AND_NEXT_ROUND_20260830.md | 2026-08-30 | 计划+审计 | 瓶颈定位为 action-to-weight transfer gap（+0.6 vs 3-5 pp）：逐查询 oracle 不是可学习统一分布；E5 日志显示 guided weight=1 时梯度裁剪仅保留 9-19% 范数；E14 teacher 把 formula_fold 错写为 outer_fold 的实现错误已修复并加回归测试 | 停止扫参；E14 重建为 M0/M1/M2 三段式；M2 过门条件 5 条（CI 下界>0、corrected−2×introduced>0 等）| 噪声微调下一轮的因果排序与执行决策。

MTBLS13729_EXTERNAL_MUCINOUS_METABOLOMICS_SEARCH_AUDIT_20260830.md | 2026-08-30 | 审计（系统检索负结果）| OmicsDI 冻结检索 88 条 CRC 代谢组（53 MetaboLights+15 MW+20 其他），逐一审计样本表：无任何独立队列有患者级 mucinous 标签；唯一 mucinous 编码是 MTBLS13729 自身 10 条 Rmu（explicit_mucinous_rows=0）| 未找到可复算队列（非"不存在"证明）| 用数据库级审计关闭"继续下载普通 CRC 队列"的循环，明确黏液型只能靠新队列/标准品升级。

MTBLS13729_CROSSCOHORT_MECHANISM_SYNTHESIS_20260830.md | 2026-08-30 | 综合（结果）| 跨队列机制综合：TCGA 32 对三轴 +0.806/+0.779/-1.669；MTBLS7387 251 对复算 9 个公开 p/q 在 0.5% 相对误差内复现（C20-C24 17 个 FDR<0.05：14 升 3 降；游离 AA 不显著 FDR=0.243）；MassBank 审计 1597 对 m7G/m2G 最佳 sqrt-cosine 0.6712/0.6667 差 0.00450 不可分；本地 5 候选 MS2 全部 30 eV 无碰撞能梯度 | 机制模型="机制支持型解释"非因果；原论文 carnitine 措辞降级为 competing hypotheses | 生物学证据矩阵综合层：正反证成对、代谢物/通路/转录四级分层。

MTBLS13729_CANDIDATE_EVIDENCE_LEDGER_20260830.md | 2026-08-30 | 结果（账本）| 八候选重审计后全部有峰界内 DDA MS2（合计 252 张），推翻"桥接表 0 link=无 MS2"；1717（+3.01，73 MS2）与 3222（59 固定窗 MS2、25 样本强 carnitine motif）为最重要两项；扩展总账升级为 15 节点（含 source 重映射 myristoylcarnitine 0.28 ppm 等、taurine 降级、feature 722 synephrine 被推翻）| 瓶颈重定义为标准品/异构体/统计边界 | 候选证据的统一机器可读账本，确立"算法候选必须受证据校准"范式。

MTBLS13729_ORIGINAL_PAPER_DELTA_AND_NOVELTY_20260830.md | 2026-08-30 | 审计（对账）| 原论文 Table S4 共 345 注释（L1=157/L2=188）已含 carnitine 24 条、purine 13 条、polyamine 6 条，禁止"首次发现"措辞；8 候选与原表严格 m/z+RT 匹配均为 0；真实增量三类：原表缺失的修饰鸟苷两家族、1717 的独立 pos-RP 正交信号、3222 的解释修正；15 候选中 9 个 source 重映射、10 个 source-linked 效应 Spearman rho=0.830（同队列技术一致性）| 新颖性边界冻结 | 防止把原论文已有结论包装成本项目首次发现。

MTBLS13729_PUBLIC_DATA_BIOLOGY_PAPER_BENCHMARK_20260830.md | 2026-08-30 | 计划（对标）| 区分算法方法/公开数据生物学发现/因果机制三类论文门槛；对标 MetDNA3/DeepMet/TidyMass2/NetID 与 AHCY/空间示踪机制论文；本项目已到"原始重处理+source 对账+外部背景"，缺独立同亚型复现、标准品、tracing/扰动 | 定位 B 类（mechanism-supporting discovery），标题禁 causal | 投稿形态与"不能承诺清单"的决策文档。

BIOAWARE_SOTA_EXECUTION_CONTRACT_20260830.md | 2026-08-30 | 合同 | SOTA 声明八条件前置禁令；P2b 现状（OOF +3.91pp、P3-main +1.07pp 但 near-core -4.23pp）；Track A 谱学主专家+BioAware 小残差架构；MoNA 身份隔离面板（1,301 query/4,070 候选分子，官方基线 Recall@1=81.245%/MRR=89.650%）；B1/B2 两类 embedding 分叉；因子图两开发集 0/0 | 禁止 SOTA 声明；工程验证（烟雾测试、梯度上限自动缩至约 0.043）通过 | BioAware 执行纪律与外部封存协议的总合同。

MTBLS13729_PROLINE_SIALIC_CROSSCOHORT_RESULT_20260830.md | 2026-08-30 | 结果 | 三个源身份正交找回：proline 345 +1.299（10/10，rho=0.814）、glutamate 374 +0.715、Neu5Ac 703 +1.975（10/10，跨面板 rho=0.959）；TCGA proline 轴 32/32 上升（q=3.73e-9）但黏液型相对更低（beta=-0.249，q=0.0130）；Neu5Ac 相反：一般 CRC 唾液酸轴下降但黏液型相对富集（synthesis/transport beta=+0.392，q=2.27e-4；secretory-mucin +0.919，q=9.59e-12）| 双轴分化 PASS：一般 CRC 程序 vs 黏液型相对糖组重塑 | 引入第六条轴并确立"双参照系"（肿瘤-正常 vs 亚型间）分析方法论。

MTBLS13729_COMPETING_MECHANISM_TREES_20260830.md | 2026-08-30 | 框架（冻结）| 冻结 5 轴 14 个竞争机制：修饰鸟苷（writer/RNA turnover/释放三分）、多胺（宿主 SAT1/相邻异构体/微生物 biofilm 三分）、LCAC（输入/利用瓶颈/组成三分）、Neu5Ac（donor/transferase/降解三分）、proline（合成 redox/基质二分）；每轴配最短区分实验 | 机器可读总账；防止"丰度→身份→通量→酶→表型"层级偷换 | 机制叙事的纪律工具：所有箭头只能 compatible with，不能实线因果。

BIOAWARE_V3_TYPED_RERANKER_AND_CONTEXT_EMBEDDING_DECISION_20260830.md | 2026-08-30 | 决策记录（多版本演化）| identity-noop 修复使 MTBLS1905 +5.56pp(2/0)→+2.78pp(1/0)；V3 冻结路由器开发集重放 +4.27pp（5/0）但属 deploy replay 非独立证据；内部 RPLC 95 query +3.16pp（5/2，风险净值=1，CI 跨 0）；首个外部面板 BV2cell-HILIC 3/3 修正/新增净 0 → 冻结更窄 V4；再两外部面板 V3 +0.0137/-0.0124、V4 0.0000/1 修正 1 新增，合并净增益为零 → V5 leave-study-out、V6 机制门；V6 前两确认面板 0/0 与 2/1（+0.00685）；tie 漏洞修复使旧 V4 数字作废 | 外部 SOTA 否定（"V3/V4 当前可称 SOTA"已排除）；开发 +4.27pp 不外推；B2 pilot 0/0 | BioAware v3→v6 完整自审演化史：每一步用外部面板否决过度声明。

MTBLS13729_INTEGRATED_BIOLOGY_RESULT_20260830.md | 2026-08-30 | 结果（整合 v2）| 18 节点四模块：多胺-MTA +3.52（10/10）、嘌呤/修饰核苷 +2.36（9/9）、LCAC +1.66（10/10）、大中性氨基酸 +0.94（10/10）；亚型交互 exact-permutation 五模块中仅 Neu5Ac 通过（raw/PQN 差 +2.209/+2.142，BH q=0.00179/0.00162）；1597/3019/3222 严格有效 Rmu 患者仅 2/2/4；18 节点 claim scorecard：1 PRIMARY_SUBTYPE_ANCHOR（703）+9 GENERAL_TUMOUR_SUPPORT 等 | 主轴重排：Neu5Ac=唯一亚型锚，其余为 general tumour 平行程序 | 亚型门裁决后疾病端故事的结构性重组。

MTBLS13729_EXTERNAL_NGLYCOMICS_SUPPLEMENT_AUDIT_20260830.md | 2026-08-30 | 审计 | 2024 CRC MALDI-MSI/N-glycomics（PMC10808565）补充表仅 101 条 m/z-组成目录（19 条含 NeuAc），无患者 ID/组织学/强度列，无法复算任何患者级比较 | 独立复制硬门 FAIL；仅结构背景 | 明确该外部资源只能作为"sialylated destination 存在性"背景。

BIOAWARE_METDNA3_INTEGRATION_CONTRACT_20260830.md | 2026-08-30 | 合同 | 源码审计发现公开 MrnAnnoAlgo3 缺 obj_mrn*.rda/info_mrn*.rda/md_mrn*.rda，不得声称完整复现 MetDNA3；合法路径=作者完整工件精确桥接或 KGMN/MetDNA2 可复现基线；唯一首轮创新=DreaMS edge reliability 四臂（作者 DP/官方/noise-tuned/保守交集），四臂共享全部其余管线 | 复现硬门冻结；M0-M3 评价顺序与停止规则 | BioAware 与成熟网络算法集成的契约，杜绝"改一处宣称改全部"。

MTBLS13729_EXTERNAL_OGLYCOMICS_MUCINOUS_AUDIT_20260831.md | 2026-08-31 | 审计 | 12 对 PGC-LC-MS/MS O-glycomics（T2/T3 为 MUC，n=2 不做检验）：两 MUC 的 core-2 居第 1/2 高（配对 +46.46/+40.13）、sLeX/A 第 2/1（+7.64/+4.49）、α2-6 倒数第 2/1（-50.08/-70.43）、core-3 第 2/1 但配对 -34.01/-34.64；披露 Table S8 将 T6 误标 MUC 的元数据冲突 | independent structural support（n=2 边界）；不能称 replicated/validated | 为 hybrid mucin glycome 提供 n=2 独立结构层："相对保留 vs 绝对损失"两参照系分离的关键证据。

MTBLS13729_NEU5AC_HYBRID_GLYCOME_AUDIT_20260831.md | 2026-08-31 | 综合（裁决）| 三层证据收敛为 hybrid mucin glycome 模型：free Neu5Ac 10/10 +1.935/+1.975 log2（唯一过亚型 BH 门的模块，但全 13,155-target FDR10 未通过）；TCGA 分支审计 donor supply beta=+0.480（q=3.30e-8）、secretory-mucin +0.922（q=5.34e-11）、core-3/Sda +0.879、core-2 composite -0.009（q=0.915）、α2-3 -0.439、ST6GAL1 -0.742、ST6GALNAC1 +0.788 | 反对 global hypersialylation；三种比较（患者内/亚型间/外部配对）不得混写 | Neu5Ac 主线的机制总纲：donor-carrier-core-linkage 解耦模型正式定稿。

MTBLS13729_ALGORITHM_TO_BIOLOGY_INCREMENT_AUDIT_20260831.md | 2026-08-31 | 审计（归属）| 三路真实应用：E6 在 pos-RP 仅 +9 个已分配 feature 但 +21 个 Level2a-supported（保守 evidence stabilizer）；P2b +171 覆盖（约 +4.79pp）但 Level2a 224→201（coverage expander 非高置信增加）；feature703 的官方 DreaMS 身份字段为空、三路 feature 级明细未同步本地，Neu5Ac 身份不能归功 E6/P2b | retained/changed/abstained 口径；归属"not locally verified" | 算法对生物学贡献的诚实归属：主轴身份来自原论文 Level-1 正交找回，算法贡献是统一协议与覆盖。

MTBLS13729_EXTERNAL_NEU5AC_BIOGEOGRAPHY_AUDIT_20260831.md | 2026-08-31 | 审计 | Jain 2024（Molecular Cancer，372 对匹配肿瘤/正常，七亚部位）：官方补充将 Neu5Ac 定为 HILIC(-) Level 1（m/z 308.0980，RT 355.7 s）；正常黏膜 cecum-to-rectum 斜率 +0.349（p<0.001），肿瘤衰减至 +0.088（p=0.091）；mucinous 关键词命中 0；分期表合计 374 与方法 372 不一致已披露 | PASS_CONTEXT（独立 Level-1 身份+疾病依赖空间背景）/ Rmu 丰度复现 FAIL_MISSING | 为本地"10 个 Rmu 全右侧"的组织学-位置混杂提供外部警示与背景。

MTBLS13729_EXTERNAL_NEU5AC_DASH_REPRODUCIBILITY_AUDIT_20260831.md | 2026-08-31 | 审计（复现性）| 官方网页 Dash 端点返回每组织 371 个值（7 亚部位×固定 53）、无 patient/pair ID；网页回归 beta 0.391/0.179 不能复现补充表 0.349/0.088（肿瘤网页 p=5.33e-4 显著 vs 论文 p=0.091 不显著，方向冲突）| 网页值仅方向性可视化；不得进入效应量合并/配对/复现计数 | 外部数据复现性边界：明确"论文补充表才是权威统计来源"。

MTBLS13729_EXTERNAL_SIALYLOME_MUCINOUS_AUDIT_20260831.md | 2026-08-31 | 审计 | 2026 Biology 论文 988 例（980 有 histology）三队列整合：mucinous 进 Sialyl-High 组 55.2%（85/154）vs non-mucinous 28.8%（238/826）；重算 OR=3.043（95% CI 2.142-4.325，Fisher p=6.58e-10）；但 score 仅由 20 个 sialyltransferase 基因构成，且三队列含 TCGA 与本地部分重叠、无分队列报告 | external pooled transcriptomic context（部分独立）；不能称独立代谢物复制 | 为黏液型-唾液酸程序优先级提供最大样本转录背景，同时保留"复合分数掩盖分支方向"的批评。

MTBLS13729_SIALIC_DONOR_DECOUPLING_AUDIT_20260831.md | 2026-08-31 | 结果 | 同患者供体解耦：原始 HILIC(-) 补充表 free Neu5Ac（Level 1）10/10 升 +2.249 log2（约 4.75x，Wilcoxon p=0.00195）；CMP-Neu5Ac（Level 2）仅 +0.556（p=0.275）、UDP-GlcNAc +0.327（p=0.492）；预设患者内差值 free−CMP +1.693、free−UDP +1.922，Holm-Wilcoxon 均为 0.0273、bootstrap 下界均>0 | free-pool-to-activated-donor decoupling PASS（同队列车内分解）| 把 donor 层从 RNA 推测升级为实测代谢物证据，修订机制主图。

MTBLS13729_OACETYL_NEU5AC_LIKE_AUDIT_20260831.md | 2026-08-31 | 审计（阴性）| 表型盲发现 m/z 350.109269 两个色谱分离峰（4.29 min：50/60 样本、47 张 MS2；5.55 min：54/60、56 张；均含强 m/z 87.0088 motif）；Rmu 配对均无稳定升高（OAc-01 完整对仅 4/10、OAc-02 8/10，两者 BH q 均 0.930）；与 free Neu5Ac 患者变化 rho 仅 0.170/-0.067 | 阴性结果；位置异构体（4/7/8/9-O-acetyl）不可分，维持 mono-O-acetyl-Neu5Ac-like | 以负结果强化 donor-carrier-core-linkage 解耦："free pool 上升≠bulk O-acetyl 产物上升"。

MTBLS13729_PXD055865_MUC2_GLYCOPEPTIDE_AUDIT_20260831.md | 2026-08-31 | 审计 | PXD055865（Nature Communications 2026）：3 块黏液癌标本实为 2 位独立患者（Colon1a/1b 同人）；去重 MUC2 糖肽 439/390/451/21（含 Neu5Ac 4/5/1/1、O-acetyl-Neu5Ac 3/5/0/1）；鉴定数≠丰度（661 vs 21 不可比）| PASS_WITH_LIMITATION：carrier/destination 结构背景，非独立复制 | 补上"游离 Neu5Ac 进入什么糖链"层的方向相容外部证据。

MTBLS13729_OACETYL_NEU5AC_STANDARD_RESOURCE_AUDIT_20260831.md | 2026-08-31 | 审计（资源）| 逐资源审计：HMDB 仅预测 QTOF 谱、ChEBI 只有结构元数据、MassBank/MoNA 精确名称/分子式查询无实验记录；文献 B1 离子（mono-OAc m/z 334.1134 等）不能定位乙酰位置，需 IM-MS/CCS | 公共资源不足；维持 mono-O-acetyl-Neu5Ac-like；P0=成对购 4-O+9-O 标准、P1 优先普通 Neu5Ac 标准 | 把标准品采购从"猜"改为资源审计驱动的优先级排序。

MTBLS13729_MECHANISM_READINESS_SCORECARD_20260830.md | 2026-08-30 | 审计（scorecard）| 对标 AHCY/SORD/ATF6/Gut 等 7 项研究逐层打分：本项目完成临床发现+MS1/MS2 重处理+外部正反证（△），缺标准品终证/独立黏液型代谢组/tracing/扰动/rescue（—）；三分支更新：分支解耦与"相对保留 vs 绝对损失"对账通过；PXD055865 载体层 PASS_WITH_LIMITATION、独立性降至 2 患者 | 定位 mechanism-supporting discovery，禁写 causal metabolic reprogramming | 论文层级与每个证据门的权威总裁决（后升级至 v9/v10 完成度总账）。

MTBLS13729_MINIMAL_VALIDATION_AND_REVIEWER_ATTACK_PLAN_20260831.md | 2026-08-31 | 计划 | 五大审稿攻击（异构体身份/小样本选择/free≠糖链/外部 MUC2 过用/通量酶靶）逐项给出当前回答与最小补强；P0=Neu5Ac 同法标准+spike-in、P1=linkage-aware carrier readout、P2=同位素+扰动+rescue；列出"不应继续投入"清单（含无新真值的 BioAware 阈值扫描、用鉴定数做丰度统计）| 投稿三级：discovery→identity/structure-supported→causal | 投稿防御预案与资源分配终审。

MTBLS13729_PXD055865_SOURCE_DATA_QUANTITATIVE_AUDIT_20260831.md | 2026-08-31 | 审计（定量边界）| 8 个补充文件全盘点：MOESM8 仅 258 张 MALDI PNG 无数值矩阵；MOESM1 图 20/22-25 的 normalized level：健康结肠 Ac/Neu5Ac 比值 11.91/44.33/182.17 vs 4 肿瘤区域 0.0094-2.2798，患者间高度异质 | NL 为图示强度标尺≠校准浓度；不能做群体统计/不能称独立丰度复现 | 用图内定量层强化 pool-carrier-modification decoupling，并给出 RAW 文件复算的下一步。

MTBLS13729_SIALIC_POOL_MECHANISM_DISCRIMINATION_20260831.md | 2026-08-31 | 审计（机制判别）| TCGA 四预定义分支：一般 CRC 中 NEU1/NEU3 释放轴 +0.854（28/32，q=9.02e-7）但黏液型相对常规型为 -0.691（q=5.58e-6）→排除"sialidase 转录上调直接释放"；CMP 轴黏液型相对 +0.449 由 SLC35A1（beta=+0.702）驱动而 CMAS 不显著；current-GDC 补齐 NXPE1：黏液型相对 +0.621（p=0.000369）但加入分泌黏蛋白程序后塌缩至 +0.064（p=0.734），50 对一般 CRC 中 47/50 肿瘤更低 | NXPE1=carrier-linked 状态非独立驱动；O-acetylation 修正为"分泌载体相关表达背景" | 对 free Neu5Ac 来源的四个竞争机制做出转录层裁决，收敛到"分泌/周转-利用失配"优先。

MTBLS13729_NXPE1_POOL_CARRIER_OACETYL_MECHANISM_AUDIT_20260831.md | 2026-08-31 | 审计（三层机制定稿）| 最终模型为三层部分解耦系统：pool（free Neu5Ac 10/10 +2.249，CI [1.641,2.866]）/carrier（NXPE1 富集被五基因分泌程序完全吸收；TPM/FPKM 双单位一致；GSE236696 六对上皮 6/6 肿瘤更低，均值 -1.084，p=0.0625；MUC2 在该数据 12 样本 feature index 全缺失记技术不可用）/destination（本地 O-acetyl-like 不升+PXD055865 肿瘤低于健康）| pool-carrier-destination decoupling 成立；NXPE1 蛋白/酶活/底物未测 | Neu5Ac 主线机制模型的英文定稿版：明确"什么已建立/什么仍开放"两张清单。

BIOAWARE_MATURE_NETWORK_FRAMEWORK_PIVOT_20260831.md | 2026-08-31 | 决策（战略转向）| 正式放弃自研完整网络算法：G3-v2"2/0"重置为 0/0 不得复用；主框架改 KGMN/MetDNA2（MetDNA3 缺资产）；修正作者基线脚本错误（is_credential 误关+误开 RT 校准）；冻结 448 配对三元组 outcome-free 校准清单（397 身份/169 式/44 组件，P3 重叠 0）；OEP003284 主线改用作者峰表（15,942/16,760 feature），80/80 Level-1 行在作者峰表内，正式真值键=polarity+peak_name | 预注册主模型=作者 DP×DreaMS 边交集；no-op 必须三表多集等价才启动实验臂；25 项合同测试通过但无外部性能结果 | BioAware 线的重大转折：从造轮子到"成熟框架内只改一处边可靠度"的可审计因果链。

GSE178341_NXPE1_MUCINOUS_PREREGISTRATION_20260831.md | 2026-08-31 | 合同（预注册）| 冻结 GSE178341 分析：6 位纯黏液型 vs 53 位纯普通腺癌（患者 PID 为单位，C130/C171 同患者切块合并）；主基因 NXPE1+12 基因固定面板；主上皮区室+4 个 goblet 谱系+30 细胞最低门；失败/停止规则（条码不一致即停、NXPE1 缺 feature 即报技术不可用、超 2/6 无计数则只报覆盖度）| 仅计划（在看组级结果前冻结）| 为 NXPE1-分泌谱系假设提供患者级独立验证的预注册纪律。

GSE178341_PORTAL_SENSITIVITY_AUDIT_20260831.md | 2026-08-31 | 结果（次级敏感性）| 等待原始计数期间用门户 100,000 细胞子样本（98,438 可与 GEO 精确连接）：NXPE1 广义上皮 +0.0279（p=0.547）、冻结匹配 +0.0444（sign-flip p=0.500）未确认；AGR2 +1.6384（q=0.00480，匹配 6/6 正）、SLC35A1 +0.1361（q=0.03432）通过；1,562 个门户细胞无 GEO 元数据已披露 | NXPE1 独立验证暂不成立 / AGR2+SLC35A1 方向信号 | 次级敏感性先行的工程排障+方向预检，明确真正裁决层是 raw-count pseudobulk。

MTBLS13729_INDEPENDENT_MUCINOUS_PROTEOMICS_PREREGISTRATION_20260831.md | 2026-08-31 | 合同（预注册）| 冻结 Xu et al. 2023 蛋白组审计：15 MC vs 15 AC（16 正常仅定向）；8 个可测固定蛋白（NXPE1/SPDEF/SLC35A1/CASD1 不可用如实报告）；log2 丰度+200,000 置换+8 蛋白 BH；四级决策语言（orthogonal support/protein-specific/directional only/not supported）| 仅计划（看组级结果前冻结）| 蛋白层正交验证的统计契约，杜绝探索性 FC>1.5 选择。

MTBLS13729_INDEPENDENT_MUCINOUS_PROTEOMICS_AUDIT_20260831.md | 2026-08-31 | 结果 | 两个预冻结模块均未过门：secretory/mucin +0.211 z（q=0.449）、sialic 模块 +0.207 z（q=0.449）；AGR2/GNE/NANS 15/15 leave-one-MC-out 方向稳定但 CI 跨零（AGR2 log2 尺度 p=0.161、q=0.643；源文原始尺度 FC=2.60、Welch p=0.0478 的分歧已解释）；TFF3 37/46 值=精确最小值（地板效应）不能作丰度证据 | directional only / not confirmed | 蛋白层给出"方向一致但欠功效"的诚实结果，支持选择性前体/分泌重塑而非全通路激活。

GSE178341_SIALIC_CELL_SOURCE_PREREGISTRATION_20260831.md | 2026-08-31 | 合同（预注册）| 冻结 7 个 compartment-module 终点（Epi/Myeloid × secretory carrier/CMP-Neu5Ac capacity/NEU1-NEU3 release/salvage-catabolism）；log2(CPM+1) 患者 pseudobulk、20,000 bootstrap、100,000 置换、7 终点 BH；支持门=bootstrap 下界>0+q<0.10+6/6 留一同号 | 仅计划 | 细胞来源定位问题的预注册（明确 RNA 不能证明释放/通量/微生物贡献）。

GSE178341_RAW_UMI_MUCINOUS_SIALIC_AUDIT_20260831.md | 2026-08-31 | 结果 | 官方 raw-count HDF5（370,115 细胞，患者为单位）：AGR2 +1.613（CI +0.962~+2.151，q=0.0068）、SLC35A1 +0.833（CI +0.462~+1.212，q=0.0068）双基因复制并通过右侧/MMR 敏感性（q=0.0179）；Epi secretory carrier +0.917（q=0.0627，5/6）与 Epi CMP capacity +0.687（q=0.0627，5/6）两模块过门；NEU1/NEU3 上皮轴 -0.547 无正支持；NXPE1 +0.837 但 p=0.152、q=0.229、匹配 4/6 未过门 | NXPE1 FAIL / AGR2+SLC35A1+两上皮模块 PASS | 独立患者级转录证据把模型收敛为"选择性上皮分泌折叠+Golgi 供体转运能力，非全通路激活、非 sialidase 释放"。

GSE178341_EPITHELIAL_COMPOSITION_DIAGNOSTIC_CONTRACT_20260831.md | 2026-08-31 | 合同（post-result 诊断）| 明示为结果后诊断非新确证终点：检验 AGR2/SLC35A1 信号是否被 goblet 谱系比例解释；三层嵌套模型（组织学/+logit goblet 比例/+右侧与 MMR）；残留状态判定=调整后系数保留正向+幅度≥25%+HC3 CI 不含零 | 仅计划（描述性规则，不升级为预注册主张）| 组成混杂与细胞内状态的拆分合同。

GSE178341_EPITHELIAL_COMPOSITION_DIAGNOSTIC_RESULT_20260831.md | 2026-08-31 | 结果 | goblet 谱系比例：黏液型 0.284 vs 常规 0.152（差 +0.132，CI -0.013~+0.287，p=0.0899；成熟 cE08 仅 +0.0050）；组成调整后 AGR2 +1.172（CI +0.483~+1.861）、SLC35A1 +0.676（CI +0.216~+1.135）保留正区间，MUC2/SPDEF/NXPE1 跨零；goblet 比例与 secretory composite rho=0.900、与 SLC35A1 仅 0.441 | AGR2/SLC35A1 残留状态 present；MUC2/SPDEF/NXPE1 composition-sensitive | 证明黏液型信号=部分组成富集+选择性上皮状态，而非"每个 goblet 细胞全上调"。

MTBLS13729_BIOLOGY_MANUSCRIPT_RESULTS_V2_20260830.md | 2026-08-30 | 结果稿（手稿级）| 论文结果稿六大结果：三路算法覆盖（P2b +171 feature 但 47 gained/370 lost tier）、345→18 候选四类增量、三个 Level-1 正交找回（proline +1.299/glutamate +0.715/Neu5Ac +1.975，均 10/10）、平行 abundance programs（模块协调 rho=0.833 q=0.0775 为边缘）、跨队列拆分（4c 匹配背景：五模块 1.414-2.582 vs 背景 0.064-0.321；4d 亚型门仅 Neu5Ac q=0.00162-0.00179；4e 372 对 PASS_CONTEXT；5b/5c 分支+raw-UMI）、终点重构（interaction 无 FDR10 feature、全 13,155-target 无 FDR10）| mechanism-supporting discovery 定稿口径 | 投稿主文结果的集大成版本，所有子审计在此汇成叙事。

MTBLS13729_BIOLOGY_MANUSCRIPT_BLUEPRINT_20260830.md | 2026-08-30 | 计划（蓝图）| 论文蓝图多轮演化：主轴从"修饰鸟苷单轴"→四程序→最终"Neu5Ac-mucin 主轴+平行 general 程序"（亚型门后重排）；Figure 1-5+Extended Data 设计；投稿三级（A 无湿实验=algorithm-enabled clinical discovery / B=structure-supported / C=causal）；完成清单（标准品/独立队列/示踪扰动三缺）；第 19-20 节冻结 v3 主图四 panel 与 25 门完成度总账 | 定位冻结：禁 drives/flux/enzyme activation/therapeutic target 措辞 | 手稿结构与完成门的总蓝图，把 30+ 份审计收敛为一份可执行的投稿计划。

---

## 二、批次总评

这一批 52 份文档覆盖 2026-08-29 至 08-31 的三天密集产出，呈现四条清晰主线及其收敛方式：

**1. 疾病端（MTBLS13729）经历了"发现→扩张→反证→重排"的完整循环。** 08-29 的修饰鸟苷模块（10/10、+2.95 log2、p=0.001953）是起点；08-30 上半天快速扩张为三轴→四模块→六轴（加入 proline/glutamate/Neu5Ac 正交找回），同日的外部 Level-1 反证（MSI -0.778 log2）、TCGA 亚型模型（黏液型嘌呤反而更低 beta=-0.232）和逆向审计连续压缩主张；08-30 深夜的亚型交互门做出结构性裁决——**只有 Neu5Ac 通过跨归一化 BH 校准（q=0.0016-0.0018）**，主轴从修饰鸟苷重排为 Neu5Ac/mucin-glycan，其余降为 general tumour 平行程序。08-31 全天围绕 Neu5Ac 深挖：donor 解耦（free-CMP 差 +1.693、Holm p=0.0273）、O-acetyl 阴性（q=0.930）、NXPE1 被分泌程序吸收（+0.621→+0.064）、GSE178341 raw-UMI 独立复制（AGR2/SLC35A1 q=0.0068 通过、NXPE1 p=0.152 失败）与组成诊断（AGR2/SLC35A1 残留状态保留），最终冻结为 **"mucinous-relative hybrid mucin glycome with donor–carrier–core–linkage decoupling"**——一个明确反对 global hypersialylation 的三层解耦模型。

**2. 算法端（BioAware）在同期完成了两次诚实降级和一次战略转向。** G3-v2 的 +1.71pp 因零权重计票与嵌套证据重复计数被撤回（0/0）；v3 路由器虽在开发集 +4.27pp、内部 RPLC +3.16pp，但首个外部面板即归零（3/3），随后 V4/V5/V6 连续用外部面板否决 SOTA 声明（两面板合并净增益为零）；08-31 的 pivot 文档正式放弃自研完整网络算法，转向"复现 KGMN/MetDNA2 + 只改 DreaMS 边可靠度"的可审计路线，并把 OEP003284 外测合同冻结到"作者峰表+预注册交集主模型"的精度。整个演化由 identity-noop 修复、tie 漏洞修复、formula_fold 错写修复等工程审计驱动，体现了"每次性能声明都要能被自己的审计推翻"的纪律。

**3. Noise 微调线确定了真实的 transfer gap。** E4-A 的 +0.635pp（15/15 CI 下界>0）是唯一可写为权重级的结果，与 3.35-4.93pp 的动作上限之间约 4pp 的差距被诊断为 action-to-weight transfer gap 而非超参问题；E14 的实现错误（formula_fold 错写）被捕获并修复，下一轮改为 M0/M1/M2 三段式，停止一切大扫参。

**4. 方法论纪律贯穿全线。** 本批文档反复执行同一套动作：预注册（GSE178341 两个合同、蛋白组合同）、负结果保留（MTBLS8090、sialidase 假设、O-acetyl 峰、蛋白模块 q=0.449）、外部资源可用性审计（N-glycomics 无患者列、Dash 网页不可复现、MetDNA3 缺资产、O-acetyl 标准谱库无记录）、层级偷换防范（竞争机制树 5 轴 14 机制）、新颖性对账（原论文 345 注释逐项核对，禁止"首次发现"）。最终所有线索汇总进手册结果稿 V2 与蓝图：项目定位冻结为 **"algorithm-enabled, evidence-calibrated clinical discovery"**，25 门完成度总账明确列出已通过（配对终点、Neu5Ac 亚型发现、raw-data 复核）、带限制通过（无 QC 归一化、匹配背景）、明确未通过（全空间 exact-FDR10、独立患者级代谢物复制、tracing/perturbation/rescue）和明确负结果（BioAware v1 无注释增益）——一个知道自己在证据阶梯哪一级的项目。

数字密度极高但口径始终统一：主效应用患者内配对+精确检验，亚型用 interaction 单独分栏，全空间 FDR 与候选面板 FDR 分栏报告，外部证据按 direct metabolite/orthogonal transcript/risk context/adversarial 四级标注。这批文档共同的最大价值，是把"我们能说什么"收敛得和"我们发现了什么"一样精确。
