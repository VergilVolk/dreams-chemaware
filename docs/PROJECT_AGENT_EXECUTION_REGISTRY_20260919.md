# DreaMS 项目 Agent 执行登记

日期：2026-09-19  
上位路线：`docs/PROJECT_INTEGRATED_EVIDENCE_ADJUDICATION_ROUTE_20260919.md`  
用途：记录每个 agent 的具体任务、输入、交付物、通过门、停止门和论文图位。任务变更必须保留旧条目，不得静默覆盖。

## 1. 当前 Phase 0：只读事实冻结

### A0-1 资产与主张审计

- **负责人**：asset-claim agent
- **状态**：`COMPLETE_READ_ONLY`（2026-09-19）
- **任务**：核对 failure atlas、Peak-token、RAW、A1/A1b、P2b、E4-A 的数字、协议和当前有效性。
- **输入**：`docs/`、相应 `data/validation/` 报告及冻结清单。
- **交付物**：每项资产的证据路径、数据/候选/split/baseline、效应量、corrected/introduced、证据等级、能否同图比较。
- **通过门**：每个拟进入 Fig.1/2 的数字均有唯一冻结来源；所有不可比较项被显式标出。
- **停止门**：缺 manifest、基线身份、候选规则或 tie policy 的结果不得进入定量主图。
- **图位**：Fig.1、Fig.2、补充资产表。

### A0-2 Noise/ChemAware 执行合同审计

- **负责人**：method-execution agent
- **状态**：`COMPLETE_READ_ONLY`（2026-09-19）
- **任务**：确认 Noise V3 与 ChemAware outer 的唯一入口、输入哈希、缺失产物、预注册门和停止规则。
- **交付物**：两张 run card；不得运行训练或开启 outer。
- **通过门**：运行命令、输入、输出、seal、恢复语义和科学门均可由仓库验证。
- **停止门**：任一依赖或哈希漂移时，不提交作业，先生成 drift 报告。
- **图位**：Fig.3、Fig.4。

### A0-3 BioAware/应用边界审计

- **负责人**：bio-application agent
- **状态**：`COMPLETE_READ_ONLY`（2026-09-19）
- **任务**：核对 B47 当前落盘状态、Bridge Panel 资格、MTBLS13729/LCNEC 候选池和 reverse 路线。
- **交付物**：B47 gap list、Bridge资格表、validation-board字段与候选纳入规则。
- **通过门**：不把旧开发 prior、真值污染 seed 或无标准候选写成外部身份确认。
- **停止门**：本地与服务器冻结产物不一致时，不进入 seed/event/performance 阶段。
- **图位**：Fig.5、条件 Fig.6/7。

## 2. Phase 1：核查完成后可下发的正式任务

### N1 Noise V3 最终裁决

- **科学问题**：targeted corrective action 是否提供超过完整 E4 continuation 和 matched shuffled 的共享表示增量？
- **当前状态**：`READY_ON_CLUSTER_ONLY / NOT_YET_RUN_AFTER_REPAIR`。
- **唯一入口**：`tasks/run_noise_e4_live_shared_hybrid_v3_1gpu.sbatch`；固定 fold 0、seed 20260830、单 GPU、4 epochs。
- **输入**：当前 corrected graph、190,324-row/7,624-query E4 base、32,114-row/3,482-query 七来源 best-action ledger、official embeddings、E8 initial checkpoint 与 decision、完整 live E4 objective、修复后的 exact-0.25 injector。当前本地缺少服务器侧 `noise_corrected_npa4_formal_fold_0_run_2332161_v3routes/routes/n_bank` 和 `noise_corrected_best_v5_canary_fold_0_run_2332784/ledger`，提交前必须在服务器核验。
- **禁止事项**：新 action sweep、复活 dynamic N+P、修改 held gate、把 action headroom 当模型增益。
- **交付物**：targeted 与 matched-shuffled 各自的 `decision.json`、`final_shared_encoder.pt`、`held_per_query.csv.gz`；汇总 `report.json`、逐 query 对照表；Slurm stdout/stderr、job ID、输入路径、optimizer-boundary receipts、action exposure/coverage 与退化 anchor 索引。
- **工程合同**：active-group action displacement 必须精确为 0.25；E4 投影保留率 `>=0.90`；总更新/E4更新范数比 `<=1.50`；两臂除 action tensor 外完全一致。任一漂移则结果作废，不解释为科学阴性。
- **严格通过门**：`strict_full_panel_four_pp_achieved=true`；targeted 相对 official、initial E8、matched shuffled 的 Bonferroni formula-cluster CI 下界均严格大于0；global/near lambda-2 risk-net 均为正；所有注册指标非负并满足边界；两臂相对 initial E8 preservation 均 `>=0.995`；targeted 相对 official Recall@1 绝对增益 `>=4 pp`。
- **停止门**：任一科学门失败即停止 Noise embedding 正向扩展与扫参；若不胜 matched shuffled，只能报告动作流/训练延续效应。即使全部通过，也只是 single-fold、`formal=false` 的 corrected-development 结果，只授权预注册 multifold。
- **论文角色**：通过则为 representation adaptation；失败则为有对照的 transfer bottleneck，不得包装成算法提升。
- **图位**：Fig.4。

### C1 ChemAware canonical outer once

- **科学问题**：truth-blind候选策略能否在未触碰 outer 上同时建立总增益、化学增量和 residualization 增量？
- **当前状态**：`OUTER_READY_SCIENTIFICALLY / BLOCKED_BY_POST_SEAL_RECOVERY`；fold 4 含16,198个未触碰 query，仍未开启。
- **唯一入口**：`tasks/run_chemaware_truthblind_outer_once.sbatch`；评价器为 `tasks/evaluate_chemaware_truthblind_outer_once.py`。
- **唯一原则**：先核对 canonical SHA、provenance 与合同测试，再精确重放 role 3 inner；创建不可覆盖的 `OUTER_FOLD_4_OPENED.lock` 后才进行 outer inference；禁止 outer tuning。
- **输入**：canonical `run_2338337` policy、冻结 manifest/token/rule library 与最终 run card 中列出的哈希。
- **交付物**：seal；release-policy 的 report/model/policy 四件套；outer 的 truth-blind predictions、truth evaluation 与 report；预测、评价、release policy、canonical 与 seal 哈希。
- **通过门**：`scientific_pass=true`，即 Recall@1 相对 official `>=3 pp`；总增益 formula-cluster CI 下界 `>0`；`corrected > 2*introduced`；相对 nuisance-only 与 same-feature-direct 的 formula-cluster CI 下界均 `>0`；Recall@1/3/5/10/20/50 全部不下降。
- **停止门**：任何一个门失败即停止以当前 residualized policy 作为主方法；不得在 outer 上改阈值或选模型。
- **恢复阻塞**：程序在 outer inference 前先写 seal，异常会删除 staging 而保留不可重入的 seal。提交前必须在合成数据上补齐持久 attempt manifest、prediction receipt、truth-open receipt、seal 后不删证据、same-attempt 受限恢复、幂等只读 validator 和全阶段故障注入。禁止删除 lock 重跑；事故时保留 seal 与日志。
- **恢复审计**：`docs/CHEMAWARE_OUTER_PREFLIGHT_RECOVERY_AUDIT_20260919.md`；当前结论 `NO-GO`。
- **论文角色**：通过则为 spectrum-only 主方法；失败则降为开发期机制/诊断结果。
- **图位**：Fig.3。

### C2 ChemAware evidence topology（仅 C1 后条件启动）

- **前置条件**：C1通过，或C1明确显示局部化学有效但现有组织方式受限；不得用于挽救已打开outer。
- **第一阶段**：零学习比较 rule presence、质量差计数、端点共享、dominant-peak集中度、重复证据和matched topology null。
- **交付物**：冻结特征定义、无训练信息量报告、matched null、候选级泄漏审计。
- **通过门**：关系组织在未消费开发折上相对相同内容的bag-of-rules有正的formula-cluster增量。
- **停止门**：无增量即停止；不训练GNN。
- **升级**：仅通过后允许 tiny edge-aware head，backbone冻结。
- **图位**：未来 Fig.3 扩展或后续论文。

### B1 B47 provenance与U0

- **科学问题**：当前 B47 是否具备可复现、公平、真值盲的 spectrum-only denominator？
- **当前可核实资产**：51,976 queries、10,578 identities、2,078,709 query-reference rows；ST001122 33,829 queries、ST003356 18,147 queries。冻结 embeddings、seeds 与 U0 当前本地缺失。
- **交付物**：本地/服务器 artifact registry；embedding/seed状态；reference multiplicity、adduct mixing、candidate exposure报告。
- **通过门**：所有工件哈希一致；无真值/表型字段；每个query/candidate的参考数量与离子形式可审计。
- **停止门**：冻结embedding或seed仅在文档中存在、落盘缺失或哈希不一致时，停止后续模型工作。
- **图位**：Fig.5方法与数据基础。

### B2 B47 ion entity与exact event（仅 B1 后）

- **科学问题**：真实sample event是否在强unary、ion-family和静态prior之外提供增量？
- **交付物**：neutral/ion entity graph、truth-blind seed表、exact event ledger、degree/coverage/seed/rewired/sample-permutation nulls。
- **通过门**：事件来源精确、truth seed leakage为零、候选/来源/formula门达标；之后才允许一次性性能评估。
- **停止门**：只胜DreaMS但不胜强unary；只胜catalogue但不胜结构化null；任一主要来源显著有害。
- **图位**：Fig.5；通过后才有资格进入Fig.6。

### B1-U1 spectrum-only unary 与 seed denominator

- **科学问题**：先在不接触 B47 真值的 corrected development graph 上建立比 molecule-max 更强且风险受控的谱学一元分数，再判断其是否有资格重建 B47 seeds。
- **U1结果**：83,619-query、6,220-formula 的 formula-OOF 中 Recall@1 `+0.0682 pp`，formula-cluster CI `[+0.0249,+0.1135] pp`，near `+0.1348 pp`；但 corrected/introduced=`131/74`、lambda-2 risk-net=`-17`，且一个 outer fold overall/near 为负。因此 `STOPPED_UNSAFE`，禁止应用到 B47。
- **U1b纠错**：v1 的 `45/0` 来自 positive-first candidate ordering 下的 `argmax` 并列泄漏，已撤销；v2 禁止 baseline tie resolution 后 corrected/introduced=`0/0`、Recall@1 delta=`0`。状态 `STOPPED_INVALIDATED_V1 / V2_NOOP`。
- **当前唯一入口**：`tasks/run_bioaware_b47_u1_program.sbatch`；U1/U1b 只重放验证，U1c 对原始冻结 seed artifact 做真值盲分母漏斗审计。
- **U1c交付**：all queries -> unique Top-1 -> absolute score/margin -> feature consensus -> Rhea safe degree/noncurrency -> sample-candidate collapse 的 query/identity 数；独立 feature-consensus 统计不得混入嵌套 first-failure 判断。
- **停止门**：任何谱学 unary 在 formula-OOF 下 `corrected <= 2*introduced`、主要 fold 反向，或需要候选顺序/真值解除并列，均不得进入 B47；U1c 若显示在谱学门之前已少于200个可用身份，停止调 unary。
- **U1c结果**：identity漏斗为 `4300 -> 4300 -> 379 -> 238 -> 127 -> 127`；首次跌破200发生在Rhea degree/noncurrency阶段。sample-collapse只减少事件、不再减少身份。固定裁决为 `RHEA_COVERAGE_OR_HUB_BOTTLENECK`。
- **U2结果**：238个谱学共识身份中，Rhea-covered=`129`、Rhea-safe=`127`、Rhea+strict-KEGG-safe provisional=`141`、含EMRN的宽松edge headroom=`149`、无任何目录边=`89`。安全过滤仅从129减至127，不是主瓶颈；strict KEGG只净补14个provisional身份。EMRN不得充当exact event。
- **协议纠正**：`200 seed identities`是多样性门，不是候选排序的统计分母。现有127个Rhea-safe身份产生3,243个sample-collapsed seed events，可能覆盖足够多的独立候选竞争。因尚未打开truth或性能，先进行U3 truth-blind atomic-event yield审计，不以U2的身份数提前否决事件模型。
- **当前状态**：`U1 STOPPED_UNSAFE / U1b STOPPED / U1c COMPLETE / U2 COVERAGE_COMPLETE / U3-V1 REJECTED / U3-V2 INTERRUPTED_INVALID / U3-V3 READY_NOT_RUN / B2 OUTCOME_BLOCKED_PENDING_U3`。U3-v1 未正确实现关键null；U3-v2首轮truth-blind null暴露“重连身份重新定义反应变换”和“wrong-identity动作数被错误作方向性门”两项逻辑错误，数字禁止引用。U3-v3冻结原始反应槽位非氢元素变换签名并将wrong-identity降为非方向诊断。

### A1 Bridge Panel 设计与冻结（条件任务）

- **前置条件**：B47或组内数据具备独立身份、MS/MS、sample event和未污染context。
- **交付物**：预注册panel manifest、Mode A/Mode B共同候选协议、seed排除规则、统计功效与最小门。
- **最低资格建议**：至少1,000 queries、200 formulas、200 baseline errors、100 reaction-reachable errors；同一 query 必须同时具备独立身份、MS/MS、冻结候选、sample event 与 context；主比较为 strong unary+ion 对 strong unary+ion+exact event。
- **停止门**：不得从已经看到context结果的query中事后挑panel。
- **图位**：条件 Fig.6。

### V1 Validation Candidate Board

- **负责人**：biology/application agent
- **范围**：组内任务、LCNEC、MTBLS13729；暂不扩新疾病。
- **每行必填**：dataset、feature ID、m/z/RT、polarity/adduct、旧/新候选、A/B结构差异、各方法排序、诊断峰/规则/context、原始MS2、标准品与样本可用性、允许身份层级、验证成本、身份成立后的生物学增量。
- **选择原则**：优先标准/资源已存在且A/B竞争真实的对象；表型不得进入身份评分。
- **首轮审计顺序**：MTBLS13729 f1597/f3019；f703 Neu5Ac；资源可得时的 LCNEC quinolinate/ascorbate。f1717 缺标准时只到 family；f3222 只到 C20:4 acylcarnitine-like class。
- **交付物**：按实验可执行性排序的候选板，以及首批不超过3个对象的验证合同。
- **当前交付**：`docs/VALIDATION_CANDIDATE_BOARD_20260919.md`；首批为 f703、f1597/f3019 竞争标准包、条件启动的 LCNEC quinolinate。所有对象当前均为 `RESOURCE_AUDIT_REQUIRED`，不代表标准或剩余样本已在手。
- **通过门**：至少一个对象可执行同平台RT+MS/MS，必要时co-injection，并预先写明失败解释。
- **停止门**：不可分位置异构体、无样本或无标准时降级结论，不强制唯一命名。
- **图位**：条件 Fig.7。

### W1 Claim registry与稿件骨架

- **负责人**：统筹 agent
- **交付物**：claim registry、两种operational modes、固定/条件主图清单、竞品边界和Results骨架。
- **每条claim字段**：claim ID、原文、允许措辞、禁止措辞、dataset、candidate protocol、split、baseline、artifact、hash、evidence level、status、figure/table。
- **停止门**：没有冻结证据条目的数字不得进入摘要、主文或图标题。
- **当前交付**：`docs/PROJECT_CLAIM_REGISTRY_20260919.md` 已登记首批12条资产声明；后续 ChemAware、B47 与应用主张需继续追加。

## 3. 状态词

- `PLANNED`：尚未完成输入审计。
- `READY`：输入、哈希、合同、资源均通过，可执行。
- `RUNNING`：正式任务已启动。
- `ENGINEERING_PASS`：代码/重放/格式通过，不代表科学门通过。
- `DEVELOPMENT_PASS`：仅开发协议通过。
- `OUTER_PASS`：未触碰外层全部注册门通过。
- `EXTERNAL_PASS`：独立来源或标准品门通过。
- `STOPPED_NEGATIVE`：科学门失败，冻结为负结果，不再调参复活。
- `BLOCKED_ENGINEERING`：科学合同已冻结，但执行完整性或恢复语义不足；只允许不改变科学合同的工程修复。
- `BLOCKED_PROVENANCE`：工件、哈希、命名空间或权限不完整。
- `RESOURCE_AUDIT_REQUIRED`：候选科学价值已登记，但标准实物、剩余样本、方法或伦理资源尚未确认。

## 4. 当前状态快照

| 任务 | 状态 | 备注 |
|---|---|---|
| A0-1 资产与主张审计 | COMPLETE_READ_ONLY | 已纠正旧cohort、Peak-token、A1/A1b、P2b、E4-A资格 |
| A0-2 Noise/ChemAware合同审计 | COMPLETE_READ_ONLY | 已冻结两个run card；未提交作业、未打开outer |
| A0-3 BioAware/应用边界审计 | COMPLETE_READ_ONLY | 已冻结B47缺口、Bridge资格和验证顺序 |
| N1 Noise V3最终裁决 | BLOCKED_PROVENANCE | 仅集群可运行；先核验两个服务器侧ledger与GPU内测试 |
| C1 ChemAware outer once | BLOCKED_ENGINEERING | 科学合同ready但恢复合同NO-GO；先补seal后状态机，不得提交 |
| C2 evidence topology | PLANNED | C1之后的条件任务 |
| B1 B47 provenance与U0 | COMPLETE | 18工件/638,372,014 bytes哈希闭环；U0确认reference multiplicity与adduct pooling为重大混杂；见 `BIOAWARE_B47_B1_U0_RESULT_20260921.md` |
| B1-U1 spectrum unary | STOPPED_WITH_DIAGNOSIS | U1有+0.0682 pp公式隔离信号但131/74、lambda-2=-17；U1b v1顺序泄漏已撤销，v2为Top-1 no-op；不再调unary |
| B1-U2 catalogue coverage | COMPLETE_WITH_LIMITED_DIVERSITY | Rhea-safe 127；Rhea+strict-KEGG provisional 141；含EMRN宽松边149；89/238无目录边。该结果限制seed多样性，但不等价于事件级功效不足 |
| B1-U3 atomic event yield | V3_READY_NOT_RUN | v1/v2均已判废；v3用127个安全seed身份/3,243个sample事件构建query-candidate-seed-reaction原子表，冻结原始反应槽位非氢元素变换签名，执行度保持重连、跨样本完整seed-event置换和同query同adduct wrong-identity诊断，并控制重复反应记录、零净化学计量参与物、query/reference独立formula质量回放、adduct与reference multiplicity；结构特异性只对有效的graph/seed-context null检验，wrong-identity不按动作数量作方向判定；另要求actionable opportunity覆盖`ceil(0.03*N)`的+3 pp数学上限；不打开truth、不拟合模型 |
| B2 exact event outcome | BLOCKED_PENDING_U3 | U3通过后才可冻结一次性event-ranking评估；+3 pp、cluster CI、corrected>2xintroduced及胜过catalogue/null的最终门不变 |
| A1 Bridge Panel | PLANNED | 当前未具备资格 |
| V1 Validation Candidate Board | RESOURCE_AUDIT_REQUIRED | 候选板已完成；先核实标准实物、剩余样本、方法与伦理资源 |
| W1 claim registry/稿件骨架 | RUNNING | 首批12条claim已冻结，等待ChemAware/B47/应用扩展 |
