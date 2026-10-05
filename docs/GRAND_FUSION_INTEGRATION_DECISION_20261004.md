# DreaMS 集大成融合：综合研判、预注册路线与实施状态

日期：2026-10-04  
状态：**实施已启动；核心代码可运行；完整三层结论尚未授权。**

## 一句话结论

值得推进，而且是目前最高性价比的主线；但真正可能产生“1+1+1>10”的不是把三个模块的分数直接相加，也不是盲目把 Noise 与 ChemAware 顺序训练，而是把三类作用层严格分开：

1. **参数层**：组合 Noise 的条件稳健性与 ChemAware 的碎裂化学表征；
2. **候选层**：按查询在互补谱学专家之间进行风险受控选择；
3. **样本层**：只在真实样本存在已授权的候选特异事件时追加 BioAware 稀疏证据。

三层的分母、可见信息和失败模式不同。它们可以形成一个部署流程，但不能把各自在不同数据上的百分点相加成论文数字。

## 新范式的准确表述与创新边界

最新同领域工作已经分别覆盖自监督谱图基础模型、谱图—结构联合表征、常规集成排序、化学反应网络传播以及额外化学实验标签。因此，本课题不能把“用了多种信息”“联合嵌入”或“代谢网络辅助”本身写成方法学首创。

本课题真正可以争取建立的是**证据自适应的分层结构注释（evidence-adaptive hierarchical annotation）**：系统先使用可普适部署的谱图证据；只有当专家分歧、候选近似或预测风险达到预注册条件时，才调用候选特异的碎裂化学证据；只有当真实样本内存在通过空模型和质量约束的精确生化事件时，才追加样本特异证据；若新增证据不足以降低风险，则保留原排序或拒绝作出唯一结构判断。

这一范式必须同时满足五项条件，缺一项只能称为集成算法：

1. **逐查询证据预算**：不是所有查询无差别调用全部模块；
2. **作用域隔离**：共享表征、候选重排和样本事件分别在合法信息边界内工作；
3. **风险控制与拒判**：没有正的预期净纠错时自动退回强基线，不强行输出唯一结构；
4. **逐查询证据账本**：记录调用了哪个专家、为何调用、改变了哪个候选以及对应冻结产物哈希；
5. **统一分母上的增量验证**：同一候选池内报告 Top-1、corrected/introduced、近结构子集、coverage-risk 曲线和公式簇置信区间，BioAware 则单独接受事件级外部门。

在完整外测和 B47 外部门通过前，文稿只能把它称为“待验证的分层证据决策框架”，不能先写成已经建立的新范式。

## 为什么这条路线足够“硬”

### 1. 互补性已经在独立外部数据上真实存在

GNPS Gold/Silver 冻结面板中，Noise V1、weighted spectral entropy（WSE）和 P2b(V1) 的单模型结果及标签知情上限如下：

| 面板 | Noise V1 | WSE | P2b(V1) | 三者 oracle union | 相对最佳单方法上限 |
|---|---:|---:|---:|---:|---:|
| identity-disjoint，n=10,995 | 86.56% | 87.37% | 87.13% | 90.59% | +3.22 pp |
| formula-disjoint，n=5,261 | 88.06% | 88.27% | 88.25% | 91.94% | +3.67 pp |

这相当于把最佳单方法的错误率理论上再降低约 25%（identity）和 31%（formula）。oracle 不是可部署结果，但它回答了最关键的立项问题：**现有资源中确实存在数个百分点的非冗余信息，不需要先开发新多模态模型。**

### 2. 现有模块不是三个同质模型

- Noise V1 改变共享谱图 embedding，已在 GNPS identity/formula 隔离面板取得 +1.20/+1.25 pp；
- P2b 是冻结谱学通道上的候选重排器，外部表现与 WSE接近，但纠错集合并不相同；
- ChemAware V2 读取特征碎片、中性丢失和质量差等候选条件证据，开发面板净纠正 76 个查询（93 corrected/17 introduced）；
- BioAware B47 读取样本内事件，作用域只覆盖存在合格样本上下文的查询，不应污染普适谱图 encoder。

因此最合理的组合不是一个大一统网络，而是“共享 encoder + 候选专家选择 + 稀疏上下文残差”。这也保留了论文中每一项增益的可解释性。

### 3. 负结果为融合划出了边界

- Noise RRF 曾在 official geometry 的探索折上得到 +1.77 pp，但 Stage-1 与 GNPS 正式确认失败，永久归入 audit-only；它可以进入完整消融表，不能进入生产路由。
- P2b 在 sealed P3-main 为正，但 near-core 为 -4.23 pp，说明固定全局重排不是答案，必须学习“何时不要用”。
- 已实现但尚未运行的 Noise MSG candidate-level fusion stack 是另一条正式竞争臂；它与本次 query-level risk router 必须在同一 GNPS 面板比较，不能因为是新代码就默认后者更优。
- BioAware 静态目录、度数/拓扑和聚合上下文已经出现外部反转或无效；B47 必须坚持真实样本、候选特异、事件级和严格空模型。

这些失败不是包袱，而是支持“风险路由 + abstention”设计的直接证据。

## 三层融合设计

### A. 参数层：Noise × ChemAware encoder

优先顺序不是先做昂贵串训，而是：

1. 对 official、Noise、Chem 三个同构 slim checkpoint 计算任务向量；
2. 先测全局余弦、逐张量余弦、活跃参数符号一致率和 top-1% 支持重叠；
3. 做很小的冻结合并网格：linear task arithmetic 与 TIES；
4. 只对通过内部保护门的 1 个合并 checkpoint 做一次 GNPS 外部评价；
5. 只有参数向量明显冲突且合并失败，才投入顺序训练。

顺序训练的必要对照：

| 实验臂 | 目的 |
|---|---|
| official → Noise | 已有 Noise 基准 |
| official → Chem | 已有 Chem 基准 |
| Noise → Chem | 检验先学条件稳健性、后学化学区分 |
| Chem → Noise | 检验反向次序及化学知识遗忘 |
| Noise → matched continuation | 排除“多训练了一轮” |
| Chem → matched continuation | 同上 |

Noise V1 slim 未保存可继续的 Adam 状态，所以 Noise→Chem 不能伪称“原优化轨迹连续”；它是以 Noise 权重重新初始化优化器的 warm start，必须与等步数 matched continuation 比较。

选择 task-vector/TIES 作为第一步有充分方法学依据：model soups 说明同一预训练起点附近的微调权重可低成本合并；task arithmetic 提供任务向量相加的形式；TIES 专门处理小幅冗余更新和符号冲突。反过来，PCGrad/CAGrad 或联合多任务训练需要同时读取两套训练任务、重写训练循环并解决梯度尺度，计算和实现风险更高，放在第二阶段。参考：[Model Soups](https://proceedings.mlr.press/v162/wortsman22a.html)、[Task Arithmetic](https://github.com/mlfoundations/task_vectors)、[TIES-Merging](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1644c9af28ab7916874f6fd6228a9bcf-Abstract-Conference.html)、[CAGrad](https://proceedings.neurips.cc/paper/2021/hash/9d27fdf2477ffbff837d73ef7ae23db9-Abstract.html)。

### B. 候选层：查询级风险路由

默认专家设为当前外部最强的 WSE。路由器只使用以下真值盲特征：

- 各专家 top-1 与 top-2 的归一化间隔；
- 某专家赢家被其他专家支持的比例；
- 该赢家在其他专家中的平均/最低分位排名；
- 专家两两是否选择同一严格赢家。

明确禁止：候选数量、参考谱数量、query edge 数、数据库度数和任何真值派生特征。训练按 molecular formula 分组；外层折评估，内层独立公式折选择切换阈值。目标函数为：

`risk_net = corrected - 2 × introduced`

若任何阈值都没有正 risk-net，阈值自动选择 1.10，即全量退回 WSE。最终 GNPS 应用只选择一个专家的完整 query score block，不在不同专家的不可比分值上做线性混合。

现有 `noise_msg_fusion_stack_v1` 则直接在候选分子层学习融合分数。两者解决同一问题但归纳偏置不同：MSG stack 更有表达力，risk router 更保守、可解释且可退回 WSE。正式结果应把 WSE、P2b(V1)、MSG stack、risk router 和（若完成）ChemAware-expanded router 五者放在同一冻结表中。

ChemAware V2 与 MSG 在本轮开始前已经登记为 provisional experts，因此二者会无条件进入这一次完整冻结系统试验；代码不会根据它们先产生的 GNPS 单模块结果决定是否纳入路由。单模块 adjudication 只回答可否单独主张该组件，完整系统 adjudication 才回答组合能否发布。这样避免用同一 GNPS 面板“先筛组件、再评价组合”的二次选择泄漏。

### C. 样本层：BioAware 稀疏残差

BioAware 不参加 GNPS 谱库基准，因为 GNPS 查询没有本课题定义的同一样本上下文。正确评价方式是：

1. 先冻结参数层和候选层形成 spectral backbone；
2. 在 B47 两个真实研究上重新产生谱学基线；
3. 只对 `repaired_action_ledger` 中授权的候选特异机会施加上下文动作；
4. 与 degree-preserving rewiring 和 metabolite-sample permutation 空模型比较；
5. R1-R5 任一不通过，BioAware 层关闭，不打开真值。

因此最终论文应有两个互补主结果：无样本上下文的外部谱库泛化，以及有样本上下文时的事件级增益。两者共同支持端到端方法，但不共享一个虚假的总分母。

## “1+1+1>10”的可检验推断

不能承诺 +10 个百分点。当前能够严谨提出的是三个乘法来源：

1. **表征增益扩大每个后续专家的输入质量**：Noise/Chem 合并若成功，P2b、WSE 与化学重排的共同基线一起提高；
2. **路由获取集合并集而非平均值**：当前三专家外部 oracle gap 已有 3.22/3.67 pp；即使只捕获其中 40%，也约为相对最佳单专家 +1.29/+1.47 pp；
3. **BioAware 只覆盖谱学难例**：稀疏上下文动作若集中在 spectral backbone 的残差错误中，其单位动作价值高于全局平均加权。

这里第二项有现成外部数据支持；第一、三项仍是待证假设。最现实的高 KPI 不是“总准确率突然 +10 pp”，而是：

- 外部 Top-1 相对最佳现成专家再提高 1–2 pp；
- near-structure 子集提高更多；
- corrected/introduced > 2；
- 在不增加新推理模型数量或只增加轻量路由的情况下，实现 10%–30% 的相对错误率下降；
- 在有真实样本上下文的子集上再获得经空模型和独立研究支持的局部增益。

这已经足以成为强论文结果，而且比把三个开发数字相加可信得多。

## 冻结判据

### 参数层继续门

- 内部公式隔离评价不劣于最佳 constituent 超过 0.10 pp；
- 至少一个预注册合并臂相对最佳 constituent ≥ +0.30 pp；
- GNPS 一次性确认中，一个面板 ≥ +0.50 pp，另一个非劣，且主要 paired CI 不跨零；
- 否则停止 encoder 合并，保留两个独立专家进入路由。

### 路由层成功门

- identity 与 formula 两面板均优于 WSE 和 P2b(V1)；
- 至少一个面板 ≥ +1.00 pp，另一个 ≥ +0.50 pp；
- formula-cluster multiplicity-corrected CI 下界 > 0；
- `corrected > 2 × introduced`，且 near 子集不回归；
- 未过门则不发布路由性能，只报告 oracle headroom 和失败分析。

### BioAware 成功门

- 先通过既有 B47 R1-R5；
- 只允许一次真值打开；
- 预注册主门为 ≥ +3 pp、formula/source CI 下界 > 0、`corrected > 2 × introduced`；
- 未过门则只保留机制性事件发现，不作注释准确率主张。

## 已实现代码

- `tasks/grand_fusion_components_v1.json`：全组件状态注册表；
- `tasks/grand_fusion_router_core.py`：真值盲特征、严格平局、风险路由；
- `tasks/train_grand_fusion_router.py`：外层公式隔离、内层阈值选择、冻结模型；
- `tasks/apply_grand_fusion_router_to_gnps.py`：封存 GNPS 应用与不可变 pair-score cache；
- `tasks/apply_grand_fusion_router_to_gnps.py` 同时写出逐查询真值盲路由账本，记录专家选择、预测风险、候选赢家与冻结来源哈希；
- `tasks/extend_grand_fusion_pair_evidence.py`：以 evidence SHA256 对齐 Chem/RRF 外部专家分数；
- `tasks/build_gnps_chemaware_kernel_cache.py`：在不读取候选真值的前提下复现 ChemAware V2 所需的峰质量、强度和前体缓存；
- `tasks/export_chemaware_v2_grand_fusion_scores.py`：用报告内 SHA256 锁定 V2 策略，分块导出 MassSpecGym 与 GNPS 候选动作；
- `tasks/analyze_dreams_task_vector_interference.py`：Noise/Chem 参数冲突诊断；
- `tasks/normalize_dreams_slim_checkpoint.py`：将 ChemAware native/slim 统一为带哈希的同构 checkpoint；
- `tasks/merge_dreams_task_vectors.py`：linear task arithmetic 与 TIES 合并；
- `tasks/adjudicate_grand_fusion_external.py`：按预注册 CI、幅度与 corrected/introduced 门自动判停；
- `tasks/audit_grand_fusion_readiness.py`：完整性审计；
- `tasks/test_grand_fusion.py`：CPU 合同测试。
- `tasks/run_noise_msg_fusion_5pp_2gpu.sbatch`：一次生成并保留 MassSpecGym evidence，同时完成现有 MSG stack；
- `tasks/run_grand_fusion_task_vector_scan_2gpu.sbatch`：七臂 Noise×Chem 参数组合开发扫描；
- `tasks/run_grand_fusion_fixed_consensus_1gpu.sbatch`：全部固定共识规则的 post-hoc 机制筛查；
- `tasks/run_grand_fusion_router_1gpu.sbatch`：冻结风险路由及 GNPS 对照；计算本身为 CPU 型，但按 E6 集群提交策略声明一张 GPU；
- `tasks/run_grand_fusion_chemaware_v2_scores.sbatch`：把 ChemAware V2 和 MSG 的 OOF/外部缓存接入同一 evidence/bundle；
- `tasks/run_bioaware_b47_gate_remediation.sbatch`：B47 R1–R5 服务器修复门；
- `tasks/freeze_bioaware_b47_repaired_action.py`：恢复丢失 U4 源码所代表的真值盲动作语义，并对未覆盖查询严格保持 unary 排序；
- `tasks/evaluate_bioaware_b47_repaired_once.py`：以不可重复创建的 seal 控制唯一一次真值打开；
- `tasks/adjudicate_bioaware_b47_external.py`：自动执行 +3 pp、聚类 CI、风险、来源、近结构和结构化空模型八项门；
- `tasks/run_bioaware_b47_confirmatory_once.sbatch`：B47 授权后的唯一确认性服务器作业；
- `tasks/run_grand_fusion_selected_encoder_gnps_1gpu.sbatch`：唯一选定 encoder 的一次性外部确认；
- `tasks/submit_grand_fusion_program.sh`：并行提交及 `afterok` 依赖编排。

## 当前实施状态与下一步唯一顺序

本地 readiness audit 的事实结论是 `PARTIAL_ONLY_FAIL_CLOSED`：现有 GNPS bundle 已包含 Noise、WSE、P2b 及全部基础谱学通道，但本地没有服务器上的 MassSpecGym pair evidence、Noise/Chem slim checkpoint、ChemAware 全图分数和 B47 修复产物。因此当前代码完成，完整实验尚不能在本机冒充已完成。

服务器执行顺序：

1. 运行 `run_noise_msg_fusion_5pp_2gpu.sbatch` 一次生成并永久保存 `RUN_ROOT/evidence`；该脚本已修正为不再随临时目录删除 620 万边 evidence；
2. 用正式 V2 truth-blind policy 导出 ChemAware train-side 与 GNPS frozen scores；同时只把 MSG 的 formula-OOF 分数用于路由训练，把其 frozen GNPS cache 用于外测；
3. 先跑 task-vector interference；只跑小网格 linear/TIES；
4. 冻结唯一 encoder 候选后编码 GNPS；
5. 在包含 Noise V1、WSE、P2b、MSG 与 ChemAware V2 的完整冻结专家集合上训练一次 grand router，再一次性打开 GNPS；
6. 独立执行 B47 remediation；只有授权后才冻结 exact-event 动作、创建一次性 seal 并运行统一 Track-C 评价。

不得倒序，不得用 GNPS 选择参数，不得把 RRF 复活为生产组件，不得在 B47 未授权时打开真值。

### 服务器提交

```bash
cd /data02/run01/scv7tsl/DreaMS
export CHEM_CHECKPOINT=data/validation/chemaware_high_coverage_native/run_2340721_resume_2340524/training/best.ckpt
test -f "$CHEM_CHECKPOINT" || { echo "missing $CHEM_CHECKPOINT" >&2; exit 1; }
export B47_U3_DIR=<可选：冻结U3-v3输出目录>
# 若要自动排队B47一次性确认，还必须显式给出以下七个规范输入：
export B47_CANDIDATE_MANIFEST=<含唯一真值的冻结候选表>
export B47_UNARY_SCORES=<冻结谱学主干候选分数>
export B47_CATALOGUE_SCORES=<目录度数对照分数>
export B47_NETWORK_SCORES=<MetDNA3/KGMN兼容网络对照分数>
export B47_DEGREE_REWIRED_SCORES=<度保持重连对照分数>
export B47_SEED_PERMUTED_SCORES=<完整seed-context置换对照分数>
export B47_MATCHED_NONNEIGHBOR_SCORES=<匹配非邻居对照分数>
bash tasks/submit_grand_fusion_program.sh
```

首批任务会并行运行 MSG/evidence、任务向量扫描和可选 B47 gate；固定共识诊断默认跳过，不占用 GPU，只有显式设置 `RUN_FIXED_CONSENSUS=1` 才提交。ChemAware V2 对齐作业通过 `afterok` 等待 MSG evidence；risk router 再等待完整的 MSG+ChemAware evidence/bundle。若 B47 的七个规范输入齐全，确认作业仅在 remediation 正常结束后排队，并在授权失败时于真值打开前停止。任务向量七臂结束后只能按预注册保护门选一个 checkpoint，再手工提交：

```bash
sbatch --export=ALL,SELECTED_CHECKPOINT=<唯一选定checkpoint> \
  tasks/run_grand_fusion_selected_encoder_gnps_1gpu.sbatch
```

这个人工停顿是科学门，不是自动化缺陷：如果自动根据 GNPS 结果继续挑第二个 checkpoint，就会失去外部确认资格。
