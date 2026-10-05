# ChemAware 2.3pp 平台期系统性分析 + GNPS 外部基线决定实验(2026-09-30)

## 0. 一句话结论

> ChemAware 的 ~2.1–2.4 pp 平台期不是运气问题,而是三个叠加的结构性事实:(1)**整个闭环关在 MassSpecGym 单一分布里**——DreaMS 在它上面训练、我们的课程从它挖掘、评估也在它的公式隔离折;(2)**1,975-query 面板的统计分辨率地板**——1 个净 query = 0.0506 pp,Phase-A 的 CI 宽 ±0.88 pp,任何真实的 <0.5 pp 改善在这个面板上不可见,我们一直在噪声带里迭代;(3)**固定 margin triplet 信道对化学信息的容量已耗尽**——Phase-A 之后连续约 10 次尝试(V3–V13、step-750、union、listwise 两臂)无一在配对 CI 意义上超过 Phase-A。下一步唯一廉价且决定性的动作:**把已封存的 GNPS gold/silver 基准(10,995 + 5,261 query,与训练语料身份/公式双零重叠)用作外部评估,先测既有资产(official / stage-1 / Phase-A / step-750)的真实迁移**——零训练,纯编码+评估,工具链现成。

---

## 1. 那个 2.3pp 到底是什么、测在哪

### 1.1 数据集全景

| 层 | 事实 |
|---|---|
| 底层数据 | `MassSpecGym_MurckoHist_split.hdf5`,231,114 谱,28,929 IK14——**DreaMS 官方 embedding 的训练/微调语料本身** |
| 候选图 | corrected candidate manifest:83,619 训练 query / 9,854 身份 / 6,220 公式 / 392,229 候选分子;严格 10 ppm、同加合物;**75.08% 负候选与真值同公式;78.06% query 面临同公式干扰;33.71% 有 MCES-near 干扰** |
| 训练面板 | role 0–1:4,032 anchor query / 2,518 公式(所有课程的挖掘池) |
| 评估面板 | role-2 选择:1,978 → **1,975 配对**;role-3 确认:1,929;role-4 outer:已被消耗 |
| 分数规则 | molecule-max cosine;并列按失败计;official replay 审计失配 0 |

所有 headline 数字的位置:

| 检查点 | role-2(1,975) | role-3(1,929) |
|---|---:|---:|
| official | 89.6203%(205 错) | 90.4614% |
| stage-1 | +1.5190(54/24) | **+1.8144(57/22,CI [+0.82,+2.87])** |
| Phase-A | **+2.1266(54/12,CI [+1.28,+3.03])** → 91.7468%(163 错) | **从未评估** |
| step-750/union(探理性) | +2.3797 / +2.4304 → 92.00% / 92.05%(158/157 错) | 从未评估 |
| 其余全部 | ≤ +0.76 且 CI 跨零,或为负 | 未开 |

### 1.2 同分布闭环(最重要的元事实)

挖掘(roles 0–1 同 manifest)→ 训练 → 评估(role-2/3 同 manifest 公式隔离折)全部在同一分布内。**Phase-A 的 +2.13 是"同语料公式隔离泛化",从未在语料外验证过一次。**GNPS 基准的 role 在整个 ChemAware encoder 线里只被规划、从未执行(V16 union sbatch 停止于覆盖门;两臂 sbatch 未运行 GNPS 段)。

---

## 2. Phase-A 那次的真实局限(六条)

1. **化学信息压缩成一位**。课程只决定"哪个 negative 被选中";证据强度/类型/冲突结构在损失里不存在(原生 cosine hinge,margin 0.1)。
2. **池的化学密度低且有向性天花板**。5,956 事件 = 3,605 安全 + 777 错误边界 + **仅 550 化学事件** + 1,024 replay;证据类诊断显示残余 151 训练折错误中**仅 42 个有可恢复化学方向,58% 无方向**。
3. **错误质量是同公式结构混淆**。75% 负候选同公式;这正是单谱→单向量 cosine 最难分辨的地方,而 product-ion/中性丢失规则对此天然稀疏。
4. **Adam 轨迹已物理丢失**。历史清理删除了原始 training 目录,reproduction gate 是 weights-only 导出(2026-09-30 审计器证实:无 optimizer_states)——"真连续续训"在资产层面不可能。
5. **从未 role-3 确认、从未外部确认**(冻结账本 evidence_class 明示)。
6. **面板分辨率**。见 §4。

---

## 3. 为什么越改越差:失败分类学(每类对应已关闭实验)

| 失败类别 | 机制 | 对号实验 |
|---|---|---|
| **A. 同信道加剂量 → 过拟合** | 更多事件/更高激活 ≠ 更多信息;边际事件是"模型已会"或"语料噪声" | V3 pair-expanded(12,333 事件,激活×4.25):+0.76 CI 跨零、R@3 −0.15、C−2I=−8 |
| **B. 自适应重挖 → 训练/ held-out 分离** | 修训练折的特定边界 = 记忆折内光谱特异;剩余错误在训练折上也是噪声 | V11(训练错 151→133,role-2 持平/负);V12(训练错 151→**73**,role-2 单调 −0.05→−0.41)——自适应 hard-negative 过拟合的直接证据 |
| **C. fresh-Adam 续训 → 先漂移后部分恢复** | 收敛模型 + 重置二阶矩 + lr 5e-6 → 先离开 Phase-A 盆地;小剂量走不出漂移段 | V6(step 100 即低于 Phase-A 5 个净 query,永不恢复);step-750/union(+0.25/+0.05,CI 跨零) |
| **D. 换目标 + 换优化器 + 十倍剂量同时发生** | 分布漂移到 60% 已正确 query 上;role-3 反向 | listwise 两臂(role-2 +0.25/+0.30,role-3 **−0.52/−0.41**;化学独立贡献仅 +0.05/+0.10) |
| **E. 点增益被安全门正确拒绝** | 修正多但引入更多/几何回退 | pair-expanded +2.28(70/25 拒);V13 +0.25(margin 0.370→0.341,拒) |

**共 ~10 次连续尝试,零次在配对 CI 意义上超过 Phase-A。这不是执行不好,是信道饱和。**

---

## 4. 统计分辨率地板(被忽视的一面)

- 1,975 query 面板:**1 个净 query = 0.0506 pp**;Phase-A 公式簇 CI 宽约 ±0.88 pp。
- 含义:(a) 真实 +0.3 pp 的改善在这个面板上**不可检出**(CI 跨零是必然);(b) 反过来,所有"探理性 +0.25"也都在噪声带内;(c) 我们用 ~2k 面板在分辨 ~0.3pp 级别的效应——**实验设计从 V5 起就在统计上注定反复得到"CI 跨零"**。
- GNPS identity 面板 10,995 query:1 净 query = 0.0091 pp,CI 宽缩小约 √5.6 ≈ 2.4×;配对设计的分辨率足以检出 ~0.3–0.5 pp 的真实效应。

---

## 5. 信息论诊断:平台期 = 三条上界的交点

1. **可学且同分布可迁移的分数被 Phase-A 拿完了**:42 个有方向错误中 54 已纠正/12 引入;再加剂量落入类别 A/B。
2. **oracle 证明余量存在但不在 triplet 信道里**:R0 切空间 +4.0039(85/3,几何可表征);ICEBERG 分数路由剂量 0.5 → +4.9805(102/0)。余量需要**候选条件化/结构信息**驱动位移,固定 margin 三元组传不过去(10 次空结果的贝叶斯证据)。
3. **错误质量向同公式结构混淆集中**(75% 同公式),单谱单向量在该子集上信息不足。

结论:**5 pp 不会来自 MassSpecGym role-2 上的任何后续课程雕琢**。要么换信息源(同公式结构证据:CSI/ICEBERG 类),要么换模型结构(多向量/候选条件化),要么换战场(更大、更难、分布外面板——那里 headroom 更大且分辨率更高)。

---

## 6. GNPS gold/silver 基准:已备好的更大战场

`data/validation/gnps_gold_silver_10ppm_benchmark_v1`(sealed,construction_model_blind=true):

| 性质 | identity-disjoint | formula-disjoint |
|---|---:|---:|
| queries | **10,995** | **5,261** |
| 身份/公式 | 5,534 / 4,204 | 2,662 / 2,210 |
| 候选分子 / 谱 | 87,518 / 175,171 | 24,401 / 47,724 |
| near 子集 | 69.0% | 56.0% |
| 每query候选身份(中位/p90) | 6 / 17 | 3 / 9 |
| 与 MSG+MoNA 重叠 | **身份 0 / 公式 0** | **身份 0 / 公式 0** |

来源:2,091,446 条 GNPS 原始记录 → Gold/Silver 质量、[M+H]+ 验证、独立采集正例、并列对真值不利。**诚实边界(report 自带)**:与 DreaMS 预训练共享公共数据生态,不可声称"全新语料";[M+H]+ only;单一库。配套还有 MoNA 极性迁移面板(2,691 query,已冻结未跑)与 MSnLib(406 query,预注册三切分)可作后续第二/第三外部面板。

### 6.1 决定实验与两个世界

对 official / stage-1 best / Phase-A / step-750 做纯编码+配对评估(零训练)。两种结果都改变下一步:

- **世界 A:Phase-A 在 GNPS 上仍为正**(哪怕衰减到 +0.5~1 pp,CI 为正)→ 微调学到了跨分布的真实结构;则 5pp 狩猎应该在 GNPS 上进行(official 基线大概率低于 89.6%,headroom 更大、分辨率更高;课程应从 GNPS+MSG 联合挖掘)。
- **世界 B:Phase-A 在 GNPS 上 ≈ 0 或为负** → 2.3 pp 是同语料打磨;整个迭代环测的是分布内记忆;正确动作是把挖掘/训练/评估全部搬到外部语料,化学与结构证据在新鲜分布上重新计量 headroom。

### 6.2 电池结果(run_2347471,2026-09-30):**世界 B 被证实**

绝对基线(首次建立):

| 模型 | identity R@1(10,995 q) | formula R@1(5,261 q) | identity near R@1(7,592 q) |
|---|---:|---:|---:|
| official | 0.8535698045 | 0.8680859152 | 0.7993940991 |
| stage-1 | 0.8565711687 | 0.8701767725 | 0.8013698630 |
| Phase-A | 0.8558435653 | 0.8720775518 | 0.8013698630 |
| step-750 | 0.8558435653 | 0.8728378635 | 0.8011064278 |

配对差(R@1,公式簇 10,000 重抽 CI,24 假设族):

| 对比 | identity ΔR@1 [CI]pp | formula ΔR@1 [CI]pp | 边注 |
|---|---|---|---|
| stage-1 vs official | +0.3001 [−0.2388,+0.8246] | +0.2091 [−0.5357,+0.9454] | **margin 显著为负**(−0.63/−0.81,CI 均不含 0) |
| Phase-A vs official | +0.2274 [−0.3109,+0.8037] | +0.3992 [−0.3234,+1.1281] | margin 显著为正(+2.59/+2.75)但零翻转 |
| step-750 vs official | +0.2274 [−0.2611,+0.7849] | +0.4752 [−0.2469,+1.1980] | 同上 |
| Phase-A vs stage-1 | **−0.0728** [−0.6134,+0.4774] | +0.1901 [−0.5695,+0.9680] | identity 171/179 净负 |

判读(全部 12 个 R@1 CI 跨零):

1. **迁移不存在**:MSG role-2 上的 +1.52/+2.13/+2.38 在 GNPS 上缩到 +0.2~0.5pp,统计上与零不可区分。8 个"我们的模型 vs official"的 R@1 CI 全部跨零。
2. **开发面板的排序也不迁移**:Phase-A 在 role-2 上比 stage-1 高 +0.61,GNPS identity 上反而 −0.073(corrected 171 / introduced 179)。
3. **risk_net(λ=2)全负**(−32 ~ −187):corrected 与 introduced 约 1:1——三个微调在分布外基本是随机重排边界。
4. **margin 动了,决策没动**:Phase-A 把正-最佳负 margin 拉大(+2.6~2.8pp,CI 正)但 R@1 不动;stage-1 甚至把 margin 拉小。几何位移没有转化成排序翻转——同语料边界放置的指纹。
5. **near 子集(69%,同公式难区)无增益**:全对 +0.17~0.20pp,CI 跨零——化学课程在它理应最强的区域也没有迁移。
6. 唯一显著的非 R@1 运动:stage-1 的 micro/pooled AUPRC(identity 0.6815→0.7086;pooled 0.6519→0.6801)——候选级聚合分离度改善,但宏观配对 CI 跨零、且不改变 Top-1。仅记录,不作为正向主张。

**结论:2.1–2.4pp 平台 = 同语料打磨。ChemAware 微调线在分布外基准上的 Recall@1 效应为零。**GNPS 正式成为主基准,official 绝对基线 85.36%(identity)/ 86.81%(formula),near 子集 79.94%——headroom 比 role-2(89.62%)大得多。

---

## 7. 行动包(本文件附带)

- `tasks/run_GLM_chemaware_gnps_external_battery.sbatch`:prepare slim ×3 → encode ×4(official、stage1、phaseA、exploratory_step750)→ 配对评估 ×4(三个 vs official;外加 phaseA vs stage1 课程前身对照)→ 四份自足评估 JSON(双面板 R@1/MRR/near/CI,descriptive,无门)+ provenance + SHA256SUMS。
- `tasks/test_GLM_chemaware_gnps_external_battery_sbatch.py`:静态合同(无 partition、gpus=1、零训练调用、双面板、无覆盖)。

## 8. 后续分岔(电池结果出来后)

**已裁决(2026-09-30,§6.2):世界 B。**因此:

- ~~A 世界:GNPS 联合课程~~ —— 微调线的 MSG 增益不迁移,先在 GNPS 上重做证据计量,再谈训练。
- **下一步(测量,先于任何 GPU 训练)**:GNPS 版错误成分诊断——对 official 在两个面板上的全部错误(identity ~1,608 + formula ~693)做同公式比例、MCES-near、化学证据覆盖(legacy 多零 + fragment/MassBank 合格源能否指向当前错误 winner)计量。这是 capacity audit 的 GNPS 重演,CPU 级。它回答:化学/结构证据在新分布上的杠杆上限有多大。
- 若覆盖可观:GNPS 训练侧课程(native triplet,候选图从 GNPS 接受池构建,公式/身份与两面板不相交,面板永不触碰)。
- 若覆盖贫瘠:候选条件化/多向量结构路线在 GNPS 上直接立项;MoNA 极性面板(2,691 q)与 MSnLib(406 q)作第二/三外部面板。
- 共同:错误 winner 覆盖率计数(SIRIUS CSI)改在 GNPS 分布上执行;role-2 迭代正式封存。

## 9. 工件

- GNPS 基准:`data/validation/gnps_gold_silver_10ppm_benchmark_v1/`(report.json 含全部 SHA)
- 编码/评估工具:`tasks/prepare_official_embedding_checkpoint.py`、`tasks/encode_gnps_gold_silver_10ppm_checkpoint.py`、`tasks/evaluate_gnps_gold_silver_10ppm_embeddings.py`(run_2347032 同链)
- 检查点:official slim;stage-1 `run_2340721_resume_2340524/training/best.ckpt`;Phase-A `run_2345481/phasea_reproduction_gate/chemaware_phasea_max_boundary_step2000.ckpt`;step-750 `run_2346408_resume_run_2346306/training/step-000750.ckpt`
- 历史证据:本文件引用的全部数字见 `docs/GLM_CHEMAWARE_ALGORITHM_POSITIVE_RESULTS_LEDGER_20260930.md`
