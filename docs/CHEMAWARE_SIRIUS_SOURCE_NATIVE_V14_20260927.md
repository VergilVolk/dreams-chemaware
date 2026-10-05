# ChemAware V14：SIRIUS 源化学证据 + Phase-A 原生 triplet 续训

## 1. 当前裁决

不再复现 Phase A。V14 直接使用已保护的 `+2.1266 pp` Phase-A checkpoint
作为初始化，完整保留其训练池，只改变新增 triplet 的化学证据来源。

不新增 adapter，不蒸馏，不改模型、采样器、loss 或 optimizer。V14 的唯一
算法变化是：用 SIRIUS fragmentation tree 与 CSI:FingerID 在完整候选图上
选择新的、对当前 Phase-A 边界有效的 native DreaMS triplet。

## 2. 为什么源化学信息才是当前瓶颈

Phase-A 训练池共有 5,956 个事件，其中 3,605 个 safety、1,024 个官方
DreaMS replay、777 个 official-error max-boundary、550 个 ChemAware-error
max-boundary。对正式候选图重新分类后：

| Phase-A 事件 | 同分子式 | 跨分子式 | 同分子式比例 |
|---|---:|---:|---:|
| official-error boundary | 683 | 94 | 87.9% |
| ChemAware-error boundary | 477 | 73 | 86.7% |

因此，继续堆叠平面质量差、峰重叠或由当前谱图相似度派生的规则，主要是在
重复 DreaMS 已经看到的几何。仅提升分子式判断也不够：绝大多数关键错误是
同分子式结构竞争。需要的新增信息是“该碎裂树更支持哪些结构子性质”，而
不是再造一组谱图距离。

SIRIUS 为每个候选分子式计算 fragmentation tree；CSI:FingerID 再从谱图与
碎裂树预测概率分子指纹并对结构数据库候选打分。这正好形成两个互不混标度
的证据层。[SIRIUS 6 CLI](https://v6.docs.sirius-ms.io/cli/)、
[SIRIUS 4](https://www.nature.com/articles/s41592-019-0344-8)、
[CSI:FingerID](https://doi.org/10.1073/pnas.1509788112)

## 3. 已完成的无泄露源面板

本地冻结资产：
`data/validation/chemaware_sirius_source_panel_v3/`

| 项目 | 数量 |
|---|---:|
| role-0/1 queries | 4,032 |
| query-formula 固定副本 | 7,939 |
| candidate rows | 16,787 |
| unique IK14 connectivities | 6,977 |
| off-formula candidate rows | 5,336（31.79%） |
| 含 off-formula 候选的 queries | 2,515（62.38%） |
| 少于两个有效峰的 queries | 7 |

每个 query 的真实分子式没有写入无约束输入；固定分子式副本对该 query 的
**所有候选分子式**逐一生成，不只生成真实分子式。候选表与 query registry
不导出 label。`>feature_id`、`>formula`、`>ionization` 与 `>ms2` 均按
[SIRIUS 6 I/O 规范](https://v6.docs.sirius-ms.io/io/)输出。

固定副本设计有两个目的：

1. 避免正确候选式因 SIRIUS top-k/soft threshold 在 fingerprint 阶段前被丢弃；
2. 7,939 个副本可以一次批处理，不需要启动 4,032 次 JVM。

## 4. 最小算法

令冻结 Phase-A embedding 为 \(e_{A}\)，query 为 \(q\)，候选 molecule 为
\(c\)，其参考谱集合为 \(R_c\)。候选检索分数为：

\[
d_A(q,c)=\max_{r\in R_c}\cos(e_A(q),e_A(r)).
\]

真实候选为 \(t\)。只考虑仍位于原生 triplet margin 内的假候选：

\[
h(q,c)=m+d_A(q,c)-d_A(q,t)>0,\qquad m=0.1.
\]

SIRIUS 证据不做加权融合：

- 若 \(f_c\ne f_t\)，只用 fragmentation-tree `TreeScore`。真实分子式必须
  是该 query 候选式集合内严格 top-1，且 \(T(q,f_t)>T(q,f_c)\)。
- 若 \(f_c=f_t\)，只用 `CSI:FingerIDScore`。真实结构必须在该分子式内严格
  top-1，且 \(C(q,t)>C(q,c)\)。

这是一条字典序化学裁决，不把 TreeScore 与 CSI 分数线性相加，不增加需要在
role 2 上调节的权重。每个 query 最多选择当前 Phase-A 分数最高的两个、且被
上述化学证据支持的 active false candidates。

对每个入选候选只生成一个检索对齐事件：

\[
(q,\ \arg\max_{r\in R_t} d_A(q,r),\
      \arg\max_{r\in R_c} d_A(q,r)).
\]

训练目标仍是原生 DreaMS cosine triplet margin loss：

\[
L=\max\{0,m-s(q,r^+)+s(q,r^-)\}.
\]

最终训练池为：

\[
\mathcal P_{V14}=\mathcal P_{PhaseA}\cup\mathcal P_{SIRIUS}.
\]

Phase-A 的 5,956 个事件作为精确保留前缀；新事件只追加。训练从 Phase-A
checkpoint 直接开始，所以已经满足的旧事件通常梯度很小，而 safety/replay
在发生漂移时重新激活。

## 5. 为什么这比 V9--V13 更接近根因

- V9 的 exact proof 只有 23 个 query，问题是原化学规则覆盖不足。
- V13 把有限证据扩成大量参考 pair，增加的是梯度重复度，不是独立化学信息。
- V14 不以 triplet 数量作为目标。一个 query-candidate 化学裁决最多贡献一个
  max-reference 对齐事件，避免 reference multiplicity 冒充证据规模。
- 历史约 `+2.3 pp` 版本证明“Phase-A 初始化 + native max-boundary 续训”这条
  注入通道可以工作，但其增量相对 Phase A 很小且未过严格门；V14 复用的是
  注入通道，不把该历史 checkpoint 当成合格冠军。

## 6. 执行与停止门

源证据阶段（CPU，不占 GPU）：

```bash
sbatch tasks/run_chemaware_sirius_source.sbatch
```

需要服务器已安装 SIRIUS 6 且已完成合法账户登录。若 launcher 不在 `PATH`，
提交时只需提供 `SIRIUS_BIN`；脚本不接受手工内存参数。

完成后直接提交一张 GPU 的原生微调：

```bash
sbatch tasks/run_chemaware_sirius_native.sbatch
```

第二个脚本自动选择最新完成的 SIRIUS score run，直接使用保护的 Phase-A
checkpoint，不重训 Phase A。固定保存 250/500/750/1000 step，role 2 只与
Phase A 做配对选择；若没有正的 formula-cluster CI，停止并保留 Phase A，
不打开 role 3。只有 role 2 通过后才执行一次 role 3 confirmation。

## 7. 目前可声称与不可声称

可声称：SIRIUS 源面板、无泄露候选命名空间、两层证据规则、native triplet
接口和两个 Slurm 入口已完成本地合同测试。

不可声称：V14 已提升 embedding。SIRIUS 真实 TreeScore/CSI 结果尚未在服务器
产生，新增事件数和最终 Recall/AUC 都必须由上述两次运行给出。
