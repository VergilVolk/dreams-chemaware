# 算法整合与生物学当前权威状态（2026-10-09）

状态：仓库证据复核后的单一入口。若旧总览、结果草稿或口头概括与本页冲突，以更晚的实现修复、原始报告和本页列出的证据资格为准。

## 1. 证据纪律

每个数字必须同时携带：数据集、分母、数据角色（development / sealed / final-unopened）、泄漏或重复使用状态、比较基线和不确定性。MassSpecGym 与 GNPS Gold/Silver 已被反复用于开发，均为 `development / consumed`。唯一计划中的最终未开封外测是 Enveda-180；它只能打开一次，且必须发生在统一模型、阈值、候选轴和工件哈希全部冻结之后。

GNPS 双面板为：identity-disjoint `10,995` queries、`87,518` candidate molecules、`175,171` directed spectrum pairs；formula-disjoint `5,261` queries、`24,401` candidates、`47,724` pairs。公式/身份隔离不等于从未接触 GNPS 生态数据。

### 算法整合治理（2026-10-09 起）

算法整合由单一证据账本治理。任何模块、checkpoint、重排器或融合规则进入训练前，必须先登记其来源 SHA-256、训练/评测数据、候选轴、独立性族、适用域、比较基线、开发/封存资格与已知风险。相同证据族的派生分数不得重复计票；失败、撤回或仅审计模块不得因“模块齐全”而复活；无真实样本事件时 BioAware 的值为 unavailable 而不是零分。每次实验须先冻结合同，再报告完整正负结果；无最强单方法增益、风险净收益和 formula-cluster CI，不得升级为统一算法性能主张或使用 Enveda。

## 2. 算法资产与统一模型准入

| 资产 | 当前最强证据 | 数据资格与风险 | 统一模型状态 |
|---|---|---|---|
| official DreaMS | GNPS R@1 `85.35698/86.80859%` | development baseline | 必须保留的共同基线与回退 |
| Noise V1 shared encoder | GNPS `86.55753/88.06311%`，比 official `+1.20055/+1.25451 pp`；双面板 micro AUROC 第一，formula pooled AUROC 第一 | GNPS development/consumed；相对 WSE 的 R@1 未胜 | 当前共享 encoder 主资产；冻结后才可进 Enveda |
| P2b on Noise | GNPS R@1 约 `87.13/88.25%`；formula MRR `0.9290` 第一；identity pooled pairwise AUROC `0.9416` 第一 | development/consumed；sealed P3 main `n=3,000` 为 `+1.07 pp`，但 near-core `n=496` 为 `−4.23 pp` | 仅作 gated 候选重排专家；冻结 composite 是 P2b 家族唯一 enabled 代表，三个原始通道不得与它同模型共现 |
| weighted spectral entropy | GNPS R@1 `87.37/88.27%`，实体找回 5% FDR coverage `67.11%` | 强经典开发基线 | 必须进入最终比较；统一模型先要胜过它 |
| ChemAware Stage-1 encoder | MSG role-3 `n=1,929`，`+1.8144 pp`，CI `[+0.8155,+2.8703]` | role-3 曾作 validation loader；GNPS 仅 `+0.3001/+0.2091 pp` 且 CI 跨 0、risk-net 负 | 保留为独立资产/消融；没有 GNPS 正迁移资格 |
| ChemAware Phase-A encoder | MSG role-2 `n=1,975`，`+2.1266 pp`，CI `[+1.2761,+3.0303]` | provisional；GNPS `+0.2274/+0.3992 pp` 且 CI 跨 0、risk-net 负 | 不与 Stage-1 合并，不作为默认 encoder |
| Noise×Phase-A TSV-Merge | GNPS identity/formula R@1 `86.5666/88.2912%`，相对 official `+1.2096/+1.4826 pp` | development/consumed；相对 Noise V1 仅 `+0.0091/+0.2281 pp`，CI 均跨 0，micro/pairwise AUC 多数下降；未达到稳定 `+3 pp` | 仅作参数合并消融；不得替代 Noise V1，不进入 Enveda；详见 2026-10-10 结果文档 |
| ChemAware V2 residual reranker | MSG role-3 `n=1,929`，`+3.9399 pp`，CI `[+2.8191,+5.1921]`；比 rotated null `+3.3178 pp` | post-outer development；冻结 official-geometry selector 接到 Noise 后，GNPS `n=10,995/5,261` 仅 `+0.4729/+0.1901 pp`，CI 均跨零，risk-net `−181/−117` | 候选级稀疏解释证据；GNPS hard-promotion 路线停止，不作为生产重排器 |
| RRF reranker | official-geometry 探针约 `+1.77 pp`，但 Stage-1/GNPS 确认失败 | audit-only | `retired_audit_only`，不得复活为生产分支 |
| BioAware B47 event module | 4,996 candidate-specific queries、2,239 opportunities；真实事件高于结构化 null | 两个外部研究，truth unopened；集中度和身份质量门失败；无 accuracy | unavailable；补救门通过前权重固定为 0 |

统一融合本身目前没有胜利结果：Noise×Phase-A TSV 在两个固定合并臂中胜过线性平均，但相对 Noise V1 的 `+0.0091/+0.2281 pp` 无统计支持且全局 AUC 多数下降；ChemAware V2 hard promotion 相对 Noise 仅 `+0.4729/+0.1901 pp`，identity/formula 的 corrected/introduced=`285/233`、`137/127`，两个 CI 跨零且 micro/pooled AUPRC 均下降；真值盲无监督融合的 CI 跨 0；监督 router 只有约 `+0.5--0.6 pp` 且 3 个 split 中仅 2 个显著；NNLS v0 相对 WSE 三个 split 分别约 `−0.23/−0.02/−0.25 pp`；candidate-differential Gate-1 corrected/introduced=`660/1120`、risk-net=`−1580`，已停止。S1 oracle `+4.93/+5.44 pp` 和 router v2 `+2.93/+4.03 pp` 因真值/面板重叠泄漏撤回，绝不引用。

因此“严格统一算法”当前首先是逐实体的分层证据系统，不是已证实的新冠军：Noise 提供共享谱图坐标；WSE 保留独立主检索结果；P2b 记录局部谱学竞争；ChemAware V2 记录候选级碎裂化学适用性和解释，但不再 hard-promote；BioAware 只有真实质量门通过的事件才能出现，没有事件上下文时完全缺失。实现见 `tasks/build_unified_entity_evidence_ledger.py` 与 `docs/UNIFIED_ENTITY_EVIDENCE_PIPELINE_IMPLEMENTATION_20261010.md`。该账本不以“模块同意”自动升级结构身份，评价真值不参与输出层级。

## 3. 从封闭候选池到实体找回

修复后的 GNPS open-set 开发实验包含 `7,695` match 与 `3,300` no-match queries，阳性分子从 no-match 候选中移除，并在混合总体上按 margin 校准。5% FDR 下以 7,695 个 match 为 coverage 分母：official `55.36%`、Noise `58.36%`、P2b-on-Noise `65.65%`、WSE `67.11%`。这说明当前最强实体找回基线仍是 WSE，也说明 Top-1 小候选池胜负不能代表真实发现能力。

P0-L 的跨仪器结果（GNPS consumed development，`n=3,420`）为 Noise R@1 `84.386%`、official `82.749%`，差 `+1.638 pp`；它支持坐标可迁移性，但不是独立外测。P0-G 旧 no-match AUC 因阳性未移除而失效。P0-M 的 margin AUC 约 0.848 仍是开发线索，但 60% coverage 下 selective risk 约 13.56%，尚非部署级结果。

## 4. 生物学资产与可主张边界

### 4.1 LCNEC

- 冻结实验对象：34 对 tumor/adjacent 患者；30 个 phenotype-blind、QC 合格的优先谱学实体。法证后为 21 个真实成分和 9 个 blank-positive 污染物。
- 暗模块反向搜索：30 个模块中 1 A、4 B1、3 B2、22 C；但所有 top hit 与候选 `[M+H]+` 均相差 23.8--6963 ppm，故分子级同质量命中为 `0/30`。A/B 的 8 个是真实信号，不等于结构鉴定；含氟/合成库邻居首先提示暴露或谱巧合。
- 全局重塑 Stage-1d：在 263 个冻结实体的 mutual kNN(k=3) 图上，34 对患者的分解为 family `0.3222`、within `0.5082`、isolated `0.1696`；208 edges，最大组件占 `0.0875`，患者 bootstrap 稳定，`pass_to_stage2=true`。它只说明该分解值得做否决性复现。
- 尚未完成：Stage-2A LIPn 独立平台复现、Stage-2B known-only 对 all-feature 的未知实体增量、Stage-2C 转换边一致性。仓库无 Stage-2 结果，近期 P2 作业也明确没有真实 LIPn 分支。
- 已失效：`lcnec_p2v2/run_2354362` 的 23 个 medium/high 和 91% dark 数字来自错误 entropy 实现，必须重算；不能作为论文结果。

允许的当前主张是“表型相关的全局化学重塑分解在单平台、患者重采样下稳定，且存在真实但未获分子结构的谱学实体”。不能写化学通量、具体酶活、跨平台复现或已识别暗代谢物。

### 4.2 MTBLS13729

- 作者原生注释为 `345/9,766=3.53%` detected features，或 `345/6,054=5.70%` MS2-bearing features；不能与算法候选覆盖直接相除。
- 共同 RPLC target 分母 `16,953`：作者坐标找回 141 (`0.83%`)，official 3,417 (`20.16%`)，E6 3,426 (`20.21%`)，P2b 3,588 (`21.16%`)，三路共识 2,162 (`12.75%`)，并集 3,599 (`21.23%`)。这些是候选覆盖，不是结构准确率。
- 更接近证据质量的 Level2a-supported 数为 official 254、E6 276、P2b 230。E6 的增量是证据稳定性；P2b 扩大候选覆盖却降低强证据层，因此 P2b-only 只能做验证 lead。
- 当前最稳的生物学节点是正交找回的 free Neu5Ac feature 703：Rmu--RN `10/10` 同向，均值 `+2.249 log2`；free-minus-CMP 与 free-minus-UDP 分别 `+1.693/+1.922 log2`，Holm `p=0.0273`。允许写 free-pool expansion with donor/destination decoupling；不能写来源酶、通量或全局 hypersialylation。
- C20:4-like acylcarnitine（坐标锚点，当前 feature 3222）只支持稳态丰度累积；Rmu--RN 丰度不等于 Rmu--Rtu interaction，更不证明 FAO flux 或酶活性。
- 该队列目前是同患者正交平台和算法辅助的证据校准，不是独立患者队列的疾病复现。候选级三路明细若本地仍缺失，不得从汇总计数反推单个 feature 的算法归属。

### 4.3 BioAware

B44 external `−0.56 pp`，B45 coverage-neutral topology `+0.2326 pp` 且 CI 跨 0，B46 dual-context `−0.7714 pp`，共同关闭静态目录成员、degree/topology、path/diffusion 和 aggregate-context 路线。B47 U3-v3 的 2,239 是潜在排序机会，不是 corrected，更不是 `+4.31 pp`。仓库中只有 2026-10-03 的 R1--R5 补救预注册，未见执行结果；`pass_to_frozen_event_ranking_evaluation=false` 继续约束揭盲。

## 5. 真正超出“注释—差异—买标准品”的主线

论文应用方法的统计对象应是冻结的实验谱学实体及其在新条件中的独立找回，而不是库名。最小可证伪链为：

1. 在 discovery QC/blank/dilution 中表型盲冻结实体原型，污染物保留为显式负对照而非生物实体；
2. 在独立仪器/条件中做 hidden-name entity retrieval，使用 match/no-match、FDR-coverage、known-only 与 known+unknown、候选数/QC/检出率匹配和 shuffled mapping；
3. 在独立患者或真实扰动数据中检验未知实体是否带来超出 known-only 的效应找回；同患者换平台只能称技术复现；
4. 只有在 disease-perturbation prospective loop 中，冻结实体、方向和 readout 后复现，才升级为机制或可操作生物学。

若第 2 步失败，停止该应用线；若第 3 步未知实体无增量，只能称检索工程扩展；若没有第 4 步，不得写机制发现。表型条件化转换程序可作为候选方法，但必须先通过跨底物留出、真实扰动恢复、LCNEC 跨平台复制和前瞻干预；PMD 加谱图阈值本身不构成主创新。

## 6. 当前执行顺序

1. 保留旧作业 `2354608` 作为构建审计；完整同步 prepare、builder、consumed-source registry 与 sbatch 后，以新作业重新生成 fail-closed Enveda-180 面板，不打开任何模型分数。
2. 在 GNPS development/consumed 上冻结唯一统一模型：最强单法基线、条件分支、缺失语义、abstention、open-set 阈值和全部哈希一次定死。
3. 一次性打开 Enveda，按预注册全指标报告；失败也不回调 Enveda。
4. 重算 LCNEC P2（固定 entropy、严格质量轴），但不把谱库邻居当结构身份。
5. 执行 LCNEC Stage-2A/B/C；没有真实 LIPn 与患者级效应联结时不做替代性伪复现。
6. B47 仅先执行 R1--R5 补救审计；全部通过后才允许一次性揭盲。
7. 将最终冻结算法用于 entity recovery 和独立效应复现，论文主结果从“候选池 R@1”推进到“未知实体是否带来可复现生物学增量”。

## 7. 明确撤回或降级清单

- 撤回：S1 oracle `+4.93/+5.44 pp`；router v2 `+2.93/+4.03 pp`。
- 失效待重算：LCNEC P2 v2 的 23 medium/high、91% dark；P0-G 旧 no-match AUC。
- 降级为开发零迁移：ChemAware Stage-1/Phase-A 的 GNPS 正点估计。
- 降级为机制探针：RRF 1.7；oracle complementarity；post-hoc RRF60。
- 未执行：Enveda 模型分数、LCNEC Stage-2、B47 gate remediation 后的 truth evaluation、独立患者 unknown-entity gain、prospective perturbation loop。
