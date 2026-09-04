# MTBLS13729：作者注释、官方 DreaMS、完整工具与生物学增量统一对照

**冻结日期：** 2026-09-01  
**审计工件：** `data/mtbls13729/annotation_biology_benchmark_v1/`  
**核心原则：** 注释覆盖、谱学证据等级、结构准确率和生物学发现是四个不同终点，不能用一个百分比替代。

## 1. 结论先行

原论文在其原生数据处理中，从 9,766 个检测 feature 中注释 345 个，原生注释率为
**3.53%**；若只以 6,054 个带 MS2 的 feature 为分母，则为 **5.70%**。这 345 个包括
157 个 Level 1 和 188 个 Level 2，覆盖 RPLC 与 HILIC。

在我们重新冻结的 RPLC MS1 重定量空间中，共有 16,953 个 target。作者的 190 个 RPLC
注释中，152 行可在 10 ppm / 20 s 内找回，折叠为 141 个当前 target。相同 target 分母上：

| 系统 | feature 数 | 16,953-target 覆盖率 | 证据含义 |
|---|---:|---:|---|
| 作者 RPLC 注释坐标找回 | 141 | 0.83% | 作者身份在当前重提取空间的坐标恢复，不是重跑作者算法 |
| 官方 DreaMS | 3,417 | 20.16% | 候选身份分配，不自动等于 MSI Level 2 |
| E6 噪声微调 shared embedding | 3,426 | 20.21% | 实验性单 formula-fold 共享权重应用 |
| 冻结 P2b | 3,588 | 21.16% | embedding 后候选专家；覆盖扩张，不是新 embedding |
| 三路稳定共识 | 2,162 | 12.75% | 三种方法给出相同身份；优先复核集合，不是结构真值 |
| 三路候选并集 | 3,599 | 21.23% | 候选发现上限；不是一个已校准部署模型 |

因此可以严谨地说：**我们的谱学系统显著扩展了可审计候选覆盖，但不能把 20%–21% 直接与
作者 3.53% 相除后声称“注释准确率提高六倍”。** 作者计数含标准品 Level 1；我们的较大分子
主要是候选覆盖。真正接近质量比较的证据层如下：

| 系统 | Level 2a-supported feature | 相对官方变化 |
|---|---:|---:|
| 官方 DreaMS | 254 | — |
| E6 embedding | **276** | **+22** |
| P2b | 230 | −24 |

E6 的价值是**证据稳定**：总覆盖只增加 9 个，但 Level2a-supported 增加 22 个。P2b 的价值是
**候选召回**：增加 171 个分配，但强证据层减少 24 个。因此当前完整工具必须分层输出：

1. E6/官方共识及 Level2a-supported 作为主要谱学候选；
2. P2b-only 作为候选 lead，不自动升级身份；
3. 三路共识 2,162 个作为高稳定人工复核队列；
4. BioAware 只提供生化上下文、离子家族折叠和冲突弃权，不增加注释分子。

## 2. 各模块对真实生物学成果的具体贡献

### 2.1 作者方法：提供标准与已知通路锚点

作者 345 个注释已经覆盖肉碱、嘌呤/核苷、多胺和脂质背景。因此本项目不能声称首次发现这些
通路。作者 Level-1/HILIC 身份还为 proline、glutamate、Neu5Ac 等正交 RPLC feature 提供了
同队列身份锚点。

### 2.2 官方 DreaMS：扩大原始 MS2 的结构候选空间

官方 DreaMS 在共同 RPLC target 空间分配 3,417 个 feature，其中 254 个达到本项目冻结的
Level2a-supported 谱学证据层。它提供了 palmitoylcarnitine、hypoxanthine、tryptophan 等稳定
谱库共识，也暴露出近异构体和条件漂移风险。

### 2.3 E6 噪声 embedding：提高跨样本谱学一致性

E6 并未靠大量翻转身份获得覆盖，而是把 Level2a-supported 从 254 提高到 276。C20:4-like
acylcarnitine 锚点中，三路身份保持一致，E6 将支持谱最大/中位相似度由约 0.8505/0.8091
提高到 0.8568/0.8166。这是共享 embedding 改进进入生物学工作流的直接证据，但 E6 当前仍是
单 formula-fold 实验模型。

### 2.4 P2b：扩充候选 lead，但不承担高置信身份

P2b 将分配数从 3,417 增至 3,588，新增 171 个候选；但 Level2a-supported 减少 24 个，且在
封存 near-core 上有明确退化。因此 P2b-only feature 进入正交验证队列，不进入主身份分子。

### 2.5 BioAware：没有提高身份率，但完成两项必要工作

BioAware v1 在 MTBLS13729 已知身份小面板为 0 修正/1 新增错误，v2 不干预。因此网络证据没有
被用于抬高注释率。其正面贡献是：

- 表型盲地将 1597/7489 折叠为 methylguanosine ion family、3019/8481 折叠为
  dimethylguanosine ion family，避免把加合物当多个发现；
- 对网络冲突进行弃权，并为候选提供反应上下文；没有网络证据时回退谱学结果。

### 2.6 峰级 MS2、经典谱库和离子家族：把候选变成可审计生物学节点

18 个冻结候选中：9 个是作者身份重映射，3 个是同队列正交 Level-1 recovery，5 个是作者表
未列出的候选离子家族，1 个作者 Level-1 锚点（taurine）因跨面板丰度不一致被主动降级为控制。
所以作者身份字段非空共有 13 个，但可用于正向结论的作者身份锚点只有 12 个。这里真正新增的是
**候选家族与证据链**，不是五个已经标准确认的新代谢物。

同理，6 个候选带有 DreaMS 投票不等于 6 个身份一致：其中 5 个与冻结最终身份一致，排除主动
降级的 taurine 后，只有 4 个属于可用于正向生物学结论的 DreaMS 身份一致节点；phenylalanine
对应 feature 722 的 DreaMS 投票是 synephrine，必须由作者 Level-1 身份覆盖，不能计作 DreaMS
正确注释。经典谱库为 proline、glutamate、Neu5Ac 提供 3 个正交支持；BioAware 对两个 modified-
guanosine 候选完成离子家族折叠，但不升级其位置异构体身份。

## 3. 当前生物学主线与辅助轴

### 3.0 初筛候选不是最终发现：6 个 DreaMS 优先 feature 的生存审计

早期 annotation-plus-statistics screen 给出 6 个优先 feature。经过作者表重叠、targeted-EIC
重新积分和最终 claim scorecard 后，仅 hypoxanthine、tryptophan、carnitine 作为已知身份的
context/general-tumour 节点保留；没有一个可作为新精确代谢物：

- L-kynurenine（41）在初筛中约 +2 log2、DreaMS median cosine 0.935，但作者表已有同 InChIKey；
  targeted EIC 后效应约 +1.1 log2，跨 raw/PQN 的最大 t-test p 约 0.20，故不保留为稳健丰度轴；
- malic acid（486）是作者 Level-1 同坐标找回，但 targeted-EIC 丰度门未通过；
- 3-phenyllactic acid（79）与作者同坐标的 2-hydroxycinnamic acid Level-2 候选冲突，且定量不稳，
  因而主动降级。

这一结果不是“算法失败后删结果”，而是完整系统的必要负向功能：DreaMS 扩大候选空间，峰级/
作者锚点/targeted-EIC 层负责阻止高相似度谱库投票被误写成新代谢物与新机制。

### 3.1 主线：free-Neu5Ac pool 与 donor/destination 解耦

当前最稳定的发现是 feature 703 对作者 Level-1 Neu5Ac 的正交 recovery：

- 10/10 Rmu–RN 患者对为正；
- free Neu5Ac 平均 `+2.249 log2`；
- CMP-Neu5Ac 与 UDP-GlcNAc 的变化分别只有 `+0.556/+0.327 log2`；
- free-minus-CMP 和 free-minus-UDP 分别为 `+1.693/+1.922 log2`，Holm `p=0.0273`；
- 独立患者级 raw-UMI 支持选择性的 AGR2 folding 与 SLC35A1 Golgi donor-transport capacity，
  不支持宿主 NEU1/NEU3 release 或整条通路统一激活。

允许的主结论是 **free-pool expansion with donor/destination decoupling**。不能写 flux、来源酶、
治疗靶点或黏液型特异性已被独立确认。

### 3.2 支持轴：不是强行拼成一条通路

- modified-guanosine family：离子家族折叠后 10/10 同向，平均约 `+2.95 log2`；为作者表未列
  的候选家族，位置异构体未解决；
- acetylated-polyamine axis：1717 有强峰级 MS2 与跨色谱一致性，但 exact positional identity
  仍需标准；
- long-chain acylcarnitine imbalance：3222 为 C20:4-like 类别锚点，支持累积而非直接证明 FAO
  flux；
- expanded amino-acid pool：proline/glutamate 等同队列正交 recovery 构成 10/10 同向模块，属于
  general CRC program，不是黏液型特异机制。

这些轴患者内并不全部协同，因此应作为并行的代谢状态，而不是拼成一个单一上游驱动。

## 4. 论文中必须同时展示的三个注释终点

1. **Native author rate：** 345/9,766 与 345/6,054；
2. **Shared-target candidate coverage：** 作者坐标、官方 DreaMS、E6、P2b、三路共识和并集；
3. **Evidence tier：** Level2a-supported、source Level 1/2 recovery、source-table-absent family、
   downgraded control。

任何只展示 P2b 21.16% 而不展示其强证据下降的图都是不完整的；任何把作者 Level 1 与候选并集
当成同一层比较的写法也是不成立的。

## 5. 当前缺口与下一步

1. 当前同协议三路只覆盖 RPLC。HILIC 尚未完成官方/E6/P2b同候选图对照；在此之前不能给四面板
   总工具注释率。
2. MTBLS13729 没有大规模结构真值；准确率必须引用封存 P3/外部标准品面板，真实应用只报告
   coverage、retained/changed/abstained 与证据等级。
3. full 13,155-target exact FDR10 为0，因此主生物学仍是预注册候选/发现级，不得包装成全空间确认。
4. 最小跨级验证仍是同法 Neu5Ac 标准+spike-in，以及同一样本 linkage-aware glycan readout。
5. 下一轮算法增量优先从 182 个三路并集新增候选和 1,437 个非三路共识候选中，按峰级谱学证据、
   离子家族一致性、跨患者复现和空白/源内碎片风险进行冻结筛选，而不是继续扩大无证据候选数。

## 6. 可复核产物

- `data/mtbls13729/annotation_biology_benchmark_v1/report.json`
- `data/mtbls13729/annotation_biology_benchmark_v1/annotation_rate_comparison.csv`
- `data/mtbls13729/annotation_biology_benchmark_v1/author_rplc_to_current_targets.csv`
- `data/mtbls13729/annotation_biology_benchmark_v1/algorithm_to_biology_module_ledger.csv`
- `data/mtbls13729/annotation_biology_benchmark_v1/frozen_biology_claim_ledger.csv`
- `data/mtbls13729/annotation_biology_benchmark_v1/candidate_method_contribution_matrix.csv`
- `data/mtbls13729/annotation_biology_benchmark_v1/biology_module_method_summary.csv`
- `data/mtbls13729/annotation_biology_benchmark_v1/method_contribution_report.json`
- `data/mtbls13729/annotation_biology_benchmark_v1/initial_annotation_priority_survival.csv`
- `data/mtbls13729/annotation_biology_benchmark_v1/initial_annotation_priority_survival_report.json`
- `data/mtbls13729/annotation_biology_benchmark_v1/annotation_benchmark.png`
- 构建脚本：`tasks/build_mtbls13729_annotation_biology_benchmark.py`
- 校验脚本：`tasks/validate_mtbls13729_annotation_biology_benchmark.py`
