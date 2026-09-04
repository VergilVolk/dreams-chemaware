# MTBLS13729：官方 DreaMS、E6 噪声 embedding 与冻结 P2b 的应用对照协议

## 目标与边界

本阶段回答三个不同问题，不再混淆：

1. **工程性能**：冻结 P2b 是否在有结构真值的检索基准上优于官方 DreaMS。
2. **生物学应用**：在 MTBLS13729 无标签队列上，冻结 P2b 是否提高可注释特征覆盖、跨样本结构一致性和可验证的谱学证据，并能否增强既有生物学发现。
3. **embedding 迁移**：噪声微调形成的共享 embedding 是否在真实组织谱图上产生可复核、方向一致的注释变化。

P2b 是官方 DreaMS embedding 后的冻结候选重排专家，不是新的 embedding checkpoint。其开发 OOF 增益为 Recall@1 `+3.91 pp`；封存 P3 主面板为 `+1.07 pp`（公式簇 bootstrap CI 为正，McNemar p=0.0101），但 P3 near-core 为 `-4.23 pp`。因此本阶段必须同时报告总体收益和困难异构体风险，不能宣称全面超越。

E6 `fixed / views=2 / safety=2` 是当前噪声路线唯一通过全部正式门的共享编码器：在一个 held-formula fold 上 Recall@1 `+0.405 pp`，near `+0.498 pp`，修正/新增 `30/6`，公式簇 CI `[+0.102,+0.812] pp`，embedding preservation `0.99517`。它确实改变最后一层 Transformer block 和官方 head，query/reference 共用同一模型，推理只输入干净谱图；但目前只有一个 formula fold，故在应用中标为**实验性 embedding**，不能称为最终超越。历史 `3.85 pp` 是动作 oracle/headroom，不是已经训练进权重的收益。

## 为什么不能复用旧 MTBLS 注释表

旧 `annotations_fdr.csv` 是全谱库先取 Top-10，再计算 precursor 质量误差。首位候选中存在大量远离质量窗的条目，所以它不是真正的质量约束候选图，也不能忠实复现 P2b 的候选组内排序。

新流程固定为：

- polarity-specific `unified_v2` 谱库；
- 在排名之前建立 strict 10 ppm precursor 候选图；
- 官方 DreaMS 和冻结 P2b 使用完全相同的候选谱；
- E6 query 和谱库两端都由同一个 E6 checkpoint 重新编码，禁止跨 embedding 空间计算 cosine；
- P2b 使用冻结公式：`0.1 × DreaMS + 0.1 × entropy + 0.8 × neutral-loss sqrt cosine`；
- 每个候选结构在其参考谱中取最大分数；
- query 没有可靠 adduct 标签，因此应用协议只能使用极性和精确 precursor m/z，必须作为迁移边界报告。

## 无表型 MS1–MS2 桥

只推理能连接到统一复定量 MS1 target 的 MS2：10 ppm、20 s 内最小联合代价。为避免一次进样中的重复 DDA 事件获得额外投票权，每个 `feature × sample` 只保留连接代价最小的一张 MS2。

本地全量预检：

- `neg_rp`：374,232 张 MS2，45,697 个原始连接；去重复后 24,846 个 feature-sample 证据，覆盖 965 个特征。
- `pos_rp`：419,676 张 MS2，116,040 个原始连接；去重复后 86,646 个 feature-sample 证据，覆盖 5,291 个特征。

排序过程完全不读取 Rmu/Rtu/RN、差异丰度、通路或预后标签。三轨比较固定为：

1. 官方 DreaMS cosine；
2. E6 fixed-v2-sw2 cosine（实验性共享 embedding）；
3. 官方 DreaMS + 冻结 P2b（下游候选专家）。

`E6 + P2b` 不列为主结果，因为 P2b 的 DreaMS 分量是在官方 embedding 上冻结选择的，尚未对 E6 空间重新校准。

## 三层验证

### A. 工程真值层

引用冻结 P3，不在 MTBLS 上伪造 accuracy。主面板和 near-core 必须并列呈现。

### B. 真实注释层

MTBLS 没有大规模结构真值，系统变化只能称为：

- retained；
- changed；
- P2b-only / DreaMS-only；
- abstained / conflicted。

每个 MS1 feature 按不同样本中的 MS2 投票形成结构共识。比较注释特征数、Level 2a-supported 数、结构一致率、tier gain/loss，并输出所有强变化候选供人工谱学复核。不得称 changed 为 corrected。

### C. 生物学层

排序冻结以后才能读取表型。优先检查：

1. 预注册坐标锚点（m/z 448.339483，RT 630.436 s，C20:4 acylcarnitine 候选）是否保留或得到更强共识；当前共识表解析为 `pos_rp feature 3222`，但程序始终按 m/z–RT 动态定位，不依赖可能漂移的 feature ID；
2. 新增强共识长链酰基肉碱能否扩展链级面板；
3. 扩展面板的 Rmu–RN 配对丰度方向、精确置换 p 值和 Rmu/Rtu interaction；
4. matched-background 和离子家族去重复后是否仍成立；
5. 仅表述稳态丰度重塑，不表述 flux 或酶活改变。

## 运行与产物

服务器提交：

```bash
sbatch tasks/run_mtbls13729_p2b_application.sbatch
```

核心产物位于：

- `data/mtbls13729/p2b_application_v1/`：官方 DreaMS 与 P2b；
- `data/mtbls13729/e6_embedding_application_v1/`：E6 embedding-only；
- `data/mtbls13729/threeway_application_v1/`：三轨逐谱与逐 feature 对账。

其中包括：

- 两个 panel 的 per-query 决策；
- 官方 DreaMS 与 P2b 的 feature-level 注释；
- 工程基准、协议和哈希报告；
- `comparison/priority_annotation_validation.csv.gz`；
- 预注册 C20:4 锚点对账结果。

后续只有在这些结果通过后，才把冻结注释接入配对 MS1 统计和生物学图表。
