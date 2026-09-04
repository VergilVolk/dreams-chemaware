# O-acetyl-Neu5Ac 标准品与谱库资源审计（2026-08-31）

## 一、执行结论

当前公共资源不足以把 MTBLS13729 的两个 `m/z 350.109269` 峰识别为具体的
4-/7-/8-/9-O-acetyl-Neu5Ac 位置异构体。可获得的是化学身份元数据、预测谱和方法学证据；
缺少能直接复用到当前 LC-MS/MS 条件的公开实验参照谱。

因此当前正确身份等级是：

> **mono-O-acetyl-Neu5Ac-like exact-mass features，位置异构体未解析。**

## 二、逐资源审计

| 资源 | 查询 | 结果 | 可用于什么 | 不能用于什么 |
|---|---|---|---|---|
| HMDB | HMDB0000794 / N-acetyl-9-O-acetylneuraminic acid | 有条目；LC-MS/MS 为预测 QTOF 谱 | 候选式、名称和预测碎裂背景 | 不能作为实验 RT/MS2 确证 |
| ChEBI | CHEBI:32844 / 4-O-acetyl-Neu5Ac | 有结构元数据；C13H21NO10，monoisotopic mass 351.11655 | 结构、分子式、InChIKey 对账 | 无同法 RT/实验谱 |
| MassBank | 精确名称与分子式 C13H21NO10 | 当前 API 查询无记录 | 证明当前公开库未提供可直接使用条目 | 不能据“无记录”证明化合物不存在 |
| MoNA | 精确名称与分子式 C13H21NO10 | 当前 REST 查询无记录 | 同上 | 同上 |
| PXD055865 | MUC2 glycopeptide source spectra | 有 glycan-bound O-acetyl-Neu5Ac HCD/ETD 证据 | 载体结合糖肽的结构背景 | 不是游离单糖标准谱或 RT 标准 |
| IM-MS 标准研究 | 合成 O-acetylated sialoside library | 能用离子淌度/CCS 区分乙酰位置和 linkage | 说明正确验证技术路线 | 普通 exact mass + 常规 MS2 仍不足 |

## 三、为何普通 LC-MS/MS 不足

O-acetyl-Neu5Ac 的 4/7/8/9 位异构体具有相同分子式和前体质量，常见碎片也高度重叠。
已有合成标准研究使用 ion mobility–mass spectrometry 和 collision cross section 才实现
乙酰位置及 glycosidic linkage 区分。文献报告的 O-acetylated Neu5Ac B1 离子包括：

- mono-OAc Neu5Ac：`m/z 334.1134`；
- di-OAc Neu5Ac：`m/z 376.1243`；
- tri-OAc Neu5Ac：`m/z 418.1348`。

这些离子可以支持“O-acetylated sialic-acid motif”，但不能单独定位乙酰位置。

## 四、对当前两个峰的影响

MTBLS13729 的两个表型盲 RT 峰：

- RT 约 257.69 s，50/60 样本、47 张 RT-resolved MS2；
- RT 约 333.19 s，54/60 样本、56 张 RT-resolved MS2；
- 均含较强 `m/z 87` motif；
- 均未表现为 Rmu 增加，完整配对 BH q=0.930；
- 均不随 Level-1 free Neu5Ac 在患者内同步变化。

所以它们当前是有价值的阴性结果，而不是待包装的阳性代谢物。

## 五、最小验证采购与实验优先级

### P0：若继续追踪位置异构体

同时购买至少 4-O- 和 9-O-acetyl-Neu5Ac，而不是只买一个标准：

1. 当前 LC 梯度下各自 RT；
2. 同碰撞能量 MS2；
3. pooled extract spike-in 共洗脱；
4. 若两者仍不可区分，停止用普通 LC-MS/MS 争论位置，转 IM-MS/CCS 或衍生化方法。

### P1：若目标是最快补强主论文

优先购买普通 Neu5Ac authentic standard 和同位素内标，而不是先追 O-acetyl 位置异构体。
原因是主阳性现象是 free Neu5Ac 上升；O-acetyl-like 游离峰目前为阴性，验证它不会直接提升主效应证据等级。

### P2：载体层

若可获得替代组织或切片，优先做 linkage-aware O-glycomics、MUC2 glycopeptide 或
lectin/MUC2 共定位。它直接回答 free pool 最终去了哪里，比继续扩 bulk RNA 更有价值。

## 六、引用资源

- HMDB0000794：https://www.hmdb.ca/metabolites/HMDB0000794
- ChEBI CHEBI:32844：https://www.ebi.ac.uk/chebi/CHEBI:32844
- MassBank API：https://massbank.eu/MassBank-api/ui/
- MoNA query API：https://massbank.us/documentation/query
- O-acetylated sialic-acid IM-MS/CCS 方法，PMID 37880209：
  https://pubmed.ncbi.nlm.nih.gov/37880209/
- PXD055865：https://proteomecentral.proteomexchange.org/cgi/GetDataset?ID=PXD055865

## 七、声明边界

谱库查询结果是 2026-08-31 的冻结快照，数据库可能更新。公共库“无记录”只说明当前没有检索到
合格实验参照，不证明标准品不可购买，也不证明化合物在样本中不存在。

