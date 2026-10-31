# Contrast Atlas：独立生物学控制资产与最高性价比顺序

日期：2026-10-31
目的：为固定的研究级响应检索问题建立真实正控、异质性控制和 no-match 控制；不是重新定义 Atlas，也不是预先宣布发现。

## 裁决

现有 15 来源开发集合没有一条可以直接当作“逐实体 signed biological truth”的严格跨母项目关系。疾病同名不等于所有代谢物同向；药物同名也不保证不同菌株、群落、盐型、剂量和时间产生相同响应。

当前最高性价比的下一步不是继续盲目扩大来源，而是冻结两组互补控制：

1. **药物—肠道微生物三干预控制：**Omeprazole 正控、Telmisartan 响应异质性、Sertraline discordant/no-match；
2. **独立 IBD 条件家族：**成人 IBD 粪便项目与独立 VEO-IBD 粪便项目，在表型盲实体连接后检验实际方向，不把疾病名称当逐实体答案。

这样同时检验“应该找回”“只在部分背景找回”和“不能因名称相同强行找回”三种能力。

## 一、药物控制轴

### Omeprazole：首选跨母项目正控

合格来源：

- ChemProp Com20：`MSV000093571` 与 `MSV000094899`，两个 accession 属于同一母项目，不互算独立验证；[原始论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC12889426/)
- Zimmermann/Goodman strain screen：`MTBLS896`，76 个 axenic strains、T0/T12、4 重复、vehicle 和 pH 无菌对照，RPLC ESI+、Agilent QTOF 6550A、公开 auto-MS/MS；[官方数据](https://ftp.ebi.ac.uk/pub/databases/metabolights/studies/public/MTBLS896/) · [原始论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC6597290/)
- 独立 community screen：`MSV000096589`，60 个人源和 29 个动物来源肠道群落、271 种药、9 时间点、4 重复及 pH 4–7 无菌对照，RPLC ESI+ Exploris 240；[MassIVE](https://massive.ucsd.edu/ProteoSAFe/dataset.jsp?task=1ab5cea1a134430a9b0899fe50a278aa) · [原始论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC12871589/) · [样本级结果](https://zenodo.org/records/13943903)

关键限制：`MTBLS896` 中 Omeprazole 在 75/76 菌株达到下降阈值，但 4 个 pH 无菌对照中 3 个也显著下降。因而允许冻结的 signed endpoint 不是简单的 `T12 < T0`，而是：

> 生物培养相对于同时间、相同理化条件无菌对照的额外母药下降。

母药身份轨可以作为正控；未知产物只能在谱学身份/化学关系证据支持后评价，不能预注册为跨研究同一产物。

### Telmisartan：响应者异质性控制

- ChemProp 将其列为低信号挑战；
- `MTBLS896` 中 12/76 菌株显著下降，4/4 无菌 pH 对照稳定；
- `MSV000096589` 中大量轨迹初始斜率为负，但群落间强度差异明显。

它检验 Atlas 能否表达“部分背景响应”，不能作为所有群落普遍同向的金标准。

### Sertraline：discordant / no-match 控制

- ChemProp 中为高信号药物；
- `MTBLS896` 的 sertraline chloride 为 0/76 菌株达到显著转化标准，无菌对照稳定；
- `MSV000096589` 显示明显群落异质性。

它用于防止系统因药名相同就机械宣布响应复现。ChemProp 制剂盐型尚未完全核实，当前只能按 active moiety 登记，不能声称精确盐型一致。

### 药物轴冻结合同

1. 检索前隐藏药名；
2. 同一母项目的多个 accession 只算一个来源家族；
3. 母药 decrease 必须相对同步 abiotic/vehicle control；
4. 身份轨与未知产物关系轨分开；
5. Omeprazole 为正控，Telmisartan 为 heterogeneity，Sertraline 为 discordant/no-match；
6. 不用 ChemProp 既有响应同时调参和验证。

## 二、独立疾病轴

### 首选 IBD 项目

- 成人 IBD：`MSV000084908 / MSV000086509`，同一母项目；粪便 103 CD、60 UC、19 healthy（n=182），Bruker Maxis qTOF、正离子 DDA；[原始论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC9400277/) · [MSV000084908](https://massive.ucsd.edu/ProteoSAFe/dataset.jsp?accession=MSV000084908) · [MSV000086509](https://massive.ucsd.edu/ProteoSAFe/dataset.jsp?accession=MSV000086509)
- VEO-IBD：`MSV000097610`；粪便 21 VEO-IBD、27 healthy（n=48），Q Exactive、正离子 DDA；[原始论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC12864830/) · [MassIVE](https://massive.ucsd.edu/ProteoSAFe/dataset.jsp?task=65cc240951884d4dadbfe2edf4172e03)

两个项目在参与者、母项目和平台上独立，样本类型均为粪便，适合作为条件家族检索。但成人与儿童的年龄、疾病亚型和治疗不同，因此不能预注册“所有实体必须同向”。

正确顺序是：

1. 各项目独立估计 `beta/SE`；
2. 在读取另一项目方向前，冻结身份或化学关系；
3. 预注册“独立粪便 IBD 项目应作为相关条件家族进入候选”，而不是逐 feature signed truth；
4. 只对高置信同一实体评价实际方向一致或相反；
5. 不同实体只能支持化学响应模式，不能合并效应；
6. 已知物、unresolved、统一算法增量分别评价。

### 次选 COVID，暂缓 HFD/AD

两项独立 COVID 血浆项目可形成第二个条件家族，但都包含 SWATH/DIA，需先确认统一谱学接口，性价比低于 IBD：`MSV000092887`（39+39）与 `MSV000089568`（32 severe COVID + 25 healthy）。

HFD 当前项目的基质、设计或论文资格不一致；AD 项目缺完整样本—诊断映射或受控标签。它们不应继续消耗近期主力。

## 三、与当前 known-anchor 负门怎样衔接

当前实体不相交开发检验说明 WSE 只显示粗相关，未通过双中心化特异性和开放 Top-1 门。外部控制不是用来“救回”WSE，而是回答开发矩阵无法回答的生物学问题：

- 正确的响应条件能否被找回；
- 异质性是否被保留，而不是被平均掉；
- 名称相同但响应不一致时是否拒识；
- unresolved 信息是否相对 known-only 增加正确答案；
- 统一算法是否相对 WSE/Official/Noise 改善这些答案。

论文级闭环应为：

> 模型改变一条合法化学连接 → unresolved 证据改变研究级回答 → 正控/异质性/no-match 控制给出预期行为 → 独立 IBD 或药物项目提供未参与调参的确认。

这仍是 Atlas 原始科学问题的实证化，不是新的项目定位。
