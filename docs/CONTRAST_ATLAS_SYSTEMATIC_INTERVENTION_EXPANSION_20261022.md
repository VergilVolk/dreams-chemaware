# Metabolomic Contrast Atlas：系统性干预家族扩展

日期：2026-10-22  
证据角色：development；任何实际使用后的来源均为 consumed  
总目标：保持研究级生物学响应检索，不把课题缩成未知峰匹配或单一疾病分析

## 1. 为什么扩展干预家族，而不是继续堆通用数据库规模

现有七来源集合已经证明，单纯增加疾病名、性别或年龄等公共元数据标签不能制造生物学真值。Pan-ReDU 的 18 个初步来源家族和 1,258 个计划查询提供了规模，但设计审计显示重复 ontology 轴被年龄、疾病、家庭结构、访问时间或药物暴露混杂。规模不能替代可裁决的响应关系。

因此，当前系统性扩展单元不是“更多 accession”，而是**多个独立来源中由实验设计直接产生的干预响应**。每个来源先在研究内估计方向、幅度和不确定性，再进入开放候选检索。干预名称只用于事后裁决，不进入谱学表示或排序。

这一策略服务于原始 Metabolomic Contrast Atlas 命题：生物学响应能否成为可检索、可积累、可证伪的公共代谢组学知识单元。它不是要求所有干预产生相同代谢物方向，也不是把相似干预名称定义为机制真值。

## 2. 已有 HFD 正控的证据边界

现有 HFD 家族来自两个独立研究/实验室：mouse urine `MSV000097426` 与 female/male mouse plasma `MSV000088023`。在冻结 official DreaMS 表示上，centered response operator 在目标来源四个响应轴中把两个 HFD-vs-normal 对比准确排为 Top-2：

- `all` 分离边际 0.000678；
- `unresolved` 分离边际 0.012180；
- `known` 因目标响应仅有 5 个 known entities 而弃权；
- 31 次响应重排中，`all` 和 `unresolved` 各有 3/31 得到同等或更大边际。

这证明一个 exact-intervention 正控中的响应轴特异性可以跨透明 fragment bins 与 official DreaMS 表示复现。它仍不是校准随机化检验，也没有建立 measurement-matched unknown 增量；obesity 家族的 post-hoc 泛化接近 null。因此必须增加不同干预来源，而不能围绕 HFD 单例优化算子。

## 3. 第二干预来源：MTBLS1066 ketogenic diet

对 MetaboLights 官方 ISA 表与官方 `FILES/` 目录进行真值盲核对后，`MTBLS1066` 通过了研究内响应准入：

- 20 个生物个体，而不是 ISA assay 表表面显示的 7 个组行；
- 4 个 LC-MS/MS 模式，每个模式 42 个文件；总计 168 个 mzML；
- 生物学组：`H_C` 4、`H_MCT` 4、`H_LCT` 5、`BC_C` 4、`BC_MCT` 3；
- 同一真实 `H_C_R1` 正离子 RP 个体文件的前 2 MiB 中已观察到 9 个 MS1 与 43 个 MS2 标签，证明个体样本确有 DDA MS2；该前缀计数不是全文件扫描分母；
- 第一轮只处理 15 个正离子 RP 生物文件，不下载完整四模式 27.8 GiB，以最低成本验证响应对象是否成立。

冻结的两个响应为：

1. `H_MCT - H_C`：健康小鼠的 LCT-MCT8 ketogenic-diet 响应，4 对 4；
2. `BC_MCT - BC_C`：乳腺癌异种移植并接受 cyclophosphamide 背景下的 ketogenic-diet 响应，3 对 4。

第二个响应不是纯饮食效应；它只表达该疾病和治疗背景下的条件响应。

`H_LCT` 不进入 claim-bearing 对比。官方文件名把它与 `H_MCT` 分开，但官方 ISA sample table 却给两者相同的 diet label。在语义冲突被独立解决前，不能按照文件名猜测其生物学含义，也不能把两组拼成更大样本量。

机器可复核产物：

- `tasks/qualify_mtbls1066_ketogenic_axis.py`
- `data/validation/contrast_atlas_mtbls1066_kd_qualification_v1_20261022/report.json`
- `official_file_manifest.tsv`
- `primary_pos_rp_manifest.tsv`

## 4. 预冻结的真实实验

### 4.1 研究内响应构建

第一轮在 15 个正离子 RP 生物文件上执行：

1. 每个样本独立、表型盲地进行 LC-MS feature detection；
2. 在固定 5 ppm / 15 s 窗口构建跨样本 consensus feature universe；
3. 保留非检测为 missing，不把它改成生物学零；
4. 对每个冻结对比计算 log2、样本中位数中心化后的组均值差及 Welch SE；每组至少 3 个有效生物单位且组内 prevalence 至少 0.75；
5. 在表型盲条件下以 10 ppm / 15 s 将 DDA precursor/RT 链接到 consensus features，并为每个 feature 选择一个代表 MS2；
6. 输出两个带 `beta/SE` 的 MS2-linked response objects。

这一阶段只建立研究级响应。Feature、同位素、加合物和源内碎片冗余不等于独立代谢物；离子家族控制和统一 targeted-EIC 重定量是后续确认要求。前体/RT 链接也不等于结构身份。

执行入口：

- `tasks/stage_mtbls1066_primary_pos_rp.py`
- `tasks/build_mtbls1066_ketogenic_response.py`
- `tasks/run_mtbls1066_ketogenic_response.sbatch`

### 4.2 研究内响应轴特异性

两个响应来自同一平台和同一研究，却处在不同生物背景中。它们首先用于检验 centered response 表示是否真正依赖有向效应，而不是只识别共同来源、仪器或 plasma composition。

禁止把二者预登记为“应当相同”。原论文已经指出 healthy 与 breast-cancer/chemotherapy 背景下 KD 调节的 feature 大量不同。正确问题是：方法能否保留共有干预成分，同时表达背景异质性，而不是强迫两条轴合并。

### 4.3 开放候选检索

响应对象通过质量门后，append-only 加入现有七来源集合；原有 18 个对比和排名保持不变。条件名称在完整候选分数、相似/相反方向和弃权状态冻结后才打开。

预登记的裁决仅为：

- ketogenic-diet 响应与 HFD 响应属于 **dietary-intervention relevance family**；
- 不预设相同实体、相同组织、相同饮食配方或相同代谢物方向；
- KD→HFD 的高排名可以证明干预家族检索的相关性，但不能证明共同机制；
- 未登记配对均为 unadjudicated，不作为阴性。

必须在相同候选宇宙比较：

1. composition-only；
2. unsigned effect；
3. signed beta；
4. signed beta/SE centered response；
5. response reassignment；
6. mass/cosine/WSE/official DreaMS/Noise V1 等冻结谱学证据提供者。

如果 response-aware 排名不能区别于 composition 或 reassignment，新增数据不能救回方法。

### 4.4 unknown-information gate

当前响应构建阶段的 identity status 是 `unpartitioned`。在统一、冻结的谱库覆盖搜索完成以前，禁止称任何 feature 为 unknown，也禁止 all/known/unresolved 比较。

分区完成后必须在同一候选全集比较：

- all；
- harmonized known-only；
- measurement-matched known；
- unresolved-only；
- 保持 measurement 的 unresolved-response reassignment。

只有 `all` 在不提高错误风险的前提下新增可独立裁决的正确回答，才能称为未知信息增量。更多 feature、更多回答或更大 margin 本身都不是增量。

## 5. MSV000100699 的当前角色

`MSV000100699` 是第二个独立 ketogenic-diet 项目，但当前不与 `MTBLS1066` 合并为 claim-bearing 跨来源对比：

- pancreas tissue metabolomics 每饮食组仅约 3 个样本；
- 肥胖、PDAC genotype、原始饮食和切换饮食形成多层条件；
- 生物样本采用 Full MS，MS/MS 来自 pooled QC 的 Top-10 DDA；
- MassIVE 定量表用 1–9 数字组编码，尚未从官方可复核元数据建立到生物设计的无歧义映射。

它可以在组映射解决后作为 `MS1 response + pooled-DDA evidence` 的独立来源，但不能被写成 sample-level DDA，也不能因为同样出现 ketogenic diet 就自动成为正确候选。

## 6. 成败标准

这一扩展只有依次通过以下门才继续扩大：

1. **response substrate**：15 个生物文件形成足量、带 SE、带代表 MS2 的研究级响应；否则停止该来源；
2. **response specificity**：response-aware score 相对 composition 和 response reassignment 提供真实区分；否则不进入开放检索主张；
3. **family retrieval**：在固定开放候选集合中，dietary-intervention family 获得可解释且非来源组成驱动的排名；不要求强行 Top-1；
4. **unknown increment**：统一分区后 all 超过两个 known 对照，并保持错误风险；否则只保留全化学响应结果，不宣称未知增量；
5. **independent accumulation**：再加入独立合格来源后，既允许支持增强，也允许异质性或反向证据削弱关系。

这条路线扩大的是可验证的公共响应知识，而不是数据库体量。局部失败只否决该来源、该算子或该关系，不再重定义 Metabolomic Contrast Atlas 的总科学目标。
