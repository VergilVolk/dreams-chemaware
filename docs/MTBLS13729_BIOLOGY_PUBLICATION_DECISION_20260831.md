# MTBLS13729 生物学部分投稿决策与最短闭环（2026-08-31）

## 一、执行结论

当前不是“所有尝试都失败”，也不是已经完成因果机制。项目已经收敛出一个可发表的主现象：

> **mucinous-relative hybrid mucin glycome**：Rmu 中可提取 free Neu5Ac pool 增大，并伴随
> Neu5Ac donor/transport 与 secretory-mucin 程序相对富集；但糖链层表现为 core-3/Sda 黏膜
> 谱系相对保留、core-2/sLeX 肿瘤相关糖抗原获得和 alpha2-6 丢失，而非全局高唾液酸化。

它目前支持一篇 **algorithm-enabled, evidence-calibrated clinical discovery**。若增加 Neu5Ac 同法
标准和 linkage-aware O-glycomics，可以升级为结构验证充分的临床糖组重塑论文；只有再加入
同位素、节点扰动和 rescue，才能称因果代谢机制。

## 二、主结果的证据链

| 层 | 当前证据 | 强度 | 不可外推 |
|---|---|---|---|
| 本地丰度 | 锁定 targeted-EIC：Rmu–RN 10/10 正向，均值 +1.935 log2，约3.82倍 | 强 | 不是全13,155-target FDR确认 |
| 亚型敏感性 | `(Rmu-RN)-(Rtu-RN)`：raw/PQN +2.209/+2.142 log2，五模块BH q=0.00179/0.00162 | 强 | n=10 Rmu，属于发现级 |
| 身份 | feature703 与 source Level-1 Neu5Ac 的同样本/患者配对相关约0.959，source rank1 | 强桥接 | 不是当前色谱同法标准终证 |
| 同患者 donor 分解 | HILIC(-) 中 free Neu5Ac 10/10升高、均值+2.249 log2；相对CMP-Neu5Ac/UDP-GlcNAc差值+1.693/+1.922 log2，Holm p均0.0273 | 强发现证据 | 同队列静态丰度；CMP-Neu5Ac为Level 2；不是通量或独立复制 |
| 来源机制分支 | 一般CRC的NEU1/NEU3轴升高，但mucinous相对conventional显著降低；CMP activation/transport RNA相对升高 | 有区分力的反证 | bulk RNA不能排除蛋白活性、其他sialidase、微生物或分泌周转 |
| O-acetyl精确质量分支 | 表型盲冻结4.29/5.55 min两个m/z 350.109峰；103张RT分层MS2多数含m/z 87；Rmu配对BH q均0.930且不随free Neu5Ac变化 | 有价值的负结果 | 未定位4/7/8/9-O-acetyl异构体；不测glycan-bound或细胞型特异O-acetylation |
| donor/carrier RNA | 42 MUC vs 329 conventional：donor beta +0.480、q=3.30e-8；secretory mucin +0.922、q=5.34e-11 | 强背景 | TCGA与既有分析重叠，不是独立代谢物复现 |
| NXPE1 carrier context | current-GDC中mucinous vs conventional的lineage beta +0.621、p=0.000369；加入五标志secretory程序后beta +0.064、p=0.734；一般CRC 50对中47/50 tumour较低 | 有区分力的载体状态证据 | 不是独立驱动、因果中介、蛋白活性或O-acetyl产物证据；GSE236696仅作低计数方向支持 |
| core/linkage RNA | core-3/Sda +0.879、q=1.76e-8；ST6GAL1 -0.742、q=8.50e-5；core-2/sLeX composite不显著 | 强分支反证 | RNA不是糖链结构或酶活 |
| 外部患者结构 | 2例MUC：core-2和sLeX/A为肿瘤队列最高端，alpha2-6最低；配对方向一致 | 有价值的结构支持 | n=2，不测free Neu5Ac，不是独立丰度复制 |
| 空间载体 | PXD055865中3块MUC标本来自2位患者；人工复核列表与source spectra确认MUC2 glycoform、O-acetyl-Neu5Ac及putative O-acetyl-GalNAc证据，并呈空间异质 | 机制边界支持 | Colon1a/1b非独立患者；鉴定数不是丰度；无非MUC人群对照，不能作亚型统计或free-Neu5Ac复现 |
| 外部患者空间代谢物 | 372对CRC–正常黏膜；Neu5Ac为HILIC(-) Level 1；正常梯度+0.349、p<0.001，肿瘤+0.088、p=0.091 | 强空间背景 | 无mucinous字段，不是Rmu独立复制；补充分期表合计374与方法372不一致 |
| 外部转录sialyltransferase背景 | 980例有组织学信息：Sialyl-High占mucinous 85/154、non-mucinous 238/826；重算OR 3.04（95% CI 2.14–4.33） | 强背景 | 20基因转移酶分数，不测Neu5Ac/糖链；含TCGA重叠且未按队列分层 |

公开网页的患者级复算不能提高这一证据等级：Dash 回调每种组织只返回371个值、每个亚部位固定53个，
没有患者或配对键；直接回归也不能复现补充表。因此外部统计只引用正式补充材料，网页仅作方向性
可视化背景，详见 `MTBLS13729_EXTERNAL_NEU5AC_DASH_REPRODUCIBILITY_AUDIT_20260831.md`。

## 三、真正的新颖性

不新颖或不能宣称：

- Neu5Ac 在 CRC 中首次出现；源论文已提到 Neu5Ac；
- CRC 或黏液癌存在 sialylation；已有大量文献；
- 由 free Neu5Ac 直接推断 ST6GAL1–PD-L1 或某个 MUC2 glycoform；
- 用 bulk RNA 证明 glycan flux。

可主张的新颖性：

1. DreaMS/P2b 驱动的原始谱图重分析把一个弱注释条目恢复为可审计的峰、MS2、患者配对丰度和
   身份桥；
2. 严格分开 Rmu 主效应与 Rmu-vs-Rtu interaction 后，Neu5Ac 是唯一跨归一化通过亚型门的充分
   覆盖模块；
3. 患者代谢物、TCGA 分支和独立 O-glycomics 共同揭示
   **donor–carrier–core–linkage decoupling**；
4. 同患者 free Neu5Ac 的升幅显著超过 CMP-Neu5Ac 和 UDP-GlcNAc，把 donor 解耦从转录推断
   推进为静态代谢物层的直接比较；
5. 预定义RNA分支排除了“黏液型NEU1/NEU3转录上调直接产生free pool”的简单解释，并暴露
   SLC35A1/CMAS转录能力与实测CMP-Neu5Ac pool的失配；
6. core-3 的肿瘤间“相对保留”与 tumour-normal“绝对下降”得到参照系对账，避免了常见的
   方向性误读；
7. 负证据被保留：ST6GAL1/alpha2-6、BioAware v1、full-space FDR、空间不共定位和非Neu5Ac平行
   模块均未被选择性隐藏；两个mono-O-acetyl-Neu5Ac-like精确质量峰也没有被包装成阳性结果。

## 四、三套投稿包

### Package A：无新湿实验，立即可完成

定位：算法增强、证据校准的临床发现。

必须包含：

1. 官方 DreaMS、最佳 embedding/候选专家和源论文注释的同协议比较；
2. feature703 原始 EIC、峰界、MS2、跨面板身份桥和三套定量协议对账；
3. primary paired endpoint 与 subtype interaction 并列，明确 full-space exact-FDR10=0；
4. TCGA donor/carrier/core/linkage 分支模型与 composition/MSI sensitivity；
5. 外部 MUC O-glycomics 的患者级配对和肿瘤间排名；
6. PXD055865 的 MUC2 carrier/destination 审计，明确三块标本仅来自两位患者且鉴定数不作丰度；
7. BioAware v1、失败候选和冲突外部队列作为主文或扩展数据；
8. 标题、摘要和图注统一使用 `abundance`、`selective remodeling`、`hybrid glycome` 和
   `mechanistic context`。

建议标题骨架：

> Evidence-calibrated reanalysis of paired colorectal tissue metabolomics reveals a hybrid mucin
> glycome in mucinous colorectal cancer

### Package B：最小湿实验，性价比最高

1. 当前 LC 方法下 Neu5Ac authentic standard 的 RT+MS2；
2. pooled sample 与至少若干代表样本 spike-in 共洗脱；
3. 最好加 isotope internal standard，给出相对或半绝对定量；
4. 同一批样本或可得替代组织上的 linkage-aware O-glycan panel，至少同时测：
   core-2/sLeX、core-3/Sda、sialyl-Tn、alpha2-3、alpha2-6；
5. 若组织量有限，优先 MUC2/StcE glycopeptide 或 lectin + MUC2 共定位，而不是再做一套 bulk RNA。

这套补强直接击中目前最薄弱的两层：同法身份和 glycan destination。

### Package C：因果机制论文

在 Package B 之上增加：

1. 在独立组织中同一样本复核 free Neu5Ac、ManNAc、authentic-standard CMP-Neu5Ac；
2. isotope-labelled precursor incorporation 和时间维度；
3. GNE/NANS/SLC35A1、ST6GALNAC1/GCNT3 或候选 sialidase 的遗传/药理扰动；
4. glycan readout、MUC2 carrier readout 和细胞表型；
5. 回补/救援与体内或类器官验证。

没有这些证据，不使用 `drives`、`flux`、`enzyme activation` 或 `therapeutic target`。

## 五、Figure 结构

1. **Figure 1：算法到真实样本。** 三方注释覆盖、冲突、置信度；feature703为何被恢复。
2. **Figure 2：患者内丰度与亚型门。** 10对Rmu折线/EIC，Rmu主效应与Rmu-vs-Rtu interaction
   分开，full-space分母同时展示。
3. **Figure 3：身份证据。** 原始MS2、跨面板source-Level-1 bridge、失败异构体/反例。
4. **Figure 4：hybrid mucin glycome。** free pool、activated donor、carrier、core/linkage分层
   并列；加入10位患者的free Neu5Ac/CMP-Neu5Ac/UDP-GlcNAc配对差值和外部O-glycomics；
   全部箭头为证据汇合而非因果。
5. **Extended Data：** 全候选总账、归一化敏感性、full-space exact检验、外部队列冲突、
   BioAware负结果、两个mono-O-acetyl-Neu5Ac-like精确质量峰的表型盲负结果、TCGA
   composition/MSI敏感性和可复现哈希；新增pool–donor–carrier边界图，显式把PXD055865
   鉴定存在性与本地丰度分开。

## 六、停止继续消耗的低价值方向

- 再增加仅有 bulk RNA 的 CRC 队列；
- 对 BioAware v1 进行无新种子/无新真值的阈值扫描；
- 用 pathway enrichment 扩充候选名单；
- 把低覆盖大效应 feature 1597/3019/3222 提升为亚型主轴；
- 用外部 n=2 O-glycomics 做显著性包装；
- 在没有同位素的情况下继续争论 flux 增强还是降解减少。

## 七、工件与可追溯性

- 主轴审计：`docs/MTBLS13729_NEU5AC_HYBRID_GLYCOME_AUDIT_20260831.md`
- 主结果：`docs/MTBLS13729_BIOLOGY_MANUSCRIPT_RESULTS_V2_20260830.md`
- 论文蓝图：`docs/MTBLS13729_BIOLOGY_MANUSCRIPT_BLUEPRINT_20260830.md`
- 机制对标：`docs/UNTARGETED_MSMS_MECHANISM_PAPER_BENCHMARK_20260830.md`
- TCGA分支：`data/external/TCGA_COADREAD_Xena_20260830/glycan_branching_v2/`
- 外部O-glycomics：`data/external/CRC_Oglycomics_PMC9254241_20260830/`
- 外部Neu5Ac生物地理：`data/external/CRC_metabolic_biogeography_PMC11438248_20260831/neu5ac_biogeography_audit_v1/`
- 主图与源数据：`data/mtbls13729/neu5ac_glycan_publication_figure_v2_final/`
- 外部sialyltransferase–mucinous审计：`docs/MTBLS13729_EXTERNAL_SIALYLOME_MUCINOUS_AUDIT_20260831.md`
- 同患者donor解耦审计：`docs/MTBLS13729_SIALIC_DONOR_DECOUPLING_AUDIT_20260831.md`
- 来源机制分支审计：`docs/MTBLS13729_SIALIC_POOL_MECHANISM_DISCRIMINATION_20260831.md`
- O-acetyl-Neu5Ac-like审计：`docs/MTBLS13729_OACETYL_NEU5AC_LIKE_AUDIT_20260831.md`
- PXD055865 MUC2糖肽审计：`docs/MTBLS13729_PXD055865_MUC2_GLYCOPEPTIDE_AUDIT_20260831.md`
- O-acetyl-Neu5Ac标准/谱库资源审计：`docs/MTBLS13729_OACETYL_NEU5AC_STANDARD_RESOURCE_AUDIT_20260831.md`
- 原始单细胞上皮组成诊断：`data/external/GSE178341_mucinous_secretory_audit/epithelial_composition_diagnostic_v1/`
- 统一生物学主图v3：`data/mtbls13729/neu5ac_evidence_convergence_figure_v3_final/`
- O-acetyl精确质量审计图：`data/mtbls13729/oacetyl_neu5ac_like_figure_v1/`
- Pool–donor–carrier边界扩展图：`data/mtbls13729/pool_carrier_boundary_figure_v1/`
- 完成度总账：`data/mtbls13729/mechanism_paper_completion_audit_v9_final/`

## 八、独立患者层转录与蛋白裁决（2026-08-31）

GSE178341原始10x UMI的患者级pseudobulk提供了目前最清楚的独立细胞来源背景。固定队列为
6例纯黏液型和53例纯常规型肿瘤，所有统计单位均为患者。12基因固定面板中，肿瘤上皮
AGR2和SLC35A1分别增加`+1.613/+0.833 log2(CPM+1)`，全肿瘤固定面板BH q均`0.0068`，
在右侧结肠/MMR分层敏感性中BH q均`0.0179`。SPDEF方向很强但分层后未通过统一门；
GNE只有趋势，NANS/CMAS没有支持。

七个预注册细胞来源端点中，只有上皮secretory-carrier和上皮CMP-Neu5Ac capacity通过门：
效应为`+0.917/+0.687 z`，bootstrap下界均大于零，BH q均`0.0627`，匹配分析5/6病例为正且
删除任意一个黏液型病例后均值保持正向。上皮和髓系NEU1/NEU3 release轴均无正支持；前者
方向反而为`-0.547 z`（BH q=`0.160`）。因此外部raw RNA支持的是**选择性的上皮分泌折叠和
Golgi CMP-Neu5Ac运输能力**，不是宿主sialidase release或整条通路统一激活。

NXPE1在广义上皮中为`+0.837 log2(CPM+1)`，但固定面板q=`0.229`；冻结匹配只有4/6病例为正，
加入secretory composite后HC3 mucinous系数降至`+0.242`、p=`0.706`。故NXPE1继续被定位为
carrier-state相关容量标志，而非独立驱动。完整结果见
`docs/GSE178341_RAW_UMI_MUCINOUS_SIALIC_AUDIT_20260831.md`。

已对独立的15例黏液型、15例常规型CRC和16例正常结肠蛋白组实施预注册固定面板审计。该审计
没有确认四蛋白secretory/mucin模块或四蛋白sialic-handling模块：两模块的MC-vs-AC差异均约
`+0.21 z`，bootstrap区间跨零，BH q均为`0.449`。因此它不能作为独立通路复现。

单蛋白层仍有可解释但受限的方向信息：AGR2、GNE和NANS在MC-vs-AC中分别为
`+0.897/+0.565/+0.502 log2`，且删除任意一位MC患者后方向始终为正；但三者的患者bootstrap
区间均跨零，固定八蛋白BH q均为`0.643`。CMAS和SIAE没有正支持。原论文把AGR2列为
`MC_AC_up`可以复现为原始尺度算术均值FC=`2.60`、Welch p=`0.0478`；在本项目冻结的log2、
置换和bootstrap口径下则为p=`0.161`。正确表述是**稳定正方向且显著性依赖尺度**，而非独立确认。

这一结果使主模型更窄而非更弱：它与选择性的secretory folding/upstream synthesis背景相容，
同时反对“整条Neu5Ac通路在黏液型CRC中统一上调”。蛋白矩阵还存在明显左删失；TFF3有37/46
个值等于全表最小值，不能承担独立丰度证据。完整审计见
`docs/MTBLS13729_INDEPENDENT_MUCINOUS_PROTEOMICS_AUDIT_20260831.md`。

## 九、最终裁决

最值得钻研的不是一个孤立代谢物，也不是一条被转录数据强行串起来的通路，而是一个经过正反证
收敛的分层现象：**free Neu5Ac pool、黏蛋白载体、糖链核心和末端连接在黏液型CRC中并不同步。**
这是当前最可发表、最不容易被审稿人一问击穿、也最适合由最少实验升级的生物学主线。完成度
总账已升级为25门的`data/mtbls13729/mechanism_paper_completion_audit_v10_final/`；Package A
保持可投稿，但独立Neu5Ac丰度复制、同法标准、同一样本glycan destination和因果实验仍未补齐。

## 十、上皮组成诊断后的最终生物学措辞（2026-08-31）

患者级raw-UMI结果现已进一步区分组成与状态。作者标注goblet-lineage epithelial fraction在
6例黏液型中平均高于53例常规型`+0.132`，但bootstrap区间跨零、冻结匹配仅4/6为正。组成调整
后MUC2、SPDEF和NXPE1均不再有独立区间支持；AGR2和SLC35A1仍分别保留`+1.172`和`+0.676`
的黏液型系数，HC3 95% CI均高于零，并在加入right-colon/MMR后保持。

因此主文的最终短句冻结为：

> Mucinous CRC shows a larger extractable free-Neu5Ac pool embedded in a partly
> composition-driven secretory lineage, with residual epithelial AGR2-mediated
> folding and SLC35A1-mediated Golgi donor-transport capacity; the data do not
> support uniform pathway activation, a host NEU1/NEU3 release mechanism, or an
> independent NXPE1 driver.

组成调整是post-result、解离敏感的诊断，不是因果中介分析。其结果不能升级Package A为因果机制
论文，但显著降低了“所有转录信号只是goblet细胞更多”这一审稿攻击的风险。
