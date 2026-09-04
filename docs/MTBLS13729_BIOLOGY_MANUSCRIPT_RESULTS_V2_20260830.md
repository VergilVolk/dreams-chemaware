# MTBLS13729 生物学论文结果稿 v2（2026-08-30）

## 结果 1：统一算法协议扩大了真实样本候选覆盖，但没有结构真值时不能称准确率提高

在相同的 10 ppm 质量候选图和 one-MS2-per-feature/sample 协议下，官方 DreaMS、实验性 E6 共享 embedding 与冻结 P2b 在 MTBLS13729 原始谱图上完成三路推理。负相 RPLC 中三种方法覆盖的 feature 数均为 345；正相 RPLC 中官方 DreaMS、E6 和 P2b 分别注释 3,072、3,081 和 3,243 个 feature。P2b 因而比官方多给出 171 个 feature-level 候选，但同时出现 47 个 tier gained 和 370 个 tier lost；在负相 RPLC 中也只有 8 个 tier gained、32 个 tier lost。E6 是改变 embedding 的实验模型，P2b 是 embedding 后的候选专家，两者不得混为同一算法改进。

这些变化说明候选空间和证据排序发生了实质改变，却不提供无标准条件下的 correctness 标签。真实样本中只能使用 `retained/changed/abstained`：负相 query 的 official–P2b 为 8,362/2,720/8，正相为 32,406/16,429/11,740；官方–E6 的 changed query 分别为 366 和 2,185。服务器端三路明细尚未完整同步到本地，当前数字来自已保存的冻结运行日志 `mtbls13729_p2b_2326596.out`；逐候选 official/E6/P2b source-concordance 审计必须等待 `threeway_application_v1` 目录同步后运行，不能从汇总日志反推。

该限制对主轴的归属尤其重要。feature703 Neu5Ac 是原论文 negative-HILIC Level-1 身份在正相 RPLC 的正交找回，并有经典谱库 median cosine `0.9054` 和31个强支持样本；它不是 E6 或 P2b 首次发现的新身份。冻结 integrated ledger 中该 feature 的官方 DreaMS 身份字段为空，而当前本地又缺少三路 feature-level 明细，因此不能把 Neu5Ac 的命名归功于任一新模型。算法的可辩护贡献是统一候选协议、改善候选覆盖或证据稳定性，并将身份候选接入患者配对定量与亚型审计。完整归属矩阵见 `docs/MTBLS13729_ALGORITHM_TO_BIOLOGY_INCREMENT_AUDIT_20260831.md`。

## 结果 2：证据校准把 345 条原论文注释转化为四类可审计增量

原论文补充表包含 345 条 UHPLC 注释，其中 Level 1 为 157 条、Level 2 为 188 条。扩展后的选择性证据面板包含 18 个 feature：9 个 source-identity remap、3 个 Level-1 身份的正交 panel recovery、5 个 source-table-absent ion-family candidate，以及 1 个主动降级的 control。由于 18 个候选经过表型和证据筛选，不能与原论文 345 条全表直接相除，也不能据此声称全局 annotation rate 提升。

真正可审核的算法增量包括三类。第一，已知源身份在另一色谱/极性中获得 raw peak-resolved MS2 与同样本丰度桥；第二，原表未给出精确身份的 methylguanosine、dimethylguanosine、long-chain acylcarnitine 和 acetylated-polyamine 等离子家族得到可复核候选；第三，taurine、leucine/isoleucine-like、proline sodium-like 和 5-HIAA-like 等看似合理身份在患者配对、质量或谱库竞争审计中被拒绝或降级。

## 结果 3：proline、glutamate 与 Neu5Ac 在正相 RPLC 中被正交找回，但其新颖性必须相对原文分层

三个原论文 Level-1 代谢物在正相 RPLC 中形成新的强身份桥。feature 345 proline 在 10 个 Rmu 患者中 10/10 升高，平均 `+1.299 log2`，患者 bootstrap 95% CI `[1.051, 1.559]`；94 张 MS2 覆盖 59 个样本，MassSpecGym median cosine 为 `0.9988`，诊断性 m/z 70.0651 在 94/94 谱中出现。它与原论文负相 HILIC proline 的 within-tissue rho 为 `0.872`、paired-delta rho 为 `0.814`，在 70 个候选中排名第 1。

feature 374 glutamic acid 平均 `+0.715 log2`，10/10 升高，95% CI `[0.423, 1.049]`；paired-delta rho 为 `0.849`，源身份 rank 1。feature 703 Neu5Ac 平均 `+1.975 log2`，10/10 升高，95% CI `[1.331, 2.607]`；33 张 MS2 覆盖 33 个样本，median cosine `0.905`，paired-delta rho `0.959`，源身份 rank 1。

feature 703 的两个丰度协议必须分开报告。跨面板审计使用 `log2(targeted-EIC+1)`，得到 10/10 正向和均值 `+1.975 log2`；与 full-space EIC 审计完全一致的 detection-masked/pseudo-count 协议同样为 10/10 正向，均值 `+1.935 log2`。候选亚型表使用早期 discovery peak-picker matrix，P24 在该矩阵中缺失，故只有 9/9 正向、均值 `+1.881 log2`。主图和定量主结果采用锁定 targeted-EIC；9/9 结果只作为缺失规则敏感性分析。三种口径的方向与 Rmu-vs-Rtu 交互结论一致，不得把其 n、均值和 p/q 值跨协议拼接。

这些结果属于同队列正交技术恢复，不是新代谢物发现，也不是独立患者复现。对原论文正文的逐轴审计进一步显示：Neu5Ac 与长链肉碱已经被正文明确点名；proline/glutamate 只出现在 pathway/family context 中，而非展开的分子机制叙事。因此 Neu5Ac 的增量是另一面板中的 raw-MS2/丰度桥及其机制边界校准，不是首次发现。相对更强的 narrative increment 是原文正文未展开的 acetylated-polyamine，以及原 identity table 未列出的 modified-guanosine ion families。正文缺席仍不等于精确化学新颖性，后者必须由标准品裁决。

feature 301 虽与 proline 共洗脱，但 `[M+Na]+` 质量误差和竞争谱库证据冲突；feature 1695 虽有 leucine/isoleucine-like MS2，却未复现原论文 leucine 的患者配对变化。反例证明身份门不是只会接受候选。

## 结果 4：Rmu 形成多个平行 abundance programs，而不是一条统一通路

候选折叠后，乙酰化多胺–MTA、嘌呤/修饰核苷、长链酰基肉碱和扩展氨基酸模块均在 Rmu 中呈正向患者级效应。扩展氨基酸模块由 proline、glutamate、isoleucine、phenylalanine 和 tryptophan 构成，平均 `+0.969 log2`，10/10 正向，bootstrap 95% CI `[0.679, 1.280]`，逐一 leave-one-feature-out 后方向保持。

但患者级协调不支持把这些模块串成单链。expanded amino-acid 与 purine/modified-nucleoside 的 Spearman rho 为 `0.833`，在 10 组模块比较后 BH q=`0.0775`；Neu5Ac 与 expanded amino-acid 的 rho=`0.164`，与 purine/modified-nucleoside 的 rho=`-0.033`，BH q 均为 `0.948`；long-chain acylcarnitine 与其他主要模块也缺乏协调。因此最符合数据的模型是多个并行 pool-size programs，共同出现在同一临床亚组，但不存在已证明的共同上游调控器。

## 结果 5：跨队列证据把 general CRC 与 mucinous-relative 程序拆开

TCGA COADREAD 的 32 对肿瘤–癌旁中，proline-synthesis 轴 32/32 上升，平均 `+0.952 standardized units`，BH q=`3.73e-9`；PYCR1 32/32、PYCRL 31/32、PYCR2 28/32 上升。独立 pooled mucinous CRC 蛋白组中，ALDH18A1/PYCR1/PYCR2/OAT 在左右侧黏液型组均 4/4 为正。相反，在 42 个黏液型与 329 个常规型肿瘤的调整比较中，proline-synthesis beta=`-0.249`，BH q=`0.0130`。因此本地 proline/glutamate 增加更符合一般 CRC 的 proline/P5C–matrix program，而不是黏液型特异增强。

GSE236696 的保守 epithelial pseudobulk 中 proline-synthesis 5/6 上升，PYCR1 5/6 上升；但轴级双侧精确 p=`0.125`，20,000 个表达、检出率和方差匹配随机轴的经验 p 为 `0.228`（幅度）和 `0.253`（5/6 同向）。这个结果只能作方向背景，不能作显著细胞来源证据。

Neu5Ac 呈现不同结构。一般 CRC 的 32 对 TCGA 中，sialic synthesis/transport、remodeling 和 mucin-sialylation 轴多数下降；但黏液型相对常规型中，GNE/NANS/SLC35A1、ST3GAL4、ST6GALNAC1/2 以及 MUC2/SPDEF/FCGBP 程序相对富集。单病例空间数据支持 tumour/goblet secretory-mucin 与 CAF/collagen context，却不支持 tumour-wide proline 或 sialic-axis 增高。因而最高强度结论是 `mucinous-relative, selective sialic/mucin-glycan remodeling`，而不是 global hypersialylation。

独立组织 glycomics 为该方向提供结构背景但不是精确复现。2024 年 CRC MALDI-MSI/N-glycomics 研究显示 mucinous 与 non-mucinous 组织可由 N-glycome 区分，主要涉及 pauci/high-mannose 与 bisecting N-glycans；其公开补充表只有101条糖链组成目录，其中19条含NeuAc，但没有患者级强度、组织学或tumour/normal列，不能据此复算 feature 703 或宣称 Neu5Ac pool 被独立复现。2026 年 MUC2 on-slide mucinase–MALDI-MSI/LC-MS 工作在三块黏液癌标本中直接观察到肿瘤内 glycoform-specific spatial pattern；严格核对后这三块标本仅来自两位独立患者，Colon1a/1b不能当作独立重复。人工复核糖肽列表与source spectra确认了MUC2上的sialylated、O-acetyl-Neu5Ac和putative O-acetyl-GalNAc证据，支持“free Neu5Ac pool 不等于具体 glycan destination”这一机制边界；但不同样本鉴定深度不等，糖肽鉴定数不能解释为丰度，无非黏液型人群对照也使其不能承担亚型验证。补充表字段级审计见 `docs/MTBLS13729_EXTERNAL_NGLYCOMICS_SUPPLEMENT_AUDIT_20260830.md` 和 `docs/MTBLS13729_PXD055865_MUC2_GLYCOPEPTIDE_AUDIT_20260831.md`。

进一步复核 2022 CRC PGC-LC-MS/MS O-glycomics 的患者级补充表后，获得了方向更直接的独立结构证据。以 Table S2 锁定 T2/T3 为 MUC 后，两例的 core-2 丰度为 `61.41/44.22`，在11个 AC/MUC 原发肿瘤中排名第1/第2；sialyl-Lewis X/A 为 `7.64/20.05`，同样排名第2/第1；α2-6 sialylation 为 `23.24/15.00`，排名倒数第2/第1。相对各自匹配正常组织，core-2 分别 `+46.46/+40.13`，sLeX/A `+7.64/+4.49`，α2-6 `−50.08/−70.43`。但是总 α2-3 sialylation 并未高于 AC，只在配对肿瘤-正常方向上两例均上升。因此外部证据支持的是 `core-2/sLeX/A expansion with α2-6 loss` 的选择性重塑，而不是 global hypersialylation。该外部队列只有2例MUC，且 Table S8 存在把T6误标为MUC的元数据冲突，故只能称 independent structural support，不能称患者级丰度复现。详见 `docs/MTBLS13729_EXTERNAL_OGLYCOMICS_MUCINOUS_AUDIT_20260831.md`。

### 组织组成敏感性把唾液酸程序进一步拆成两层

在 TCGA 的 42 个黏液型与 329 个常规型原发肿瘤中，我们为每个代谢轴单独构造了不含该轴基因的 epithelial、myeloid、B/plasma、T/NK、endothelial 和 fibroblast 表达代理，避免用 `COL1A1/COL1A2` 等重叠基因机械性校正结果。加入六类 lineage proxy 后，sialic-acid synthesis/transport 的黏液型效应由 beta=`+0.392` 增强为 `+0.480`，HC3 p=`6.60e-9`、八轴 BH q=`2.64e-8`；再加入 MSI 的完整病例模型仍为 beta=`+0.448`、p=`1.45e-7`。secretory-mucin program 同样保持 beta=`+0.922`、q=`4.27e-11`。

相反，mucin-sialylation transferase 轴在 lineage 校正后由 beta=`+0.252` 降至 `+0.113`，p=`0.0800`、q=`0.1067`，说明其 bulk 信号对组织组成更敏感。proline-synthesis 方向仍为负，但由 `-0.249` 衰减至 `-0.179`，p=`0.0526`、q=`0.0842`；只能称方向保留而非确认的黏液型特异差异。glutamate-supply 轴则稳健保持较低（beta=`-0.296`, q=`0.00235`）。这些结果把“黏液型唾液酸程序”收敛为较稳健的前体合成/胞内转运与分泌型黏蛋白背景，而不是所有唾液酸转移反应一致上调。

这些 broad-lineage scores 只是 bulk composition sensitivity proxies，不是实测细胞比例。信号保留支持对粗粒度组成的稳健性；信号衰减也可能反映真实的基质或上皮生态位，而不是应被删除的技术混杂。

### 结果 4c：五个冻结丰度模块均超出 phenotype-blind 匹配背景，但仍受同队列选择限制

为检验模块效应是否可由 m/z、RT、检测率、离子家族大小或 DDA/MS2 覆盖简单解释，我们在 1,018 个非候选正相 feature 中，为每个冻结节点建立 outcome-blind 近邻背景，并对每个模块抽取 50,000 个离子家族不重复的匹配集合。三套匹配定义分别使用 acquisition covariates、`±0.10` prevalence caliper 和额外 MS2-support sensitivity；由于 DDA support 会受离子丰度影响，acquisition-only 版本为主分析。

在主分析中，acetylated-polyamine–MTA、purine/modified-guanosine、long-chain acylcarnitine、expanded amino-acid 和 Neu5Ac 的观测平均 log2 效应分别为 `2.582`、`1.414`、`1.588`、`0.914` 和 `1.833`，对应匹配背景均值为 `0.155`、`0.126`、`0.064`、`0.321` 和 `0.185`。五个模块在三套定义中的描述性上尾比例均低于 `0.05`；前三个模块最稳定，expanded amino-acid 与 Neu5Ac 的尾部偏离较弱。

该审计排除了这些前置技术变量作为全部解释，但它没有修正同队列候选选择，因此这些 tail areas 不作为 confirmatory p 值，也不替代独立队列。工件见 `data/mtbls13729/module_matched_background_sensitivity_v2/`。

### 结果 4d：只有 Neu5Ac 通过了预定义的 Rmu 相对 Rtu 亚型敏感性门

为避免把一般肿瘤效应误写成黏液型特异效应，我们把 primary endpoint `Rmu-RN` 与 subtype-sensitivity endpoint `(Rmu-RN)-(Rtu-RN)` 分开。五个冻结模块在 Rmu 中均为正，但只有 Neu5Ac 在两种归一化下通过五模块 exact-permutation BH 校正：`log_raw` 的 Rmu、Rtu 平均配对变化分别为 `+1.881` 与 `-0.327 log2`，差值 `+2.209`、BH q=`0.00179`；PQN 对应为 `+1.798` 与 `-0.344`，差值 `+2.142`、BH q=`0.00162`。其余模块的跨归一化最大 q 分别为 acetylated-polyamine–MTA `0.181`、purine/modified-guanosine `0.125`、long-chain acylcarnitine `0.125`、expanded amino-acid `0.647`。

候选级复核进一步确认，在17个正相冻结候选中，feature 703 Neu5Ac 是唯一同时满足充分覆盖、Rmu 配对升高、Rmu 相对 Rtu 方向一致且两种归一化下候选面板 BH q<`0.01` 的节点（最大 q=`0.00607`）。feature 1597、3019 和 3222 的严格有效 Rmu 患者数分别只有 `2`、`2` 和 `4`，不能用表观大效应替代覆盖门。该候选面板统计不等同全13,155-target空间校正；feature 703 在全空间 exact-FDR10 仍未通过。

因此当前生物学结构应写为：**一个 mucinous-relative Neu5Ac/mucin-glycan 主轴，加上多个主要属于 general tumour abundance 的平行程序**，而不是“所有候选都具有黏液型特异性”。对应工件为 `data/mtbls13729/module_subtype_interactions_v1/`、`data/mtbls13729/candidate_subtype_interactions_v1/` 和 `data/mtbls13729/candidate_claim_scorecard_v3/`。

### 结果 4e：372 对外部 CRC 组织确认 Neu5Ac 的疾病依赖空间调控，但没有黏液型分层

为检验解剖位置是否构成 Neu5Ac 结果的真实生物学背景，我们复核了 Jain 等 2024 年覆盖七个
结直肠亚部位的 372 对患者匹配肿瘤–正常黏膜队列及其官方补充材料。该研究用标准品、RT 和
CRC 样本 MS/MS 将 N-acetylneuraminic acid 定为 HILIC negative 的 Level 1 代谢物
（m/z `308.0980`，RT `355.7 s`）。在线性生物地理分析中，匹配正常黏膜的 cecum-to-rectum
斜率为 `+0.349`、p `<0.001`，肿瘤斜率仅 `+0.088`、p=`0.091`。这提供了独立患者组织中的
Neu5Ac 身份和疾病依赖空间调控背景，说明配对及位置敏感分析不可省略。

该证据不能升级为 Rmu 复现。补充患者特征只有 sex、age、stage 和 anatomical subsite，全文及
表格没有 `mucinous` 命中，也没有患者级黏液型 Neu5Ac 结果；Neu5Ac 还未进入该研究各亚部位共同
tumour-vs-normal 差异汇总表。故外部结果只能作为 `PASS_CONTEXT`：它支持位置是必要协变量，
但既不能证明本地 interaction 完全来自组织学，也不能替代独立黏液型丰度队列。补充分期表格
单元格合计为374、与方法报告372不一致，该源文件差异已保留在审计中。完整工件见
`docs/MTBLS13729_EXTERNAL_NEU5AC_BIOGEOGRAPHY_AUDIT_20260831.md`。

公开网页不能作为该统计的患者级复算来源。Dash 回调对normal和tumour各返回371个值，七个
亚部位均固定为53个，且没有patient/pair ID。按网页值回归得到的normal/tumour标准化系数为
`0.391/0.179`，不能复现补充表的`0.349/0.088`，尤其tumour网页回归显著而补充表p=`0.091`。
因此网页值不进入效应量合并、配对分析或独立复现计数，正式补充表仍是该队列的权威统计来源。
复现性工件见 `docs/MTBLS13729_EXTERNAL_NEU5AC_DASH_REPRODUCIBILITY_AUDIT_20260831.md`。

## 结果 5b：分支级转录审计把主轴收敛为 hybrid mucin glycome

一份额外的三队列转录整合结果为黏液型关联提供背景支持。其公开补充表在980例有组织学信息的
CRC中记录Sialyl-High为mucinous `85/154`、non-mucinous `238/826`，由原始计数重算OR=`3.04`
（95% CI `2.14–4.33`）。但该score只由20个sialyltransferase genes组成，不测free Neu5Ac或
糖链结构；三队列还包含本地已经使用的TCGA，且没有按队列分别报告mucinous效应。因此它只作为
外部 pooled transcriptomic context，不计为独立代谢物复制。

为了检验 Neu5Ac 丰度升高是否对应统一的高唾液酸化程序，我们依据独立 CRC O-glycomics 的
生物合成结构，预先冻结 donor supply/transport、secretory-mucin carrier、core-2/sLeX、
core-3/Sda 和 linkage-specific 分支，并在 TCGA COAD/READ 的 42 例黏液型与 329 例常规型
肿瘤中依次调整临床变量、非重叠 broad-lineage expression proxies 和 MSI。Neu5Ac donor
supply/transport 在 clinical+lineage 模型中为 beta `+0.480`、BH q=`3.30e-8`，加 MSI 后仍为
`+0.448`、q=`5.36e-7`；secretory-mucin carrier 为 `+0.922`、q=`5.34e-11`；core-3/Sda
mucosal-lineage 为 `+0.879`、q=`1.76e-8`。相反，core-2/sLeX transcript composite 接近零
（beta `-0.009`、q=`0.915`），alpha2-3 O-glycan sialylation 为 `-0.439`、q=`0.0093`，
ST6GAL1 为 `-0.742`、q=`8.50e-5`；ST6GALNAC1 和 GCNT3 则分别为 `+0.788`、q=`7.99e-8`
及 `+0.417`、q=`0.0096`。这些方向在 MSI 完整病例中保持。

转录分支与外部患者结构层并非矛盾，而是揭示了不同参照系。两个外部 MUC 肿瘤的 core-3 在
AC/MUC 肿瘤间排名最高，但相对各自正常组织仍下降约 `34%`；core-2、sLeX/A 和
core-2+alpha2-3 则配对上升，alpha2-6 配对大幅下降。因此 core-3/Sda 的正向肿瘤间系数应解释
为黏膜分泌谱系的**相对保留**，不能解释为肿瘤转化中的绝对增加。

综合 feature 703 的 10/10 Rmu 配对升高、TCGA 分支解耦和独立 O-glycomics，当前最小充分模型
是 hybrid mucin glycome：扩大或重新分配的游离 Neu5Ac 资源池与 MUC2/secretory-mucin 程序
共存，同时发生 core-2/sLeX 肿瘤相关糖抗原获得、core-3 黏膜谱系相对保留及 alpha2-6 丢失。
一项 2026 MUC2 空间糖肽研究还在来自2位患者的3块黏液型肿瘤中观察到以低唾液酸化/非唾液酸化
MUC2 glycopeptide 为主，以及相对健康结肠较低的 mono-/di-O-acetylated Neu5Ac glycans；公开
Supplementary Data 2与source spectra的本地审计确认这些载体级结构证据存在，但不把不均衡的
鉴定数转写为丰度。这为“free pool 与 carrier-specific destination 解耦”提供了正交反证，
但不是亚型人群复现或游离Neu5Ac独立复制。
三层证据已并排呈现在`data/mtbls13729/pool_carrier_boundary_figure_v1/`：Panel A为同患者
free-pool/donor丰度，Panel B为bulk mono-O-acetyl-Neu5Ac-like阴性结果，Panel C仅展示
PXD055865人工复核MUC2结构的存在性并明确标注`NOT abundance`。

同一患者内的代谢物分解进一步把“donor”从转录背景推进到可检验的静态 pool-size 层。原始
HILIC(-) 补充表中，Level-1 free Neu5Ac 在10/10 Rmu患者中升高，平均 `+2.249 log2`；
Level-2 CMP-Neu5Ac 与 Level-1 UDP-GlcNAc 分别仅为 `+0.556` 和 `+0.327 log2`，均未达到
名义显著。预设的患者内差值显示，free Neu5Ac 的升幅分别比 CMP-Neu5Ac 和 UDP-GlcNAc 高
`+1.693/+1.922 log2`，两个比较的 Holm-Wilcoxon p 均为 `0.0273`，bootstrap 95% CI 下界
均大于零。因此数据支持 `free-pool-to-activated-donor decoupling`，并反对把 RNA 层面的
GNE/NANS/SLC35A1 富集直接解释为 CMP-Neu5Ac 同步扩增。该结果仍是同队列静态丰度分解，且
CMP-Neu5Ac 仅为 Level 2；它不能区分释放、回收、活化、摄取或糖链利用。患者级审计见
`docs/MTBLS13729_SIALIC_DONOR_DECOUPLING_AUDIT_20260831.md`。

一个按生化反应预定义的TCGA分支审计进一步裁掉了最简单的“sialidase释放”解释。在32对一般
CRC tumour-normal中，NEU1/NEU3轴显著升高（平均`+0.854 z`，BH q=`9.02e-7`）；但在42个
mucinous与329个conventional primary tumour中，该轴经lineage/MSI校正后反而降低
（beta=`-0.691/-0.654`，BH q=`5.58e-6/1.53e-5`）。CMP activation/transport轴则在
mucinous中相对升高（lineage beta=`+0.449`，q=`1.61e-4`），主要由SLC35A1而非CMAS驱动。
因此NEU1/NEU3转录上调不是Rmu free Neu5Ac积累的简单解释；RNA capacity与实测CMP-Neu5Ac
pool也并不同步。current-GDC对同一锁定队列补齐NXPE1后，NXPE1在mucinous相对conventional中
经临床+lineage校正为beta=`+0.621`、p=`0.000369`，加入MSI后仍为`+0.530`、p=`0.00134`；
但加入预定义`MUC2/TFF3/SPDEF/FCGBP/AGR2` secretory-mucin程序后降为`+0.064`、p=`0.734`
（再加MSI为`-0.048`、p=`0.782`）。删除任一secretory marker及所有双marker敏感性结果均未恢复
独立显著性，说明这是分布式carrier state而非单个MUC2共线性；它仍是协变量敏感性，不能写成
因果中介。另一方面，current-GDC一般CRC的50对tumour-normal中NXPE1为47/50肿瘤较低，
GSE236696六对黏液癌的保守上皮pseudobulk也为6/6肿瘤较低（低计数，精确双侧p=`0.0625`）。
因此最严谨的解释是：一般肿瘤转化伴随NXPE1降低，而黏液型相对常规型保留/富集一个与分泌黏蛋白
状态相连的O-acetylation capacity。CASD1−SIAE未通过统一校正，且NXPE1蛋白、酶活、具体产物位置
和glycan carrier均未测，O-acetylation通量仍属未决机制。2025年两项原始研究分别支持free Neu5Ac
和CMP-Neu5Ac acceptor context，故本地Level-1 free Neu5Ac不得被直接指定为体内NXPE1底物。
完整审计见`docs/MTBLS13729_SIALIC_POOL_MECHANISM_DISCRIMINATION_20260831.md`和
`docs/MTBLS13729_NXPE1_POOL_CARRIER_OACETYL_MECHANISM_AUDIT_20260831.md`。

为直接检验bulk游离O-acetylated sialic-acid pool，negative-HILIC原始数据在结果分组不可见时
对`m/z 350.109269 [M-H]-`做了全RT峰发现。只保留两个互不重叠的共识峰：4.29分钟峰由50/60
样本和47张RT分层MS2支持，5.55分钟峰由54/60样本和56张MS2支持；m/z 87分别出现于
47/47和54/56张谱。但两个峰在Rmu中均无稳定配对升高（两个BH q均`0.930`），与Level-1 free
Neu5Ac的患者变化rho仅`0.170/-0.067`。因此free pool扩张不伴随这两个bulk
mono-O-acetyl-Neu5Ac-like exact-mass features同步增加。由于位置异构体、标准品和glycan carrier
均未解决，该结果只排除一个简单pool模型，不排除NXPE1/CASD1/SIAE活性或glycan-bound
O-acetylation。现有HMDB条目仅提供预测谱，MassBank/MoNA的精确名称/分子式检索未得到实验记录；
4/7/8/9位异构体需要成对标准品，必要时需IM-MS/CCS。详见
`docs/MTBLS13729_OACETYL_NEU5AC_LIKE_AUDIT_20260831.md`和
`docs/MTBLS13729_OACETYL_NEU5AC_STANDARD_RESOURCE_AUDIT_20260831.md`。

该模型反对 `global hypersialylation`，也不把 free Neu5Ac、CMP-Neu5Ac donor、具体 glycan
linkage 或 ST6GAL1–PD-L1 路线互相等同。完整审计见
`docs/MTBLS13729_NEU5AC_HYBRID_GLYCOME_AUDIT_20260831.md` 和
`data/external/TCGA_COADREAD_Xena_20260830/glycan_branching_v2/`。

### 结果 5c：独立 raw-UMI 患者审计把组成效应与上皮状态拆开

我们随后直接读取 GSE178341 官方 10x raw-count HDF5，并以患者而非细胞作为推断单位。在6例
纯黏液型与53例纯常规腺癌的肿瘤上皮中，AGR2和SLC35A1分别增加`+1.613`和`+0.833
log2(CPM+1)`，固定基因面板BH q均为`0.0068`；右侧/MMR分层敏感性q均为`0.0179`。
预定义的上皮secretory-carrier和CMP-Neu5Ac-capacity模块均通过患者bootstrap、固定七端点
BH<`0.10`、5/6匹配方向及leave-one-case-out门。相反，上皮和髓系NEU1/NEU3 release轴均无
正支持；上皮轴的方向为`-0.547 z`。NXPE1在广义上皮中虽为`+0.837`，但患者置换p=`0.152`、
固定面板q=`0.229`，冻结匹配仅4/6为正，不能作为独立驱动。

考虑到黏液型CRC的goblet-like组成，我们又冻结了一个明确标注为post-result的患者级组成诊断。
作者标注的`cE02/cE06/cE07/cE08`在黏液型中的平均比例为`0.284`，常规型为`0.152`，差值
`+0.132`，但bootstrap 95% CI为`-0.013至+0.287`，六例匹配中仅4例为正。加入logit goblet
fraction后，MUC2、SPDEF和NXPE1的黏液型系数区间均跨零；AGR2仍为`+1.172`（95% CI
`+0.483至+1.861`），SLC35A1仍为`+0.676`（`+0.216至+1.135`）。再加入right-colon与MMR
后两者仍保持正区间。故独立转录证据支持的是**部分组成富集之上的选择性上皮分泌折叠和Golgi
donor-transport capacity**，而不是每个goblet cell统一上调全部通路成员，也不是宿主sialidase
释放机制。完整结果见`docs/GSE178341_RAW_UMI_MUCINOUS_SIALIC_AUDIT_20260831.md`和
`docs/GSE178341_EPITHELIAL_COMPOSITION_DIAGNOSTIC_RESULT_20260831.md`。

独立15例黏液型对15例常规型蛋白组没有确认预冻结的两个蛋白模块。AGR2、GNE和NANS均保持
15/15 leave-one-mucinous-patient-out正方向，但bootstrap区间跨零、八蛋白BH q均为`0.643`；
CMAS和SIAE没有正支持。AGR2在原始丰度尺度可复现源文探索性FC=`2.60`、Welch p=`0.0478`，
但在预冻结log2/permutation分析中p=`0.161`，提示结论对尺度敏感。该结果作为受限方向背景保留，
不能称为Neu5Ac独立丰度复制或蛋白通路确认。

## 结果 6：当前证据达到 mechanism-supporting discovery，而非代谢因果机制

高水平非靶向机制研究并不在差异 feature 或通路富集处结束。CRC AHCY 研究将 untargeted LC-MS/MS 与标准、空间定位、同位素追踪、类器官、shRNA/药理和体内表型连接；DeepMet 对 67 个购买/合成候选仅有 25 个同时通过组织峰 RT 与 MS/MS，表明强生成模型的 top-1 仍需实验终证；患者肿瘤–PDX 代谢研究甚至对同一患者来源样本实施平行代谢组与体内 `[U-13C]glucose` tracing，以区分 pool size 与 nutrient fate。

本项目已经具备患者内配对定量、原始峰界 MS2、跨面板身份桥、失败候选、TCGA/患者级raw-UMI单细胞/空间/蛋白组正反证、43-row evidence matrix和25门完成度总账。仍缺同法 authentic-standard RT/MS2/spike-in、独立 Rmu 组织 metabolomics、同一样本 glycan linkage/glycoproteomics、同位素 tracing、酶扰动、rescue 和体内功能。因此论文应使用 `abundance programs`、`remodeling`、`mechanistic context` 和 `testable hypotheses`，不使用 `flux`, `enzyme activation`, `causal reprogramming` 或 `therapeutic target established`。

原论文自身以跨患者 Rmu-vs-Rtu Student t-test 报告 92 个 nominal differential metabolites（其中 83 个升高），并使用了 `altered flux`、`activated carnitine shuttle/FAO` 等强机制措辞。本研究的 primary biology endpoint 改为同患者 Rmu-vs-RN 配对丰度；subtype specificity 必须由 `(Rmu-RN) - (Rtu-RN)` interaction 单独判断。在冻结的 555 个 phenotype-blind positive-RP annotation targets 中，primary endpoint 有 6 个跨归一化 FDR10、3 个 FDR05 feature，而 interaction endpoint 没有 FDR10 feature；这不是对全部 13,155 个 MS1 target 的全空间 FDR。完整 13,155-target discovery-matrix 审计中没有 feature 通过 FDR10，只有 132 个 nominal exact-gate feature。因此六个候选必须表述为预定义注释候选面板内的 FDR，而不是全非靶向空间确认。预选择 modified-guanosine module 的 exact interaction p 约 `0.019–0.025` 也只能作为探索性模块信号，不能替代全空间 FDR。这个终点与分母重构是本项目相对原文的重要统计增量。

## 当前主文工件

- `data/mtbls13729/integrated_biology_ledger_v2/`
- `data/mtbls13729/original_paper_delta_v2/`
- `data/mtbls13729/mechanism_evidence_matrix_v2/`
- `data/mtbls13729/manuscript_evidence_matrix_v2/`
- `data/mtbls13729/module_coordination_v2/`
- `data/mtbls13729/proline_sialic_summary_figure_v1/`
- `data/mtbls13729/source_narrative_audit_v1/`
- `data/external/TCGA_COADREAD_Xena_20260830/proline_sialic_lineage_sensitivity_v1/`
- `data/mtbls13729/module_matched_background_v1/`
- `data/mtbls13729/module_matched_background_sensitivity_v2/`
- `data/mtbls13729/competing_mechanism_trees_v1/`
- `data/mtbls13729/neu5ac_glycan_publication_figure_v2_final/`
- `data/mtbls13729/oacetyl_neu5ac_like_v2/`
- `data/mtbls13729/oacetyl_neu5ac_like_figure_v1/`
- `data/external/GSE178341_mucinous_secretory_audit/nxpe1_mucinous_patient_pseudobulk_v1/`
- `data/external/GSE178341_mucinous_secretory_audit/sialic_cell_source_patient_pseudobulk_v1/`
- `data/external/GSE178341_mucinous_secretory_audit/epithelial_composition_diagnostic_v1/`
- `data/external/GSE178341_mucinous_secretory_audit/independent_proteomics_fixed_panel_v1/`
- `data/mtbls13729/mechanism_paper_completion_audit_v10_final/`

## 主要方法学和机制对标

- CRC AHCY mechanism: https://www.nature.com/articles/s42255-023-00857-0
- CRC proline/PYCR biology: https://pubmed.ncbi.nlm.nih.gov/35130302/
- DeepMet prediction-to-standard success: https://www.nature.com/articles/s41586-025-09969-x
- Patient–PDX metabolomics plus in-vivo tracing: https://www.nature.com/articles/s42255-025-01338-2
- Human mucinome with defined O-glycans: https://www.nature.com/articles/s41467-021-24366-4
- Colon sialoglycome O-acetylation: https://www.nature.com/articles/s41467-025-59671-9
- Mucinous CRC collagen–integrin polarity: https://www.nature.com/articles/s41467-026-75127-0
