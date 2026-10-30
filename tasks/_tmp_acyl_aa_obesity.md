# 脂肪酸-氨基酸缀合物（N-acyl amino acids / lipoamino acids）在肥胖与高脂饮食中的角色 — 文献调研

> 调研时间：2026-10（基于 web 检索与全文/摘要核对）。缩写：NAA = N-acyl amino acid；NAGly = N-acyl glycine；O-Gly/N-OleGly = N-oleoyl glycine；S-Gly = N-stearoyl glycine；NAryGly = N-arachidonoyl glycine；OEA = 油酰乙醇酰胺；HFD = high-fat diet；eCBome = endocannabinoidome（内大麻素系统扩展家族）。

---

## 一句话总结（TL;DR）

| 问题 | 简答 |
|---|---|
| 1. NAA 在肥胖/HFD 中是否变化？ | **是**。人队列显示多种 N-油酰氨基酸（N-oleoyl-leucine/-phenylalanine/-glycine 等）与 BMI、腰围、糖尿病显著相关（JHS 队列 n=2351）；T2D 患者血浆 N-oleoyl glycine 显著升高；HFD 小鼠/肥胖干预模型中 N-oleoyl glycine 等多次被报道变化。 |
| 2. 是否为已知 GPCR 配体？ | **是，但受体依赖分子与细胞体系**。油酰/花生四烯酰甘氨酸是 GPR40/55/119/120 的激动剂（GLP-1 细胞体系，2024）；但在 HEK293-GPR119 体系中 N-oleoyl glycine（带羧基头基）**不能**激活 GPR119。N-arachidonoyl glycine 是 GPR18 内源性配体（GPR55 活性有争议）。N-oleoyl glycine 还激活脂肪细胞 CB1 受体。 |
| 3. 是否在尿液中被检出？ | **部分是**。酰基甘氨酸（acylglycines）作为一类在尿液中早有检出（1975 年 TLC；现为有机酸尿症临床 UPLC-MS/MS 常规指标），但明确针对**长链** NAA（如 O-Gly/S-Gly）的尿液定量文献很少；O-Gly 在 HMDB 有收录（HMDB0013631），已在粪便、血浆、脑中检出。 |
| 4. 饱和/不饱和 NAA 比值是否为已知生物标志物？ | **尚无被广泛公认的"命名"比值标志物**。最接近的证据：Grapov 2012 中 T2D 仅升高不饱和的 O-Gly 而饱和的另一种 LAA（stearoyl 系列）不变（提示不饱和/饱和模式偏移）；同源概念已确立的是脂肪酸层面的 SCD 指数（16:1/16:0、18:1/18:0）与 NAE 层面的 OEA/PEA。NAA 层面的比值可视为"合理但未成型"的候选标志物。 |

---

## 1. N-acyl amino acids 在肥胖 / HFD 中的变化（问题 1）

### 1.1 人类流行病学/临床证据

**Tanzo et al., J Biol Chem 2023（Jackson Heart Study, n=2351，非洲裔美国人）** — 目前最强的人类证据：
- 系统分析了 4 种血浆 NAA：N-oleoyl-leucine、N-oleoyl-phenylalanine、N-oleoyl-serine、**N-oleoyl-glycine**。
- N-oleoyl-leucine / N-oleoyl-phenylalanine 与 **BMI、腰围、皮下脂肪量正相关**。
- FAAH 调控的 N-oleoyl-glycine / N-oleoyl-serine 与**血糖/血脂稳态指标负相关**。
- 发现 **CYP4F2 位点**是血浆 N-oleoyl-leucine/-phenylalanine 的人类特异性遗传决定因子（ω-羟化代谢多样性）。
- 结论：单分子 NAA 按氨基酸头基分组呈现不同的心脏代谢疾病关联模式。
- 链接：https://pmc.ncbi.nlm.nih.gov/articles/PMC10318452/ ；DOI: 10.1016/j.jbc.2023.104764

**Grapov et al., PLoS One 2012（e48852）** — BMI 匹配的肥胖非洲裔美国女性，非糖尿病(n=12) vs T2D(n=43)：
- 靶向定量 >150 种血浆脂质，含 NAE、单酰甘油和 **lipoamino acids（LAA）**。
- 所测 2 种 LAA 中，**仅 N-oleoyl glycine（NO-Gly）在 T2D 显著升高**；ΣLAA 升高（8.67 → 21.0 nM 量级）。全部 NAE（stearoyl-EA 除外）也升高；MAG 与 BMI 负相关。
- 同期 SCD 活性指标（16:1n7/16:0、18:1n9/18:0）升高约 60%。
- 链接：https://pmc.ncbi.nlm.nih.gov/articles/PMC3493609/ ；Table 5: https://journals.plos.org/plosone/article/figure?id=10.1371/journal.pone.0048852.t005

**其他人体出现场景**：
- 癌症恶病质小鼠（eCBome 靶向脂组学）：N-oleoyl-glycine、N-acyl-serotonin、N-acyl-taurine 在肠/肝改变并与体重丢失、肝脂关联（Degraeve/Di Marzo 2026, BJC Rep；https://europepmc.org/articles/PMC12960972）。
- 儿童肥胖表观遗传学：**PM20D1**（NAA 生物合成酶）高甲基化见于肥胖儿童（Slapnik 2026, Clin Epigenetics；https://pmc.ncbi.nlm.nih.gov/articles/PMC13343863/ —— 间接提示 NAA 通路与肥胖相关）。

### 1.2 机制/动物模型证据（NAAs 是"抗肥胖"脂质信号家族）

**Long et al., Cell 2016, 166(2):424-435（被引 >230 次，里程碑）**：
- 分泌型酶 **PM20D1** 在褐色/米色脂肪富集，双向催化脂肪酸+氨基酸 ⇌ **N-acyl amino acids**。
- NAAs 直接结合线粒体，作为 **UCP1 非依赖的解偶联剂**；给小鼠注射 NAA 改善葡萄糖稳态、增加能量消耗。
- 链接：https://europepmc.org/articles/PMC4947008 ；新闻稿：https://beta-stage.dana-farber.org/newsroom/news-releases/2016/researchers-identify-calorie-burning-pathway-in-fat-cells-with-potential-for-obesity-treatment

**后续验证/修正**：
- Herrnhold et al., Redox Biol 2023：中性氨基酸头基的 NAAs（N-oleoyl-Phe/Leu/Ile）在人脂肪细胞中确实是强解偶联剂（UCP1/ANT 非依赖），但 >25 µM 损伤脂肪细胞功能与活力，治疗指数低于 NEN/DNP。链接：https://europepmc.org/articles/PMC10493596
- 综述：Verkerke & Kajimura, Dev Cell 2021（脂质在产热脂肪中的多面角色，含 NAA 通路）：https://europepmc.org/articles/PMC8132468

**HFD/肥胖干预动物模型**：
- Yu et al., Front Nutr 2026（Gnaphalium affine 提取物抗 HFD 肥胖）：多组学显示 **N-oleoyl glycine** 是介导"肠菌-代谢"改善的关键生物活性代谢物之一；FMT 证实可转移表型。链接：https://europepmc.org/articles/PMC13021598
- Wang et al., BBRC 2015：**N-oleoyl glycine 本身促进 3T3-L1 脂肪细胞分化**（脂质累积、PPARγ/aP2↑），经 **CB1 受体 + Akt 信号**（SR141716 可阻断）；作者提出其可能增敏胰岛素、抑制肥胖/糖尿病（注意：与"促脂肪生成"字面张力，论文结论偏正面代谢效应）。DOI: 10.1016/j.bbrc.2015.09.046
- Murine C57BL/6 模型中 N-arachidonoyl glycine (NAGly) / oleoyl glycine (OlGly) 对神经递质、焦虑样行为与肥胖发生的影响（Archivos de Neurociencias 会议/期刊报告，2020；https://www.archivosdeneurociencias.mx/files/10996amex253-arneuro-30-107uk.html ）——证据级别较低（摘要级）。

**交叉关联（代谢/微生态）**：
- Drzazga et al., Biomed Pharmacother 2024（详见 §2）：**肠道 Bacteroides spp. 产生 N-acyl glycines**，其产量取决于培养底物（动物源 vs 酵母源培养基）→ 饮食脂肪来源直接影响 NAGly 产量，是"HFD ↔ NAA"的微生态纽带。DOI: 10.1016/j.biopha.2024.117467；PMID: 39362066。
- 定量方法学：Karin et al., J Sep Sci 2023 建立 O-Gly/O-Ala 的 HPLC-MS/MS 脑/血浆定量（>90% 回收）：https://pmc.ncbi.nlm.nih.gov/articles/PMC10872932/ ；3-硝基苯肼衍生化 NAGly 靶向代谢组学用于小鼠糖尿病进展研究：https://pmc.ncbi.nlm.nih.gov/articles/PMC9893217/

**小结 Q1**：NAA（尤其 N-oleoyl-glycine/leucine/phenylalanine）在肥胖、T2D、HFD 相关状态下的变化有**多层次证据**（人大队列关联、HFD 干预多组学、细胞/小鼠机制），方向依氨基酸头基而异（Leu/Phe 系与脂肪量正相关；FAAH 调控的 Gly/Ser 系与血糖代谢改善关联）。

---

## 2. GPCR 配体属性（问题 2）

### 2.1 正面证据：N-acyl glycines 是脂质 GPCR 激动剂

**Drzazga et al., Biomed Pharmacother 2024, 180:117467**（核心文献）：
- 肠道共生 **Bacteroides**（B. thetaiotaomicron、B. intestinalis 等）产生 NAGlys。
- **长链不饱和的油酰甘氨酸（oleoyl glycine）与花生四烯酰甘氨酸（arachidonoyl glycine）最强**，在 GLUTag 与 NCI-H716 细胞中促进 GLP-1 分泌。
- 关键受体数据：这些 NAGlys 是 **GPR40/FFAR1、GPR55、GPR119、GPR120/FFAR4 的激动剂**（在表达这些受体的 GLP-1 分泌细胞中验证）。
- 强调"均衡饮食作为共生菌生产有效脂质 GPCR 激动剂底物"的重要性。

### 2.2 反面/限定证据：GPR119 对头基带羧基的配体不敏感

**Kumar, Kumar & Song, Eur J Pharmacol 2013, 723:465-472**：
- HEK293 稳定表达人 GPR119 的 cAMP（HTRF）检测系统。
- OEA 激活 GPR119（EC50 ≈ 7.65 µM）；oleamide、OLDA 也能激活（弱）。
- 但 **N-oleoyl glycine、N-oleoyl alanine、N-oleoyl GABA 在浓度高达 100 µM 时均不能激活 GPR119**，也不能促敏化失活。
- 原因分析：带电荷的羧基头基（酸性/空间位阻）不利于 GPR119 激活；乙酰胺头基（乙醇酰胺）才是高效能所需。
- 链接：https://pmc.ncbi.nlm.nih.gov/articles/PMC5086809/

### 2.3 其他受体

- **GPR18**：N-arachidonoyl glycine（NAGly）是公认的内源性配体（Kohno et al. 2006 起；PubMed 检索 "n-arachidonoyl glycine" 有专目，https://pubmed.ncbi.nlm.nih.gov/?term=%22n-arachidonoyl%20glycine%22 ）。
- **GPR55**：内源配体公认是 LPI；NAGly 对 GPR55 的作用文献有争议（部分筛选报道弱活性）。Drzazga 2024 支持油酰/花生四烯酰甘氨酸作为 GPR55 激动剂（GLP-1 细胞背景）。
- **GPR55/GPR119 与 acylethanolamides** 的系统综述（Pertwee 系，Pharmacol Rev 2010）：https://pmc.ncbi.nlm.nih.gov/articles/PMC2993256/ ；"Receptors for acylethanolamides—GPR55 and GPR119"（PMC2751869）。
- **CB1**：N-oleoyl glycine 促进 3T3-L1 脂肪生成的作用被 SR141716 阻断且上调 CB1R mRNA（Wang 2015，见 §1.2）。
- **GLP-1 分泌增强**本身是 NAGlys 的重要代谢效应通路（Drzazga 2024；肠内分泌 L 细胞）。

**小结 Q2**：是——NAA 是"脂质 GPCR 配体家族"成员，但**受体选择性高度依赖氨基酸头基与检测体系**：不饱和长链 NAGly（油酰/花生四烯酰）在肠内分泌细胞中激动 GPR40/55/119/120；而在异源表达体系（HEK293-GPR119）中，带羧基的 N-oleoyl glycine 对 GPR119 无活性（GPR119 的内源配体主要是 OEA 等 NAE）。引用时需注明体系差异，避免绝对化。

---

## 3. 尿液检出（问题 3）

- **酰基甘氨酸类（acylglycines）在尿液中的检出历史悠久**：Van den Berg & Hommes, *J Chromatogr* 1975, 104:217-222，"Thin-layer chromatographic detection of some N-acylglycine conjugates in urine"（人尿，含婴幼儿；DOI: 10.1016/S0021-9673(01)85517-5）。
- **临床常规**：中短链酰基甘氨酸（isovalerylglycine 等）是有机酸血症（organic acidemias）的**经典尿标志物**，有成熟 LC-APCI-MS（https://www.sciencedirect.com/science/article/abs/pii/0378434795001743 ）与 UPLC-MS/MS 定量方案（Current Protocols《Acylglycine Analysis by UPLC-MS/MS》：https://currentprotocols.onlinelibrary.wiley.com/doi/abs/10.1002/cphg.19 ）。
- **长链 NAA（O-Gly/S-Gly 等）的尿液定量证据薄弱**：O-Gly 收录于 HMDB（HMDB0013631，https://hmdb.ca/metabolites/HMDB0013631 ），已确认的生物标本为**粪便**（Goedert et al., Carcinogenesis 2014，GC-MS/LC-MS，"detected but not quantified"，https://fecalmetabolome.ca/concentrations/157958 ）；亦见于 FOODB（FDB029608）与牛乳数据库。血浆（Grapov 2012；Tanzo 2023）与小鼠脑（Karin 2023）均有定量，但**未检索到针对人尿中长链 N-油酰/硬脂酰甘氨酸的专门定量研究**——非靶向尿液代谢组学中这类高疏水性、低丰度脂质通常难被常规 RP-LC-MS 覆盖（或仅 Level 3/4 注释）。

**小结 Q3**：酰基甘氨酸作为"类"在尿液中检出是确定的（1975 TLC 起，现为有机酸尿症临床指标）；但**长链** NAA 在尿液中的明确检出/定量文献极少，主要在粪便/血浆/脑中报告。若课题需要尿液证据，这是一个可填补的空白点。

---

## 4. 饱和/不饱和 NAA 比值作为代谢标志物（问题 4）

- **未发现被公认命名的"N-stearoyl/N-oleoyl glycine 比值"类生物标志物**（多轮检索无果）。
- **最接近的直接证据**：Grapov 2012（BMI 匹配、肥胖人群）中 T2D **仅升高不饱和的 O-Gly，而另一种所测 LAA（饱和硬脂酰系）不变** → 不饱和/饱和 LAA 模式在糖尿病中发生偏移（尽管原文未将该比值定义为独立标志物）。
- **平行概念（已确立，可类比引用）**：
  - 脂肪酸层：SCD（硬脂酰-CoA 去饱和酶）活性指数 = 16:1n7/16:0、18:1n9/18:0，T2D 中升高 ~60%（Grapov 2012 同一研究）；
  - NAE 层：OEA/PEA 比值等在代谢研究中使用；
  - GPR119 SAR：脂肪酰链不饱和度决定活性（1-2 个双键 > 3-6 个双键）（Kumar 2013）——即"饱和度比值"有**药理功能**基础，但尚无"临床比值标志物"定名。
- benchchem 商业文档称 N-stearoylglycine 为"潜在 NAFLD 生物标志物"（https://www.benchchem.com/fr/product/B1329668/docs ）——厂商二级来源，非同行评议，谨慎引用。

**小结 Q4**：不是已确立的命名标志物。但 (a) 不饱和 NAA（O-Gly）在 T2D 选择性升高的方向性证据存在；(b) 同一家族的饱和/不饱和比值概念（SCD 指数、OEA/PEA）已确立；(c) 因此"oleoyl/stearoyl-glycine 比值"是**有合理生物学基础但未被正式提出/验证的候选标志物**——若你的数据支持，可主张为 novel contribution 而非引用既有标志物。

---

## 5. 关键文献速查表

| # | 文献 | 与四问的对应 |
|---|---|---|
| 1 | Tanzo JT et al., *JBC* 2023;299(6):104764 (JHS n=2351) | Q1（人: N-oleoyl-Leu/Phe ↔ BMI/腰围正相关；O-Gly/O-Ser ↔ 血糖血脂负相关；CYP4F2 遗传决定） |
| 2 | Grapov D et al., *PLoS One* 2012;7:e48852 | Q1/Q4（T2D 血浆 O-Gly↑、另一种 LAA 不变；SCD 指数） |
| 3 | Long JZ et al., *Cell* 2016;166:424-435 | Q1 机制（PM20D1-NAAs-线粒体解偶联-能量消耗/血糖） |
| 4 | Herrnhold M et al., *Redox Biol* 2023;66:102874 | Q1 机制修正（人脂肪细胞：解偶联但治疗指数低） |
| 5 | Drzazga A et al., *Biomed Pharmacother* 2024;180:117467 | Q1/Q2（Bacteroides 产 NAGly；O-Gly/NAGly 激动 GPR40/55/119/120，促 GLP-1） |
| 6 | Kumar P et al., *Eur J Pharmacol* 2013;723:465-472 | Q2（O-Gly/O-Ala/O-GABA 不能激活 HEK293-GPR119；OEA 为配体；SAR） |
| 7 | Wang S et al., *BBRC* 2015;466:438-443 | Q1/Q2（O-Gly 经 CB1+Akt 促 3T3-L1 脂肪生成） |
| 8 | Van den Berg H & Hommes FA, *J Chromatogr* 1975;104:217-222 | Q3（尿中 N-acylglycine 结合物的 TLC 检出） |
| 9 | Goedert JJ et al., *Carcinogenesis* 2014;35:2089-96（HMDB0013631） | Q3（O-Gly 于人粪便检出；HMDB 收录） |
| 10 | Karin K.N. et al., *J Sep Sci* 2023;46:e2300395 | 方法（O-Gly/O-Ala 的 LC-MS/MS 定量） |
| 11 | Yu Z et al., *Front Nutr* 2026;13:1779459 | Q1（HFD 小鼠抗肥胖干预中 N-oleoyl glycine 为关键中介代谢物） |
| 12 | Degraeve AL et al., *BJC Rep* 2026;4:7 | Q1（恶病质 eCBome：O-Gly 改变，与体重丢失/肠菌关联） |
| 13 | Kohno et al. 2006（GPR18/NAGly）及 PubMed 专题 | Q2（NAGly = GPR18 内源配体） |
| 14 | Pertwee RG 系综述, *Pharmacol Rev* 2010 (PMC2993256) | Q2（GPR55/GPR119 与 acylethanolamides 总论） |

---

## 6. 对下游分析/写作的提示

1. **"NAA 变化于肥胖/HFD"有坚实文献基础**（人队列 + 机制 + 动物），可放心作为背景陈述，建议引用 #1、#2、#3、#5。
2. **GPCR 配体表述必须带限定**："N-acyl amino acids act as ligands for lipid-sensing GPCRs (GPR40/55/119/120)，但受体激活依赖头基电荷与体系；在 HEK293-GPR119 中带羧基的 N-oleoyl glycine 无活性"。GPR119 的经典内源配体仍是 OEA。
3. **尿液证据是短板**：若你的数据是尿液检出长链 NAA，这是相对新颖的发现（现有文献以粪便/血浆为主），值得强调；酰基甘氨酸尿检的经典文献（#8、有机酸尿症临床方案）可作类比支撑。
4. **饱和/不饱和比值**：不要声称其为"已确立标志物"；正确的表述是"受 SCD 指数与 OEA/PEA 类比启发、有 Grapov 2012 方向性数据支持的候选比值"，可作为你结果的 novelty。
