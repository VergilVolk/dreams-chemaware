# MSV000097426 原始论文调研结果

> 调研日期：2026-07（基于网络检索）
> 数据集：MassIVE MSV000097426 — "Effects of drug, microbiome and diet on the glucuronidome of mice"
> 提交人：Andrew D. Patterson (Penn State, adp117@psu.edu)，2025-03-24；仪器 Orbitrap Exploris 120；物种 Mus musculus (C57BL6/J, 雄性)

## 1. 原始论文标题和 DOI

- **标题**: Glucuronidation metabolomic fingerprinting to map host-microbe metabolism
- **作者**: Boyle, N.R., Sekela, J.J., Wang, M., Mannochio-Russo, H., ... Dorrestein, P.C., Bisanz, J.E., Redinbo, M.R., Patterson, A.D.（通讯）
- **期刊**: Nature Communications 17, 6646 (2026)，Open Access
- **DOI**: [10.1038/s41467-026-73398-1](https://doi.org/10.1038/s41467-026-73398-1)
- **发表**: Received 2026-01-09; Accepted 2026-05-08; Published 2026-05-20 (version of record 2026-07-20)
- 论文 Data availability 中把登录号写作 "MSV00097426"（笔误，少写一个 0），实际即 **MSV000097426**
- 关联数据: 小鼠宏基因组测序 SRA: PRJNA1241896；IGRAM MS/MS: MSV000098859

研究内容摘要：该研究聚焦**葡萄糖醛酸化组 (glucuronidome)** —— 即 UDP-葡萄糖醛酸转移酶(UGT)产生的葡糖醛酸苷结合型代谢物，与肠道菌群 β-葡萄糖醛酸苷酶 (GUS) 的去结合作用之间的平衡。涵盖 GF(无菌)/FMT/常规小鼠、抗生素(庆大霉素/万古霉素)处理、APAP(对乙酰氨基酚)给药、以及 **18 周高脂 vs 常脂饮食(HFD vs NFD)** 小鼠的尿液、血清、结肠内容物/粪便代谢组 + 宏基因组配对分析，并有人群数据验证。

## 2. 鉴定的代谢物（重点 m/z 150–300 Da）

分析平台 MS1 扫描范围 m/z **90–1000**（MS1 分辨率 120K；MS/MS stepped NCE 20/35/50 @30K），因此 150–300 Da 区间完全覆盖。论文核心是**葡糖醛酸苷 (glucuronides)** 鉴定：

- 标准品库：**134 种商品化葡糖醛酸苷标准**（Supplementary Data 12），其中 131 种(负模式)/124 种(正模式)获得 MS/MS
- 检索规则 (MassQL)：葡糖醛酸苷中性丢失 **NL 176.0321 Da**(及较少见的 194.0425 Da，正负模式均有)；负模式特征碎片离子 **m/z 113.0244, 85.0295, 75.0087, 71.0139**（葡糖醛酸基团碎裂产物）
- 因此葡糖醛酸苷与其苷元 (aglycone) 恒差 Δm/z = 176.03

论文中点名的代谢物（按 m/z 归类，[M±H]± 近似值）：

**苷元在 150–300 Da 区间（即葡糖醛酸苷本身在 ~330–480 Da）**：
| 代谢物 | 类别 | 苷元 MW | 葡糖醛酸苷近似 [M-H]- |
|---|---|---|---|
| APAP / 对乙酰氨基酚 | 药物(验证用) | 151 | ~326.1 |
| 3,5-dihydroxyphenylpropanoic acid | 黄酮微生物代谢物 | 182 | ~357.1 |
| equol（雌马酚，R,S equol 7-GlcA & 4′-GlcA） | 大豆异黄酮-daidzein 的菌群代谢物 | 242 | ~417.1 |
| daidzein（7-GlcA, 4′-GlcA, 还有 daidzein-7-sulfate-4′-GlcA） | 大豆异黄酮 | 254 | ~429.1 |
| formononetin | 大豆异黄酮(甲基化) | 268 | ~443.1 |
| genistein（7-GlcA, 4′-GlcA）及其 sulfate | 大豆异黄酮 | 270 | ~445.1 |
| dihydrogenistein glucuronide（暂定） | 大豆异黄酮代谢物 | 274 | ~449.1 |
| naringenin-7-O-β-D-glucuronide（dihydrogenistein 的异构替代物，GUS 酶验证用） | 黄烷酮 | 272 | ~447.1 |
| biochanin A | 大豆异黄酮(甲基化) | 284 | ~459.1 |
| hispidulin glucuronide (6-O-methyl scutellarin) | 黄酮 | 300 | ~475.1 |
| **HFD 关联未鉴定苷元 m/z 220.1366**（配对葡糖醛酸苷 m/z 396.1684） | 未知 | — | 396.1684 |

**葡糖醛酸苷本身落在 150–300 Da 区间**：
- ethyl β-D-glucuronide（乙基葡糖醛酸苷，MW≈222，酒精暴露标志物）
- phenyl β-D-glucuronide（苯基葡糖醛酸苷，MW≈270，苯酚结合物）
- p-cresol glucuronide（对甲酚葡糖醛酸苷，MW≈284，菌群酪氨酸代谢物）
- indoxyl β-D-glucuronide（吲哚酚葡糖醛酸苷，MW≈309，[M-H]- 308.1，略高于 300）

**大分子量端（苷元 >300 Da，供参考）**：bilirubin glucuronide、biliverdin glucuronide（血红素代谢物，正/负模式分子网络分别注释）、β-muricholic acid glucuronide（胆汁酸）、pregnanolone 3-β-D-glucuronide、etonogestrel β-D-glucuronide（甾体）、cotinine N-α-D-glucuronide（尼古丁代谢物）、25-hydroxyvitamin D3 3-glucuronide。

## 3. HFD vs NFD 的主要代谢变化（化学类别）

**主要变化的化学类别 = 葡糖醛酸苷结合型代谢物（phenolic/glucuronidated conjugates，II 相结合物）**，而不是游离脂质本身。具体：

- 实验设计：18 周 purified HFD vs NFD，配对尿液代谢组 + 粪便宏基因组（参考 #63）
- HFD 后 26 个检测到的 GUS 基因中 **5 个显著改变**（2 个 CTD 型、2 个 Loop 1 型——均为小分子葡糖醛酸苷去结合的重要 loop 类型）
- 核心发现：HFD/肥胖与一个 CTD 型 GUS 基因及**葡糖醛酸苷:苷元对 m/z 396.1684 : 220.1366**（MS/MS cosine 0.8734）统计关联；粪便苷元与尿液葡糖醛酸苷均与该 GUS 正相关 → 提示该 GUS 参与去结合并改变该对物质的粪:尿排泄分配
- 该 m/z 396.1684 葡糖醛酸苷在人肥胖/糖尿病数据集 MSV000084112 中同样检测到，且**肥胖组高于对照**（人鼠 MS/MS cosine 0.9892）→ 跨物种保守
- 旁证（同数据集中 GF vs FMT 部分，说明该数据集里饮食/菌群如何塑造葡糖醛酸化组）：FMT 定植后尿葡糖醛酸化组多样性扩大；血清/尿中 indoxyl-、phenyl-、p-cresol-glucuronide（菌群色氨酸/酪氨酸/苯酚代谢物）升高；大豆异黄酮源 equol/dihydrogenistein 葡糖醛酸苷升高；β-muricholic acid 葡糖醛酸苷结肠降低、血清升高（肠肝循环）

**一句话**：HFD 引起的主要是**酚类/异黄酮类/胆汁酸等葡糖醛酸苷结合物**（含吲哚酚、对甲酚、苯基等微生物来源酚类）的丰度与 compartments（粪便↔尿液）再分配变化，由菌群 β-葡萄糖醛酸苷酶(GUS)基因组成变化驱动。

## 4. 离子模式与色谱柱

- **离子模式**：**正、负两种模式都用了**（OmicsDI 摘要原话 "positive and/or negative ionization mode"）。负模式是葡糖醛酸苷鉴定的主模式（特征碎片 113.0244/85.0295/75.0087/71.0139 只在负模式出现；NL 176/194 正负模式均可）；bilirubin/biliverdin 葡糖醛酸苷在正、负模式分子网络中独立注释。尿样 PCA 分析展示的是负模式数据。
- **色谱**：Thermo Vanquish Horizon UHPLC；**Waters ACQUITY UPLC BEH C18 柱**（130 Å, 1.7 µm, **2.1 × 100 mm**），柱温/在线预热 55 °C，样品 5 °C；流动相 A = 水+0.1% 甲酸，B = 乙腈+0.1% 甲酸；梯度 3%B(0 min)→45%B(10 min)→75%B(12 min)→83%B(14 min 保持至 17.5 min)→3%B(18 min 至 20 min 结束)；流速 **250 µL/min**（**反相 C18，非 HILIC**）
- **质谱**：Orbitrap Exploris 120；MS1 m/z 90–1000 @ 120K 分辨率；DDA MS/MS stepped NCE 20/35/50 @ 30K
- （注：仅 GUS 酶活体外实验用 Agilent 1260 Infinity II + Poroshell 120 EC-C18 4.6×150 mm 柱测 naringenin-7-O-β-D-glucuronide/naringenin 定量，非数据集主体方法）

## 来源链接

- MassIVE 数据集页: https://massive.ucsd.edu/ProteoSAFe/dataset.jsp?task=5c07d6ae3144479fb094a56edc28698b
- OmicsDI 条目: https://www.omicsdi.org/dataset/massive/MSV000097426
- 原始论文 (Nature Communications): https://www.nature.com/articles/s41467-026-73398-1
- 论文引用格式: Boyle, N.R., Sekela, J.J., Wang, M. et al. Glucuronidation metabolomic fingerprinting to map host-microbe metabolism. Nat Commun 17, 6646 (2026). https://doi.org/10.1038/s41467-026-73398-1
