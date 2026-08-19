# 癌症代谢组候选数据集清单（草稿，汇总中）

> 状态：4 路并行搜索进行中。本文件逐路追加，全部到齐后去重、合并成最终 ~50 条清单。
> 字段：癌种 / 样本类型 / 物种 / 数据集 ID / MS2-DDA 状态 / 仪器 / 论文(期刊·年·DOI) / 下载链接。
> MS2 状态含义：confirmed=明确 DDA/MS2；likely=非靶 LC-MS 杂交仪器推断；unknown=未标注；absent=明确 MS1-only 或靶向。
> 更新：2026-08-19。

## 路 1：MassIVE + GNPS（18 条，已到）

| # | cancer_type | sample_type | species | accession | MS2/DDA | instrument | paper (journal, year, DOI) | download URL |
|---|---|---|---|---|---|---|---|---|
| 1 | Colorectal | tissue (GEMM intestine + human CRC + MSI) | human + mouse | MSV000092468 | likely | Q Exactive; Xevo G2 XS/SYNAPT G2-Si; TSQ Altis | Metabolic profiling stratifies colorectal cancer and reveals adenosylhomocysteinase as a therapeutic target. Nat Metab, 2023, 10.1038/s42255-023-00857-0 | ftp://massive.ucsd.edu/v06/MSV000092468/ |
| 2 | Hepatocellular carcinoma (liver) | tissue (mouse liver cancer models) | mouse | MSV000100015 | likely | Q Exactive Plus | Impaired nitrogenous waste clearance promotes hepatocellular carcinoma. Sci Adv, 2026, 10.1126/sciadv.aec0766 | ftp://massive-ftp.ucsd.edu/v11/MSV000100015/ |
| 3 | Prostate | tissue (mouse prostate + xenograft) | mouse | MSV000091405 | likely | Q-Exactive | Lipidomics datasets of mouse prostate cancer with AMPK activation (Carling). DOI not confirmed | ftp://massive.ucsd.edu/v05/MSV000091405/ |
| 4 | Colorectal (adipose) | tissue (visceral/subcutaneous adipose) | human | MSV000084042 | absent (GC-TOF MS1) | LECO Pegasus GC-TOF | Metabolomics Workbench ST000061 (Lampe, Fred Hutchinson). DOI not confirmed | ftp://massive-ftp.ucsd.edu/v02/MSV000084042/ |
| 5 | Hepatocellular carcinoma (NASH-HCC) | serum | human | MSV000088003 | likely | TripleTOF 5600+ (SCIEX) | Metabolomics and Lipidomics Screening… in Non-Alcoholic Steatohepatitis. Int J Mol Sci, 2023, 24(1):210, 10.3390/ijms24010210 | ftp://massive.ucsd.edu/v03/MSV000088003/ |
| 6 | Brain glioma | plasma | human | MSV000085693 | likely | Q-Exactive; 6500 QTRAP | Metabolic detection of malignant brain gliomas through plasma lipidomic analysis… EBioMedicine, 2022, 10.1016/j.ebiom.2022.104097 | ftp://massive-ftp.ucsd.edu/v03/MSV000085693/ |
| 7 | Breast | plasma | human | MSV000093325 | unknown (targeted absolute-quant) | Q-Exactive | Multi-omic analysis identifies metabolic biomarkers for early detection of breast cancer… iScience, 2024, 10.1016/j.isci.2024.110682 | ftp://massive-ftp.ucsd.edu/v06/MSV000093325/ |
| 8 | Pancreatic (PDAC) | serum | human | MSV000087756 | likely | Q-Exactive | Profiling of PDAC using serum lipidomics (Wang). DOI not found | ftp://massive.ucsd.edu/v03/MSV000087756/ |
| 9 | Lung | plasma | human | MSV000084032 | absent (GC-TOF MS1) | LECO Pegasus GC-TOF | Metabolomics Workbench ST000396 Lung Cancer Plasma Discovery. DOI not confirmed | ftp://massive-ftp.ucsd.edu/v02/MSV000084032/ |
| 10 | Colorectal | serum | human | MSV000084038 | absent (GC-TOF MS1) | LECO Pegasus GC-TOF | Metabolomics Workbench ST000065 (Lampe). DOI not confirmed | ftp://massive-ftp.ucsd.edu/v02/MSV000084038/ |
| 11 | Lung (NSCLC, post-surgery) | serum | human | MSV000092213 | unknown | unknown | Untargeted Metabolomic Study of Lung Cancer Patients after Surgery with Curative Intent. 2023 (PMC10629266) | ftp://massive.ucsd.edu/MSV000092213/ |
| 12 | Leukemia (AML) | bone-marrow HSPC (cells) | human | MSV000097228 | likely | Bruker impact II Q-TOF | Differentiation, ageing and leukaemia alter the metabolic profile of human bone marrow HSPCs. Nat Cell Biol, 2025, 10.1038/s41556-025-01709-7 | ftp://massive-ftp.ucsd.edu/v09/MSV000097228/ |
| 13 | Leukemia (AML) | cell-line (MOLM14) | human | MSV000087892 | unknown | LTQ Orbitrap Velos; Q Exactive Plus | Metabolomic/13C-glucose profiling of venetoclax+cytarabine in MOLM14. DOI not confirmed | ftp://massive.ucsd.edu/MSV000087892/ |
| 14 | Colorectal | cell-line (HT29) + HepG2 | human | MSV000098191 | likely | Q Exactive | HT29/HepG2 small-molecule incubation metabolomics. Paper not confirmed | ftp://massive-ftp.ucsd.edu/v10/MSV000098191/ |
| 15 | Pancreatic (PDA) | cell-line | mouse | MSV000089514 | unknown (isotope tracing) | 6530A Q-TOF LC/MS | 13C6-glucose tracing in pancreatic cancer cells. Paper not confirmed | ftp://massive.ucsd.edu/MSV000089514/ |
| 16 | Lung (A549) | cell-line | human | MSV000086879 | likely | 6545 Q-TOF LC/MS | Metabolomics Workbench ST001610 (DRB18-treated A549). DOI not confirmed | ftp://massive-ftp.ucsd.edu/v03/MSV000086879/ |
| 17 | Ovarian | cell-line | human | MSV000096771 | likely | Orbitrap Fusion Lumos | Metabolic consequences of erastin-induced ferroptosis in human ovarian cancer cells. Front Mol Biosci, 2025, 10.3389/fmolb.2024.1520876 | ftp://massive-ftp.ucsd.edu/v06/MSV000096771/ |
| 18 | Melanoma | cell-line | human | MSV000098805 | unknown (single-cell MS) | LTQ Orbitrap XL | Investigating metabolomics features in primary/metastatic melanoma cell invasion. Paper not confirmed | ftp://massive-ftp.ucsd.edu/v10/MSV000098805/ |

> 路 1 备注：18 条里 17 条 accession 经 OmicsDI 结构化记录核验（accession/名称/仪器/物种/FTP 均确认）；MSV000092213 仅经其论文(PMC10629266)找到，仪器/MS2/FTP 较不确定。**无一条明确标注 DDA**——描述均未显式写 DDA/MS2，仅从"非靶 LC-MS + 杂交仪器(Q-Exactive/Q-TOF/Orbitrap)"推断为 likely。人肿瘤**组织**代谢组极稀，多为血清/血浆/细胞系。

## 路 2：MetaboLights（专用搜索卡死，未单独返回；但 MTBLS 关键候选已由路 4 覆盖：MTBLS13729/4861/9283/9288/737/3873）

## 路 3：Metabolomics Workbench + Zenodo + OMIX（15 条，已到）

| # | cancer_type | sample_type | species | accession | MS2/DDA | instrument | paper (journal, year, DOI) | download URL |
|---|---|---|---|---|---|---|---|---|
| 1 | Colorectal | tissue (tumor + adjacent mucosa, paired) | human | MW ST001087 | unknown | Agilent 6540 QTOF (LC-MS, ESI±) | Loke MF et al., PLoS ONE 2018;13(12):e0208584. 10.1371/journal.pone.0208584 | metabolomicsworkbench.org/data/DRCCMetadata.php?Mode=Study&StudyID=ST001087 |
| 2 | NSCLC | tissue (lung) | human | MW ST002996 | likely | Thermo LTQ XL (Orbitrap) | not confirmed (n=230; maybe Yamashita 2023 Sci Rep 13:12092) | metabolomicsworkbench.org/data/DRCCMetadata.php?Mode=Study&StudyID=ST002996 |
| 3 | Esophageal adenocarcinoma | tissue (normal/Barrett/EAC) | human | MW ST001942 | unknown | Agilent 6550 QTOF (+ 6490 QQQ) | Molendijk J et al., Clin Transl Med 2022;12(5):e810. 10.1002/ctm2.810 | metabolomicsworkbench.org/data/DRCCMetadata.php?Mode=Study&StudyID=ST001942 |
| 4 | Medulloblastoma (MYC-amp) | tissue (tumor) | human | MW ST002806 | unknown | Agilent 6520 QTOF | Pham K et al., Cancers 2022;14(5):1311. 10.3390/cancers14051311 | metabolomicsworkbench.org/data/DRCCMetadata.php?Mode=Study&StudyID=ST002806 |
| 5 | HCC | tissue (tumor vs non-tumor) | human | MW ST001152 | unknown | Leco GC-TOF + Waters Synapt G2-Si QTOF | Ressom HW et al., J Proteome Res 2019;18(8):3067–3076. 10.1021/acs.jproteome.9b00185 | metabolomicsworkbench.org/data/DRCCMetadata.php?Mode=Study&StudyID=ST001152 |
| 6 | Intrahepatic CCA + HCC | tissue (ICC/HCC vs non-tumor) | human | MW ST000230 | absent (CE-TOF) | Agilent 6220 TOF | not found | metabolomicsworkbench.org/data/DRCCMetadata.php?Mode=Study&StudyID=ST000230 |
| 7 | Breast | tissue | human | MW ST001111 | unknown | ABI Sciex 5600 TripleTOF | not found | metabolomicsworkbench.org/data/DRCCMetadata.php?Mode=Study&StudyID=ST001111 |
| 8 | Prostate | tissue (PCa + benign adjacent, matched) | human | MW ST000784 | confirmed (targeted MRM, NOT DDA) | Agilent 6490 QQQ | Gohlke JH et al., JNCI Cancer Spectr 2019;3(2):pkz019. 10.1093/jncics/pkz019 | metabolomicsworkbench.org/data/DRCCMetadata.php?Mode=Study&StudyID=ST000784 |
| 9 | PDAC | plasma | human | MW ST004767 | likely | Thermo Orbitrap Exploris 120 | not found ("pancreatitis→PDAC" multi-omics) | metabolomicsworkbench.org/data/DRCCMetadata.php?Mode=Study&StudyID=ST004767 |
| 10 | Lung adenocarcinoma | plasma | human | MW ST000386 | absent (GC-MS) | GC-MS | not found | metabolomicsworkbench.org/data/DRCCMetadata.php?Mode=Study&StudyID=ST000386 |
| 11 | Ovarian (serous) | tissue (tumor vs benign cyst) | human | NGDC OMIX007054 | likely | Q Exactive Orbitrap (UHPLC) | "SLC7A5 regulates tryptophan uptake and PD-L1...", Oncology Letters 2025 | ngdc.cncb.ac.cn/omix/release/OMIX007054 |
| 12 | Prostate | unspecified (patient samples) | human | NGDC OMIX008183 | unknown (targeted) | SCIEX QTRAP 6500 | "Integrated proteogenomic characterization of localized prostate cancer...", Nat Commun 2025 | ngdc.cncb.ac.cn/omix/release/OMIX008183 |
| 13 | HCC | plasma | human | NGDC OMIX001067 | unknown | LC-MS (raw POS/NEG) | "Untargeted plasma metabolomics for risk prediction of HCC...", Int J Cancer 2022 | ngdc.cncb.ac.cn/omix/release/OMIX001067 |
| 14 | Pancancer (11 types) | tissue (988 tumor/control) | human | Zenodo 10.5281/zenodo.7348459 | unknown (processed atlas) | multi-platform (processed) | Benedetti E et al., Nat Metab 2023;5(6):1029–1044. 10.1038/s42255-023-00817-8 | doi.org/10.5281/zenodo.7348459 |
| 15 | Cancer metabolism (EGF cells) | cell-line | human | Zenodo 10.5281/zenodo.15605273 | unknown | LC-MS | not found | doi.org/10.5281/zenodo.15605273 |

> 路 3 备注：15 条 accession/仪器/物种均经各库自身元数据(REST API/页面)核验。**无一条明确为 untargeted DDA-MS2**；#8 前列腺是靶向 MRM(三级杆，非 DDA)、#6/#10 是 CE-TOF/GC-MS(无 MS2)。#14 是 Nat Metab 2023 的泛癌处理图集(988 样本)，但为处理后的 atlas 包、非原始谱。

## 路 4：近三年泛癌种（14 条，已到）

| # | cancer_type | sample_type | species | accession | MS2/DDA | instrument | paper (journal, year, DOI) | download URL |
|---|---|---|---|---|---|---|---|---|
| 1 | Colorectal | tissue (tumor + matched normal, 30 pts) | human | MTBLS13729 | confirmed (UHPLC-MS/MS stated) | Q Exactive Orbitrap UHPLC-MS/MS + MALDI-MSI | J. Proteome Res., 2025, 10.1021/acs.jproteome.5c01260 | https://www.ebi.ac.uk/metabolights/MTBLS13729 |
| 2 | Ovarian | uterine fluid | human | MTBLS4861 | unknown | Q Exactive | Cell Rep. Med., 2023, 10.1016/j.xcrm.2023.101061 | https://www.ebi.ac.uk/metabolights/MTBLS4861 |
| 3 | Breast | tissue — PDX tumor (basal/luminal) | human (xenograft) | MSV000098703 | confirmed (DDA stated) | Agilent 6560 IMS-QTOF + Orbitrap Fusion Lumos | LipidIMEA method paper (dataset DOI 10.25345/C55T3GC44) | https://massive.ucsd.edu/ProteoSAFe/dataset.jsp?task=eef7361fa490451e8def7612c94a58cf |
| 4 | Pancreatic | tissue — PDAC GEMM pancreas | mouse | MSV000097601 | likely | Q Exactive | bioRxiv preprint, 2025, 10.1101/2025.05.20.655200 | https://massive.ucsd.edu (MSV000097601) |
| 5 | Pancreatic | cell line (PANC-1 / SW1990) | human | MTBLS9283 / MTBLS9288 | unknown | Thermo LC-MS (RP pos/neg) | alkaliptosis study, 2025 | https://www.ebi.ac.uk/metabolights/MTBLS9283 |
| 6 | Liver (HCC) | tissue (post-resection) | human | PXD019864 (iProX) | unknown | Q Exactive HF | J. Proteome Res., 2020, 19(8):3533–3541 | http://www.iprox.org (PXD019864) |
| 7 | Lung (NSCLC) | tissue — lung adenocarcinoma | human | PXD047520 / IPX0007556000 (iProX) | unknown | not stated | iProX deposition, 2023 | http://www.iprox.org/page/project.html?id=IPX0007556000 |
| 8 | Lung (NSCLC) | tissue (cancer/benign/adjacent/distal, 131 pts) | human | PXD019969 (iProX) | unknown | LC-MS | iProX, 2020 (Guowang Xu group) | https://proteomecentral.proteomexchange.org/cgi/GetDataset?ID=PXD019969 |
| 9 | Lung (NSCLC) | tissue — lung adenocarcinoma | human | OMIX010825 (GSA) | unknown | not stated | Cell Rep., 2026, 10.1016/j.celrep.2026.117427 | https://ngdc.cncb.ac.cn/omix/release/OMIX010825 |
| 10 | Prostate | tissue (localized PCa) | human | OMIX008183 | unknown (restricted access) | not stated | proteogenomic study, 2025 (PMC11968977) | https://ngdc.cncb.ac.cn/omix/release/OMIX008183 |
| 11 | Gastric | tissue (tumor) | human | OMIX004317 | unknown | not stated | OMIX, 2025 (PRJCA017614) | https://ngdc.cncb.ac.cn/omix/release/OMIX004317 |
| 12 | Kidney (ccRCC) | tissue (paired ccRCC reference set) | human | Zenodo 10.5281/zenodo.11286535 | unknown | not stated | Nature Cancer, 2025, 10.1038/s43018-025-00943-0 | https://doi.org/10.5281/zenodo.11286535 |
| 13 | Kidney (ccRCC) | cell line (conditioned media) | human | MTBLS737 | unknown | Waters Xevo G2S QTOF | J. Proteome Res., 2018, 10.1021/acs.jproteome.8b00538 | https://www.ebi.ac.uk/metabolights/MTBLS737 |
| 14 | Brain (glioma) | tissue (astrocytoma/oligo/GBM, 101 samples) | human | MTBLS3873 | absent (NMR, not MS2) | NMR | JCI Insight, 2022, 10.1172/jci.insight.153526 | https://www.ebi.ac.uk/metabolights/MTBLS3873 |

> 路 4 备注：仅 #1(MTBLS13729 CRC 组织) 与 #3(MSV000098703 乳腺 PDX 组织) 明确标注 MS2/DDA；其余 acquisition mode 均未标注故标 unknown。#1 是**「人癌组织 + 明确 MS/MS + 已发表(J Proteome Res 2025) + MetaboLights 可下载」**的唯一命中的组合。脑瘤组织只有 NMR(#14)、肾组织只有 Zenodo 参考集(#12)；#10 前列腺为受限访问。
