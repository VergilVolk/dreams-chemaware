# MTBLS13729 × DreaMS 分析预案：如何改进作者结论、回答他们的开放问题

> 数据集：MTBLS13729（MetaboLights），对应论文 "Integrated Spatial and Bulk Untargeted
> Metabolomics Characterize Location- and Histology-Associated Metabolic Heterogeneity in
> Colorectal Cancer"，J Proteome Res 2026，DOI 10.1021/acs.jproteome.5c01260。

---

## 1. 标准非靶代谢组学文章到底怎么做的（回答"注释完库就没了吗"）

非靶代谢组学的标准流程有 6 步，**注释只是第 3 步，不是终点**：

1. 采集（LC-MS/MS，DDA 或 DIA）
2. 峰检测 / 特征提取（XCMS / MZmine / MS-DIAL）
3. **注释**（按 Schymanski 2014 五级置信度，DOI 10.1021/es5002105）
   - L1：标准品 + RT 确证（最高）
   - L2：MS/MS 谱图匹配谱库（结构注释）
   - L3：分子式（精确质量 + 同位素）
   - L4：精确质量（m/z）匹配
   - L5：未知
4. 统计分析（单变量 + 多变量，差异/聚类）
5. 通路富集（KEGG / MetaboAnalyst）
6. 生物学解释 + 验证（多组学整合、靶向验证）

**关键瓶颈**：非靶代谢组学的注释率通常 **<10%**，>90% 的特征是"暗物质"（dark matter，
da Silva et al. 2015，DOI 10.1073/pnas.1506877112）。所以一篇标准文章**真正的结论只建立
在少数已注释分子上**——"注释完库之后"不是没了，而是**剩下 90% 没注释的信号被静默丢弃**，
通路富集和生物学解释都只看到冰山一角。这正是我们的切入点。

---

## 2. 这篇论文作者的实际做法（实测核验，非推测）

**技术组合**（两套数据）：
- **空间代谢组（MALDI-MSI）**：仅 1 例罕见"双侧同时原发 CRC"（47 岁女性，左右侧 TNM 分期
  相同，用于最小化遗传/宿主混杂）——探索性、假设生成。
- **整体非靶（UHPLC-HRMS/MS，DDA）**：30 对配对肿瘤/癌旁组织，按部位 + 组织学分层
  （左半管状 / 右半管状 / 右半黏液）——即 MTBLS13729 的 240 个 mzML（4 面板）。

**作者注释（实测 maf.tsv，345 条）**：
- 只填了 `mass_to_charge` + `retention_time` + `metabolite_identification`（HMDB 名字）；
- `chemical_formula` / `smiles` / `inchi` / `search_engine_score` / `reliability` **全空（0/86）**；
- 98% 有 HMDB ID，但**没有结构、没有谱图打分、没有置信度** → 相当于 Schymanski **L3–L4**（m/z+RT），
  连 L2（MS2 谱匹配）都没到；
- **异构体歧义直接暴露**：`2-Aminobutyric acid 或 2-Aminoisobutyric acid`、
  `Glycoursodeoxycholic acid 或 Glycochenodeoxycholic acid`——UDCA vs CDCA 两种胆酸生物学
  意义截然不同，m/z 分不出来，作者自己写"或"。

**作者结论**：
- 部位：左半管状腺癌富集**醚脂 + 磷脂酰胆碱**代谢；右半富集**核苷酸 + 脂肪酸**代谢。
- 组织学：右半**黏液腺癌**呈独特**鞘脂**改变，提示去饱和酶相关的"促凋亡→结构功能"转变。

**作者明确承认的局限**：空间结果是单病例、探索性；需要多中心/多患者验证。

---

## 3. 我们的改进路径（DreaMS 差异化）

DreaMS（Bushuiev et al. Nat Biotechnol 2025，DOI 10.1038/s41587-025-02663-3）做的是
**MS2 谱图级结构注释**：precursor token + 线性头编码，cosine 相似 + **m/z 硬约束** +
**target-decoy FDR**（Elias & Gygi 2007，DOI 10.1038/nmeth1019，蛋白组转代谢组）。

四条具体改进，逐条可量化：

1. **从"名字"到"结构 + 置信度"**：把作者 345 条 L3/L4 注释升到 **L2a**（MS2 匹配 + InChIKey +
   FDR 控制），每条带 cosine / qvalue，可筛假阳性。
2. **解开异构体歧义**：作者写"或"的（胆酸 UDCA/CDCA、氨基丁酸异构体）用 MS2 谱图区分——
   这是 m/z 维度做不到、恰好 DreaMS 能做的地方。
3. **攻暗物质**：作者 240 个 mzML 只注释了 345 条（每样本 ~5300 张 MS2 × 60 = ~32 万张谱），
   绝大多数字段特征没注释。我们用 DreaMS 重注释，目标是把**差异显著但作者没注释**的特征找出来。
4. **更严统计**：作者没给 FDR；我们用样本级 presence/absence Fisher + BH（已有的 diff 管线）。

**回答他们的开放问题**（论文自己没答的）：
- "鞘脂改变"具体是哪些鞘脂分子（作者只给了类别，没给结构）？
- "醚脂/磷脂酰胆碱"差异是哪些具体分子？
- 暗物质里有没有部位/组织学差异更强的、作者完全看不见的信号？

---

## 4. 分阶段分析预案

- **阶段 A（已完成）**：pos-RP 60 mzML 下载 + 管线脚本就绪 + 作者注释基线（本文件 §2）。
- **阶段 B（进行中）**：4 面板 × 60 = 240 mzML 全量下载（neg_rp / pos_hilic / neg_hilic）。
- **阶段 C**：服务器 GPU 跑 DreaMS 全流程（mzML→hdf5→嵌入→MONA 注释→FDR），**4 面板各自**
  注释（正/负离子互补覆盖不同代谢物类别；RP/HILIC 互补覆盖不同极性）。
- **阶段 D**：差异分析（tissue / location / histology 三维，4 面板合并），得显著差异特征。
- **阶段 E（对照作者，核心交付）**：
  1. **复现**：我们的差异特征能否复现作者的"左醚脂/右核苷酸/黏液鞘脂"方向？
  2. **升级**：把作者 345 条名字注释升级为结构 + 置信度（尤其解开胆酸/氨基丁酸歧义）。
  3. **新发现**：作者未注释、我们新注释、且部位/组织学差异显著的分子——即"暗物质里的新结论"。
  4. **交叉验证**：与作者的 MALDI-MSI 空间结论（左半肿瘤核心→侵袭边缘梯度）对照。
- **阶段 F**：通路富集（KEGG）+ 生物学解读，形成可发表的"改进版"结论。

## 5. 怎么证明我们"改进了"（可量化指标）

| 指标 | 作者 | 我们（目标） |
|---|---|---|
| 注释置信度 | L3/L4（m/z+RT，无结构） | L2a（MS2 匹配 + FDR + InChIKey） |
| 结构信息 | 0（SMILES/InChI 全空） | 每条带 InChIKey |
| 假阳性控制 | 无 | target-decoy FDR（qvalue） |
| 异构体区分 | 写"或" | MS2 谱图区分 |
| 注释覆盖 | 345 条（浅） | 目标 >345，攻暗物质 |

**判定成功的门槛**：能在作者 345 条之外，给出（a）结构确认、（b）解开歧义、（c）新增若干
"部位/组织学差异显著且作者未注释"的分子——三者至少成立其一且 FDR 受控。
