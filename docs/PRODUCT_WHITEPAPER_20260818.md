# Chem-aware DreaMS — LC-MS/MS 质谱注释平台


> 权威代码记录见 `docs/ANNOTATION_PLATFORM_ARCHITECTURE_20260817.md`；本白皮书是面向外部的产品叙事 + 工程说明书。


## 0. 一句话定位

**DreaMS 把一张 MS/MS 谱图压成一个 1024 维向量。** 我们的平台是在这个向量之上，补上 DreaMS 原论文里没有交付的一整条「从原始谱到可发表的注释结论」的流水线——**可审计的 FDR、可校准的概率、Schymanski 置信度分级、组间差异、通路富集、暗物质挖掘**；并在这个冻结编码器之外，做了一套**「化学规则注入编码器注意力」**的研究层，目标是让编码器本身对碎片化学更敏感。



## 1. 与 DreaMS 的关系：照搬了什么、多了什么（核心对比）


### 1.1 照搬（原样使用，不声称是我们的贡献）
| 项 | 说明 | 出处 |
|---|---|---|
| 自监督 MS/MS 编码器 backbone | Graphormer 风格 Transformer，输出 1024 维，position-0 前体 token 作为谱表示 | Bushuiev et al., *Nat Biotechnol* 2025, DOI 10.1038/s41587-025-02663-3 |
| 官方嵌入公式 | `precursor = backbone(batch, None)[:, 0]`；`e = L2_norm(linear(precursor, W, b))` | 同上 |
| 峰值预处理 | top-100 峰、按最大峰归一化到 [0,1]、前置合成前体峰 `[precursor_mz, 1.1]` | 同上 |
| 权重 | `dreams/models/pretrained/ssl_model_server.pt`（464 MB backbone）+ 线性 head `data/e1/official_embedding_slim.pt`（468 MB） | 官方 checkpoint |


### 1.2 平台层新增（DreaMS 没交付、我们补上的完整工作流）
DreaMS 论文做的是「训练编码器 + 检索评测」，它**没有**把这些做成可复现、可审计、可对外交付注释结论的管线。我们补的每一块都挂文献：

| 模块 | 我们加的东西 | 文献 |
|---|---|---|
| M1 检索 | **前体 m/z 硬约束**（±ppm）叠加在 cosine 排序上 | DreaMS 评测同款策略；实测 raw cosine 会过度注释约 4× |
| M2 置信度 | **Schymanski 1–5 级**（诚实封顶 2a+3） | Schymanski 2014, DOI 10.1021/es5002105 |
| M2b 规则证据 | 335 条化学规则 → 诊断证据（语义升级，非置信度升级） | McLafferty；Kind & Fiehn 2007；Tsugawa 2016 |
| M3 FDR | **目标-诱饵 FDR → q-value** | Elias & Gygi 2007, DOI 10.1038/nmeth1013；Scheubert 2017, DOI 10.1038/s41467-017-01318-5 |
| M4 校准 | **score → P(correct)**（Platt / isotonic） | Platt 1999；Zadrozny & Elkan 2002；Hoffmann 2022, DOI 10.1038/s41587-021-01045-9 |
| M5 差异 | DDA 谱计数 + Fisher 精确检验 + BH | Fisher 1925；Benjamini & Hochberg 1995 |
| M6 通路 | 超几何 ORA + mummichog 式 m/z 富集 | Li 2013, DOI 10.1371/journal.pcbi.1003123 |
| M7 暗物质 | 暗谱聚类 + 候选 lead | Cao 2025, DOI 10.1021/jacsau.5c01063 |
| M8 消融 | 模块 on/off 阶梯，量化每步贡献 | — |

### 1.3 研究层（真正「超越 DreaMS 编码器」的方向，**进行中、有因果门**）
`dreams/models/chem_aware/` 下是「化学感知编码器」的研究分支（module-1 分支）：

- **ChemicalRuleEngine v4**：6 类机制性化学规则（NL 中性丢失 / CF 特征碎片 / ISO 同位素 / NR 氮规则 / EE 偶电子 / HR 氢重排，约 335 条主库），每条带 source 溯源。
- **ChemAwareDreaMS v3**：继承 DreaMS，**只在最后一层**注入化学规则注意力奖励偏置；每条规则维度独立可学习权重（softplus 非负）；带 `chem_attn_enabled` 开关——`False` 时与原版 DreaMS **逐位一致**（这是做干净消融的前提）。
- 配套：门控/协同损失（v3 已弃用，保留供参考）、训练脚本、反事实/因果残差审计等实验。

> **重要诚实声明**：这套化学感知编码器**尚未证明在检索上超越冻结 DreaMS**，正处于 `docs/ChemAware_SOTA_literature_audit_20260807.md` 定义的六道因果门（G0–G5）之下。当前交付管线用的是**冻结 DreaMS backbone**，不是 ChemAwareDreaMS。二者共用同一套 335 规则库——一套化学先验、两个落点：**注释层的证据（已交付）** 和 **编码器层的注意力偏置（研究中）**
---

## 2. 系统架构与数据流（端到端）

```
[原始 .raw] ──ThermoRawFileParser(v1.4.5, 已验证与 GNPS 逐位一致)──► [.mzML]
     │                                                                   │
     │                                                        MSData.load(in_mem=False)
     ▼                                                                   ▼
 [hdf5: RT/charge/precursor_mz/ms_level/polarity/scan_number/spectrum]   [参考库 MGF]
     │                                                                   │
     ▼ preprocess_spectrum (top-100, max-norm, 前置前体峰)  ◄─────────────┘  parse_mgf
     │                                                                   → records {peaks(2,n), precursor_mz, smiles, inchikey, name}
     ▼                                                                          │
 embed_hdf5 / embed_records  ◄──────────────────────────────────────────────────┘
 (冻结 DreaMS backbone + 线性 head, 取 position-0 前体 token, L2 归一化)
     │                                                        │
     ▼                                                        ▼
 embeddings.npy [N×1024]           同一个 backbone+head      embeddings.npy [M×1024]
 manifest.csv (file/scan/pmz/...)                          manifest.csv (smiles/inchikey/name/pmz)
     \_______________________________________________________/
                           │
                 retrieve.chunked_topk (cosine top-k)
                 + 前体 m/z ±20 ppm 硬约束  ◄── 关键：杀掉 76% 的 raw-cosine 错误 top-1
                           │
                           ▼
               annotations.csv（long 表：每 query × rank 一行）
               cosine / dppm / mz_pass / lib_* / query_*
                           │
   ┌───────────┬───────────┼────────────┬────────────┐
   ▼           ▼           ▼            ▼            ▼
 M2 分级    M2b 规则    M3 FDR      M4 校准      M7 暗物质
 +schymanski +diagnostic  +qvalue     +calibrated   dark_mask→cluster→lead
  _level(2/3/5) _rule_ev   +fdr_pass   _prob
   │           │           │            │            │
   └───────────┴───────────┴────────────┴────────────┘
                           │
                           ▼
        M5 差异（Fisher+BH）        M6 通路（ORA/mummichog）
        diff.csv（化合物×组差异）    pathway.csv（通路富集）
                           │
                           ▼
        M8 消融阶梯（raw cosine → +m/z → +FDR）
                           │
                           ▼
              report.json / annotations.csv（可审计中间产物全落盘）
```

**设计原则：每一步中间产物独立落盘、可单独重跑、可审计。** 没有黑箱串联——任何一个模块坏了，其它模块照常。

---

## 3. 模块清单（M0–M9）

| 模块 | 文件 | 职责 | 关键文献 |
|---|---|---|---|
| M0 | `params.py` | 全部阈值/方法，逐条挂 DOI，`frozen dataclass` | `SOURCES` 字典 |
| M1 | `_inference.py` `embed.py` `retrieve.py` | 预处理 → 官方嵌入 → cosine top-k + m/z 硬约束 | Bushuiev 2025 |
| M2 | `confidence.py` | Schymanski 1–5 级（封顶 2a+3） | Schymanski 2014 |
| M2b | `rule_evidence.py` | 335 规则 → 诊断证据（Level 3 语义升级） | McLafferty / Kind&Fiehn / Tsugawa |
| M3 | `fdr.py` | shuffle 诱饵 + TDA q-value | Elias&Gygi 2007；Scheubert 2017 |
| M4 | `calibrate.py` | Platt/isotonic → P(correct) | Platt 1999；Zadrozny&Elkan 2002；Hoffmann 2022 |
| M5 | `diff.py` | DDA 谱计数 + Fisher + BH | Fisher 1925；BH 1995 |
| M6 | `pathway.py` | 超几何 ORA + mummichog | Li 2013 |
| M7 | `darkmatter.py` | 暗谱聚类 + 候选 lead | Cao 2025；Schymanski 2014 |
| M8 | `ablation.py` | 模块 on/off 阶梯量化 | — |
| M9 | `cli.py` | 端到端命令入口 | — |

---

## 4. 怎么解析（注释方法论）

### 4.1 预处理（与 DreaMS 官方一致）
1. 每张 MS2 谱取 top-100 峰（按强度，稳定排序）；
2. 峰强度按最大峰归一化到 [0,1]；
3. 前置一个合成前体峰 `[precursor_mz, 1.1]`——让模型「知道」这张谱的前体质量。

### 4.2 嵌入
- 冻结 DreaMS backbone 前向，取 **position-0 前体 token**；
- 过官方线性 head，L2 归一化 → 1024 维单位向量。
- 查询谱与参考库用**同一个** backbone+head 嵌入，cosine 才有意义。

### 4.3 检索 + 硬约束（平台最关键的一步）
- cosine top-k 候选；
- **前体 m/z 必须在 ±20 ppm 内**（同加合物）。这一步是平台与「裸 DreaMS cosine」的本质区别：实测 raw cosine 的 top-1 有 73% 是 m/z 差 >1000 ppm 的假阳性，硬约束把这批错误直接杀到 0（FP 代理 0.764 → 0.000）。

### 4.4 置信度分级（Schymanski，诚实封顶）
| 级别 | 含义 | 平台是否输出 |
|---|---|---|
| Level 1 | 标准品同机 RT+MS/MS 确认 | ❌ 永不输出（需标准品） |
| Level 2a | 谱库匹配（cos≥0.7 且 m/z 通过） | ✅ 平台上限 |
| Level 3 | 推测候选（m/z 通过 + 结构线索：低 cosine 或规则证据） | ✅ |
| Level 4 | 明确分子式（需 SIRIUS 类预测器） | ❌ 永不输出 |
| Level 5 | 仅精确质量 | ✅（无结构线索时） |

### 4.5 FDR（q-value）
- shuffle 诱饵：保持前体 m/z、打乱碎片强度，1:1 嵌入；
- q(s) = min_{s'≥s} (N_decoy(s')+1)/(N_target(s')+1)；
- q ≤ 0.01 才叫 confident。

### 4.6 校准（score → P(correct)）
- 用参考库 **leave-one-out 自检索**构造标注集（命中同 InChIKey14 = 正确）；
- Platt 或 isotonic 拟合，输出 `calibrated_prob = P(correct | cosine)`。
- 诚实：raw cosine 不是概率（正确命中均值 0.884 vs 错误 0.781），校准后才能对外报「这个注释有多大概率对」。

### 4.7 差异与通路
- 差异：DDA MS2 谱计数（半定量）→ Fisher 精确检验 → BH 校正；
- 通路：注释化合物走超几何 ORA；未注释 m/z 走 mummichog 式富集。

### 4.8 暗物质
- 94.1% 的谱无 confident 注释 → 暗谱按 embedding cosine 贪心聚类 → 每个簇报低 cosine 结构 lead（候选，不是确认）。



## 5. 差异化模块

### 5.1 前体 m/z 硬约束（M1，已交付，最朴素但最有用）
DreaMS 的嵌入编码的是**碎裂（结构）远多于质量**（前体 m/z 只是前置的一个峰），所以裸 cosine 会过度注释。硬约束是平台准确率的第一来源，也是消融里从 0.248 拉到 0.059、FP 从 0.764 拉到 0.000 的那一步。

### 5.2 目标-诱饵 FDR + 校准（M3+M4，已交付）
DreaMS 给的是 cosine 分数，**没有错误率、没有概率**。我们补上「q-value（1% FDR 线）」和「P(correct)」，让注释结论第一次可量化地诚实。

### 5.3 化学规则证据（M2b，已交付，但诚实定位）
335 条机制性规则（NL/CF/ISO/NR/EE/HR），每条带 source 溯源。实测结论（非假设）：
- 100% 谱命中 ≥1 规则 → 「命中任意规则」无区分度；
- CF 命中稀疏（1.29/谱）但**不预测**注释正确率；
- 因此规则证据是 **Schymanski 语义升级**（精确质量 → 有碎裂解释的 tentative candidate），**不是置信度升级**。

**这条结论本身就是科研价值**：它证伪了「加规则=更准」的朴素直觉，并写死在 `rule_evidence.py` 的 docstring 里。

### 5.4 化学感知编码器（研究层，进行中，有因果门）
`ChemAwareDreaMS v3`：把同一套规则编码成**最后一层注意力偏置**，每条规则维度独立可学习、匹配加分、不匹配不罚。设计上有几个 DreaMS 没有的点：
- **仅注入最后一层**（避免跨层复合放大，前 6 层自由学习）；
- **逐规则独立权重**（好规则不被坏规则拖累，训练完打印「哪些规则有用」即科学发现）；
- **等价开关**（`chem_attn_enabled=False` 与原版 DreaMS 完全一致，消融干净）。

当前方向（2026-08-17 定案）：**带权规则挖样本 + MS2DeepScore 噪声微调**，仍在 `ChemAware_SOTA_literature_audit` 的 G0–G5 门控下，**不承诺超越**。

---

## 6. 怎么超越 DreaMS

**第一层（已验证）：DreaMS 是编码器，我们是平台。**
DreaMS 交付的是「谱 → 向量」；我们交付的是「谱 → 带 FDR + 校准概率 + Schymanski 分级 + 组间差异 + 通路 + 暗物质候选的注释结论」。这一层**不依赖任何未验证的创新**，纯工程 + 标准统计方法，是当下就能交付的硬超越。

**第二层（已验证的科学发现）：我们证伪了一个直觉。**
「往 DreaMS 加化学规则会变准」——我们实测了，规则证据不预测正确率，只做语义升级。这条**负结果**本身是 DreaMS 生态里少有的、可复现的经验证据，说明我们不是「堆规则」而是「验证规则」。

**第三层（研究前沿，未验证，不对外承诺）：化学感知编码器。**
目标是让编码器本身对碎片化学更敏感，超越冻结 backbone。这是论文级的方向，但按纪律**只有通过 G0–G5 六道因果门、且在 MassSpecGym v1.5 同协议下显著超过最强修正基线，才允许声称「检索超越 DreaMS」**。现在不承诺、不写进「已验证」表。

**一句话对外讲法**：
> 「我们不是重训一个编码器去碰运气。我们把 DreaMS 的向量真正用起来——补上它论文里没有的 FDR、概率校准和置信度分级，让它第一次能给医学/非靶向代谢组一个可量化、可审计的注释结论；同时在做一套把机制性化学规则注入编码器注意力的研究，方向明确、验证严格，但不拿未验证的承诺当卖点。」



## 7. 预留的新模块接口（扩展点，代码里真实存在）

以下每个接口都在代码里，标注「现状」与「怎么扩展」。接入**无需改动其它模块**。

| # | 扩展点 | 代码位置 | 现状 → 扩展 |
|---|---|---|---|
| 7.1 | 规则证据接入 | `confidence.assign_schymanski(rules_evidence=...)` | 已接 335 主库 → 可换成 mokapot 式 rescoring 特征（`fdr.FDR_CITATIONS["rescoring"]` 已挂 DOI，打分器未实现） |
| 7.2 | 诱饵策略 | `params.decoy_strategy: "shuffle"\|"fragment-tree"\|"precursor-swap"` | 只实现 shuffle → 可加 fragment-tree（passatutto 主路线）/ precursor-swap（复用 target 嵌入、零额外成本） |
| 7.3 | 校准器 | `calibrate.fit_calibrator(method=...)` | platt/isotonic → 加 temperature scaling、Bayesian binning、`none` 直通 |
| 7.4 | 定量后端 | `diff.group_counts` | DDA 谱计数（半定量）→ 换 MS1 peak-area / DIA / MaxQuant，Fisher/BH 层不变 |
| 7.5 | 通路数据库 | `pathway.enrich_annotated(mapping=...)` | 算法完整，**不随代码交付 compound→pathway 表**（避免编造）→ 传入 HMDB/KEGG/Reactome DataFrame 即用 |
| 7.6 | 暗物质类别先验 | `darkmatter.DARK_CITATIONS["class_prior"]` | 标注「not yet wired」→ 接 CANOPUS（Dührkop 2021）给暗簇化合物类别先验 |
| 7.7 | 检索后端 | `retrieve.chunked_topk(chunk=512)` | 朴素 numpy argpartition → 换 faiss IVF/HNSW / GPU matmul / 多机分片，对外只暴露 `(topk_vals, topk_idx)`，下游无感 |
| 7.8 | 嵌入后端 | `embed.DEFAULT_RAW` / `DEFAULT_OFFICIAL` | 指向官方冻结 backbone → 换自定义 backbone / 对 head 领域微调（`reconstruct_backbone` 已分离 backbone/head） |
| 7.9 | CLI 子命令 | `cli.build_parser` | 现 `embed`/`annotate` → 加 `diff`/`pathway`/`darkmatter`/`report`；加 `--params-json` 覆盖任意 `Params` 字段 |
| 7.10 | 报告输出 | `retrieve.save` / `cli.cmd_annotate` 末尾 | `annotations.csv`+`report.json`+消融文本 → 加 HTML/PDF 渲染（面向医学科学交付） |
| 7.11 | 化学感知编码器接入 | `embed.load_embedder` ↔ `dreams/models/chem_aware/ChemAwareDreaMS` | 当前管线用冻结 backbone → 待 G0–G5 通过后，把 `load_embedder` 换成 `ChemAwareDreaMS`（`chem_attn_enabled=True`）即可无缝替换 |

---

## 8. 部署

### 8.1 依赖（`requirements.txt`）
```
torch==2.2.1（GPU 服务器改 +cu121）· numpy==1.26.4 · pandas==2.3.3 · h5py==3.11.0
pyteomics==5.0 · psims==1.3.6 · lxml==4.9.4 · scipy==1.15.3 · networkx==3.6.1
scikit-learn==1.9.0 · matplotlib==3.11.0 · tqdm==4.68.3 · pytorch-lightning==2.0.8
```
验证：`python -c "import torch,numpy,pandas,h5py,pyteomics,scipy,sklearn; print('ok')"`

### 8.2 要迁移的资产（本地 → 服务器）
- 代码：`annotation/`（236 KB）+ `dreams/`（含 backbone 权重 464 MB）+ `tasks/`；
- 权重：`data/e1/official_embedding_slim.pt`（468 MB）；
- 参考库：`data/models/mona_neg_full.mgf`（141 MB）+ `data/models/mona_neg_dreams_emb/`（155 MB）；
- 规则：`dreams/models/chem_aware/chem_rules_data.json`（100 KB，335 主规则）；
- 数据：`.mzML`（Met/neg ~7 GB，本地已转好）或 `.raw`（120 GB，服务器转）。

### 8.3 运行顺序（服务器 SLURM，见 `run_annotate_met_neg.sbatch`）
```
(1) mzML→hdf5   python tasks/convert_met_neg_hdf5.py --mzml-dir ...
(2) 嵌入(GPU)   python tasks/encode_msv100574_spectra.py ... --device cuda
(3) 注释+规则   python -m annotation.cli annotate ... --rules
(4) FDR(GPU)    python tasks/run_fdr_met_neg.py --device cuda
(5) 差异分析    python tasks/run_diff.py --group-a PF --group-b HF
```
提交：`sbatch run_annotate_met_neg.sbatch`（`#SBATCH --gpus=1` / `--time=24:00:00`，照仓库 `run_e1_identity.sbatch` 模式）。
本地 Windows cmd 版 FDR 命令见 `docs/SERVER_DEPLOYMENT_20260818.md` 第 5 节（**不要用 bash 的 `nohup`/`$!`/`tail`**）。

---

## 9. 优势与前沿程度（怎么对外讲）

**可辩护的四点优势：**

1. **可量化、可审计**——每个注释带 q-value（1% FDR 线）和校准概率 P(correct)，每个阈值可追溯到 DOI，中间产物全落盘。这是「医学科学/非靶向代谢组」最需要的诚实。

2. **硬约束的工程价值**——前体 m/z ±ppm 硬约束是平台准确率第一来源，直接杀掉裸 cosine 76% 的假阳性。DreaMS 论文评测里有这个策略，但没有把它做成可复现交付。

3. **负结果也是成果**——实测证伪「加规则=更准」，把规则证据正确定位为 Schymanski 语义升级。这套「验证而非堆砌」的纪律，是区分「工程堆料」和「严谨方法学」的分水岭。

4. **明确的前沿路线 + 严格门控**——化学感知编码器方向清晰（规则注入最后层注意力 + 逐规则可学习权重 + 等价开关做干净消融），但只走 `ChemAware_SOTA_literature_audit` 的六道因果门，不拿未验证的承诺当卖点。

**前沿程度（诚实版）：**
- 平台层 = 标准方法学 + 工程整合，**成熟、可交付**，前沿性在于「把 DreaMS 真正用起来并做到可审计」，而非算法原创；
- 研究层 = 机制规则锚定 + 因果忠实解释，属当前质谱检索的**开放前沿**，但按 2026 MassSpecGym v1.5 审计的标准，**不承诺通用检索 SOTA**，只在困难异构体/解释忠实性上争取可辩护的增益。

---

## 10. 已验证数字 + 已知限制（诚实声明）

**已验证数字（Met/neg，13,770 谱 × 36,663 谱 MONA-neg 库）：**
- 消融：raw cosine 0.248（FP 代理 0.764）→ +m/z 0.059（FP 代理 0.000）
- 校准：库内 leave-one-out 70.5% 同化合物命中；正确 hit 均 cos 0.884 vs 错误 0.781
- 暗物质：94.1% 谱无注释
- 规则证据：100% 谱命中 ≥1 规则；NL 平均 18.6 条/谱；CF 不预测正确率；实际只翻转 1 个 Level（5→3）

**已知限制（不隐瞒）：**
1. **Level 1 / Level 4 永不输出**（需标准品 / 分子式预测器 SIRIUS）。
2. **M6 不随代码交付 compound→pathway 表**（HMDB/KEGG/Reactome 是外部资源，需提供，避免编造映射）。
3. **M5 是半定量**（DDA 谱计数，精度低于 MS1 peak-area / DIA），且需每组 replicate。
4. **FDR 需 1:1 全量诱饵**（36,663 诱饵 ≈ 68 min CPU / 1–2 h GPU 之前）；子集诱饵 q-value 塌缩、无区分度，不能代替。
5. **参考库覆盖有限**——注释率 5.9%，94% 为暗物质；鞘脂等需专用库（脂质组）。

---

## 附：关键文献 DOI 速查

| 方法 | 文献 | DOI |
|---|---|---|
| DreaMS 编码器 | Bushuiev et al., Nat Biotechnol 2025 | 10.1038/s41587-025-02663-3 |
| Schymanski 分级 | Environ Sci Technol 2014 | 10.1021/es5002105 |
| 目标-诱饵 FDR | Elias & Gygi, Nat Methods 2007 | 10.1038/nmeth1013 |
| passatutto | Scheubert et al., Nat Commun 2017 | 10.1038/s41467-017-01318-5 |
| mokapot rescoring | Fondrie & Noble, J Proteome Res 2021 | 10.1021/acs.jproteome.1c00410 |
| 校准/高置信注释 | Hoffmann et al., Nat Biotechnol 2022 | 10.1038/s41587-021-01045-9 |
| mummichog | Li et al., PLoS Comput Biol 2013 | 10.1371/journal.pcbi.1003123 |
| 暗代谢物组 | Cao et al., JACS Au 2025 | 10.1021/jacsau.5c01063 |
| CANOPUS | Dührkop et al., Nat Biotechnol 2021 | 10.1038/s41587-020-0740-8 |
