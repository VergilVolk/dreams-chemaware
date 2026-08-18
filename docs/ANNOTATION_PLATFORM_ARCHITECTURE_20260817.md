# Chem-aware DreaMS 注释平台 — 架构 / 数据流 / 改进接口

> 状态日期：2026-08-17。本文件是平台的**权威记录**：架构、数据流、每个模块的输入输出、预留的改进接口、以及诚实的完成状态。所有阈值与方法均挂文献 DOI（见 `annotation/params.py` 的 `SOURCES`）。

---

## 1. 一句话回答「全流程能工作了吗」

**核心注释流程（谱 → 带置信度 + 校准概率 + 暗物质候选的注释表）已端到端实测跑通，可交付。** 三块下游能力代码就绪、但依赖外部条件：

| 能力 | 状态 | 卡点 |
|---|---|---|
| 嵌入 → 检索 → m/z 约束 → Schymanski 分级 → 消融 | ✅ 实测跑通 | — |
| 校准（score → P(correct)） | ✅ 实测跑通 | — |
| 暗物质聚类 + 候选 lead | ✅ 实测跑通 | — |
| 诱饵 FDR（q-value） | ⚠️ 链路已验证，未全量 | 1:1 诱饵嵌入 CPU ~68 min |
| 差异分析（组间） | ⚠️ 代码就绪 | 需分组 replicate（当前 n=1） |
| 通路富集 | ⚠️ 代码就绪 | 需外部 compound→pathway 表 |

---

## 2. 数据流（端到端）

```
[查询谱 hdf5]                              [参考库 MGF]
 dreams.utils.data.MSData                   SMILES/INCHIKEY/PEPMASS/peaks
      |                                          |
      v                                          v
 preprocess_spectrum                      parse_mgf -> records
 (top-100 peaks, max-norm,                {precursor_mz, smiles,
  前置合成 precursor 峰)                    inchikey, name, peaks(2,n)}
      |                                          |
      v                                          v
 embed_hdf5 / embed_records              embed_records
 (冻结 DreaMS backbone + 线性 head,        (同一 backbone + head)
  取 position-0 precursor token)
      |                                          |
      v                                          v
 embeddings.npy [N×1024 L2]              embeddings.npy [M×1024 L2]
 manifest.csv (file_name, scan_number,   manifest.csv (smiles, inchikey,
   precursor_mz, charge, RT, row_in_file)  name, precursor_mz)
      \_________________________________________/
                        |
                 retrieve.chunked_topk
                 cosine top-k + precursor m/z ±ppm 硬约束
                        |
                        v
              annotations.csv (long: 每 query × rank 一行)
              列: query_idx, query_file, query_scan,
                  query_precursor_mz, query_group, rank, cosine,
                  lib_smiles, lib_inchikey, lib_name,
                  lib_precursor_mz, dppm, mz_pass
                        |
        +---------------+----------------+----------------+
        v               v                v                v
  confidence       fdr (可选)        calibrate (可选)   darkmatter
  +schymanski_    +qvalue          +calibrated_prob   dark_mask -> 暗谱
   level(2/3/5)   +fdr_pass        (Platt/isotonic)   cluster_dark + lead
        |               |                |                |
        v               v                v                v
  (分级列)       (FDR 列)          (概率列)         dark_lead 候选
        |___________________________________________________|
                        |
                  annotations.csv (含全部追加列)
                        |
        +---------------+----------------+
        v                                v
  diff (需分组)                     pathway (需外部 DB)
  confident_top1 ->                enrich_annotated (超几何)
  Fisher exact + BH               enrich_mz (mummichog)
        |                                |
        v                                v
  diff.csv (化合物 × 组差异)      pathway.csv (通路富集)
                        |
                        v
              ablation.csv (raw -> +mz -> +fdr 阶梯)
                        |
                        v
              report.json / 本 README
```

**中间产物全部落盘、可审计**（每个阶段独立文件，可单独重跑）：
- `embeddings.npy` / `manifest.csv` / `report.json` — 嵌入层
- `annotations.csv` — 检索 + 追加列
- `annotations_fdr.csv` — FDR 追加（`tasks/run_fdr_met_neg.py`）
- `mona_neg_decoy_emb.npy` — 诱饵嵌入缓存

---

## 3. 模块清单（M0–M9）

| 模块 | 文件 | 职责 | 关键文献 |
|---|---|---|---|
| M0 | `params.py` | 全部阈值/方法，逐条挂 DOI | `SOURCES` 字典 |
| M1 | `_inference.py` `embed.py` `retrieve.py` | 预处理 → 官方嵌入 → cosine top-k + m/z 硬约束 | Bushuiev 2025 |
| M2 | `confidence.py` | Schymanski 1–5 级（平台上限 2a+3） | Schymanski 2014 |
| M3 | `fdr.py` | shuffle 诱饵 + TDA q-value | Elias&Gygi 2007; Scheubert 2017 |
| M4 | `calibrate.py` | Platt/isotonic → P(correct) | Platt 1999; Zadrozny&Elkan 2002; Hoffmann 2022 |
| M5 | `diff.py` | DDA 谱计数 + Fisher + BH | Fisher 1925; BH 1995 |
| M6 | `pathway.py` | 超几何 ORA + mummichog | Li 2013 |
| M7 | `darkmatter.py` | 暗谱聚类 + 候选 lead | Cao 2025; Schymanski 2014 |
| M8 | `ablation.py` | 模块 on/off 阶梯量化 | — |
| M9 | `cli.py` | 端到端命令入口 | — |

---

## 4. 预留的改进接口（扩展点）

以下每个接口都在代码里**真实存在**，标注了「现状」与「如何扩展」。接入时无需改动其他模块。

### 4.1 规则证据接入（已实现；`rule_evidence.py`）
- 位置：`confidence.assign_schymanski(rules_evidence=...)` + `annotation/rule_evidence.py`
- 规则库：**主库 335 条**（`chem_rules_data.json`，NL/CF/ISO/NR/EE/HR）。MassBank 3,151 条被规则引擎自己标为 `tier=extended, evidence=medium` 噪声规则，故排除。之前文档/记忆里的"3,486 条"是 335+3151 相加，非可直接用于注释的证据。
- 诊断证据 = 命中 ≥1 条稀疏规则（CF 特征碎片 / ISO 同位素），经 `spectrum_rule_vector`（与 `build_spectrum_rule_label_cache` / `FastRuleMatcher` bit-for-bit 一致）。
- **实测结论（Met/neg，13,770 谱，非假设）**：
  - 100% 谱命中 ≥1 规则；NL 平均 18.6 条/谱 → "命中任意规则"无区分度。
  - CF 命中（稀疏 1.29 条/谱）**不预测** confident 注释率（CF=0→6.05%，CF≥4→4.19% 反而更低）。
  - 因此规则证据是 Schymanski **语义升级**（精确质量→有碎裂解释的 tentative candidate），**不是置信度升级**。对 13,770 个 top1 命中，实际只翻转 1 个 Level（5→3），因 m/z 硬约束已把可升级空间压缩到 ~246 个 Level 3。
  - 与规则引擎自带 `claim_limit`（"匹配的质量模式不是唯一的碎片结构/断键机制"）一致。
- 扩展：同一证据可作为 `mokapot` 式 rescoring 的额外特征（`fdr.FDR_CITATIONS["rescoring"]` 已挂 DOI，尚未实现打分器）。

### 4.2 诱饵策略（decoy strategy）
- 位置：`params.decoy_strategy: "shuffle" | "fragment-tree" | "precursor-swap"`
- 现状：只实现了 `shuffle`（`fdr.make_shuffle_decoys`）。
- 扩展：`fragment-tree`（断裂树重排，错误更逼真，passatutto 主路线）；`precursor-swap`（交换真实谱的前体 m/z，可复用 target 嵌入、**零额外嵌入成本**——是绕过 68 min 瓶颈的首选，但需在嵌入前改写 precursor 峰，故当前仍需重嵌）。

### 4.3 校准器（calibrator）
- 位置：`calibrate.fit_calibrator(method=...)`
- 现状：`platt` / `isotonic`。
- 扩展：加 temperature scaling、Bayesian binning、`none` 直通；`apply_calibrator` 已按 `params.calibration_method` 分发。

### 4.4 定量后端（差异分析）
- 位置：`diff.py` 的 `confident_top1` / `group_counts`
- 现状：DDA MS2 **谱计数**（半定量，`DIFF_CITATIONS["dda_counts"]` 已声明 caveat）。
- 扩展：把 `group_counts` 的输入从谱计数换成外部定量矩阵（MS1 peak-area / DIA / MaxQuant 输出），Fisher/BH 层不变。

### 4.5 通路数据库（database-agnostic）
- 位置：`pathway.enrich_annotated(mapping=...)` / `enrich_mz(metabolites=...)`
- 现状：算法实现完整，但**不随代码交付任何 compound→pathway 表**（避免编造映射，`PATHWAY_CITATIONS["database"]` 已声明）。
- 扩展：传入 HMDB / KEGG / Reactome / GNPS 的两列或三列 DataFrame 即用。

### 4.6 暗物质类别先验（class-level prior）
- 位置：`darkmatter.DARK_CITATIONS["class_prior"]`
- 现状：标注 `"(optional class-level prior, not yet wired)"`。
- 扩展：接入 CANOPUS（Dührkop 2021）或自训的分类器，给 dark cluster 一个化合物类别先验，收窄候选空间。

### 4.7 检索后端（faiss / GPU / 分片）
- 位置：`retrieve.chunked_topk(chunk=512)`
- 现状：分块 `argpartition` 的朴素实现，纯 numpy。
- 扩展：替换为 faiss IVF/HNSW（大规模库）、GPU matmul、或多机分片；对外只暴露 `(topk_vals, topk_idx)`，下游无感。

### 4.8 嵌入后端（换 backbone / 微调）
- 位置：`embed.DEFAULT_RAW` / `DEFAULT_OFFICIAL` / `load_embedder`
- 现状：指向官方冻结 backbone + 线性 head。
- 扩展：换自定义 backbone；或对 head 做领域微调（`reconstruct_backbone` 已把 backbone/head 分离，head 权重可单独训练）。

### 4.9 CLI 子命令与参数覆盖
- 位置：`cli.build_parser`
- 现状：`embed` / `annotate` 两个子命令。
- 扩展：加 `diff` / `pathway` / `darkmatter` / `report` 子命令；加 `--params-json` 覆盖任意 `Params` 字段（现 `Params` 是 frozen dataclass，可用 `dataclasses.replace`）。

### 4.10 报告输出（HTML/PDF）
- 位置：`retrieve.save` / `cli.cmd_annotate` 末尾
- 现状：`annotations.csv` + `report.json` + 消融文本表。
- 扩展：加 HTML/PDF 报告渲染（消融阶梯、注释率、校准曲线、暗物质候选），面向医学科学交付。

---

## 5. 已知限制（诚实声明）

1. **Level 1 / Level 4 永不输出**——Level 1 需标准品同机 RT+MS/MS，Level 4 需分子式预测器（SIRIUS），平台均不消费。
2. **M6 不随代码交付 compound→pathway 表**——HMDB/KEGG/Reactome 是外部资源，需提供（避免编造）。
3. **M5 是半定量**——DDA 谱计数，精度低于 MS1 peak-area / DIA；且需每组 replicate。
4. **FDR 需 1:1 全量诱饵**——实测 9 谱/s，36,663 诱饵 ≈ 68 min；子集诱饵 q-value 塌缩到下限、无区分度，不能代替。
5. **参考库覆盖有限**——注释率 5.9%（Met/neg vs MONA-neg），94% 为暗物质；鞘脂等需专用库（见 `memory/dreams-sphingolipid-neg-adduct-obstacle`）。

---

## 6. 已验证数字（Met/neg，13,770 谱 × 36,663 谱）

- 消融：raw cosine 0.248（fp 代理 0.764）→ +m/z 0.059（fp 代理 0.000）
- 校准：库内 leave-one-out 70.5% 同化合物命中；正确 hit 均 cos 0.884 vs 错误 0.781
- 暗物质：94.1% 谱无注释
