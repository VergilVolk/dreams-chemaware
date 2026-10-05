# GNPS Gold/Silver 10-ppm 通用评估集

## 定位

这是一个从本地 `ALL_GNPS.mgf` 构建的、模型盲的、有结构标签的
`[M+H]+` MS/MS 检索与谱图对评估集。它复用 NIST20 Figure 4b 的核心数学关系：

- 同一 IK14 为 positive；
- 不同 IK14 且 observed precursor m/z 在 10 ppm 内为 negative；
- pooled cosine score 计算 AUROC/AUPRC。

它不是 NIST20，也不得称为 NIST20 精确复现。GNPS 与 DreaMS 预训练来自同一公共数据
生态，因此该集合证明的是标注 GNPS 上的 identity/formula-disjoint transfer 与鲁棒性，
不是原始预训练语料级别的完全外部独立性。

## 冻结规模

输出目录：`data/validation/gnps_gold_silver_10ppm_benchmark_v1`

- 原始 GNPS：2,091,446 spectra；
- Gold/Silver、结构与 `[M+H]+` 检查后且排除训练身份：329,607 spectra；
- 排除 31,971 个 MassSpecGym/MoNA IK14；
- identity-disjoint 主集：10,995 queries、5,534 query identities、4,204 formulas、
  175,171 spectrum pairs；
- formula-disjoint 核心：5,261 queries、2,662 identities、2,210 formulas、
  47,724 spectrum pairs；
- 主集 7,592 个 query 含同公式异分子 near-negative；
- 所有 positive query/reference 来自不同原始 `FILENAME`；
- exact duplicate positive 与跨身份冲突谱均未进入评估图。

## 文件

- `spectra.mgf`：329,607 条经过质量与泄漏过滤的谱；
- `manifest.csv.gz`：逐谱身份、公式、precursor、来源文件、质量和 hash；
- `panel_identity_disjoint.npz`：主检索图；
- `panel_formula_disjoint.npz`：严格 formula-disjoint 检索图；
- `pairs_identity_disjoint.npz` / `pairs_formula_disjoint.npz`：pooled pair ledgers；
- `report.json`：协议、规模、门和 provenance；
- `checksums.sha256`：传输校验。

## 本地构建与独立复核

```bash
python -u tasks/build_gnps_gold_silver_10ppm_benchmark.py
python -u tasks/validate_gnps_gold_silver_10ppm_benchmark.py
```

独立复核器不相信 `report.json` 的门，而是重新读取 MGF、manifest、两个 panel、两个
pair ledger 和 MassSpecGym/MoNA 排除源，逐 query 检查 IK14、10 ppm、positive 文件独立性、
near 标签和公式泄漏。

## 模型无关评价

输入 `.npz` 必须包含对齐的 `rows` 与单位归一化 `embeddings`；也支持与 manifest 全行
对齐的 `.npy`。

```bash
python -u tasks/evaluate_gnps_gold_silver_10ppm_embeddings.py \
  --baseline-embeddings /path/official_embeddings.npz \
  --candidate-embeddings /path/candidate_embeddings.npz \
  --output /path/gnps_evaluation
```

评价器同时输出 Recall@1/2/3/5/10/20、MRR、mean/median rank、macro-query
AUROC/AUPRC、micro-candidate AUROC/AUPRC、pooled pairwise AUROC/AUPRC、positive-vs-best-
negative margin、Top1-Top2 gap、corrected/introduced/risk-net、near subset 以及按 formula
cluster 的 paired CI。

## 正确措辞

允许：

> GNPS Gold/Silver identity-disjoint 10-ppm benchmark。

> GNPS `[M+H]+` 10-ppm pooled pairwise AUROC。

禁止：

> NIST20 0.85 精确复现。

> 完全未见过的外部预训练语料测试。
