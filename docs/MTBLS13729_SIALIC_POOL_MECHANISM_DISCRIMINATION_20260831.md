# MTBLS13729 free Neu5Ac 来源机制的预定义转录分支审计

## 问题

Rmu 中 free Neu5Ac 明显升高，而 CMP-Neu5Ac 与 UDP-GlcNAc 未同步。为避免继续用通路名称
猜测来源，本审计预先定义四个可区分的转录分支，并在 TCGA COAD/READ 中使用相同的临床、
六类非重叠 broad-lineage proxy 和 MSI 校正：

1. de novo supply：GNE/NANS/NANP；
2. CMP activation/transport：CMAS/SLC35A1；
3. selected sialidase release：NEU1/NEU3；
4. O-acetylation protection balance：CASD1 − SIAE。

同时保留每个基因单独结果，并把一般 CRC 的 32 对 tumour-normal 与 42 个 mucinous、329 个
conventional primary tumour 的组织学比较分开。共 15 个预设 outcome 统一做 BH 校正。

## 为什么加入 O-acetylation

2025 年两项原始研究确定 NXPE1 能在结肠黏液糖链上催化 Neu5Ac O-acetylation，其中一项
进一步确定产物为 9-O-acetyl-Neu5Ac。O-acetylation 会改变黏液屏障和sialic-acid-binding
protein识别，因此它是 free Neu5Ac 与 glycan destination 之间不能忽略的调节层：

- Nature Communications 2025: https://www.nature.com/articles/s41467-025-59671-9
- JACS 2025: https://doi.org/10.1021/jacs.5c00769

旧版 TCGA HiSeqV2 矩阵没有 NXPE1，故没有用别的基因事后替代。CASD1/SIAE 只构成部分的
O-acetylation protection/removal proxy。

## 主要结果

### 一般 CRC tumour-normal

| 分支 | 平均 tumour-normal z 差 | 升高患者 | BH q | 解释 |
|---|---:|---:|---:|---|
| de novo supply | -0.241 | 13/32 | 0.0455 | 复合轴轻度下降，内部方向相反 |
| CMP activation/transport | -1.078 | 4/32 | 5.18e-7 | 一般CRC中明显下降 |
| NEU1/NEU3 release | +0.854 | 28/32 | 9.02e-7 | 一般CRC中强一致升高 |
| CASD1−SIAE balance | +0.040 | 16/32 | 0.573 | signed balance无变化 |

一般CRC中NEU1与NEU3分别升高，而NEU4、CASD1和SIAE下降。这证明sialidase/O-acetylation
重塑确实存在，但不同亚细胞和反应方向并不一致，不能压缩成一个“sialic remodeling score”。

### Mucinous 相对 conventional primary tumour

| 分支 | lineage-adjusted beta | BH q | MSI-adjusted beta | BH q |
|---|---:|---:|---:|---:|
| de novo supply | +0.465 | 8.08e-8 | +0.436 | 1.53e-6 |
| CMP activation/transport | +0.449 | 1.61e-4 | +0.419 | 7.97e-4 |
| NEU1/NEU3 release | -0.691 | 5.58e-6 | -0.654 | 1.53e-5 |
| CASD1−SIAE balance | +0.151 | 0.124 | +0.167 | 0.0879 |

单基因拆分揭示：CMP轴的正向主要由SLC35A1驱动（lineage beta `+0.702`、q=`1.61e-4`），
CMAS本身不显著（beta `+0.196`、q=`0.180`）。NEU1和NEU3在黏液型相对常规型中均为负，
其中NEU1在MSI校正后仍显著；CASD1为正而SIAE不显著，但signed O-acetyl balance在统一的
15-outcome校正后未通过。

## 对竞争机制的裁决

1. **不支持“Rmu free Neu5Ac由NEU1/NEU3转录上调直接释放”。** 该轴在一般CRC上升，
   但在mucinous相对conventional中方向相反。
2. **支持转录能力与代谢池脱耦。** SLC35A1及de novo supply RNA在mucinous中相对富集，
   但同患者实测CMP-Neu5Ac没有同步扩张。
3. **O-acetylation仍是未决分支。** CASD1的相对升高与signed balance的非显著并存；缺少
   NXPE1、9-O-acetyl-Neu5Ac和酶活性，不能判断保护/去保护方向。
4. 当前更合理的解释优先级是：黏液分泌/周转或亚细胞转运—利用失配，随后才是未测量的
   sialidase activity、微生物释放或O-acetylation改变。它们仍是竞争假说，不是已证机制。

## 不能外推

- bulk RNA 不能报告蛋白/酶活、亚细胞定位、宿主与微生物来源或净通量；
- 42 vs 329 是mucinous-relative背景，不是10位Rmu患者的独立代谢组复现；
- NEU1/NEU3复合轴不包含所有底物和sialidase，不能排除蛋白活性或其他酶介导释放；
- CASD1−SIAE不等于完整O-acetylation，NXPE1在该旧矩阵中缺失；
- SLC35A1表达增加不证明Golgi CMP-Neu5Ac transport增加。

## 可复核工件

- 脚本：`tasks/audit_tcga_sialic_pool_mechanisms_v1.py`
- 结果：`data/external/TCGA_COADREAD_Xena_20260830/sialic_pool_mechanisms_v1/`
- 同患者代谢物审计：`docs/MTBLS13729_SIALIC_DONOR_DECOUPLING_AUDIT_20260831.md`

## 2026-08-31 current-GDC NXPE1 resolution

The earlier statement that NXPE1 was unavailable applied only to the legacy HiSeqV2 matrix. A current-GDC
STAR TPM/FPKM-UQ audit now resolves this branch in the exact locked 371-tumour cohort (42 mucinous, 329
conventional):

- NXPE1 is higher in mucinous tumours after clinical + broad-lineage adjustment (TPM beta `+0.621`,
  `p=0.000369`) and after additional MSI adjustment (`+0.530`, `p=0.00134`);
- the direction and significance reproduce in FPKM-UQ;
- after adding the pre-defined secretory-mucin programme (`MUC2/TFF3/SPDEF/FCGBP/AGR2`), the NXPE1
  coefficient collapses to `+0.064`, `p=0.734` (with MSI: `-0.048`, `p=0.782`);
- in 50 paired general-CRC tumour/normal cases, NXPE1 is lower in tumour in 47/50 pairs.

Two adversarial checks further narrow this statement. First, deleting any one marker from the
pre-defined five-gene secretory programme still removes the mucinous NXPE1 association in both TPM and
FPKM-UQ, and no two-marker adjustment remains significant. Second, patient-level broad-epithelial
pseudobulk in the independent six-pair mucinous GSE236696 cohort shows NXPE1 lower in tumour in 6/6
pairs (mean `-1.084 log2`; exact two-sided `p=0.0625`). NXPE1 is sparse in this dataset and `MUC2` is
absent from all deposited feature indices, so the single-cell result is directional tumour-normal
context only, not a mucinous-versus-conventional replication or an enzyme-activity readout.

Therefore O-acetylation is no longer simply “unmeasured because NXPE1 is absent”. The updated conclusion is
more specific: mucinous-relative NXPE1 enrichment is a **secretory-mucin/carrier-linked state**, not an
independent driver signal. It remains expression context, not protein activity or O-acetylation flux.

The substrate description is also corrected. 2025 primary studies support both free-Neu5Ac and CMP-Neu5Ac
acceptor contexts in vitro; local free Neu5Ac cannot be assigned as the direct in-vivo NXPE1 substrate.

Detailed audit: `docs/MTBLS13729_NXPE1_POOL_CARRIER_OACETYL_MECHANISM_AUDIT_20260831.md`.
